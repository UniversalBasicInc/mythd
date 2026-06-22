//! # mythd-inference — Layer 3: inference orchestrator
//!
//! Drives generation through llama.cpp (Option A: linked `libllama`/`libggml`
//! from the BitNet SYCL build, via the thin C ABI shim in `kernels/`). The BitNet
//! INT2 kernel is invoked automatically when llama.cpp loads an i2_s GGUF on the
//! SYCL device; other models ride the same path on the [`mythd_xpu`] fallback
//! ladder (dense INT8 → FP16). The security perimeter (Layers 1/4/5) is identical
//! regardless of which kernel runs.
//!
//! ## Unsafe is confined and justified
//! The only `unsafe` here is the FFI to the C shim ([`engine`]); every block
//! carries a `// SAFETY:` comment. Layers 1, 4, 5 are `#![forbid(unsafe_code)]`.

use mythd_xpu::select_kernel;
use thiserror::Error;

// These types appear in this crate's public API (`load_model`, `InferenceResponse`,
// `device_caps`), so downstream layers must be able to name them.
pub use mythd_xpu::{DeviceCaps, KernelSelection, Precision};

#[derive(Debug, Error)]
pub enum InferenceError {
    #[error("core error: {0}")]
    Core(#[from] mythd_core::CoreError),
    #[error("xpu error: {0}")]
    Xpu(#[from] mythd_xpu::XpuError),
    #[error("llama backend not built (enable the `llama-cpp` feature)")]
    EngineNotBuilt,
    #[error("model load failed")]
    ModelLoad,
    #[error("no model loaded")]
    NoModel,
    #[error("generation failed (code {0})")]
    Generate(i32),
    #[error("prompt contained an interior NUL byte")]
    BadPrompt,
}

pub type Result<T> = std::result::Result<T, InferenceError>;

/// Sampling parameters. `temperature <= 0` selects greedy (argmax) decoding,
/// which is fully deterministic; `seed < 0` means a random seed.
#[derive(Debug, Clone, Copy)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub seed: i64,
}

impl Default for Sampling {
    /// Greedy + deterministic — matches mythd's historical behavior.
    fn default() -> Self {
        Self { temperature: 0.0, top_p: 1.0, seed: -1 }
    }
}

/// One generation request: text (and optionally an image) in, text out, plus how
/// to sample. When `image` is set, `prompt` must contain the `<__media__>` marker
/// where the image belongs, and the model must have been loaded with a vision
/// projector (mmproj).
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub prompt: String,
    pub max_new_tokens: u32,
    pub sampling: Sampling,
    pub image: Option<Vec<u8>>,
}

/// Raw engine output: text plus token counts.
#[derive(Debug, Clone)]
pub struct Generation {
    pub text: String,
    pub prompt_tokens: u32,
    pub generated_tokens: u32,
}

/// Generation result + the kernel path taken. The token counts and `kernel`
/// feed Layer 4's signed audit record.
#[derive(Debug, Clone)]
pub struct InferenceResponse {
    pub text: String,
    pub prompt_tokens: u32,
    pub generated_tokens: u32,
    /// Honest execution-backend label recorded in the audit (e.g.
    /// `"llama.cpp/sycl (Intel(R) Graphics [0xe223])"` or `"llama.cpp/cpu"`).
    /// mythd delegates per-tensor kernel choice to llama.cpp, so this records the
    /// backend the inference actually ran on — not a fixed kernel guess.
    pub backend: String,
}

pub mod engine;
pub use engine::LlamaEngine;

/// The orchestrator owns the engine + detected device caps and routes work.
pub struct Orchestrator {
    engine: Option<LlamaEngine>,
    caps: Option<mythd_xpu::DeviceCaps>,
    /// Weight precision of the loaded model (drives kernel selection / logging).
    weight_precision: Precision,
    n_gpu_layers: i32,
    n_ctx: i32,
    /// Store the KV cache as q8_0 (INT8) instead of f16 — ~half KV memory/bandwidth.
    kv_q8: bool,
    /// True when the loaded model has a vision projector (accepts images).
    vision: bool,
}

impl Orchestrator {
    pub fn new() -> Self {
        let caps = mythd_xpu::DeviceCaps::detect().ok();
        match &caps {
            Some(c) => tracing::info!(target: "mythd::infer", device = %c.name, is_b70 = c.is_b70, "xpu detected"),
            None => tracing::warn!(target: "mythd::infer", "no XPU detected — host fallback path"),
        }
        Self {
            engine: None,
            caps,
            weight_precision: Precision::Int2, // BitNet reference default
            n_gpu_layers: 999,                 // offload all layers when a GPU is present
            n_ctx: 4096,
            kv_q8: false,
            vision: false,
        }
    }

    /// Load a model (GGUF) through llama.cpp, optionally with a vision projector
    /// (`mmproj`) so it accepts images. `weight_precision` is recorded for logging.
    pub fn load_model(
        &mut self,
        path: impl AsRef<std::path::Path>,
        mmproj: Option<impl AsRef<std::path::Path>>,
        weight_precision: Precision,
    ) -> Result<()> {
        let mut eng = LlamaEngine::new()?;
        let mmproj_ref = mmproj.as_ref().map(|p| p.as_ref());
        eng.load(path.as_ref(), mmproj_ref, self.n_gpu_layers, self.n_ctx, self.kv_q8)?;
        self.engine = Some(eng);
        self.weight_precision = weight_precision;
        self.vision = mmproj_ref.is_some();
        Ok(())
    }

    /// Set the context window (tokens) used when loading the model. Call before
    /// [`load_model`](Self::load_model). The KV cache scales with this; on a
    /// GQA model (few KV heads) even 32K is cheap.
    pub fn set_context_size(&mut self, n_ctx: u32) {
        self.n_ctx = n_ctx.max(512) as i32;
    }

    /// The configured context window (tokens).
    pub fn context_size(&self) -> u32 {
        self.n_ctx as u32
    }

    /// Store the KV cache as q8_0 (INT8). Call before [`load_model`](Self::load_model).
    pub fn set_kv_q8(&mut self, on: bool) {
        self.kv_q8 = on;
    }

    /// Whether the KV cache is stored as q8_0 (INT8).
    pub fn kv_q8(&self) -> bool {
        self.kv_q8
    }

    /// Whether the loaded model can accept images.
    pub fn supports_vision(&self) -> bool {
        self.vision
    }

    /// The detected device capabilities (for `/health` and routing introspection).
    pub fn device_caps(&self) -> Option<&DeviceCaps> {
        self.caps.as_ref()
    }

    /// The ternary fast-path kernel a BitNet model *would* take on this device
    /// (capability hint, not a claim about what ran). The actual per-tensor kernel
    /// for a general GGUF is chosen by llama.cpp — see [`backend_label`](Self::backend_label).
    pub fn kernel_selection(&self) -> KernelSelection {
        self.caps
            .as_ref()
            .map(|c| select_kernel(c, self.weight_precision))
            .unwrap_or(KernelSelection::Fp16)
    }

    /// Honest label for the backend inference actually runs on (recorded in the
    /// audit). mythd dispatches through llama.cpp; with an Intel GPU detected the
    /// SYCL backend handles offloaded layers, otherwise CPU.
    pub fn backend_label(&self) -> String {
        match self.caps.as_ref() {
            Some(c) => format!("llama.cpp/sycl ({})", c.name),
            None => "llama.cpp/cpu".to_string(),
        }
    }

    /// Run one generation.
    pub fn infer(&mut self, req: &InferenceRequest) -> Result<InferenceResponse> {
        let backend = self.backend_label();
        let engine = self.engine.as_mut().ok_or(InferenceError::NoModel)?;
        tracing::info!(target: "mythd::infer", %backend, max = req.max_new_tokens, temp = req.sampling.temperature, image = req.image.is_some(), "generate");
        let gen = engine.generate(&req.prompt, req.max_new_tokens as i32, req.sampling, req.image.as_deref())?;
        Ok(InferenceResponse {
            text: gen.text,
            prompt_tokens: gen.prompt_tokens,
            generated_tokens: gen.generated_tokens,
            backend,
        })
    }

    /// Run one generation, invoking `on_token` with each token's text as it is
    /// produced (true incremental streaming), and returning the assembled
    /// response. `on_token` returns `false` to stop early (e.g. the client
    /// disconnected) — generation halts and the partial response is returned.
    /// `on_token` is called on the calling thread between decode steps.
    pub fn infer_stream(
        &mut self,
        req: &InferenceRequest,
        mut on_token: impl FnMut(&str) -> bool,
    ) -> Result<InferenceResponse> {
        let backend = self.backend_label();
        let engine = self.engine.as_mut().ok_or(InferenceError::NoModel)?;
        tracing::info!(target: "mythd::infer", %backend, max = req.max_new_tokens, temp = req.sampling.temperature, image = req.image.is_some(), "stream");
        let prompt_tokens = engine.stream_begin(&req.prompt, req.max_new_tokens as i32, req.sampling, req.image.as_deref())?;
        let mut text = String::new();
        while let Some(piece) = engine.stream_next()? {
            text.push_str(&piece);
            if !on_token(&piece) {
                tracing::debug!(target: "mythd::infer", "stream consumer cancelled — stopping generation");
                break;
            }
        }
        Ok(InferenceResponse {
            text,
            prompt_tokens,
            generated_tokens: engine.stream_gen_count(),
            backend,
        })
    }
}

impl Default for Orchestrator {
    fn default() -> Self {
        Self::new()
    }
}
