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

/// One generation request. `messages` is an OpenAI chat `messages` array (text +
/// optional inline-`data:` image_url parts), forwarded verbatim to the engine,
/// which applies the model's chat template and handles vision. `has_image` records
/// whether an image is present (for the vision-capability gate + the audit).
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub messages: serde_json::Value,
    pub max_new_tokens: u32,
    pub sampling: Sampling,
    pub has_image: bool,
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

pub mod gateway;
pub use gateway::LlamaServerClient;

// The embedded llama.cpp FFI engine is retained behind the `llama-cpp` feature
// (the standalone / Intel-demo path). The production daemon uses the gateway.
#[cfg(feature = "llama-cpp")]
pub mod engine;
#[cfg(feature = "llama-cpp")]
pub use engine::LlamaEngine;

/// The orchestrator holds the gateway client to llama-server + detected device
/// caps. Inference forwards over HTTP, so the hot path is `&self` (shared) — many
/// requests run concurrently as llama-server batch slots.
pub struct Orchestrator {
    client: Option<gateway::LlamaServerClient>,
    caps: Option<mythd_xpu::DeviceCaps>,
    /// Weight precision recorded for the kernel-capability hint / logging.
    weight_precision: Precision,
    /// Configured context window (tokens) — informational; the live KV pool is
    /// owned by llama-server. Kept for /health + logging.
    n_ctx: i32,
    /// Whether the KV cache is q8_0 (INT8) — informational (set on llama-server).
    kv_q8: bool,
    /// True when the engine accepts images (a vision mmproj is loaded server-side).
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
            client: None,
            caps,
            weight_precision: Precision::Int2, // BitNet reference default
            n_ctx: 8192,
            kv_q8: false,
            vision: false,
        }
    }

    /// Point the orchestrator at a llama-server instance (loopback HTTP). `vision`
    /// declares whether that server has a multimodal projector loaded. No network
    /// call is made here; reachability is exercised on the first request.
    pub fn connect_gateway(&mut self, url: &str, vision: bool) -> Result<()> {
        self.client = Some(gateway::LlamaServerClient::new(url)?);
        self.vision = vision;
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
            Some(c) => format!("llama.cpp/sycl via llama-server ({})", c.name),
            None => "llama.cpp via llama-server".to_string(),
        }
    }

    /// Run one generation by forwarding to llama-server. `&self` so many requests
    /// forward concurrently (no exclusive engine lock).
    pub fn infer(&self, req: &InferenceRequest) -> Result<InferenceResponse> {
        let backend = self.backend_label();
        let client = self.client.as_ref().ok_or(InferenceError::NoModel)?;
        tracing::info!(target: "mythd::infer", %backend, max = req.max_new_tokens, temp = req.sampling.temperature, image = req.has_image, "generate");
        let gen = client.infer(req)?;
        Ok(InferenceResponse {
            text: gen.text,
            prompt_tokens: gen.prompt_tokens,
            generated_tokens: gen.generated_tokens,
            backend,
        })
    }

    /// Run one generation, invoking `on_token` with each token's text as it is
    /// produced (true incremental streaming). `on_token` returns `false` to stop
    /// early (e.g. the client disconnected). `&self` for concurrent forwards.
    pub fn infer_stream(
        &self,
        req: &InferenceRequest,
        on_token: impl FnMut(&str) -> bool,
    ) -> Result<InferenceResponse> {
        let backend = self.backend_label();
        let client = self.client.as_ref().ok_or(InferenceError::NoModel)?;
        tracing::info!(target: "mythd::infer", %backend, max = req.max_new_tokens, temp = req.sampling.temperature, image = req.has_image, "stream");
        let gen = client.infer_stream(req, on_token)?;
        Ok(InferenceResponse {
            text: gen.text,
            prompt_tokens: gen.prompt_tokens,
            generated_tokens: gen.generated_tokens,
            backend,
        })
    }
}

impl Default for Orchestrator {
    fn default() -> Self {
        Self::new()
    }
}
