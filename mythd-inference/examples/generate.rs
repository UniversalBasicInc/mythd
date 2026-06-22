//! Live end-to-end Layer 3 test: load a GGUF through mythd → llama.cpp → the
//! BitNet INT2 SYCL kernel on the XPU, and generate.
//!
//!   cargo run -p mythd-inference --example generate --features llama-cpp -- \
//!       /path/to/model.gguf "Your prompt here" 64
//!
//! Requires the `llama-cpp` feature, the linked shim + libllama/libggml, oneAPI
//! sourced, and an Intel GPU. Without an XPU it still runs on the host fallback.

use mythd_inference::{InferenceRequest, Orchestrator};
use mythd_xpu::Precision;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(true)
        .init();

    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .expect("usage: generate <model.gguf> [prompt] [max_tokens]");
    let prompt = args
        .next()
        .unwrap_or_else(|| "Explain what makes a runtime secure in one sentence.".to_string());
    let max_new_tokens: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(64);

    let mut orch = Orchestrator::new();
    println!("kernel route (pre-load): {:?}", orch.kernel_selection());

    println!("loading model: {model}");
    orch.load_model(&model, None::<&str>, Precision::Int2)?;

    let req = InferenceRequest {
        prompt: prompt.clone(),
        max_new_tokens,
        sampling: Default::default(),
        image: None,
    };
    println!("kernel route (in use): {:?}", orch.kernel_selection());
    println!("--- prompt ---\n{prompt}\n--- output ---");

    let resp = orch.infer(&req)?;
    println!("{}", resp.text);
    println!(
        "\n--- via backend: {} | prompt_tokens={} generated_tokens={} ---",
        resp.backend, resp.prompt_tokens, resp.generated_tokens
    );
    Ok(())
}
