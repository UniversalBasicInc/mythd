//! Live Layer 3 test: forward one chat completion through the mythd
//! orchestrator to a running llama-server (the gateway path the daemon uses).
//!
//!   cargo run -p mythd-inference --example generate -- \
//!       http://127.0.0.1:8091 "Your prompt here" 64
//!
//! Requires a llama-server reachable at the given URL (see
//! `deploy/llama-server.service`). On an Intel GPU host the response's backend
//! label reports the SYCL device the engine ran on; elsewhere it reports CPU.

use mythd_inference::{InferenceRequest, Orchestrator};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(true)
        .init();

    let mut args = std::env::args().skip(1);
    let url = args
        .next()
        .expect("usage: generate <llama-server url> [prompt] [max_tokens]");
    let prompt = args
        .next()
        .unwrap_or_else(|| "Explain what makes a runtime secure in one sentence.".to_string());
    let max_new_tokens: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(64);

    let mut orch = Orchestrator::new();
    println!("kernel route (detected): {:?}", orch.kernel_selection());

    println!("gateway: {url}");
    orch.connect_gateway(&url, false)?;

    let req = InferenceRequest {
        messages: serde_json::json!([{ "role": "user", "content": prompt }]),
        max_new_tokens,
        sampling: Default::default(),
        has_image: false,
    };
    println!("--- prompt ---\n{prompt}\n--- output ---");

    let resp = orch.infer(&req)?;
    println!("{}", resp.text);
    println!(
        "\n--- via backend: {} | prompt_tokens={} generated_tokens={} ---",
        resp.backend, resp.prompt_tokens, resp.generated_tokens
    );
    Ok(())
}
