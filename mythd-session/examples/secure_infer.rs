//! Live MVP demonstration of the full mythd perimeter:
//! open session → infer through BitNet INT2 SYCL → append signed audit entry →
//! close session, then verify (and tamper-test) the post-quantum signed log.
//!
//!   cargo run -p mythd-session --example secure_infer --features llama-cpp -- \
//!       /path/to/ggml-model-i2_s.gguf
//!
//! (Without --features llama-cpp it exercises the audit/session logic but the
//! inference call returns "engine not built".)

use mythd_session::{verify_chain, Capabilities, Precision, Request, SecureRuntime};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(true)
        .init();

    let model = std::env::args()
        .nth(1)
        .expect("usage: secure_infer <model.gguf>");

    let mut rt = SecureRuntime::new();
    println!("loading model: {model}");
    rt.load_model(&model, None::<&str>, Precision::Int2)?;

    // A session permitted to generate up to 48 tokens (deny-by-default otherwise).
    let caps = Capabilities::allow_infer(48);

    let prompts = [
        "Q: What is the capital of France? A:",
        "Q: Name a primary color. A:",
        "Q: What is 2 + 2? A:",
    ];

    for p in prompts {
        let req = Request {
            prompt: p.to_string(),
            max_new_tokens: 48,
            sampling: Default::default(),
            image: None,
        };
        let r = rt.infer(&req, caps)?;
        println!("\n=== session {} (audit seq {}) ===", r.session_id, r.audit_seq);
        println!("kernel      : {}", r.kernel);
        println!("tokens      : prompt={} generated={}", r.prompt_tokens, r.generated_tokens);
        println!("audit sig   : {}…", &r.audit_signature_hex[..32]);
        println!("output      : {}", r.text.replace('\n', " "));
    }

    // Verify the whole post-quantum-signed, hash-chained log.
    let vk = rt.audit_verifying_key();
    let entries = rt.audit_entries();
    println!(
        "\n--- audit log: {} entries, ML-DSA-65 verifying key {} bytes ---",
        entries.len(),
        vk.len()
    );
    match verify_chain(entries, &vk) {
        Ok(()) => println!("CHAIN VERIFY: PASS (every entry signed + linked)"),
        Err(e) => println!("CHAIN VERIFY: FAIL — {e}"),
    }

    // Tamper demonstration: flip one byte of one signed record → chain must reject.
    let mut tampered = entries.to_vec();
    if let Some(first) = tampered.first_mut() {
        first.record_json[12] ^= 0x01;
    }
    match verify_chain(&tampered, &vk) {
        Ok(()) => println!("TAMPER TEST: FAIL — tamper went undetected!"),
        Err(e) => println!("TAMPER TEST: PASS — detected: {e}"),
    }

    Ok(())
}
