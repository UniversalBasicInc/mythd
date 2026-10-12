//! Live demonstration of the full mythd perimeter:
//! open session → infer via the llama-server gateway → append signed audit
//! entry → close session, then verify (and tamper-test) the post-quantum
//! signed log.
//!
//!   cargo run -p mythd-session --example secure_infer -- http://127.0.0.1:8091
//!
//! Requires a llama-server reachable at the given URL. Without one, the
//! inference calls fail and the audit log stays empty.

use mythd_session::{verify_chain, Capabilities, Request, SecureRuntime};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(true)
        .init();

    let url = std::env::args()
        .nth(1)
        .expect("usage: secure_infer <llama-server url>");

    let mut rt = SecureRuntime::new();
    println!("gateway: {url}");
    rt.connect_gateway(&url, false)?;

    // A session permitted to generate up to 48 tokens (deny-by-default otherwise).
    let caps = Capabilities::allow_infer(48);

    let prompts = [
        "Q: What is the capital of France? A:",
        "Q: Name a primary color. A:",
        "Q: What is 2 + 2? A:",
    ];

    for p in prompts {
        let req = Request {
            messages: serde_json::json!([{ "role": "user", "content": p }]),
            max_new_tokens: 48,
            sampling: Default::default(),
            has_image: false,
        };
        let r = rt.infer(&req, caps)?;
        println!("\n=== session {} (audit seq {}) ===", r.session_id, r.audit_seq);
        println!("backend     : {}", r.kernel);
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
    match verify_chain(&entries, &vk) {
        Ok(()) => println!("CHAIN VERIFY: PASS (every entry signed + linked)"),
        Err(e) => println!("CHAIN VERIFY: FAIL — {e}"),
    }

    // Tamper demonstration: flip one byte of one signed record → chain must reject.
    let mut tampered = entries.clone();
    if let Some(first) = tampered.first_mut() {
        first.record_json[12] ^= 0x01;
    }
    match verify_chain(&tampered, &vk) {
        Ok(()) => println!("TAMPER TEST: FAIL — tamper went undetected!"),
        Err(e) => println!("TAMPER TEST: PASS — detected: {e}"),
    }

    Ok(())
}
