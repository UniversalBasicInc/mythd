//! End-to-end integration test of the full mythd MVP, as a real client:
//!
//!   1. GET  /health                — confirm the B70 / BitNetInt2 is detected
//!   2. POST /v1/session/init       — genuine ML-KEM-768 handshake; client
//!                                    decapsulates and confirms the shared key
//!   3. POST /v1/chat/completions   — OpenAI JSON, bound to the PQ session →
//!                                    BitNet INT2 inference (non-stream + SSE)
//!   4. GET  /v1/audit/{session}    — fetch the ML-DSA-65 signed entries and
//!                                    verify every signature client-side
//!
//!   cargo run -p mythd-bin --example integration_client -- http://127.0.0.1:8080
//!
//! Exits non-zero if any check fails.

use base64::Engine;
use mythd_core::{verify_audit_signature, KemKeypair};
use sha2::{Digest, Sha256};

fn b64e(b: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(b)
}
fn b64d(s: &str) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD.decode(s.trim()).expect("base64")
}
fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// A reqwest client. For https targets it uses rustls with the post-quantum
/// `X25519MLKEM768` provider (so the transport is PQ). If `api_key` is set, every
/// request carries `Authorization: Bearer <key>`. (Cert verification is disabled
/// only because the test uses a self-signed cert.)
fn make_client(base: &str, api_key: Option<&str>) -> reqwest::blocking::Client {
    let mut builder = reqwest::blocking::Client::builder();
    if let Some(k) = api_key {
        let mut h = reqwest::header::HeaderMap::new();
        let mut v = reqwest::header::HeaderValue::from_str(&format!("Bearer {k}")).unwrap();
        v.set_sensitive(true);
        h.insert(reqwest::header::AUTHORIZATION, v);
        builder = builder.default_headers(h);
    }
    if base.starts_with("https://") {
        use std::sync::Arc;
        let provider = Arc::new(rustls_post_quantum::provider());
        let config = rustls::ClientConfig::builder_with_provider(provider.clone())
            .with_safe_default_protocol_versions()
            .unwrap()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(NoVerify(provider)))
            .with_no_client_auth();
        builder = builder.use_preconfigured_tls(config);
    }
    builder.build().unwrap()
}

/// Self-signed-cert acceptor for the integration test ONLY (the PQ transport
/// guarantee comes from the provider above, not from cert validation).
#[derive(Debug)]
struct NoVerify(std::sync::Arc<rustls::crypto::CryptoProvider>);
impl rustls::client::danger::ServerCertVerifier for NoVerify {
    fn verify_server_cert(
        &self,
        _e: &rustls::pki_types::CertificateDer,
        _i: &[rustls::pki_types::CertificateDer],
        _s: &rustls::pki_types::ServerName,
        _o: &[u8],
        _n: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }
    fn verify_tls12_signature(
        &self,
        _m: &[u8],
        _c: &rustls::pki_types::CertificateDer,
        _d: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }
    fn verify_tls13_signature(
        &self,
        _m: &[u8],
        _c: &rustls::pki_types::CertificateDer,
        _d: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }
    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.0.signature_verification_algorithms.supported_schemes()
    }
}

fn main() {
    let base = std::env::args().nth(1).unwrap_or_else(|| "http://127.0.0.1:8080".into());
    let api_key = std::env::var("MYTHD_API_KEY").ok();
    println!(
        "target: {base}  (transport: {}; auth: {})",
        if base.starts_with("https") { "PQ-TLS X25519MLKEM768" } else { "plain HTTP" },
        if api_key.is_some() { "bearer token" } else { "none" }
    );
    let http = make_client(&base, api_key.as_deref());
    let mut failures = 0;
    macro_rules! check {
        ($cond:expr, $msg:expr) => {
            if $cond {
                println!("  ✓ {}", $msg);
            } else {
                println!("  ✗ {}", $msg);
                failures += 1;
            }
        };
    }

    // 0. auth gate — when a key is configured, an unauthenticated call is rejected
    if api_key.is_some() {
        println!("\n[0] auth gate — request WITHOUT a token must be 401");
        let bare = make_client(&base, None);
        let code = bare
            .post(format!("{base}/v1/session/init"))
            .json(&serde_json::json!({ "client_public_key": "" }))
            .send()
            .unwrap()
            .status();
        check!(code.as_u16() == 401, "unauthenticated request rejected (401)");
    }

    // 1. health -----------------------------------------------------------
    println!("\n[1] GET /health");
    let health: serde_json::Value =
        http.get(format!("{base}/health")).send().unwrap().json().unwrap();
    println!("    {}", health);
    let dev = &health["device"];
    check!(health["kem"] == "ML-KEM-768", "server advertises ML-KEM-768");
    check!(dev["bitnet_int2"] == true, "device reports BitNetInt2 capable");
    check!(dev["device_id"] == "0xe223", "device is the B70 (0xe223)");

    // 2. ML-KEM-768 handshake --------------------------------------------
    println!("\n[2] POST /v1/session/init  (ML-KEM-768)");
    let kp = KemKeypair::generate(); // client keypair; client holds the secret
    let init: serde_json::Value = http
        .post(format!("{base}/v1/session/init"))
        .json(&serde_json::json!({ "client_public_key": b64e(&kp.public_key_bytes()) }))
        .send()
        .unwrap()
        .json()
        .unwrap();
    let session_id = init["session_id"].as_str().unwrap().to_string();
    let ciphertext = b64d(init["ciphertext"].as_str().unwrap());
    let server_conf = init["key_confirmation"].as_str().unwrap();

    // client decapsulates → derives the same session key
    let session_key = kp.decapsulate(&ciphertext).expect("decapsulate");
    let mut h = Sha256::new();
    h.update(session_key.expose());
    h.update(session_id.as_bytes());
    h.update(b"mythd/kem-confirm/v1");
    let client_conf = hex(&h.finalize());
    println!("    session_id = {session_id}");
    check!(client_conf == server_conf, "both sides derived the SAME session key (PQ handshake)");

    // 3a. chat completion (non-stream) -----------------------------------
    println!("\n[3a] POST /v1/chat/completions  (JSON, session-bound)");
    let chat: serde_json::Value = http
        .post(format!("{base}/v1/chat/completions"))
        .header("x-mythd-session", &session_id)
        .json(&serde_json::json!({
            "model": "bitnet",
            "messages": [{ "role": "user", "content": "Q: What is the capital of France? A:" }],
            "max_tokens": 48
        }))
        .send()
        .unwrap()
        .json()
        .unwrap();
    let content = chat["choices"][0]["message"]["content"].as_str().unwrap_or("");
    let kernel = chat["mythd"]["kernel"].as_str().unwrap_or("");
    println!("    kernel  = {kernel}");
    println!("    usage   = {}", chat["usage"]);
    println!("    output  = {}", content.replace('\n', " "));
    check!(kernel == "BitNetInt2", "inference ran on the BitNetInt2 kernel");
    check!(content.to_lowercase().contains("paris"), "answer mentions Paris");

    // 3b. chat completion (SSE stream) -----------------------------------
    println!("\n[3b] POST /v1/chat/completions  (stream: true, SSE)");
    let sse = http
        .post(format!("{base}/v1/chat/completions"))
        .header("x-mythd-session", &session_id)
        .json(&serde_json::json!({
            "model": "bitnet",
            "messages": [{ "role": "user", "content": "Q: Name a primary color. A:" }],
            "max_tokens": 32,
            "stream": true
        }))
        .send()
        .unwrap()
        .text()
        .unwrap();
    let mut streamed = String::new();
    let mut saw_done = false;
    for line in sse.lines() {
        let Some(data) = line.strip_prefix("data: ") else { continue };
        if data.trim() == "[DONE]" {
            saw_done = true;
            continue;
        }
        if let Ok(chunk) = serde_json::from_str::<serde_json::Value>(data) {
            if let Some(c) = chunk["choices"][0]["delta"]["content"].as_str() {
                streamed.push_str(c);
            }
        }
    }
    println!("    reassembled = {}", streamed.replace('\n', " "));
    check!(saw_done, "SSE stream terminated with [DONE]");
    check!(!streamed.is_empty(), "SSE stream produced content");

    // 3c. encrypted chat (/v1/secure/chat) — AEAD under the ML-KEM session key
    println!("\n[3c] POST /v1/secure/chat  (XChaCha20-Poly1305 under the ML-KEM session key)");
    let req_json = serde_json::json!({
        "model": "bitnet",
        "messages": [{ "role": "user", "content": "Q: What is 2 + 2? A:" }],
        "max_tokens": 24
    })
    .to_string();
    // client seals the request under the SAME session key it derived above
    let sealed = session_key.seal(req_json.as_bytes(), session_id.as_bytes()).unwrap();
    let env: serde_json::Value = http
        .post(format!("{base}/v1/secure/chat"))
        .json(&serde_json::json!({ "session_id": session_id.clone(), "ciphertext": b64e(&sealed) }))
        .send()
        .unwrap()
        .json()
        .unwrap();
    let resp_sealed = b64d(env["ciphertext"].as_str().expect("server returns an encrypted envelope"));
    let resp_plain = session_key.open(&resp_sealed, session_id.as_bytes()).expect("decrypts response");
    let resp_json: serde_json::Value = serde_json::from_slice(&resp_plain).unwrap();
    let sc_content = resp_json["choices"][0]["message"]["content"].as_str().unwrap_or("");
    let sc_kernel = resp_json["mythd"]["kernel"].as_str().unwrap_or("");
    println!("    response was returned ENCRYPTED (not plaintext JSON)");
    println!("    decrypted kernel = {sc_kernel}");
    println!("    decrypted output = {}", sc_content.replace('\n', " "));
    check!(sc_kernel == "BitNetInt2", "encrypted inference ran on the BitNetInt2 kernel");
    check!(!sc_content.is_empty(), "encrypted response decrypts to a valid completion");

    // 4. audit verification ----------------------------------------------
    println!("\n[4] GET /v1/audit/{{session}}  (verify ML-DSA-65 signatures)");
    let audit: serde_json::Value =
        http.get(format!("{base}/v1/audit/{session_id}")).send().unwrap().json().unwrap();
    let vk = b64d(audit["audit_verifying_key"].as_str().unwrap());
    let entries = audit["entries"].as_array().cloned().unwrap_or_default();
    println!("    {} signed entries, scheme = {}", entries.len(), audit["signature_scheme"]);
    let mut all_ok = !entries.is_empty();
    for e in &entries {
        let record = b64d(e["record_b64"].as_str().unwrap());
        let sig = {
            let s = e["signature"].as_str().unwrap();
            (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect::<Vec<u8>>()
        };
        let ok = verify_audit_signature(&vk, &record, &sig).is_ok();
        if !ok {
            all_ok = false;
        }
        println!(
            "    seq {} kernel={} gen_tokens={} sig_valid={}",
            e["seq"], e["kernel"], e["generated_tokens"], ok
        );
    }
    check!(all_ok, "every audit entry's ML-DSA-65 signature verifies");
    check!(entries.len() >= 2, "audit recorded both inferences for this session");

    println!("\n========================================");
    if failures == 0 {
        println!("ALL CHECKS PASSED — mythd MVP is real.");
    } else {
        println!("{failures} CHECK(S) FAILED");
        std::process::exit(1);
    }
}
