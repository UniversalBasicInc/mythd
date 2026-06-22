//! # mythd — secure inference daemon
//!
//! Wires all five layers and serves the OpenAI-compatible API:
//!
//! > establish a post-quantum ML-KEM session → run inference through the BitNet
//! > INT2 SYCL kernel → append an ML-DSA-65-signed audit entry → return the
//! > response, optionally over post-quantum (`X25519MLKEM768`) TLS.
//!
//! Config: `/etc/myth/mythd.toml` (override with `MYTHD_CONFIG`).

#![forbid(unsafe_code)]

use anyhow::{Context, Result};
use serde::Deserialize;

/// Daemon configuration. Parsed with serde/toml — no executable deserialization.
#[derive(Debug, Deserialize)]
struct Config {
    #[serde(default = "default_bind")]
    bind: String,
    /// Path to the GGUF model to serve.
    #[serde(default)]
    model_path: Option<String>,
    /// Optional vision projector (mmproj GGUF). When set, the model accepts images
    /// (multimodal); required for every NoFootprint mode to support vision.
    #[serde(default)]
    mmproj_path: Option<String>,
    #[serde(default = "default_model_id")]
    model_id: String,
    /// PQ-TLS cert/key (PEM). If both are set and the `pq-tls` feature is on, the
    /// listener is wrapped in rustls with X25519MLKEM768.
    #[serde(default)]
    tls_cert: Option<String>,
    #[serde(default)]
    tls_key: Option<String>,
    #[serde(default = "default_max_sessions")]
    max_sessions: usize,
    #[serde(default = "default_max_tokens")]
    default_max_tokens: u32,
    /// Context window in tokens (KV cache size). Default 8192; on a GQA model the
    /// KV cache is small, so 16K–32K is cheap on a 32 GB card.
    #[serde(default = "default_n_ctx")]
    n_ctx: u32,
    /// KV cache precision: "f16" (default) or "q8_0" (INT8 — ~half the KV
    /// memory/bandwidth, big long-context win; forces flash attention on).
    #[serde(default = "default_kv_cache_type")]
    kv_cache_type: String,
    /// Path to the persistent ML-DSA-65 audit signing key (32-byte seed, mode
    /// 0600). Generated on first run if absent; reused after, so the audit log's
    /// verifying key is stable across restarts.
    #[serde(default = "default_audit_key_path")]
    audit_key_path: String,
    /// Path to the persistent, signed, hash-chained audit log. If set, entries are
    /// appended + fsync'd and the existing log is integrity-verified on startup.
    /// Omitted → in-memory only (entries lost on restart).
    #[serde(default)]
    audit_log_path: Option<String>,
    /// Accepted API keys (bearer tokens). If empty, authentication is DISABLED
    /// (open mode) and any reachable client may use the inference endpoints.
    #[serde(default)]
    api_keys: Vec<String>,
    /// Max in-flight + queued inference requests before the server returns 503
    /// (backpressure). Inference is serial regardless; this bounds the queue.
    #[serde(default = "default_max_inflight")]
    max_inflight: usize,
}

fn default_bind() -> String {
    // Loopback by default: never network-exposed unless the operator opts in
    // (and the auth interlock in main() then requires api_keys).
    "127.0.0.1:8080".to_string()
}
fn default_model_id() -> String {
    "bitnet-b1.58-2b".to_string()
}
fn default_max_sessions() -> usize {
    256
}
fn default_max_tokens() -> u32 {
    256
}
fn default_n_ctx() -> u32 {
    8192
}
fn default_kv_cache_type() -> String {
    "f16".to_string()
}
fn default_audit_key_path() -> String {
    "/etc/myth/keys/audit-mldsa65.seed".to_string()
}
fn default_max_inflight() -> usize {
    16
}

impl Default for Config {
    fn default() -> Self {
        Self {
            bind: default_bind(),
            model_path: None,
            model_id: default_model_id(),
            mmproj_path: None,
            tls_cert: None,
            tls_key: None,
            max_sessions: default_max_sessions(),
            default_max_tokens: default_max_tokens(),
            n_ctx: default_n_ctx(),
            kv_cache_type: default_kv_cache_type(),
            audit_key_path: default_audit_key_path(),
            audit_log_path: None,
            api_keys: Vec::new(),
            max_inflight: default_max_inflight(),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let config = load_config().unwrap_or_default();
    tracing::info!(?config, "mythd starting");

    // Layer 1: load (or first-time generate) the PERSISTENT ML-DSA-65 audit
    // signing key, so the audit log's verifying key is stable across restarts.
    let (signer, created) =
        mythd_core::load_or_create_audit_signer(std::path::Path::new(&config.audit_key_path))
            .with_context(|| format!("audit key: {}", config.audit_key_path))?;

    // Layers 1/3/4: the secure runtime. With audit_log_path set, the signed
    // hash-chained log is persisted + integrity-verified on startup; otherwise
    // it is in-memory (entries lost on restart).
    let mut runtime = match &config.audit_log_path {
        Some(p) => {
            let rt = mythd_session::SecureRuntime::with_persistent_audit(signer, std::path::Path::new(p))
                .with_context(|| format!("audit log: {p} (failed integrity verification or i/o)"))?;
            tracing::info!(audit_log = %p, entries = rt.audit_entries().len(), "audit log opened + verified");
            rt
        }
        None => {
            tracing::warn!("no audit_log_path configured — audit log is IN-MEMORY (entries lost on restart)");
            mythd_session::SecureRuntime::with_audit_signer(signer)
        }
    };
    let vk = runtime.audit_verifying_key();
    let vk_fp: String = vk.iter().take(4).map(|b| format!("{b:02x}")).collect();
    tracing::info!(
        audit_key = %config.audit_key_path,
        status = if created { "generated" } else { "loaded" },
        verifying_key_fp = %vk_fp,
        audit_pubkey_len = vk.len(),
        "secure runtime initialized (persistent ML-DSA-65 audit key; ML-KEM-768 sessions)"
    );

    // Context window + KV precision — applied before load.
    runtime.set_context_size(config.n_ctx);
    runtime.set_kv_q8(config.kv_cache_type.eq_ignore_ascii_case("q8_0"));

    // Load the model if configured. Without the `llama-cpp` feature this returns
    // "engine not built" — the daemon still serves /health and /v1/models.
    if let Some(path) = &config.model_path {
        match runtime.load_model(path, config.mmproj_path.as_ref(), mythd_inference::Precision::Int2) {
            Ok(()) => tracing::info!(
                model = %path,
                vision = runtime.supports_vision(),
                n_ctx = runtime.context_size(),
                kv_q8 = runtime.kv_q8(),
                mmproj = config.mmproj_path.as_deref().unwrap_or("(none)"),
                "model loaded"
            ),
            Err(e) => tracing::warn!(error = %e, model = %path, "model load failed — inference disabled"),
        }
    } else {
        tracing::warn!("no model_path configured — inference disabled");
    }

    // Layer 5: state + router.
    let state = mythd_api::AppState::new(
        runtime,
        config.model_id.clone(),
        config.max_sessions,
        config.default_max_tokens,
        config.api_keys.clone(),
        config.max_inflight,
    );
    if state.auth_enabled() {
        tracing::info!(api_keys = config.api_keys.len(), "bearer-token auth ENABLED");
    } else {
        tracing::warn!(
            "NO api_keys configured — auth DISABLED (open mode); any reachable client can use the inference endpoints"
        );
    }
    let router = mythd_api::router(state);

    let addr: std::net::SocketAddr = config
        .bind
        .parse()
        .with_context(|| format!("invalid bind address: {}", config.bind))?;

    // Secure-by-default safety interlock: refuse to expose an UNAUTHENTICATED
    // endpoint on a non-loopback interface. Binding to the network requires either
    // api_keys (auth) — strongly also TLS. This fails closed instead of silently
    // serving the world.
    if !addr.ip().is_loopback() && config.api_keys.is_empty() {
        anyhow::bail!(
            "refusing to bind {addr} (non-loopback) with NO api_keys — that would expose an \
             unauthenticated inference endpoint to the network. Set api_keys (and tls_cert/tls_key), \
             or bind a loopback address (127.0.0.1)."
        );
    }
    if !addr.ip().is_loopback() && config.tls_cert.is_none() {
        tracing::warn!(%addr, "exposing on a non-loopback interface WITHOUT TLS — set tls_cert/tls_key (+ build --features pq-tls) for transport encryption; the app-layer ML-KEM session is not server-authenticated on its own");
    }

    match (&config.tls_cert, &config.tls_key) {
        (Some(cert), Some(key)) => serve_tls(addr, router, cert, key).await,
        _ => serve_plain(addr, router).await,
    }
}

/// Resolves when the process receives SIGINT (Ctrl-C) or SIGTERM (systemd stop) —
/// the trigger for graceful shutdown.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        if let Ok(mut s) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            s.recv().await;
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = term => {}
    }
    tracing::info!("shutdown signal received — draining in-flight requests");
}

/// Serve plain HTTP (the ML-KEM application-layer session still applies).
async fn serve_plain(addr: std::net::SocketAddr, router: axum::Router) -> Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::warn!(%addr, "serving WITHOUT transport TLS (set tls_cert/tls_key + build --features pq-tls for PQ-TLS)");
    tracing::info!(%addr, "mythd ready");
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    tracing::info!("mythd stopped cleanly");
    Ok(())
}

#[cfg(feature = "pq-tls")]
async fn serve_tls(
    addr: std::net::SocketAddr,
    router: axum::Router,
    cert_path: &str,
    key_path: &str,
) -> Result<()> {
    use std::sync::Arc;

    // Crypto provider with the post-quantum X25519MLKEM768 hybrid key exchange.
    let provider = Arc::new(rustls_post_quantum::provider());

    let certs = load_certs(cert_path)?;
    let key = load_key(key_path)?;

    let server_config = rustls::ServerConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .context("rustls protocol versions")?
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .context("rustls cert/key")?;

    let tls = axum_server::tls_rustls::RustlsConfig::from_config(Arc::new(server_config));
    tracing::info!(%addr, "mythd ready (PQ-TLS: X25519MLKEM768)");

    // axum-server drains via a Handle: on signal, stop accepting and give
    // in-flight requests a grace period to finish.
    let handle = axum_server::Handle::new();
    let shutdown_handle = handle.clone();
    tokio::spawn(async move {
        shutdown_signal().await;
        shutdown_handle.graceful_shutdown(Some(std::time::Duration::from_secs(30)));
    });
    axum_server::bind_rustls(addr, tls)
        .handle(handle)
        .serve(router.into_make_service())
        .await?;
    tracing::info!("mythd stopped cleanly");
    Ok(())
}

#[cfg(not(feature = "pq-tls"))]
async fn serve_tls(
    addr: std::net::SocketAddr,
    router: axum::Router,
    _cert_path: &str,
    _key_path: &str,
) -> Result<()> {
    tracing::error!("tls_cert/tls_key set but binary built without the `pq-tls` feature — falling back to plain HTTP");
    serve_plain(addr, router).await
}

#[cfg(feature = "pq-tls")]
fn load_certs(path: &str) -> Result<Vec<rustls::pki_types::CertificateDer<'static>>> {
    let data = std::fs::read(path).with_context(|| format!("read cert {path}"))?;
    let certs = rustls_pemfile::certs(&mut &data[..]).collect::<std::result::Result<Vec<_>, _>>()?;
    anyhow::ensure!(!certs.is_empty(), "no certificates in {path}");
    Ok(certs)
}

#[cfg(feature = "pq-tls")]
fn load_key(path: &str) -> Result<rustls::pki_types::PrivateKeyDer<'static>> {
    let data = std::fs::read(path).with_context(|| format!("read key {path}"))?;
    rustls_pemfile::private_key(&mut &data[..])?
        .with_context(|| format!("no private key in {path}"))
}

fn load_config() -> Result<Config> {
    let path = std::env::var("MYTHD_CONFIG").unwrap_or_else(|_| "/etc/myth/mythd.toml".to_string());
    let text = std::fs::read_to_string(&path)?;
    let cfg: Config = toml::from_str(&text)?;
    Ok(cfg)
}
