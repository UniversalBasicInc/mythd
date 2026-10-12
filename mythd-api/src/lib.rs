//! # mythd-api — Layer 5: API surface
//!
//! OpenAI-compatible HTTP API (Axum) over the secure runtime. Endpoints:
//!
//! * `POST /v1/session/init` — **ML-KEM-768 post-quantum handshake.** The client
//!   sends its encapsulation (public) key; the server encapsulates a shared secret
//!   to it, returns the ciphertext, and both sides derive the same [`SessionKey`]
//!   via HKDF-SHA256 ([`mythd_core::encapsulate_to`] / `KemKeypair::decapsulate`).
//!   A key-confirmation tag lets the client prove both derived the same key.
//! * `POST /v1/chat/completions` — OpenAI chat completion (JSON or SSE stream),
//!   routed through [`SecureRuntime`]; works with standard OpenAI client libs. An
//!   optional `X-Mythd-Session` header binds the call to a handshaked session, so
//!   the signed audit record carries that session id.
//! * `GET  /v1/models` — model list in OpenAI format.
//! * `GET  /v1/audit/{session_id}` — the signed, hash-chained audit entries for a
//!   session, so a client can verify its inference was handled and recorded.
//! * `GET  /health` — liveness + detected kernel capabilities (Xe2/XMX/BitNetInt2).
//!
//! Transport TLS (post-quantum `X25519MLKEM768`) is applied by the daemon
//! (`mythd-bin`); this crate is transport-agnostic. The ML-KEM session key here
//! protects the *application* layer; PQ-TLS protects the *transport* layer.

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use mythd_core::SessionKey;
use mythd_inference::{InferenceRequest, Sampling};
use mythd_session::{Capabilities, SecureRuntime, SessionId};

pub mod openai {
    //! Minimal OpenAI-compatible request types (responses are built with
    //! `serde_json` so the wire shape exactly matches what clients expect).
    use super::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    pub struct ChatMessage {
        pub role: String,
        pub content: Content,
    }

    /// OpenAI message content: either a plain string, or an array of parts (the
    /// vision format, where parts can be text or an image).
    #[derive(Debug, Serialize, Deserialize)]
    #[serde(untagged)]
    pub enum Content {
        Text(String),
        Parts(Vec<ContentPart>),
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(tag = "type")]
    pub enum ContentPart {
        #[serde(rename = "text")]
        Text { text: String },
        #[serde(rename = "image_url")]
        ImageUrl { image_url: ImageUrl },
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct ImageUrl {
        pub url: String,
    }

    #[derive(Debug, Deserialize)]
    pub struct ChatCompletionRequest {
        #[serde(default)]
        pub model: String,
        pub messages: Vec<ChatMessage>,
        #[serde(default)]
        pub max_tokens: Option<u32>,
        #[serde(default)]
        pub stream: bool,
        /// OpenAI sampling controls. Omitted → mythd defaults (greedy/deterministic).
        #[serde(default)]
        pub temperature: Option<f32>,
        #[serde(default)]
        pub top_p: Option<f32>,
        #[serde(default)]
        pub seed: Option<i64>,
    }

    /// ML-KEM session init request: the client's encapsulation key (base64).
    #[derive(Debug, Deserialize)]
    pub struct SessionInitRequest {
        pub client_public_key: String,
    }

    /// Encrypted ("secure") chat request: a standard OpenAI
    /// [`ChatCompletionRequest`] JSON, AEAD-sealed under the session key.
    #[derive(Debug, Deserialize)]
    pub struct SecureChatRequest {
        pub session_id: String,
        /// base64( nonce(24) || XChaCha20-Poly1305 ciphertext+tag ); AAD = session id.
        pub ciphertext: String,
    }

    // `Serialize` is used by response helpers in the parent module.
    #[allow(unused_imports)]
    use Serialize as _;
}

use openai::{ChatCompletionRequest, SecureChatRequest, SessionInitRequest};

// ---------------------------------------------------------------------------
// Shared application state
// ---------------------------------------------------------------------------

/// How long an idle handshaked session lives before it is evicted. Bounds the
/// session table (memory) and prevents a client from permanently exhausting it.
const SESSION_TTL: std::time::Duration = std::time::Duration::from_secs(1800);

/// One handshaked transport session: the ML-KEM-derived key plus its grant.
struct TransportSession {
    // The ML-KEM-derived key now encrypts /v1/secure/chat payloads (AEAD);
    // zeroized on drop.
    key: SessionKey,
    caps: Capabilities,
    /// Last time this session was created or used — drives idle eviction.
    last_used: std::time::Instant,
}

/// Drop sessions idle longer than [`SESSION_TTL`]. Returns the number evicted.
fn evict_expired(sessions: &mut HashMap<String, TransportSession>, now: std::time::Instant) -> usize {
    let before = sessions.len();
    sessions.retain(|_, s| now.duration_since(s.last_used) < SESSION_TTL);
    before - sessions.len()
}

/// Snapshot of detected device capabilities, for `/health`.
#[derive(Clone, Serialize)]
pub struct DeviceInfo {
    pub detected: bool,
    pub name: String,
    pub device_id: String,
    pub is_xe2: bool,
    pub has_xmx: bool,
    pub bitnet_int2: bool,
}

struct AppInner {
    // No outer lock: SecureRuntime is Send+Sync and its inference path is &self,
    // so requests forward concurrently (bounded by `infer_sem`). The brief
    // sessions/audit locks live inside SecureRuntime, never held across a forward.
    runtime: SecureRuntime,
    sessions: tokio::sync::Mutex<HashMap<String, TransportSession>>,
    model_id: String,
    max_sessions: usize,
    default_max_tokens: u32,
    max_tokens_cap: u32,
    device: DeviceInfo,
    /// SHA-256 hashes of accepted API keys. Empty = auth disabled (open mode).
    api_key_hashes: Vec<[u8; 32]>,
    /// Audit log's public verifying key, cached at startup (stable for the life of
    /// the process), so session handshakes don't contend the inference lock.
    audit_verifying_key: Vec<u8>,
    /// Admission control: caps total concurrent inference so the server sheds load
    /// (503) instead of piling up. Inference now runs concurrently — each request
    /// forwards to a llama-server batch slot — so this bounds how many at once.
    infer_sem: Arc<tokio::sync::Semaphore>,
    /// True when the loaded model accepts images (a vision mmproj is loaded).
    vision: bool,
}

/// Cloneable handle to the server state (Axum requires `Clone + Send + Sync`).
#[derive(Clone)]
pub struct AppState {
    inner: Arc<AppInner>,
}

impl AppState {
    /// Build server state around an initialized [`SecureRuntime`].
    ///
    /// `api_keys` are the accepted bearer tokens. If empty, authentication is
    /// **disabled** (open mode) and every request is allowed.
    pub fn new(
        runtime: SecureRuntime,
        model_id: impl Into<String>,
        max_sessions: usize,
        default_max_tokens: u32,
        api_keys: Vec<String>,
        max_inflight: usize,
    ) -> Self {
        let device = match runtime.device_caps() {
            Some(c) => DeviceInfo {
                detected: true,
                name: c.name.clone(),
                device_id: format!("0x{:04x}", c.device_id),
                is_xe2: c.is_xe2,
                has_xmx: c.has_xmx,
                bitnet_int2: c.supports_bitnet_int2(),
            },
            None => DeviceInfo {
                detected: false,
                name: "none".into(),
                device_id: "0x0000".into(),
                is_xe2: false,
                has_xmx: false,
                bitnet_int2: false,
            },
        };
        let api_key_hashes = api_keys.iter().map(|k| sha256_32(k.as_bytes())).collect();
        // Capture the (stable) audit verifying key + vision capability before
        // `runtime` is moved into the mutex.
        let audit_verifying_key = runtime.audit_verifying_key();
        let vision = runtime.supports_vision();
        let infer_sem = Arc::new(tokio::sync::Semaphore::new(max_inflight.max(1)));
        Self {
            inner: Arc::new(AppInner {
                runtime,
                sessions: tokio::sync::Mutex::new(HashMap::new()),
                model_id: model_id.into(),
                max_sessions,
                default_max_tokens,
                max_tokens_cap: default_max_tokens.max(512),
                device,
                api_key_hashes,
                audit_verifying_key,
                infer_sem,
                vision,
            }),
        }
    }

    /// Whether bearer-token auth is enforced (any API keys configured).
    pub fn auth_enabled(&self) -> bool {
        !self.inner.api_key_hashes.is_empty()
    }

    /// Require a valid `Authorization: Bearer <key>` when API keys are configured;
    /// open (always Ok) when none are. Constant-time over the key set.
    fn authorize(&self, headers: &HeaderMap) -> std::result::Result<(), ApiError> {
        if self.inner.api_key_hashes.is_empty() {
            return Ok(()); // open mode
        }
        let token = headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.strip_prefix("Bearer ").or_else(|| s.strip_prefix("bearer ")))
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .ok_or_else(|| {
                ApiError::new(
                    StatusCode::UNAUTHORIZED,
                    "authentication_error",
                    "missing or malformed bearer token",
                )
            })?;
        let presented = sha256_32(token.as_bytes());
        let mut matched = 0u8;
        for h in &self.inner.api_key_hashes {
            matched |= ct_eq_32(&presented, h);
        }
        if matched == 1 {
            Ok(())
        } else {
            Err(ApiError::new(
                StatusCode::UNAUTHORIZED,
                "authentication_error",
                "invalid api key",
            ))
        }
    }
}

/// Build the OpenAI-compatible router bound to `state`.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/session/init", post(session_init))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/secure/chat", post(secure_chat))
        .route("/v1/audit/{session_id}", get(audit))
        .with_state(state)
}

/// Back-compat shim used by the daemon skeleton; prefer [`router`].
pub async fn build_router() -> Router {
    Router::new().route("/health", get(health_static))
}

async fn health_static() -> &'static str {
    "ok"
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// API error. Messages are intentionally coarse — no internal/infra detail on the
/// wire; specifics are logged server-side.
#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self { status, code, message: message.into() }
    }
    fn bad_request(msg: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_request_error", msg)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({
            "error": { "message": self.message, "type": self.code }
        });
        (self.status, Json(body)).into_response()
    }
}

impl From<mythd_session::SessionError> for ApiError {
    fn from(e: mythd_session::SessionError) -> Self {
        use mythd_session::SessionError as E;
        match e {
            E::Denied(what) => ApiError::new(
                StatusCode::FORBIDDEN,
                "permission_error",
                format!("capability denied: {what}"),
            ),
            E::UnknownSession => {
                ApiError::new(StatusCode::NOT_FOUND, "not_found_error", "unknown session")
            }
            E::Inference(ie) => {
                tracing::warn!(target: "mythd::api", error = %ie, "inference failed");
                // Coarse, infra-free message on the wire.
                ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "server_error", "inference unavailable")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn health(State(st): State<AppState>) -> Json<serde_json::Value> {
    let d = &st.inner.device;
    Json(serde_json::json!({
        "status": "ok",
        "service": "mythd",
        "model": st.inner.model_id,
        "vision": st.inner.vision,
        "kem": "ML-KEM-768",
        "audit_signatures": "ML-DSA-65",
        "device": {
            "detected": d.detected,
            "name": d.name,
            "device_id": d.device_id,
            "is_xe2": d.is_xe2,
            "has_xmx": d.has_xmx,
            "bitnet_int2": d.bitnet_int2,
        }
    }))
}

async fn models(
    State(st): State<AppState>,
    headers: HeaderMap,
) -> std::result::Result<Json<serde_json::Value>, ApiError> {
    st.authorize(&headers)?;
    Ok(Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": st.inner.model_id,
            "object": "model",
            "created": 0,
            "owned_by": "mythd",
        }]
    })))
}

/// ML-KEM-768 handshake: client sends its public key; server encapsulates a
/// shared secret to it and returns the ciphertext. Both sides derive the same
/// session key; the returned confirmation tag lets the client check the match.
async fn session_init(
    State(st): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<SessionInitRequest>,
) -> std::result::Result<Json<serde_json::Value>, ApiError> {
    st.authorize(&headers)?;
    let client_pk = b64_decode(&req.client_public_key)
        .map_err(|_| ApiError::bad_request("client_public_key must be base64"))?;

    // Server side of the KEM: encapsulate to the client's public key.
    let (ciphertext, session_key) = mythd_core::encapsulate_to(&client_pk)
        .map_err(|_| ApiError::bad_request("invalid ML-KEM public key"))?;

    let session_id = SessionId::random();
    let confirmation = key_confirmation(&session_key, &session_id.0);

    // Default grant for a handshaked session.
    let caps = Capabilities::allow_infer(st.inner.default_max_tokens);
    let audit_vk = {
        let now = std::time::Instant::now();
        let mut sessions = st.inner.sessions.lock().await;
        // Reclaim idle sessions first, so a transient burst can't permanently
        // exhaust the table (and so it doesn't grow without bound).
        let evicted = evict_expired(&mut sessions, now);
        if evicted > 0 {
            tracing::debug!(target: "mythd::api", evicted, "expired sessions reclaimed");
        }
        if sessions.len() >= st.inner.max_sessions {
            return Err(ApiError::new(
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
                "session limit reached",
            ));
        }
        sessions.insert(
            session_id.0.clone(),
            TransportSession { key: session_key, caps, last_used: now },
        );
        // cached verifying key — no runtime lock, so handshakes stay responsive
        // even while an inference is in flight
        st.inner.audit_verifying_key.clone()
    };

    tracing::info!(target: "mythd::api", session = %session_id.0, "ML-KEM session established");
    Ok(Json(serde_json::json!({
        "session_id": session_id.0,
        "kem": "ML-KEM-768",
        "kdf": "HKDF-SHA256",
        "ciphertext": b64_encode(&ciphertext),
        "key_confirmation": confirmation,
        "audit_verifying_key": b64_encode(&audit_vk),
    })))
}

async fn chat_completions(
    State(st): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> std::result::Result<Response, ApiError> {
    st.authorize(&headers)?;
    if req.messages.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty"));
    }

    // Bind to a handshaked session if the client supplied one.
    let session_id = headers
        .get("x-mythd-session")
        .and_then(|v| v.to_str().ok())
        .map(|s| SessionId(s.to_string()));

    let requested = req.max_tokens.unwrap_or(st.inner.default_max_tokens);
    let max_tokens = requested.min(st.inner.max_tokens_cap).max(1);

    // Capabilities from the handshaked session, else a default grant.
    let caps = match &session_id {
        Some(sid) => {
            let mut sessions = st.inner.sessions.lock().await;
            match sessions.get_mut(&sid.0) {
                Some(ts) => {
                    ts.last_used = std::time::Instant::now(); // sliding TTL
                    ts.caps
                }
                None => {
                    return Err(ApiError::new(
                        StatusCode::UNAUTHORIZED,
                        "authentication_error",
                        "unknown or expired session",
                    ))
                }
            }
        }
        None => Capabilities::allow_infer(max_tokens),
    };

    let (messages, has_image) = prepare_messages(&req.messages)?;
    if has_image && !st.inner.vision {
        return Err(ApiError::bad_request("this model has no vision support (no mmproj loaded)"));
    }
    let inf_req = InferenceRequest {
        messages,
        max_new_tokens: max_tokens,
        sampling: sampling_from(req.temperature, req.top_p, req.seed),
        has_image,
    };

    // Admission control: cap total in-flight + queued inference (backpressure →
    // 503 when saturated, instead of unbounded pileup). Held for the whole
    // request (including the full stream duration).
    let permit = st.inner.infer_sem.clone().try_acquire_owned().map_err(|_| {
        ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "server_busy",
            "inference queue full, retry shortly",
        )
    })?;

    let model = if req.model.is_empty() {
        st.inner.model_id.clone()
    } else {
        req.model.clone()
    };

    if req.stream {
        // True per-token streaming: generation runs on a blocking thread holding
        // the runtime lock + the permit, emitting each token to the SSE channel as
        // it is decoded; the final chunk carries the signed-audit provenance.
        return Ok(stream_response(st.clone(), session_id, inf_req, caps, model, permit));
    }

    // Non-streaming: the forward to llama-server is blocking, so run it on a
    // blocking thread (no shared lock) → many forwards run concurrently.
    let st2 = st.clone();
    let resp = tokio::task::spawn_blocking(move || {
        let _permit = permit; // held for the duration of the forward
        st2.inner.runtime.infer_with_session(session_id, &inf_req, caps)
    })
    .await
    .map_err(|_| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "server_error", "inference task failed"))??;
    Ok(Json(json_completion(&resp, &model)).into_response())
}

/// Build a true-streaming SSE response. Spawns a blocking task that holds the
/// runtime lock for the generation and pushes each token into the channel the SSE
/// body drains. The opening chunk carries the assistant role; each token is its
/// own `chat.completion.chunk`; the final chunk has `finish_reason:"stop"` plus
/// mythd audit provenance, followed by the `[DONE]` sentinel.
fn stream_response(
    st: AppState,
    session_id: Option<SessionId>,
    inf_req: InferenceRequest,
    caps: Capabilities,
    model: String,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> Response {
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;
    type Ev = std::result::Result<Event, std::convert::Infallible>;

    let (tx, rx) = mpsc::channel::<Ev>(64);
    // Bind to the supplied session, or mint one — so the chunk id and the audit
    // session id are consistent.
    let sid = session_id.unwrap_or_else(SessionId::random);
    let id = format!("chatcmpl-{}", sid.0);
    let created = now_unix_secs();

    tokio::task::spawn_blocking(move || {
        let _permit = permit; // held until the stream completes
        // opening role delta
        let _ = tx.blocking_send(Ok(stream_chunk(&id, created, &model, serde_json::json!({ "role": "assistant" }), None)));

        let res = st.inner.runtime.infer_stream(Some(sid.clone()), &inf_req, caps, |piece| {
            // `blocking_send` errors when the receiver is gone (client
            // disconnected) → returning false stops generation, freeing the
            // inference slot + permit instead of decoding into the void.
            tx.blocking_send(Ok(stream_chunk(
                &id, created, &model, serde_json::json!({ "content": piece }), None,
            )))
            .is_ok()
        });
        match res {
            Ok(resp) => {
                // final chunk: stop + provenance, then [DONE]
                let chunk = serde_json::json!({
                    "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                    "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
                    "mythd": {
                        "session_id": resp.session_id,
                        "kernel": resp.kernel,
                        "audit_seq": resp.audit_seq,
                        "audit_signature": resp.audit_signature_hex,
                    }
                });
                let _ = tx.blocking_send(Ok(Event::default().data(chunk.to_string())));
                let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
            }
            Err(e) => {
                tracing::warn!(target: "mythd::api", error = %e, "stream inference failed");
                let err = serde_json::json!({ "error": { "message": "inference unavailable", "type": "server_error" } });
                let _ = tx.blocking_send(Ok(Event::default().data(err.to_string())));
                let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
            }
        }
    });

    Sse::new(ReceiverStream::new(rx)).keep_alive(KeepAlive::default()).into_response()
}

/// Encrypted chat: the request body is an OpenAI chat completion sealed under the
/// session's ML-KEM-derived key (AEAD), and the response is sealed the same way.
/// Requires a prior `/v1/session/init`. App-layer confidentiality end-to-end
/// between client and runtime (request/response only — no streaming here).
async fn secure_chat(
    State(st): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<SecureChatRequest>,
) -> std::result::Result<Json<serde_json::Value>, ApiError> {
    st.authorize(&headers)?;
    let sealed = b64_decode(&req.ciphertext)
        .map_err(|_| ApiError::bad_request("ciphertext must be base64"))?;
    let aad = req.session_id.as_bytes();

    // 1. decrypt the request under the session key (AAD = session id).
    let (plaintext, caps) = {
        let mut sessions = st.inner.sessions.lock().await;
        let ts = sessions.get_mut(&req.session_id).ok_or_else(|| {
            ApiError::new(StatusCode::UNAUTHORIZED, "authentication_error", "unknown or expired session")
        })?;
        let pt = ts
            .key
            .open(&sealed, aad)
            .map_err(|_| ApiError::bad_request("decryption failed (bad key, ciphertext, or aad)"))?;
        ts.last_used = std::time::Instant::now(); // sliding TTL
        (pt, ts.caps)
    };

    // 2. the decrypted payload is a standard OpenAI chat request.
    let chat: ChatCompletionRequest = serde_json::from_slice(&plaintext)
        .map_err(|_| ApiError::bad_request("decrypted body is not a chat request"))?;
    if chat.messages.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty"));
    }
    let max_tokens = chat
        .max_tokens
        .unwrap_or(st.inner.default_max_tokens)
        .min(st.inner.max_tokens_cap)
        .max(1);
    let (messages, has_image) = prepare_messages(&chat.messages)?;
    if has_image && !st.inner.vision {
        return Err(ApiError::bad_request("this model has no vision support (no mmproj loaded)"));
    }
    let inf_req = InferenceRequest {
        messages,
        max_new_tokens: max_tokens,
        sampling: sampling_from(chat.temperature, chat.top_p, chat.seed),
        has_image,
    };

    // admission control (same backpressure as the plaintext path)
    let permit = st.inner.infer_sem.clone().try_acquire_owned().map_err(|_| {
        ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "server_busy",
            "inference queue full, retry shortly",
        )
    })?;
    // 3. run inference, bound to (and audited under) this session. Blocking forward
    // on a blocking thread → concurrent with other requests.
    let st2 = st.clone();
    let sid2 = req.session_id.clone();
    let resp = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        st2.inner.runtime.infer_with_session(Some(SessionId(sid2)), &inf_req, caps)
    })
    .await
    .map_err(|_| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "server_error", "inference task failed"))??;

    let model = if chat.model.is_empty() {
        st.inner.model_id.clone()
    } else {
        chat.model.clone()
    };
    let body = json_completion(&resp, &model);
    let body_bytes = serde_json::to_vec(&body)
        .map_err(|_| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "server_error", "encode failed"))?;

    // 4. seal the response under the same session key.
    let sealed_resp = {
        let sessions = st.inner.sessions.lock().await;
        let ts = sessions.get(&req.session_id).ok_or_else(|| {
            ApiError::new(StatusCode::UNAUTHORIZED, "authentication_error", "session closed mid-request")
        })?;
        ts.key
            .seal(&body_bytes, aad)
            .map_err(|_| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "server_error", "encrypt failed"))?
    };

    Ok(Json(serde_json::json!({ "ciphertext": b64_encode(&sealed_resp) })))
}

async fn audit(
    State(st): State<AppState>,
    headers: HeaderMap,
    Path(session_id): Path<String>,
) -> std::result::Result<Json<serde_json::Value>, ApiError> {
    st.authorize(&headers)?;
    let vk = st.inner.audit_verifying_key.clone(); // cached at startup (stable key)

    // The audit log is lock-guarded; read it on a blocking thread (owned copies).
    let st2 = st.clone();
    let sid = session_id.clone();
    let records = tokio::task::spawn_blocking(move || st2.inner.runtime.audit_records_for(&sid))
        .await
        .map_err(|_| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "server_error", "audit read failed"))?;

    let mut entries = Vec::new();
    for r in records {
        entries.push(serde_json::json!({
            "seq": r.seq,
            "timestamp_ms": r.timestamp_ms,
            "kernel": r.kernel,
            "prompt_tokens": r.prompt_tokens,
            "generated_tokens": r.generated_tokens,
            "prev_hash": r.prev_hash,
            "signature": to_hex(&r.signature),
            // exact signed bytes (base64) so the client can re-verify the signature
            "record_b64": b64_encode(&r.record_json),
        }));
    }

    if entries.is_empty() {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "not_found_error",
            "no audit entries for session",
        ));
    }

    Ok(Json(serde_json::json!({
        "session_id": session_id,
        "signature_scheme": "ML-DSA-65",
        "chain_hash": "SHA-256",
        "audit_verifying_key": b64_encode(&vk),
        "entries": entries,
    })))
}

// ---------------------------------------------------------------------------
// Response builders
// ---------------------------------------------------------------------------

fn json_completion(resp: &mythd_session::SecureResponse, model: &str) -> serde_json::Value {
    serde_json::json!({
        "id": format!("chatcmpl-{}", resp.session_id),
        "object": "chat.completion",
        "created": now_unix_secs(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": resp.text },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.generated_tokens,
            "total_tokens": resp.prompt_tokens + resp.generated_tokens,
        },
        // mythd extension: cryptographic provenance of this completion.
        "mythd": {
            "session_id": resp.session_id,
            "kernel": resp.kernel,
            "audit_seq": resp.audit_seq,
            "audit_signature": resp.audit_signature_hex,
        }
    })
}

/// One OpenAI `chat.completion.chunk` SSE event.
fn stream_chunk(
    id: &str,
    created: u64,
    model: &str,
    delta: serde_json::Value,
    finish: Option<&str>,
) -> Event {
    let chunk = serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
    });
    Event::default().data(chunk.to_string())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map OpenAI sampling fields to mythd's [`Sampling`]. Absent `temperature` ⇒
/// greedy/deterministic (mythd's default); absent `top_p` ⇒ no nucleus cutoff;
/// absent `seed` ⇒ random. Values are clamped to sane ranges.
fn sampling_from(temperature: Option<f32>, top_p: Option<f32>, seed: Option<i64>) -> Sampling {
    Sampling {
        temperature: temperature.unwrap_or(0.0).clamp(0.0, 4.0),
        top_p: top_p.unwrap_or(1.0).clamp(0.0, 1.0),
        seed: seed.unwrap_or(-1),
    }
}

/// Validate + prepare the chat messages for forwarding to llama-server.
///
/// The OpenAI `messages` array is forwarded **verbatim**, so llama-server applies
/// the model's own chat template (`--jinja`) and handles vision via `mtmd` — which
/// retires mythd's hand-rolled ChatML and gets the correct Qwen3-VL template for
/// free. Two rules are enforced here: (1) only inline `data:` image URLs are
/// allowed — mythd never fetches remote URLs (privacy/SSRF); (2) report whether an
/// image is present so the caller can gate on the model's vision capability.
fn prepare_messages(
    messages: &[openai::ChatMessage],
) -> std::result::Result<(serde_json::Value, bool), ApiError> {
    use openai::{Content, ContentPart};
    let mut has_image = false;
    for m in messages {
        if let Content::Parts(parts) = &m.content {
            for part in parts {
                if let ContentPart::ImageUrl { image_url } = part {
                    if !image_url.url.starts_with("data:") {
                        return Err(ApiError::bad_request(
                            "only inline data: image URLs are accepted (mythd does not fetch remote URLs)",
                        ));
                    }
                    has_image = true;
                }
            }
        }
    }
    let value = serde_json::to_value(messages).map_err(|_| {
        ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "server_error", "encode failed")
    })?;
    Ok((value, has_image))
}

/// Key-confirmation tag: SHA-256(session_key ‖ session_id ‖ label). Both sides
/// compute it; equality proves the ML-KEM exchange yielded the same key, without
/// revealing the key.
fn key_confirmation(key: &SessionKey, session_id: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(key.expose());
    h.update(session_id.as_bytes());
    h.update(b"mythd/kem-confirm/v1");
    to_hex(&h.finalize())
}

fn now_unix_secs() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn b64_encode(bytes: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn b64_decode(s: &str) -> std::result::Result<Vec<u8>, base64::DecodeError> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.decode(s.trim())
}

/// SHA-256 of `bytes` as a fixed 32-byte array.
fn sha256_32(bytes: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut out = [0u8; 32];
    out.copy_from_slice(&Sha256::digest(bytes));
    out
}

/// Constant-time equality of two 32-byte arrays: 1 if equal, else 0.
fn ct_eq_32(a: &[u8; 32], b: &[u8; 32]) -> u8 {
    let mut diff = 0u8;
    for i in 0..32 {
        diff |= a[i] ^ b[i];
    }
    (((diff as u32).wrapping_sub(1)) >> 31) as u8
}

fn to_hex(bytes: &[u8]) -> String {
    const H: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(H[(b >> 4) as usize] as char);
        s.push(H[(b & 0x0f) as usize] as char);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_forward_in_order_no_image() {
        use openai::Content;
        let msgs = vec![
            openai::ChatMessage { role: "system".into(), content: Content::Text("Be terse.".into()) },
            openai::ChatMessage { role: "user".into(), content: Content::Text("Hi".into()) },
        ];
        let (value, has_image) = prepare_messages(&msgs).unwrap();
        assert!(!has_image);
        let arr = value.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["role"], "system");
        assert_eq!(arr[0]["content"], "Be terse.");
        assert_eq!(arr[1]["content"], "Hi");
    }

    #[test]
    fn inline_image_ok_remote_rejected() {
        use openai::{Content, ContentPart, ImageUrl};
        let png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
        let ok = vec![openai::ChatMessage {
            role: "user".into(),
            content: Content::Parts(vec![
                ContentPart::Text { text: "what is this?".into() },
                ContentPart::ImageUrl { image_url: ImageUrl { url: png.into() } },
            ]),
        }];
        let (value, has_image) = prepare_messages(&ok).unwrap();
        assert!(has_image);
        let parts = value[0]["content"].as_array().unwrap();
        assert_eq!(parts[1]["type"], "image_url");
        assert!(parts[1]["image_url"]["url"].as_str().unwrap().starts_with("data:"));

        // a remote URL must be rejected (no SSRF / no remote fetch)
        let remote = vec![openai::ChatMessage {
            role: "user".into(),
            content: Content::Parts(vec![ContentPart::ImageUrl {
                image_url: ImageUrl { url: "https://example.com/x.png".into() },
            }]),
        }];
        assert!(prepare_messages(&remote).is_err());
    }

    #[test]
    fn base64_roundtrip() {
        let data = b"\x00\x01\xfe\xff mythd";
        assert_eq!(b64_decode(&b64_encode(data)).unwrap(), data);
    }

    #[test]
    fn key_confirmation_matches_for_same_inputs() {
        // The two sides derive identical session keys; the tag must agree.
        let server = mythd_core::KemKeypair::generate();
        let pk = server.public_key_bytes();
        let (ct, client_key) = mythd_core::encapsulate_to(&pk).unwrap();
        let server_key = server.decapsulate(&ct).unwrap();
        let sid = "deadbeef";
        assert_eq!(
            key_confirmation(&client_key, sid),
            key_confirmation(&server_key, sid)
        );
    }

    #[test]
    fn authorize_enforces_api_keys() {
        use axum::http::header::AUTHORIZATION;
        let st = AppState::new(
            mythd_session::SecureRuntime::new(),
            "m",
            8,
            64,
            vec!["sk-secret-123".to_string()],
            16,
        );
        assert!(st.auth_enabled());

        let mut good = HeaderMap::new();
        good.insert(AUTHORIZATION, "Bearer sk-secret-123".parse().unwrap());
        assert!(st.authorize(&good).is_ok());

        let mut wrong = HeaderMap::new();
        wrong.insert(AUTHORIZATION, "Bearer nope".parse().unwrap());
        assert!(st.authorize(&wrong).is_err());
        assert!(st.authorize(&HeaderMap::new()).is_err()); // missing token

        // open mode: no keys configured → every request allowed
        let open = AppState::new(mythd_session::SecureRuntime::new(), "m", 8, 64, vec![], 16);
        assert!(!open.auth_enabled());
        assert!(open.authorize(&HeaderMap::new()).is_ok());
    }
}
