//! # mythd-session — Layer 4: session manager + tamper-evident audit
//!
//! Three security guarantees live here, all enforced by construction:
//!
//! 1. **Cryptographically-random session identity.** [`SessionManager::create`]
//!    mints each [`SessionId`] from 128 bits of OS CSPRNG entropy
//!    ([`mythd_core::fill_random`]); ids are unguessable and effectively unique.
//!
//! 2. **Compile-time KV-cache isolation.** Each [`Session`] *owns* its
//!    [`KvCache`] as a private field. There is no global/shared KV pool; the only
//!    way to reach a session's cache is through an exclusive `&mut Session`, which
//!    the manager hands out for one id at a time. Rust ownership makes
//!    cross-session KV access *unrepresentable* — a type error, not a runtime
//!    check. (`KvCache` is not `Clone`, so it cannot be copied into another
//!    session either.)
//!
//! 3. **Tamper-evident audit log.** Every inference appends an
//!    [`audit::SignedEntry`]: an ML-DSA-65 (post-quantum) signature over a record
//!    binding `session_id + timestamp + kernel + token counts + previous entry
//!    hash`. The SHA-256 hash chain makes the log append-only — editing,
//!    reordering, or deleting any entry breaks every signature/link after it, with
//!    no signing key required to *detect* the tamper.
//!
//! [`SecureRuntime`] wires it together: every inference opens a session, runs,
//! appends a signed audit entry, and closes the session cleanly.

#![forbid(unsafe_code)]

use mythd_inference::{InferenceRequest, InferenceResponse, Orchestrator};
use thiserror::Error;

pub use mythd_inference::{InferenceRequest as Request, Precision};

#[derive(Debug, Error)]
pub enum SessionError {
    #[error("inference error: {0}")]
    Inference(#[from] mythd_inference::InferenceError),
    #[error("unknown session id")]
    UnknownSession,
    #[error("capability denied: {0}")]
    Denied(&'static str),
}

pub type Result<T> = std::result::Result<T, SessionError>;

/// Opaque, cryptographically-random session identifier (32 hex chars = 128 bits).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SessionId(pub String);

impl SessionId {
    /// Mint a fresh id from 128 bits of OS CSPRNG entropy.
    pub fn random() -> Self {
        let mut bytes = [0u8; 16];
        mythd_core::fill_random(&mut bytes);
        SessionId(hex(&bytes))
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Per-session KV cache. Private contents; constructed and owned exclusively by
/// its [`Session`]. Not `Clone`, so it cannot be duplicated into another session.
///
/// The physical attention KV tensors live inside the inference engine's context;
/// for the MVP that context is accessed only while this session holds the
/// exclusive `&mut Orchestrator`, so sessions are serialized and never share live
/// KV. This struct is the *type-level* ownership token for that allocation — the
/// guarantee mythd enforces structurally. (Production step: bind each session to a
/// distinct llama.cpp `seq_id` for physical partitioning within one context.)
pub struct KvCache {
    capacity_tokens: u32,
    used_tokens: u32,
}

impl KvCache {
    fn new(capacity_tokens: u32) -> Self {
        Self { capacity_tokens, used_tokens: 0 }
    }

    /// Account tokens against this session's cache (saturating at capacity).
    fn account(&mut self, tokens: u32) {
        self.used_tokens = self.used_tokens.saturating_add(tokens).min(self.capacity_tokens);
    }

    pub fn used_tokens(&self) -> u32 {
        self.used_tokens
    }
    pub fn capacity_tokens(&self) -> u32 {
        self.capacity_tokens
    }
}

/// Capabilities a session is permitted (deny-by-default capability model).
#[derive(Debug, Clone, Copy)]
pub struct Capabilities {
    pub may_infer: bool,
    pub max_new_tokens: u32,
    pub kv_capacity_tokens: u32,
}

impl Default for Capabilities {
    fn default() -> Self {
        // Deny by default.
        Self { may_infer: false, max_new_tokens: 0, kv_capacity_tokens: 0 }
    }
}

impl Capabilities {
    /// Grant inference up to `max_new_tokens`, with a KV budget of `4096 + cap`.
    pub fn allow_infer(max_new_tokens: u32) -> Self {
        Self {
            may_infer: true,
            max_new_tokens,
            kv_capacity_tokens: max_new_tokens.saturating_add(4096),
        }
    }
}

/// A single client session. Owns its KV cache; isolation is enforced by this
/// ownership (no `Clone`, cache is private, only reachable via `&mut self`).
pub struct Session {
    id: SessionId,
    caps: Capabilities,
    kv: KvCache,
}

impl Session {
    fn new(id: SessionId, caps: Capabilities) -> Self {
        let kv = KvCache::new(caps.kv_capacity_tokens);
        Self { id, caps, kv }
    }

    pub fn id(&self) -> &SessionId {
        &self.id
    }

    pub fn kv(&self) -> &KvCache {
        &self.kv
    }

    /// Run inference for *this* session against *this* session's KV cache. The
    /// `&mut self` receiver is the isolation boundary: the borrow checker
    /// guarantees no other session can be touching `self.kv` concurrently.
    pub fn infer(
        &mut self,
        orch: &mut Orchestrator,
        req: &InferenceRequest,
    ) -> Result<InferenceResponse> {
        // Deny-by-default capability checks.
        if !self.caps.may_infer {
            return Err(SessionError::Denied("inference not permitted"));
        }
        if req.max_new_tokens > self.caps.max_new_tokens {
            return Err(SessionError::Denied("token budget exceeded"));
        }

        let resp = orch.infer(req)?;
        // Charge this session's own KV cache (the per-session allocation).
        self.kv.account(resp.prompt_tokens.saturating_add(resp.generated_tokens));
        Ok(resp)
    }

    /// As [`Self::infer`], but streams each token to `on_token` as it is produced
    /// (`on_token` returns `false` to stop early). Same deny-by-default capability
    /// checks and per-session KV accounting.
    pub fn infer_stream(
        &mut self,
        orch: &mut Orchestrator,
        req: &InferenceRequest,
        on_token: impl FnMut(&str) -> bool,
    ) -> Result<InferenceResponse> {
        if !self.caps.may_infer {
            return Err(SessionError::Denied("inference not permitted"));
        }
        if req.max_new_tokens > self.caps.max_new_tokens {
            return Err(SessionError::Denied("token budget exceeded"));
        }
        let resp = orch.infer_stream(req, on_token)?;
        self.kv.account(resp.prompt_tokens.saturating_add(resp.generated_tokens));
        Ok(resp)
    }
}

/// Owns all sessions and lends exclusive access by id.
pub struct SessionManager {
    sessions: std::collections::HashMap<SessionId, Session>,
}

impl SessionManager {
    pub fn new() -> Self {
        Self { sessions: std::collections::HashMap::new() }
    }

    /// Create a new session with a fresh random id and the given capabilities.
    /// Returns the id; the `Session` itself is owned by the manager.
    pub fn create(&mut self, caps: Capabilities) -> SessionId {
        let id = SessionId::random();
        self.create_with_id(id, caps)
    }

    /// Create a session under a caller-supplied id (e.g. one established by the
    /// ML-KEM handshake at Layer 5). Returns the id.
    pub fn create_with_id(&mut self, id: SessionId, caps: Capabilities) -> SessionId {
        self.sessions.insert(id.clone(), Session::new(id.clone(), caps));
        id
    }

    /// Exclusive borrow of one session. Two different sessions can only ever be
    /// `&mut`-borrowed one at a time through this manager, so their KV caches are
    /// never simultaneously reachable.
    pub fn get_mut(&mut self, id: &SessionId) -> Result<&mut Session> {
        self.sessions.get_mut(id).ok_or(SessionError::UnknownSession)
    }

    /// Close a session, dropping it and its `KvCache` (the allocation is freed).
    pub fn close(&mut self, id: &SessionId) {
        self.sessions.remove(id);
    }

    pub fn len(&self) -> usize {
        self.sessions.len()
    }
    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

pub mod audit;
pub use audit::{verify_chain, AuditLog, AuditOpenError, AuditRecord, AuditVerifyError, SignedEntry};

/// The full secure inference path. Owns the orchestrator (Layer 3), the session
/// manager, and the audit log — the complete MVP perimeter.
pub struct SecureRuntime {
    orch: Orchestrator,
    sessions: SessionManager,
    audit: AuditLog,
}

/// Result of a [`SecureRuntime::infer`] call: the generated text plus the proof
/// it was recorded — a signed, chained audit entry.
#[derive(Debug, Clone)]
pub struct SecureResponse {
    pub session_id: String,
    pub text: String,
    pub prompt_tokens: u32,
    pub generated_tokens: u32,
    /// The kernel that handled this inference (e.g. `"BitNetInt2"` on the B70).
    pub kernel: String,
    pub audit_seq: u64,
    /// hex of the ML-DSA-65 signature over this entry's record.
    pub audit_signature_hex: String,
    /// the exact canonical record bytes that were signed.
    pub audit_record_json: Vec<u8>,
}

impl SecureRuntime {
    /// Build a runtime with a fresh (ephemeral) audit signing key. Fine for tests
    /// and the standalone demo; in production prefer
    /// [`with_audit_signer`](Self::with_audit_signer) + a persisted key so the
    /// audit verifying key survives restarts.
    pub fn new() -> Self {
        Self::with_audit_signer(mythd_core::AuditSigner::generate())
    }

    /// Build a runtime with a specific audit signing key — e.g. one loaded by
    /// [`mythd_core::load_or_create_audit_signer`] so the audit log's verifying
    /// key is stable across restarts. The audit log is in-memory only.
    pub fn with_audit_signer(signer: mythd_core::AuditSigner) -> Self {
        Self {
            orch: Orchestrator::new(),
            sessions: SessionManager::new(),
            audit: AuditLog::new(signer),
        }
    }

    /// As [`with_audit_signer`](Self::with_audit_signer), but persists the signed,
    /// hash-chained audit log to `log_path` (loaded + integrity-verified on
    /// startup, appended + fsync'd on each entry). Errors if the existing on-disk
    /// log fails verification — a tamper/corruption signal that should stop boot.
    pub fn with_persistent_audit(
        signer: mythd_core::AuditSigner,
        log_path: &std::path::Path,
    ) -> std::result::Result<Self, AuditOpenError> {
        Ok(Self {
            orch: Orchestrator::new(),
            sessions: SessionManager::new(),
            audit: AuditLog::open(signer, log_path)?,
        })
    }

    /// Load the model into the orchestrator (Layer 3), optionally with a vision
    /// projector (`mmproj`) so it accepts images.
    pub fn load_model(
        &mut self,
        path: impl AsRef<std::path::Path>,
        mmproj: Option<impl AsRef<std::path::Path>>,
        weight_precision: Precision,
    ) -> Result<()> {
        self.orch.load_model(path, mmproj, weight_precision)?;
        Ok(())
    }

    /// Whether the loaded model can accept images.
    pub fn supports_vision(&self) -> bool {
        self.orch.supports_vision()
    }

    /// Set the context window (tokens) before [`load_model`](Self::load_model).
    pub fn set_context_size(&mut self, n_ctx: u32) {
        self.orch.set_context_size(n_ctx);
    }

    /// Store the KV cache as q8_0 (INT8) — call before [`load_model`](Self::load_model).
    pub fn set_kv_q8(&mut self, on: bool) {
        self.orch.set_kv_q8(on);
    }

    /// Whether the KV cache is stored as q8_0 (INT8).
    pub fn kv_q8(&self) -> bool {
        self.orch.kv_q8()
    }

    /// The configured context window (tokens).
    pub fn context_size(&self) -> u32 {
        self.orch.context_size()
    }

    /// The audit log's public verifying key (publishable; used to verify the log).
    pub fn audit_verifying_key(&self) -> Vec<u8> {
        self.audit.verifying_key()
    }

    /// The full signed, chained audit log.
    pub fn audit_entries(&self) -> &[SignedEntry] {
        self.audit.entries()
    }

    /// Detected XPU capabilities (for `/health`); `None` on a non-Intel host.
    pub fn device_caps(&self) -> Option<&mythd_inference::DeviceCaps> {
        self.orch.device_caps()
    }

    /// Run one inference through the complete perimeter, minting a fresh random
    /// session id: open session → infer → append signed audit entry → close.
    pub fn infer(
        &mut self,
        req: &InferenceRequest,
        caps: Capabilities,
    ) -> Result<SecureResponse> {
        self.infer_with_session(None, req, caps)
    }

    /// As [`Self::infer`], but binds the work (and its audit record) to a
    /// caller-supplied session id when given — e.g. one established by the ML-KEM
    /// handshake at Layer 5. `None` mints a fresh random id.
    pub fn infer_with_session(
        &mut self,
        session_id: Option<SessionId>,
        req: &InferenceRequest,
        caps: Capabilities,
    ) -> Result<SecureResponse> {
        // 1. open a session (supplied id, or a fresh random one)
        let sid = match session_id {
            Some(id) => self.sessions.create_with_id(id, caps),
            None => self.sessions.create(caps),
        };

        // 2. run inference under this session's exclusive borrow
        let run = {
            let session = self.sessions.get_mut(&sid).expect("just created");
            session.infer(&mut self.orch, req)
        };
        let response = match run {
            Ok(r) => r,
            Err(e) => {
                self.sessions.close(&sid); // still close cleanly on failure
                return Err(e);
            }
        };

        // 3. append a signed, chained audit entry
        let kernel = response.backend.clone();
        let seq = self.audit.append(
            &sid,
            now_unix_ms(),
            &kernel,
            response.prompt_tokens,
            response.generated_tokens,
        );
        let entry = self.audit.entries().last().expect("just appended");
        let out = SecureResponse {
            session_id: sid.0.clone(),
            text: response.text.clone(),
            prompt_tokens: response.prompt_tokens,
            generated_tokens: response.generated_tokens,
            kernel,
            audit_seq: seq,
            audit_signature_hex: hex(&entry.signature),
            audit_record_json: entry.record_json.clone(),
        };

        // 4. close the session cleanly (drops the Session + its KvCache)
        self.sessions.close(&sid);
        tracing::info!(
            target: "mythd::session",
            session = %out.session_id, seq = out.audit_seq, kernel = %out.kernel,
            prompt_tokens = out.prompt_tokens, generated_tokens = out.generated_tokens,
            "inference recorded + signed"
        );
        Ok(out)
    }

    /// Streaming variant of [`Self::infer_with_session`]: each token is delivered
    /// to `on_token` as it is produced, then the full perimeter completes —
    /// per-session KV accounting + a signed, chained audit entry over the assembled
    /// response — and the session is closed. The returned [`SecureResponse`]
    /// carries the same audit provenance as the non-streaming path.
    pub fn infer_stream(
        &mut self,
        session_id: Option<SessionId>,
        req: &InferenceRequest,
        caps: Capabilities,
        on_token: impl FnMut(&str) -> bool,
    ) -> Result<SecureResponse> {
        let sid = match session_id {
            Some(id) => self.sessions.create_with_id(id, caps),
            None => self.sessions.create(caps),
        };

        let run = {
            let session = self.sessions.get_mut(&sid).expect("just created");
            session.infer_stream(&mut self.orch, req, on_token)
        };
        let response = match run {
            Ok(r) => r,
            Err(e) => {
                self.sessions.close(&sid);
                return Err(e);
            }
        };

        let kernel = response.backend.clone();
        let seq = self.audit.append(
            &sid,
            now_unix_ms(),
            &kernel,
            response.prompt_tokens,
            response.generated_tokens,
        );
        let entry = self.audit.entries().last().expect("just appended");
        let out = SecureResponse {
            session_id: sid.0.clone(),
            text: response.text.clone(),
            prompt_tokens: response.prompt_tokens,
            generated_tokens: response.generated_tokens,
            kernel,
            audit_seq: seq,
            audit_signature_hex: hex(&entry.signature),
            audit_record_json: entry.record_json.clone(),
        };

        self.sessions.close(&sid);
        tracing::info!(
            target: "mythd::session",
            session = %out.session_id, seq = out.audit_seq, kernel = %out.kernel,
            prompt_tokens = out.prompt_tokens, generated_tokens = out.generated_tokens,
            "streamed inference recorded + signed"
        );
        Ok(out)
    }
}

impl Default for SecureRuntime {
    fn default() -> Self {
        Self::new()
    }
}

/// Milliseconds since the Unix epoch (0 if the clock is before the epoch).
fn now_unix_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Lowercase hex encoding (no external dep, keeps the trust surface small).
pub(crate) fn hex(bytes: &[u8]) -> String {
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
    fn session_ids_are_random_and_distinct() {
        let mut mgr = SessionManager::new();
        let a = mgr.create(Capabilities::allow_infer(8));
        let b = mgr.create(Capabilities::allow_infer(8));
        assert_ne!(a, b, "ids must differ");
        assert_eq!(a.0.len(), 32, "128-bit id = 32 hex chars");
        assert!(a.0.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(mgr.len(), 2);
        mgr.close(&a);
        mgr.close(&b);
        assert!(mgr.is_empty());
    }

    #[test]
    fn deny_by_default_capabilities() {
        let caps = Capabilities::default();
        assert!(!caps.may_infer);
        assert_eq!(caps.max_new_tokens, 0);
    }
}
