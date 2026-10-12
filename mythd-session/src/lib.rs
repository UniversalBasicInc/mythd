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
    // Inference forwards over HTTP (`&self`), so the orchestrator needs no lock.
    orch: Orchestrator,
    // Brief locks only — never held across an inference forward. The audit chain
    // is sequential, so its append is the one serialized step.
    sessions: std::sync::Mutex<SessionManager>,
    audit: std::sync::Mutex<AuditLog>,
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

/// An owned, presentation-ready view of one signed audit entry (the live log is
/// lock-guarded, so reads hand back owned copies rather than borrows).
#[derive(Debug, Clone)]
pub struct AuditEntryView {
    pub seq: u64,
    pub timestamp_ms: u64,
    pub kernel: String,
    pub prompt_tokens: u32,
    pub generated_tokens: u32,
    pub prev_hash: String,
    pub signature: Vec<u8>,
    pub record_json: Vec<u8>,
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
            sessions: std::sync::Mutex::new(SessionManager::new()),
            audit: std::sync::Mutex::new(AuditLog::new(signer)),
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
            sessions: std::sync::Mutex::new(SessionManager::new()),
            audit: std::sync::Mutex::new(AuditLog::open(signer, log_path)?),
        })
    }

    /// Point the runtime's orchestrator at a llama-server instance (loopback HTTP).
    /// `vision` declares whether that server has a multimodal projector loaded.
    pub fn connect_gateway(&mut self, url: &str, vision: bool) -> Result<()> {
        self.orch.connect_gateway(url, vision)?;
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
        self.audit.lock().unwrap().verifying_key()
    }

    /// Owned snapshot of every signed audit entry, in chain order — the input
    /// [`verify_chain`] expects. The live log is lock-guarded, so this copies.
    pub fn audit_entries(&self) -> Vec<SignedEntry> {
        self.audit.lock().unwrap().entries().to_vec()
    }

    /// Number of audit entries recorded (brief lock).
    pub fn audit_entry_count(&self) -> usize {
        self.audit.lock().unwrap().len()
    }

    /// Owned, filtered view of the signed audit entries for one session — for the
    /// API's `/v1/audit/{id}` endpoint. The live log is lock-guarded, so this hands
    /// back owned copies rather than a borrow.
    pub fn audit_records_for(&self, session_id: &str) -> Vec<AuditEntryView> {
        let audit = self.audit.lock().unwrap();
        let mut out = Vec::new();
        for e in audit.entries() {
            if let Ok(rec) = serde_json::from_slice::<AuditRecord>(&e.record_json) {
                if rec.session_id == session_id {
                    out.push(AuditEntryView {
                        seq: rec.seq,
                        timestamp_ms: rec.timestamp_ms,
                        kernel: rec.kernel,
                        prompt_tokens: rec.prompt_tokens,
                        generated_tokens: rec.generated_tokens,
                        prev_hash: rec.prev_hash,
                        signature: e.signature.clone(),
                        record_json: e.record_json.clone(),
                    });
                }
            }
        }
        out
    }

    /// Detected XPU capabilities (for `/health`); `None` on a non-Intel host.
    pub fn device_caps(&self) -> Option<&mythd_inference::DeviceCaps> {
        self.orch.device_caps()
    }

    /// Run one inference through the complete perimeter, minting a fresh random
    /// session id: open session → infer → append signed audit entry → close.
    pub fn infer(
        &self,
        req: &InferenceRequest,
        caps: Capabilities,
    ) -> Result<SecureResponse> {
        self.infer_with_session(None, req, caps)
    }

    /// As [`Self::infer`], but binds the work (and its audit record) to a
    /// caller-supplied session id when given — e.g. one established by the ML-KEM
    /// handshake at Layer 5. `None` mints a fresh random id.
    pub fn infer_with_session(
        &self,
        session_id: Option<SessionId>,
        req: &InferenceRequest,
        caps: Capabilities,
    ) -> Result<SecureResponse> {
        // Deny-by-default capability checks (formerly inside Session::infer).
        if !caps.may_infer {
            return Err(SessionError::Denied("inference not permitted"));
        }
        if req.max_new_tokens > caps.max_new_tokens {
            return Err(SessionError::Denied("token budget exceeded"));
        }
        // Open a session (supplied id, or a fresh random one). Brief lock.
        let sid = {
            let mut s = self.sessions.lock().unwrap();
            match session_id {
                Some(id) => s.create_with_id(id, caps),
                None => s.create(caps),
            }
        };
        // Inference forwards over HTTP holding NO lock → concurrent across requests.
        let response = match self.orch.infer(req) {
            Ok(r) => r,
            Err(e) => {
                self.sessions.lock().unwrap().close(&sid);
                return Err(e.into());
            }
        };
        Ok(self.record_and_close(sid, response))
    }

    /// Streaming variant of [`Self::infer_with_session`]: each token is delivered
    /// to `on_token` as it is produced, then the full perimeter completes —
    /// per-session KV accounting + a signed, chained audit entry over the assembled
    /// response — and the session is closed. The returned [`SecureResponse`]
    /// carries the same audit provenance as the non-streaming path.
    pub fn infer_stream(
        &self,
        session_id: Option<SessionId>,
        req: &InferenceRequest,
        caps: Capabilities,
        on_token: impl FnMut(&str) -> bool,
    ) -> Result<SecureResponse> {
        if !caps.may_infer {
            return Err(SessionError::Denied("inference not permitted"));
        }
        if req.max_new_tokens > caps.max_new_tokens {
            return Err(SessionError::Denied("token budget exceeded"));
        }
        let sid = {
            let mut s = self.sessions.lock().unwrap();
            match session_id {
                Some(id) => s.create_with_id(id, caps),
                None => s.create(caps),
            }
        };
        // Streaming forward holds NO session/audit lock → concurrent.
        let response = match self.orch.infer_stream(req, on_token) {
            Ok(r) => r,
            Err(e) => {
                self.sessions.lock().unwrap().close(&sid);
                return Err(e.into());
            }
        };
        Ok(self.record_and_close(sid, response))
    }

    /// Append a signed, chained audit entry for `response`, then close the session.
    /// The audit append is the only serialized step (the chain is sequential); it
    /// is never held across an inference forward.
    fn record_and_close(&self, sid: SessionId, response: InferenceResponse) -> SecureResponse {
        let kernel = response.backend.clone();
        let session_id = sid.0.clone();
        let (seq, sig_hex, record_json) = {
            let mut a = self.audit.lock().unwrap();
            let seq = a.append(
                &sid,
                now_unix_ms(),
                &kernel,
                response.prompt_tokens,
                response.generated_tokens,
            );
            let entry = a.entries().last().expect("just appended");
            (seq, hex(&entry.signature), entry.record_json.clone())
        };
        self.sessions.lock().unwrap().close(&sid);
        tracing::info!(
            target: "mythd::session",
            session = %session_id, seq, kernel = %kernel,
            prompt_tokens = response.prompt_tokens, generated_tokens = response.generated_tokens,
            "inference recorded + signed"
        );
        SecureResponse {
            session_id,
            text: response.text,
            prompt_tokens: response.prompt_tokens,
            generated_tokens: response.generated_tokens,
            kernel,
            audit_seq: seq,
            audit_signature_hex: sig_hex,
            audit_record_json: record_json,
        }
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
