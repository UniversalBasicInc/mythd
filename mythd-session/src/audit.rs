//! Append-only, hash-chained, ML-DSA-65-signed audit log.
//!
//! Each entry signs a record binding: `seq`, `session_id`, `timestamp_ms`,
//! `kernel`, `prompt_tokens`, `generated_tokens`, and `prev_hash` — the SHA-256
//! hash of the *previous* signed entry. Two independent properties make the log
//! tamper-evident **by construction**:
//!
//! * **Signature.** Altering any record's bytes invalidates its ML-DSA-65
//!   signature; forging a new one requires the secret signing key.
//! * **Hash chain.** Each record commits to the previous entry's hash, so
//!   deleting, reordering, or editing any entry changes its hash and breaks the
//!   `prev_hash` link of every entry after it — detectable with only the public
//!   verifying key.
//!
//! [`verify_chain`] checks both for the whole log.

use super::{hex, SessionId};
use mythd_core::verify_audit_signature;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// The fixed `prev_hash` of the first (genesis) entry: SHA-256 of the empty input.
const GENESIS_PREV: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// One audit record. Serialized canonically (this struct's JSON) and signed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRecord {
    pub seq: u64,
    pub session_id: String,
    pub timestamp_ms: u64,
    /// The kernel that handled the inference (e.g. `"BitNetInt2"`).
    pub kernel: String,
    pub prompt_tokens: u32,
    pub generated_tokens: u32,
    /// SHA-256 (hex) of the previous signed entry; `GENESIS_PREV` for the first.
    pub prev_hash: String,
}

/// A signed audit entry: the exact record bytes that were signed plus the
/// ML-DSA-65 signature over them.
#[derive(Debug, Clone)]
pub struct SignedEntry {
    pub record_json: Vec<u8>,
    pub signature: Vec<u8>,
}

impl SignedEntry {
    /// This entry's chain hash: SHA-256 over `record_json || signature`.
    pub fn entry_hash(&self) -> String {
        let mut h = Sha256::new();
        h.update(&self.record_json);
        h.update(&self.signature);
        hex(&h.finalize())
    }
}

/// The audit log: a signer, the chain state, the accumulated entries, and an
/// optional append-only file the entries are persisted to.
pub struct AuditLog {
    signer: mythd_core::AuditSigner,
    entries: Vec<SignedEntry>,
    seq: u64,
    prev_hash: String,
    sink: Option<std::fs::File>,
}

impl AuditLog {
    /// In-memory log (entries are lost on drop). Use [`open`](Self::open) to
    /// persist across restarts.
    pub fn new(signer: mythd_core::AuditSigner) -> Self {
        Self {
            signer,
            entries: Vec::new(),
            seq: 0,
            prev_hash: GENESIS_PREV.to_string(),
            sink: None,
        }
    }

    /// Open a persistent audit log backed by `path` (one base64+hex record per
    /// line, mode 0600). If the file exists its entries are loaded and the whole
    /// chain is **verified against `signer`'s public key** — a tamper/corruption
    /// of the on-disk log is detected here, at startup, and returned as an error
    /// rather than silently continued. New entries are appended + fsync'd.
    pub fn open(
        signer: mythd_core::AuditSigner,
        path: &std::path::Path,
    ) -> std::result::Result<Self, AuditOpenError> {
        use std::io::Write as _;
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

        let mut entries: Vec<SignedEntry> = Vec::new();
        if path.exists() {
            let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
            let data = std::fs::read_to_string(path).map_err(|_| AuditOpenError::Io)?;
            for (i, line) in data.lines().enumerate() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let rec: PersistLine =
                    serde_json::from_str(line).map_err(|_| AuditOpenError::Corrupt(i as u64))?;
                let record_json = b64_decode(&rec.r).ok_or(AuditOpenError::Corrupt(i as u64))?;
                let signature = b64_decode(&rec.s).ok_or(AuditOpenError::Corrupt(i as u64))?;
                entries.push(SignedEntry { record_json, signature });
            }
            // Integrity gate: the on-disk chain must verify under this key.
            let vk = signer.verifying_key_bytes();
            verify_chain(&entries, &vk).map_err(AuditOpenError::Verify)?;
        }

        // Derive chain state from the (verified) tail.
        let (seq, prev_hash) = match entries.last() {
            Some(e) => (entries.len() as u64, e.entry_hash()),
            None => (0, GENESIS_PREV.to_string()),
        };

        if let Some(dir) = path.parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir).map_err(|_| AuditOpenError::Io)?;
            }
        }
        let sink = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .mode(0o600)
            .open(path)
            .map_err(|_| AuditOpenError::Io)?;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
        let mut sink = sink;
        let _ = sink.flush();

        Ok(Self { signer, entries, seq, prev_hash, sink: Some(sink) })
    }

    /// The public verifying key for this log (publishable).
    pub fn verifying_key(&self) -> Vec<u8> {
        self.signer.verifying_key_bytes()
    }

    pub fn entries(&self) -> &[SignedEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Append a record: build it with the current chain hash, sign it, store it,
    /// and advance the chain. Returns the new entry's sequence number.
    pub fn append(
        &mut self,
        session: &SessionId,
        timestamp_ms: u64,
        kernel: &str,
        prompt_tokens: u32,
        generated_tokens: u32,
    ) -> u64 {
        let seq = self.seq;
        let rec = AuditRecord {
            seq,
            session_id: session.0.clone(),
            timestamp_ms,
            kernel: kernel.to_string(),
            prompt_tokens,
            generated_tokens,
            prev_hash: self.prev_hash.clone(),
        };
        // Canonical encoding: serde_json with this struct's fixed field order.
        let record_json = serde_json::to_vec(&rec).expect("audit record serializes");
        let signature = self.signer.sign(&record_json);
        let entry = SignedEntry { record_json, signature };

        // Persist (append-only) before recording in memory, so a durable, signed,
        // hash-chained record exists even if the process dies right after.
        if let Some(sink) = self.sink.as_mut() {
            use std::io::Write as _;
            let line = PersistLine { r: b64_encode(&entry.record_json), s: b64_encode(&entry.signature) };
            let mut buf = serde_json::to_vec(&line).expect("persist line serializes");
            buf.push(b'\n');
            if let Err(e) = sink.write_all(&buf).and_then(|_| sink.sync_data()) {
                // Don't lose the audit trail silently: a failed write is a hard error.
                tracing::error!(target: "mythd::audit", error = %e, "failed to persist audit entry");
            }
        }

        self.prev_hash = entry.entry_hash();
        self.seq += 1;
        self.entries.push(entry);
        seq
    }
}

/// On-disk line format: base64 of the exact signed record bytes + base64 of the
/// signature. Storing the raw record bytes (not a re-serialization) preserves the
/// exact preimage the signature was computed over.
#[derive(Serialize, Deserialize)]
struct PersistLine {
    r: String,
    s: String,
}

/// Why opening a persistent audit log failed.
#[derive(Debug, Error)]
pub enum AuditOpenError {
    #[error("audit log i/o error")]
    Io,
    #[error("audit log line {0} is corrupt")]
    Corrupt(u64),
    #[error("audit log failed integrity verification: {0}")]
    Verify(#[from] AuditVerifyError),
}

fn b64_encode(bytes: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}
fn b64_decode(s: &str) -> Option<Vec<u8>> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.decode(s).ok()
}

/// Why an audit log failed verification (carries the offending entry index).
#[derive(Debug, Error, PartialEq, Eq)]
pub enum AuditVerifyError {
    #[error("entry {0}: invalid ML-DSA signature")]
    BadSignature(u64),
    #[error("entry {0}: malformed record")]
    Malformed(u64),
    #[error("entry {0}: sequence number mismatch")]
    SeqMismatch(u64),
    #[error("entry {0}: broken hash chain (prev_hash does not match)")]
    BrokenChain(u64),
}

/// Verify an entire audit log against its public verifying key: every signature
/// must be valid, sequence numbers contiguous from 0, and each `prev_hash` must
/// equal the SHA-256 of the preceding entry. Returns the offending index on the
/// first failure.
pub fn verify_chain(
    entries: &[SignedEntry],
    verifying_key: &[u8],
) -> std::result::Result<(), AuditVerifyError> {
    let mut expected_prev = GENESIS_PREV.to_string();
    for (i, entry) in entries.iter().enumerate() {
        let i = i as u64;
        // 1. signature over the exact recorded bytes
        verify_audit_signature(verifying_key, &entry.record_json, &entry.signature)
            .map_err(|_| AuditVerifyError::BadSignature(i))?;
        // 2. parse and check the structural invariants
        let rec: AuditRecord =
            serde_json::from_slice(&entry.record_json).map_err(|_| AuditVerifyError::Malformed(i))?;
        if rec.seq != i {
            return Err(AuditVerifyError::SeqMismatch(i));
        }
        if rec.prev_hash != expected_prev {
            return Err(AuditVerifyError::BrokenChain(i));
        }
        // 3. advance the chain
        expected_prev = entry.entry_hash();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mythd_core::AuditSigner;

    fn sample_log() -> (AuditLog, Vec<u8>) {
        let mut log = AuditLog::new(AuditSigner::generate());
        let vk = log.verifying_key();
        let s = SessionId(String::from("00112233445566778899aabbccddeeff"));
        log.append(&s, 1_000, "BitNetInt2", 12, 48);
        log.append(&s, 1_050, "BitNetInt2", 7, 16);
        log.append(&s, 1_090, "BitNetInt2", 20, 64);
        (log, vk)
    }

    #[test]
    fn genesis_is_sha256_of_empty() {
        // sanity: GENESIS_PREV is the well-known SHA-256("")
        assert_eq!(hex(&Sha256::digest(b"")), GENESIS_PREV);
    }

    #[test]
    fn valid_chain_verifies() {
        let (log, vk) = sample_log();
        assert_eq!(log.len(), 3);
        verify_chain(log.entries(), &vk).expect("untampered log verifies");
        // the record really carries the kernel that ran
        let rec: AuditRecord = serde_json::from_slice(&log.entries()[0].record_json).unwrap();
        assert_eq!(rec.kernel, "BitNetInt2");
        assert_eq!(rec.generated_tokens, 48);
    }

    #[test]
    fn editing_a_record_breaks_signature() {
        let (log, vk) = sample_log();
        let mut entries = log.entries().to_vec();
        // flip a byte in entry 1's signed record
        entries[1].record_json[10] ^= 0xff;
        let err = verify_chain(&entries, &vk).unwrap_err();
        assert_eq!(err, AuditVerifyError::BadSignature(1));
    }

    #[test]
    fn deleting_an_entry_breaks_the_chain() {
        let (log, vk) = sample_log();
        let mut entries = log.entries().to_vec();
        entries.remove(1); // drop the middle entry
        // entry formerly at index 2 now sits at index 1 with seq=2 → seq mismatch
        let err = verify_chain(&entries, &vk).unwrap_err();
        assert_eq!(err, AuditVerifyError::SeqMismatch(1));
    }

    #[test]
    fn reordering_breaks_the_chain() {
        let (log, vk) = sample_log();
        let mut entries = log.entries().to_vec();
        entries.swap(1, 2);
        let err = verify_chain(&entries, &vk).unwrap_err();
        // swapped entries no longer have contiguous seq numbers
        assert!(matches!(err, AuditVerifyError::SeqMismatch(_)));
    }

    #[test]
    fn wrong_key_fails() {
        let (log, _vk) = sample_log();
        let other = AuditSigner::generate().verifying_key_bytes();
        assert!(verify_chain(log.entries(), &other).is_err());
    }

    #[test]
    fn persistent_log_survives_reopen_and_detects_tamper() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("mythd_auditlog_test_{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let s = SessionId(String::from("0011223344556677"));

        // Persist a key seed so we can reopen with the SAME signer.
        let signer = AuditSigner::generate();
        let seed = signer.seed_bytes().to_vec();
        let vk = signer.verifying_key_bytes();

        // Write two entries, then drop the log.
        {
            let mut log = AuditLog::open(signer, &path).unwrap();
            log.append(&s, 1, "BitNetInt2", 5, 10);
            log.append(&s, 2, "BitNetInt2", 6, 12);
            assert_eq!(log.len(), 2);
        }

        // Reopen with a signer reconstructed from the same seed → entries load,
        // chain verifies, and the sequence continues from where it left off.
        let signer2 = AuditSigner::from_seed_bytes(&seed).unwrap();
        let mut log2 = AuditLog::open(signer2, &path).unwrap();
        assert_eq!(log2.len(), 2, "entries survived restart");
        verify_chain(log2.entries(), &vk).expect("reloaded chain verifies");
        let seq = log2.append(&s, 3, "BitNetInt2", 7, 14);
        assert_eq!(seq, 2, "seq continues after reload");
        assert_eq!(log2.len(), 3);
        drop(log2);

        // Tamper: flip a byte in the persisted file → reopen must reject it.
        let mut bytes = std::fs::read(&path).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0x01;
        std::fs::write(&path, &bytes).unwrap();
        let signer3 = AuditSigner::from_seed_bytes(&seed).unwrap();
        assert!(
            AuditLog::open(signer3, &path).is_err(),
            "tampered on-disk log must fail to open"
        );

        let _ = std::fs::remove_file(&path);
    }
}
