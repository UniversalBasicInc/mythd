//! Persistent storage for the audit signing key.
//!
//! The ML-DSA-65 audit key is persisted as its **32-byte FIPS-204 seed**
//! (Algorithm 6, `KeyGen_internal`) and regenerated deterministically on load, so
//! the audit log's public verifying key is **stable across daemon restarts** — the
//! durable anchor a tamper-evident log needs. (Generating a fresh key each boot
//! would silently rotate the verifying key and break audit continuity.)
//!
//! The seed file is created and kept at mode `0600`; if an existing file is more
//! permissive it is tightened on load. No `unsafe`: the Unix permission APIs used
//! here (`PermissionsExt` / `OpenOptionsExt`) are safe.

use crate::error::Result;
use crate::sign::AuditSigner;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::Path;

/// Load the audit signer from `path`, or generate one and persist it there if the
/// file does not exist. The seed file is enforced to mode `0600` in both cases.
///
/// Returns the signer plus `true` if it was freshly **created** (vs. loaded), so
/// the caller can log which happened.
pub fn load_or_create_audit_signer(path: &Path) -> Result<(AuditSigner, bool)> {
    if path.exists() {
        // Defensively tighten perms before reading existing key material.
        let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o600));
        let bytes = zeroize::Zeroizing::new(fs::read(path)?);
        let signer = AuditSigner::from_seed_bytes(&bytes)?;
        Ok((signer, false))
    } else {
        if let Some(dir) = path.parent() {
            if !dir.as_os_str().is_empty() {
                fs::create_dir_all(dir)?;
                let _ = fs::set_permissions(dir, fs::Permissions::from_mode(0o700));
            }
        }
        let signer = AuditSigner::generate();
        let seed = signer.seed_bytes(); // Zeroizing<Vec<u8>>
        // Create with 0600 from the start (no race window where the seed is
        // world-readable), then fsync.
        let mut f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)?;
        f.write_all(&seed)?;
        f.sync_all()?;
        // Belt-and-suspenders in case of a permissive umask interaction.
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
        Ok((signer, true))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persists_a_stable_key_at_0600() {
        let path =
            std::env::temp_dir().join(format!("mythd_audit_test_{}.seed", std::process::id()));
        let _ = fs::remove_file(&path);

        // First call generates + persists.
        let (s1, created1) = load_or_create_audit_signer(&path).unwrap();
        assert!(created1, "first call should create the key");
        let vk1 = s1.verifying_key_bytes();

        // Seed file must be 0600 and exactly 32 bytes.
        let meta = fs::metadata(&path).unwrap();
        assert_eq!(meta.permissions().mode() & 0o777, 0o600, "seed must be 0600");
        assert_eq!(meta.len(), 32, "ML-DSA seed is 32 bytes");

        // Second call (a "restart") loads the SAME key — stable verifying key.
        let (s2, created2) = load_or_create_audit_signer(&path).unwrap();
        assert!(!created2, "second call should load, not create");
        assert_eq!(vk1, s2.verifying_key_bytes(), "verifying key must survive restart");

        // A signature from the reloaded key verifies under the original vk —
        // audit continuity across restarts.
        let sig = s2.sign(b"audit entry across restart");
        crate::sign::verify(&vk1, b"audit entry across restart", &sig).unwrap();

        let _ = fs::remove_file(&path);
    }
}
