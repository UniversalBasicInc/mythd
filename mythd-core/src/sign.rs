//! ML-DSA-65 (FIPS 204) signatures for the tamper-evident audit log.
//!
//! Every audit-log entry (and, at higher layers, every emitted token batch) is
//! signed with ML-DSA-65 so the log is post-quantum tamper-evident: an attacker
//! who alters a record cannot forge a matching signature without the secret key.
//!
//! The signing key never leaves this layer; callers receive only signatures and
//! the public verifying key.

use crate::error::{CoreError, Result};
use crate::rng::OsRng;
use ml_dsa::signature::{Keypair, Signer, Verifier};
use ml_dsa::{
    EncodedSignature, EncodedVerifyingKey, Generate, MlDsa65, Seed, Signature, SigningKey,
    VerifyingKey,
};
use zeroize::Zeroizing;

type P = MlDsa65;

/// Holds an ML-DSA-65 signing key and signs audit records.
pub struct AuditSigner {
    signing_key: SigningKey<P>,
}

impl AuditSigner {
    /// Generate a fresh signing key from the OS CSPRNG.
    pub fn generate() -> Self {
        // `generate_from_rng` is the infallible variant (our OsRng is CryptoRng).
        let signing_key = SigningKey::<P>::generate_from_rng(&mut OsRng);
        Self { signing_key }
    }

    /// The public verifying key, serialized (publishable; used by `verify`).
    pub fn verifying_key_bytes(&self) -> Vec<u8> {
        self.signing_key.verifying_key().encode().to_vec()
    }

    /// Sign a message (e.g. a serialized audit record).
    pub fn sign(&self, message: &[u8]) -> Vec<u8> {
        self.signing_key.sign(message).encode().to_vec()
    }

    /// Export the 32-byte FIPS-204 seed for persistence. **Key material** — the
    /// returned buffer zeroizes on drop; never log it or copy it into plain memory.
    /// Persisting the seed (not the expanded key) keeps the audit log's verifying
    /// key stable across restarts via [`from_seed_bytes`](Self::from_seed_bytes).
    pub fn seed_bytes(&self) -> Zeroizing<Vec<u8>> {
        Zeroizing::new(self.signing_key.as_seed().as_slice().to_vec())
    }

    /// Reconstruct the signer deterministically from a 32-byte seed (the inverse
    /// of [`seed_bytes`](Self::seed_bytes)). Rejects any non-32-byte input.
    pub fn from_seed_bytes(seed: &[u8]) -> Result<Self> {
        let seed = Seed::try_from(seed).map_err(|_| CoreError::MalformedKey)?;
        Ok(Self {
            signing_key: SigningKey::<P>::from_seed(&seed),
        })
    }
}

/// Verify an ML-DSA-65 signature against a serialized verifying key. Returns
/// `Ok(())` only on a valid signature; the error is intentionally opaque.
pub fn verify(verifying_key: &[u8], message: &[u8], signature: &[u8]) -> Result<()> {
    let vk_encoded =
        EncodedVerifyingKey::<P>::try_from(verifying_key).map_err(|_| CoreError::MalformedKey)?;
    let vk = VerifyingKey::<P>::decode(&vk_encoded);

    let sig_encoded =
        EncodedSignature::<P>::try_from(signature).map_err(|_| CoreError::MalformedKey)?;
    let sig = Signature::<P>::decode(&sig_encoded).ok_or(CoreError::MalformedKey)?;

    vk.verify(message, &sig).map_err(|_| CoreError::BadSignature)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_then_verify_roundtrip() {
        let signer = AuditSigner::generate();
        let vk = signer.verifying_key_bytes();
        let msg = b"audit: session=abc req=1 tokens=42";

        let sig = signer.sign(msg);
        verify(&vk, msg, &sig).expect("valid signature should verify");
    }

    #[test]
    fn tampered_message_fails() {
        let signer = AuditSigner::generate();
        let vk = signer.verifying_key_bytes();
        let sig = signer.sign(b"original record");

        let err = verify(&vk, b"tampered record", &sig).unwrap_err();
        assert!(matches!(err, CoreError::BadSignature));
    }

    #[test]
    fn wrong_key_fails() {
        let a = AuditSigner::generate();
        let b = AuditSigner::generate();
        let msg = b"record";
        let sig = a.sign(msg);
        assert!(verify(&b.verifying_key_bytes(), msg, &sig).is_err());
    }
}
