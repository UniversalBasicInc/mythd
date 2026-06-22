//! Zeroizing key material.
//!
//! Every symmetric secret in mythd lives in a type that scrubs its bytes on
//! drop. These types are deliberately not `Clone`, not `Serialize`, and have a
//! redacting `Debug` so a secret cannot be copied into non-zeroizing memory,
//! logged, or accidentally serialized into the (signed) audit log.

use crate::error::{CoreError, Result};
use chacha20poly1305::aead::{Aead, Payload};
use chacha20poly1305::{KeyInit, XChaCha20Poly1305, XNonce};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// XChaCha20-Poly1305 nonce length (192-bit; large enough that random nonces are
/// collision-safe, so no per-message counter state is needed).
const NONCE_LEN: usize = 24;

/// A 32-byte symmetric session key derived (via HKDF) from an ML-KEM-768 shared
/// secret. Scrubbed on drop.
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct SessionKey([u8; 32]);

impl SessionKey {
    pub(crate) fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Borrow the raw key for use by an AEAD/KDF. The borrow keeps the key inside
    /// the zeroizing allocation; callers must not copy it into a plain `[u8; 32]`
    /// that outlives this borrow.
    pub fn expose(&self) -> &[u8; 32] {
        &self.0
    }

    /// AEAD-seal `plaintext` under this session key with XChaCha20-Poly1305,
    /// authenticating `aad` (e.g. the session id, so a ciphertext can't be
    /// replayed into a different session). Returns `nonce(24) || ciphertext+tag`;
    /// the nonce is fresh-random per call.
    pub fn seal(&self, plaintext: &[u8], aad: &[u8]) -> Result<Vec<u8>> {
        let cipher = XChaCha20Poly1305::new_from_slice(&self.0).map_err(|_| CoreError::Aead)?;
        let mut nonce = [0u8; NONCE_LEN];
        crate::rng::fill_random(&mut nonce);
        let ct = cipher
            .encrypt(XNonce::from_slice(&nonce), Payload { msg: plaintext, aad })
            .map_err(|_| CoreError::Aead)?;
        let mut out = Vec::with_capacity(NONCE_LEN + ct.len());
        out.extend_from_slice(&nonce);
        out.extend_from_slice(&ct);
        Ok(out)
    }

    /// Inverse of [`seal`](Self::seal): verify + decrypt `sealed`
    /// (`nonce(24) || ciphertext+tag`) under this key and `aad`. Fails opaquely on
    /// a wrong key, tampered ciphertext/aad, or truncated input.
    pub fn open(&self, sealed: &[u8], aad: &[u8]) -> Result<Vec<u8>> {
        if sealed.len() < NONCE_LEN {
            return Err(CoreError::Aead);
        }
        let (nonce, ct) = sealed.split_at(NONCE_LEN);
        let cipher = XChaCha20Poly1305::new_from_slice(&self.0).map_err(|_| CoreError::Aead)?;
        cipher
            .decrypt(XNonce::from_slice(nonce), Payload { msg: ct, aad })
            .map_err(|_| CoreError::Aead)
    }
}

impl core::fmt::Debug for SessionKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("SessionKey(<redacted>)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_is_redacted() {
        let k = SessionKey::from_bytes([7u8; 32]);
        assert_eq!(format!("{k:?}"), "SessionKey(<redacted>)");
    }

    #[test]
    fn aead_seal_open_roundtrip_and_tamper() {
        let k = SessionKey::from_bytes([9u8; 32]);
        let aad = b"session-abc";
        let sealed = k.seal(b"secret prompt", aad).unwrap();
        assert_ne!(&sealed[NONCE_LEN..], b"secret prompt"); // genuinely encrypted
        assert_eq!(k.open(&sealed, aad).unwrap(), b"secret prompt");

        // wrong AAD (different session) is rejected
        assert!(k.open(&sealed, b"session-xyz").is_err());
        // tampered ciphertext is rejected
        let mut bad = sealed.clone();
        *bad.last_mut().unwrap() ^= 0x01;
        assert!(k.open(&bad, aad).is_err());
        // wrong key is rejected
        let k2 = SessionKey::from_bytes([1u8; 32]);
        assert!(k2.open(&sealed, aad).is_err());
    }
}
