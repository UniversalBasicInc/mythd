//! ML-KEM-768 (FIPS 203) session-key establishment.
//!
//! Roles:
//! * The **server** (mythd) holds a [`KemKeypair`]; it publishes the encapsulation
//!   (public) key and uses the decapsulation (secret) key to recover the shared
//!   secret a client encapsulated to it.
//! * The **client** calls [`encapsulate_to`] with the server's public key,
//!   producing a ciphertext (sent to the server) and a session key (kept).
//!
//! The raw ML-KEM shared secret is never used directly as a session key; it is
//! run through HKDF-SHA256 with a context label so the key is bound to mythd's
//! protocol version. Secret intermediates live in zeroizing storage (and ml-kem
//! is built with its `zeroize` feature so its own key material scrubs on drop).

use crate::error::{CoreError, Result};
use crate::rng::OsRng;
use crate::secret::SessionKey;
use hkdf::Hkdf;
use ml_kem::kem::{Decapsulate, Encapsulate, KeyExport};
use ml_kem::{EncapsulationKey, Kem, Key, MlKem768};
use sha2::Sha256;
use zeroize::Zeroizing;

const HKDF_INFO: &[u8] = b"mythd/ml-kem-768/session-key/v1";

type Ek = EncapsulationKey<MlKem768>;

/// A server-side ML-KEM-768 keypair.
pub struct KemKeypair {
    decap: <MlKem768 as Kem>::DecapsulationKey,
    encap: Ek,
}

impl KemKeypair {
    /// Generate a fresh keypair from the OS CSPRNG.
    pub fn generate() -> Self {
        let (decap, encap) = MlKem768::generate_keypair_from_rng(&mut OsRng);
        Self { decap, encap }
    }

    /// The encapsulation (public) key, serialized for transport to clients.
    pub fn public_key_bytes(&self) -> Vec<u8> {
        self.encap.to_bytes().to_vec()
    }

    /// Recover the session key from a client's ciphertext.
    pub fn decapsulate(&self, ciphertext: &[u8]) -> Result<SessionKey> {
        // `decapsulate_slice` validates the ciphertext length for us.
        let shared = self
            .decap
            .decapsulate_slice(ciphertext)
            .map_err(|_| CoreError::MalformedKey)?;
        let ss = Zeroizing::new(shared.to_vec());
        derive_session_key(&ss)
    }
}

/// Client side: encapsulate to a server's public key, yielding the ciphertext to
/// send and the session key to keep.
pub fn encapsulate_to(server_public_key: &[u8]) -> Result<(Vec<u8>, SessionKey)> {
    let key = Key::<Ek>::try_from(server_public_key).map_err(|_| CoreError::MalformedKey)?;
    let encap = Ek::new(&key).map_err(|_| CoreError::MalformedKey)?;
    let (ct, shared) = encap.encapsulate_with_rng(&mut OsRng);
    let ss = Zeroizing::new(shared.to_vec());
    let key = derive_session_key(&ss)?;
    Ok((ct.to_vec(), key))
}

/// HKDF-SHA256 the raw ML-KEM shared secret into a 32-byte session key.
fn derive_session_key(shared_secret: &[u8]) -> Result<SessionKey> {
    let hk = Hkdf::<Sha256>::new(None, shared_secret);
    let mut okm = Zeroizing::new([0u8; 32]);
    hk.expand(HKDF_INFO, okm.as_mut_slice()).map_err(|_| CoreError::Kdf)?;
    Ok(SessionKey::from_bytes(*okm))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kem_roundtrip_yields_matching_session_keys() {
        let server = KemKeypair::generate();
        let pk = server.public_key_bytes();

        let (ct, client_key) = encapsulate_to(&pk).expect("encapsulate");
        let server_key = server.decapsulate(&ct).expect("decapsulate");

        assert_eq!(client_key.expose(), server_key.expose(), "session keys must match");
    }

    #[test]
    fn malformed_public_key_is_rejected() {
        let err = encapsulate_to(&[0u8; 10]).unwrap_err();
        assert!(matches!(err, CoreError::MalformedKey));
    }
}
