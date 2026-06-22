//! Error types for Layer 1. Deliberately coarse: callers above this layer should
//! not be able to branch on cryptographic failure details (avoids oracles).

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("i/o error")]
    Io(#[from] std::io::Error),

    /// A model file was rejected before any parsing (e.g. it looked like a
    /// Python pickle). This is a hard security boundary.
    #[error("model file rejected: {0}")]
    RejectedModel(String),

    /// The safetensors header/body failed to parse. No code is executed on parse
    /// failure; the file is simply discarded.
    #[error("safetensors parse error: {0}")]
    SafeTensors(String),

    /// ML-KEM encapsulate/decapsulate failed, or key material was malformed.
    #[error("kem failure")]
    Kem,

    /// HKDF expansion failed.
    #[error("key derivation failure")]
    Kdf,

    /// ML-DSA signature failed to verify, or a key/signature was malformed.
    /// Intentionally opaque.
    #[error("signature verification failed")]
    BadSignature,

    /// Key/ciphertext bytes had the wrong length or encoding.
    #[error("malformed key material")]
    MalformedKey,

    /// AEAD seal/open failed (wrong key, tampered ciphertext, or bad framing).
    /// Intentionally opaque.
    #[error("aead failure")]
    Aead,
}

pub type Result<T> = std::result::Result<T, CoreError>;
