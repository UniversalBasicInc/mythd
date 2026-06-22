//! # mythd-core — Layer 1: security primitives
//!
//! The trust root of mythd. No other mythd crate sits below this one; everything
//! above depends on it, never the reverse.
//!
//! Provides:
//! * [`kem`] — ML-KEM-768 (FIPS 203) post-quantum session-key establishment.
//! * [`sign`] — ML-DSA-65 (FIPS 204) post-quantum signatures for the audit log.
//! * [`keystore`] — persistent audit-key storage (32-byte seed, mode 0600) so the
//!   audit verifying key is stable across restarts.
//! * [`safetensors_loader`] — safe model loading. The **only** model format that
//!   can be parsed is safetensors; there is no pickle / `torch.load` /
//!   arbitrary-deserialization path anywhere in this binary.
//! * [`secret`] — zeroizing key material.
//! * [`rng`] — the OS CSPRNG, the sole entropy source.
//!
//! ## No unsafe in the trust root
//! This crate is `#![forbid(unsafe_code)]`. All `unsafe` in mythd is confined to
//! the XPU FFI layer (`mythd-xpu`), where each block is individually justified.

#![forbid(unsafe_code)]

pub mod error;
pub mod kem;
pub mod keystore;
pub mod rng;
pub mod safetensors_loader;
pub mod secret;
pub mod sign;

pub use error::{CoreError, Result};
pub use kem::{encapsulate_to, KemKeypair};
pub use keystore::load_or_create_audit_signer;
pub use rng::{fill_random, OsRng};
pub use safetensors_loader::LoadedModel;
pub use secret::SessionKey;
pub use sign::{verify as verify_audit_signature, AuditSigner};
