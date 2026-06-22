//! Operating-system CSPRNG, exposed as a `rand_core` generator for the ML-KEM /
//! ML-DSA key generation and encapsulation routines.
//!
//! We deliberately do NOT pull in the `rand` crate: the only entropy source mythd
//! trusts is the OS CSPRNG via `getrandom`. Keeping the dependency surface small
//! and explicit matters for a security-critical crate.
//!
//! rand_core 0.10 derives `Rng`/`RngCore`/`CryptoRng` automatically from a
//! `TryRng<Error = Infallible>` + marker `TryCryptoRng` implementation, so we
//! implement those two and get the rest for free.

use core::convert::Infallible;
use rand_core::{TryCryptoRng, TryRng};

/// Stateless wrapper over the OS CSPRNG (`getrandom`).
///
/// If the OS RNG ever fails (which on a healthy system effectively never
/// happens), we abort rather than continue with degraded entropy — producing key
/// material from a failed RNG would be catastrophic, and the crate is built with
/// `panic = "abort"`.
pub struct OsRng;

impl TryRng for OsRng {
    // Infallible: a failed OS draw aborts inside `try_fill_bytes` rather than
    // surfacing an error, so this generator never returns `Err`.
    type Error = Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        let mut b = [0u8; 4];
        self.try_fill_bytes(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        let mut b = [0u8; 8];
        self.try_fill_bytes(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Infallible> {
        getrandom::fill(dst)
            .expect("OS CSPRNG (getrandom) failed — refusing to produce key material");
        Ok(())
    }
}

// Marker: asserts this generator is cryptographically secure. Backed solely by
// the OS CSPRNG, so the assertion holds. This (plus `TryRng<Error = Infallible>`)
// gives `CryptoRng` via rand_core's blanket impls.
impl TryCryptoRng for OsRng {}

/// Fill `dst` with cryptographically secure random bytes from the OS CSPRNG.
///
/// A convenience so callers (e.g. session-id generation in Layer 4) need not pull
/// in `rand_core` just to reach the `Rng` trait method.
pub fn fill_random(dst: &mut [u8]) {
    use rand_core::Rng;
    OsRng.fill_bytes(dst);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_core::Rng;

    #[test]
    fn produces_distinct_bytes() {
        let mut rng = OsRng;
        let mut a = [0u8; 32];
        let mut b = [0u8; 32];
        rng.fill_bytes(&mut a);
        rng.fill_bytes(&mut b);
        assert_ne!(a, b, "two OS RNG draws should not collide");
        assert_ne!(a, [0u8; 32]);
    }
}
