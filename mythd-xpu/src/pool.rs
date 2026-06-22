//! Session-scoped device memory.
//!
//! Each session owns its own [`SessionPool`]. There is no global or `static`
//! device arena, and the pool is neither `Clone` nor shared, so two sessions can
//! never alias each other's device memory — the same ownership-based isolation
//! the Layer-4 KV cache relies on, pushed down to the device allocator.

use crate::{Result, XpuError};

/// A device scratch arena owned by exactly one session.
///
/// Reusing one pool across requests in a session avoids the per-call
/// `malloc_device`/`free` that throttled the BitNet GEMM before scratch was
/// cached; isolating one pool *per session* keeps that reuse from leaking state
/// across tenants.
pub struct SessionPool {
    capacity: usize,
    high_water: usize,
    // TODO(L2): hold the device allocation handle here once the kernel runtime
    // FFI exposes alloc/free. Until then this tracks intended usage so the
    // session/orchestrator wiring above can be exercised end-to-end.
}

impl SessionPool {
    /// Create a pool for one session with a fixed byte capacity.
    pub fn new(capacity: usize) -> Result<Self> {
        Ok(Self { capacity, high_water: 0 })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn high_water(&self) -> usize {
        self.high_water
    }

    /// Reserve `bytes` from this session's pool, failing if it would exceed
    /// capacity. (Device-backed allocation is wired with the kernel runtime.)
    pub fn reserve(&mut self, bytes: usize) -> Result<()> {
        let need = self
            .high_water
            .checked_add(bytes)
            .ok_or(XpuError::PoolExhausted)?;
        if need > self.capacity {
            return Err(XpuError::PoolExhausted);
        }
        self.high_water = need;
        Ok(())
    }

    /// Reset usage between requests (retains the underlying capacity).
    pub fn reset(&mut self) {
        self.high_water = 0;
    }
}

// Intentionally NOT Clone/Copy: a pool belongs to a single session.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve_respects_capacity_and_resets() {
        let mut p = SessionPool::new(1024).unwrap();
        p.reserve(600).unwrap();
        assert_eq!(p.high_water(), 600);
        assert!(matches!(p.reserve(500), Err(XpuError::PoolExhausted)));
        p.reset();
        assert_eq!(p.high_water(), 0);
        p.reserve(500).unwrap();
    }
}
