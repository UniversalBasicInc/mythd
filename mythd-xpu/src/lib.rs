//! # mythd-xpu — Layer 2: XPU kernel interface
//!
//! Three jobs:
//! 1. [`caps`] — detect the XPU (Xe2 / B70 / XMX) via a runtime Level-Zero query.
//! 2. [`kernel`] — the safe wrapper over the BitNet INT2 SYCL kernel, plus the
//!    precision **fallback ladder** (BitNet INT2 → dense INT8 DPAS → FP16).
//! 3. [`pool`] — session-scoped device memory (no shared mutable state).
//!
//! ## Unsafe is confined and justified
//! This is one of only two crates that contain `unsafe` (the other is
//! `mythd-inference`, for the llama.cpp C API). The trust root (`mythd-core`) and
//! Layers 4–5 are `#![forbid(unsafe_code)]`. **Every `unsafe` block in this crate
//! carries a `// SAFETY:` comment.**

use thiserror::Error;

#[derive(Debug, Error)]
pub enum XpuError {
    #[error("no compatible XPU device found (Level-Zero unavailable or no Intel GPU)")]
    NoDevice,
    #[error("xpu kernel support not built (enable the `xpu-kernels` feature)")]
    KernelsNotBuilt,
    #[error("device kernel call failed (code {0})")]
    KernelCall(i32),
    #[error("buffer size mismatch: {0}")]
    BadShape(&'static str),
    #[error("requested precision {0:?} not supported on this device")]
    UnsupportedPrecision(Precision),
    #[error("session pool capacity exceeded")]
    PoolExhausted,
}

pub type Result<T> = std::result::Result<T, XpuError>;

/// Native integer/float matrix precisions a device's matrix engine exposes.
/// Drives kernel routing and the fallback ladder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// Ternary BitNet weights (1.58-bit, i2_s). Not a native XMX precision on
    /// Xe2 today — realized by the BitNet custom kernel (decode → INT8 DPAS).
    Int2,
    Int4,
    Int8,
    Fp16,
    Bf16,
    Fp32,
}

pub mod caps;
pub mod ffi;
pub mod kernel;
pub mod pool;

pub use caps::DeviceCaps;
pub use kernel::{select_kernel, KernelSelection};
pub use pool::SessionPool;
