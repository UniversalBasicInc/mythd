//! Raw C ABI of the compiled BitNet INT2 SYCL kernel.
//!
//! Declarations only — the definition lives in the kernel object that `build.rs`
//! links when the `xpu-kernels` feature is enabled. Every call is `unsafe`; the
//! safe wrapper is [`crate::kernel::bitnet_i2s_dpas`].

#[cfg(feature = "xpu-kernels")]
extern "C" {
    /// BitNet i2_s ternary × f16 DPAS GEMM. Returns 0 on success, nonzero error
    /// code otherwise.
    ///
    /// ABI (fixed by the kernel):
    /// `bitnet_i2s_dpas(weights: *const i8, activations: *const f16,
    ///                  output: *mut f32, m: usize, k: usize, n: usize) -> i32`
    ///
    /// `half::f16` is `#[repr(transparent)]` over `u16`, hence FFI-compatible
    /// with the C `_Float16*` / `uint16_t*` activation operand.
    pub fn bitnet_i2s_dpas(
        weights: *const i8,
        activations: *const half::f16,
        output: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    ) -> i32;
}
