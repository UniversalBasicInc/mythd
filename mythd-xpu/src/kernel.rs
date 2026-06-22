//! Safe wrapper over the BitNet kernel, plus the precision fallback ladder.

use crate::caps::DeviceCaps;
use crate::{Precision, Result, XpuError};
use half::f16;

/// Which kernel path the orchestrator should dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelSelection {
    /// BitNet ternary kernel (decode → INT8 DPAS) — Xe2 + XMX.
    BitNetInt2,
    /// Native INT8 XMX DPAS (dense weights / non-ternary).
    DenseInt8,
    /// FP16 fallback (no usable integer matrix path).
    Fp16,
}

/// The fallback ladder, in priority order:
/// 1. ternary weights **and** Xe2+XMX present → [`KernelSelection::BitNetInt2`]
/// 2. INT8 matrix path available (XMX)        → [`KernelSelection::DenseInt8`]
/// 3. otherwise                               → [`KernelSelection::Fp16`]
pub fn select_kernel(caps: &DeviceCaps, weight_precision: Precision) -> KernelSelection {
    match weight_precision {
        Precision::Int2 if caps.supports_bitnet_int2() => KernelSelection::BitNetInt2,
        _ if caps.supports(Precision::Int8) => KernelSelection::DenseInt8,
        _ => KernelSelection::Fp16,
    }
}

/// Safe wrapper over the BitNet i2_s DPAS GEMM.
///
/// Shapes: `output` is M×N (f32), `activations` is M×K (f16). `weights` is the
/// kernel's i8 weight buffer; its exact packing/size is part of the kernel ABI,
/// so the caller guarantees it is correctly sized for (K, N). We strictly
/// validate the two unambiguous buffers and require `weights` non-empty.
pub fn bitnet_i2s_dpas(
    weights: &[i8],
    activations: &[f16],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    let mk = m.checked_mul(k).ok_or(XpuError::BadShape("m*k overflow"))?;
    let mn = m.checked_mul(n).ok_or(XpuError::BadShape("m*n overflow"))?;
    if activations.len() < mk {
        return Err(XpuError::BadShape("activations shorter than m*k"));
    }
    if output.len() < mn {
        return Err(XpuError::BadShape("output shorter than m*n"));
    }
    if weights.is_empty() {
        return Err(XpuError::BadShape("weights empty"));
    }

    #[cfg(feature = "xpu-kernels")]
    {
        // SAFETY: all three pointers come from live slices that outlive this
        // call. We verified `activations.len() >= m*k` and `output.len() >= m*n`
        // above; the caller guarantees `weights` is sized for (k, n) per the
        // kernel ABI. The callee only reads weights/activations and writes within
        // output[0..m*n]; it allocates nothing and does not retain the pointers
        // past return.
        let code = unsafe {
            crate::ffi::bitnet_i2s_dpas(
                weights.as_ptr(),
                activations.as_ptr(),
                output.as_mut_ptr(),
                m,
                k,
                n,
            )
        };
        return if code == 0 {
            Ok(())
        } else {
            Err(XpuError::KernelCall(code))
        };
    }

    #[cfg(not(feature = "xpu-kernels"))]
    {
        let _ = (weights, activations, output, m, k, n);
        Err(XpuError::KernelsNotBuilt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::DeviceCaps;

    fn xe2_caps() -> DeviceCaps {
        DeviceCaps {
            name: "B70".into(),
            vendor_id: 0x8086,
            device_id: 0xe223,
            is_xe2: true,
            is_b70: true,
            has_xmx: true,
            max_alloc_bytes: 0,
            native_matrix_precisions: vec![
                Precision::Int8,
                Precision::Fp16,
                Precision::Bf16,
                Precision::Fp32,
            ],
        }
    }

    fn cpu_caps() -> DeviceCaps {
        DeviceCaps {
            name: "none".into(),
            vendor_id: 0,
            device_id: 0,
            is_xe2: false,
            is_b70: false,
            has_xmx: false,
            max_alloc_bytes: 0,
            native_matrix_precisions: vec![],
        }
    }

    #[test]
    fn ladder_picks_bitnet_then_int8_then_fp16() {
        let xe2 = xe2_caps();
        assert_eq!(select_kernel(&xe2, Precision::Int2), KernelSelection::BitNetInt2);
        assert_eq!(select_kernel(&xe2, Precision::Int8), KernelSelection::DenseInt8);
        assert_eq!(select_kernel(&xe2, Precision::Fp16), KernelSelection::DenseInt8);

        // No XMX: ternary can't use the BitNet kernel; nothing INT8 → FP16.
        let cpu = cpu_caps();
        assert_eq!(select_kernel(&cpu, Precision::Int2), KernelSelection::Fp16);
        assert_eq!(select_kernel(&cpu, Precision::Fp16), KernelSelection::Fp16);
    }

    #[test]
    fn wrapper_validates_shapes_without_kernel() {
        let w = vec![0i8; 4];
        let a = vec![f16::from_f32(0.0); 2]; // too short for m*k = 2*4
        let mut o = vec![0f32; 2];
        assert!(matches!(
            bitnet_i2s_dpas(&w, &a, &mut o, 2, 4, 1),
            Err(XpuError::BadShape(_))
        ));
    }
}
