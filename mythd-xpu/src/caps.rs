//! Device capability detection via Level-Zero.
//!
//! We `dlopen` the Level-Zero loader at runtime (rather than link against it) so
//! that on a host with no Intel GPU stack the call simply returns
//! [`XpuError::NoDevice`] and the caller falls back — instead of the binary
//! failing to load. This is the foundation of the fallback ladder in
//! [`crate::kernel`].
//!
//! Layout/ABI constants below come straight from `level_zero/ze_api.h`.

use crate::{Precision, Result, XpuError};
use core::ffi::c_void;
use core::ptr;

// --- Level-Zero ABI constants (ze_api.h) ---
const ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES: u32 = 0x3;
const ZE_DEVICE_TYPE_GPU: u32 = 1;
const ZE_RESULT_SUCCESS: i32 = 0;
const INTEL_VENDOR_ID: u32 = 0x8086;

/// Intel Battlemage (Xe2) discrete GPUs report PCI device ids in the 0xe2xx
/// range. B580 = 0xe20b, **B70 (Arc Pro) = 0xe223** (confirmed via sycl-ls /
/// torch.xpu device string "Intel(R) Graphics [0xe223]").
const B70_DEVICE_ID: u32 = 0xe223;
const XE2_DEVICE_ID_MASK: u32 = 0xff00;
const XE2_DEVICE_ID_BASE: u32 = 0xe200;

// Opaque Level-Zero handles (typedef of `struct _ze_*_handle_t *`).
type Handle = *mut c_void;
type ZeResult = i32;

/// `ze_device_properties_t` — field-for-field `#[repr(C)]` mirror of the C struct
/// (ze_api.h). Padding is handled by `repr(C)` to match the C ABI.
#[repr(C)]
struct ZeDeviceProperties {
    stype: u32,
    p_next: *mut c_void,
    type_: u32,
    vendor_id: u32,
    device_id: u32,
    flags: u32,
    subdevice_id: u32,
    core_clock_rate: u32,
    max_mem_alloc_size: u64,
    max_hardware_contexts: u32,
    max_command_queue_priority: u32,
    num_threads_per_eu: u32,
    physical_eu_simd_width: u32,
    num_eus_per_subslice: u32,
    num_subslices_per_slice: u32,
    num_slices: u32,
    timer_resolution: u64,
    timestamp_valid_bits: u32,
    kernel_timestamp_valid_bits: u32,
    uuid: [u8; 16], // ze_device_uuid_t { uint8_t id[ZE_MAX_DEVICE_UUID_SIZE=16] }
    name: [u8; 256], // char name[ZE_MAX_DEVICE_NAME=256]
}

impl Default for ZeDeviceProperties {
    fn default() -> Self {
        // stype MUST be set before the call; pNext must be null.
        Self {
            stype: ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES,
            p_next: ptr::null_mut(),
            type_: 0,
            vendor_id: 0,
            device_id: 0,
            flags: 0,
            subdevice_id: 0,
            core_clock_rate: 0,
            max_mem_alloc_size: 0,
            max_hardware_contexts: 0,
            max_command_queue_priority: 0,
            num_threads_per_eu: 0,
            physical_eu_simd_width: 0,
            num_eus_per_subslice: 0,
            num_subslices_per_slice: 0,
            num_slices: 0,
            timer_resolution: 0,
            timestamp_valid_bits: 0,
            kernel_timestamp_valid_bits: 0,
            uuid: [0u8; 16],
            name: [0u8; 256],
        }
    }
}

/// Detected capabilities of the selected XPU.
#[derive(Debug, Clone)]
pub struct DeviceCaps {
    pub name: String,
    pub vendor_id: u32,
    /// PCI device id (low 16 bits), e.g. 0xe223 for the B70.
    pub device_id: u32,
    pub is_xe2: bool,
    pub is_b70: bool,
    /// XMX matrix engines present (true for Battlemage/Xe2 discrete GPUs).
    pub has_xmx: bool,
    /// Largest single device allocation (proxy for memory size; total VRAM would
    /// need `zeDeviceGetMemoryProperties`).
    pub max_alloc_bytes: u64,
    /// Native matrix precisions the XMX engine exposes via `joint_matrix`.
    pub native_matrix_precisions: Vec<Precision>,
}

impl DeviceCaps {
    pub fn supports(&self, p: Precision) -> bool {
        self.native_matrix_precisions.contains(&p)
    }

    /// True when the BitNet INT2 custom kernel can run here (Xe2 + XMX). INT2 is
    /// not a native `joint_matrix` precision, so this is a capability, not a
    /// `native_matrix_precisions` entry.
    pub fn supports_bitnet_int2(&self) -> bool {
        self.is_xe2 && self.has_xmx
    }

    fn from_props(p: &ZeDeviceProperties) -> Self {
        let device_id = p.device_id & 0xffff;
        let is_xe2 = (device_id & XE2_DEVICE_ID_MASK) == XE2_DEVICE_ID_BASE;
        let is_b70 = device_id == B70_DEVICE_ID;
        let has_xmx = is_xe2; // Battlemage/Xe2 discrete GPUs ship XMX
        let mut native = Vec::new();
        if has_xmx {
            // From the BitNet work: Xe2 XMX exposes INT8 (and fp16/bf16) via
            // joint_matrix; INT4 is convert-only, INT2 not surfaced.
            native = vec![Precision::Int8, Precision::Fp16, Precision::Bf16, Precision::Fp32];
        }
        Self {
            name: cstr_to_string(&p.name),
            vendor_id: p.vendor_id,
            device_id,
            is_xe2,
            is_b70,
            has_xmx,
            max_alloc_bytes: p.max_mem_alloc_size,
            native_matrix_precisions: native,
        }
    }

    /// Detect the first Intel GPU via Level-Zero. Returns [`XpuError::NoDevice`]
    /// if the loader is absent, `zeInit` fails, or no Intel GPU is present.
    pub fn detect() -> Result<Self> {
        // SAFETY: `Library::new` dlopens the Level-Zero loader by soname. It is
        // unsafe because loading an arbitrary library can run initializers; here
        // the name is a fixed, trusted system library. On failure we map to
        // NoDevice (the fallback path). `lib` is held in scope for the whole
        // function so every Symbol borrowed from it stays valid.
        let lib = unsafe { libloading::Library::new("libze_loader.so.1") }
            .or_else(|_| unsafe { libloading::Library::new("libze_loader.so") })
            .map_err(|_| XpuError::NoDevice)?;

        // SAFETY: each `lib.get` resolves a symbol whose Rust signature we
        // declare to match the C ABI in ze_api.h exactly (verified against the
        // header). The returned `Symbol` borrows `lib`.
        unsafe {
            let ze_init: libloading::Symbol<unsafe extern "C" fn(u32) -> ZeResult> =
                lib.get(b"zeInit\0").map_err(|_| XpuError::NoDevice)?;
            let ze_driver_get: libloading::Symbol<
                unsafe extern "C" fn(*mut u32, *mut Handle) -> ZeResult,
            > = lib.get(b"zeDriverGet\0").map_err(|_| XpuError::NoDevice)?;
            let ze_device_get: libloading::Symbol<
                unsafe extern "C" fn(Handle, *mut u32, *mut Handle) -> ZeResult,
            > = lib.get(b"zeDeviceGet\0").map_err(|_| XpuError::NoDevice)?;
            let ze_device_get_properties: libloading::Symbol<
                unsafe extern "C" fn(Handle, *mut ZeDeviceProperties) -> ZeResult,
            > = lib.get(b"zeDeviceGetProperties\0").map_err(|_| XpuError::NoDevice)?;

            // SAFETY: zeInit takes an init-flags bitfield (0 = default); no pointers.
            if ze_init(0) != ZE_RESULT_SUCCESS {
                return Err(XpuError::NoDevice);
            }

            // Query driver count, then the drivers themselves.
            let mut n_drivers: u32 = 0;
            // SAFETY: Level-Zero contract — a null array pointer with a valid
            // count pointer returns the count in *n_drivers.
            if ze_driver_get(&mut n_drivers, ptr::null_mut()) != ZE_RESULT_SUCCESS || n_drivers == 0
            {
                return Err(XpuError::NoDevice);
            }
            let mut drivers: Vec<Handle> = vec![ptr::null_mut(); n_drivers as usize];
            // SAFETY: `drivers` has capacity `n_drivers`; we pass its pointer and
            // the same count, matching the contract.
            if ze_driver_get(&mut n_drivers, drivers.as_mut_ptr()) != ZE_RESULT_SUCCESS {
                return Err(XpuError::NoDevice);
            }

            for &driver in &drivers {
                let mut n_devices: u32 = 0;
                // SAFETY: null device array → returns the device count.
                if ze_device_get(driver, &mut n_devices, ptr::null_mut()) != ZE_RESULT_SUCCESS
                    || n_devices == 0
                {
                    continue;
                }
                let mut devices: Vec<Handle> = vec![ptr::null_mut(); n_devices as usize];
                // SAFETY: `devices` sized to `n_devices`.
                if ze_device_get(driver, &mut n_devices, devices.as_mut_ptr()) != ZE_RESULT_SUCCESS {
                    continue;
                }

                for &device in &devices {
                    let mut props = ZeDeviceProperties::default();
                    // SAFETY: `props` is a valid, correctly-laid-out, mutable
                    // `ze_device_properties_t` with `stype` set and `pNext` null,
                    // as the API requires.
                    if ze_device_get_properties(device, &mut props) != ZE_RESULT_SUCCESS {
                        continue;
                    }
                    if props.type_ == ZE_DEVICE_TYPE_GPU && props.vendor_id == INTEL_VENDOR_ID {
                        let caps = DeviceCaps::from_props(&props);
                        tracing::info!(
                            target: "mythd::xpu",
                            name = %caps.name,
                            device_id = format_args!("0x{:04x}", caps.device_id),
                            is_xe2 = caps.is_xe2,
                            is_b70 = caps.is_b70,
                            has_xmx = caps.has_xmx,
                            "detected Intel GPU"
                        );
                        return Ok(caps);
                    }
                }
            }
            Err(XpuError::NoDevice)
        }
    }
}

fn cstr_to_string(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xe2_classification() {
        // B70 and B580 are Xe2; B70 specifically flagged.
        let mk = |id: u32| {
            let mut p = ZeDeviceProperties::default();
            p.type_ = ZE_DEVICE_TYPE_GPU;
            p.vendor_id = INTEL_VENDOR_ID;
            p.device_id = id;
            DeviceCaps::from_props(&p)
        };
        let b70 = mk(0xe223);
        assert!(b70.is_xe2 && b70.is_b70 && b70.has_xmx);
        assert!(b70.supports_bitnet_int2());
        assert!(b70.supports(Precision::Int8));

        let b580 = mk(0xe20b);
        assert!(b580.is_xe2 && !b580.is_b70);

        // An older non-Xe2 id is not flagged Xe2.
        let alch = mk(0x56a0);
        assert!(!alch.is_xe2 && !alch.has_xmx);
    }

    #[test]
    fn detect_is_graceful_without_device() {
        // On a host with no Intel GPU / no Level-Zero, detect() must return
        // NoDevice rather than panic. (On the B70 host it returns Ok.)
        let _ = DeviceCaps::detect();
    }
}
