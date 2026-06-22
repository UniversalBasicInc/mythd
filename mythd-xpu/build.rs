//! Build script for mythd-xpu.
//!
//! With the `xpu-kernels` feature OFF (default), this does nothing and the crate
//! builds on any host. With it ON, it links the compiled BitNet kernel object
//! that exports the C-ABI `bitnet_i2s_dpas` symbol. Configure its location via:
//!
//!   MYTHD_XPU_KERNEL_DIR  — directory containing the kernel library (required)
//!   MYTHD_XPU_KERNEL_LIB  — library name (default "bitnet_xpu" → libbitnet_xpu.so)
//!   MYTHD_XPU_SYCL_DIR    — optional: dir of the SYCL runtime the kernel needs
//!                           (adds a link-search path + links libsycl)

fn main() {
    println!("cargo:rerun-if-env-changed=MYTHD_XPU_KERNEL_DIR");
    println!("cargo:rerun-if-env-changed=MYTHD_XPU_KERNEL_LIB");
    println!("cargo:rerun-if-env-changed=MYTHD_XPU_SYCL_DIR");

    // CARGO_FEATURE_XPU_KERNELS is set by Cargo iff the feature is enabled.
    if std::env::var_os("CARGO_FEATURE_XPU_KERNELS").is_none() {
        return;
    }

    let dir = std::env::var("MYTHD_XPU_KERNEL_DIR").expect(
        "feature `xpu-kernels` is enabled but MYTHD_XPU_KERNEL_DIR is not set \
         (point it at the directory containing the compiled BitNet kernel)",
    );
    let lib = std::env::var("MYTHD_XPU_KERNEL_LIB").unwrap_or_else(|_| "bitnet_xpu".to_string());

    println!("cargo:rustc-link-search=native={dir}");
    println!("cargo:rustc-link-lib=dylib={lib}");

    if let Ok(sycl_dir) = std::env::var("MYTHD_XPU_SYCL_DIR") {
        println!("cargo:rustc-link-search=native={sycl_dir}");
        println!("cargo:rustc-link-lib=dylib=sycl");
    }
}
