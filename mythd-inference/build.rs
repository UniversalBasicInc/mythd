// mythd-inference build script.
//
// With the `llama-cpp` feature OFF (default) this is a no-op, so the workspace
// builds on any host. With it ON, it links the `mythd_llama` shim plus libllama
// and libggml from the BitNet SYCL build (Option A). Paths default to a local
// layout and are overridable by env vars so the build is portable:
//
//   MYTHD_LLAMA_SHIM_DIR  dir containing libmythd_llama.so   (the C ABI shim)
//   MYTHD_LLAMA_DIR       dir containing libllama.so
//   MYTHD_GGML_DIR        dir containing libggml.so
//   MYTHD_SYCL_DIR        dir containing the SYCL runtime    (optional)
//
// The shim is compiled separately on the GPU box (see kernels/mythd_llama.cpp's
// header for the icpx command); this script only wires the linker.

use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=MYTHD_LLAMA_SHIM_DIR");
    println!("cargo:rerun-if-env-changed=MYTHD_LLAMA_DIR");
    println!("cargo:rerun-if-env-changed=MYTHD_GGML_DIR");
    println!("cargo:rerun-if-env-changed=MYTHD_SYCL_DIR");

    if std::env::var_os("CARGO_FEATURE_LLAMA_CPP").is_none() {
        return; // feature off → nothing to link
    }

    const BITNET: &str = "/opt/BitNet";
    let llama_dir = env_or(
        "MYTHD_LLAMA_DIR",
        &format!("{BITNET}/build-sycl/3rdparty/llama.cpp/src"),
    );
    let ggml_dir = env_or(
        "MYTHD_GGML_DIR",
        &format!("{BITNET}/build-sycl/3rdparty/llama.cpp/ggml/src"),
    );
    let shim_dir = env_or("MYTHD_LLAMA_SHIM_DIR", "kernels");

    // Search paths + rpath so the .so's resolve at runtime without LD_LIBRARY_PATH.
    for dir in [&shim_dir, &llama_dir, &ggml_dir] {
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
    }
    if let Some(sycl) = std::env::var_os("MYTHD_SYCL_DIR") {
        let sycl = sycl.to_string_lossy().into_owned();
        println!("cargo:rustc-link-search=native={sycl}");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{sycl}");
    }

    // Core libraries.
    println!("cargo:rustc-link-lib=dylib=mythd_llama");
    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-lib=dylib=ggml");

    // The BitNet build sometimes splits ggml into base/backend/sycl shards — link
    // whichever exist so symbol resolution is complete.
    for lib in ["ggml-base", "ggml-sycl", "ggml-cpu"] {
        if Path::new(&format!("{ggml_dir}/lib{lib}.so")).exists() {
            println!("cargo:rustc-link-lib=dylib={lib}");
        }
    }
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}
