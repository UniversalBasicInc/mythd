// mythd daemon build script.
//
// When built with `--features llama-cpp`, the final `mythd` binary must link the
// shim + libllama + libggml. Native link-search paths emitted by a dependency's
// build script (mythd-inference) do NOT propagate to this downstream binary, so
// the daemon re-emits them here — and bakes an rpath so the daemon resolves the
// libraries at runtime without LD_LIBRARY_PATH. No-op without the feature.
//
// Paths default to a local layout; override with the same env vars mythd-inference
// uses: MYTHD_LLAMA_SHIM_DIR / MYTHD_LLAMA_DIR / MYTHD_GGML_DIR / MYTHD_SYCL_DIR.

use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    for v in ["MYTHD_LLAMA_SHIM_DIR", "MYTHD_LLAMA_DIR", "MYTHD_GGML_DIR", "MYTHD_SYCL_DIR"] {
        println!("cargo:rerun-if-env-changed={v}");
    }

    if std::env::var_os("CARGO_FEATURE_LLAMA_CPP").is_none() {
        return;
    }

    const BITNET: &str = "/opt/BitNet";
    let llama_dir = env_or("MYTHD_LLAMA_DIR", &format!("{BITNET}/build-sycl/3rdparty/llama.cpp/src"));
    let ggml_dir = env_or("MYTHD_GGML_DIR", &format!("{BITNET}/build-sycl/3rdparty/llama.cpp/ggml/src"));
    let shim_dir = env_or("MYTHD_LLAMA_SHIM_DIR", "kernels");

    for dir in [&shim_dir, &llama_dir, &ggml_dir] {
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
    }
    if let Some(sycl) = std::env::var_os("MYTHD_SYCL_DIR") {
        let sycl = sycl.to_string_lossy().into_owned();
        println!("cargo:rustc-link-search=native={sycl}");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{sycl}");
    }

    println!("cargo:rustc-link-lib=dylib=mythd_llama");
    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-lib=dylib=ggml");
    for lib in ["ggml-base", "ggml-sycl", "ggml-cpu"] {
        if Path::new(&format!("{ggml_dir}/lib{lib}.so")).exists() {
            println!("cargo:rustc-link-lib=dylib={lib}");
        }
    }
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}
