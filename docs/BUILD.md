# Building & deploying mythd

mythd is a security/serving wrapper over **llama.cpp's SYCL backend**, so the build
is two stages: **(1)** build a llama.cpp SYCL engine, **(2)** build mythd against
it with `scripts/build.sh`. Use **mainline llama.cpp** (`LLAMA_DIR`) for the widest
model support — Qwen3 MoE, Llama, Mistral, Gemma, BitNet/TQ, etc. (The older
Microsoft BitNet fork, `BITNET_DIR`, is still accepted for its legacy i2_s path.)
Stage 2 derives every path from the engine dir — no hardcoded locations.

## Prerequisites
- Intel Arc **Xe2 / Battlemage** GPU (B580 / B70) on Linux with the in-tree `xe` driver.
- Intel **oneAPI** (DPC++ `icpx`/`icx` + Level-Zero loader) — `source /opt/intel/oneapi/setvars.sh`.
- **Rust** 1.85+ (https://rustup.rs).
- **cmake** (for stage 1, and for the optional `pq-tls` feature's aws-lc-rs).
- A coherent Intel GPU userspace stack (Level-Zero loader / NEO / IGC / gmmlib).
  The runtime needs a matched stack — the same kind built by the companion
  `boltz2-intel-arc` `setup_arc_stack.sh` (loader 1.29 + NEO 26.05 + IGC 2.28.4 +
  gmmlib 22.9.0), shadowed via `LD_LIBRARY_PATH`; the stock distro stack often
  segfaults on SYCL fp-config queries.

## 1. Build the llama.cpp SYCL engine (mainline — recommended)

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build-sycl \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL
cmake --build build-sycl -j
```

This produces the engine libraries in `build-sycl/bin/` (`libllama.so`,
`libggml*.so`). Mainline supports the full model set incl. Qwen3 MoE.

*Verified: llama.cpp @ `ba1df05` (2026-06), oneAPI 2026.0, running
Qwen3-30B-A3B-Instruct-2507 (Q4_K_M) on an Arc Pro B70.*

*(Legacy: the Microsoft BitNet fork builds to
`build-sycl/3rdparty/llama.cpp/{src,ggml/src}` and is accepted via `BITNET_DIR`
for its i2_s ternary path; mainline also runs BitNet via TQ quants.)*

## 2. Build mythd

```bash
source /opt/intel/oneapi/setvars.sh
LLAMA_DIR=/path/to/llama.cpp ./scripts/build.sh
```

Post-quantum TLS as well (needs cmake for aws-lc-rs):

```bash
MYTHD_FEATURES="llama-cpp pq-tls" LLAMA_DIR=/path/to/llama.cpp ./scripts/build.sh
```

`build.sh` compiles the thin C-ABI shim (`kernels/libmythd_llama.so`, with the
engine's rpath baked in so the daemon needs no `LD_LIBRARY_PATH`) and then
`cargo build --release`s the daemon → `target/release/mythd`.

## 3. Configure & run

```bash
cp docs/mythd.toml.example /etc/myth/mythd.toml    # set model_path, api_keys, tls_*, audit_key_path
source /opt/intel/oneapi/setvars.sh
ONEAPI_DEVICE_SELECTOR=level_zero:gpu MYTHD_CONFIG=/etc/myth/mythd.toml ./target/release/mythd
```

On first run the daemon generates the persistent ML-DSA-65 audit key at
`audit_key_path` (mode 0600); subsequent runs reuse it so the audit verifying key
is stable across restarts.

## Models
A BitNet **i2_s** (ternary) GGUF takes the Arc INT2 fast path (e.g. BitNet-2B
`ggml-model-i2_s.gguf`). Any other GGUF that llama.cpp's SYCL backend supports
also runs, via the dense FP16 / INT8 path.
