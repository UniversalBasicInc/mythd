#!/usr/bin/env bash
#
# Reproducibly build mythd (with the BitNet INT2 SYCL inference engine) on an
# Intel Arc box — no hardcoded paths; everything is derived from $BITNET_DIR.
# See docs/BUILD.md for the full deploy flow (incl. building the engine itself).
#
#   source /opt/intel/oneapi/setvars.sh
#   BITNET_DIR=/path/to/BitNet ./scripts/build.sh
#
# Env:
#   BITNET_DIR      (required) a built Microsoft BitNet checkout whose llama.cpp
#                   was compiled with the SYCL backend (must contain build-sycl/).
#   MYTHD_FEATURES  cargo features (default "llama-cpp"; use "llama-cpp pq-tls"
#                   for post-quantum TLS — needs cmake for aws-lc-rs).
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FEATURES="${MYTHD_FEATURES:-llama-cpp}"

echo "== mythd build  (features: $FEATURES) =="

# Pick up rustup's cargo if a non-interactive shell hasn't put it on PATH.
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"

# --- preflight ---
command -v cargo >/dev/null || { echo "!! Rust/cargo not found — see https://rustup.rs"; exit 1; }
command -v icpx  >/dev/null || { echo "!! icpx not found — 'source <oneAPI>/setvars.sh' first"; exit 1; }
case "$FEATURES" in
  *pq-tls*) command -v cmake >/dev/null || { echo "!! pq-tls needs cmake (for aws-lc-rs)"; exit 1; } ;;
esac
[ -e /dev/dri/renderD128 ] || echo "   note: no /dev/dri/renderD128 — builds fine, but needs an Intel GPU to run"

# Engine source: a llama.cpp checkout built with the SYCL backend. Prefer
# mainline llama.cpp (LLAMA_DIR) — it supports the widest model set (Qwen3 MoE,
# Llama, Mistral, Gemma, BitNet/TQ, …). BITNET_DIR (the older fork) is still
# accepted for the legacy i2_s path. Two on-disk layouts are auto-detected.
MTMD_INC=""   # multimodal (vision) headers dir, if available
if [ -n "${LLAMA_DIR:-}" ]; then
  # Mainline layout: libs in build-sycl/bin, headers in include/ + ggml/include.
  L_INC="$LLAMA_DIR/include"; G_INC="$LLAMA_DIR/ggml/include"
  L_LIB="$LLAMA_DIR/build-sycl/bin"; G_LIB="$LLAMA_DIR/build-sycl/bin"
  [ -f "$LLAMA_DIR/tools/mtmd/mtmd.h" ] && MTMD_INC="$LLAMA_DIR/tools/mtmd"
elif [ -n "${BITNET_DIR:-}" ]; then
  # BitNet-fork layout.
  L_INC="$BITNET_DIR/3rdparty/llama.cpp/include"; G_INC="$BITNET_DIR/3rdparty/llama.cpp/ggml/include"
  L_LIB="$BITNET_DIR/build-sycl/3rdparty/llama.cpp/src"; G_LIB="$BITNET_DIR/build-sycl/3rdparty/llama.cpp/ggml/src"
else
  echo "!! set LLAMA_DIR (mainline llama.cpp, recommended) or BITNET_DIR (legacy fork) — see docs/BUILD.md"; exit 1
fi
# llama.cpp may name the lib libllama.so or libllama.so.0 (versioned).
ls "$L_INC/llama.h" >/dev/null 2>&1 || { echo "!! missing $L_INC/llama.h — is the checkout + SYCL build present? (docs/BUILD.md §1)"; exit 1; }
ls "$L_LIB"/libllama.so* >/dev/null 2>&1 || { echo "!! missing libllama.so in $L_LIB — build with the SYCL backend (build-sycl/)"; exit 1; }
# Vision (mtmd) is required by the shim source; needs libmtmd + headers.
if [ -z "$MTMD_INC" ] || ! ls "$L_LIB"/libmtmd.so* >/dev/null 2>&1; then
  echo "!! missing libmtmd / mtmd.h — mythd's shim needs multimodal support."
  echo "   Build mainline llama.cpp with the mtmd tool (default) so build-sycl/bin/libmtmd.so + tools/mtmd/mtmd.h exist."
  exit 1
fi
SYCL_LIB="$(realpath "$(dirname "$(command -v icpx)")/../lib")"

# --- 1. compile the C-ABI shim (rpath-baked so the daemon needs no LD_LIBRARY_PATH) ---
echo ">> [1/2] compiling C-ABI shim → kernels/libmythd_llama.so"
# The shim references llama_* and mtmd_* symbols; libggml resolves transitively via
# rpath at runtime. rpaths bake the engine + SYCL runtime locations into the .so.
icpx -O2 -fPIC -shared "$ROOT/kernels/mythd_llama.cpp" -o "$ROOT/kernels/libmythd_llama.so" \
  -I"$L_INC" -I"$G_INC" -I"$MTMD_INC" -L"$L_LIB" -lllama -lmtmd \
  -Wl,-rpath,"$L_LIB" -Wl,-rpath,"$G_LIB" -Wl,-rpath,"$SYCL_LIB"

# --- 2. build mythd (paths handed to the Rust build scripts via env) ---
echo ">> [2/2] cargo build --release"
export MYTHD_LLAMA_SHIM_DIR="$ROOT/kernels"
export MYTHD_LLAMA_DIR="$L_LIB"
export MYTHD_GGML_DIR="$G_LIB"
export MYTHD_SYCL_DIR="$SYCL_LIB"
cargo build --release --manifest-path "$ROOT/Cargo.toml" -p mythd-bin --bin mythd --features "$FEATURES"

echo
echo "✅ Built: $ROOT/target/release/mythd"
echo "   1) cp docs/mythd.toml.example /etc/myth/mythd.toml   # set model_path, api_keys, tls_*"
echo "   2) source <oneAPI>/setvars.sh"
echo "   3) ONEAPI_DEVICE_SELECTOR=level_zero:gpu MYTHD_CONFIG=/etc/myth/mythd.toml $ROOT/target/release/mythd"
