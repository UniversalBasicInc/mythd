#!/usr/bin/env bash
# Batched llama-server — the inference engine behind the mythd gateway.
#
# Architecture: mythd (public :8090, PQC/auth/signed-audit) proxies to THIS
# process on loopback. llama-server owns the model weights, the per-slot KV
# cache, and continuous batching — which is what gives NoFootprint Lightning
# real concurrency (the old embedded-mythd path was strictly serial).
#
# oneAPI runtime (libsvml etc.) + the Intel GPU selector are sourced here so the
# SYCL backend offloads all layers to the B70.
# NOTE: intentionally NOT `set -u` — oneAPI's setvars.sh references unset vars,
# which under strict mode aborts the source and leaves libsvml unconfigured (the
# service then crash-loops with "libsvml.so: cannot open shared object file").
set -o pipefail

# --- paths (override via env) ---
LLAMA_BIN="${LLAMA_BIN:-$HOME/llama.cpp/build-sycl/bin/llama-server}"
MODEL="${MODEL:-$HOME/models/qwen3vl-30b-a3b/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf}"
MMPROJ="${MMPROJ:-$HOME/models/qwen3vl-30b-a3b/mmproj-F16.gguf}"
ONEAPI_SETVARS="${ONEAPI_SETVARS:-/opt/intel/oneapi/setvars.sh}"

# --- serving config ---
HOST="${HOST:-127.0.0.1}"            # loopback ONLY — mythd is the public face
PORT="${PORT:-8091}"
PARALLEL="${PARALLEL:-4}"            # concurrent slots; start 4, raise to 6-8 after a VRAM check
CTX_PER_SLOT="${CTX_PER_SLOT:-32768}"
CTX_SIZE="${CTX_SIZE:-$((PARALLEL * CTX_PER_SLOT))}"   # total KV pool = slots x per-slot

# Intel oneAPI runtime + GPU selection for the SYCL backend.
# shellcheck disable=SC1090
source "$ONEAPI_SETVARS" >/dev/null 2>&1 || true
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"

echo "llama-server: parallel=$PARALLEL ctx_per_slot=$CTX_PER_SLOT total_ctx=$CTX_SIZE kv=q8_0 host=$HOST:$PORT"
exec "$LLAMA_BIN" \
  -m "$MODEL" --mmproj "$MMPROJ" \
  -ngl 99 \
  --ctx-size "$CTX_SIZE" --parallel "$PARALLEL" --cont-batching \
  -ctk q8_0 -ctv q8_0 --flash-attn on \
  --no-context-shift \
  --cache-reuse 256 \
  --jinja \
  --host "$HOST" --port "$PORT" \
  --metrics --slots \
  --threads 8
