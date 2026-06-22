# mythd

**A privacy-first, post-quantum-secured inference runtime for Intel Arc.**

mythd is a Rust-native **secure inference runtime** for Intel Arc (Xe2 / Battlemage) GPUs: a memory-safe **security perimeter** that wraps a proven inference engine (llama.cpp on the SYCL backend) and adds the guarantees a privacy- or sovereignty-focused deployment needs — post-quantum key exchange, compile-time session-isolation typing, data-only model loading (safetensors / GGUF, no code-execution path), and a post-quantum-signed, tamper-evident audit log. It runs ternary (BitNet 1.58-bit) and other GGUF models on the Arc XMX engines **via llama.cpp's kernels** (it does not ship its own GPU kernel). For teams already running on Arc, mythd is a security-first runtime option — complementary to Intel's acceleration tooling such as [ipex-llm](https://github.com/intel/ipex-llm) (now archived), which solved the orthogonal problem of raw acceleration.

> **Status:** working MVP, live-verified end-to-end on an Intel Arc Pro B70 (see [The live proof](#the-live-proof)). Greedy decoding, single model per daemon. Production hardening (Option B standalone kernel, externalized key management, per-request AEAD) is on the [roadmap](#roadmap).

---

## The five-layer architecture

Each crate may only depend on the layers below it. **The dependency graph _is_ the security architecture** — a layer literally cannot reach around the trust root.

```
                          ┌──────────────────────────────────────────────┐
   client  ──PQ-TLS──▶    │  L5  mythd-api / mythd-bin   #![forbid(unsafe)]│
  (X25519MLKEM768)        │      Axum: /v1/chat/completions (JSON + SSE),  │
                          │      /v1/session/init (ML-KEM-768 handshake),  │
                          │      /v1/models, /v1/audit/{id}, /health       │
                          └───────────────────────┬──────────────────────┘
                          ┌───────────────────────▼──────────────────────┐
                          │  L4  mythd-session          #![forbid(unsafe)]│
                          │      • per-session KV cache isolated by Rust   │
                          │        OWNERSHIP (cross-session access = a     │
                          │        compile error, not a runtime check)     │
                          │      • ML-DSA-65 signed, SHA-256 hash-chained   │
                          │        audit log (tamper-evident by design)    │
                          └───────────────────────┬──────────────────────┘
                          ┌───────────────────────▼──────────────────────┐
                          │  L3  mythd-inference                           │
                          │      orchestrator: model load, kernel routing  │
                          │      (BitNetInt2 → DenseInt8 → FP16 ladder),   │
                          │      llama.cpp via a thin, audited C-ABI shim  │
                          └───────────────────────┬──────────────────────┘
                          ┌───────────────────────▼──────────────────────┐
                          │  L2  mythd-xpu                                 │
                          │      Xe2 capability detection (device 0xe223), │
                          │      kernel selection ladder + session mempool │
                          │      ← `unsafe` GPU/engine FFI (L2–L3); each    │
                          │        block carries a `// SAFETY:` proof       │
                          └───────────────────────┬──────────────────────┘
                          ┌───────────────────────▼──────────────────────┐
                          │  L1  mythd-core   TRUST ROOT  #![forbid(unsafe)]│
                          │      ML-KEM-768 (FIPS 203) session keys,        │
                          │      ML-DSA-65 (FIPS 204) signatures,           │
                          │      HKDF-SHA256, safetensors-only loader       │
                          │      (NO pickle / torch.load anywhere),         │
                          │      zeroizing key material                     │
                          └────────────────────────────────────────────────┘
```

All crypto is pure Rust ([RustCrypto](https://github.com/RustCrypto): `ml-kem`, `ml-dsa`, `hkdf`, `sha2`, `zeroize`) — **no C/FFI in the cryptographic core.** The only `unsafe` in the entire workspace is the GPU FFI confined to L2/L3, and every block is individually justified with a `// SAFETY:` comment. L1, L4, and L5 are `#![forbid(unsafe_code)]`.

---

## Security properties

mythd is a *security* runtime, not an acceleration library — its value is in the wrapper, which protects **any** model run through it (Llama, Qwen, Mistral, BitNet, …):

| Property | mythd |
|---|---|
| Implementation | Rust, memory-safe by default — `#![forbid(unsafe_code)]` in the crypto core, session, and API layers; `unsafe` confined to the GPU/engine FFI (L2–L3), each block `// SAFETY:`-justified |
| Model loading | data-only formats only — safetensors (mythd-core's own loader) and GGUF (via llama.cpp); no pickle / `torch.load`, no code-execution path |
| Session isolation | compile-time KV-ownership typing (cross-session access is a type error), enforced at runtime by serialized execution + KV-clear between requests; physical `seq_id` partitioning is on the roadmap |
| Audit trail | ML-DSA-65-signed, SHA-256 hash-chained, tamper-evident |
| Application-layer key exchange | ML-KEM-768 (FIPS 203) per session |
| Transport | post-quantum TLS (`X25519MLKEM768`) |
| Ternary (1.58-bit) on Arc | runs BitNet / ternary GGUF on the XMX engines via llama.cpp's i2_s kernel |

These guarantees come from the wrapper, not any one kernel — so they hold for whatever model or engine kernel runs underneath. mythd is **complementary** to Arc acceleration tooling (such as Intel's now-archived ipex-llm): it doesn't aim to be faster, it aims to be a runtime you can put sensitive workloads through.

---

## The live proof

A single client driving the **entire** stack — post-quantum TLS, post-quantum session handshake, ternary inference on the Arc XMX engines, and signed-audit verification — against the daemon running on an Intel Arc Pro B70:

```
$ curl -v https://mythd:8443/health
* SSL connection using TLSv1.3 / TLS_AES_256_GCM_SHA384 / X25519MLKEM768 / RSASSA-PSS
```

```
target: https://127.0.0.1:8443  (transport: PQ-TLS X25519MLKEM768)

[1] GET /health
    {"audit_signatures":"ML-DSA-65","device":{"bitnet_int2":true,"detected":true,
     "device_id":"0xe223","has_xmx":true,"is_xe2":true,
     "name":"Intel(R) Graphics [0xe223]"},"kem":"ML-KEM-768",
     "model":"bitnet-b1.58-2b","service":"mythd","status":"ok"}
  ✓ server advertises ML-KEM-768
  ✓ device reports BitNetInt2 capable
  ✓ device is the B70 (0xe223)

[2] POST /v1/session/init  (ML-KEM-768)
    session_id = cce08764277afbd9cd673a38bc845da4
  ✓ both sides derived the SAME session key (PQ handshake)

[3a] POST /v1/chat/completions  (JSON, session-bound)
    kernel  = BitNetInt2
    usage   = {"completion_tokens":48,"prompt_tokens":17,"total_tokens":65}
    output  =  The capital of France is Paris. Paris is a city located in the
               northern part of the country and is known for its rich history,
               culture, and landmarks. It is the largest city in France ...
  ✓ inference ran on the BitNetInt2 kernel
  ✓ answer mentions Paris

[3b] POST /v1/chat/completions  (stream: true, SSE)
    reassembled =  The primary color of France is blue. ...
  ✓ SSE stream terminated with [DONE]
  ✓ SSE stream produced content

[4] GET /v1/audit/{session}  (verify ML-DSA-65 signatures)
    2 signed entries, scheme = "ML-DSA-65"
    seq 0 kernel="BitNetInt2" gen_tokens=48 sig_valid=true
    seq 1 kernel="BitNetInt2" gen_tokens=32 sig_valid=true
  ✓ every audit entry's ML-DSA-65 signature verifies
  ✓ audit recorded both inferences for this session

========================================
ALL CHECKS PASSED — mythd MVP is real.
```

The client (`mythd-bin/examples/integration_client.rs`) performs a genuine ML-KEM-768 keypair generation → decapsulation → key-confirmation, and verifies every audit signature with the published ML-DSA-65 key — entirely independent of the server.

---

## Hardware requirements

| | |
|---|---|
| **GPU** | Intel Arc **Xe2 / Battlemage** — Arc Pro **B70** (device `0xe223`, verified) or Arc **B580** (`0xe20b`). XMX (matrix) engines required for the INT2 fast path. |
| **GPU memory** | ≥ 4 GB for the BitNet-2B model; more for larger models. |
| **Toolchain** | Intel **oneAPI 2025.x or newer** (verified on 2026.0) — provides `icpx` (DPC++) and the Level-Zero loader. |
| **OS** | **Debian 13** (verified). Any Linux with a recent Level-Zero / `xe` KMD and Arc firmware should work. |
| **Rust** | 1.85+ (edition 2021). |

Non-Arc hosts still build and run mythd (the workspace compiles anywhere); device detection degrades gracefully and the inference engine is feature-gated.

---

## Quick start

mythd reaches the BitNet INT2 kernel through llama.cpp's SYCL backend, so you first build that engine, then a thin C-ABI shim, then mythd.

**1. Build the BitNet/llama.cpp SYCL engine** (Microsoft BitNet fork) per its instructions, producing `libllama.so` and `libggml.so` on your Arc host with oneAPI sourced.

**2. Compile the shim** (`kernels/mythd_llama.cpp`) against that build:

```sh
source /opt/intel/oneapi/setvars.sh
B=/path/to/BitNet
icpx -O2 -fPIC -shared kernels/mythd_llama.cpp -o kernels/libmythd_llama.so \
  -I$B/3rdparty/llama.cpp/include -I$B/3rdparty/llama.cpp/ggml/include \
  -L$B/build-sycl/3rdparty/llama.cpp/src       -lllama \
  -L$B/build-sycl/3rdparty/llama.cpp/ggml/src  -lggml \
  -Wl,-rpath,$B/build-sycl/3rdparty/llama.cpp/src \
  -Wl,-rpath,$B/build-sycl/3rdparty/llama.cpp/ggml/src \
  -Wl,-rpath,/opt/intel/oneapi/compiler/latest/lib
```

**3. Build mythd.** Point the build at the libraries via env vars; enable `llama-cpp` for the engine and (optionally) `pq-tls` for post-quantum transport:

```sh
export MYTHD_LLAMA_SHIM_DIR=$PWD/kernels
export MYTHD_LLAMA_DIR=$B/build-sycl/3rdparty/llama.cpp/src
export MYTHD_GGML_DIR=$B/build-sycl/3rdparty/llama.cpp/ggml/src
export MYTHD_SYCL_DIR=/opt/intel/oneapi/compiler/latest/lib

cargo build --release -p mythd-bin --bin mythd --features "llama-cpp pq-tls"
```

**4. Configure** — copy [`docs/mythd.toml.example`](docs/mythd.toml.example) to `/etc/myth/mythd.toml` and set `model_path` (a BitNet i2_s GGUF) and, for PQ-TLS, `tls_cert` / `tls_key`.

**5. Run:**

```sh
source /opt/intel/oneapi/setvars.sh
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export MYTHD_CONFIG=/etc/myth/mythd.toml
./target/release/mythd
```

It serves an OpenAI-compatible API — point any OpenAI client at it:

```sh
curl -k https://localhost:8443/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"bitnet","messages":[{"role":"user","content":"Hello"}],"max_tokens":48}'
```

Run the full end-to-end check:

```sh
cargo run -p mythd-bin --example integration_client -- https://127.0.0.1:8443
```

**Workspace tests** (run anywhere, no GPU needed):

```sh
cargo test --workspace      # 30 tests: PQC round-trips, tamper-evidence, isolation
```

---

## Why this needs native INT2 in Intel's compiler

On Xe2, the smallest integer operand the XMX/DPAS matrix engine exposes through Intel's SYCL `joint_matrix` API (in [Intel LLVM](https://github.com/intel/llvm)) is **INT8**. BitNet weights are **ternary (1.58 bits, packed 2-bit)**, so the i2_s kernel mythd runs (llama.cpp's SYCL path) must **unpack 2-bit → INT8 at runtime** and feed the INT8 matrix path. This already beats CPU comfortably — but it pays an unpack cost and spends INT8-width matrix slots on 2-bit data.

A **native INT2 `joint_matrix` type** in Intel LLVM (e.g. a `precision::s2` operand with Xe2 lowering for DPAS) would let ternary weights feed the XMX engine **directly**:

- **~4× operand density** per matrix tile vs. unpacked INT8,
- **no runtime unpack step**, and
- **lower memory-bandwidth pressure** — the binding constraint for low-bit decode.

That is the upstream contribution mythd is built to motivate. Ternary/1.58-bit models are the natural fit for memory-constrained edge GPUs, and Arc is positioned to be *the* open, NVIDIA-free platform for them — if the matrix engine can speak INT2 natively. mythd is the secure runtime that would consume it on day one.

---

## Roadmap

- [x] **L1–L5 MVP** — PQ session → BitNet INT2 SYCL inference → signed audit → SSE, over PQ-TLS. ✅ live
- [ ] **Option B: standalone kernel** — extract the INT2 GEMM into a self-contained `libbitnet_xpu.so` with a clean ABI, removing the coupling to the BitNet llama.cpp build and making mythd engine-agnostic.
- [ ] **Key-management hardening** — load audit/TLS keys from `/etc/myth/keys/` with strict permissions instead of generating per-boot; key rotation; HSM/TPM option.
- [ ] **Per-request application-layer AEAD** — encrypt request/response payloads under the established ML-KEM session key (the key is currently established and confirmed; payload encryption is the next step).
- [ ] **Physical KV partitioning** — bind each session to a distinct llama.cpp `seq_id` for in-context isolation alongside the existing ownership guarantee.
- [ ] **Native INT2 joint_matrix** — track/contribute the Intel LLVM work above.
- [ ] **FHE inference (research)** — explore fully-homomorphic-encryption inference paths so the server can compute on ciphertext it cannot read. Long-horizon; the PQC perimeter is the first step toward it.

---

## License

[MIT](LICENSE) © 2026 Universal Basic Inc.

---

Dylan Holdren — Lead developer & founder, Universal Basic Inc. · Built with AI assistance.
