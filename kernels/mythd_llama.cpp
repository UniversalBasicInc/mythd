// mythd llama.cpp shim — a minimal, stable C ABI over a llama.cpp SYCL build
// (libllama/libggml). Keeping the by-value llama.cpp param/batch structs on the
// C++ side (where the compiler guarantees their ABI) means mythd's Rust side only
// sees a few clean functions — minimizing the unsafe surface, which is the point
// of mythd.
//
// Targets MAINLINE llama.cpp (2025+ API: vocab handle, llama_memory_*,
// llama_model_load_from_file, NULL-position llama_batch_get_one). Mainline's SYCL
// backend runs any architecture llama.cpp supports on Intel Arc — Qwen3 MoE,
// Llama, Mistral, Gemma, BitNet (TQ), etc. mythd is the security/serving wrapper;
// the model is whatever GGUF you load. (The older BitNet fork used a different
// API — see git history if you need that i2_s path.)
//
// SPECULATIVE DECODING (opt-in, self-contained): if the env var MYTHD_DRAFT_MODEL
// points at a small same-vocab GGUF, the shim loads it as a draft model and the
// generation path uses greedy draft-and-verify (lossless: the accepted tokens are
// exactly the target's own greedy tokens). Activates only for greedy (temp<=0),
// text-only requests; otherwise the normal single-model path runs unchanged. No
// Rust changes needed — set the env var and rebuild this .so. (MYTHD_DRAFT_N sets
// the draft length K, default 5.)
//
// Build (with oneAPI sourced, against a llama.cpp SYCL build):
//   icpx -O2 -fPIC -shared mythd_llama.cpp -o libmythd_llama.so \
//     -I<llama.cpp>/include -I<llama.cpp>/ggml/include \
//     -L<llama.cpp>/build-sycl/bin -lllama \
//     -Wl,-rpath,<llama.cpp>/build-sycl/bin -Wl,-rpath,<oneapi compiler lib>

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {
// Per-generation streaming state, owned by the handle. mythd reuses one
// llama_context across requests (serialized by the runtime lock), so one in-flight
// generation at a time — a single state instance per handle is sufficient.
struct GenState {
    bool active = false;
    int remaining = 0; // generation budget left
    int n_prompt = 0;
    int n_gen = 0;
    float temperature = 0.0f;
    float top_p = 1.0f;
    std::mt19937 rng;

    // --- speculative decoding (greedy) state, used when `spec` is true ---
    bool spec = false;
    int n_past = 0;                  // tokens committed to both KVs (excludes last_seed)
    llama_token last_seed = 0;       // last confirmed token, not yet in either KV
    std::vector<llama_token> emitq;  // confirmed tokens awaiting emission
    size_t emit_pos = 0;
    bool eog_hit = false;
    long n_drafted = 0;
    long n_accepted = 0;
};

struct MythdLlama {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    const llama_vocab* vocab = nullptr;
    int n_vocab = 0;
    mtmd_context* mtmd = nullptr; // non-null when a vision mmproj is loaded

    // optional draft model for speculative decoding (non-null = enabled)
    llama_model* draft_model = nullptr;
    llama_context* draft_ctx = nullptr;
    const llama_vocab* draft_vocab = nullptr;
    int draft_n_vocab = 0;
    int draft_len = 5; // K

    GenState gen;
};

// Argmax over the logits vector.
int argmax(const float* logits, int n_vocab) {
    int best = 0;
    float best_v = logits[0];
    for (int t = 1; t < n_vocab; ++t) {
        if (logits[t] > best_v) { best_v = logits[t]; best = t; }
    }
    return best;
}

// Temperature + nucleus (top-p) sampling.
//   temperature <= 0     → greedy (argmax), fully deterministic.
//   top_p <= 0 or >= 1   → no nucleus truncation (sample the full softmax).
// Self-contained (no dependency on llama.cpp's sampler API, which varies by
// version): softmax-with-temperature → optional top-p cutoff → inverse-CDF sample.
int sample_token(const float* logits, int n_vocab, float temperature, float top_p,
                 std::mt19937& rng) {
    if (temperature <= 0.0f) {
        return argmax(logits, n_vocab);
    }
    float maxl = logits[0];
    for (int t = 1; t < n_vocab; ++t) maxl = std::max(maxl, logits[t]);

    std::vector<std::pair<float, int>> probs;
    probs.reserve((size_t) n_vocab);
    double sum = 0.0;
    for (int t = 0; t < n_vocab; ++t) {
        double p = std::exp((double) (logits[t] - maxl) / (double) temperature);
        probs.emplace_back((float) p, t);
        sum += p;
    }
    for (auto& pr : probs) pr.first = (float) ((double) pr.first / sum);

    size_t cut = probs.size();
    if (top_p > 0.0f && top_p < 1.0f) {
        std::sort(probs.begin(), probs.end(),
                  [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                      return a.first > b.first;
                  });
        double cum = 0.0;
        for (size_t i = 0; i < probs.size(); ++i) {
            cum += probs[i].first;
            if (cum >= (double) top_p) { cut = i + 1; break; }
        }
    }

    double tsum = 0.0;
    for (size_t i = 0; i < cut; ++i) tsum += probs[i].first;
    std::uniform_real_distribution<double> dist(0.0, tsum);
    double r = dist(rng), acc = 0.0;
    for (size_t i = 0; i < cut; ++i) {
        acc += probs[i].first;
        if (r <= acc) return probs[i].second;
    }
    return probs[cut - 1].second;
}

// Chunked prefill of pre-tokenized prompt into `ctx` (NULL-position batches
// auto-advance from the cleared memory). Returns 0 on success, <0 on error.
int prefill_tokens(llama_context* ctx, const std::vector<llama_token>& tokens) {
    const int n = (int) tokens.size();
    const int n_batch = (int) llama_n_batch(ctx);
    for (int i = 0; i < n; i += n_batch) {
        int chunk = std::min(n_batch, n - i);
        llama_batch b = llama_batch_get_one(const_cast<llama_token*>(tokens.data()) + i, chunk);
        if (llama_decode(ctx, b) != 0) return -3;
    }
    return 0;
}

// One greedy draft-and-verify round. Drafts up to K tokens on the draft model,
// verifies them in a single target batch, accepts the matching prefix + the
// target's correction, rolls back both KV caches to the committed length, and
// appends the newly-confirmed tokens to g.emitq. Returns true if it produced at
// least one token, false if generation is finished (EOG/budget/decode error).
//
// Invariant maintained: both KVs hold exactly g.n_past tokens (prompt + all
// confirmed except g.last_seed), and g.last_seed is the confirmed-but-not-yet-
// cached token to draft from.
bool spec_round(MythdLlama* h) {
    GenState& g = h->gen;
    if (g.eog_hit || g.remaining <= 0) return false;
    llama_context* tctx = h->ctx;
    llama_context* dctx = h->draft_ctx;
    const int K = h->draft_len;

    // 1) draft up to K tokens from last_seed on the draft model (greedy)
    std::vector<llama_token> d;
    d.reserve((size_t) K);
    llama_token cur = g.last_seed;
    for (int k = 0; k < K; ++k) {
        llama_batch b = llama_batch_get_one(&cur, 1);
        if (llama_decode(dctx, b) != 0) break;
        const float* dl = llama_get_logits_ith(dctx, -1);
        if (!dl) break;
        int nx = argmax(dl, h->draft_n_vocab);
        d.push_back(nx);
        cur = nx;
        if (llama_vocab_is_eog(h->draft_vocab, nx)) break; // don't draft past EOG
    }
    const int m = (int) d.size();
    g.n_drafted += m;

    // 2) verify [last_seed, d0..d_{m-1}] on the target in one batch (logits at all)
    const int nb = m + 1;
    llama_batch vb = llama_batch_init(nb, 0, 1);
    vb.n_tokens = nb;
    vb.token[0] = g.last_seed;
    for (int k = 0; k < m; ++k) vb.token[k + 1] = d[k];
    for (int i = 0; i < nb; ++i) {
        vb.pos[i] = g.n_past + i;
        vb.n_seq_id[i] = 1;
        vb.seq_id[i][0] = 0;
        vb.logits[i] = 1;
    }
    int rc = llama_decode(tctx, vb);
    llama_batch_free(vb);
    if (rc != 0) return false;

    // 3) greedy accept run; first mismatch (or after-last) gives the correction
    int n_acc = 0;
    llama_token next = 0;
    bool have_next = false;
    for (int i = 0; i < nb; ++i) {
        const float* tl = llama_get_logits_ith(tctx, i);
        if (!tl) break;
        int p = argmax(tl, h->n_vocab);
        if (i < m && p == d[i]) {
            n_acc++;
        } else {
            next = p; have_next = true; break;
        }
    }
    if (!have_next) { // all m drafts matched → token after the whole draft
        const float* tl = llama_get_logits_ith(tctx, m);
        if (tl) { next = argmax(tl, h->n_vocab); have_next = true; }
    }
    g.n_accepted += n_acc;

    const int n_past_new = g.n_past + n_acc + 1; // committed: last_seed + n_acc drafts

    // 4a) keep the target KV to the committed length
    llama_memory_seq_rm(llama_get_memory(tctx), 0, n_past_new, -1);

    // 4b) sync the draft KV to the committed length. The draft cached
    // [last_seed, d0..d_{m-2}] (m tokens past n_past); we need [last_seed,
    // d0..d_{n_acc-1}] (n_acc+1 tokens). If n_acc==m the last accepted draft was
    // never fed to the draft model — feed it now so the caches stay in lockstep.
    const int draft_have = (m >= 1) ? m : 0;
    for (int j = draft_have; j < n_acc + 1; ++j) {
        llama_token t = (j == 0) ? g.last_seed : d[j - 1];
        llama_batch b = llama_batch_get_one(&t, 1); // auto-position = next slot
        if (llama_decode(dctx, b) != 0) break;
    }
    llama_memory_seq_rm(llama_get_memory(dctx), 0, n_past_new, -1);
    g.n_past = n_past_new;

    // 5) enqueue newly-confirmed tokens (accepted drafts, then the correction),
    // stopping at EOG or the budget.
    bool produced = false;
    for (int i = 0; i < n_acc && g.remaining > 0; ++i) {
        if (llama_vocab_is_eog(h->vocab, d[i])) { g.eog_hit = true; return produced; }
        g.emitq.push_back(d[i]);
        g.n_gen++; g.remaining--; produced = true;
    }
    if (!g.eog_hit && have_next && g.remaining > 0) {
        if (llama_vocab_is_eog(h->vocab, next)) { g.eog_hit = true; }
        else { g.emitq.push_back(next); g.n_gen++; g.remaining--; g.last_seed = next; produced = true; }
    }
    return produced;
}
} // namespace

extern "C" {

// One-time backend init. Returns 0 on success.
int mythd_llama_init() {
    llama_backend_init();
    return 0;
}

// Load a GGUF model with `n_gpu_layers` offloaded and a `n_ctx` context. If
// `mmproj_path` is non-null/non-empty, also load the vision projector (multimodal).
// `kv_q8` != 0 stores the KV cache as q8_0 (INT8) instead of f16 — ~half the KV
// memory + bandwidth (big long-context win); requires flash attention, so we
// force-enable it. If env MYTHD_DRAFT_MODEL is set, also load that as a draft
// model for speculative decoding. Returns an opaque handle, or nullptr on failure.
void* mythd_llama_load(const char* path, const char* mmproj_path, int n_gpu_layers,
                       int n_ctx, int kv_q8) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    llama_model* model = llama_model_load_from_file(path, mp);
    if (!model) return nullptr;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) n_ctx;
    if (kv_q8) {
        cp.type_k = GGML_TYPE_Q8_0;
        cp.type_v = GGML_TYPE_Q8_0;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED; // required for quantized KV
    }
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        llama_model_free(model);
        return nullptr;
    }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    auto* h = new MythdLlama{};
    h->model = model;
    h->ctx = ctx;
    h->vocab = vocab;
    h->n_vocab = llama_vocab_n_tokens(vocab);

    // Optional vision: load the multimodal projector (mmproj). On failure we keep
    // the text-only model rather than aborting the whole load.
    if (mmproj_path && mmproj_path[0] != '\0') {
        mtmd_context_params mctl = mtmd_context_params_default();
        mctl.use_gpu = true;
        mctl.print_timings = false;
        mctl.n_threads = 4;
        h->mtmd = mtmd_init_from_file(mmproj_path, model, mctl);
        // h->mtmd stays null on failure → image requests will be rejected.
    }

    // Optional speculative decoding: load a draft model from MYTHD_DRAFT_MODEL.
    // On any failure we keep the (working) single-model path rather than failing.
    const char* draft_path = std::getenv("MYTHD_DRAFT_MODEL");
    if (draft_path && draft_path[0] != '\0') {
        llama_model_params dmp = llama_model_default_params();
        dmp.n_gpu_layers = n_gpu_layers;
        llama_model* dmodel = llama_model_load_from_file(draft_path, dmp);
        if (dmodel) {
            llama_context_params dcp = llama_context_default_params();
            dcp.n_ctx = (uint32_t) n_ctx;
            dcp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            llama_context* dctx = llama_init_from_model(dmodel, dcp);
            const llama_vocab* dvocab = llama_model_get_vocab(dmodel);
            if (dctx && dvocab &&
                llama_vocab_n_tokens(dvocab) == h->n_vocab) { // vocab must match
                h->draft_model = dmodel;
                h->draft_ctx = dctx;
                h->draft_vocab = dvocab;
                h->draft_n_vocab = llama_vocab_n_tokens(dvocab);
                const char* kenv = std::getenv("MYTHD_DRAFT_N");
                if (kenv && kenv[0]) { int k = std::atoi(kenv); if (k >= 1 && k <= 16) h->draft_len = k; }
                std::fprintf(stderr, "mythd: speculative decoding ENABLED (draft=%s, K=%d)\n",
                             draft_path, h->draft_len);
            } else {
                if (dctx) llama_free(dctx);
                llama_model_free(dmodel);
                std::fprintf(stderr, "mythd: draft model vocab mismatch — speculative decoding OFF\n");
            }
        } else {
            std::fprintf(stderr, "mythd: draft model load failed — speculative decoding OFF\n");
        }
    }
    return h;
}

// --- streaming generation (the primitives the one-shot path is built on) ---
//
// Begin a generation: reset the KV/memory, tokenize, prefill. Sampling params:
// temperature <= 0 → greedy/deterministic; seed < 0 → random. Reports the prompt
// token count via the optional `out_n_prompt`. Returns 0 on success, negative on
// error. Then call `mythd_llama_stream_next` repeatedly.
// `image_data`/`image_len`: optional image bytes (PNG/JPEG/etc.) for a vision
// request. When provided (and a vision mmproj was loaded), the prompt MUST contain
// the media marker "<__media__>" where the image belongs; the image is encoded and
// interleaved via mtmd. When absent, this is the plain text path.
int mythd_llama_stream_begin(void* handle, const char* prompt, int max_tokens,
                             float temperature, float top_p, long long seed,
                             const unsigned char* image_data, size_t image_len,
                             int* out_n_prompt) {
    if (out_n_prompt) *out_n_prompt = 0;
    auto* h = static_cast<MythdLlama*>(handle);
    if (!h || !h->model || !h->ctx) return -1;
    llama_context* ctx = h->ctx;
    const llama_vocab* vocab = h->vocab;

    // Reset the context's memory so each request starts clean. mythd reuses one
    // llama_context across requests (serialized by the runtime lock); without this,
    // residual state contaminates the next request — breaking determinism and
    // seeded reproducibility. Also resets sequence positions to 0. CRITICAL.
    llama_memory_clear(llama_get_memory(ctx), true);
    h->gen = GenState{};

    // Speculative decoding eligibility: draft loaded, greedy, and text-only.
    const bool use_spec = (h->draft_ctx != nullptr) && (temperature <= 0.0f) && (image_len == 0);

    int n_prompt = 0;

    if (image_len > 0) {
        // --- vision (multimodal) prefill via mtmd ---
        if (!h->mtmd || !mtmd_support_vision(h->mtmd)) return -4; // no vision loaded
        mtmd_helper_bitmap_wrapper bw =
            mtmd_helper_bitmap_init_from_buf(h->mtmd, image_data, image_len, false);
        if (!bw.bitmap) return -5; // undecodable image
        mtmd_input_chunks* chunks = mtmd_input_chunks_init();
        mtmd_input_text it;
        it.text = prompt;
        it.add_special = true;
        it.parse_special = true;
        const mtmd_bitmap* bmps[1] = { bw.bitmap };
        int trc = mtmd_tokenize(h->mtmd, chunks, &it, bmps, 1);
        if (trc != 0) { mtmd_bitmap_free(bw.bitmap); mtmd_input_chunks_free(chunks); return -6; }
        // Encode the vision tokens + prefill the text, leaving last-token logits.
        llama_pos new_n_past = 0;
        int erc = mtmd_helper_eval_chunks(h->mtmd, ctx, chunks, /*n_past*/ 0,
                                          /*seq_id*/ 0, /*n_batch*/ (int) llama_n_batch(ctx),
                                          /*logits_last*/ true, &new_n_past);
        n_prompt = (int) mtmd_helper_get_n_tokens(chunks);
        mtmd_bitmap_free(bw.bitmap);
        mtmd_input_chunks_free(chunks);
        if (erc != 0) return -7;
    } else {
        // --- text-only prefill ---
        n_prompt = -llama_tokenize(vocab, prompt, (int) std::strlen(prompt),
                                   nullptr, 0, true, true);
        if (n_prompt <= 0) return -2;
        std::vector<llama_token> tokens((size_t) n_prompt);
        if (llama_tokenize(vocab, prompt, (int) std::strlen(prompt),
                           tokens.data(), (int) tokens.size(), true, true) < 0) {
            return -2;
        }
        if (prefill_tokens(ctx, tokens) != 0) return -3;
        if (use_spec) {
            // also prefill the draft model with the same prompt, and seed the
            // speculative state from the target's first predicted token.
            llama_memory_clear(llama_get_memory(h->draft_ctx), true);
            if (prefill_tokens(h->draft_ctx, tokens) != 0) return -3;
        }
    }

    h->gen.active = true;
    h->gen.remaining = max_tokens;
    h->gen.n_prompt = n_prompt;
    h->gen.n_gen = 0;
    h->gen.temperature = temperature;
    h->gen.top_p = top_p;
    h->gen.rng.seed(seed < 0 ? std::random_device{}() : (std::uint32_t) seed);

    if (use_spec) {
        // first confirmed token = target's greedy prediction after the prompt
        const float* logits = llama_get_logits_ith(ctx, -1);
        if (logits) {
            llama_token first = (llama_token) argmax(logits, h->n_vocab);
            h->gen.spec = true;
            h->gen.n_past = n_prompt;       // both KVs hold the prompt; first not cached
            h->gen.last_seed = first;
            if (!llama_vocab_is_eog(vocab, first) && max_tokens > 0) {
                h->gen.emitq.push_back(first);
                h->gen.n_gen = 1;
                h->gen.remaining = max_tokens - 1;
            } else {
                h->gen.eog_hit = true;
            }
        }
        // if logits were null, fall through as a (degenerate) non-spec stream
    }

    if (out_n_prompt) *out_n_prompt = n_prompt;
    return 0;
}

// Produce the next token's text into `out` (up to out_cap-1 bytes + NUL).
// Returns: >0 = bytes written; 0 = generation complete (EOS or budget reached);
// <0 = error. After 0 or <0 the stream is finished.
int mythd_llama_stream_next(void* handle, char* out, int out_cap) {
    auto* h = static_cast<MythdLlama*>(handle);
    if (!h || !h->model || !h->ctx) return -1;
    GenState& g = h->gen;
    if (!g.active) { if (out_cap > 0) out[0] = '\0'; return 0; }
    const llama_vocab* vocab = h->vocab;

    // ---- speculative path: emit from the confirmed-token queue, refilling via a
    // draft-and-verify round when it drains ----
    if (g.spec) {
        if (g.emit_pos >= g.emitq.size()) {
            if (g.eog_hit || g.remaining <= 0 || !spec_round(h)) {
                std::fprintf(stderr, "mythd-spec: drafted=%ld accepted=%ld accept-rate=%.0f%% gen=%d\n",
                             g.n_drafted, g.n_accepted,
                             g.n_drafted ? 100.0 * g.n_accepted / g.n_drafted : 0.0, g.n_gen);
                g.active = false; if (out_cap > 0) out[0] = '\0'; return 0;
            }
        }
        llama_token tok = g.emitq[g.emit_pos++];
        char piece[512];
        int np = llama_token_to_piece(vocab, tok, piece, (int) sizeof(piece), 0, true);
        if (np < 0) np = 0;
        int n = np; if (n > out_cap - 1) n = out_cap - 1; if (n < 0) n = 0;
        std::memcpy(out, piece, (size_t) n);
        out[n] = '\0';
        return n;
    }

    // ---- normal single-model path ----
    if (g.remaining <= 0) { g.active = false; if (out_cap > 0) out[0] = '\0'; return 0; }
    llama_context* ctx = h->ctx;
    const float* logits = llama_get_logits_ith(ctx, -1);
    if (!logits) { g.active = false; if (out_cap > 0) out[0] = '\0'; return 0; }

    llama_token tok =
        (llama_token) sample_token(logits, h->n_vocab, g.temperature, g.top_p, g.rng);
    if (llama_vocab_is_eog(vocab, tok)) { g.active = false; if (out_cap > 0) out[0] = '\0'; return 0; }

    char piece[512];
    int np = llama_token_to_piece(vocab, tok, piece, (int) sizeof(piece), 0, true);
    if (np < 0) np = 0;
    g.n_gen += 1;
    g.remaining -= 1;

    llama_batch step = llama_batch_get_one(&tok, 1);
    if (llama_decode(ctx, step) != 0) {
        g.active = false; // still return this token's piece; next call returns 0
    }

    int n = np;
    if (n > out_cap - 1) n = out_cap - 1;
    if (n < 0) n = 0;
    std::memcpy(out, piece, (size_t) n);
    out[n] = '\0';
    return n;
}

// Tokens generated so far in the current/just-finished stream (for the audit record).
int mythd_llama_stream_gen_count(void* handle) {
    auto* h = static_cast<MythdLlama*>(handle);
    return (h) ? h->gen.n_gen : 0;
}

// One-shot generation — built on the streaming primitives so the two paths share
// identical memory-clear + sampling behavior (no divergence). Writes up to
// out_cap-1 bytes + NUL into `out`; reports prompt/generated token counts via the
// optional out pointers. Returns the number of bytes written, or a negative error.
int mythd_llama_generate(void* handle, const char* prompt, int max_tokens,
                         float temperature, float top_p, long long seed,
                         const unsigned char* image_data, size_t image_len,
                         char* out, int out_cap, int* out_n_prompt, int* out_n_gen) {
    if (out_n_prompt) *out_n_prompt = 0;
    if (out_n_gen) *out_n_gen = 0;

    int n_prompt = 0;
    int rc = mythd_llama_stream_begin(handle, prompt, max_tokens, temperature, top_p,
                                      seed, image_data, image_len, &n_prompt);
    if (rc != 0) { if (out_cap > 0) out[0] = '\0'; return rc; }
    if (out_n_prompt) *out_n_prompt = n_prompt;

    std::string result;
    char piece[512];
    for (;;) {
        int b = mythd_llama_stream_next(handle, piece, (int) sizeof(piece));
        if (b < 0) { if (out_cap > 0) out[0] = '\0'; return b; }
        if (b == 0) break;
        result.append(piece, (size_t) b);
    }
    if (out_n_gen) *out_n_gen = mythd_llama_stream_gen_count(handle);

    int n = (int) result.size();
    if (n > out_cap - 1) n = out_cap - 1;
    if (n < 0) n = 0;
    std::memcpy(out, result.data(), (size_t) n);
    out[n] = '\0';
    return n;
}

void mythd_llama_free(void* handle) {
    auto* h = static_cast<MythdLlama*>(handle);
    if (!h) return;
    if (h->mtmd) mtmd_free(h->mtmd);
    if (h->draft_ctx) llama_free(h->draft_ctx);
    if (h->draft_model) llama_model_free(h->draft_model);
    if (h->ctx) llama_free(h->ctx);
    if (h->model) llama_model_free(h->model);
    delete h;
}

} // extern "C"
