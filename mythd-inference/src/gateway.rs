//! Gateway client — forwards inference to a local `llama-server` over loopback.
//!
//! This is what gives mythd real concurrency. `llama-server` owns the model
//! weights, the per-slot KV cache, and the **continuous-batching scheduler**;
//! mythd stays the secure front door (PQC handshake, bearer auth, ML-DSA-signed
//! audit). The old embedded path decoded one sequence at a time behind a global
//! lock — this path lets many requests run as concurrent llama-server slots.
//!
//! The blocking reqwest client is created once and shared via `&self`, so many
//! forwards run at once. Every call must run on a blocking thread (the API layer
//! wraps them in `spawn_blocking`) — never on the async reactor.
//!
//! Messages are forwarded verbatim to `/v1/chat/completions`, so `llama-server`
//! applies the model's own chat template (`--jinja`) and handles vision via
//! `mtmd` (inline `data:` image URLs). mythd no longer hand-rolls ChatML.

use crate::{Generation, InferenceError, InferenceRequest, Result};
use std::io::BufRead;

/// HTTP client to a `llama-server` instance (OpenAI-compatible API).
pub struct LlamaServerClient {
    http: reqwest::blocking::Client,
    base_url: String,
}

impl LlamaServerClient {
    /// Configure the client. Makes **no** network call (so it is safe to build
    /// from within the async runtime at startup); reachability is exercised on the
    /// first request.
    pub fn new(base_url: impl Into<String>) -> Result<Self> {
        let http = reqwest::blocking::Client::builder()
            .build()
            .map_err(|_| InferenceError::ModelLoad)?;
        Ok(Self {
            http,
            base_url: base_url.into().trim_end_matches('/').to_string(),
        })
    }

    fn chat_url(&self) -> String {
        format!("{}/v1/chat/completions", self.base_url)
    }

    /// Build the OpenAI chat body. `stream` toggles SSE (with usage in the final
    /// chunk so the audit gets real token counts).
    fn body(&self, req: &InferenceRequest, stream: bool) -> serde_json::Value {
        let s = &req.sampling;
        let mut b = serde_json::json!({
            "messages": req.messages,
            "max_tokens": req.max_new_tokens,
            "stream": stream,
            // temperature <= 0 ⇒ deterministic/greedy (mythd's convention).
            "temperature": s.temperature.max(0.0),
            "top_p": s.top_p,
        });
        if s.seed >= 0 {
            b["seed"] = serde_json::json!(s.seed);
        }
        if stream {
            b["stream_options"] = serde_json::json!({ "include_usage": true });
        }
        b
    }

    /// Non-streaming generation.
    pub fn infer(&self, req: &InferenceRequest) -> Result<Generation> {
        let resp = self
            .http
            .post(self.chat_url())
            .json(&self.body(req, false))
            .send()
            .map_err(|e| {
                tracing::warn!(target: "mythd::gateway", error = %e, "llama-server request failed");
                InferenceError::Generate(-1)
            })?;
        if !resp.status().is_success() {
            let code = resp.status().as_u16() as i32;
            tracing::warn!(target: "mythd::gateway", status = code, "llama-server non-2xx");
            return Err(InferenceError::Generate(code));
        }
        let v: serde_json::Value = resp.json().map_err(|_| InferenceError::Generate(-2))?;
        let text = v["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string();
        let prompt_tokens = v["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32;
        let generated_tokens = v["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32;
        Ok(Generation { text, prompt_tokens, generated_tokens })
    }

    /// Streaming generation: parse `llama-server`'s SSE, invoke `on_token` for each
    /// content delta. `on_token` returns `false` → stop (client disconnected).
    /// Returns the assembled text + token counts (from the final usage chunk; falls
    /// back to the delta count if the server omits usage).
    pub fn infer_stream(
        &self,
        req: &InferenceRequest,
        mut on_token: impl FnMut(&str) -> bool,
    ) -> Result<Generation> {
        let resp = self
            .http
            .post(self.chat_url())
            .json(&self.body(req, true))
            .send()
            .map_err(|e| {
                tracing::warn!(target: "mythd::gateway", error = %e, "llama-server stream request failed");
                InferenceError::Generate(-1)
            })?;
        if !resp.status().is_success() {
            return Err(InferenceError::Generate(resp.status().as_u16() as i32));
        }

        let mut text = String::new();
        let mut prompt_tokens = 0u32;
        let mut generated_tokens = 0u32;
        let mut n_deltas = 0u32;
        let reader = std::io::BufReader::new(resp);
        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => break,
            };
            let data = match line.strip_prefix("data:") {
                Some(d) => d.trim(),
                None => continue,
            };
            if data == "[DONE]" {
                break;
            }
            let chunk: serde_json::Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };
            // Final usage chunk (we request include_usage) → real token counts.
            if let Some(u) = chunk.get("usage") {
                if !u.is_null() {
                    prompt_tokens = u["prompt_tokens"].as_u64().unwrap_or(prompt_tokens as u64) as u32;
                    generated_tokens =
                        u["completion_tokens"].as_u64().unwrap_or(generated_tokens as u64) as u32;
                }
            }
            if let Some(piece) = chunk["choices"][0]["delta"]["content"].as_str() {
                if !piece.is_empty() {
                    text.push_str(piece);
                    n_deltas += 1;
                    if !on_token(piece) {
                        tracing::debug!(target: "mythd::gateway", "stream consumer cancelled");
                        break;
                    }
                }
            }
        }
        if generated_tokens == 0 {
            generated_tokens = n_deltas; // fallback if the server omitted usage
        }
        Ok(Generation { text, prompt_tokens, generated_tokens })
    }
}
