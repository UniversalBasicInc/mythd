//! Safe wrapper over the `mythd_llama` C ABI shim (`kernels/mythd_llama.cpp`).
//!
//! All FFI — the only `unsafe` in Layer 3 — is confined to this module. The shim
//! exposes exactly four C functions taking C scalars + raw pointers; no by-value
//! llama.cpp structs cross the boundary, so the ABI risk that motivated the shim
//! does not reach Rust. Each `unsafe` block carries a `// SAFETY:` note.
//!
//! Without the `llama-cpp` feature the engine is a stub returning
//! [`InferenceError::EngineNotBuilt`], so the workspace builds anywhere.

use crate::{Generation, InferenceError, Result, Sampling};
use std::path::Path;

#[cfg(feature = "llama-cpp")]
mod ffi {
    use core::ffi::{c_char, c_float, c_int, c_longlong, c_uchar, c_void};

    // SAFETY (declaration contract): these signatures must match
    // `kernels/mythd_llama.cpp` exactly. Every argument is a C scalar or a raw
    // pointer — there are deliberately no C++/llama.cpp aggregate types here. The
    // shim, libllama, libggml and libmtmd are linked by `build.rs` under this feature.
    extern "C" {
        pub fn mythd_llama_init() -> c_int;
        pub fn mythd_llama_load(
            path: *const c_char,
            mmproj_path: *const c_char,
            n_gpu_layers: c_int,
            n_ctx: c_int,
            kv_q8: c_int,
        ) -> *mut c_void;
        pub fn mythd_llama_generate(
            handle: *mut c_void,
            prompt: *const c_char,
            max_tokens: c_int,
            temperature: c_float,
            top_p: c_float,
            seed: c_longlong,
            image_data: *const c_uchar,
            image_len: usize,
            out: *mut c_char,
            out_cap: c_int,
            out_n_prompt: *mut c_int,
            out_n_gen: *mut c_int,
        ) -> c_int;
        pub fn mythd_llama_stream_begin(
            handle: *mut c_void,
            prompt: *const c_char,
            max_tokens: c_int,
            temperature: c_float,
            top_p: c_float,
            seed: c_longlong,
            image_data: *const c_uchar,
            image_len: usize,
            out_n_prompt: *mut c_int,
        ) -> c_int;
        pub fn mythd_llama_stream_next(handle: *mut c_void, out: *mut c_char, out_cap: c_int)
            -> c_int;
        pub fn mythd_llama_stream_gen_count(handle: *mut c_void) -> c_int;
        pub fn mythd_llama_free(handle: *mut c_void);
    }
}

/// An owned llama.cpp model/context behind the shim. Not `Clone`: there is one
/// native context per engine, freed on drop.
pub struct LlamaEngine {
    #[cfg(feature = "llama-cpp")]
    handle: *mut core::ffi::c_void,
}

// SAFETY: the native handle is only ever dereferenced through `&mut self`, so the
// borrow checker already serializes access within a process. mythd's higher layers
// (L4 session manager / L5 server) additionally hold the engine behind a Mutex —
// one request touches the context at a time. No interior mutability escapes.
#[cfg(feature = "llama-cpp")]
unsafe impl Send for LlamaEngine {}

impl LlamaEngine {
    /// Initialize the llama.cpp backend (idempotent within a process).
    #[cfg(feature = "llama-cpp")]
    pub fn new() -> Result<Self> {
        // SAFETY: `mythd_llama_init` takes no arguments and only initializes the
        // llama.cpp backend; it returns 0 on success.
        let rc = unsafe { ffi::mythd_llama_init() };
        if rc != 0 {
            return Err(InferenceError::ModelLoad);
        }
        Ok(Self {
            handle: core::ptr::null_mut(),
        })
    }

    #[cfg(not(feature = "llama-cpp"))]
    pub fn new() -> Result<Self> {
        Err(InferenceError::EngineNotBuilt)
    }

    /// Load a GGUF model with `n_gpu_layers` offloaded and an `n_ctx` context.
    /// `mmproj` (optional) loads a vision projector so the model accepts images.
    /// llama.cpp's SYCL backend chooses per-tensor kernels by quant type.
    #[cfg(feature = "llama-cpp")]
    pub fn load(&mut self, path: &Path, mmproj: Option<&Path>, n_gpu_layers: i32, n_ctx: i32, kv_q8: bool) -> Result<()> {
        use std::ffi::CString;
        let path_str = path.to_str().ok_or(InferenceError::ModelLoad)?;
        let c_path = CString::new(path_str).map_err(|_| InferenceError::ModelLoad)?;
        // Empty CString → the shim treats it as "no mmproj" (text-only).
        let c_mmproj = match mmproj {
            Some(p) => CString::new(p.to_str().ok_or(InferenceError::ModelLoad)?)
                .map_err(|_| InferenceError::ModelLoad)?,
            None => CString::new("").unwrap(),
        };

        // Free any previously loaded model first.
        self.free_handle();

        // SAFETY: `c_path`/`c_mmproj` are valid NUL-terminated C strings that
        // outlive the call. The shim copies what it needs and returns an owned
        // opaque handle (or null on failure). The scalars are passed by value.
        let handle = unsafe {
            ffi::mythd_llama_load(
                c_path.as_ptr(),
                c_mmproj.as_ptr(),
                n_gpu_layers,
                n_ctx,
                if kv_q8 { 1 } else { 0 },
            )
        };
        if handle.is_null() {
            return Err(InferenceError::ModelLoad);
        }
        self.handle = handle;
        Ok(())
    }

    #[cfg(not(feature = "llama-cpp"))]
    pub fn load(&mut self, _path: &Path, _mmproj: Option<&Path>, _n_gpu_layers: i32, _n_ctx: i32, _kv_q8: bool) -> Result<()> {
        Err(InferenceError::EngineNotBuilt)
    }

    /// Decode up to `max_tokens` tokens from `prompt` with the given sampling
    /// params (`temperature <= 0` → greedy; `seed < 0` → random), reporting the
    /// prompt/generated token counts (which feed the L4 signed audit record).
    #[cfg(feature = "llama-cpp")]
    pub fn generate(&mut self, prompt: &str, max_tokens: i32, sampling: Sampling, image: Option<&[u8]>) -> Result<Generation> {
        use core::ffi::{c_float, c_int, c_longlong};
        use std::ffi::CString;
        if self.handle.is_null() {
            return Err(InferenceError::NoModel);
        }
        let c_prompt = CString::new(prompt).map_err(|_| InferenceError::BadPrompt)?;
        let (img_ptr, img_len) = match image {
            Some(b) if !b.is_empty() => (b.as_ptr() as *const core::ffi::c_uchar, b.len()),
            _ => (core::ptr::null(), 0usize),
        };

        // Generous output cap; generation stops at EOS or max_tokens regardless.
        let cap: usize = (max_tokens.max(1) as usize)
            .saturating_mul(32)
            .max(8192)
            + 1;
        let mut out = vec![0u8; cap];
        let mut n_prompt: c_int = 0;
        let mut n_gen: c_int = 0;

        // SAFETY: `c_prompt` is a valid NUL-terminated C string for the call.
        // `out` is a writable buffer of `cap` bytes; the shim writes at most
        // `out_cap - 1` bytes plus a NUL terminator and returns the byte count
        // (or a negative error code). `n_prompt`/`n_gen` are valid `c_int`
        // out-pointers the shim writes the token counts into. The sampling params
        // are plain scalars. `self.handle` is a live, non-null handle produced by
        // `mythd_llama_load` and not yet freed.
        let n = unsafe {
            ffi::mythd_llama_generate(
                self.handle,
                c_prompt.as_ptr(),
                max_tokens,
                sampling.temperature as c_float,
                sampling.top_p as c_float,
                sampling.seed as c_longlong,
                img_ptr,
                img_len,
                out.as_mut_ptr() as *mut core::ffi::c_char,
                cap as c_int,
                &mut n_prompt,
                &mut n_gen,
            )
        };
        if n < 0 {
            return Err(InferenceError::Generate(n));
        }
        out.truncate(n as usize);
        Ok(Generation {
            text: String::from_utf8_lossy(&out).into_owned(),
            prompt_tokens: n_prompt.max(0) as u32,
            generated_tokens: n_gen.max(0) as u32,
        })
    }

    #[cfg(not(feature = "llama-cpp"))]
    pub fn generate(&mut self, _prompt: &str, _max_tokens: i32, _sampling: Sampling, _image: Option<&[u8]>) -> Result<Generation> {
        Err(InferenceError::EngineNotBuilt)
    }

    /// Begin a streaming generation (optionally with an image). Returns the prompt
    /// token count. Follow with repeated [`stream_next`](Self::stream_next).
    #[cfg(feature = "llama-cpp")]
    pub fn stream_begin(&mut self, prompt: &str, max_tokens: i32, sampling: Sampling, image: Option<&[u8]>) -> Result<u32> {
        use core::ffi::{c_float, c_int, c_longlong};
        use std::ffi::CString;
        if self.handle.is_null() {
            return Err(InferenceError::NoModel);
        }
        let c_prompt = CString::new(prompt).map_err(|_| InferenceError::BadPrompt)?;
        let (img_ptr, img_len) = match image {
            Some(b) if !b.is_empty() => (b.as_ptr() as *const core::ffi::c_uchar, b.len()),
            _ => (core::ptr::null(), 0usize),
        };
        let mut n_prompt: c_int = 0;
        // SAFETY: `c_prompt` is a valid NUL-terminated C string for the call;
        // `img_ptr`/`img_len` describe an optional image buffer valid for the call;
        // `n_prompt` is a valid out-pointer; sampling args are plain scalars;
        // `self.handle` is a live handle from `mythd_llama_load`. The shim stores
        // generation state in the handle for subsequent `stream_next` calls.
        let rc = unsafe {
            ffi::mythd_llama_stream_begin(
                self.handle,
                c_prompt.as_ptr(),
                max_tokens,
                sampling.temperature as c_float,
                sampling.top_p as c_float,
                sampling.seed as c_longlong,
                img_ptr,
                img_len,
                &mut n_prompt,
            )
        };
        if rc != 0 {
            return Err(InferenceError::Generate(rc));
        }
        Ok(n_prompt.max(0) as u32)
    }

    /// Produce the next token's text, or `None` when the stream is complete.
    /// Must be preceded by [`stream_begin`](Self::stream_begin).
    #[cfg(feature = "llama-cpp")]
    pub fn stream_next(&mut self) -> Result<Option<String>> {
        use core::ffi::c_int;
        if self.handle.is_null() {
            return Err(InferenceError::NoModel);
        }
        // A token piece is small; this buffer matches the shim's internal cap.
        let mut buf = [0u8; 512];
        // SAFETY: `buf` is a writable 512-byte buffer; the shim writes at most
        // `out_cap - 1` bytes + NUL and returns the byte count (>0), 0 (done), or
        // <0 (error). `self.handle` is a live handle mid-stream.
        let n = unsafe {
            ffi::mythd_llama_stream_next(self.handle, buf.as_mut_ptr() as *mut core::ffi::c_char, buf.len() as c_int)
        };
        if n < 0 {
            return Err(InferenceError::Generate(n));
        }
        if n == 0 {
            return Ok(None);
        }
        Ok(Some(String::from_utf8_lossy(&buf[..n as usize]).into_owned()))
    }

    /// Tokens generated so far in the current/just-finished stream.
    #[cfg(feature = "llama-cpp")]
    pub fn stream_gen_count(&self) -> u32 {
        if self.handle.is_null() {
            return 0;
        }
        // SAFETY: read-only query against a live handle; no aliasing.
        let n = unsafe { ffi::mythd_llama_stream_gen_count(self.handle) };
        n.max(0) as u32
    }

    #[cfg(not(feature = "llama-cpp"))]
    pub fn stream_begin(&mut self, _prompt: &str, _max_tokens: i32, _sampling: Sampling, _image: Option<&[u8]>) -> Result<u32> {
        Err(InferenceError::EngineNotBuilt)
    }
    #[cfg(not(feature = "llama-cpp"))]
    pub fn stream_next(&mut self) -> Result<Option<String>> {
        Err(InferenceError::EngineNotBuilt)
    }
    #[cfg(not(feature = "llama-cpp"))]
    pub fn stream_gen_count(&self) -> u32 {
        0
    }

    #[cfg(feature = "llama-cpp")]
    fn free_handle(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: `self.handle` was returned by `mythd_llama_load` and has not
            // been freed; the shim frees the llama context/model and we null it so
            // it is never double-freed.
            unsafe { ffi::mythd_llama_free(self.handle) };
            self.handle = core::ptr::null_mut();
        }
    }
}

#[cfg(feature = "llama-cpp")]
impl Drop for LlamaEngine {
    fn drop(&mut self) {
        self.free_handle();
    }
}
