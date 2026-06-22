//! Safe model loading.
//!
//! SECURITY INVARIANT: the only model format mythd can parse is `safetensors`
//! (a length-prefixed JSON header followed by raw tensor bytes — no code is ever
//! executed during parsing). There is deliberately **no** pickle / `torch.load`
//! / numpy `.npy`-object / arbitrary-deserialization path anywhere in this
//! binary. A pickle handed to this loader is rejected before parsing and, even
//! if it weren't, there is no deserializer that could execute it.

use crate::error::{CoreError, Result};
use safetensors::SafeTensors;
use std::path::Path;

/// A model file whose bytes have been read and validated as well-formed
/// safetensors. The raw buffer is retained so individual tensors can be borrowed
/// on demand (`tensors()`); re-parsing the header is cheap.
pub struct LoadedModel {
    raw: Vec<u8>,
    source: String,
}

impl LoadedModel {
    /// Read and validate a `.safetensors` file. Returns an error (never panics,
    /// never executes file contents) for anything that is not well-formed
    /// safetensors.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw = std::fs::read(path)?;
        reject_if_not_safetensors(&raw, path)?;

        // Validate by fully parsing the header. `SafeTensors::deserialize` only
        // reads the JSON metadata and bounds-checks tensor offsets; it does not
        // execute anything. We discard the borrow and keep the buffer.
        SafeTensors::deserialize(&raw)
            .map_err(|e| CoreError::SafeTensors(e.to_string()))?;

        tracing::info!(target: "mythd::loader", model = %path.display(), bytes = raw.len(), "loaded safetensors model");
        Ok(Self { raw, source: path.display().to_string() })
    }

    /// Borrow the parsed tensor view (header re-parsed; cheap).
    pub fn tensors(&self) -> Result<SafeTensors<'_>> {
        SafeTensors::deserialize(&self.raw).map_err(|e| CoreError::SafeTensors(e.to_string()))
    }

    /// Names of the tensors in the model.
    pub fn tensor_names(&self) -> Result<Vec<String>> {
        Ok(self.tensors()?.names().into_iter().map(|s| s.to_string()).collect())
    }

    pub fn source(&self) -> &str {
        &self.source
    }
}

impl core::fmt::Debug for LoadedModel {
    /// Redacting: never dumps the raw tensor bytes.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LoadedModel")
            .field("source", &self.source)
            .field("bytes", &self.raw.len())
            .finish()
    }
}

/// Reject obvious non-safetensors inputs before parsing. This is belt-and-
/// suspenders on top of the architectural guarantee (no pickle deserializer
/// exists): we additionally refuse files bearing the Python pickle magic and
/// require a plausible safetensors header.
fn reject_if_not_safetensors(bytes: &[u8], path: &Path) -> Result<()> {
    // A safetensors file starts with an 8-byte little-endian header length,
    // followed by a UTF-8 JSON object beginning with '{'.
    if bytes.len() < 9 {
        return Err(CoreError::RejectedModel(format!(
            "{}: too small to be safetensors",
            path.display()
        )));
    }

    // Python pickle protocol 2+ starts with 0x80; protocol 0/1 commonly start
    // with '(' (0x28), 'c' (0x63), or '}' (0x7d). None of these are a valid
    // safetensors header-length leading byte for a real model (the header length
    // is small relative to the file, so byte 0 is the low byte of that length).
    let b0 = bytes[0];
    if b0 == 0x80 {
        return Err(CoreError::RejectedModel(format!(
            "{}: starts with the Python pickle magic (0x80) — refused",
            path.display()
        )));
    }

    // Sanity-check the declared header length against the file size.
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().expect("8 bytes")) as usize;
    if header_len == 0 || header_len.saturating_add(8) > bytes.len() {
        return Err(CoreError::RejectedModel(format!(
            "{}: implausible safetensors header length ({header_len})",
            path.display()
        )));
    }
    if bytes.get(8) != Some(&b'{') {
        return Err(CoreError::RejectedModel(format!(
            "{}: header is not a JSON object — not safetensors",
            path.display()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp(bytes: &[u8]) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("mythd_test_{}.bin", std::process::id().wrapping_add(bytes.len() as u32)));
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(bytes).unwrap();
        p
    }

    #[test]
    fn rejects_pickle_magic() {
        let p = tmp(&[0x80, 0x04, 0x95, 0, 0, 0, 0, 0, 0, 0]);
        let err = LoadedModel::load(&p).unwrap_err();
        assert!(matches!(err, CoreError::RejectedModel(_)));
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn loads_minimal_safetensors() {
        // Minimal valid safetensors: header_len + {"__metadata__":{}}
        let header = br#"{"__metadata__":{}}"#;
        let mut buf = (header.len() as u64).to_le_bytes().to_vec();
        buf.extend_from_slice(header);
        let p = tmp(&buf);
        let m = LoadedModel::load(&p).expect("should load");
        assert!(m.tensor_names().unwrap().is_empty());
        let _ = std::fs::remove_file(p);
    }
}
