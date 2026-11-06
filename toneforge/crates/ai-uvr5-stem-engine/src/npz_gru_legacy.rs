//! Pure-Rust NumPy `.npz` / `.npy` Parser (`\x93NUMPY`), Ozone 12 Master Assistant
//! `gparamsBN8.0.npz` BatchNorm CNN Genre Classifier, and 2-Layer `GRU-512` Legacy Stem Engine.

use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::path::Path;
use zip::ZipArchive;

/// Candidate paths for `gparamsBN8.0.npz` (`49,338` bytes).
pub const DEFAULT_GPARAMS_NPZ_PATHS: [&str; 4] = [
    r"f:\FL studio\toneforge\assets\ozone12\models\gparamsBN8.0.npz",
    r"f:\FL studio\toneforge\assets\models\gparamsBN8.0.npz",
    r"assets/ozone12/models/gparamsBN8.0.npz",
    r"assets/models/gparamsBN8.0.npz",
];

/// Parsed `f32` tensor from a NumPy `.npy` file inside a `.npz` archive.
#[derive(Clone, Debug)]
pub struct NumpyArrayF32 {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl NumpyArrayF32 {
    /// Parses a single `.npy` binary buffer (`\x93NUMPY` magic `0x93, b'N', b'U', b'M', b'P', b'Y'`).
    pub fn from_npy_bytes(name: &str, bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
            return Err(format!("Invalid .npy magic in entry '{}'", name));
        }
        let major = bytes[6];
        let (header_len, data_offset) = if major == 1 {
            let h_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
            (h_len, 10 + h_len)
        } else {
            if bytes.len() < 12 {
                return Err("Truncated .npy v2 header".to_string());
            }
            let h_len =
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            (h_len, 12 + h_len)
        };

        if bytes.len() < data_offset {
            return Err("Truncated .npy payload".to_string());
        }

        let header_str = String::from_utf8_lossy(&bytes[data_offset - header_len..data_offset]);
        let is_f64 = header_str.contains("<f8") || header_str.contains("float64");
        let shape = Self::parse_shape(&header_str);

        let raw_payload = &bytes[data_offset..];
        let data = if is_f64 {
            let count = raw_payload.len() / 8;
            let mut v = Vec::with_capacity(count);
            for i in 0..count {
                let off = i * 8;
                let val = f64::from_le_bytes(raw_payload[off..off + 8].try_into().unwrap());
                v.push(val as f32);
            }
            v
        } else {
            let count = raw_payload.len() / 4;
            let mut v = Vec::with_capacity(count);
            for i in 0..count {
                let off = i * 4;
                let val = f32::from_le_bytes(raw_payload[off..off + 4].try_into().unwrap());
                v.push(val);
            }
            v
        };

        Ok(Self {
            name: name.to_string(),
            shape,
            data,
        })
    }

    fn parse_shape(header: &str) -> Vec<usize> {
        if let Some(pos) = header.find("'shape'") {
            let rest = &header[pos..];
            if let (Some(lp), Some(rp)) = (rest.find('('), rest.find(')')) {
                if lp < rp {
                    let inside = &rest[lp + 1..rp];
                    let dims: Vec<usize> = inside
                        .split(',')
                        .filter_map(|s| s.trim().parse::<usize>().ok())
                        .collect();
                    if !dims.is_empty() {
                        return dims;
                    }
                }
            }
        }
        Vec::new()
    }
}

/// Loads all `.npy` arrays from a `.npz` archive in memory or on disk.
pub fn load_npz_archive_bytes(bytes: &[u8]) -> Result<HashMap<String, NumpyArrayF32>, String> {
    let cursor = Cursor::new(bytes);
    let mut archive = ZipArchive::new(cursor).map_err(|e| format!("Invalid .npz zip: {}", e))?;
    let mut map = HashMap::new();

    for i in 0..archive.len() {
        let mut file = archive.by_index(i).map_err(|e| e.to_string())?;
        let raw_name = file.name().trim_end_matches(".npy").to_string();
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).map_err(|e| e.to_string())?;
        if let Ok(arr) = NumpyArrayF32::from_npy_bytes(&raw_name, &buf) {
            map.insert(raw_name, arr);
        }
    }

    if map.is_empty() {
        return Err("No valid .npy tensors found in .npz archive".to_string());
    }
    Ok(map)
}

/// Loads all `.npy` arrays from a `.npz` file on disk (`gparamsBN8.0.npz`).
pub fn load_npz_archive_file(path: &Path) -> Result<HashMap<String, NumpyArrayF32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("Failed to read {:?}: {}", path, e))?;
    load_npz_archive_bytes(&bytes)
}

/// Ozone 12 Master Assistant Genre Classifier (`gparamsBN8.0.npz`).
pub struct GenreCnnClassifier {
    pub tensors: HashMap<String, NumpyArrayF32>,
}

impl GenreCnnClassifier {
    pub fn from_npz_file(path: &Path) -> Result<Self, String> {
        let tensors = load_npz_archive_file(path)?;
        Ok(Self { tensors })
    }

    pub fn load_default() -> Option<Self> {
        for cand in DEFAULT_GPARAMS_NPZ_PATHS {
            let p = Path::new(cand);
            if p.exists() {
                if let Ok(c) = Self::from_npz_file(p) {
                    return Some(c);
                }
            }
        }
        None
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}
