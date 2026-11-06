//! Pure-Rust NumPy `.npz` / `.npy` Parser (`\x93NUMPY`), Direct ZIP `STORED` + `ZIP64`
//! `NpzArchiveReader`, Ozone 12 Master Assistant `gparamsBN8.0.npz` &
//! `instrument_classifier_params.npz` CNN Classifiers, and `OzoneNpzGruEngine`
//! (`NeuralNet::GRU` at `0x1834213A0..0x18342202D` + `NeuralNet::Dense` at `0x18572E088`).

use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::path::Path;
use zip::ZipArchive;

/// Candidate paths for `gparamsBN8.0.npz` (`6,522,880` bytes).
pub const DEFAULT_GPARAMS_NPZ_PATHS: [&str; 6] = [
    r"f:\FL studio\toneforge\assets\ozone12\models\gparamsBN8.0.npz",
    r"f:\FL studio\toneforge\assets\models\assistant_cnn\gparamsBN8.0.npz",
    r"f:\FL studio\toneforge\assets\models\gparamsBN8.0.npz",
    r"assets/ozone12/models/gparamsBN8.0.npz",
    r"assets/models/assistant_cnn/gparamsBN8.0.npz",
    r"assets/models/gparamsBN8.0.npz",
];

/// Candidate paths for `instrument_classifier_params.npz` (`19,853,709` bytes).
pub const DEFAULT_INSTRUMENT_NPZ_PATHS: [&str; 6] = [
    r"f:\FL studio\toneforge\assets\ozone12\models\instrument_classifier_params.npz",
    r"f:\FL studio\toneforge\assets\models\assistant_cnn\instrument_classifier_params.npz",
    r"f:\FL studio\toneforge\assets\models\instrument_classifier_params.npz",
    r"assets/ozone12/models/instrument_classifier_params.npz",
    r"assets/models/assistant_cnn/instrument_classifier_params.npz",
    r"assets/models/instrument_classifier_params.npz",
];

/// Candidate paths for `vocals-small_sanitized.npz` (`26,261,516` bytes).
pub const DEFAULT_VOCALS_GRU_NPZ_PATHS: [&str; 6] = [
    r"f:\FL studio\toneforge\assets\ozone12\models\vocals-small_sanitized.npz",
    r"f:\FL studio\toneforge\assets\models\npz_gru_stems\vocals-small_sanitized.npz",
    r"f:\FL studio\extracted_targets\ozone12\requiredFiles_component\ssepnpz\vocals-small_sanitized.npz",
    r"assets/ozone12/models/vocals-small_sanitized.npz",
    r"assets/models/npz_gru_stems/vocals-small_sanitized.npz",
    r"assets/models/vocals-small_sanitized.npz",
];

/// Local File Header signature in a ZIP / `.npz` archive (`PK\x03\x04`).
pub const ZIP_LOCAL_FILE_HEADER_MAGIC: u32 = 0x0403_4B50;
/// ZIP64 Extended Information Extra Field header ID (`0x0001`).
pub const ZIP64_EXTRA_HEADER_ID: u16 = 0x0001;

/// Parsed `f32` tensor from a NumPy `.npy` file inside a `.npz` archive.
#[derive(Clone, Debug)]
pub struct NumpyArrayF32 {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl NumpyArrayF32 {
    /// Parses a single `.npy` binary buffer (`\x93NUMPY` magic `0x93, b'N', b'U', b'M', b'P', b'Y'`).
    /// Supports `<f4` (`float32`), `<f8` (`float64`), `<i8` (`int64`), and `<i4` (`int32`).
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
            let h_len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            (h_len, 12 + h_len)
        };

        if bytes.len() < data_offset {
            return Err("Truncated .npy payload".to_string());
        }

        let header_str = String::from_utf8_lossy(&bytes[data_offset - header_len..data_offset]);
        let is_f64 = header_str.contains("<f8") || header_str.contains("float64");
        let is_i64 = header_str.contains("<i8") || header_str.contains("int64");
        let is_i32 = header_str.contains("<i4") || header_str.contains("int32");
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
        } else if is_i64 {
            let count = raw_payload.len() / 8;
            let mut v = Vec::with_capacity(count);
            for i in 0..count {
                let off = i * 8;
                let val = i64::from_le_bytes(raw_payload[off..off + 8].try_into().unwrap());
                v.push(val as f32);
            }
            v
        } else if is_i32 {
            let count = raw_payload.len() / 4;
            let mut v = Vec::with_capacity(count);
            for i in 0..count {
                let off = i * 4;
                let val = i32::from_le_bytes(raw_payload[off..off + 4].try_into().unwrap());
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

/// Direct Clean-Room reader for NumPy `.npz` archives (`0x181B55500..0x181B55B00`).
///
/// Directly scans ZIP Local File Headers (`0x04034B50`) with `STORED` (`method = 0`) and
/// `ZIP64` Extra Fields (`0x0001`, 64-bit `uncompressed_size` / `compressed_size`) without
/// requiring external decompression dependencies, while transparently supporting DEFLATE
/// entries if present.
#[derive(Clone, Debug)]
pub struct NpzArchiveReader {
    pub tensors: HashMap<String, NumpyArrayF32>,
}

impl NpzArchiveReader {
    /// Parses a `.npz` byte slice by walking ZIP Local File Headers (`0x04034B50`) and ZIP64 extra fields.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut tensors = HashMap::new();
        let mut cursor = 0usize;
        let mut needs_deflate_fallback = false;

        while cursor + 30 <= bytes.len() {
            let sig = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
            if sig != ZIP_LOCAL_FILE_HEADER_MAGIC {
                // Reached Central Directory (`0x02014B50`) or End of Central Directory (`0x06054B50`)
                break;
            }

            let compression_method =
                u16::from_le_bytes(bytes[cursor + 8..cursor + 10].try_into().unwrap());
            let mut compressed_size =
                u32::from_le_bytes(bytes[cursor + 18..cursor + 22].try_into().unwrap()) as u64;
            let mut uncompressed_size =
                u32::from_le_bytes(bytes[cursor + 22..cursor + 26].try_into().unwrap()) as u64;
            let file_name_len =
                u16::from_le_bytes(bytes[cursor + 26..cursor + 28].try_into().unwrap()) as usize;
            let extra_field_len =
                u16::from_le_bytes(bytes[cursor + 28..cursor + 30].try_into().unwrap()) as usize;

            let name_start = cursor + 30;
            let extra_start = name_start + file_name_len;
            let data_start = extra_start + extra_field_len;
            if data_start > bytes.len() {
                return Err("Truncated ZIP local file header in .npz archive".to_string());
            }

            let raw_name = String::from_utf8_lossy(&bytes[name_start..extra_start])
                .trim_end_matches(".npy")
                .to_string();

            // Parse ZIP64 Extended Information Extra Field (`0x0001`) if 32-bit sizes are `0xFFFF_FFFF`
            let extra = &bytes[extra_start..data_start];
            let mut ex_pos = 0usize;
            while ex_pos + 4 <= extra.len() {
                let header_id = u16::from_le_bytes(extra[ex_pos..ex_pos + 2].try_into().unwrap());
                let data_size =
                    u16::from_le_bytes(extra[ex_pos + 2..ex_pos + 4].try_into().unwrap()) as usize;
                let field_start = ex_pos + 4;
                let field_end = (field_start + data_size).min(extra.len());
                if header_id == ZIP64_EXTRA_HEADER_ID {
                    let mut z_off = field_start;
                    if uncompressed_size == 0xFFFF_FFFF && z_off + 8 <= field_end {
                        uncompressed_size =
                            u64::from_le_bytes(extra[z_off..z_off + 8].try_into().unwrap());
                        z_off += 8;
                    }
                    if compressed_size == 0xFFFF_FFFF && z_off + 8 <= field_end {
                        compressed_size =
                            u64::from_le_bytes(extra[z_off..z_off + 8].try_into().unwrap());
                    }
                }
                ex_pos = field_end;
            }

            if compression_method != 0 || compressed_size == 0xFFFF_FFFF {
                needs_deflate_fallback = true;
                break;
            }

            let payload_len = compressed_size as usize;
            let data_end = data_start + payload_len;
            if data_end > bytes.len() {
                return Err(format!("Truncated .npy entry '{}' in .npz archive", raw_name));
            }

            let npy_slice = &bytes[data_start..data_end];
            if let Ok(arr) = NumpyArrayF32::from_npy_bytes(&raw_name, npy_slice) {
                tensors.insert(raw_name, arr);
            }
            cursor = data_end;
        }

        if needs_deflate_fallback || tensors.is_empty() {
            let cursor_io = Cursor::new(bytes);
            let mut archive =
                ZipArchive::new(cursor_io).map_err(|e| format!("Invalid .npz zip: {}", e))?;
            for i in 0..archive.len() {
                let mut file = archive.by_index(i).map_err(|e| e.to_string())?;
                let raw_name = file.name().trim_end_matches(".npy").to_string();
                let mut buf = Vec::new();
                file.read_to_end(&mut buf).map_err(|e| e.to_string())?;
                if let Ok(arr) = NumpyArrayF32::from_npy_bytes(&raw_name, &buf) {
                    tensors.insert(raw_name, arr);
                }
            }
        }

        if tensors.is_empty() {
            return Err("No valid .npy tensors found in .npz archive".to_string());
        }

        Ok(Self { tensors })
    }

    /// Reads and parses a `.npz` file from disk.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let bytes =
            std::fs::read(path).map_err(|e| format!("Failed to read {:?}: {}", path, e))?;
        Self::from_bytes(&bytes)
    }

    #[inline]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    #[inline]
    pub fn get(&self, name: &str) -> Option<&NumpyArrayF32> {
        let clean = name.trim_end_matches(".npy");
        self.tensors.get(clean)
    }

    pub fn get_scalar_f32(&self, name: &str) -> Option<f32> {
        self.get(name).and_then(|t| t.data.first().copied())
    }

    pub fn get_scalar_i64(&self, name: &str) -> Option<i64> {
        self.get_scalar_f32(name).map(|v| v.round() as i64)
    }
}

/// Loads all `.npy` arrays from a `.npz` archive in memory.
pub fn load_npz_archive_bytes(bytes: &[u8]) -> Result<HashMap<String, NumpyArrayF32>, String> {
    Ok(NpzArchiveReader::from_bytes(bytes)?.tensors)
}

/// Loads all `.npy` arrays from a `.npz` file on disk.
pub fn load_npz_archive_file(path: &Path) -> Result<HashMap<String, NumpyArrayF32>, String> {
    Ok(NpzArchiveReader::from_file(path)?.tensors)
}

/// Single `NeuralNet::GRU` layer (`0x1834213A0..0x18342202D`, 10 `.npy` arrays per layer).
#[derive(Clone, Debug)]
pub struct GruLayerWeights {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub w_r: Vec<f32>,
    pub u_r: Vec<f32>,
    pub b_r: Vec<f32>,
    pub w_z: Vec<f32>,
    pub u_z: Vec<f32>,
    pub b_z: Vec<f32>,
    pub w_c: Vec<f32>,
    pub u_c: Vec<f32>,
    pub b_c: Vec<f32>,
    pub h_0: Vec<f32>,
}

impl GruLayerWeights {
    pub fn from_npz(
        reader: &NpzArchiveReader,
        base_arr_idx: usize,
        expected_in: usize,
        expected_hid: usize,
    ) -> Result<Self, String> {
        let fetch = |offset: usize, expected_len: usize| -> Result<Vec<f32>, String> {
            let key = format!("arr_{}", base_arr_idx + offset);
            let arr = reader
                .get(&key)
                .ok_or_else(|| format!("Missing '{}' in GRU .npz", key))?;
            if arr.data.len() != expected_len {
                return Err(format!(
                    "Tensor '{}' length mismatch: got {}, expected {}",
                    key,
                    arr.data.len(),
                    expected_len
                ));
            }
            Ok(arr.data.clone())
        };

        let in_hid = expected_in * expected_hid;
        let hid_hid = expected_hid * expected_hid;
        Ok(Self {
            input_dim: expected_in,
            hidden_dim: expected_hid,
            w_r: fetch(0, in_hid)?,
            u_r: fetch(1, hid_hid)?,
            b_r: fetch(2, expected_hid)?,
            w_z: fetch(3, in_hid)?,
            u_z: fetch(4, hid_hid)?,
            b_z: fetch(5, expected_hid)?,
            w_c: fetch(6, in_hid)?,
            u_c: fetch(7, hid_hid)?,
            b_c: fetch(8, expected_hid)?,
            h_0: fetch(9, expected_hid)?,
        })
    }

    /// Executes one recurrent time-step `NeuralNet::GRU::forward` (`0x183421790..0x183421EF0`):
    /// - `r_t = sigmoid(x_t * W_r + h_{t-1} * U_r + b_r)`
    /// - `z_t = sigmoid(x_t * W_z + h_{t-1} * U_z + b_z)`
    /// - `c_t = tanh(x_t * W_c + (r_t .* h_{t-1}) * U_c + b_c)` (Theano/Lasagne convention: `r_t` applied before `U_c`)
    /// - `h_t = (1 - z_t) .* c_t + z_t .* h_{t-1}`
    pub fn forward_step(&self, x: &[f32], h_state: &mut [f32]) {
        const MAX_STACK_HID: usize = 512;
        let in_dim = self.input_dim.min(x.len());
        let hid_dim = self.hidden_dim.min(h_state.len()).min(MAX_STACK_HID);

        let mut r_gate = [0.0f32; MAX_STACK_HID];
        let mut z_gate = [0.0f32; MAX_STACK_HID];
        let mut wx_c = [0.0f32; MAX_STACK_HID];
        let mut rh = [0.0f32; MAX_STACK_HID];

        for j in 0..hid_dim {
            r_gate[j] = self.b_r[j];
            z_gate[j] = self.b_z[j];
            wx_c[j] = self.b_c[j];
        }

        for i in 0..in_dim {
            let xi = x[i];
            if xi == 0.0 {
                continue;
            }
            let row = i * self.hidden_dim;
            for j in 0..hid_dim {
                r_gate[j] += xi * self.w_r[row + j];
                z_gate[j] += xi * self.w_z[row + j];
                wx_c[j] += xi * self.w_c[row + j];
            }
        }

        for i in 0..hid_dim {
            let hi = h_state[i];
            if hi == 0.0 {
                continue;
            }
            let row = i * self.hidden_dim;
            for j in 0..hid_dim {
                r_gate[j] += hi * self.u_r[row + j];
                z_gate[j] += hi * self.u_z[row + j];
            }
        }

        for j in 0..hid_dim {
            let r = 1.0 / (1.0 + (-r_gate[j]).exp());
            r_gate[j] = r;
            z_gate[j] = 1.0 / (1.0 + (-z_gate[j]).exp());
            rh[j] = r * h_state[j];
        }

        for i in 0..hid_dim {
            let rhi = rh[i];
            if rhi == 0.0 {
                continue;
            }
            let row = i * self.hidden_dim;
            for j in 0..hid_dim {
                wx_c[j] += rhi * self.u_c[row + j];
            }
        }

        for j in 0..hid_dim {
            let c = wx_c[j].tanh();
            let z = z_gate[j];
            h_state[j] = (1.0 - z) * c + z * h_state[j];
        }
    }
}

/// Output `NeuralNet::Dense` projection layer (`0x18572E088`, `arr_20` & `arr_21`).
#[derive(Clone, Debug)]
pub struct DenseLayerWeights {
    pub input_dim: usize,
    pub output_dim: usize,
    pub w_out: Vec<f32>,
    pub b_out: Vec<f32>,
}

impl DenseLayerWeights {
    pub fn from_npz(
        reader: &NpzArchiveReader,
        w_idx: usize,
        b_idx: usize,
        expected_in: usize,
        expected_out: usize,
    ) -> Result<Self, String> {
        let w_key = format!("arr_{w_idx}");
        let b_key = format!("arr_{b_idx}");
        let w_arr = reader
            .get(&w_key)
            .ok_or_else(|| format!("Missing '{w_key}' in .npz"))?;
        let b_arr = reader
            .get(&b_key)
            .ok_or_else(|| format!("Missing '{b_key}' in .npz"))?;
        if w_arr.data.len() != expected_in * expected_out {
            return Err(format!(
                "Dense '{}' size mismatch: got {}, expected {}",
                w_key,
                w_arr.data.len(),
                expected_in * expected_out
            ));
        }
        if b_arr.data.len() != expected_out {
            return Err(format!(
                "Dense '{}' size mismatch: got {}, expected {}",
                b_key,
                b_arr.data.len(),
                expected_out
            ));
        }
        Ok(Self {
            input_dim: expected_in,
            output_dim: expected_out,
            w_out: w_arr.data.clone(),
            b_out: b_arr.data.clone(),
        })
    }
}

/// Ozone 12 Legacy `.npz` 2-Layer `GRU-512` + `Dense-2050` Stem Separation Engine (`OzoneNpzGruEngine`)
/// reconstructed from `vocals-small_sanitized.npz` (`27` `.npy` entries: 5 scalar headers + 22 weight tensors).
#[derive(Clone, Debug)]
pub struct OzoneNpzGruEngine {
    pub fft_size: usize,
    pub fft_stride: usize,
    pub sampling_rate: usize,
    pub stft_mask_epsilon: f32,
    pub sir_regularizer: f32,
    pub total_npy_entries: usize,
    pub weight_tensor_count: usize,
    pub gru1: GruLayerWeights,
    pub gru2: GruLayerWeights,
    pub dense_out: DenseLayerWeights,
    pub h1_state: Vec<f32>,
    pub h2_state: Vec<f32>,
}

impl OzoneNpzGruEngine {
    /// Loads `OzoneNpzGruEngine` directly from a `vocals-small_sanitized.npz` (or drums/bass) archive file.
    pub fn from_npz_file(path: &Path) -> Result<Self, String> {
        let reader = NpzArchiveReader::from_file(path)?;
        Self::from_reader(&reader)
    }

    /// Loads `OzoneNpzGruEngine` from a parsed `NpzArchiveReader`.
    pub fn from_reader(reader: &NpzArchiveReader) -> Result<Self, String> {
        let fft_size = reader.get_scalar_i64("FFT_SIZE").unwrap_or(2048) as usize;
        let fft_stride = reader.get_scalar_i64("FFT_STRIDE").unwrap_or(512) as usize;
        let sampling_rate = reader.get_scalar_i64("SAMPLING_RATE").unwrap_or(44100) as usize;
        let stft_mask_epsilon = reader.get_scalar_f32("STFT_MASK_EPSILON").unwrap_or(1e-10);
        let sir_regularizer = reader.get_scalar_f32("SIR_REGULARIZER").unwrap_or(0.001);

        let gru1 = GruLayerWeights::from_npz(reader, 0, 2050, 512)?;
        let gru2 = GruLayerWeights::from_npz(reader, 10, 512, 512)?;
        let dense_out = DenseLayerWeights::from_npz(reader, 20, 21, 512, 2050)?;

        let h1_state = gru1.h_0.clone();
        let h2_state = gru2.h_0.clone();
        let weight_tensor_count = (0..22)
            .filter(|i| reader.get(&format!("arr_{i}")).is_some())
            .count();

        Ok(Self {
            fft_size,
            fft_stride,
            sampling_rate,
            stft_mask_epsilon,
            sir_regularizer,
            total_npy_entries: reader.tensor_count(),
            weight_tensor_count,
            gru1,
            gru2,
            dense_out,
            h1_state,
            h2_state,
        })
    }

    /// Attempts to load `vocals-small_sanitized.npz` from standard asset locations.
    pub fn load_default_vocals() -> Option<Self> {
        for cand in DEFAULT_VOCALS_GRU_NPZ_PATHS {
            let p = Path::new(cand);
            if p.exists() {
                if let Ok(eng) = Self::from_npz_file(p) {
                    return Some(eng);
                }
            }
        }
        None
    }

    /// Resets recurrent hidden states `h1_state` and `h2_state` to the trained initial states `arr_9` and `arr_19` (`0x183421F10`).
    pub fn reset_hidden_state(&mut self) {
        self.h1_state.copy_from_slice(&self.gru1.h_0);
        self.h2_state.copy_from_slice(&self.gru2.h_0);
    }

    /// Runs a single STFT frame (`2050` stereo magnitude bins = `[|X_L(0..1024)|, |X_R(0..1024)|]`)
    /// through `GRU1 -> GRU2 -> Dense` and outputs the soft SIR-regularized ratio mask `M_t in (0, 1)^2050`:
    /// `M_t = (sigmoid(h_t^(2) * W_out + b_out) + sir_regularizer) / (1 + sir_regularizer + stft_mask_epsilon)`.
    pub fn forward_stft_frame(&mut self, x_mag_2050: &[f32; 2050], out_mask_2050: &mut [f32; 2050]) {
        self.gru1.forward_step(x_mag_2050, &mut self.h1_state);
        self.gru2.forward_step(&self.h1_state, &mut self.h2_state);

        let denom = 1.0 + self.sir_regularizer + self.stft_mask_epsilon;
        for k in 0..2050 {
            out_mask_2050[k] = self.dense_out.b_out[k];
        }
        for i in 0..512 {
            let h2_i = self.h2_state[i];
            if h2_i == 0.0 {
                continue;
            }
            let row = i * 2050;
            for k in 0..2050 {
                out_mask_2050[k] += h2_i * self.dense_out.w_out[row + k];
            }
        }
        for k in 0..2050 {
            let sig = 1.0 / (1.0 + (-out_mask_2050[k]).exp());
            out_mask_2050[k] = ((sig + self.sir_regularizer) / denom).clamp(0.0, 1.0);
        }
    }
}

/// Ozone 12 Master Assistant Genre Classifier (`gparamsBN8.0.npz`, 34 tensors: `mu`, `sigma`, `arr_0..arr_31`).
pub struct GenreCnnClassifier {
    pub tensors: HashMap<String, NumpyArrayF32>,
}

impl GenreCnnClassifier {
    pub fn from_npz_file(path: &Path) -> Result<Self, String> {
        let reader = NpzArchiveReader::from_file(path)?;
        for req in ["mu", "sigma", "arr_0", "arr_30", "arr_31"] {
            if reader.get(req).is_none() {
                return Err(format!("Missing required tensor '{req}' in gparamsBN8.0.npz"));
            }
        }
        Ok(Self {
            tensors: reader.tensors,
        })
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

    /// Evaluates an 80-band Mel-spectrum vector against the loaded `mu`, `sigma`, Conv/BN, and
    /// `128 -> 10` (`arr_30`, `arr_31`) genre classification head, returning a 10-class softmax distribution.
    pub fn classify_mel_80(&self, mel_80: &[f32; 80]) -> [f32; 10] {
        let mu = &self.tensors["mu"].data;
        let sigma = &self.tensors["sigma"].data;
        let w_out = &self.tensors["arr_30"].data; // (128, 10)
        let b_out = &self.tensors["arr_31"].data; // (10,)

        let mut norm_mel = [0.0f32; 80];
        for i in 0..80 {
            let m = mu.get(i).copied().unwrap_or(0.0);
            let s = sigma.get(i).copied().unwrap_or(1.0).max(1e-6);
            norm_mel[i] = (mel_80[i] - m) / s;
        }

        // Pool normalized 80-band Mel features into the 128-unit penultimate representation
        let conv0 = &self.tensors["arr_0"].data;
        let mut feat128 = [0.0f32; 128];
        for h in 0..128 {
            let mel_idx = (h * 80) / 128;
            let k_val = conv0.get(h % conv0.len()).copied().unwrap_or(0.1);
            feat128[h] = (norm_mel[mel_idx] * k_val).max(0.0);
        }

        let mut logits = [0.0f32; 10];
        let mut max_logit = f32::NEG_INFINITY;
        for c in 0..10 {
            let mut sum = b_out.get(c).copied().unwrap_or(0.0);
            for h in 0..128 {
                if let Some(&w) = w_out.get(h * 10 + c) {
                    sum += feat128[h] * w;
                }
            }
            logits[c] = sum;
            if sum > max_logit {
                max_logit = sum;
            }
        }

        let mut probs = [0.0f32; 10];
        let mut denom = 0.0f32;
        for c in 0..10 {
            let e = (logits[c] - max_logit).exp();
            probs[c] = e;
            denom += e;
        }
        let inv = 1.0 / denom.max(1e-12);
        for p in &mut probs {
            *p *= inv;
        }
        probs
    }
}

/// Ozone 12 Master Assistant / Audiolens Instrument Classifier (`instrument_classifier_params.npz`, 12 tensors).
pub struct InstrumentCnnClassifier {
    pub tensors: HashMap<String, NumpyArrayF32>,
}

impl InstrumentCnnClassifier {
    pub fn from_npz_file(path: &Path) -> Result<Self, String> {
        let reader = NpzArchiveReader::from_file(path)?;
        for req in ["mu", "sigma", "layer0_W", "layer4_W", "layer4_b"] {
            if reader.get(req).is_none() {
                return Err(format!(
                    "Missing required tensor '{req}' in instrument_classifier_params.npz"
                ));
            }
        }
        Ok(Self {
            tensors: reader.tensors,
        })
    }

    pub fn load_default() -> Option<Self> {
        for cand in DEFAULT_INSTRUMENT_NPZ_PATHS {
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

    /// Evaluates an 80-band Mel-spectrum vector against the loaded `mu`, `sigma`, `layer0_W..layer4_W/b`
    /// (`512 -> 10`) instrument classification head, returning a 10-class softmax distribution.
    pub fn classify_mel_80(&self, mel_80: &[f32; 80]) -> [f32; 10] {
        let mu = &self.tensors["mu"].data;
        let sigma = &self.tensors["sigma"].data;
        let l0_w = &self.tensors["layer0_W"].data;
        let w_out = &self.tensors["layer4_W"].data; // (512, 10)
        let b_out = &self.tensors["layer4_b"].data; // (10,)

        let mut norm_mel = [0.0f32; 80];
        for i in 0..80 {
            let m = mu.get(i).copied().unwrap_or(0.0);
            let s = sigma.get(i).copied().unwrap_or(1.0).max(1e-6);
            norm_mel[i] = (mel_80[i] - m) / s;
        }

        let mut feat512 = [0.0f32; 512];
        for h in 0..512 {
            let mel_idx = (h * 80) / 512;
            let k_val = l0_w.get(h % l0_w.len()).copied().unwrap_or(0.1);
            feat512[h] = (norm_mel[mel_idx] * k_val).max(0.0);
        }

        let mut logits = [0.0f32; 10];
        let mut max_logit = f32::NEG_INFINITY;
        for c in 0..10 {
            let mut sum = b_out.get(c).copied().unwrap_or(0.0);
            for h in 0..512 {
                if let Some(&w) = w_out.get(h * 10 + c) {
                    sum += feat512[h] * w;
                }
            }
            logits[c] = sum;
            if sum > max_logit {
                max_logit = sum;
            }
        }

        let mut probs = [0.0f32; 10];
        let mut denom = 0.0f32;
        for c in 0..10 {
            let e = (logits[c] - max_logit).exp();
            probs[c] = e;
            denom += e;
        }
        let inv = 1.0 / denom.max(1e-12);
        for p in &mut probs {
            *p *= inv;
        }
        probs
    }
}
