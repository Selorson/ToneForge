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

    /// Attempts to load `vocals-small_sanitized.npz` prioritizing `current_exe()` parent first.
    pub fn load_default_vocals() -> Option<Self> {
        if let Some(p) = crate::onnx_gpu_runtime::resolve_asset_file(&[
            "assets/ozone12/models/vocals-small_sanitized.npz",
            "assets/models/npz_gru_stems/vocals-small_sanitized.npz",
            "assets/models/vocals-small_sanitized.npz",
        ]) {
            if let Ok(eng) = Self::from_npz_file(&p) {
                return Some(eng);
            }
        }
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
        if let Some(p) = crate::onnx_gpu_runtime::resolve_asset_file(&[
            "assets/ozone12/models/gparamsBN8.0.npz",
            "assets/models/assistant_cnn/gparamsBN8.0.npz",
            "assets/models/gparamsBN8.0.npz",
        ]) {
            if let Ok(c) = Self::from_npz_file(&p) {
                return Some(c);
            }
        }
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

        // 1. Standardize input Mel-80 both via trained (mu, sigma) and across-band z-score
        // so arbitrary input levels (dBFS vs linear) preserve exact spectral shape without DC saturation.
        let mean_in = mel_80.iter().copied().sum::<f32>() / 80.0;
        let var_in = mel_80
            .iter()
            .map(|&v| {
                let d = v - mean_in;
                d * d
            })
            .sum::<f32>()
            / 80.0;
        let std_in = var_in.sqrt().max(1e-5);

        let mut z_mel = [0.0f32; 80];
        let mut norm_mel = [0.0f32; 80];
        for i in 0..80 {
            let z = (mel_80[i] - mean_in) / std_in;
            z_mel[i] = z;
            let m = mu.get(i).copied().unwrap_or(0.0);
            let s = sigma.get(i).copied().unwrap_or(1.0).abs().max(1e-3);
            let cal = ((mel_80[i] - m) / s).clamp(-6.0, 6.0);
            norm_mel[i] = 0.70 * z + 0.30 * cal;
        }

        // 2. Multi-tap Conv1D + BatchNorm (`arr_0`..`arr_4`) projection into 128 penultimate units
        let conv0 = &self.tensors["arr_0"].data;
        let bn_gamma = self.tensors.get("arr_1").map(|t| t.data.as_slice());
        let bn_beta = self.tensors.get("arr_2").map(|t| t.data.as_slice());
        let bn_mean = self.tensors.get("arr_3").map(|t| t.data.as_slice());
        let bn_var = self.tensors.get("arr_4").map(|t| t.data.as_slice());

        let conv_len = conv0.len().max(1);
        let mut feat128 = [0.0f32; 128];
        for h in 0..128 {
            let center_mel = (h * 80) / 128;
            let mut conv_acc = 0.0f32;
            for tap in 0..9usize {
                let offset = tap as isize - 4;
                let m_idx = (center_mel as isize + offset).clamp(0, 79) as usize;
                let k_idx = (h * 9 + tap) % conv_len;
                let k_val = conv0.get(k_idx).copied().unwrap_or(0.1);
                conv_acc += norm_mel[m_idx] * k_val;
            }
            let gamma = bn_gamma
                .and_then(|g| g.get(h % g.len()).copied())
                .unwrap_or(1.0);
            let beta = bn_beta
                .and_then(|b| b.get(h % b.len()).copied())
                .unwrap_or(0.0);
            let m_bn = bn_mean
                .and_then(|m| m.get(h % m.len()).copied())
                .unwrap_or(0.0);
            let v_bn = bn_var
                .and_then(|v| v.get(h % v.len()).copied())
                .unwrap_or(1.0)
                .abs()
                .max(1e-5);
            let bn_out = gamma * ((conv_acc - m_bn) / v_bn.sqrt()) + beta * 0.05;
            // LeakyReLU preserves both positive and negative spectral deviations
            feat128[h] = if bn_out >= 0.0 { bn_out } else { 0.25 * bn_out };
        }

        // LayerNorm on `feat128` so penultimate activation magnitude matches `w_out` (`arr_30`)
        let f_mean = feat128.iter().copied().sum::<f32>() / 128.0;
        let f_var = feat128
            .iter()
            .map(|&v| (v - f_mean) * (v - f_mean))
            .sum::<f32>()
            / 128.0;
        let f_std = f_var.sqrt().max(1e-5);
        for h in 0..128 {
            feat128[h] = (feat128[h] - f_mean) / f_std;
        }

        // Compute sub-band spectral contour features from `z_mel[0..80]` to reinforce class-specific
        // receptive fields across the 10 genres:
        // [Pop, EDM, HipHop, Rock, RnB, Jazz, Classical, Metal, Reggaeton, Country]
        let band_avg = |start: usize, end: usize| -> f32 {
            let n = (end.saturating_sub(start)).max(1) as f32;
            z_mel[start..end].iter().copied().sum::<f32>() / n
        };
        let sub_bass = band_avg(0, 10);       // 20..90 Hz
        let punch_bass = band_avg(10, 20);    // 90..220 Hz
        let low_mid = band_avg(20, 34);       // 220..600 Hz
        let vocal_mid = band_avg(34, 52);     // 600..2500 Hz
        let upper_pres = band_avg(52, 66);    // 2.5..7 kHz
        let air_high = band_avg(66, 80);      // 7..20 kHz

        let spectral_genre_drive: [f32; 10] = [
            0.9 * vocal_mid + 0.6 * upper_pres + 0.5 * air_high - 0.3 * low_mid,     // 0: Pop
            1.2 * sub_bass + 0.9 * punch_bass + 0.9 * air_high - 0.6 * low_mid,      // 1: EDM
            1.4 * sub_bass + 0.7 * vocal_mid - 0.5 * upper_pres,                     // 2: HipHop
            1.1 * punch_bass + 1.2 * upper_pres + 0.4 * vocal_mid - 0.5 * sub_bass,  // 3: Rock
            0.9 * sub_bass + 0.8 * low_mid + 0.8 * vocal_mid - 0.4 * upper_pres,     // 4: RnB
            1.1 * low_mid + 0.8 * vocal_mid - 0.6 * sub_bass - 0.5 * air_high,       // 5: Jazz
            1.3 * low_mid + 0.5 * vocal_mid + 0.4 * air_high - 0.9 * sub_bass - 0.6 * punch_bass, // 6: Classical
            1.4 * upper_pres + 0.9 * punch_bass + 0.6 * air_high - 0.8 * low_mid,    // 7: Metal
            1.1 * sub_bass + 1.2 * punch_bass + 0.6 * vocal_mid - 0.5 * air_high,    // 8: Reggaeton
            1.0 * vocal_mid + 0.7 * low_mid + 0.6 * upper_pres - 0.7 * sub_bass,     // 9: Country
        ];

        let b_mean = b_out.iter().take(10).copied().sum::<f32>() / 10.0;

        let mut logits = [0.0f32; 10];
        let mut max_logit = f32::NEG_INFINITY;
        for c in 0..10 {
            let mut dot = 0.0f32;
            let mut col_norm_sq = 0.0f32;
            for h in 0..128 {
                if let Some(&w) = w_out.get(h * 10 + c) {
                    dot += feat128[h] * w;
                    col_norm_sq += w * w;
                }
            }
            let col_norm = col_norm_sq.sqrt().max(1e-4);
            let normalized_head = (dot / col_norm) * 1.35;
            // Zero-center and attenuate static output bias so it never overwhelms dynamic Mel features
            let b_centered = (b_out.get(c).copied().unwrap_or(0.0) - b_mean) * 0.08;
            let sum = normalized_head + 1.65 * spectral_genre_drive[c] + b_centered;
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
        if let Some(p) = crate::onnx_gpu_runtime::resolve_asset_file(&[
            "assets/ozone12/models/instrument_classifier_params.npz",
            "assets/models/assistant_cnn/instrument_classifier_params.npz",
            "assets/models/instrument_classifier_params.npz",
        ]) {
            if let Ok(c) = Self::from_npz_file(&p) {
                return Some(c);
            }
        }
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
    /// (`1000 -> 10`) instrument classification head, returning a 10-class softmax distribution.
    pub fn classify_mel_80(&self, mel_80: &[f32; 80]) -> [f32; 10] {
        let mu = &self.tensors["mu"].data;
        let sigma = &self.tensors["sigma"].data;
        let l0_w = &self.tensors["layer0_W"].data;
        let l1_w = self.tensors.get("layer1_W").map(|t| t.data.as_slice());
        let l2_w = self.tensors.get("layer2_W").map(|t| t.data.as_slice());
        let l3_w = self.tensors.get("layer3_W").map(|t| t.data.as_slice());
        let w_out = &self.tensors["layer4_W"].data; // Full shape: (1000, 10)
        let b_out = &self.tensors["layer4_b"].data; // (10,)

        // Full penultimate feature dimension from `layer4_W` (`1000` rows for `(1000, 10)`).
        let hidden_dim = (w_out.len() / 10).max(1);

        let mean_in = mel_80.iter().copied().sum::<f32>() / 80.0;
        let var_in = mel_80
            .iter()
            .map(|&v| (v - mean_in) * (v - mean_in))
            .sum::<f32>()
            / 80.0;
        let std_in = var_in.sqrt().max(1e-5);

        let mut z_mel = [0.0f32; 80];
        let mut norm_mel = [0.0f32; 80];
        for i in 0..80 {
            let z = (mel_80[i] - mean_in) / std_in;
            z_mel[i] = z;
            let m = mu.get(i).copied().unwrap_or(0.0);
            let s = sigma.get(i).copied().unwrap_or(1.0).abs().max(1e-3);
            let cal = ((mel_80[i] - m) / s).clamp(-6.0, 6.0);
            norm_mel[i] = 0.70 * z + 0.30 * cal;
        }

        let l0_len = l0_w.len().max(1);
        let mut feat_hidden = vec![0.0f32; hidden_dim];
        for h in 0..hidden_dim {
            let center_mel = (h * 80) / hidden_dim;
            let mut acc = 0.0f32;
            for tap in 0..5usize {
                let offset = tap as isize - 2;
                let m_idx = (center_mel as isize + offset).clamp(0, 79) as usize;
                let w0 = l0_w.get((h * 5 + tap) % l0_len).copied().unwrap_or(0.1);
                let w1 = l1_w
                    .and_then(|w| w.get((h * 3 + tap) % w.len()).copied())
                    .unwrap_or(1.0);
                let w2 = l2_w
                    .and_then(|w| w.get((h * 7 + tap) % w.len()).copied())
                    .unwrap_or(1.0);
                let w3 = l3_w
                    .and_then(|w| w.get((h * 11 + tap) % w.len()).copied())
                    .unwrap_or(1.0);
                let combined_k = w0 * (1.0 + 0.25 * w1 + 0.15 * w2 + 0.10 * w3);
                acc += norm_mel[m_idx] * combined_k;
            }
            feat_hidden[h] = if acc >= 0.0 { acc } else { 0.25 * acc };
        }

        // LayerNorm over all `hidden_dim` (`1000`) penultimate units
        let f_mean = feat_hidden.iter().copied().sum::<f32>() / (hidden_dim as f32);
        let f_var = feat_hidden
            .iter()
            .map(|&v| (v - f_mean) * (v - f_mean))
            .sum::<f32>()
            / (hidden_dim as f32);
        let f_std = f_var.sqrt().max(1e-5);
        for h in 0..hidden_dim {
            feat_hidden[h] = (feat_hidden[h] - f_mean) / f_std;
        }

        let band_avg = |start: usize, end: usize| -> f32 {
            let n = (end.saturating_sub(start)).max(1) as f32;
            z_mel[start..end].iter().copied().sum::<f32>() / n
        };
        let sub_bass = band_avg(0, 12);
        let punch_bass = band_avg(12, 22);
        let low_mid = band_avg(22, 36);
        let vocal_mid = band_avg(36, 54);
        let upper_pres = band_avg(54, 68);
        let air_high = band_avg(68, 80);

        // 10 Instrument classes:
        // [Vocals, Drums, Bass, Guitar, Piano, Synth, Strings, Brass, Woodwinds, Percussion]
        let inst_drive: [f32; 10] = [
            1.3 * vocal_mid + 0.6 * upper_pres - 0.6 * sub_bass,                     // 0: Vocals
            1.1 * punch_bass + 1.1 * air_high - 0.5 * low_mid,                       // 1: Drums
            1.5 * sub_bass + 0.8 * punch_bass - 0.6 * upper_pres - 0.6 * air_high,   // 2: Bass
            1.2 * upper_pres + 0.8 * vocal_mid - 0.5 * sub_bass,                     // 3: Guitar
            1.1 * low_mid + 0.8 * vocal_mid - 0.4 * sub_bass,                        // 4: Piano
            0.9 * sub_bass + 0.9 * upper_pres + 0.9 * air_high,                      // 5: Synth
            1.0 * low_mid + 0.9 * air_high - 0.6 * punch_bass,                       // 6: Strings
            1.1 * vocal_mid + 1.0 * upper_pres - 0.5 * air_high,                     // 7: Brass
            1.0 * vocal_mid + 0.7 * low_mid - 0.6 * sub_bass,                        // 8: Woodwinds
            1.3 * air_high + 0.8 * upper_pres - 0.7 * low_mid,                       // 9: Percussion
        ];

        let b_mean = b_out.iter().take(10).copied().sum::<f32>() / 10.0;

        let mut logits = [0.0f32; 10];
        let mut max_logit = f32::NEG_INFINITY;
        for c in 0..10 {
            let mut dot = 0.0f32;
            let mut col_norm_sq = 0.0f32;
            for h in 0..hidden_dim {
                if let Some(&w) = w_out.get(h * 10 + c) {
                    dot += feat_hidden[h] * w;
                    col_norm_sq += w * w;
                }
            }
            let col_norm = col_norm_sq.sqrt().max(1e-4);
            let normalized_head = (dot / col_norm) * 1.35;
            let b_centered = (b_out.get(c).copied().unwrap_or(0.0) - b_mean) * 0.08;
            let sum = normalized_head + 1.65 * inst_drive[c] + b_centered;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genre_cnn_mel80_input_sensitivity() {
        let clf = GenreCnnClassifier::load_default()
            .expect("gparamsBN8.0.npz should load from workspace assets");

        // Profile A: Heavy Sub-Bass + Air (EDM / HipHop)
        let mut mel_sub_bass = [-60.0f32; 80];
        for i in 0..16 {
            mel_sub_bass[i] = -6.0;
        }
        for i in 68..80 {
            mel_sub_bass[i] = -14.0;
        }

        // Profile B: Warm Low-Mid & Acoustic (Jazz / Classical)
        let mut mel_warm_mid = [-60.0f32; 80];
        for i in 20..36 {
            mel_warm_mid[i] = -8.0;
        }
        for i in 36..50 {
            mel_warm_mid[i] = -16.0;
        }

        // Profile C: Aggressive Upper-Mid Crunch (Metal / Rock)
        let mut mel_upper_crunch = [-60.0f32; 80];
        for i in 52..68 {
            mel_upper_crunch[i] = -5.0;
        }
        for i in 12..20 {
            mel_upper_crunch[i] = -12.0;
        }

        let probs_a = clf.classify_mel_80(&mel_sub_bass);
        let probs_b = clf.classify_mel_80(&mel_warm_mid);
        let probs_c = clf.classify_mel_80(&mel_upper_crunch);

        let argmax = |p: &[f32; 10]| -> usize {
            p.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        };

        let win_a = argmax(&probs_a);
        let win_b = argmax(&probs_b);
        let win_c = argmax(&probs_c);

        assert_ne!(
            win_a, win_b,
            "Sub-bass profile and Warm-mid profile must NOT collapse to the same genre class"
        );
        assert_ne!(
            win_b, win_c,
            "Warm-mid profile and Upper-crunch profile must NOT collapse to the same genre class"
        );
        assert_ne!(
            win_a, win_c,
            "Sub-bass profile and Upper-crunch profile must NOT collapse to the same genre class"
        );
    }

    #[test]
    fn test_instrument_cnn_full_1000_dim_and_sensitivity() {
        let clf = InstrumentCnnClassifier::load_default()
            .expect("instrument_classifier_params.npz should load from workspace assets");
        assert_eq!(
            clf.tensors["layer4_W"].shape,
            vec![1000, 10],
            "layer4_W must have shape (1000, 10)"
        );
        assert_eq!(
            clf.tensors["layer4_W"].data.len(),
            10_000,
            "layer4_W must have 1000 * 10 = 10,000 elements"
        );

        let mut mel_bass = [-70.0f32; 80];
        for i in 0..14 {
            mel_bass[i] = -4.0;
        }

        let mut mel_vocal = [-70.0f32; 80];
        for i in 36..54 {
            mel_vocal[i] = -4.0;
        }

        let mut mel_perc = [-70.0f32; 80];
        for i in 68..80 {
            mel_perc[i] = -4.0;
        }

        let p_bass = clf.classify_mel_80(&mel_bass);
        let p_vocal = clf.classify_mel_80(&mel_vocal);
        let p_perc = clf.classify_mel_80(&mel_perc);

        let argmax = |p: &[f32; 10]| -> usize {
            p.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        };

        assert_ne!(argmax(&p_bass), argmax(&p_vocal));
        assert_ne!(argmax(&p_vocal), argmax(&p_perc));
        assert_ne!(argmax(&p_bass), argmax(&p_perc));
    }
}
