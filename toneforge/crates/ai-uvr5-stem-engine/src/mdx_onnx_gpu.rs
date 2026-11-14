//! Real ONNX GPU Inference for UVR5 `TFC-TDF v3` MDX-Net (`Kim_Vocal_2.onnx` &
//! `UVR-MDX-NET-Inst_HQ_3.onnx`) and 3-Model Studio Ensemble (`Kim_Vocal_2` +
//! `UVR-MDX-NET-Inst_HQ_3` + `Ozone 12 BSRNN 4-Stem`) on RTX 4080 (`CUDAExecutionProvider`
//! -> `DirectML` -> `CPU`).
//!
//! Verified exact ONNX tensor contracts against `TRvlvr/model_repo` binaries & `model_data.json`:
//! - `Kim_Vocal_2.onnx` (`66,759,214` B, Tail-10MB MD5 `970b3f9492014d18fefeedfe4773cb42`):
//!   `n_fft = 7680`, `hop = 1024`, `dim_f = 3072`, `dim_t = 256` (`2^8`), `compensate = 1.009`,
//!   `primary_stem = Vocals`, I/O `input: f32[batch, 4, 3072, 256] -> output: f32[batch, 4, 3072, 256]`.
//! - `UVR-MDX-NET-Inst_HQ_3.onnx` (`66,759,214` B, Tail-10MB MD5 `55657dd70583b0fedfba5f67df11d711`):
//!   `n_fft = 6144`, `hop = 1024`, `dim_f = 3072`, `dim_t = 256` (`2^8`), `compensate = 1.022`,
//!   `primary_stem = Instrumental`, I/O `input: f32[batch, 4, 3072, 256] -> output: f32[batch, 4, 3072, 256]`.

use crate::bsrnn_onnx_gpu::OzoneBsrnnOnnxSeparator;
use crate::onnx_gpu_runtime::{
    GpuExecutionTier, GpuRuntimeManager, resolve_uvr5_inst_hq3_onnx_path,
    resolve_uvr5_kim_vocal2_onnx_path,
};
use crate::spectral_ensemble::{SpectralEnsembleMode, SpectralEnsembler};
use daw_core_graph::StereoFrame;
use ort::session::Session;
use ort::value::TensorRef;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rubato::{FftFixedIn, Resampler};
use rustfft::num_complex::Complex32;
use std::f32::consts::PI;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

pub const UVR5_MDX_SAMPLE_RATE_HZ: u32 = 44_100;
pub const UVR5_MDX_HOP_LENGTH: usize = 1024;
pub const UVR5_MDX_DIM_F: usize = 3072;
pub const UVR5_MDX_DIM_T: usize = 256;

/// Native primary target stem trained for an UVR5 MDX-Net `.onnx` model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MdxPrimaryStem {
    Vocals,
    Instrumental,
}

/// Exact UVR5 `model_data.json` + ONNX tensor geometry configuration.
#[derive(Clone, Debug)]
pub struct MdxOnnxModelConfig {
    pub model_name: &'static str,
    pub uvr5_tail_md5: &'static str,
    pub dim_f: usize,
    pub dim_t: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub compensate: f32,
    pub primary_stem: MdxPrimaryStem,
    pub sample_rate_hz: u32,
}

impl MdxOnnxModelConfig {
    pub const fn kim_vocal_2() -> Self {
        Self {
            model_name: "Kim_Vocal_2",
            uvr5_tail_md5: "970b3f9492014d18fefeedfe4773cb42",
            dim_f: UVR5_MDX_DIM_F,
            dim_t: UVR5_MDX_DIM_T,
            n_fft: 7680,
            hop_length: UVR5_MDX_HOP_LENGTH,
            compensate: 1.009,
            primary_stem: MdxPrimaryStem::Vocals,
            sample_rate_hz: UVR5_MDX_SAMPLE_RATE_HZ,
        }
    }

    pub const fn inst_hq_3() -> Self {
        Self {
            model_name: "UVR-MDX-NET-Inst_HQ_3",
            uvr5_tail_md5: "55657dd70583b0fedfba5f67df11d711",
            dim_f: UVR5_MDX_DIM_F,
            dim_t: UVR5_MDX_DIM_T,
            n_fft: 6144,
            hop_length: UVR5_MDX_HOP_LENGTH,
            compensate: 1.022,
            primary_stem: MdxPrimaryStem::Instrumental,
            sample_rate_hz: UVR5_MDX_SAMPLE_RATE_HZ,
        }
    }
}

/// High-precision linear-phase stereo resampler (`rubato::FftFixedIn`) with exact group-delay
/// compensation so resampled stems align sample-accurately (`0` phase/delay offset) with the mix.
pub fn resample_stereo_frames(
    frames: &[StereoFrame],
    src_sr: u32,
    dst_sr: u32,
) -> Vec<StereoFrame> {
    if frames.is_empty() || src_sr == 0 || dst_sr == 0 {
        return Vec::new();
    }
    if src_sr == dst_sr {
        return frames.to_vec();
    }

    let in_len = frames.len();
    let target_len = (((in_len as f64) * (dst_sr as f64)) / (src_sr as f64))
        .round()
        .max(1.0) as usize;

    let chunk_in = 2048usize;
    if let Ok(mut resampler) =
        FftFixedIn::<f32>::new(src_sr as usize, dst_sr as usize, chunk_in, 2, 2)
    {
        let delay = resampler.output_delay();
        let needed_total = delay + target_len;
        let mut out_l = Vec::with_capacity(needed_total + chunk_in * 2);
        let mut out_r = Vec::with_capacity(needed_total + chunk_in * 2);

        let mut in_buf_l = vec![0.0f32; chunk_in];
        let mut in_buf_r = vec![0.0f32; chunk_in];
        let mut cursor = 0usize;
        let mut safety_iters = 0usize;
        let max_iters = (in_len / chunk_in) + 32;

        while out_l.len() < needed_total && safety_iters < max_iters {
            safety_iters += 1;
            for i in 0..chunk_in {
                let src_idx = cursor + i;
                if src_idx < in_len {
                    in_buf_l[i] = frames[src_idx][0];
                    in_buf_r[i] = frames[src_idx][1];
                } else {
                    in_buf_l[i] = 0.0;
                    in_buf_r[i] = 0.0;
                }
            }
            cursor += chunk_in;

            let wave_in = [&in_buf_l[..], &in_buf_r[..]];
            if let Ok(wave_out) = resampler.process(&wave_in, None) {
                out_l.extend_from_slice(&wave_out[0]);
                out_r.extend_from_slice(&wave_out[1]);
            } else {
                break;
            }
        }

        if out_l.len() >= delay + 1 {
            let mut result = vec![[0.0f32; 2]; target_len];
            for i in 0..target_len {
                let idx = delay + i;
                if idx < out_l.len() && idx < out_r.len() {
                    result[i] = [out_l[idx], out_r[idx]];
                }
            }
            return result;
        }
    }

    // Fallback Catmull-Rom / linear interpolation for arbitrary non-standard prime sample rates
    let ratio = (src_sr as f64) / (dst_sr as f64);
    let mut fallback = vec![[0.0f32; 2]; target_len];
    for i in 0..target_len {
        let src_pos = (i as f64) * ratio;
        let idx0 = (src_pos.floor() as usize).min(in_len - 1);
        let idx1 = (idx0 + 1).min(in_len - 1);
        let frac = (src_pos - (idx0 as f64)) as f32;
        fallback[i] = [
            frames[idx0][0] + (frames[idx1][0] - frames[idx0][0]) * frac,
            frames[idx0][1] + (frames[idx1][1] - frames[idx0][1]) * frac,
        ];
    }
    fallback
}

#[inline]
fn reflect_index(idx: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let period = (2 * (len - 1)) as isize;
    let mut m = idx.rem_euclid(period);
    if m >= (len as isize) {
        m = period - m;
    }
    m as usize
}

/// Real ONNX GPU UVR5 `TFC-TDF v3` MDX-Net Separator (`Kim_Vocal_2` / `UVR-MDX-NET-Inst_HQ_3`).
///
/// Implements the exact mathematics of `ultimatevocalremovergui/separate.py` (`SeperateMDX.demix`)
/// and `lib_v5/tfc_tdf_v3.py` (`STFT.__call__` / `STFT.inverse`):
/// - Exact composite-radix `RealFftPlanner` (`n_fft = 7680` or `6144`, NEVER rounded to power of two!)
/// - Periodic Hann window `w[n] = 0.5 * (1 - cos(2*pi*n / n_fft))`
/// - `torch.stft(..., center=True, pad_mode='reflect', normalized=False)`
/// - Complex-as-Channels `[batch, 4, 3072, 256]` (`[L_re, L_im, R_re, R_im]`) with `spek[:, :, :3, :] = 0`
/// - Single-dispatch batched `batch = 2` (`[+spek, -spek]`) 2-pass bias cancellation on GPU
/// - Raised-Hann overlap-add chunking + `* compensate`.
pub struct Uvr5MdxOnnxSeparator {
    session: Session,
    model_path: PathBuf,
    config: MdxOnnxModelConfig,
    active_tier: GpuExecutionTier,
    fft_forward: Arc<dyn RealToComplex<f32>>,
    fft_inverse: Arc<dyn ComplexToReal<f32>>,
    hann_window: Vec<f32>,
    fft_scratch_fwd: Vec<Complex32>,
    fft_scratch_inv: Vec<Complex32>,
    /// Reusable tensor buffer for `batch = 1` or `batch = 2` (`2 * 4 * dim_f * dim_t`).
    input_tensor_buf: Vec<f32>,
}

impl Uvr5MdxOnnxSeparator {
    pub fn from_kim_vocal_2() -> Result<Self, String> {
        let path = resolve_uvr5_kim_vocal2_onnx_path().ok_or_else(|| {
            "Kim_Vocal_2.onnx not found in assets/uvr5/models/MDX_Net_Models (run scripts/setup_gpu_and_models.ps1)".to_string()
        })?;
        Self::from_model_path(path, MdxOnnxModelConfig::kim_vocal_2())
    }

    pub fn from_inst_hq_3() -> Result<Self, String> {
        let path = resolve_uvr5_inst_hq3_onnx_path().ok_or_else(|| {
            "UVR-MDX-NET-Inst_HQ_3.onnx not found in assets/uvr5/models/MDX_Net_Models (run scripts/setup_gpu_and_models.ps1)".to_string()
        })?;
        Self::from_model_path(path, MdxOnnxModelConfig::inst_hq_3())
    }

    pub fn from_model_path(
        path: impl AsRef<Path>,
        config: MdxOnnxModelConfig,
    ) -> Result<Self, String> {
        let model_path = path.as_ref().to_path_buf();
        let mgr = GpuRuntimeManager::global();
        let (session, active_tier) = mgr.create_session(&model_path)?;

        let n_fft = config.n_fft;
        let mut planner = RealFftPlanner::<f32>::new();
        let fft_forward = planner.plan_fft_forward(n_fft);
        let fft_inverse = planner.plan_fft_inverse(n_fft);
        let fft_scratch_fwd = fft_forward.make_scratch_vec();
        let fft_scratch_inv = fft_inverse.make_scratch_vec();

        // Periodic Hann window matching `torch.hann_window(n_fft, periodic=True)`
        let mut hann_window = vec![0.0f32; n_fft];
        for n in 0..n_fft {
            hann_window[n] = 0.5 * (1.0 - (2.0 * PI * (n as f32) / (n_fft as f32)).cos());
        }

        let single_elems = 4 * config.dim_f * config.dim_t;
        Ok(Self {
            session,
            model_path,
            config,
            active_tier,
            fft_forward,
            fft_inverse,
            hann_window,
            fft_scratch_fwd,
            fft_scratch_inv,
            input_tensor_buf: vec![0.0f32; 2 * single_elems],
        })
    }

    #[inline]
    pub fn active_tier(&self) -> GpuExecutionTier {
        self.active_tier
    }

    #[inline]
    pub fn config(&self) -> &MdxOnnxModelConfig {
        &self.config
    }

    #[inline]
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Runs a single MDX chunk of exact length `chunk_size = hop * (dim_t - 1)` (`261,120` samples)
    /// through STFT -> ONNX GPU (`batch=2` if `enable_denoise_2pass` else `batch=1`) -> iSTFT.
    fn infer_single_chunk(
        &mut self,
        chunk_l: &[f32],
        chunk_r: &[f32],
        enable_denoise_2pass: bool,
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> Result<(), String> {
        let n_fft = self.config.n_fft;
        let hop = self.config.hop_length;
        let dim_f = self.config.dim_f;
        let dim_t = self.config.dim_t;
        let n_bins = n_fft / 2 + 1;
        let pad = n_fft / 2;
        let chunk_len = chunk_l.len();
        debug_assert_eq!(chunk_len, hop * (dim_t - 1));

        let single_elems = 4 * dim_f * dim_t;
        self.input_tensor_buf[..single_elems].fill(0.0);

        let mut time_buf = vec![0.0f32; n_fft];
        let mut spec_l = vec![Complex32::new(0.0, 0.0); n_bins];
        let mut spec_r = vec![Complex32::new(0.0, 0.0); n_bins];

        let ch_stride = dim_f * dim_t;

        // Forward STFT (`center=True`, reflection padding, `spek[:, :, :3, :] *= 0`)
        for t in 0..dim_t {
            let center_start = (t * hop) as isize - (pad as isize);

            for n in 0..n_fft {
                let idx = reflect_index(center_start + (n as isize), chunk_len);
                time_buf[n] = chunk_l[idx] * self.hann_window[n];
            }
            let _ = self.fft_forward.process_with_scratch(
                &mut time_buf,
                &mut spec_l,
                &mut self.fft_scratch_fwd,
            );

            for n in 0..n_fft {
                let idx = reflect_index(center_start + (n as isize), chunk_len);
                time_buf[n] = chunk_r[idx] * self.hann_window[n];
            }
            let _ = self.fft_forward.process_with_scratch(
                &mut time_buf,
                &mut spec_r,
                &mut self.fft_scratch_fwd,
            );

            // Bins 0, 1, 2 remain 0.0 per `spek[:, :, :3, :] *= 0` in `separate.py` line 630
            for f in 3..dim_f {
                let ft = f * dim_t + t;
                self.input_tensor_buf[0 * ch_stride + ft] = spec_l[f].re;
                self.input_tensor_buf[1 * ch_stride + ft] = spec_l[f].im;
                self.input_tensor_buf[2 * ch_stride + ft] = spec_r[f].re;
                self.input_tensor_buf[3 * ch_stride + ft] = spec_r[f].im;
            }
        }

        let mut pred_cac = vec![0.0f32; single_elems];

        if enable_denoise_2pass {
            // Pack batch[1] = -batch[0] so both `+spek` and `-spek` execute in a single GPU batch=2 call!
            for i in 0..single_elems {
                self.input_tensor_buf[single_elems + i] = -self.input_tensor_buf[i];
            }
            let t_in = TensorRef::from_array_view((
                [2usize, 4, dim_f, dim_t],
                &self.input_tensor_buf[..2 * single_elems],
            ))
            .map_err(|e| e.to_string())?;

            let outputs = self
                .session
                .run(ort::inputs!["input" => t_in])
                .map_err(|e| format!("{} ONNX batch=2 run failed: {}", self.config.model_name, e))?;

            let (_, out_slice) = outputs["output"]
                .try_extract_tensor::<f32>()
                .map_err(|e| e.to_string())?;

            // `spec_pred = 0.5 * M(spek) - 0.5 * M(-spek)` (`separate.py` line 635)
            for i in 0..single_elems {
                pred_cac[i] = 0.5 * (out_slice[i] - out_slice[single_elems + i]);
            }
        } else {
            let t_in = TensorRef::from_array_view((
                [1usize, 4, dim_f, dim_t],
                &self.input_tensor_buf[..single_elems],
            ))
            .map_err(|e| e.to_string())?;

            let outputs = self
                .session
                .run(ort::inputs!["input" => t_in])
                .map_err(|e| format!("{} ONNX batch=1 run failed: {}", self.config.model_name, e))?;

            let (_, out_slice) = outputs["output"]
                .try_extract_tensor::<f32>()
                .map_err(|e| e.to_string())?;
            pred_cac.copy_from_slice(&out_slice[..single_elems]);
        }

        // Inverse STFT (`torch.istft(..., center=True)` with exact COLA `sum w^2` normalization)
        let padded_len = chunk_len + 2 * pad;
        let mut ola_l = vec![0.0f32; padded_len];
        let mut ola_r = vec![0.0f32; padded_len];
        let mut win_sq = vec![0.0f32; padded_len];
        let inv_n = 1.0 / (n_fft as f32);

        for t in 0..dim_t {
            spec_l.fill(Complex32::new(0.0, 0.0));
            spec_r.fill(Complex32::new(0.0, 0.0));

            for f in 0..dim_f {
                let ft = f * dim_t + t;
                spec_l[f] = Complex32::new(
                    pred_cac[0 * ch_stride + ft],
                    pred_cac[1 * ch_stride + ft],
                );
                spec_r[f] = Complex32::new(
                    pred_cac[2 * ch_stride + ft],
                    pred_cac[3 * ch_stride + ft],
                );
            }
            spec_l[0].im = 0.0;
            spec_l[n_bins - 1].im = 0.0;
            spec_r[0].im = 0.0;
            spec_r[n_bins - 1].im = 0.0;

            let start = t * hop;

            let _ = self.fft_inverse.process_with_scratch(
                &mut spec_l,
                &mut time_buf,
                &mut self.fft_scratch_inv,
            );
            for n in 0..n_fft {
                let w = self.hann_window[n];
                ola_l[start + n] += time_buf[n] * inv_n * w;
                win_sq[start + n] += w * w;
            }

            let _ = self.fft_inverse.process_with_scratch(
                &mut spec_r,
                &mut time_buf,
                &mut self.fft_scratch_inv,
            );
            for n in 0..n_fft {
                let w = self.hann_window[n];
                ola_r[start + n] += time_buf[n] * inv_n * w;
            }
        }

        for i in 0..chunk_len {
            let idx = pad + i;
            let norm = win_sq[idx].max(1e-8);
            out_l[i] = ola_l[idx] / norm;
            out_r[i] = ola_r[idx] / norm;
        }

        Ok(())
    }

    /// Separates the model's native primary stem (`Vocals` for `Kim_Vocal_2`,
    /// `Instrumental` for `UVR-MDX-NET-Inst_HQ_3`) at `44,100 Hz`.
    pub fn separate_primary_44k(
        &mut self,
        mix_44k: &[StereoFrame],
        enable_denoise_2pass: bool,
    ) -> Result<Vec<StereoFrame>, String> {
        let orig_len = mix_44k.len();
        if orig_len == 0 {
            return Ok(Vec::new());
        }

        let n_fft = self.config.n_fft;
        let hop = self.config.hop_length;
        let dim_t = self.config.dim_t;
        let trim = n_fft / 2;
        let chunk_size = hop * (dim_t - 1); // 261,120 samples
        let gen_size = chunk_size - 2 * trim;
        let step = chunk_size - n_fft; // UVR5 `DEFAULT` overlap step

        let pad = gen_size + trim - (orig_len % gen_size);
        let mix_padded_len = trim + orig_len + pad;

        let mut mix_l = vec![0.0f32; mix_padded_len];
        let mut mix_r = vec![0.0f32; mix_padded_len];
        for (i, f) in mix_44k.iter().enumerate() {
            mix_l[trim + i] = f[0];
            mix_r[trim + i] = f[1];
        }

        let mut res_l = vec![0.0f32; mix_padded_len];
        let mut res_r = vec![0.0f32; mix_padded_len];
        let mut divider = vec![0.0f32; mix_padded_len];

        let mut chunk_in_l = vec![0.0f32; chunk_size];
        let mut chunk_in_r = vec![0.0f32; chunk_size];
        let mut chunk_out_l = vec![0.0f32; chunk_size];
        let mut chunk_out_r = vec![0.0f32; chunk_size];

        let mut start = 0usize;
        while start < mix_padded_len {
            let end = (start + chunk_size).min(mix_padded_len);
            let actual = end - start;

            chunk_in_l.fill(0.0);
            chunk_in_r.fill(0.0);
            chunk_in_l[..actual].copy_from_slice(&mix_l[start..end]);
            chunk_in_r[..actual].copy_from_slice(&mix_r[start..end]);

            self.infer_single_chunk(
                &chunk_in_l,
                &chunk_in_r,
                enable_denoise_2pass,
                &mut chunk_out_l,
                &mut chunk_out_r,
            )?;

            // Midpoint-sampled Hann crossfade window (strictly > 0 at endpoints, never 0/0)
            let denom = actual.max(1) as f32;
            for i in 0..actual {
                let w = 0.5 * (1.0 - (2.0 * PI * ((i as f32) + 0.5) / denom).cos());
                res_l[start + i] += chunk_out_l[i] * w;
                res_r[start + i] += chunk_out_r[i] * w;
                divider[start + i] += w;
            }

            if end == mix_padded_len {
                break;
            }
            start += step;
        }

        let comp = self.config.compensate;
        let mut out = vec![[0.0f32; 2]; orig_len];
        for i in 0..orig_len {
            let idx = trim + i;
            let div = divider[idx].max(1e-8);
            out[i] = [(res_l[idx] / div) * comp, (res_r[idx] / div) * comp];
        }

        Ok(out)
    }

    /// Separates `(vocals, instrumental)` at arbitrary `sample_rate_hz` (automatically resampling
    /// to/from `44,100 Hz` via linear-phase `rubato` when `sample_rate_hz != 44,100`).
    pub fn separate_vocal_instrumental(
        &mut self,
        mix_frames: &[StereoFrame],
        sample_rate_hz: u32,
        enable_denoise_2pass: bool,
    ) -> Result<(Vec<StereoFrame>, Vec<StereoFrame>), String> {
        if mix_frames.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let sr = sample_rate_hz.max(8_000);
        let mix_44k = resample_stereo_frames(mix_frames, sr, UVR5_MDX_SAMPLE_RATE_HZ);
        let primary_44k = self.separate_primary_44k(&mix_44k, enable_denoise_2pass)?;

        match self.config.primary_stem {
            MdxPrimaryStem::Vocals => {
                let vocals = resample_stereo_frames(&primary_44k, UVR5_MDX_SAMPLE_RATE_HZ, sr);
                let inst = SpectralEnsembler::invert_audio_waveform(mix_frames, &vocals);
                Ok((vocals, inst))
            }
            MdxPrimaryStem::Instrumental => {
                let vocals_44k = SpectralEnsembler::invert_audio_waveform(&mix_44k, &primary_44k);
                let vocals = resample_stereo_frames(&vocals_44k, UVR5_MDX_SAMPLE_RATE_HZ, sr);
                let inst = SpectralEnsembler::invert_audio_waveform(mix_frames, &vocals);
                Ok((vocals, inst))
            }
        }
    }
}

/// Model routing selector for Khu 6 (`UVR5 Ensemble Studio`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuStemModelSelection {
    /// Full 3-Model Studio Ensemble (`Kim_Vocal_2` + `UVR-MDX-NET-Inst_HQ_3` + `Ozone 12 BSRNN`).
    Ensemble3Model,
    /// `Kim_Vocal_2` primary vocal extraction + BSRNN sub-stem projection for Drums/Bass/Other.
    KimVocal2Only,
    /// `UVR-MDX-NET-Inst_HQ_3` primary instrumental extraction + BSRNN sub-stem projection.
    InstHq3Only,
    /// Single-pass `Ozone 12 BSRNN` 4-stem separation.
    OzoneBsrnnOnly,
}

impl GpuStemModelSelection {
    pub fn from_model_tag(tag: &str) -> Self {
        let lower = tag.trim().to_ascii_lowercase();
        if lower.contains("kim_vocal")
            || lower.contains("970b3f9492014d18fefeedfe4773cb42")
            || lower == "mdx23c"
        {
            if lower.contains("ensemble") || lower.contains("+") {
                Self::Ensemble3Model
            } else {
                Self::KimVocal2Only
            }
        } else if lower.contains("inst_hq_3")
            || lower.contains("55657dd70583b0fedfba5f67df11d711")
        {
            if lower.contains("ensemble") || lower.contains("+") {
                Self::Ensemble3Model
            } else {
                Self::InstHq3Only
            }
        } else if lower.contains("bsrnn") && !lower.contains("ensemble") {
            Self::OzoneBsrnnOnly
        } else {
            Self::Ensemble3Model
        }
    }
}

/// Configuration for `GpuStemEnsembleEngine`.
#[derive(Clone, Debug)]
pub struct GpuStemEnsembleConfig {
    pub model_selection: GpuStemModelSelection,
    pub ensemble_mode: SpectralEnsembleMode,
    pub weights: [f32; 3],
    pub enable_denoise_2pass: bool,
    pub vr_aggressiveness: f32,
    pub phase_denoise_strength: f32,
    pub invert_instrumental: bool,
}

impl Default for GpuStemEnsembleConfig {
    fn default() -> Self {
        Self {
            model_selection: GpuStemModelSelection::Ensemble3Model,
            ensemble_mode: SpectralEnsembleMode::Weighted,
            weights: [0.45, 0.35, 0.20],
            enable_denoise_2pass: true,
            vr_aggressiveness: 0.15,
            phase_denoise_strength: 0.20,
            invert_instrumental: true,
        }
    }
}

/// Complete 4-Stem + Inverted-Instrumental output from `GpuStemEnsembleEngine`.
#[derive(Clone, Debug)]
pub struct GpuStemEnsembleOutput {
    /// Separated lead + backing vocals (`len == mix.len()`).
    pub vocals: Vec<StereoFrame>,
    /// Separated drum stem projected onto vocal-free instrumental (`len == mix.len()`).
    pub drums: Vec<StereoFrame>,
    /// Separated bass stem projected onto vocal-free instrumental (`len == mix.len()`).
    pub bass: Vec<StereoFrame>,
    /// Separated melodic/harmonic `Other` stem such that `vocals + drums + bass + other == mix`!
    pub other: Vec<StereoFrame>,
    /// Exact phase-inverted full instrumental `inv_instrumental = mix - vocals`.
    pub inv_instrumental: Vec<StereoFrame>,
    /// Hardware execution tier used (`CudaTensorCore` on RTX 4080).
    pub active_tier: GpuExecutionTier,
    /// Total GPU inference + STFT/ensemble wall-clock time in milliseconds.
    pub elapsed_ms: f64,
}

/// Projects BSRNN's `[drums, bass, other]` onto `inst_master` via complex STFT ratio masking so
/// `drums` and `bass` inherit zero vocal bleed from `inst_master` AND `drums + bass + other == inst_master`
/// identically to `f32` precision!
fn project_instrumental_substems(
    inst_master: &[StereoFrame],
    bsrnn_drums: &[StereoFrame],
    bsrnn_bass: &[StereoFrame],
    bsrnn_other: &[StereoFrame],
) -> (Vec<StereoFrame>, Vec<StereoFrame>, Vec<StereoFrame>) {
    let len = inst_master.len();
    if len == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let n_fft = 2048usize;
    let hop = 512usize;
    let f_bins = n_fft / 2 + 1;
    let pad = n_fft / 2;
    let padded_len = len + 2 * pad;
    let num_frames = (padded_len.saturating_sub(n_fft)) / hop + 1;

    let mut planner = RealFftPlanner::<f32>::new();
    let fft_fwd = planner.plan_fft_forward(n_fft);
    let fft_inv = planner.plan_fft_inverse(n_fft);

    let mut hann = vec![0.0f32; n_fft];
    for i in 0..n_fft {
        hann[i] = 0.5 * (1.0 - (2.0 * PI * (i as f32) / (n_fft as f32)).cos());
    }

    let mut drm_ola_l = vec![0.0f32; padded_len + n_fft];
    let mut drm_ola_r = vec![0.0f32; padded_len + n_fft];
    let mut bas_ola_l = vec![0.0f32; padded_len + n_fft];
    let mut bas_ola_r = vec![0.0f32; padded_len + n_fft];
    let mut win_norm = vec![0.0f32; padded_len + n_fft];

    let mut time_buf = vec![0.0f32; n_fft];
    let mut spec_inst = vec![Complex32::new(0.0, 0.0); f_bins];
    let mut spec_d = vec![Complex32::new(0.0, 0.0); f_bins];
    let mut spec_b = vec![Complex32::new(0.0, 0.0); f_bins];
    let mut spec_o = vec![Complex32::new(0.0, 0.0); f_bins];
    let mut out_d = vec![Complex32::new(0.0, 0.0); f_bins];
    let mut out_b = vec![Complex32::new(0.0, 0.0); f_bins];
    let inv_n = 1.0 / (n_fft as f32);

    for t in 0..num_frames {
        let start = t * hop;
        for i in 0..n_fft {
            win_norm[start + i] += hann[i] * hann[i];
        }

        for ch in 0..2 {
            let fill_and_fft = |src: &[StereoFrame],
                                tb: &mut [f32],
                                dst: &mut [Complex32]| {
                for i in 0..n_fft {
                    let idx = start + i;
                    let s = if idx >= pad && (idx - pad) < src.len() {
                        src[idx - pad][ch]
                    } else {
                        0.0
                    };
                    tb[i] = s * hann[i];
                }
                let _ = fft_fwd.process(tb, dst);
            };

            fill_and_fft(inst_master, &mut time_buf, &mut spec_inst);
            fill_and_fft(bsrnn_drums, &mut time_buf, &mut spec_d);
            fill_and_fft(bsrnn_bass, &mut time_buf, &mut spec_b);
            fill_and_fft(bsrnn_other, &mut time_buf, &mut spec_o);

            for k in 0..f_bins {
                let md = spec_d[k].norm();
                let mb = spec_b[k].norm();
                let mo = spec_o[k].norm();
                let msum = md + mb + mo;
                if msum > 1e-9 {
                    let wd = (md / msum).clamp(0.0, 1.0);
                    let wb = (mb / msum).clamp(0.0, 1.0 - wd);
                    out_d[k] = spec_inst[k] * wd;
                    out_b[k] = spec_inst[k] * wb;
                } else {
                    out_d[k] = Complex32::new(0.0, 0.0);
                    out_b[k] = Complex32::new(0.0, 0.0);
                }
            }
            out_d[0].im = 0.0;
            out_d[f_bins - 1].im = 0.0;
            out_b[0].im = 0.0;
            out_b[f_bins - 1].im = 0.0;

            let _ = fft_inv.process(&mut out_d, &mut time_buf);
            let dst_d = if ch == 0 {
                &mut drm_ola_l
            } else {
                &mut drm_ola_r
            };
            for i in 0..n_fft {
                dst_d[start + i] += time_buf[i] * inv_n * hann[i];
            }

            let _ = fft_inv.process(&mut out_b, &mut time_buf);
            let dst_b = if ch == 0 {
                &mut bas_ola_l
            } else {
                &mut bas_ola_r
            };
            for i in 0..n_fft {
                dst_b[start + i] += time_buf[i] * inv_n * hann[i];
            }
        }
    }

    let mut drums = vec![[0.0f32; 2]; len];
    let mut bass = vec![[0.0f32; 2]; len];
    let mut other = vec![[0.0f32; 2]; len];
    for i in 0..len {
        let idx = pad + i;
        let norm = win_norm[idx].max(1e-8);
        let dl = drm_ola_l[idx] / norm;
        let dr = drm_ola_r[idx] / norm;
        let bl = bas_ola_l[idx] / norm;
        let br = bas_ola_r[idx] / norm;
        drums[i] = [dl, dr];
        bass[i] = [bl, br];
        other[i] = [inst_master[i][0] - dl - bl, inst_master[i][1] - dr - br];
    }

    (drums, bass, other)
}

/// Applies spectral phase-coherence / soft Wiener gating when `phase_denoise_strength > 0`
/// in modes where `SpectralEnsembler` does not already apply inter-model circular gating
/// (e.g. `MaxSpec`, `MinSpec`, `Median`, or single-model runs).
fn apply_spectral_phase_denoise_vs_mix(
    vocals: &[StereoFrame],
    mix: &[StereoFrame],
    strength: f32,
) -> Vec<StereoFrame> {
    let denoise = strength.clamp(0.0, 1.0);
    let len = vocals.len().min(mix.len());
    if len == 0 || denoise <= 1e-6 {
        return vocals.to_vec();
    }

    let n_fft = 1024usize;
    let hop = 256usize;
    let f_bins = n_fft / 2 + 1;
    let pad = n_fft / 2;
    let padded_len = len + 2 * pad;
    let num_frames = (padded_len.saturating_sub(n_fft)) / hop + 1;

    let mut planner = RealFftPlanner::<f32>::new();
    let fft_fwd = planner.plan_fft_forward(n_fft);
    let fft_inv = planner.plan_fft_inverse(n_fft);

    let mut hann = vec![0.0f32; n_fft];
    for i in 0..n_fft {
        hann[i] = 0.5 * (1.0 - (2.0 * PI * (i as f32) / (n_fft as f32)).cos());
    }

    let mut ola_l = vec![0.0f32; padded_len + n_fft];
    let mut ola_r = vec![0.0f32; padded_len + n_fft];
    let mut win_norm = vec![0.0f32; padded_len + n_fft];
    let mut tb = vec![0.0f32; n_fft];
    let mut sv = vec![Complex32::new(0.0, 0.0); f_bins];
    let mut sm = vec![Complex32::new(0.0, 0.0); f_bins];
    let inv_n = 1.0 / (n_fft as f32);

    for t in 0..num_frames {
        let start = t * hop;
        for i in 0..n_fft {
            win_norm[start + i] += hann[i] * hann[i];
        }
        for ch in 0..2 {
            for i in 0..n_fft {
                let idx = start + i;
                let v = if idx >= pad && (idx - pad) < len {
                    vocals[idx - pad][ch]
                } else {
                    0.0
                };
                tb[i] = v * hann[i];
            }
            let _ = fft_fwd.process(&mut tb, &mut sv);

            for i in 0..n_fft {
                let idx = start + i;
                let m = if idx >= pad && (idx - pad) < len {
                    mix[idx - pad][ch]
                } else {
                    0.0
                };
                tb[i] = m * hann[i];
            }
            let _ = fft_fwd.process(&mut tb, &mut sm);

            for k in 0..f_bins {
                let mv = sv[k].norm();
                let mm = sm[k].norm();
                if mv > 1e-12 && mm > 1e-12 {
                    let cos_phi = ((sv[k].re * sm[k].re + sv[k].im * sm[k].im) / (mv * mm))
                        .clamp(0.0, 1.0);
                    let ratio = (mv / (mm + 1e-9)).clamp(0.0, 1.0);
                    let coherence = (0.5 * cos_phi + 0.5 * ratio).clamp(0.0, 1.0);
                    let gate = (1.0 - denoise * 0.65) + (denoise * 0.65) * coherence;
                    sv[k] *= gate;
                }
            }
            sv[0].im = 0.0;
            sv[f_bins - 1].im = 0.0;

            let _ = fft_inv.process(&mut sv, &mut tb);
            let dst = if ch == 0 { &mut ola_l } else { &mut ola_r };
            for i in 0..n_fft {
                dst[start + i] += tb[i] * inv_n * hann[i];
            }
        }
    }

    let mut out = vec![[0.0f32; 2]; len];
    for i in 0..len {
        let idx = pad + i;
        let norm = win_norm[idx].max(1e-8);
        out[i] = [ola_l[idx] / norm, ola_r[idx] / norm];
    }
    out
}

/// Complete 3-Model GPU Ensemble Studio Engine combining:
/// 1. `Kim_Vocal_2.onnx` (UVR5 TFC-TDF v3 Vocal flagship)
/// 2. `UVR-MDX-NET-Inst_HQ_3.onnx` (UVR5 TFC-TDF v3 Instrumental flagship)
/// 3. `ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx` (Ozone 12 BSRNN 4-Stem separator)
pub struct GpuStemEnsembleEngine {
    kim_vocal: Uvr5MdxOnnxSeparator,
    inst_hq3: Uvr5MdxOnnxSeparator,
    ozone_bsrnn: OzoneBsrnnOnnxSeparator,
    active_tier: GpuExecutionTier,
}

impl GpuStemEnsembleEngine {
    pub fn new() -> Result<Self, String> {
        let kim_vocal = Uvr5MdxOnnxSeparator::from_kim_vocal_2()?;
        let inst_hq3 = Uvr5MdxOnnxSeparator::from_inst_hq_3()?;
        let ozone_bsrnn = OzoneBsrnnOnnxSeparator::from_default_assets()?;
        let active_tier = kim_vocal.active_tier();
        Ok(Self {
            kim_vocal,
            inst_hq3,
            ozone_bsrnn,
            active_tier,
        })
    }

    #[inline]
    pub fn active_tier(&self) -> GpuExecutionTier {
        self.active_tier
    }

    pub fn separate_ensemble(
        &mut self,
        mix_frames: &[StereoFrame],
        sample_rate_hz: u32,
        cfg: &GpuStemEnsembleConfig,
    ) -> Result<GpuStemEnsembleOutput, String> {
        let t0 = Instant::now();
        let len = mix_frames.len();
        if len == 0 {
            return Ok(GpuStemEnsembleOutput {
                vocals: Vec::new(),
                drums: Vec::new(),
                bass: Vec::new(),
                other: Vec::new(),
                inv_instrumental: Vec::new(),
                active_tier: self.active_tier,
                elapsed_ms: 0.0,
            });
        }

        let sr = sample_rate_hz.max(8_000);

        // Always run Ozone 12 BSRNN (at its native 48 kHz) so we have genuine neural
        // 4-stem sub-components (`Vocals, Drums, Bass, Other`)!
        let mix_48k = resample_stereo_frames(mix_frames, sr, 48_000);
        let [bsrnn_v_48k, bsrnn_d_48k, bsrnn_b_48k, bsrnn_o_48k] =
            self.ozone_bsrnn.separate_4stems_offline(&mix_48k)?;
        let bsrnn_vocals = resample_stereo_frames(&bsrnn_v_48k, 48_000, sr);
        let bsrnn_drums = resample_stereo_frames(&bsrnn_d_48k, 48_000, sr);
        let bsrnn_bass = resample_stereo_frames(&bsrnn_b_48k, 48_000, sr);
        let bsrnn_other = resample_stereo_frames(&bsrnn_o_48k, 48_000, sr);

        let raw_vocals = match cfg.model_selection {
            GpuStemModelSelection::Ensemble3Model => {
                let (kim_vocals, _) = self.kim_vocal.separate_vocal_instrumental(
                    mix_frames,
                    sr,
                    cfg.enable_denoise_2pass,
                )?;
                let (inst_hq3_vocals, _) = self.inst_hq3.separate_vocal_instrumental(
                    mix_frames,
                    sr,
                    cfg.enable_denoise_2pass,
                )?;

                let n_fft_ens = if len >= 4096 { 2048 } else { 512 };
                let hop_ens = n_fft_ens / 4;
                let mut ensembler =
                    SpectralEnsembler::new(n_fft_ens, hop_ens, cfg.ensemble_mode);
                ensembler.phase_denoise_strength = cfg.phase_denoise_strength.clamp(0.0, 1.0);
                let combined = ensembler.ensemble_waveforms(
                    &[&kim_vocals, &inst_hq3_vocals, &bsrnn_vocals],
                    Some(&cfg.weights),
                );

                // For modes where `SpectralEnsembler` doesn't gate (`MaxSpec`, `MinSpec`, `Median`),
                // apply explicit spectral phase-coherence gating when requested.
                if cfg.phase_denoise_strength > 0.0
                    && matches!(
                        cfg.ensemble_mode,
                        SpectralEnsembleMode::MaxSpec
                            | SpectralEnsembleMode::MinSpec
                            | SpectralEnsembleMode::Median
                    )
                {
                    apply_spectral_phase_denoise_vs_mix(
                        &combined,
                        mix_frames,
                        cfg.phase_denoise_strength,
                    )
                } else {
                    combined
                }
            }
            GpuStemModelSelection::KimVocal2Only => {
                let (v, _) = self.kim_vocal.separate_vocal_instrumental(
                    mix_frames,
                    sr,
                    cfg.enable_denoise_2pass,
                )?;
                apply_spectral_phase_denoise_vs_mix(&v, mix_frames, cfg.phase_denoise_strength)
            }
            GpuStemModelSelection::InstHq3Only => {
                let (v, _) = self.inst_hq3.separate_vocal_instrumental(
                    mix_frames,
                    sr,
                    cfg.enable_denoise_2pass,
                )?;
                apply_spectral_phase_denoise_vs_mix(&v, mix_frames, cfg.phase_denoise_strength)
            }
            GpuStemModelSelection::OzoneBsrnnOnly => apply_spectral_phase_denoise_vs_mix(
                &bsrnn_vocals,
                mix_frames,
                cfg.phase_denoise_strength,
            ),
        };

        // Exact phase-inverted full instrumental: `inv_instrumental = mix - vocals`
        let inv_instrumental = SpectralEnsembler::invert_audio_waveform(mix_frames, &raw_vocals);

        // Project BSRNN's `[Drums, Bass, Other]` onto `inv_instrumental` so:
        // 1) `drums` and `bass` contain zero vocal bleed, AND
        // 2) `vocals + drums + bass + other == mix_frames` identically!
        let (drums, bass, other) = project_instrumental_substems(
            &inv_instrumental,
            &bsrnn_drums,
            &bsrnn_bass,
            &bsrnn_other,
        );

        Ok(GpuStemEnsembleOutput {
            vocals: raw_vocals,
            drums,
            bass,
            other,
            inv_instrumental,
            active_tier: self.active_tier,
            elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
        })
    }
}

static GLOBAL_GPU_ENSEMBLE_ENGINE: OnceLock<Mutex<Option<GpuStemEnsembleEngine>>> = OnceLock::new();

/// Runs the cached global RTX 4080 `GpuStemEnsembleEngine` (lazily initializing the 3 ONNX
/// sessions once on first offline separation and keeping them warm in VRAM for instant reuse).
/// When ONNX Runtime DLLs are unavailable on a headless CPU host, cleanly falls back to the
/// real 3-model trained `.npz` `GruThreeStemSeparator` (`PureRustFallback`).
pub fn run_global_gpu_stem_ensemble(
    mix_frames: &[StereoFrame],
    sample_rate_hz: u32,
    cfg: &GpuStemEnsembleConfig,
) -> Result<GpuStemEnsembleOutput, String> {
    let mutex = GLOBAL_GPU_ENSEMBLE_ENGINE.get_or_init(|| Mutex::new(None));
    let mut guard = mutex
        .lock()
        .map_err(|_| "Global GpuStemEnsembleEngine mutex poisoned".to_string())?;
    if guard.is_none() {
        match GpuStemEnsembleEngine::new() {
            Ok(engine) => {
                *guard = Some(engine);
            }
            Err(onnx_err) => {
                let t0 = Instant::now();
                if let Some(mut gru) = crate::npz_gru_legacy::GruThreeStemSeparator::from_global_cache()
                {
                    let sr = sample_rate_hz.max(8_000);
                    let [raw_vocals, drums_raw, bass_raw, other_raw] =
                        gru.separate_4stems(mix_frames, sr);
                    let vocals = apply_spectral_phase_denoise_vs_mix(
                        &raw_vocals,
                        mix_frames,
                        cfg.phase_denoise_strength,
                    );
                    let inv_instrumental =
                        SpectralEnsembler::invert_audio_waveform(mix_frames, &vocals);
                    let (drums, bass, other) = project_instrumental_substems(
                        &inv_instrumental,
                        &drums_raw,
                        &bass_raw,
                        &other_raw,
                    );
                    return Ok(GpuStemEnsembleOutput {
                        vocals,
                        drums,
                        bass,
                        other,
                        inv_instrumental,
                        active_tier: GpuExecutionTier::PureRustFallback,
                        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
                    });
                }
                return Err(onnx_err);
            }
        }
    }
    guard
        .as_mut()
        .expect("GpuStemEnsembleEngine initialized")
        .separate_ensemble(mix_frames, sample_rate_hz, cfg)
}
