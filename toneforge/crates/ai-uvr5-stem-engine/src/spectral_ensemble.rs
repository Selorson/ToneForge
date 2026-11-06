//! UVR5 Multi-Model Spectral Ensembler (`SpectralEnsembler`, `lib_v5/spec_utils.py`).
//!
//! Supports all UVR5 spectral combination algorithms:
//! - `MaxSpec` (`MAX_SPEC`): Selects the complex bin `z_m*(t, f)` with maximum magnitude `|z_m|`,
//!   preserving exact complex phase (`arg(z_m*)`).
//! - `MinSpec` (`MIN_SPEC`): Selects the complex bin `z_m*(t, f)` with minimum magnitude `|z_m|`,
//!   suppressing model-specific bleed while preserving complex phase.
//! - `Average` (`AVERAGE`): Complex / phase-aligned average across models.
//! - `Median` (`MEDIAN_SPEC`): Selects the model bin with median magnitude `|z_m|`.
//! - `Weighted` (`WEIGHTED_ENSEMBLE`): Weighted magnitude & circular mean phase with inter-model
//!   phase coherence denoising (`R(t,f) = |sum w_m e^{j*phi_m}|`).
//! - `invert_audio`: Time-domain and spectral-domain phase-inverted residual stem extraction.

use daw_core_graph::StereoFrame;
use realfft::RealFftPlanner;
use rustfft::num_complex::Complex32;
use std::f32::consts::PI;

/// UVR5 Spectral Ensemble combination mode (`lib_v5/spec_utils.py` `ensembler`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpectralEnsembleMode {
    /// `MAX_SPEC`: `z_out(t, f) = z_{argmax_m |z_m(t, f)|}(t, f)`
    MaxSpec,
    /// `MIN_SPEC`: `z_out(t, f) = z_{argmin_m |z_m(t, f)|}(t, f)`
    MinSpec,
    /// `AVERAGE`: `z_out(t, f) = (1 / M) * sum_m z_m(t, f)`
    Average,
    /// `MEDIAN`: `z_out(t, f) = z_{median_m |z_m(t, f)|}(t, f)`
    Median,
    /// `WEIGHTED`: Weighted ensemble with inter-model phase coherence denoising.
    Weighted,
}

/// Combines multiple complex STFT spectrogram frames `model_frames[m][bin]` into `out_frame[bin]`
/// using UVR5's phase-preserving `MAX_SPEC` / `MIN_SPEC` / `AVERAGE` / `Median` / `Weighted` rules.
///
/// Strictly **Zero Heap Allocation**.
#[inline]
pub fn combine_complex_spectrogram_frame(
    mode: SpectralEnsembleMode,
    model_frames: &[&[Complex32]],
    out_frame: &mut [Complex32],
) {
    combine_complex_spectrogram_frame_weighted(mode, model_frames, None, 0.0, out_frame);
}

/// Weighted & phase-denoised variant of `combine_complex_spectrogram_frame`.
/// Strictly **Zero Heap Allocation** (uses a fixed stack scratch buffer for up to 16 models).
pub fn combine_complex_spectrogram_frame_weighted(
    mode: SpectralEnsembleMode,
    model_frames: &[&[Complex32]],
    weights: Option<&[f32]>,
    phase_denoise_strength: f32,
    out_frame: &mut [Complex32],
) {
    if model_frames.is_empty() {
        out_frame.fill(Complex32::new(0.0, 0.0));
        return;
    }

    let bins = out_frame.len();
    let num_models = model_frames.len().min(16);
    let inv_m = 1.0 / (num_models as f32);
    let denoise = phase_denoise_strength.clamp(0.0, 1.0);

    for k in 0..bins {
        match mode {
            SpectralEnsembleMode::MaxSpec => {
                let mut best = model_frames[0][k];
                let mut best_mag_sq = best.norm_sqr();
                for m in 1..num_models {
                    let c = model_frames[m][k];
                    let mag_sq = c.norm_sqr();
                    if mag_sq > best_mag_sq {
                        best_mag_sq = mag_sq;
                        best = c;
                    }
                }
                out_frame[k] = best;
            }
            SpectralEnsembleMode::MinSpec => {
                let mut best = model_frames[0][k];
                let mut best_mag_sq = best.norm_sqr();
                for m in 1..num_models {
                    let c = model_frames[m][k];
                    let mag_sq = c.norm_sqr();
                    if mag_sq < best_mag_sq {
                        best_mag_sq = mag_sq;
                        best = c;
                    }
                }
                out_frame[k] = best;
            }
            SpectralEnsembleMode::Average => {
                let mut sum_c = Complex32::new(0.0, 0.0);
                let mut sum_mag = 0.0f32;
                for m in 0..num_models {
                    let c = model_frames[m][k];
                    sum_c += c;
                    sum_mag += c.norm();
                }
                let avg_c = sum_c * inv_m;
                if denoise > 0.0 && sum_mag > 1e-12 {
                    // Inter-model phase coherence `R = |sum z_m| / sum |z_m| in [0, 1]`
                    let coherence = (sum_c.norm() / sum_mag).clamp(0.0, 1.0);
                    let gate = (1.0 - denoise) + denoise * coherence;
                    out_frame[k] = avg_c * gate;
                } else {
                    out_frame[k] = avg_c;
                }
            }
            SpectralEnsembleMode::Median => {
                let mut buf = [(0.0f32, Complex32::new(0.0, 0.0)); 16];
                for m in 0..num_models {
                    let c = model_frames[m][k];
                    buf[m] = (c.norm_sqr(), c);
                }
                // Insertion sort on stack array `buf[..num_models]`
                for i in 1..num_models {
                    let key = buf[i];
                    let mut j = i;
                    while j > 0 && buf[j - 1].0 > key.0 {
                        buf[j] = buf[j - 1];
                        j -= 1;
                    }
                    buf[j] = key;
                }
                let mid = num_models / 2;
                if (num_models & 1) == 1 {
                    out_frame[k] = buf[mid].1;
                } else {
                    out_frame[k] = (buf[mid - 1].1 + buf[mid].1) * 0.5;
                }
            }
            SpectralEnsembleMode::Weighted => {
                let mut w_sum = 0.0f32;
                let mut sum_c = Complex32::new(0.0, 0.0);
                let mut sum_mag = 0.0f32;
                for m in 0..num_models {
                    let w = weights
                        .and_then(|ws| ws.get(m).copied())
                        .unwrap_or(1.0)
                        .max(0.0);
                    let c = model_frames[m][k];
                    w_sum += w;
                    sum_c += c * w;
                    sum_mag += c.norm() * w;
                }
                let norm_w = if w_sum > 1e-9 { 1.0 / w_sum } else { inv_m };
                let target_mag = sum_mag * norm_w;
                let vec_len = sum_c.norm();
                if vec_len > 1e-12 {
                    // Preserve weighted circular mean phase (`sum_c / |sum_c|`) and apply phase coherence denoising
                    let unit_phase = sum_c / vec_len;
                    let coherence = (vec_len / sum_mag.max(1e-12)).clamp(0.0, 1.0);
                    let gate = (1.0 - denoise) + denoise * coherence;
                    out_frame[k] = unit_phase * (target_mag * gate);
                } else {
                    out_frame[k] = Complex32::new(0.0, 0.0);
                }
            }
        }
    }
}

/// Complete Multi-Model UVR5 Spectral Ensembler (`SpectralEnsembler`).
///
/// Combines multiple separated stem waveforms or spectrograms in the complex STFT domain
/// while preserving complex phase and suppressing inter-model phase noise.
pub struct SpectralEnsembler {
    pub n_fft: usize,
    pub hop_size: usize,
    pub mode: SpectralEnsembleMode,
    /// Inter-model circular phase coherence denoising strength in `[0.0, 1.0]`.
    pub phase_denoise_strength: f32,
}

impl Default for SpectralEnsembler {
    fn default() -> Self {
        Self::new(2048, 512, SpectralEnsembleMode::MaxSpec)
    }
}

impl SpectralEnsembler {
    pub fn new(n_fft: usize, hop_size: usize, mode: SpectralEnsembleMode) -> Self {
        let n = n_fft.max(256).next_power_of_two();
        let h = hop_size.clamp(64, n / 2);
        Self {
            n_fft: n,
            hop_size: h,
            mode,
            phase_denoise_strength: 0.0,
        }
    }

    /// Combines a slice of complex spectrogram frames in-place with Zero Heap Allocation.
    #[inline]
    pub fn combine_frames(
        &self,
        model_frames: &[&[Complex32]],
        weights: Option<&[f32]>,
        out_frame: &mut [Complex32],
    ) {
        combine_complex_spectrogram_frame_weighted(
            self.mode,
            model_frames,
            weights,
            self.phase_denoise_strength,
            out_frame,
        );
    }

    /// Ensembles multiple time-domain stereo waveforms `model_waveforms` in the complex STFT domain
    /// (`lib_v5/spec_utils.py` `ensembler`) and reconstructs the phase-coherent stereo waveform.
    pub fn ensemble_waveforms(
        &self,
        model_waveforms: &[&[StereoFrame]],
        weights: Option<&[f32]>,
    ) -> Vec<StereoFrame> {
        if model_waveforms.is_empty() {
            return Vec::new();
        }
        let len = model_waveforms.iter().map(|w| w.len()).min().unwrap_or(0);
        if len == 0 {
            return Vec::new();
        }
        if model_waveforms.len() == 1 {
            return model_waveforms[0][..len].to_vec();
        }

        let num_models = model_waveforms.len().min(16);
        let n_fft = self.n_fft;
        let hop = self.hop_size;
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

        let mut time_buf = vec![0.0f32; n_fft];
        let mut model_specs_l = vec![vec![Complex32::new(0.0, 0.0); f_bins]; num_models];
        let mut model_specs_r = vec![vec![Complex32::new(0.0, 0.0); f_bins]; num_models];
        let mut out_spec_l = vec![Complex32::new(0.0, 0.0); f_bins];
        let mut out_spec_r = vec![Complex32::new(0.0, 0.0); f_bins];
        let inv_n = 1.0 / (n_fft as f32);

        for t in 0..num_frames {
            let start = t * hop;
            for i in 0..n_fft {
                win_norm[start + i] += hann[i] * hann[i];
            }

            for m in 0..num_models {
                let wf = model_waveforms[m];
                for i in 0..n_fft {
                    let idx = start + i;
                    let s = if idx >= pad && (idx - pad) < len {
                        wf[idx - pad][0]
                    } else {
                        0.0
                    };
                    time_buf[i] = s * hann[i];
                }
                let _ = fft_fwd.process(&mut time_buf, &mut model_specs_l[m]);

                for i in 0..n_fft {
                    let idx = start + i;
                    let s = if idx >= pad && (idx - pad) < len {
                        wf[idx - pad][1]
                    } else {
                        0.0
                    };
                    time_buf[i] = s * hann[i];
                }
                let _ = fft_fwd.process(&mut time_buf, &mut model_specs_r[m]);
            }

            {
                let refs_l: Vec<&[Complex32]> =
                    model_specs_l.iter().map(|v| v.as_slice()).collect();
                let refs_r: Vec<&[Complex32]> =
                    model_specs_r.iter().map(|v| v.as_slice()).collect();
                self.combine_frames(&refs_l, weights, &mut out_spec_l);
                self.combine_frames(&refs_r, weights, &mut out_spec_r);
            }

            out_spec_l[0].im = 0.0;
            out_spec_l[f_bins - 1].im = 0.0;
            out_spec_r[0].im = 0.0;
            out_spec_r[f_bins - 1].im = 0.0;

            let _ = fft_inv.process(&mut out_spec_l, &mut time_buf);
            for i in 0..n_fft {
                ola_l[start + i] += time_buf[i] * inv_n * hann[i];
            }
            let _ = fft_inv.process(&mut out_spec_r, &mut time_buf);
            for i in 0..n_fft {
                ola_r[start + i] += time_buf[i] * inv_n * hann[i];
            }
        }

        let mut out = vec![[0.0f32; 2]; len];
        for i in 0..len {
            let idx = pad + i;
            let norm = win_norm[idx].max(1e-6);
            out[i] = [ola_l[idx] / norm, ola_r[idx] / norm];
        }
        out
    }

    /// UVR5 `invert_audio`: Extracts the secondary complement stem via phase-inverted subtraction
    /// `residual[n] = mix[n] - primary_stem[n]`.
    #[inline]
    pub fn invert_audio_waveform(
        mix: &[StereoFrame],
        primary_stem: &[StereoFrame],
    ) -> Vec<StereoFrame> {
        let len = mix.len().min(primary_stem.len());
        let mut out = vec![[0.0f32; 2]; len];
        for i in 0..len {
            out[i] = [
                mix[i][0] - primary_stem[i][0],
                mix[i][1] - primary_stem[i][1],
            ];
        }
        out
    }
}
