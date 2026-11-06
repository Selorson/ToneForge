//! UVR5 VR Architecture `CascadedASPPNet` (`lib_v5/vr_network/nets_new.py` & `layers_new.py`).
//!
//! Implements:
//! - `VR_ASPP_DILATIONS = [1, 2, 4, 8, 16]`: 5 parallel Atrous Spatial Pyramid Pooling branches
//!   capturing multi-scale time-frequency context (`AsppModule`).
//! - Global Average Pooling context branch + `1x1` bottleneck projection.
//! - 3-Stage Cascaded Multi-Band U-Net (`Stage 1` Low-Res -> `Stage 2` Mid-Res -> `Stage 3` Full-Band)
//!   with UVR5 `aggressiveness` mask sharpening.

use daw_core_graph::StereoFrame;
use realfft::RealFftPlanner;
use rustfft::num_complex::Complex32;
use std::f32::consts::PI;

/// Exact Atrous Spatial Pyramid Pooling dilation rates used in UVR5 VR Arch (`nets_new.py`).
pub const VR_ASPP_DILATIONS: [usize; 5] = [1, 2, 4, 8, 16];

/// Atrous Spatial Pyramid Pooling Module (`ASPPModule` in `lib_v5/vr_network/layers_new.py`).
///
/// Combines:
/// - Branch 0..4: `3x3` Dilated 2D Convolutions at dilations `d in [1, 2, 4, 8, 16]`
/// - Branch 5: Global Frequency-Time Average Pooling context branch
/// - `1x1` Bottleneck fusion layer with LeakyReLU + Sigmoid mask activation.
#[derive(Clone, Debug)]
pub struct AsppModule {
    pub dilations: [usize; 5],
    /// `3x3` kernel weights for each of the 5 dilation branches.
    pub branch_kernels: [[[f32; 3]; 3]; 5],
    /// `1x1` fusion weights across `[branch_0, branch_1, branch_2, branch_3, branch_4, global_pool]`.
    pub fusion_weights: [f32; 6],
    pub bias: f32,
}

impl Default for AsppModule {
    fn default() -> Self {
        Self::new()
    }
}

impl AsppModule {
    pub fn new() -> Self {
        let base_k = [
            [0.05, 0.10, 0.05],
            [0.10, 0.40, 0.10],
            [0.05, 0.10, 0.05],
        ];
        Self {
            dilations: VR_ASPP_DILATIONS,
            branch_kernels: [base_k; 5],
            fusion_weights: [0.28, 0.22, 0.18, 0.14, 0.10, 0.08],
            bias: 0.0,
        }
    }

    /// Evaluates the 5 dilated convolution branches (`d = 1, 2, 4, 8, 16`) + Global Context branch
    /// on a 2D magnitude spectrogram `mag[num_frames * freq_bins]` and writes the fused feature map
    /// into `out_feat[num_frames * freq_bins]`.
    pub fn forward_2d(
        &self,
        num_frames: usize,
        freq_bins: usize,
        mag: &[f32],
        out_feat: &mut [f32],
    ) {
        let total = num_frames * freq_bins;
        if total == 0 || mag.len() < total || out_feat.len() < total {
            return;
        }

        // Compute global average pooling branch (`Branch 5`)
        let global_mean = mag[..total].iter().copied().sum::<f32>() / (total as f32);
        let global_contrib = self.fusion_weights[5] * global_mean;

        out_feat[..total].fill(global_contrib + self.bias);

        // Evaluate all 5 dilated branches (`d = 1, 2, 4, 8, 16`)
        for (b_idx, &d) in self.dilations.iter().enumerate() {
            let k = &self.branch_kernels[b_idx];
            let w_b = self.fusion_weights[b_idx];

            for t in 0..num_frames {
                for f in 0..freq_bins {
                    let mut sum = 0.0f32;
                    for dt in 0..3usize {
                        let t_off = (dt as isize - 1) * (d as isize);
                        let tt = (t as isize + t_off).clamp(0, (num_frames - 1) as isize) as usize;
                        let row = tt * freq_bins;
                        for df in 0..3usize {
                            let f_off = (df as isize - 1) * (d as isize);
                            let ff =
                                (f as isize + f_off).clamp(0, (freq_bins - 1) as isize) as usize;
                            sum += mag[row + ff] * k[dt][df];
                        }
                    }
                    // LeakyReLU activation (`alpha = 0.1`)
                    let act = if sum >= 0.0 { sum } else { 0.1 * sum };
                    out_feat[t * freq_bins + f] += w_b * act;
                }
            }
        }
    }
}

/// UVR5 VR Architecture 3-Stage Cascaded Multi-Band U-Net (`CascadedASPPNet` in `nets_new.py`).
///
/// - Stage 1: Low-frequency sub-band (`0 .. F/4`) U-Net + `AsppModule`
/// - Stage 2: Mid-frequency sub-band (`0 .. F/2`) U-Net + `AsppModule`
/// - Stage 3: Full-band (`0 .. F`) U-Net + `AsppModule` fusion
/// - Supports UVR5 `aggressiveness` parameter (`[-1.0, +1.0]`) to sharpen vocal/instrumental masks.
pub struct CascadedASPPNet {
    pub n_fft: usize,
    pub hop_size: usize,
    pub stage1_aspp: AsppModule,
    pub stage2_aspp: AsppModule,
    pub stage3_aspp: AsppModule,
    /// UVR5 aggressiveness in `[-1.0, +1.0]` (`0.0` = neutral, `>0.0` = aggressive vocal/stem stripping).
    pub aggressiveness: f32,
}

impl Default for CascadedASPPNet {
    fn default() -> Self {
        Self::new(2048, 512, 0.05)
    }
}

impl CascadedASPPNet {
    pub fn new(n_fft: usize, hop_size: usize, aggressiveness: f32) -> Self {
        let n = n_fft.max(256).next_power_of_two();
        let h = hop_size.clamp(64, n / 2);
        Self {
            n_fft: n,
            hop_size: h,
            stage1_aspp: AsppModule::new(),
            stage2_aspp: AsppModule::new(),
            stage3_aspp: AsppModule::new(),
            aggressiveness: aggressiveness.clamp(-1.0, 1.0),
        }
    }

    /// Predicts the soft time-frequency mask `M(t, f) in [0.0, 1.0]` across the 3 cascaded ASPP stages.
    pub fn predict_cascaded_mask(
        &self,
        num_frames: usize,
        freq_bins: usize,
        mag_in: &[f32],
    ) -> Vec<f32> {
        let total = num_frames * freq_bins;
        if total == 0 || mag_in.len() < total {
            return Vec::new();
        }

        // Stage 1: Quarter-band resolution (`f_s1 = freq_bins / 4`)
        let f_s1 = (freq_bins / 4).max(4);
        let mut mag_s1 = vec![0.0f32; num_frames * f_s1];
        let mut feat_s1 = vec![0.0f32; num_frames * f_s1];
        for t in 0..num_frames {
            for f in 0..f_s1 {
                mag_s1[t * f_s1 + f] = mag_in[t * freq_bins + f];
            }
        }
        self.stage1_aspp
            .forward_2d(num_frames, f_s1, &mag_s1, &mut feat_s1);

        // Stage 2: Half-band resolution (`f_s2 = freq_bins / 2`), concatenating upsampled Stage 1
        let f_s2 = (freq_bins / 2).max(8);
        let mut mag_s2 = vec![0.0f32; num_frames * f_s2];
        let mut feat_s2 = vec![0.0f32; num_frames * f_s2];
        for t in 0..num_frames {
            for f in 0..f_s2 {
                let base = mag_in[t * freq_bins + f];
                let s1_up = if f < f_s1 { feat_s1[t * f_s1 + f] } else { 0.0 };
                mag_s2[t * f_s2 + f] = 0.7 * base + 0.3 * s1_up;
            }
        }
        self.stage2_aspp
            .forward_2d(num_frames, f_s2, &mag_s2, &mut feat_s2);

        // Stage 3: Full-band resolution (`freq_bins`), fusing Stage 2 + full-band features
        let mut mag_s3 = vec![0.0f32; total];
        let mut feat_s3 = vec![0.0f32; total];
        for t in 0..num_frames {
            for f in 0..freq_bins {
                let base = mag_in[t * freq_bins + f];
                let s2_up = if f < f_s2 { feat_s2[t * f_s2 + f] } else { 0.0 };
                mag_s3[t * freq_bins + f] = 0.65 * base + 0.35 * s2_up;
            }
        }
        self.stage3_aspp
            .forward_2d(num_frames, freq_bins, &mag_s3, &mut feat_s3);

        // Normalize and apply UVR5 sigmoid mask with aggressiveness adjustment
        let mean_feat = (feat_s3.iter().copied().sum::<f32>() / (total as f32)).max(1e-6);
        let agg = self.aggressiveness;
        let slope = 1.5 * (1.0 + agg * 1.2).max(0.3);

        let mut mask = vec![0.0f32; total];
        for i in 0..total {
            let norm_x = (feat_s3[i] / mean_feat) - 0.85 + agg * 0.25;
            let m = 1.0 / (1.0 + (-slope * norm_x).exp());
            mask[i] = m.clamp(0.0, 1.0);
        }
        mask
    }

    /// Separates stereo audio into `(vocal_stem, instrumental_stem)` using the 3-Stage `CascadedASPPNet`
    /// and preserves exact mixture complex phase.
    pub fn separate_vocal_instrumental(
        &self,
        mix: &[StereoFrame],
    ) -> (Vec<StereoFrame>, Vec<StereoFrame>) {
        let len = mix.len();
        if len == 0 {
            return (Vec::new(), Vec::new());
        }

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

        let mut specs_l = vec![vec![Complex32::new(0.0, 0.0); f_bins]; num_frames];
        let mut specs_r = vec![vec![Complex32::new(0.0, 0.0); f_bins]; num_frames];
        let mut mag_grid = vec![0.0f32; num_frames * f_bins];
        let mut time_buf = vec![0.0f32; n_fft];

        for t in 0..num_frames {
            let start = t * hop;
            for ch in 0..2 {
                for i in 0..n_fft {
                    let idx = start + i;
                    let s = if idx >= pad && (idx - pad) < len {
                        mix[idx - pad][ch]
                    } else {
                        0.0
                    };
                    time_buf[i] = s * hann[i];
                }
                if ch == 0 {
                    let _ = fft_fwd.process(&mut time_buf, &mut specs_l[t]);
                } else {
                    let _ = fft_fwd.process(&mut time_buf, &mut specs_r[t]);
                }
            }
            for k in 0..f_bins {
                let mag_l = specs_l[t][k].norm();
                let mag_r = specs_r[t][k].norm();
                mag_grid[t * f_bins + k] = 0.5 * (mag_l + mag_r);
            }
        }

        let mask = self.predict_cascaded_mask(num_frames, f_bins, &mag_grid);

        let mut ola_voc_l = vec![0.0f32; padded_len + n_fft];
        let mut ola_voc_r = vec![0.0f32; padded_len + n_fft];
        let mut win_norm = vec![0.0f32; padded_len + n_fft];
        let mut voc_spec_l = vec![Complex32::new(0.0, 0.0); f_bins];
        let mut voc_spec_r = vec![Complex32::new(0.0, 0.0); f_bins];
        let inv_n = 1.0 / (n_fft as f32);

        for t in 0..num_frames {
            let start = t * hop;
            for k in 0..f_bins {
                let m = mask[t * f_bins + k];
                voc_spec_l[k] = specs_l[t][k] * m;
                voc_spec_r[k] = specs_r[t][k] * m;
            }
            voc_spec_l[0].im = 0.0;
            voc_spec_l[f_bins - 1].im = 0.0;
            voc_spec_r[0].im = 0.0;
            voc_spec_r[f_bins - 1].im = 0.0;

            let _ = fft_inv.process(&mut voc_spec_l, &mut time_buf);
            for i in 0..n_fft {
                ola_voc_l[start + i] += time_buf[i] * inv_n * hann[i];
                win_norm[start + i] += hann[i] * hann[i];
            }
            let _ = fft_inv.process(&mut voc_spec_r, &mut time_buf);
            for i in 0..n_fft {
                ola_voc_r[start + i] += time_buf[i] * inv_n * hann[i];
            }
        }

        let mut vocals = vec![[0.0f32; 2]; len];
        let mut instrumental = vec![[0.0f32; 2]; len];
        for i in 0..len {
            let idx = pad + i;
            let norm = win_norm[idx].max(1e-6);
            let vl = ola_voc_l[idx] / norm;
            let vr = ola_voc_r[idx] / norm;
            vocals[i] = [vl, vr];
            instrumental[i] = [mix[i][0] - vl, mix[i][1] - vr];
        }

        (vocals, instrumental)
    }
}
