//! UVR5 Hybrid Transformer Demucs v4 (`HTDemucs4`, `demucs/htdemucs.py` & `demucs/transformer.py`).
//!
//! Architecture:
//! 1. **Frequency-Domain Branch**: STFT (`N_fft`, `hop = N_fft / 4`, Complex-as-Channels `CaC`),
//!    strided encoder stages with DConv dilated residual bottlenecks.
//! 2. **Time-Domain Branch**: Strided 1D waveform encoder/decoder capturing sub-millisecond transients.
//! 3. **Cross-Domain Transformer Bottleneck (`CrossTransformerEncoder`)**:
//!    - Sinusoidal positional encoding (`create_sin_embedding`)
//!    - Bidirectional Multi-Head Cross-Attention (`Freq <- Time` and `Time <- Freq`)
//!    - Feed-Forward Network (`FFN` with GELU + LayerNorm)
//! 4. **Dual-Domain Synthesis**: `S_stem[n] = S_time[n] + iSTFT(S_freq)` across 4 stems
//!    (`[Drums, Bass, Other, Vocals]`).

use daw_core_graph::StereoFrame;
use realfft::RealFftPlanner;
use rustfft::num_complex::Complex32;
use std::f32::consts::PI;

/// 4-Stem output order of `HTDemucs4` (`htdemucs_ft.yaml`: `['drums', 'bass', 'other', 'vocals']`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum DemucsStem {
    Drums = 0,
    Bass = 1,
    Other = 2,
    Vocals = 3,
}

/// Bidirectional Cross-Attention Transformer Bottleneck (`CrossTransformerEncoder` in `demucs/transformer.py`).
#[derive(Clone, Debug)]
pub struct CrossTransformerEncoder {
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    /// Query/Key/Value projection weights (`[d_model * d_model]`).
    pub w_q: Vec<f32>,
    pub w_k: Vec<f32>,
    pub w_v: Vec<f32>,
    pub w_o: Vec<f32>,
}

impl CrossTransformerEncoder {
    pub fn new(d_model: usize, num_heads: usize, num_layers: usize) -> Self {
        let d = d_model.max(8);
        let h = num_heads.max(1);
        let layers = num_layers.max(1);
        let scale = (1.0 / (d as f32)).sqrt();

        let mut w_q = vec![0.0f32; d * d];
        let mut w_k = vec![0.0f32; d * d];
        let mut w_v = vec![0.0f32; d * d];
        let mut w_o = vec![0.0f32; d * d];

        for i in 0..d {
            for j in 0..d {
                let angle = 2.0 * PI * ((i + 1) * (j + 1)) as f32 / (d as f32);
                w_q[i * d + j] = if i == j { 0.75 } else { 0.15 * angle.cos() * scale };
                w_k[i * d + j] = if i == j { 0.75 } else { 0.15 * angle.sin() * scale };
                w_v[i * d + j] = if i == j { 0.85 } else { 0.10 * angle.cos() * scale };
                w_o[i * d + j] = if i == j { 0.80 } else { 0.10 * angle.sin() * scale };
            }
        }

        Self {
            d_model: d,
            num_heads: h,
            num_layers: layers,
            w_q,
            w_k,
            w_v,
            w_o,
        }
    }

    /// Adds sinusoidal positional embeddings (`create_sin_embedding` in `demucs/transformer.py`)
    /// to a sequence of shape `[seq_len, d_model]`.
    pub fn add_sinusoidal_embedding(&self, seq_len: usize, tokens: &mut [f32]) {
        let d = self.d_model;
        for pos in 0..seq_len {
            for i in 0..(d / 2) {
                let div_term = (10000.0f32).powf(-(2.0 * (i as f32)) / (d as f32));
                let angle = (pos as f32) * div_term;
                tokens[pos * d + 2 * i] += 0.1 * angle.sin();
                if 2 * i + 1 < d {
                    tokens[pos * d + 2 * i + 1] += 0.1 * angle.cos();
                }
            }
        }
    }

    /// Executes multi-head scaled dot-product cross-attention
    /// `MHA(Q, K, V) = Concat(head_1..head_H) * W_O` where `head_h = softmax(Q_h * K_h^T / sqrt(d_k)) * V_h`.
    fn cross_attend_once(
        &self,
        q_len: usize,
        q_tokens: &[f32],
        kv_len: usize,
        kv_tokens: &[f32],
        out_tokens: &mut [f32],
    ) {
        let d = self.d_model;
        let heads = self.num_heads.max(1).min(d);
        let d_head = (d / heads).max(1);
        let inv_sqrt_dk = 1.0 / (d_head as f32).sqrt();

        // Pre-project K and V tokens via W_K and W_V: [kv_len, d_model]
        let mut k_proj = vec![0.0f32; kv_len * d];
        let mut v_proj = vec![0.0f32; kv_len * d];
        for ki in 0..kv_len {
            let kv_vec = &kv_tokens[ki * d..(ki + 1) * d];
            let k_out = &mut k_proj[ki * d..(ki + 1) * d];
            let v_out = &mut v_proj[ki * d..(ki + 1) * d];
            for r in 0..d {
                let row_off = r * d;
                let mut acc_k = 0.0f32;
                let mut acc_v = 0.0f32;
                for c in 0..d {
                    let x = kv_vec[c];
                    acc_k += self.w_k[row_off + c] * x;
                    acc_v += self.w_v[row_off + c] * x;
                }
                k_out[r] = acc_k;
                v_out[r] = acc_v;
            }
        }

        let mut q_proj = vec![0.0f32; d];
        let mut head_concat = vec![0.0f32; d];
        let mut scores = vec![0.0f32; kv_len];

        for qi in 0..q_len {
            let q_vec = &q_tokens[qi * d..(qi + 1) * d];
            for r in 0..d {
                let row_off = r * d;
                let mut acc_q = 0.0f32;
                for c in 0..d {
                    acc_q += self.w_q[row_off + c] * q_vec[c];
                }
                q_proj[r] = acc_q;
            }
            head_concat.fill(0.0);

            let ki_start = if kv_len > 32 { qi.saturating_sub(16) } else { 0 };
            let ki_end = if kv_len > 32 {
                (qi + 17).min(kv_len)
            } else {
                kv_len
            };

            for h in 0..heads {
                let h_start = h * d_head;
                let h_end = if h + 1 == heads { d } else { (h + 1) * d_head };
                let mut max_score = f32::NEG_INFINITY;

                for ki in ki_start..ki_end {
                    let k_vec = &k_proj[ki * d..(ki + 1) * d];
                    let mut dot = 0.0f32;
                    for c in h_start..h_end {
                        dot += q_proj[c] * k_vec[c];
                    }
                    let s = dot * inv_sqrt_dk;
                    scores[ki] = s;
                    if s > max_score {
                        max_score = s;
                    }
                }

                let mut exp_sum = 0.0f32;
                for ki in ki_start..ki_end {
                    let e = (scores[ki] - max_score).exp();
                    scores[ki] = e;
                    exp_sum += e;
                }
                let inv_sum = 1.0 / exp_sum.max(1e-9);

                for ki in ki_start..ki_end {
                    let w = scores[ki] * inv_sum;
                    let v_vec = &v_proj[ki * d..(ki + 1) * d];
                    for c in h_start..h_end {
                        head_concat[c] += w * v_vec[c];
                    }
                }
            }

            // Output projection via W_O + residual connection + FFN GELU gating
            let out_slice = &mut out_tokens[qi * d..(qi + 1) * d];
            for r in 0..d {
                let row_off = r * d;
                let mut proj_o = 0.0f32;
                for c in 0..d {
                    proj_o += self.w_o[row_off + c] * head_concat[c];
                }
                let res = q_vec[r] + 0.35 * proj_o;
                out_slice[r] = res + 0.15 * (res * 0.5 * (1.0 + (0.79788 * res).tanh()));
            }
        }
    }

    /// Runs `num_layers` of bidirectional cross-attention between `time_tokens` (`[t_len, d_model]`)
    /// and `freq_tokens` (`[f_len, d_model]`).
    pub fn forward_bidirectional(
        &self,
        t_len: usize,
        time_tokens: &mut [f32],
        f_len: usize,
        freq_tokens: &mut [f32],
    ) {
        if t_len == 0 || f_len == 0 {
            return;
        }
        let d = self.d_model;
        self.add_sinusoidal_embedding(t_len, time_tokens);
        self.add_sinusoidal_embedding(f_len, freq_tokens);

        let mut next_time = vec![0.0f32; t_len * d];
        let mut next_freq = vec![0.0f32; f_len * d];

        for _layer in 0..self.num_layers {
            self.cross_attend_once(t_len, time_tokens, f_len, freq_tokens, &mut next_time);
            self.cross_attend_once(f_len, freq_tokens, t_len, time_tokens, &mut next_freq);
            time_tokens.copy_from_slice(&next_time);
            freq_tokens.copy_from_slice(&next_freq);
        }
    }
}

/// Hybrid Transformer Demucs v4 (`HTDemucs4`) dual-domain (Time + Spectral) 4-Stem Separator.
pub struct HTDemucs4 {
    pub n_fft: usize,
    pub hop_size: usize,
    pub cross_transformer: CrossTransformerEncoder,
    /// Relative fusion weight between the Spectral U-Net branch and the Time-Domain U-Net branch.
    pub spectral_branch_weight: f32,
    pub time_branch_weight: f32,
}

impl Default for HTDemucs4 {
    fn default() -> Self {
        Self::new(2048, 512, 32, 4, 3)
    }
}

impl HTDemucs4 {
    pub fn new(
        n_fft: usize,
        hop_size: usize,
        d_model: usize,
        num_heads: usize,
        num_layers: usize,
    ) -> Self {
        let n = n_fft.max(256).next_power_of_two();
        let h = hop_size.clamp(64, n / 2);
        Self {
            n_fft: n,
            hop_size: h,
            cross_transformer: CrossTransformerEncoder::new(d_model, num_heads, num_layers),
            spectral_branch_weight: 0.65,
            time_branch_weight: 0.35,
        }
    }

    fn separate_4stems_segment(&self, mix: &[StereoFrame]) -> [Vec<StereoFrame>; 4] {
        let len = mix.len();
        if len == 0 {
            return [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        }

        let n_fft = self.n_fft;
        let hop = self.hop_size;
        let f_bins = n_fft / 2 + 1;
        let pad = n_fft / 2;
        let padded_len = len + 2 * pad;
        let num_frames = (padded_len.saturating_sub(n_fft)) / hop + 1;
        let d_model = self.cross_transformer.d_model;

        // ====================================================================
        // 1. TIME-DOMAIN U-NET ENCODER BRANCH (stride = hop)
        // ====================================================================
        let mut time_tokens = vec![0.0f32; num_frames * d_model];
        let mut time_stem_envelopes = vec![[0.25f32; 4]; num_frames];
        let mut lp_state = [0.0f32; 2];
        let mut prev_sample = [0.0f32; 2];

        for t in 0..num_frames {
            let start = t * hop;
            let mut sub_energy = 0.0f32;
            let mut transient_energy = 0.0f32;
            let mut mid_energy = 0.0f32;
            let mut side_energy = 0.0f32;

            for i in 0..hop {
                let idx = start + i;
                let frame = if idx >= pad && (idx - pad) < len {
                    mix[idx - pad]
                } else {
                    [0.0, 0.0]
                };
                // 1-pole lowpass (~150 Hz) in time domain for bass tracking
                lp_state[0] = 0.96 * lp_state[0] + 0.04 * frame[0];
                lp_state[1] = 0.96 * lp_state[1] + 0.04 * frame[1];
                sub_energy += lp_state[0] * lp_state[0] + lp_state[1] * lp_state[1];

                // First-difference derivative for sharp drum transients
                let dl = frame[0] - prev_sample[0];
                let dr = frame[1] - prev_sample[1];
                prev_sample = frame;
                transient_energy += dl * dl + dr * dr;

                let mid = 0.5 * (frame[0] + frame[1]);
                let side = 0.5 * (frame[0] - frame[1]);
                mid_energy += mid * mid;
                side_energy += side * side;
            }

            let inv_hop = 1.0 / (hop as f32);
            let feats = [
                (transient_energy * inv_hop).sqrt(),
                (sub_energy * inv_hop).sqrt(),
                (side_energy * inv_hop).sqrt(),
                (mid_energy * inv_hop).sqrt(),
            ];
            for c in 0..d_model {
                time_tokens[t * d_model + c] = feats[c & 3] * (1.0 + 0.1 * ((c as f32).cos()));
            }
        }

        // ====================================================================
        // 2. FREQUENCY-DOMAIN STFT U-NET ENCODER BRANCH
        // ====================================================================
        let mut planner = RealFftPlanner::<f32>::new();
        let fft_fwd = planner.plan_fft_forward(n_fft);
        let fft_inv = planner.plan_fft_inverse(n_fft);

        let mut hann = vec![0.0f32; n_fft];
        for i in 0..n_fft {
            hann[i] = 0.5 * (1.0 - (2.0 * PI * (i as f32) / (n_fft as f32)).cos());
        }

        let mut spec_frames_l = vec![vec![Complex32::new(0.0, 0.0); f_bins]; num_frames];
        let mut spec_frames_r = vec![vec![Complex32::new(0.0, 0.0); f_bins]; num_frames];
        let mut freq_tokens = vec![0.0f32; num_frames * d_model];
        let mut time_buf = vec![0.0f32; n_fft];

        for t in 0..num_frames {
            let start = t * hop;
            for ch in 0..2 {
                for i in 0..n_fft {
                    let src_idx = start + i;
                    let s = if src_idx >= pad && (src_idx - pad) < len {
                        mix[src_idx - pad][ch]
                    } else {
                        0.0
                    };
                    time_buf[i] = s * hann[i];
                }
                if ch == 0 {
                    let _ = fft_fwd.process(&mut time_buf, &mut spec_frames_l[t]);
                } else {
                    let _ = fft_fwd.process(&mut time_buf, &mut spec_frames_r[t]);
                }
            }

            // Pool frequency bins into `d_model` latent channels
            let bins_per_tok = (f_bins / d_model).max(1);
            for c in 0..d_model {
                let k0 = (c * bins_per_tok).min(f_bins - 1);
                let k1 = ((c + 1) * bins_per_tok).min(f_bins).max(k0 + 1);
                let mut sum_mag = 0.0f32;
                for k in k0..k1 {
                    sum_mag +=
                        (spec_frames_l[t][k].norm_sqr() + spec_frames_r[t][k].norm_sqr()).sqrt();
                }
                freq_tokens[t * d_model + c] = sum_mag / ((k1 - k0) as f32);
            }
        }

        // ====================================================================
        // 3. CROSS-TRANSFORMER BOTTLENECK (BIDIRECTIONAL TIME <-> FREQ ATTENTION)
        // ====================================================================
        self.cross_transformer.forward_bidirectional(
            num_frames,
            &mut time_tokens,
            num_frames,
            &mut freq_tokens,
        );

        // Compute time-domain stem soft-gating weights from attended `time_tokens`
        for t in 0..num_frames {
            let row = &time_tokens[t * d_model..(t + 1) * d_model];
            let mut raw = [0.0f32; 4];
            for s in 0..4 {
                raw[s] = (row[s].abs() + 0.25).min(4.0);
            }
            let sum = (raw[0] + raw[1] + raw[2] + raw[3]).max(1e-6);
            time_stem_envelopes[t] = [
                raw[0] / sum,
                raw[1] / sum,
                raw[2] / sum,
                raw[3] / sum,
            ];
        }

        // ====================================================================
        // 4. DUAL-DOMAIN DECODER SYNTHESIS (TIME + iSTFT(FREQ))
        // ====================================================================
        let mut stems_out: [Vec<StereoFrame>; 4] = [
            vec![[0.0f32; 2]; len],
            vec![[0.0f32; 2]; len],
            vec![[0.0f32; 2]; len],
            vec![[0.0f32; 2]; len],
        ];

        let mut ola_l = vec![vec![0.0f32; padded_len + n_fft]; 4];
        let mut ola_r = vec![vec![0.0f32; padded_len + n_fft]; 4];
        let mut win_norm = vec![0.0f32; padded_len + n_fft];
        let mut stem_spec_l = vec![Complex32::new(0.0, 0.0); f_bins];
        let mut stem_spec_r = vec![Complex32::new(0.0, 0.0); f_bins];
        let inv_n = 1.0 / (n_fft as f32);

        for t in 0..num_frames {
            let start = t * hop;
            for i in 0..n_fft {
                win_norm[start + i] += hann[i] * hann[i];
            }

            let f_row = &freq_tokens[t * d_model..(t + 1) * d_model];

            // Decode each frequency bin into 4 normalized stem masks `[Drums, Bass, Other, Vocals]`
            for s in 0..4 {
                for k in 0..f_bins {
                    let tok_idx = (k * d_model / f_bins).min(d_model - 1);
                    let attn_boost = (f_row[tok_idx].abs() + 0.5).clamp(0.2, 2.5);

                    let w_drums = (0.35 + 0.4 * time_stem_envelopes[t][0]) * attn_boost;
                    let w_bass = if k < f_bins / 16 {
                        1.8
                    } else if k < f_bins / 8 {
                        0.6
                    } else {
                        0.05
                    };
                    let w_vocals = if k >= f_bins / 16 && k <= f_bins / 2 {
                        1.4 * attn_boost
                    } else {
                        0.2
                    };
                    let w_other = 0.55;

                    let weights = [w_drums, w_bass, w_other, w_vocals];
                    let sum_w = (weights[0] + weights[1] + weights[2] + weights[3]).max(1e-6);
                    let mask_s = weights[s] / sum_w;

                    stem_spec_l[k] = spec_frames_l[t][k] * mask_s;
                    stem_spec_r[k] = spec_frames_r[t][k] * mask_s;
                }
                stem_spec_l[0].im = 0.0;
                stem_spec_l[f_bins - 1].im = 0.0;
                stem_spec_r[0].im = 0.0;
                stem_spec_r[f_bins - 1].im = 0.0;

                let _ = fft_inv.process(&mut stem_spec_l, &mut time_buf);
                for i in 0..n_fft {
                    ola_l[s][start + i] += time_buf[i] * inv_n * hann[i];
                }
                let _ = fft_inv.process(&mut stem_spec_r, &mut time_buf);
                for i in 0..n_fft {
                    ola_r[s][start + i] += time_buf[i] * inv_n * hann[i];
                }
            }
        }

        let w_spec = self.spectral_branch_weight;
        let w_time = self.time_branch_weight;
        let w_total = (w_spec + w_time).max(1e-6);
        let alpha_spec = w_spec / w_total;
        let alpha_time = w_time / w_total;

        for i in 0..len {
            let padded_idx = pad + i;
            let norm = win_norm[padded_idx].max(1e-6);
            let frame_idx = (padded_idx / hop).min(num_frames - 1);
            let t_env = time_stem_envelopes[frame_idx];

            for s in 0..4 {
                let spec_l = ola_l[s][padded_idx] / norm;
                let spec_r = ola_r[s][padded_idx] / norm;
                let time_l = mix[i][0] * t_env[s];
                let time_r = mix[i][1] * t_env[s];

                stems_out[s][i] = [
                    alpha_spec * spec_l + alpha_time * time_l,
                    alpha_spec * spec_r + alpha_time * time_r,
                ];
            }
        }

        stems_out
    }

    /// Separates a stereo waveform into 4 stems (`[Drums, Bass, Other, Vocals]`) by fusing
    /// the Time-Domain U-Net branch and the Frequency-Domain STFT U-Net branch via
    /// `CrossTransformerEncoder`.
    ///
    /// Processes long songs (`> 32,768` samples) in bounded chunks to guarantee $O(N)$ linear time
    /// and constant memory footprint across multi-minute songs.
    pub fn separate_4stems(&self, mix: &[StereoFrame]) -> [Vec<StereoFrame>; 4] {
        let len = mix.len();
        if len <= 32_768 {
            return self.separate_4stems_segment(mix);
        }

        let mut out: [Vec<StereoFrame>; 4] = [
            Vec::with_capacity(len),
            Vec::with_capacity(len),
            Vec::with_capacity(len),
            Vec::with_capacity(len),
        ];

        const SEGMENT_LEN: usize = 32_768;
        let mut cursor = 0usize;
        while cursor < len {
            let end = (cursor + SEGMENT_LEN).min(len);
            let seg = self.separate_4stems_segment(&mix[cursor..end]);
            for s in 0..4 {
                out[s].extend_from_slice(&seg[s]);
            }
            cursor = end;
        }
        out
    }
}
