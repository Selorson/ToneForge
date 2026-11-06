//! `spectral_dynamics`: Per-Band Spectral Dynamics STFT Processor (`0x18019F1D0`, `0x1801A28F0`).
//!
//! Specifications from `FabFilter Pro-Q 4.clap`:
//! - Window size: `N_STFT = 2048 * scale`, Hop size: `H = 512 * scale` (75% overlap)
//! - Additional latency when active: `Latency_spectral = 1024 * scale` samples (`0x18018D8B9`)
//! - Processes bins in 4-bin SIMD `wide::f32x4` chunks (`shr r8, 2` at `0x1801A2990`)
//! - Frequency-axis kernel smoothing controlled by `Spectral Density` (`0% .. 100%`).

use crate::params_601::{EqBandParams, PROQ4_NUM_BANDS};
use num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::f32::consts::TAU;
use std::sync::Arc;
use wide::f32x4;

pub const SPECTRAL_BASE_FFT_SIZE: usize = 2048;
pub const SPECTRAL_BASE_HOP_SIZE: usize = 256;
pub const SPECTRAL_BASE_LATENCY: usize = 1024;
pub const SPECTRAL_BASE_BINS: usize = (SPECTRAL_BASE_FFT_SIZE / 2) + 1; // 1025 bins
pub const SPECTRAL_SIMD_PADDED_BINS: usize = 1028; // Multiple of 4 for `f32x4` SIMD loop
const MAX_SCALE_EXTRA_DELAY: usize = SPECTRAL_BASE_LATENCY * 4; // Power-of-two ring for up to 4x scale

/// Per-Band Spectral Dynamics STFT Core (`0x18019F1D0`, `0x1801A28F0`) with Zero Heap Allocation.
pub struct SpectralDynamicsCore {
    r2c: Arc<dyn RealToComplex<f32>>,
    c2r: Arc<dyn ComplexToReal<f32>>,
    r2c_scratch: Vec<Complex32>,
    c2r_scratch: Vec<Complex32>,
    window: [f32; SPECTRAL_BASE_LATENCY],
    window_power_norm: f32,
    in_ring: [[f32; SPECTRAL_BASE_LATENCY]; 2],
    out_ola: [[f32; SPECTRAL_BASE_FFT_SIZE]; 2],
    scale_delay_ring: [[f32; MAX_SCALE_EXTRA_DELAY]; 2],
    scale_delay_cursor: usize,
    ring_cursor: usize,
    ola_cursor: usize,
    hop_counter: usize,
    fft_time_buf: [f32; SPECTRAL_BASE_FFT_SIZE],
    fft_spec_buf: [Complex32; SPECTRAL_BASE_BINS],
    bin_mag_sq: [f32; SPECTRAL_SIMD_PADDED_BINS],
    bin_baseline_sq: [f32; SPECTRAL_SIMD_PADDED_BINS],
    bin_env_gain: [[f32; SPECTRAL_SIMD_PADDED_BINS]; 2],
    band_weight_mask: [f32; SPECTRAL_SIMD_PADDED_BINS],
    band_max_dyn_db: [f32; SPECTRAL_SIMD_PADDED_BINS],
    band_thresh_lin_sq: [f32; SPECTRAL_SIMD_PADDED_BINS],
    smoothing_radius_bins: usize,
    sample_rate_scale: usize,
}

impl Default for SpectralDynamicsCore {
    fn default() -> Self {
        Self::new()
    }
}

impl SpectralDynamicsCore {
    pub fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(SPECTRAL_BASE_FFT_SIZE);
        let c2r = planner.plan_fft_inverse(SPECTRAL_BASE_FFT_SIZE);
        let r2c_scratch = vec![Complex32::new(0.0, 0.0); r2c.get_scratch_len()];
        let c2r_scratch = vec![Complex32::new(0.0, 0.0); c2r.get_scratch_len()];

        // COLA-compliant root-Hann window of length `W = 1024` (`SPECTRAL_BASE_LATENCY`)
        // with 4x overlap (`H = 256`): sum_{m=0..3} Hann(i + m*W/4) == 2.0, so norm = 1/sqrt(2)
        // yields exact unity passband reconstruction (`sum w^2 == 1.000000`) and exact 1024-sample latency.
        let mut window = [0.0f32; SPECTRAL_BASE_LATENCY];
        let norm = std::f32::consts::FRAC_1_SQRT_2;
        let mut win_sum = 0.0f32;
        for (i, w) in window.iter_mut().enumerate() {
            let h = 0.5 * (1.0 - (TAU * (i as f32) / (SPECTRAL_BASE_LATENCY as f32)).cos());
            let val = h.sqrt() * norm;
            *w = val;
            win_sum += val;
        }
        let window_power_norm = 4.0 / (win_sum * win_sum).max(1.0);

        Self {
            r2c,
            c2r,
            r2c_scratch,
            c2r_scratch,
            window,
            window_power_norm,
            in_ring: [[0.0; SPECTRAL_BASE_LATENCY]; 2],
            out_ola: [[0.0; SPECTRAL_BASE_FFT_SIZE]; 2],
            scale_delay_ring: [[0.0; MAX_SCALE_EXTRA_DELAY]; 2],
            scale_delay_cursor: 0,
            ring_cursor: 0,
            ola_cursor: 0,
            hop_counter: 0,
            fft_time_buf: [0.0; SPECTRAL_BASE_FFT_SIZE],
            fft_spec_buf: [Complex32::new(0.0, 0.0); SPECTRAL_BASE_BINS],
            bin_mag_sq: [0.0; SPECTRAL_SIMD_PADDED_BINS],
            bin_baseline_sq: [0.0; SPECTRAL_SIMD_PADDED_BINS],
            bin_env_gain: [[1.0; SPECTRAL_SIMD_PADDED_BINS]; 2],
            band_weight_mask: [0.0; SPECTRAL_SIMD_PADDED_BINS],
            band_max_dyn_db: [0.0; SPECTRAL_SIMD_PADDED_BINS],
            band_thresh_lin_sq: [1e-4; SPECTRAL_SIMD_PADDED_BINS],
            smoothing_radius_bins: 4,
            sample_rate_scale: 1,
        }
    }

    #[inline]
    pub fn latency_samples(&self) -> usize {
        SPECTRAL_BASE_LATENCY * self.sample_rate_scale
    }

    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        let scale = ((sample_rate / 48_000.0).round().max(1.0) as usize).next_power_of_two();
        let clamped = scale.clamp(1, 4);
        if clamped != self.sample_rate_scale {
            self.sample_rate_scale = clamped;
            self.scale_delay_ring = [[0.0; MAX_SCALE_EXTRA_DELAY]; 2];
            self.scale_delay_cursor = 0;
        }
    }

    pub fn reset(&mut self) {
        self.in_ring = [[0.0; SPECTRAL_BASE_LATENCY]; 2];
        self.out_ola = [[0.0; SPECTRAL_BASE_FFT_SIZE]; 2];
        self.scale_delay_ring = [[0.0; MAX_SCALE_EXTRA_DELAY]; 2];
        self.scale_delay_cursor = 0;
        self.ring_cursor = 0;
        self.ola_cursor = 0;
        self.hop_counter = 0;
        self.bin_env_gain = [[1.0; SPECTRAL_SIMD_PADDED_BINS]; 2];
    }

    /// Recomputes the spectral frequency mask `W_band[k]` (`0x1801A2C4B..0x1801A2D08`)
    /// from all active bands that have `spectral_enabled == true`.
    pub fn update_spectral_masks(
        &mut self,
        bands: &[EqBandParams; PROQ4_NUM_BANDS],
        global_gain_scale: f32,
        sample_rate: f64,
    ) {
        self.band_weight_mask.fill(0.0);
        self.band_max_dyn_db.fill(0.0);
        self.band_thresh_lin_sq.fill(1.0);

        let fs = sample_rate.max(8000.0) as f32;
        let bin_hz = fs / (SPECTRAL_BASE_FFT_SIZE as f32);
        let mut avg_density = 50.0f32;
        let mut active_count = 0usize;

        for b in bands {
            if !b.used
                || !b.enabled
                || !b.dyn_enabled
                || !b.spectral_enabled
                || !b.shape.supports_spectral_dynamics()
            {
                continue;
            }
            let dyn_db = b.effective_dyn_range_db(global_gain_scale);
            if dyn_db.abs() < 1e-3 {
                continue;
            }
            active_count += 1;
            avg_density += b.spectral_density;
            let f0 = b.freq_hz.max(20.0);
            let q = b.q_ui.max(0.1);
            let bw_oct = (1.0 / q).clamp(0.05, 4.0);
            let thresh_lin = 10.0f32.powf(b.dyn_threshold_db / 20.0);
            let thresh_sq = (thresh_lin * thresh_lin).max(1e-10);

            for k in 1..SPECTRAL_BASE_BINS {
                let fk = (k as f32) * bin_hz;
                let w_k = match b.shape {
                    crate::params_601::FilterShape::Bell => {
                        let oct_dist = (fk / f0).log2() / bw_oct;
                        (-0.5 * oct_dist * oct_dist).exp()
                    }
                    crate::params_601::FilterShape::LowShelf => {
                        1.0 / (1.0 + (fk / f0).powf(2.0 * q.min(4.0)))
                    }
                    crate::params_601::FilterShape::HighShelf
                    | crate::params_601::FilterShape::TiltShelf => {
                        let r = (fk / f0).powf(2.0 * q.min(4.0));
                        r / (1.0 + r)
                    }
                    _ => 0.0,
                };
                if w_k > self.band_weight_mask[k] {
                    self.band_weight_mask[k] = w_k;
                    self.band_max_dyn_db[k] = dyn_db * w_k;
                    self.band_thresh_lin_sq[k] = thresh_sq;
                }
            }
        }

        if active_count > 0 {
            let density_norm = ((avg_density / (active_count as f32 + 1.0)) * 0.01).clamp(0.0, 1.0);
            // Low Spectral Density -> wide smoothing kernel; High Spectral Density -> sharp per-bin action
            self.smoothing_radius_bins = ((1.0 - density_norm) * 14.0).round() as usize + 1;
        }
    }

    #[inline]
    pub fn process_stereo_sample(&mut self, in_l: f32, in_r: f32) -> [f32; 2] {
        let rc = self.ring_cursor;
        self.in_ring[0][rc] = in_l;
        self.in_ring[1][rc] = in_r;
        self.ring_cursor = (rc + 1) & (SPECTRAL_BASE_LATENCY - 1);

        let oc = self.ola_cursor;
        let out_base = [self.out_ola[0][oc], self.out_ola[1][oc]];
        self.out_ola[0][oc] = 0.0;
        self.out_ola[1][oc] = 0.0;
        self.ola_cursor = (oc + 1) & (SPECTRAL_BASE_FFT_SIZE - 1);

        self.hop_counter += 1;
        if self.hop_counter >= SPECTRAL_BASE_HOP_SIZE {
            self.hop_counter = 0;
            self.process_stft_hop();
        }

        if self.sample_rate_scale <= 1 {
            out_base
        } else {
            let extra_delay = (self.sample_rate_scale - 1) * SPECTRAL_BASE_LATENCY;
            let mask = MAX_SCALE_EXTRA_DELAY - 1;
            let wc = self.scale_delay_cursor;
            self.scale_delay_ring[0][wc] = out_base[0];
            self.scale_delay_ring[1][wc] = out_base[1];
            let rd = (wc + MAX_SCALE_EXTRA_DELAY - extra_delay) & mask;
            self.scale_delay_cursor = (wc + 1) & mask;
            [self.scale_delay_ring[0][rd], self.scale_delay_ring[1][rd]]
        }
    }

    fn process_stft_hop(&mut self) {
        let n = SPECTRAL_BASE_FFT_SIZE;
        let w_len = SPECTRAL_BASE_LATENCY;
        let inv_n = 1.0 / (n as f32);
        let p_norm = self.window_power_norm;
        let rad = self.smoothing_radius_bins.clamp(1, 16);

        for ch in 0..2 {
            // 1. Windowed 1024-sample analysis frame from circular `in_ring`, zero-padded to 2048
            for i in 0..w_len {
                let idx = (self.ring_cursor + i) & (w_len - 1);
                self.fft_time_buf[i] = self.in_ring[ch][idx] * self.window[i];
            }
            self.fft_time_buf[w_len..n].fill(0.0);

            let _ = self.r2c.process_with_scratch(
                &mut self.fft_time_buf,
                &mut self.fft_spec_buf,
                &mut self.r2c_scratch,
            );

            // 2. Compute per-bin calibrated power spectrum
            for k in 0..SPECTRAL_BASE_BINS {
                self.bin_mag_sq[k] = self.fft_spec_buf[k].norm_sqr() * p_norm;
            }
            for k in SPECTRAL_BASE_BINS..SPECTRAL_SIMD_PADDED_BINS {
                self.bin_mag_sq[k] = 0.0;
            }

            // 3. Smooth power spectrum across `rad` bins controlled by `Spectral Density`
            let mut running = 0.0f32;
            for k in 0..SPECTRAL_BASE_BINS {
                let lo = k.saturating_sub(rad);
                let hi = (k + rad).min(SPECTRAL_BASE_BINS - 1);
                if k == 0 {
                    running = self.bin_mag_sq[lo..=hi].iter().sum();
                } else {
                    if k + rad < SPECTRAL_BASE_BINS {
                        running += self.bin_mag_sq[k + rad];
                    }
                    if k > rad {
                        running -= self.bin_mag_sq[k - rad - 1];
                    }
                }
                let count = (hi - lo + 1) as f32;
                self.bin_baseline_sq[k] = (running / count).max(1e-12);
            }

            // 4. 4-Bin SIMD (`f32x4`) Spectral Dynamics Evaluation (`0x1801A2990..0x1801A2AE6`)
            let eps_v = f32x4::splat(1e-12);
            let one_v = f32x4::splat(1.0);
            let alpha_smooth = f32x4::splat(0.65);
            let one_minus_alpha = f32x4::splat(0.35);

            for chunk_idx in (0..SPECTRAL_SIMD_PADDED_BINS).step_by(4) {
                let p_bin = f32x4::from([
                    self.bin_baseline_sq[chunk_idx],
                    self.bin_baseline_sq[chunk_idx + 1],
                    self.bin_baseline_sq[chunk_idx + 2],
                    self.bin_baseline_sq[chunk_idx + 3],
                ]) + eps_v;

                let p_thresh = f32x4::from([
                    self.band_thresh_lin_sq[chunk_idx],
                    self.band_thresh_lin_sq[chunk_idx + 1],
                    self.band_thresh_lin_sq[chunk_idx + 2],
                    self.band_thresh_lin_sq[chunk_idx + 3],
                ]) + eps_v;

                let ratio = (p_bin / p_thresh).max(one_v);
                let ratio_arr: [f32; 4] = ratio.into();
                let mut target_gains = [1.0f32; 4];

                for lane in 0..4 {
                    let k = chunk_idx + lane;
                    let w = self.band_weight_mask[k];
                    if w > 1e-4 && ratio_arr[lane] > 1.0 {
                        let over_db = 10.0 * ratio_arr[lane].log10();
                        let max_dyn_db = self.band_max_dyn_db[k];
                        let drive = over_db / (over_db + 12.0);
                        let g_db = (max_dyn_db * drive).clamp(-30.0, 30.0);
                        target_gains[lane] = 10.0f32.powf(g_db / 20.0);
                    }
                }

                let prev_g = f32x4::from([
                    self.bin_env_gain[ch][chunk_idx],
                    self.bin_env_gain[ch][chunk_idx + 1],
                    self.bin_env_gain[ch][chunk_idx + 2],
                    self.bin_env_gain[ch][chunk_idx + 3],
                ]);
                let tgt_g = f32x4::from(target_gains);
                let smoothed_g = prev_g * alpha_smooth + tgt_g * one_minus_alpha;
                let smoothed_arr: [f32; 4] = smoothed_g.into();

                for lane in 0..4 {
                    let k = chunk_idx + lane;
                    self.bin_env_gain[ch][k] = smoothed_arr[lane];
                    if k < SPECTRAL_BASE_BINS {
                        self.fft_spec_buf[k] *= smoothed_arr[lane] * inv_n;
                    }
                }
            }

            // 5. Enforce real DC/Nyquist and IFFT + Overlap-Add over the 1024-sample synthesis window
            self.fft_spec_buf[0].im = 0.0;
            self.fft_spec_buf[SPECTRAL_BASE_BINS - 1].im = 0.0;
            let _ = self.c2r.process_with_scratch(
                &mut self.fft_spec_buf,
                &mut self.fft_time_buf,
                &mut self.c2r_scratch,
            );

            for i in 0..w_len {
                let dst = (self.ola_cursor + i) & (n - 1);
                self.out_ola[ch][dst] += self.fft_time_buf[i] * self.window[i];
            }
        }
    }
}
