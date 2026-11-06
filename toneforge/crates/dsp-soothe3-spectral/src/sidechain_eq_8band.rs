//! `sidechain_eq_8band`: 8-Band Hybrid Series / Parallel Sidechain Weighting EQ (`dsp::WeightFilter` / `dsp::SidechainEQ`).
//!
//! Key Topology (`02_proq4_and_soothe3_dsp.md` Section 2.1.B & Section 2.3.B Step 1):
//! - Bands with `shape != Bandpass (3)` (`LowCut`, `LowShelf`, `Bell`, `BandReject`, `Tilt`, `HighShelf`, `HighCut`)
//!   are cascaded in **SERIES**: `H_series(f_k) = \prod_{b \notin BP} H_b(f_k)`.
//! - Enabled bands with `shape == Bandpass (3)` are summed in **PARALLEL** and multiplied by `H_series(f_k)`:
//!   `W_EQ(f_k) = H_series(f_k) * \sum_{b \in BP} H_b(f_k)`.
//! - Supports per-band `focus` (`0.0..1.0` L/R or Mid/Side sensitivity balance) and global `stereo_focus` (`0.0..1.0`).

use crate::params_143::{SidechainEqBandParams, SidechainFilterShape, SOOTHE3_NUM_SC_BANDS};

pub const SC_EQ_NUM_BINS: usize = 2049;

/// Evaluates the 8-Band Hybrid Series / Parallel Sidechain EQ across all `2,049` STFT bins
/// for both channels (`0 = Left / Mid`, `1 = Right / Side`).
pub struct SidechainEq8Band {
    /// Squared magnitude weighting `|W_EQ[ch, k]|^2` for `ch in 0..2` and `k in 0..2049`
    pub weight_sq_per_ch: [[f32; SC_EQ_NUM_BINS]; 2],
    /// Combined linear magnitude response `|W_EQ[k]|` for UI curve / inspection
    pub combined_mag_response: [f32; SC_EQ_NUM_BINS],
}

impl Default for SidechainEq8Band {
    fn default() -> Self {
        Self::new()
    }
}

impl SidechainEq8Band {
    pub fn new() -> Self {
        Self {
            weight_sq_per_ch: [[1.0; SC_EQ_NUM_BINS]; 2],
            combined_mag_response: [1.0; SC_EQ_NUM_BINS],
        }
    }

    /// Evaluates the magnitude transfer function `|H_b(f)|` (linear amplitude) of a single Sidechain EQ band at `freq_hz`.
    #[inline]
    pub fn evaluate_band_magnitude(
        band: &SidechainEqBandParams,
        freq_hz: f32,
        channel_idx: usize,
    ) -> f32 {
        // Check per-band channel routing (`0 = Stereo`, `1 = Ch 0 only (L/Mid)`, `2 = Ch 1 only (R/Side)`)
        if (band.channels == 1 && channel_idx == 1) || (band.channels == 2 && channel_idx == 0) {
            return if band.shape == SidechainFilterShape::Bandpass {
                0.0
            } else {
                1.0
            };
        }

        let f = freq_hz.max(10.0);
        let f0 = band.freq_hz.clamp(20.0, 20_000.0);
        let q = band.q.clamp(0.1, 16.0);
        let ratio = f / f0;
        let log_oct = ratio.log2();

        // Focus balance: 0.5 -> equal (1.0, 1.0); < 0.5 favors ch 0 (Left/Mid); > 0.5 favors ch 1 (Right/Side)
        let focus = band.focus.clamp(0.0, 1.0);
        let ch_focus_scale = if channel_idx == 0 {
            ((1.0 - focus) * 2.0).clamp(0.0, 1.5)
        } else {
            (focus * 2.0).clamp(0.0, 1.5)
        };

        let gain_db = band.depth_to_db() * ch_focus_scale;
        // Continuous slope 0..10 maps to order 1..8 (6 dB/oct .. 48 dB/oct)
        let order = 1.0 + 0.7 * band.slope.clamp(0.0, 10.0);

        match band.shape {
            SidechainFilterShape::LowCut => {
                let steepness = (2.0 * order).max(1.0);
                let hp = 1.0 / (1.0 + (f0 / f).powf(steepness)).sqrt();
                if q > 0.707 {
                    let peak_db = 20.0 * (q / 0.707).log10() * (-4.0 * log_oct * log_oct).exp();
                    hp * 10.0f32.powf(peak_db / 20.0)
                } else {
                    hp
                }
            }
            SidechainFilterShape::HighCut => {
                let steepness = (2.0 * order).max(1.0);
                let lp = 1.0 / (1.0 + (f / f0).powf(steepness)).sqrt();
                if q > 0.707 {
                    let peak_db = 20.0 * (q / 0.707).log10() * (-4.0 * log_oct * log_oct).exp();
                    lp * 10.0f32.powf(peak_db / 20.0)
                } else {
                    lp
                }
            }
            SidechainFilterShape::LowShelf => {
                let s_low = 1.0 / (1.0 + (f / f0).powf(2.0 * (0.5 + 0.35 * q)));
                10.0f32.powf((gain_db * s_low) / 20.0)
            }
            SidechainFilterShape::HighShelf => {
                let r_pow = (f / f0).powf(2.0 * (0.5 + 0.35 * q));
                let s_high = r_pow / (1.0 + r_pow);
                10.0f32.powf((gain_db * s_high) / 20.0)
            }
            SidechainFilterShape::Bell => {
                let bw_oct = (1.0 / q).max(0.04);
                let bell_shape = 1.0 / (1.0 + (2.0 * log_oct / bw_oct).powi(2));
                10.0f32.powf((gain_db * bell_shape) / 20.0)
            }
            SidechainFilterShape::Bandpass => {
                let bw_oct = (1.0 / q).max(0.05);
                let bp = 1.0 / (1.0 + (2.0 * log_oct / bw_oct).powi(2)).sqrt();
                let depth_mult = 10.0f32.powf(gain_db / 20.0);
                bp * depth_mult
            }
            SidechainFilterShape::BandReject => {
                let bw_oct = (1.0 / q).max(0.05);
                let x = 2.0 * log_oct / bw_oct;
                (x * x / (1.0 + x * x)).sqrt()
            }
            SidechainFilterShape::Tilt => {
                let r_pow = (f / f0).powf(1.2);
                let tilt_norm = (r_pow - 1.0) / (r_pow + 1.0);
                10.0f32.powf((gain_db * tilt_norm) / 20.0)
            }
        }
    }

    /// Recomputes `weight_sq_per_ch` and `combined_mag_response` across all `2,049` bins
    /// for a 4096-point STFT at `base_sample_rate`.
    pub fn update_weights(
        &mut self,
        bands: &[SidechainEqBandParams; SOOTHE3_NUM_SC_BANDS],
        stereo_focus: f32,
        fft_size: usize,
        base_sample_rate: f32,
    ) {
        let fs = base_sample_rate.max(8000.0);
        let hz_per_bin = fs / (fft_size as f32);

        // Global stereo focus (0.0..1.0, 0.5 = equal sensitivity on both channels):
        let bal = (stereo_focus.clamp(0.0, 1.0) - 0.5) * 2.0; // [-1.0, +1.0]
        let global_ch_gain = [
            if bal > 0.0 { 1.0 - bal } else { 1.0 },
            if bal < 0.0 { 1.0 + bal } else { 1.0 },
        ];

        for ch in 0..2 {
            let ch_scale = global_ch_gain[ch];
            for k in 0..SC_EQ_NUM_BINS {
                let fk = (k as f32) * hz_per_bin;
                let mut h_series = 1.0f32;
                let mut h_parallel_bp_sum = 0.0f32;
                let mut has_parallel_bp = false;

                for band in bands {
                    if !band.enabled {
                        continue;
                    }
                    let mag = Self::evaluate_band_magnitude(band, fk, ch);
                    if band.shape == SidechainFilterShape::Bandpass {
                        has_parallel_bp = true;
                        h_parallel_bp_sum += mag;
                    } else {
                        h_series *= mag;
                    }
                }

                let total_mag = if has_parallel_bp {
                    h_series * h_parallel_bp_sum * ch_scale
                } else {
                    h_series * ch_scale
                };
                let clamped_mag = total_mag.clamp(0.0, 64.0);
                self.weight_sq_per_ch[ch][k] = clamped_mag * clamped_mag;
                if ch == 0 {
                    self.combined_mag_response[k] = clamped_mag;
                } else {
                    self.combined_mag_response[k] =
                        0.5 * (self.combined_mag_response[k] + clamped_mag);
                }
            }
        }
    }
}
