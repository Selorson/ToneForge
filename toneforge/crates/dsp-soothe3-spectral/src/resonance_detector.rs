//! `resonance_detector`: ERB / Bark Psychoacoustic Local Resonance Detector & Dynamic Spectral Suppressor
//! (`dsp::SpectralAnalyzer & ResonanceSuppressor` in `soothe3_symbols.map`).
//!
//! Implements Steps 3..6 of `02_proq4_and_soothe3_dsp.md` Section 2.3.B & Section 3.2:
//! - O(N) prefix-sum local spectral baseline `\overline{P}_{ref}[k]` on the ERB/Bark psychoacoustic scale
//! - Detail-dependent narrow resonance estimator `P_{det}[k]`
//! - Dual-mode (`Soft` level-independent prominence vs `Hard` level-dependent dynamic threshold)
//! - Spectral notch sharpness controlled by per-bin `Detail` (`D_eff[k]`) and `Max cut` ceiling
//! - Per-bin `Attack` / `Release` ballistics and `Stereo link`

use crate::params_143::{Soothe3GlobalParams, SootheProcessMode};
use crate::tilt_curves::{integrate_asym, update_per_bin_tilt_tables};

pub const DETECTOR_FFT_SIZE: usize = 4096;
pub const DETECTOR_NUM_BINS: usize = (DETECTOR_FFT_SIZE / 2) + 1; // 2049

/// Psychoacoustic ERB/Bark resonance detector and per-bin envelope follower (`Soothe3SpectralCore`).
pub struct ResonanceDetector {
    pub detail_per_bin: [f32; DETECTOR_NUM_BINS],
    pub attack_alpha_per_bin: [f32; DETECTOR_NUM_BINS],
    pub release_alpha_per_bin: [f32; DETECTOR_NUM_BINS],
    /// Precomputed ERB reference window half-width in bins for each bin `k`
    pub erb_ref_radius_bins: [usize; DETECTOR_NUM_BINS],
    /// Precomputed Bark coordinate for each bin `k`
    pub bark_per_bin: [f32; DETECTOR_NUM_BINS],
    /// Smoothed gain reduction envelope in dB (`<= 0.0`) per channel `[ch][k]`
    pub env_gr_db: [[f32; DETECTOR_NUM_BINS]; 2],
    /// Stereo-linked gain reduction in dB (`<= 0.0`) per channel `[ch][k]`
    pub linked_gr_db: [[f32; DETECTOR_NUM_BINS]; 2],
    /// Linear gain mask (`0.0 ..= 1.0`) per channel `[ch][k]` (after `Inverse` if enabled)
    pub linear_gain_mask: [[f32; DETECTOR_NUM_BINS]; 2],
    /// Prefix-sum scratch buffer for O(N) ERB baseline & detail smoothing (Zero Allocation)
    prefix_power: [f64; DETECTOR_NUM_BINS + 1],
    baseline_power_bins: [f32; DETECTOR_NUM_BINS],
    detail_power_bins: [f32; DETECTOR_NUM_BINS],
    raw_cut_db_scratch: [f32; DETECTOR_NUM_BINS],
    smooth_cut_db_scratch: [f32; DETECTOR_NUM_BINS],
}

impl Default for ResonanceDetector {
    fn default() -> Self {
        Self::new(48000.0)
    }
}

impl ResonanceDetector {
    pub fn new(base_sample_rate: f32) -> Self {
        let mut det = Self {
            detail_per_bin: [5.0; DETECTOR_NUM_BINS],
            attack_alpha_per_bin: [0.8; DETECTOR_NUM_BINS],
            release_alpha_per_bin: [0.95; DETECTOR_NUM_BINS],
            erb_ref_radius_bins: [8; DETECTOR_NUM_BINS],
            bark_per_bin: [0.0; DETECTOR_NUM_BINS],
            env_gr_db: [[0.0; DETECTOR_NUM_BINS]; 2],
            linked_gr_db: [[0.0; DETECTOR_NUM_BINS]; 2],
            linear_gain_mask: [[1.0; DETECTOR_NUM_BINS]; 2],
            prefix_power: [0.0; DETECTOR_NUM_BINS + 1],
            baseline_power_bins: [0.0; DETECTOR_NUM_BINS],
            detail_power_bins: [0.0; DETECTOR_NUM_BINS],
            raw_cut_db_scratch: [0.0; DETECTOR_NUM_BINS],
            smooth_cut_db_scratch: [0.0; DETECTOR_NUM_BINS],
        };
        det.update_psychoacoustic_tables(base_sample_rate);
        det
    }

    pub fn reset(&mut self) {
        self.env_gr_db = [[0.0; DETECTOR_NUM_BINS]; 2];
        self.linked_gr_db = [[0.0; DETECTOR_NUM_BINS]; 2];
        self.linear_gain_mask = [[1.0; DETECTOR_NUM_BINS]; 2];
    }

    /// Glasberg & Moore Equivalent Rectangular Bandwidth `ERB(f)` in Hz.
    #[inline]
    pub fn erb_bandwidth_hz(freq_hz: f32) -> f32 {
        24.7 * (4.37 * (freq_hz.max(20.0) * 0.001) + 1.0)
    }

    /// Zwicker critical-band Bark rate `Bark(f) in [0..24]`.
    #[inline]
    pub fn freq_to_bark(freq_hz: f32) -> f32 {
        let f = freq_hz.max(0.0);
        13.0 * (0.00076 * f).atan() + 3.5 * ((f / 7500.0).powi(2)).atan()
    }

    /// Precomputes ERB reference window radii and Bark coordinates for `base_sample_rate`.
    pub fn update_psychoacoustic_tables(&mut self, base_sample_rate: f32) {
        let fs = base_sample_rate.max(8000.0);
        let hz_per_bin = fs / (DETECTOR_FFT_SIZE as f32);
        for k in 0..DETECTOR_NUM_BINS {
            let fk = (k as f32) * hz_per_bin;
            self.bark_per_bin[k] = Self::freq_to_bark(fk);
            let erb_hz = Self::erb_bandwidth_hz(fk);
            // Reference window spans ~2.2 ERBs on each side to capture local spectral continuum
            let radius = ((2.2 * erb_hz) / hz_per_bin).round() as usize;
            self.erb_ref_radius_bins[k] = radius.clamp(6, 256);
        }
    }

    /// Updates per-bin `Detail`, `Attack`, and `Release` tables from `Soothe3GlobalParams`.
    pub fn update_tilt_tables(
        &mut self,
        global: &Soothe3GlobalParams,
        hop_size_1x: usize,
        base_sample_rate: f32,
    ) {
        update_per_bin_tilt_tables(
            DETECTOR_FFT_SIZE,
            hop_size_1x,
            base_sample_rate,
            global.detail,
            &global.detail_tilt,
            global.attack,
            &global.attack_tilt,
            global.release,
            &global.release_tilt,
            &mut self.detail_per_bin,
            &mut self.attack_alpha_per_bin,
            &mut self.release_alpha_per_bin,
        );
    }

    /// Computes per-bin dynamic resonance suppression for channel `ch` (`0` or `1`)
    /// given raw sidechain power spectrum `sc_power_bins` and Sidechain EQ weights `sc_eq_weight_sq`.
    pub fn process_channel_spectrum(
        &mut self,
        ch: usize,
        sc_power_bins: &[f32; DETECTOR_NUM_BINS],
        sc_eq_weight_sq: &[f32; DETECTOR_NUM_BINS],
        global: &Soothe3GlobalParams,
    ) {
        // Step 1: Build prefix sum of weighted sidechain power for O(1) range averages
        self.prefix_power[0] = 0.0;
        for k in 0..DETECTOR_NUM_BINS {
            let pw = (sc_power_bins[k] * sc_eq_weight_sq[k]).max(0.0) as f64;
            self.prefix_power[k + 1] = self.prefix_power[k] + pw;
        }

        // Step 2: Compute ERB local baseline power `\overline{P}_{ref}[k]` and narrow detail power `P_{det}[k]`
        for k in 0..DETECTOR_NUM_BINS {
            let ref_rad = self.erb_ref_radius_bins[k];
            let r_lo = k.saturating_sub(ref_rad);
            let r_hi = (k + ref_rad).min(DETECTOR_NUM_BINS - 1);
            let r_count = (r_hi - r_lo + 1) as f64;

            let detail_k = self.detail_per_bin[k].clamp(0.1, 25.0);
            // Below ~260 Hz (`k <= 22` at 48 kHz, where 1 bin = 11.7 Hz), keep single-bin detail resolution
            // so adjacent musical bass notes (e.g. 55 Hz, 80 Hz, 110 Hz) stay cleanly resolved.
            let det_rad = if k <= 22 || detail_k >= 4.0 {
                0
            } else {
                (((4.0 - detail_k) * 0.25 * (ref_rad as f32)).round() as usize)
                    .min(ref_rad.saturating_sub(2))
            };
            let d_lo = k.saturating_sub(det_rad);
            let d_hi = (k + det_rad).min(DETECTOR_NUM_BINS - 1);
            let d_count = (d_hi - d_lo + 1) as f64;

            let d_sum = self.prefix_power[d_hi + 1] - self.prefix_power[d_lo];
            let r_sum = self.prefix_power[r_hi + 1] - self.prefix_power[r_lo];

            let p_det = (d_sum / d_count) as f32;
            let pw_curr = self.prefix_power[k + 1] - self.prefix_power[k];
            let pw_prev = if k > 0 {
                self.prefix_power[k] - self.prefix_power[k - 1]
            } else {
                0.0
            };
            let pw_next = if k + 1 < DETECTOR_NUM_BINS {
                self.prefix_power[k + 2] - self.prefix_power[k + 1]
            } else {
                0.0
            };
            let is_local_peak = pw_curr >= pw_prev && pw_curr >= pw_next;
            let min_hole = if !is_local_peak {
                0
            } else if k <= 22 {
                1
            } else {
                2
            };
            let hole_rad = det_rad.max(min_hole).min(ref_rad.saturating_sub(2));
            let h_lo = k.saturating_sub(hole_rad);
            let h_hi = (k + hole_rad).min(DETECTOR_NUM_BINS - 1);
            let h_count = (h_hi - h_lo + 1) as f64;
            let h_sum = self.prefix_power[h_hi + 1] - self.prefix_power[h_lo];
            let p_ref_erb = if r_count > h_count {
                ((r_sum - h_sum) / (r_count - h_count)).max(0.0) as f32
            } else {
                (r_sum / r_count) as f32
            };
            let broad_mean =
                (self.prefix_power[DETECTOR_NUM_BINS] / (DETECTOR_NUM_BINS as f64)) as f32;
            let p_ref = p_ref_erb.max(broad_mean * 6.0);

            self.detail_power_bins[k] = p_det;
            self.baseline_power_bins[k] = p_ref;
        }

        // Step 3: Evaluate Soft vs Hard mode target gain reduction in dB
        let eps = 1e-12f32;
        let depth_norm = (global.depth * 0.1).clamp(0.0, 1.0);
        let unlimited_cut = global.max_cut_db <= -35.99;

        for k in 0..DETECTOR_NUM_BINS {
            let p_det = self.detail_power_bins[k];
            let p_ref = self.baseline_power_bins[k];
            let l_det_db = 10.0 * (p_det + eps).log10();
            let l_ref_db = 10.0 * (p_ref + eps).log10();

            if l_det_db < -90.0 || depth_norm <= 1e-5 {
                self.raw_cut_db_scratch[k] = 0.0;
                continue;
            }

            let prominence_db = (l_det_db - l_ref_db).max(0.0);
            let detail_k = self.detail_per_bin[k].clamp(0.1, 25.0);
            let selectivity_thresh_db = 6.0 / (0.5 + 0.25 * detail_k);

            let raw_cut_db = match global.mode {
                SootheProcessMode::Soft => {
                    let excess = (prominence_db - selectivity_thresh_db).max(0.0);
                    let knee = (excess * excess) / (excess + 3.0);
                    -(depth_norm * 2.4) * knee
                }
                SootheProcessMode::Hard => {
                    let prom_gate = ((prominence_db - 0.5 * selectivity_thresh_db)
                        / (0.5 * selectivity_thresh_db).max(0.5))
                        .clamp(0.0, 1.0);
                    let abs_thresh_db = -12.0 - 48.0 * depth_norm;
                    let abs_excess = (l_det_db - abs_thresh_db).max(0.0);
                    let combined = 0.70 * prominence_db + 0.30 * abs_excess * prom_gate;
                    let excess = (combined - 0.7 * selectivity_thresh_db).max(0.0);
                    -(depth_norm * 3.2) * excess * prom_gate
                }
            };

            self.raw_cut_db_scratch[k] = raw_cut_db;
        }

        // Step 4: Constant-Q zero-phase forward-backward 1-pole spectral smoothing (`apply_filtfilt`)
        // controlled by per-bin `Detail` & frequency, then clamp to `Max cut`.
        // Sub-bass bins (`k <= 15`, `< 175 Hz`) use minimal bin-to-bin bleed so narrow sub resonances
        // (e.g. 80 Hz between 55 Hz and 110 Hz) are surgically isolated without brickwall ringing.
        let mut fwd_state = self.raw_cut_db_scratch[0];
        for k in 0..DETECTOR_NUM_BINS {
            let detail_k = self.detail_per_bin[k].clamp(0.1, 25.0);
            let freq_scale = ((k as f32) / 48.0).clamp(0.02, 2.5);
            let pole = ((0.65 * freq_scale) / (1.0 + 1.35 * detail_k)).clamp(0.0, 0.82);
            fwd_state = pole * fwd_state + (1.0 - pole) * self.raw_cut_db_scratch[k];
            self.smooth_cut_db_scratch[k] = fwd_state;
        }
        let mut bwd_state = self.smooth_cut_db_scratch[DETECTOR_NUM_BINS - 1];
        for k in (0..DETECTOR_NUM_BINS).rev() {
            let detail_k = self.detail_per_bin[k].clamp(0.1, 25.0);
            let freq_scale = ((k as f32) / 48.0).clamp(0.02, 2.5);
            let pole = ((0.65 * freq_scale) / (1.0 + 1.35 * detail_k)).clamp(0.0, 0.82);
            bwd_state = pole * bwd_state + (1.0 - pole) * self.smooth_cut_db_scratch[k];
            let clamped_cut_db = if unlimited_cut {
                bwd_state.max(-60.0)
            } else {
                bwd_state.max(global.max_cut_db)
            };
            self.smooth_cut_db_scratch[k] = clamped_cut_db.min(0.0);
        }

        // Step 5: Temporal Attack / Release ballistics per bin (`integrate_asym`)
        for k in 0..DETECTOR_NUM_BINS {
            self.env_gr_db[ch][k] = integrate_asym(
                self.env_gr_db[ch][k],
                self.smooth_cut_db_scratch[k],
                self.attack_alpha_per_bin[k],
                self.release_alpha_per_bin[k],
            );
        }
    }

    /// Applies `Stereo link` between Channel 0 and Channel 1 and computes linear gain masks `G_lin[ch, k]`.
    pub fn link_channels_and_compute_linear_masks(&mut self, global: &Soothe3GlobalParams) {
        let link = (global.stereo_link_pct * 0.01).clamp(0.0, 1.0);
        let inverse = global.inverse;
        let wet_trim_lin = 10.0f32.powf(global.wet_trim_db / 20.0);

        for k in 0..DETECTOR_NUM_BINS {
            let gr0 = self.env_gr_db[0][k];
            let gr1 = self.env_gr_db[1][k];
            let max_cut_both = gr0.min(gr1);

            let linked0 = (1.0 - link) * gr0 + link * max_cut_both;
            let linked1 = (1.0 - link) * gr1 + link * max_cut_both;

            self.linked_gr_db[0][k] = linked0;
            self.linked_gr_db[1][k] = linked1;

            let g0 = 10.0f32.powf(linked0 / 20.0).clamp(0.0, 1.0);
            let g1 = 10.0f32.powf(linked1 / 20.0).clamp(0.0, 1.0);

            if inverse {
                self.linear_gain_mask[0][k] = ((1.0 - g0) * wet_trim_lin).clamp(0.0, 4.0);
                self.linear_gain_mask[1][k] = ((1.0 - g1) * wet_trim_lin).clamp(0.0, 4.0);
            } else {
                self.linear_gain_mask[0][k] = (g0 * wet_trim_lin).clamp(0.0, 4.0);
                self.linear_gain_mask[1][k] = (g1 * wet_trim_lin).clamp(0.0, 4.0);
            }
        }
    }
}
