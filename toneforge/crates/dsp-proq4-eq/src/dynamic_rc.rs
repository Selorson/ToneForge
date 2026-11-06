//! `dynamic_rc`: Wideband Dynamic EQ (`Detector` at `0x180221778` / `0x1801095A0` and
//! `Capacitor` at `0x180221710` / `0x18010A9B0`) running in double-precision `f64`.
//!
//! Includes:
//! - Sidechain bandpass isolation (`sc_low_hz .. sc_high_hz` or band's own `freq_hz` & `q_ui`)
//! - `SoftKneeDetector`: quadratic soft-knee static transfer curve with Auto Threshold adaptation
//! - `ProgramDependentCapacitor`: double-precision `f64` RC envelope follower with crest-factor-modulated
//!   attack and dual-stage program-dependent release time constants (`0x18010B2C0..0x18010B611`).

use crate::biquad_frac_slope::FractionalSlopeBiquadBand;
use crate::params_601::{EqBandParams, FilterShape};
use std::f64::consts::TAU;

/// Double-precision Program-Dependent RC Envelope Follower (`Capacitor` at `0x18010A9B0`).
#[derive(Clone, Copy, Debug)]
pub struct ProgramDependentCapacitor {
    pub env_state_db: f64,
    pub slow_crest_state_db: f64,
    pub attack_coeff_base: f64,  // Base attack time constant in seconds (e.g. 0.010 s)
    pub release_coeff_base: f64, // Base release time constant in seconds (e.g. 0.080 s)
}

impl Default for ProgramDependentCapacitor {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgramDependentCapacitor {
    pub fn new() -> Self {
        Self {
            env_state_db: 0.0,
            slow_crest_state_db: 0.0,
            attack_coeff_base: 0.010,
            release_coeff_base: 0.080,
        }
    }

    pub fn reset(&mut self) {
        self.env_state_db = 0.0;
        self.slow_crest_state_db = 0.0;
    }

    /// Configures base attack (`0.0005 .. 0.150 s`) and release (`0.005 .. 1.000 s`) time constants
    /// from the UI percentages `attack_pct` and `release_pct` (`0.0 ..= 100.0`).
    #[inline]
    pub fn set_ballistics_pct(&mut self, attack_pct: f32, release_pct: f32) {
        let a_norm = (attack_pct as f64 * 0.01).clamp(0.0, 1.0);
        let r_norm = (release_pct as f64 * 0.01).clamp(0.0, 1.0);
        self.attack_coeff_base = 0.0005 * (300.0f64).powf(a_norm);
        self.release_coeff_base = 0.005 * (200.0f64).powf(r_norm);
    }

    /// Processes one sample or sub-block step of the program-dependent RC follower (`0x18010B2C0`).
    #[inline]
    pub fn step(&mut self, target_gr_db: f64, sample_rate: f64) -> f64 {
        let fs = sample_rate.max(8000.0);
        let delta = target_gr_db - self.env_state_db;
        if delta.abs() <= 1e-7 {
            return self.env_state_db;
        }
        // Program-dependent time constant modulation (`0x18010B2C0..0x18010B611`)
        let coeff = if target_gr_db.abs() > self.env_state_db.abs() {
            let crest_factor = (target_gr_db.abs() - self.slow_crest_state_db).max(0.0);
            let tau_att = (self.attack_coeff_base / (1.0 + 0.25 * crest_factor)).max(0.0001);
            (-1.0 / (tau_att * fs)).exp()
        } else {
            let transient_ratio = (self.env_state_db.abs() - self.slow_crest_state_db).max(0.0);
            let tau_rel = (self.release_coeff_base / (1.0 + 0.5 * transient_ratio)).max(0.0005);
            (-1.0 / (tau_rel * fs)).exp()
        };
        self.env_state_db = coeff * self.env_state_db + (1.0 - coeff) * target_gr_db;
        let slow_alpha = (-1.0 / (0.150 * fs)).exp();
        self.slow_crest_state_db =
            slow_alpha * self.slow_crest_state_db + (1.0 - slow_alpha) * self.env_state_db.abs();
        self.env_state_db
    }
}

/// Sidechain level & quadratic soft-knee transfer curve evaluator (`Detector` at `0x1801095A0`).
#[derive(Clone, Copy, Debug)]
pub struct SoftKneeDetector {
    pub auto_bg_level_db: f64,
    pub sc_biquad: [f64; 5],
    pub sc_z: [[f64; 2]; 2],
}

impl Default for SoftKneeDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl SoftKneeDetector {
    pub fn new() -> Self {
        Self {
            auto_bg_level_db: -36.0,
            sc_biquad: FractionalSlopeBiquadBand::UNITY_SECTION,
            sc_z: [[0.0; 2]; 2],
        }
    }

    pub fn reset(&mut self) {
        self.auto_bg_level_db = -36.0;
        self.sc_z = [[0.0; 2]; 2];
    }

    /// Updates the sidechain bandpass isolation filter from `EqBandParams`.
    pub fn update_sidechain_filter(&mut self, band: &EqBandParams, sample_rate: f64) {
        let fs = sample_rate.max(8000.0);
        let (center_hz, q) = if band.sc_filtering {
            let f_lo = (band.sc_low_hz as f64).clamp(10.0, 0.48 * fs);
            let f_hi = (band.sc_high_hz as f64).clamp(f_lo + 10.0, 0.49 * fs);
            let f_c = (f_lo * f_hi).sqrt();
            let bw_oct = (f_hi / f_lo).log2().max(0.1);
            let q_bp = (1.0 / bw_oct).clamp(0.25, 16.0);
            (f_c, q_bp)
        } else {
            (
                (band.freq_hz as f64).clamp(10.0, 0.49 * fs),
                (band.q_ui as f64).clamp(0.25, 16.0),
            )
        };
        let w0 = (TAU * center_hz) / fs;
        self.sc_biquad = FractionalSlopeBiquadBand::biquad_bandpass(w0, q);
    }

    #[inline(always)]
    pub fn filter_sidechain_sample(&mut self, ch: usize, x: f64) -> f64 {
        let [b0, b1, b2, a1, a2] = self.sc_biquad;
        let [z1, z2] = self.sc_z[ch];
        let out = b0 * x + z1;
        self.sc_z[ch][0] = b1 * x - a1 * out + z2;
        self.sc_z[ch][1] = b2 * x - a2 * out;
        out
    }

    /// Computes the target dynamic gain offset in dB (`-30.0 ..= +30.0 dB`) using a quadratic
    /// soft knee (`0x180109F70` & `0x18010A102`).
    #[inline]
    pub fn compute_target_dynamic_gain_db(
        &mut self,
        sc_level_lin: f64,
        band: &EqBandParams,
        global_gain_scale: f32,
        sample_rate: f64,
    ) -> f64 {
        let dyn_range_db = band.effective_dyn_range_db(global_gain_scale) as f64;
        if !band.dyn_enabled || dyn_range_db.abs() < 1e-6 {
            return 0.0;
        }
        let level_db = 20.0 * sc_level_lin.max(1e-7).log10();

        // Update slow background level when `Dynamics Auto = 1`
        let eff_thresh_db = if band.dyn_auto {
            let alpha_bg = (-1.0 / (0.350 * sample_rate.max(8000.0))).exp();
            self.auto_bg_level_db =
                alpha_bg * self.auto_bg_level_db + (1.0 - alpha_bg) * level_db.max(-72.0);
            let manual_offset = (band.dyn_threshold_db as f64 + 24.0) * 0.5;
            (self.auto_bg_level_db + manual_offset).clamp(-80.0, -3.0)
        } else {
            (band.dyn_threshold_db as f64).clamp(-90.0, 0.0)
        };

        let knee_width_db = 6.0f64;
        let over_db = level_db - eff_thresh_db;
        if over_db <= -0.5 * knee_width_db {
            return 0.0;
        }

        // Quadratic soft-knee excess `E_db`
        let excess_db = if over_db < 0.5 * knee_width_db {
            let t = over_db + 0.5 * knee_width_db;
            (t * t) / (2.0 * knee_width_db)
        } else {
            over_db
        };

        // Asymptotic saturation towards `dyn_range_db` (`-30..+30 dB`)
        let norm_drive = excess_db / (excess_db + 12.0);
        (dyn_range_db * norm_drive).clamp(-30.0, 30.0)
    }
}

/// Complete per-band Dynamic EQ state combining `SoftKneeDetector` + `ProgramDependentCapacitor`.
#[derive(Clone, Copy, Debug, Default)]
pub struct DynamicEqBandProcessor {
    pub detector: SoftKneeDetector,
    pub capacitor: ProgramDependentCapacitor,
    pub current_dyn_gain_db: f64,
    pub update_counter: u32,
}

impl DynamicEqBandProcessor {
    pub fn new() -> Self {
        Self {
            detector: SoftKneeDetector::new(),
            capacitor: ProgramDependentCapacitor::new(),
            current_dyn_gain_db: 0.0,
            update_counter: 0,
        }
    }

    pub fn reset(&mut self) {
        self.detector.reset();
        self.capacitor.reset();
        self.current_dyn_gain_db = 0.0;
        self.update_counter = 0;
    }

    #[inline]
    pub fn step_dynamic_band(
        &mut self,
        band_params: &EqBandParams,
        biquad_band: &mut FractionalSlopeBiquadBand,
        sc_l: f64,
        sc_r: f64,
        global_gain_scale: f32,
        gain_q_interaction: bool,
        sample_rate: f64,
    ) -> f64 {
        let filt_l = self.detector.filter_sidechain_sample(0, sc_l);
        let filt_r = self.detector.filter_sidechain_sample(1, sc_r);
        let peak_sc = filt_l.abs().max(filt_r.abs());

        let target_db = self.detector.compute_target_dynamic_gain_db(
            peak_sc,
            band_params,
            global_gain_scale,
            sample_rate,
        );
        let smoothed_db = self.capacitor.step(target_db, sample_rate);

        // Update biquad coefficients every 8 samples when dynamic gain moves by > 0.01 dB
        self.update_counter = self.update_counter.wrapping_add(1);
        if (self.update_counter & 7) == 0 && (smoothed_db - self.current_dyn_gain_db).abs() > 0.01 {
            self.current_dyn_gain_db = smoothed_db;
            let total_gain_db =
                (band_params.effective_static_gain_db(global_gain_scale) as f64) + smoothed_db;
            if matches!(
                band_params.shape,
                FilterShape::Bell
                    | FilterShape::LowShelf
                    | FilterShape::HighShelf
                    | FilterShape::TiltShelf
                    | FilterShape::FlatTilt
            ) {
                biquad_band.configure_shape(
                    band_params.shape,
                    band_params.freq_hz as f64,
                    total_gain_db,
                    band_params.q_ui as f64,
                    band_params.slope_db_oct as f64,
                    sample_rate,
                    gain_q_interaction,
                );
            }
        }
        smoothed_db
    }
}
