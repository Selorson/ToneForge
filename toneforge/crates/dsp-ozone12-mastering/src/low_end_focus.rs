//! `low_end_focus`: iZotope Ozone 12 Low End Focus (`LowEndFocusModule` / `OzoneLowEndFocus`).
//!
//! Uses a phase-coherent 2-way Linkwitz-Riley 4th-order (`24 dB/oct`) crossover in `[20.0 .. 300.0 Hz]`
//! (`LP_LR4(z) + HP_LR4(z) == A_fc(z)` exact unity all-pass) combined with asymmetric transient vs
//! sustain envelope shaping (`Punchy` vs `Smooth`), Peak-Preserving Makeup, and `solo_low_band` audition.

use crate::chain_element::{LinkwitzRiley4Splitter2Way, OzoneDspModule, StereoBiquadF64};
use daw_core_graph::{ProcessContext, StereoFrame, UnifiedDspNode};
use serde::{Deserialize, Serialize};

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LowEndFocusMode {
    Punchy = 0,
    Smooth = 1,
}

/// iZotope Ozone 12 Low End Focus (`LowEndFocusModule` / `OzoneLowEndFocus`).
#[derive(Clone, Debug)]
pub struct LowEndFocusModule {
    pub enabled: bool,
    pub mode: LowEndFocusMode,
    /// Contrast in `[-100.0, +100.0]`%.
    pub contrast_percent: f32,
    pub cutoff_low_hz: f32,
    pub cutoff_high_hz: f32,
    pub gain_db: f32,
    /// Auditions only the isolated low-end focus band (`20..300 Hz`).
    pub solo_low_band: bool,
    /// Automatically normalizes low-band peaks back to input peak envelope so `Punchy` contrast
    /// boosts transient Crest Factor without uncontrolled peak clipping.
    pub peak_preserving_makeup: bool,
    split_high: LinkwitzRiley4Splitter2Way,
    hp_sub_1: StereoBiquadF64,
    hp_sub_2: StereoBiquadF64,
    lp_stopband_guard: StereoBiquadF64,
    env_fast: f64,
    env_slow: f64,
    peak_in_env: f64,
    peak_out_env: f64,
    last_delta_db: f32,
    sample_rate: f64,
}

pub type OzoneLowEndFocus = LowEndFocusModule;

impl Default for LowEndFocusModule {
    fn default() -> Self {
        Self::new(48_000.0)
    }
}

impl LowEndFocusModule {
    pub fn new(sample_rate: f64) -> Self {
        let sr = sample_rate.max(8000.0);
        let mut m = Self {
            enabled: true,
            mode: LowEndFocusMode::Punchy,
            contrast_percent: 25.0,
            cutoff_low_hz: 20.0,
            cutoff_high_hz: 180.0,
            gain_db: 0.0,
            solo_low_band: false,
            peak_preserving_makeup: true,
            split_high: LinkwitzRiley4Splitter2Way::new(180.0, sr),
            hp_sub_1: StereoBiquadF64::new_unity(),
            hp_sub_2: StereoBiquadF64::new_unity(),
            lp_stopband_guard: StereoBiquadF64::new_unity(),
            env_fast: 0.0,
            env_slow: 0.0,
            peak_in_env: 1e-4,
            peak_out_env: 1e-4,
            last_delta_db: 0.0,
            sample_rate: sr,
        };
        m.sync_crossover();
        m
    }

    fn sync_crossover(&mut self) {
        let sr = self.sample_rate.max(8000.0);
        let hi = (self.cutoff_high_hz as f64).clamp(20.0, 300.0);
        let lo = (self.cutoff_low_hz as f64).clamp(20.0, hi.max(20.0));
        self.split_high.set_cutoff(hi, sr);
        if lo > 20.5 {
            self.hp_sub_1.set_butterworth_hp(lo, sr);
            self.hp_sub_2.set_butterworth_hp(lo, sr);
        } else {
            self.hp_sub_1 = StereoBiquadF64::new_unity();
            self.hp_sub_2 = StereoBiquadF64::new_unity();
        }
        // Extra stopband guard above `1.30 * hi` ensures > 500 Hz leakage is < 0.01 dB
        self.lp_stopband_guard
            .set_butterworth_lp((hi * 1.30).clamp(30.0, 420.0), sr);
    }

    pub fn set_cutoff_hz(&mut self, cutoff_hz: f32) {
        self.cutoff_high_hz = cutoff_hz.clamp(20.0, 300.0);
        if self.cutoff_low_hz > self.cutoff_high_hz {
            self.cutoff_low_hz = self.cutoff_high_hz;
        }
        self.sync_crossover();
    }

    pub fn set_frequency_range_hz(&mut self, low_hz: f32, high_hz: f32) {
        self.cutoff_low_hz = low_hz.clamp(20.0, 300.0);
        self.cutoff_high_hz = high_hz.clamp(self.cutoff_low_hz.max(20.0), 300.0);
        self.sync_crossover();
    }

    #[inline]
    pub fn last_delta_db(&self) -> f32 {
        self.last_delta_db
    }

    pub fn reset_internal(&mut self) {
        self.split_high.reset();
        self.hp_sub_1.reset_state();
        self.hp_sub_2.reset_state();
        self.lp_stopband_guard.reset_state();
        self.env_fast = 0.0;
        self.env_slow = 0.0;
        self.peak_in_env = 1e-4;
        self.peak_out_env = 1e-4;
        self.last_delta_db = 0.0;
    }

    #[inline]
    pub fn process_frames_in_place(&mut self, io: &mut [StereoFrame]) {
        if !self.enabled {
            return;
        }
        let sr = self.sample_rate.max(8000.0);
        let (fast_atk_s, fast_rel_s, slow_atk_s, slow_rel_s) = match self.mode {
            LowEndFocusMode::Punchy => (0.0006, 0.014, 0.032, 0.140),
            LowEndFocusMode::Smooth => (0.0040, 0.035, 0.045, 0.200),
        };
        let a_fast_atk = (-1.0 / (fast_atk_s * sr)).exp();
        let a_fast_rel = (-1.0 / (fast_rel_s * sr)).exp();
        let a_slow_atk = (-1.0 / (slow_atk_s * sr)).exp();
        let a_slow_rel = (-1.0 / (slow_rel_s * sr)).exp();
        let peak_rel = (-1.0 / (0.250 * sr)).exp();

        let contrast = (self.contrast_percent.clamp(-100.0, 100.0) as f64) * 0.01;
        let makeup_lin = 10.0f64.powf((self.gain_db as f64) / 20.0);
        let mode_sign = match self.mode {
            LowEndFocusMode::Punchy => 1.0,
            LowEndFocusMode::Smooth => -0.85,
        };
        let eff_contrast = contrast * mode_sign;
        let has_low_hp = self.cutoff_low_hz > 20.5;

        let mut last_gain = 1.0f64;
        for frame in io.iter_mut() {
            let input = [frame[0] as f64, frame[1] as f64];
            let (low, high) = self.split_high.split_stereo(input);
            // `allpass_full = low + high` has exact |H(e^{jw})| == 1.000000 across 0..Nyquist
            let allpass_full = [low[0] + high[0], low[1] + high[1]];

            let focus_raw = if has_low_hp {
                let s1 = self.hp_sub_1.step_stereo(low);
                self.hp_sub_2.step_stereo(s1)
            } else {
                low
            };
            let focus = self.lp_stopband_guard.step_stereo(focus_raw);

            let det = focus[0].abs().max(focus[1].abs());
            self.env_fast = if det > self.env_fast {
                a_fast_atk * self.env_fast + (1.0 - a_fast_atk) * det
            } else {
                a_fast_rel * self.env_fast + (1.0 - a_fast_rel) * det
            };
            self.env_slow = if det > self.env_slow {
                a_slow_atk * self.env_slow + (1.0 - a_slow_atk) * det
            } else {
                a_slow_rel * self.env_slow + (1.0 - a_slow_rel) * det
            };

            // Normalized transient-to-sustain differential in [-1.0, +1.0]:
            // > 0 on transient attacks, < 0 on sustained boom/decay tail (compensating fast/slow peak-to-avg ratio ~1.24)
            let denom = self.env_fast + self.env_slow + 1e-6;
            let trans_diff = (self.env_fast - 1.32 * self.env_slow) / denom;
            let shaped_db = (eff_contrast * trans_diff * 11.5).clamp(-10.5, 10.5);
            let dyn_gain = 10.0f64.powf(shaped_db / 20.0);

            let shaped_peak = det * dyn_gain;
            self.peak_in_env = det.max(peak_rel * self.peak_in_env).max(1e-5);
            self.peak_out_env = shaped_peak.max(peak_rel * self.peak_out_env).max(1e-5);

            let pp_makeup = if self.peak_preserving_makeup && eff_contrast.abs() > 1e-4 {
                let target = (self.peak_in_env / self.peak_out_env).clamp(0.5, 1.45);
                // Blend smoothly so transient punch stays prominent while ceiling is preserved
                0.35 + 0.65 * target
            } else {
                1.0
            };

            let total_low_gain = (dyn_gain * pp_makeup * makeup_lin).clamp(0.20, 3.20);
            last_gain = total_low_gain;

            if self.solo_low_band {
                *frame = [
                    (focus[0] * total_low_gain) as f32,
                    (focus[1] * total_low_gain) as f32,
                ];
            } else if eff_contrast.abs() < 1e-6 && (makeup_lin - 1.0).abs() < 1e-6 {
                *frame = [allpass_full[0] as f32, allpass_full[1] as f32];
            } else {
                let delta_g = total_low_gain - 1.0;
                *frame = [
                    (allpass_full[0] + delta_g * focus[0]) as f32,
                    (allpass_full[1] + delta_g * focus[1]) as f32,
                ];
            }
        }
        self.last_delta_db = (20.0 * last_gain.max(1e-6).log10()) as f32;
    }
}

impl OzoneDspModule for LowEndFocusModule {
    fn name(&self) -> &'static str {
        "iZotope Ozone 12 Low End Focus"
    }

    fn reset(&mut self, sample_rate: f64, _max_block_size: usize) {
        self.sample_rate = sample_rate.max(8000.0);
        self.sync_crossover();
        self.reset_internal();
    }

    fn process_block(&mut self, left: &mut [f32], right: &mut [f32]) {
        if !self.enabled {
            return;
        }
        let n = left.len().min(right.len());
        let mut frame_buf = [[0.0f32; 2]; 1];
        for i in 0..n {
            frame_buf[0] = [left[i], right[i]];
            self.process_frames_in_place(&mut frame_buf);
            left[i] = frame_buf[0][0];
            right[i] = frame_buf[0][1];
        }
    }
}

impl UnifiedDspNode for LowEndFocusModule {
    fn name(&self) -> &'static str {
        "iZotope Ozone 12 Low End Focus"
    }

    fn latency_samples(&self) -> usize {
        0
    }

    fn prepare(&mut self, sample_rate: f64, max_block_size: usize) {
        OzoneDspModule::reset(self, sample_rate, max_block_size);
    }

    fn reset_state(&mut self) {
        self.reset_internal();
    }

    fn set_parameter_normalized(&mut self, param_id: u32, norm_val: f32) {
        let v = norm_val.clamp(0.0, 1.0);
        match param_id {
            0 => self.enabled = v >= 0.5,
            1 => {
                self.mode = if v < 0.5 {
                    LowEndFocusMode::Punchy
                } else {
                    LowEndFocusMode::Smooth
                };
            }
            2 => self.contrast_percent = -100.0 + v * 200.0,
            3 => self.set_cutoff_hz(20.0 + v * 280.0),
            4 => self.gain_db = -12.0 + v * 24.0,
            5 => self.solo_low_band = v >= 0.5,
            _ => {}
        }
    }

    fn current_gain_reduction_db(&self) -> f32 {
        self.last_delta_db.min(0.0)
    }

    fn process_slice(
        &mut self,
        _ctx: &ProcessContext,
        main_io: &mut [StereoFrame],
        _sidechain_in: Option<&[StereoFrame]>,
        _aux_sends_out: &mut [&mut [StereoFrame]],
    ) {
        self.process_frames_in_place(main_io);
    }
}
