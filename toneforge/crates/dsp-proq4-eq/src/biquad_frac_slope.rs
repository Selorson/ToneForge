//! `biquad_frac_slope`: Fractional-Slope Dual Biquad Cascade (`EQFilter` at `0x180221958`, `0x180112260`)
//! supporting all 9 Pro-Q 4 EQ filter shapes (`Bell`, `LowShelf`, `HighShelf`, `LowCut`, `HighCut`,
//! `Notch`, `BandPass`, `TiltShelf`, `FlatTilt`) plus `AllPass` across continuous slopes `6..96 dB/oct`,
//! with Analog-Console Gain-Q Interaction (`1.0905077326652577`, bitmask `0x1074`).

use crate::params_601::{FilterShape, PROQ4_GAIN_Q_INTERACTION_BASE};
use num_complex::Complex64;
use std::f64::consts::{FRAC_1_SQRT_2, PI, TAU};

/// Fractional-Slope Dual Biquad Cascade (`EQFilter` at `0x180221958`, `0x180112260`).
#[derive(Clone, Copy, Debug)]
pub struct FractionalSlopeBiquadBand {
    pub cascade_n1: [[f64; 5]; 8], // Lower integer order N1 (`rbx + 0xFB0`)
    pub cascade_n2: [[f64; 5]; 8], // Upper integer order N2 (`rbx + 0x10F0`)
    pub z_n1: [[[f64; 2]; 8]; 2],  // DF-II Transposed state [ch][sec][z1, z2]
    pub z_n2: [[[f64; 2]; 8]; 2],
    pub sections_n1: usize,
    pub sections_n2: usize,
    pub alpha_frac: f64, // Fractional slope blend frac(S / 6.0) in [0.0, 1.0)
}

impl Default for FractionalSlopeBiquadBand {
    fn default() -> Self {
        Self::new_unity()
    }
}

impl FractionalSlopeBiquadBand {
    pub const UNITY_SECTION: [f64; 5] = [1.0, 0.0, 0.0, 0.0, 0.0];

    pub fn new_unity() -> Self {
        Self {
            cascade_n1: [Self::UNITY_SECTION; 8],
            cascade_n2: [Self::UNITY_SECTION; 8],
            z_n1: [[[0.0; 2]; 8]; 2],
            z_n2: [[[0.0; 2]; 8]; 2],
            sections_n1: 1,
            sections_n2: 1,
            alpha_frac: 0.0,
        }
    }

    /// Analog-console Gain-Q interaction (`0x18010FF70`, bitmask `0x1074`):
    /// `Q_eff = (0.7071067811865475 * Q_UI) * 1.0905077326652577^(min(|G_dB|, 12.0) * 0.25)`
    #[inline]
    pub fn compute_effective_q(q_ui: f64, gain_db: f64, gain_q_interaction: bool) -> f64 {
        let q_internal = FRAC_1_SQRT_2 * q_ui;
        if !gain_q_interaction {
            return q_internal;
        }
        let clamped_db = gain_db.abs().min(12.0);
        q_internal * PROQ4_GAIN_Q_INTERACTION_BASE.powf(clamped_db * 0.25)
    }

    /// Configures a Bell peaking filter with continuous fractional slope `slope_db_oct` (`6.0..=96.0`).
    pub fn configure_bell(
        &mut self,
        freq_hz: f64,
        gain_db: f64,
        q_ui: f64,
        slope_db_oct: f64,
        sample_rate: f64,
        gain_q_interaction: bool,
    ) {
        self.configure_shape(
            FilterShape::Bell,
            freq_hz,
            gain_db,
            q_ui,
            slope_db_oct,
            sample_rate,
            gain_q_interaction,
        );
    }

    /// Configures any of the 9 EQ filter shapes (+ `AllPass`) with continuous fractional slope `6.0..=96.0 dB/oct`.
    pub fn configure_shape(
        &mut self,
        shape: FilterShape,
        freq_hz: f64,
        gain_db: f64,
        q_ui: f64,
        slope_db_oct: f64,
        sample_rate: f64,
        gain_q_interaction: bool,
    ) {
        let fs = sample_rate.max(8000.0);
        let f0 = freq_hz.clamp(10.0, 0.495 * fs);
        let w0 = (TAU * f0) / fs;

        let s_norm = (slope_db_oct.clamp(6.0, 96.0) / 6.0).max(1.0);
        let order_low = (s_norm.floor() as usize).clamp(1, 16);
        let order_high = (order_low + 1).min(16);
        self.alpha_frac = if order_low == order_high {
            0.0
        } else {
            s_norm.fract()
        };

        self.cascade_n1 = [Self::UNITY_SECTION; 8];
        self.cascade_n2 = [Self::UNITY_SECTION; 8];

        match shape {
            FilterShape::Bell => {
                let n1 = order_low.clamp(1, 8);
                let n2 = order_high.clamp(1, 8);
                self.sections_n1 = n1;
                self.sections_n2 = n2;
                let q_eff = Self::compute_effective_q(q_ui, gain_db, gain_q_interaction).max(0.025);
                let sec1 = Self::biquad_peaking(w0, gain_db / (n1 as f64), q_eff);
                let sec2 = Self::biquad_peaking(w0, gain_db / (n2 as f64), q_eff);
                for s in 0..n1 {
                    self.cascade_n1[s] = sec1;
                }
                for s in 0..n2 {
                    self.cascade_n2[s] = sec2;
                }
            }
            FilterShape::LowShelf => {
                let n1 = order_low.clamp(1, 8);
                let n2 = order_high.clamp(1, 8);
                self.sections_n1 = n1;
                self.sections_n2 = n2;
                let q_eff =
                    Self::compute_effective_q(q_ui, gain_db, gain_q_interaction).clamp(0.125, 16.0);
                let sec1 = Self::biquad_low_shelf(w0, gain_db / (n1 as f64), q_eff);
                let sec2 = Self::biquad_low_shelf(w0, gain_db / (n2 as f64), q_eff);
                for s in 0..n1 {
                    self.cascade_n1[s] = sec1;
                }
                for s in 0..n2 {
                    self.cascade_n2[s] = sec2;
                }
            }
            FilterShape::HighShelf => {
                let n1 = order_low.clamp(1, 8);
                let n2 = order_high.clamp(1, 8);
                self.sections_n1 = n1;
                self.sections_n2 = n2;
                let q_eff =
                    Self::compute_effective_q(q_ui, gain_db, gain_q_interaction).clamp(0.125, 16.0);
                let sec1 = Self::biquad_high_shelf(w0, gain_db / (n1 as f64), q_eff);
                let sec2 = Self::biquad_high_shelf(w0, gain_db / (n2 as f64), q_eff);
                for s in 0..n1 {
                    self.cascade_n1[s] = sec1;
                }
                for s in 0..n2 {
                    self.cascade_n2[s] = sec2;
                }
            }
            FilterShape::LowCut => {
                let n1 = Self::build_butterworth_cut_cascade(
                    &mut self.cascade_n1,
                    order_low,
                    w0,
                    q_ui,
                    true,
                );
                self.sections_n1 = n1;
                self.cascade_n2 = self.cascade_n1;
                if self.alpha_frac > 1e-6 && n1 < 8 {
                    self.cascade_n2[n1] = Self::first_order_cut(w0, true);
                    self.sections_n2 = n1 + 1;
                } else {
                    self.sections_n2 = n1;
                }
            }
            FilterShape::HighCut => {
                let n1 = Self::build_butterworth_cut_cascade(
                    &mut self.cascade_n1,
                    order_low,
                    w0,
                    q_ui,
                    false,
                );
                self.sections_n1 = n1;
                self.cascade_n2 = self.cascade_n1;
                if self.alpha_frac > 1e-6 && n1 < 8 {
                    self.cascade_n2[n1] = Self::first_order_cut(w0, false);
                    self.sections_n2 = n1 + 1;
                } else {
                    self.sections_n2 = n1;
                }
            }
            FilterShape::Notch => {
                let n1 = order_low.div_ceil(2).clamp(1, 8);
                let n2 = order_high.div_ceil(2).clamp(1, 8);
                self.sections_n1 = n1;
                self.sections_n2 = n2;
                let q_eff = (FRAC_1_SQRT_2 * q_ui).clamp(0.05, 40.0);
                let sec1 = Self::biquad_notch(w0, q_eff);
                let sec2 = Self::biquad_notch(w0, q_eff);
                for s in 0..n1 {
                    self.cascade_n1[s] = sec1;
                }
                for s in 0..n2 {
                    self.cascade_n2[s] = sec2;
                }
            }
            FilterShape::BandPass => {
                let n1 = order_low.div_ceil(2).clamp(1, 8);
                let n2 = order_high.div_ceil(2).clamp(1, 8);
                self.sections_n1 = n1;
                self.sections_n2 = n2;
                let q_eff = (FRAC_1_SQRT_2 * q_ui).clamp(0.05, 40.0);
                let sec1 = Self::biquad_bandpass(w0, q_eff);
                let sec2 = Self::biquad_bandpass(w0, q_eff);
                for s in 0..n1 {
                    self.cascade_n1[s] = sec1;
                }
                for s in 0..n2 {
                    self.cascade_n2[s] = sec2;
                }
            }
            FilterShape::TiltShelf => {
                // Symmetric tilt around f0: -gain_db/2 at DC, +gain_db/2 at Nyquist
                let n1 = order_low.div_ceil(2).clamp(1, 8);
                let n2 = order_high.div_ceil(2).clamp(1, 8);
                self.sections_n1 = n1;
                self.sections_n2 = n2;
                let q_eff = (FRAC_1_SQRT_2 * q_ui).clamp(0.125, 8.0);
                let sec1 = Self::biquad_tilt_shelf(w0, gain_db / (n1 as f64), q_eff);
                let sec2 = Self::biquad_tilt_shelf(w0, gain_db / (n2 as f64), q_eff);
                for s in 0..n1 {
                    self.cascade_n1[s] = sec1;
                }
                for s in 0..n2 {
                    self.cascade_n2[s] = sec2;
                }
            }
            FilterShape::FlatTilt => {
                // Spectral tilt across 20 Hz..20 kHz pivoted at f0 (4 log-spaced shelf pairs)
                let num_pivots = 4usize;
                self.sections_n1 = num_pivots;
                self.sections_n2 = num_pivots;
                self.alpha_frac = 0.0;
                Self::build_flat_tilt_cascade(&mut self.cascade_n1, f0, gain_db, fs);
                self.cascade_n2 = self.cascade_n1;
            }
            FilterShape::AllPass => {
                let n1 = order_low.div_ceil(2).clamp(1, 8);
                let n2 = order_high.div_ceil(2).clamp(1, 8);
                self.sections_n1 = n1;
                self.sections_n2 = n2;
                let q_eff = (FRAC_1_SQRT_2 * q_ui).clamp(0.05, 40.0);
                let sec = Self::biquad_allpass(w0, q_eff);
                for s in 0..n1 {
                    self.cascade_n1[s] = sec;
                }
                for s in 0..n2 {
                    self.cascade_n2[s] = sec;
                }
            }
        }
    }

    #[inline]
    pub fn biquad_peaking(w0: f64, gain_db: f64, q: f64) -> [f64; 5] {
        if gain_db.abs() < 1e-9 {
            return Self::UNITY_SECTION;
        }
        let a = 10.0f64.powf(gain_db / 40.0);
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q.max(0.025));
        let b0 = 1.0 + alpha * a;
        let b1 = -2.0 * cos_w0;
        let b2 = 1.0 - alpha * a;
        let a0 = 1.0 + alpha / a;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha / a;
        let inv_a0 = 1.0 / a0;
        [b0 * inv_a0, b1 * inv_a0, b2 * inv_a0, a1 * inv_a0, a2 * inv_a0]
    }

    #[inline]
    pub fn biquad_low_shelf(w0: f64, gain_db: f64, q: f64) -> [f64; 5] {
        if gain_db.abs() < 1e-9 {
            return Self::UNITY_SECTION;
        }
        let a = 10.0f64.powf(gain_db / 40.0);
        let sqrt_a = a.sqrt();
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q.max(0.1));
        let two_sqrt_a_alpha = 2.0 * sqrt_a * alpha;

        let b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha);
        let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0);
        let b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha);
        let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
        let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0);
        let a2 = (a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha;
        let inv_a0 = 1.0 / a0;
        [b0 * inv_a0, b1 * inv_a0, b2 * inv_a0, a1 * inv_a0, a2 * inv_a0]
    }

    #[inline]
    pub fn biquad_high_shelf(w0: f64, gain_db: f64, q: f64) -> [f64; 5] {
        if gain_db.abs() < 1e-9 {
            return Self::UNITY_SECTION;
        }
        let a = 10.0f64.powf(gain_db / 40.0);
        let sqrt_a = a.sqrt();
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q.max(0.1));
        let two_sqrt_a_alpha = 2.0 * sqrt_a * alpha;

        let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha);
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
        let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha);
        let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
        let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
        let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha;
        let inv_a0 = 1.0 / a0;
        [b0 * inv_a0, b1 * inv_a0, b2 * inv_a0, a1 * inv_a0, a2 * inv_a0]
    }

    #[inline]
    pub fn biquad_tilt_shelf(w0: f64, gain_db: f64, q: f64) -> [f64; 5] {
        if gain_db.abs() < 1e-9 {
            return Self::UNITY_SECTION;
        }
        // Symmetric Tilt Shelf: HighShelf(+gain_db) scaled by 10^(-gain_db/40) so DC is -gain_db/2 and HF is +gain_db/2
        let hs = Self::biquad_high_shelf(w0, gain_db, q);
        let dc_scale = 10.0f64.powf(-gain_db / 40.0);
        [
            hs[0] * dc_scale,
            hs[1] * dc_scale,
            hs[2] * dc_scale,
            hs[3],
            hs[4],
        ]
    }

    #[inline]
    pub fn biquad_notch(w0: f64, q: f64) -> [f64; 5] {
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q.max(0.05));
        let b0 = 1.0;
        let b1 = -2.0 * cos_w0;
        let b2 = 1.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        let inv_a0 = 1.0 / a0;
        [b0 * inv_a0, b1 * inv_a0, b2 * inv_a0, a1 * inv_a0, a2 * inv_a0]
    }

    #[inline]
    pub fn biquad_bandpass(w0: f64, q: f64) -> [f64; 5] {
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q.max(0.05));
        let b0 = alpha;
        let b1 = 0.0;
        let b2 = -alpha;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        let inv_a0 = 1.0 / a0;
        [b0 * inv_a0, b1 * inv_a0, b2 * inv_a0, a1 * inv_a0, a2 * inv_a0]
    }

    #[inline]
    pub fn biquad_allpass(w0: f64, q: f64) -> [f64; 5] {
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q.max(0.05));
        let b0 = 1.0 - alpha;
        let b1 = -2.0 * cos_w0;
        let b2 = 1.0 + alpha;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        let inv_a0 = 1.0 / a0;
        [b0 * inv_a0, b1 * inv_a0, b2 * inv_a0, a1 * inv_a0, a2 * inv_a0]
    }

    #[inline]
    pub fn first_order_cut(w0: f64, is_highpass: bool) -> [f64; 5] {
        let k = (0.5 * w0).tan();
        let inv = 1.0 / (1.0 + k);
        if is_highpass {
            [inv, -inv, 0.0, (k - 1.0) * inv, 0.0]
        } else {
            let b = k * inv;
            [b, b, 0.0, (k - 1.0) * inv, 0.0]
        }
    }

    #[inline]
    pub fn second_order_cut(w0: f64, q: f64, is_highpass: bool) -> [f64; 5] {
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q.max(0.05));
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        let inv_a0 = 1.0 / a0;
        if is_highpass {
            let b0 = 0.5 * (1.0 + cos_w0);
            let b1 = -(1.0 + cos_w0);
            let b2 = b0;
            [b0 * inv_a0, b1 * inv_a0, b2 * inv_a0, a1 * inv_a0, a2 * inv_a0]
        } else {
            let b0 = 0.5 * (1.0 - cos_w0);
            let b1 = 1.0 - cos_w0;
            let b2 = b0;
            [b0 * inv_a0, b1 * inv_a0, b2 * inv_a0, a1 * inv_a0, a2 * inv_a0]
        }
    }

    /// Builds an N-pole (`N = 1..=16`, corresponding to `6..96 dB/oct`) Butterworth / Q-shaped
    /// High-Pass (`is_highpass = true`) or Low-Pass (`is_highpass = false`) cascade.
    pub fn build_butterworth_cut_cascade(
        cascade: &mut [[f64; 5]; 8],
        num_poles: usize,
        w0: f64,
        q_ui: f64,
        is_highpass: bool,
    ) -> usize {
        let poles = num_poles.clamp(1, 16);
        let pairs = poles / 2;
        let has_odd_pole = (poles % 2) == 1;
        let mut sec_idx = 0usize;

        let odd_offset = (poles % 2) as f64;
        for k in 0..pairs {
            let angle = PI * (2.0 * (k as f64) + 1.0 + odd_offset) / (2.0 * (poles as f64));
            let mut q_sec = 1.0 / (2.0 * angle.cos());
            // Apply UI resonance Q scaling on the last (highest-Q) pole pair
            if k + 1 == pairs {
                q_sec *= q_ui.clamp(0.1, 20.0);
            }
            cascade[sec_idx] = Self::second_order_cut(w0, q_sec, is_highpass);
            sec_idx += 1;
        }

        if has_odd_pole && sec_idx < 8 {
            cascade[sec_idx] = Self::first_order_cut(w0, is_highpass);
            sec_idx += 1;
        }

        sec_idx.max(1)
    }

    /// Builds a 4-section logarithmic `FlatTilt` cascade producing a linear-in-dB/octave
    /// spectral tilt across `20 Hz .. 20 kHz` with 0 dB pivot at `f0_hz`.
    pub fn build_flat_tilt_cascade(
        cascade: &mut [[f64; 5]; 8],
        f0_hz: f64,
        tilt_gain_db: f64,
        sample_rate: f64,
    ) {
        if tilt_gain_db.abs() < 1e-9 {
            return;
        }
        // Distribute 4 wide tilt-shelf sections at 62.5 Hz, 250 Hz, 1000 Hz, 4000 Hz
        let pivot_freqs = [62.5f64, 250.0, 1000.0, 4000.0];
        let sec_gain_db = tilt_gain_db * 0.5;
        for (i, &fp) in pivot_freqs.iter().enumerate() {
            let w = (TAU * fp.min(0.49 * sample_rate)) / sample_rate;
            cascade[i] = Self::biquad_tilt_shelf(w, sec_gain_db, 0.38);
        }
        // Normalize gain at pivot frequency f0_hz to 0 dB (1.0 linear)
        let w_pivot = (TAU * f0_hz.clamp(20.0, 0.49 * sample_rate)) / sample_rate;
        let mag_at_pivot = Self::eval_cascade_complex(cascade, 4, w_pivot).norm().max(1e-9);
        let norm_scale = 1.0 / mag_at_pivot;
        cascade[0][0] *= norm_scale;
        cascade[0][1] *= norm_scale;
        cascade[0][2] *= norm_scale;
    }

    pub fn reset_state(&mut self) {
        self.z_n1 = [[[0.0; 2]; 8]; 2];
        self.z_n2 = [[[0.0; 2]; 8]; 2];
    }

    #[inline(always)]
    pub fn process_sample(&mut self, ch: usize, x: f64) -> f64 {
        let mut y1 = x;
        for s in 0..self.sections_n1 {
            let [b0, b1, b2, a1, a2] = self.cascade_n1[s];
            let [z1, z2] = self.z_n1[ch][s];
            let out = b0 * y1 + z1;
            self.z_n1[ch][s][0] = b1 * y1 - a1 * out + z2;
            self.z_n1[ch][s][1] = b2 * y1 - a2 * out;
            y1 = out;
        }
        if self.alpha_frac <= 1e-6 {
            return y1;
        }
        let mut y2 = x;
        for s in 0..self.sections_n2 {
            let [b0, b1, b2, a1, a2] = self.cascade_n2[s];
            let [z1, z2] = self.z_n2[ch][s];
            let out = b0 * y2 + z1;
            self.z_n2[ch][s][0] = b1 * y2 - a1 * out + z2;
            self.z_n2[ch][s][1] = b2 * y2 - a2 * out;
            y2 = out;
        }
        (1.0 - self.alpha_frac) * y1 + self.alpha_frac * y2
    }

    /// Evaluates the exact complex transfer function `H(e^{jw})` of a biquad cascade.
    #[inline]
    pub fn eval_cascade_complex(
        cascade: &[[f64; 5]; 8],
        num_sections: usize,
        w: f64,
    ) -> Complex64 {
        let z1 = Complex64::from_polar(1.0, -w);
        let z2 = Complex64::from_polar(1.0, -2.0 * w);
        let mut h = Complex64::new(1.0, 0.0);
        for s in 0..num_sections.min(8) {
            let [b0, b1, b2, a1, a2] = cascade[s];
            let num = Complex64::new(b0, 0.0) + z1 * b1 + z2 * b2;
            let den = Complex64::new(1.0, 0.0) + z1 * a1 + z2 * a2;
            h *= num / den;
        }
        h
    }

    /// Evaluates the exact fractional-slope blended complex frequency response `H(f)`.
    #[inline]
    pub fn eval_complex_response(&self, freq_hz: f64, sample_rate: f64) -> Complex64 {
        let fs = sample_rate.max(8000.0);
        let w = (TAU * freq_hz.clamp(1.0, 0.4999 * fs)) / fs;
        let h1 = Self::eval_cascade_complex(&self.cascade_n1, self.sections_n1, w);
        if self.alpha_frac <= 1e-6 {
            return h1;
        }
        let h2 = Self::eval_cascade_complex(&self.cascade_n2, self.sections_n2, w);
        h1 * (1.0 - self.alpha_frac) + h2 * self.alpha_frac
    }

    #[inline]
    pub fn eval_magnitude_db(&self, freq_hz: f64, sample_rate: f64) -> f64 {
        let mag = self.eval_complex_response(freq_hz, sample_rate).norm().max(1e-12);
        20.0 * mag.log10()
    }

    #[inline]
    pub fn eval_phase_rad(&self, freq_hz: f64, sample_rate: f64) -> f64 {
        self.eval_complex_response(freq_hz, sample_rate).arg()
    }
}
