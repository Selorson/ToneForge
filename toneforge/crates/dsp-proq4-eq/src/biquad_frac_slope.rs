//! `biquad_frac_slope`: Fractional-Slope Dual Biquad Cascade (`EQFilter` at `0x180221958`, `0x180112260`)
//! supporting all 9 Pro-Q 4 EQ filter shapes (`Bell`, `LowShelf`, `HighShelf`, `LowCut`, `HighCut`,
//! `Notch`, `BandPass`, `TiltShelf`, `FlatTilt`) plus `AllPass` across continuous slopes `6..96 dB/oct`,
//! with Analog-Console Gain-Q Interaction (`1.0905077326652577`, bitmask `0x1074`).

use crate::params_601::{FilterShape, PROQ4_GAIN_Q_INTERACTION_BASE};
use num_complex::Complex64;
use std::f64::consts::{FRAC_1_SQRT_2, PI, TAU};

/// Trapezoidal Integrator State-Variable Filter (Cytomic SVF, `0x180111550`) realization
/// bijectively mapped from any causal stable biquad `[b0, b1, b2, a1, a2]`.
/// Preserves modulation stability and `f64` low-frequency precision on dynamic & oversampled bands.
#[derive(Clone, Copy, Debug)]
pub struct CytomicSvfCoeffs {
    pub a1: f64,
    pub a2: f64,
    pub a3: f64,
    pub m0: f64,
    pub m1: f64,
    pub m2: f64,
    pub is_unity: bool,
}

impl Default for CytomicSvfCoeffs {
    fn default() -> Self {
        Self::UNITY
    }
}

impl CytomicSvfCoeffs {
    pub const UNITY: Self = Self {
        a1: 0.25,
        a2: 0.25,
        a3: 0.25,
        m0: 1.0,
        m1: 0.0,
        m2: 0.0,
        is_unity: true,
    };

    /// Exact algebraic bijection from normalized DF-II biquad `[b0, b1, b2, a1, a2]`
    /// (`H(z) = (b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2)`) onto Cytomic SVF coefficients.
    #[inline]
    pub fn from_biquad(sec: [f64; 5]) -> Self {
        let [b0, b1, b2, a1, a2] = sec;
        if (b0 - 1.0).abs() < 1e-14
            && b1.abs() < 1e-14
            && b2.abs() < 1e-14
            && a1.abs() < 1e-14
            && a2.abs() < 1e-14
        {
            return Self::UNITY;
        }
        let d_plus = (1.0 + a1 + a2).max(1e-15);
        let d_minus = (1.0 - a1 + a2).max(1e-15);
        let a1_svf = 0.25 * d_minus;
        let a2_svf = 0.25 * (d_plus * d_minus).sqrt();
        let a3_svf = 0.25 * d_plus;

        let m0 = (b0 - b1 + b2) / d_minus;
        let m2 = (b0 + b1 + b2) / d_plus - m0;
        let m1 = if a2_svf > 1e-15 {
            ((b0 - b2) - m0 * (1.0 - a2)) / (2.0 * a2_svf)
        } else {
            0.0
        };

        Self {
            a1: a1_svf,
            a2: a2_svf,
            a3: a3_svf,
            m0,
            m1,
            m2,
            is_unity: false,
        }
    }

    #[inline(always)]
    pub fn step(&self, x: f64, state: &mut [f64; 2]) -> f64 {
        if self.is_unity {
            return x;
        }
        let ic1 = state[0];
        let ic2 = state[1];
        let v3 = x - ic2;
        let v1 = self.a1 * ic1 + self.a2 * v3;
        let v2 = ic2 + self.a2 * ic1 + self.a3 * v3;
        state[0] = 2.0 * v1 - ic1;
        state[1] = 2.0 * v2 - ic2;
        self.m0 * x + self.m1 * v1 + self.m2 * v2
    }
}

/// Continuous-time Laplace 1st/2nd-order prototype section in normalized `s_bar = j * f / f0_hz`:
/// `H_sec(s_bar) = (n2 * s_bar^2 + n1 * s_bar + n0) / (d2 * s_bar^2 + d1 * s_bar + d0)`.
#[derive(Clone, Copy, Debug)]
pub struct AnalogSection2 {
    pub n2: f64,
    pub n1: f64,
    pub n0: f64,
    pub d2: f64,
    pub d1: f64,
    pub d0: f64,
    pub f0_hz: f64,
}

impl Default for AnalogSection2 {
    fn default() -> Self {
        Self::UNITY
    }
}

impl AnalogSection2 {
    pub const UNITY: Self = Self {
        n2: 0.0,
        n1: 0.0,
        n0: 1.0,
        d2: 0.0,
        d1: 0.0,
        d0: 1.0,
        f0_hz: 1000.0,
    };

    #[inline]
    pub fn eval_complex(&self, eval_freq_hz: f64) -> Complex64 {
        let omega = eval_freq_hz.max(0.0) / self.f0_hz.max(1.0);
        let omega_sq = omega * omega;
        let num = Complex64::new(self.n0 - self.n2 * omega_sq, self.n1 * omega);
        let den = Complex64::new(self.d0 - self.d2 * omega_sq, self.d1 * omega);
        if den.norm_sqr() < 1e-30 {
            Complex64::new(1.0, 0.0)
        } else {
            num / den
        }
    }

    #[inline]
    pub fn eval_mag_sq(&self, eval_freq_hz: f64) -> f64 {
        let omega = eval_freq_hz.max(0.0) / self.f0_hz.max(1.0);
        let omega_sq = omega * omega;
        let nr = self.n0 - self.n2 * omega_sq;
        let ni = self.n1 * omega;
        let dr = self.d0 - self.d2 * omega_sq;
        let di = self.d1 * omega;
        let den = dr * dr + di * di;
        if den < 1e-30 {
            1.0
        } else {
            (nr * nr + ni * ni) / den
        }
    }
}

/// Fractional-Slope Dual Biquad Cascade (`EQFilter` at `0x180221958`, `0x180112260`).
#[derive(Clone, Copy, Debug)]
pub struct FractionalSlopeBiquadBand {
    pub cascade_n1: [[f64; 5]; 8], // Lower integer order N1 (`rbx + 0xFB0`)
    pub cascade_n2: [[f64; 5]; 8], // Upper integer order N2 (`rbx + 0x10F0`)
    pub z_n1: [[[f64; 2]; 8]; 2],  // DF-II Transposed state [ch][sec][z1, z2]
    pub z_n2: [[[f64; 2]; 8]; 2],
    pub svf_n1: [CytomicSvfCoeffs; 8], // Cytomic Trapezoidal SVF sections for N1
    pub svf_n2: [CytomicSvfCoeffs; 8], // Cytomic Trapezoidal SVF sections for N2
    pub svf_z_n1: [[[f64; 2]; 8]; 2],  // SVF integrator state [ch][sec][ic1eq, ic2eq]
    pub svf_z_n2: [[[f64; 2]; 8]; 2],
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
            svf_n1: [CytomicSvfCoeffs::UNITY; 8],
            svf_n2: [CytomicSvfCoeffs::UNITY; 8],
            svf_z_n1: [[[0.0; 2]; 8]; 2],
            svf_z_n2: [[[0.0; 2]; 8]; 2],
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
        self.sync_svf_from_cascades();
    }

    /// Recomputes Cytomic Trapezoidal SVF parameters from `cascade_n1` and `cascade_n2`.
    #[inline]
    pub fn sync_svf_from_cascades(&mut self) {
        self.svf_n1 = [CytomicSvfCoeffs::UNITY; 8];
        self.svf_n2 = [CytomicSvfCoeffs::UNITY; 8];
        for s in 0..self.sections_n1.min(8) {
            self.svf_n1[s] = CytomicSvfCoeffs::from_biquad(self.cascade_n1[s]);
        }
        if self.alpha_frac > 1e-6 {
            for s in 0..self.sections_n2.min(8) {
                self.svf_n2[s] = CytomicSvfCoeffs::from_biquad(self.cascade_n2[s]);
            }
        } else {
            self.svf_n2 = self.svf_n1;
        }
    }

    /// Builds the exact continuous-time Laplace prototype dual-cascade `(sec_n1, len_n1, sec_n2, len_n2, alpha_frac)`
    /// shared between `eval_analog_s_plane_response` and `configure_shape_natural_phase` (`0x180110dc0`).
    pub fn build_analog_prototype_cascades(
        shape: FilterShape,
        freq_hz: f64,
        gain_db: f64,
        q_ui: f64,
        slope_db_oct: f64,
        gain_q_interaction: bool,
    ) -> ([AnalogSection2; 8], usize, [AnalogSection2; 8], usize, f64) {
        let f0 = freq_hz.clamp(10.0, 30_000.0);
        let slope = slope_db_oct.clamp(6.0, 96.0);
        let mut proto_n1 = [AnalogSection2::UNITY; 8];
        let mut proto_n2 = [AnalogSection2::UNITY; 8];

        match shape {
            FilterShape::LowCut | FilterShape::HighCut => {
                let is_hp = matches!(shape, FilterShape::LowCut);
                let s_norm = (slope / 6.0).max(1.0);
                let poles_low = (s_norm.floor() as usize).clamp(1, 16);
                let poles_high = (poles_low + 1).min(16);
                let alpha = if poles_low == poles_high {
                    0.0
                } else {
                    s_norm.fract()
                };
                let len1 = Self::fill_butterworth_analog_cut(&mut proto_n1, poles_low, f0, q_ui, is_hp);
                let len2 = if alpha > 1e-6 {
                    Self::fill_butterworth_analog_cut(&mut proto_n2, poles_high, f0, q_ui, is_hp)
                } else {
                    proto_n2 = proto_n1;
                    len1
                };
                (proto_n1, len1, proto_n2, len2, alpha)
            }
            FilterShape::FlatTilt => {
                if gain_db.abs() < 1e-9 {
                    return (proto_n1, 1, proto_n2, 1, 0.0);
                }
                let pivots = [62.5f64, 250.0, 1000.0, 4000.0];
                let sec_gain_db = gain_db * 0.5;
                let a = 10.0f64.powf(sec_gain_db / 40.0);
                let sqrt_a = a.sqrt();
                let q_sec = 0.38f64;
                for (i, &fp) in pivots.iter().enumerate() {
                    proto_n1[i] = AnalogSection2 {
                        n2: a,
                        n1: sqrt_a / q_sec,
                        n0: 1.0,
                        d2: 1.0,
                        d1: sqrt_a / q_sec,
                        d0: a,
                        f0_hz: fp,
                    };
                }
                let mut pivot_h = Complex64::new(1.0, 0.0);
                for s in &proto_n1[..4] {
                    pivot_h *= s.eval_complex(f0);
                }
                let norm_scale = 1.0 / pivot_h.norm().max(1e-12);
                proto_n1[0].n2 *= norm_scale;
                proto_n1[0].n1 *= norm_scale;
                proto_n1[0].n0 *= norm_scale;
                proto_n2 = proto_n1;
                (proto_n1, 4, proto_n2, 4, 0.0)
            }
            _ => {
                // Standard 12 dB/oct = 1 2nd-order section; 12..96 dB/oct cascades 1..8 sections;
                // 6..12 dB/oct scales Q smoothly while keeping 1 continuous-time biquad section.
                let sec_f = (slope / 12.0).clamp(0.5, 8.0);
                let (n1, n2, alpha, q_slope_scale) = if sec_f < 1.0 {
                    (1usize, 1usize, 0.0, sec_f)
                } else {
                    let lo = (sec_f.floor() as usize).clamp(1, 8);
                    let hi = (lo + 1).min(8);
                    let frac = if lo == hi { 0.0 } else { sec_f.fract() };
                    (lo, hi, frac, 1.0)
                };

                let gq_active = gain_q_interaction && shape.has_gain_q_interaction();
                let q_base = Self::compute_effective_q(q_ui, gain_db, gq_active) * q_slope_scale;

                let sec_lo = Self::make_single_analog_section(shape, f0, gain_db / (n1 as f64), q_base);
                for s in 0..n1 {
                    proto_n1[s] = sec_lo;
                }
                if alpha > 1e-6 {
                    let sec_hi =
                        Self::make_single_analog_section(shape, f0, gain_db / (n2 as f64), q_base);
                    for s in 0..n2 {
                        proto_n2[s] = sec_hi;
                    }
                } else {
                    proto_n2 = proto_n1;
                }
                (proto_n1, n1, proto_n2, n2, alpha)
            }
        }
    }

    #[inline]
    fn fill_butterworth_analog_cut(
        out: &mut [AnalogSection2; 8],
        num_poles: usize,
        f0_hz: f64,
        q_ui: f64,
        is_highpass: bool,
    ) -> usize {
        let poles = num_poles.clamp(1, 16);
        let pairs = poles / 2;
        let has_odd = (poles % 2) == 1;
        let odd_offset = (poles % 2) as f64;
        let mut idx = 0usize;

        for k in 0..pairs {
            let angle = PI * (2.0 * (k as f64) + 1.0 + odd_offset) / (2.0 * (poles as f64));
            let mut q_sec = 1.0 / (2.0 * angle.cos());
            if k + 1 == pairs {
                q_sec *= q_ui.clamp(0.1, 20.0);
            }
            q_sec = q_sec.max(0.05);
            out[idx] = if is_highpass {
                AnalogSection2 {
                    n2: 1.0,
                    n1: 0.0,
                    n0: 0.0,
                    d2: 1.0,
                    d1: 1.0 / q_sec,
                    d0: 1.0,
                    f0_hz,
                }
            } else {
                AnalogSection2 {
                    n2: 0.0,
                    n1: 0.0,
                    n0: 1.0,
                    d2: 1.0,
                    d1: 1.0 / q_sec,
                    d0: 1.0,
                    f0_hz,
                }
            };
            idx += 1;
        }

        if has_odd && idx < 8 {
            out[idx] = if is_highpass {
                AnalogSection2 {
                    n2: 0.0,
                    n1: 1.0,
                    n0: 0.0,
                    d2: 0.0,
                    d1: 1.0,
                    d0: 1.0,
                    f0_hz,
                }
            } else {
                AnalogSection2 {
                    n2: 0.0,
                    n1: 0.0,
                    n0: 1.0,
                    d2: 0.0,
                    d1: 1.0,
                    d0: 1.0,
                    f0_hz,
                }
            };
            idx += 1;
        }
        idx.max(1)
    }

    #[inline]
    fn make_single_analog_section(
        shape: FilterShape,
        f0_hz: f64,
        sec_gain_db: f64,
        q_eff: f64,
    ) -> AnalogSection2 {
        match shape {
            FilterShape::Bell => {
                if sec_gain_db.abs() < 1e-9 {
                    return AnalogSection2 {
                        f0_hz,
                        ..AnalogSection2::UNITY
                    };
                }
                let q = q_eff.max(0.025);
                let a = 10.0f64.powf(sec_gain_db / 40.0);
                AnalogSection2 {
                    n2: 1.0,
                    n1: a / q,
                    n0: 1.0,
                    d2: 1.0,
                    d1: 1.0 / (a * q),
                    d0: 1.0,
                    f0_hz,
                }
            }
            FilterShape::LowShelf => {
                if sec_gain_db.abs() < 1e-9 {
                    return AnalogSection2 {
                        f0_hz,
                        ..AnalogSection2::UNITY
                    };
                }
                let q = q_eff.clamp(0.125, 16.0);
                let a = 10.0f64.powf(sec_gain_db / 40.0);
                let sqrt_a = a.sqrt();
                AnalogSection2 {
                    n2: a,
                    n1: a * sqrt_a / q,
                    n0: a * a,
                    d2: a,
                    d1: sqrt_a / q,
                    d0: 1.0,
                    f0_hz,
                }
            }
            FilterShape::HighShelf => {
                if sec_gain_db.abs() < 1e-9 {
                    return AnalogSection2 {
                        f0_hz,
                        ..AnalogSection2::UNITY
                    };
                }
                let q = q_eff.clamp(0.125, 16.0);
                let a = 10.0f64.powf(sec_gain_db / 40.0);
                let sqrt_a = a.sqrt();
                AnalogSection2 {
                    n2: a * a,
                    n1: a * sqrt_a / q,
                    n0: a,
                    d2: 1.0,
                    d1: sqrt_a / q,
                    d0: a,
                    f0_hz,
                }
            }
            FilterShape::TiltShelf => {
                if sec_gain_db.abs() < 1e-9 {
                    return AnalogSection2 {
                        f0_hz,
                        ..AnalogSection2::UNITY
                    };
                }
                let q = q_eff.clamp(0.125, 8.0);
                let a = 10.0f64.powf(sec_gain_db / 40.0);
                let sqrt_a = a.sqrt();
                AnalogSection2 {
                    n2: a,
                    n1: sqrt_a / q,
                    n0: 1.0,
                    d2: 1.0,
                    d1: sqrt_a / q,
                    d0: a,
                    f0_hz,
                }
            }
            FilterShape::Notch => {
                let q = q_eff.clamp(0.05, 40.0);
                AnalogSection2 {
                    n2: 1.0,
                    n1: 0.0,
                    n0: 1.0,
                    d2: 1.0,
                    d1: 1.0 / q,
                    d0: 1.0,
                    f0_hz,
                }
            }
            FilterShape::BandPass => {
                let q = q_eff.clamp(0.05, 40.0);
                AnalogSection2 {
                    n2: 0.0,
                    n1: 1.0 / q,
                    n0: 0.0,
                    d2: 1.0,
                    d1: 1.0 / q,
                    d0: 1.0,
                    f0_hz,
                }
            }
            FilterShape::AllPass => {
                let q = q_eff.clamp(0.05, 40.0);
                AnalogSection2 {
                    n2: 1.0,
                    n1: -1.0 / q,
                    n0: 1.0,
                    d2: 1.0,
                    d1: 1.0 / q,
                    d0: 1.0,
                    f0_hz,
                }
            }
            _ => AnalogSection2 {
                f0_hz,
                ..AnalogSection2::UNITY
            },
        }
    }

    /// Configures an Orfanidis / Vicanek 5-Anchor Analog Magnitude & Phase Matched Biquad + Cytomic SVF
    /// cascade (`0x180110dc0` & `0x180111550`) operating at `oversampled_rate` (`2 * host_sample_rate`).
    pub fn configure_shape_natural_phase(
        &mut self,
        shape: FilterShape,
        freq_hz: f64,
        gain_db: f64,
        q_ui: f64,
        slope_db_oct: f64,
        oversampled_rate: f64,
        gain_q_interaction: bool,
    ) {
        let fs_os = oversampled_rate.max(16_000.0);
        let (proto_n1, len1, proto_n2, len2, alpha) = Self::build_analog_prototype_cascades(
            shape,
            freq_hz,
            gain_db,
            q_ui,
            slope_db_oct,
            gain_q_interaction,
        );

        self.sections_n1 = len1;
        self.sections_n2 = len2;
        self.alpha_frac = alpha;
        self.cascade_n1 = [Self::UNITY_SECTION; 8];
        self.cascade_n2 = [Self::UNITY_SECTION; 8];

        for s in 0..len1 {
            self.cascade_n1[s] = Self::discretize_analog_section_matched(&proto_n1[s], fs_os);
        }
        if alpha > 1e-6 {
            for s in 0..len2 {
                self.cascade_n2[s] = Self::discretize_analog_section_matched(&proto_n2[s], fs_os);
            }
        } else {
            self.cascade_n2 = self.cascade_n1;
        }
        self.sync_svf_from_cascades();
    }

    /// Discretizes a continuous-time `AnalogSection2` at `fs_os` via 5-Anchor Rational Squared-Magnitude
    /// Matching in `\phi = \sin^2(\pi f / fs_os)` + exact quadratic Fejér-Riesz minimum-phase spectral
    /// factorization, with Vicanek impulse-invariant pole + 3-anchor numerator fallback (`0x180110dc0`).
    pub fn discretize_analog_section_matched(sec: &AnalogSection2, fs_os: f64) -> [f64; 5] {
        // 1. Exact unity bypass
        if (sec.n2 - sec.d2).abs() < 1e-12
            && (sec.n1 - sec.d1).abs() < 1e-12
            && (sec.n0 - sec.d0).abs() < 1e-12
        {
            return Self::UNITY_SECTION;
        }

        let f_top = (0.44 * fs_os).min(20_200.0).max(3_500.0);
        let f0 = sec.f0_hz.clamp(10.0, 30_000.0);

        // 2. Exact AllPass (`|H(e^{jw})| == 1.0` with 2-point analog phase matching)
        if (sec.n2 - sec.d2).abs() < 1e-12
            && (sec.n0 - sec.d0).abs() < 1e-12
            && (sec.n1 + sec.d1).abs() < 1e-12
            && sec.d2 > 1e-12
        {
            let q = (sec.d2 / sec.d1.max(1e-6)).clamp(0.05, 40.0);
            if f0 < 0.46 * fs_os {
                let w0 = TAU * f0 / fs_os;
                let cos_w0 = w0.cos();
                let f1 = if f0 <= 0.65 * f_top {
                    (f0 * (1.0 + 0.5 / q.max(0.35))).clamp(f0 * 1.08, f_top)
                } else {
                    (f0 / (1.0 + 0.5 / q.max(0.35))).clamp(40.0, f0 * 0.92)
                };
                let w1 = TAU * f1 / fs_os;
                let h_a = sec.eval_complex(f1);
                // Continuous unwrapped half-phase `-0.5 * arg(H_a(j f1))` in `(0, pi)`
                let omega1 = f1 / f0;
                let phi_pole = (sec.d1 * omega1).atan2(sec.d0 - sec.d2 * omega1 * omega1);
                let _ = h_a;
                let eta = phi_pole.tan() * (w1.cos() - cos_w0) / w1.sin().max(1e-12);
                if eta > 1e-9 && eta.is_finite() {
                    let a2 = (1.0 - eta) / (1.0 + eta);
                    if a2.abs() < 0.999_999 {
                        let a1 = -cos_w0 * (1.0 + a2);
                        return [a2, a1, 1.0, a1, a2];
                    }
                }
            }
            let w0 = TAU * f0.min(0.49 * fs_os) / fs_os;
            return Self::biquad_allpass(w0, q);
        }

        // 3. 1st-Order Section (`n2 == 0 && d2 == 0`)
        if sec.n2.abs() < 1e-14 && sec.d2.abs() < 1e-14 {
            return Self::discretize_first_order_matched(sec, fs_os, f_top);
        }

        // 4. 2nd-Order HighPass (`LowCut`: `n0 == 0 && n1 == 0 && n2 != 0`) -> exact double zero at DC
        if sec.n0.abs() < 1e-14 && sec.n1.abs() < 1e-14 && sec.n2.abs() > 1e-14 {
            if let Some(biq) = Self::discretize_highpass2_matched(sec, fs_os, f_top) {
                return biq;
            }
        }

        // 5. 2nd-Order Notch (`n1 == 0 && n2 == n0 > 0`) -> exact transmission zero on unit circle
        if sec.n1.abs() < 1e-14 && (sec.n2 - sec.n0).abs() < 1e-12 && sec.n0 > 1e-12 {
            if let Some(biq) = Self::discretize_notch2_matched(sec, fs_os, f_top) {
                return biq;
            }
        }

        // 6. General 2nd-Order 5-Anchor Rational Fit (`Bell`, `LowShelf`, `HighShelf`, `HighCut`, `BandPass`, `TiltShelf`, `FlatTilt`)
        if let Some(biq) = Self::discretize_general2_five_anchor(sec, fs_os, f_top) {
            return biq;
        }

        // 7. Guaranteed-stable Vicanek Impulse-Invariant Poles + 3-Anchor Numerator fallback
        Self::discretize_vicanek_fallback(sec, fs_os, f_top)
    }

    #[inline]
    fn phi_of_freq(freq_hz: f64, fs_os: f64) -> f64 {
        let s = (PI * freq_hz / fs_os).sin();
        s * s
    }

    /// Exact minimum-phase quadratic Fejér-Riesz spectral factorization of `P(\phi) = c0 + c1 \phi + c2 \phi^2`
    /// (`\phi = \sin^2(\omega / 2)`) into `p0 + p1 z^-1 + p2 z^-2` with `|p(e^{j\omega})|^2 == P(\phi)`.
    #[inline]
    fn factor_fejer_riesz_quadratic(c0: f64, c1: f64, c2: f64, strict_den: bool) -> Option<[f64; 3]> {
        let tol = if strict_den { 1e-11 } else { -1e-12 };
        let v1 = c0 + c1 + c2;
        if c0 < tol || v1 < tol {
            return None;
        }
        let g0 = c0.max(0.0).sqrt();
        let g1 = v1.max(0.0).sqrt();
        let s_plus = 0.5 * (g0 + g1);
        let p1 = 0.5 * (g0 - g1);
        let disc = s_plus * s_plus - 0.25 * c2;
        if disc < tol {
            return None;
        }
        let root = disc.max(0.0).sqrt();
        let p0 = 0.5 * (s_plus + root);
        let p2 = 0.5 * (s_plus - root);
        if strict_den {
            if p0 <= 1e-12 || (p2 / p0).abs() >= 0.999_999_5 {
                return None;
            }
        }
        Some([p0, p1, p2])
    }

    fn discretize_first_order_matched(sec: &AnalogSection2, fs_os: f64, f_top: f64) -> [f64; 5] {
        let f0 = sec.f0_hz.clamp(10.0, 30_000.0);
        let f1 = if f0 < 0.6 * f_top {
            f0.max(20.0)
        } else {
            0.45 * f_top
        };
        let f2 = f_top;
        let phi1 = Self::phi_of_freq(f1, fs_os);
        let phi2 = Self::phi_of_freq(f2, fs_os);
        let m1 = sec.eval_mag_sq(f1);
        let m2 = sec.eval_mag_sq(f2);

        if sec.n0.abs() < 1e-14 {
            // 1st-order HighPass: `B0 = 0`, `M_i (1 + A1 phi_i) = B1 phi_i`
            let denom = phi1 * phi2 * (m1 - m2);
            if denom.abs() > 1e-18 {
                let a1_cap = (m2 * phi1 - m1 * phi2) / denom;
                let b1_cap = m1 * (1.0 + a1_cap * phi1) / phi1.max(1e-18);
                if 1.0 + a1_cap > 1e-9 && b1_cap > 0.0 {
                    let r1 = (1.0 + a1_cap).sqrt();
                    let q1 = b1_cap.sqrt();
                    let a1 = (1.0 - r1) / (1.0 + r1);
                    let b0 = q1 / (1.0 + r1);
                    return [b0, -b0, 0.0, a1, 0.0];
                }
            }
            let w0 = TAU * f0.min(0.49 * fs_os) / fs_os;
            return Self::first_order_cut(w0, true);
        }

        let b0_cap = sec.eval_mag_sq(0.0);
        let diff_m = m2 - m1;
        if diff_m.abs() > 1e-15 && phi1 > 1e-15 && phi2 > phi1 + 1e-12 {
            let a1_cap = ((m1 - b0_cap) / phi1 - (m2 - b0_cap) / phi2) / diff_m;
            let b1_cap = (m1 * (1.0 + a1_cap * phi1) - b0_cap) / phi1;
            if 1.0 + a1_cap > 1e-9 && b0_cap + b1_cap >= 0.0 {
                let r1 = (1.0 + a1_cap).sqrt();
                let q0 = b0_cap.max(0.0).sqrt();
                let q1 = (b0_cap + b1_cap).max(0.0).sqrt();
                let a1 = (1.0 - r1) / (1.0 + r1);
                let b0 = (q0 + q1) / (1.0 + r1);
                let b1 = (q0 - q1) / (1.0 + r1);
                return [b0, b1, 0.0, a1, 0.0];
            }
        }
        let w0 = TAU * f0.min(0.49 * fs_os) / fs_os;
        Self::first_order_cut(w0, false)
    }

    fn discretize_highpass2_matched(
        sec: &AnalogSection2,
        fs_os: f64,
        f_top: f64,
    ) -> Option<[f64; 5]> {
        let f0 = sec.f0_hz.clamp(10.0, 30_000.0);
        let anchors = if f0 <= 0.72 * f_top {
            [
                (0.55 * f0).clamp(15.0, 0.30 * f_top),
                f0.max(35.0),
                f_top,
            ]
        } else {
            [0.25 * f_top, 0.62 * f_top, f_top]
        };

        let mut mat = [[0.0f64; 4]; 3];
        for (i, &fi) in anchors.iter().enumerate() {
            let phi = Self::phi_of_freq(fi, fs_os);
            let phi2 = phi * phi;
            let m = sec.eval_mag_sq(fi);
            mat[i] = [phi2, -m * phi, -m * phi2, m];
        }
        let [b2_cap, a1_cap, a2_cap] = Self::solve_linear_3x3(mat)?;
        if b2_cap <= 1e-18 {
            return None;
        }
        let [p0, p1, p2] = Self::factor_fejer_riesz_quadratic(1.0, a1_cap, a2_cap, true)?;
        let b0 = b2_cap.sqrt() / (4.0 * p0);
        Some([b0, -2.0 * b0, b0, p1 / p0, p2 / p0])
    }

    fn discretize_notch2_matched(
        sec: &AnalogSection2,
        fs_os: f64,
        f_top: f64,
    ) -> Option<[f64; 5]> {
        let f0 = sec.f0_hz.clamp(10.0, 0.48 * fs_os);
        let w0 = TAU * f0 / fs_os;
        let phi0 = Self::phi_of_freq(f0, fs_os).max(1e-15);
        let q = (sec.d0.max(1e-6).sqrt() / sec.d1.max(1e-6)).clamp(0.05, 40.0);
        let half_bw = 0.5 / q;

        let (f1, f2) = if f0 <= 0.75 * f_top {
            let lo = (f0 / (1.0 + half_bw).max(1.06)).clamp(15.0, f0 * 0.94);
            let hi = (f0 * (1.0 + half_bw).max(1.06)).clamp(f0 * 1.06, f_top);
            (lo, hi)
        } else {
            let hi = (f0 / (1.0 + 0.35 * half_bw).max(1.05)).clamp(80.0, f_top.min(f0 * 0.95));
            let lo = (0.45 * hi).max(20.0);
            (lo, hi)
        };

        let phi1 = Self::phi_of_freq(f1, fs_os);
        let phi2 = Self::phi_of_freq(f2, fs_os);
        let n1 = (1.0 - phi1 / phi0).powi(2);
        let n2 = (1.0 - phi2 / phi0).powi(2);
        let m1 = sec.eval_mag_sq(f1).max(1e-18);
        let m2 = sec.eval_mag_sq(f2).max(1e-18);
        let rhs1 = n1 / m1 - 1.0;
        let rhs2 = n2 / m2 - 1.0;

        let det = phi1 * phi2 * (phi2 - phi1);
        if det.abs() < 1e-20 {
            return None;
        }
        let a1_cap = (rhs1 * phi2 * phi2 - rhs2 * phi1 * phi1) / det;
        let a2_cap = (phi1 * rhs2 - phi2 * rhs1) / det;
        let [p0, p1, p2] = Self::factor_fejer_riesz_quadratic(1.0, a1_cap, a2_cap, true)?;
        let b0 = 1.0 / (4.0 * phi0 * p0);
        let b1 = -2.0 * w0.cos() * b0;
        Some([b0, b1, b0, p1 / p0, p2 / p0])
    }

    fn discretize_general2_five_anchor(
        sec: &AnalogSection2,
        fs_os: f64,
        f_top: f64,
    ) -> Option<[f64; 5]> {
        let f0 = sec.f0_hz.clamp(10.0, 30_000.0);
        let anchors = if f0 <= 3_500.0 {
            let f1 = (0.45 * f0).max(12.0);
            let f2 = f0.max(f1 + 10.0);
            let f3 = (f2 * 2.4).clamp(f2 + 40.0, 0.55 * f_top);
            [f1, f2, f3, f_top]
        } else if f0 <= 0.90 * f_top {
            let f1 = (0.35 * f0).max(40.0);
            let f2 = (0.72 * f0).max(f1 + 50.0);
            let f3 = f0.max(f2 + 50.0);
            [f1, f2, f3, f_top]
        } else {
            [0.18 * f_top, 0.48 * f_top, 0.76 * f_top, f_top]
        };

        let is_bp_dc_zero = sec.n0.abs() < 1e-14 && sec.n2.abs() < 1e-14 && sec.n1.abs() > 1e-14;
        let b0_cap = if is_bp_dc_zero {
            0.0
        } else {
            sec.eval_mag_sq(0.0)
        };

        let mut mat = [[0.0f64; 5]; 4];
        for (i, &fi) in anchors.iter().enumerate() {
            let phi = Self::phi_of_freq(fi, fs_os);
            let phi2 = phi * phi;
            let m = sec.eval_mag_sq(fi);
            mat[i] = [phi, phi2, -m * phi, -m * phi2, m - b0_cap];
        }

        let [b1_cap, b2_cap, a1_cap, a2_cap] = Self::solve_linear_4x4(mat)?;
        let [p0, p1, p2] = Self::factor_fejer_riesz_quadratic(1.0, a1_cap, a2_cap, true)?;

        let [q0, q1, q2] = if is_bp_dc_zero {
            // Exact DC zero `(1 - z^-1)(c0 + c1 z^-1)` where `|B(e^{jw})|^2 = B1 \phi + B2 \phi^2`
            if b1_cap < 0.0 || (b1_cap + b2_cap) < -1e-12 {
                return None;
            }
            let u0 = 0.5 * b1_cap.max(0.0).sqrt();
            let u1 = 0.5 * (b1_cap + b2_cap).max(0.0).sqrt();
            let c0 = 0.5 * (u0 + u1);
            let c1 = 0.5 * (u0 - u1);
            [c0, c1 - c0, -c1]
        } else {
            Self::factor_fejer_riesz_quadratic(b0_cap, b1_cap, b2_cap, false)?
        };

        Some([q0 / p0, q1 / p0, q2 / p0, p1 / p0, p2 / p0])
    }

    /// Vicanek impulse-invariant poles + 3-anchor matched numerator (`0, f0, f_top`).
    fn discretize_vicanek_fallback(sec: &AnalogSection2, fs_os: f64, f_top: f64) -> [f64; 5] {
        if sec.d2.abs() < 1e-14 {
            return Self::discretize_first_order_matched(sec, fs_os, f_top);
        }
        let f0 = sec.f0_hz.clamp(10.0, 30_000.0);
        let w0 = TAU * f0 / fs_os;
        let p1_a = sec.d1 / sec.d2;
        let p0_a = (sec.d0 / sec.d2).max(1e-12);
        let disc_s = p1_a * p1_a - 4.0 * p0_a;

        let (a1, a2) = if disc_s < 0.0 {
            let sigma = 0.5 * p1_a.max(1e-6);
            let wd = 0.5 * (-disc_s).sqrt();
            let r = (-sigma * w0).exp().min(0.999_99);
            let theta = wd * w0;
            (-2.0 * r * theta.cos(), r * r)
        } else {
            let root = disc_s.sqrt();
            let lam1 = (0.5 * (p1_a + root)).max(1e-6);
            let lam2 = (0.5 * (p1_a - root)).max(1e-6);
            let r1 = (-lam1 * w0).exp().min(0.999_99);
            let r2 = (-lam2 * w0).exp().min(0.999_99);
            (-(r1 + r2), r1 * r2)
        };

        let d0_sq = (1.0 + a1 + a2).powi(2).max(1e-24);
        let a1_cap = (-4.0 * a1 * (1.0 + a2) - 16.0 * a2) / d0_sq;
        let a2_cap = (16.0 * a2) / d0_sq;

        let is_bp_dc_zero = sec.n0.abs() < 1e-14 && sec.n2.abs() < 1e-14 && sec.n1.abs() > 1e-14;
        let b0_cap = if is_bp_dc_zero {
            0.0
        } else {
            sec.eval_mag_sq(0.0)
        };

        let f_mid = if f0 < 0.80 * f_top {
            f0.max(25.0)
        } else {
            0.50 * f_top
        };
        let phi1 = Self::phi_of_freq(f_mid, fs_os);
        let phi2 = Self::phi_of_freq(f_top, fs_os);
        let rhs1 = sec.eval_mag_sq(f_mid) * (1.0 + a1_cap * phi1 + a2_cap * phi1 * phi1) - b0_cap;
        let rhs2 = sec.eval_mag_sq(f_top) * (1.0 + a1_cap * phi2 + a2_cap * phi2 * phi2) - b0_cap;

        let det = phi1 * phi2 * (phi2 - phi1);
        if det.abs() > 1e-20 {
            let b1_cap = (rhs1 * phi2 * phi2 - rhs2 * phi1 * phi1) / det;
            let b2_cap = (phi1 * rhs2 - phi2 * rhs1) / det;
            let p0 = 1.0 / (1.0 + a1 + a2).abs().max(1e-12);
            let _ = p0;
            let den_scale = (1.0 + a1 + a2).abs();
            if is_bp_dc_zero && b1_cap >= 0.0 && (b1_cap + b2_cap) >= 0.0 {
                let u0 = 0.5 * b1_cap.sqrt();
                let u1 = 0.5 * (b1_cap + b2_cap).sqrt();
                let c0 = 0.5 * (u0 + u1) * den_scale;
                let c1 = 0.5 * (u0 - u1) * den_scale;
                return [c0, c1 - c0, -c1, a1, a2];
            }
            if let Some([q0, q1, q2]) =
                Self::factor_fejer_riesz_quadratic(b0_cap, b1_cap, b2_cap, false)
            {
                return [q0 * den_scale, q1 * den_scale, q2 * den_scale, a1, a2];
            }
        }

        // Ultimate closed-form prewarped fallback
        let g = (b0_cap.max(0.0).sqrt()) * (1.0 + a1 + a2);
        [g, 0.0, 0.0, a1, a2]
    }

    #[inline]
    fn solve_linear_3x3(mut a: [[f64; 4]; 3]) -> Option<[f64; 3]> {
        for col in 0..3 {
            let mut piv = col;
            let mut max_v = a[col][col].abs();
            for row in (col + 1)..3 {
                let v = a[row][col].abs();
                if v > max_v {
                    max_v = v;
                    piv = row;
                }
            }
            if max_v < 1e-13 {
                return None;
            }
            if piv != col {
                a.swap(col, piv);
            }
            let inv = 1.0 / a[col][col];
            for j in col..4 {
                a[col][j] *= inv;
            }
            for row in 0..3 {
                if row != col {
                    let f = a[row][col];
                    for j in col..4 {
                        a[row][j] -= f * a[col][j];
                    }
                }
            }
        }
        Some([a[0][3], a[1][3], a[2][3]])
    }

    #[inline]
    fn solve_linear_4x4(mut a: [[f64; 5]; 4]) -> Option<[f64; 4]> {
        for col in 0..4 {
            let mut piv = col;
            let mut max_v = a[col][col].abs();
            for row in (col + 1)..4 {
                let v = a[row][col].abs();
                if v > max_v {
                    max_v = v;
                    piv = row;
                }
            }
            if max_v < 1e-13 {
                return None;
            }
            if piv != col {
                a.swap(col, piv);
            }
            let inv = 1.0 / a[col][col];
            for j in col..5 {
                a[col][j] *= inv;
            }
            for row in 0..4 {
                if row != col {
                    let f = a[row][col];
                    for j in col..5 {
                        a[row][j] -= f * a[col][j];
                    }
                }
            }
        }
        Some([a[0][4], a[1][4], a[2][4], a[3][4]])
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
        self.svf_z_n1 = [[[0.0; 2]; 8]; 2];
        self.svf_z_n2 = [[[0.0; 2]; 8]; 2];
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

    /// Processes a single sample through the Cytomic Trapezoidal Integrator SVF dual-cascade (`0x180111550`).
    #[inline(always)]
    pub fn process_sample_svf(&mut self, ch: usize, x: f64) -> f64 {
        let mut y1 = x;
        for s in 0..self.sections_n1 {
            y1 = self.svf_n1[s].step(y1, &mut self.svf_z_n1[ch][s]);
        }
        if self.alpha_frac <= 1e-6 {
            return y1;
        }
        let mut y2 = x;
        for s in 0..self.sections_n2 {
            y2 = self.svf_n2[s].step(y2, &mut self.svf_z_n2[ch][s]);
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
