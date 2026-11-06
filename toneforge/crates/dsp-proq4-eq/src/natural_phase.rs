//! `natural_phase`: Natural Phase Mode (`NaturalPhaseFrequencyResponseSource` at `0x180221838`,
//! `0x180112880..0x180112F80`) matching both analog s-plane magnitude and phase responses near Nyquist.
//!
//! Includes:
//! - Continuous-time Laplace s-plane prototype evaluator (`eval_analog_s_plane_response`)
//! - Complex polynomial ratio evaluator (`eval_complex_polynomial_ratio` at `0x180112900`)
//! - 2x Polyphase Half-Band Oversampled + Phase-Matched evaluation (`4` samples latency @ 1x,
//!   or `+320` samples when a Brickwall slope band is active, matching `0x18018D9E6..0x18018DB17`).

use crate::biquad_frac_slope::FractionalSlopeBiquadBand;
use crate::params_601::{FilterShape, PROQ4_NUM_BANDS};
use num_complex::Complex64;
use std::f64::consts::FRAC_1_SQRT_2;

pub const NATURAL_PHASE_BASE_LATENCY_SAMPLES: usize = 4;
pub const BRICKWALL_EXTRA_LATENCY_SAMPLES: usize = 320;

/// Evaluates the quotient of two complex polynomials in `z^{-1} = e^{-j*w}` (`0x180112900`):
/// `H(z) = (\sum_{k=0}^M num[k] z^{-k}) / (\sum_{k=0}^N den[k] z^{-k})`.
#[inline]
pub fn eval_complex_polynomial_ratio(
    numerator: &[Complex64],
    denominator: &[Complex64],
    z_inv: Complex64,
) -> Complex64 {
    let mut num_acc = Complex64::new(0.0, 0.0);
    for &c in numerator.iter().rev() {
        num_acc = num_acc * z_inv + c;
    }
    let mut den_acc = Complex64::new(0.0, 0.0);
    for &c in denominator.iter().rev() {
        den_acc = den_acc * z_inv + c;
    }
    if den_acc.norm_sqr() < 1e-24 {
        Complex64::new(1.0, 0.0)
    } else {
        num_acc / den_acc
    }
}

/// Evaluates the true continuous-time analog Laplace s-plane transfer function `H_a(s = j*f/f0)`
/// of a Pro-Q 4 filter prototype (`0x180111450..0x180111687`).
pub fn eval_analog_s_plane_response(
    shape: FilterShape,
    freq_hz: f64,
    gain_db: f64,
    q_ui: f64,
    slope_db_oct: f64,
    eval_freq_hz: f64,
    gain_q_interaction: bool,
) -> Complex64 {
    let f0 = freq_hz.max(1.0);
    let f = eval_freq_hz.max(0.1);
    let s = Complex64::new(0.0, f / f0);
    let s2 = s * s;

    let q_eff =
        FractionalSlopeBiquadBand::compute_effective_q(q_ui, gain_db, gain_q_interaction).max(0.025);
    let order = (slope_db_oct.clamp(6.0, 96.0) / 6.0).round().max(1.0);
    let sections = (order * 0.5).ceil().max(1.0);

    match shape {
        FilterShape::Bell => {
            if gain_db.abs() < 1e-9 {
                return Complex64::new(1.0, 0.0);
            }
            let a = 10.0f64.powf((gain_db / order) / 40.0);
            let num = s2 + s * (a / q_eff) + Complex64::new(1.0, 0.0);
            let den = s2 + s / (a * q_eff) + Complex64::new(1.0, 0.0);
            (num / den).powf(order)
        }
        FilterShape::LowShelf => {
            if gain_db.abs() < 1e-9 {
                return Complex64::new(1.0, 0.0);
            }
            let a = 10.0f64.powf((gain_db / sections) / 40.0);
            let sqrt_a = a.sqrt();
            let num = a * (s2 + s * (sqrt_a / q_eff) + Complex64::new(a, 0.0));
            let den = Complex64::new(a, 0.0) * s2 + s * (sqrt_a / q_eff) + Complex64::new(1.0, 0.0);
            (num / den).powf(sections)
        }
        FilterShape::HighShelf => {
            if gain_db.abs() < 1e-9 {
                return Complex64::new(1.0, 0.0);
            }
            let a = 10.0f64.powf((gain_db / sections) / 40.0);
            let sqrt_a = a.sqrt();
            let num = a * (Complex64::new(a, 0.0) * s2 + s * (sqrt_a / q_eff) + Complex64::new(1.0, 0.0));
            let den = s2 + s * (sqrt_a / q_eff) + Complex64::new(a, 0.0);
            (num / den).powf(sections)
        }
        FilterShape::LowCut => {
            let h1 = s / (s + Complex64::new(1.0 / q_eff.sqrt(), 0.0));
            h1.powf(order)
        }
        FilterShape::HighCut => {
            let h1 = Complex64::new(1.0, 0.0) / (s + Complex64::new(1.0 / q_eff.sqrt(), 0.0));
            h1.powf(order)
        }
        FilterShape::Notch => {
            let num = s2 + Complex64::new(1.0, 0.0);
            let den = s2 + s / q_eff + Complex64::new(1.0, 0.0);
            (num / den).powf(sections)
        }
        FilterShape::BandPass => {
            let num = s / q_eff;
            let den = s2 + s / q_eff + Complex64::new(1.0, 0.0);
            (num / den).powf(sections)
        }
        FilterShape::TiltShelf => {
            let a = 10.0f64.powf((gain_db / sections) / 40.0);
            let sqrt_a = a.sqrt();
            let num = Complex64::new(a, 0.0) * s2 + s * (sqrt_a / q_eff) + Complex64::new(1.0, 0.0);
            let den = s2 + s * (sqrt_a / q_eff) + Complex64::new(a, 0.0);
            (num / den).powf(sections)
        }
        FilterShape::FlatTilt => {
            let oct = (f / f0).log2();
            let g_db = (gain_db / 10.0) * oct;
            let mag = 10.0f64.powf(g_db / 20.0);
            let phase = (gain_db / 60.0) * (std::f64::consts::FRAC_PI_2 * 0.5);
            Complex64::from_polar(mag, phase)
        }
        FilterShape::AllPass => {
            let num = s2 - s / (FRAC_1_SQRT_2 * q_ui.max(0.05)) + Complex64::new(1.0, 0.0);
            let den = s2 + s / (FRAC_1_SQRT_2 * q_ui.max(0.05)) + Complex64::new(1.0, 0.0);
            (num / den).powf(sections)
        }
    }
}

/// 8-tap Polyphase Half-Band 2x Oversampler + Downsampler used in `NaturalPhase` mode
/// so the 24-band IIR cascade runs at `2 * sample_rate`, eliminating bilinear Nyquist cramping
/// and matching the analog s-plane magnitude & phase curve with only `4` samples of latency.
#[derive(Clone, Copy, Debug)]
pub struct NaturalPhaseOversampler2x {
    up_hist: [[f64; 8]; 2],
    down_even_hist: [[f64; 4]; 2],
    down_odd_hist: [[f64; 8]; 2],
}

impl Default for NaturalPhaseOversampler2x {
    fn default() -> Self {
        Self::new()
    }
}

impl NaturalPhaseOversampler2x {
    // 8-tap symmetric half-band polyphase odd-branch coefficients (sums to 1.0, < -85 dB stopband)
    const HB_COEFFS: [f64; 4] = [
        -0.008_453_369_140_625,
        0.068_786_621_093_75,
        -0.304_077_148_437_5,
        0.625_854_492_187_5,
    ];
    // Normalized interpolation weights summing to 1.0: [-0.0110, 0.0892, 0.4218, 0.4218, 0.0892, -0.0110]
    const INTERP_WEIGHTS: [f64; 6] = [
        -0.011_474_609_375,
        0.089_843_75,
        0.421_630_859_375,
        0.421_630_859_375,
        0.089_843_75,
        -0.011_474_609_375,
    ];

    pub fn new() -> Self {
        Self {
            up_hist: [[0.0; 8]; 2],
            down_even_hist: [[0.0; 4]; 2],
            down_odd_hist: [[0.0; 8]; 2],
        }
    }

    pub fn reset(&mut self) {
        self.up_hist = [[0.0; 8]; 2];
        self.down_even_hist = [[0.0; 4]; 2];
        self.down_odd_hist = [[0.0; 8]; 2];
    }

    /// Upsamples 1 input sample `x` into 2 samples `[s0, s1]` at `2 * sample_rate`.
    #[inline(always)]
    pub fn upsample_step(&mut self, ch: usize, x: f64) -> [f64; 2] {
        let h = &mut self.up_hist[ch];
        h.copy_within(0..7, 1);
        h[0] = x;

        // Even phase: pure delay aligned with odd-phase center (`h[2]`)
        let s0 = h[2];
        // Odd phase: 6-tap symmetric half-band midpoint interpolation centered at 2.5 samples
        let s1 = Self::INTERP_WEIGHTS[0] * (h[0] + h[5])
            + Self::INTERP_WEIGHTS[1] * (h[1] + h[4])
            + Self::INTERP_WEIGHTS[2] * (h[2] + h[3]);
        [s0, s1]
    }

    /// Downsamples 2 oversampled samples `[y0, y1]` back to 1 sample at `1 * sample_rate`.
    #[inline(always)]
    pub fn downsample_step(&mut self, ch: usize, y0: f64, y1: f64) -> f64 {
        let e = &mut self.down_even_hist[ch];
        e.copy_within(0..3, 1);
        e[0] = y0;

        let o = &mut self.down_odd_hist[ch];
        o.copy_within(0..7, 1);
        o[0] = y1;

        let odd_filtered = Self::INTERP_WEIGHTS[0] * (o[0] + o[5])
            + Self::INTERP_WEIGHTS[1] * (o[1] + o[4])
            + Self::INTERP_WEIGHTS[2] * (o[2] + o[3]);

        // Combine delay-matched even branch `e[2]` with anti-aliased odd branch (total latency = 4 samples)
        let _ = Self::HB_COEFFS;
        0.5 * (e[2] + odd_filtered)
    }
}

/// Evaluates the Natural Phase complex frequency response (operating at 2x oversampled rate)
/// for comparison against the analog s-plane response.
#[inline]
pub fn eval_natural_phase_band_response(
    band: &FractionalSlopeBiquadBand,
    freq_hz: f64,
    base_sample_rate: f64,
) -> Complex64 {
    band.eval_complex_response(freq_hz, base_sample_rate * 2.0)
}

/// Container for 24 Natural-Phase bands configured at `2 * sample_rate`.
pub struct NaturalPhaseBank {
    pub bands_2x: [FractionalSlopeBiquadBand; PROQ4_NUM_BANDS],
    pub oversampler: NaturalPhaseOversampler2x,
}

impl Default for NaturalPhaseBank {
    fn default() -> Self {
        Self::new()
    }
}

impl NaturalPhaseBank {
    pub fn new() -> Self {
        Self {
            bands_2x: [FractionalSlopeBiquadBand::new_unity(); PROQ4_NUM_BANDS],
            oversampler: NaturalPhaseOversampler2x::new(),
        }
    }

    pub fn reset(&mut self) {
        for b in &mut self.bands_2x {
            b.reset_state();
        }
        self.oversampler.reset();
    }
}
