//! `natural_phase`: Natural Phase Mode (`NaturalPhaseFrequencyResponseSource` at `0x180221838`,
//! `0x180112880..0x180112F80`) matching both analog s-plane magnitude and phase responses near Nyquist.
//!
//! Includes:
//! - FabFilter Pro-Q 4 Jacobi-Elliptic Polyphase Allpass Half-Band IIR (`sr::IIRConverterStyle`,
//!   `0x18011d4e0` / `0x18011d830`) with `> 113 dB` stopband rejection and exact allpass flat cascade
//! - Continuous-time Laplace s-plane prototype evaluator (`eval_analog_s_plane_response`)
//! - Complex polynomial ratio evaluator (`eval_complex_polynomial_ratio` at `0x180112900`)
//! - `2x` (`NaturalPhaseOversampler2x`) and `4x` (`JacobiEllipticOversampler4x`) zero-allocation IIR oversamplers

use crate::biquad_frac_slope::FractionalSlopeBiquadBand;
use crate::params_601::{FilterShape, PROQ4_NUM_BANDS};
use num_complex::Complex64;
use std::f64::consts::PI;

pub const NATURAL_PHASE_BASE_LATENCY_SAMPLES: usize = 4;
pub const BRICKWALL_EXTRA_LATENCY_SAMPLES: usize = 320;

pub const MAX_HALFBAND_POLES_PER_BRANCH: usize = 6;
pub const PROQ4_STAGE1_OMEGA_P: f64 = 1.382_300_7; // 0.44 * pi (21.12 kHz passband @ 48 kHz)
pub const PROQ4_STAGE2_OMEGA_P: f64 = 1.036_725_6; // 0.33 * pi

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

/// Evaluates the true continuous-time analog Laplace s-plane transfer function `H_a(s = j*2*pi*f)`
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
    let f = eval_freq_hz.max(0.1);
    let (proto_n1, len1, proto_n2, len2, alpha) =
        FractionalSlopeBiquadBand::build_analog_prototype_cascades(
            shape,
            freq_hz,
            gain_db,
            q_ui,
            slope_db_oct,
            gain_q_interaction,
        );

    let mut h1 = Complex64::new(1.0, 0.0);
    for s in 0..len1 {
        h1 *= proto_n1[s].eval_complex(f);
    }
    if alpha <= 1e-6 {
        return h1;
    }
    let mut h2 = Complex64::new(1.0, 0.0);
    for s in 0..len2 {
        h2 *= proto_n2[s].eval_complex(f);
    }
    h1 * (1.0 - alpha) + h2 * alpha
}

/// Convenience helper returning `20 * log10(|H_a(j * 2 * pi * eval_freq_hz)|)` in dB.
#[inline]
pub fn analog_prototype_mag_db(
    shape: FilterShape,
    freq_hz: f64,
    gain_db: f64,
    q_ui: f64,
    slope_db_oct: f64,
    eval_freq_hz: f64,
    gain_q_interaction: bool,
) -> f64 {
    let mag = eval_analog_s_plane_response(
        shape,
        freq_hz,
        gain_db,
        q_ui,
        slope_db_oct,
        eval_freq_hz,
        gain_q_interaction,
    )
    .norm()
    .max(1e-15);
    20.0 * mag.log10()
}

/// Pole specification for a Jacobi-Elliptic Polyphase Allpass Half-Band IIR (`0x18011d4e0`):
/// `H(z) = 0.5 * [A_0(z^2) + z^{-1} A_1(z^2)]`, where each `A_m(w)` is a cascade of
/// first-order allpass sections `(c_k + w^{-1}) / (1 + c_k w^{-1})`.
#[derive(Clone, Copy, Debug)]
pub struct JacobiEllipticHalfbandSpec {
    pub order_n: usize,
    pub omega_p: f64,
    pub branch0_poles: [f64; MAX_HALFBAND_POLES_PER_BRANCH],
    pub branch0_len: usize,
    pub branch1_poles: [f64; MAX_HALFBAND_POLES_PER_BRANCH],
    pub branch1_len: usize,
}

impl JacobiEllipticHalfbandSpec {
    /// Evaluates the exact complex frequency response `H(e^{j * omega_rad})` (`omega_rad in [0, pi]`).
    #[inline]
    pub fn eval_frequency_response(&self, omega_rad: f64) -> Complex64 {
        let z_inv = Complex64::from_polar(1.0, -omega_rad);
        let z_inv2 = Complex64::from_polar(1.0, -2.0 * omega_rad);

        let mut a0 = Complex64::new(1.0, 0.0);
        for i in 0..self.branch0_len {
            let c = self.branch0_poles[i];
            a0 *= (Complex64::new(c, 0.0) + z_inv2) / (Complex64::new(1.0, 0.0) + z_inv2 * c);
        }

        let mut a1 = Complex64::new(1.0, 0.0);
        for i in 0..self.branch1_len {
            let c = self.branch1_poles[i];
            a1 *= (Complex64::new(c, 0.0) + z_inv2) / (Complex64::new(1.0, 0.0) + z_inv2 * c);
        }

        0.5 * (a0 + z_inv * a1)
    }
}

/// Exact AGM + backward Landen Jacobi-Elliptic half-band allpass pole solver (`0x18011d4e0`).
pub fn design_jacobi_elliptic_halfband(order_n: usize, omega_p: f64) -> JacobiEllipticHalfbandSpec {
    let n_odd = (order_n | 1).clamp(3, 4 * MAX_HALFBAND_POLES_PER_BRANCH + 1);
    let num_poles = (n_odd - 1) / 2;
    let wp = omega_p.clamp(0.05 * PI, 0.495 * PI);

    let t = (0.5 * wp).tan();
    let d4 = 1.0 / (t * t);
    let k = (1.0 / d4).clamp(1e-14, 1.0 - 1e-14);

    // 8-step Arithmetic-Geometric Mean (AGM) for complete elliptic integral K(k^2) & Landen ladder
    let mut a_agm = [0.0f64; 9];
    let mut b_agm = [0.0f64; 9];
    let mut c_agm = [0.0f64; 9];
    a_agm[0] = 1.0;
    b_agm[0] = (1.0 - k * k).max(1e-30).sqrt();
    c_agm[0] = k;
    for step in 0..8 {
        a_agm[step + 1] = 0.5 * (a_agm[step] + b_agm[step]);
        b_agm[step + 1] = (a_agm[step] * b_agm[step]).sqrt();
        c_agm[step + 1] = 0.5 * (a_agm[step] - b_agm[step]);
    }
    let quarter_period_k = PI / (2.0 * a_agm[8]);

    let mut poles_sorted = [0.0f64; 2 * MAX_HALFBAND_POLES_PER_BRANCH];
    for i in 0..num_poles {
        let u = (((2 * i + 1) as f64) / (n_odd as f64) + 1.0) * quarter_period_k;
        let mut phi = 256.0 * a_agm[8] * u;
        for step in (1..=8).rev() {
            let ratio = (c_agm[step] / a_agm[step]) * phi.sin();
            phi = 0.5 * (phi + ratio.clamp(-1.0, 1.0).asin());
        }
        let sn = phi.sin();
        let d6 = sn * sn;
        let d7 = ((d4 * d4 - d6).max(0.0) * (1.0 - d6).max(0.0)).sqrt();
        poles_sorted[i] = ((d6 + d4 - d7) / (d6 + d4 + d7)).clamp(1e-12, 1.0 - 1e-12);
    }

    // Sort strictly ascending: `0 < c_0 < c_1 < ... < c_{M-1} < 1`.
    // Even indices go to undelayed Branch 0 (`A_0`); odd indices go to `z^{-1}`-delayed Branch 1 (`A_1`).
    for i in 0..num_poles {
        for j in (i + 1)..num_poles {
            if poles_sorted[j] < poles_sorted[i] {
                poles_sorted.swap(i, j);
            }
        }
    }

    let mut branch0_poles = [0.0f64; MAX_HALFBAND_POLES_PER_BRANCH];
    let mut branch1_poles = [0.0f64; MAX_HALFBAND_POLES_PER_BRANCH];
    let mut b0_len = 0usize;
    let mut b1_len = 0usize;

    for i in 0..num_poles {
        if (i & 1) == 0 {
            branch0_poles[b0_len] = poles_sorted[i];
            b0_len += 1;
        } else {
            branch1_poles[b1_len] = poles_sorted[i];
            b1_len += 1;
        }
    }

    JacobiEllipticHalfbandSpec {
        order_n: n_odd,
        omega_p: wp,
        branch0_poles,
        branch0_len: b0_len,
        branch1_poles,
        branch1_len: b1_len,
    }
}

/// Stateful stereo `1x <-> 2x` Polyphase Allpass Half-Band Stage (`0` heap allocations).
#[derive(Clone, Copy, Debug)]
pub struct AllpassHalfbandStage {
    pub spec: JacobiEllipticHalfbandSpec,
    up_b0_state: [[f64; MAX_HALFBAND_POLES_PER_BRANCH]; 2],
    up_b1_state: [[f64; MAX_HALFBAND_POLES_PER_BRANCH]; 2],
    down_b0_state: [[f64; MAX_HALFBAND_POLES_PER_BRANCH]; 2],
    down_b1_state: [[f64; MAX_HALFBAND_POLES_PER_BRANCH]; 2],
}

impl AllpassHalfbandStage {
    pub fn new(order_n: usize, omega_p: f64) -> Self {
        Self {
            spec: design_jacobi_elliptic_halfband(order_n, omega_p),
            up_b0_state: [[0.0; MAX_HALFBAND_POLES_PER_BRANCH]; 2],
            up_b1_state: [[0.0; MAX_HALFBAND_POLES_PER_BRANCH]; 2],
            down_b0_state: [[0.0; MAX_HALFBAND_POLES_PER_BRANCH]; 2],
            down_b1_state: [[0.0; MAX_HALFBAND_POLES_PER_BRANCH]; 2],
        }
    }

    pub fn reconfigure(&mut self, order_n: usize, omega_p: f64) {
        if self.spec.order_n != (order_n | 1) || (self.spec.omega_p - omega_p).abs() > 1e-12 {
            self.spec = design_jacobi_elliptic_halfband(order_n, omega_p);
            self.reset();
        }
    }

    pub fn reset(&mut self) {
        self.up_b0_state = [[0.0; MAX_HALFBAND_POLES_PER_BRANCH]; 2];
        self.up_b1_state = [[0.0; MAX_HALFBAND_POLES_PER_BRANCH]; 2];
        self.down_b0_state = [[0.0; MAX_HALFBAND_POLES_PER_BRANCH]; 2];
        self.down_b1_state = [[0.0; MAX_HALFBAND_POLES_PER_BRANCH]; 2];
    }

    #[inline(always)]
    fn step_branch(
        x: f64,
        poles: &[f64; MAX_HALFBAND_POLES_PER_BRANCH],
        len: usize,
        state: &mut [f64; MAX_HALFBAND_POLES_PER_BRANCH],
    ) -> f64 {
        let mut v = x;
        for i in 0..len {
            let c = poles[i];
            let s = state[i];
            let y = c * v + s;
            state[i] = v - c * y;
            v = y;
        }
        v
    }

    /// Upsamples 1 input sample `x` into `[s0, s1]` (`u[2n], u[2n+1]`).
    #[inline(always)]
    pub fn upsample_step(&mut self, ch: usize, x: f64) -> [f64; 2] {
        let s0 = Self::step_branch(
            x,
            &self.spec.branch0_poles,
            self.spec.branch0_len,
            &mut self.up_b0_state[ch],
        );
        let s1 = Self::step_branch(
            x,
            &self.spec.branch1_poles,
            self.spec.branch1_len,
            &mut self.up_b1_state[ch],
        );
        [s0, s1]
    }

    /// Anti-alias filters and decimates `[y0, y1]` back to `1x` (`0.5 * [A_0(y1) + A_1(y0)]`),
    /// making `upsample_step -> flat -> downsample_step` strictly allpass (`A_0(z) A_1(z)`).
    #[inline(always)]
    pub fn downsample_step(&mut self, ch: usize, y0: f64, y1: f64) -> f64 {
        let f_odd = Self::step_branch(
            y1,
            &self.spec.branch0_poles,
            self.spec.branch0_len,
            &mut self.down_b0_state[ch],
        );
        let f_even = Self::step_branch(
            y0,
            &self.spec.branch1_poles,
            self.spec.branch1_len,
            &mut self.down_b1_state[ch],
        );
        0.5 * (f_odd + f_even)
    }
}

/// Jacobi-Elliptic Polyphase Allpass Half-Band `2x` Oversampler + Downsampler (`sr::IIRConverterStyle`,
/// `0x18011d4e0` / `0x18011d830`) used in `NaturalPhase` mode (`> 113 dB` stopband attenuation,
/// `< 1e-12 dB` flat passband ripple, `~4` samples latency @ `1x`, `0` heap allocations).
#[derive(Clone, Copy, Debug)]
pub struct NaturalPhaseOversampler2x {
    stage: AllpassHalfbandStage,
}

pub type NaturalPhaseOversampler = NaturalPhaseOversampler2x;

impl Default for NaturalPhaseOversampler2x {
    fn default() -> Self {
        Self::new()
    }
}

impl NaturalPhaseOversampler2x {
    pub fn new() -> Self {
        Self {
            stage: AllpassHalfbandStage::new(17, PROQ4_STAGE1_OMEGA_P),
        }
    }

    pub fn with_order_and_omega(order_n: usize, omega_p: f64) -> Self {
        Self {
            stage: AllpassHalfbandStage::new(order_n, omega_p),
        }
    }

    /// Adapts half-band elliptic order to host sample rate (`<= 48,001 Hz`: `N = 17, wp = 0.44*pi`;
    /// `> 48,001 Hz`: `N = 11, wp = 0.33*pi`, both exceeding `112 dB` stopband rejection).
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        if sample_rate <= 48_001.0 {
            self.stage.reconfigure(17, PROQ4_STAGE1_OMEGA_P);
        } else {
            self.stage.reconfigure(11, PROQ4_STAGE2_OMEGA_P);
        }
    }

    pub fn reset(&mut self) {
        self.stage.reset();
    }

    #[inline]
    pub fn spec(&self) -> &JacobiEllipticHalfbandSpec {
        &self.stage.spec
    }

    #[inline]
    pub fn eval_one_way_response(&self, omega_2x_rad: f64) -> Complex64 {
        self.stage.spec.eval_frequency_response(omega_2x_rad)
    }

    /// Upsamples 1 input sample `x` into 2 samples `[s0, s1]` at `2 * sample_rate`.
    #[inline(always)]
    pub fn upsample_step(&mut self, ch: usize, x: f64) -> [f64; 2] {
        self.stage.upsample_step(ch, x)
    }

    /// Downsamples 2 oversampled samples `[y0, y1]` back to 1 sample at `1 * sample_rate`.
    #[inline(always)]
    pub fn downsample_step(&mut self, ch: usize, y0: f64, y1: f64) -> f64 {
        self.stage.downsample_step(ch, y0, y1)
    }
}

/// Two-Stage Cascaded `4x` Jacobi-Elliptic Polyphase Allpass Half-Band Oversampler (`1x -> 2x -> 4x`
/// and `4x -> 2x -> 1x`) in `f64` precision (`> 113 dB` stopband rejection, `0` heap allocations).
#[derive(Clone, Copy, Debug)]
pub struct JacobiEllipticOversampler4x {
    pub stage1: AllpassHalfbandStage,
    pub stage2: AllpassHalfbandStage,
}

impl Default for JacobiEllipticOversampler4x {
    fn default() -> Self {
        Self::new(48_000.0)
    }
}

impl JacobiEllipticOversampler4x {
    pub fn new(sample_rate: f64) -> Self {
        let mut s = Self {
            stage1: AllpassHalfbandStage::new(17, PROQ4_STAGE1_OMEGA_P),
            stage2: AllpassHalfbandStage::new(11, PROQ4_STAGE2_OMEGA_P),
        };
        s.set_sample_rate(sample_rate);
        s
    }

    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        if sample_rate <= 48_001.0 {
            self.stage1.reconfigure(17, PROQ4_STAGE1_OMEGA_P);
        } else {
            self.stage1.reconfigure(11, PROQ4_STAGE2_OMEGA_P);
        }
        self.stage2.reconfigure(11, PROQ4_STAGE2_OMEGA_P);
    }

    pub fn reset(&mut self) {
        self.stage1.reset();
        self.stage2.reset();
    }

    #[inline(always)]
    pub fn upsample_4x(&mut self, ch: usize, x: f64) -> [f64; 4] {
        let [u0, u1] = self.stage1.upsample_step(ch, x);
        let [s0, s1] = self.stage2.upsample_step(ch, u0);
        let [s2, s3] = self.stage2.upsample_step(ch, u1);
        [s0, s1, s2, s3]
    }

    #[inline(always)]
    pub fn downsample_4x(&mut self, ch: usize, y: [f64; 4]) -> f64 {
        let d0 = self.stage2.downsample_step(ch, y[0], y[1]);
        let d1 = self.stage2.downsample_step(ch, y[2], y[3]);
        self.stage1.downsample_step(ch, d0, d1)
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

    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.oversampler.set_sample_rate(sample_rate);
    }

    pub fn reset(&mut self) {
        for b in &mut self.bands_2x {
            b.reset_state();
        }
        self.oversampler.reset();
    }
}
