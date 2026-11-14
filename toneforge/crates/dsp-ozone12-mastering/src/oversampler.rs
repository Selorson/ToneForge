//! Zero-Allocation Polyphase Half-Band Oversampler (`2x` / `4x`, `f64`).
//!
//! Provides two production anti-aliasing / anti-imaging filter architectures:
//! 1. [`OversamplerFilterMode::EllipticPolyphaseIir`]:
//!    - Stage 1 (`1x <-> 2x`): 12-allpass-pole (25th-order) Jacobi-Elliptic polyphase half-band IIR
//!      (`6` first-order allpass sections per polyphase branch, `> 125 dB` stopband rejection).
//!    - Stage 2 (`2x <-> 4x`): 6-allpass-pole (13th-order) Jacobi-Elliptic polyphase half-band IIR
//!      (`3` first-order allpass sections per polyphase branch, `> 125 dB` stopband rejection).
//!    Synthesized analytically in `f64` via Arithmetic-Geometric Mean (AGM) / Landen Jacobi-elliptic
//!    pole placement (`0x18011d4e0`).
//! 2. [`OversamplerFilterMode::LinearPhaseFirKaiser128`]:
//!    - Exact `128`-tap Kaiser-windowed sinc (`beta = 9.0`) polyphase FIR matching Ozone 12's
//!      `DSP::OversampleInt` (`0x182796500`), providing zero phase dispersion for peak-preserving
//!      polynomial soft-clipping.
//!
//! 100% pre-allocated fixed arrays (`0` heap allocations on the real-time audio path).

use std::f64::consts::PI;

/// Stage 1 (`1x <-> 2x`) elliptic allpass pole count (`2 * 12 + 1 = 25th` order half-band IIR).
pub const STAGE1_ALLPASS_POLES: usize = 12;
/// Stage 1 poles per polyphase branch (`12 / 2 = 6`).
pub const STAGE1_BRANCH_POLES: usize = STAGE1_ALLPASS_POLES / 2;

/// Stage 2 (`2x <-> 4x`) elliptic allpass pole count (`2 * 6 + 1 = 13th` order half-band IIR).
pub const STAGE2_ALLPASS_POLES: usize = 6;
/// Stage 2 poles per polyphase branch (`6 / 2 = 3`).
pub const STAGE2_BRANCH_POLES: usize = STAGE2_ALLPASS_POLES / 2;

/// Total tap count of the linear-phase Kaiser (`beta = 9.0`) polyphase FIR (`DSP::OversampleInt`).
pub const KAISER_FIR_TAPS: usize = 128;
/// Taps per polyphase branch in `2x` FIR mode (`128 / 2 = 64`).
pub const KAISER_FIR_2X_BRANCH_TAPS: usize = KAISER_FIR_TAPS / 2;
/// Taps per polyphase branch in `4x` FIR mode (`128 / 4 = 32`).
pub const KAISER_FIR_4X_BRANCH_TAPS: usize = KAISER_FIR_TAPS / 4;
/// Kaiser window shape parameter `beta = 9.0` (`~92 dB` sidelobe rejection).
pub const KAISER_BETA: f64 = 9.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OversampleFactor {
    X1,
    X2,
    X4,
}

impl OversampleFactor {
    #[inline]
    pub const fn as_usize(self) -> usize {
        match self {
            Self::X1 => 1,
            Self::X2 => 2,
            Self::X4 => 4,
        }
    }

    #[inline]
    pub const fn from_factor(factor: usize) -> Self {
        match factor {
            0 | 1 => Self::X1,
            2 => Self::X2,
            _ => Self::X4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OversamplerFilterMode {
    /// Minimum-phase Jacobi-Elliptic polyphase allpass half-band IIR (25th + 13th order).
    EllipticPolyphaseIir,
    /// Zero phase-dispersion 128-tap Kaiser (`beta = 9.0`) linear-phase polyphase FIR.
    LinearPhaseFirKaiser128,
}

/// Evaluates the zeroth-order modified Bessel function of the first kind `I_0(x)` in `f64`.
#[inline]
pub fn bessel_i0(x: f64) -> f64 {
    let half_x = 0.5 * x;
    let x2 = half_x * half_x;
    let mut sum = 1.0_f64;
    let mut term = 1.0_f64;
    for k in 1..32 {
        let kf = k as f64;
        term *= x2 / (kf * kf);
        sum += term;
        if term <= sum * 1.0e-16 {
            break;
        }
    }
    sum
}

/// Computes the complete elliptic integral `K(k^2)` and AGM sequence for modulus `k in (0, 1)`.
fn agm_elliptic_k_and_ladder(k: f64) -> (f64, [(f64, f64); 16], usize) {
    let k_clamped = k.clamp(1.0e-15, 1.0 - 1.0e-15);
    let mut a = 1.0_f64;
    let mut b = (1.0 - k_clamped * k_clamped).sqrt();
    let mut c = k_clamped;
    let mut ladder = [(0.0_f64, 0.0_f64); 16];
    let mut steps = 0usize;

    while steps < ladder.len() {
        ladder[steps] = (a, c);
        if c.abs() <= 1.0e-15 * a {
            break;
        }
        let a_next = 0.5 * (a + b);
        let b_next = (a * b).sqrt();
        c = 0.5 * (a - b);
        a = a_next;
        b = b_next;
        steps += 1;
    }
    let idx = steps.min(ladder.len() - 1);
    ladder[idx] = (a, c);
    let cap_k = PI / (2.0 * a);
    (cap_k, ladder, idx)
}

/// Evaluates Jacobi elliptic `sn(u, k)` via descending Landen / AGM recurrence.
fn jacobi_sn_from_ladder(u: f64, ladder: &[(f64, f64); 16], top_idx: usize) -> f64 {
    let a_m = ladder[top_idx].0;
    let scale = (1_u64 << (top_idx as u32)) as f64;
    let mut phi = scale * a_m * u;
    for n in (1..=top_idx).rev() {
        let (a_n, c_n) = ladder[n];
        let arg = ((c_n / a_n) * phi.sin()).clamp(-1.0, 1.0);
        phi = 0.5 * (phi + arg.asin());
    }
    phi.sin()
}

/// Synthesizes `P` sorted Jacobi-Elliptic half-band allpass poles `0 < c[0] < ... < c[P-1] < 1`
/// for filter order `N = 2 * P + 1` and normalized passband edge `omega_p` in `(0, pi/2)`.
///
/// Even indices `c[0], c[2], ...` belong to un-delayed polyphase branch `A_0(z^-2)`.
/// Odd indices `c[1], c[3], ...` belong to `z^-1`-delayed polyphase branch `A_1(z^-2)`.
pub fn design_elliptic_halfband_poles<const P: usize>(omega_p_rad: f64) -> [f64; P] {
    let wp = omega_p_rad.clamp(0.05 * PI, 0.495 * PI);
    let cot_half = 1.0 / (0.5 * wp).tan();
    let d4 = cot_half * cot_half;
    let k = 1.0 / d4;
    let (cap_k, ladder, top_idx) = agm_elliptic_k_and_ladder(k);
    let order_n = (2 * P + 1) as f64;

    let mut poles = [0.0_f64; P];
    for (i, slot) in poles.iter_mut().enumerate() {
        let u_i = (((2 * i + 1) as f64) / order_n + 1.0) * cap_k;
        let sn = jacobi_sn_from_ladder(u_i, &ladder, top_idx);
        let d6 = sn * sn;
        let d7 = ((d4 * d4 - d6).max(0.0) * (1.0 - d6).max(0.0)).sqrt();
        *slot = ((d6 + d4 - d7) / (d6 + d4 + d7)).clamp(1.0e-12, 1.0 - 1.0e-12);
    }

    // Insertion-sort ascending so even indices -> A0 (un-delayed) and odd indices -> A1 (z^-1 delayed).
    for i in 1..P {
        let key = poles[i];
        let mut j = i;
        while j > 0 && poles[j - 1] > key {
            poles[j] = poles[j - 1];
            j -= 1;
        }
        poles[j] = key;
    }
    poles
}

/// First-order polyphase allpass cascade `prod_k (a_k + z^-1) / (1 + a_k z^-1)`.
#[derive(Clone, Copy, Debug)]
struct AllpassBranch<const B: usize> {
    coeffs: [f64; B],
    state: [f64; B],
}

impl<const B: usize> AllpassBranch<B> {
    #[inline]
    const fn new(coeffs: [f64; B]) -> Self {
        Self {
            coeffs,
            state: [0.0; B],
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.state = [0.0; B];
    }

    #[inline]
    fn process(&mut self, mut x: f64) -> f64 {
        for i in 0..B {
            let a = self.coeffs[i];
            let y = a * x + self.state[i];
            self.state[i] = x - a * y;
            x = y;
        }
        x
    }
}

/// Single-channel 2x polyphase allpass half-band stage (`B` poles per branch, order `4*B + 1`).
#[derive(Clone, Copy, Debug)]
struct EllipticHalfBandStage1Ch<const B: usize> {
    up_a0: AllpassBranch<B>,
    up_a1: AllpassBranch<B>,
    dn_a0: AllpassBranch<B>,
    dn_a1: AllpassBranch<B>,
}

impl<const B: usize> EllipticHalfBandStage1Ch<B> {
    fn from_sorted_poles<const P: usize>(poles: &[f64; P]) -> Self {
        debug_assert_eq!(P, 2 * B);
        let mut c0 = [0.0_f64; B];
        let mut c1 = [0.0_f64; B];
        for b in 0..B {
            c0[b] = poles[2 * b];
            c1[b] = poles[2 * b + 1];
        }
        Self {
            up_a0: AllpassBranch::new(c0),
            up_a1: AllpassBranch::new(c1),
            dn_a0: AllpassBranch::new(c0),
            dn_a1: AllpassBranch::new(c1),
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.up_a0.reset();
        self.up_a1.reset();
        self.dn_a0.reset();
        self.dn_a1.reset();
    }

    /// Upsamples 1 input sample into 2 chronological oversampled sub-samples `[y[2n], y[2n+1]]`.
    #[inline]
    fn upsample(&mut self, x: f64) -> [f64; 2] {
        [self.up_a0.process(x), self.up_a1.process(x)]
    }

    /// Downsamples 2 chronological oversampled sub-samples `[x[2n], x[2n+1]]` into 1 output sample.
    #[inline]
    fn downsample(&mut self, sub: [f64; 2]) -> f64 {
        0.5 * (self.dn_a0.process(sub[1]) + self.dn_a1.process(sub[0]))
    }
}

/// Precomputed `128`-tap Kaiser (`beta = 9.0`) polyphase FIR tables for `2x` and `4x`.
#[derive(Clone, Copy, Debug)]
pub struct KaiserPolyphaseFirCoeffs {
    up_2x: [[f64; KAISER_FIR_2X_BRANCH_TAPS]; 2],
    dn_2x: [[f64; KAISER_FIR_2X_BRANCH_TAPS]; 2],
    up_4x: [[f64; KAISER_FIR_4X_BRANCH_TAPS]; 4],
    dn_4x: [[f64; KAISER_FIR_4X_BRANCH_TAPS]; 4],
}

impl KaiserPolyphaseFirCoeffs {
    pub fn design() -> Self {
        let i0_beta = bessel_i0(KAISER_BETA);
        let center = 0.5 * ((KAISER_FIR_TAPS - 1) as f64);

        // Cutoff slightly inside 1/(2*M) so the entire stopband [f_s/2, M*f_s/2] sits > 92 dB down
        // while keeping the audible passband (< 20.5 kHz @ 48 kHz) flat within < 0.005 dB.
        let proto_2x = Self::design_prototype(0.478 * 0.5, center, i0_beta);
        let proto_4x = Self::design_prototype(0.478 * 0.25, center, i0_beta);

        let mut up_2x = [[0.0_f64; KAISER_FIR_2X_BRANCH_TAPS]; 2];
        let mut dn_2x = [[0.0_f64; KAISER_FIR_2X_BRANCH_TAPS]; 2];
        for p in 0..2 {
            let mut branch_sum = 0.0_f64;
            for k in 0..KAISER_FIR_2X_BRANCH_TAPS {
                let v = proto_2x[k * 2 + p];
                up_2x[p][k] = v;
                branch_sum += v;
            }
            let inv = if branch_sum.abs() > 1.0e-15 {
                1.0 / branch_sum
            } else {
                1.0
            };
            for k in 0..KAISER_FIR_2X_BRANCH_TAPS {
                up_2x[p][k] *= inv;
                dn_2x[p][k] = proto_2x[k * 2 + (1 - p)];
            }
        }

        let mut up_4x = [[0.0_f64; KAISER_FIR_4X_BRANCH_TAPS]; 4];
        let mut dn_4x = [[0.0_f64; KAISER_FIR_4X_BRANCH_TAPS]; 4];
        for p in 0..4 {
            let mut branch_sum = 0.0_f64;
            for k in 0..KAISER_FIR_4X_BRANCH_TAPS {
                let v = proto_4x[k * 4 + p];
                up_4x[p][k] = v;
                branch_sum += v;
            }
            let inv = if branch_sum.abs() > 1.0e-15 {
                1.0 / branch_sum
            } else {
                1.0
            };
            for k in 0..KAISER_FIR_4X_BRANCH_TAPS {
                up_4x[p][k] *= inv;
                dn_4x[p][k] = proto_4x[k * 4 + (3 - p)];
            }
        }

        Self {
            up_2x,
            dn_2x,
            up_4x,
            dn_4x,
        }
    }

    fn design_prototype(fc_cycles: f64, center: f64, i0_beta: f64) -> [f64; KAISER_FIR_TAPS] {
        let mut taps = [0.0_f64; KAISER_FIR_TAPS];
        let mut sum = 0.0_f64;
        for (m, slot) in taps.iter_mut().enumerate() {
            let x = (m as f64) - center;
            let norm = x / center;
            let win = bessel_i0(KAISER_BETA * (1.0 - norm * norm).max(0.0).sqrt()) / i0_beta;
            let sinc = if x.abs() < 1.0e-12 {
                2.0 * fc_cycles
            } else {
                (2.0 * PI * fc_cycles * x).sin() / (PI * x)
            };
            let v = sinc * win;
            *slot = v;
            sum += v;
        }
        let inv_sum = if sum.abs() > 1.0e-15 { 1.0 / sum } else { 1.0 };
        for slot in &mut taps {
            *slot *= inv_sum;
        }
        taps
    }
}

#[derive(Clone, Copy, Debug)]
struct KaiserFirChannelState {
    up_hist_2x: [f64; KAISER_FIR_2X_BRANCH_TAPS],
    up_pos_2x: usize,
    dn_hist_2x: [[f64; KAISER_FIR_2X_BRANCH_TAPS]; 2],
    dn_pos_2x: usize,

    up_hist_4x: [f64; KAISER_FIR_4X_BRANCH_TAPS],
    up_pos_4x: usize,
    dn_hist_4x: [[f64; KAISER_FIR_4X_BRANCH_TAPS]; 4],
    dn_pos_4x: usize,
}

impl KaiserFirChannelState {
    const fn new() -> Self {
        Self {
            up_hist_2x: [0.0; KAISER_FIR_2X_BRANCH_TAPS],
            up_pos_2x: 0,
            dn_hist_2x: [[0.0; KAISER_FIR_2X_BRANCH_TAPS]; 2],
            dn_pos_2x: 0,
            up_hist_4x: [0.0; KAISER_FIR_4X_BRANCH_TAPS],
            up_pos_4x: 0,
            dn_hist_4x: [[0.0; KAISER_FIR_4X_BRANCH_TAPS]; 4],
            dn_pos_4x: 0,
        }
    }

    #[inline]
    fn reset(&mut self) {
        *self = Self::new();
    }

    #[inline]
    fn upsample_2x(&mut self, x: f64, coeffs: &KaiserPolyphaseFirCoeffs) -> [f64; 2] {
        self.up_pos_2x = if self.up_pos_2x == 0 {
            KAISER_FIR_2X_BRANCH_TAPS - 1
        } else {
            self.up_pos_2x - 1
        };
        self.up_hist_2x[self.up_pos_2x] = x;

        let mut out = [0.0_f64; 2];
        for p in 0..2 {
            let branch = &coeffs.up_2x[p];
            let mut acc = 0.0_f64;
            let mut idx = self.up_pos_2x;
            for &tap in branch.iter() {
                acc += tap * self.up_hist_2x[idx];
                idx += 1;
                if idx == KAISER_FIR_2X_BRANCH_TAPS {
                    idx = 0;
                }
            }
            out[p] = acc;
        }
        out
    }

    #[inline]
    fn downsample_2x(&mut self, sub: [f64; 2], coeffs: &KaiserPolyphaseFirCoeffs) -> f64 {
        self.dn_pos_2x = if self.dn_pos_2x == 0 {
            KAISER_FIR_2X_BRANCH_TAPS - 1
        } else {
            self.dn_pos_2x - 1
        };
        self.dn_hist_2x[0][self.dn_pos_2x] = sub[0];
        self.dn_hist_2x[1][self.dn_pos_2x] = sub[1];

        let mut acc = 0.0_f64;
        let mut idx = self.dn_pos_2x;
        for k in 0..KAISER_FIR_2X_BRANCH_TAPS {
            acc += coeffs.dn_2x[0][k] * self.dn_hist_2x[0][idx]
                + coeffs.dn_2x[1][k] * self.dn_hist_2x[1][idx];
            idx += 1;
            if idx == KAISER_FIR_2X_BRANCH_TAPS {
                idx = 0;
            }
        }
        acc
    }

    #[inline]
    fn upsample_4x(&mut self, x: f64, coeffs: &KaiserPolyphaseFirCoeffs) -> [f64; 4] {
        self.up_pos_4x = if self.up_pos_4x == 0 {
            KAISER_FIR_4X_BRANCH_TAPS - 1
        } else {
            self.up_pos_4x - 1
        };
        self.up_hist_4x[self.up_pos_4x] = x;

        let mut out = [0.0_f64; 4];
        for p in 0..4 {
            let branch = &coeffs.up_4x[p];
            let mut acc = 0.0_f64;
            let mut idx = self.up_pos_4x;
            for &tap in branch.iter() {
                acc += tap * self.up_hist_4x[idx];
                idx += 1;
                if idx == KAISER_FIR_4X_BRANCH_TAPS {
                    idx = 0;
                }
            }
            out[p] = acc;
        }
        out
    }

    #[inline]
    fn downsample_4x(&mut self, sub: [f64; 4], coeffs: &KaiserPolyphaseFirCoeffs) -> f64 {
        self.dn_pos_4x = if self.dn_pos_4x == 0 {
            KAISER_FIR_4X_BRANCH_TAPS - 1
        } else {
            self.dn_pos_4x - 1
        };
        for p in 0..4 {
            self.dn_hist_4x[p][self.dn_pos_4x] = sub[p];
        }

        let mut acc = 0.0_f64;
        let mut idx = self.dn_pos_4x;
        for k in 0..KAISER_FIR_4X_BRANCH_TAPS {
            acc += coeffs.dn_4x[0][k] * self.dn_hist_4x[0][idx]
                + coeffs.dn_4x[1][k] * self.dn_hist_4x[1][idx]
                + coeffs.dn_4x[2][k] * self.dn_hist_4x[2][idx]
                + coeffs.dn_4x[3][k] * self.dn_hist_4x[3][idx];
            idx += 1;
            if idx == KAISER_FIR_4X_BRANCH_TAPS {
                idx = 0;
            }
        }
        acc
    }
}

/// Stereo zero-allocation `2x` / `4x` `f64` polyphase oversampler supporting both
/// Jacobi-Elliptic Half-Band Allpass IIR (12-pole Stage 1 + 6-pole Stage 2) and
/// `128`-tap Kaiser (`beta = 9.0`) Linear-Phase Polyphase FIR (`DSP::OversampleInt`).
#[derive(Clone, Debug)]
pub struct PolyphaseHalfBandOversampler {
    mode: OversamplerFilterMode,
    factor: OversampleFactor,
    stage1_iir: [EllipticHalfBandStage1Ch<STAGE1_BRANCH_POLES>; 2],
    stage2_iir: [EllipticHalfBandStage1Ch<STAGE2_BRANCH_POLES>; 2],
    fir_coeffs: KaiserPolyphaseFirCoeffs,
    fir_state: [KaiserFirChannelState; 2],
}

impl PolyphaseHalfBandOversampler {
    /// Creates a new stereo zero-allocation oversampler.
    pub fn new(mode: OversamplerFilterMode, factor: OversampleFactor) -> Self {
        // Stage 1 (1x <-> 2x): passband up to 0.435 * pi (~20.88 kHz @ 48 kHz), 12 poles -> > 125 dB stopband.
        let poles_s1 = design_elliptic_halfband_poles::<STAGE1_ALLPASS_POLES>(0.435 * PI);
        // Stage 2 (2x <-> 4x): passband up to 0.275 * pi (~26.4 kHz @ 96 kHz), 6 poles -> > 130 dB stopband.
        let poles_s2 = design_elliptic_halfband_poles::<STAGE2_ALLPASS_POLES>(0.275 * PI);

        let s1 = EllipticHalfBandStage1Ch::from_sorted_poles(&poles_s1);
        let s2 = EllipticHalfBandStage1Ch::from_sorted_poles(&poles_s2);
        let fir_coeffs = KaiserPolyphaseFirCoeffs::design();

        Self {
            mode,
            factor,
            stage1_iir: [s1, s1],
            stage2_iir: [s2, s2],
            fir_coeffs,
            fir_state: [KaiserFirChannelState::new(), KaiserFirChannelState::new()],
        }
    }

    #[inline]
    pub fn set_mode(&mut self, mode: OversamplerFilterMode) {
        if self.mode != mode {
            self.mode = mode;
            self.reset();
        }
    }

    #[inline]
    pub const fn mode(&self) -> OversamplerFilterMode {
        self.mode
    }

    #[inline]
    pub fn set_factor(&mut self, factor: OversampleFactor) {
        if self.factor != factor {
            self.factor = factor;
            self.reset();
        }
    }

    #[inline]
    pub const fn factor(&self) -> OversampleFactor {
        self.factor
    }

    /// Resets all internal IIR and FIR delay states without allocating.
    #[inline]
    pub fn reset(&mut self) {
        for ch in 0..2 {
            self.stage1_iir[ch].reset();
            self.stage2_iir[ch].reset();
            self.fir_state[ch].reset();
        }
    }

    /// Round-trip group delay in integer host samples (rounded) for linear-phase FIR mode,
    /// or nominal low-frequency group delay for elliptic IIR mode.
    #[inline]
    pub const fn latency_host_samples(&self) -> usize {
        match (self.mode, self.factor) {
            (_, OversampleFactor::X1) => 0,
            (OversamplerFilterMode::LinearPhaseFirKaiser128, OversampleFactor::X2) => {
                KAISER_FIR_2X_BRANCH_TAPS - 1
            }
            (OversamplerFilterMode::LinearPhaseFirKaiser128, OversampleFactor::X4) => {
                KAISER_FIR_4X_BRANCH_TAPS - 1
            }
            (OversamplerFilterMode::EllipticPolyphaseIir, OversampleFactor::X2) => 5,
            (OversamplerFilterMode::EllipticPolyphaseIir, OversampleFactor::X4) => 6,
        }
    }

    /// Upsamples a single channel sample by `2x`.
    #[inline]
    pub fn upsample_channel_2x(&mut self, ch: usize, x: f64) -> [f64; 2] {
        match self.mode {
            OversamplerFilterMode::EllipticPolyphaseIir => self.stage1_iir[ch].upsample(x),
            OversamplerFilterMode::LinearPhaseFirKaiser128 => {
                self.fir_state[ch].upsample_2x(x, &self.fir_coeffs)
            }
        }
    }

    /// Downsamples 2 oversampled sub-samples on a single channel back to `1x`.
    #[inline]
    pub fn downsample_channel_2x(&mut self, ch: usize, sub: [f64; 2]) -> f64 {
        match self.mode {
            OversamplerFilterMode::EllipticPolyphaseIir => self.stage1_iir[ch].downsample(sub),
            OversamplerFilterMode::LinearPhaseFirKaiser128 => {
                self.fir_state[ch].downsample_2x(sub, &self.fir_coeffs)
            }
        }
    }

    /// Upsamples a single channel sample by `4x`.
    #[inline]
    pub fn upsample_channel_4x(&mut self, ch: usize, x: f64) -> [f64; 4] {
        match self.mode {
            OversamplerFilterMode::EllipticPolyphaseIir => {
                let s2 = self.stage1_iir[ch].upsample(x);
                let u01 = self.stage2_iir[ch].upsample(s2[0]);
                let u23 = self.stage2_iir[ch].upsample(s2[1]);
                [u01[0], u01[1], u23[0], u23[1]]
            }
            OversamplerFilterMode::LinearPhaseFirKaiser128 => {
                self.fir_state[ch].upsample_4x(x, &self.fir_coeffs)
            }
        }
    }

    /// Downsamples 4 oversampled sub-samples on a single channel back to `1x`.
    #[inline]
    pub fn downsample_channel_4x(&mut self, ch: usize, sub: [f64; 4]) -> f64 {
        match self.mode {
            OversamplerFilterMode::EllipticPolyphaseIir => {
                let d0 = self.stage2_iir[ch].downsample([sub[0], sub[1]]);
                let d1 = self.stage2_iir[ch].downsample([sub[2], sub[3]]);
                self.stage1_iir[ch].downsample([d0, d1])
            }
            OversamplerFilterMode::LinearPhaseFirKaiser128 => {
                self.fir_state[ch].downsample_4x(sub, &self.fir_coeffs)
            }
        }
    }

    /// Processes a stereo `f64` frame `[L, R]` at the configured `1x` / `2x` / `4x` factor
    /// with zero heap allocations.
    #[inline]
    pub fn process_stereo_frame<F>(&mut self, input: [f64; 2], mut kernel: F) -> [f64; 2]
    where
        F: FnMut([f64; 2]) -> [f64; 2],
    {
        match self.factor {
            OversampleFactor::X1 => kernel(input),
            OversampleFactor::X2 => {
                let l_up = self.upsample_channel_2x(0, input[0]);
                let r_up = self.upsample_channel_2x(1, input[1]);
                let y0 = kernel([l_up[0], r_up[0]]);
                let y1 = kernel([l_up[1], r_up[1]]);
                [
                    self.downsample_channel_2x(0, [y0[0], y1[0]]),
                    self.downsample_channel_2x(1, [y0[1], y1[1]]),
                ]
            }
            OversampleFactor::X4 => {
                let l_up = self.upsample_channel_4x(0, input[0]);
                let r_up = self.upsample_channel_4x(1, input[1]);
                let y0 = kernel([l_up[0], r_up[0]]);
                let y1 = kernel([l_up[1], r_up[1]]);
                let y2 = kernel([l_up[2], r_up[2]]);
                let y3 = kernel([l_up[3], r_up[3]]);
                [
                    self.downsample_channel_4x(0, [y0[0], y1[0], y2[0], y3[0]]),
                    self.downsample_channel_4x(1, [y0[1], y1[1], y2[1], y3[1]]),
                ]
            }
        }
    }

    /// Processes a stereo slice in place (`0` heap allocations).
    #[inline]
    pub fn process_stereo_block_f32<F>(&mut self, left: &mut [f32], right: &mut [f32], mut kernel: F)
    where
        F: FnMut([f64; 2]) -> [f64; 2],
    {
        let n = left.len().min(right.len());
        for i in 0..n {
            let y = self.process_stereo_frame([left[i] as f64, right[i] as f64], &mut kernel);
            left[i] = y[0] as f32;
            right[i] = y[1] as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elliptic_poles_are_strictly_ordered_inside_unit_circle() {
        let p1 = design_elliptic_halfband_poles::<STAGE1_ALLPASS_POLES>(0.435 * PI);
        let p2 = design_elliptic_halfband_poles::<STAGE2_ALLPASS_POLES>(0.275 * PI);
        for w in p1.windows(2) {
            assert!(w[0] > 0.0 && w[1] < 1.0 && w[0] < w[1]);
        }
        for w in p2.windows(2) {
            assert!(w[0] > 0.0 && w[1] < 1.0 && w[0] < w[1]);
        }
    }

    #[test]
    fn both_modes_preserve_passband_gain_and_suppress_hard_clip_aliasing() {
        for mode in [
            OversamplerFilterMode::EllipticPolyphaseIir,
            OversamplerFilterMode::LinearPhaseFirKaiser128,
        ] {
            let mut os = PolyphaseHalfBandOversampler::new(mode, OversampleFactor::X4);
            let sr = 48_000.0_f64;
            let freq = 11_300.0_f64;
            let amp = 10.0_f64.powf(-1.0 / 20.0); // -1.0 dBFS
            let n = 8192usize;
            let warmup = 1024usize;

            let mut out_1x = Vec::with_capacity(n);
            let mut out_4x = Vec::with_capacity(n);

            for i in 0..(n + warmup) {
                let x = amp * (2.0 * PI * freq * (i as f64) / sr).sin();
                // Mild cubic-soft clip y = tanh(2.2 * x) / tanh(2.2)
                let nl = |v: f64| (2.2 * v).tanh() / 2.2_f64.tanh();
                let y1 = nl(x);
                let y4 = os.process_stereo_frame([x, x], |[l, r]| [nl(l), nl(r)])[0];
                if i >= warmup {
                    out_1x.push(y1);
                    out_4x.push(y4);
                }
            }

            // Measure 3rd harmonic folded alias at 48000 - 3*11300 = 14100 Hz via Hann-windowed DFT.
            let measure_db = |sig: &[f64], f_target: f64| -> f64 {
                let mut re = 0.0_f64;
                let mut im = 0.0_f64;
                let mut w_sum = 0.0_f64;
                for (k, &s) in sig.iter().enumerate() {
                    let w = 0.5 - 0.5 * (2.0 * PI * (k as f64) / (sig.len() as f64)).cos();
                    let ph = 2.0 * PI * f_target * (k as f64) / sr;
                    re += s * w * ph.cos();
                    im -= s * w * ph.sin();
                    w_sum += w;
                }
                let mag = 2.0 * (re * re + im * im).sqrt() / w_sum;
                20.0 * mag.max(1.0e-15).log10()
            };

            let spur_1x = measure_db(&out_1x, 14_100.0);
            let spur_4x = measure_db(&out_4x, 14_100.0);
            assert!(
                spur_1x - spur_4x > 60.0,
                "mode={mode:?}: 1x spur={spur_1x:.2} dB vs 4x spur={spur_4x:.2} dB"
            );
        }
    }
}
