//! `character_waveshaper`: Oversampled ADAA Analog Harmonic Saturation (`Character` Modes:
//! `Clean`, `Subtle`, `Warm` at `0x18019DA1F..0x18019DB82`, `WaveShaper` `0x180221BF8` /
//! `WaveShaperVecLL<&vectorClipTanHyperbolicLLFast>` `0x1801234e0`).
//!
//! Architecture:
//! - `Clean`: Bit-transparent linear bypass.
//! - `Subtle`: `4x` Jacobi-Elliptic Polyphase Allpass Oversampling + 1st-order Antiderivative
//!   Anti-Aliasing (`ADAA1`) on FabFilter's linear-low-level (`|x| <= L`, exact `f'(0) = 1.0`) +
//!   asymmetric transformer `tanh` transfer function (`C^2` continuous) + `f64` 5 Hz DC blocker.
//! - `Warm`: `4x` Jacobi-Elliptic Polyphase Allpass Oversampling + 2nd-order Antiderivative
//!   Anti-Aliasing (`ADAA2`) on a globally `C^2`-continuous asymmetric triode transfer function
//!   normalized to exact unity small-signal slope `f'(0) = 1.0` with rich 2nd-order even harmonics +
//!   `f64` 5 Hz DC blocker.

use crate::natural_phase::JacobiEllipticOversampler4x;
use crate::params_601::CharacterMode;
use std::f64::consts::{LN_2, TAU};

pub const SUBTLE_C0: f64 = 1.975;
pub const SUBTLE_C1: f64 = 2.1419137;
pub const SUBTLE_ASYM: f64 = -0.074325;

pub const WARM_DRIVE: f64 = 0.562745;
pub const WARM_C0: f64 = 1.9856194;
pub const WARM_C1: f64 = 2.607522;

/// Linear low-level threshold `L` for `Subtle` (`WaveShaperVecLL` `0x1801234e0`).
pub const SUBTLE_LINEAR_THRESHOLD: f64 = 0.35;
const SUBTLE_W_POS: f64 = 1.0 - SUBTLE_LINEAR_THRESHOLD; // 0.65
const SUBTLE_W_NEG: f64 = 0.72; // Asymmetric transformer rail

/// Positive & negative saturation knees for `Warm` (`C^2`-continuous asymmetric tube curve).
const WARM_U_POS: f64 = 2.2;
const WARM_U_NEG: f64 = 1.8;
const WARM_BETA_NEG: f64 = (2.0 / WARM_U_POS) + (2.0 / WARM_U_NEG);
const WARM_A2: f64 = -1.0 / WARM_U_POS;
const WARM_A3: f64 = 1.0 / (3.0 * WARM_U_POS * WARM_U_POS);
const WARM_B2: f64 = -1.0 / WARM_U_POS;
const WARM_B3: f64 = (1.0 - 2.0 * WARM_BETA_NEG * WARM_U_NEG) / (3.0 * WARM_U_NEG * WARM_U_NEG);
const WARM_B4: f64 = -WARM_BETA_NEG / (4.0 * WARM_U_NEG * WARM_U_NEG);

const ADAA1_DEBLUR_A: f64 = 0.133;
const ADAA2_DEBLUR_A: f64 = 0.17875;
const ADAA_DELTA_EPS: f64 = 1e-6;
const ADAA2_SEG_EPS: f64 = 1e-5;

/// Stateful stereo `4x`-oversampled `ADAA1`/`ADAA2` analog saturation & `f64` DC-blocker.
#[derive(Clone, Copy, Debug)]
pub struct CharacterWaveShaper {
    oversampler_4x: JacobiEllipticOversampler4x,
    adaa_x1: [f64; 2],
    adaa_x2: [f64; 2],
    adaa_f1_x1: [f64; 2],
    deblur_z1: [f64; 2],
    deblur_z2: [f64; 2],
    dc_x1: [f64; 2],
    dc_y1: [f64; 2],
    dc_r: f64,
    dc_b0: f64,
    active_mode: CharacterMode,
}

/// Alias matching both capitalization conventions (`CharacterWaveShaper` / `CharacterWaveshaper`).
pub type CharacterWaveshaper = CharacterWaveShaper;

impl Default for CharacterWaveShaper {
    fn default() -> Self {
        Self::new(48_000.0)
    }
}

impl CharacterWaveShaper {
    pub fn new(sample_rate: f64) -> Self {
        let mut s = Self {
            oversampler_4x: JacobiEllipticOversampler4x::new(sample_rate),
            adaa_x1: [0.0; 2],
            adaa_x2: [0.0; 2],
            adaa_f1_x1: [0.0; 2],
            deblur_z1: [0.0; 2],
            deblur_z2: [0.0; 2],
            dc_x1: [0.0; 2],
            dc_y1: [0.0; 2],
            dc_r: 0.99935,
            dc_b0: 0.999675,
            active_mode: CharacterMode::Clean,
        };
        s.set_sample_rate(sample_rate);
        s
    }

    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        let fs = sample_rate.max(8000.0);
        self.oversampler_4x.set_sample_rate(fs);
        // Exact bilinear 5 Hz high-pass DC blocker pole & HF-unity gain factor
        let k = (0.5 * TAU * 5.0 / fs).tan();
        let inv = 1.0 / (1.0 + k);
        self.dc_b0 = inv;
        self.dc_r = ((1.0 - k) * inv).clamp(0.990, 0.999_99);
    }

    pub fn reset(&mut self) {
        self.oversampler_4x.reset();
        self.adaa_x1 = [0.0; 2];
        self.adaa_x2 = [0.0; 2];
        self.adaa_f1_x1 = [0.0; 2];
        self.deblur_z1 = [0.0; 2];
        self.deblur_z2 = [0.0; 2];
        self.dc_x1 = [0.0; 2];
        self.dc_y1 = [0.0; 2];
    }

    /// Overflow- and cancellation-free evaluation of `ln(cosh(z))` for `z >= 0`.
    #[inline(always)]
    fn ln_cosh_nonneg(z: f64) -> f64 {
        if z < 0.12 {
            let z2 = z * z;
            z2 * (0.5
                - z2 * ((1.0 / 12.0)
                    - z2 * ((1.0 / 45.0) - z2 * (17.0 / 2520.0))))
        } else {
            z + (-2.0 * z).exp().ln_1p() - LN_2
        }
    }

    /// `Subtle` memoryless transfer function (`WaveShaperVecLL<&vectorClipTanHyperbolicLLFast>`, `0x1801234e0`):
    /// strictly linear (`f(x) = x`, `f'(0) = 1.0`) for `|x| <= L`, `C^2`-smoothly joining asymmetric `tanh` rails.
    #[inline(always)]
    pub fn subtle_f(x: f64) -> f64 {
        let l = SUBTLE_LINEAR_THRESHOLD;
        if x > l {
            l + SUBTLE_W_POS * ((x - l) / SUBTLE_W_POS).tanh()
        } else if x < -l {
            -(l + SUBTLE_W_NEG * ((-x - l) / SUBTLE_W_NEG).tanh())
        } else {
            x
        }
    }

    /// Exact closed-form 1st antiderivative `F_1(x) = \int_0^x subtle_f(t) dt`.
    #[inline(always)]
    pub fn subtle_f1(x: f64) -> f64 {
        let l = SUBTLE_LINEAR_THRESHOLD;
        if x > l {
            let d = x - l;
            0.5 * l * l + l * d + (SUBTLE_W_POS * SUBTLE_W_POS) * Self::ln_cosh_nonneg(d / SUBTLE_W_POS)
        } else if x < -l {
            let d = -x - l;
            0.5 * l * l + l * d + (SUBTLE_W_NEG * SUBTLE_W_NEG) * Self::ln_cosh_nonneg(d / SUBTLE_W_NEG)
        } else {
            0.5 * x * x
        }
    }

    #[inline(always)]
    fn warm_pos_poly(x: f64) -> (f64, f64, f64) {
        let x2 = x * x;
        let x3 = x2 * x;
        let x4 = x2 * x2;
        let x5 = x4 * x;
        let f = x + WARM_A2 * x2 + WARM_A3 * x3;
        let f1 = 0.5 * x2 + (WARM_A2 / 3.0) * x3 + (WARM_A3 * 0.25) * x4;
        let f2 = (1.0 / 6.0) * x3 + (WARM_A2 / 12.0) * x4 + (WARM_A3 * 0.05) * x5;
        (f, f1, f2)
    }

    #[inline(always)]
    fn warm_neg_poly(x: f64) -> (f64, f64, f64) {
        let x2 = x * x;
        let x3 = x2 * x;
        let x4 = x2 * x2;
        let x5 = x4 * x;
        let x6 = x3 * x3;
        let f = x + WARM_B2 * x2 + WARM_B3 * x3 + WARM_B4 * x4;
        let f1 = 0.5 * x2 + (WARM_B2 / 3.0) * x3 + (WARM_B3 * 0.25) * x4 + (WARM_B4 * 0.2) * x5;
        let f2 = (1.0 / 6.0) * x3
            + (WARM_B2 / 12.0) * x4
            + (WARM_B3 * 0.05) * x5
            + (WARM_B4 / 30.0) * x6;
        (f, f1, f2)
    }

    /// `Warm` memoryless asymmetric tube transfer function (`f(0) = 0`, `f'(0) = 1.0`, `f''(0) = -2/u_+`).
    #[inline(always)]
    pub fn warm_f(x: f64) -> f64 {
        if x >= 0.0 {
            if x <= WARM_U_POS {
                Self::warm_pos_poly(x).0
            } else {
                Self::warm_pos_poly(WARM_U_POS).0
            }
        } else if x >= -WARM_U_NEG {
            Self::warm_neg_poly(x).0
        } else {
            Self::warm_neg_poly(-WARM_U_NEG).0
        }
    }

    /// Exact closed-form 1st antiderivative `F_1(x) = \int_0^x warm_f(t) dt`.
    #[inline(always)]
    pub fn warm_f1(x: f64) -> f64 {
        if x >= 0.0 {
            if x <= WARM_U_POS {
                Self::warm_pos_poly(x).1
            } else {
                let (yp, f1p, _) = Self::warm_pos_poly(WARM_U_POS);
                f1p + yp * (x - WARM_U_POS)
            }
        } else if x >= -WARM_U_NEG {
            Self::warm_neg_poly(x).1
        } else {
            let (yn, f1n, _) = Self::warm_neg_poly(-WARM_U_NEG);
            f1n + yn * (x + WARM_U_NEG)
        }
    }

    /// Exact closed-form 2nd antiderivative `F_2(x) = \int_0^x F_1(t) dt`.
    #[inline(always)]
    pub fn warm_f2(x: f64) -> f64 {
        if x >= 0.0 {
            if x <= WARM_U_POS {
                Self::warm_pos_poly(x).2
            } else {
                let (yp, f1p, f2p) = Self::warm_pos_poly(WARM_U_POS);
                let d = x - WARM_U_POS;
                f2p + f1p * d + 0.5 * yp * d * d
            }
        } else if x >= -WARM_U_NEG {
            Self::warm_neg_poly(x).2
        } else {
            let (yn, f1n, f2n) = Self::warm_neg_poly(-WARM_U_NEG);
            let d = x + WARM_U_NEG;
            f2n + f1n * d + 0.5 * yn * d * d
        }
    }

    /// Evaluates the static memoryless saturation transfer curve (`1x` reference for alias testing).
    #[inline(always)]
    pub fn eval_transfer_curve(mode: CharacterMode, x: f64) -> f64 {
        match mode {
            CharacterMode::Clean => x,
            CharacterMode::Subtle => Self::subtle_f(x),
            CharacterMode::Warm => Self::warm_f(x),
        }
    }

    #[inline(always)]
    fn warm_adaa2_half_kernel(a: f64, b: f64) -> f64 {
        let delta = b - a;
        if delta.abs() < ADAA2_SEG_EPS {
            // Exact 2nd-order quadrature fallback when `|b - a| < 1e-5` (smooth to < 1e-10)
            0.5 * Self::warm_f((2.0 * a + b) * (1.0 / 3.0))
        } else {
            let f1_a = Self::warm_f1(a);
            let f2_a = Self::warm_f2(a);
            let f2_b = Self::warm_f2(b);
            (f2_b - f2_a - delta * f1_a) / (delta * delta)
        }
    }

    #[inline(always)]
    fn step_oversampled_subtle(&mut self, ch: usize, x: f64) -> f64 {
        let x1 = self.adaa_x1[ch];
        let f1_x0 = Self::subtle_f1(x);
        let delta = x - x1;
        let raw = if delta.abs() < ADAA_DELTA_EPS {
            Self::subtle_f(0.5 * (x + x1))
        } else {
            (f1_x0 - self.adaa_f1_x1[ch]) / delta
        };
        self.adaa_x1[ch] = x;
        self.adaa_f1_x1[ch] = f1_x0;

        // 3-tap linear-phase ADAA1 passband compensation `[-a, 1 + 2a, -a]` at `4x` rate
        let z1 = self.deblur_z1[ch];
        let z2 = self.deblur_z2[ch];
        self.deblur_z2[ch] = z1;
        self.deblur_z1[ch] = raw;
        (1.0 + 2.0 * ADAA1_DEBLUR_A) * z1 - ADAA1_DEBLUR_A * (raw + z2)
    }

    #[inline(always)]
    fn step_oversampled_warm(&mut self, ch: usize, x0: f64) -> f64 {
        let x1 = self.adaa_x1[ch];
        let x2 = self.adaa_x2[ch];
        let raw = if (x0 - x1).abs() < ADAA_DELTA_EPS && (x1 - x2).abs() < ADAA_DELTA_EPS {
            Self::warm_f(x1)
        } else {
            Self::warm_adaa2_half_kernel(x1, x2) + Self::warm_adaa2_half_kernel(x1, x0)
        };
        self.adaa_x2[ch] = x1;
        self.adaa_x1[ch] = x0;

        // 3-tap linear-phase ADAA2 passband compensation `[-a, 1 + 2a, -a]` at `4x` rate
        let z1 = self.deblur_z1[ch];
        let z2 = self.deblur_z2[ch];
        self.deblur_z2[ch] = z1;
        self.deblur_z1[ch] = raw;
        (1.0 + 2.0 * ADAA2_DEBLUR_A) * z1 - ADAA2_DEBLUR_A * (raw + z2)
    }

    #[inline(always)]
    pub fn process_sample(&mut self, mode: CharacterMode, ch: usize, x: f64) -> f64 {
        if matches!(mode, CharacterMode::Clean) {
            return x;
        }
        if self.active_mode != mode {
            // Re-prime antiderivative state when switching between Subtle and Warm
            self.active_mode = mode;
            for c in 0..2 {
                self.adaa_f1_x1[c] = Self::subtle_f1(self.adaa_x1[c]);
            }
        }

        let up = self.oversampler_4x.upsample_4x(ch, x);
        let mut shaped = [0.0f64; 4];
        match mode {
            CharacterMode::Clean => return x,
            CharacterMode::Subtle => {
                for i in 0..4 {
                    shaped[i] = self.step_oversampled_subtle(ch, up[i]);
                }
            }
            CharacterMode::Warm => {
                for i in 0..4 {
                    shaped[i] = self.step_oversampled_warm(ch, up[i]);
                }
            }
        }
        let sat = self.oversampler_4x.downsample_4x(ch, shaped);

        // 64-bit (`f64`) 5 Hz High-Pass DC Blocker removing asymmetric even-harmonic DC offset
        let y = self.dc_b0 * (sat - self.dc_x1[ch]) + self.dc_r * self.dc_y1[ch];
        self.dc_x1[ch] = sat;
        self.dc_y1[ch] = y;
        y
    }

    /// Processes a stereo frame `[l, r]` (`0` heap allocations).
    #[inline(always)]
    pub fn process_frame(&mut self, mode: CharacterMode, l: f64, r: f64) -> [f64; 2] {
        if matches!(mode, CharacterMode::Clean) {
            return [l, r];
        }
        [
            self.process_sample(mode, 0, l),
            self.process_sample(mode, 1, r),
        ]
    }

    /// Processes an in-place block of stereo `f32` frames (`0` heap allocations).
    #[inline]
    pub fn process_block(&mut self, mode: CharacterMode, frames: &mut [[f32; 2]]) {
        if matches!(mode, CharacterMode::Clean) {
            return;
        }
        for frame in frames.iter_mut() {
            let [yl, yr] = self.process_frame(mode, frame[0] as f64, frame[1] as f64);
            frame[0] = yl as f32;
            frame[1] = yr as f32;
        }
    }
}
