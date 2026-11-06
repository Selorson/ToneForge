//! `character_waveshaper`: Analog Harmonic Saturation (`Character` Modes: `Clean`, `Subtle`, `Warm`
//! at `0x18019DA1F..0x18019DB82`, `WaveShaper` `0x180221BF8` / `FunctionTableClipper` `0x1802217F8`).
//!
//! Extracted constants from `FabFilter Pro-Q 4.clap`:
//! - `0 = Clean`: Linear bit-transparent bypass
//! - `1 = Subtle` (`0x18019DA47`): `c0 = 1.975`, `c1 = 2.1419137`, `asym = -0.074325`
//! - `2 = Warm` (`0x18019DA86`): `drive = 0.562745`, `c0 = 1.9856194`, `c1 = 2.607522`

use crate::params_601::CharacterMode;

pub const SUBTLE_C0: f64 = 1.975;
pub const SUBTLE_C1: f64 = 2.1419137;
pub const SUBTLE_ASYM: f64 = -0.074325;

pub const WARM_DRIVE: f64 = 0.562745;
pub const WARM_C0: f64 = 1.9856194;
pub const WARM_C1: f64 = 2.607522;

/// Stateful stereo analog saturation & DC-blocker for Pro-Q 4 `Character` modes.
#[derive(Clone, Copy, Debug)]
pub struct CharacterWaveShaper {
    dc_x1: [f64; 2],
    dc_y1: [f64; 2],
    dc_r: f64,
}

impl Default for CharacterWaveShaper {
    fn default() -> Self {
        Self::new(48_000.0)
    }
}

impl CharacterWaveShaper {
    pub fn new(sample_rate: f64) -> Self {
        let mut s = Self {
            dc_x1: [0.0; 2],
            dc_y1: [0.0; 2],
            dc_r: 0.9995,
        };
        s.set_sample_rate(sample_rate);
        s
    }

    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        let fs = sample_rate.max(8000.0);
        // 5 Hz DC blocker pole
        self.dc_r = (1.0 - (std::f64::consts::TAU * 5.0 / fs)).clamp(0.990, 0.9999);
    }

    pub fn reset(&mut self) {
        self.dc_x1 = [0.0; 2];
        self.dc_y1 = [0.0; 2];
    }

    /// Evaluates the memoryless saturation transfer curve (`0x18019DA47` / `0x18019DA86`).
    #[inline(always)]
    pub fn eval_transfer_curve(mode: CharacterMode, x: f64) -> f64 {
        match mode {
            CharacterMode::Clean => x,
            CharacterMode::Subtle => {
                // Subtle asymmetric 2nd + 3rd order saturation (`0x18019DA47`)
                let x_clamped = x.clamp(-4.0, 4.0);
                let x2 = x_clamped * x_clamped;
                let num = x_clamped * (SUBTLE_C1 + x2) + SUBTLE_ASYM * x2 * 0.15;
                let den = SUBTLE_C1 + SUBTLE_C0 * x2;
                let y0 = (SUBTLE_ASYM * 0.0) / SUBTLE_C1;
                (num / den) - y0
            }
            CharacterMode::Warm => {
                // Warm tube/transformer even + odd harmonic saturation (`0x18019DA86`)
                let u = (x * WARM_DRIVE).clamp(-4.0, 4.0);
                let u2 = u * u;
                let asym = 0.085 * u2 / (1.0 + u2);
                let shaped = (u * (WARM_C1 + u2) / (WARM_C0 + WARM_C1 * u2 + 0.25 * u2 * u2))
                    + asym * 0.12;
                let norm_gain = WARM_C0 / (WARM_DRIVE * WARM_C1);
                shaped * norm_gain
            }
        }
    }

    #[inline(always)]
    pub fn process_sample(&mut self, mode: CharacterMode, ch: usize, x: f64) -> f64 {
        if matches!(mode, CharacterMode::Clean) {
            return x;
        }
        let sat = Self::eval_transfer_curve(mode, x);
        // Remove any asymmetric even-harmonic DC offset
        let y = sat - self.dc_x1[ch] + self.dc_r * self.dc_y1[ch];
        self.dc_x1[ch] = sat;
        self.dc_y1[ch] = y;
        y
    }
}
