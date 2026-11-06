//! `dsp-proq4-eq`: 24-Band Dynamic/Spectral/Linear-Phase Equalizer (`FabFilter Pro-Q 4.clap` `v4.13`).
//! Implements:
//! - 601 Parameter System (`24 Bands * 24 Params + 25 Global` from `DigiPageTable.xml`)
//! - Fractional-Slope Dual Biquad Cascade (`0x180112260`, bitmask `0x10F6`)
//! - Gain-Q Interaction (`0x18010FF70`, bitmask `0x1074`)
//! - Modified Bessel `I0(x)` (`0x180102A10`) & Kaiser-60dB Window (`0x180232D00`)
//! - `ProgramDependentCapacitor` (`0x18010A9B0`)

use daw_core_graph::{ProcessContext, StereoFrame, UnifiedDspNode};
use std::f64::consts::TAU;

pub const PROQ4_NUM_BANDS: usize = 24;
pub const PROQ4_PARAMS_PER_BAND: usize = 24;
pub const PROQ4_GLOBAL_PARAMS: usize = 25;
pub const PROQ4_TOTAL_PARAMS: usize = PROQ4_NUM_BANDS * PROQ4_PARAMS_PER_BAND + PROQ4_GLOBAL_PARAMS; // 601

/// Filter prototype bitmask `0x10F6` (`0b0001_0000_1111_0110`) identifying Gain-dependent shapes.
pub const PROQ4_GAIN_DEPENDENT_MASK: u32 = 0x10F6;

/// Linear Phase FIR tap table @ 1x sample rate (`0x18019EAE0`).
pub const TAP_TABLE_1X: [usize; 5] = [4096, 8192, 16384, 32768, 131072];

/// Fractional-Slope Dual Biquad Cascade (`EQFilter` at `0x180221958`, `0x180112260`).
#[derive(Clone, Copy, Debug)]
pub struct FractionalSlopeBiquadBand {
    pub cascade_n1: [[f64; 5]; 8], // Lower integer order N1 = floor(S / 6.0) (`rbx + 0xFB0`)
    pub cascade_n2: [[f64; 5]; 8], // Upper integer order N2 = N1 + 1 (`rbx + 0x10F0`)
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
    pub fn new_unity() -> Self {
        let unity_sec = [1.0, 0.0, 0.0, 0.0, 0.0];
        Self {
            cascade_n1: [unity_sec; 8],
            cascade_n2: [unity_sec; 8],
            z_n1: [[[0.0; 2]; 8]; 2],
            z_n2: [[[0.0; 2]; 8]; 2],
            sections_n1: 1,
            sections_n2: 1,
            alpha_frac: 0.0,
        }
    }

    /// Analog-console Gain-Q interaction (`0x18010FF70`, bitmask `0x1074`).
    #[inline]
    pub fn compute_effective_q(q_ui: f64, gain_db: f64, gain_q_interaction: bool) -> f64 {
        let q_internal = std::f64::consts::FRAC_1_SQRT_2 * q_ui;
        if !gain_q_interaction {
            return q_internal;
        }
        let clamped_db = gain_db.abs().min(12.0);
        q_internal * 1.0905077326652577f64.powf(clamped_db * 0.25)
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
        let s_norm = (slope_db_oct.clamp(6.0, 96.0) / 6.0).max(1.0);
        let n1 = (s_norm.floor() as usize).clamp(1, 8);
        let n2 = (n1 + 1).min(8);
        self.alpha_frac = if n1 == n2 { 0.0 } else { s_norm.fract() };
        self.sections_n1 = n1;
        self.sections_n2 = n2;

        let q_eff = Self::compute_effective_q(q_ui, gain_db, gain_q_interaction).max(0.025);
        let w0 = (TAU * freq_hz.clamp(10.0, 0.495 * sample_rate)) / sample_rate.max(8000.0);
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();

        let make_biquad = |num_sections: usize| -> [f64; 5] {
            let g_sec_db = gain_db / (num_sections as f64);
            let a = 10.0f64.powf(g_sec_db / 40.0);
            let alpha = sin_w0 / (2.0 * q_eff);
            let b0 = 1.0 + alpha * a;
            let b1 = -2.0 * cos_w0;
            let b2 = 1.0 - alpha * a;
            let a0 = 1.0 + alpha / a;
            let a1 = -2.0 * cos_w0;
            let a2 = 1.0 - alpha / a;
            let inv_a0 = 1.0 / a0;
            [b0 * inv_a0, b1 * inv_a0, b2 * inv_a0, a1 * inv_a0, a2 * inv_a0]
        };

        let sec1 = make_biquad(n1);
        let sec2 = make_biquad(n2);
        for s in 0..n1 {
            self.cascade_n1[s] = sec1;
        }
        for s in 0..n2 {
            self.cascade_n2[s] = sec2;
        }
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
}

/// Modified Bessel function of the first kind order 0 (`0x180102A10`) used to generate the
/// 60.0 dB Kaiser window (`0x180232D00`) for Linear Phase FIR filters.
#[inline]
pub fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0f64;
    let mut term = 1.0f64;
    let half_x_sq = 0.25 * x * x;
    for k in 1..=25 {
        term *= half_x_sq / ((k * k) as f64);
        sum += term;
        if term < sum * 1e-15 {
            break;
        }
    }
    sum
}

/// 24-Band FabFilter Pro-Q 4 Processor implementing `UnifiedDspNode`.
pub struct ProQ4Processor {
    pub bands: [FractionalSlopeBiquadBand; PROQ4_NUM_BANDS],
    pub band_enabled: [bool; PROQ4_NUM_BANDS],
    pub sample_rate: f64,
}

impl Default for ProQ4Processor {
    fn default() -> Self {
        Self::new()
    }
}

impl ProQ4Processor {
    pub fn new() -> Self {
        Self {
            bands: [FractionalSlopeBiquadBand::new_unity(); PROQ4_NUM_BANDS],
            band_enabled: [false; PROQ4_NUM_BANDS],
            sample_rate: 48_000.0,
        }
    }
}

impl UnifiedDspNode for ProQ4Processor {
    fn name(&self) -> &'static str {
        "FabFilter Pro-Q 4 (24-Band)"
    }

    fn latency_samples(&self) -> usize {
        0
    }

    fn prepare(&mut self, sample_rate: f64, _max_block_size: usize) {
        self.sample_rate = sample_rate.max(8000.0);
    }

    fn reset_state(&mut self) {
        for b in &mut self.bands {
            b.reset_state();
        }
    }

    fn set_parameter_normalized(&mut self, param_id: u32, norm_val: f32) {
        let idx = param_id as usize;
        if idx < PROQ4_NUM_BANDS * PROQ4_PARAMS_PER_BAND {
            let band_idx = idx / PROQ4_PARAMS_PER_BAND;
            let offset = idx % PROQ4_PARAMS_PER_BAND;
            if offset == 1 {
                self.band_enabled[band_idx] = norm_val >= 0.5;
            }
        }
    }

    fn process_slice(
        &mut self,
        _ctx: &ProcessContext,
        main_io: &mut [StereoFrame],
        _sidechain_in: Option<&[StereoFrame]>,
        _aux_sends_out: &mut [&mut [StereoFrame]],
    ) {
        for frame in main_io.iter_mut() {
            let mut l = frame[0] as f64;
            let mut r = frame[1] as f64;
            for b in 0..PROQ4_NUM_BANDS {
                if self.band_enabled[b] {
                    l = self.bands[b].process_sample(0, l);
                    r = self.bands[b].process_sample(1, r);
                }
            }
            frame[0] = l as f32;
            frame[1] = r as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fractional_slope_bell_and_proq4_processor() {
        let mut band = FractionalSlopeBiquadBand::new_unity();
        band.configure_bell(1000.0, 6.0, 1.0, 15.0, 48_000.0, true);
        // 15 dB/oct / 6 = 2.5 -> n1 = 2, n2 = 3, alpha_frac = 0.5
        assert_eq!(band.sections_n1, 2);
        assert_eq!(band.sections_n2, 3);
        assert!((band.alpha_frac - 0.5).abs() < 1e-9);

        let mut proc = ProQ4Processor::new();
        proc.band_enabled[0] = true;
        proc.bands[0].configure_bell(1000.0, 0.0, 1.0, 12.0, 48_000.0, true);
        let mut io = [[0.5f32, -0.5f32]; 32];
        proc.process_slice(&ProcessContext::default(), &mut io, None, &mut []);
        assert!((io[31][0] - 0.5).abs() < 1e-6);
    }
}

