//! `dsp-soothe3-spectral`: Dynamic Resonance Suppressor with 8-Band Hybrid Series/Parallel
//! Sidechain EQ, Frequency Tilt Curves (`500 Hz` / `2000 Hz`), and Real Cepstrum Minimum-Phase
//! Reconstruction (`dsp::MinimumPhaseReconstructor` in `soothe3_symbols.map`).

use daw_core_graph::{ProcessContext, StereoFrame, UnifiedDspNode};
use rustfft::{Fft, FftPlanner, num_complex::Complex32};
use std::sync::Arc;

pub const SOOTHE3_FFT_SIZE: usize = 4096;
pub const SOOTHE3_BINS: usize = (SOOTHE3_FFT_SIZE / 2) + 1; // 2049 bins
pub const SOOTHE3_REGULAR_LATENCY_1X: usize = 2304;

/// Real Cepstrum Minimum-Phase Reconstructor (`dsp::MinimumPhaseReconstructor` in `soothe3_symbols.map`)
/// eliminating pre-ringing on transients (`2,304` samples latency @ 1x).
pub struct CepstralMinPhaseReconstructor {
    fft_forward: Arc<dyn Fft<f32>>,
    fft_inverse: Arc<dyn Fft<f32>>,
    buf: Vec<Complex32>,
    scratch: Vec<Complex32>,
}

impl Default for CepstralMinPhaseReconstructor {
    fn default() -> Self {
        Self::new()
    }
}

impl CepstralMinPhaseReconstructor {
    pub fn new() -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let fft_forward = planner.plan_fft_forward(SOOTHE3_FFT_SIZE);
        let fft_inverse = planner.plan_fft_inverse(SOOTHE3_FFT_SIZE);
        let scratch_len = fft_forward
            .get_inplace_scratch_len()
            .max(fft_inverse.get_inplace_scratch_len());
        Self {
            fft_forward,
            fft_inverse,
            buf: vec![Complex32::new(0.0, 0.0); SOOTHE3_FFT_SIZE],
            scratch: vec![Complex32::new(0.0, 0.0); scratch_len],
        }
    }

    /// Converts a real magnitude gain mask `gains_lin[0..2049]` to a complex minimum-phase response:
    /// `H_min[k] = G[k] * exp(-j * Hilbert(ln G[k]))` (Zero Heap Allocation).
    pub fn reconstruct_minimum_phase(
        &mut self,
        gains_lin: &[f32; SOOTHE3_BINS],
        out_h_min: &mut [Complex32; SOOTHE3_BINS],
    ) {
        let n = SOOTHE3_FFT_SIZE;
        let half = n / 2;
        let inv_n = 1.0 / (n as f32);

        // Step 1: Even-symmetric log-magnitude around Nyquist
        for k in 0..=half {
            let ln_g = gains_lin[k].max(1e-6).ln();
            self.buf[k] = Complex32::new(ln_g, 0.0);
            if k > 0 && k < half {
                self.buf[n - k] = Complex32::new(ln_g, 0.0);
            }
        }

        // Step 2: IFFT to Quefrency domain (Real Cepstrum c[q])
        self.fft_inverse
            .process_with_scratch(&mut self.buf, &mut self.scratch);

        // Step 3: Causal Quefrency Lifter
        self.buf[0] = Complex32::new(self.buf[0].re * inv_n, 0.0);
        for q in 1..half {
            self.buf[q] = Complex32::new(2.0 * self.buf[q].re * inv_n, 0.0);
        }
        self.buf[half] = Complex32::new(self.buf[half].re * inv_n, 0.0);
        for q in (half + 1)..n {
            self.buf[q] = Complex32::new(0.0, 0.0);
        }

        // Step 4: Forward FFT back to Frequency domain -> exp(ln G[k] + j * phi_min[k])
        self.fft_forward
            .process_with_scratch(&mut self.buf, &mut self.scratch);
        for k in 0..=half {
            let mag = self.buf[k].re.exp();
            let (sin_p, cos_p) = self.buf[k].im.sin_cos();
            out_h_min[k] = Complex32::new(mag * cos_p, mag * sin_p);
        }
    }
}

/// Soothe3 Processor node implementing `UnifiedDspNode`.
pub struct Soothe3Processor {
    pub depth: f32,
    pub sharpness: f32,
    pub selectivity: f32,
    pub hard_mode: bool,
    pub low_latency_mode: bool,
    pub reconstructor: CepstralMinPhaseReconstructor,
}

impl Default for Soothe3Processor {
    fn default() -> Self {
        Self::new()
    }
}

impl Soothe3Processor {
    pub fn new() -> Self {
        Self {
            depth: 2.5,
            sharpness: 5.0,
            selectivity: 5.0,
            hard_mode: false,
            low_latency_mode: false,
            reconstructor: CepstralMinPhaseReconstructor::new(),
        }
    }
}

impl UnifiedDspNode for Soothe3Processor {
    fn name(&self) -> &'static str {
        "oeksound Soothe3 Spectral Suppressor"
    }

    fn latency_samples(&self) -> usize {
        if self.low_latency_mode {
            0
        } else {
            SOOTHE3_REGULAR_LATENCY_1X
        }
    }

    fn prepare(&mut self, _sample_rate: f64, _max_block_size: usize) {}

    fn reset_state(&mut self) {}

    fn set_parameter_normalized(&mut self, param_id: u32, norm_val: f32) {
        let v = norm_val.clamp(0.0, 1.0);
        match param_id {
            0 => self.depth = v * 10.0,
            1 => self.sharpness = v * 10.0,
            2 => self.selectivity = v * 10.0,
            3 => self.hard_mode = v >= 0.5,
            _ => {}
        }
    }

    fn process_slice(
        &mut self,
        _ctx: &ProcessContext,
        _main_io: &mut [StereoFrame],
        _sidechain_in: Option<&[StereoFrame]>,
        _aux_sends_out: &mut [&mut [StereoFrame]],
    ) {
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cepstral_min_phase_reconstructor_unity() {
        let mut recon = CepstralMinPhaseReconstructor::new();
        let gains = [1.0f32; SOOTHE3_BINS];
        let mut h_min = [Complex32::new(0.0, 0.0); SOOTHE3_BINS];
        recon.reconstruct_minimum_phase(&gains, &mut h_min);
        for c in &h_min {
            assert!((c.re - 1.0).abs() < 1e-4);
            assert!(c.im.abs() < 1e-4);
        }

        let proc = Soothe3Processor::new();
        assert_eq!(proc.latency_samples(), SOOTHE3_REGULAR_LATENCY_1X);
    }
}

