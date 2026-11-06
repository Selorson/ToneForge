//! Base Chain Element Trait (`DSP::ElementImpl` & `DSP::OzoneChainElement`) and
//! Phase-Coherent 4-Band Linkwitz-Riley (`LR4`, `24 dB/oct`) Crossover Network.

use daw_core_graph::StereoFrame;
use std::f64::consts::{FRAC_1_SQRT_2, PI};

/// Number of frequency bands in Ozone 12 multi-band processors (`Exciter`, `Imager`, `Impact`, `Maximizer`).
pub const OZONE_MULTIBAND_COUNT: usize = 4;

/// Default 4-band crossover frequencies in Hz (`[Low/LowMid, LowMid/HighMid, HighMid/High]`).
pub const DEFAULT_CROSSOVER_FREQS_HZ: [f64; 3] = [120.0, 1000.0, 6000.0];

/// Base trait equivalent to `DSP::ElementImpl` (`VT = 0x1856DF158`) and
/// `DSP::OzoneChainElement` (`VT = 0x18572E258`) in `iZOzone12Core.dll`.
pub trait OzoneDspModule: Send + Sync {
    /// Module identifier name.
    fn name(&self) -> &'static str;

    /// Resets and pre-allocates state (`vfn[58]` / `vfn[61]`).
    fn reset(&mut self, sample_rate: f64, max_block_size: usize);

    /// Processes split stereo buffers in-place (`vfn[59]` / `vfn[62]`, Zero Heap Allocation).
    fn process_block(&mut self, left: &mut [f32], right: &mut [f32]);

    /// Processes interleaved stereo frames in-place (Zero Heap Allocation).
    #[inline]
    fn process_frames(&mut self, frames: &mut [StereoFrame]) {
        // Process in fixed stack chunks to avoid any heap allocation when bridging split/interleaved
        const CHUNK: usize = 256;
        let mut l_buf = [0.0f32; CHUNK];
        let mut r_buf = [0.0f32; CHUNK];
        let mut offset = 0;
        let total = frames.len();
        while offset < total {
            let n = (total - offset).min(CHUNK);
            for i in 0..n {
                l_buf[i] = frames[offset + i][0];
                r_buf[i] = frames[offset + i][1];
            }
            self.process_block(&mut l_buf[..n], &mut r_buf[..n]);
            for i in 0..n {
                frames[offset + i][0] = l_buf[i];
                frames[offset + i][1] = r_buf[i];
            }
            offset += n;
        }
    }

    /// Intrinsic processing latency in samples at the current sample rate.
    fn latency_samples(&self) -> usize {
        0
    }
}

/// Double-precision Direct-Form II Transposed Stereo Biquad filter (Zero Heap Allocation).
#[derive(Clone, Copy, Debug)]
pub struct StereoBiquadF64 {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
    pub z1: [f64; 2],
    pub z2: [f64; 2],
}

impl Default for StereoBiquadF64 {
    fn default() -> Self {
        Self::new_unity()
    }
}

impl StereoBiquadF64 {
    #[inline]
    pub const fn new_unity() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            z1: [0.0; 2],
            z2: [0.0; 2],
        }
    }

    #[inline]
    pub fn reset_state(&mut self) {
        self.z1 = [0.0; 2];
        self.z2 = [0.0; 2];
    }

    /// Configures a 2nd-order Butterworth Lowpass (`Q = 1 / sqrt(2)`) used as half of an LR4 lowpass.
    #[inline]
    pub fn set_butterworth_lp(&mut self, cutoff_hz: f64, sample_rate: f64) {
        let sr = sample_rate.max(8000.0);
        let fc = cutoff_hz.clamp(10.0, 0.49 * sr);
        let k = (PI * fc / sr).tan();
        let k2 = k * k;
        let sqrt2_k = std::f64::consts::SQRT_2 * k;
        let norm = 1.0 / (1.0 + sqrt2_k + k2);
        self.b0 = k2 * norm;
        self.b1 = 2.0 * self.b0;
        self.b2 = self.b0;
        self.a1 = 2.0 * (k2 - 1.0) * norm;
        self.a2 = (1.0 - sqrt2_k + k2) * norm;
    }

    /// Configures a 2nd-order Butterworth Highpass (`Q = 1 / sqrt(2)`) used as half of an LR4 highpass.
    #[inline]
    pub fn set_butterworth_hp(&mut self, cutoff_hz: f64, sample_rate: f64) {
        let sr = sample_rate.max(8000.0);
        let fc = cutoff_hz.clamp(10.0, 0.49 * sr);
        let k = (PI * fc / sr).tan();
        let k2 = k * k;
        let sqrt2_k = std::f64::consts::SQRT_2 * k;
        let norm = 1.0 / (1.0 + sqrt2_k + k2);
        self.b0 = norm;
        self.b1 = -2.0 * self.b0;
        self.b2 = self.b0;
        self.a1 = 2.0 * (k2 - 1.0) * norm;
        self.a2 = (1.0 - sqrt2_k + k2) * norm;
    }

    /// Configures the exact 2nd-order Butterworth All-Pass filter `A_fc(z) = LP_LR4(z) + HP_LR4(z)`.
    /// Satisfies `|A_fc(e^{j*w})| == 1.0` for all `w` and matches the exact phase of an LR4 split at `cutoff_hz`.
    #[inline]
    pub fn set_lr4_phase_allpass(&mut self, cutoff_hz: f64, sample_rate: f64) {
        let sr = sample_rate.max(8000.0);
        let fc = cutoff_hz.clamp(10.0, 0.49 * sr);
        let k = (PI * fc / sr).tan();
        let k2 = k * k;
        let sqrt2_k = std::f64::consts::SQRT_2 * k;
        let norm = 1.0 / (1.0 + sqrt2_k + k2);
        let a1 = 2.0 * (k2 - 1.0) * norm;
        let a2 = (1.0 - sqrt2_k + k2) * norm;
        self.b0 = a2;
        self.b1 = a1;
        self.b2 = 1.0;
        self.a1 = a1;
        self.a2 = a2;
    }

    /// Configures a peaking / bell EQ filter (`gain_db`, `q`, `freq_hz`).
    #[inline]
    pub fn set_peaking_eq(&mut self, freq_hz: f64, q: f64, gain_db: f64, sample_rate: f64) {
        let sr = sample_rate.max(8000.0);
        let fc = freq_hz.clamp(15.0, 0.48 * sr);
        let q_safe = q.clamp(0.1, 30.0);
        let a = 10.0f64.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * fc / sr;
        let (sin_w0, cos_w0) = w0.sin_cos();
        let alpha = sin_w0 / (2.0 * q_safe);

        let b0 = 1.0 + alpha * a;
        let b1 = -2.0 * cos_w0;
        let b2 = 1.0 - alpha * a;
        let a0 = 1.0 + alpha / a;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha / a;
        let inv_a0 = 1.0 / a0;

        self.b0 = b0 * inv_a0;
        self.b1 = b1 * inv_a0;
        self.b2 = b2 * inv_a0;
        self.a1 = a1 * inv_a0;
        self.a2 = a2 * inv_a0;
    }

    /// Configures a high-shelf filter (`gain_db`, `q`, `freq_hz`).
    #[inline]
    pub fn set_high_shelf(&mut self, freq_hz: f64, q: f64, gain_db: f64, sample_rate: f64) {
        let sr = sample_rate.max(8000.0);
        let fc = freq_hz.clamp(20.0, 0.48 * sr);
        let q_safe = q.clamp(0.1, 10.0);
        let a = 10.0f64.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * fc / sr;
        let (sin_w0, cos_w0) = w0.sin_cos();
        let alpha = sin_w0 / (2.0 * q_safe);
        let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;

        let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha);
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
        let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha);
        let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
        let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
        let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha;
        let inv_a0 = 1.0 / a0;

        self.b0 = b0 * inv_a0;
        self.b1 = b1 * inv_a0;
        self.b2 = b2 * inv_a0;
        self.a1 = a1 * inv_a0;
        self.a2 = a2 * inv_a0;
    }

    /// Configures a low-shelf filter (`gain_db`, `q`, `freq_hz`).
    #[inline]
    pub fn set_low_shelf(&mut self, freq_hz: f64, q: f64, gain_db: f64, sample_rate: f64) {
        let sr = sample_rate.max(8000.0);
        let fc = freq_hz.clamp(15.0, 0.48 * sr);
        let q_safe = q.clamp(0.1, 10.0);
        let a = 10.0f64.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * fc / sr;
        let (sin_w0, cos_w0) = w0.sin_cos();
        let alpha = sin_w0 / (2.0 * q_safe);
        let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;

        let b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha);
        let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0);
        let b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha);
        let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
        let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0);
        let a2 = (a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha;
        let inv_a0 = 1.0 / a0;

        self.b0 = b0 * inv_a0;
        self.b1 = b1 * inv_a0;
        self.b2 = b2 * inv_a0;
        self.a1 = a1 * inv_a0;
        self.a2 = a2 * inv_a0;
    }

    #[inline]
    pub fn reset(&mut self) {
        self.reset_state();
    }

    /// Configures a 2nd-order Band-Pass filter (`H(s) = (s / Q) / (s^2 + s / Q + 1)`).
    #[inline]
    pub fn set_bandpass(&mut self, freq_hz: f64, q: f64, sample_rate: f64) {
        let sr = sample_rate.max(8000.0);
        let fc = freq_hz.clamp(15.0, 0.48 * sr);
        let q_safe = q.clamp(0.1, 30.0);
        let w0 = 2.0 * PI * fc / sr;
        let (sin_w0, cos_w0) = w0.sin_cos();
        let alpha = sin_w0 / (2.0 * q_safe);
        let inv_a0 = 1.0 / (1.0 + alpha);

        self.b0 = alpha * inv_a0;
        self.b1 = 0.0;
        self.b2 = -alpha * inv_a0;
        self.a1 = -2.0 * cos_w0 * inv_a0;
        self.a2 = (1.0 - alpha) * inv_a0;
    }

    /// Configures a 2nd-order All-Pass filter at `freq_hz` with quality factor `q`.
    #[inline]
    pub fn set_allpass_2nd(&mut self, freq_hz: f64, q: f64, sample_rate: f64) {
        let sr = sample_rate.max(8000.0);
        let fc = freq_hz.clamp(15.0, 0.48 * sr);
        let q_safe = q.clamp(0.1, 30.0);
        let w0 = 2.0 * PI * fc / sr;
        let (sin_w0, cos_w0) = w0.sin_cos();
        let alpha = sin_w0 / (2.0 * q_safe);
        let inv_a0 = 1.0 / (1.0 + alpha);

        self.b0 = (1.0 - alpha) * inv_a0;
        self.b1 = -2.0 * cos_w0 * inv_a0;
        self.b2 = 1.0;
        self.a1 = self.b1;
        self.a2 = self.b0;
    }

    #[inline(always)]
    pub fn step_channel(&mut self, ch: usize, x: f64) -> f64 {
        let y = self.b0 * x + self.z1[ch];
        let z1_next = self.b1 * x - self.a1 * y + self.z2[ch];
        let z2_next = self.b2 * x - self.a2 * y;
        self.z1[ch] = if z1_next.abs() < 1e-20 { 0.0 } else { z1_next };
        self.z2[ch] = if z2_next.abs() < 1e-20 { 0.0 } else { z2_next };
        y
    }

    #[inline(always)]
    pub fn step_stereo(&mut self, frame: [f64; 2]) -> [f64; 2] {
        [
            self.step_channel(0, frame[0]),
            self.step_channel(1, frame[1]),
        ]
    }

    #[inline(always)]
    pub fn tick(&mut self, l: f64, r: f64) -> (f64, f64) {
        (self.step_channel(0, l), self.step_channel(1, r))
    }
}

/// 2-Way Linkwitz-Riley 4th-Order (`24 dB/oct`) Splitter at a single cutoff frequency `f_c`.
/// Consists of two cascaded Butterworth 2nd-order LPFs and two cascaded Butterworth 2nd-order HPFs.
#[derive(Clone, Copy, Debug, Default)]
pub struct LinkwitzRiley4Splitter2Way {
    lp1: StereoBiquadF64,
    lp2: StereoBiquadF64,
    hp1: StereoBiquadF64,
    hp2: StereoBiquadF64,
}

impl LinkwitzRiley4Splitter2Way {
    pub fn new(cutoff_hz: f64, sample_rate: f64) -> Self {
        let mut s = Self::default();
        s.set_cutoff(cutoff_hz, sample_rate);
        s
    }

    #[inline]
    pub fn set_cutoff(&mut self, cutoff_hz: f64, sample_rate: f64) {
        self.lp1.set_butterworth_lp(cutoff_hz, sample_rate);
        self.lp2.set_butterworth_lp(cutoff_hz, sample_rate);
        self.hp1.set_butterworth_hp(cutoff_hz, sample_rate);
        self.hp2.set_butterworth_hp(cutoff_hz, sample_rate);
    }

    #[inline]
    pub fn reset(&mut self) {
        self.lp1.reset_state();
        self.lp2.reset_state();
        self.hp1.reset_state();
        self.hp2.reset_state();
    }

    /// Splits input `[L, R]` into `(low, high)` 4th-order Linkwitz-Riley components.
    /// Their sum `low + high` equals the exact 2nd-order All-Pass response `A_fc(z)`.
    #[inline(always)]
    pub fn split_stereo(&mut self, input: [f64; 2]) -> ([f64; 2], [f64; 2]) {
        let l1 = self.lp1.step_stereo(input);
        let low = self.lp2.step_stereo(l1);
        let h1 = self.hp1.step_stereo(input);
        let high = self.hp2.step_stereo(h1);
        (low, high)
    }
}

/// Phase-Coherent 4-Band Linkwitz-Riley 4th-Order (`24 dB/oct`) Crossover Network.
///
/// Tree topology:
/// 1. Mid split at `f2` into `low12` and `high34`.
/// 2. `low12` split at `f1` into `Band 0 (Low)` and `Band 1 (Low-Mid)`, phase-aligned with `A_f3(z)`.
/// 3. `high34` split at `f3` into `Band 2 (High-Mid)` and `Band 3 (High)`, phase-aligned with `A_f1(z)`.
///
/// Algebraically guarantees `|Band0 + Band1 + Band2 + Band3| === 1.0` (`0.000000 dB` ripple)
/// across the entire Nyquist range `[0, fs/2]`.
#[derive(Clone, Debug)]
pub struct LinkwitzRiley4BandCrossover {
    pub cutoffs_hz: [f64; 3],
    pub sample_rate: f64,
    split_mid_f2: LinkwitzRiley4Splitter2Way,
    split_low_f1: LinkwitzRiley4Splitter2Way,
    split_high_f3: LinkwitzRiley4Splitter2Way,
    allpass_f3_band0: StereoBiquadF64,
    allpass_f3_band1: StereoBiquadF64,
    allpass_f1_band2: StereoBiquadF64,
    allpass_f1_band3: StereoBiquadF64,
}

impl Default for LinkwitzRiley4BandCrossover {
    fn default() -> Self {
        Self::new(DEFAULT_CROSSOVER_FREQS_HZ, 48_000.0)
    }
}

impl LinkwitzRiley4BandCrossover {
    pub fn new(cutoffs_hz: [f64; 3], sample_rate: f64) -> Self {
        let mut xover = Self {
            cutoffs_hz,
            sample_rate: sample_rate.max(8000.0),
            split_mid_f2: LinkwitzRiley4Splitter2Way::default(),
            split_low_f1: LinkwitzRiley4Splitter2Way::default(),
            split_high_f3: LinkwitzRiley4Splitter2Way::default(),
            allpass_f3_band0: StereoBiquadF64::new_unity(),
            allpass_f3_band1: StereoBiquadF64::new_unity(),
            allpass_f1_band2: StereoBiquadF64::new_unity(),
            allpass_f1_band3: StereoBiquadF64::new_unity(),
        };
        xover.configure(cutoffs_hz, sample_rate);
        xover
    }

    pub fn configure(&mut self, cutoffs_hz: [f64; 3], sample_rate: f64) {
        let sr = sample_rate.max(8000.0);
        let f1 = cutoffs_hz[0].clamp(20.0, 0.45 * sr);
        let f2 = cutoffs_hz[1].clamp(f1 + 20.0, 0.46 * sr);
        let f3 = cutoffs_hz[2].clamp(f2 + 50.0, 0.48 * sr);
        self.cutoffs_hz = [f1, f2, f3];
        self.sample_rate = sr;

        self.split_mid_f2.set_cutoff(f2, sr);
        self.split_low_f1.set_cutoff(f1, sr);
        self.split_high_f3.set_cutoff(f3, sr);

        self.allpass_f3_band0.set_lr4_phase_allpass(f3, sr);
        self.allpass_f3_band1.set_lr4_phase_allpass(f3, sr);
        self.allpass_f1_band2.set_lr4_phase_allpass(f1, sr);
        self.allpass_f1_band3.set_lr4_phase_allpass(f1, sr);
    }

    pub fn reset(&mut self) {
        self.split_mid_f2.reset();
        self.split_low_f1.reset();
        self.split_high_f3.reset();
        self.allpass_f3_band0.reset_state();
        self.allpass_f3_band1.reset_state();
        self.allpass_f1_band2.reset_state();
        self.allpass_f1_band3.reset_state();
    }

    /// Splits a single stereo frame `[L, R]` into 4 phase-coherent bands `[Low, LowMid, HighMid, High]`.
    #[inline(always)]
    pub fn split_frame_f64(&mut self, input: [f64; 2]) -> [[f64; 2]; OZONE_MULTIBAND_COUNT] {
        let (low12, high34) = self.split_mid_f2.split_stereo(input);
        let (b0_raw, b1_raw) = self.split_low_f1.split_stereo(low12);
        let (b2_raw, b3_raw) = self.split_high_f3.split_stereo(high34);

        let b0 = self.allpass_f3_band0.step_stereo(b0_raw);
        let b1 = self.allpass_f3_band1.step_stereo(b1_raw);
        let b2 = self.allpass_f1_band2.step_stereo(b2_raw);
        let b3 = self.allpass_f1_band3.step_stereo(b3_raw);

        [b0, b1, b2, b3]
    }

    /// Splits a single `f32` stereo frame `[L, R]` into 4 phase-coherent bands.
    #[inline(always)]
    pub fn split_frame(&mut self, input: StereoFrame) -> [StereoFrame; OZONE_MULTIBAND_COUNT] {
        let bands = self.split_frame_f64([input[0] as f64, input[1] as f64]);
        [
            [bands[0][0] as f32, bands[0][1] as f32],
            [bands[1][0] as f32, bands[1][1] as f32],
            [bands[2][0] as f32, bands[2][1] as f32],
            [bands[3][0] as f32, bands[3][1] as f32],
        ]
    }

    #[inline]
    pub fn set_crossover_frequencies(&mut self, sample_rate: f64, f1: f64, f2: f64, f3: f64) {
        self.configure([f1, f2, f3], sample_rate);
    }

    #[inline(always)]
    pub fn split_stereo(&mut self, l: f64, r: f64) -> [[f64; 2]; OZONE_MULTIBAND_COUNT] {
        self.split_frame_f64([l, r])
    }

    /// Evaluates the all-pass reference path `A_f1(A_f2(A_f3(x)))` if needed for parallel dry comparison.
    #[inline]
    pub const fn inv_sqrt2() -> f64 {
        FRAC_1_SQRT_2
    }
}
