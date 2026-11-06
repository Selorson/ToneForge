//! iZotope Ozone 12 Multiband Imager (`iZOzone12Imager.dll`, `DSP::MultibandImager` at `0x181DF4200`).
//!
//! Features:
//! - 4-Band Linkwitz-Riley 4th-order (24 dB/oct) phase-compensated M/S width control (`[-100%, +100%]`).
//! - Phase-Safe Mono-Compatible Stereoizer (Mode I: Velvet-Noise FIR decorrelator;
//!   Mode II: Cascaded Schroeder All-Pass phase decorrelator). Because the synthesized
//!   side component `D_b[n]` is injected in anti-phase (`L' = L + g*D`, `R' = R - g*D`),
//!   summing to mono yields `L' + R' == L + R` with 0.0 dB comb-filtering cancellation.
//! - `Recover Sides`: Recovers attenuated side energy `(1 - w_b)` through a 90-degree phase-shifted
//!   all-pass network into Mid when narrowing stereo width (`w_b < 1.0`).
//! - Per-band real-time Pearson phase correlation meter (`imager_corr_bands: [f32; 4]`).
//! - Strictly 0 heap allocations on the real-time audio thread.

use crate::chain_element::{LinkwitzRiley4BandCrossover, OzoneDspModule, StereoBiquadF64};
use daw_core_graph::{ProcessContext, StereoFrame, UnifiedDspNode};

/// Stereoizer synthesis mode (`AutomatableParamList_Imager.txt`: `Stereoizer Mode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StereoizerMode {
    /// Mode I: Sparse Velvet-Noise FIR decorrelator for crisp transient preservation.
    ModeI = 0,
    /// Mode II: Cascaded All-Pass Schroeder phase decorrelator for smooth ambient width.
    ModeII = 1,
}

impl StereoizerMode {
    #[inline]
    pub fn from_index(idx: usize) -> Self {
        match idx {
            1 => Self::ModeII,
            _ => Self::ModeI,
        }
    }
}

/// Fixed-capacity circular delay line used by the Velvet-Noise / All-Pass Stereoizer.
#[derive(Clone, Debug)]
struct FixedMonoDelay {
    buf: Vec<f64>,
    write_pos: usize,
    mask: usize,
}

impl FixedMonoDelay {
    fn new(capacity_pow2: usize) -> Self {
        let cap = capacity_pow2.max(64).next_power_of_two();
        Self {
            buf: vec![0.0; cap],
            write_pos: 0,
            mask: cap - 1,
        }
    }

    fn reset(&mut self) {
        self.buf.fill(0.0);
        self.write_pos = 0;
    }

    #[inline(always)]
    fn push(&mut self, x: f64) {
        self.buf[self.write_pos] = x;
        self.write_pos = (self.write_pos + 1) & self.mask;
    }

    #[inline(always)]
    fn read_tap(&self, delay_samples: usize) -> f64 {
        let idx = self.write_pos.wrapping_sub(delay_samples.max(1)) & self.mask;
        self.buf[idx]
    }
}

/// First-order all-pass stage for `Recover Sides` 90-degree quadrature shift and Mode II Stereoizer.
#[derive(Clone, Copy, Debug, Default)]
struct FirstOrderAllPass {
    coeff: f64,
    x1: f64,
    y1: f64,
}

impl FirstOrderAllPass {
    #[inline]
    fn new(coeff: f64) -> Self {
        Self {
            coeff,
            x1: 0.0,
            y1: 0.0,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.x1 = 0.0;
        self.y1 = 0.0;
    }

    #[inline(always)]
    fn tick(&mut self, x: f64) -> f64 {
        let y = self.coeff * (x - self.y1) + self.x1;
        self.x1 = x;
        self.y1 = y;
        y
    }
}

/// Per-band configuration for the 4-band Imager.
#[derive(Clone, Copy, Debug)]
pub struct ImagerBandConfig {
    /// Width percentage in `[-100.0, +100.0]` (`0.0` = unity `w=1.0`, `-100.0` = mono `w=0.0`, `+100.0` = `w=2.0`).
    pub width_percent: f32,
    /// Stereoize synthesis amount in `[0.0, 100.0]`%.
    pub stereoize_percent: f32,
    /// Recover Sides amount in `[0.0, 100.0]`% when narrowing (`width_percent < 0.0`).
    pub recover_sides_percent: f32,
    /// Band mute flag.
    pub mute: bool,
    /// Band solo flag.
    pub solo: bool,
}

impl Default for ImagerBandConfig {
    fn default() -> Self {
        Self {
            width_percent: 0.0,
            stereoize_percent: 0.0,
            recover_sides_percent: 0.0,
            mute: false,
            solo: false,
        }
    }
}

/// iZotope Ozone 12 4-Band Stereo Imager (`MultibandImagerModule` / `OzoneImager`).
#[derive(Clone, Debug)]
pub struct MultibandImagerModule {
    pub enabled: bool,
    pub multiband_enabled: bool,
    pub stereoizer_enable: bool,
    pub stereoizer_mode: StereoizerMode,
    /// Global Stereoize Haas/Velvet delay width in milliseconds (`[0.5, 30.0]` ms).
    pub stereoize_delay_ms: f32,
    pub bands: [ImagerBandConfig; 4],
    pub crossover: LinkwitzRiley4BandCrossover,

    sample_rate: f64,
    /// Velvet-noise sparse tap offsets (normalized `[0.08 .. 1.00]`) and signs (`+1.0` / `-1.0`).
    velvet_delays: [FixedMonoDelay; 4],
    /// Mode II cascaded all-pass networks per band.
    mode2_allpass: [[FirstOrderAllPass; 4]; 4],
    mode2_biquad: [StereoBiquadF64; 4],
    /// Quadrature all-pass filters for `Recover Sides` Mid injection.
    recover_allpass: [[FirstOrderAllPass; 2]; 4],

    /// Smoothed power accumulators for Pearson correlation per band: `[E[L*R], E[L^2], E[R^2]]`.
    corr_accum: [[f64; 3]; 4],
    /// Latest per-band phase correlation values in `[-1.0, +1.0]`.
    pub band_correlations: [f32; 4],
    /// Smoothed vectorscope polar histogram (36 bins).
    pub vectorscope_polar: [f32; 36],
}

pub type OzoneImager = MultibandImagerModule;

const VELVET_TAP_RATIOS: [f64; 8] = [0.11, 0.23, 0.37, 0.49, 0.61, 0.73, 0.86, 1.00];
const VELVET_TAP_SIGNS: [f64; 8] = [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0];

impl Default for MultibandImagerModule {
    fn default() -> Self {
        Self::new(48_000.0)
    }
}

impl MultibandImagerModule {
    pub fn new(sample_rate: f64) -> Self {
        let sr = sample_rate.max(8000.0);
        let max_delay_samples = ((0.050 * sr).ceil() as usize).max(2048).next_power_of_two();
        let mut module = Self {
            enabled: true,
            multiband_enabled: true,
            stereoizer_enable: false,
            stereoizer_mode: StereoizerMode::ModeI,
            stereoize_delay_ms: 12.0,
            bands: [ImagerBandConfig::default(); 4],
            crossover: LinkwitzRiley4BandCrossover::new([120.0, 1200.0, 7500.0], sr),
            sample_rate: sr,
            velvet_delays: [
                FixedMonoDelay::new(max_delay_samples),
                FixedMonoDelay::new(max_delay_samples),
                FixedMonoDelay::new(max_delay_samples),
                FixedMonoDelay::new(max_delay_samples),
            ],
            mode2_allpass: [
                [
                    FirstOrderAllPass::new(0.618),
                    FirstOrderAllPass::new(-0.414),
                    FirstOrderAllPass::new(0.732),
                    FirstOrderAllPass::new(-0.535),
                ];
                4
            ],
            mode2_biquad: [StereoBiquadF64::default(); 4],
            recover_allpass: [
                [FirstOrderAllPass::new(0.4794), FirstOrderAllPass::new(0.9365)];
                4
            ],
            corr_accum: [[0.0; 3]; 4],
            band_correlations: [1.0; 4],
            vectorscope_polar: [0.0; 36],
        };
        module.update_decorrelator_filters();
        module
    }

    pub fn set_crossover_frequencies(&mut self, f1_hz: f64, f2_hz: f64, f3_hz: f64) {
        self.crossover
            .set_crossover_frequencies(self.sample_rate, f1_hz, f2_hz, f3_hz);
    }

    fn update_decorrelator_filters(&mut self) {
        let freqs = [180.0, 650.0, 2400.0, 6800.0];
        for b in 0..4 {
            self.mode2_biquad[b].set_allpass_2nd(freqs[b], 0.7071, self.sample_rate);
        }
    }

    #[inline(always)]
    fn synthesize_stereoize_side(&mut self, band_idx: usize, mid: f64) -> f64 {
        let base_delay_samples =
            ((self.stereoize_delay_ms.clamp(0.5, 30.0) as f64) * 0.001 * self.sample_rate)
                .round() as usize;
        let base_delay = base_delay_samples.max(8);

        self.velvet_delays[band_idx].push(mid);

        match self.stereoizer_mode {
            StereoizerMode::ModeI => {
                // Sparse 8-tap Velvet-Noise decorrelator (normalized energy = 1/sqrt(8)).
                let mut acc = 0.0;
                for t in 0..8 {
                    let tap_delay =
                        ((base_delay as f64 * VELVET_TAP_RATIOS[t]).round() as usize).max(1);
                    acc += VELVET_TAP_SIGNS[t] * self.velvet_delays[band_idx].read_tap(tap_delay);
                }
                acc * 0.3535533905932738
            }
            StereoizerMode::ModeII => {
                // Delayed Schroeder all-pass cascade for smooth phase-dispersed stereo width.
                let delayed = self.velvet_delays[band_idx].read_tap(base_delay / 2 + 4);
                let (mut y, _) = self.mode2_biquad[band_idx].tick(delayed, 0.0);
                for ap in &mut self.mode2_allpass[band_idx] {
                    y = ap.tick(y);
                }
                y
            }
        }
    }

    #[inline]
    pub fn process_frame(&mut self, l_in: f32, r_in: f32) -> [f32; 2] {
        if !self.enabled {
            return [l_in, r_in];
        }

        let split_bands = if self.multiband_enabled {
            self.crossover.split_stereo(l_in as f64, r_in as f64)
        } else {
            [
                [l_in as f64, r_in as f64],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        };

        let any_solo = self.bands.iter().any(|b| b.solo);
        let corr_alpha = (-1.0 / (0.050 * self.sample_rate)).exp();
        let active_bands = if self.multiband_enabled { 4 } else { 1 };

        let mut out_l = 0.0f64;
        let mut out_r = 0.0f64;

        for b in 0..active_bands {
            let cfg = &self.bands[b];
            if cfg.mute || (any_solo && !cfg.solo) {
                continue;
            }

            let bl = split_bands[b][0];
            let br = split_bands[b][1];

            // 1. Mid/Side decomposition (orthonormal 1/sqrt(2) scaling).
            const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;
            let mut mid = (bl + br) * INV_SQRT2;
            let mut side = (bl - br) * INV_SQRT2;

            // 2. Width scaling: [-100%, +100%] -> [0.0, 2.0].
            let w = (1.0 + (cfg.width_percent.clamp(-100.0, 100.0) as f64) * 0.01).clamp(0.0, 2.0);

            // 3. Recover Sides: when narrowing (w < 1.0), recover attenuated side energy into Mid
            //    via 90-degree phase shift to prevent panned instrument dropout.
            if w < 1.0 && cfg.recover_sides_percent > 0.0 {
                let recover_gain =
                    (1.0 - w) * (cfg.recover_sides_percent.clamp(0.0, 100.0) as f64 * 0.01);
                let mut shifted_side = self.recover_allpass[b][0].tick(side);
                shifted_side = self.recover_allpass[b][1].tick(shifted_side);
                mid += recover_gain * shifted_side;
            }

            side *= w;

            // 4. Phase-Safe Mono-Compatible Stereoizer:
            //    Synthesized decorrelated signal is added purely into `side`.
            //    Since L' = (Mid + Side)/sqrt(2) and R' = (Mid - Side)/sqrt(2),
            //    L' + R' = sqrt(2) * Mid == bl + br!
            //    Therefore mono summation cancels the Stereoizer signal to 0.0 (-inf dB).
            if self.stereoizer_enable && cfg.stereoize_percent > 0.0 {
                let st_gain = (cfg.stereoize_percent.clamp(0.0, 100.0) as f64) * 0.01;
                let decorr = self.synthesize_stereoize_side(b, mid);
                side += st_gain * decorr;
            } else {
                // Keep delay buffer warm even when stereoize is bypassed for click-free toggling.
                self.velvet_delays[b].push(mid);
            }

            let proc_l = (mid + side) * INV_SQRT2;
            let proc_r = (mid - side) * INV_SQRT2;

            // 5. Update per-band Pearson correlation meter.
            let acc = &mut self.corr_accum[b];
            acc[0] = corr_alpha * acc[0] + (1.0 - corr_alpha) * (proc_l * proc_r);
            acc[1] = corr_alpha * acc[1] + (1.0 - corr_alpha) * (proc_l * proc_l);
            acc[2] = corr_alpha * acc[2] + (1.0 - corr_alpha) * (proc_r * proc_r);
            let denom = (acc[1] * acc[2]).sqrt();
            if denom > 1e-12 {
                self.band_correlations[b] = (acc[0] / denom).clamp(-1.0, 1.0) as f32;
            } else {
                self.band_correlations[b] = 1.0;
            }

            out_l += proc_l;
            out_r += proc_r;
        }

        // Update vectorscope polar histogram bin.
        let mag = (out_l * out_l + out_r * out_r).sqrt() as f32;
        if mag > 1e-6 {
            let angle = (out_r.atan2(out_l) + std::f64::consts::PI)
                / (2.0 * std::f64::consts::PI);
            let bin = ((angle * 36.0).floor() as usize).min(35);
            self.vectorscope_polar[bin] = 0.98 * self.vectorscope_polar[bin] + 0.02 * mag.min(1.5);
        }

        [out_l as f32, out_r as f32]
    }

    pub fn reset_internal(&mut self) {
        self.crossover.reset();
        for d in &mut self.velvet_delays {
            d.reset();
        }
        for b in 0..4 {
            self.mode2_biquad[b].reset();
            for ap in &mut self.mode2_allpass[b] {
                ap.reset();
            }
            for ap in &mut self.recover_allpass[b] {
                ap.reset();
            }
        }
        self.corr_accum = [[0.0; 3]; 4];
        self.band_correlations = [1.0; 4];
        self.vectorscope_polar.fill(0.0);
    }
}

impl OzoneDspModule for MultibandImagerModule {
    fn name(&self) -> &'static str {
        "iZotope Ozone 12 Multiband Imager"
    }

    fn reset(&mut self, sample_rate: f64, _max_block_size: usize) {
        self.sample_rate = sample_rate.max(8000.0);
        let [f1, f2, f3] = self.crossover.cutoffs_hz;
        self.crossover
            .set_crossover_frequencies(self.sample_rate, f1, f2, f3);
        self.update_decorrelator_filters();
        self.reset_internal();
    }

    fn process_block(&mut self, left: &mut [f32], right: &mut [f32]) {
        if !self.enabled {
            return;
        }
        let n = left.len().min(right.len());
        for i in 0..n {
            let [l, r] = self.process_frame(left[i], right[i]);
            left[i] = l;
            right[i] = r;
        }
    }
}

impl UnifiedDspNode for MultibandImagerModule {
    fn name(&self) -> &'static str {
        "iZotope Ozone 12 Multiband Imager"
    }

    fn latency_samples(&self) -> usize {
        0
    }

    fn prepare(&mut self, sample_rate: f64, max_block_size: usize) {
        OzoneDspModule::reset(self, sample_rate, max_block_size);
    }

    fn reset_state(&mut self) {
        self.reset_internal();
    }

    fn set_parameter_normalized(&mut self, param_id: u32, norm_val: f32) {
        let v = norm_val.clamp(0.0, 1.0);
        match param_id {
            0..=3 => {
                self.bands[param_id as usize].width_percent = -100.0 + v * 200.0;
            }
            4..=7 => {
                self.bands[(param_id - 4) as usize].stereoize_percent = v * 100.0;
            }
            8 => {
                self.stereoizer_enable = v >= 0.5;
            }
            9 => {
                self.stereoizer_mode = if v >= 0.5 {
                    StereoizerMode::ModeII
                } else {
                    StereoizerMode::ModeI
                };
            }
            10 => {
                self.stereoize_delay_ms = 0.5 + v * 29.5;
            }
            _ => {}
        }
    }

    fn process_slice(
        &mut self,
        _ctx: &ProcessContext,
        main_io: &mut [StereoFrame],
        _sidechain_in: Option<&[StereoFrame]>,
        _aux_sends_out: &mut [&mut [StereoFrame]],
    ) {
        if !self.enabled {
            return;
        }
        for frame in main_io.iter_mut() {
            *frame = self.process_frame(frame[0], frame[1]);
        }
    }
}
