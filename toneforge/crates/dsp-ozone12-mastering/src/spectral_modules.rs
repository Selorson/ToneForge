//! iZotope Ozone 12 Spectral Mastering Modules:
//! 1. `StabilizerModule` (`iZOzone12Stabilizer.dll`, `0x181DF3E00`): Adaptive Tonal Balance
//!    (`Shape` & `Cut` modes + `Tame Transients` + 256-bar Tonal Balance telemetry).
//! 2. `ClarityModule` (`iZOzone12Clarity.dll`): Psychoacoustic ISO 226 Spectral Tilt & Unmasking.
//! 3. `LowEndFocusModule` (`iZOzone12LowEndFocus.dll`): Low-frequency spectral contrast (`Punchy`/`Smooth`).
//! 4. `SpectralShaperModule` (`iZOzone12SpectralShaper.dll`): Dynamic spectral resonance suppression.
//!
//! All modules operate with strictly 0 heap allocations on the real-time audio thread.

use crate::chain_element::{LinkwitzRiley4BandCrossover, OzoneDspModule, StereoBiquadF64};
use daw_core_graph::{ProcessContext, StereoFrame, UnifiedDspNode};

// ============================================================================
// 1. OZONE 12 STABILIZER (ADAPTIVE TONAL BALANCE)
// ============================================================================

/// Stabilizer correction mode (`AutomatableParamList_Stabilizer.txt`: `Mode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StabilizerMode {
    /// Bidirectional tonal shaping (allows both dynamic boost and cut).
    Shape = 0,
    /// Subtractive-only resonance & mud suppression (`Delta G <= 0.0 dB`).
    Cut = 1,
}

/// Genre target profile for Stabilizer adaptive tonal balance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StabilizerTarget {
    AllAround = 0,
    Electronic = 1,
    HipHop = 2,
    Pop = 3,
    Rock = 4,
    Classical = 5,
    Jazz = 6,
    Folk = 7,
}

impl StabilizerTarget {
    /// Returns the target relative band energy profile in dB `[Low, LowMid, HighMid, High]`.
    #[inline]
    pub fn target_band_offsets_db(self) -> [f64; 4] {
        match self {
            Self::AllAround => [0.0, -1.5, -3.0, -5.0],
            Self::Electronic => [1.5, -2.0, -3.5, -4.5],
            Self::HipHop => [2.0, -2.5, -3.5, -5.0],
            Self::Pop => [0.5, -1.5, -2.5, -4.0],
            Self::Rock => [-0.5, -0.5, -2.0, -4.5],
            Self::Classical => [-1.5, -1.0, -3.5, -6.5],
            Self::Jazz => [-0.5, -1.0, -3.0, -5.5],
            Self::Folk => [-1.0, -0.5, -2.5, -5.0],
        }
    }
}

/// iZotope Ozone 12 Stabilizer (`StabilizerModule` / `OzoneStabilizer`).
#[derive(Clone, Debug)]
pub struct StabilizerModule {
    pub enabled: bool,
    pub mode: StabilizerMode,
    pub target: StabilizerTarget,
    /// Global correction amount in `[0.0, 100.0]`%.
    pub amount_percent: f32,
    /// Adaptation speed in `[0.0, 100.0]`%.
    pub speed_percent: f32,
    /// Spectral smoothing in `[0.0, 100.0]`%.
    pub smoothing_percent: f32,
    /// Transient taming amount in `[0.0, 100.0]`%.
    pub tame_transients_percent: f32,
    /// Per-band sensitivity delta in `[-100.0, +100.0]`% (`[Low, LowMid, HighMid, High]`).
    pub band_delta_sensitivity: [f32; 4],
    /// Auto-level RMS makeup enable.
    pub auto_gain_enabled: bool,
    /// Delta audition mode (outputs only the difference `Y - X`).
    pub delta_audition: bool,

    crossover: LinkwitzRiley4BandCrossover,
    /// 8-band secondary shaping EQ filters for fine spectral smoothing.
    fine_filters: [StereoBiquadF64; 8],
    band_rms_sq: [f64; 4],
    band_fast_peak: [f64; 4],
    band_slow_peak: [f64; 4],
    band_gain_lin: [f64; 4],
    in_rms_sq: f64,
    out_rms_sq: f64,
    sample_rate: f64,
    /// 256-bar tonal balance delta curve exported to `Ozone12MeterFrame`.
    pub tonal_balance_bars: [f32; 256],
}

pub type OzoneStabilizer = StabilizerModule;

const FINE_BAND_HZ: [f64; 8] = [60.0, 140.0, 320.0, 750.0, 1700.0, 3800.0, 8200.0, 15000.0];

impl Default for StabilizerModule {
    fn default() -> Self {
        Self::new(48_000.0)
    }
}

impl StabilizerModule {
    pub fn new(sample_rate: f64) -> Self {
        let sr = sample_rate.max(8000.0);
        let mut s = Self {
            enabled: true,
            mode: StabilizerMode::Shape,
            target: StabilizerTarget::AllAround,
            amount_percent: 50.0,
            speed_percent: 50.0,
            smoothing_percent: 50.0,
            tame_transients_percent: 25.0,
            band_delta_sensitivity: [0.0; 4],
            auto_gain_enabled: true,
            delta_audition: false,
            crossover: LinkwitzRiley4BandCrossover::new([140.0, 1200.0, 6500.0], sr),
            fine_filters: [StereoBiquadF64::default(); 8],
            band_rms_sq: [1e-6; 4],
            band_fast_peak: [0.0; 4],
            band_slow_peak: [0.0; 4],
            band_gain_lin: [1.0; 4],
            in_rms_sq: 1e-6,
            out_rms_sq: 1e-6,
            sample_rate: sr,
            tonal_balance_bars: [0.0; 256],
        };
        s.update_fine_filters([0.0; 4]);
        s
    }

    fn update_fine_filters(&mut self, band_gains_db: [f64; 4]) {
        let smooth_q = 0.7 + (1.0 - (self.smoothing_percent.clamp(0.0, 100.0) as f64) * 0.01) * 1.3;
        for i in 0..8 {
            let b_idx = i / 2;
            let next_idx = (b_idx + 1).min(3);
            let frac = (i & 1) as f64 * 0.5;
            let interp_db =
                (band_gains_db[b_idx] * (1.0 - frac) + band_gains_db[next_idx] * frac) * 0.35;
            self.fine_filters[i].set_peaking_eq(
                FINE_BAND_HZ[i],
                smooth_q,
                interp_db,
                self.sample_rate,
            );
        }
    }

    pub fn reset_internal(&mut self) {
        self.crossover.reset();
        for f in &mut self.fine_filters {
            f.reset();
        }
        self.band_rms_sq = [1e-6; 4];
        self.band_fast_peak = [0.0; 4];
        self.band_slow_peak = [0.0; 4];
        self.band_gain_lin = [1.0; 4];
        self.in_rms_sq = 1e-6;
        self.out_rms_sq = 1e-6;
        self.tonal_balance_bars.fill(0.0);
    }

    #[inline]
    pub fn process_frame(&mut self, l_in: f32, r_in: f32) -> [f32; 2] {
        if !self.enabled {
            return [l_in, r_in];
        }

        let xl = l_in as f64;
        let xr = r_in as f64;
        let in_energy = 0.5 * (xl * xl + xr * xr);

        let speed_norm = (self.speed_percent.clamp(0.0, 100.0) as f64) * 0.01;
        let tau_sec = 0.250 - speed_norm * 0.220; // 30 ms .. 250 ms
        let env_alpha = (-1.0 / (tau_sec * self.sample_rate)).exp();
        let fast_alpha = (-1.0 / (0.0015 * self.sample_rate)).exp();
        let slow_alpha = (-1.0 / (0.0450 * self.sample_rate)).exp();

        let split = self.crossover.split_stereo(xl, xr);
        let mut total_rms_db = 0.0f64;
        let mut band_db = [0.0f64; 4];

        for b in 0..4 {
            let bl = split[b][0];
            let br = split[b][1];
            let p_sq = 0.5 * (bl * bl + br * br);
            self.band_rms_sq[b] = env_alpha * self.band_rms_sq[b] + (1.0 - env_alpha) * p_sq;
            let db = 10.0 * (self.band_rms_sq[b] + 1e-12).log10();
            band_db[b] = db;
            total_rms_db += 0.25 * db;

            let peak = bl.abs().max(br.abs());
            self.band_fast_peak[b] = if peak > self.band_fast_peak[b] {
                fast_alpha * self.band_fast_peak[b] + (1.0 - fast_alpha) * peak
            } else {
                slow_alpha * self.band_fast_peak[b] + (1.0 - slow_alpha) * peak
            };
            self.band_slow_peak[b] = slow_alpha * self.band_slow_peak[b] + (1.0 - slow_alpha) * peak;
        }

        let target_offsets = self.target.target_band_offsets_db();
        let amount = (self.amount_percent.clamp(0.0, 100.0) as f64) * 0.01;
        let tame_amt = (self.tame_transients_percent.clamp(0.0, 100.0) as f64) * 0.01;

        let mut sum_l = 0.0f64;
        let mut sum_r = 0.0f64;
        let mut applied_db = [0.0f64; 4];

        for b in 0..4 {
            let sens = 1.0 + (self.band_delta_sensitivity[b].clamp(-100.0, 100.0) as f64) * 0.008;
            let desired_rel_db = target_offsets[b];
            let actual_rel_db = band_db[b] - total_rms_db;
            let mut err_db = (desired_rel_db - actual_rel_db) * amount * sens * 0.65;

            match self.mode {
                StabilizerMode::Shape => {
                    err_db = err_db.clamp(-6.0, 6.0);
                }
                StabilizerMode::Cut => {
                    err_db = err_db.min(0.0).max(-9.0);
                }
            }

            if tame_amt > 0.0 && self.band_fast_peak[b] > self.band_slow_peak[b] * 1.15 {
                let ratio = (self.band_fast_peak[b] / (self.band_slow_peak[b] + 1e-9)).min(4.0);
                let tame_cut_db = -20.0 * ratio.log10() * tame_amt * 0.5;
                err_db = (err_db + tame_cut_db).max(-12.0);
            }

            applied_db[b] = err_db;
            let target_lin = 10.0f64.powf(err_db / 20.0);
            self.band_gain_lin[b] =
                env_alpha * self.band_gain_lin[b] + (1.0 - env_alpha) * target_lin;

            sum_l += split[b][0] * self.band_gain_lin[b];
            sum_r += split[b][1] * self.band_gain_lin[b];
        }

        for f in &mut self.fine_filters {
            let (nl, nr) = f.tick(sum_l, sum_r);
            sum_l = nl;
            sum_r = nr;
        }

        let out_energy = 0.5 * (sum_l * sum_l + sum_r * sum_r);
        self.in_rms_sq = env_alpha * self.in_rms_sq + (1.0 - env_alpha) * in_energy;
        self.out_rms_sq = env_alpha * self.out_rms_sq + (1.0 - env_alpha) * out_energy;

        if self.auto_gain_enabled && self.in_rms_sq > 1e-8 && self.out_rms_sq > 1e-8 {
            let makeup = (self.in_rms_sq / self.out_rms_sq).sqrt().clamp(0.5, 2.0);
            sum_l *= makeup;
            sum_r *= makeup;
        }

        for idx in 0..256 {
            let pos = (idx as f64 / 255.0) * 3.0;
            let b0 = (pos.floor() as usize).min(2);
            let frac = pos - b0 as f64;
            let val_db = applied_db[b0] * (1.0 - frac) + applied_db[b0 + 1] * frac;
            self.tonal_balance_bars[idx] = val_db as f32;
        }

        if self.delta_audition {
            [(sum_l - xl) as f32, (sum_r - xr) as f32]
        } else {
            [sum_l as f32, sum_r as f32]
        }
    }

    #[inline]
    pub fn process_frames_in_place(&mut self, io: &mut [StereoFrame]) {
        if !self.enabled {
            return;
        }
        for frame in io.iter_mut() {
            *frame = self.process_frame(frame[0], frame[1]);
        }
        let band_db = [
            20.0 * self.band_gain_lin[0].max(1e-6).log10(),
            20.0 * self.band_gain_lin[1].max(1e-6).log10(),
            20.0 * self.band_gain_lin[2].max(1e-6).log10(),
            20.0 * self.band_gain_lin[3].max(1e-6).log10(),
        ];
        self.update_fine_filters(band_db);
    }
}

impl OzoneDspModule for StabilizerModule {
    fn name(&self) -> &'static str {
        "iZotope Ozone 12 Stabilizer"
    }

    fn reset(&mut self, sample_rate: f64, _max_block_size: usize) {
        self.sample_rate = sample_rate.max(8000.0);
        let [f1, f2, f3] = self.crossover.cutoffs_hz;
        self.crossover
            .set_crossover_frequencies(self.sample_rate, f1, f2, f3);
        self.update_fine_filters([0.0; 4]);
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
        let band_db = [
            20.0 * self.band_gain_lin[0].max(1e-6).log10(),
            20.0 * self.band_gain_lin[1].max(1e-6).log10(),
            20.0 * self.band_gain_lin[2].max(1e-6).log10(),
            20.0 * self.band_gain_lin[3].max(1e-6).log10(),
        ];
        self.update_fine_filters(band_db);
    }
}

impl UnifiedDspNode for StabilizerModule {
    fn name(&self) -> &'static str {
        "iZotope Ozone 12 Stabilizer"
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
            0 => self.amount_percent = v * 100.0,
            1 => self.speed_percent = v * 100.0,
            2 => self.smoothing_percent = v * 100.0,
            3 => self.tame_transients_percent = v * 100.0,
            4 => {
                self.mode = if v >= 0.5 {
                    StabilizerMode::Cut
                } else {
                    StabilizerMode::Shape
                };
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
        self.process_frames_in_place(main_io);
    }
}

// ============================================================================
// 2. OZONE 12 CLARITY (PSYCHOACOUSTIC SPECTRAL TILT & UNMASKING)
// ============================================================================

const CLARITY_BARK_FREQS: [f64; 6] = [250.0, 630.0, 1600.0, 3150.0, 6300.0, 12500.0];

/// iZotope Ozone 12 Clarity (`ClarityModule` / `OzoneClarity`).
/// Uses an ISO 226 psychoacoustic Bark-band filterbank to dynamically unmask high-mid
/// and air frequencies towards a configurable spectral tilt target (`dB/octave`).
#[derive(Clone, Debug)]
pub struct ClarityModule {
    pub enabled: bool,
    /// Unmasking amount in `[0.0, 100.0]`%.
    pub amount_percent: f32,
    /// Psychoacoustic target tilt in `[-6.0, +6.0]` dB/octave (`0.0` = pink-noise -3 dB/oct reference).
    pub tilt_db_per_oct: f32,
    /// Envelope attack time in milliseconds (`[1.0, 200.0]` ms).
    pub attack_ms: f32,
    /// Envelope release time in milliseconds (`[10.0, 1000.0]` ms).
    pub release_ms: f32,
    /// Focus band lower bound (`[100.0, 5000.0]` Hz).
    pub focus_low_hz: f32,
    /// Focus band upper bound (`[1000.0, 20000.0]` Hz).
    pub focus_high_hz: f32,

    detector_bp: [StereoBiquadF64; 6],
    shaper_eq: [StereoBiquadF64; 6],
    band_env: [f64; 6],
    band_gain_db: [f64; 6],
    broadband_env: f64,
    sample_rate: f64,
}

pub type OzoneClarity = ClarityModule;

impl Default for ClarityModule {
    fn default() -> Self {
        Self::new(48_000.0)
    }
}

impl ClarityModule {
    pub fn new(sample_rate: f64) -> Self {
        let sr = sample_rate.max(8000.0);
        let mut m = Self {
            enabled: true,
            amount_percent: 35.0,
            tilt_db_per_oct: 1.5,
            attack_ms: 15.0,
            release_ms: 120.0,
            focus_low_hz: 300.0,
            focus_high_hz: 16000.0,
            detector_bp: [StereoBiquadF64::default(); 6],
            shaper_eq: [StereoBiquadF64::default(); 6],
            band_env: [1e-6; 6],
            band_gain_db: [0.0; 6],
            broadband_env: 1e-6,
            sample_rate: sr,
        };
        m.init_filters();
        m
    }

    fn init_filters(&mut self) {
        for i in 0..6 {
            self.detector_bp[i].set_bandpass(CLARITY_BARK_FREQS[i], 1.2, self.sample_rate);
            self.shaper_eq[i].set_peaking_eq(
                CLARITY_BARK_FREQS[i],
                1.0,
                self.band_gain_db[i],
                self.sample_rate,
            );
        }
    }

    fn refresh_shaper_eq(&mut self) {
        for i in 0..6 {
            self.shaper_eq[i].set_peaking_eq(
                CLARITY_BARK_FREQS[i],
                1.0,
                self.band_gain_db[i],
                self.sample_rate,
            );
        }
    }

    pub fn reset_internal(&mut self) {
        for i in 0..6 {
            self.detector_bp[i].reset();
            self.shaper_eq[i].reset();
        }
        self.band_env = [1e-6; 6];
        self.band_gain_db = [0.0; 6];
        self.broadband_env = 1e-6;
    }

    #[inline]
    pub fn process_frames_in_place(&mut self, io: &mut [StereoFrame]) {
        if !self.enabled || self.amount_percent <= 0.0 {
            return;
        }

        let atk_alpha =
            (-1.0 / ((self.attack_ms.clamp(1.0, 200.0) as f64) * 0.001 * self.sample_rate)).exp();
        let rel_alpha =
            (-1.0 / ((self.release_ms.clamp(10.0, 1000.0) as f64) * 0.001 * self.sample_rate))
                .exp();
        let amt = (self.amount_percent.clamp(0.0, 100.0) as f64) * 0.01;
        let tilt = self.tilt_db_per_oct.clamp(-6.0, 6.0) as f64;
        let f_lo = self.focus_low_hz as f64;
        let f_hi = self.focus_high_hz as f64;

        for frame in io.iter_mut() {
            let mut l = frame[0] as f64;
            let mut r = frame[1] as f64;
            let bb_sq = 0.5 * (l * l + r * r);
            self.broadband_env = rel_alpha * self.broadband_env + (1.0 - rel_alpha) * bb_sq;

            for i in 0..6 {
                let (bl, br) = self.detector_bp[i].tick(l, r);
                let e = 0.5 * (bl * bl + br * br);
                let c = if e > self.band_env[i] {
                    atk_alpha
                } else {
                    rel_alpha
                };
                self.band_env[i] = c * self.band_env[i] + (1.0 - c) * e;
            }

            for s in &mut self.shaper_eq {
                let (nl, nr) = s.tick(l, r);
                l = nl;
                r = nr;
            }
            *frame = [l as f32, r as f32];
        }

        let ref_db = 10.0 * (self.broadband_env + 1e-12).log10();
        for i in 0..6 {
            let freq = CLARITY_BARK_FREQS[i];
            if freq < f_lo || freq > f_hi {
                self.band_gain_db[i] *= 0.9;
                continue;
            }
            let octaves_from_1k = (freq / 1000.0).log2();
            let target_rel_db = (-3.0 + tilt) * octaves_from_1k;
            let band_rel_db = 10.0 * (self.band_env[i] + 1e-12).log10() - ref_db;
            let deficit_db = (target_rel_db - band_rel_db).clamp(-4.5, 6.0);
            self.band_gain_db[i] = 0.75 * self.band_gain_db[i] + 0.25 * (deficit_db * amt);
        }
        self.refresh_shaper_eq();
    }
}

impl OzoneDspModule for ClarityModule {
    fn name(&self) -> &'static str {
        "iZotope Ozone 12 Clarity"
    }

    fn reset(&mut self, sample_rate: f64, _max_block_size: usize) {
        self.sample_rate = sample_rate.max(8000.0);
        self.init_filters();
        self.reset_internal();
    }

    fn process_block(&mut self, left: &mut [f32], right: &mut [f32]) {
        if !self.enabled || self.amount_percent <= 0.0 {
            return;
        }
        let n = left.len().min(right.len());
        let atk_alpha =
            (-1.0 / ((self.attack_ms.clamp(1.0, 200.0) as f64) * 0.001 * self.sample_rate)).exp();
        let rel_alpha =
            (-1.0 / ((self.release_ms.clamp(10.0, 1000.0) as f64) * 0.001 * self.sample_rate))
                .exp();
        let amt = (self.amount_percent.clamp(0.0, 100.0) as f64) * 0.01;
        let tilt = self.tilt_db_per_oct.clamp(-6.0, 6.0) as f64;
        let f_lo = self.focus_low_hz as f64;
        let f_hi = self.focus_high_hz as f64;

        for idx in 0..n {
            let mut l = left[idx] as f64;
            let mut r = right[idx] as f64;
            let bb_sq = 0.5 * (l * l + r * r);
            self.broadband_env = rel_alpha * self.broadband_env + (1.0 - rel_alpha) * bb_sq;

            for i in 0..6 {
                let (bl, br) = self.detector_bp[i].tick(l, r);
                let e = 0.5 * (bl * bl + br * br);
                let c = if e > self.band_env[i] {
                    atk_alpha
                } else {
                    rel_alpha
                };
                self.band_env[i] = c * self.band_env[i] + (1.0 - c) * e;
            }

            for s in &mut self.shaper_eq {
                let (nl, nr) = s.tick(l, r);
                l = nl;
                r = nr;
            }
            left[idx] = l as f32;
            right[idx] = r as f32;
        }

        let ref_db = 10.0 * (self.broadband_env + 1e-12).log10();
        for i in 0..6 {
            let freq = CLARITY_BARK_FREQS[i];
            if freq < f_lo || freq > f_hi {
                self.band_gain_db[i] *= 0.9;
                continue;
            }
            let octaves_from_1k = (freq / 1000.0).log2();
            let target_rel_db = (-3.0 + tilt) * octaves_from_1k;
            let band_rel_db = 10.0 * (self.band_env[i] + 1e-12).log10() - ref_db;
            let deficit_db = (target_rel_db - band_rel_db).clamp(-4.5, 6.0);
            self.band_gain_db[i] = 0.75 * self.band_gain_db[i] + 0.25 * (deficit_db * amt);
        }
        self.refresh_shaper_eq();
    }
}

impl UnifiedDspNode for ClarityModule {
    fn name(&self) -> &'static str {
        "iZotope Ozone 12 Clarity"
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
            0 => self.amount_percent = v * 100.0,
            1 => self.tilt_db_per_oct = -6.0 + v * 12.0,
            2 => self.attack_ms = 1.0 + v * 199.0,
            3 => self.release_ms = 10.0 + v * 990.0,
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
        self.process_frames_in_place(main_io);
    }
}

// ============================================================================
// 3. OZONE 12 LOW END FOCUS & SPECTRAL SHAPER
// ============================================================================

pub use crate::low_end_focus::{LowEndFocusMode, LowEndFocusModule, OzoneLowEndFocus};

/// Tone compression ratios for `SpectralShaperModule`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpectralShaperToneMode {
    /// Light (`2:1` ratio)
    Light = 0,
    /// Medium (`4:1` ratio)
    Medium = 1,
    /// Heavy (`8:1` ratio)
    Heavy = 2,
}

/// iZotope Ozone 12 Spectral Shaper (`SpectralShaperModule` / `OzoneSpectralShaper`).
#[derive(Clone, Debug)]
pub struct SpectralShaperModule {
    pub enabled: bool,
    pub tone_mode: SpectralShaperToneMode,
    pub threshold_db: f32,
    pub ceiling_reduction_db: f32,
    pub center_hz: f32,
    pub attack_ms: f32,
    pub release_ms: f32,
    detector_bp: StereoBiquadF64,
    dynamic_eq: StereoBiquadF64,
    env: f64,
    last_reduction_db: f32,
    sample_rate: f64,
}

pub type OzoneSpectralShaper = SpectralShaperModule;

impl Default for SpectralShaperModule {
    fn default() -> Self {
        Self::new(48_000.0)
    }
}

impl SpectralShaperModule {
    pub fn new(sample_rate: f64) -> Self {
        let sr = sample_rate.max(8000.0);
        let mut m = Self {
            enabled: true,
            tone_mode: SpectralShaperToneMode::Medium,
            threshold_db: -18.0,
            ceiling_reduction_db: -12.0,
            center_hz: 4200.0,
            attack_ms: 5.0,
            release_ms: 45.0,
            detector_bp: StereoBiquadF64::default(),
            dynamic_eq: StereoBiquadF64::default(),
            env: 0.0,
            last_reduction_db: 0.0,
            sample_rate: sr,
        };
        m.detector_bp.set_bandpass(4200.0, 1.5, sr);
        m.dynamic_eq.set_peaking_eq(4200.0, 1.5, 0.0, sr);
        m
    }

    pub fn set_center_hz(&mut self, center_hz: f32) {
        self.center_hz = center_hz.clamp(20.0, 20_000.0);
        self.detector_bp
            .set_bandpass(self.center_hz as f64, 1.5, self.sample_rate);
    }

    #[inline]
    pub fn last_reduction_db(&self) -> f32 {
        self.last_reduction_db
    }

    pub fn reset_internal(&mut self) {
        self.detector_bp.reset();
        self.dynamic_eq.reset();
        self.env = 0.0;
        self.last_reduction_db = 0.0;
    }

    #[inline]
    pub fn process_frames_in_place(&mut self, io: &mut [StereoFrame]) {
        if !self.enabled {
            return;
        }
        let thresh_lin = 10.0f64.powf((self.threshold_db as f64) / 20.0);
        let atk_alpha =
            (-1.0 / ((self.attack_ms.clamp(0.5, 200.0) as f64) * 0.001 * self.sample_rate)).exp();
        let rel_alpha =
            (-1.0 / ((self.release_ms.clamp(5.0, 1000.0) as f64) * 0.001 * self.sample_rate)).exp();
        let slope = match self.tone_mode {
            SpectralShaperToneMode::Light => 0.50,  // 2:1
            SpectralShaperToneMode::Medium => 0.75, // 4:1
            SpectralShaperToneMode::Heavy => 0.875, // 8:1
        };
        let center_f64 = self.center_hz as f64;
        let ceil_f64 = self.ceiling_reduction_db as f64;

        for (idx, frame) in io.iter_mut().enumerate() {
            let (dl, dr) = self.detector_bp.tick(frame[0] as f64, frame[1] as f64);
            let p = dl.abs().max(dr.abs());
            let coeff = if p > self.env { atk_alpha } else { rel_alpha };
            self.env = coeff * self.env + (1.0 - coeff) * p;
            if (idx & 15) == 0 {
                let step_cut_db = if self.env > thresh_lin {
                    let over_db = 20.0 * (self.env / thresh_lin).log10();
                    (-slope * over_db).max(ceil_f64)
                } else {
                    0.0
                };
                self.dynamic_eq
                    .set_peaking_eq(center_f64, 1.5, step_cut_db, self.sample_rate);
            }
            let (yl, yr) = self.dynamic_eq.tick(frame[0] as f64, frame[1] as f64);
            *frame = [yl as f32, yr as f32];
        }

        let cut_db = if self.env > thresh_lin {
            let over_db = 20.0 * (self.env / thresh_lin).log10();
            (-slope * over_db).max(ceil_f64)
        } else {
            0.0
        };
        self.last_reduction_db = cut_db as f32;
        self.dynamic_eq
            .set_peaking_eq(center_f64, 1.5, cut_db, self.sample_rate);
    }
}

impl OzoneDspModule for SpectralShaperModule {
    fn name(&self) -> &'static str {
        "iZotope Ozone 12 Spectral Shaper"
    }

    fn reset(&mut self, sample_rate: f64, _max_block_size: usize) {
        self.sample_rate = sample_rate.max(8000.0);
        self.detector_bp
            .set_bandpass(self.center_hz as f64, 1.5, self.sample_rate);
        self.dynamic_eq
            .set_peaking_eq(self.center_hz as f64, 1.5, 0.0, self.sample_rate);
        self.reset_internal();
    }

    fn process_block(&mut self, left: &mut [f32], right: &mut [f32]) {
        if !self.enabled {
            return;
        }
        let n = left.len().min(right.len());
        let mut frame_buf = [[0.0f32; 2]; 1];
        for i in 0..n {
            frame_buf[0] = [left[i], right[i]];
            self.process_frames_in_place(&mut frame_buf);
            left[i] = frame_buf[0][0];
            right[i] = frame_buf[0][1];
        }
    }
}

impl UnifiedDspNode for SpectralShaperModule {
    fn name(&self) -> &'static str {
        "iZotope Ozone 12 Spectral Shaper"
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
            0 => self.enabled = v >= 0.5,
            1 => self.threshold_db = -60.0 + v * 60.0,
            2 => self.ceiling_reduction_db = -24.0 + v * 24.0,
            3 => self.set_center_hz(20.0 * 1000.0f32.powf(v)),
            4 => {
                self.tone_mode = if v < 0.33 {
                    SpectralShaperToneMode::Light
                } else if v < 0.67 {
                    SpectralShaperToneMode::Medium
                } else {
                    SpectralShaperToneMode::Heavy
                };
            }
            _ => {}
        }
    }

    fn current_gain_reduction_db(&self) -> f32 {
        self.last_reduction_db
    }

    fn process_slice(
        &mut self,
        _ctx: &ProcessContext,
        main_io: &mut [StereoFrame],
        _sidechain_in: Option<&[StereoFrame]>,
        _aux_sends_out: &mut [&mut [StereoFrame]],
    ) {
        self.process_frames_in_place(main_io);
    }
}
