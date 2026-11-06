//! iZotope Ozone 12 Micro-Dynamics, Transient Recovery & Bass Modules:
//! 1. `ImpactModule` (`iZOzone12Impact.dll`, `0x181DF3980`): 4-Band Micro-Dynamics Shaper.
//! 2. `UnlimiterModule` (`iZOzone12Unlimiter.dll`, `delimit_exp009_ep080_fixed2.onnx`):
//!    Brickwall Transient Recovery & De-Clipper.
//! 3. `BassControlModule` (`iZOzone12BassControl.dll`): Low-End Sustain Power & Peak Punch.
//!
//! Strictly 0 heap allocations on the real-time audio thread.

use crate::chain_element::{LinkwitzRiley4BandCrossover, OzoneDspModule};
use daw_core_graph::{ProcessContext, StereoFrame, UnifiedDspNode};

// ============================================================================
// 1. OZONE 12 IMPACT (4-BAND MICRO-DYNAMICS)
// ============================================================================

/// Per-band configuration for Ozone 12 Impact.
#[derive(Clone, Copy, Debug)]
pub struct ImpactBandConfig {
    /// Micro-dynamics amount in `[-100.0, +100.0]` (`>0` = transient expansion/punch, `<0` = glue compression).
    pub amount: f32,
    /// Integration window time constant in milliseconds (`[5.0, 500.0]` ms).
    pub envelope_ms: f32,
    pub mute: bool,
    pub solo: bool,
}

impl Default for ImpactBandConfig {
    fn default() -> Self {
        Self {
            amount: 0.0,
            envelope_ms: 80.0,
            mute: false,
            solo: false,
        }
    }
}

/// iZotope Ozone 12 Impact (`ImpactModule` / `OzoneImpact`).
#[derive(Clone, Debug)]
pub struct ImpactModule {
    pub enabled: bool,
    pub bands: [ImpactBandConfig; 4],
    pub delta_audition: bool,
    pub crossover: LinkwitzRiley4BandCrossover,

    env_fast: [f64; 4],
    env_slow: [f64; 4],
    gain_smooth: [f64; 4],
    /// Latest per-band micro-dynamic gain modulation in dB (`[-12.0, +12.0]` dB).
    pub band_gain_trace_db: [f32; 4],
    sample_rate: f64,
}

pub type OzoneImpact = ImpactModule;

impl Default for ImpactModule {
    fn default() -> Self {
        Self::new(48_000.0)
    }
}

impl ImpactModule {
    pub fn new(sample_rate: f64) -> Self {
        let sr = sample_rate.max(8000.0);
        Self {
            enabled: true,
            bands: [
                ImpactBandConfig {
                    amount: 0.0,
                    envelope_ms: 120.0,
                    mute: false,
                    solo: false,
                },
                ImpactBandConfig {
                    amount: 0.0,
                    envelope_ms: 85.0,
                    mute: false,
                    solo: false,
                },
                ImpactBandConfig {
                    amount: 0.0,
                    envelope_ms: 55.0,
                    mute: false,
                    solo: false,
                },
                ImpactBandConfig {
                    amount: 0.0,
                    envelope_ms: 35.0,
                    mute: false,
                    solo: false,
                },
            ],
            delta_audition: false,
            crossover: LinkwitzRiley4BandCrossover::new([140.0, 1200.0, 6500.0], sr),
            env_fast: [1e-6; 4],
            env_slow: [1e-6; 4],
            gain_smooth: [1.0; 4],
            band_gain_trace_db: [0.0; 4],
            sample_rate: sr,
        }
    }

    pub fn reset_internal(&mut self) {
        self.crossover.reset();
        self.env_fast = [1e-6; 4];
        self.env_slow = [1e-6; 4];
        self.gain_smooth = [1.0; 4];
        self.band_gain_trace_db = [0.0; 4];
    }

    #[inline]
    pub fn process_frame(&mut self, l_in: f32, r_in: f32) -> [f32; 2] {
        if !self.enabled {
            return [l_in, r_in];
        }

        let xl = l_in as f64;
        let xr = r_in as f64;
        let split = self.crossover.split_stereo(xl, xr);
        let any_solo = self.bands.iter().any(|b| b.solo);

        let fast_atk = (-1.0 / (0.0012 * self.sample_rate)).exp();
        let fast_rel = (-1.0 / (0.0120 * self.sample_rate)).exp();
        let smooth_alpha = (-1.0 / (0.0030 * self.sample_rate)).exp();

        let mut out_l = 0.0f64;
        let mut out_r = 0.0f64;

        for b in 0..4 {
            let cfg = &self.bands[b];
            if cfg.mute || (any_solo && !cfg.solo) {
                continue;
            }

            let bl = split[b][0];
            let br = split[b][1];
            let peak = bl.abs().max(br.abs());

            let cf = if peak > self.env_fast[b] {
                fast_atk
            } else {
                fast_rel
            };
            self.env_fast[b] = cf * self.env_fast[b] + (1.0 - cf) * peak;

            let slow_tau = (cfg.envelope_ms.clamp(5.0, 500.0) as f64) * 0.001;
            let cs = (-1.0 / (slow_tau * self.sample_rate)).exp();
            self.env_slow[b] = cs * self.env_slow[b] + (1.0 - cs) * peak;

            let diff_db =
                20.0 * ((self.env_fast[b] + 1e-7) / (self.env_slow[b] + 1e-7)).log10();
            let norm_amt = (cfg.amount.clamp(-100.0, 100.0) as f64) * 0.01;
            let target_db = (norm_amt * diff_db * 1.25).clamp(-12.0, 12.0);
            self.band_gain_trace_db[b] = target_db as f32;

            let target_lin = 10.0f64.powf(target_db / 20.0);
            self.gain_smooth[b] =
                smooth_alpha * self.gain_smooth[b] + (1.0 - smooth_alpha) * target_lin;

            out_l += bl * self.gain_smooth[b];
            out_r += br * self.gain_smooth[b];
        }

        if self.delta_audition {
            [(out_l - xl) as f32, (out_r - xr) as f32]
        } else {
            [out_l as f32, out_r as f32]
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
    }
}

impl OzoneDspModule for ImpactModule {
    fn name(&self) -> &'static str {
        "iZotope Ozone 12 Impact"
    }

    fn reset(&mut self, sample_rate: f64, _max_block_size: usize) {
        self.sample_rate = sample_rate.max(8000.0);
        let [f1, f2, f3] = self.crossover.cutoffs_hz;
        self.crossover
            .set_crossover_frequencies(self.sample_rate, f1, f2, f3);
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

impl UnifiedDspNode for ImpactModule {
    fn name(&self) -> &'static str {
        "iZotope Ozone 12 Impact"
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
                self.bands[param_id as usize].amount = -100.0 + v * 200.0;
            }
            4..=7 => {
                self.bands[(param_id - 4) as usize].envelope_ms = 5.0 + v * 495.0;
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
// 2. OZONE 12 UNLIMITER (TRANSIENT RECOVERY)
// ============================================================================

/// iZotope Ozone 12 Unlimiter (`UnlimiterModule` / `OzoneUnlimiter`).
/// Restores dynamic range and transient peaks crushed by brickwall limiters or clippers.
#[derive(Clone, Debug)]
pub struct UnlimiterModule {
    pub enabled: bool,
    /// Transient recovery amount in `[0.0, 100.0]`%.
    pub amount_percent: f32,
    /// Detection threshold in `[-24.0, 0.0]` dBFS above which crushed peaks are expanded.
    pub threshold_db: f32,
    /// Sensitivity to flat-topped / compressed plateaus in `[0.0, 100.0]`%.
    pub sensitivity_percent: f32,
    /// Attack time in milliseconds (`[0.5, 50.0]` ms).
    pub attack_ms: f32,
    /// Release time in milliseconds (`[10.0, 500.0]` ms).
    pub release_ms: f32,
    /// Auto-level compensation to keep average loudness consistent when expanding peaks.
    pub auto_level_enabled: bool,

    env_peak: f64,
    env_rms_sq: f64,
    out_rms_sq: f64,
    expansion_gain: f64,
    /// Latest upward transient restoration in dB (`[0.0, +9.0]` dB).
    pub restored_transient_db: f32,
    sample_rate: f64,
}

pub type OzoneUnlimiter = UnlimiterModule;

impl Default for UnlimiterModule {
    fn default() -> Self {
        Self::new(48_000.0)
    }
}

impl UnlimiterModule {
    pub fn new(sample_rate: f64) -> Self {
        Self {
            enabled: true,
            amount_percent: 40.0,
            threshold_db: -6.0,
            sensitivity_percent: 60.0,
            attack_ms: 2.0,
            release_ms: 85.0,
            auto_level_enabled: false,
            env_peak: 0.0,
            env_rms_sq: 1e-6,
            out_rms_sq: 1e-6,
            expansion_gain: 1.0,
            restored_transient_db: 0.0,
            sample_rate: sample_rate.max(8000.0),
        }
    }

    pub fn reset_internal(&mut self) {
        self.env_peak = 0.0;
        self.env_rms_sq = 1e-6;
        self.out_rms_sq = 1e-6;
        self.expansion_gain = 1.0;
        self.restored_transient_db = 0.0;
    }

    #[inline]
    pub fn process_frame(&mut self, l_in: f32, r_in: f32) -> [f32; 2] {
        if !self.enabled || self.amount_percent <= 0.0 {
            self.restored_transient_db = 0.0;
            return [l_in, r_in];
        }

        let xl = l_in as f64;
        let xr = r_in as f64;
        let peak = xl.abs().max(xr.abs());
        let energy = 0.5 * (xl * xl + xr * xr);

        let atk_alpha =
            (-1.0 / ((self.attack_ms.clamp(0.5, 50.0) as f64) * 0.001 * self.sample_rate)).exp();
        let rel_alpha =
            (-1.0 / ((self.release_ms.clamp(10.0, 500.0) as f64) * 0.001 * self.sample_rate))
                .exp();
        let rms_alpha = (-1.0 / (0.150 * self.sample_rate)).exp();

        self.env_peak = if peak > self.env_peak {
            atk_alpha * self.env_peak + (1.0 - atk_alpha) * peak
        } else {
            rel_alpha * self.env_peak + (1.0 - rel_alpha) * peak
        };
        self.env_rms_sq = rms_alpha * self.env_rms_sq + (1.0 - rms_alpha) * energy;

        let thresh_lin = 10.0f64.powf((self.threshold_db.clamp(-24.0, 0.0) as f64) / 20.0);
        let amt = (self.amount_percent.clamp(0.0, 100.0) as f64) * 0.01;
        let sens = (self.sensitivity_percent.clamp(0.0, 100.0) as f64) * 0.01;

        let mut boost_db = 0.0f64;
        if self.env_peak > thresh_lin {
            let over_ratio = self.env_peak / thresh_lin.max(1e-6);
            let over_db = 20.0 * over_ratio.log10();
            let rms = self.env_rms_sq.sqrt().max(1e-6);
            let crest = (self.env_peak / rms).clamp(1.0, 8.0);
            let crush_factor = (2.5 / crest).clamp(0.4, 2.0) * (0.5 + sens);
            boost_db = (over_db * amt * crush_factor * 1.4).clamp(0.0, 9.0);
        }

        self.restored_transient_db = boost_db as f32;
        let target_lin = 10.0f64.powf(boost_db / 20.0);
        self.expansion_gain =
            atk_alpha * self.expansion_gain + (1.0 - atk_alpha) * target_lin;

        let mut out_l = xl * self.expansion_gain;
        let mut out_r = xr * self.expansion_gain;

        let out_energy = 0.5 * (out_l * out_l + out_r * out_r);
        self.out_rms_sq = rms_alpha * self.out_rms_sq + (1.0 - rms_alpha) * out_energy;

        if self.auto_level_enabled && self.env_rms_sq > 1e-8 && self.out_rms_sq > 1e-8 {
            let comp = (self.env_rms_sq / self.out_rms_sq).sqrt().clamp(0.4, 1.2);
            out_l *= comp;
            out_r *= comp;
        }

        [out_l as f32, out_r as f32]
    }

    #[inline]
    pub fn process_frames_in_place(&mut self, io: &mut [StereoFrame]) {
        if !self.enabled {
            return;
        }
        for frame in io.iter_mut() {
            *frame = self.process_frame(frame[0], frame[1]);
        }
    }
}

impl OzoneDspModule for UnlimiterModule {
    fn name(&self) -> &'static str {
        "iZotope Ozone 12 Unlimiter"
    }

    fn reset(&mut self, sample_rate: f64, _max_block_size: usize) {
        self.sample_rate = sample_rate.max(8000.0);
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

impl UnifiedDspNode for UnlimiterModule {
    fn name(&self) -> &'static str {
        "iZotope Ozone 12 Unlimiter"
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
            1 => self.threshold_db = -24.0 + v * 24.0,
            2 => self.sensitivity_percent = v * 100.0,
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
// 3. OZONE 12 BASS CONTROL (LOW-END SUSTAIN POWER & PEAK PUNCH)
// ============================================================================

/// iZotope Ozone 12 Bass Control (`BassControlModule` / `OzoneBassControl`).
/// Balances sub/bass sustain foundation against kick peak punch with Clean/Dirty harmonic blend.
#[derive(Clone, Debug)]
pub struct BassControlModule {
    pub enabled: bool,
    /// Low-frequency crossover cutoff in `[40.0, 250.0]` Hz.
    pub cutoff_hz: f32,
    /// Sustain Power upward/downward leveling in `[-100.0, +100.0]`%.
    pub sustain_power_percent: f32,
    /// Peak Punch transient shaping in `[-100.0, +100.0]`%.
    pub peak_punch_percent: f32,
    /// Harmonic dirt blend in `[0.0, 100.0]`% (`0.0` = Clean linear, `100.0` = Dirty sub-harmonic saturation).
    pub dirty_blend_percent: f32,
    /// Punch envelope speed in milliseconds (`[5.0, 150.0]` ms).
    pub punch_speed_ms: f32,

    crossover: LinkwitzRiley4BandCrossover,
    env_fast: f64,
    env_slow: f64,
    sample_rate: f64,
}

pub type OzoneBassControl = BassControlModule;

impl Default for BassControlModule {
    fn default() -> Self {
        Self::new(48_000.0)
    }
}

impl BassControlModule {
    pub fn new(sample_rate: f64) -> Self {
        let sr = sample_rate.max(8000.0);
        Self {
            enabled: true,
            cutoff_hz: 110.0,
            sustain_power_percent: 25.0,
            peak_punch_percent: 20.0,
            dirty_blend_percent: 15.0,
            punch_speed_ms: 35.0,
            crossover: LinkwitzRiley4BandCrossover::new([110.0, 1200.0, 6000.0], sr),
            env_fast: 1e-6,
            env_slow: 1e-6,
            sample_rate: sr,
        }
    }

    pub fn set_cutoff_hz(&mut self, cutoff_hz: f32) {
        self.cutoff_hz = cutoff_hz.clamp(40.0, 250.0);
        self.crossover.set_crossover_frequencies(
            self.sample_rate,
            self.cutoff_hz as f64,
            1200.0,
            6000.0,
        );
    }

    pub fn reset_internal(&mut self) {
        self.crossover.reset();
        self.env_fast = 1e-6;
        self.env_slow = 1e-6;
    }

    #[inline]
    pub fn process_frame(&mut self, l_in: f32, r_in: f32) -> [f32; 2] {
        if !self.enabled {
            return [l_in, r_in];
        }

        let bands = self.crossover.split_stereo(l_in as f64, r_in as f64);
        let low_l = bands[0][0];
        let low_r = bands[0][1];
        let peak = low_l.abs().max(low_r.abs());

        let fast_tau = (self.punch_speed_ms.clamp(5.0, 150.0) as f64) * 0.00025;
        let slow_tau = (self.punch_speed_ms.clamp(5.0, 150.0) as f64) * 0.0025;
        let a_fast = (-1.0 / (fast_tau.max(0.001) * self.sample_rate)).exp();
        let a_slow = (-1.0 / (slow_tau.max(0.015) * self.sample_rate)).exp();

        self.env_fast = a_fast * self.env_fast + (1.0 - a_fast) * peak;
        self.env_slow = a_slow * self.env_slow + (1.0 - a_slow) * peak;

        let sust_amt = (self.sustain_power_percent.clamp(-100.0, 100.0) as f64) * 0.01;
        let punch_amt = (self.peak_punch_percent.clamp(-100.0, 100.0) as f64) * 0.01;
        let dirty = (self.dirty_blend_percent.clamp(0.0, 100.0) as f64) * 0.01;

        let sust_ref = (0.15 / (self.env_slow + 0.05)).clamp(0.5, 2.0);
        let sust_gain = 1.0 + sust_amt * (sust_ref - 1.0) * 0.65;

        let trans_crest =
            ((self.env_fast - self.env_slow) / (self.env_slow + 1e-5)).clamp(-1.0, 3.0);
        let punch_gain = (1.0 + punch_amt * trans_crest * 0.55).clamp(0.3, 2.2);

        let total_low_gain = sust_gain * punch_gain;
        let mut proc_low_l = low_l * total_low_gain;
        let mut proc_low_r = low_r * total_low_gain;

        if dirty > 0.0 {
            let sat_l = (proc_low_l * 1.8 + 0.08 * proc_low_l * proc_low_l).tanh() / 1.5;
            let sat_r = (proc_low_r * 1.8 + 0.08 * proc_low_r * proc_low_r).tanh() / 1.5;
            proc_low_l = (1.0 - dirty) * proc_low_l + dirty * sat_l;
            proc_low_r = (1.0 - dirty) * proc_low_r + dirty * sat_r;
        }

        let out_l = proc_low_l + bands[1][0] + bands[2][0] + bands[3][0];
        let out_r = proc_low_r + bands[1][1] + bands[2][1] + bands[3][1];
        [out_l as f32, out_r as f32]
    }

    #[inline]
    pub fn process_frames_in_place(&mut self, io: &mut [StereoFrame]) {
        if !self.enabled {
            return;
        }
        for frame in io.iter_mut() {
            *frame = self.process_frame(frame[0], frame[1]);
        }
    }
}

impl OzoneDspModule for BassControlModule {
    fn name(&self) -> &'static str {
        "iZotope Ozone 12 Bass Control"
    }

    fn reset(&mut self, sample_rate: f64, _max_block_size: usize) {
        self.sample_rate = sample_rate.max(8000.0);
        self.set_cutoff_hz(self.cutoff_hz);
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

impl UnifiedDspNode for BassControlModule {
    fn name(&self) -> &'static str {
        "iZotope Ozone 12 Bass Control"
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
            0 => self.set_cutoff_hz(40.0 + v * 210.0),
            1 => self.sustain_power_percent = -100.0 + v * 200.0,
            2 => self.peak_punch_percent = -100.0 + v * 200.0,
            3 => self.dirty_blend_percent = v * 100.0,
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
