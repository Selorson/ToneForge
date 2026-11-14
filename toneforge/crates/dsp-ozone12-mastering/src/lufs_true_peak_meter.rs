//! Broadcast-Standard Loudness & True-Peak Meter (`LufsTruePeakMeter` — ITU-R BS.1770-4 / EBU R128).
//!
//! Features:
//! 1. Exact 2-stage K-Weighting IIR filter (`Stage 1` head-acoustics High-Shelf + `Stage 2` RLB High-Pass)
//!    matching ITU-R BS.1770-4 Tables 1 & 2 at `48 kHz` and analytical bilinear transform at arbitrary rates.
//! 2. 4-Phase 12-Tap (`48-tap`) Polyphase FIR `4x` Oversampling True-Peak Detector (`TruePeakDetector4x`)
//!    matching ITU-R BS.1770-4 Annex 2 Table 3 to capture inter-sample peaks (`dBTP`).
//! 3. Momentary (`400 ms`), Short-Term (`3.0 s`), and Integrated (`-70 LUFS` absolute + `-10 LU` relative gated)
//!    LUFS loudness computation with **Zero Heap Allocation** on the real-time audio thread.
//! 4. Direct zero-copy export to `Ozone12MeterFrame` (`Ozone12MasterTelemetry`) and `VisualTelemetryFrame`.

use crate::chain_element::{OzoneDspModule, StereoBiquadF64};
use daw_core_graph::{
    Ozone12MasterTelemetry, ProcessContext, StereoFrame, UnifiedDspNode, VisualTelemetryFrame,
};
use std::f64::consts::PI;

/// Type alias matching the `Ozone12MeterFrame` name in the Phase 3 specification.
pub type Ozone12MeterFrame = Ozone12MasterTelemetry;

/// ITU-R BS.1770-4 Annex 2 Table 3: 4-phase x 12-tap (48-tap) polyphase FIR interpolation filter
/// for `4x` oversampled True-Peak detection.
pub const ITU_BS1770_4_POLYPHASE_TAPS: [[f32; 12]; 4] = [
    // Phase 0
    [
        0.0017089844,
        0.010986328,
        -0.01965332,
        0.033203125,
        -0.059448242,
        0.1373291,
        0.97216797,
        -0.10229492,
        0.047607422,
        -0.026611328,
        0.014892578,
        -0.008300781,
    ],
    // Phase 1
    [
        -0.029174805,
        0.029296875,
        -0.051757812,
        0.08911133,
        -0.1665039,
        0.4650879,
        0.77978516,
        -0.20031738,
        0.09082031,
        -0.05029297,
        0.028076172,
        -0.015625,
    ],
    // Phase 2
    [
        -0.018920898,
        0.033081055,
        -0.05822754,
        0.1015625,
        -0.20031738,
        0.77978516,
        0.4650879,
        -0.1665039,
        0.08911133,
        -0.051757812,
        0.029296875,
        -0.029174805,
    ],
    // Phase 3
    [
        -0.008300781,
        0.014892578,
        -0.026611328,
        0.047607422,
        -0.10229492,
        0.97216797,
        0.1373291,
        -0.059448242,
        0.033203125,
        -0.01965332,
        0.010986328,
        0.0017089844,
    ],
];

/// Evaluates the instantaneous 4x oversampled True-Peak magnitude over a 12-sample history `[x[n], x[n-1], ..., x[n-11]]`.
#[inline(always)]
pub fn eval_4x_true_peak_window(hist: &[f32; 12]) -> f32 {
    let mut max_abs = hist[0].abs();
    for m in 1..12 {
        let a = hist[m].abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    for phase in &ITU_BS1770_4_POLYPHASE_TAPS {
        let mut sum = 0.0f32;
        for m in 0..12 {
            sum += phase[m] * hist[m];
        }
        let abs_v = sum.abs();
        if abs_v > max_abs {
            max_abs = abs_v;
        }
    }
    max_abs
}

/// 4-Phase 12-Tap (`48-tap`) `4x` Oversampling True-Peak Detector (`ITU-R BS.1770-4`).
/// Zero Heap Allocation.
#[derive(Clone, Copy, Debug)]
pub struct TruePeakDetector4x {
    hist: [[f32; 12]; 2],
    pub max_true_peak_lin: [f32; 2],
    pub max_sample_peak_lin: [f32; 2],
}

impl Default for TruePeakDetector4x {
    fn default() -> Self {
        Self::new()
    }
}

impl TruePeakDetector4x {
    #[inline]
    pub const fn new() -> Self {
        Self {
            hist: [[0.0; 12]; 2],
            max_true_peak_lin: [0.0; 2],
            max_sample_peak_lin: [0.0; 2],
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.hist = [[0.0; 12]; 2];
        self.max_true_peak_lin = [0.0; 2];
        self.max_sample_peak_lin = [0.0; 2];
    }

    /// Pushes a single sample for channel `ch` (`0 = Left, 1 = Right`) and returns the instantaneous
    /// 4x oversampled True-Peak linear amplitude at this sample step.
    #[inline(always)]
    pub fn step_channel(&mut self, ch: usize, x: f32) -> f32 {
        let h = &mut self.hist[ch];
        for m in (1..12).rev() {
            h[m] = h[m - 1];
        }
        h[0] = x;

        let sp = x.abs();
        if sp > self.max_sample_peak_lin[ch] {
            self.max_sample_peak_lin[ch] = sp;
        }

        let mut tp = sp;
        for phase in &ITU_BS1770_4_POLYPHASE_TAPS {
            let mut sum = 0.0f32;
            for m in 0..12 {
                sum += phase[m] * h[m];
            }
            let a = sum.abs();
            if a > tp {
                tp = a;
            }
        }
        if tp > self.max_true_peak_lin[ch] {
            self.max_true_peak_lin[ch] = tp;
        }
        tp
    }

    /// Pushes a stereo frame `[L, R]` and returns `[tp_l, tp_r]` linear true peaks for this frame.
    #[inline(always)]
    pub fn step_stereo(&mut self, l: f32, r: f32) -> [f32; 2] {
        [self.step_channel(0, l), self.step_channel(1, r)]
    }

    /// Returns the maximum True Peak in dBTP (`[Left dBTP, Right dBTP]`).
    #[inline]
    pub fn true_peak_dbtp(&self) -> [f32; 2] {
        [
            linear_to_dbtp(self.max_true_peak_lin[0]),
            linear_to_dbtp(self.max_true_peak_lin[1]),
        ]
    }

    /// Returns the highest True Peak across both stereo channels in dBTP.
    #[inline]
    pub fn max_stereo_dbtp(&self) -> f32 {
        linear_to_dbtp(self.max_true_peak_lin[0].max(self.max_true_peak_lin[1]))
    }
}

#[inline]
pub fn linear_to_dbtp(lin: f32) -> f32 {
    if lin <= 1e-7 {
        -140.0
    } else {
        20.0 * lin.log10()
    }
}

/// ITU-R BS.1770-4 2-Stage K-Weighting IIR Filter (`Stage 1` Acoustic Head High-Shelf + `Stage 2` RLB High-Pass).
#[derive(Clone, Copy, Debug)]
pub struct KWeightingStereoFilter {
    stage1_pre: StereoBiquadF64,
    stage2_rlb: StereoBiquadF64,
}

impl Default for KWeightingStereoFilter {
    fn default() -> Self {
        Self::new(48_000.0)
    }
}

impl KWeightingStereoFilter {
    pub fn new(sample_rate: f64) -> Self {
        let mut f = Self {
            stage1_pre: StereoBiquadF64::new_unity(),
            stage2_rlb: StereoBiquadF64::new_unity(),
        };
        f.set_sample_rate(sample_rate);
        f
    }

    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        let sr = sample_rate.max(8000.0);
        if (sr - 48_000.0).abs() < 0.5 {
            // Exact ITU-R BS.1770-4 Table 1 (Stage 1 pre-filter @ 48 kHz)
            self.stage1_pre.b0 = 1.53512485958697;
            self.stage1_pre.b1 = -2.69169618940638;
            self.stage1_pre.b2 = 1.19839281085285;
            self.stage1_pre.a1 = -1.69065929318241;
            self.stage1_pre.a2 = 0.73248077421585;

            // Exact ITU-R BS.1770-4 Table 2 (Stage 2 RLB filter @ 48 kHz)
            self.stage2_rlb.b0 = 1.0;
            self.stage2_rlb.b1 = -2.0;
            self.stage2_rlb.b2 = 1.0;
            self.stage2_rlb.a1 = -1.99004745483398;
            self.stage2_rlb.a2 = 0.99007225036621;
        } else {
            // Analytical bilinear transformation of ITU-R BS.1770-4 analog prototypes:
            // Stage 1: f0 = 1681.974450955533 Hz, G = +3.999843853973347 dB, Q = 0.7071752369554196
            let f0_1 = 1681.974450955533;
            let g_db_1 = 3.999843853973347;
            let q_1 = 0.7071752369554196;
            let vh = 10.0f64.powf(g_db_1 / 20.0);
            let vb = vh.powf(0.4996667741545416);
            let k1 = (PI * f0_1 / sr).tan();
            let k1_sq = k1 * k1;
            let a0_1 = 1.0 + k1 / q_1 + k1_sq;
            self.stage1_pre.b0 = (vh + vb * k1 / q_1 + k1_sq) / a0_1;
            self.stage1_pre.b1 = 2.0 * (k1_sq - vh) / a0_1;
            self.stage1_pre.b2 = (vh - vb * k1 / q_1 + k1_sq) / a0_1;
            self.stage1_pre.a1 = 2.0 * (k1_sq - 1.0) / a0_1;
            self.stage1_pre.a2 = (1.0 - k1 / q_1 + k1_sq) / a0_1;

            // Stage 2: f0 = 38.13547087602444 Hz, Q = 0.5003270373238773
            let f0_2 = 38.13547087602444;
            let q_2 = 0.5003270373238773;
            let k2 = (PI * f0_2 / sr).tan();
            let k2_sq = k2 * k2;
            let a0_2 = 1.0 + k2 / q_2 + k2_sq;
            self.stage2_rlb.b0 = 1.0 / a0_2;
            self.stage2_rlb.b1 = -2.0 / a0_2;
            self.stage2_rlb.b2 = 1.0 / a0_2;
            self.stage2_rlb.a1 = 2.0 * (k2_sq - 1.0) / a0_2;
            self.stage2_rlb.a2 = (1.0 - k2 / q_2 + k2_sq) / a0_2;
        }
    }

    pub fn reset(&mut self) {
        self.stage1_pre.reset_state();
        self.stage2_rlb.reset_state();
    }

    #[inline(always)]
    pub fn step_stereo(&mut self, l: f64, r: f64) -> [f64; 2] {
        let s1 = self.stage1_pre.step_stereo([l, r]);
        self.stage2_rlb.step_stereo(s1)
    }
}

/// Number of `100 ms` sub-blocks kept for Short-Term (`3.0 s` = 30 sub-blocks) and Momentary (`400 ms` = 4 sub-blocks).
const SHORT_TERM_SUBBLOCKS: usize = 30;
/// Integrated LUFS gating histogram: `400 ms` block loudness from `-70` to `+10` LUFS in `0.01 LU`
/// bins. Every gating block of the whole programme is kept (exact energy sum per bin), so Integrated
/// loudness covers the full measurement period as ITU-R BS.1770-4 / EBU R128 require, with fixed
/// memory and no allocation on the audio thread.
const GATING_HIST_MIN_LUFS: f64 = -70.0;
const GATING_HIST_BIN_LU: f64 = 0.01;
const GATING_HIST_BINS: usize = 8_000;

/// Broadcast-standard ITU-R BS.1770-4 Loudness (`Momentary`, `Short-Term`, `Integrated` LUFS)
/// and `4x` Oversampled True-Peak (`dBTP`) Meter with Gain Reduction trace.
///
/// Zero Heap Allocation on the real-time audio thread.
#[derive(Clone, Debug)]
pub struct LufsTruePeakMeter {
    pub sample_rate: f64,
    pub k_filter: KWeightingStereoFilter,
    pub true_peak_detector: TruePeakDetector4x,

    // 100ms sub-block energy accumulator (`z_L + z_R`)
    subblock_target_samples: usize,
    subblock_sample_count: usize,
    subblock_energy_sum: f64,

    // Total cumulative energy since last reset (for immediate sub-400ms accuracy)
    total_samples: u64,
    total_energy_sum: f64,

    // Circular buffer of 100ms sub-block mean-square powers `z_sub = (mean(y_L^2) + mean(y_R^2))`
    subblock_ring: [f64; SHORT_TERM_SUBBLOCKS],
    subblock_write_pos: usize,
    subblock_filled: usize,

    // Whole-programme histogram of 400ms block powers for ITU-R BS.1770-4 Integrated LUFS gating
    gating_hist_energy: Vec<f64>,
    gating_hist_count: Vec<u32>,
    gating_total_energy: f64,
    gating_total_count: u64,

    // Published LUFS readings
    pub lufs_momentary: f32,
    pub lufs_short_term: f32,
    pub lufs_integrated: f32,
    pub current_gr_db: f32,

    // Telemetry rings for `Ozone12MasterTelemetry` (`Ozone12MeterFrame`)
    pub irc_gr_trace: [f32; 512],
    pub polar_sample_xy: [[f32; 2]; 512],
    trace_cursor: usize,
}

impl Default for LufsTruePeakMeter {
    fn default() -> Self {
        Self::new(48_000.0)
    }
}

impl LufsTruePeakMeter {
    pub fn new(sample_rate: f64) -> Self {
        let sr = sample_rate.max(8000.0);
        let subblock_target_samples = ((0.100 * sr).round() as usize).max(64);
        Self {
            sample_rate: sr,
            k_filter: KWeightingStereoFilter::new(sr),
            true_peak_detector: TruePeakDetector4x::new(),
            subblock_target_samples,
            subblock_sample_count: 0,
            subblock_energy_sum: 0.0,
            total_samples: 0,
            total_energy_sum: 0.0,
            subblock_ring: [0.0; SHORT_TERM_SUBBLOCKS],
            subblock_write_pos: 0,
            subblock_filled: 0,
            gating_hist_energy: vec![0.0; GATING_HIST_BINS],
            gating_hist_count: vec![0; GATING_HIST_BINS],
            gating_total_energy: 0.0,
            gating_total_count: 0,
            lufs_momentary: -70.0,
            lufs_short_term: -70.0,
            lufs_integrated: -70.0,
            current_gr_db: 0.0,
            irc_gr_trace: [0.0; 512],
            polar_sample_xy: [[0.0; 2]; 512],
            trace_cursor: 0,
        }
    }

    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        let sr = sample_rate.max(8000.0);
        self.sample_rate = sr;
        self.subblock_target_samples = ((0.100 * sr).round() as usize).max(64);
        self.k_filter.set_sample_rate(sr);
        self.reset();
    }

    pub fn reset(&mut self) {
        self.k_filter.reset();
        self.true_peak_detector.reset();
        self.subblock_sample_count = 0;
        self.subblock_energy_sum = 0.0;
        self.total_samples = 0;
        self.total_energy_sum = 0.0;
        self.subblock_ring = [0.0; SHORT_TERM_SUBBLOCKS];
        self.subblock_write_pos = 0;
        self.subblock_filled = 0;
        self.gating_hist_energy.fill(0.0);
        self.gating_hist_count.fill(0);
        self.gating_total_energy = 0.0;
        self.gating_total_count = 0;
        self.lufs_momentary = -70.0;
        self.lufs_short_term = -70.0;
        self.lufs_integrated = -70.0;
        self.current_gr_db = 0.0;
        self.irc_gr_trace = [0.0; 512];
        self.polar_sample_xy = [[0.0; 2]; 512];
        self.trace_cursor = 0;
    }

    /// Converts summed stereo mean-square K-weighted energy `z = z_L + z_R` to LUFS per ITU-R BS.1770-4:
    /// `LUFS = -0.691 + 10 * log10(z_L + z_R)`.
    #[inline]
    pub fn energy_to_lufs(z_sum: f64) -> f32 {
        if z_sum <= 1e-14 {
            -100.0
        } else {
            (-0.691 + 10.0 * z_sum.log10()).max(-100.0) as f32
        }
    }

    /// Pushes a single stereo frame `[L, R]` and optional instantaneous gain reduction `gr_db` (`<= 0.0`).
    /// Zero Heap Allocation.
    #[inline]
    pub fn push_frame(&mut self, l: f32, r: f32, gr_db: f32) {
        // 1. Update 4x oversampled True-Peak detector
        let _ = self.true_peak_detector.step_stereo(l, r);

        // 2. K-Weighting IIR filter + mean-square accumulation
        let [kl, kr] = self.k_filter.step_stereo(l as f64, r as f64);
        let frame_power = kl * kl + kr * kr;
        self.subblock_energy_sum += frame_power;
        self.subblock_sample_count += 1;
        self.total_energy_sum += frame_power;
        self.total_samples += 1;

        // 3. Record Gain Reduction trace and Vectorscope polar XY sample
        self.current_gr_db = gr_db;
        let tc = self.trace_cursor;
        self.irc_gr_trace[tc] = gr_db;
        self.polar_sample_xy[tc] = [l, r];
        self.trace_cursor = (tc + 1) & 511;

        // 4. Commit 100ms sub-block when full
        if self.subblock_sample_count >= self.subblock_target_samples {
            let z_sub = self.subblock_energy_sum / (self.subblock_sample_count as f64);
            self.subblock_energy_sum = 0.0;
            self.subblock_sample_count = 0;

            self.subblock_ring[self.subblock_write_pos] = z_sub;
            self.subblock_write_pos = (self.subblock_write_pos + 1) % SHORT_TERM_SUBBLOCKS;
            if self.subblock_filled < SHORT_TERM_SUBBLOCKS {
                self.subblock_filled += 1;
            }

            self.recompute_windowed_lufs();
        } else if self.subblock_filled == 0 && (self.total_samples & 63) == 0 {
            // Provide immediate K-weighted loudness reading during the first 100ms window
            let z_inst = self.total_energy_sum / (self.total_samples as f64);
            let inst_lufs = Self::energy_to_lufs(z_inst);
            self.lufs_momentary = inst_lufs;
            self.lufs_short_term = inst_lufs;
        }
    }

    /// Processes a slice of stereo frames (Zero Heap Allocation).
    #[inline]
    pub fn push_block(&mut self, frames: &[StereoFrame], gr_db: f32) {
        for &[l, r] in frames {
            self.push_frame(l, r, gr_db);
        }
        // Refresh momentary/short-term if still within initial sub-block
        if self.subblock_filled == 0 && self.total_samples > 0 {
            let z_inst = self.total_energy_sum / (self.total_samples as f64);
            let inst_lufs = Self::energy_to_lufs(z_inst);
            self.lufs_momentary = inst_lufs;
            self.lufs_short_term = inst_lufs;
        }
    }

    fn recompute_windowed_lufs(&mut self) {
        if self.subblock_filled == 0 {
            return;
        }
        // Momentary LUFS: average of up to 4 most recent 100ms sub-blocks (400ms)
        let mom_count = self.subblock_filled.min(4);
        let mut mom_sum = 0.0f64;
        for k in 0..mom_count {
            let idx = (self.subblock_write_pos + SHORT_TERM_SUBBLOCKS - 1 - k) % SHORT_TERM_SUBBLOCKS;
            mom_sum += self.subblock_ring[idx];
        }
        let z_mom = mom_sum / (mom_count as f64);
        self.lufs_momentary = Self::energy_to_lufs(z_mom);

        // Short-Term LUFS: average of up to 30 most recent 100ms sub-blocks (3.0s)
        let st_count = self.subblock_filled.min(SHORT_TERM_SUBBLOCKS);
        let mut st_sum = 0.0f64;
        for k in 0..st_count {
            let idx = (self.subblock_write_pos + SHORT_TERM_SUBBLOCKS - 1 - k) % SHORT_TERM_SUBBLOCKS;
            st_sum += self.subblock_ring[idx];
        }
        let z_st = st_sum / (st_count as f64);
        self.lufs_short_term = Self::energy_to_lufs(z_st);

        // When at least 4 sub-blocks (400ms) are available, the 400ms block power `z_mom` (75%
        // overlap, 100ms hop) enters Integrated gating.
        if self.subblock_filled >= 4 {
            // Absolute gate: -70 LUFS
            let z_abs_gate = 10.0f64.powf((GATING_HIST_MIN_LUFS + 0.691) / 10.0);
            if z_mom > z_abs_gate {
                let block_lufs = -0.691 + 10.0 * z_mom.log10();
                let bin = (((block_lufs - GATING_HIST_MIN_LUFS) / GATING_HIST_BIN_LU) as usize)
                    .min(GATING_HIST_BINS - 1);
                self.gating_hist_energy[bin] += z_mom;
                self.gating_hist_count[bin] += 1;
                self.gating_total_energy += z_mom;
                self.gating_total_count += 1;
            }

            if self.gating_total_count > 0 {
                // Relative gate: -10 LU below the absolute-gated mean loudness of the programme.
                let mean_abs = self.gating_total_energy / (self.gating_total_count as f64);
                let rel_gate_lufs = -0.691 + 10.0 * (mean_abs * 0.1).log10();
                let first_bin = (((rel_gate_lufs - GATING_HIST_MIN_LUFS) / GATING_HIST_BIN_LU)
                    .ceil()
                    .max(0.0) as usize)
                    .min(GATING_HIST_BINS);
                let mut sum_rel = 0.0f64;
                let mut count_rel = 0u64;
                for bin in first_bin..GATING_HIST_BINS {
                    sum_rel += self.gating_hist_energy[bin];
                    count_rel += self.gating_hist_count[bin] as u64;
                }
                self.lufs_integrated = if count_rel > 0 {
                    Self::energy_to_lufs(sum_rel / (count_rel as f64))
                } else {
                    Self::energy_to_lufs(mean_abs)
                };
            }
        }
    }

    /// `true` once at least one 400 ms gating block exists (Integrated LUFS is defined).
    #[inline]
    pub fn integrated_is_defined(&self) -> bool {
        self.gating_total_count > 0
    }

    /// Ungated K-weighted loudness of everything measured so far (used only to describe
    /// programmes shorter than one 400 ms gating block, where BS.1770 Integrated is undefined).
    #[inline]
    pub fn ungated_programme_lufs(&self) -> f32 {
        if self.total_samples == 0 {
            -70.0
        } else {
            Self::energy_to_lufs(self.total_energy_sum / (self.total_samples as f64))
        }
    }

    /// Returns `[Left dBTP, Right dBTP]` from the 4x oversampling True-Peak detector.
    #[inline]
    pub fn true_peak_dbtp(&self) -> [f32; 2] {
        self.true_peak_detector.true_peak_dbtp()
    }

    /// Returns the maximum True Peak across L/R in dBTP.
    #[inline]
    pub fn max_true_peak_dbtp(&self) -> f32 {
        self.true_peak_detector.max_stereo_dbtp()
    }

    /// Exports Gain Reduction trace and Polar XY vectorscope directly into `Ozone12MeterFrame` (`8 KB`).
    #[inline]
    pub fn export_to_meter_frame(&self, meter_frame: &mut Ozone12MeterFrame) {
        meter_frame.irc_gr_trace = self.irc_gr_trace;
        meter_frame.polar_sample_xy = self.polar_sample_xy;
    }

    /// Exports LUFS (`Momentary`, `Short-Term`, `Integrated`), True Peak (`dBTP`), Master GR,
    /// and `Ozone12MeterFrame` directly into `VisualTelemetryFrame` (`128 KB`).
    #[inline]
    pub fn export_to_visual_telemetry(&self, frame: &mut VisualTelemetryFrame) {
        frame.header.lufs_momentary = self.lufs_momentary;
        frame.header.lufs_short_term = self.lufs_short_term;
        frame.header.lufs_integrated = self.lufs_integrated;
        frame.header.true_peak_db = self.true_peak_dbtp();
        frame.header.master_gr_db = self.current_gr_db;
        self.export_to_meter_frame(&mut frame.ozone12_telemetry);
    }
}

impl OzoneDspModule for LufsTruePeakMeter {
    fn name(&self) -> &'static str {
        "ITU-R BS.1770-4 K-Weighted LUFS & 4x True-Peak Meter"
    }

    fn reset(&mut self, sample_rate: f64, _max_block_size: usize) {
        self.set_sample_rate(sample_rate);
    }

    fn process_block(&mut self, left: &mut [f32], right: &mut [f32]) {
        let n = left.len().min(right.len());
        for i in 0..n {
            self.push_frame(left[i], right[i], self.current_gr_db);
        }
    }
}

impl UnifiedDspNode for LufsTruePeakMeter {
    fn name(&self) -> &'static str {
        "ITU-R BS.1770-4 K-Weighted LUFS & 4x True-Peak Meter"
    }

    fn latency_samples(&self) -> usize {
        0
    }

    fn prepare(&mut self, sample_rate: f64, _max_block_size: usize) {
        self.set_sample_rate(sample_rate);
    }

    fn reset_state(&mut self) {
        self.reset();
    }

    fn set_parameter_normalized(&mut self, _param_id: u32, _norm_val: f32) {}

    fn current_gain_reduction_db(&self) -> f32 {
        self.current_gr_db
    }

    fn export_visual_telemetry(&self, frame: &mut VisualTelemetryFrame) {
        self.export_to_visual_telemetry(frame);
    }

    fn process_slice(
        &mut self,
        _ctx: &ProcessContext,
        main_io: &mut [StereoFrame],
        _sidechain_in: Option<&[StereoFrame]>,
        _aux_sends_out: &mut [&mut [StereoFrame]],
    ) {
        self.push_block(main_io, self.current_gr_db);
    }
}
