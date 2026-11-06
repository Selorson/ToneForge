//! Automation Clips, Signed-Exponential Tension Spline (`TMEnvelope` `VMT = 0x00b97508`),
//! LFO Modulator (`TMLFO` `VMT = 0x00b976a8`), and 30-Bit Fixed-Point Internal Controller Bus
//! (`0x40000000` / `1_073_741_824.0`, `TMixVoiceAutomationClip` `sub_d24d80` at `0x00d24d80`).
//!
//! All evaluation and controller propagation routines operate with **Zero Heap Allocation**.

use crate::{FIXED_POINT_30BIT_MAX, FIXED_POINT_30BIT_SCALE};
use std::f64::consts::TAU;

/// Number of internal controller channels on the 30-bit fixed-point bus
/// (matches `AutomationAndSplineTelemetry::controller_values_norm: [f32; 512]`).
pub const NUM_INTERNAL_CONTROLLERS: usize = 512;

/// Maximum pre-allocated parameter links in `InternalControllerBus`.
pub const MAX_CONTROLLER_LINKS: usize = 1024;

/// Converts a normalized value `[0.0, 1.0]` to FL Studio 26's 30-bit fixed-point range
/// `0..=0x40000000` (`sub_d24d80` -> `sub_40a1e0` `round`).
#[inline(always)]
pub fn normalized_to_fixed30(norm: f64) -> i32 {
    let clamped = norm.clamp(0.0, 1.0);
    ((clamped * FIXED_POINT_30BIT_SCALE).round() as i32).clamp(0, FIXED_POINT_30BIT_MAX)
}

/// Converts a 30-bit fixed-point internal controller value `0..=0x40000000` back to `[0.0, 1.0]`.
#[inline(always)]
pub fn fixed30_to_normalized(fixed30: i32) -> f64 {
    (fixed30.clamp(0, FIXED_POINT_30BIT_MAX) as f64) / FIXED_POINT_30BIT_SCALE
}

/// Single control point on a `TensionSpline` (`TMEnvelope` `VMT = 0x00b97508`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TensionControlPoint {
    pub x: f64,
    pub y: f64,
    /// Tension parameter in `[-1.0, 1.0]`:
    /// - `0.0` = strictly linear interpolation
    /// - `> 0.0` = concave exponential curve (`((k*t).exp() - 1) / (k.exp() - 1)` with `k = 10 * tension`)
    /// - `< 0.0` = convex exponential curve (`k < 0`)
    pub tension: f64,
}

impl TensionControlPoint {
    #[inline]
    pub const fn new(x: f64, y: f64, tension: f64) -> Self {
        Self { x, y, tension }
    }
}

/// Signed-exponential tension spline evaluator (`TMEnvelope` `VMT = 0x00b97508` in `FLEngine_x64.dll`
/// and `Maximus_x64.dll` `0x007958b8`).
#[derive(Clone, Debug, Default)]
pub struct TensionSpline {
    pub points: Vec<TensionControlPoint>,
}

impl TensionSpline {
    pub fn new(mut points: Vec<TensionControlPoint>) -> Self {
        points.sort_by(|a, b| a.x.total_cmp(&b.x));
        Self { points }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            points: Vec::with_capacity(capacity),
        }
    }

    /// Evaluates the signed exponential shaping function `S(t, tension)` on `t in [0.0, 1.0]`.
    #[inline(always)]
    pub fn shape_normalized_t(t: f64, tension: f64) -> f64 {
        let t_clamped = t.clamp(0.0, 1.0);
        let ten = tension.clamp(-1.0, 1.0);
        if ten.abs() < 1e-5 {
            t_clamped
        } else {
            let k = ten * 10.0;
            ((k * t_clamped).exp() - 1.0) / (k.exp() - 1.0)
        }
    }

    /// Evaluates the spline at position `x` (Zero Heap Allocation).
    #[inline]
    pub fn eval(&self, x: f64) -> f64 {
        if self.points.is_empty() {
            return 0.0;
        }
        if x <= self.points[0].x {
            return self.points[0].y;
        }
        let last = self.points.len() - 1;
        if x >= self.points[last].x {
            return self.points[last].y;
        }
        let idx = self.points.partition_point(|p| p.x <= x).saturating_sub(1);
        let p0 = self.points[idx];
        let p1 = self.points[idx + 1];
        let dx = (p1.x - p0.x).max(1e-12);
        let t = ((x - p0.x) / dx).clamp(0.0, 1.0);
        let shaped_t = Self::shape_normalized_t(t, p1.tension);
        p0.y + (p1.y - p0.y) * shaped_t
    }

    /// Evaluates the spline at `x` and quantizes the output to FL Studio 26's 30-bit Fixed-Point
    /// Internal Controller representation `0..=0x40000000` (`sub_d24d80`).
    #[inline]
    pub fn eval_fixed30(&self, x: f64) -> i32 {
        normalized_to_fixed30(self.eval(x))
    }
}

/// LFO waveform types (`TMLFO` `VMT = 0x00b976a8`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LfoWaveform {
    Sine,
    Triangle,
    Square,
    Saw,
}

/// LFO Modulator (`TMLFO` `VMT = 0x00b976a8`) superimposed on `TensionSpline` in `TACArticulator`.
#[derive(Clone, Copy, Debug)]
pub struct LfoModulator {
    pub enabled: bool,
    pub waveform: LfoWaveform,
    pub frequency_cycles_per_tick: f64,
    pub phase_offset: f64,
    pub amount: f64, // [-1.0, 1.0]
}

impl Default for LfoModulator {
    fn default() -> Self {
        Self {
            enabled: false,
            waveform: LfoWaveform::Sine,
            frequency_cycles_per_tick: 1.0 / 96.0,
            phase_offset: 0.0,
            amount: 0.0,
        }
    }
}

impl LfoModulator {
    #[inline]
    pub fn eval_bipolar(&self, x_tick: f64) -> f64 {
        if !self.enabled || self.amount.abs() < 1e-9 {
            return 0.0;
        }
        let phase = (x_tick * self.frequency_cycles_per_tick + self.phase_offset).rem_euclid(1.0);
        let raw = match self.waveform {
            LfoWaveform::Sine => (TAU * phase).sin(),
            LfoWaveform::Triangle => 1.0 - 4.0 * (phase - 0.5).abs(),
            LfoWaveform::Square => {
                if phase < 0.5 {
                    1.0
                } else {
                    -1.0
                }
            }
            LfoWaveform::Saw => 2.0 * phase - 1.0,
        };
        raw * self.amount
    }
}

/// Combined Automation Clip Articulator (`TMixVoiceAutomationClip` `VMT = 0x00d1ff48`, `sub_d24d80`).
#[derive(Clone, Debug, Default)]
pub struct AutomationClipArticulator {
    pub controller_id: u16,
    pub spline: TensionSpline,
    pub lfo: LfoModulator,
    pub min_level: f64,
    pub max_level: f64,
}

impl AutomationClipArticulator {
    pub fn new(controller_id: u16, spline: TensionSpline) -> Self {
        Self {
            controller_id,
            spline,
            lfo: LfoModulator::default(),
            min_level: 0.0,
            max_level: 1.0,
        }
    }

    /// Evaluates `TACArticulator` (`spline + lfo`) at position `x_tick` and outputs 30-bit
    /// fixed-point controller value `0..=0x40000000` (`sub_d24d80`).
    #[inline]
    pub fn evaluate_fixed30(&self, x_tick: f64) -> i32 {
        let env_val = self.spline.eval(x_tick).clamp(0.0, 1.0);
        let scaled = self.min_level + env_val * (self.max_level - self.min_level);
        let with_lfo = (scaled + self.lfo.eval_bipolar(x_tick)).clamp(0.0, 1.0);
        normalized_to_fixed30(with_lfo)
    }
}

/// Link mapping an internal 30-bit fixed-point controller (`0..512`) to a track/slot parameter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ControllerLink {
    pub controller_id: u16,
    pub target_track: u16,
    pub slot_index: u8,
    pub param_id: u32,
    pub min_out_norm: f32,
    pub max_out_norm: f32,
    pub inverted: bool,
}

impl ControllerLink {
    /// Maps a 30-bit fixed-point controller value `0..=0x40000000` to the target normalized parameter.
    #[inline(always)]
    pub fn map_fixed30_to_norm(&self, value_fixed30: i32) -> f32 {
        let mut t = fixed30_to_normalized(value_fixed30) as f32;
        if self.inverted {
            t = 1.0 - t;
        }
        (self.min_out_norm + t * (self.max_out_norm - self.min_out_norm)).clamp(0.0, 1.0)
    }
}

/// 30-Bit Fixed-Point Internal Controller Bus (`0x40000000` scale, `sub_d24d80`).
/// Stores 512 controller slots and propagates values to linked parameters with Zero Heap Allocation.
pub struct InternalControllerBus {
    values_fixed30: [i32; NUM_INTERNAL_CONTROLLERS],
    links: Vec<ControllerLink>,
    link_count: usize,
}

impl Default for InternalControllerBus {
    fn default() -> Self {
        Self::new()
    }
}

impl InternalControllerBus {
    pub fn new() -> Self {
        let dummy_link = ControllerLink {
            controller_id: 0,
            target_track: 0,
            slot_index: 0,
            param_id: 0,
            min_out_norm: 0.0,
            max_out_norm: 1.0,
            inverted: false,
        };
        Self {
            values_fixed30: [0; NUM_INTERNAL_CONTROLLERS],
            links: vec![dummy_link; MAX_CONTROLLER_LINKS],
            link_count: 0,
        }
    }

    /// Publishes a 30-bit fixed-point value (`0..=0x40000000`) to `controller_id`.
    #[inline(always)]
    pub fn publish_fixed30(&mut self, controller_id: u16, value_fixed30: i32) {
        let idx = (controller_id as usize) & (NUM_INTERNAL_CONTROLLERS - 1);
        self.values_fixed30[idx] = value_fixed30.clamp(0, FIXED_POINT_30BIT_MAX);
    }

    /// Publishes a normalized value `[0.0, 1.0]` quantized to 30-bit fixed-point (`0x40000000`).
    #[inline(always)]
    pub fn publish_normalized(&mut self, controller_id: u16, norm: f64) {
        self.publish_fixed30(controller_id, normalized_to_fixed30(norm));
    }

    /// Reads the current 30-bit fixed-point value (`0..=0x40000000`) of `controller_id`.
    #[inline(always)]
    pub fn get_fixed30(&self, controller_id: u16) -> i32 {
        let idx = (controller_id as usize) & (NUM_INTERNAL_CONTROLLERS - 1);
        self.values_fixed30[idx]
    }

    /// Reads the normalized `[0.0, 1.0]` value of `controller_id`.
    #[inline(always)]
    pub fn get_normalized(&self, controller_id: u16) -> f64 {
        fixed30_to_normalized(self.get_fixed30(controller_id))
    }

    /// Registers a controller-to-parameter link (Zero Heap Allocation).
    pub fn add_link(&mut self, link: ControllerLink) -> bool {
        if self.link_count >= self.links.len() {
            return false;
        }
        self.links[self.link_count] = link;
        self.link_count += 1;
        true
    }

    pub fn clear_links(&mut self) {
        self.link_count = 0;
    }

    #[inline]
    pub fn active_links(&self) -> &[ControllerLink] {
        &self.links[..self.link_count]
    }

    /// Exports all 512 controller values as normalized `f32` into `VisualTelemetryFrame`
    /// (`AutomationAndSplineTelemetry::controller_values_norm`).
    #[inline]
    pub fn export_telemetry_norm(&self, out_norm: &mut [f32; NUM_INTERNAL_CONTROLLERS]) {
        let inv_scale = (1.0 / FIXED_POINT_30BIT_SCALE) as f32;
        for (dst, &val) in out_norm.iter_mut().zip(self.values_fixed30.iter()) {
            *dst = (val as f32) * inv_scale;
        }
    }
}
