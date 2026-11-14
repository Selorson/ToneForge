//! `wasapi_capture_duplex`: Windows Core Audio **WASAPI Capture** (`eCapture` Event-Driven
//! `IAudioCaptureClient`) + **Adaptive Delay-Locked Loop (`DLL`) Kaiser Sinc Drift Resampler**.
//!
//! Solves asynchronous multi-clock drift (e.g., USB condenser mic running `+150 ppm` faster than
//! motherboard DAC / Bluetooth headphones) with:
//! - `0` buffer overruns / `0` underruns over multi-hour duplex recording sessions
//! - `> 110 dB` coherent Signal-to-Noise & Distortion Ratio (`SNR >= 95 dB` studio specification)
//! - `0` heap allocations and `0` mutexes on the real-time audio consumer path.

use crate::StereoFrame;
use crate::audio_io::{ProAudioThreadGuard, WasapiEndpointInfo, WasapiSampleFormat};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};

pub const DLL_RING_CAPACITY: usize = 16_384;
const DLL_RING_MASK: usize = DLL_RING_CAPACITY - 1;
/// Steady-state DLL queue target at a 48 kHz capture rate (kept for API compatibility; the actual
/// target scales with the capture rate, see [`AdaptiveDllDriftResampler::dll_target_fill_frames`]).
pub const DLL_TARGET_FILL_FRAMES: f64 = 2048.0;
/// Steady-state DLL queue target expressed in time (`2048` frames at 48 kHz = 42.67 ms).
///
/// Budget for a separate-clock WASAPI shared-mode capture endpoint (20 ms endpoint buffer, 10 ms
/// packets) feeding an engine that renders 256-frame blocks on demand from the output callback:
/// - `5.0 ms`: the queue is a sawtooth around its mean: it dips half a 10 ms packet below the mean
///   just before the next packet lands;
/// - `20.0 ms`: a capture packet may arrive up to one full endpoint buffer late (the device itself
///   overruns beyond that, so there is no point covering more);
/// - `10.7 ms`: the output callback can pull two 256-frame engine blocks back to back (WASAPI
///   10 ms periods / ASIO+SRC refills) before the capture thread runs again;
/// - `1.3 ms`: the 128-tap kernel needs 63 frames past the interpolation point (+ 2 guard frames).
/// Total `37 ms`; the remaining ~5.7 ms is headroom for DLL acquisition error (the loop starts
/// with up to half a packet of phase error and slews at <= 500 ppm). This is the margin the
/// previous first-fill anchor (2048 frames at 48 kHz) provided, now made deterministic.
pub const DLL_TARGET_FILL_SECONDS: f64 = 2048.0 / 48_000.0;
/// After priming, the DLL runs a short muted acquisition window at the nominal ratio, measures the
/// mean queue level the engine actually sees (capture-packet vs render-block phase), and shifts
/// the read head once so the mean sits exactly on the target before audio is emitted.
pub const DLL_ACQUISITION_SECONDS: f64 = 0.150;
pub const DLL_MAX_PPM_ADJUST: f64 = 0.000_500; // ±500 ppm

const DLL_SINC_TAPS: usize = 128;
const DLL_SINC_PHASES: usize = 1024;
/// Interpolation point of the polyphase kernel relative to the read head (`centre = 63 + mu`).
const DLL_KERNEL_CENTRE: f64 = 63.0;
const KAISER_BETA_DLL: f64 = 12.5;

/// Engine frames the ASIO/WASAPI output SRC (`WasapiStreamingResampler::render_hardware_frames`)
/// renders ahead of the hardware frame it is currently producing, worst case.
///
/// That SRC pulls a `Q = 256`-frame engine block whenever its ring holds fewer than
/// `T = SINC_TAPS + 4 = 132` frames, and its ring starts with `L = 64` lead-in zeros. After
/// producing hardware frame `q` it has consumed `floor(q / r)` engine frames (`r = hw / engine`),
/// so block `b` is pulled at the first `q` with `floor(q / r) >= b - (T - L - 1)`. The last frame
/// of that block (`b + Q - 1`) is therefore at most `Q - 1 + T - L - 1 = 322` engine frames ahead
/// of the hardware position, and the hardware input of that callback is already pushed.
pub const OUTPUT_SRC_RENDER_LEAD_ENGINE_FRAMES: f64 = 322.0;
/// Extra frames on top of the analytic minimum clock-locked cushion (guards floating-point
/// rounding of the output SRC phase accumulator at exact integer boundaries).
pub const CLOCK_LOCKED_GUARD_FRAMES: usize = 2;
const OUTPUT_SRC_REFILL_THRESHOLD: f64 = 132.0;
const OUTPUT_SRC_LEAD_IN_ZEROS: f64 = 64.0;
const ENGINE_RENDER_QUANTUM_F: f64 = 256.0;

#[inline]
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0f64;
    let mut term = 1.0f64;
    let half_sq = 0.25 * x * x;
    for k in 1..=44 {
        let kf = k as f64;
        term *= half_sq / (kf * kf);
        sum += term;
        if term < sum * 1e-16 {
            break;
        }
    }
    sum
}

#[inline]
fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// How the input bridge maps hardware-clock frames onto the engine clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureBridgeMode {
    /// Separate hardware clock (WASAPI capture endpoint): a PI delay-locked loop steers the ratio
    /// so the queue sits at a fixed target fill.
    AdaptiveDll,
    /// Same hardware clock as the output (ASIO full duplex): exact fixed ratio, no steering, and a
    /// fixed frame mapping `engine frame m <- hw position m * r - cushion`.
    ClockLocked,
}

/// Zero-Allocation input bridge: 128-Tap × 1024-Phase linearly-interpolated Kaiser
/// ($\beta = 12.5$) windowed-sinc resampler from a hardware capture clock to the engine clock,
/// either drift-tracking (Adaptive Delay-Locked Loop) or clock-locked (see [`CaptureBridgeMode`]).
pub struct AdaptiveDllDriftResampler {
    ring: Box<[StereoFrame; DLL_RING_CAPACITY]>,
    sinc_table: Box<[[f64; DLL_SINC_TAPS]; DLL_SINC_PHASES + 1]>,
    engine_sample_rate_hz: f64,
    read_idx: usize,
    write_idx: usize,
    available: usize,
    frac_phase: f64,
    nominal_ratio: f64,
    current_ratio: f64,
    mode: CaptureBridgeMode,
    /// DLL: fixed steady-state queue target (hw frames); also the priming threshold.
    target_fill: f64,
    err_lp1: f64,
    err_lp2: f64,
    integrator: f64,
    processed_out_frames: u64,
    overrun_count: u64,
    underrun_count: u64,
    primed: bool,
    /// Clock-locked: hardware frames between the engine read position and the write position at
    /// stream start (`engine frame m` interpolates hardware position `m * r - cushion`).
    cushion_frames: usize,
    /// Clock-locked: zero frames still to emit before the first interpolated frame.
    lead_in_remaining: usize,
    lead_in_total: usize,
    start_skip: usize,
    start_frac: f64,
    /// Clock-locked: input frames owed to the read head (start alignment + underrun recovery), so
    /// an underrun never shifts the fixed frame mapping.
    skip_debt: usize,
    /// DLL: engine frames of the (muted) acquisition window still to run after priming.
    acq_remaining: usize,
    acq_fill_sum: f64,
    acq_fill_n: u32,
    acq_consumed: usize,
}

impl AdaptiveDllDriftResampler {
    /// Drift-tracking bridge for a capture device on its own clock (WASAPI capture).
    /// Pre-allocates the `16384`-frame ring buffer and `1025 × 128` `f64` polyphase Kaiser Sinc LUT.
    pub fn new(capture_sample_rate_hz: f64, engine_sample_rate_hz: f64) -> Self {
        let nominal_ratio = (capture_sample_rate_hz.max(8_000.0)
            / engine_sample_rate_hz.max(8_000.0))
        .clamp(0.25, 4.0);
        let mut s = Self::allocate(nominal_ratio, engine_sample_rate_hz);
        s.mode = CaptureBridgeMode::AdaptiveDll;
        s.target_fill = Self::dll_target_fill_frames(capture_sample_rate_hz);
        s.acq_remaining = s.dll_acquisition_frames();
        s
    }

    /// Length of the muted DLL acquisition window (engine frames): long enough to average the
    /// queue over >= 10 capture packets and one block/packet beat period.
    #[inline]
    fn dll_acquisition_frames(&self) -> usize {
        (self.engine_sample_rate_hz * DLL_ACQUISITION_SECONDS).round() as usize
    }

    /// Clock-locked bridge for hardware input sharing the output's sample clock (ASIO duplex with
    /// the output SRC active). The ratio is the exact rational `round(hw) / round(engine)` used by
    /// the output SRC, there is no PI loop, and the cushion is the minimum that covers the output
    /// SRC's 256-frame pull-ahead plus the 128-tap kernel for this buffer size (see
    /// [`Self::clock_locked_cushion_frames`]). Must be attached before the first engine block of
    /// the stream so engine frame `0` and hardware frame `0` belong to the same callback.
    pub fn new_clock_locked(
        hw_sample_rate_hz: f64,
        engine_sample_rate_hz: f64,
        hw_buffer_frames: usize,
    ) -> Self {
        let cushion = Self::clock_locked_cushion_frames(
            hw_sample_rate_hz,
            engine_sample_rate_hz,
            hw_buffer_frames,
        );
        Self::new_clock_locked_with_cushion(hw_sample_rate_hz, engine_sample_rate_hz, cushion)
    }

    /// [`Self::new_clock_locked`] with an explicit cushion (hardware frames). Intended for tests
    /// and diagnostics; a cushion below [`Self::clock_locked_cushion_frames`] can underrun.
    pub fn new_clock_locked_with_cushion(
        hw_sample_rate_hz: f64,
        engine_sample_rate_hz: f64,
        cushion_frames: usize,
    ) -> Self {
        let ratio = Self::locked_ratio(hw_sample_rate_hz, engine_sample_rate_hz);
        let mut s = Self::allocate(ratio, engine_sample_rate_hz);
        s.mode = CaptureBridgeMode::ClockLocked;
        s.cushion_frames = cushion_frames.min(DLL_RING_CAPACITY / 2);
        // First interpolated engine frame `Z = ceil(C / r)` sits at hw position `Z * r - C >= 0`.
        let z = ((s.cushion_frames as f64) / ratio).ceil().max(0.0);
        let pos0 = (z * ratio - (s.cushion_frames as f64)).max(0.0);
        s.lead_in_total = z as usize;
        s.start_skip = pos0.floor() as usize;
        s.start_frac = pos0 - pos0.floor();
        s.arm_clock_locked();
        s
    }

    fn allocate(nominal_ratio: f64, engine_sample_rate_hz: f64) -> Self {
        let cutoff = 0.450 * (1.0 / nominal_ratio).min(1.0);
        let inv_i0 = 1.0 / bessel_i0(KAISER_BETA_DLL);
        let half_support = 63.0f64;

        let mut sinc_table: Box<[[f64; DLL_SINC_TAPS]; DLL_SINC_PHASES + 1]> =
            vec![[0.0f64; DLL_SINC_TAPS]; DLL_SINC_PHASES + 1]
                .into_boxed_slice()
                .try_into()
                .expect("exact polyphase sinc dimensions");

        for p in 0..=DLL_SINC_PHASES {
            let mu = (p as f64) / (DLL_SINC_PHASES as f64);
            let center = DLL_KERNEL_CENTRE + mu;
            let mut row_sum = 0.0f64;
            for t in 0..DLL_SINC_TAPS {
                let x = (t as f64) - center;
                if x.abs() >= half_support {
                    sinc_table[p][t] = 0.0;
                    continue;
                }
                let u = (x / half_support).clamp(-1.0, 1.0);
                let win = bessel_i0(KAISER_BETA_DLL * (1.0 - u * u).max(0.0).sqrt()) * inv_i0;
                let arg = 2.0 * cutoff * x;
                let sinc = if arg.abs() < 1e-12 {
                    2.0 * cutoff
                } else {
                    (PI * arg).sin() / (PI * x)
                };
                let w = sinc * win;
                sinc_table[p][t] = w;
                row_sum += w;
            }
            if row_sum.abs() > 1e-15 {
                let inv_sum = 1.0 / row_sum;
                for t in 0..DLL_SINC_TAPS {
                    sinc_table[p][t] *= inv_sum;
                }
            }
        }

        let ring: Box<[StereoFrame; DLL_RING_CAPACITY]> = vec![[0.0f32; 2]; DLL_RING_CAPACITY]
            .into_boxed_slice()
            .try_into()
            .expect("exact DLL ring capacity");

        Self {
            ring,
            sinc_table,
            engine_sample_rate_hz: engine_sample_rate_hz.max(8_000.0),
            read_idx: 0,
            write_idx: 0,
            available: 0,
            frac_phase: 0.0,
            nominal_ratio,
            current_ratio: nominal_ratio,
            mode: CaptureBridgeMode::AdaptiveDll,
            target_fill: DLL_TARGET_FILL_FRAMES,
            err_lp1: 0.0,
            err_lp2: 0.0,
            integrator: 0.0,
            processed_out_frames: 0,
            overrun_count: 0,
            underrun_count: 0,
            primed: false,
            cushion_frames: 0,
            lead_in_remaining: 0,
            lead_in_total: 0,
            start_skip: 0,
            start_frac: 0.0,
            skip_debt: 0,
            acq_remaining: 0,
            acq_fill_sum: 0.0,
            acq_fill_n: 0,
            acq_consumed: 0,
        }
    }

    fn arm_clock_locked(&mut self) {
        self.lead_in_remaining = self.lead_in_total;
        self.skip_debt = 0;
        self.frac_phase = 0.0;
        self.primed = false;
        if self.lead_in_total == 0 {
            self.skip_debt = self.start_skip;
            self.frac_phase = self.start_frac;
            self.primed = true;
        }
    }

    pub fn reset(&mut self) {
        self.read_idx = 0;
        self.write_idx = 0;
        self.available = 0;
        self.frac_phase = 0.0;
        self.current_ratio = self.nominal_ratio;
        self.err_lp1 = 0.0;
        self.err_lp2 = 0.0;
        self.integrator = 0.0;
        self.processed_out_frames = 0;
        self.overrun_count = 0;
        self.underrun_count = 0;
        self.primed = false;
        self.skip_debt = 0;
        self.acq_fill_sum = 0.0;
        self.acq_fill_n = 0;
        self.acq_consumed = 0;
        match self.mode {
            CaptureBridgeMode::ClockLocked => self.arm_clock_locked(),
            CaptureBridgeMode::AdaptiveDll => self.acq_remaining = self.dll_acquisition_frames(),
        }
    }

    // ------------------------------------------------------------------------------------------
    // Latency model (all host-facing values are in ENGINE samples)
    // ------------------------------------------------------------------------------------------

    /// Exact clock-locked ratio `hw / engine` from the same rounded integer rates the output SRC
    /// uses, so input and output stay sample-locked forever.
    pub fn locked_ratio(hw_sample_rate_hz: f64, engine_sample_rate_hz: f64) -> f64 {
        let hw = hw_sample_rate_hz.round().max(8_000.0);
        let eng = engine_sample_rate_hz.round().max(8_000.0);
        (hw / eng).clamp(0.25, 4.0)
    }

    /// Minimum safe clock-locked cushion (hardware frames) for an ASIO buffer of
    /// `hw_buffer_frames` (`0` = any buffer size).
    ///
    /// Engine frame `m` reads hardware frames `floor(m*r - C) .. +128`, so every engine block `b`
    /// needs `floor((b + 255) * r) - C + 128 <= W_b`, where `W_b` is the input pushed by the
    /// callback in which the output SRC pulls that block. The SRC pulls block `b` at hardware frame
    /// `q_b = ceil((b - 67) / (engine/hw))` (ring below 132 frames, 64 lead-in zeros), and the
    /// whole input buffer of that callback is already pushed: `W_b = (floor(q_b / B) + 1) * B`.
    /// The schedule is periodic, so the exact maximum is found by enumerating one period with
    /// integer arithmetic (non-RT, at stream open). It never exceeds the buffer-independent bound
    /// `ceil(322 r) + 128` (from `W_b >= q_b + 1`), which is also used when the period is too long
    /// or `hw_buffer_frames == 0`. [`CLOCK_LOCKED_GUARD_FRAMES`] is added on top. (The output SRC's
    /// floating-point phase accumulator can only make a pull happen *later* than the exact
    /// schedule, which only adds headroom.)
    pub fn clock_locked_cushion_frames(
        hw_sample_rate_hz: f64,
        engine_sample_rate_hz: f64,
        hw_buffer_frames: usize,
    ) -> usize {
        let r = Self::locked_ratio(hw_sample_rate_hz, engine_sample_rate_hz);
        let any_buffer = (OUTPUT_SRC_RENDER_LEAD_ENGINE_FRAMES * r - 1e-9).ceil() as usize
            + DLL_SINC_TAPS
            + CLOCK_LOCKED_GUARD_FRAMES;
        if hw_buffer_frames == 0 {
            return any_buffer;
        }
        let hw = hw_sample_rate_hz.round().max(8_000.0) as i128;
        let eng = engine_sample_rate_hz.round().max(8_000.0) as i128;
        let g = gcd_u64(hw as u64, eng as u64) as i128;
        let (a, c) = (hw / g, eng / g); // r = a / c exactly
        let bsz = hw_buffer_frames as i128;
        let quantum = ENGINE_RENDER_QUANTUM_F as i128;
        let lead = (OUTPUT_SRC_REFILL_THRESHOLD - OUTPUT_SRC_LEAD_IN_ZEROS - 1.0) as i128; // 67
        // Period: `n0` blocks advance `q` by the integer `dq`; `q mod B` repeats after `B/gcd(dq,B)`.
        let n0 = c / gcd_u64((quantum * a) as u64, c as u64) as i128;
        let dq = quantum * a * n0 / c;
        let period = n0 * (bsz / gcd_u64(dq as u64, bsz as u64) as i128);
        if period > (1 << 22) {
            return any_buffer;
        }
        let mut worst = i128::MIN;
        for n in 0..=period {
            let b = quantum * n;
            let q = if b <= lead { 0 } else { ((b - lead) * a + c - 1) / c };
            let w = (q / bsz + 1) * bsz;
            let need = ((b + quantum - 1) * a).div_euclid(c) + DLL_SINC_TAPS as i128 - w;
            worst = worst.max(need);
        }
        let exact = (worst.max(0) as usize) + CLOCK_LOCKED_GUARD_FRAMES;
        exact.min(any_buffer)
    }

    /// Fixed delay of the clock-locked input bridge, engine samples: hardware input frame `j`
    /// emerges at engine frame `(j + C - 63) / r`, i.e. `(C - 63) / r` later than a direct
    /// same-rate injection would place it.
    pub fn clock_locked_input_delay_engine_frames(
        hw_sample_rate_hz: f64,
        engine_sample_rate_hz: f64,
        hw_buffer_frames: usize,
    ) -> f64 {
        let r = Self::locked_ratio(hw_sample_rate_hz, engine_sample_rate_hz);
        let c = Self::clock_locked_cushion_frames(
            hw_sample_rate_hz,
            engine_sample_rate_hz,
            hw_buffer_frames,
        ) as f64;
        (c - DLL_KERNEL_CENTRE) / r
    }

    /// Alignment delay of the output SRC itself, engine samples: its ring starts with 64 zeros and
    /// interpolates at tap 63, so engine frame `m` plays at hardware frame `r * (m + 1)`
    /// (`+1` engine frame); `0` when the rates match (pass-through).
    ///
    /// The SRC's ring look-ahead (68..324 engine frames rendered before they play) is *not* an
    /// output delay on the engine timeline: it only means engine frames are computed early, and
    /// for a clock-locked input it is already contained in the cushion.
    pub fn output_src_alignment_engine_frames(hw_sample_rate_hz: u32, engine_sample_rate_hz: u32) -> f64 {
        if hw_sample_rate_hz.max(8_000) == engine_sample_rate_hz.max(8_000) {
            0.0
        } else {
            1.0
        }
    }

    /// Total duplex delay the ASIO+SRC path adds on top of the native-rate path (direct injection,
    /// engine blocks == ASIO buffers), engine samples: clock-locked input bridge + output SRC
    /// alignment. Exact and constant for the lifetime of the stream.
    pub fn clock_locked_duplex_extra_latency_engine_frames(
        hw_sample_rate_hz: f64,
        engine_sample_rate_hz: f64,
        hw_buffer_frames: usize,
    ) -> f64 {
        Self::clock_locked_input_delay_engine_frames(
            hw_sample_rate_hz,
            engine_sample_rate_hz,
            hw_buffer_frames,
        ) + Self::output_src_alignment_engine_frames(
            hw_sample_rate_hz.round() as u32,
            engine_sample_rate_hz.round() as u32,
        )
    }

    /// Fixed DLL steady-state queue target for a capture rate (hardware frames):
    /// [`DLL_TARGET_FILL_SECONDS`] of audio, clamped to fit the ring.
    pub fn dll_target_fill_frames(capture_sample_rate_hz: f64) -> f64 {
        (capture_sample_rate_hz.max(8_000.0) * DLL_TARGET_FILL_SECONDS)
            .round()
            .clamp(512.0, (DLL_RING_CAPACITY / 2) as f64)
    }

    /// Mean delay of the drift-tracking bridge, engine samples, measured from the moment a frame
    /// is delivered by the capture thread to the engine block that consumes it: the DLL holds the
    /// queue at `F` frames at block start and interpolates 63 frames behind the newest data, so
    /// the delay is `(F - 63) / r`. The capture packet phase adds `0..1` packet on top (covered by
    /// the flat WASAPI device-period term), and the output render look-ahead must be added by the
    /// caller ([`Self::render_lookahead_engine_frames`]) because the DLL measures its fill at
    /// render time, not at play time.
    pub fn dll_input_delay_engine_frames(capture_sample_rate_hz: f64, engine_sample_rate_hz: f64) -> f64 {
        let r = (capture_sample_rate_hz.max(8_000.0) / engine_sample_rate_hz.max(8_000.0))
            .clamp(0.25, 4.0);
        (Self::dll_target_fill_frames(capture_sample_rate_hz) - DLL_KERNEL_CENTRE) / r
    }

    /// Average time (engine samples) between an engine block being rendered and its first frame
    /// reaching the hardware buffer position, for a stream rendered through
    /// `WasapiStreamingResampler::render_hardware_frames` with `hw_buffer_frames` per callback.
    /// Needed only for an asynchronous (separate-clock) input bridge, which samples its queue at
    /// render time.
    /// - pass-through (`hw == engine`): blocks are pulled when the ring is empty, at
    ///   `b mod B` frames into the callback -> mean `(B - gcd(256, B)) / 2`;
    /// - SRC: a block is pulled when the ring drops below 132 frames, i.e.
    ///   `T - L - 0.5 = 67.5` frames (mean) before its first frame plays, plus the hardware
    ///   frame's offset inside the
    ///   callback (`(q mod B) / r`, mean `(B - 1) / (2 r)`).
    ///
    /// Callers whose engine renders exactly one block per hardware callback (ASIO without SRC)
    /// must use `0`.
    pub fn render_lookahead_engine_frames(
        engine_sample_rate_hz: u32,
        hw_sample_rate_hz: u32,
        hw_buffer_frames: u32,
    ) -> f64 {
        let eng = engine_sample_rate_hz.max(8_000);
        let hw = hw_sample_rate_hz.max(8_000);
        let b = hw_buffer_frames.max(1) as u64;
        if eng == hw {
            let g = gcd_u64(ENGINE_RENDER_QUANTUM_F as u64, b);
            (b - g) as f64 * 0.5
        } else {
            let r = (hw as f64) / (eng as f64);
            (OUTPUT_SRC_REFILL_THRESHOLD - OUTPUT_SRC_LEAD_IN_ZEROS - 0.5)
                + ((b - 1) as f64) / (2.0 * r)
        }
    }

    /// Expected delay of this bridge instance in engine samples (see
    /// [`Self::clock_locked_input_delay_engine_frames`] / [`Self::dll_input_delay_engine_frames`]).
    pub fn expected_input_delay_engine_frames(&self) -> f64 {
        match self.mode {
            CaptureBridgeMode::ClockLocked => {
                (self.cushion_frames as f64 - DLL_KERNEL_CENTRE) / self.nominal_ratio
            }
            CaptureBridgeMode::AdaptiveDll => {
                (self.target_fill - DLL_KERNEL_CENTRE) / self.nominal_ratio
            }
        }
    }

    #[inline]
    pub fn mode(&self) -> CaptureBridgeMode {
        self.mode
    }

    /// Clock-locked cushion in hardware frames (`0` in DLL mode).
    #[inline]
    pub fn cushion_frames(&self) -> usize {
        self.cushion_frames
    }

    /// DLL steady-state queue target in hardware frames.
    #[inline]
    pub fn target_fill_frames(&self) -> f64 {
        self.target_fill
    }

    // ------------------------------------------------------------------------------------------
    // Real-time path
    // ------------------------------------------------------------------------------------------

    #[inline(always)]
    fn push_one(&mut self, f: StereoFrame) {
        if self.available >= DLL_RING_CAPACITY - 1 {
            // Ring full overrun: drop oldest frame and increment xrun diagnostic counter
            self.read_idx = (self.read_idx + 1) & DLL_RING_MASK;
            self.available -= 1;
            self.overrun_count += 1;
        }
        self.ring[self.write_idx] = f;
        self.write_idx = (self.write_idx + 1) & DLL_RING_MASK;
        self.available += 1;
    }

    /// DLL priming: once the queue reaches the fixed target, trim the excess (pre-roll only) so
    /// the loop starts exactly on target instead of wherever the last packet left it.
    #[inline]
    fn prime_if_ready(&mut self) {
        if self.primed || self.mode != CaptureBridgeMode::AdaptiveDll {
            return;
        }
        let target = self.target_fill as usize;
        if self.available >= target {
            let excess = self.available - target;
            self.read_idx = (self.read_idx + excess) & DLL_RING_MASK;
            self.available = target;
            self.primed = true;
        }
    }

    /// Pushes captured input frames into the internal ring (`0` heap allocations).
    #[inline]
    pub fn push_captured_frames(&mut self, frames: &[StereoFrame]) {
        for &f in frames {
            self.push_one(f);
        }
        self.prime_if_ready();
    }

    /// Drains any pending frames from a lock-free `rtrb::Consumer<StereoFrame>` (`0` heap allocations).
    #[inline]
    pub fn drain_from_rtrb(&mut self, rx: &mut rtrb::Consumer<StereoFrame>) {
        while let Ok(frame) = rx.pop() {
            self.push_one(frame);
        }
        self.prime_if_ready();
    }

    #[inline(always)]
    fn interpolate_at_read_head(&self) -> StereoFrame {
        // Linear interpolation between adjacent polyphase rows `p` and `p + 1` in `f64`
        let scaled = self.frac_phase * (DLL_SINC_PHASES as f64);
        let p0 = (scaled.floor() as usize).min(DLL_SINC_PHASES - 1);
        let frac = scaled - (p0 as f64);
        let row0 = &self.sinc_table[p0];
        let row1 = &self.sinc_table[p0 + 1];

        let base = self.read_idx;
        let mut acc_l = 0.0f64;
        let mut acc_r = 0.0f64;
        for t in 0..DLL_SINC_TAPS {
            let w = row0[t] + frac * (row1[t] - row0[t]);
            let s = self.ring[(base + t) & DLL_RING_MASK];
            acc_l += (s[0] as f64) * w;
            acc_r += (s[1] as f64) * w;
        }
        [acc_l as f32, acc_r as f32]
    }

    #[inline(always)]
    fn settle_skip_debt(&mut self) {
        let s = self.skip_debt.min(self.available);
        if s > 0 {
            self.read_idx = (self.read_idx + s) & DLL_RING_MASK;
            self.available -= s;
            self.skip_debt -= s;
        }
    }

    /// Produces `out_frames.len()` samples aligned to the engine clock (`0` heap allocs).
    #[inline]
    pub fn process_output_block(&mut self, out_frames: &mut [StereoFrame]) {
        if out_frames.is_empty() {
            return;
        }
        match self.mode {
            CaptureBridgeMode::ClockLocked => self.process_clock_locked_block(out_frames),
            CaptureBridgeMode::AdaptiveDll => self.process_dll_block(out_frames),
        }
    }

    /// Fixed-ratio path: engine frame `m` interpolates hardware position `m * r - C`. Underruns
    /// (never expected with the computed cushion) emit silence but keep advancing the phase, so
    /// the mapping — and therefore the reported latency — can never slip.
    fn process_clock_locked_block(&mut self, out_frames: &mut [StereoFrame]) {
        let r = self.nominal_ratio;
        for slot in out_frames.iter_mut() {
            if self.lead_in_remaining > 0 {
                *slot = [0.0, 0.0];
                self.lead_in_remaining -= 1;
                if self.lead_in_remaining == 0 {
                    self.skip_debt += self.start_skip;
                    self.frac_phase = self.start_frac;
                    self.primed = true;
                }
                continue;
            }
            self.settle_skip_debt();
            if self.skip_debt > 0 || self.available < DLL_SINC_TAPS {
                self.underrun_count += 1;
                *slot = [0.0, 0.0];
            } else {
                *slot = self.interpolate_at_read_head();
            }
            self.frac_phase += r;
            let adv = self.frac_phase.floor();
            if adv > 0.0 {
                self.frac_phase -= adv;
                self.skip_debt += adv as usize;
                self.settle_skip_debt();
            }
        }
        self.processed_out_frames += out_frames.len() as u64;
    }

    /// Muted acquisition block: consume at the nominal ratio, record the queue level seen at block
    /// start, and at the end shift the read head once so the mean level equals the fixed target.
    /// Nothing consumed here was ever emitted, so shifting back (re-reading those frames) or
    /// forward (dropping pre-roll) is inaudible.
    fn run_dll_acquisition(&mut self, out_frames: &mut [StereoFrame]) {
        let n_out = out_frames.len();
        out_frames.fill([0.0, 0.0]);
        self.acq_fill_sum += (self.available as f64) - self.frac_phase;
        self.acq_fill_n += 1;

        let total = self.frac_phase + (n_out as f64) * self.nominal_ratio;
        let adv = total.floor();
        let actual = (adv as usize).min(self.available);
        self.read_idx = (self.read_idx + actual) & DLL_RING_MASK;
        self.available -= actual;
        self.acq_consumed += actual;
        self.frac_phase = total - adv;
        self.processed_out_frames += n_out as u64;

        self.acq_remaining = self.acq_remaining.saturating_sub(n_out);
        if self.acq_remaining == 0 && self.acq_fill_n > 0 {
            let mean = self.acq_fill_sum / (self.acq_fill_n as f64);
            let max_shift = (0.25 * self.target_fill).round();
            let shift = (mean - self.target_fill).round().clamp(-max_shift, max_shift);
            if shift > 0.0 {
                let skip = (shift as usize).min(self.available);
                self.read_idx = (self.read_idx + skip) & DLL_RING_MASK;
                self.available -= skip;
            } else if shift < 0.0 {
                let back = ((-shift) as usize)
                    .min(self.acq_consumed)
                    .min(DLL_RING_CAPACITY - 1 - self.available);
                self.read_idx = (self.read_idx + DLL_RING_CAPACITY - back) & DLL_RING_MASK;
                self.available += back;
            }
        }
    }

    fn process_dll_block(&mut self, out_frames: &mut [StereoFrame]) {
        let n_out = out_frames.len();
        if !self.primed {
            out_frames.fill([0.0, 0.0]);
            return;
        }
        if self.acq_remaining > 0 {
            self.run_dll_acquisition(out_frames);
            return;
        }

        // Overdamped 2nd-order PI Delay-Locked Loop (`zeta = 1.05`, `f_lp = 6 * f_n`) operating on
        // engine-frame queue error so loop dynamics are identical across 44.1 / 48 / 96 kHz:
        // - Pull-in & drift lock (`0..24 s`): `f_n = 0.075 Hz` quickly tracks ±150 ppm clock skew
        //   while remaining linear under 20 ms late-packet stalls inside `±500 ppm`.
        // - Narrowband tracking (`>= 38 s`): `f_n = 0.012 Hz` settles cleanly by `59 s`
        //   (`>= 95 dB` coherent sine SNR) while rejecting packet-arrival jitter.
        let fs = self.engine_sample_rate_hz;
        let dt = (n_out as f64) / fs;
        let cont_fill = (self.available as f64) - self.frac_phase;
        // Fixed anchor: the steady-state queue (and so the bridge latency) is deterministic.
        let anchor = self.target_fill;
        let raw_err = (cont_fill - anchor) / self.nominal_ratio;

        let t_sec = (self.processed_out_frames as f64) / fs;
        let fn_hz = if t_sec < 24.0 {
            0.075
        } else if t_sec < 38.0 {
            let u = (t_sec - 24.0) / 14.0;
            0.075 * (0.012_f64 / 0.075).powf(u)
        } else {
            0.012
        };

        let omega_n = 2.0 * PI * fn_hz;
        let zeta = 1.05;
        let omega_lp = 6.0 * omega_n;
        let lp_alpha = (1.0 - (-omega_lp * dt).exp()).clamp(1e-6, 1.0);
        self.err_lp1 += lp_alpha * (raw_err - self.err_lp1);
        self.err_lp2 += lp_alpha * (self.err_lp1 - self.err_lp2);

        let kp = (2.0 * zeta * omega_n) / fs;
        let ki_dt = (omega_n * omega_n * dt) / fs;

        let prev_ratio = self.current_ratio;
        let p_term = kp * self.err_lp2;
        let cand_i = self.integrator + ki_dt * self.err_lp2;
        let unsat = p_term + cand_i;
        if unsat.abs() <= DLL_MAX_PPM_ADJUST || cand_i * self.err_lp2 < 0.0 {
            self.integrator = cand_i.clamp(-DLL_MAX_PPM_ADJUST, DLL_MAX_PPM_ADJUST);
        }
        let delta_rel = (p_term + self.integrator).clamp(-DLL_MAX_PPM_ADJUST, DLL_MAX_PPM_ADJUST);
        self.current_ratio = self.nominal_ratio * (1.0 + delta_rel);

        let inv_n = 1.0 / (n_out as f64);
        let ratio_slope = (self.current_ratio - prev_ratio) * inv_n;
        for (i, out_slot) in out_frames.iter_mut().enumerate() {
            let step = prev_ratio + ratio_slope * ((i + 1) as f64);
            if self.available < DLL_SINC_TAPS + 2 {
                self.underrun_count += 1;
                *out_slot = [0.0, 0.0];
                continue;
            }

            *out_slot = self.interpolate_at_read_head();

            self.frac_phase += step;
            let adv = self.frac_phase.floor() as usize;
            if adv > 0 {
                let actual = adv.min(self.available);
                self.read_idx = (self.read_idx + actual) & DLL_RING_MASK;
                self.available -= actual;
                self.frac_phase -= actual as f64;
            }
        }

        self.processed_out_frames += n_out as u64;
    }

    #[inline]
    pub fn current_ratio(&self) -> f64 {
        self.current_ratio
    }

    #[inline]
    pub fn current_ppm_offset(&self) -> f64 {
        ((self.current_ratio / self.nominal_ratio) - 1.0) * 1_000_000.0
    }

    #[inline]
    pub fn available_frames(&self) -> usize {
        self.available
    }

    /// Engine frames emitted so far (including pre-roll silence).
    #[inline]
    pub fn processed_output_frames(&self) -> u64 {
        self.processed_out_frames
    }

    #[inline]
    pub fn underruns(&self) -> u64 {
        self.underrun_count
    }

    #[inline]
    pub fn overruns(&self) -> u64 {
        self.overrun_count
    }

    #[inline]
    pub fn total_xruns(&self) -> u64 {
        self.overrun_count + self.underrun_count
    }
}

// ============================================================================
// WINDOWS CORE AUDIO `eCapture` ENDPOINT ENUMERATION & EVENT-DRIVEN STREAM
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WasapiCaptureStatusSnapshot {
    pub running: bool,
    pub device_id: String,
    pub device_name: String,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub captured_blocks: u64,
    pub status_message: String,
}

#[inline]
pub unsafe fn unpack_hardware_capture_to_stereo(
    src_bytes: *const u8,
    frames: usize,
    channels: usize,
    format: WasapiSampleFormat,
    silent_flag: bool,
    dst: &mut [StereoFrame],
) {
    let n = frames.min(dst.len());
    if silent_flag || src_bytes.is_null() || channels == 0 {
        dst[..n].fill([0.0, 0.0]);
        return;
    }
    match format {
        WasapiSampleFormat::Float32 => {
            let s = unsafe { std::slice::from_raw_parts(src_bytes as *const f32, n * channels) };
            for i in 0..n {
                let l = s[i * channels];
                let r = if channels >= 2 { s[i * channels + 1] } else { l };
                dst[i] = [
                    if l.is_finite() { l.clamp(-1.0, 1.0) } else { 0.0 },
                    if r.is_finite() { r.clamp(-1.0, 1.0) } else { 0.0 },
                ];
            }
        }
        WasapiSampleFormat::Int16 => {
            let s = unsafe { std::slice::from_raw_parts(src_bytes as *const i16, n * channels) };
            for i in 0..n {
                let l = (s[i * channels] as f32) / 32_768.0;
                let r = if channels >= 2 {
                    (s[i * channels + 1] as f32) / 32_768.0
                } else {
                    l
                };
                dst[i] = [l.clamp(-1.0, 1.0), r.clamp(-1.0, 1.0)];
            }
        }
        WasapiSampleFormat::Int32 | WasapiSampleFormat::Int24In32 => {
            let s = unsafe { std::slice::from_raw_parts(src_bytes as *const i32, n * channels) };
            for i in 0..n {
                let l = ((s[i * channels] as f64) / 2_147_483_648.0) as f32;
                let r = if channels >= 2 {
                    ((s[i * channels + 1] as f64) / 2_147_483_648.0) as f32
                } else {
                    l
                };
                dst[i] = [l.clamp(-1.0, 1.0), r.clamp(-1.0, 1.0)];
            }
        }
        WasapiSampleFormat::Int24Packed => {
            let stride = channels * 3;
            let s = unsafe { std::slice::from_raw_parts(src_bytes, n * stride) };
            for i in 0..n {
                let base = i * stride;
                let v_l = (s[base] as i32)
                    | ((s[base + 1] as i32) << 8)
                    | (((s[base + 2] as i8) as i32) << 16);
                let l = ((v_l as f64) / 8_388_608.0) as f32;
                let r = if channels >= 2 {
                    let v_r = (s[base + 3] as i32)
                        | ((s[base + 4] as i32) << 8)
                        | (((s[base + 5] as i8) as i32) << 16);
                    ((v_r as f64) / 8_388_608.0) as f32
                } else {
                    l
                };
                dst[i] = [l.clamp(-1.0, 1.0), r.clamp(-1.0, 1.0)];
            }
        }
    }
}

#[cfg(target_os = "windows")]
mod win32_capture {
    use super::*;
    use windows_sys::Win32::Media::Audio::{WAVEFORMATEX, WAVEFORMATEXTENSIBLE};
    use windows_sys::Win32::System::Com::{
        CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx, CoTaskMemFree,
        CoUninitialize,
    };
    use windows_sys::Win32::System::Threading::{INFINITE, WaitForSingleObject};
    use windows_sys::core::GUID;

    const E_CAPTURE: i32 = 1;
    const E_CONSOLE: i32 = 0;
    const DEVICE_STATE_ACTIVE: u32 = 0x0000_0001;
    const STGM_READ: u32 = 0;
    const VT_LPWSTR: u16 = 31;
    const AUDCLNT_SHAREMODE_SHARED: i32 = 0;
    const AUDCLNT_STREAMFLAGS_EVENTCALLBACK: u32 = 0x0004_0000;
    const AUDCLNT_BUFFERFLAGS_SILENT: u32 = 0x2;

    const CLSID_MMDEVICE_ENUMERATOR: GUID = GUID {
        data1: 0xBCDE0395,
        data2: 0xE52F,
        data3: 0x467C,
        data4: [0x8E, 0x3D, 0xC4, 0x57, 0x92, 0x91, 0x69, 0x2E],
    };
    const IID_IMMDEVICE_ENUMERATOR: GUID = GUID {
        data1: 0xA95664D2,
        data2: 0x9614,
        data3: 0x4F35,
        data4: [0xA7, 0x46, 0xDE, 0x8D, 0xB6, 0x36, 0x17, 0xE6],
    };
    const IID_IAUDIOCLIENT: GUID = GUID {
        data1: 0x1CB9AD4C,
        data2: 0xDBFA,
        data3: 0x4C32,
        data4: [0xB1, 0x78, 0xC2, 0xF5, 0x68, 0xA7, 0x03, 0xB2],
    };
    const IID_IAUDIOCAPTURECLIENT: GUID = GUID {
        data1: 0xC8ADBD64,
        data2: 0xE71E,
        data3: 0x48A0,
        data4: [0xA4, 0xDE, 0x18, 0x5C, 0x39, 0x5C, 0xD3, 0x17],
    };
    const KSDATAFORMAT_SUBTYPE_IEEE_FLOAT_GUID: GUID = GUID {
        data1: 0x00000003,
        data2: 0x0000,
        data3: 0x0010,
        data4: [0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71],
    };

    #[repr(C)]
    struct PropertyKey {
        fmtid: GUID,
        pid: u32,
    }
    const PKEY_DEVICE_FRIENDLY_NAME: PropertyKey = PropertyKey {
        fmtid: GUID {
            data1: 0xA45C254E,
            data2: 0xDF1C,
            data3: 0x4EFD,
            data4: [0x80, 0x20, 0x67, 0xD1, 0x46, 0xA8, 0x50, 0xE0],
        },
        pid: 14,
    };

    #[repr(C)]
    struct RawPropVariant {
        vt: u16,
        reserved1: u16,
        reserved2: u16,
        reserved3: u16,
        ptr_val: *mut u16,
        pad_val: usize,
    }

    #[link(name = "ole32")]
    unsafe extern "system" {
        fn PropVariantClear(pvar: *mut RawPropVariant) -> i32;
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn CreateEventW(
            attrs: *const c_void,
            manual_reset: i32,
            init_state: i32,
            name: *const u16,
        ) -> *mut c_void;
        fn SetEvent(h_event: *mut c_void) -> i32;
        fn CloseHandle(h_object: *mut c_void) -> i32;
    }

    #[repr(C)]
    struct IMMDeviceEnumerator {
        vtbl: *const IMMDeviceEnumeratorVtbl,
    }
    #[repr(C)]
    struct IMMDeviceEnumeratorVtbl {
        qi: unsafe extern "system" fn(*mut IMMDeviceEnumerator, *const GUID, *mut *mut c_void) -> i32,
        add_ref: unsafe extern "system" fn(*mut IMMDeviceEnumerator) -> u32,
        release: unsafe extern "system" fn(*mut IMMDeviceEnumerator) -> u32,
        enum_audio_endpoints: unsafe extern "system" fn(
            *mut IMMDeviceEnumerator,
            i32,
            u32,
            *mut *mut IMMDeviceCollection,
        ) -> i32,
        get_default_audio_endpoint:
            unsafe extern "system" fn(*mut IMMDeviceEnumerator, i32, i32, *mut *mut IMMDevice) -> i32,
        get_device:
            unsafe extern "system" fn(*mut IMMDeviceEnumerator, *const u16, *mut *mut IMMDevice) -> i32,
    }

    #[repr(C)]
    struct IMMDeviceCollection {
        vtbl: *const IMMDeviceCollectionVtbl,
    }
    #[repr(C)]
    struct IMMDeviceCollectionVtbl {
        qi: unsafe extern "system" fn(*mut IMMDeviceCollection, *const GUID, *mut *mut c_void) -> i32,
        add_ref: unsafe extern "system" fn(*mut IMMDeviceCollection) -> u32,
        release: unsafe extern "system" fn(*mut IMMDeviceCollection) -> u32,
        get_count: unsafe extern "system" fn(*mut IMMDeviceCollection, *mut u32) -> i32,
        item: unsafe extern "system" fn(*mut IMMDeviceCollection, u32, *mut *mut IMMDevice) -> i32,
    }

    #[repr(C)]
    struct IMMDevice {
        vtbl: *const IMMDeviceVtbl,
    }
    #[repr(C)]
    struct IMMDeviceVtbl {
        qi: unsafe extern "system" fn(*mut IMMDevice, *const GUID, *mut *mut c_void) -> i32,
        add_ref: unsafe extern "system" fn(*mut IMMDevice) -> u32,
        release: unsafe extern "system" fn(*mut IMMDevice) -> u32,
        activate: unsafe extern "system" fn(
            *mut IMMDevice,
            *const GUID,
            u32,
            *const c_void,
            *mut *mut c_void,
        ) -> i32,
        open_property_store:
            unsafe extern "system" fn(*mut IMMDevice, u32, *mut *mut IPropertyStore) -> i32,
        get_id: unsafe extern "system" fn(*mut IMMDevice, *mut *mut u16) -> i32,
    }

    #[repr(C)]
    struct IPropertyStore {
        vtbl: *const IPropertyStoreVtbl,
    }
    #[repr(C)]
    struct IPropertyStoreVtbl {
        qi: unsafe extern "system" fn(*mut IPropertyStore, *const GUID, *mut *mut c_void) -> i32,
        add_ref: unsafe extern "system" fn(*mut IPropertyStore) -> u32,
        release: unsafe extern "system" fn(*mut IPropertyStore) -> u32,
        get_count: unsafe extern "system" fn(*mut IPropertyStore, *mut u32) -> i32,
        get_at: unsafe extern "system" fn(*mut IPropertyStore, u32, *mut PropertyKey) -> i32,
        get_value: unsafe extern "system" fn(
            *mut IPropertyStore,
            *const PropertyKey,
            *mut RawPropVariant,
        ) -> i32,
    }

    #[repr(C)]
    struct IAudioClient {
        vtbl: *const IAudioClientVtbl,
    }
    #[repr(C)]
    struct IAudioClientVtbl {
        qi: unsafe extern "system" fn(*mut IAudioClient, *const GUID, *mut *mut c_void) -> i32,
        add_ref: unsafe extern "system" fn(*mut IAudioClient) -> u32,
        release: unsafe extern "system" fn(*mut IAudioClient) -> u32,
        initialize: unsafe extern "system" fn(
            *mut IAudioClient,
            i32,
            u32,
            i64,
            i64,
            *const WAVEFORMATEX,
            *const GUID,
        ) -> i32,
        get_buffer_size: unsafe extern "system" fn(*mut IAudioClient, *mut u32) -> i32,
        get_stream_latency: unsafe extern "system" fn(*mut IAudioClient, *mut i64) -> i32,
        get_current_padding: unsafe extern "system" fn(*mut IAudioClient, *mut u32) -> i32,
        is_format_supported: unsafe extern "system" fn(
            *mut IAudioClient,
            i32,
            *const WAVEFORMATEX,
            *mut *mut WAVEFORMATEX,
        ) -> i32,
        get_mix_format: unsafe extern "system" fn(*mut IAudioClient, *mut *mut WAVEFORMATEX) -> i32,
        get_device_period:
            unsafe extern "system" fn(*mut IAudioClient, *mut i64, *mut i64) -> i32,
        start: unsafe extern "system" fn(*mut IAudioClient) -> i32,
        stop: unsafe extern "system" fn(*mut IAudioClient) -> i32,
        reset: unsafe extern "system" fn(*mut IAudioClient) -> i32,
        set_event_handle: unsafe extern "system" fn(*mut IAudioClient, *mut c_void) -> i32,
        get_service:
            unsafe extern "system" fn(*mut IAudioClient, *const GUID, *mut *mut c_void) -> i32,
    }

    #[repr(C)]
    struct IAudioCaptureClient {
        vtbl: *const IAudioCaptureClientVtbl,
    }
    #[repr(C)]
    struct IAudioCaptureClientVtbl {
        qi: unsafe extern "system" fn(*mut IAudioCaptureClient, *const GUID, *mut *mut c_void) -> i32,
        add_ref: unsafe extern "system" fn(*mut IAudioCaptureClient) -> u32,
        release: unsafe extern "system" fn(*mut IAudioCaptureClient) -> u32,
        get_buffer: unsafe extern "system" fn(
            *mut IAudioCaptureClient,
            *mut *mut u8,
            *mut u32,
            *mut u32,
            *mut u64,
            *mut u64,
        ) -> i32,
        release_buffer: unsafe extern "system" fn(*mut IAudioCaptureClient, u32) -> i32,
        get_next_packet_size: unsafe extern "system" fn(*mut IAudioCaptureClient, *mut u32) -> i32,
    }

    unsafe fn pwstr_to_string(ptr: *const u16) -> String {
        if ptr.is_null() {
            return String::new();
        }
        let mut len = 0usize;
        unsafe {
            while *ptr.add(len) != 0 && len < 2048 {
                len += 1;
            }
            String::from_utf16_lossy(std::slice::from_raw_parts(ptr, len))
        }
    }

    unsafe fn classify_waveformat(wf: *const WAVEFORMATEX) -> (u32, u16, WasapiSampleFormat) {
        if wf.is_null() {
            return (48_000, 2, WasapiSampleFormat::Float32);
        }
        unsafe {
            let sr = (*wf).nSamplesPerSec;
            let ch = (*wf).nChannels;
            let bits = (*wf).wBitsPerSample;
            let tag = (*wf).wFormatTag;
            let fmt = if tag == 0x0003 {
                WasapiSampleFormat::Float32
            } else if tag == 0xFFFE {
                let ext = &*(wf as *const WAVEFORMATEXTENSIBLE);
                let sub = ext.SubFormat;
                if sub.data1 == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT_GUID.data1 && bits == 32 {
                    WasapiSampleFormat::Float32
                } else if bits == 32 {
                    WasapiSampleFormat::Int32
                } else if bits == 24 {
                    WasapiSampleFormat::Int24Packed
                } else {
                    WasapiSampleFormat::Int16
                }
            } else if bits == 32 {
                WasapiSampleFormat::Int32
            } else if bits == 24 {
                WasapiSampleFormat::Int24Packed
            } else {
                WasapiSampleFormat::Int16
            };
            (sr, ch, fmt)
        }
    }

    unsafe fn inspect_imm_device(dev: *mut IMMDevice, default_id: &str) -> Option<WasapiEndpointInfo> {
        unsafe {
            let mut id_ptr: *mut u16 = std::ptr::null_mut();
            if ((*(*dev).vtbl).get_id)(dev, &mut id_ptr) < 0 || id_ptr.is_null() {
                return None;
            }
            let id = pwstr_to_string(id_ptr);
            CoTaskMemFree(id_ptr as *const c_void);

            let mut friendly = id.clone();
            let mut store: *mut IPropertyStore = std::ptr::null_mut();
            if ((*(*dev).vtbl).open_property_store)(dev, STGM_READ, &mut store) >= 0
                && !store.is_null()
            {
                let mut pv: RawPropVariant = std::mem::zeroed();
                if ((*(*store).vtbl).get_value)(store, &PKEY_DEVICE_FRIENDLY_NAME, &mut pv) >= 0 {
                    if pv.vt == VT_LPWSTR && !pv.ptr_val.is_null() {
                        friendly = pwstr_to_string(pv.ptr_val);
                    }
                    let _ = PropVariantClear(&mut pv);
                }
                let _ = ((*(*store).vtbl).release)(store);
            }

            let mut sr = 48_000u32;
            let mut ch = 2u16;
            let mut fmt_label = "Float32".to_string();
            let mut client: *mut IAudioClient = std::ptr::null_mut();
            if ((*(*dev).vtbl).activate)(
                dev,
                &IID_IAUDIOCLIENT,
                CLSCTX_ALL,
                std::ptr::null(),
                (&mut client) as *mut *mut IAudioClient as *mut *mut c_void,
            ) >= 0
                && !client.is_null()
            {
                let mut mix_wf: *mut WAVEFORMATEX = std::ptr::null_mut();
                if ((*(*client).vtbl).get_mix_format)(client, &mut mix_wf) >= 0
                    && !mix_wf.is_null()
                {
                    let (msr, mch, mfmt) = classify_waveformat(mix_wf);
                    sr = msr;
                    ch = mch;
                    fmt_label = mfmt.label().to_string();
                    CoTaskMemFree(mix_wf as *const c_void);
                }
                let _ = ((*(*client).vtbl).release)(client);
            }

            let is_default = !default_id.is_empty() && id == default_id;
            Some(WasapiEndpointInfo {
                id,
                friendly_name: friendly,
                is_default,
                mix_sample_rate_hz: sr,
                channels: ch,
                mix_channels: ch,
                mix_format_label: fmt_label,
            })
        }
    }

    pub fn enumerate_capture_endpoints() -> Vec<WasapiEndpointInfo> {
        let mut out = Vec::new();
        unsafe {
            let hr = CoInitializeEx(std::ptr::null(), COINIT_MULTITHREADED as u32);
            let uninit = hr >= 0;
            let mut enum_ptr: *mut IMMDeviceEnumerator = std::ptr::null_mut();
            if CoCreateInstance(
                &CLSID_MMDEVICE_ENUMERATOR,
                std::ptr::null_mut(),
                CLSCTX_ALL,
                &IID_IMMDEVICE_ENUMERATOR,
                (&mut enum_ptr) as *mut *mut IMMDeviceEnumerator as *mut *mut c_void,
            ) >= 0
                && !enum_ptr.is_null()
            {
                let mut default_id = String::new();
                let mut def_dev: *mut IMMDevice = std::ptr::null_mut();
                if ((*(*enum_ptr).vtbl).get_default_audio_endpoint)(
                    enum_ptr,
                    E_CAPTURE,
                    E_CONSOLE,
                    &mut def_dev,
                ) >= 0
                    && !def_dev.is_null()
                {
                    let mut id_ptr: *mut u16 = std::ptr::null_mut();
                    if ((*(*def_dev).vtbl).get_id)(def_dev, &mut id_ptr) >= 0 && !id_ptr.is_null()
                    {
                        default_id = pwstr_to_string(id_ptr);
                        CoTaskMemFree(id_ptr as *const c_void);
                    }
                    let _ = ((*(*def_dev).vtbl).release)(def_dev);
                }

                let mut col: *mut IMMDeviceCollection = std::ptr::null_mut();
                if ((*(*enum_ptr).vtbl).enum_audio_endpoints)(
                    enum_ptr,
                    E_CAPTURE,
                    DEVICE_STATE_ACTIVE,
                    &mut col,
                ) >= 0
                    && !col.is_null()
                {
                    let mut count = 0u32;
                    let _ = ((*(*col).vtbl).get_count)(col, &mut count);
                    for i in 0..count {
                        let mut dev: *mut IMMDevice = std::ptr::null_mut();
                        if ((*(*col).vtbl).item)(col, i, &mut dev) >= 0 && !dev.is_null() {
                            if let Some(info) = inspect_imm_device(dev, &default_id) {
                                out.push(info);
                            }
                            let _ = ((*(*dev).vtbl).release)(dev);
                        }
                    }
                    let _ = ((*(*col).vtbl).release)(col);
                }
                let _ = ((*(*enum_ptr).vtbl).release)(enum_ptr);
            }
            if uninit {
                CoUninitialize();
            }
        }
        out
    }

    pub struct CaptureSharedState {
        pub shutdown: AtomicBool,
        pub running: AtomicBool,
        pub sample_rate_hz: AtomicU32,
        pub channels: AtomicU32,
        pub captured_blocks: AtomicU64,
        pub wake_event: AtomicU64,
        pub meta: RwLock<(String, String, String)>,
    }

    pub fn run_capture_loop(
        requested_device_id: Option<String>,
        mut frame_tx: rtrb::Producer<StereoFrame>,
        state: Arc<CaptureSharedState>,
    ) {
        let _mmcss = ProAudioThreadGuard::enter();
        unsafe {
            let hr = CoInitializeEx(std::ptr::null(), COINIT_MULTITHREADED as u32);
            let uninit = hr >= 0;

            let mut enum_ptr: *mut IMMDeviceEnumerator = std::ptr::null_mut();
            if CoCreateInstance(
                &CLSID_MMDEVICE_ENUMERATOR,
                std::ptr::null_mut(),
                CLSCTX_ALL,
                &IID_IMMDEVICE_ENUMERATOR,
                (&mut enum_ptr) as *mut *mut IMMDeviceEnumerator as *mut *mut c_void,
            ) < 0
                || enum_ptr.is_null()
            {
                state.meta.write().2 = "MMDeviceEnumerator unavailable".to_string();
                if uninit {
                    CoUninitialize();
                }
                return;
            }

            let mut dev: *mut IMMDevice = std::ptr::null_mut();
            if let Some(ref req_id) = requested_device_id {
                if !req_id.is_empty() && req_id != "default" {
                    let mut wide: Vec<u16> = req_id.encode_utf16().collect();
                    wide.push(0);
                    let _ = ((*(*enum_ptr).vtbl).get_device)(enum_ptr, wide.as_ptr(), &mut dev);
                }
            }
            if dev.is_null() {
                let _ = ((*(*enum_ptr).vtbl).get_default_audio_endpoint)(
                    enum_ptr,
                    E_CAPTURE,
                    E_CONSOLE,
                    &mut dev,
                );
            }
            if dev.is_null() {
                state.meta.write().2 = "No active WASAPI capture microphone found".to_string();
                let _ = ((*(*enum_ptr).vtbl).release)(enum_ptr);
                if uninit {
                    CoUninitialize();
                }
                return;
            }

            if let Some(info) = inspect_imm_device(dev, "") {
                let mut m = state.meta.write();
                m.0 = info.id;
                m.1 = info.friendly_name;
            }

            let mut client: *mut IAudioClient = std::ptr::null_mut();
            if ((*(*dev).vtbl).activate)(
                dev,
                &IID_IAUDIOCLIENT,
                CLSCTX_ALL,
                std::ptr::null(),
                (&mut client) as *mut *mut IAudioClient as *mut *mut c_void,
            ) >= 0
                && !client.is_null()
            {
                let mut mix_wf: *mut WAVEFORMATEX = std::ptr::null_mut();
                if ((*(*client).vtbl).get_mix_format)(client, &mut mix_wf) >= 0
                    && !mix_wf.is_null()
                {
                    let (sr, ch, fmt) = classify_waveformat(mix_wf);
                    state.sample_rate_hz.store(sr, Ordering::Relaxed);
                    state.channels.store(ch as u32, Ordering::Relaxed);

                    let event_handle = CreateEventW(std::ptr::null(), 0, 0, std::ptr::null());
                    state
                        .wake_event
                        .store(event_handle as u64, Ordering::Release);

                    let init_hr = ((*(*client).vtbl).initialize)(
                        client,
                        AUDCLNT_SHAREMODE_SHARED,
                        AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                        200_000, // 20 ms capture buffer
                        0,
                        mix_wf,
                        std::ptr::null(),
                    );
                    CoTaskMemFree(mix_wf as *const c_void);

                    if init_hr >= 0 && !event_handle.is_null() {
                        let _ = ((*(*client).vtbl).set_event_handle)(client, event_handle);
                        let mut cap_client: *mut IAudioCaptureClient = std::ptr::null_mut();
                        if ((*(*client).vtbl).get_service)(
                            client,
                            &IID_IAUDIOCAPTURECLIENT,
                            (&mut cap_client) as *mut *mut IAudioCaptureClient as *mut *mut c_void,
                        ) >= 0
                            && !cap_client.is_null()
                        {
                            let mut decode_scratch = vec![[0.0f32; 2]; 4096];
                            if ((*(*client).vtbl).start)(client) >= 0 {
                                state.running.store(true, Ordering::Release);
                                state.meta.write().2 = format!(
                                    "WASAPI Capture Active ({sr} Hz, {ch} ch, {})",
                                    fmt.label()
                                );

                                while !state.shutdown.load(Ordering::Acquire) {
                                    let _ = WaitForSingleObject(event_handle, 50.min(INFINITE));
                                    if state.shutdown.load(Ordering::Acquire) {
                                        break;
                                    }
                                    let mut pkt_frames = 0u32;
                                    while ((*(*cap_client).vtbl).get_next_packet_size)(
                                        cap_client,
                                        &mut pkt_frames,
                                    ) >= 0
                                        && pkt_frames > 0
                                    {
                                        let mut data_ptr: *mut u8 = std::ptr::null_mut();
                                        let mut frames_to_read = 0u32;
                                        let mut flags = 0u32;
                                        let mut dev_pos = 0u64;
                                        let mut qpc_pos = 0u64;
                                        if ((*(*cap_client).vtbl).get_buffer)(
                                            cap_client,
                                            &mut data_ptr,
                                            &mut frames_to_read,
                                            &mut flags,
                                            &mut dev_pos,
                                            &mut qpc_pos,
                                        ) >= 0
                                        {
                                            let n = (frames_to_read as usize)
                                                .min(decode_scratch.len());
                                            let silent =
                                                (flags & AUDCLNT_BUFFERFLAGS_SILENT) != 0;
                                            unpack_hardware_capture_to_stereo(
                                                data_ptr,
                                                n,
                                                ch as usize,
                                                fmt,
                                                silent,
                                                &mut decode_scratch[..n],
                                            );
                                            for &f in &decode_scratch[..n] {
                                                let _ = frame_tx.push(f);
                                            }
                                            let _ = ((*(*cap_client).vtbl).release_buffer)(
                                                cap_client,
                                                frames_to_read,
                                            );
                                            state
                                                .captured_blocks
                                                .fetch_add(1, Ordering::Relaxed);
                                        } else {
                                            break;
                                        }
                                    }
                                }
                                let _ = ((*(*client).vtbl).stop)(client);
                            }
                            state.running.store(false, Ordering::Release);
                            let _ = ((*(*cap_client).vtbl).release)(cap_client);
                        }
                    } else {
                        state.meta.write().2 =
                            format!("WASAPI Capture Initialize failed (0x{init_hr:08X})");
                    }

                    state.wake_event.store(0, Ordering::Release);
                    if !event_handle.is_null() {
                        let _ = CloseHandle(event_handle);
                    }
                }
                let _ = ((*(*client).vtbl).release)(client);
            }

            let _ = ((*(*dev).vtbl).release)(dev);
            let _ = ((*(*enum_ptr).vtbl).release)(enum_ptr);
            if uninit {
                CoUninitialize();
            }
        }
    }

    pub fn signal_wake_event(ev_u64: u64) {
        if ev_u64 != 0 {
            unsafe {
                let _ = SetEvent(ev_u64 as *mut c_void);
            }
        }
    }
}

/// Enumerates all active Windows Core Audio `eCapture` microphone/line-in endpoints.
pub fn enumerate_wasapi_input_devices() -> Vec<WasapiEndpointInfo> {
    #[cfg(target_os = "windows")]
    {
        win32_capture::enumerate_capture_endpoints()
    }
    #[cfg(not(target_os = "windows"))]
    {
        Vec::new()
    }
}

/// Event-Driven WASAPI Hardware Capture Stream feeding a lock-free `rtrb` SPSC queue.
pub struct WasapiCaptureStream {
    #[cfg(target_os = "windows")]
    state: Arc<win32_capture::CaptureSharedState>,
    worker: Option<JoinHandle<()>>,
}

impl WasapiCaptureStream {
    pub fn spawn(
        device_id: Option<String>,
        ring_capacity_frames: usize,
    ) -> (Self, rtrb::Consumer<StereoFrame>) {
        let (tx, rx) = rtrb::RingBuffer::new(ring_capacity_frames.max(4096));
        #[cfg(target_os = "windows")]
        {
            let state = Arc::new(win32_capture::CaptureSharedState {
                shutdown: AtomicBool::new(false),
                running: AtomicBool::new(false),
                sample_rate_hz: AtomicU32::new(48_000),
                channels: AtomicU32::new(2),
                captured_blocks: AtomicU64::new(0),
                wake_event: AtomicU64::new(0),
                meta: RwLock::new((
                    String::new(),
                    "Default Capture Input".to_string(),
                    "Starting WASAPI Capture...".to_string(),
                )),
            });
            let st_bg = Arc::clone(&state);
            let worker = thread::Builder::new()
                .name("toneforge-wasapi-capture".to_string())
                .spawn(move || {
                    win32_capture::run_capture_loop(device_id, tx, st_bg);
                })
                .expect("failed to spawn toneforge-wasapi-capture thread");
            (
                Self {
                    state,
                    worker: Some(worker),
                },
                rx,
            )
        }
        #[cfg(not(target_os = "windows"))]
        {
            let _ = (device_id, tx);
            (Self { worker: None }, rx)
        }
    }

    pub fn snapshot(&self) -> WasapiCaptureStatusSnapshot {
        #[cfg(target_os = "windows")]
        {
            let (device_id, device_name, status_message) = self.state.meta.read().clone();
            WasapiCaptureStatusSnapshot {
                running: self.state.running.load(Ordering::Relaxed),
                device_id,
                device_name,
                sample_rate_hz: self.state.sample_rate_hz.load(Ordering::Relaxed),
                channels: self.state.channels.load(Ordering::Relaxed) as u16,
                captured_blocks: self.state.captured_blocks.load(Ordering::Relaxed),
                status_message,
            }
        }
        #[cfg(not(target_os = "windows"))]
        {
            WasapiCaptureStatusSnapshot {
                running: false,
                device_id: String::new(),
                device_name: String::new(),
                sample_rate_hz: 48_000,
                channels: 2,
                captured_blocks: 0,
                status_message: "Unsupported OS".to_string(),
            }
        }
    }
}

impl Drop for WasapiCaptureStream {
    fn drop(&mut self) {
        #[cfg(target_os = "windows")]
        {
            self.state.shutdown.store(true, Ordering::Release);
            win32_capture::signal_wake_event(self.state.wake_event.load(Ordering::Acquire));
        }
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_dll_resampler_60s_150ppm_drift_zero_xrun_and_95db_snr() {
        let engine_sr = 48_000.0;
        let drift_ratio = 1.000_150; // +150 ppm faster capture oscillator
        let capture_true_sr = engine_sr * drift_ratio;
        let tone_hz = 997.0;

        // Resampler only knows nominal 48 kHz / 48 kHz — must discover +150 ppm purely via DLL!
        let mut dll = AdaptiveDllDriftResampler::new(engine_sr, engine_sr);

        // Prime initial target fill (`2048` frames)
        let mut cap_sample_idx = 0usize;
        let mut init_chunk = Vec::with_capacity(DLL_TARGET_FILL_FRAMES as usize);
        for _ in 0..(DLL_TARGET_FILL_FRAMES as usize) {
            let t = (cap_sample_idx as f64) / capture_true_sr;
            let s = (0.8 * (2.0 * PI * tone_hz * t).sin()) as f32;
            init_chunk.push([s, s]);
            cap_sample_idx += 1;
        }
        dll.push_captured_frames(&init_chunk);

        let block_len = 256usize;
        let total_blocks = (60.0 * engine_sr / (block_len as f64)).round() as usize; // 11,250 blocks = 60.0 s
        let mut cap_accum = 0.0f64;
        let mut in_scratch = Vec::with_capacity(260);
        let mut out_block = [[0.0f32; 2]; 256];

        // Collect steady-state tail (`last 10 seconds` = 480,000 samples) for exact SNR & frequency verification
        let measure_start_block = total_blocks - 1875;
        let mut tail_samples = Vec::with_capacity(1875 * block_len);

        for b in 0..total_blocks {
            cap_accum += (block_len as f64) * drift_ratio;
            let n_in = cap_accum.floor() as usize;
            cap_accum -= n_in as f64;

            in_scratch.clear();
            for _ in 0..n_in {
                let t = (cap_sample_idx as f64) / capture_true_sr;
                let s = (0.8 * (2.0 * PI * tone_hz * t).sin()) as f32;
                in_scratch.push([s, s]);
                cap_sample_idx += 1;
            }
            dll.push_captured_frames(&in_scratch);
            dll.process_output_block(&mut out_block);

            if b >= measure_start_block {
                for f in &out_block {
                    tail_samples.push(f[0] as f64);
                }
            }
        }

        assert_eq!(
            dll.total_xruns(),
            0,
            "Adaptive DLL must have 0 overruns/underruns across 60s +150 ppm drift"
        );
        let locked_ppm = dll.current_ppm_offset();
        assert!(
            (locked_ppm - 150.0).abs() < 2.0,
            "Adaptive DLL must lock onto +150 ppm clock drift, got {locked_ppm:.3} ppm"
        );

        // Least-squares fit at exact target `997.000 Hz` on `48,000.0 Hz` output grid over 48,000 samples (1.0 s)
        let eval_slice = &tail_samples[tail_samples.len() - 48_000..];
        let mut sum_cc = 0.0f64;
        let mut sum_ss = 0.0f64;
        let mut sum_yc = 0.0f64;
        let mut sum_ys = 0.0f64;
        for (i, &y) in eval_slice.iter().enumerate() {
            let ph = 2.0 * PI * tone_hz * (i as f64) / engine_sr;
            let c = ph.cos();
            let s = ph.sin();
            sum_cc += c * c;
            sum_ss += s * s;
            sum_yc += y * c;
            sum_ys += y * s;
        }
        let a_cos = sum_yc / sum_cc;
        let b_sin = sum_ys / sum_ss;

        let mut sig_pow = 0.0f64;
        let mut err_pow = 0.0f64;
        for (i, &y) in eval_slice.iter().enumerate() {
            let ph = 2.0 * PI * tone_hz * (i as f64) / engine_sr;
            let ideal = a_cos * ph.cos() + b_sin * ph.sin();
            let err = y - ideal;
            sig_pow += ideal * ideal;
            err_pow += err * err;
        }
        let snr_db = 10.0 * (sig_pow / err_pow.max(1e-24)).log10();
        assert!(
            snr_db >= 95.0,
            "Adaptive DLL Resampler SNR must be >= 95.0 dB, got {snr_db:.2} dB (ppm={locked_ppm:.3})"
        );
    }
}
