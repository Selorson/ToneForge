//! Production-Grade Windows Core Audio **WASAPI** Hardware Driver (`wasapi_hardware_driver.rs`).
//!
//! Replaces all simulated sleep loops with genuine Win32 COM WASAPI audio output:
//! - **WASAPI Shared Event-Driven Mode** (default on startup; auto-adapts to `44.1 / 48 / 88.2 / 96 / 176.4 / 192 kHz`).
//! - **WASAPI Exclusive Low-Latency Event-Driven Mode** (hardware-direct bypass of Windows Audio Engine mixer,
//!   including Intel/Realtek HD Audio `AUDCLNT_E_BUFFER_SIZE_NOT_ALIGNED` buffer alignment negotiation and
//!   automatic Shared fallback if another process holds an exclusive lock).
//! - **Automatic Hot-Plug Stream Recovery** on `AUDCLNT_E_DEVICE_INVALIDATED` (`0x88890004`) and
//!   `AUDCLNT_E_RESOURCES_INVALIDATED` (`0x88890026`) when headphones or USB DACs are plugged/unplugged.
//! - **MMCSS `"Pro Audio"` Critical Thread Elevation** via [`ProAudioThreadGuard`].
//! - **Zero-Allocation Real-Time Packet Loop**: Exclusive ownership of `Box<MixerEngine>` on the RT thread,
//!   pre-allocated 16-tap 512-phase Kaiser-windowed sinc resampler when endpoint rate differs from engine rate,
//!   and lock-free format packing (`Float32`, `Int24Packed`, `Int24In32`, `Int32`, `Int16`).

use crate::audio_io::{
    ProAudioThreadGuard, convert_f32_to_int16_lsb, convert_f32_to_int24_lsb,
    convert_f32_to_int32_lsb,
};
use crate::mixer_dag::MixerEngine;
use crate::StereoFrame;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Engine internal block size rendered per `MixerEngine::process_audio_block` invocation.
pub const WASAPI_ENGINE_RENDER_QUANTUM: usize = 256;

/// Ring capacity (power of two) for the zero-allocation hardware resampler / block adapter.
const RESAMPLER_RING_CAPACITY: usize = 16_384;
const RESAMPLER_RING_MASK: usize = RESAMPLER_RING_CAPACITY - 1;

/// Polyphase Kaiser-windowed sinc resampler geometry (`1024` fractional phases × `128` taps, with
/// linear interpolation between adjacent phase rows). 128 taps at Kaiser β = 9.5 give ≈95 dB stopband
/// and a transition band of ≈0.1 × Nyquist, so 44.1 kHz endpoints stay flat to ≈19.6 kHz with
/// images/aliases pushed below the output Nyquist.
const SINC_TAPS: usize = 128;
const SINC_PHASES: usize = 1024;

/// Negotiated hardware sample container format on the active WASAPI endpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WasapiSampleFormat {
    Float32,
    Int32,
    Int24In32,
    Int24Packed,
    Int16,
}

impl WasapiSampleFormat {
    #[inline]
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            Self::Float32 | Self::Int32 | Self::Int24In32 => 4,
            Self::Int24Packed => 3,
            Self::Int16 => 2,
        }
    }

    #[inline]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Float32 => "Float32",
            Self::Int32 => "PCM32",
            Self::Int24In32 => "PCM24-in-32",
            Self::Int24Packed => "PCM24",
            Self::Int16 => "PCM16",
        }
    }
}

/// Information about an active Windows Core Audio output endpoint (`IMMDevice`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WasapiEndpointInfo {
    pub id: String,
    pub friendly_name: String,
    pub is_default: bool,
    pub mix_sample_rate_hz: u32,
    pub channels: u16,
    pub mix_channels: u16,
    pub mix_format_label: String,
}

/// Live snapshot of the running WASAPI hardware output stream exposed to the UI.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WasapiStreamStatusSnapshot {
    pub stream_running: bool,
    pub using_hardware: bool,
    pub exclusive_mode_requested: bool,
    pub exclusive_mode_active: bool,
    pub active_device_id: String,
    pub active_device_name: String,
    pub hardware_sample_rate_hz: u32,
    pub engine_sample_rate_hz: u32,
    pub buffer_frames: u32,
    pub channels: u16,
    pub hardware_channels: u16,
    pub sample_format: String,
    pub mmcss_pro_audio: bool,
    pub mmcss_pro_audio_elevated: bool,
    pub resampler_active: bool,
    pub xrun_count: u64,
    pub hotplug_recoveries: u64,
    pub rendered_blocks: u64,
    pub avg_block_render_us: u32,
    pub speaker_volume_pct: f32,
    pub speaker_muted: bool,
    pub status_message: String,
}

impl Default for WasapiStreamStatusSnapshot {
    fn default() -> Self {
        Self {
            stream_running: false,
            using_hardware: false,
            exclusive_mode_requested: false,
            exclusive_mode_active: false,
            active_device_id: String::new(),
            active_device_name: "Windows Default Output".to_string(),
            hardware_sample_rate_hz: 48_000,
            engine_sample_rate_hz: 48_000,
            buffer_frames: 480,
            channels: 2,
            hardware_channels: 2,
            sample_format: "Float32".to_string(),
            mmcss_pro_audio: false,
            mmcss_pro_audio_elevated: false,
            resampler_active: false,
            xrun_count: 0,
            hotplug_recoveries: 0,
            rendered_blocks: 0,
            avg_block_render_us: 0,
            speaker_volume_pct: 85.0,
            speaker_muted: false,
            status_message: "Initializing WASAPI...".to_string(),
        }
    }
}

/// Desired WASAPI stream configuration updated atomically by the Tauri host when the user selects
/// a new output device or toggles Shared / Exclusive Low-Latency mode.
#[derive(Clone, Debug)]
struct DesiredWasapiConfig {
    /// `None` or empty string means Windows Default Multimedia/Console Render Endpoint.
    device_id: Option<String>,
    exclusive_mode: bool,
    /// Exclusive mode: device rate tried first (the engine keeps its own rate; SRC bridges them).
    preferred_sample_rate_hz: Option<u32>,
    /// Requested device buffer (frames at the device rate); Shared mode never goes below the
    /// endpoint's default period.
    desired_buffer_frames: Option<u32>,
}

/// Shared lock-free / off-RT control block between `WasapiStreamHandle` (Host) and the WASAPI thread.
pub struct WasapiSharedControl {
    shutdown: AtomicBool,
    reopen_requested: AtomicBool,
    stream_running: AtomicBool,
    using_hardware: AtomicBool,
    exclusive_active: AtomicBool,
    mmcss_elevated: AtomicBool,
    hw_sample_rate_hz: AtomicU32,
    engine_sample_rate_hz: AtomicU32,
    buffer_frames: AtomicU32,
    channels: AtomicU32,
    speaker_volume_bits: AtomicU32,
    speaker_muted: AtomicBool,
    xrun_count: AtomicU64,
    hotplug_recoveries: AtomicU64,
    rendered_blocks: AtomicU64,
    avg_block_render_us: AtomicU32,
    desired_config: RwLock<DesiredWasapiConfig>,
    metadata: RwLock<(String, String, String, String)>, // (device_id, device_name, sample_format, status_message)
}

impl WasapiSharedControl {
    fn new(engine_sample_rate_hz: u32, initial_device_id: Option<String>, exclusive_mode: bool) -> Self {
        Self {
            shutdown: AtomicBool::new(false),
            reopen_requested: AtomicBool::new(false),
            stream_running: AtomicBool::new(false),
            using_hardware: AtomicBool::new(false),
            exclusive_active: AtomicBool::new(false),
            mmcss_elevated: AtomicBool::new(false),
            hw_sample_rate_hz: AtomicU32::new(engine_sample_rate_hz),
            engine_sample_rate_hz: AtomicU32::new(engine_sample_rate_hz),
            buffer_frames: AtomicU32::new(WASAPI_ENGINE_RENDER_QUANTUM as u32),
            channels: AtomicU32::new(2),
            speaker_volume_bits: AtomicU32::new(0.85_f32.to_bits()),
            speaker_muted: AtomicBool::new(false),
            xrun_count: AtomicU64::new(0),
            hotplug_recoveries: AtomicU64::new(0),
            rendered_blocks: AtomicU64::new(0),
            avg_block_render_us: AtomicU32::new(0),
            desired_config: RwLock::new(DesiredWasapiConfig {
                device_id: initial_device_id,
                exclusive_mode,
                preferred_sample_rate_hz: None,
                desired_buffer_frames: None,
            }),
            metadata: RwLock::new((
                String::new(),
                "Windows Default Audio Output".to_string(),
                "Float32".to_string(),
                "Starting WASAPI Shared Event-Driven...".to_string(),
            )),
        }
    }

    /// Reads a complete `WasapiStreamStatusSnapshot` without touching or blocking the RT audio loop.
    pub fn snapshot(&self) -> WasapiStreamStatusSnapshot {
        let desired = self.desired_config.read().clone();
        let (active_device_id, active_device_name, sample_format, status_message) =
            self.metadata.read().clone();
        let vol_lin = f32::from_bits(self.speaker_volume_bits.load(Ordering::Relaxed));
        let hw_sr = self.hw_sample_rate_hz.load(Ordering::Relaxed);
        let eng_sr = self.engine_sample_rate_hz.load(Ordering::Relaxed);
        let ch = self.channels.load(Ordering::Relaxed) as u16;
        let mmcss = self.mmcss_elevated.load(Ordering::Relaxed);
        WasapiStreamStatusSnapshot {
            stream_running: self.stream_running.load(Ordering::Relaxed),
            using_hardware: self.using_hardware.load(Ordering::Relaxed),
            exclusive_mode_requested: desired.exclusive_mode,
            exclusive_mode_active: self.exclusive_active.load(Ordering::Relaxed),
            active_device_id,
            active_device_name,
            hardware_sample_rate_hz: hw_sr,
            engine_sample_rate_hz: eng_sr,
            buffer_frames: self.buffer_frames.load(Ordering::Relaxed),
            channels: ch,
            hardware_channels: ch,
            sample_format,
            mmcss_pro_audio: mmcss,
            mmcss_pro_audio_elevated: mmcss,
            resampler_active: hw_sr != eng_sr,
            xrun_count: self.xrun_count.load(Ordering::Relaxed),
            hotplug_recoveries: self.hotplug_recoveries.load(Ordering::Relaxed),
            rendered_blocks: self.rendered_blocks.load(Ordering::Relaxed),
            avg_block_render_us: self.avg_block_render_us.load(Ordering::Relaxed),
            speaker_volume_pct: (vol_lin * 100.0).clamp(0.0, 150.0),
            speaker_muted: self.speaker_muted.load(Ordering::Relaxed),
            status_message,
        }
    }
}

/// Handle held by Tauri Host (`ToneForgeHostState`) to control and inspect the live WASAPI hardware stream.
pub struct WasapiHardwareStream {
    control: Arc<WasapiSharedControl>,
    worker_thread: Option<JoinHandle<Box<MixerEngine>>>,
}

impl WasapiHardwareStream {
    /// Transfers exclusive ownership of `Box<MixerEngine>` to a dedicated MMCSS `"Pro Audio"` WASAPI
    /// hardware thread. Defaults to **WASAPI Shared Event-Driven** when `exclusive_mode == false`.
    pub fn spawn(
        engine: Box<MixerEngine>,
        initial_device_id: Option<String>,
        exclusive_mode: bool,
    ) -> Self {
        let engine_sr = engine.slicer.sample_rate.round().max(8_000.0) as u32;
        let control = Arc::new(WasapiSharedControl::new(
            engine_sr,
            initial_device_id,
            exclusive_mode,
        ));
        let ctrl_clone = Arc::clone(&control);

        let worker_thread = thread::Builder::new()
            .name("toneforge-wasapi-rt".to_string())
            .stack_size(16 * 1024 * 1024)
            .spawn(move || run_wasapi_rt_thread(engine, ctrl_clone))
            .expect("failed to spawn toneforge-wasapi-rt hardware thread");

        Self {
            control,
            worker_thread: Some(worker_thread),
        }
    }

    /// Requests switching the active WASAPI output endpoint and/or Shared vs Exclusive Low-Latency mode.
    /// Signals the WASAPI thread via an atomic flag (`reopen_requested`) with zero audio-thread mutexes.
    pub fn reconfigure_endpoint(&self, device_id: Option<String>, exclusive_mode: bool) {
        let normalized_id = device_id.and_then(|s| {
            let trimmed = s.trim().to_string();
            if trimmed.is_empty() || trimmed == "default" {
                None
            } else {
                Some(trimmed)
            }
        });
        {
            let mut cfg = self.control.desired_config.write();
            cfg.device_id = normalized_id;
            cfg.exclusive_mode = exclusive_mode;
        }
        self.control.reopen_requested.store(true, Ordering::Release);
    }

    /// Sets the preferred Exclusive-mode device rate and the requested device buffer size; applied
    /// on the next (re)open, e.g. by a following [`Self::reconfigure_endpoint`].
    pub fn set_stream_preferences(
        &self,
        preferred_sample_rate_hz: Option<u32>,
        desired_buffer_frames: Option<u32>,
    ) {
        let mut cfg = self.control.desired_config.write();
        cfg.preferred_sample_rate_hz = preferred_sample_rate_hz.filter(|r| (8_000..=384_000).contains(r));
        // Bounded by the packet scratch the render thread pre-allocates (16 384 frames).
        cfg.desired_buffer_frames = desired_buffer_frames.filter(|f| (16..=8_192).contains(f));
    }

    /// Updates master hardware output volume (`0.0..=1.5` linear) and mute state atomically (wait-free).
    #[inline]
    pub fn set_volume_and_mute(&self, volume_linear: f32, muted: bool) {
        let clamped = volume_linear.clamp(0.0, 1.5);
        self.control
            .speaker_volume_bits
            .store(clamped.to_bits(), Ordering::Relaxed);
        self.control.speaker_muted.store(muted, Ordering::Relaxed);
    }

    /// Returns a live snapshot of hardware sample rate, buffer size, mode, device name, and xrun counter.
    #[inline]
    pub fn status_snapshot(&self) -> WasapiStreamStatusSnapshot {
        self.control.snapshot()
    }

    /// Returns `true` while an asynchronous endpoint/mode switch (`reconfigure_endpoint`) is still in flight.
    #[inline]
    pub fn is_reopen_pending(&self) -> bool {
        self.control.reopen_requested.load(Ordering::Acquire)
    }

    /// Cleanly stops the WASAPI stream and recovers ownership of `Box<MixerEngine>` (e.g. for hot-switching to ASIO).
    pub fn stop_and_take_engine(mut self) -> Option<Box<MixerEngine>> {
        self.control.shutdown.store(true, Ordering::Release);
        self.worker_thread.take().and_then(|h| h.join().ok())
    }

    /// Cleanly stops the WASAPI hardware stream and joins the worker thread.
    pub fn shutdown(&mut self) {
        self.control.shutdown.store(true, Ordering::Release);
        if let Some(handle) = self.worker_thread.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for WasapiHardwareStream {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// ============================================================================
// ZERO-ALLOCATION POLYPHASE KAISER-WINDOWED SINC HARDWARE RESAMPLER
// ============================================================================

/// Pre-allocated 16-tap × 512-phase Kaiser-windowed sinc resampler & FIFO ring adapter.
/// Bridges `engine_sample_rate` (`48 kHz`) to any hardware endpoint sample rate
/// (`44.1 / 48 / 88.2 / 96 / 176.4 / 192 kHz`) with **Zero Heap Allocation** in the RT packet loop.
pub struct WasapiStreamingResampler {
    ring: Box<[StereoFrame; RESAMPLER_RING_CAPACITY]>,
    sinc_table: Box<[[f32; SINC_TAPS]; SINC_PHASES + 1]>,
    engine_block: Box<[StereoFrame; WASAPI_ENGINE_RENDER_QUANTUM]>,
    write_idx: usize,
    read_idx: usize,
    available: usize,
    frac_phase: f64,
    step_ratio: f64,
    passthrough_1to1: bool,
    configured_in_sr: u32,
    configured_out_sr: u32,
    smoothed_gain: f32,
}

impl WasapiStreamingResampler {
    pub fn new(engine_sr: u32, hw_sr: u32) -> Self {
        // Heap-allocate fixed arrays directly without placing large temporaries on stack
        let ring_vec = vec![[0.0f32; 2]; RESAMPLER_RING_CAPACITY];
        let ring: Box<[StereoFrame; RESAMPLER_RING_CAPACITY]> = ring_vec
            .into_boxed_slice()
            .try_into()
            .expect("exact ring capacity");

        let table_vec = vec![[0.0f32; SINC_TAPS]; SINC_PHASES + 1];
        let sinc_table: Box<[[f32; SINC_TAPS]; SINC_PHASES + 1]> = table_vec
            .into_boxed_slice()
            .try_into()
            .expect("exact sinc table size");

        let block_vec = vec![[0.0f32; 2]; WASAPI_ENGINE_RENDER_QUANTUM];
        let engine_block: Box<[StereoFrame; WASAPI_ENGINE_RENDER_QUANTUM]> = block_vec
            .into_boxed_slice()
            .try_into()
            .expect("exact engine block size");

        let mut resampler = Self {
            ring,
            sinc_table,
            engine_block,
            write_idx: 0,
            read_idx: 0,
            available: 0,
            frac_phase: 0.0,
            step_ratio: 1.0,
            passthrough_1to1: true,
            configured_in_sr: 0,
            configured_out_sr: 0,
            smoothed_gain: 0.85,
        };
        resampler.reconfigure_rates(engine_sr, hw_sr);
        resampler
    }

    #[inline]
    fn bessel_i0(x: f64) -> f64 {
        let mut sum = 1.0_f64;
        let mut term = 1.0_f64;
        let half_x_sq = 0.25 * x * x;
        for k in 1..=24 {
            let kf = k as f64;
            term *= half_x_sq / (kf * kf);
            sum += term;
            if term < 1e-15 * sum {
                break;
            }
        }
        sum
    }

    /// Recomputes the 512-phase Kaiser-windowed sinc table when endpoint sample rate changes
    /// (called at stream open boundary outside the steady-state RT packet wait loop; 0 allocations).
    pub fn reconfigure_rates(&mut self, engine_sr: u32, hw_sr: u32) {
        let safe_in = engine_sr.max(8_000);
        let safe_out = hw_sr.max(8_000);
        self.write_idx = 0;
        self.read_idx = 0;
        self.available = 0;
        self.frac_phase = 0.0;
        self.passthrough_1to1 = safe_in == safe_out;
        self.step_ratio = (safe_in as f64) / (safe_out as f64);

        if self.configured_in_sr == safe_in && self.configured_out_sr == safe_out {
            return;
        }
        self.configured_in_sr = safe_in;
        self.configured_out_sr = safe_out;

        // Anti-aliasing cutoff (transition-band centre) normalized to the input Nyquist: the
        // stopband edge lands at the lower of the two Nyquist frequencies.
        let min_ratio = ((safe_out as f64) / (safe_in as f64)).min(1.0);
        let cutoff = 0.94 * min_ratio;
        let beta = 9.5_f64;
        let inv_i0_beta = 1.0 / Self::bessel_i0(beta);
        let half_taps = (SINC_TAPS / 2) as f64;

        // `SINC_PHASES + 1` rows so the render loop can interpolate between row `k` and `k + 1`.
        for phase_idx in 0..=SINC_PHASES {
            let mu = (phase_idx as f64) / (SINC_PHASES as f64);
            let mut weight_sum = 0.0_f64;
            let mut raw = [0.0_f64; SINC_TAPS];
            for tap in 0..SINC_TAPS {
                // Tap centered around `idx0 + mu`: offsets `-(SINC_TAPS/2 - 1) ..= SINC_TAPS/2`
                let rel_x = ((tap as isize) - ((SINC_TAPS as isize) / 2 - 1)) as f64 - mu;
                let win_pos = (rel_x / half_taps).clamp(-1.0, 1.0);
                let kaiser =
                    Self::bessel_i0(beta * (1.0 - win_pos * win_pos).max(0.0).sqrt()) * inv_i0_beta;
                let arg = std::f64::consts::PI * cutoff * rel_x;
                let sinc = if arg.abs() < 1e-12 {
                    cutoff
                } else {
                    arg.sin() / (std::f64::consts::PI * rel_x)
                };
                let coeff = sinc * kaiser;
                raw[tap] = coeff;
                weight_sum += coeff;
            }
            let inv_sum = if weight_sum.abs() > 1e-12 {
                1.0 / weight_sum
            } else {
                1.0
            };
            for tap in 0..SINC_TAPS {
                self.sinc_table[phase_idx][tap] = (raw[tap] * inv_sum) as f32;
            }
        }

        // Prime lead-in zeros for centered 16-tap sinc kernel
        if !self.passthrough_1to1 {
            for _ in 0..(SINC_TAPS / 2) {
                self.push_frame([0.0, 0.0]);
            }
        }
    }

    #[inline(always)]
    fn push_frame(&mut self, frame: StereoFrame) {
        if self.available < RESAMPLER_RING_CAPACITY {
            self.ring[self.write_idx] = frame;
            self.write_idx = (self.write_idx + 1) & RESAMPLER_RING_MASK;
            self.available += 1;
        }
    }

    /// Renders `out_frames.len()` hardware stereo frames by pulling `WASAPI_ENGINE_RENDER_QUANTUM`
    /// blocks from `engine` on demand. Strictly **Zero Heap Allocation**.
    pub fn render_hardware_frames(
        &mut self,
        engine: &mut MixerEngine,
        out_frames: &mut [StereoFrame],
        target_volume: f32,
        avg_block_render_us: &AtomicU32,
        rendered_blocks: &AtomicU64,
    ) {
        let gain_smooth_coeff = 0.008_f32;

        if self.passthrough_1to1 {
            let mut written = 0usize;
            let total = out_frames.len();
            while written < total {
                if self.available == 0 {
                    let t0 = Instant::now();
                    engine.process_audio_block(&mut self.engine_block[..]);
                    let us = t0.elapsed().as_micros() as u32;
                    let prev_us = avg_block_render_us.load(Ordering::Relaxed);
                    let ema_us = if prev_us == 0 {
                        us
                    } else {
                        ((prev_us as u64 * 7 + us as u64) / 8) as u32
                    };
                    avg_block_render_us.store(ema_us, Ordering::Relaxed);
                    rendered_blocks.fetch_add(1, Ordering::Relaxed);

                    for i in 0..WASAPI_ENGINE_RENDER_QUANTUM {
                        self.push_frame(self.engine_block[i]);
                    }
                }

                let chunk = (total - written).min(self.available);
                for slot in &mut out_frames[written..written + chunk] {
                    let f = self.ring[self.read_idx];
                    self.read_idx = (self.read_idx + 1) & RESAMPLER_RING_MASK;
                    self.smoothed_gain += gain_smooth_coeff * (target_volume - self.smoothed_gain);
                    let g = self.smoothed_gain;
                    *slot = [f[0] * g, f[1] * g];
                }
                self.available -= chunk;
                written += chunk;
            }
            return;
        }

        // Polyphase 16-tap Kaiser-windowed sinc resampling path (`44.1 / 88.2 / 96 / 176.4 / 192 kHz`)
        for slot in out_frames.iter_mut() {
            while self.available < SINC_TAPS + 4 {
                let t0 = Instant::now();
                engine.process_audio_block(&mut self.engine_block[..]);
                let us = t0.elapsed().as_micros() as u32;
                let prev_us = avg_block_render_us.load(Ordering::Relaxed);
                let ema_us = if prev_us == 0 {
                    us
                } else {
                    ((prev_us as u64 * 7 + us as u64) / 8) as u32
                };
                avg_block_render_us.store(ema_us, Ordering::Relaxed);
                rendered_blocks.fetch_add(1, Ordering::Relaxed);

                for i in 0..WASAPI_ENGINE_RENDER_QUANTUM {
                    self.push_frame(self.engine_block[i]);
                }
            }

            let phase_scaled = self.frac_phase * (SINC_PHASES as f64);
            let phase_idx = (phase_scaled as usize).min(SINC_PHASES - 1);
            let phase_frac = (phase_scaled - phase_idx as f64).clamp(0.0, 1.0) as f32;
            let taps0 = &self.sinc_table[phase_idx];
            let taps1 = &self.sinc_table[phase_idx + 1];

            let mut acc_l = 0.0_f32;
            let mut acc_r = 0.0_f32;
            let base = self.read_idx;
            for t in 0..SINC_TAPS {
                let s = self.ring[(base + t) & RESAMPLER_RING_MASK];
                let w = taps0[t] + phase_frac * (taps1[t] - taps0[t]);
                acc_l += s[0] * w;
                acc_r += s[1] * w;
            }

            self.smoothed_gain += gain_smooth_coeff * (target_volume - self.smoothed_gain);
            let g = self.smoothed_gain;
            *slot = [acc_l * g, acc_r * g];

            self.frac_phase += self.step_ratio;
            let advance = self.frac_phase.floor() as usize;
            if advance > 0 {
                let step = advance.min(self.available);
                self.read_idx = (self.read_idx + step) & RESAMPLER_RING_MASK;
                self.available -= step;
                self.frac_phase -= advance as f64;
            }
        }
    }
}

/// Packs normalized `StereoFrame`s into raw hardware endpoint buffer memory (`data_ptr`) across
/// arbitrary channel counts (`1`, `2`, `4`, `6`, `8` ch) and sample formats (`Float32`, `Int32`,
/// `Int24In32`, `Int24Packed`, `Int16`). Strictly **Zero Heap Allocation**.
///
/// # Safety
/// `dst_bytes` must point to at least `frames.len() * (channels as usize) * format.bytes_per_sample()` valid bytes.
#[inline]
pub unsafe fn pack_stereo_frames_to_hardware_buffer(
    frames: &[StereoFrame],
    dst_bytes: *mut u8,
    channels: usize,
    format: WasapiSampleFormat,
) {
    if dst_bytes.is_null() || channels == 0 || frames.is_empty() {
        return;
    }
    match format {
        WasapiSampleFormat::Float32 => {
            let out = unsafe {
                std::slice::from_raw_parts_mut(dst_bytes as *mut f32, frames.len() * channels)
            };
            if channels == 2 {
                for (i, f) in frames.iter().enumerate() {
                    let l = if f[0].is_finite() {
                        f[0].clamp(-1.0, 1.0)
                    } else {
                        0.0
                    };
                    let r = if f[1].is_finite() {
                        f[1].clamp(-1.0, 1.0)
                    } else {
                        0.0
                    };
                    out[i * 2] = l;
                    out[i * 2 + 1] = r;
                }
            } else if channels == 1 {
                for (i, f) in frames.iter().enumerate() {
                    let mono = 0.5 * (f[0] + f[1]);
                    out[i] = if mono.is_finite() {
                        mono.clamp(-1.0, 1.0)
                    } else {
                        0.0
                    };
                }
            } else {
                for (i, f) in frames.iter().enumerate() {
                    let base = i * channels;
                    out[base] = if f[0].is_finite() {
                        f[0].clamp(-1.0, 1.0)
                    } else {
                        0.0
                    };
                    out[base + 1] = if f[1].is_finite() {
                        f[1].clamp(-1.0, 1.0)
                    } else {
                        0.0
                    };
                    for c in 2..channels {
                        out[base + c] = 0.0;
                    }
                }
            }
        }
        WasapiSampleFormat::Int32 | WasapiSampleFormat::Int24In32 => {
            let out = unsafe {
                std::slice::from_raw_parts_mut(dst_bytes as *mut i32, frames.len() * channels)
            };
            let mask = if format == WasapiSampleFormat::Int24In32 {
                !0xFF_i32
            } else {
                !0_i32
            };
            for (i, f) in frames.iter().enumerate() {
                let base = i * channels;
                if channels == 1 {
                    out[base] = convert_f32_to_int32_lsb(0.5 * (f[0] + f[1])) & mask;
                } else {
                    out[base] = convert_f32_to_int32_lsb(f[0]) & mask;
                    out[base + 1] = convert_f32_to_int32_lsb(f[1]) & mask;
                    for c in 2..channels {
                        out[base + c] = 0;
                    }
                }
            }
        }
        WasapiSampleFormat::Int24Packed => {
            let stride = channels * 3;
            let out =
                unsafe { std::slice::from_raw_parts_mut(dst_bytes, frames.len() * stride) };
            for (i, f) in frames.iter().enumerate() {
                let base = i * stride;
                if channels == 1 {
                    let b = convert_f32_to_int24_lsb(0.5 * (f[0] + f[1]));
                    out[base..base + 3].copy_from_slice(&b);
                } else {
                    let bl = convert_f32_to_int24_lsb(f[0]);
                    let br = convert_f32_to_int24_lsb(f[1]);
                    out[base..base + 3].copy_from_slice(&bl);
                    out[base + 3..base + 6].copy_from_slice(&br);
                    if stride > 6 {
                        out[base + 6..base + stride].fill(0);
                    }
                }
            }
        }
        WasapiSampleFormat::Int16 => {
            let out = unsafe {
                std::slice::from_raw_parts_mut(dst_bytes as *mut i16, frames.len() * channels)
            };
            for (i, f) in frames.iter().enumerate() {
                let base = i * channels;
                if channels == 1 {
                    out[base] = convert_f32_to_int16_lsb(0.5 * (f[0] + f[1]));
                } else {
                    out[base] = convert_f32_to_int16_lsb(f[0]);
                    out[base + 1] = convert_f32_to_int16_lsb(f[1]);
                    for c in 2..channels {
                        out[base + c] = 0;
                    }
                }
            }
        }
    }
}

// ============================================================================
// WIN32 CORE AUDIO COM VTABLES & WASAPI IMPLEMENTATION
// ============================================================================

#[cfg(target_os = "windows")]
mod win32_wasapi {
    use super::*;
    use std::ffi::c_void;
    use std::ptr;
    use windows_sys::Win32::Media::Audio::{
        AUDCLNT_SHAREMODE_EXCLUSIVE, AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
        WAVEFORMATEX, WAVEFORMATEXTENSIBLE,
    };
    use windows_sys::Win32::System::Com::{
        CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx, CoTaskMemFree,
        CoUninitialize,
    };
    use windows_sys::Win32::System::Threading::{INFINITE, WaitForSingleObject};

    const WAIT_OBJECT_0: u32 = 0;
    const WAIT_TIMEOUT: u32 = 0x0000_0102;
    use windows_sys::core::GUID;

    const S_OK: i32 = 0;
    const S_FALSE: i32 = 1;
    const RPC_E_CHANGED_MODE: i32 = 0x8001_0106_u32 as i32;

    const E_RENDER: i32 = 0;
    const E_CONSOLE: i32 = 0;
    const E_MULTIMEDIA: i32 = 1;
    const DEVICE_STATE_ACTIVE: u32 = 0x0000_0001;
    const STGM_READ: u32 = 0;
    const VT_LPWSTR: u16 = 31;

    const WAVE_FORMAT_PCM_TAG: u16 = 0x0001;
    const WAVE_FORMAT_IEEE_FLOAT_TAG: u16 = 0x0003;
    const WAVE_FORMAT_EXTENSIBLE_TAG: u16 = 0xFFFE;

    // HRESULTs from `Audioclient.h`
    const AUDCLNT_E_DEVICE_INVALIDATED: i32 = 0x8889_0004_u32 as i32;
    const AUDCLNT_E_DEVICE_IN_USE: i32 = 0x8889_000A_u32 as i32;
    const AUDCLNT_E_EXCLUSIVE_MODE_NOT_ALLOWED: i32 = 0x8889_000E_u32 as i32;
    const AUDCLNT_E_BUFFER_SIZE_NOT_ALIGNED: i32 = 0x8889_0019_u32 as i32;
    const AUDCLNT_E_RESOURCES_INVALIDATED: i32 = 0x8889_0026_u32 as i32;

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
    const IID_IAUDIORENDERCLIENT: GUID = GUID {
        data1: 0xF294ACFC,
        data2: 0x3146,
        data3: 0x4483,
        data4: [0xA7, 0xBF, 0xAD, 0xDC, 0xA7, 0xC2, 0x60, 0xE2],
    };
    const IID_IAUDIOCLOCK: GUID = GUID {
        data1: 0xCD63314F,
        data2: 0x3FBA,
        data3: 0x4A1B,
        data4: [0x81, 0x2C, 0xEF, 0x96, 0x35, 0x87, 0x28, 0xE7],
    };
    const KSDATAFORMAT_SUBTYPE_PCM_GUID: GUID = GUID {
        data1: 0x00000001,
        data2: 0x0000,
        data3: 0x0010,
        data4: [0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71],
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
            lp_event_attributes: *const c_void,
            b_manual_reset: i32,
            b_initial_state: i32,
            lp_name: *const u16,
        ) -> *mut c_void;
        fn CloseHandle(h_object: *mut c_void) -> i32;
    }

    // Raw COM Vtables matching Windows SDK `mmdeviceapi.h`, `propsys.h`, and `audioclient.h`
    #[repr(C)]
    struct IMMDeviceEnumerator {
        lp_vtbl: *const IMMDeviceEnumeratorVtbl,
    }

    #[repr(C)]
    struct IMMDeviceEnumeratorVtbl {
        query_interface: unsafe extern "system" fn(*mut IMMDeviceEnumerator, *const GUID, *mut *mut c_void) -> i32,
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
        register_endpoint_notification_callback:
            unsafe extern "system" fn(*mut IMMDeviceEnumerator, *mut c_void) -> i32,
        unregister_endpoint_notification_callback:
            unsafe extern "system" fn(*mut IMMDeviceEnumerator, *mut c_void) -> i32,
    }

    #[repr(C)]
    struct IMMDeviceCollection {
        lp_vtbl: *const IMMDeviceCollectionVtbl,
    }

    #[repr(C)]
    struct IMMDeviceCollectionVtbl {
        query_interface: unsafe extern "system" fn(*mut IMMDeviceCollection, *const GUID, *mut *mut c_void) -> i32,
        add_ref: unsafe extern "system" fn(*mut IMMDeviceCollection) -> u32,
        release: unsafe extern "system" fn(*mut IMMDeviceCollection) -> u32,
        get_count: unsafe extern "system" fn(*mut IMMDeviceCollection, *mut u32) -> i32,
        item: unsafe extern "system" fn(*mut IMMDeviceCollection, u32, *mut *mut IMMDevice) -> i32,
    }

    #[repr(C)]
    struct IMMDevice {
        lp_vtbl: *const IMMDeviceVtbl,
    }

    #[repr(C)]
    struct IMMDeviceVtbl {
        query_interface: unsafe extern "system" fn(*mut IMMDevice, *const GUID, *mut *mut c_void) -> i32,
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
        get_state: unsafe extern "system" fn(*mut IMMDevice, *mut u32) -> i32,
    }

    #[repr(C)]
    struct IPropertyStore {
        lp_vtbl: *const IPropertyStoreVtbl,
    }

    #[repr(C)]
    struct IPropertyStoreVtbl {
        query_interface: unsafe extern "system" fn(*mut IPropertyStore, *const GUID, *mut *mut c_void) -> i32,
        add_ref: unsafe extern "system" fn(*mut IPropertyStore) -> u32,
        release: unsafe extern "system" fn(*mut IPropertyStore) -> u32,
        get_count: unsafe extern "system" fn(*mut IPropertyStore, *mut u32) -> i32,
        get_at: unsafe extern "system" fn(*mut IPropertyStore, u32, *mut PropertyKey) -> i32,
        get_value:
            unsafe extern "system" fn(*mut IPropertyStore, *const PropertyKey, *mut RawPropVariant) -> i32,
        set_value: unsafe extern "system" fn(*mut IPropertyStore, *const PropertyKey, *const RawPropVariant) -> i32,
        commit: unsafe extern "system" fn(*mut IPropertyStore) -> i32,
    }

    #[repr(C)]
    struct IAudioClient {
        lp_vtbl: *const IAudioClientVtbl,
    }

    #[repr(C)]
    struct IAudioClientVtbl {
        query_interface: unsafe extern "system" fn(*mut IAudioClient, *const GUID, *mut *mut c_void) -> i32,
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
        get_device_period: unsafe extern "system" fn(*mut IAudioClient, *mut i64, *mut i64) -> i32,
        start: unsafe extern "system" fn(*mut IAudioClient) -> i32,
        stop: unsafe extern "system" fn(*mut IAudioClient) -> i32,
        reset: unsafe extern "system" fn(*mut IAudioClient) -> i32,
        set_event_handle: unsafe extern "system" fn(*mut IAudioClient, *mut c_void) -> i32,
        get_service: unsafe extern "system" fn(*mut IAudioClient, *const GUID, *mut *mut c_void) -> i32,
    }

    #[repr(C)]
    struct IAudioRenderClient {
        lp_vtbl: *const IAudioRenderClientVtbl,
    }

    #[repr(C)]
    struct IAudioRenderClientVtbl {
        query_interface:
            unsafe extern "system" fn(*mut IAudioRenderClient, *const GUID, *mut *mut c_void) -> i32,
        add_ref: unsafe extern "system" fn(*mut IAudioRenderClient) -> u32,
        release: unsafe extern "system" fn(*mut IAudioRenderClient) -> u32,
        get_buffer: unsafe extern "system" fn(*mut IAudioRenderClient, u32, *mut *mut u8) -> i32,
        release_buffer: unsafe extern "system" fn(*mut IAudioRenderClient, u32, u32) -> i32,
    }

    #[repr(C)]
    struct IAudioClock {
        lp_vtbl: *const IAudioClockVtbl,
    }

    #[repr(C)]
    struct IAudioClockVtbl {
        query_interface: unsafe extern "system" fn(*mut IAudioClock, *const GUID, *mut *mut c_void) -> i32,
        add_ref: unsafe extern "system" fn(*mut IAudioClock) -> u32,
        release: unsafe extern "system" fn(*mut IAudioClock) -> u32,
        get_frequency: unsafe extern "system" fn(*mut IAudioClock, *mut u64) -> i32,
        get_position: unsafe extern "system" fn(*mut IAudioClock, *mut u64, *mut u64) -> i32,
        get_characteristics: unsafe extern "system" fn(*mut IAudioClock, *mut u32) -> i32,
    }

    struct ComInitGuard {
        should_uninit: bool,
    }

    impl ComInitGuard {
        fn enter() -> Result<Self, String> {
            let hr = unsafe { CoInitializeEx(ptr::null(), COINIT_MULTITHREADED as u32) };
            if hr == S_OK || hr == S_FALSE {
                Ok(Self {
                    should_uninit: true,
                })
            } else if hr == RPC_E_CHANGED_MODE {
                Ok(Self {
                    should_uninit: false,
                })
            } else {
                Err(format!("CoInitializeEx failed: 0x{:08X}", hr as u32))
            }
        }
    }

    impl Drop for ComInitGuard {
        fn drop(&mut self) {
            if self.should_uninit {
                unsafe {
                    CoUninitialize();
                }
            }
        }
    }

    #[inline]
    fn guid_eq(a: &GUID, b: &GUID) -> bool {
        a.data1 == b.data1 && a.data2 == b.data2 && a.data3 == b.data3 && a.data4 == b.data4
    }

    unsafe fn pwstr_to_string(pwstr: *const u16) -> String {
        if pwstr.is_null() {
            return String::new();
        }
        let mut len = 0usize;
        while unsafe { *pwstr.add(len) } != 0 {
            len += 1;
            if len > 4096 {
                break;
            }
        }
        String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(pwstr, len) })
    }

    fn string_to_utf16_null(s: &str) -> Vec<u16> {
        let mut v: Vec<u16> = s.encode_utf16().collect();
        v.push(0);
        v
    }

    unsafe fn query_device_id_and_name(device: *mut IMMDevice) -> (String, String) {
        let mut id_str = String::new();
        let mut pwstr_id: *mut u16 = ptr::null_mut();
        if unsafe { ((*(*device).lp_vtbl).get_id)(device, &mut pwstr_id) } >= 0 && !pwstr_id.is_null() {
            id_str = unsafe { pwstr_to_string(pwstr_id) };
            unsafe {
                CoTaskMemFree(pwstr_id as *const c_void);
            }
        }

        let mut friendly = String::from("Windows Audio Endpoint");
        let mut prop_store: *mut IPropertyStore = ptr::null_mut();
        if unsafe { ((*(*device).lp_vtbl).open_property_store)(device, STGM_READ, &mut prop_store) } >= 0
            && !prop_store.is_null()
        {
            let mut pv = RawPropVariant {
                vt: 0,
                reserved1: 0,
                reserved2: 0,
                reserved3: 0,
                ptr_val: ptr::null_mut(),
                pad_val: 0,
            };
            if unsafe {
                ((*(*prop_store).lp_vtbl).get_value)(prop_store, &PKEY_DEVICE_FRIENDLY_NAME, &mut pv)
            } >= 0
            {
                if pv.vt == VT_LPWSTR && !pv.ptr_val.is_null() {
                    let parsed = unsafe { pwstr_to_string(pv.ptr_val) };
                    if !parsed.trim().is_empty() {
                        friendly = parsed;
                    }
                }
                unsafe {
                    let _ = PropVariantClear(&mut pv);
                }
            }
            unsafe {
                ((*(*prop_store).lp_vtbl).release)(prop_store);
            }
        }
        (id_str, friendly)
    }

    unsafe fn parse_waveformat(fmt_ptr: *const WAVEFORMATEX) -> Option<(u32, u16, WasapiSampleFormat)> {
        if fmt_ptr.is_null() {
            return None;
        }
        let wf = unsafe { ptr::read_unaligned(fmt_ptr) };
        let sr = wf.nSamplesPerSec;
        let ch = wf.nChannels;
        let bits = wf.wBitsPerSample;
        let tag = wf.wFormatTag;

        if sr < 8_000 || ch == 0 {
            return None;
        }

        if tag == WAVE_FORMAT_IEEE_FLOAT_TAG && bits == 32 {
            return Some((sr, ch, WasapiSampleFormat::Float32));
        }
        if tag == WAVE_FORMAT_PCM_TAG {
            return match bits {
                16 => Some((sr, ch, WasapiSampleFormat::Int16)),
                24 => Some((sr, ch, WasapiSampleFormat::Int24Packed)),
                32 => Some((sr, ch, WasapiSampleFormat::Int32)),
                _ => None,
            };
        }
        if tag == WAVE_FORMAT_EXTENSIBLE_TAG && wf.cbSize >= 22 {
            let ext = unsafe { ptr::read_unaligned(fmt_ptr as *const WAVEFORMATEXTENSIBLE) };
            let sub = unsafe { ptr::addr_of!(ext.SubFormat).read_unaligned() };
            let valid_bits = unsafe { ptr::addr_of!(ext.Samples.wValidBitsPerSample).read_unaligned() };
            if guid_eq(&sub, &KSDATAFORMAT_SUBTYPE_IEEE_FLOAT_GUID) && bits == 32 {
                return Some((sr, ch, WasapiSampleFormat::Float32));
            }
            if guid_eq(&sub, &KSDATAFORMAT_SUBTYPE_PCM_GUID) {
                return match (bits, valid_bits) {
                    (32, 24) => Some((sr, ch, WasapiSampleFormat::Int24In32)),
                    (32, _) => Some((sr, ch, WasapiSampleFormat::Int32)),
                    (24, _) => Some((sr, ch, WasapiSampleFormat::Int24Packed)),
                    (16, _) => Some((sr, ch, WasapiSampleFormat::Int16)),
                    _ => None,
                };
            }
        }
        None
    }

    fn build_waveformat_extensible(
        sample_rate: u32,
        channels: u16,
        format: WasapiSampleFormat,
    ) -> WAVEFORMATEXTENSIBLE {
        let (bits_per_sample, valid_bits, sub_format) = match format {
            WasapiSampleFormat::Float32 => (32u16, 32u16, KSDATAFORMAT_SUBTYPE_IEEE_FLOAT_GUID),
            WasapiSampleFormat::Int32 => (32u16, 32u16, KSDATAFORMAT_SUBTYPE_PCM_GUID),
            WasapiSampleFormat::Int24In32 => (32u16, 24u16, KSDATAFORMAT_SUBTYPE_PCM_GUID),
            WasapiSampleFormat::Int24Packed => (24u16, 24u16, KSDATAFORMAT_SUBTYPE_PCM_GUID),
            WasapiSampleFormat::Int16 => (16u16, 16u16, KSDATAFORMAT_SUBTYPE_PCM_GUID),
        };
        let block_align = channels * (bits_per_sample / 8);
        let avg_bytes_per_sec = sample_rate * (block_align as u32);
        let channel_mask: u32 = match channels {
            1 => 0x4,   // SPEAKER_FRONT_CENTER
            2 => 0x3,   // SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT
            4 => 0x33,  // Quad
            6 => 0x3F,  // 5.1
            8 => 0x63F, // 7.1
            _ => (1u32 << channels.min(31)) - 1,
        };

        let mut ext: WAVEFORMATEXTENSIBLE = unsafe { std::mem::zeroed() };
        ext.Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE_TAG;
        ext.Format.nChannels = channels;
        ext.Format.nSamplesPerSec = sample_rate;
        ext.Format.nAvgBytesPerSec = avg_bytes_per_sec;
        ext.Format.nBlockAlign = block_align;
        ext.Format.wBitsPerSample = bits_per_sample;
        ext.Format.cbSize = 22;
        ext.Samples.wValidBitsPerSample = valid_bits;
        ext.dwChannelMask = channel_mask;
        ext.SubFormat = sub_format;
        ext
    }

    pub fn enumerate_endpoints_impl() -> Result<Vec<WasapiEndpointInfo>, String> {
        let _com = ComInitGuard::enter()?;
        let mut enumerator: *mut IMMDeviceEnumerator = ptr::null_mut();
        let hr = unsafe {
            CoCreateInstance(
                &CLSID_MMDEVICE_ENUMERATOR,
                ptr::null_mut(),
                CLSCTX_ALL,
                &IID_IMMDEVICE_ENUMERATOR,
                &mut enumerator as *mut *mut IMMDeviceEnumerator as *mut *mut c_void,
            )
        };
        if hr < 0 || enumerator.is_null() {
            return Err(format!("CoCreateInstance(MMDeviceEnumerator) failed: 0x{:08X}", hr as u32));
        }

        let mut default_id = String::new();
        let mut default_dev: *mut IMMDevice = ptr::null_mut();
        if unsafe {
            ((*(*enumerator).lp_vtbl).get_default_audio_endpoint)(
                enumerator,
                E_RENDER,
                E_MULTIMEDIA,
                &mut default_dev,
            )
        } >= 0
            && !default_dev.is_null()
        {
            let (did, _) = unsafe { query_device_id_and_name(default_dev) };
            default_id = did;
            unsafe {
                ((*(*default_dev).lp_vtbl).release)(default_dev);
            }
        }

        let mut collection: *mut IMMDeviceCollection = ptr::null_mut();
        let hr_enum = unsafe {
            ((*(*enumerator).lp_vtbl).enum_audio_endpoints)(
                enumerator,
                E_RENDER,
                DEVICE_STATE_ACTIVE,
                &mut collection,
            )
        };
        if hr_enum < 0 || collection.is_null() {
            unsafe {
                ((*(*enumerator).lp_vtbl).release)(enumerator);
            }
            return Err(format!("EnumAudioEndpoints failed: 0x{:08X}", hr_enum as u32));
        }

        let mut count: u32 = 0;
        let _ = unsafe { ((*(*collection).lp_vtbl).get_count)(collection, &mut count) };
        let mut out = Vec::with_capacity(count as usize);

        for idx in 0..count {
            let mut dev: *mut IMMDevice = ptr::null_mut();
            if unsafe { ((*(*collection).lp_vtbl).item)(collection, idx, &mut dev) } >= 0
                && !dev.is_null()
            {
                let (id, friendly_name) = unsafe { query_device_id_and_name(dev) };
                let mut mix_sr = 48_000u32;
                let mut mix_ch = 2u16;
                let mut fmt_label = String::from("Float32");

                let mut client: *mut IAudioClient = ptr::null_mut();
                if unsafe {
                    ((*(*dev).lp_vtbl).activate)(
                        dev,
                        &IID_IAUDIOCLIENT,
                        CLSCTX_ALL,
                        ptr::null(),
                        &mut client as *mut *mut IAudioClient as *mut *mut c_void,
                    )
                } >= 0
                    && !client.is_null()
                {
                    let mut mix_fmt: *mut WAVEFORMATEX = ptr::null_mut();
                    if unsafe { ((*(*client).lp_vtbl).get_mix_format)(client, &mut mix_fmt) } >= 0
                        && !mix_fmt.is_null()
                    {
                        if let Some((sr, ch, fmt)) = unsafe { parse_waveformat(mix_fmt) } {
                            mix_sr = sr;
                            mix_ch = ch;
                            fmt_label = fmt.label().to_string();
                        }
                        unsafe {
                            CoTaskMemFree(mix_fmt as *const c_void);
                        }
                    }
                    unsafe {
                        ((*(*client).lp_vtbl).release)(client);
                    }
                }

                let is_default = !default_id.is_empty() && id == default_id;
                out.push(WasapiEndpointInfo {
                    id,
                    friendly_name,
                    is_default,
                    mix_sample_rate_hz: mix_sr,
                    channels: mix_ch,
                    mix_channels: mix_ch,
                    mix_format_label: fmt_label,
                });

                unsafe {
                    ((*(*dev).lp_vtbl).release)(dev);
                }
            }
        }

        unsafe {
            ((*(*collection).lp_vtbl).release)(collection);
            ((*(*enumerator).lp_vtbl).release)(enumerator);
        }

        // Put default endpoint first for UI convenience
        out.sort_by(|a, b| b.is_default.cmp(&a.is_default).then_with(|| a.friendly_name.cmp(&b.friendly_name)));
        Ok(out)
    }

    struct OpenedWasapiSession {
        audio_client: *mut IAudioClient,
        render_client: *mut IAudioRenderClient,
        audio_clock: *mut IAudioClock,
        event_handle: *mut c_void,
        device_id: String,
        device_name: String,
        sample_rate_hz: u32,
        channels: u16,
        buffer_frames: u32,
        sample_format: WasapiSampleFormat,
        exclusive_active: bool,
        status_message: String,
    }

    impl Drop for OpenedWasapiSession {
        fn drop(&mut self) {
            unsafe {
                if !self.audio_client.is_null() {
                    let _ = ((*(*self.audio_client).lp_vtbl).stop)(self.audio_client);
                    let _ = ((*(*self.audio_client).lp_vtbl).reset)(self.audio_client);
                }
                if !self.audio_clock.is_null() {
                    ((*(*self.audio_clock).lp_vtbl).release)(self.audio_clock);
                    self.audio_clock = ptr::null_mut();
                }
                if !self.render_client.is_null() {
                    ((*(*self.render_client).lp_vtbl).release)(self.render_client);
                    self.render_client = ptr::null_mut();
                }
                if !self.audio_client.is_null() {
                    ((*(*self.audio_client).lp_vtbl).release)(self.audio_client);
                    self.audio_client = ptr::null_mut();
                }
                if !self.event_handle.is_null() {
                    let _ = CloseHandle(self.event_handle);
                    self.event_handle = ptr::null_mut();
                }
            }
        }
    }

    unsafe fn activate_audio_client(device: *mut IMMDevice) -> Result<*mut IAudioClient, i32> {
        let mut client: *mut IAudioClient = ptr::null_mut();
        let hr = unsafe {
            ((*(*device).lp_vtbl).activate)(
                device,
                &IID_IAUDIOCLIENT,
                CLSCTX_ALL,
                ptr::null(),
                &mut client as *mut *mut IAudioClient as *mut *mut c_void,
            )
        };
        if hr < 0 || client.is_null() {
            Err(hr)
        } else {
            Ok(client)
        }
    }

    unsafe fn try_open_exclusive_on_device(
        device: *mut IMMDevice,
        engine_sr: u32,
        mix_channels: u16,
        device_id: &str,
        device_name: &str,
        desired: &DesiredWasapiConfig,
    ) -> Result<OpenedWasapiSession, String> {
        let mut probe_client =
            unsafe { activate_audio_client(device) }.map_err(|hr| format!("Activate failed: 0x{:08X}", hr as u32))?;

        let mut default_period: i64 = 100_000;
        let mut min_period: i64 = 30_000;
        let _ = unsafe {
            ((*(*probe_client).lp_vtbl).get_device_period)(
                probe_client,
                &mut default_period,
                &mut min_period,
            )
        };
        let target_period_hns = default_period.max(min_period).max(30_000);

        let candidate_rates = [
            desired.preferred_sample_rate_hz.unwrap_or(engine_sr),
            engine_sr,
            48_000,
            44_100,
            96_000,
            88_200,
            192_000,
            176_400,
        ];
        let candidate_formats = [
            WasapiSampleFormat::Float32,
            WasapiSampleFormat::Int24In32,
            WasapiSampleFormat::Int32,
            WasapiSampleFormat::Int24Packed,
            WasapiSampleFormat::Int16,
        ];
        let candidate_channels = if mix_channels > 2 {
            [2u16, mix_channels]
        } else {
            [2u16, 2u16]
        };

        let mut chosen: Option<(WAVEFORMATEXTENSIBLE, u32, u16, WasapiSampleFormat)> = None;
        'search: for &sr in &candidate_rates {
            for &ch in &candidate_channels {
                for &fmt in &candidate_formats {
                    let wfxt = build_waveformat_extensible(sr, ch, fmt);
                    let hr = unsafe {
                        ((*(*probe_client).lp_vtbl).is_format_supported)(
                            probe_client,
                            AUDCLNT_SHAREMODE_EXCLUSIVE,
                            &wfxt.Format,
                            ptr::null_mut(),
                        )
                    };
                    if hr == S_OK {
                        chosen = Some((wfxt, sr, ch, fmt));
                        break 'search;
                    }
                }
            }
        }

        let Some((wfxt, sr, ch, sample_fmt)) = chosen else {
            unsafe {
                ((*(*probe_client).lp_vtbl).release)(probe_client);
            }
            return Err("No Exclusive PCM/Float format supported by endpoint".to_string());
        };
        let target_period_hns = desired
            .desired_buffer_frames
            .map(|f| ((10_000_000.0 * f as f64) / sr as f64).round() as i64)
            .map(|hns| hns.clamp(min_period.max(10_000), 5_000_000))
            .unwrap_or(target_period_hns);

        let mut hr_init = unsafe {
            ((*(*probe_client).lp_vtbl).initialize)(
                probe_client,
                AUDCLNT_SHAREMODE_EXCLUSIVE,
                AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                target_period_hns,
                target_period_hns,
                &wfxt.Format,
                ptr::null(),
            )
        };

        // Handle HD Audio hardware buffer alignment requirement (`AUDCLNT_E_BUFFER_SIZE_NOT_ALIGNED`)
        if hr_init == AUDCLNT_E_BUFFER_SIZE_NOT_ALIGNED {
            let mut aligned_frames: u32 = 0;
            let _ = unsafe { ((*(*probe_client).lp_vtbl).get_buffer_size)(probe_client, &mut aligned_frames) };
            unsafe {
                ((*(*probe_client).lp_vtbl).release)(probe_client);
            }
            if aligned_frames > 0 {
                probe_client = unsafe { activate_audio_client(device) }
                    .map_err(|hr| format!("Re-activate failed: 0x{:08X}", hr as u32))?;
                let aligned_hns =
                    ((10_000_000.0_f64 * (aligned_frames as f64)) / (sr as f64) + 0.5) as i64;
                hr_init = unsafe {
                    ((*(*probe_client).lp_vtbl).initialize)(
                        probe_client,
                        AUDCLNT_SHAREMODE_EXCLUSIVE,
                        AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                        aligned_hns,
                        aligned_hns,
                        &wfxt.Format,
                        ptr::null(),
                    )
                };
            }
        }

        if hr_init < 0 {
            unsafe {
                ((*(*probe_client).lp_vtbl).release)(probe_client);
            }
            let reason = match hr_init {
                AUDCLNT_E_DEVICE_IN_USE => "Endpoint in use by another app (AUDCLNT_E_DEVICE_IN_USE)",
                AUDCLNT_E_EXCLUSIVE_MODE_NOT_ALLOWED => "Exclusive mode disabled in Windows Sound settings",
                _ => "Exclusive Initialize rejected by driver",
            };
            return Err(format!("{reason} (0x{:08X})", hr_init as u32));
        }

        unsafe {
            finalize_wasapi_client_setup(
                probe_client,
                device_id,
                device_name,
                sr,
                ch,
                sample_fmt,
                true,
                format!(
                    "WASAPI Exclusive Low-Latency • {} Hz • {}",
                    sr,
                    sample_fmt.label()
                ),
            )
        }
    }

    unsafe fn try_open_shared_on_device(
        device: *mut IMMDevice,
        device_id: &str,
        device_name: &str,
        fallback_note: Option<&str>,
        desired: &DesiredWasapiConfig,
    ) -> Result<OpenedWasapiSession, String> {
        let client =
            unsafe { activate_audio_client(device) }.map_err(|hr| format!("Activate failed: 0x{:08X}", hr as u32))?;

        let mut mix_fmt_ptr: *mut WAVEFORMATEX = ptr::null_mut();
        let hr_mix = unsafe { ((*(*client).lp_vtbl).get_mix_format)(client, &mut mix_fmt_ptr) };
        if hr_mix < 0 || mix_fmt_ptr.is_null() {
            unsafe {
                ((*(*client).lp_vtbl).release)(client);
            }
            return Err(format!("GetMixFormat failed: 0x{:08X}", hr_mix as u32));
        }

        let Some((sr, ch, sample_fmt)) = (unsafe { parse_waveformat(mix_fmt_ptr) }) else {
            unsafe {
                CoTaskMemFree(mix_fmt_ptr as *const c_void);
                ((*(*client).lp_vtbl).release)(client);
            }
            return Err("Unsupported Shared Mode WAVEFORMATEX".to_string());
        };

        let mut default_period: i64 = 100_000;
        let mut min_period: i64 = 30_000;
        let _ = unsafe {
            ((*(*client).lp_vtbl).get_device_period)(client, &mut default_period, &mut min_period)
        };
        // Event-driven Shared Mode: pass `0` for periodicity and `default_period` (~10 ms) for buffer duration
        let requested_hns = desired
            .desired_buffer_frames
            .map(|f| ((10_000_000.0 * f as f64) / sr.max(8_000) as f64).round() as i64)
            .unwrap_or(0)
            .min(5_000_000);
        let buf_duration_hns = default_period.max(100_000).max(requested_hns);

        let hr_init = unsafe {
            ((*(*client).lp_vtbl).initialize)(
                client,
                AUDCLNT_SHAREMODE_SHARED,
                AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                buf_duration_hns,
                0,
                mix_fmt_ptr,
                ptr::null(),
            )
        };
        unsafe {
            CoTaskMemFree(mix_fmt_ptr as *const c_void);
        }

        if hr_init < 0 {
            unsafe {
                ((*(*client).lp_vtbl).release)(client);
            }
            return Err(format!("Shared Initialize failed: 0x{:08X}", hr_init as u32));
        }

        let msg = match fallback_note {
            Some(note) => format!(
                "WASAPI Shared Event-Driven ({note}) • {} Hz • {}",
                sr,
                sample_fmt.label()
            ),
            None => format!(
                "WASAPI Shared Event-Driven • {} Hz • {}",
                sr,
                sample_fmt.label()
            ),
        };

        unsafe {
            finalize_wasapi_client_setup(client, device_id, device_name, sr, ch, sample_fmt, false, msg)
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn finalize_wasapi_client_setup(
        client: *mut IAudioClient,
        device_id: &str,
        device_name: &str,
        sr: u32,
        ch: u16,
        sample_fmt: WasapiSampleFormat,
        exclusive_active: bool,
        status_message: String,
    ) -> Result<OpenedWasapiSession, String> {
        let event_handle = unsafe { CreateEventW(ptr::null(), 0, 0, ptr::null()) };
        if event_handle.is_null() {
            unsafe {
                ((*(*client).lp_vtbl).release)(client);
            }
            return Err("CreateEventW failed".to_string());
        }

        let hr_ev = unsafe { ((*(*client).lp_vtbl).set_event_handle)(client, event_handle) };
        if hr_ev < 0 {
            unsafe {
                let _ = CloseHandle(event_handle);
                ((*(*client).lp_vtbl).release)(client);
            }
            return Err(format!("SetEventHandle failed: 0x{:08X}", hr_ev as u32));
        }

        let mut buffer_frames: u32 = 0;
        let hr_buf = unsafe { ((*(*client).lp_vtbl).get_buffer_size)(client, &mut buffer_frames) };
        if hr_buf < 0 || buffer_frames == 0 {
            unsafe {
                let _ = CloseHandle(event_handle);
                ((*(*client).lp_vtbl).release)(client);
            }
            return Err(format!("GetBufferSize failed: 0x{:08X}", hr_buf as u32));
        }

        let mut render_client: *mut IAudioRenderClient = ptr::null_mut();
        let hr_rc = unsafe {
            ((*(*client).lp_vtbl).get_service)(
                client,
                &IID_IAUDIORENDERCLIENT,
                &mut render_client as *mut *mut IAudioRenderClient as *mut *mut c_void,
            )
        };
        if hr_rc < 0 || render_client.is_null() {
            unsafe {
                let _ = CloseHandle(event_handle);
                ((*(*client).lp_vtbl).release)(client);
            }
            return Err(format!("GetService(IAudioRenderClient) failed: 0x{:08X}", hr_rc as u32));
        }

        let mut audio_clock: *mut IAudioClock = ptr::null_mut();
        let _ = unsafe {
            ((*(*client).lp_vtbl).get_service)(
                client,
                &IID_IAUDIOCLOCK,
                &mut audio_clock as *mut *mut IAudioClock as *mut *mut c_void,
            )
        };

        // Pre-fill initial hardware buffer with silence (`AUDCLNT_BUFFERFLAGS_SILENT = 0x2`) so DMA starts glitch-free
        let mut init_data: *mut u8 = ptr::null_mut();
        if unsafe { ((*(*render_client).lp_vtbl).get_buffer)(render_client, buffer_frames, &mut init_data) } >= 0 {
            let _ = unsafe {
                ((*(*render_client).lp_vtbl).release_buffer)(render_client, buffer_frames, 0x2)
            };
        }

        let hr_start = unsafe { ((*(*client).lp_vtbl).start)(client) };
        if hr_start < 0 {
            let session = OpenedWasapiSession {
                audio_client: client,
                render_client,
                audio_clock,
                event_handle,
                device_id: device_id.to_string(),
                device_name: device_name.to_string(),
                sample_rate_hz: sr,
                channels: ch,
                buffer_frames,
                sample_format: sample_fmt,
                exclusive_active,
                status_message,
            };
            drop(session);
            return Err(format!("IAudioClient::Start failed: 0x{:08X}", hr_start as u32));
        }

        Ok(OpenedWasapiSession {
            audio_client: client,
            render_client,
            audio_clock,
            event_handle,
            device_id: device_id.to_string(),
            device_name: device_name.to_string(),
            sample_rate_hz: sr,
            channels: ch,
            buffer_frames,
            sample_format: sample_fmt,
            exclusive_active,
            status_message,
        })
    }

    fn open_configured_session(
        desired: &DesiredWasapiConfig,
        engine_sr: u32,
    ) -> Result<OpenedWasapiSession, String> {
        let mut enumerator: *mut IMMDeviceEnumerator = ptr::null_mut();
        let hr = unsafe {
            CoCreateInstance(
                &CLSID_MMDEVICE_ENUMERATOR,
                ptr::null_mut(),
                CLSCTX_ALL,
                &IID_IMMDEVICE_ENUMERATOR,
                &mut enumerator as *mut *mut IMMDeviceEnumerator as *mut *mut c_void,
            )
        };
        if hr < 0 || enumerator.is_null() {
            return Err(format!("CoCreateInstance(MMDeviceEnumerator) failed: 0x{:08X}", hr as u32));
        }

        let mut device: *mut IMMDevice = ptr::null_mut();
        if let Some(ref wanted_id) = desired.device_id {
            let utf16 = string_to_utf16_null(wanted_id);
            let hr_dev =
                unsafe { ((*(*enumerator).lp_vtbl).get_device)(enumerator, utf16.as_ptr(), &mut device) };
            if hr_dev < 0 || device.is_null() {
                device = ptr::null_mut();
            }
        }

        if device.is_null() {
            let mut hr_def = unsafe {
                ((*(*enumerator).lp_vtbl).get_default_audio_endpoint)(
                    enumerator,
                    E_RENDER,
                    E_MULTIMEDIA,
                    &mut device,
                )
            };
            if hr_def < 0 || device.is_null() {
                hr_def = unsafe {
                    ((*(*enumerator).lp_vtbl).get_default_audio_endpoint)(
                        enumerator,
                        E_RENDER,
                        E_CONSOLE,
                        &mut device,
                    )
                };
            }
            if hr_def < 0 || device.is_null() {
                unsafe {
                    ((*(*enumerator).lp_vtbl).release)(enumerator);
                }
                return Err(format!(
                    "No active Windows audio output device found (0x{:08X})",
                    hr_def as u32
                ));
            }
        }

        let (device_id, device_name) = unsafe { query_device_id_and_name(device) };

        // Query shared mix channels as hint
        let mut mix_channels = 2u16;
        if let Ok(probe) = unsafe { activate_audio_client(device) } {
            let mut mix_fmt: *mut WAVEFORMATEX = ptr::null_mut();
            if unsafe { ((*(*probe).lp_vtbl).get_mix_format)(probe, &mut mix_fmt) } >= 0
                && !mix_fmt.is_null()
            {
                if let Some((_, ch, _)) = unsafe { parse_waveformat(mix_fmt) } {
                    mix_channels = ch;
                }
                unsafe {
                    CoTaskMemFree(mix_fmt as *const c_void);
                }
            }
            unsafe {
                ((*(*probe).lp_vtbl).release)(probe);
            }
        }

        let result = if desired.exclusive_mode {
            match unsafe {
                try_open_exclusive_on_device(device, engine_sr, mix_channels, &device_id, &device_name, desired)
            } {
                Ok(sess) => Ok(sess),
                Err(excl_err) => unsafe {
                    try_open_shared_on_device(
                        device,
                        &device_id,
                        &device_name,
                        Some(&format!("Exclusive fallback: {excl_err}")),
                        desired,
                    )
                },
            }
        } else {
            unsafe { try_open_shared_on_device(device, &device_id, &device_name, None, desired) }
        };

        unsafe {
            ((*(*device).lp_vtbl).release)(device);
            ((*(*enumerator).lp_vtbl).release)(enumerator);
        }
        result
    }

    pub fn run_wasapi_rt_thread_windows(
        mut engine: Box<MixerEngine>,
        control: Arc<WasapiSharedControl>,
    ) -> Box<MixerEngine> {
        let _com = match ComInitGuard::enter() {
            Ok(g) => g,
            Err(e) => {
                *control.metadata.write() = (
                    String::new(),
                    "COM Error".to_string(),
                    "None".to_string(),
                    e,
                );
                return engine;
            }
        };

        // Elevate thread to Windows MMCSS "Pro Audio" (`AVRT_PRIORITY_CRITICAL`)
        let pro_audio_guard = ProAudioThreadGuard::enter();
        control
            .mmcss_elevated
            .store(pro_audio_guard.is_elevated(), Ordering::Relaxed);

        let engine_sr = engine.slicer.sample_rate.round().max(8_000.0) as u32;
        control.engine_sample_rate_hz.store(engine_sr, Ordering::Relaxed);

        // Pre-allocate packet scratch buffer (up to 16,384 stereo frames = 85 ms @ 192 kHz) and streaming resampler ONCE
        let max_packet_frames: usize = 16_384;
        let mut packet_scratch: Vec<StereoFrame> = vec![[0.0; 2]; max_packet_frames];
        let mut resampler = WasapiStreamingResampler::new(engine_sr, engine_sr);

        while !control.shutdown.load(Ordering::Acquire) {
            control.reopen_requested.store(false, Ordering::Release);
            let desired = control.desired_config.read().clone();

            match open_configured_session(&desired, engine_sr) {
                Ok(session) => {
                    resampler.reconfigure_rates(engine_sr, session.sample_rate_hz);
                    control.stream_running.store(true, Ordering::Relaxed);
                    control.using_hardware.store(true, Ordering::Relaxed);
                    control
                        .exclusive_active
                        .store(session.exclusive_active, Ordering::Relaxed);
                    control
                        .hw_sample_rate_hz
                        .store(session.sample_rate_hz, Ordering::Relaxed);
                    control
                        .buffer_frames
                        .store(session.buffer_frames, Ordering::Relaxed);
                    control
                        .channels
                        .store(session.channels as u32, Ordering::Relaxed);
                    *control.metadata.write() = (
                        session.device_id.clone(),
                        session.device_name.clone(),
                        session.sample_format.label().to_string(),
                        session.status_message.clone(),
                    );

                    // ====================================================================
                    // STEADY-STATE REAL-TIME EVENT-DRIVEN PACKET LOOP (0 ALLOC, 0 MUTEX)
                    // ====================================================================
                    let mut device_invalidated = false;
                    let mut packets_written: u64 = 0;
                    while !control.shutdown.load(Ordering::Acquire)
                        && !control.reopen_requested.load(Ordering::Acquire)
                    {
                        let wait_rc = unsafe { WaitForSingleObject(session.event_handle as *mut c_void, 100) };
                        if wait_rc == WAIT_TIMEOUT {
                            // Poll clock just in case hardware stalled
                            if !session.audio_clock.is_null() {
                                let mut pos = 0u64;
                                let hr_clk = unsafe {
                                    ((*(*session.audio_clock).lp_vtbl).get_position)(
                                        session.audio_clock,
                                        &mut pos,
                                        ptr::null_mut(),
                                    )
                                };
                                if hr_clk == AUDCLNT_E_DEVICE_INVALIDATED
                                    || hr_clk == AUDCLNT_E_RESOURCES_INVALIDATED
                                {
                                    device_invalidated = true;
                                    break;
                                }
                            }
                            continue;
                        }
                        if wait_rc != WAIT_OBJECT_0 && wait_rc != INFINITE {
                            device_invalidated = true;
                            break;
                        }

                        let frames_to_write: u32 = if session.exclusive_active {
                            session.buffer_frames
                        } else {
                            let mut padding: u32 = 0;
                            let hr_pad = unsafe {
                                ((*(*session.audio_client).lp_vtbl).get_current_padding)(
                                    session.audio_client,
                                    &mut padding,
                                )
                            };
                            if hr_pad == AUDCLNT_E_DEVICE_INVALIDATED
                                || hr_pad == AUDCLNT_E_RESOURCES_INVALIDATED
                            {
                                device_invalidated = true;
                                break;
                            }
                            if hr_pad < 0 {
                                control.xrun_count.fetch_add(1, Ordering::Relaxed);
                                continue;
                            }
                            if padding == 0 && session.buffer_frames > 0 && packets_written > 2 {
                                // The endpoint consumed everything we queued before we were woken
                                // again: the device played silence (render underrun / glitch).
                                control.xrun_count.fetch_add(1, Ordering::Relaxed);
                            }
                            session.buffer_frames.saturating_sub(padding)
                        };

                        if frames_to_write == 0 {
                            continue;
                        }

                        let clamped_frames = (frames_to_write as usize).min(max_packet_frames);
                        let mut hw_data_ptr: *mut u8 = ptr::null_mut();
                        let hr_get = unsafe {
                            ((*(*session.render_client).lp_vtbl).get_buffer)(
                                session.render_client,
                                clamped_frames as u32,
                                &mut hw_data_ptr,
                            )
                        };
                        if hr_get == AUDCLNT_E_DEVICE_INVALIDATED
                            || hr_get == AUDCLNT_E_RESOURCES_INVALIDATED
                        {
                            device_invalidated = true;
                            break;
                        }
                        if hr_get < 0 || hw_data_ptr.is_null() {
                            control.xrun_count.fetch_add(1, Ordering::Relaxed);
                            continue;
                        }

                        let target_vol = if control.speaker_muted.load(Ordering::Relaxed) {
                            0.0_f32
                        } else {
                            f32::from_bits(control.speaker_volume_bits.load(Ordering::Relaxed))
                        };

                        resampler.render_hardware_frames(
                            &mut engine,
                            &mut packet_scratch[..clamped_frames],
                            target_vol,
                            &control.avg_block_render_us,
                            &control.rendered_blocks,
                        );

                        unsafe {
                            pack_stereo_frames_to_hardware_buffer(
                                &packet_scratch[..clamped_frames],
                                hw_data_ptr,
                                session.channels as usize,
                                session.sample_format,
                            );
                        }

                        let hr_rel = unsafe {
                            ((*(*session.render_client).lp_vtbl).release_buffer)(
                                session.render_client,
                                clamped_frames as u32,
                                0,
                            )
                        };
                        if hr_rel == AUDCLNT_E_DEVICE_INVALIDATED
                            || hr_rel == AUDCLNT_E_RESOURCES_INVALIDATED
                        {
                            device_invalidated = true;
                            break;
                        }
                        if hr_rel < 0 {
                            control.xrun_count.fetch_add(1, Ordering::Relaxed);
                        } else {
                            packets_written += 1;
                        }
                    }

                    drop(session);
                    if device_invalidated && !control.shutdown.load(Ordering::Acquire) {
                        control.hotplug_recoveries.fetch_add(1, Ordering::Relaxed);
                        // If a specific unplugged device was selected, clear to default if it disappeared
                        thread::sleep(Duration::from_millis(80));
                    }
                }
                Err(open_err) => {
                    // Headless CI / zero-endpoint fallback: keep telemetry & command draining alive
                    // until an audio device is plugged in!
                    control.stream_running.store(true, Ordering::Relaxed);
                    control.using_hardware.store(false, Ordering::Relaxed);
                    control.exclusive_active.store(false, Ordering::Relaxed);
                    *control.metadata.write() = (
                        String::new(),
                        "No Active Audio Endpoint (Headless Fallback)".to_string(),
                        "Float32".to_string(),
                        open_err,
                    );

                    let quantum_dur = Duration::from_secs_f64(
                        (WASAPI_ENGINE_RENDER_QUANTUM as f64) / (engine_sr as f64),
                    );
                    // Retry opening WASAPI hardware every ~500 ms (~90 blocks)
                    for _ in 0..90 {
                        if control.shutdown.load(Ordering::Acquire)
                            || control.reopen_requested.load(Ordering::Acquire)
                        {
                            break;
                        }
                        let t0 = Instant::now();
                        engine.process_audio_block(&mut packet_scratch[..WASAPI_ENGINE_RENDER_QUANTUM]);
                        let elapsed = t0.elapsed();
                        control
                            .avg_block_render_us
                            .store(elapsed.as_micros() as u32, Ordering::Relaxed);
                        control.rendered_blocks.fetch_add(1, Ordering::Relaxed);
                        if let Some(rem) = quantum_dur.checked_sub(elapsed) {
                            thread::sleep(rem);
                        }
                    }
                }
            }
        }

        control.stream_running.store(false, Ordering::Relaxed);
        engine
    }
}

/// Enumerates all active Windows Core Audio WASAPI render endpoints (`IMMDevice`s).
pub fn enumerate_wasapi_output_devices() -> Result<Vec<WasapiEndpointInfo>, String> {
    #[cfg(target_os = "windows")]
    {
        win32_wasapi::enumerate_endpoints_impl()
    }
    #[cfg(not(target_os = "windows"))]
    {
        Ok(vec![WasapiEndpointInfo {
            id: "default".to_string(),
            friendly_name: "Default System Output".to_string(),
            is_default: true,
            mix_sample_rate_hz: 48_000,
            channels: 2,
            mix_channels: 2,
            mix_format_label: "Float32".to_string(),
        }])
    }
}

fn run_wasapi_rt_thread(
    #[allow(unused_mut)] mut engine: Box<MixerEngine>,
    control: Arc<WasapiSharedControl>,
) -> Box<MixerEngine> {
    #[cfg(target_os = "windows")]
    {
        win32_wasapi::run_wasapi_rt_thread_windows(engine, control)
    }
    #[cfg(not(target_os = "windows"))]
    {
        let sr = engine.slicer.sample_rate.max(8_000.0);
        let quantum_dur = Duration::from_secs_f64((WASAPI_ENGINE_RENDER_QUANTUM as f64) / sr);
        let mut scratch = vec![[0.0f32; 2]; WASAPI_ENGINE_RENDER_QUANTUM];
        control.stream_running.store(true, Ordering::Relaxed);
        while !control.shutdown.load(Ordering::Acquire) {
            let t0 = Instant::now();
            engine.process_audio_block(&mut scratch);
            let elapsed = t0.elapsed();
            control
                .avg_block_render_us
                .store(elapsed.as_micros() as u32, Ordering::Relaxed);
            control.rendered_blocks.fetch_add(1, Ordering::Relaxed);
            if let Some(rem) = quantum_dur.checked_sub(elapsed) {
                thread::sleep(rem);
            }
        }
        control.stream_running.store(false, Ordering::Relaxed);
        engine
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SineNode {
        freq: f64,
        phase: f64,
        sr: f64,
    }
    impl crate::UnifiedDspNode for SineNode {
        fn name(&self) -> &'static str {
            "SineProbe"
        }
        fn latency_samples(&self) -> usize {
            0
        }
        fn prepare(&mut self, sr: f64, _max: usize) {
            self.sr = sr;
        }
        fn reset_state(&mut self) {}
        fn set_parameter_normalized(&mut self, _p: u32, _v: f32) {}
        fn process_slice(
            &mut self,
            _ctx: &crate::ProcessContext,
            io: &mut [StereoFrame],
            _sc: Option<&[StereoFrame]>,
            _aux: &mut [&mut [StereoFrame]],
        ) {
            for f in io.iter_mut() {
                let v = (0.5 * self.phase.sin()) as f32;
                self.phase += 2.0 * std::f64::consts::PI * self.freq / self.sr;
                *f = [v, v];
            }
        }
    }

    /// Least-squares sine fit at a known frequency; returns (amplitude, residual RMS).
    fn fit_sine(x: &[f32], freq: f64, sr: f64) -> (f64, f64) {
        let (mut ss, mut sc, mut cc, mut xs, mut xc) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for (n, &v) in x.iter().enumerate() {
            let w = 2.0 * std::f64::consts::PI * freq * (n as f64) / sr;
            let (s, c) = (w.sin(), w.cos());
            ss += s * s;
            sc += s * c;
            cc += c * c;
            xs += (v as f64) * s;
            xc += (v as f64) * c;
        }
        let det = ss * cc - sc * sc;
        let a = (xs * cc - xc * sc) / det;
        let b = (xc * ss - xs * sc) / det;
        let mut res = 0.0;
        for (n, &v) in x.iter().enumerate() {
            let w = 2.0 * std::f64::consts::PI * freq * (n as f64) / sr;
            let e = (v as f64) - (a * w.sin() + b * w.cos());
            res += e * e;
        }
        ((a * a + b * b).sqrt(), (res / (x.len() as f64)).sqrt())
    }

    /// 48 kHz engine -> 44.1 kHz endpoint: flat passband to 18 kHz and >= 85 dB SINAD.
    #[test]
    fn test_output_resampler_48k_to_44k1_passband_and_sinad() {
        for &freq in &[1_000.0f64, 18_000.0f64] {
            let (mut engine, _cmd, _tel) = crate::MixerEngine::new(48_000.0, 120.0, 0);
            engine.channels.push(crate::ChannelGenerator::new(
                1,
                1,
                Box::new(SineNode { freq, phase: 0.0, sr: 48_000.0 }),
            ));
            engine.tracks[1].add_send(crate::MASTER_TRACK_INDEX, 1.0, false);
            engine.rebuild_graph().unwrap();
            let control = WasapiSharedControl::new(48_000, None, false);
            let mut rs = WasapiStreamingResampler::new(48_000, 44_100);
            let mut out = vec![[0.0f32; 2]; 44_100];
            for chunk in out.chunks_mut(441) {
                rs.render_hardware_frames(
                    &mut engine,
                    chunk,
                    1.0,
                    &control.avg_block_render_us,
                    &control.rendered_blocks,
                );
            }
            let tail: Vec<f32> = out[22_050..].iter().map(|f| f[0]).collect();
            let (amp, res) = fit_sine(&tail, freq, 44_100.0);
            let gain_db = 20.0 * (amp / 0.5).log10();
            let sinad_db = 20.0 * (amp / std::f64::consts::SQRT_2 / res).log10();
            assert!(gain_db.abs() < 0.1, "{freq} Hz passband gain {gain_db:.3} dB");
            assert!(sinad_db > 85.0, "{freq} Hz SINAD {sinad_db:.1} dB");
        }
    }

    #[test]
    fn test_wasapi_format_packing_roundtrip_and_multichannel() {
        let frames: [StereoFrame; 4] = [
            [0.5, -0.5],
            [1.0, -1.0],
            [0.0, 0.25],
            [-0.75, 0.75],
        ];

        // 1. Float32 6-channel (5.1) packing places stereo in ch 0,1 and zeros ch 2..6
        let mut buf_f32 = vec![0u8; 4 * 6 * 4];
        unsafe {
            pack_stereo_frames_to_hardware_buffer(
                &frames,
                buf_f32.as_mut_ptr(),
                6,
                WasapiSampleFormat::Float32,
            );
        }
        let words = unsafe { std::slice::from_raw_parts(buf_f32.as_ptr() as *const f32, 24) };
        assert!((words[0] - 0.5).abs() < 1e-6);
        assert!((words[1] + 0.5).abs() < 1e-6);
        assert_eq!(words[2], 0.0);
        assert_eq!(words[5], 0.0);

        // 2. Int24Packed 2-channel
        let mut buf_i24 = vec![0u8; 4 * 2 * 3];
        unsafe {
            pack_stereo_frames_to_hardware_buffer(
                &frames,
                buf_i24.as_mut_ptr(),
                2,
                WasapiSampleFormat::Int24Packed,
            );
        }
        let max_pos = i32::from_le_bytes([buf_i24[6], buf_i24[7], buf_i24[8], 0]);
        assert_eq!(max_pos, 8_388_607);
    }

    #[test]
    fn test_wasapi_endpoint_enumeration_smoke() {
        let res = enumerate_wasapi_output_devices();
        assert!(res.is_ok(), "WASAPI endpoint enumeration failed: {:?}", res);
    }
}
