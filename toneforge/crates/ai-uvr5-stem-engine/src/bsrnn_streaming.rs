//! Real-Time Low-Latency (`120 ms` = `5,760` samples @ 48 kHz) 4-Stem BSRNN Separation Pipeline
//! (`RealTimeBsrnnStemNode`, `ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx`).
//!
//! Architecture (`03_ozone12_and_uvr5_dsp_ai.md` & `05_master_rust_svelte_blueprint.md`):
//! - 4 Stems: `[Vocals, Drums, Bass, Other]`
//! - STFT window `N_fft = 2048` (`F = 1025` bins), hop `H = 512` (`10.67 ms` @ 48 kHz)
//! - 13 Psychoacoustic Bark-like subbands (`BSRNN_SUBBAND_BINS`, sum = `1025`)
//! - Recurrent hidden state shape `[2, 2, 2, 1, 13, 128]` (`13,312` floats)
//! - Lock-Free `rtrb::RingBuffer` SPSC queues between the Real-Time Audio Thread and the
//!   BSRNN separation worker (`0` heap allocations on the audio thread)
//! - Unity-Identity Additive Mixing (`DSP::OzoneStemEQ` `0x1857397B0`):
//!   `Y[n] = X_delayed[n] + sum_{s=0..3} (G_s * H_s(S_s[n]) - S_s[n])`
//!   Guarantees bit-exact transparency (`Y[n] === X_delayed[n]`) when all stem gains/EQ are `0.0 dB`.

use daw_core_graph::{ApdcDelayLine, ProcessContext, StereoFrame, UnifiedDspNode, VisualTelemetryFrame};
use rtrb::{Consumer, Producer, RingBuffer};
use std::path::Path;
use std::sync::Arc;

/// Exact lookahead latency in seconds (`120 ms`).
pub const BSRNN_REALTIME_LATENCY_SEC: f64 = 0.120;

/// Standard latency in samples at `48,000 Hz` (`0.120 * 48000 = 5,760` samples).
pub const BSRNN_LATENCY_SAMPLES_48K: usize = 5760;

/// STFT FFT size (`2048` samples -> `1025` complex frequency bins).
pub const BSRNN_N_FFT: usize = 2048;
pub const BSRNN_FREQ_BINS: usize = BSRNN_N_FFT / 2 + 1; // 1025

/// STFT hop size (`512` samples = `10.67 ms` @ 48 kHz).
pub const BSRNN_HOP_SIZE: usize = 512;

/// Number of separated stems (`[Vocals, Drums, Bass, Other]`).
pub const BSRNN_STEM_COUNT: usize = 4;

/// Exact 13 subband widths in FFT bins from `/band_split_module/Split` (`onnx::Split_335`, sum = `1025`).
pub const BSRNN_SUBBAND_BINS: [usize; 13] = [
    5, 6, 9, 12, 16, 22, 30, 41, 55, 74, 100, 135, 520,
];

/// Recurrent state tensor shape `[batch_lr=2, num_time_blocks=2, hc=2, dir=1, subbands=13, hidden=128]`.
pub const BSRNN_HIDDEN_STATE_SHAPE: [usize; 6] = [2, 2, 2, 1, 13, 128];
pub const BSRNN_HIDDEN_STATE_LEN: usize = 2 * 2 * 2 * 1 * 13 * 128; // 13,312 floats

/// Candidate paths for `ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx` (`7,230,143` bytes).
pub const DEFAULT_BSRNN_ONNX_PATHS: [&str; 4] = [
    r"f:\FL studio\toneforge\assets\ozone12\models\ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx",
    r"f:\FL studio\toneforge\assets\models\ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx",
    r"assets/ozone12/models/ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx",
    r"assets/models/ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx",
];

/// Candidate paths for `delimit_exp009_ep080_fixed2.onnx` (`5,580,442` bytes).
pub const DEFAULT_DELIMIT_ONNX_PATHS: [&str; 4] = [
    r"f:\FL studio\toneforge\assets\ozone12\models\delimit_exp009_ep080_fixed2.onnx",
    r"f:\FL studio\toneforge\assets\models\delimit_exp009_ep080_fixed2.onnx",
    r"assets/ozone12/models/delimit_exp009_ep080_fixed2.onnx",
    r"assets/models/delimit_exp009_ep080_fixed2.onnx",
];

/// Stem index identifier for Ozone 12 4-Stem separation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum StemKind {
    Vocals = 0,
    Drums = 1,
    Bass = 2,
    Other = 3,
}

/// Metadata extracted from a verified ONNX model file (`ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx`).
#[derive(Clone, Debug)]
pub struct OnnxModelDescriptor {
    pub file_size_bytes: usize,
    pub ir_version: u64,
    pub has_input_tensor: bool,
    pub has_hidden_state_io: bool,
    pub weight_fingerprint_f32: [f32; 13],
}

impl OnnxModelDescriptor {
    /// Inspects and validates an ONNX protobuf binary (`ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx`
    /// or `delimit_exp009_ep080_fixed2.onnx`) without external C++ dependencies.
    pub fn from_onnx_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() < 64 {
            return Err("ONNX file too small (< 64 bytes)");
        }
        // ONNX ModelProto starts with field 1 (`ir_version`, varint tag `0x08`).
        if bytes[0] != 0x08 {
            return Err("Invalid ONNX ModelProto header (expected tag 0x08)");
        }
        let ir_version = bytes[1] as u64;
        let has_input = Self::contains_ascii(bytes, b"input") || Self::contains_ascii(bytes, b"mix");
        let has_hidden = Self::contains_ascii(bytes, b"h_")
            || Self::contains_ascii(bytes, b"hidden")
            || Self::contains_ascii(bytes, b"GRU")
            || Self::contains_ascii(bytes, b"LSTM");

        // Extract deterministic 13-subband calibration fingerprint from model weight payload
        let mut fp = [1.0f32; 13];
        let step = (bytes.len() / 16).max(4);
        for b in 0..13 {
            let off = ((b + 1) * step).min(bytes.len() - 4);
            let raw_u32 = u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
            let norm = ((raw_u32 & 0xFFFF) as f32) / 65535.0;
            fp[b] = 0.85 + 0.30 * norm;
        }

        Ok(Self {
            file_size_bytes: bytes.len(),
            ir_version,
            has_input_tensor: has_input,
            has_hidden_state_io: has_hidden,
            weight_fingerprint_f32: fp,
        })
    }

    pub fn from_onnx_file(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|e| format!("Failed to read ONNX file {:?}: {}", path, e))?;
        Self::from_onnx_bytes(&bytes).map_err(|e| e.to_string())
    }

    fn contains_ascii(haystack: &[u8], needle: &[u8]) -> bool {
        if needle.is_empty() || haystack.len() < needle.len() {
            return false;
        }
        haystack.windows(needle.len()).any(|w| w == needle)
    }
}

/// Fixed-size input hop (`512` stereo frames) sent over the lock-free `rtrb::RingBuffer`
/// from the Real-Time Audio Thread to the BSRNN worker.
#[derive(Clone, Copy)]
pub struct BsrnnInputHop {
    pub epoch: u64,
    pub seq: u64,
    pub frames: [StereoFrame; BSRNN_HOP_SIZE],
}

impl Default for BsrnnInputHop {
    fn default() -> Self {
        Self {
            epoch: 0,
            seq: 0,
            frames: [[0.0; 2]; BSRNN_HOP_SIZE],
        }
    }
}

/// Fixed-size output hop (`4` stems x `512` stereo frames) returned over the lock-free
/// `rtrb::RingBuffer` from the BSRNN worker to the Real-Time Audio Thread.
#[derive(Clone, Copy)]
pub struct BsrnnOutputHop {
    pub epoch: u64,
    pub seq: u64,
    pub stems: [[StereoFrame; BSRNN_HOP_SIZE]; BSRNN_STEM_COUNT],
}

impl Default for BsrnnOutputHop {
    fn default() -> Self {
        Self {
            epoch: 0,
            seq: 0,
            stems: [[[0.0; 2]; BSRNN_HOP_SIZE]; BSRNN_STEM_COUNT],
        }
    }
}

/// Number of pre-seeded hops (`4 * 512 = 2,048` samples = `42.7 ms` @ 48 kHz) cushioning the
/// asynchronous ONNX GPU worker thread inside the `5,760`-sample (`120 ms`) APDC latency budget.
const ASYNC_GPU_CUSHION_HOPS: usize = 4;

/// Peak level below which an input frame counts as digital silence (`1e-6` = `-120 dBFS`).
const IDLE_SILENCE_PEAK: f32 = 1.0e-6;

/// `|applied delta gain|` below which a unity (0 dB / no solo / no mute) rebalance counts as settled.
const IDLE_DELTA_EPS: f32 = 1.0e-4;

/// Algorithmic delay of `OzoneBsrnnOnnxSeparator::process_hop` in hops: output hop `k` carries the
/// stems of input hop `k - 5` (`2048` STFT/OLA + `1024` lookahead = `3072` = `6` hops, minus the
/// hop-framing hop). See `separate_4stems_offline` (`algo_delay = 5 * BSRNN_HOP_SIZE`).
const SEPARATOR_ALGO_DELAY_HOPS: usize =
    (BSRNN_N_FFT + crate::bsrnn_onnx_gpu::OZONE_BSRNN_LOOKAHEAD_SAMPLES) / BSRNN_HOP_SIZE - 1;

/// Hops of true input history the separator must see before the first stem hop that is mixed after
/// an idle restart, so every `2048`-sample analysis frame overlapping that hop is complete
/// (`2048 / 512 - 1 = 3`). Without it the first frames would see a fake all-zero history (hard edge).
const RESTART_WARMUP_HOPS: usize = BSRNN_N_FFT / BSRNN_HOP_SIZE - 1;

/// History hops (`8 * 512 = 4,096` samples) re-fed to the freshly reset separator when the pipeline
/// restarts after idle while the input was not silent.
const RESTART_PRIME_HOPS: usize = SEPARATOR_ALGO_DELAY_HOPS + RESTART_WARMUP_HOPS;

/// Raw input history ring (power of two) used to prime the separator on an idle restart.
const INPUT_HISTORY_LEN: usize = 8192;
const _: () = assert!(RESTART_PRIME_HOPS * BSRNN_HOP_SIZE <= INPUT_HISTORY_LEN);
const _: () = assert!(INPUT_HISTORY_LEN.is_power_of_two());

/// GPU worker idle backoff: after this long without input hops the poll sleep grows progressively.
const WORKER_IDLE_BACKOFF_AFTER: std::time::Duration = std::time::Duration::from_millis(100);
const WORKER_POLL_SLEEP_US: u64 = 250;
/// Upper bound of the idle poll sleep (`2 ms`, far inside the `42.7 ms` async cushion).
const WORKER_MAX_IDLE_SLEEP_US: u64 = 2_000;

struct AsyncOnnxWorkerHandle {
    stop_flag: Arc<std::sync::atomic::AtomicBool>,
    worker_epoch: Arc<std::sync::atomic::AtomicU64>,
    join_handle: Option<std::thread::JoinHandle<()>>,
    tier: crate::onnx_gpu_runtime::GpuExecutionTier,
}

impl Drop for AsyncOnnxWorkerHandle {
    fn drop(&mut self) {
        self.stop_flag
            .store(true, std::sync::atomic::Ordering::Release);
        if let Some(jh) = self.join_handle.take() {
            let _ = jh.join();
        }
    }
}

/// Real-Time `120 ms` (`5,760` samples @ 48 kHz) Unity-Identity Additive 4-Stem BSRNN Node
/// (`RealTimeBsrnnStemNode` / `RealtimeBsrnnStemNode`, `DSP::OzoneStemEQ` at `0x1857397B0`).
///
/// Uses lock-free `rtrb::RingBuffer` SPSC queues to decouple hop framing from sample-by-sample
/// audio processing, guaranteeing **0 heap allocations** on the real-time audio thread in both
/// `DeterministicLockstep` mode and `AsyncOnnxGpuWorker` mode (`enable_onnx_gpu_worker()`).
pub struct RealTimeBsrnnStemNode {
    /// Per-stem gain in dB (`[-24.0, +12.0]` dB) for `[Vocals, Drums, Bass, Other]`.
    pub stem_gains_db: [f32; BSRNN_STEM_COUNT],
    /// Per-stem solo flags.
    pub stem_solo: [bool; BSRNN_STEM_COUNT],
    /// Per-stem mute flags.
    pub stem_mute: [bool; BSRNN_STEM_COUNT],
    /// Per-stem RMS telemetry in dB (`[Vocals, Drums, Bass, Other]`).
    pub stem_rms_db: [f32; BSRNN_STEM_COUNT],

    sample_rate: f64,
    latency_samples: usize,
    /// Exact `5,760`-sample (`120 ms` @ 48 kHz) dry reference delay line for Unity-Identity mixing.
    delay_line: ApdcDelayLine,
    /// Sample-accurate alignment delay lines aligning separated stems to `latency_samples`.
    stem_delay_lines: [ApdcDelayLine; BSRNN_STEM_COUNT],

    /// Lock-free SPSC ring buffers (`rtrb`) for input hops (`Audio -> Worker`) and output hops (`Worker -> Audio`).
    in_tx: Producer<BsrnnInputHop>,
    in_rx: Option<Consumer<BsrnnInputHop>>,
    out_tx: Option<Producer<BsrnnOutputHop>>,
    out_rx: Consumer<BsrnnOutputHop>,

    /// Pre-allocated staging hop on the audio side (`512` frames).
    staging_in_hop: BsrnnInputHop,
    staging_in_pos: usize,
    current_out_hop: BsrnnOutputHop,
    current_out_pos: usize,
    pending_out_hop: Option<BsrnnOutputHop>,

    /// Monotonic reset epoch & hop sequence counters preserving exact phase lock across resets and underflows.
    active_epoch: u64,
    next_in_seq: u64,
    expected_out_seq: u64,
    cushion_hops_remaining: usize,

    /// Synchronous ONNX separator serviced inline (offline render / deterministic verification).
    /// Never used on the live real-time thread, which runs the asynchronous GPU worker instead.
    lockstep_onnx: Option<Box<crate::bsrnn_onnx_gpu::OzoneBsrnnOnnxSeparator>>,
    /// Hops whose separated stems were not ready in time (GPU underflow); rebalance fades out for them.
    pub underflow_hops: u64,
    /// `1.0` while separated stems are flowing, ramps to `0.0` on underflow (de-clicked).
    stems_available_gain: f32,
    /// Per-stem rebalance delta actually applied (one-pole ramp towards the target, de-zippered).
    applied_delta_gains: [f32; BSRNN_STEM_COUNT],
    current_hop_valid: bool,
    /// Optional dedicated ONNX Runtime GPU worker thread spawned strictly outside `process_slice()`.
    async_gpu_worker: Option<AsyncOnnxWorkerHandle>,

    // --- Idle gating: the separator receives NO hops while the node is idle -------------------
    /// `true` while hop submission is gated (unity bypass settled, or long digital silence).
    idle: bool,
    /// Whether a settled unity bypass gates the separator (default: on for the async GPU worker,
    /// off for lockstep/offline so renders keep reacting instantly to automation leaving 0 dB).
    unity_idle_gating: bool,
    /// Consecutive input frames with peak `< IDLE_SILENCE_PEAK` (saturating).
    silent_run: usize,
    /// Raw input history (`INPUT_HISTORY_LEN` frames) used to prime the separator on restart.
    input_history: Box<[StereoFrame]>,
    history_pos: usize,
    /// Whether the zero cushion hops after a (re)start count as valid stems (silent history).
    cushion_hops_valid: bool,
    /// Samples during which `stems_available_gain` is held at 0 after a primed restart, until the
    /// first valid stems have travelled through `stem_delay_lines` (no un-faded step).
    avail_hold_samples: usize,
    /// Input hops successfully submitted to the separator (lockstep or async GPU worker).
    pushed_hops: u64,
    #[cfg(test)]
    test_mock: Option<Box<tests::MockSeparator>>,
}

/// Alias preserving backwards compatibility with Phase 1/2 naming.
pub type RealtimeBsrnnStemNode = RealTimeBsrnnStemNode;

impl Default for RealTimeBsrnnStemNode {
    fn default() -> Self {
        Self::new(48_000.0)
    }
}

impl RealTimeBsrnnStemNode {
    pub fn new(sample_rate: f64) -> Self {
        let sr = sample_rate.max(8000.0);
        let lat = (BSRNN_REALTIME_LATENCY_SEC * sr).round() as usize;
        let cap = lat.max(16384);
        let mut delay_line = ApdcDelayLine::with_capacity(cap);
        delay_line.set_delay_samples(lat);

        let stem_align_delay = lat.saturating_sub(BSRNN_N_FFT);
        let stem_delay_lines = std::array::from_fn(|_| {
            let mut dl = ApdcDelayLine::with_capacity(cap);
            dl.set_delay_samples(stem_align_delay);
            dl
        });

        let (in_tx, in_rx) = RingBuffer::<BsrnnInputHop>::new(32);
        let (out_tx, out_rx) = RingBuffer::<BsrnnOutputHop>::new(32);

        Self {
            stem_gains_db: [0.0; BSRNN_STEM_COUNT],
            stem_solo: [false; BSRNN_STEM_COUNT],
            stem_mute: [false; BSRNN_STEM_COUNT],
            stem_rms_db: [-60.0; BSRNN_STEM_COUNT],
            sample_rate: sr,
            latency_samples: lat,
            delay_line,
            stem_delay_lines,
            in_tx,
            in_rx: Some(in_rx),
            out_tx: Some(out_tx),
            out_rx,
            staging_in_hop: BsrnnInputHop::default(),
            staging_in_pos: 0,
            current_out_hop: BsrnnOutputHop::default(),
            current_out_pos: 0,
            pending_out_hop: None,
            active_epoch: 1,
            next_in_seq: 0,
            expected_out_seq: 0,
            cushion_hops_remaining: 0,
            lockstep_onnx: None,
            underflow_hops: 0,
            stems_available_gain: 0.0,
            applied_delta_gains: [0.0; BSRNN_STEM_COUNT],
            current_hop_valid: false,
            async_gpu_worker: None,
            idle: false,
            unity_idle_gating: false,
            // Freshly constructed: every delay line / separator history is zero == silence.
            silent_run: usize::MAX,
            input_history: vec![[0.0f32; 2]; INPUT_HISTORY_LEN].into_boxed_slice(),
            history_pos: 0,
            cushion_hops_valid: false,
            avail_hold_samples: 0,
            pushed_hops: 0,
            #[cfg(test)]
            test_mock: None,
        }
    }

    /// Number of input hops submitted to the separator so far (diagnostics / idle-gating tests).
    #[inline]
    pub fn pushed_hops(&self) -> u64 {
        self.pushed_hops
    }

    /// `true` while separator hop submission is gated (no GPU inference).
    #[inline]
    pub fn is_idle_gated(&self) -> bool {
        self.idle
    }

    /// Enables/disables gating the separator while a unity bypass has settled. Enabled by default
    /// for the async GPU worker, disabled by default for lockstep (offline render). Call outside the
    /// real-time thread or between `process_slice` calls.
    #[inline]
    pub fn set_unity_idle_gating(&mut self, enabled: bool) {
        self.unity_idle_gating = enabled;
    }

    #[inline]
    pub fn unity_idle_gating(&self) -> bool {
        self.unity_idle_gating
    }

    /// Constructs a `RealTimeBsrnnStemNode` and immediately activates the dedicated ONNX GPU
    /// background worker thread (`ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx`) outside the RT thread.
    pub fn new_with_onnx_gpu(sample_rate: f64) -> Result<Self, String> {
        let mut node = Self::new(sample_rate);
        node.enable_onnx_gpu_worker()?;
        Ok(node)
    }

    /// Constructs a node that runs the real `ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx` separator
    /// synchronously inside `process_slice` (offline render and deterministic verification only).
    pub fn new_with_onnx_lockstep(sample_rate: f64) -> Result<Self, String> {
        let mut node = Self::new(sample_rate);
        node.enable_onnx_lockstep()?;
        Ok(node)
    }

    /// Loads the ONNX separator for inline (lockstep) servicing. MUST NOT be used on the live
    /// real-time thread (ONNX Runtime inference allocates); offline render threads only.
    pub fn enable_onnx_lockstep(
        &mut self,
    ) -> Result<crate::onnx_gpu_runtime::GpuExecutionTier, String> {
        if self.async_gpu_worker.is_some() {
            return Err("BSRNN async GPU worker already active".to_string());
        }
        let mut sep = crate::bsrnn_onnx_gpu::OzoneBsrnnOnnxSeparator::from_default_assets()?;
        let tier = sep.active_tier();
        let mut warm_out = BsrnnOutputHop::default();
        let _ = sep.process_hop(&BsrnnInputHop::default(), &mut warm_out);
        sep.reset();
        self.lockstep_onnx = Some(Box::new(sep));
        // Offline render: never delay automation leaving 0 dB (silence gating still applies).
        self.unity_idle_gating = false;
        let stem_align_delay = self.latency_samples.saturating_sub(self.current_pre_stem_delay());
        for dl in &mut self.stem_delay_lines {
            dl.set_delay_samples(stem_align_delay);
        }
        self.reset_state();
        Ok(tier)
    }

    /// `true` when a real separator (async GPU worker or lockstep ONNX) feeds the stems. Without one
    /// the node is an exact latency-compensated pass-through and stem rebalancing is unavailable.
    #[inline]
    pub fn separation_available(&self) -> bool {
        self.async_gpu_worker.is_some() || self.lockstep_onnx.is_some() || self.has_test_mock()
    }

    /// Spawns a dedicated background worker thread running `OzoneBsrnnOnnxSeparator` on RTX 4080
    /// (`CUDAExecutionProvider` -> `DirectML` -> `CPU`), communicating with `process_slice()`
    /// exclusively over lock-free `rtrb::RingBuffer` queues (**0 allocations on the RT thread**).
    ///
    /// MUST ONLY be called outside the Real-Time audio callback thread!
    pub fn enable_onnx_gpu_worker(
        &mut self,
    ) -> Result<crate::onnx_gpu_runtime::GpuExecutionTier, String> {
        if let Some(existing) = &self.async_gpu_worker {
            return Ok(existing.tier);
        }

        let mut onnx_sep = crate::bsrnn_onnx_gpu::OzoneBsrnnOnnxSeparator::from_default_assets()?;
        let tier = onnx_sep.active_tier();

        let mut in_rx = self
            .in_rx
            .take()
            .ok_or_else(|| "Input SPSC consumer already taken".to_string())?;
        let mut out_tx = self
            .out_tx
            .take()
            .ok_or_else(|| "Output SPSC producer already taken".to_string())?;

        while in_rx.pop().is_ok() {}
        while self.out_rx.pop().is_ok() {}
        self.pending_out_hop = None;
        self.active_epoch = self.active_epoch.wrapping_add(1);
        self.next_in_seq = 0;
        self.expected_out_seq = 0;
        self.staging_in_pos = 0;
        self.current_out_hop = BsrnnOutputHop::default();
        self.current_out_pos = 0;
        self.cushion_hops_remaining = ASYNC_GPU_CUSHION_HOPS;
        self.delay_line.reset();
        self.current_hop_valid = false;
        self.stems_available_gain = 0.0;
        self.idle = false;
        self.unity_idle_gating = true;
        self.silent_run = usize::MAX;
        self.input_history.fill([0.0; 2]);
        self.history_pos = 0;
        self.cushion_hops_valid = false;
        self.avail_hold_samples = 0;

        // Update stem alignment delay to account for 2-hop ONNX lookahead + async cushion:
        // Total pre-stem delay = 2048 (STFT) + 1024 (2-hop lookahead) + 1024 (2-hop cushion) = 4,096 samples.
        let pre_delay = BSRNN_N_FFT
            + crate::bsrnn_onnx_gpu::OZONE_BSRNN_LOOKAHEAD_SAMPLES
            + ASYNC_GPU_CUSHION_HOPS * BSRNN_HOP_SIZE;
        let stem_align_delay = self.latency_samples.saturating_sub(pre_delay);
        for dl in &mut self.stem_delay_lines {
            dl.set_delay_samples(stem_align_delay);
            dl.reset();
        }

        let initial_epoch = self.active_epoch;
        let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let worker_epoch = Arc::new(std::sync::atomic::AtomicU64::new(initial_epoch));
        let ready_barrier = Arc::new(std::sync::Barrier::new(2));
        let stop_clone = Arc::clone(&stop_flag);
        let epoch_clone = Arc::clone(&worker_epoch);
        let ready_clone = Arc::clone(&ready_barrier);

        let join_handle = std::thread::Builder::new()
            .name("toneforge-bsrnn-onnx-gpu".to_string())
            .stack_size(8 * 1024 * 1024)
            .spawn(move || {
                let mut out_hop = BsrnnOutputHop::default();
                // Pre-warm thread-local CUDA context & cuDNN handles on this OS worker thread
                // outside real-time playback so hop #0 never suffers a cold-start kernel stall.
                let warm_in = BsrnnInputHop::default();
                let _ = onnx_sep.process_hop(&warm_in, &mut out_hop);
                onnx_sep.reset();
                out_hop = BsrnnOutputHop::default();
                ready_clone.wait();

                let mut local_epoch = initial_epoch;
                let mut idle_spins = 0u32;
                let mut idle_since: Option<std::time::Instant> = None;
                let mut idle_sleep_us = WORKER_POLL_SLEEP_US;
                while !stop_clone.load(std::sync::atomic::Ordering::Acquire) {
                    let target_epoch = epoch_clone.load(std::sync::atomic::Ordering::Acquire);
                    if target_epoch != local_epoch {
                        onnx_sep.reset();
                        local_epoch = target_epoch;
                    }
                    if let Ok(in_hop) = in_rx.pop() {
                        idle_spins = 0;
                        idle_since = None;
                        idle_sleep_us = WORKER_POLL_SLEEP_US;
                        // Discard stale input hops queued prior to the latest `reset_state()` epoch
                        let latest_epoch = epoch_clone.load(std::sync::atomic::Ordering::Acquire);
                        if latest_epoch != local_epoch {
                            onnx_sep.reset();
                            local_epoch = latest_epoch;
                        }
                        if in_hop.epoch != local_epoch {
                            continue;
                        }
                        if onnx_sep.process_hop(&in_hop, &mut out_hop).is_ok() {
                            out_hop.epoch = in_hop.epoch;
                            out_hop.seq = in_hop.seq;
                            let _ = out_tx.push(out_hop);
                        } else {
                            let mut zero_hop = BsrnnOutputHop::default();
                            zero_hop.epoch = in_hop.epoch;
                            zero_hop.seq = in_hop.seq;
                            let _ = out_tx.push(zero_hop);
                        }
                    } else {
                        idle_spins = idle_spins.saturating_add(1);
                        if idle_spins < 64 {
                            std::hint::spin_loop();
                        } else {
                            // Progressive idle backoff: the RT side submits no hops while idle-gated,
                            // so after ~100 ms without work poll at up to 2 ms (resume needs a hop
                            // within the 4-hop / 42.7 ms cushion, so 2 ms is still ample).
                            let since = *idle_since.get_or_insert_with(std::time::Instant::now);
                            if since.elapsed() >= WORKER_IDLE_BACKOFF_AFTER {
                                idle_sleep_us = (idle_sleep_us * 2).min(WORKER_MAX_IDLE_SLEEP_US);
                            }
                            std::thread::sleep(std::time::Duration::from_micros(idle_sleep_us));
                        }
                    }
                }
            })
            .map_err(|e| format!("Failed to spawn ONNX GPU worker thread: {}", e))?;

        ready_barrier.wait();

        self.async_gpu_worker = Some(AsyncOnnxWorkerHandle {
            stop_flag,
            worker_epoch,
            join_handle: Some(join_handle),
            tier,
        });

        Ok(tier)
    }

    #[inline]
    pub fn is_onnx_gpu_worker_active(&self) -> bool {
        self.async_gpu_worker.is_some()
    }

    #[inline]
    pub fn active_onnx_tier(&self) -> Option<crate::onnx_gpu_runtime::GpuExecutionTier> {
        self.async_gpu_worker.as_ref().map(|w| w.tier)
    }

    /// Returns `true` when all stems are at `0.0 dB` with no mute/solo active.
    #[inline]
    pub fn is_unity_bypass(&self) -> bool {
        let any_solo = self.stem_solo.iter().any(|&s| s);
        let any_mute = self.stem_mute.iter().any(|&m| m);
        !any_solo
            && !any_mute
            && self.stem_gains_db.iter().all(|g| g.abs() < 1e-5)
    }

    /// Lockstep mode: drains pending input hops and runs the ONNX separator inline. Without any
    /// separator the input hops are discarded (no output hops -> stems read as unavailable).
    /// When the async GPU worker is active this is a no-op (the worker owns the queues).
    #[inline]
    pub fn service_worker_queue(&mut self) {
        if self.out_tx.is_none() {
            return;
        }
        let inline_separator = self.lockstep_onnx.is_some() || self.has_test_mock();
        loop {
            let in_hop = match self.in_rx.as_mut() {
                Some(in_rx) => match in_rx.pop() {
                    Ok(hop) => hop,
                    Err(_) => break,
                },
                None => break,
            };
            if in_hop.epoch != self.active_epoch || !inline_separator {
                continue;
            }
            let mut out_hop = BsrnnOutputHop::default();
            let produced = match &mut self.lockstep_onnx {
                Some(sep) => sep.process_hop(&in_hop, &mut out_hop).is_ok(),
                None => self.test_mock_process(&in_hop, &mut out_hop),
            };
            if produced {
                out_hop.epoch = in_hop.epoch;
                out_hop.seq = in_hop.seq;
                if let Some(out_tx) = &mut self.out_tx {
                    let _ = out_tx.push(out_hop);
                }
            }
        }
    }

    /// `true` when the async GPU worker cushion (`ASYNC_GPU_CUSHION_HOPS`) is part of the pipeline.
    #[inline]
    fn uses_async_cushion(&self) -> bool {
        self.async_gpu_worker.is_some() || self.test_mock_emulates_async()
    }

    #[inline]
    fn base_cushion_hops(&self) -> usize {
        if self.uses_async_cushion() {
            ASYNC_GPU_CUSHION_HOPS
        } else {
            0
        }
    }

    /// Resets the inline (lockstep / test) separator state. Allocation-free.
    #[inline]
    fn reset_inline_separator(&mut self) {
        if let Some(sep) = &mut self.lockstep_onnx {
            sep.reset();
        }
        self.reset_test_mock();
    }

    /// Restarts the separation pipeline when leaving idle, at the exact sample the idle condition
    /// flips (the new input hop starts at this sample). Like `reset_state()` but the dry
    /// `delay_line` (and therefore the output / reported latency) is left untouched.
    ///
    /// - `history_silent`: the whole dry window (`latency_samples`) was below `-120 dBFS`, so the
    ///   separator's zero history is faithful: cushion/first hops are valid (zero) stems and the
    ///   availability ramp starts immediately -> rebalance is fully applied at the first onset.
    /// - otherwise: the last `RESTART_PRIME_HOPS` hops of real input are re-fed to the reset
    ///   separator (true history, complete analysis frames), the hops carrying that warm-up are
    ///   skipped, and the availability ramp is held until the first valid stems leave
    ///   `stem_delay_lines` (`lat - 2560` samples), then fades in over the 10 ms ramp.
    ///
    /// Zero heap allocations (queue drains, atomics and fills of pre-allocated buffers only).
    fn restart_pipeline(&mut self, history_silent: bool) {
        self.active_epoch = self.active_epoch.wrapping_add(1);
        if let Some(bg) = &self.async_gpu_worker {
            bg.worker_epoch
                .store(self.active_epoch, std::sync::atomic::Ordering::Release);
        }
        self.reset_inline_separator();
        if let Some(in_rx) = &mut self.in_rx {
            while in_rx.pop().is_ok() {}
        }
        while self.out_rx.pop().is_ok() {}
        self.pending_out_hop = None;
        for dl in &mut self.stem_delay_lines {
            dl.reset();
        }
        self.current_out_hop = BsrnnOutputHop::default();
        self.current_out_pos = 0;
        self.staging_in_pos = 0;
        self.next_in_seq = 0;
        self.expected_out_seq = 0;
        self.cushion_hops_remaining = self.base_cushion_hops();

        if history_silent {
            self.cushion_hops_valid = true;
            self.current_hop_valid = true;
            self.avail_hold_samples = 0;
            return;
        }

        self.cushion_hops_valid = false;
        self.current_hop_valid = false;
        // Prime: virtual pipeline start `RESTART_PRIME_HOPS` hops in the past. Hop `k` carries input
        // hop `k - SEPARATOR_ALGO_DELAY_HOPS`, so the first popped hop `RESTART_PRIME_HOPS` carries
        // primed hop `RESTART_WARMUP_HOPS` (3 hops of true history before it), and its stems land
        // exactly on the dry path via the unchanged `stem_delay_lines` alignment.
        let mask = INPUT_HISTORY_LEN - 1;
        let mut read = (self.history_pos + INPUT_HISTORY_LEN - RESTART_PRIME_HOPS * BSRNN_HOP_SIZE) & mask;
        for h in 0..RESTART_PRIME_HOPS {
            for i in 0..BSRNN_HOP_SIZE {
                self.staging_in_hop.frames[i] = self.input_history[read];
                read = (read + 1) & mask;
            }
            self.staging_in_hop.epoch = self.active_epoch;
            self.staging_in_hop.seq = h as u64;
            if self.in_tx.push(self.staging_in_hop).is_ok() {
                self.pushed_hops = self.pushed_hops.wrapping_add(1);
            }
        }
        self.next_in_seq = RESTART_PRIME_HOPS as u64;
        self.expected_out_seq = RESTART_PRIME_HOPS as u64;
        let stem_align_delay = self.latency_samples.saturating_sub(self.current_pre_stem_delay());
        self.avail_hold_samples =
            (self.cushion_hops_remaining + 1) * BSRNN_HOP_SIZE + stem_align_delay;
        self.service_worker_queue();
    }

    #[cfg(test)]
    #[inline]
    fn has_test_mock(&self) -> bool {
        self.test_mock.is_some()
    }
    #[cfg(not(test))]
    #[inline(always)]
    fn has_test_mock(&self) -> bool {
        false
    }
    #[cfg(test)]
    #[inline]
    fn test_mock_emulates_async(&self) -> bool {
        self.test_mock.as_ref().is_some_and(|m| m.emulate_async)
    }
    #[cfg(not(test))]
    #[inline(always)]
    fn test_mock_emulates_async(&self) -> bool {
        false
    }
    #[cfg(test)]
    #[inline]
    fn reset_test_mock(&mut self) {
        if let Some(m) = &mut self.test_mock {
            m.reset();
        }
    }
    #[cfg(not(test))]
    #[inline(always)]
    fn reset_test_mock(&mut self) {}
    #[cfg(test)]
    #[inline]
    fn test_mock_process(&mut self, in_hop: &BsrnnInputHop, out_hop: &mut BsrnnOutputHop) -> bool {
        match &mut self.test_mock {
            Some(m) => {
                m.process_hop(in_hop, out_hop);
                true
            }
            None => false,
        }
    }
    #[cfg(not(test))]
    #[inline(always)]
    fn test_mock_process(&mut self, _in_hop: &BsrnnInputHop, _out_hop: &mut BsrnnOutputHop) -> bool {
        false
    }

    /// Installs a deterministic inline mock separator (test only, no GPU / model needed).
    #[cfg(test)]
    fn enable_test_mock_separator(&mut self, emulate_async: bool) {
        self.test_mock = Some(Box::new(tests::MockSeparator::new(emulate_async)));
        self.unity_idle_gating = true;
        let stem_align_delay = self.latency_samples.saturating_sub(self.current_pre_stem_delay());
        for dl in &mut self.stem_delay_lines {
            dl.set_delay_samples(stem_align_delay);
        }
        self.reset_state();
    }

    fn current_pre_stem_delay(&self) -> usize {
        if self.uses_async_cushion() {
            BSRNN_N_FFT
                + crate::bsrnn_onnx_gpu::OZONE_BSRNN_LOOKAHEAD_SAMPLES
                + ASYNC_GPU_CUSHION_HOPS * BSRNN_HOP_SIZE
        } else if self.lockstep_onnx.is_some() || self.has_test_mock() {
            BSRNN_N_FFT + crate::bsrnn_onnx_gpu::OZONE_BSRNN_LOOKAHEAD_SAMPLES
        } else {
            BSRNN_N_FFT
        }
    }

    /// Pops the exact output hop matching `(self.active_epoch, target_seq)` without heap allocations,
    /// discarding any stale epoch/late underflow hops and buffering at most 1 future hop.
    #[inline]
    fn pop_aligned_output_hop(&mut self, target_seq: u64) -> Option<BsrnnOutputHop> {
        if let Some(pending) = self.pending_out_hop {
            if pending.epoch < self.active_epoch
                || (pending.epoch == self.active_epoch && pending.seq < target_seq)
            {
                self.pending_out_hop = None;
            } else if pending.epoch == self.active_epoch && pending.seq == target_seq {
                self.pending_out_hop = None;
                return Some(pending);
            } else {
                return None;
            }
        }

        while let Ok(hop) = self.out_rx.pop() {
            if hop.epoch < self.active_epoch
                || (hop.epoch == self.active_epoch && hop.seq < target_seq)
            {
                continue;
            }
            if hop.epoch == self.active_epoch && hop.seq == target_seq {
                return Some(hop);
            }
            // Future hop (`hop.seq > target_seq` or newer epoch): preserve it for its exact sequence slot
            self.pending_out_hop = Some(hop);
            break;
        }

        None
    }
}

impl UnifiedDspNode for RealTimeBsrnnStemNode {
    fn name(&self) -> &'static str {
        "Ozone 12 Real-Time 120ms BSRNN 4-Stem Rebalance"
    }

    fn latency_samples(&self) -> usize {
        self.latency_samples
    }

    fn prepare(&mut self, sample_rate: f64, _max_block_size: usize) {
        self.sample_rate = sample_rate.max(8000.0);
        self.latency_samples = (BSRNN_REALTIME_LATENCY_SEC * self.sample_rate).round() as usize;
        let cap = self.latency_samples.max(16384);
        self.delay_line.ensure_capacity(cap);
        self.delay_line.set_delay_samples(self.latency_samples);

        let stem_align_delay = self.latency_samples.saturating_sub(self.current_pre_stem_delay());
        for dl in &mut self.stem_delay_lines {
            dl.ensure_capacity(cap);
            dl.set_delay_samples(stem_align_delay);
        }
    }

    fn reset_state(&mut self) {
        self.delay_line.reset();
        for dl in &mut self.stem_delay_lines {
            dl.reset();
        }
        self.reset_inline_separator();
        self.stems_available_gain = 0.0;
        self.applied_delta_gains = [0.0; BSRNN_STEM_COUNT];
        self.current_hop_valid = false;
        self.active_epoch = self.active_epoch.wrapping_add(1);
        self.next_in_seq = 0;
        self.expected_out_seq = 0;
        if let Some(in_rx) = &mut self.in_rx {
            while in_rx.pop().is_ok() {}
        }
        while self.out_rx.pop().is_ok() {}
        self.pending_out_hop = None;
        if let Some(bg) = &self.async_gpu_worker {
            bg.worker_epoch
                .store(self.active_epoch, std::sync::atomic::Ordering::Release);
        }
        self.cushion_hops_remaining = self.base_cushion_hops();
        self.staging_in_hop = BsrnnInputHop::default();
        self.staging_in_pos = 0;
        self.current_out_hop = BsrnnOutputHop::default();
        self.current_out_pos = 0;
        // Everything is zero again == the node has "heard" silence forever.
        self.idle = false;
        self.silent_run = usize::MAX;
        self.input_history.fill([0.0; 2]);
        self.history_pos = 0;
        self.cushion_hops_valid = false;
        self.avail_hold_samples = 0;
    }

    fn set_parameter_normalized(&mut self, param_id: u32, norm_val: f32) {
        let v = norm_val.clamp(0.0, 1.0);
        match param_id {
            0..=3 => self.stem_gains_db[param_id as usize] = -12.0 + v * 24.0,
            4..=7 => self.stem_solo[(param_id - 4) as usize] = v >= 0.5,
            8..=11 => self.stem_mute[(param_id - 8) as usize] = v >= 0.5,
            _ => {}
        }
    }

    fn export_visual_telemetry(&self, frame: &mut VisualTelemetryFrame) {
        frame.ozone12_telemetry.stem_rms_db = self.stem_rms_db;
        let mut solo_mask = 0u32;
        let mut mute_mask = 0u32;
        for s in 0..BSRNN_STEM_COUNT {
            if self.stem_solo[s] {
                solo_mask |= 1 << s;
            }
            if self.stem_mute[s] {
                mute_mask |= 1 << s;
            }
        }
        frame.header.active_stem_mode = if self.is_unity_bypass() {
            0
        } else if (solo_mask | mute_mask) != 0 {
            (solo_mask << 4) | mute_mask
        } else {
            1
        };
    }

    fn process_slice(
        &mut self,
        ctx: &ProcessContext,
        main_io: &mut [StereoFrame],
        _sidechain_in: Option<&[StereoFrame]>,
        _aux_sends_out: &mut [&mut [StereoFrame]],
    ) {
        let any_solo = self.stem_solo.iter().any(|&s| s);
        let mut target_delta = [0.0f32; BSRNN_STEM_COUNT];
        for s in 0..BSRNN_STEM_COUNT {
            target_delta[s] = if self.stem_mute[s] || (any_solo && !self.stem_solo[s]) {
                // Mute subtracts 100% of the stem (`Delta = -1.0`)
                -1.0
            } else {
                10.0f32.powf(self.stem_gains_db[s].clamp(-24.0, 12.0) / 20.0) - 1.0
            };
        }
        // ~10 ms de-zipper ramp for gain moves and for stem availability (GPU underflow fades).
        let coeff = (1.0 - (-1.0 / (0.010 * ctx.sample_rate.max(8_000.0))).exp()) as f32;
        let separation = self.separation_available();
        // Idle gating (only with a real separator): no hops reach the separator while idle.
        // Silence gate: every stem hop influenced by real audio has been played out.
        let silence_gate = self.latency_samples + self.current_pre_stem_delay();
        let unity_gate = separation && self.unity_idle_gating && self.is_unity_bypass();
        let history_mask = INPUT_HISTORY_LEN - 1;

        for frame in main_io.iter_mut() {
            let raw_in = *frame;

            // 0. Idle gating state machine (evaluated per sample so a restart happens exactly at
            //    the first non-silent sample / first sample after leaving unity).
            let prev_silent_run = self.silent_run;
            // NaN compares false -> counted as signal (never gates on garbage).
            let quiet = raw_in[0].abs() < IDLE_SILENCE_PEAK && raw_in[1].abs() < IDLE_SILENCE_PEAK;
            self.silent_run = if quiet { prev_silent_run.saturating_add(1) } else { 0 };
            if separation {
                let want_idle = self.silent_run >= silence_gate
                    || (unity_gate
                        && self
                            .applied_delta_gains
                            .iter()
                            .all(|d| d.abs() < IDLE_DELTA_EPS));
                if want_idle != self.idle {
                    self.idle = want_idle;
                    if !want_idle {
                        self.restart_pipeline(prev_silent_run >= self.latency_samples);
                    }
                }
            }
            self.input_history[self.history_pos] = raw_in;
            self.history_pos = (self.history_pos + 1) & history_mask;

            // 1. Feed lock-free input hop staging buffer & service worker (lockstep or async SPSC)
            if !self.idle {
                self.staging_in_hop.frames[self.staging_in_pos] = raw_in;
                self.staging_in_pos += 1;
                if self.staging_in_pos >= BSRNN_HOP_SIZE {
                    self.staging_in_pos = 0;
                    self.staging_in_hop.epoch = self.active_epoch;
                    self.staging_in_hop.seq = self.next_in_seq;
                    self.next_in_seq = self.next_in_seq.wrapping_add(1);
                    if self.in_tx.push(self.staging_in_hop).is_ok() {
                        self.pushed_hops = self.pushed_hops.wrapping_add(1);
                    }
                    self.service_worker_queue();
                }
            }

            // 2. Advance exact 5,760-sample (`120 ms` @ 48 kHz) dry delay line
            let mut single = [raw_in];
            self.delay_line.process_in_place(&mut single);
            let x_delayed = single[0];

            // 3. Pop next separated stem hop from lock-free `out_rx` when needed
            if self.current_out_pos >= BSRNN_HOP_SIZE {
                if self.idle {
                    // Gated: no hops in flight, no underflow accounting; stems fade out.
                    self.current_out_hop = BsrnnOutputHop::default();
                    self.current_hop_valid = false;
                } else if self.cushion_hops_remaining > 0 {
                    self.cushion_hops_remaining -= 1;
                    self.current_out_hop = BsrnnOutputHop::default();
                    self.current_hop_valid = self.cushion_hops_valid;
                } else {
                    let target_seq = self.expected_out_seq;
                    self.expected_out_seq = self.expected_out_seq.wrapping_add(1);
                    match self.pop_aligned_output_hop(target_seq) {
                        Some(hop) => {
                            self.current_out_hop = hop;
                            self.current_hop_valid = true;
                        }
                        None => {
                            self.current_out_hop = BsrnnOutputHop::default();
                            self.current_hop_valid = false;
                            if separation {
                                self.underflow_hops = self.underflow_hops.wrapping_add(1);
                            }
                        }
                    }
                }
                self.current_out_pos = 0;
            }

            let pos = self.current_out_pos;
            self.current_out_pos += 1;

            let avail_target = if self.current_hop_valid && self.avail_hold_samples == 0 {
                1.0
            } else {
                0.0
            };
            self.avail_hold_samples = self.avail_hold_samples.saturating_sub(1);
            self.stems_available_gain += coeff * (avail_target - self.stems_available_gain);
            let avail = self.stems_available_gain;

            let mut out_l = x_delayed[0];
            let mut out_r = x_delayed[1];
            for s in 0..BSRNN_STEM_COUNT {
                let mut stem_single = [self.current_out_hop.stems[s][pos]];
                self.stem_delay_lines[s].process_in_place(&mut stem_single);
                let stem_frame = stem_single[0];

                let p_sq = 0.5 * (stem_frame[0] * stem_frame[0] + stem_frame[1] * stem_frame[1]);
                let db = 10.0 * (p_sq + 1e-12).log10();
                self.stem_rms_db[s] = 0.995 * self.stem_rms_db[s] + 0.005 * db;

                self.applied_delta_gains[s] +=
                    coeff * (target_delta[s] - self.applied_delta_gains[s]);
                let g = self.applied_delta_gains[s] * avail;
                out_l += g * stem_frame[0];
                out_r += g * stem_frame[1];
            }
            *frame = [out_l, out_r];
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    //! Idle-gating tests. They use a deterministic inline mock separator with the exact
    //! `OzoneBsrnnOnnxSeparator` hop structure (output hop `k` carries input hop `k - 5`) whose 4
    //! stems are power-of-two fractions of the input, so `sum(stems) == input` bit-exactly. No GPU
    //! or ONNX model is needed. `emulate_async = true` additionally runs the async GPU-worker
    //! scheduling (4-hop cushion, async stem alignment) with inline servicing.
    use super::*;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    const MOCK_RING_HOPS: usize = SEPARATOR_ALGO_DELAY_HOPS + 1;
    const MOCK_SPLIT: [f32; BSRNN_STEM_COUNT] = [0.5, 0.25, 0.125, 0.125];

    pub(crate) struct MockSeparator {
        ring: Box<[[StereoFrame; BSRNN_HOP_SIZE]; MOCK_RING_HOPS]>,
        hops_seen: usize,
        pub(crate) emulate_async: bool,
        pub(crate) processed: u64,
    }

    impl MockSeparator {
        pub(crate) fn new(emulate_async: bool) -> Self {
            Self {
                ring: Box::new([[[0.0; 2]; BSRNN_HOP_SIZE]; MOCK_RING_HOPS]),
                hops_seen: 0,
                emulate_async,
                processed: 0,
            }
        }

        pub(crate) fn reset(&mut self) {
            for hop in self.ring.iter_mut() {
                *hop = [[0.0; 2]; BSRNN_HOP_SIZE];
            }
            self.hops_seen = 0;
        }

        pub(crate) fn process_hop(&mut self, in_hop: &BsrnnInputHop, out_hop: &mut BsrnnOutputHop) {
            self.ring[self.hops_seen % MOCK_RING_HOPS] = in_hop.frames;
            // == (hops_seen - SEPARATOR_ALGO_DELAY_HOPS) mod ring; zeros until then (reset).
            let src = (self.hops_seen + 1) % MOCK_RING_HOPS;
            for s in 0..BSRNN_STEM_COUNT {
                for i in 0..BSRNN_HOP_SIZE {
                    let x = self.ring[src][i];
                    out_hop.stems[s][i] = [x[0] * MOCK_SPLIT[s], x[1] * MOCK_SPLIT[s]];
                }
            }
            self.hops_seen += 1;
            self.processed += 1;
        }
    }

    // ---- Per-thread allocation counter (verifies the RT path stays allocation-free) ----------
    struct CountingAlloc;
    thread_local! {
        static THREAD_ALLOCS: Cell<u64> = const { Cell::new(0) };
    }
    #[inline]
    fn bump_allocs() {
        let _ = THREAD_ALLOCS.try_with(|c| c.set(c.get() + 1));
    }
    unsafe impl GlobalAlloc for CountingAlloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            bump_allocs();
            unsafe { System.alloc(layout) }
        }
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            bump_allocs();
            unsafe { System.alloc_zeroed(layout) }
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            bump_allocs();
            unsafe { System.realloc(ptr, layout, new_size) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }
    }
    #[global_allocator]
    static GLOBAL_COUNTING_ALLOC: CountingAlloc = CountingAlloc;
    fn thread_allocs() -> u64 {
        THREAD_ALLOCS.with(|c| c.get())
    }

    const SR: f64 = 48_000.0;

    fn ctx() -> ProcessContext {
        ProcessContext {
            sample_rate: SR,
            ..Default::default()
        }
    }

    fn tone(n: usize) -> StereoFrame {
        let t = n as f64 / SR;
        let tau = 2.0 * std::f64::consts::PI;
        [
            (0.4 * (tau * 220.0 * t).sin() + 0.1 * (tau * 1375.0 * t).sin()) as f32,
            (0.35 * (tau * 330.0 * t).cos()) as f32,
        ]
    }

    fn mock_node(emulate_async: bool) -> RealTimeBsrnnStemNode {
        let mut node = RealTimeBsrnnStemNode::new(SR);
        node.prepare(SR, 256);
        node.enable_test_mock_separator(emulate_async);
        assert!(node.separation_available());
        node
    }

    fn run(node: &mut RealTimeBsrnnStemNode, buf: &mut [StereoFrame], block: usize) {
        let c = ctx();
        for chunk in buf.chunks_mut(block) {
            node.process_slice(&c, chunk, None, &mut []);
        }
    }

    fn mock_processed(node: &RealTimeBsrnnStemNode) -> u64 {
        node.test_mock.as_ref().unwrap().processed
    }

    #[test]
    fn unity_bypass_submits_no_hops_and_is_exact_dry_delay() {
        for emulate_async in [false, true] {
            let mut node = mock_node(emulate_async);
            let lat = node.latency_samples();
            let n = 2 * 48_000;
            let input: Vec<StereoFrame> = (0..n).map(tone).collect();
            let mut buf = input.clone();
            run(&mut node, &mut buf, 256);
            assert!(node.is_idle_gated());
            assert_eq!(node.pushed_hops(), 0, "async={emulate_async}");
            assert_eq!(mock_processed(&node), 0);
            assert_eq!(node.underflow_hops, 0);
            assert_eq!(node.latency_samples(), lat);
            for i in 0..n {
                let exp = if i >= lat { input[i - lat] } else { [0.0; 2] };
                assert_eq!(buf[i], exp, "async={emulate_async} i={i}");
            }
        }
    }

    #[test]
    fn long_silence_stops_hop_submission_after_tail() {
        for emulate_async in [false, true] {
            let mut node = mock_node(emulate_async);
            node.stem_gains_db = [6.0, -3.0, 0.0, 2.0];
            let gate = node.latency_samples() + node.current_pre_stem_delay();

            // Silence from construction: gated immediately.
            let mut sil = vec![[0.0f32; 2]; 48_000];
            run(&mut node, &mut sil, 256);
            assert!(node.is_idle_gated());
            assert_eq!(node.pushed_hops(), 0);

            // 0.5 s of audio streams normally (1 hop per 512 samples).
            let mut loud: Vec<StereoFrame> = (0..24_064).map(tone).collect();
            run(&mut node, &mut loud, 256);
            assert!(!node.is_idle_gated());
            let at_silence_start = node.pushed_hops();
            assert_eq!(at_silence_start, 24_064 / 512);

            // Tail: keeps submitting until `latency + pre_stem_delay` silent samples, then stops.
            let mut tail = vec![[0.0f32; 2]; gate + 512];
            run(&mut node, &mut tail, 256);
            assert!(node.is_idle_gated());
            let after_tail = node.pushed_hops();
            assert!(after_tail - at_silence_start <= (gate / 512 + 1) as u64);
            let underflows = node.underflow_hops;

            let mut long = vec![[0.0f32; 2]; 3 * 48_000];
            run(&mut node, &mut long, 256);
            assert_eq!(node.pushed_hops(), after_tail, "async={emulate_async}");
            assert_eq!(node.underflow_hops, underflows);
            assert!(long.iter().all(|f| *f == [0.0, 0.0]));
        }
    }

    #[test]
    fn resume_from_silence_is_seamless_and_sample_exact() {
        for emulate_async in [false, true] {
            let mut node = mock_node(emulate_async);
            let gain_db = 6.0f32;
            let g = 10.0f32.powf(gain_db / 20.0);
            node.stem_gains_db = [gain_db; BSRNN_STEM_COUNT];
            let lat = node.latency_samples();
            let onset = 48_000 + 77; // mid-block, not hop aligned
            let n = onset + 72_000;
            let input: Vec<StereoFrame> =
                (0..n).map(|i| if i < onset { [0.0; 2] } else { tone(i) }).collect();
            let mut buf = input.clone();
            run(&mut node, &mut buf[..onset], 256);
            assert!(node.is_idle_gated());
            assert_eq!(node.pushed_hops(), 0);
            run(&mut node, &mut buf[onset..], 256);
            assert!(!node.is_idle_gated());
            // Silent history -> no priming; the first hop starts exactly at the onset sample.
            assert_eq!(node.pushed_hops(), ((n - onset) / 512) as u64, "async={emulate_async}");
            assert_eq!(node.underflow_hops, 0);
            for i in onset..onset + lat {
                assert_eq!(buf[i], [0.0, 0.0], "async={emulate_async} i={i}");
            }
            // Rebalance is fully applied from the very first delayed onset sample.
            let mut max_err = 0.0f32;
            for i in onset + lat..n {
                let x = input[i - lat];
                max_err = max_err
                    .max((buf[i][0] - g * x[0]).abs())
                    .max((buf[i][1] - g * x[1]).abs());
            }
            assert!(max_err < 5e-5, "async={emulate_async} max_err={max_err}");
        }
    }

    #[test]
    fn resume_from_unity_primes_history_holds_dry_then_fades_in() {
        for emulate_async in [false, true] {
            let mut node = mock_node(emulate_async);
            let lat = node.latency_samples();
            let gain_db = 6.0f32;
            let g = 10.0f32.powf(gain_db / 20.0);
            let flip = 96_000 + 64 * 3; // block boundary for 64-frame blocks, not hop aligned
            let n = flip + 48_000;
            let input: Vec<StereoFrame> = (0..n).map(tone).collect();
            let mut buf = input.clone();

            run(&mut node, &mut buf[..flip], 64);
            assert!(node.is_idle_gated());
            assert_eq!(node.pushed_hops(), 0);

            node.stem_gains_db = [gain_db; BSRNN_STEM_COUNT];
            // Restart happens exactly at the flip sample: 8 history hops are primed at once.
            run(&mut node, &mut buf[flip..flip + 1], 1);
            assert!(!node.is_idle_gated());
            assert_eq!(node.pushed_hops(), RESTART_PRIME_HOPS as u64);
            // The first live hop starts at `flip`, so it completes at `flip + 511`.
            run(&mut node, &mut buf[flip + 1..flip + 511], 1);
            assert_eq!(node.pushed_hops(), RESTART_PRIME_HOPS as u64);
            run(&mut node, &mut buf[flip + 511..flip + 512], 1);
            assert_eq!(node.pushed_hops(), RESTART_PRIME_HOPS as u64 + 1);
            run(&mut node, &mut buf[flip + 512..], 64);
            assert_eq!(
                node.pushed_hops(),
                (RESTART_PRIME_HOPS + (n - flip) / 512) as u64,
                "async={emulate_async}"
            );
            assert_eq!(node.underflow_hops, 0);

            // Stems invalid (warm-up / in flight): output is exactly the dry delayed input.
            let hold = lat - SEPARATOR_ALGO_DELAY_HOPS * BSRNN_HOP_SIZE; // 3,200 samples
            for i in flip..flip + hold {
                assert_eq!(buf[i], input[i - lat], "async={emulate_async} i={i}");
            }
            // Fade-in: monotone, bounded by the target gain (no step / click).
            for i in flip + hold..n {
                let x = input[i - lat];
                assert!(buf[i][0].abs() <= g * x[0].abs() + 1e-6, "i={i}");
                assert!(buf[i][1].abs() <= g * x[1].abs() + 1e-6, "i={i}");
            }
            // Settled: primed stems are sample-aligned with the dry path.
            let mut max_err = 0.0f32;
            for i in flip + hold + 6_000..n {
                let x = input[i - lat];
                max_err = max_err
                    .max((buf[i][0] - g * x[0]).abs())
                    .max((buf[i][1] - g * x[1]).abs());
            }
            // f32 one-pole ramps stall ~1.4e-5 short of their target (same 5e-5 floor as the
            // lockstep ONNX verification test).
            assert!(max_err < 5e-5, "async={emulate_async} max_err={max_err}");
        }
    }

    #[test]
    fn lockstep_without_unity_gating_keeps_streaming() {
        let mut node = mock_node(false);
        node.set_unity_idle_gating(false);
        let lat = node.latency_samples();
        let input: Vec<StereoFrame> = (0..48_128).map(tone).collect();
        let mut buf = input.clone();
        run(&mut node, &mut buf, 256);
        assert!(!node.is_idle_gated());
        assert_eq!(node.pushed_hops(), 48_128 / 512);
        for i in lat..buf.len() {
            assert_eq!(buf[i], input[i - lat]);
        }
    }

    #[test]
    fn idle_gating_transitions_are_allocation_free() {
        for emulate_async in [false, true] {
            let mut node = mock_node(emulate_async);
            node.stem_gains_db = [6.0; BSRNN_STEM_COUNT];
            let total = 6 * 48_000;
            let mut buf: Vec<StereoFrame> = (0..total)
                .map(|i| if i < 48_000 { [0.0; 2] } else { tone(i) })
                .collect();
            let c = ctx();
            let mut restarts_seen = 0u32;
            let mut was_idle = false;

            let before = thread_allocs();
            for (b, chunk) in buf.chunks_mut(256).enumerate() {
                let pos = b * 256;
                if pos == 2 * 48_000 {
                    node.stem_gains_db = [0.0; BSRNN_STEM_COUNT]; // -> unity idle
                }
                if pos == 4 * 48_000 {
                    node.stem_gains_db = [-6.0; BSRNN_STEM_COUNT]; // -> primed restart
                }
                node.process_slice(&c, chunk, None, &mut []);
                if was_idle && !node.is_idle_gated() {
                    restarts_seen += 1;
                }
                was_idle = node.is_idle_gated();
            }
            let allocs = thread_allocs() - before;

            assert_eq!(allocs, 0, "async={emulate_async}: RT path allocated {allocs} times");
            assert_eq!(restarts_seen, 2, "silence->audio and unity->rebalance restarts");
            assert!(!node.is_idle_gated());
            assert!(node.pushed_hops() > 0);
        }
    }
}
