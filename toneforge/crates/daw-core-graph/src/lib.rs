//! `daw-core-graph`: Production-grade Core Audio Graph for **ToneForge** (Clean-Room FL Studio 26
//! + FabFilter Pro-Q 4 + oeksound Soothe3 + iZotope Ozone 12 + UVR5 Architecture).
//!
//! Modules:
//! - [`apdc_engine`]: 5-Point Automatic Plugin Delay Compensation (`ApdcDelayLine`, `PluginRackSlot`,
//!   `SampleAccurateAutomationQueue`).
//! - [`slicer_ppq`]: Hybrid PPQ Tick & Sample Slicer (`HybridPpqSlicer`, 8-sample SIMD alignment, `TempoMap`).
//! - [`automation_spline`]: Signed-Exponential `TensionSpline`, `LfoModulator`, and 30-Bit Fixed-Point
//!   Internal Controller Bus (`0x40000000`).
//! - [`mixer_dag`]: 502-Track Topological Mixer DAG (`0=Master, 1..=500=Inserts, 501=Preview`),
//!   Kahn Level-Order Scheduler, Cycle Detection, and Lock-Free Work-Stealing Pool.
//! - [`patcher_modular`]: Modular Node Graph (`PatcherGraphNode`) + `SugiyamaLayout`.
//! - [`native_fx`]: Zero-allocation `SparseCompressorNode` (Gather/Scatter `log2`/`exp2`),
//!   `MaximusMultibandNode`, `EmphasisLimiterNode`, `GrossBeatTimeBufferNode`, `LuxeVerbFdnShimmerNode`.
//! - [`telemetry_shm`]: 128 KB (`131,072` B) `VisualTelemetryFrame` exchanged via `TripleBuffer`
//!   and wait-free SPSC `rtrb::RingBuffer<EngineCommand>`.
//! - [`audio_io`]: Windows MMCSS `"Pro Audio"` thread elevation & WASAPI-to-ASIO block ring adapter.

pub mod apdc_engine;
pub mod audio_io;
pub mod automation_spline;
pub mod mixer_dag;
pub mod native_fx;
pub mod patcher_modular;
pub mod slicer_ppq;
pub mod telemetry_shm;

/// Total number of Mixer Tracks (`0x15e06a0` in `FLEngine_x64.dll`: `0`=Master, `1..=500`=Inserts, `501`=Preview).
pub const MAX_MIXER_TRACKS: usize = 502;
pub const MASTER_TRACK_INDEX: usize = 0;
pub const FIRST_INSERT_TRACK_INDEX: usize = 1;
pub const LAST_INSERT_TRACK_INDEX: usize = 500;
pub const PREVIEW_TRACK_INDEX: usize = 501;

/// Maximum FX Insert Slots per Mixer Track (`+0x134c` on `TMixerTrack`).
pub const MAX_INSERT_SLOTS: usize = 10;

/// Maximum real-time audio block size in frames.
pub const MAX_BLOCK_SIZE: usize = 4096;

/// 30-Bit Fixed-Point Internal Controller scale (`0x40000000` = `1,073,741,824.0`, `sub_d24d80`).
pub const FIXED_POINT_30BIT_SCALE: f64 = 1_073_741_824.0;
pub const FIXED_POINT_30BIT_MAX: i32 = 0x4000_0000;

/// Interleaved stereo sample frame `[Left, Right]`.
pub type StereoFrame = [f32; 2];

/// Real-time execution context passed to every `UnifiedDspNode` in each PPQ sub-slice (`sub_d0ece0`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProcessContext {
    pub sample_rate: f64,
    pub current_sample_pos: i64,
    pub ppq_tick: i64,
    pub bpm: f64,
    pub slice_frames: usize,
    pub is_playing: bool,
    pub is_offline_rendering: bool,
}

impl Default for ProcessContext {
    fn default() -> Self {
        Self {
            sample_rate: 48_000.0,
            current_sample_pos: 0,
            ppq_tick: 0,
            bpm: 128.0,
            slice_frames: 64,
            is_playing: true,
            is_offline_rendering: false,
        }
    }
}

/// Unified DSP Node trait combining `TFruityPlug` (`04_flstudio26`), `ProQProcessor` (`02_proq4`),
/// `dsp::MainProcessor` (`02_soothe3`), and `DSP::OzoneChainElement` (`03_ozone12`).
pub trait UnifiedDspNode: Send {
    fn name(&self) -> &'static str;

    /// Intrinsic processing latency in samples at the current `sample_rate` (used by 5-Point APDC).
    fn latency_samples(&self) -> usize;

    /// Pre-allocates all internal buffers when sample rate or max block size changes.
    /// Heap allocation is strictly forbidden inside `process_slice`.
    fn prepare(&mut self, sample_rate: f64, max_block_size: usize);

    /// Clears delay lines, IIR state, and FFT overlap buffers on transport stop or seek.
    fn reset_state(&mut self);

    /// Sample-accurate normalized parameter update (`0.0..=1.0` or dispatched from `TAutomationDelay`).
    fn set_parameter_normalized(&mut self, param_id: u32, norm_val: f32);

    /// Invoked at PPQ Tick boundaries (`TFruityPlug::NewTick` `vfn[14]`).
    fn on_ppq_tick(&mut self, _ctx: &ProcessContext) {}

    /// Current instantaneous gain reduction in dB (`<= 0.0` when attenuating) for 120 FPS metering.
    fn current_gain_reduction_db(&self) -> f32 {
        0.0
    }

    /// Processes a real-time audio slice (Zero Heap Allocation).
    fn process_slice(
        &mut self,
        ctx: &ProcessContext,
        main_io: &mut [StereoFrame],
        sidechain_in: Option<&[StereoFrame]>,
        aux_sends_out: &mut [&mut [StereoFrame]],
    );
}

// Re-export primary public API types at the crate root
pub use apdc_engine::{
    ApdcDelayLine, BasicDelay32FS, DelayedAutomationEvent, DelayedVoiceEvent, PluginRackSlot,
    SampleAccurateAutomationQueue, VoiceEventKind,
};
pub use audio_io::{
    AsioSampleFormat, ProAudioThreadGuard, WasapiAsioBlockAdapter, convert_f32_to_int32_lsb,
};
pub use automation_spline::{
    AutomationClipArticulator, ControllerLink, InternalControllerBus, LfoModulator, LfoWaveform,
    NUM_INTERNAL_CONTROLLERS, TensionControlPoint, TensionSpline, fixed30_to_normalized,
    normalized_to_fixed30,
};
pub use mixer_dag::{
    ChannelGenerator, GraphCycleError, LockFreeAudioThreadPool, MixerEngine, MixerSendRoute,
    MixerTrack, TRACK_STATE_ACTIVE, TRACK_STATE_SILENT, TopologicalMixerScheduler,
};
pub use native_fx::{
    DeterministicGeneratorNode, EmphasisLimiterNode, FixedLatencyTransformNode,
    GrossBeatTimeBufferNode, LuxeVerbFdnShimmerNode, MaximusMultibandNode, SparseCompressorNode,
    sparse_compressor_gain_reduction, sparse_compressor_gain_reduction_rt,
};
pub use patcher_modular::{
    ModularAudioConnection, ModularParamConnection, NodeLayoutPosition, PATCHER_HOST_INPUT_NODE,
    PATCHER_HOST_OUTPUT_NODE, PatcherGraphNode, SugiyamaLayout,
};
pub use slicer_ppq::{DEFAULT_PPQ, HybridPpqSlicer, HybridTickSlicer, TempoMap, TempoPoint};
pub use telemetry_shm::{
    AutomationAndSplineTelemetry, EngineCommand, Ozone12MasterTelemetry, ProQ4SpectrumFrame,
    Soothe3ReductionFrame, TrackMeterEntry, TransportAndMasterHeader,
    VISUAL_TELEMETRY_FRAME_BYTES, VisualTelemetryFrame, create_command_ring_buffer,
    create_telemetry_triple_buffer,
};
