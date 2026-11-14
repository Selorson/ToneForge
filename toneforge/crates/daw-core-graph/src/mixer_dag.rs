//! 502-Track Topological Mixer DAG (`FLEngine_x64.dll` `0x15e06a0`, stride `0x1478 = 5,240 B`),
//! Kahn Level-Order Scheduler (`sub_cf42f0`), Cycle Detection, 5-Point APDC Solver, and
//! Lock-Free Work-Stealing Execution (`TSharedThreadBank` `0x00584720` / `sub_d0d4c0`).
//!
//! Track Index Assignment (`sub_d0cee0`):
//! - `0` = Master Track (`MASTER_TRACK_INDEX`)
//! - `1..=500` = 500 Insert Tracks (`FIRST_INSERT_TRACK_INDEX..=LAST_INSERT_TRACK_INDEX`)
//! - `501` = Preview / Current Track (`PREVIEW_TRACK_INDEX`)

use crate::apdc_engine::{
    ApdcDelayLine, DEFAULT_APDC_CAPACITY, PluginRackSlot, SampleAccurateAutomationQueue,
};
use crate::automation_spline::InternalControllerBus;
use crate::sequencer_pianoroll::{PlaylistSequencer, SequencerVoiceEvent};
use crate::slicer_ppq::{DEFAULT_PPQ, HybridPpqSlicer};
use crate::telemetry_shm::{
    DEFAULT_COMMAND_RING_CAPACITY, EngineCommand, TrackMeterEntry, VisualTelemetryFrame,
    create_command_ring_buffer, create_telemetry_triple_buffer,
};
use crate::{
    FIRST_INSERT_TRACK_INDEX, LAST_INSERT_TRACK_INDEX, MASTER_TRACK_INDEX, MAX_BLOCK_SIZE,
    MAX_INSERT_SLOTS, MAX_MIXER_TRACKS, PREVIEW_TRACK_INDEX, ProcessContext, StereoFrame,
    UnifiedDspNode,
};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};

/// Activity state constants for `TMixerTrack.BufferActivityState` (`+0x148`).
pub const TRACK_STATE_SILENT: u32 = 0;
pub const TRACK_STATE_ACTIVE: u32 = 2;

/// Pre-reserved maximum channels on the real-time `MixerEngine` (`Zero-Allocation` guarantee).
pub const MAX_RT_CHANNELS: usize = 128;

/// Pre-reserved maximum outgoing Send/Sidechain routes per `MixerTrack` (`Zero-Allocation` guarantee).
pub const MAX_SENDS_PER_TRACK: usize = 16;

/// Default capacity of the lock-free `RtGraphMutation` SPSC ring buffer (`Host -> RT`).
pub const DEFAULT_RT_MUTATION_RING_CAPACITY: usize = 4096;

/// Default capacity of the lock-free `RetiredRtResource` SPSC ring buffer (`RT -> toneforge-rt-gc`).
pub const DEFAULT_RT_GC_RING_CAPACITY: usize = 8192;

/// Heap-backed resources retired from the Real-Time WASAPI thread without ever invoking `drop()`
/// or `HeapFree` on the RT thread. Dropped asynchronously by [`RtGarbageCollector`] (`toneforge-rt-gc`).
pub enum RetiredRtResource {
    PluginSlot(Box<Option<PluginRackSlot>>),
    Channel(Box<Option<ChannelGenerator>>),
    ChannelBatch(Box<Vec<ChannelGenerator>>),
    SendRoute(Box<Option<MixerSendRoute>>),
    Pattern(Box<crate::sequencer_pianoroll::PianoRollPattern>),
    Articulator(Box<crate::automation_spline::AutomationClipArticulator>),
    AudioFrames(Box<Vec<StereoFrame>>),
    RecordTap(Box<Option<crate::audio_io::RtRecordTap>>),
    HwCaptureBridge(
        Box<Option<(rtrb::Consumer<StereoFrame>, crate::audio_io::AdaptiveDllDriftResampler)>>,
    ),
    MidiInputConsumer(Box<Option<rtrb::Consumer<SequencerVoiceEvent>>>),
    StemRegions(Box<Vec<crate::sequencer_pianoroll::StemClipRegion>>),
}

/// Lock-free, Zero-Allocation, Zero-Drop Real-Time Graph Mutation message (`Host -> RT`).
///
/// Every heap-bearing variant wraps its payload in a `Box<Option<T>>` or `Box<T>` carrier allocated
/// on the non-RT host thread. On the RT thread, values are moved/swapped in-place into pre-reserved
/// engine slots and the carrier `Box` (holding any displaced object) is forwarded to
/// `rtrb::Producer<RetiredRtResource>` (`0` RT `alloc`, `0` RT `dealloc`).
pub enum RtGraphMutation {
    /// Inserts or replaces FX slot `slot_index` (`0..10`) on `track_index` (`0..502`).
    /// Slots are fixed positions: `slot_index == len` appends, `slot_index > len` is rejected
    /// (the carrier returns the slot to the GC) — hosts pad the gap with `EmptySlotNode` slots first.
    SetOrReplacePluginSlot {
        track_index: u16,
        slot_index: u8,
        slot: Box<Option<PluginRackSlot>>,
    },
    /// Removes FX slot `slot_index` on `track_index`, storing the removed slot in `recycle_slot` (`Box::new(None)`).
    RemovePluginSlot {
        track_index: u16,
        slot_index: u8,
        recycle_slot: Box<Option<PluginRackSlot>>,
    },
    /// Adds or updates a Send/Sidechain route `src_track -> dest_track` (cycle-checked on RT with `0` alloc).
    AddOrUpdateSendRoute {
        src_track: u16,
        dest_track: u16,
        gain: f32,
        is_sidechain_only: bool,
        route_carrier: Box<Option<MixerSendRoute>>,
    },
    /// Removes a Send/Sidechain route `src_track -> dest_track`, retiring it via `recycle_carrier` (`Box::new(None)`).
    RemoveSendRoute {
        src_track: u16,
        dest_track: u16,
        is_sidechain_only: bool,
        recycle_carrier: Box<Option<MixerSendRoute>>,
    },
    /// Inserts or replaces a `ChannelGenerator` matched by `channel.id`.
    UpsertChannel {
        channel: Box<Option<ChannelGenerator>>,
    },
    /// Removes a `ChannelGenerator` by `channel_id`, retiring it via `recycle_channel` (`Box::new(None)`).
    RemoveChannel {
        channel_id: u32,
        recycle_channel: Box<Option<ChannelGenerator>>,
    },
    /// Atomically replaces stem playback generators on tracks selected by `target_tracks_mask`
    /// (`bit i` corresponds to Mixer Track `i`, e.g. `0b1_1110` for Insert Tracks `1..=4`).
    /// `retired_channels` must be pre-allocated on the host with `Vec::with_capacity(16)`.
    ReplaceStemChannels {
        target_tracks_mask: u16,
        new_channels: Box<Vec<ChannelGenerator>>,
        retired_channels: Box<Vec<ChannelGenerator>>,
        reset_slicer: bool,
    },
    /// Inserts or replaces a `PianoRollPattern` in `PlaylistSequencer` via `std::mem::swap`.
    UpsertPattern {
        pattern: Box<crate::sequencer_pianoroll::PianoRollPattern>,
    },
    /// Adds or updates a single `PianoRollNote` inside `pattern_id` in-place.
    AddOrUpdateNote {
        pattern_id: u32,
        note: crate::sequencer_pianoroll::PianoRollNote,
    },
    /// Removes `note_id` from `pattern_id` in-place.
    RemoveNote {
        pattern_id: u32,
        note_id: u32,
    },
    /// Inserts or updates a `PlaylistClip` in-place.
    UpsertPlaylistClip {
        clip: crate::sequencer_pianoroll::PlaylistClip,
    },
    /// Removes `clip_id` from `PlaylistSequencer` in-place.
    RemovePlaylistClip {
        clip_id: u32,
    },
    /// Updates `PlaylistSequencer` playback mode and loop parameters.
    SetSequencerMode {
        song_mode: bool,
        active_pattern_id: u32,
        loop_enabled: bool,
        loop_length_ticks: i64,
    },
    /// Inserts or updates an `AutomationClipArticulator` via `std::mem::swap`.
    UpsertAutomationArticulator {
        articulator: Box<crate::automation_spline::AutomationClipArticulator>,
    },
    /// Registers or updates a 30-bit fixed-point `ControllerLink` in `InternalControllerBus`.
    UpsertControllerLink {
        link: crate::automation_spline::ControllerLink,
    },
    /// Clears all active `ControllerLink` entries on `InternalControllerBus` (`0` RT alloc).
    ClearControllerLinks,
    /// Updates the engine sample rate (`slicer.sample_rate`) and resets APDC delay lines without allocating.
    UpdateSampleRate {
        sample_rate: f64,
    },
    /// Configures hardware audio input routing (`input_source_bus`) and record-arming on `track_index`.
    SetTrackInputAndRecordArm {
        track_index: u16,
        input_source_bus: Option<u8>,
        record_armed: bool,
        hw_input_latency_samples: u32,
    },
    /// Attaches or replaces (or detaches when `None`) the lock-free `RtRecordTap` (`0` RT alloc/drop).
    AttachRecordTap {
        tap: Box<Option<crate::audio_io::RtRecordTap>>,
    },
    /// Attaches or replaces (or detaches when `None`) the WASAPI Capture + Adaptive DLL Drift Resampler bridge.
    AttachHwCaptureBridge {
        bridge: Box<
            Option<(
                rtrb::Consumer<StereoFrame>,
                crate::audio_io::AdaptiveDllDriftResampler,
            )>,
        >,
    },
    /// Attaches or replaces (or detaches when `None`) the Win32 MIDI hardware input consumer.
    AttachMidiInputConsumer {
        consumer: Box<Option<rtrb::Consumer<SequencerVoiceEvent>>>,
    },
    /// Replaces the timeline regions of the `AudioStemClipPlayerNode` owned by `channel_id` with
    /// `regions[..count]` (Playlist audio clip sync). `regions` must be pre-sized on the host to
    /// `MAX_STEM_CLIP_REGIONS`; the displaced region table is retired to `toneforge-rt-gc`.
    SetStemClipRegions {
        channel_id: u32,
        regions: Box<Vec<crate::sequencer_pianoroll::StemClipRegion>>,
        count: usize,
    },
}

/// Background Non-Real-Time Garbage Collector (`toneforge-rt-gc`) that drains [`RetiredRtResource`]
/// objects pushed by the Real-Time WASAPI audio thread and drops them safely on a background thread.
pub struct RtGarbageCollector {
    shutdown: Arc<AtomicBool>,
    freed_count: Arc<AtomicU64>,
    worker: Option<JoinHandle<()>>,
    disposer: Arc<parking_lot::Mutex<Option<RetiredPluginDisposer>>>,
}

/// Receives retired resources that own plug-in instances (FX slots, channel generators) so the host
/// can tear them down on its plug-in thread (VST3 `terminate` / CLAP `destroy` belong on the thread
/// that runs the editors), instead of on `toneforge-rt-gc`.
pub type RetiredPluginDisposer = Box<dyn FnMut(RetiredRtResource) + Send>;

impl RtGarbageCollector {
    /// Spawns the `"toneforge-rt-gc"` background thread consuming from `gc_consumer`.
    pub fn spawn(mut gc_consumer: rtrb::Consumer<RetiredRtResource>) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let freed_count = Arc::new(AtomicU64::new(0));
        let shutdown_bg = Arc::clone(&shutdown);
        let freed_bg = Arc::clone(&freed_count);
        let disposer: Arc<parking_lot::Mutex<Option<RetiredPluginDisposer>>> =
            Arc::new(parking_lot::Mutex::new(None));
        let disposer_bg = Arc::clone(&disposer);

        let worker = thread::Builder::new()
            .name("toneforge-rt-gc".to_string())
            .spawn(move || {
                let release = |retired: RetiredRtResource| {
                    let owns_plugin = matches!(
                        retired,
                        RetiredRtResource::PluginSlot(_)
                            | RetiredRtResource::Channel(_)
                            | RetiredRtResource::ChannelBatch(_)
                    );
                    if owns_plugin {
                        if let Some(d) = disposer_bg.lock().as_mut() {
                            d(retired);
                            return;
                        }
                    }
                    drop(retired);
                };
                while !shutdown_bg.load(Ordering::Acquire) {
                    let mut drained = 0u64;
                    while let Ok(retired) = gc_consumer.pop() {
                        release(retired);
                        drained += 1;
                    }
                    if drained > 0 {
                        freed_bg.fetch_add(drained, Ordering::Relaxed);
                    } else {
                        thread::park_timeout(std::time::Duration::from_millis(1));
                    }
                }
                let mut final_drained = 0u64;
                while let Ok(retired) = gc_consumer.pop() {
                    release(retired);
                    final_drained += 1;
                }
                if final_drained > 0 {
                    freed_bg.fetch_add(final_drained, Ordering::Relaxed);
                }
            })
            .expect("failed to spawn toneforge-rt-gc thread");

        Self {
            shutdown,
            freed_count,
            worker: Some(worker),
            disposer,
        }
    }

    /// Installs (or removes) the host's plug-in disposer; see [`RetiredPluginDisposer`].
    pub fn set_plugin_disposer(&self, disposer: Option<RetiredPluginDisposer>) {
        *self.disposer.lock() = disposer;
    }

    /// Wakes the GC thread immediately so retired RT resources are reclaimed without waiting for timeout.
    #[inline]
    pub fn wake(&self) {
        if let Some(w) = &self.worker {
            w.thread().unpark();
        }
    }

    /// Total number of [`RetiredRtResource`] items freed by `toneforge-rt-gc`.
    #[inline]
    pub fn freed_count(&self) -> u64 {
        self.freed_count.load(Ordering::Relaxed)
    }
}

impl Drop for RtGarbageCollector {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(w) = &self.worker {
            w.thread().unpark();
        }
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

/// Creates the paired lock-free `RtGraphMutation` SPSC channel (`Host -> RT`) and
/// `RetiredRtResource` SPSC channel + running `RtGarbageCollector` (`RT -> toneforge-rt-gc`).
pub fn create_rt_mutation_channels(
    mutation_capacity: usize,
    gc_capacity: usize,
) -> (
    rtrb::Producer<RtGraphMutation>,
    rtrb::Consumer<RtGraphMutation>,
    rtrb::Producer<RetiredRtResource>,
    RtGarbageCollector,
) {
    let (mut_prod, mut_cons) = rtrb::RingBuffer::new(mutation_capacity.max(64));
    let (gc_prod, gc_cons) = rtrb::RingBuffer::new(gc_capacity.max(128));
    let gc_worker = RtGarbageCollector::spawn(gc_cons);
    (mut_prod, mut_cons, gc_prod, gc_worker)
}

/// Error returned when a routing change or graph build encounters a directed cycle in the Mixer DAG.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphCycleError {
    pub cyclic_tracks: Vec<usize>,
}

impl fmt::Display for GraphCycleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Cycle detected in Mixer routing graph involving tracks: {:?}",
            self.cyclic_tracks
        )
    }
}

impl std::error::Error for GraphCycleError {}

/// Inter-Track Send / Sidechain Route equipped with **Point 5 APDC** (`[TPluginRackModule + 0x4c]`,
/// `0x00cf2cb0` & `0x00cf33e5`).
pub struct MixerSendRoute {
    pub dest_track_index: usize,
    pub gain: f32,
    pub is_sidechain_only: bool,
    pub send_apdc_delay: ApdcDelayLine, // Point 5: Inter-Track Send / Sidechain APDC
    pub send_buffer: Vec<StereoFrame>,  // Pre-allocated to `MAX_BLOCK_SIZE`
}

impl MixerSendRoute {
    pub fn new(dest_track_index: usize, gain: f32, is_sidechain_only: bool) -> Self {
        Self {
            dest_track_index,
            gain,
            is_sidechain_only,
            send_apdc_delay: ApdcDelayLine::with_capacity(DEFAULT_APDC_CAPACITY),
            send_buffer: vec![[0.0; 2]; MAX_BLOCK_SIZE],
        }
    }
}

/// Generator / Audio Clip Channel (`TChannel` / `TChannelModule` `VMT = 0x00cf2830`) equipped with
/// **Point 1: Channel Output APDC** (`[TChannel + 0x130]`, `0x00cf33b0`).
pub struct ChannelGenerator {
    pub id: u32,
    pub target_mixer_track: usize,
    pub enabled: bool,
    pub gain: f32,
    pub generator: Box<dyn UnifiedDspNode>,
    pub channel_apdc_delay: ApdcDelayLine, // Point 1: Channel Output Delay (`[TChannel + 0x130]`)
    pub output_buf: Vec<StereoFrame>,      // Pre-allocated to `MAX_BLOCK_SIZE`
}

impl ChannelGenerator {
    pub fn new(id: u32, target_mixer_track: usize, generator: Box<dyn UnifiedDspNode>) -> Self {
        let lat = generator.latency_samples();
        Self {
            id,
            target_mixer_track: target_mixer_track.min(MAX_MIXER_TRACKS - 1),
            enabled: true,
            gain: 1.0,
            generator,
            channel_apdc_delay: ApdcDelayLine::with_capacity(lat.max(DEFAULT_APDC_CAPACITY)),
            output_buf: vec![[0.0; 2]; MAX_BLOCK_SIZE],
        }
    }

    pub fn prepare(&mut self, sample_rate: f64, max_block_size: usize) {
        let cap = max_block_size.max(MAX_BLOCK_SIZE);
        if self.output_buf.len() < cap {
            self.output_buf.resize(cap, [0.0; 2]);
        }
        self.generator.prepare(sample_rate, cap);
        self.channel_apdc_delay
            .ensure_capacity(self.generator.latency_samples().max(DEFAULT_APDC_CAPACITY));
    }

    pub fn reset_state(&mut self) {
        self.generator.reset_state();
        self.channel_apdc_delay.reset();
    }

    /// Renders `slice_len` frames from the generator and applies **Point 1 APDC** in-place
    /// (`0x00cf33b0`: `sub_9803f0(*(TBasicDelay_32FS**)(Channel + 0x130), OutBuf, SliceLen)`).
    /// Zero Heap Allocation.
    #[inline]
    pub fn render_slice(&mut self, ctx: &ProcessContext, slice_offset: usize, slice_len: usize) {
        if !self.enabled {
            return;
        }
        let buf = &mut self.output_buf[slice_offset..slice_offset + slice_len];
        buf.fill([0.0; 2]);
        self.generator.process_slice(ctx, buf, None, &mut []);
        if (self.gain - 1.0).abs() > 1e-7 {
            let g = self.gain;
            for frame in buf.iter_mut() {
                frame[0] *= g;
                frame[1] *= g;
            }
        }
        // Point 1: Channel Output APDC
        self.channel_apdc_delay.process_in_place(buf);
    }
}

/// Equivalent to `TMixerTrack` (`Size = 0x1478 = 5,240 B` in `FLEngine_x64.dll` at `0x15e06a0`).
pub struct MixerTrack {
    pub index: usize,
    pub sub_block_buf: Vec<StereoFrame>,      // `+0x34` Direct input -> processed track output
    pub upstream_mix_buf: Vec<StereoFrame>,   // `+0x3c` Sum of incoming upstream track Sends
    pub sidechain_buf: Vec<StereoFrame>,      // `+0x44` Sum of incoming Sidechain routes
    pub hw_input_buf: Vec<StereoFrame>,       // `+0xe4`/`+0xec` Hardware ASIO Input buffer
    pub has_hw_input: bool,                   // `+0xe0` (`ASIOInChannelIndex >= 0`)
    pub input_source_bus: Option<u8>,         // Hardware stereo input bus index (`Some(0)` = In 1-2)
    pub record_armed: bool,                   // Track armed for `DiskTrackRecorder` `.wav` recording
    pub hw_input_latency_samples: usize,      // Intrinsic latency of the hardware input source
    pub has_channel_input: bool,              // Set when at least one `ChannelGenerator` targets this track
    pub direct_source_latency_samples: usize, // Latency of direct input entering Point 2
    pub has_upstream_audio: bool,             // True if any upstream track accumulated audio this block
    pub has_sidechain_input: bool,            // True if any upstream track routed sidechain this block
    pub input_apdc_delay: ApdcDelayLine,      // Point 2: Hardware / Direct Input APDC (`+0xa8`, `0x00d0e2ea`)
    pub activity_state: AtomicU32,            // `+0x148` (`0` = Silent, `2` = Active)
    pub needs_clear: bool,                    // `+0x150` (`NeedsClearFlag`)
    pub slots: Vec<PluginRackSlot>,           // `+0x134c` (up to `MAX_INSERT_SLOTS = 10`, with Point 3 APDC)
    pub post_rack_apdc_delay: ApdcDelayLine,  // Point 4: Post-Chain APDC (`[TPluginRackModule + 0x5c]`, `0x00cf2d00`)
    pub sends: Vec<MixerSendRoute>,           // `+0x130` (`TMixerRoutingTable`, with Point 5 APDC)
    pub gain: f32,
    pub polarity_inverted: bool,
    /// Stereo balance (`-1.0..=1.0`, unity at centre).
    pub pan: f32,
    /// User mute switch.
    pub muted: bool,
    /// User solo switch.
    pub soloed: bool,
    /// Implicit mute applied while another track is soloed (recomputed by `MixerEngine::recompute_solo_state`).
    pub solo_muted: bool,
    /// De-clicked per-channel gain actually applied (one-pole ramp towards the fader/pan/mute target).
    applied_gain_lr: [f32; 2],
    pub input_arrival_latency: usize, // Target aligned latency at the entrance of `slots`
    pub rack_output_latency: usize,   // Latency immediately after `slots` (before Point 4)
    pub accumulated_latency: usize,   // Latency after Point 4 (`post_rack_apdc_delay`)
    pub meter: TrackMeterEntry,
}

impl MixerTrack {
    /// Creates a pre-allocated `MixerTrack` at `index` (`0..502`).
    /// Insert tracks `1..=500` route to Master (`0`) by default only when explicitly connected or via helper.
    pub fn new(index: usize) -> Self {
        Self {
            index,
            sub_block_buf: vec![[0.0; 2]; MAX_BLOCK_SIZE],
            upstream_mix_buf: vec![[0.0; 2]; MAX_BLOCK_SIZE],
            sidechain_buf: vec![[0.0; 2]; MAX_BLOCK_SIZE],
            hw_input_buf: vec![[0.0; 2]; MAX_BLOCK_SIZE],
            has_hw_input: false,
            input_source_bus: None,
            record_armed: false,
            hw_input_latency_samples: 0,
            has_channel_input: false,
            direct_source_latency_samples: 0,
            has_upstream_audio: false,
            has_sidechain_input: false,
            input_apdc_delay: ApdcDelayLine::with_capacity(DEFAULT_APDC_CAPACITY),
            activity_state: AtomicU32::new(TRACK_STATE_SILENT),
            needs_clear: true,
            slots: Vec::with_capacity(MAX_INSERT_SLOTS),
            post_rack_apdc_delay: ApdcDelayLine::with_capacity(DEFAULT_APDC_CAPACITY),
            sends: Vec::with_capacity(MAX_SENDS_PER_TRACK),
            gain: 1.0,
            polarity_inverted: false,
            pan: 0.0,
            muted: false,
            soloed: false,
            solo_muted: false,
            applied_gain_lr: [1.0, 1.0],
            input_arrival_latency: 0,
            rack_output_latency: 0,
            accumulated_latency: 0,
            meter: TrackMeterEntry::default(),
        }
    }

    /// Inserts an FX plugin into the track's 10-slot rack (`+0x134c`).
    pub fn add_plugin_slot(
        &mut self,
        plugin: Box<dyn UnifiedDspNode>,
        wet_mix: f32,
    ) -> Result<usize, &'static str> {
        if self.slots.len() >= MAX_INSERT_SLOTS {
            return Err("MixerTrack already has MAX_INSERT_SLOTS (10) plugins");
        }
        let idx = self.slots.len();
        self.slots.push(PluginRackSlot::new(plugin, wet_mix));
        Ok(idx)
    }

    /// Inserts or replaces an FX plugin in the track's 10-slot rack (`+0x134c`) at `slot_index` (`0..10`),
    /// preparing its internal buffers for `sample_rate` and `max_block_size`.
    pub fn set_or_replace_plugin_slot(
        &mut self,
        slot_index: usize,
        mut plugin: Box<dyn UnifiedDspNode>,
        wet_mix: f32,
        sample_rate: f64,
        max_block_size: usize,
    ) -> Result<usize, &'static str> {
        if slot_index >= MAX_INSERT_SLOTS {
            return Err("slot_index out of range (0..10)");
        }
        let cap = max_block_size.max(MAX_BLOCK_SIZE);
        plugin.prepare(sample_rate, cap);
        let mut rack_slot = PluginRackSlot::new(plugin, wet_mix);
        rack_slot.prepare(sample_rate, cap);
        // Fixed slot positions: fill any gap with empty pass-through slots so the plug-in lands
        // exactly at `slot_index` (never at `len`).
        while self.slots.len() < slot_index {
            let mut empty = PluginRackSlot::new(Box::new(crate::EmptySlotNode::new()), 1.0);
            empty.prepare(sample_rate, cap);
            self.slots.push(empty);
        }
        if slot_index < self.slots.len() {
            self.slots[slot_index] = rack_slot;
            Ok(slot_index)
        } else if self.slots.len() < MAX_INSERT_SLOTS {
            let idx = self.slots.len();
            self.slots.push(rack_slot);
            Ok(idx)
        } else {
            Err("MixerTrack already has MAX_INSERT_SLOTS (10) plugins")
        }
    }

    /// Adds a Send or Sidechain route from this track to `dest_track_index`.
    pub fn add_send(&mut self, dest_track_index: usize, gain: f32, is_sidechain_only: bool) {
        if dest_track_index < MAX_MIXER_TRACKS && dest_track_index != self.index {
            // Update existing route if already present
            for route in &mut self.sends {
                if route.dest_track_index == dest_track_index
                    && route.is_sidechain_only == is_sidechain_only
                {
                    route.gain = gain;
                    return;
                }
            }
            self.sends
                .push(MixerSendRoute::new(dest_track_index, gain, is_sidechain_only));
        }
    }

    /// Sum of latencies of all enabled FX slots on this track.
    #[inline]
    pub fn internal_rack_latency(&self) -> usize {
        let mut lat = 0usize;
        for slot in &self.slots {
            if slot.enabled {
                lat += slot.plugin.latency_samples();
            }
        }
        lat
    }

    /// Cumulative APDC latency at the entrance of `slot_index` (`0..10`), or at the post-rack
    /// track fader when `slot_index == 0xFF`. Used by `TAutomationDelay` so parameter changes
    /// on later FX slots account for the latency of earlier FX slots on the same track.
    #[inline]
    pub fn slot_input_arrival_latency(&self, slot_index: u8) -> usize {
        if slot_index == 0xFF {
            return self.rack_output_latency;
        }
        let mut lat = self.input_arrival_latency;
        let limit = (slot_index as usize).min(self.slots.len());
        for i in 0..limit {
            lat += self.slots[i].effective_latency_samples();
        }
        lat
    }

    /// Pre-allocates all track buffers, slot buffers, and APDC delay lines for `max_block_size`.
    pub fn prepare(&mut self, sample_rate: f64, max_block_size: usize) {
        let cap = max_block_size.max(MAX_BLOCK_SIZE);
        if self.sub_block_buf.len() < cap {
            self.sub_block_buf.resize(cap, [0.0; 2]);
        }
        if self.upstream_mix_buf.len() < cap {
            self.upstream_mix_buf.resize(cap, [0.0; 2]);
        }
        if self.sidechain_buf.len() < cap {
            self.sidechain_buf.resize(cap, [0.0; 2]);
        }
        if self.hw_input_buf.len() < cap {
            self.hw_input_buf.resize(cap, [0.0; 2]);
        }
        for slot in &mut self.slots {
            slot.prepare(sample_rate, cap);
        }
        for send in &mut self.sends {
            if send.send_buffer.len() < cap {
                send.send_buffer.resize(cap, [0.0; 2]);
            }
        }
    }

    /// Resets all delay lines and plugin states on this track.
    pub fn reset_state(&mut self) {
        self.applied_gain_lr = self.target_gain_lr();
        self.input_apdc_delay.reset();
        for slot in &mut self.slots {
            slot.reset_state();
        }
        self.post_rack_apdc_delay.reset();
        for send in &mut self.sends {
            send.send_apdc_delay.reset();
        }
    }

    /// Jumps the de-click gain ramp straight to the current target (off-RT configuration).
    #[inline]
    pub fn snap_gain_ramp(&mut self) {
        self.applied_gain_lr = self.target_gain_lr();
    }

    /// Per-channel target gain combining fader, polarity, mute/solo and stereo balance
    /// (balance law: unity at centre, the opposite channel is attenuated linearly).
    #[inline]
    pub fn target_gain_lr(&self) -> [f32; 2] {
        if self.muted || self.solo_muted {
            return [0.0, 0.0];
        }
        let g = if self.polarity_inverted { -self.gain } else { self.gain };
        let p = self.pan.clamp(-1.0, 1.0);
        [g * (1.0 - p).min(1.0), g * (1.0 + p).min(1.0)]
    }

    /// Clears transient slice buffers for `[slice_offset..slice_offset + slice_len]` before a slice render.
    #[inline]
    pub fn clear_slice_buffers(&mut self, slice_offset: usize, slice_len: usize) {
        let end = slice_offset + slice_len;
        self.sub_block_buf[slice_offset..end].fill([0.0; 2]);
        self.upstream_mix_buf[slice_offset..end].fill([0.0; 2]);
        self.sidechain_buf[slice_offset..end].fill([0.0; 2]);
        self.has_upstream_audio = false;
        self.has_sidechain_input = false;
    }

    /// Equivalent to `TPluginRackModule.Render` (`0x00cf2a90`) + `TMixerTrackModule` (`0x01276868`).
    /// Executes Points 2, 3, 4, and 5 of the 5-Point APDC architecture on `[slice_offset..slice_offset + slice_len]`.
    /// Zero Heap Allocation.
    #[inline]
    pub fn render_slice(&mut self, ctx: &ProcessContext, slice_offset: usize, slice_len: usize) {
        let end = slice_offset + slice_len;
        let target = self.target_gain_lr();
        let buf = &mut self.sub_block_buf[slice_offset..end];

        // 1 & 2. Hardware Input (`+0xe4`/`+0xec`) + Point 2 APDC (`[TMixerTrack + 0xa8]`, `0x00d0e2ea`):
        // - If both `has_hw_input` and `has_channel_input` are active on this track, Point 2 delays
        //   `hw_input_buf` from `hw_input_latency_samples` to `input_arrival_latency` BEFORE mixing
        //   with `buf` (which already holds `ChannelGenerator` audio aligned to `input_arrival_latency`
        //   by Point 1).
        // - Otherwise, `hw_input_buf` (if any) is summed into `buf` and Point 2 delays `buf` to
        //   `input_arrival_latency`.
        if self.has_hw_input {
            if self.has_channel_input {
                let hw = &mut self.hw_input_buf[slice_offset..end];
                self.input_apdc_delay.process_in_place(hw);
                for i in 0..slice_len {
                    buf[i][0] += hw[i][0];
                    buf[i][1] += hw[i][1];
                }
            } else {
                let hw = &self.hw_input_buf[slice_offset..end];
                for i in 0..slice_len {
                    buf[i][0] += hw[i][0];
                    buf[i][1] += hw[i][1];
                }
                self.input_apdc_delay.process_in_place(buf);
            }
        } else {
            self.input_apdc_delay.process_in_place(buf);
        }

        // 3. Merge upstream track Sends (`upstream_mix_buf`, already aligned to `self.input_arrival_latency`
        // by upstream Point 4 + Point 5 APDC).
        if self.has_upstream_audio {
            let up = &self.upstream_mix_buf[slice_offset..end];
            for i in 0..slice_len {
                buf[i][0] += up[i][0];
                buf[i][1] += up[i][1];
            }
        }

        // 4. Execute the 10 FX Slots (`+0x134c`) with Point 3 Parallel Dry/Wet APDC
        let sc_opt = if self.has_sidechain_input {
            Some(&self.sidechain_buf[slice_offset..end])
        } else {
            None
        };
        let mut min_gr_db = if slice_offset == 0 {
            0.0f32
        } else {
            self.meter.gr_db
        };
        for slot in self.slots.iter_mut() {
            slot.process_with_sidechain(ctx, buf, sc_opt);
            if slot.enabled {
                let gr = slot.plugin.current_gain_reduction_db();
                if gr < min_gr_db {
                    min_gr_db = gr;
                }
            }
        }

        // 5. Apply post-fader gain, polarity, mute/solo and balance with a ~5 ms de-click ramp
        let mut cur = self.applied_gain_lr;
        if (cur[0] - target[0]).abs() > 1e-6 || (cur[1] - target[1]).abs() > 1e-6 {
            let coeff = (1.0 - (-1.0 / (0.005 * ctx.sample_rate.max(8_000.0))).exp()) as f32;
            for frame in buf.iter_mut() {
                cur[0] += coeff * (target[0] - cur[0]);
                cur[1] += coeff * (target[1] - cur[1]);
                frame[0] *= cur[0];
                frame[1] *= cur[1];
            }
            if (cur[0] - target[0]).abs() <= 1e-6 && (cur[1] - target[1]).abs() <= 1e-6 {
                cur = target;
            }
            self.applied_gain_lr = cur;
        } else if (target[0] - 1.0).abs() > 1e-7 || (target[1] - 1.0).abs() > 1e-7 {
            self.applied_gain_lr = target;
            for frame in buf.iter_mut() {
                frame[0] *= target[0];
                frame[1] *= target[1];
            }
        }

        // 6. Point 4: Post-Rack APDC (`[TPluginRackModule + 0x5c]`, `0x00cf2d00`)
        self.post_rack_apdc_delay.process_in_place(buf);

        // 7. Point 5: Prepare inter-track Send & Sidechain buffers (`[TPluginRackModule + 0x4c]`, `0x00cf2cb0`)
        for send in self.sends.iter_mut() {
            let sbuf = &mut send.send_buffer[..slice_len];
            let g = send.gain;
            for i in 0..slice_len {
                sbuf[i][0] = buf[i][0] * g;
                sbuf[i][1] = buf[i][1] * g;
            }
            send.send_apdc_delay.process_in_place(sbuf);
        }

        // 8. Update real-time Peak/RMS meters over `[0..end]` of `sub_block_buf` so multi-slice
        // PPQ blocks never overwrite and lose earlier sub-slice peaks (`+0x148`).
        let metered_frames = &self.sub_block_buf[..end];
        let mut peak_l = 0.0f32;
        let mut peak_r = 0.0f32;
        let mut sum_sq_l = 0.0f32;
        let mut sum_sq_r = 0.0f32;
        for frame in metered_frames.iter() {
            let al = frame[0].abs();
            let ar = frame[1].abs();
            if al > peak_l {
                peak_l = al;
            }
            if ar > peak_r {
                peak_r = ar;
            }
            sum_sq_l += frame[0] * frame[0];
            sum_sq_r += frame[1] * frame[1];
        }
        let is_active = peak_l > 1e-9 || peak_r > 1e-9;
        let state = if is_active {
            TRACK_STATE_ACTIVE
        } else {
            TRACK_STATE_SILENT
        };
        self.activity_state.store(state, Ordering::Relaxed);

        let inv_n = 1.0 / (end.max(1) as f32);
        let lin_to_db = |x: f32| -> f32 {
            if x <= 1e-9 {
                -144.0
            } else {
                20.0 * x.log10()
            }
        };
        self.meter = TrackMeterEntry {
            peak_l_db: lin_to_db(peak_l),
            peak_r_db: lin_to_db(peak_r),
            rms_l_db: lin_to_db((sum_sq_l * inv_n).sqrt()),
            rms_r_db: lin_to_db((sum_sq_r * inv_n).sqrt()),
            gr_db: min_gr_db,
            activity_flags: state,
        };
    }
}

// ============================================================================
// LOCK-FREE WORK-STEALING THREAD BANK (`TSharedThreadBank` `0x00584720`)
// ============================================================================

struct SharedLevelWorkGroup {
    track_indices: [AtomicUsize; MAX_MIXER_TRACKS],
    /// Single 64-bit atomic word packing `[epoch: u32 (bits 32..63) | task_count: u16 (bits 16..31) | next_cursor: u16 (bits 0..15)]`
    /// so task claiming via `compare_exchange_weak` is immune to epoch-tearing / ABA preemption races.
    queue_state: AtomicU64,
    remaining_tasks: AtomicUsize,
    tracks_base_ptr: AtomicUsize,
    ctx_sample_rate_bits: AtomicUsize,
    ctx_sample_pos: AtomicUsize,
    ctx_ppq_tick: AtomicUsize,
    ctx_bpm_bits: AtomicUsize,
    /// `bit 0` = `is_playing`, `bit 1` = `is_offline_rendering`, `bits 2..` = `ppq_resolution`.
    ctx_flags: AtomicUsize,
    slice_offset: AtomicUsize,
    slice_len: AtomicUsize,
    shutdown: AtomicBool,
}

impl SharedLevelWorkGroup {
    fn new() -> Self {
        Self {
            track_indices: std::array::from_fn(|_| AtomicUsize::new(0)),
            queue_state: AtomicU64::new(0),
            remaining_tasks: AtomicUsize::new(0),
            tracks_base_ptr: AtomicUsize::new(0),
            ctx_sample_rate_bits: AtomicUsize::new(48_000.0f64.to_bits() as usize),
            ctx_sample_pos: AtomicUsize::new(0),
            ctx_ppq_tick: AtomicUsize::new(0),
            ctx_bpm_bits: AtomicUsize::new(128.0f64.to_bits() as usize),
            ctx_flags: AtomicUsize::new(1),
            slice_offset: AtomicUsize::new(0),
            slice_len: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
        }
    }

    #[inline(always)]
    fn pack_queue_state(epoch: u32, count: u16, cursor: u16) -> u64 {
        ((epoch as u64) << 32) | ((count as u64) << 16) | (cursor as u64)
    }

    #[inline(always)]
    fn unpack_queue_state(state: u64) -> (u32, usize, usize) {
        let epoch = (state >> 32) as u32;
        let count = ((state >> 16) & 0xFFFF) as usize;
        let cursor = (state & 0xFFFF) as usize;
        (epoch, count, cursor)
    }

    /// Steals and executes one pending track from `expected_epoch` (`sub_584ed0`).
    /// Returns `true` if a task was atomically claimed and executed.
    ///
    /// # Safety
    /// All track indices in `track_indices[0..task_count]` belong to the SAME topological level
    /// (verified disjoint indices with zero edges between them), and each index `idx < task_count`
    /// is claimed by at most one thread via `queue_state.compare_exchange_weak`.
    #[inline]
    unsafe fn try_steal_and_execute(&self, expected_epoch: u32) -> bool {
        let mut q = self.queue_state.load(Ordering::Acquire);
        let claimed_idx = loop {
            let (epoch, count, cursor) = Self::unpack_queue_state(q);
            if epoch != expected_epoch || cursor >= count {
                return false;
            }
            match self.queue_state.compare_exchange_weak(
                q,
                q + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break cursor,
                Err(actual) => q = actual,
            }
        };

        let track_id = self.track_indices[claimed_idx].load(Ordering::Relaxed);
        let base_ptr = self.tracks_base_ptr.load(Ordering::Acquire) as *mut MixerTrack;
        let slice_offset = self.slice_offset.load(Ordering::Relaxed);
        let slice_len = self.slice_len.load(Ordering::Relaxed);
        let ctx = ProcessContext {
            sample_rate: f64::from_bits(self.ctx_sample_rate_bits.load(Ordering::Relaxed) as u64),
            current_sample_pos: self.ctx_sample_pos.load(Ordering::Relaxed) as i64,
            ppq_tick: self.ctx_ppq_tick.load(Ordering::Relaxed) as i64,
            bpm: f64::from_bits(self.ctx_bpm_bits.load(Ordering::Relaxed) as u64),
            slice_frames: slice_len,
            is_playing: (self.ctx_flags.load(Ordering::Relaxed) & 1) != 0,
            is_offline_rendering: (self.ctx_flags.load(Ordering::Relaxed) & 2) != 0,
            ppq_resolution: (self.ctx_flags.load(Ordering::Relaxed) >> 2) as u32,
        };
        unsafe {
            let track_ref = &mut *base_ptr.add(track_id);
            track_ref.render_slice(&ctx, slice_offset, slice_len);
        }
        self.remaining_tasks.fetch_sub(1, Ordering::Release);
        true
    }
}

/// Lock-Free Work-Stealing Thread Bank (`TSharedThreadBank` `VMT = 0x00584720` in `QuickFontCache_x64.dll`).
/// Executes independent tracks within a topological level across worker threads while the main
/// Audio Thread actively work-steals (`WaitForTaskGroup` `0x005850b0`) with Zero Heap Allocation.
pub struct LockFreeAudioThreadPool {
    shared: Arc<SharedLevelWorkGroup>,
    workers: Vec<JoinHandle<()>>,
}

impl LockFreeAudioThreadPool {
    pub fn new(num_workers: usize) -> Self {
        let shared = Arc::new(SharedLevelWorkGroup::new());
        let mut workers = Vec::with_capacity(num_workers);
        for worker_id in 0..num_workers {
            let sh = Arc::clone(&shared);
            let handle = thread::Builder::new()
                .name(format!("ToneForge-STBWorker-{worker_id}"))
                .spawn(move || {
                    let _pro_audio = crate::audio_io::ProAudioThreadGuard::enter();
                    let mut last_epoch = 0u32;
                    while !sh.shutdown.load(Ordering::Acquire) {
                        let (cur_epoch, count, cursor) =
                            SharedLevelWorkGroup::unpack_queue_state(sh.queue_state.load(Ordering::Acquire));
                        if cur_epoch != last_epoch && cursor < count {
                            while unsafe { sh.try_steal_and_execute(cur_epoch) } {}
                            last_epoch = cur_epoch;
                        } else {
                            std::hint::spin_loop();
                            thread::park_timeout(std::time::Duration::from_micros(250));
                        }
                    }
                })
                .expect("failed to spawn ToneForge STB worker thread");
            workers.push(handle);
        }
        Self { shared, workers }
    }

    /// Submits a topological level of disjoint track indices and participates in work-stealing
    /// on the calling thread until all tracks in the level complete (`0x00584f60` + `0x005850b0`).
    /// Zero Heap Allocation.
    #[inline]
    pub fn execute_level(
        &self,
        tracks: &mut [MixerTrack],
        level_tracks: &[usize],
        ctx: &ProcessContext,
        slice_offset: usize,
        slice_len: usize,
    ) {
        let m = level_tracks.len();
        if m == 0 {
            return;
        }
        if m == 1 || self.workers.is_empty() {
            for &u in level_tracks {
                tracks[u].render_slice(ctx, slice_offset, slice_len);
            }
            return;
        }

        for (i, &track_idx) in level_tracks.iter().enumerate() {
            self.shared.track_indices[i].store(track_idx, Ordering::Relaxed);
        }
        self.shared
            .tracks_base_ptr
            .store(tracks.as_mut_ptr() as usize, Ordering::Release);
        self.shared
            .ctx_sample_rate_bits
            .store(ctx.sample_rate.to_bits() as usize, Ordering::Relaxed);
        self.shared
            .ctx_sample_pos
            .store(ctx.current_sample_pos as usize, Ordering::Relaxed);
        self.shared
            .ctx_ppq_tick
            .store(ctx.ppq_tick as usize, Ordering::Relaxed);
        self.shared
            .ctx_bpm_bits
            .store(ctx.bpm.to_bits() as usize, Ordering::Relaxed);
        self.shared.ctx_flags.store(
            (ctx.is_playing as usize)
                | ((ctx.is_offline_rendering as usize) << 1)
                | ((ctx.ppq_resolution as usize) << 2),
            Ordering::Relaxed,
        );
        self.shared
            .slice_offset
            .store(slice_offset, Ordering::Relaxed);
        self.shared.slice_len.store(slice_len, Ordering::Relaxed);
        self.shared.remaining_tasks.store(m, Ordering::Release);

        let (prev_epoch, _, _) =
            SharedLevelWorkGroup::unpack_queue_state(self.shared.queue_state.load(Ordering::Relaxed));
        let new_epoch = prev_epoch.wrapping_add(1);
        let new_state = SharedLevelWorkGroup::pack_queue_state(new_epoch, m as u16, 0);
        self.shared.queue_state.store(new_state, Ordering::Release);

        for w in &self.workers {
            w.thread().unpark();
        }

        // Calling Audio Thread actively steals tasks (`WaitForTaskGroup` `0x005850b0`)
        while unsafe { self.shared.try_steal_and_execute(new_epoch) } {}

        // Spin-wait until any in-flight worker finishes its last stolen track
        while self.shared.remaining_tasks.load(Ordering::Acquire) > 0 {
            std::hint::spin_loop();
        }
    }
}

impl Drop for LockFreeAudioThreadPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        for w in &self.workers {
            w.thread().unpark();
        }
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

// ============================================================================
// TOPOLOGICAL MIXER SCHEDULER (`sub_cf42f0` & 5-POINT APDC SOLVER)
// ============================================================================

/// Pre-allocated flat topology & Kahn/BFS scratch buffers so live real-time routing mutations
/// (`AddOrUpdateSendRoute`, `RemoveSendRoute`, `SetOrReplacePluginSlot`, `UpsertChannel`, etc.)
/// execute full Kahn topological sorting, cycle detection, and 5-Point APDC with **0 allocations and 0 drops**.
#[derive(Clone, Debug)]
struct RtTopologyScratch {
    flat_levels: Vec<usize>,
    level_ranges: Vec<(usize, usize)>,
    flat_active_levels: Vec<usize>,
    active_level_ranges: Vec<(usize, usize)>,
    use_rt_flat_levels: bool,
    in_degree: Vec<u16>,
    out_degree: Vec<u16>,
    has_predecessor: Vec<bool>,
    adj_flat: Vec<u16>,
    adj_len: Vec<u16>,
    curr_queue: Vec<u16>,
    next_queue: Vec<u16>,
    bfs_visited: Vec<bool>,
    bfs_queue: Vec<u16>,
}

impl RtTopologyScratch {
    fn new() -> Self {
        Self {
            flat_levels: Vec::with_capacity(MAX_MIXER_TRACKS),
            level_ranges: Vec::with_capacity(MAX_MIXER_TRACKS),
            flat_active_levels: Vec::with_capacity(MAX_MIXER_TRACKS),
            active_level_ranges: Vec::with_capacity(MAX_MIXER_TRACKS),
            use_rt_flat_levels: false,
            in_degree: vec![0u16; MAX_MIXER_TRACKS],
            out_degree: vec![0u16; MAX_MIXER_TRACKS],
            has_predecessor: vec![false; MAX_MIXER_TRACKS],
            adj_flat: vec![0u16; MAX_MIXER_TRACKS * MAX_SENDS_PER_TRACK],
            adj_len: vec![0u16; MAX_MIXER_TRACKS],
            curr_queue: vec![0u16; MAX_MIXER_TRACKS],
            next_queue: vec![0u16; MAX_MIXER_TRACKS],
            bfs_visited: vec![false; MAX_MIXER_TRACKS],
            bfs_queue: vec![0u16; MAX_MIXER_TRACKS],
        }
    }
}

/// Topological Level-Order Mixer DAG Scheduler (`sub_cf42f0` at `0x00cf42f0` in `FLEngine_x64.dll`).
#[derive(Clone, Debug)]
pub struct TopologicalMixerScheduler {
    /// Topological levels (`0x15e06f8`, level count at `0x16b8dcc`): tracks in the same level
    /// have zero dependencies on each other and execute in parallel.
    pub levels: Vec<Vec<usize>>,
    /// Active topological levels (excluding completely unrouted idle tracks) for fast real-time dispatch.
    pub active_levels: Vec<Vec<usize>>,
    /// Total compensated latency at Master Track 0 (`tracks[0].accumulated_latency`).
    pub master_output_latency_samples: usize,
    rt_scratch: RtTopologyScratch,
}

impl TopologicalMixerScheduler {
    #[inline]
    pub fn num_levels(&self) -> usize {
        if self.rt_scratch.use_rt_flat_levels {
            self.rt_scratch.level_ranges.len()
        } else {
            self.levels.len()
        }
    }

    #[inline]
    pub fn level_at(&self, level_idx: usize) -> &[usize] {
        if self.rt_scratch.use_rt_flat_levels {
            let (start, end) = self.rt_scratch.level_ranges[level_idx];
            &self.rt_scratch.flat_levels[start..end]
        } else {
            &self.levels[level_idx]
        }
    }

    #[inline]
    pub fn num_active_levels(&self) -> usize {
        if self.rt_scratch.use_rt_flat_levels {
            self.rt_scratch.active_level_ranges.len()
        } else {
            self.active_levels.len()
        }
    }

    #[inline]
    pub fn active_level_at(&self, level_idx: usize) -> &[usize] {
        if self.rt_scratch.use_rt_flat_levels {
            let (start, end) = self.rt_scratch.active_level_ranges[level_idx];
            &self.rt_scratch.flat_active_levels[start..end]
        } else {
            &self.active_levels[level_idx]
        }
    }

    /// Zero-allocation RT BFS cycle check (`0` heap alloc, `0` heap free).
    #[inline]
    pub fn would_create_cycle_rt(
        &mut self,
        tracks: &[MixerTrack],
        src_track: usize,
        dst_track: usize,
    ) -> bool {
        let n = tracks.len().min(MAX_MIXER_TRACKS);
        if src_track >= n || dst_track >= n {
            return false;
        }
        if src_track == dst_track {
            return true;
        }
        let visited = &mut self.rt_scratch.bfs_visited[..n];
        visited.fill(false);
        let queue = &mut self.rt_scratch.bfs_queue[..n];
        visited[dst_track] = true;
        queue[0] = dst_track as u16;
        let mut head = 0usize;
        let mut tail = 1usize;
        while head < tail {
            let u = queue[head] as usize;
            head += 1;
            for route in &tracks[u].sends {
                let v = route.dest_track_index;
                if v == src_track {
                    return true;
                }
                if v < n && !visited[v] {
                    visited[v] = true;
                    queue[tail] = v as u16;
                    tail += 1;
                }
            }
        }
        false
    }

    /// Zero-allocation, Zero-drop Real-Time Kahn Topological Sort + 5-Point APDC Solver.
    /// Returns `true` if `tracks` is acyclic and `flat_levels` / `flat_active_levels` + APDC were updated.
    pub fn rebuild_levels_and_apdc_rt(
        &mut self,
        tracks: &mut [MixerTrack],
        channels: &mut [ChannelGenerator],
    ) -> bool {
        let n = tracks.len().min(MAX_MIXER_TRACKS);
        {
            let sc = &mut self.rt_scratch;
            sc.in_degree[..n].fill(0);
            sc.out_degree[..n].fill(0);
            sc.has_predecessor[..n].fill(false);
            sc.adj_len[..n].fill(0);

            for t in 0..n {
                tracks[t].has_channel_input = false;
                let adj_base = t * MAX_SENDS_PER_TRACK;
                for route in &tracks[t].sends {
                    let dst = route.dest_track_index;
                    if dst < n && dst != t {
                        let cur_len = sc.adj_len[t] as usize;
                        if cur_len < MAX_SENDS_PER_TRACK {
                            let mut dup = false;
                            for k in 0..cur_len {
                                if sc.adj_flat[adj_base + k] as usize == dst {
                                    dup = true;
                                    break;
                                }
                            }
                            if !dup {
                                sc.adj_flat[adj_base + cur_len] = dst as u16;
                                sc.adj_len[t] = (cur_len + 1) as u16;
                                sc.in_degree[dst] += 1;
                                sc.out_degree[t] += 1;
                                sc.has_predecessor[dst] = true;
                            }
                        }
                    }
                }
            }

            for ch in channels.iter() {
                if ch.enabled && ch.target_mixer_track < n {
                    tracks[ch.target_mixer_track].has_channel_input = true;
                }
            }

            let mut curr_len = 0usize;
            for i in 0..n {
                if sc.in_degree[i] == 0 {
                    sc.curr_queue[curr_len] = i as u16;
                    curr_len += 1;
                }
            }

            // First dry-run check via `in_degree` & temporary flat_levels write; since `flat_levels` is
            // Copy (`Vec<usize>`), `.clear()` never deallocates.
            sc.flat_levels.clear();
            sc.level_ranges.clear();
            sc.flat_active_levels.clear();
            sc.active_level_ranges.clear();

            let mut visited_count = 0usize;
            while curr_len > 0 {
                visited_count += curr_len;
                let lvl_start = sc.flat_levels.len();
                let act_start = sc.flat_active_levels.len();
                let mut next_len = 0usize;

                for q_idx in 0..curr_len {
                    let u = sc.curr_queue[q_idx] as usize;
                    sc.flat_levels.push(u);
                    if u == MASTER_TRACK_INDEX
                        || u == PREVIEW_TRACK_INDEX
                        || sc.out_degree[u] > 0
                        || sc.has_predecessor[u]
                        || tracks[u].has_channel_input
                        || tracks[u].has_hw_input
                        || tracks[u].input_source_bus.is_some()
                        || tracks[u].record_armed
                        || !tracks[u].slots.is_empty()
                    {
                        sc.flat_active_levels.push(u);
                    }

                    let adj_base = u * MAX_SENDS_PER_TRACK;
                    let deg_out = sc.adj_len[u] as usize;
                    for k in 0..deg_out {
                        let v = sc.adj_flat[adj_base + k] as usize;
                        sc.in_degree[v] -= 1;
                        if sc.in_degree[v] == 0 {
                            sc.next_queue[next_len] = v as u16;
                            next_len += 1;
                        }
                    }
                }

                let lvl_end = sc.flat_levels.len();
                sc.level_ranges.push((lvl_start, lvl_end));
                let act_end = sc.flat_active_levels.len();
                if act_end > act_start {
                    sc.active_level_ranges.push((act_start, act_end));
                }

                sc.curr_queue[..next_len].copy_from_slice(&sc.next_queue[..next_len]);
                curr_len = next_len;
            }

            if visited_count < n {
                // Restore fallback to pre-existing `self.levels` if a cycle ever slipped through
                sc.use_rt_flat_levels = false;
                return false;
            }
            sc.use_rt_flat_levels = true;
        }

        self.recompute_apdc_rt(tracks, channels);
        true
    }

    /// Checks whether adding a directed route `src_track -> dst_track` would introduce a cycle.
    pub fn would_create_cycle(tracks: &[MixerTrack], src_track: usize, dst_track: usize) -> bool {
        let n = tracks.len();
        if src_track >= n || dst_track >= n {
            return false;
        }
        if src_track == dst_track {
            return true;
        }
        // BFS from `dst_track` to see if `src_track` is reachable
        let mut visited = vec![false; n];
        let mut queue = Vec::with_capacity(n);
        visited[dst_track] = true;
        queue.push(dst_track);
        let mut head = 0usize;
        while head < queue.len() {
            let u = queue[head];
            head += 1;
            for route in &tracks[u].sends {
                let v = route.dest_track_index;
                if v == src_track {
                    return true;
                }
                if v < n && !visited[v] {
                    visited[v] = true;
                    queue.push(v);
                }
            }
        }
        false
    }

    /// Adds a route `src_track -> dst_track` only if it does not introduce a cycle, then rebuilds
    /// topological levels and 5-Point APDC.
    pub fn add_route_checked(
        tracks: &mut [MixerTrack],
        src_track: usize,
        dest_track: usize,
        gain: f32,
        is_sidechain_only: bool,
    ) -> Result<Self, GraphCycleError> {
        if Self::would_create_cycle(tracks, src_track, dest_track) {
            return Err(GraphCycleError {
                cyclic_tracks: vec![src_track, dest_track],
            });
        }
        tracks[src_track].add_send(dest_track, gain, is_sidechain_only);
        Self::rebuild_levels_and_apdc_checked(tracks, &mut [])
    }

    /// Rebuilds the Kahn topological level order and solves all 5 APDC points across `tracks`
    /// (compatible with the direct signature in `04_flstudio26_daw_architecture.md`).
    pub fn rebuild_levels_and_apdc(tracks: &mut [MixerTrack]) -> Self {
        Self::rebuild_levels_and_apdc_checked(tracks, &mut [])
            .expect("Mixer routing graph contains a cycle")
    }

    /// Recomputes all 5 APDC points across `channels` and `tracks` in-place using the existing
    /// topological `levels` without any heap allocation (safe to call on the real-time audio thread
    /// when a slot is bypassed/enabled or a plugin's latency changes dynamically).
    pub fn recompute_apdc_rt(
        &mut self,
        tracks: &mut [MixerTrack],
        channels: &mut [ChannelGenerator],
    ) {
        let n = tracks.len().min(MAX_MIXER_TRACKS);
        let mut max_incoming_pred_latency = [0usize; MAX_MIXER_TRACKS];
        let mut has_predecessor = [false; MAX_MIXER_TRACKS];

        for t in 0..n {
            tracks[t].has_channel_input = false;
        }

        let mut max_channel_latency = 0usize;
        for ch in channels.iter() {
            if ch.enabled && ch.target_mixer_track < n {
                max_channel_latency = max_channel_latency.max(ch.generator.latency_samples());
                tracks[ch.target_mixer_track].has_channel_input = true;
            }
        }

        let mut max_hw_latency = 0usize;
        for t in 0..n {
            if tracks[t].has_hw_input {
                max_hw_latency = max_hw_latency.max(tracks[t].hw_input_latency_samples);
            }
        }
        let global_source_floor = max_channel_latency.max(max_hw_latency);

        // Determine raw direct source latency entering Point 2 on each track
        for t in 0..n {
            let raw_direct = if tracks[t].has_hw_input {
                tracks[t].hw_input_latency_samples
            } else if tracks[t].has_channel_input {
                max_channel_latency
            } else {
                tracks[t].hw_input_latency_samples
            };
            tracks[t].direct_source_latency_samples = raw_direct;
        }

        // Traverse precomputed topological levels to solve Point 2 and Point 3
        let total_levels = self.num_levels();
        for lvl_idx in 0..total_levels {
            let lvl_len = self.level_at(lvl_idx).len();
            for item_idx in 0..lvl_len {
                let u = self.level_at(lvl_idx)[item_idx];
                if u >= n {
                    continue;
                }
                let has_direct_src = tracks[u].has_channel_input
                    || tracks[u].has_hw_input
                    || tracks[u].hw_input_latency_samples > 0
                    || !has_predecessor[u];

                let direct_target = if has_direct_src {
                    global_source_floor.max(tracks[u].direct_source_latency_samples)
                } else {
                    0
                };

                let input_arrival_u = max_incoming_pred_latency[u].max(direct_target);
                tracks[u].input_arrival_latency = input_arrival_u;

                // Point 2: Track Hardware / Direct Input APDC (`input_apdc_delay`)
                let p2_slack = if has_direct_src {
                    input_arrival_u.saturating_sub(tracks[u].direct_source_latency_samples)
                } else {
                    0
                };
                tracks[u].input_apdc_delay.set_delay_samples(p2_slack);

                // Point 3: Configure Dry/Wet Parallel APDC on each enabled FX slot
                let mut internal_rack_lat = 0usize;
                for slot in &mut tracks[u].slots {
                    if slot.enabled {
                        let s_lat = slot.plugin.latency_samples();
                        slot.dry_apdc_delay.set_delay_samples(s_lat);
                        internal_rack_lat += s_lat;
                    } else {
                        slot.dry_apdc_delay.set_delay_samples(0);
                    }
                }

                let rack_out_u = input_arrival_u + internal_rack_lat;
                tracks[u].rack_output_latency = rack_out_u;
                tracks[u].accumulated_latency = rack_out_u;

                let num_sends = tracks[u].sends.len();
                for s_idx in 0..num_sends {
                    let v = tracks[u].sends[s_idx].dest_track_index;
                    if v < n && v != u {
                        has_predecessor[v] = true;
                        max_incoming_pred_latency[v] = max_incoming_pred_latency[v].max(rack_out_u);
                    }
                }
            }
        }

        // Point 1: Configure `channel_apdc_delay` on every `ChannelGenerator`
        for ch in channels.iter_mut() {
            if ch.enabled && ch.target_mixer_track < n {
                let gen_lat = ch.generator.latency_samples();
                let dst = ch.target_mixer_track;
                let p1_target = if tracks[dst].has_hw_input {
                    tracks[dst].input_arrival_latency
                } else {
                    max_channel_latency
                };
                let p1_slack = p1_target.saturating_sub(gen_lat);
                ch.channel_apdc_delay.set_delay_samples(p1_slack);
            } else {
                ch.channel_apdc_delay.set_delay_samples(0);
            }
        }

        // Solve Point 4 (Post-Rack Common APDC) and Point 5 (Inter-Track Per-Send/Sidechain APDC)
        for u in 0..n {
            let rack_out_u = tracks[u].rack_output_latency;
            if tracks[u].sends.is_empty() {
                tracks[u].post_rack_apdc_delay.set_delay_samples(0);
                tracks[u].accumulated_latency = rack_out_u;
                continue;
            }

            let mut min_downstream_slack = usize::MAX;
            for send in &tracks[u].sends {
                let dst = send.dest_track_index;
                if dst < n {
                    let target_lat = tracks[dst].input_arrival_latency;
                    let total_needed = target_lat.saturating_sub(rack_out_u);
                    min_downstream_slack = min_downstream_slack.min(total_needed);
                }
            }
            if min_downstream_slack == usize::MAX {
                min_downstream_slack = 0;
            }

            // Point 4: Post-Rack APDC (`[TPluginRackModule + 0x5c]`)
            tracks[u]
                .post_rack_apdc_delay
                .set_delay_samples(min_downstream_slack);

            let post4_lat_u = rack_out_u + min_downstream_slack;
            tracks[u].accumulated_latency = post4_lat_u;

            // Point 5: Residual Per-Route Inter-Track Send / Sidechain APDC (`[TPluginRackModule + 0x4c]`)
            for send in &mut tracks[u].sends {
                let dst = send.dest_track_index;
                if dst < n {
                    let target_lat = tracks[dst].input_arrival_latency;
                    let residual_slack = target_lat.saturating_sub(post4_lat_u);
                    send.send_apdc_delay.set_delay_samples(residual_slack);
                }
            }
        }

        self.master_output_latency_samples = tracks
            .get(MASTER_TRACK_INDEX)
            .map(|t| t.accumulated_latency)
            .unwrap_or(0);
    }

    /// Full Kahn topological sort with cycle detection (`Err(GraphCycleError)`) and 5-Point APDC
    /// solver across both `ChannelGenerator`s (Point 1) and `MixerTrack`s (Points 2, 3, 4, 5).
    pub fn rebuild_levels_and_apdc_checked(
        tracks: &mut [MixerTrack],
        channels: &mut [ChannelGenerator],
    ) -> Result<Self, GraphCycleError> {
        let n = tracks.len();
        let mut in_degree = vec![0usize; n];
        let mut adj = vec![Vec::<usize>::new(); n];
        let mut out_degree = vec![0usize; n];
        let mut has_predecessor = vec![false; n];

        for t in 0..n {
            tracks[t].has_channel_input = false;
            for route in &tracks[t].sends {
                let dst = route.dest_track_index;
                if dst < n && dst != t {
                    if !adj[t].contains(&dst) {
                        adj[t].push(dst);
                        in_degree[dst] += 1;
                        out_degree[t] += 1;
                        has_predecessor[dst] = true;
                    }
                }
            }
        }

        for ch in channels.iter() {
            if ch.enabled && ch.target_mixer_track < n {
                tracks[ch.target_mixer_track].has_channel_input = true;
            }
        }

        // Kahn's topological level-order traversal
        let mut current_level: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut levels: Vec<Vec<usize>> = Vec::new();
        let mut visited_count = 0usize;

        while !current_level.is_empty() {
            visited_count += current_level.len();
            let mut next_level = Vec::new();

            for &u in &current_level {
                for &v in &adj[u] {
                    in_degree[v] -= 1;
                    if in_degree[v] == 0 {
                        next_level.push(v);
                    }
                }
            }

            levels.push(current_level);
            current_level = next_level;
        }

        // Cycle check: if any track still has `in_degree > 0`, a directed cycle exists!
        if visited_count < n {
            let cyclic_tracks: Vec<usize> = (0..n).filter(|&i| in_degree[i] > 0).collect();
            return Err(GraphCycleError { cyclic_tracks });
        }

        // Build `active_levels`: tracks that participate in routing, have plugins/inputs, or are Master/Preview
        let mut active_levels = Vec::with_capacity(levels.len());
        for lvl in &levels {
            let active_in_lvl: Vec<usize> = lvl
                .iter()
                .copied()
                .filter(|&u| {
                    u == MASTER_TRACK_INDEX
                        || u == PREVIEW_TRACK_INDEX
                        || out_degree[u] > 0
                        || has_predecessor[u]
                        || tracks[u].has_channel_input
                        || tracks[u].has_hw_input
                        || tracks[u].input_source_bus.is_some()
                        || tracks[u].record_armed
                        || !tracks[u].slots.is_empty()
                })
                .collect();
            if !active_in_lvl.is_empty() {
                active_levels.push(active_in_lvl);
            }
        }

        let mut rt_scratch = RtTopologyScratch::new();
        for lvl in &levels {
            let s = rt_scratch.flat_levels.len();
            rt_scratch.flat_levels.extend_from_slice(lvl);
            let e = rt_scratch.flat_levels.len();
            rt_scratch.level_ranges.push((s, e));
        }
        for act in &active_levels {
            let s = rt_scratch.flat_active_levels.len();
            rt_scratch.flat_active_levels.extend_from_slice(act);
            let e = rt_scratch.flat_active_levels.len();
            rt_scratch.active_level_ranges.push((s, e));
        }

        let mut sched = Self {
            levels,
            active_levels,
            master_output_latency_samples: 0,
            rt_scratch,
        };

        // Compute all 5 APDC points
        sched.recompute_apdc_rt(tracks, channels);

        // Ensure delay line ring capacities are sufficient when building/rebuilding graph topology
        for ch in channels.iter_mut() {
            let d = ch.channel_apdc_delay.delay_samples();
            ch.channel_apdc_delay
                .ensure_capacity(d.max(DEFAULT_APDC_CAPACITY));
        }
        for u in 0..n {
            let d2 = tracks[u].input_apdc_delay.delay_samples();
            tracks[u]
                .input_apdc_delay
                .ensure_capacity(d2.max(DEFAULT_APDC_CAPACITY));
            for slot in &mut tracks[u].slots {
                let d3 = slot.dry_apdc_delay.delay_samples();
                slot.dry_apdc_delay
                    .ensure_capacity(d3.max(DEFAULT_APDC_CAPACITY));
            }
            let d4 = tracks[u].post_rack_apdc_delay.delay_samples();
            tracks[u]
                .post_rack_apdc_delay
                .ensure_capacity(d4.max(DEFAULT_APDC_CAPACITY));
            for send in &mut tracks[u].sends {
                let d5 = send.send_apdc_delay.delay_samples();
                send.send_apdc_delay
                    .ensure_capacity(d5.max(DEFAULT_APDC_CAPACITY));
            }
        }
        // Re-apply delay sample offsets in case `ensure_capacity` resized any ring buffer
        sched.recompute_apdc_rt(tracks, channels);

        Ok(sched)
    }

    /// Executes a sub-block slice `[slice_offset..slice_offset + slice_len]` across all topological levels
    /// (`sub_cf42f0`), routing Point-5-compensated Sends and Sidechains between levels.
    /// Zero Heap Allocation.
    pub fn process_slice_levels(
        &self,
        ctx: &ProcessContext,
        tracks: &mut [MixerTrack],
        thread_pool: Option<&LockFreeAudioThreadPool>,
        slice_offset: usize,
        slice_len: usize,
        process_all_502_tracks: bool,
    ) {
        let level_count = if process_all_502_tracks {
            self.num_levels()
        } else {
            self.num_active_levels()
        };

        for lvl_idx in 0..level_count {
            let level_tracks = if process_all_502_tracks {
                self.level_at(lvl_idx)
            } else {
                self.active_level_at(lvl_idx)
            };

            // Step A: Render all independent tracks in `level_tracks` (parallel or inline)
            if let Some(pool) = thread_pool {
                pool.execute_level(tracks, level_tracks, ctx, slice_offset, slice_len);
            } else {
                for &u in level_tracks {
                    tracks[u].render_slice(ctx, slice_offset, slice_len);
                }
            }

            // Step B: Accumulate Point-5-compensated `send_buffer`s into destination tracks
            // (all destinations `dst` belong to strictly later levels `> k`).
            let end = slice_offset + slice_len;
            for &u in level_tracks {
                let num_sends = tracks[u].sends.len();
                for s_idx in 0..num_sends {
                    let dst = tracks[u].sends[s_idx].dest_track_index;
                    let is_sc = tracks[u].sends[s_idx].is_sidechain_only;
                    if dst < tracks.len() && dst != u {
                        // Split borrow between `tracks[u]` and `tracks[dst]` safely via indices
                        let (src_track, dst_track) = if u < dst {
                            let (left, right) = tracks.split_at_mut(dst);
                            (&left[u], &mut right[0])
                        } else {
                            let (left, right) = tracks.split_at_mut(u);
                            (&right[0], &mut left[dst])
                        };
                        let sbuf = &src_track.sends[s_idx].send_buffer[..slice_len];
                        if is_sc {
                            let sc_dst = &mut dst_track.sidechain_buf[slice_offset..end];
                            for i in 0..slice_len {
                                sc_dst[i][0] += sbuf[i][0];
                                sc_dst[i][1] += sbuf[i][1];
                            }
                            dst_track.has_sidechain_input = true;
                        } else {
                            let up_dst = &mut dst_track.upstream_mix_buf[slice_offset..end];
                            for i in 0..slice_len {
                                up_dst[i][0] += sbuf[i][0];
                                up_dst[i][1] += sbuf[i][1];
                            }
                            dst_track.has_upstream_audio = true;
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// UNIFIED 502-TRACK REAL-TIME MIXER ENGINE (`sub_d0ece0` + `sub_d0d4c0`)
// ============================================================================

struct MixerSliceRefs<'a> {
    sequencer: &'a mut PlaylistSequencer,
    automation_queue: &'a mut SampleAccurateAutomationQueue,
    controller_bus: &'a mut InternalControllerBus,
    channels: &'a mut Vec<ChannelGenerator>,
    tracks: &'a mut Vec<MixerTrack>,
    scheduler: &'a mut TopologicalMixerScheduler,
    thread_pool: Option<&'a LockFreeAudioThreadPool>,
}

impl<'a> MixerSliceRefs<'a> {
    #[inline]
    fn recompute_apdc_rt(&mut self) {
        self.scheduler
            .recompute_apdc_rt(self.tracks, self.channels);
    }

    #[inline]
    fn propagate_internal_controllers(&mut self) {
        let links = self.controller_bus.active_links();
        let mut latency_changed = false;
        for i in 0..links.len() {
            let link = links[i];
            let fixed_val = self.controller_bus.get_fixed30(link.controller_id);
            let norm_val = link.map_fixed30_to_norm(fixed_val);
            if let Some(track) = self.tracks.get_mut(link.target_track as usize) {
                if let Some(slot) = track.slots.get_mut(link.slot_index as usize) {
                    let prev_lat = slot.plugin.latency_samples();
                    slot.plugin.set_parameter_normalized(link.param_id, norm_val);
                    if slot.plugin.latency_samples() != prev_lat {
                        latency_changed = true;
                    }
                }
            }
        }
        if latency_changed {
            self.recompute_apdc_rt();
        }
    }
}

/// Complete 502-Track Real-Time Audio Engine (`FLEngine_x64.dll` `sub_d0ece0` & `sub_d0d4c0`).
/// Integrates `HybridPpqSlicer`, `SampleAccurateAutomationQueue`, `InternalControllerBus`,
/// `ChannelGenerator`s (Phase 1 + Point 1 APDC), 502 `MixerTrack`s (Phase 2 + Point 2 APDC,
/// Phase 3 + Points 3/4/5 APDC), lock-free `rtrb` command + `RtGraphMutation` ingestion,
/// `toneforge-rt-gc` background retirement, and 128 KB `TripleBuffer` telemetry.
pub struct MixerEngine {
    pub slicer: HybridPpqSlicer,
    pub sequencer: PlaylistSequencer,
    pub automation_queue: SampleAccurateAutomationQueue,
    pub controller_bus: InternalControllerBus,
    pub channels: Vec<ChannelGenerator>,
    pub tracks: Vec<MixerTrack>,
    pub scheduler: TopologicalMixerScheduler,
    pub thread_pool: Option<LockFreeAudioThreadPool>,
    pub cmd_consumer: rtrb::Consumer<EngineCommand>,
    pub mutation_consumer: Option<rtrb::Consumer<RtGraphMutation>>,
    pub gc_producer: Option<rtrb::Producer<RetiredRtResource>>,
    pub telemetry_input: triple_buffer::Input<VisualTelemetryFrame>,
    pub hw_capture_scratch: Vec<StereoFrame>,
    pub record_tap: Option<crate::audio_io::RtRecordTap>,
    pub hw_capture_bridge:
        Option<(rtrb::Consumer<StereoFrame>, crate::audio_io::AdaptiveDllDriftResampler)>,
    pub midi_in_consumer: Option<rtrb::Consumer<SequencerVoiceEvent>>,
    pub is_playing: bool,
    pub process_all_502_tracks: bool,
    /// Set by offline render so every `ProcessContext` reports `is_offline_rendering == true`.
    pub is_offline_rendering: bool,
    pub last_hw_input_peak: f32,
    hw_input_injected_this_block: bool,
    frame_seq: u64,
    audition_note_seq: u32,
    /// Length (frames) of the de-click fade applied when the transport stops.
    stop_fade_total: u32,
    stop_fade_remaining: u32,
    stop_request_pos: (i64, i64),
    /// Track whose plugins own the shared Pro-Q 4 / Soothe3 telemetry slots.
    pub telemetry_focus_track: usize,
    /// Exponentially averaged measured DSP load (% of the real-time block budget).
    dsp_load_ema_pct: f32,
    /// 4x true-peak interpolator history of the master output and the last block's peak (linear).
    master_tp_hist: [[f32; 12]; 2],
    master_tp_block_lin: [f32; 2],
    /// `midi_arrival_clock_ns` at the start of the previous block: live MIDI that arrived during
    /// that block plays at the same offset in this one.
    pub midi_prev_block_ns: i64,
}

/// ITU-R BS.1770-4 Annex 2 4-phase x 12-tap interpolation filter (4x oversampled true peak of the
/// master bus, used when no true-peak meter runs on the master chain).
const MASTER_TRUE_PEAK_TAPS: [[f32; 12]; 4] = [
    [0.0017089844, 0.010986328, -0.01965332, 0.033203125, -0.059448242, 0.1373291, 0.97216797, -0.10229492, 0.047607422, -0.026611328, 0.014892578, -0.008300781],
    [-0.029174805, 0.029296875, -0.051757812, 0.08911133, -0.1665039, 0.4650879, 0.77978516, -0.20031738, 0.09082031, -0.05029297, 0.028076172, -0.015625],
    [-0.018920898, 0.033081055, -0.05822754, 0.1015625, -0.20031738, 0.77978516, 0.4650879, -0.1665039, 0.08911133, -0.051757812, 0.029296875, -0.029174805],
    [-0.008300781, 0.014892578, -0.026611328, 0.047607422, -0.10229492, 0.97216797, 0.1373291, -0.059448242, 0.033203125, -0.01965332, 0.010986328, 0.0017089844],
];

impl MixerEngine {
    /// Creates a new 502-Track `MixerEngine` along with its UI/Host handles:
    /// `(MixerEngine, rtrb::Producer<EngineCommand>, triple_buffer::Output<VisualTelemetryFrame>)`.
    pub fn new(
        sample_rate: f64,
        bpm: f64,
        worker_threads: usize,
    ) -> (
        Self,
        rtrb::Producer<EngineCommand>,
        triple_buffer::Output<VisualTelemetryFrame>,
    ) {
        let mut tracks = Vec::with_capacity(MAX_MIXER_TRACKS);
        for idx in 0..MAX_MIXER_TRACKS {
            tracks.push(MixerTrack::new(idx));
        }
        let mut channels = Vec::with_capacity(MAX_RT_CHANNELS);
        let scheduler =
            TopologicalMixerScheduler::rebuild_levels_and_apdc_checked(&mut tracks, &mut channels)
                .expect("initial 502 tracks are acyclic");
        let thread_pool = if worker_threads > 0 {
            Some(LockFreeAudioThreadPool::new(worker_threads))
        } else {
            None
        };
        let (cmd_producer, cmd_consumer) = create_command_ring_buffer(DEFAULT_COMMAND_RING_CAPACITY);
        let (telemetry_input, telemetry_output) = create_telemetry_triple_buffer();

        let engine = Self {
            slicer: HybridPpqSlicer::new(sample_rate, bpm, DEFAULT_PPQ),
            sequencer: PlaylistSequencer::new(),
            automation_queue: SampleAccurateAutomationQueue::default(),
            controller_bus: InternalControllerBus::new(),
            channels,
            tracks,
            scheduler,
            thread_pool,
            cmd_consumer,
            mutation_consumer: None,
            gc_producer: None,
            telemetry_input,
            hw_capture_scratch: vec![[0.0; 2]; MAX_BLOCK_SIZE],
            record_tap: None,
            hw_capture_bridge: None,
            midi_in_consumer: None,
            is_playing: true,
            process_all_502_tracks: false,
            is_offline_rendering: false,
            last_hw_input_peak: 0.0,
            hw_input_injected_this_block: false,
            frame_seq: 0,
            audition_note_seq: 1,
            stop_fade_total: 0,
            stop_fade_remaining: 0,
            stop_request_pos: (0, 0),
            telemetry_focus_track: FIRST_INSERT_TRACK_INDEX,
            dsp_load_ema_pct: 0.0,
            master_tp_hist: [[0.0; 12]; 2],
            master_tp_block_lin: [0.0; 2],
            midi_prev_block_ns: crate::audio_io::midi_arrival_clock_ns(),
        };
        (engine, cmd_producer, telemetry_output)
    }

    /// Creates a new 502-Track `MixerEngine` pre-wired with lock-free `RtGraphMutation` ingestion
    /// and a running `RtGarbageCollector` (`toneforge-rt-gc`) so the RT thread achieves
    /// **0 Mutex locks, 0 allocations, and 0 deallocations/drops**.
    pub fn new_with_rt_gc(
        sample_rate: f64,
        bpm: f64,
        worker_threads: usize,
    ) -> (
        Self,
        rtrb::Producer<EngineCommand>,
        rtrb::Producer<RtGraphMutation>,
        RtGarbageCollector,
        triple_buffer::Output<VisualTelemetryFrame>,
    ) {
        let (mut engine, cmd_producer, telemetry_output) =
            Self::new(sample_rate, bpm, worker_threads);
        let (mut_prod, mut_cons, gc_prod, gc_worker) = create_rt_mutation_channels(
            DEFAULT_RT_MUTATION_RING_CAPACITY,
            DEFAULT_RT_GC_RING_CAPACITY,
        );
        engine.attach_rt_mutation_channels(mut_cons, gc_prod);
        (
            engine,
            cmd_producer,
            mut_prod,
            gc_worker,
            telemetry_output,
        )
    }

    /// Attaches lock-free `RtGraphMutation` and `RetiredRtResource` SPSC rings to this `MixerEngine`.
    pub fn attach_rt_mutation_channels(
        &mut self,
        mutation_consumer: rtrb::Consumer<RtGraphMutation>,
        gc_producer: rtrb::Producer<RetiredRtResource>,
    ) {
        if self.channels.capacity() < MAX_RT_CHANNELS {
            self.channels
                .reserve(MAX_RT_CHANNELS - self.channels.len());
        }
        self.mutation_consumer = Some(mutation_consumer);
        self.gc_producer = Some(gc_producer);
    }

    #[inline(always)]
    fn is_stem_generator_name(name: &str) -> bool {
        matches!(
            name,
            "UVR5 Separated Stem Playback Generator"
                | "Custom Audio Stem"
                | "Vocals"
                | "Drums"
                | "Bass"
                | "Other"
                | "FL26 Sample Kit"
        )
    }

    /// Drains pending lock-free [`RtGraphMutation`] messages with **0 heap allocations** and
    /// **0 heap frees/drops** on the Real-Time audio thread.
    #[inline]
    pub fn drain_rt_mutations(&mut self) {
        if self.mutation_consumer.is_none() || self.gc_producer.is_none() {
            return;
        }
        let mut topology_dirty = false;

        loop {
            let gc_ready = self
                .gc_producer
                .as_ref()
                .map(|gc| gc.slots() >= 4)
                .unwrap_or(false);
            if !gc_ready {
                break;
            }
            let Some(mutation) = self
                .mutation_consumer
                .as_mut()
                .and_then(|cons| cons.pop().ok())
            else {
                break;
            };

            match mutation {
                RtGraphMutation::SetOrReplacePluginSlot {
                    track_index,
                    slot_index,
                    mut slot,
                } => {
                    let t_idx = track_index as usize;
                    let s_idx = slot_index as usize;
                    if t_idx < self.tracks.len() && s_idx < MAX_INSERT_SLOTS {
                        if let Some(incoming_slot) = slot.take() {
                            let track = &mut self.tracks[t_idx];
                            if s_idx < track.slots.len() {
                                let displaced =
                                    std::mem::replace(&mut track.slots[s_idx], incoming_slot);
                                *slot = Some(displaced);
                                topology_dirty = true;
                            } else if s_idx == track.slots.len()
                                && track.slots.len() < track.slots.capacity()
                            {
                                // Appends only at the exact index: slots are fixed positions, so a
                                // gap (`s_idx > len`) is rejected instead of silently landing the
                                // plug-in at `len`. Hosts pad gaps with `EmptySlotNode` first.
                                track.slots.push(incoming_slot);
                                topology_dirty = true;
                            } else {
                                *slot = Some(incoming_slot);
                            }
                        }
                    }
                    if let Some(gc) = &mut self.gc_producer {
                        let _ = gc.push(RetiredRtResource::PluginSlot(slot));
                    }
                }
                RtGraphMutation::RemovePluginSlot {
                    track_index,
                    slot_index,
                    mut recycle_slot,
                } => {
                    let t_idx = track_index as usize;
                    let s_idx = slot_index as usize;
                    if recycle_slot.is_none()
                        && t_idx < self.tracks.len()
                        && s_idx < self.tracks[t_idx].slots.len()
                    {
                        let removed = self.tracks[t_idx].slots.remove(s_idx);
                        *recycle_slot = Some(removed);
                        topology_dirty = true;
                    }
                    if let Some(gc) = &mut self.gc_producer {
                        let _ = gc.push(RetiredRtResource::PluginSlot(recycle_slot));
                    }
                }
                RtGraphMutation::AddOrUpdateSendRoute {
                    src_track,
                    dest_track,
                    gain,
                    is_sidechain_only,
                    mut route_carrier,
                } => {
                    let src = src_track as usize;
                    let dst = dest_track as usize;
                    if src < self.tracks.len()
                        && dst < self.tracks.len()
                        && src != dst
                        && !self
                            .scheduler
                            .would_create_cycle_rt(&self.tracks, src, dst)
                    {
                        let mut updated_existing = false;
                        for route in &mut self.tracks[src].sends {
                            if route.dest_track_index == dst
                                && route.is_sidechain_only == is_sidechain_only
                            {
                                route.gain = gain;
                                updated_existing = true;
                                break;
                            }
                        }
                        if !updated_existing
                            && self.tracks[src].sends.len() < self.tracks[src].sends.capacity()
                        {
                            if let Some(mut prepared_route) = route_carrier.take() {
                                prepared_route.dest_track_index = dst;
                                prepared_route.gain = gain;
                                prepared_route.is_sidechain_only = is_sidechain_only;
                                self.tracks[src].sends.push(prepared_route);
                                topology_dirty = true;
                            }
                        }
                    }
                    if let Some(gc) = &mut self.gc_producer {
                        let _ = gc.push(RetiredRtResource::SendRoute(route_carrier));
                    }
                }
                RtGraphMutation::RemoveSendRoute {
                    src_track,
                    dest_track,
                    is_sidechain_only,
                    mut recycle_carrier,
                } => {
                    let src = src_track as usize;
                    let dst = dest_track as usize;
                    if recycle_carrier.is_none() && src < self.tracks.len() {
                        if let Some(pos) = self.tracks[src].sends.iter().position(|r| {
                            r.dest_track_index == dst && r.is_sidechain_only == is_sidechain_only
                        }) {
                            let removed = self.tracks[src].sends.remove(pos);
                            *recycle_carrier = Some(removed);
                            topology_dirty = true;
                        }
                    }
                    if let Some(gc) = &mut self.gc_producer {
                        let _ = gc.push(RetiredRtResource::SendRoute(recycle_carrier));
                    }
                }
                RtGraphMutation::UpsertChannel { mut channel } => {
                    if let Some(incoming_ch) = channel.take() {
                        if let Some(pos) =
                            self.channels.iter().position(|c| c.id == incoming_ch.id)
                        {
                            let displaced =
                                std::mem::replace(&mut self.channels[pos], incoming_ch);
                            *channel = Some(displaced);
                            topology_dirty = true;
                        } else if self.channels.len() < self.channels.capacity() {
                            self.channels.push(incoming_ch);
                            topology_dirty = true;
                        } else {
                            *channel = Some(incoming_ch);
                        }
                    }
                    if let Some(gc) = &mut self.gc_producer {
                        let _ = gc.push(RetiredRtResource::Channel(channel));
                    }
                }
                RtGraphMutation::RemoveChannel {
                    channel_id,
                    mut recycle_channel,
                } => {
                    if recycle_channel.is_none() {
                        if let Some(pos) = self.channels.iter().position(|c| c.id == channel_id) {
                            let removed = self.channels.remove(pos);
                            *recycle_channel = Some(removed);
                            topology_dirty = true;
                        }
                    }
                    if let Some(gc) = &mut self.gc_producer {
                        let _ = gc.push(RetiredRtResource::Channel(recycle_channel));
                    }
                }
                RtGraphMutation::ReplaceStemChannels {
                    target_tracks_mask,
                    mut new_channels,
                    mut retired_channels,
                    reset_slicer,
                } => {
                    let mut i = 0usize;
                    while i < self.channels.len() {
                        let trk = self.channels[i].target_mixer_track;
                        let in_mask = trk < 16 && ((target_tracks_mask >> trk) & 1) != 0;
                        let is_stem = Self::is_stem_generator_name(self.channels[i].generator.name());
                        if in_mask
                            && is_stem
                            && retired_channels.len() < retired_channels.capacity()
                        {
                            let removed = self.channels.remove(i);
                            retired_channels.push(removed);
                            topology_dirty = true;
                        } else {
                            i += 1;
                        }
                    }

                    while let Some(ch) = new_channels.pop() {
                        if self.channels.len() < self.channels.capacity() {
                            self.channels.push(ch);
                            topology_dirty = true;
                        } else if retired_channels.len() < retired_channels.capacity() {
                            retired_channels.push(ch);
                        }
                    }

                    if reset_slicer {
                        self.slicer.reset();
                    }
                    if let Some(gc) = &mut self.gc_producer {
                        let _ = gc.push(RetiredRtResource::ChannelBatch(new_channels));
                        let _ = gc.push(RetiredRtResource::ChannelBatch(retired_channels));
                    }
                }
                RtGraphMutation::UpsertPattern { mut pattern } => {
                    let is_empty = pattern.active_notes().is_empty();
                    let _ = self.sequencer.upsert_pattern_in_place_box(&mut pattern);
                    if is_empty {
                        self.release_all_voices();
                    }
                    if let Some(gc) = &mut self.gc_producer {
                        let _ = gc.push(RetiredRtResource::Pattern(pattern));
                    }
                }
                RtGraphMutation::AddOrUpdateNote { pattern_id, note } => {
                    if let Some(pat) = self.sequencer.find_pattern_mut(pattern_id) {
                        if !pat.update_note(note) {
                            let _ = pat.add_note(note);
                        }
                    }
                }
                RtGraphMutation::RemoveNote {
                    pattern_id,
                    note_id,
                } => {
                    if let Some(pat) = self.sequencer.find_pattern_mut(pattern_id) {
                        let _ = pat.remove_note(note_id);
                    }
                }
                RtGraphMutation::UpsertPlaylistClip { clip } => {
                    let _ = self.sequencer.upsert_clip(clip);
                }
                RtGraphMutation::RemovePlaylistClip { clip_id } => {
                    let _ = self.sequencer.remove_clip(clip_id);
                    self.release_all_voices();
                }
                RtGraphMutation::SetSequencerMode {
                    song_mode,
                    active_pattern_id,
                    loop_enabled,
                    loop_length_ticks,
                } => {
                    self.sequencer.song_mode = song_mode;
                    self.sequencer.active_pattern_id = active_pattern_id;
                    self.sequencer.loop_enabled = loop_enabled;
                    self.sequencer.loop_length_ticks = loop_length_ticks;
                }
                RtGraphMutation::UpsertAutomationArticulator { mut articulator } => {
                    let _ = self
                        .sequencer
                        .upsert_articulator_in_place_box(&mut articulator);
                    if let Some(gc) = &mut self.gc_producer {
                        let _ = gc.push(RetiredRtResource::Articulator(articulator));
                    }
                }
                RtGraphMutation::UpsertControllerLink { link } => {
                    let _ = self.controller_bus.add_link(link);
                }
                RtGraphMutation::ClearControllerLinks => {
                    self.controller_bus.clear_links();
                }
                RtGraphMutation::UpdateSampleRate { sample_rate } => {
                    let sr = sample_rate.clamp(8_000.0, 384_000.0);
                    if (self.slicer.sample_rate - sr).abs() > 0.5 {
                        self.slicer.sample_rate = sr;
                        for trk in &mut self.tracks {
                            trk.reset_state();
                        }
                        for ch in &mut self.channels {
                            ch.reset_state();
                        }
                    }
                }
                RtGraphMutation::SetTrackInputAndRecordArm {
                    track_index,
                    input_source_bus,
                    record_armed,
                    hw_input_latency_samples,
                } => {
                    let t_idx = track_index as usize;
                    if t_idx < self.tracks.len() {
                        // A single hardware input bus feeds one armed/monitored track at a time:
                        // disarm and stop monitoring on every other track first.
                        for (i, trk) in self.tracks.iter_mut().enumerate() {
                            if i != t_idx && (trk.record_armed || trk.input_source_bus.is_some()) {
                                trk.record_armed = false;
                                trk.input_source_bus = None;
                                trk.has_hw_input = false;
                                trk.hw_input_buf.fill([0.0, 0.0]);
                            }
                        }
                        let trk = &mut self.tracks[t_idx];
                        trk.input_source_bus = input_source_bus;
                        trk.record_armed = record_armed;
                        trk.has_hw_input = input_source_bus.is_some();
                        trk.hw_input_latency_samples = hw_input_latency_samples as usize;
                        topology_dirty = true;
                    }
                }
                RtGraphMutation::AttachRecordTap { mut tap } => {
                    std::mem::swap(&mut self.record_tap, &mut *tap);
                    if let Some(gc) = &mut self.gc_producer {
                        let _ = gc.push(RetiredRtResource::RecordTap(tap));
                    }
                }
                RtGraphMutation::AttachHwCaptureBridge { mut bridge } => {
                    std::mem::swap(&mut self.hw_capture_bridge, &mut *bridge);
                    if self.hw_capture_bridge.is_none() {
                        for trk in &mut self.tracks {
                            trk.hw_input_buf.fill([0.0, 0.0]);
                        }
                    }
                    if let Some(gc) = &mut self.gc_producer {
                        let _ = gc.push(RetiredRtResource::HwCaptureBridge(bridge));
                    }
                }
                RtGraphMutation::AttachMidiInputConsumer { mut consumer } => {
                    std::mem::swap(&mut self.midi_in_consumer, &mut *consumer);
                    if let Some(gc) = &mut self.gc_producer {
                        let _ = gc.push(RetiredRtResource::MidiInputConsumer(consumer));
                    }
                }
                RtGraphMutation::SetStemClipRegions {
                    channel_id,
                    mut regions,
                    count,
                } => {
                    if let Some(ch) = self.channels.iter_mut().find(|c| c.id == channel_id) {
                        ch.generator.swap_stem_clip_regions(&mut regions, count);
                    }
                    if let Some(gc) = &mut self.gc_producer {
                        let _ = gc.push(RetiredRtResource::StemRegions(regions));
                    }
                }
            }
        }

        if topology_dirty {
            let _ = self
                .scheduler
                .rebuild_levels_and_apdc_rt(&mut self.tracks, &mut self.channels);
            self.recompute_solo_state();
        }
    }

    /// Configures default FL Studio routing where Insert Tracks `1..=count` route to Master Track `0`.
    pub fn route_inserts_to_master(&mut self, insert_count: usize) -> Result<(), GraphCycleError> {
        let limit = insert_count.clamp(FIRST_INSERT_TRACK_INDEX, LAST_INSERT_TRACK_INDEX);
        for idx in FIRST_INSERT_TRACK_INDEX..=limit {
            self.tracks[idx].add_send(MASTER_TRACK_INDEX, 1.0, false);
        }
        self.rebuild_graph()
    }

    /// Adds a `ChannelGenerator` and rebuilds the 5-Point APDC graph.
    pub fn add_channel_generator(
        &mut self,
        channel: ChannelGenerator,
    ) -> Result<usize, GraphCycleError> {
        let idx = self.channels.len();
        self.channels.push(channel);
        self.rebuild_graph()?;
        Ok(idx)
    }

    /// Removes any stem playback generators (`UVR5 Separated Stem Playback Generator`, `Custom Audio Stem`,
    /// `Vocals`, `Drums`, `Bass`, `Other`, `FL26 Sample Kit`) targeting `target_tracks` while preserving
    /// `PolyphonicSynthGeneratorNode` channels.
    pub fn remove_stem_channels_on_tracks(&mut self, target_tracks: &[usize]) {
        self.channels.retain(|ch| {
            if !target_tracks.contains(&ch.target_mixer_track) {
                return true;
            }
            !Self::is_stem_generator_name(ch.generator.name())
        });
    }

    /// Replaces any startup `demo_stem` or previous UVR5 stem generators on Insert Tracks `1..=4`
    /// with a single full-length custom audio stem on Track 1 (`FIRST_INSERT_TRACK_INDEX`), ensuring
    /// Track 1 routes to Master Track `0` and rebuilding the 5-Point APDC graph.
    pub fn replace_with_custom_audio_stem(
        &mut self,
        frames: Vec<StereoFrame>,
        loop_playback: bool,
    ) -> Result<usize, GraphCycleError> {
        self.remove_stem_channels_on_tracks(&[1, 2, 3, 4]);
        let mut player = crate::sequencer_pianoroll::AudioStemClipPlayerNode::new(
            900,
            "Custom Audio Stem",
            frames,
        );
        player.continuous_mode = true;
        player.loop_enabled = loop_playback;
        player.prepare(self.slicer.sample_rate, MAX_BLOCK_SIZE);

        let mut ch = ChannelGenerator::new(900, FIRST_INSERT_TRACK_INDEX, Box::new(player));
        ch.prepare(self.slicer.sample_rate, MAX_BLOCK_SIZE);
        let idx = self.channels.len();
        self.channels.push(ch);
        self.tracks[FIRST_INSERT_TRACK_INDEX].add_send(MASTER_TRACK_INDEX, 1.0, false);
        self.slicer.reset();
        self.rebuild_graph()?;
        Ok(idx)
    }

    /// Rebuilds the Kahn topological levels and 5-Point APDC delays across all channels and 502 tracks.
    pub fn rebuild_graph(&mut self) -> Result<(), GraphCycleError> {
        self.scheduler = TopologicalMixerScheduler::rebuild_levels_and_apdc_checked(
            &mut self.tracks,
            &mut self.channels,
        )?;
        // Off-RT graph (re)build: gains configured directly take effect immediately; only live
        // changes arriving through the command queue are de-click ramped.
        self.recompute_solo_state();
        for t in &mut self.tracks {
            t.snap_gain_ramp();
        }
        Ok(())
    }

    /// Recomputes all 5 APDC points in-place with Zero Heap Allocation (used when a slot is bypassed
    /// or a plugin reports a dynamic latency change on the real-time audio thread).
    #[inline]
    pub fn recompute_apdc_rt(&mut self) {
        self.scheduler
            .recompute_apdc_rt(&mut self.tracks, &mut self.channels);
    }

    /// Ingests all pending lock-free `EngineCommand`s from `rtrb::Consumer` (Zero Heap Allocation).
    #[inline]
    pub fn drain_commands(&mut self) {
        while let Ok(cmd) = self.cmd_consumer.pop() {
            match cmd {
                EngineCommand::SetTransportPlaying(playing) => {
                    if playing {
                        self.is_playing = true;
                        self.stop_fade_remaining = 0;
                    } else if self.is_playing && self.stop_fade_remaining == 0 {
                        // Keep rendering for ~6 ms while fading the master out, then stop and
                        // return the transport to the exact position where stop was requested.
                        self.stop_fade_total =
                            ((self.slicer.sample_rate * 0.006).round() as u32).max(32);
                        self.stop_fade_remaining = self.stop_fade_total;
                        self.stop_request_pos =
                            (self.slicer.current_sample_pos, self.slicer.global_ppq_tick);
                    }
                }
                EngineCommand::SetBpm(bpm) => {
                    self.slicer.set_bpm(bpm);
                }
                EngineCommand::SeekPosition {
                    sample_pos,
                    ppq_tick,
                } => {
                    self.slicer.seek(sample_pos, ppq_tick);
                    self.release_all_voices();
                }
                EngineCommand::SetTrackGain {
                    track_index,
                    gain_linear,
                } => {
                    if let Some(t) = self.tracks.get_mut(track_index as usize) {
                        t.gain = gain_linear;
                    }
                }
                EngineCommand::SetTrackPolarityInverted {
                    track_index,
                    inverted,
                } => {
                    if let Some(t) = self.tracks.get_mut(track_index as usize) {
                        t.polarity_inverted = inverted;
                    }
                }
                EngineCommand::SetSlotWetMix {
                    track_index,
                    slot_index,
                    wet_mix,
                } => {
                    if let Some(t) = self.tracks.get_mut(track_index as usize) {
                        if let Some(s) = t.slots.get_mut(slot_index as usize) {
                            s.wet_mix = wet_mix.clamp(0.0, 1.0);
                        }
                    }
                }
                EngineCommand::SetSlotEnabled {
                    track_index,
                    slot_index,
                    enabled,
                } => {
                    let mut changed = false;
                    if let Some(t) = self.tracks.get_mut(track_index as usize) {
                        if let Some(s) = t.slots.get_mut(slot_index as usize) {
                            if s.enabled != enabled {
                                s.enabled = enabled;
                                changed = true;
                            }
                        }
                    }
                    if changed {
                        self.recompute_apdc_rt();
                    }
                }
                EngineCommand::SetPluginParamNormalized {
                    track_index,
                    slot_index,
                    param_id,
                    norm_val,
                    target_sample_pos,
                } => {
                    let trk_idx = track_index as usize;
                    if let Some(raw_pos) = target_sample_pos {
                        let apdc_lat = self
                            .tracks
                            .get(trk_idx)
                            .map(|t| t.slot_input_arrival_latency(slot_index))
                            .unwrap_or(0);
                        let _ = self.automation_queue.schedule_param_normalized(
                            raw_pos,
                            apdc_lat,
                            track_index,
                            slot_index,
                            0,
                            param_id,
                            norm_val,
                        );
                    } else if let Some(t) = self.tracks.get_mut(trk_idx) {
                        if slot_index == 0xFF {
                            t.gain = norm_val;
                        } else if let Some(s) = t.slots.get_mut(slot_index as usize) {
                            let prev_lat = s.plugin.latency_samples();
                            s.plugin.set_parameter_normalized(param_id, norm_val);
                            if s.plugin.latency_samples() != prev_lat {
                                self.recompute_apdc_rt();
                            }
                        }
                    }
                }
                EngineCommand::SetInternalControllerFixed30 {
                    controller_id,
                    value_fixed30,
                } => {
                    self.controller_bus
                        .publish_fixed30(controller_id, value_fixed30);
                }
                EngineCommand::SetPpq(ppq) => {
                    self.slicer.set_ppq(ppq);
                }
                EngineCommand::TriggerVoiceEvent(ev) => {
                    PlaylistSequencer::deliver_event_to_channels(&ev, &mut self.channels);
                }
                EngineCommand::TriggerAuditionNote {
                    key,
                    velocity,
                    fine_pitch_cents,
                    pan,
                    gate_samples,
                } => {
                    let target_ch_id = self
                        .channels
                        .iter()
                        .find(|c| {
                            c.enabled && !Self::is_stem_generator_name(c.generator.name())
                        })
                        .map(|c| c.id)
                        .unwrap_or(1);
                    let gate = if gate_samples > 0 {
                        gate_samples
                    } else {
                        ((self.slicer.sample_rate * 0.24).round() as u32).max(2_048)
                    };
                    let note_id = 0x8000_0000 | (self.audition_note_seq & 0x7FFF_FFFF);
                    self.audition_note_seq = self.audition_note_seq.wrapping_add(1);
                    let ev = crate::sequencer_pianoroll::SequencerVoiceEvent {
                        sample_pos: self.slicer.current_sample_pos,
                        ppq_tick: self.slicer.global_ppq_tick,
                        channel_id: target_ch_id,
                        note_id,
                        kind: crate::VoiceEventKind::NoteOn,
                        key: key.min(127),
                        velocity: velocity.clamp(0.05, 1.0),
                        pan: pan.clamp(-1.0, 1.0),
                        fine_pitch_cents: fine_pitch_cents.clamp(-1200.0, 1200.0),
                        slide_duration_ticks: 0,
                        slide_duration_samples: gate as usize,
                        color_group: 15,
                    };
                    PlaylistSequencer::deliver_event_to_channels(&ev, &mut self.channels);
                }
                EngineCommand::RefreshApdc => {
                    self.recompute_apdc_rt();
                }
                EngineCommand::SetTrackMute { track_index, muted } => {
                    if let Some(t) = self.tracks.get_mut(track_index as usize) {
                        t.muted = muted;
                    }
                }
                EngineCommand::SetTrackSolo {
                    track_index,
                    soloed,
                } => {
                    if let Some(t) = self.tracks.get_mut(track_index as usize) {
                        t.soloed = soloed && (track_index as usize) != MASTER_TRACK_INDEX;
                    }
                    self.recompute_solo_state();
                }
                EngineCommand::SetTrackPan { track_index, pan } => {
                    if let Some(t) = self.tracks.get_mut(track_index as usize) {
                        t.pan = if pan.is_finite() { pan.clamp(-1.0, 1.0) } else { 0.0 };
                    }
                }
                EngineCommand::SetTelemetryFocusTrack(track_index) => {
                    self.telemetry_focus_track =
                        (track_index as usize).min(MAX_MIXER_TRACKS - 1);
                }
                EngineCommand::SetChannelParamNormalized {
                    channel_id,
                    param_id,
                    norm_val,
                } => {
                    if let Some(ch) = self.channels.iter_mut().find(|c| c.id == channel_id) {
                        ch.generator.set_parameter_normalized(param_id, norm_val);
                    }
                }
            }
        }
    }

    /// Injects a block of hardware input audio (`ASIO` `bufferSwitch` or duplex test harness)
    /// directly into every `MixerTrack` configured for hardware audio input (`0` heap allocations).
    #[inline]
    pub fn inject_hardware_input_block(&mut self, frames: &[StereoFrame]) {
        let n = frames.len().min(MAX_BLOCK_SIZE);
        if n == 0 {
            return;
        }
        let mut peak = 0.0f32;
        for f in &frames[..n] {
            let p = f[0].abs().max(f[1].abs());
            if p > peak {
                peak = p;
            }
        }
        self.last_hw_input_peak = peak;
        self.hw_capture_scratch[..n].copy_from_slice(&frames[..n]);
        self.hw_capture_scratch[n..].fill([0.0, 0.0]);
        for trk in &mut self.tracks {
            if trk.has_hw_input || trk.input_source_bus.is_some() {
                trk.hw_input_buf[..n].copy_from_slice(&frames[..n]);
            }
        }
        self.hw_input_injected_this_block = true;
    }

    #[inline]
    fn compute_record_telemetry_word(&self) -> u32 {
        let is_rec = self
            .record_tap
            .as_ref()
            .map(|t| t.is_recording())
            .unwrap_or(false);
        let mut any_armed = false;
        let mut armed_idx = 0u16;
        for (idx, trk) in self.tracks.iter().enumerate() {
            if trk.record_armed {
                any_armed = true;
                armed_idx = idx as u16;
                break;
            }
        }
        crate::audio_io::pack_telemetry_record_status(
            is_rec,
            any_armed,
            armed_idx,
            self.last_hw_input_peak,
        )
    }

    /// Executes a complete hardware audio block (`master_out.len() <= MAX_BLOCK_SIZE`) through:
    /// 1. `rtrb` Command & `RtGraphMutation` Queue Ingestion (`0` alloc, `0` drop)
    /// 2. Hardware input acquisition (ASIO injection or WASAPI capture bridge) + input-sourced record tap
    /// 3. While playing: `HybridPpqSlicer` + `PlaylistSequencer` + sample-accurate automation + 3-phase DAG.
    ///    While stopped: the full mixer DAG keeps running (input monitoring, FX tails, auditioned notes go
    ///    through track FX and the master chain) with a frozen transport position.
    /// 4. A short de-click fade when the transport stops.
    /// 5. Wait-free 128 KB `VisualTelemetryFrame` publish via `TripleBuffer` with measured DSP load.
    ///
    /// Strictly **Zero Heap Allocation** and **Zero Heap Deallocation**.
    pub fn process_audio_block(&mut self, master_out: &mut [StereoFrame]) {
        let total_frames = master_out.len().min(MAX_BLOCK_SIZE);
        if total_frames == 0 {
            return;
        }
        let block_t0 = std::time::Instant::now();

        // 1. Drain wait-free SPSC commands & zero-alloc/zero-drop graph mutations
        self.drain_commands();
        self.drain_rt_mutations();

        // 1b. Drain hardware Win32 MIDI Input events (`0` alloc). Each event carries its arrival
        // time; it plays at the same offset into this block as it arrived into the previous one, so
        // live notes keep their timing with a constant one-block latency.
        let block_ns = crate::audio_io::midi_arrival_clock_ns();
        let prev_block_ns = self.midi_prev_block_ns;
        self.midi_prev_block_ns = block_ns;
        if let Some(midi_rx) = &mut self.midi_in_consumer {
            let block_start = self.slicer.current_sample_pos;
            let ns_to_samples = self.slicer.sample_rate / 1.0e9;
            let ctx = ProcessContext {
                sample_rate: self.slicer.sample_rate,
                current_sample_pos: block_start,
                ppq_tick: self.slicer.global_ppq_tick,
                bpm: self.slicer.bpm,
                slice_frames: total_frames,
                is_playing: self.is_playing,
                is_offline_rendering: self.is_offline_rendering,
                ppq_resolution: self.slicer.ppq,
            };
            while let Ok(mut ev) = midi_rx.pop() {
                let offset = (((ev.sample_pos - prev_block_ns) as f64) * ns_to_samples)
                    .round()
                    .clamp(0.0, total_frames.saturating_sub(1) as f64) as i64;
                ev.sample_pos = block_start + offset;
                ev.ppq_tick = self.slicer.global_ppq_tick;
                if self.is_playing {
                    // The running transport renders in sub-slices; the automation queue splits them
                    // exactly at the event.
                    let dv = crate::apdc_engine::DelayedVoiceEvent {
                        target_sample_pos: ev.sample_pos,
                        target_id: ev.channel_id,
                        kind: ev.kind,
                        note: ev.key,
                        velocity: (ev.velocity * 127.0).round().clamp(0.0, 127.0) as u8,
                        value: SequencerVoiceEvent::pack_delayed_value(
                            ev.color_group,
                            ev.pan,
                            ev.fine_pitch_cents,
                        ),
                    };
                    if !self.automation_queue.schedule_voice_event(ev.sample_pos, 0, dv) {
                        PlaylistSequencer::deliver_event_to_channels(&ev, &mut self.channels);
                    }
                } else {
                    // Stopped: the generators render the whole block in one call and place the
                    // event at its sample offset.
                    for ch in self.channels.iter_mut() {
                        if ch.enabled && ch.id == ev.channel_id {
                            ch.generator.on_voice_event(&ev, &ctx);
                        }
                    }
                }
            }
        }

        // 1c. Hardware input: ASIO already injected into `hw_capture_scratch`; otherwise drain the
        // WASAPI Capture + Adaptive DLL Drift Resampler bridge. Detaching the bridge zeroes every
        // track input buffer (see `AttachHwCaptureBridge`) so a removed device never loops its last block.
        if !self.hw_input_injected_this_block {
            if let Some((rx, dll)) = &mut self.hw_capture_bridge {
                dll.drain_from_rtrb(rx);
                let scratch = &mut self.hw_capture_scratch[..total_frames];
                dll.process_output_block(scratch);
                let mut peak = 0.0f32;
                for f in scratch.iter() {
                    let p = f[0].abs().max(f[1].abs());
                    if p > peak {
                        peak = p;
                    }
                }
                self.last_hw_input_peak = peak;
                let scratch = &self.hw_capture_scratch[..total_frames];
                for trk in &mut self.tracks {
                    if trk.has_hw_input || trk.input_source_bus.is_some() {
                        trk.hw_input_buf[..total_frames].copy_from_slice(scratch);
                    }
                }
            } else {
                self.hw_capture_scratch[..total_frames].fill([0.0, 0.0]);
                self.last_hw_input_peak = 0.0;
            }
        }

        // 1d. Record tap captures the raw hardware input (pre-FX, pre-fader), never the track output.
        if let Some(tap) = &mut self.record_tap {
            tap.push_rt_slice(&self.hw_capture_scratch[..total_frames]);
        }

        if self.is_playing {
            self.render_playing_block(total_frames);
        } else {
            self.render_stopped_block(total_frames);
        }

        // Copy Master Track 0 (`MASTER_TRACK_INDEX`) output to `master_out`
        master_out[..total_frames]
            .copy_from_slice(&self.tracks[MASTER_TRACK_INDEX].sub_block_buf[..total_frames]);

        // De-click fade-out when the transport stops (the graph keeps running afterwards for tails).
        if self.stop_fade_remaining > 0 {
            let total = self.stop_fade_total.max(1) as f32;
            for frame in master_out[..total_frames].iter_mut() {
                let g = if self.stop_fade_remaining > 0 {
                    self.stop_fade_remaining -= 1;
                    (self.stop_fade_remaining as f32) / total
                } else {
                    0.0
                };
                frame[0] *= g;
                frame[1] *= g;
            }
            if self.stop_fade_remaining == 0 {
                self.is_playing = false;
                let (pos, tick) = self.stop_request_pos;
                self.slicer.seek(pos, tick);
                self.release_all_voices();
            }
        }
        self.hw_input_injected_this_block = false;

        // 4x oversampled (BS.1770-4) true peak of what actually leaves the master bus.
        let mut tp_block = [0.0f32; 2];
        for frame in master_out[..total_frames].iter() {
            for ch in 0..2 {
                let h = &mut self.master_tp_hist[ch];
                h.copy_within(0..11, 1);
                h[0] = frame[ch];
                let mut tp = frame[ch].abs();
                for phase in &MASTER_TRUE_PEAK_TAPS {
                    let mut sum = 0.0f32;
                    for m in 0..12 {
                        sum += phase[m] * h[m];
                    }
                    tp = tp.max(sum.abs());
                }
                tp_block[ch] = tp_block[ch].max(tp);
            }
        }
        self.master_tp_block_lin = tp_block;

        // Measured DSP load: wall-clock render time relative to the real-time budget of this block.
        let budget_s = (total_frames as f64) / self.slicer.sample_rate.max(8_000.0);
        let load_pct = (block_t0.elapsed().as_secs_f64() / budget_s * 100.0) as f32;
        self.dsp_load_ema_pct = if self.dsp_load_ema_pct <= 0.0 {
            load_pct
        } else {
            0.9 * self.dsp_load_ema_pct + 0.1 * load_pct
        };

        self.publish_telemetry_frame();
    }

    /// Sends an all-notes-off (`ChannelMute`, velocity `0`) to every generator so voices started by
    /// the sequencer do not keep sounding once the transport has stopped (Zero Heap Allocation).
    fn release_all_voices(&mut self) {
        let ctx = ProcessContext {
            sample_rate: self.slicer.sample_rate,
            current_sample_pos: self.slicer.current_sample_pos,
            ppq_tick: self.slicer.global_ppq_tick,
            bpm: self.slicer.bpm,
            slice_frames: 0,
            is_playing: false,
            is_offline_rendering: self.is_offline_rendering,
            ppq_resolution: self.slicer.ppq,
        };
        for ch in self.channels.iter_mut() {
            let ev = SequencerVoiceEvent {
                sample_pos: ctx.current_sample_pos,
                ppq_tick: ctx.ppq_tick,
                channel_id: ch.id,
                note_id: 0,
                kind: crate::VoiceEventKind::ChannelMute,
                key: 0,
                velocity: 0.0,
                pan: 0.0,
                fine_pitch_cents: 0.0,
                slide_duration_ticks: 0,
                slide_duration_samples: 0,
                color_group: 0,
            };
            ch.generator.on_voice_event(&ev, &ctx);
        }
    }

    /// Transport-running render: PPQ slicer, sequencer dispatch, sample-accurate automation and the
    /// 3-phase mixer DAG (Zero Heap Allocation).
    fn render_playing_block(&mut self, total_frames: usize) {
        let sample_rate = self.slicer.sample_rate;
        let ppq_res = self.slicer.ppq;
        let is_playing = true;
        let process_all = self.process_all_502_tracks;
        let is_offline = self.is_offline_rendering;

        // Clear track slice buffers for `[0..total_frames]`
        if process_all {
            for track in &mut self.tracks {
                track.clear_slice_buffers(0, total_frames);
            }
        } else {
            let act_lvl_count = self.scheduler.num_active_levels();
            for lvl_idx in 0..act_lvl_count {
                let lvl = self.scheduler.active_level_at(lvl_idx);
                for &u in lvl {
                    self.tracks[u].clear_slice_buffers(0, total_frames);
                }
            }
        }

        // 2. Slice hardware block at PPQ Tick boundaries (`sub_d0ece0`) via split borrow (0 drops!)
        let mut cur_sample_cursor = self.slicer.current_sample_pos;
        {
            let Self {
                slicer,
                sequencer,
                automation_queue,
                controller_bus,
                channels,
                tracks,
                scheduler,
                thread_pool,
                ..
            } = self;

            let mut refs = MixerSliceRefs {
                sequencer,
                automation_queue,
                controller_bus,
                channels,
                tracks,
                scheduler,
                thread_pool: thread_pool.as_ref(),
            };

            slicer.process_buffer_state(
                &mut refs,
                total_frames,
                |engine, ppq_tick, tick_sample_pos, bpm| {
                    if is_playing {
                        engine.sequencer.on_ppq_tick(
                            ppq_tick,
                            tick_sample_pos,
                            bpm,
                            sample_rate,
                            ppq_res,
                            engine.channels,
                            engine.controller_bus,
                        );
                    }
                    let tick_ctx = ProcessContext {
                        sample_rate,
                        current_sample_pos: tick_sample_pos,
                        ppq_tick,
                        bpm,
                        slice_frames: 0,
                        is_playing,
                        is_offline_rendering: is_offline,
                        ppq_resolution: ppq_res,
                    };
                    for ch in engine.channels.iter_mut() {
                        if ch.enabled {
                            ch.generator.on_ppq_tick(&tick_ctx);
                        }
                    }
                    let act_lvl_count = engine.scheduler.num_active_levels();
                    for lvl_idx in 0..act_lvl_count {
                        let lvl = engine.scheduler.active_level_at(lvl_idx);
                        for &u in lvl {
                            for slot in &mut engine.tracks[u].slots {
                                if slot.enabled {
                                    slot.plugin.on_ppq_tick(&tick_ctx);
                                }
                            }
                        }
                    }
                },
                |engine, slice_offset, slice_len, active_ppq_tick, bpm| {
                    let slice_end_pos = cur_sample_cursor + (slice_len as i64);

                    // Propagate 30-bit fixed-point internal controllers (`sub_d24d80`) at PPQ boundary
                    engine.propagate_internal_controllers();

                    // Sub-slice `[slice_offset..slice_offset + slice_len]` at exact `SampleAccurateAutomationQueue`
                    // event timestamps (`sub_c61630`) so scheduled events take effect with 0-sample error.
                    let mut sub_offset = slice_offset;
                    let mut sub_remaining = slice_len;
                    let mut sub_cur_pos = cur_sample_cursor;

                    while sub_remaining > 0 {
                        let mut latency_changed = false;
                        {
                            let tracks_ref: &mut [MixerTrack] = engine.tracks;
                            let channels_ref: &mut [ChannelGenerator] = engine.channels;
                            engine.automation_queue.drain_up_to(
                                sub_cur_pos,
                                |param_ev| {
                                    if let Some(t) =
                                        tracks_ref.get_mut(param_ev.target_track as usize)
                                    {
                                        if param_ev.slot_index == 0xFF {
                                            t.gain = param_ev.norm_value;
                                        } else if let Some(s) =
                                            t.slots.get_mut(param_ev.slot_index as usize)
                                        {
                                            let prev_lat = s.plugin.latency_samples();
                                            s.plugin.set_parameter_normalized(
                                                param_ev.param_index,
                                                param_ev.norm_value,
                                            );
                                            if s.plugin.latency_samples() != prev_lat {
                                                latency_changed = true;
                                            }
                                        }
                                    }
                                },
                                |voice_ev| {
                                    let seq_ev =
                                        SequencerVoiceEvent::from_delayed_voice_event(voice_ev);
                                    PlaylistSequencer::deliver_event_to_channels(
                                        &seq_ev,
                                        channels_ref,
                                    );
                                },
                            );
                        }
                        if latency_changed {
                            engine.recompute_apdc_rt();
                        }

                        let step_len = match engine.automation_queue.next_event_sample_pos() {
                            Some(next_ev_pos)
                                if next_ev_pos > sub_cur_pos && next_ev_pos < slice_end_pos =>
                            {
                                ((next_ev_pos - sub_cur_pos) as usize).min(sub_remaining)
                            }
                            _ => sub_remaining,
                        };

                        let ctx = ProcessContext {
                            sample_rate,
                            current_sample_pos: sub_cur_pos,
                            ppq_tick: active_ppq_tick,
                            bpm,
                            slice_frames: step_len,
                            is_playing,
                            is_offline_rendering: is_offline,
                            ppq_resolution: ppq_res,
                        };

                        // Phase 1 (`0xd0e22f`): Render Generators + Point 1 APDC (`[TChannel + 0x130]`)
                        // Phase 2 (`0xd0e3c7`): Accumulate Channel outputs into target Mixer Tracks
                        let end = sub_offset + step_len;
                        for ch in engine.channels.iter_mut() {
                            if ch.enabled {
                                ch.render_slice(&ctx, sub_offset, step_len);
                                let dst_idx = ch.target_mixer_track;
                                if let Some(dst_track) = engine.tracks.get_mut(dst_idx) {
                                    let ch_out = &ch.output_buf[sub_offset..end];
                                    let trk_in = &mut dst_track.sub_block_buf[sub_offset..end];
                                    for i in 0..step_len {
                                        trk_in[i][0] += ch_out[i][0];
                                        trk_in[i][1] += ch_out[i][1];
                                    }
                                }
                            }
                        }

                        // Phase 3 (`sub_cf42f0`): Topological Level-Order Mixer Graph + Points 2, 3, 4, 5 APDC
                        engine.scheduler.process_slice_levels(
                            &ctx,
                            engine.tracks,
                            engine.thread_pool,
                            sub_offset,
                            step_len,
                            process_all,
                        );

                        sub_offset += step_len;
                        sub_remaining -= step_len;
                        sub_cur_pos += step_len as i64;
                    }

                    // Drain any automation events scheduled exactly at `slice_end_pos`
                    let mut end_latency_changed = false;
                    {
                        let tracks_ref: &mut [MixerTrack] = engine.tracks;
                        let channels_ref: &mut [ChannelGenerator] = engine.channels;
                        engine.automation_queue.drain_up_to(
                            slice_end_pos,
                            |param_ev| {
                                if let Some(t) =
                                    tracks_ref.get_mut(param_ev.target_track as usize)
                                {
                                    if param_ev.slot_index == 0xFF {
                                        t.gain = param_ev.norm_value;
                                    } else if let Some(s) =
                                        t.slots.get_mut(param_ev.slot_index as usize)
                                    {
                                        let prev_lat = s.plugin.latency_samples();
                                        s.plugin.set_parameter_normalized(
                                            param_ev.param_index,
                                            param_ev.norm_value,
                                        );
                                        if s.plugin.latency_samples() != prev_lat {
                                            end_latency_changed = true;
                                        }
                                    }
                                }
                            },
                            |voice_ev| {
                                let seq_ev =
                                    SequencerVoiceEvent::from_delayed_voice_event(voice_ev);
                                PlaylistSequencer::deliver_event_to_channels(
                                    &seq_ev,
                                    channels_ref,
                                );
                            },
                        );
                    }
                    if end_latency_changed {
                        engine.recompute_apdc_rt();
                    }

                    cur_sample_cursor = slice_end_pos;
                },
            );
        }

    }

    /// Transport-stopped render: the transport position stays frozen, generators receive
    /// `is_playing == false` (stem/clip players fall silent, synth voices still sound), and the whole
    /// mixer DAG keeps processing so input monitoring, FX tails and auditioned notes are heard
    /// through every track insert and the master chain (Zero Heap Allocation).
    fn render_stopped_block(&mut self, total_frames: usize) {
        let process_all = self.process_all_502_tracks;
        if process_all {
            for track in &mut self.tracks {
                track.clear_slice_buffers(0, total_frames);
            }
        } else {
            let act_lvl_count = self.scheduler.num_active_levels();
            for lvl_idx in 0..act_lvl_count {
                let lvl = self.scheduler.active_level_at(lvl_idx);
                for &u in lvl {
                    self.tracks[u].clear_slice_buffers(0, total_frames);
                }
            }
        }
        let ctx = ProcessContext {
            sample_rate: self.slicer.sample_rate,
            current_sample_pos: self.slicer.current_sample_pos,
            ppq_tick: self.slicer.global_ppq_tick,
            bpm: self.slicer.bpm,
            slice_frames: total_frames,
            is_playing: false,
            is_offline_rendering: self.is_offline_rendering,
            ppq_resolution: self.slicer.ppq,
        };
        for ch in self.channels.iter_mut() {
            if ch.enabled {
                ch.render_slice(&ctx, 0, total_frames);
                if let Some(dst_track) = self.tracks.get_mut(ch.target_mixer_track) {
                    let ch_out = &ch.output_buf[..total_frames];
                    let trk_in = &mut dst_track.sub_block_buf[..total_frames];
                    for i in 0..total_frames {
                        trk_in[i][0] += ch_out[i][0];
                        trk_in[i][1] += ch_out[i][1];
                    }
                }
            }
        }
        self.scheduler.process_slice_levels(
            &ctx,
            &mut self.tracks,
            self.thread_pool.as_ref(),
            0,
            total_frames,
            process_all,
        );
    }

    /// Recomputes the implicit solo mutes: when any track is soloed, every track that is neither
    /// soloed, upstream of a soloed track (feeding it), nor downstream of one (receiving from it) is
    /// muted. Master is never solo-muted. Zero Heap Allocation (stack-only scratch).
    pub fn recompute_solo_state(&mut self) {
        let n = self.tracks.len().min(MAX_MIXER_TRACKS);
        let any_solo = self.tracks[..n].iter().any(|t| t.soloed);
        if !any_solo {
            for t in &mut self.tracks[..n] {
                t.solo_muted = false;
            }
            return;
        }
        let mut keep_up = [false; MAX_MIXER_TRACKS];
        let mut keep_down = [false; MAX_MIXER_TRACKS];
        for (i, t) in self.tracks[..n].iter().enumerate() {
            if t.soloed {
                keep_up[i] = true;
                keep_down[i] = true;
            }
        }
        // Relaxation over the send graph (acyclic, so it converges within `n` passes).
        for _ in 0..n {
            let mut changed = false;
            for u in 0..n {
                for send in &self.tracks[u].sends {
                    let d = send.dest_track_index;
                    if d >= n {
                        continue;
                    }
                    if keep_down[u] && !keep_down[d] {
                        keep_down[d] = true;
                        changed = true;
                    }
                    if keep_up[d] && !keep_up[u] {
                        keep_up[u] = true;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        for (i, t) in self.tracks[..n].iter_mut().enumerate() {
            t.solo_muted = i != MASTER_TRACK_INDEX && !(keep_up[i] || keep_down[i]);
        }
    }

    /// Publishes the 120 FPS `VisualTelemetryFrame` (`131,072` bytes) in-place via `TripleBuffer`.
    /// Every track meter is rewritten each block (inactive tracks read as silent), and plugins on the
    /// telemetry focus track export last so the shared Pro-Q 4 / Soothe3 slots show that instance.
    fn publish_telemetry_frame(&mut self) {
        let rec_word = self.compute_record_telemetry_word();
        self.frame_seq = self.frame_seq.wrapping_add(1);
        let process_all = self.process_all_502_tracks;
        let silent_meter = TrackMeterEntry {
            peak_l_db: -120.0,
            peak_r_db: -120.0,
            rms_l_db: -120.0,
            rms_r_db: -120.0,
            gr_db: 0.0,
            activity_flags: TRACK_STATE_SILENT,
        };
        let focus = self.telemetry_focus_track;
        let frame = self.telemetry_input.input_buffer_mut();
        frame.header.frame_seq = self.frame_seq;
        frame.header.sample_pos = self.slicer.current_sample_pos;
        frame.header.ppq_tick = self.slicer.global_ppq_tick;
        frame.header.bpm = self.slicer.bpm as f32;
        frame.header.master_gr_db = self.tracks[MASTER_TRACK_INDEX].meter.gr_db;
        // Block true peak (dBTP) of the master output; a true-peak meter on the master chain
        // overwrites this below with its own reading.
        let tp_db = |x: f32| if x > 1.0e-6 { 20.0 * x.log10() } else { -120.0 };
        frame.header.true_peak_db = [tp_db(self.master_tp_block_lin[0]), tp_db(self.master_tp_block_lin[1])];
        frame.header.reserved[0] = rec_word;
        frame.header.cpu_load_pct = self.dsp_load_ema_pct.clamp(0.0, 999.0);
        frame.track_meters.fill(silent_meter);
        // Shared plugin views rest at their floor unless a live instance exports this block.
        frame.proq4_spectrum.pre_db.fill(-120.0);
        frame.proq4_spectrum.post_db.fill(-120.0);
        frame.proq4_spectrum.sc_db.fill(-120.0);
        frame.proq4_spectrum.collision_mask.fill(0.0);
        frame.soothe3_reduction.reduction_db_1024_mut().fill(0.0);
        frame.soothe3_reduction.input_spec_db.fill(-120.0);
        if process_all {
            for i in 0..MAX_MIXER_TRACKS {
                frame.track_meters[i] = self.tracks[i].meter;
                if i == focus {
                    continue;
                }
                for slot in &self.tracks[i].slots {
                    if slot.enabled {
                        slot.plugin.export_visual_telemetry(frame);
                    }
                }
            }
        } else {
            let act_lvl_count = self.scheduler.num_active_levels();
            for lvl_idx in 0..act_lvl_count {
                let lvl = self.scheduler.active_level_at(lvl_idx);
                for &u in lvl {
                    frame.track_meters[u] = self.tracks[u].meter;
                    if u == focus {
                        continue;
                    }
                    for slot in &self.tracks[u].slots {
                        if slot.enabled {
                            slot.plugin.export_visual_telemetry(frame);
                        }
                    }
                }
            }
        }
        if let Some(track) = self.tracks.get(focus) {
            for slot in &track.slots {
                if slot.enabled {
                    slot.plugin.export_visual_telemetry(frame);
                }
            }
        }
        self.controller_bus
            .export_telemetry_norm(&mut frame.automation_spline.controller_values_norm);
        self.telemetry_input.publish();
    }
}
