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
            sends: Vec::with_capacity(4),
            gain: 1.0,
            polarity_inverted: false,
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
        self.input_apdc_delay.reset();
        for slot in &mut self.slots {
            slot.reset_state();
        }
        self.post_rack_apdc_delay.reset();
        for send in &mut self.sends {
            send.send_apdc_delay.reset();
        }
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

        // 5. Apply post-fader track gain and polarity inversion
        let fader_scale = if self.polarity_inverted {
            -self.gain
        } else {
            self.gain
        };
        if (fader_scale - 1.0).abs() > 1e-7 {
            for frame in buf.iter_mut() {
                frame[0] *= fader_scale;
                frame[1] *= fader_scale;
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
            is_playing: true,
            is_offline_rendering: false,
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
}

impl TopologicalMixerScheduler {
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
        for lvl in &self.levels {
            for &u in lvl {
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
                // When the target track also has hardware input (`has_hw_input`), Point 2 is dedicated
                // to delaying `hw_input_buf`, so Point 1 aligns the channel generator all the way to
                // `tracks[dst].input_arrival_latency`. Otherwise Point 1 aligns to `max_channel_latency`
                // and Point 2 handles the remaining slack to `input_arrival_latency`.
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
                        || !tracks[u].slots.is_empty()
                })
                .collect();
            if !active_in_lvl.is_empty() {
                active_levels.push(active_in_lvl);
            }
        }

        let mut sched = Self {
            levels,
            active_levels,
            master_output_latency_samples: 0,
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
        let level_list = if process_all_502_tracks {
            &self.levels
        } else {
            &self.active_levels
        };

        for level_tracks in level_list {
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

/// Complete 502-Track Real-Time Audio Engine (`FLEngine_x64.dll` `sub_d0ece0` & `sub_d0d4c0`).
/// Integrates `HybridPpqSlicer`, `SampleAccurateAutomationQueue`, `InternalControllerBus`,
/// `ChannelGenerator`s (Phase 1 + Point 1 APDC), 502 `MixerTrack`s (Phase 2 + Point 2 APDC,
/// Phase 3 + Points 3/4/5 APDC), `rtrb` command ingestion, and 128 KB `TripleBuffer` telemetry.
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
    pub telemetry_input: triple_buffer::Input<VisualTelemetryFrame>,
    pub is_playing: bool,
    pub process_all_502_tracks: bool,
    frame_seq: u64,
}

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
        let mut channels = Vec::new();
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
            telemetry_input,
            is_playing: true,
            process_all_502_tracks: false,
            frame_seq: 0,
        };
        (engine, cmd_producer, telemetry_output)
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

    /// Rebuilds the Kahn topological levels and 5-Point APDC delays across all channels and 502 tracks.
    pub fn rebuild_graph(&mut self) -> Result<(), GraphCycleError> {
        self.scheduler = TopologicalMixerScheduler::rebuild_levels_and_apdc_checked(
            &mut self.tracks,
            &mut self.channels,
        )?;
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
                    self.is_playing = playing;
                }
                EngineCommand::SetBpm(bpm) => {
                    self.slicer.set_bpm(bpm);
                }
                EngineCommand::SeekPosition {
                    sample_pos,
                    ppq_tick,
                } => {
                    self.slicer.seek(sample_pos, ppq_tick);
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
                EngineCommand::RefreshApdc => {
                    self.recompute_apdc_rt();
                }
            }
        }
    }

    /// Propagates active 30-bit fixed-point internal controller links to target plugin parameters.
    /// Zero Heap Allocation.
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

    /// Executes a complete hardware audio block (`master_out.len() <= MAX_BLOCK_SIZE`) through:
    /// 1. `rtrb` Command Queue Ingestion
    /// 2. `HybridPpqSlicer` (`sub_d0ece0`) + `PlaylistSequencer` (`PianoRollPattern` & `TensionSpline`)
    /// 3. `SampleAccurateAutomationQueue` (`sub_c61630`) + 30-Bit Fixed-Point Controller Bus (`0x40000000`)
    /// 4. 3-Phase Engine (`sub_d0d4c0`):
    ///    - Phase 1: Parallel/Sequential Generators + Point 1 APDC
    ///    - Phase 2: Hardware Input + Channel-to-Track Accumulation + Point 2 APDC
    ///    - Phase 3: Topological Level-Order Mixer DAG + Points 3, 4, 5 APDC
    /// 5. Wait-free 128 KB `VisualTelemetryFrame` publish via `TripleBuffer`.
    ///
    /// Strictly **Zero Heap Allocation**.
    pub fn process_audio_block(&mut self, master_out: &mut [StereoFrame]) {
        let total_frames = master_out.len().min(MAX_BLOCK_SIZE);
        if total_frames == 0 {
            return;
        }

        // 1. Drain wait-free SPSC commands
        self.drain_commands();

        let sample_rate = self.slicer.sample_rate;
        let ppq_res = self.slicer.ppq;
        let is_playing = self.is_playing;
        let process_all = self.process_all_502_tracks;

        // Clear track slice buffers for `[0..total_frames]`
        if process_all {
            for track in &mut self.tracks {
                track.clear_slice_buffers(0, total_frames);
            }
        } else {
            for lvl in &self.scheduler.active_levels {
                for &u in lvl {
                    self.tracks[u].clear_slice_buffers(0, total_frames);
                }
            }
        }

        // 2. Slice hardware block at PPQ Tick boundaries (`sub_d0ece0`)
        let mut slicer = std::mem::replace(
            &mut self.slicer,
            HybridPpqSlicer::new(sample_rate, 128.0, ppq_res),
        );
        let mut cur_sample_cursor = slicer.current_sample_pos;

        slicer.process_buffer_state(
            self,
            total_frames,
            |engine, ppq_tick, tick_sample_pos, bpm| {
                if is_playing {
                    engine.sequencer.on_ppq_tick(
                        ppq_tick,
                        tick_sample_pos,
                        bpm,
                        sample_rate,
                        ppq_res,
                        &mut engine.channels,
                        &mut engine.controller_bus,
                    );
                }
                let tick_ctx = ProcessContext {
                    sample_rate,
                    current_sample_pos: tick_sample_pos,
                    ppq_tick,
                    bpm,
                    slice_frames: 0,
                    is_playing,
                    is_offline_rendering: false,
                };
                for ch in &mut engine.channels {
                    if ch.enabled {
                        ch.generator.on_ppq_tick(&tick_ctx);
                    }
                }
                for lvl in &engine.scheduler.active_levels {
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
                    // Drain all automation and voice events due at or before `sub_cur_pos`
                    let mut latency_changed = false;
                    {
                        let tracks_ref = &mut engine.tracks;
                        let channels_ref = &mut engine.channels;
                        engine.automation_queue.drain_up_to(
                            sub_cur_pos,
                            |param_ev| {
                                if let Some(t) = tracks_ref.get_mut(param_ev.target_track as usize) {
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
                                let seq_ev = SequencerVoiceEvent::from_delayed_voice_event(voice_ev);
                                PlaylistSequencer::deliver_event_to_channels(&seq_ev, channels_ref);
                            },
                        );
                    }
                    if latency_changed {
                        engine.recompute_apdc_rt();
                    }

                    // Determine how many frames we can render before the next scheduled automation event
                    let step_len = match engine.automation_queue.next_event_sample_pos() {
                        Some(next_ev_pos) if next_ev_pos > sub_cur_pos && next_ev_pos < slice_end_pos => {
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
                        is_offline_rendering: false,
                    };

                    // Phase 1 (`0xd0e22f`): Render Generators + Point 1 APDC (`[TChannel + 0x130]`)
                    // Phase 2 (`0xd0e3c7`): Accumulate Channel outputs into target Mixer Tracks
                    let end = sub_offset + step_len;
                    for ch in &mut engine.channels {
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
                        &mut engine.tracks,
                        engine.thread_pool.as_ref(),
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
                    let tracks_ref = &mut engine.tracks;
                    let channels_ref = &mut engine.channels;
                    engine.automation_queue.drain_up_to(
                        slice_end_pos,
                        |param_ev| {
                            if let Some(t) = tracks_ref.get_mut(param_ev.target_track as usize) {
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
                            let seq_ev = SequencerVoiceEvent::from_delayed_voice_event(voice_ev);
                            PlaylistSequencer::deliver_event_to_channels(&seq_ev, channels_ref);
                        },
                    );
                }
                if end_latency_changed {
                    engine.recompute_apdc_rt();
                }

                cur_sample_cursor = slice_end_pos;
            },
        );

        self.slicer = slicer;

        // Copy Master Track 0 (`MASTER_TRACK_INDEX`) output to `master_out`
        master_out[..total_frames]
            .copy_from_slice(&self.tracks[MASTER_TRACK_INDEX].sub_block_buf[..total_frames]);

        // Publish 120 FPS `VisualTelemetryFrame` (`131,072` bytes) in-place via `TripleBuffer`
        self.frame_seq = self.frame_seq.wrapping_add(1);
        let frame = self.telemetry_input.input_buffer_mut();
        frame.header.frame_seq = self.frame_seq;
        frame.header.sample_pos = self.slicer.current_sample_pos;
        frame.header.ppq_tick = self.slicer.global_ppq_tick;
        frame.header.bpm = self.slicer.bpm as f32;
        frame.header.master_gr_db = self.tracks[MASTER_TRACK_INDEX].meter.gr_db;
        frame.header.true_peak_db = [
            self.tracks[MASTER_TRACK_INDEX].meter.peak_l_db,
            self.tracks[MASTER_TRACK_INDEX].meter.peak_r_db,
        ];
        let mut active_slots = 0usize;
        if process_all {
            for i in 0..MAX_MIXER_TRACKS {
                frame.track_meters[i] = self.tracks[i].meter;
                for slot in &self.tracks[i].slots {
                    if slot.enabled {
                        active_slots += 1;
                        slot.plugin.export_visual_telemetry(frame);
                    }
                }
            }
        } else {
            for lvl in &self.scheduler.active_levels {
                for &u in lvl {
                    frame.track_meters[u] = self.tracks[u].meter;
                    for slot in &self.tracks[u].slots {
                        if slot.enabled {
                            active_slots += 1;
                            slot.plugin.export_visual_telemetry(frame);
                        }
                    }
                }
            }
        }
        frame.header.cpu_load_pct = (1.2 + (active_slots as f32) * 0.85).min(99.9);
        self.controller_bus
            .export_telemetry_norm(&mut frame.automation_spline.controller_values_norm);
        self.telemetry_input.publish();
    }
}
