//! 5-Point Automatic Plugin Delay Compensation (`ApdcDelayLine` / `TBasicDelay_32FS`),
//! `PluginRackSlot` (Point 3 Parallel Dry/Wet APDC), and `SampleAccurateAutomationQueue`
//! (`TAutomationDelay` `0x00c5e568`, `sub_c61630`).
//!
//! All real-time processing methods (`process_in_place`, `process`, `schedule_*`, `drain_up_to`)
//! operate strictly on pre-allocated buffers with **Zero Heap Allocation**.

use crate::{FIXED_POINT_30BIT_SCALE, MAX_BLOCK_SIZE, ProcessContext, StereoFrame, UnifiedDspNode};

/// Default pre-allocated capacity in stereo frames for track/slot APDC delay lines.
pub const DEFAULT_APDC_CAPACITY: usize = 8192;

/// Default pre-allocated capacity for sample-accurate automation & voice event queues.
pub const DEFAULT_AUTOMATION_QUEUE_CAPACITY: usize = 4096;

/// Size of the fixed-capacity deduplication hash table (`AutomationDelay.TEventIDValueRec`).
const DEDUP_TABLE_SIZE: usize = 1024;

/// Sample-accurate circular delay line equivalent to `TBasicDelay_32FS`
/// (`VMT = 0x009287a0`, `Size = 0x58` in `FLEngine_x64.dll`).
#[derive(Clone, Debug)]
pub struct ApdcDelayLine {
    ring: Vec<StereoFrame>,
    cursor: usize,
    delay_samples: usize,
}

/// Alias matching the clean-room specification in `04_flstudio26_daw_architecture.md`.
pub type BasicDelay32FS = ApdcDelayLine;

impl ApdcDelayLine {
    /// Pre-allocates a delay line capable of holding up to `max_delay_samples` frames
    /// without reallocating during real-time playback.
    pub fn with_capacity(max_delay_samples: usize) -> Self {
        Self {
            ring: vec![[0.0; 2]; max_delay_samples.max(1)],
            cursor: 0,
            delay_samples: 0,
        }
    }

    /// Creates a delay line with initial active delay `delay_samples` and pre-allocated capacity.
    pub fn new(delay_samples: usize) -> Self {
        let cap = delay_samples.max(DEFAULT_APDC_CAPACITY).max(1);
        Self {
            ring: vec![[0.0; 2]; cap],
            cursor: 0,
            delay_samples,
        }
    }

    /// Ensures the internal ring buffer can accommodate `required_delay_samples`.
    /// Must be called during `prepare()` or `rebuild_levels_and_apdc()`, never on the audio thread.
    #[inline]
    pub fn ensure_capacity(&mut self, required_delay_samples: usize) {
        let needed = required_delay_samples.max(1);
        if self.ring.len() < needed {
            self.ring.resize(needed.next_power_of_two(), [0.0; 2]);
        }
    }

    /// Sets the active delay in samples (Zero Heap Allocation).
    /// Panics in debug/release if `delay_samples > self.ring.len()`.
    #[inline]
    pub fn set_delay_samples(&mut self, delay_samples: usize) {
        assert!(
            delay_samples <= self.ring.len(),
            "ApdcDelayLine capacity ({}) exceeded by requested delay ({})",
            self.ring.len(),
            delay_samples
        );
        if self.delay_samples != delay_samples {
            self.delay_samples = delay_samples;
            self.cursor = 0;
            self.ring[..delay_samples.max(1)].fill([0.0; 2]);
        }
    }

    /// Alias for `set_delay_samples`.
    #[inline]
    pub fn set_delay(&mut self, delay_samples: usize) {
        self.set_delay_samples(delay_samples);
    }

    /// Current active delay in samples.
    #[inline]
    pub fn delay_samples(&self) -> usize {
        self.delay_samples
    }

    /// Maximum delay capacity pre-allocated in the ring buffer.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.ring.len()
    }

    /// Clears the ring buffer state to silence without deallocating memory.
    #[inline]
    pub fn reset(&mut self) {
        self.cursor = 0;
        self.ring[..self.delay_samples.max(1)].fill([0.0; 2]);
    }

    /// Equivalent to `sub_928b00` (`TBasicDelay_32FS` closure at `+0x30`):
    /// Swaps `buf` in-place with the oldest delayed samples in the circular ring buffer.
    /// Zero Heap Allocation.
    #[inline(always)]
    pub fn process_in_place(&mut self, buf: &mut [StereoFrame]) {
        let d = self.delay_samples;
        if d == 0 {
            return;
        }
        let mut c = self.cursor;
        for frame in buf.iter_mut() {
            let delayed = self.ring[c];
            self.ring[c] = *frame;
            *frame = delayed;
            c += 1;
            if c >= d {
                c = 0;
            }
        }
        self.cursor = c;
    }

    /// Out-of-place delay (`ProcessReplace` at `0x00928b60` in `FLEngine_x64.dll`).
    #[inline(always)]
    pub fn process_replace(&mut self, input: &[StereoFrame], output: &mut [StereoFrame]) {
        let n = input.len().min(output.len());
        let d = self.delay_samples;
        if d == 0 {
            output[..n].copy_from_slice(&input[..n]);
            return;
        }
        let mut c = self.cursor;
        for i in 0..n {
            let delayed = self.ring[c];
            self.ring[c] = input[i];
            output[i] = delayed;
            c += 1;
            if c >= d {
                c = 0;
            }
        }
        self.cursor = c;
    }
}

/// 1 of 10 FX Slots on `TMixerTrack` (`+0x134c`), integrating **Point 3: Parallel Dry/Wet APDC**
/// (`TPluginRackModule` `0x00cf26e0`, `[Module + 0x10]` at `0x00cf2c60`).
pub struct PluginRackSlot {
    pub plugin: Box<dyn UnifiedDspNode>,
    pub enabled: bool,
    pub wet_mix: f32,
    pub dry_apdc_delay: ApdcDelayLine, // Point 3 APDC: `[TPluginRackModule + 0x10]`
    pub dry_scratch: Vec<StereoFrame>, // Pre-allocated once in `new()` / `prepare()`
}

impl PluginRackSlot {
    /// Creates a new FX slot wrapping `plugin`, pre-allocating `dry_scratch` and `dry_apdc_delay`.
    pub fn new(plugin: Box<dyn UnifiedDspNode>, wet_mix: f32) -> Self {
        let lat = plugin.latency_samples();
        let cap = lat.max(DEFAULT_APDC_CAPACITY);
        let mut dry_apdc_delay = ApdcDelayLine::with_capacity(cap);
        dry_apdc_delay.set_delay_samples(lat);
        Self {
            plugin,
            enabled: true,
            wet_mix: wet_mix.clamp(0.0, 1.0),
            dry_apdc_delay,
            dry_scratch: vec![[0.0; 2]; MAX_BLOCK_SIZE],
        }
    }

    /// Pre-allocates internal plugin buffers and Point 3 Dry APDC buffer.
    pub fn prepare(&mut self, sample_rate: f64, max_block_size: usize) {
        let block_cap = max_block_size.max(MAX_BLOCK_SIZE);
        if self.dry_scratch.len() < block_cap {
            self.dry_scratch.resize(block_cap, [0.0; 2]);
        }
        self.plugin.prepare(sample_rate, block_cap);
        let lat = self.plugin.latency_samples();
        self.dry_apdc_delay.ensure_capacity(lat.max(DEFAULT_APDC_CAPACITY));
        self.dry_apdc_delay.set_delay_samples(lat);
    }

    /// Resets both the plugin state and the Point 3 Dry delay line.
    pub fn reset_state(&mut self) {
        self.plugin.reset_state();
        self.dry_apdc_delay.reset();
    }

    /// Effective latency contributed by this slot to the Mixer Track's FX chain.
    #[inline]
    pub fn effective_latency_samples(&self) -> usize {
        if self.enabled {
            self.plugin.latency_samples()
        } else {
            0
        }
    }

    /// Processes a real-time slice through the FX slot with Point 3 Dry/Wet Parallel APDC.
    /// Zero Heap Allocation.
    #[inline]
    pub fn process(&mut self, ctx: &ProcessContext, track_buf: &mut [StereoFrame]) {
        self.process_with_sidechain(ctx, track_buf, None);
    }

    /// Processes a real-time slice with optional external sidechain input and Point 3 Dry/Wet APDC.
    /// Zero Heap Allocation.
    #[inline]
    pub fn process_with_sidechain(
        &mut self,
        ctx: &ProcessContext,
        track_buf: &mut [StereoFrame],
        sidechain_in: Option<&[StereoFrame]>,
    ) {
        if !self.enabled {
            return;
        }
        let n = track_buf.len();
        let lat = self.plugin.latency_samples();
        self.dry_apdc_delay.set_delay_samples(lat);

        let w = self.wet_mix.clamp(0.0, 1.0);
        if w <= 0.0 {
            // Even at 0% wet, an enabled slot maintains its reported latency `lat` via Point 3
            // so automating `wet_mix` from 0.0 to 1.0 never causes a graph latency jump.
            self.dry_apdc_delay.process_in_place(track_buf);
            return;
        }

        if (w - 1.0).abs() > 1e-6 {
            let dry = &mut self.dry_scratch[..n];
            dry.copy_from_slice(track_buf);
            // Point 3 APDC: delay Dry branch by exact plugin latency `lat`
            self.dry_apdc_delay.process_in_place(dry);
            self.plugin.process_slice(ctx, track_buf, sidechain_in, &mut []);
            let inv_w = 1.0 - w;
            for i in 0..n {
                track_buf[i][0] = dry[i][0] * inv_w + track_buf[i][0] * w;
                track_buf[i][1] = dry[i][1] * inv_w + track_buf[i][1] * w;
            }
        } else {
            self.plugin.process_slice(ctx, track_buf, sidechain_in, &mut []);
        }
    }
}

/// Sample-accurate delayed parameter automation event (`TAutomationDelay` `VMT = 0x00c5e568`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DelayedAutomationEvent {
    pub target_sample_pos: i64,
    pub target_track: u16,
    pub slot_index: u8, // 0..10 for insert slot, 0xFF for track parameter
    pub plugin_id: u32,
    pub param_index: u32,
    pub value_fixed30: i32, // 0..=0x40000000 (`sub_d24d80`)
    pub norm_value: f32,    // 0.0..=1.0
}

/// Voice / MIDI / Mute event kind for `TAutomationDelay` voice queue (`+0x20`, `sub_c61970`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoiceEventKind {
    NoteOn,
    NoteOff,
    SlideNote,
    ChannelMute,
    SlotEnable,
}

/// Sample-accurate delayed voice / MIDI / mute event (`sub_c61970`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DelayedVoiceEvent {
    pub target_sample_pos: i64,
    pub target_id: u32,
    pub kind: VoiceEventKind,
    pub note: u8,
    pub velocity: u8,
    pub value: i32,
}

#[derive(Clone, Copy, Debug, Default)]
struct EventIdValueRec {
    occupied: bool,
    event_key: u64,
    last_value_fixed30: i32,
}

/// Sample-accurate event & parameter automation queue (`TAutomationDelay` `0x00c5e568`, `sub_c61630`).
/// Compensates parameter and MIDI/Voice events by the exact APDC delay of their target module
/// and flushes them in sample-accurate order without any heap allocation on the audio thread.
pub struct SampleAccurateAutomationQueue {
    param_events: Vec<DelayedAutomationEvent>,
    param_len: usize,
    voice_events: Vec<DelayedVoiceEvent>,
    voice_len: usize,
    dedup_table: Vec<EventIdValueRec>,
    pub enabled: bool,
}

impl Default for SampleAccurateAutomationQueue {
    fn default() -> Self {
        Self::with_capacity(DEFAULT_AUTOMATION_QUEUE_CAPACITY)
    }
}

impl SampleAccurateAutomationQueue {
    /// Pre-allocates both parameter (`+0x18`) and voice (`+0x20`) queues plus the deduplication
    /// dictionary (`+0x30`, `TDictionary<Integer, AutomationDelay.TEventIDValueRec>`).
    pub fn with_capacity(capacity: usize) -> Self {
        let cap = capacity.max(64);
        let dummy_param = DelayedAutomationEvent {
            target_sample_pos: 0,
            target_track: 0,
            slot_index: 0,
            plugin_id: 0,
            param_index: 0,
            value_fixed30: 0,
            norm_value: 0.0,
        };
        let dummy_voice = DelayedVoiceEvent {
            target_sample_pos: 0,
            target_id: 0,
            kind: VoiceEventKind::NoteOn,
            note: 0,
            velocity: 0,
            value: 0,
        };
        Self {
            param_events: vec![dummy_param; cap],
            param_len: 0,
            voice_events: vec![dummy_voice; cap],
            voice_len: 0,
            dedup_table: vec![EventIdValueRec::default(); DEDUP_TABLE_SIZE],
            enabled: true,
        }
    }

    #[inline]
    pub fn pending_param_count(&self) -> usize {
        self.param_len
    }

    #[inline]
    pub fn pending_voice_count(&self) -> usize {
        self.voice_len
    }

    /// Returns the earliest `target_sample_pos` among pending parameter and voice events, if any.
    #[inline]
    pub fn next_event_sample_pos(&self) -> Option<i64> {
        if !self.enabled {
            return None;
        }
        let next_p = if self.param_len > 0 {
            Some(self.param_events[0].target_sample_pos)
        } else {
            None
        };
        let next_v = if self.voice_len > 0 {
            Some(self.voice_events[0].target_sample_pos)
        } else {
            None
        };
        match (next_p, next_v) {
            (Some(p), Some(v)) => Some(p.min(v)),
            (Some(p), None) => Some(p),
            (None, Some(v)) => Some(v),
            (None, None) => None,
        }
    }

    /// Clears all pending events and deduplication state without deallocating.
    pub fn clear(&mut self) {
        self.param_len = 0;
        self.voice_len = 0;
        self.dedup_table.fill(EventIdValueRec::default());
    }

    #[inline]
    fn make_event_key(target_track: u16, slot_index: u8, param_index: u32) -> u64 {
        ((target_track as u64) << 40) | ((slot_index as u64) << 32) | (param_index as u64)
    }

    /// Checks and updates the fixed-size open-addressed deduplication table (`+0x30`).
    /// Returns `true` if the value changed and should be dispatched.
    #[inline]
    fn check_and_record_dedup(&mut self, key: u64, value_fixed30: i32) -> bool {
        let mask = self.dedup_table.len() - 1;
        let mut idx = (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) as usize) & mask;
        for _ in 0..8 {
            let entry = &mut self.dedup_table[idx];
            if !entry.occupied {
                entry.occupied = true;
                entry.event_key = key;
                entry.last_value_fixed30 = value_fixed30;
                return true;
            }
            if entry.event_key == key {
                if entry.last_value_fixed30 == value_fixed30 {
                    return false;
                }
                entry.last_value_fixed30 = value_fixed30;
                return true;
            }
            idx = (idx + 1) & mask;
        }
        // Fallback replacement on probe limit
        self.dedup_table[idx] = EventIdValueRec {
            occupied: true,
            event_key: key,
            last_value_fixed30: value_fixed30,
        };
        true
    }

    /// Schedules a 30-bit fixed-point parameter change (`0..=0x40000000`) compensated by
    /// `apdc_delay_samples`. Maintains sorted order by `target_sample_pos` with Zero Heap Allocation.
    #[inline]
    pub fn schedule_param_fixed30(
        &mut self,
        raw_sample_pos: i64,
        apdc_delay_samples: usize,
        target_track: u16,
        slot_index: u8,
        plugin_id: u32,
        param_index: u32,
        value_fixed30: i32,
    ) -> bool {
        if self.param_len >= self.param_events.len() {
            return false;
        }
        let clamped_fixed = value_fixed30.clamp(0, 0x4000_0000);
        let norm_value = ((clamped_fixed as f64) / FIXED_POINT_30BIT_SCALE) as f32;
        let target_sample_pos = raw_sample_pos + (apdc_delay_samples as i64);

        let ev = DelayedAutomationEvent {
            target_sample_pos,
            target_track,
            slot_index,
            plugin_id,
            param_index,
            value_fixed30: clamped_fixed,
            norm_value,
        };

        // Insertion sort into pre-allocated array (keeps earliest target_sample_pos at index 0)
        let mut insert_pos = self.param_len;
        while insert_pos > 0
            && self.param_events[insert_pos - 1].target_sample_pos > target_sample_pos
        {
            self.param_events[insert_pos] = self.param_events[insert_pos - 1];
            insert_pos -= 1;
        }
        self.param_events[insert_pos] = ev;
        self.param_len += 1;
        true
    }

    /// Schedules a normalized (`0.0..=1.0`) parameter event quantized to the 30-bit fixed-point bus.
    #[inline]
    pub fn schedule_param_normalized(
        &mut self,
        raw_sample_pos: i64,
        apdc_delay_samples: usize,
        target_track: u16,
        slot_index: u8,
        plugin_id: u32,
        param_index: u32,
        norm_val: f32,
    ) -> bool {
        let clamped = norm_val.clamp(0.0, 1.0) as f64;
        let fixed30 = (clamped * FIXED_POINT_30BIT_SCALE).round() as i32;
        self.schedule_param_fixed30(
            raw_sample_pos,
            apdc_delay_samples,
            target_track,
            slot_index,
            plugin_id,
            param_index,
            fixed30,
        )
    }

    /// Schedules a sample-accurate Voice/MIDI/Mute event compensated by `apdc_delay_samples`.
    #[inline]
    pub fn schedule_voice_event(
        &mut self,
        raw_sample_pos: i64,
        apdc_delay_samples: usize,
        mut event: DelayedVoiceEvent,
    ) -> bool {
        if self.voice_len >= self.voice_events.len() {
            return false;
        }
        event.target_sample_pos = raw_sample_pos + (apdc_delay_samples as i64);
        let mut insert_pos = self.voice_len;
        while insert_pos > 0
            && self.voice_events[insert_pos - 1].target_sample_pos > event.target_sample_pos
        {
            self.voice_events[insert_pos] = self.voice_events[insert_pos - 1];
            insert_pos -= 1;
        }
        self.voice_events[insert_pos] = event;
        self.voice_len += 1;
        true
    }

    /// Equivalent to `sub_c61630(AutomationDelay, CurrentSamplePos + SliceLen)`:
    /// Flushes all voice (`sub_c61970`) and parameter (`sub_c61660`) events whose `target_sample_pos`
    /// falls within `..=max_sample_pos` (Zero Heap Allocation).
    #[inline]
    pub fn drain_up_to<FParam, FVoice>(
        &mut self,
        max_sample_pos: i64,
        mut on_param: FParam,
        mut on_voice: FVoice,
    ) where
        FParam: FnMut(&DelayedAutomationEvent),
        FVoice: FnMut(&DelayedVoiceEvent),
    {
        if !self.enabled {
            return;
        }

        // 1. Flush Voice/MIDI/Mute queue (`sub_c61970`)
        let mut v_consumed = 0usize;
        while v_consumed < self.voice_len {
            let ev = self.voice_events[v_consumed];
            if ev.target_sample_pos <= max_sample_pos {
                on_voice(&ev);
                v_consumed += 1;
            } else {
                break;
            }
        }
        if v_consumed > 0 {
            let remaining = self.voice_len - v_consumed;
            for i in 0..remaining {
                self.voice_events[i] = self.voice_events[i + v_consumed];
            }
            self.voice_len = remaining;
        }

        // 2. Flush Parameter queue (`sub_c61660`) with `TEventIDValueRec` deduplication
        let mut p_consumed = 0usize;
        while p_consumed < self.param_len {
            let ev = self.param_events[p_consumed];
            if ev.target_sample_pos <= max_sample_pos {
                let key = Self::make_event_key(ev.target_track, ev.slot_index, ev.param_index);
                if self.check_and_record_dedup(key, ev.value_fixed30) {
                    on_param(&ev);
                }
                p_consumed += 1;
            } else {
                break;
            }
        }
        if p_consumed > 0 {
            let remaining = self.param_len - p_consumed;
            for i in 0..remaining {
                self.param_events[i] = self.param_events[i + p_consumed];
            }
            self.param_len = remaining;
        }
    }
}
