//! Hybrid PPQ 96–960 Sequencer, Sample-Accurate Piano Roll (with FL Studio 26 Slide/Portamento
//! Notes), Playlist Arrangement (`Pattern`, `AudioStem`, `TensionSpline` Automation Clips),
//! `PolyphonicSynthGeneratorNode`, and `AudioStemClipPlayerNode`.
//!
//! All real-time dispatch and audio rendering methods (`on_ppq_tick`, `process_slice`,
//! `dispatch_tick_events`, `schedule_voice_event`) operate on pre-allocated buffers with
//! **Zero Heap Allocation** on the real-time audio thread.

use crate::apdc_engine::{
    ApdcDelayLine, DEFAULT_APDC_CAPACITY, DelayedVoiceEvent, SampleAccurateAutomationQueue,
    VoiceEventKind,
};
use crate::automation_spline::{
    AutomationClipArticulator, InternalControllerBus, TensionSpline,
};
use crate::{ProcessContext, StereoFrame, UnifiedDspNode};
use std::f64::consts::TAU;

/// Maximum pre-allocated Piano Roll notes per pattern (`Zero-Allocation` real-time guarantee).
pub const MAX_NOTES_PER_PATTERN: usize = 512;

/// Maximum pre-allocated Patterns in the `PlaylistSequencer`.
pub const MAX_SEQUENCER_PATTERNS: usize = 64;

/// Maximum pre-allocated Playlist Arrangement Clips in the `PlaylistSequencer`.
pub const MAX_PLAYLIST_CLIPS: usize = 256;

/// Maximum pre-allocated Automation Clip Articulators in the `PlaylistSequencer`.
pub const MAX_AUTOMATION_ARTICULATORS: usize = 64;

/// Maximum polyphonic voices per `PolyphonicSynthGeneratorNode`.
pub const MAX_SYNTH_VOICES: usize = 16;

/// Maximum pre-allocated sample-accurate voice events queued inside a generator node.
pub const MAX_GENERATOR_VOICE_EVENTS: usize = 512;

/// Maximum pre-allocated scheduled regions inside an `AudioStemClipPlayerNode`.
pub const MAX_STEM_CLIP_REGIONS: usize = 128;

// ============================================================================
// PIANO ROLL NOTE & PATTERN DATA STRUCTURES (FL STUDIO 26 PIANO ROLL SPEC)
// ============================================================================

/// Single Piano Roll Note supporting FL Studio 26 per-note expression (`velocity`, `pan`,
/// `fine_pitch_cents`, 16 `color_group` MIDI channels, and `is_slide` Portamento glide).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PianoRollNote {
    pub id: u32,
    pub start_tick: i64,
    pub duration_ticks: i64,
    pub key: u8,               // MIDI note 0..=127 (60 = C4)
    pub velocity: f32,         // Normalized 0.0..=1.0 (default 0.8)
    pub pan: f32,              // Stereo pan -1.0 (Left) .. 0.0 (Center) .. +1.0 (Right)
    pub fine_pitch_cents: f32, // Per-note fine pitch offset in cents (-1200.0..=+1200.0)
    pub is_slide: bool,        // FL Studio 26 Slide/Portamento note flag
    pub color_group: u8,       // FL Studio 26 16-color note group (0..16)
}

impl Default for PianoRollNote {
    fn default() -> Self {
        Self {
            id: 0,
            start_tick: 0,
            duration_ticks: 96,
            key: 60,
            velocity: 0.8,
            pan: 0.0,
            fine_pitch_cents: 0.0,
            is_slide: false,
            color_group: 0,
        }
    }
}

impl PianoRollNote {
    #[inline]
    pub fn new(id: u32, start_tick: i64, duration_ticks: i64, key: u8, velocity: f32) -> Self {
        Self {
            id,
            start_tick,
            duration_ticks,
            key,
            velocity,
            pan: 0.0,
            fine_pitch_cents: 0.0,
            is_slide: false,
            color_group: 0,
        }
    }

    #[inline]
    pub fn new_slide(id: u32, start_tick: i64, duration_ticks: i64, key: u8, velocity: f32) -> Self {
        Self {
            id,
            start_tick,
            duration_ticks,
            key,
            velocity,
            pan: 0.0,
            fine_pitch_cents: 0.0,
            is_slide: true,
            color_group: 0,
        }
    }

    #[inline]
    pub fn end_tick(&self) -> i64 {
        self.start_tick + self.duration_ticks.max(1)
    }
}

/// Sample-accurate Voice / MIDI / Slide event dispatched to a `PolyphonicSynthGeneratorNode`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SequencerVoiceEvent {
    pub sample_pos: i64,
    pub ppq_tick: i64,
    pub channel_id: u32,
    pub note_id: u32,
    pub kind: VoiceEventKind,
    pub key: u8,
    pub velocity: f32,
    pub pan: f32,
    pub fine_pitch_cents: f32,
    pub slide_duration_ticks: i64,
    pub slide_duration_samples: usize,
    pub color_group: u8,
}

impl Default for SequencerVoiceEvent {
    fn default() -> Self {
        Self {
            sample_pos: 0,
            ppq_tick: 0,
            channel_id: 0,
            note_id: 0,
            kind: VoiceEventKind::NoteOn,
            key: 60,
            velocity: 0.8,
            pan: 0.0,
            fine_pitch_cents: 0.0,
            slide_duration_ticks: 0,
            slide_duration_samples: 0,
            color_group: 0,
        }
    }
}

impl SequencerVoiceEvent {
    /// Converts a `DelayedVoiceEvent` from `SampleAccurateAutomationQueue` into a `SequencerVoiceEvent`.
    /// `DelayedVoiceEvent.value` encodes `(slide_blocks_256 << 28) | (color_group as i32) << 24 | (pan_i8 as u8 as i32) << 16 | (fine_pitch_cents_i16 as u16 as i32)`.
    #[inline]
    pub fn from_delayed_voice_event(ev: &DelayedVoiceEvent) -> Self {
        let slide_blocks_256 = (((ev.value as u32) >> 28) & 0x0F) as usize;
        let color_group = ((ev.value >> 24) & 0x0F) as u8;
        let pan_i8 = ((ev.value >> 16) & 0xFF) as i8;
        let cents_i16 = (ev.value & 0xFFFF) as i16;
        let pan = (pan_i8 as f32 / 100.0).clamp(-1.0, 1.0);
        let fine_pitch_cents = cents_i16 as f32;
        Self {
            sample_pos: ev.target_sample_pos,
            ppq_tick: 0,
            channel_id: ev.target_id,
            note_id: ev.note as u32,
            kind: ev.kind,
            key: ev.note,
            velocity: (ev.velocity as f32 / 127.0).clamp(0.0, 1.0),
            pan,
            fine_pitch_cents,
            slide_duration_ticks: 0,
            slide_duration_samples: slide_blocks_256 * 256,
            color_group,
        }
    }

    /// Packs pan (`-1.0..=1.0`), fine pitch cents (`-1200..=1200`), and `color_group` (`0..16`)
    /// into the 32-bit `value` field of `DelayedVoiceEvent`.
    #[inline]
    pub fn pack_delayed_value(color_group: u8, pan: f32, fine_pitch_cents: f32) -> i32 {
        Self::pack_delayed_value_with_slide(color_group, pan, fine_pitch_cents, 0)
    }

    /// Packs `color_group` (`0..16`), `pan` (`-1.0..=1.0`), `fine_pitch_cents` (`-2400..=2400`),
    /// and `slide_duration_samples` (quantized to 256-sample units in bits `28..31`) into `DelayedVoiceEvent::value`.
    #[inline]
    pub fn pack_delayed_value_with_slide(
        color_group: u8,
        pan: f32,
        fine_pitch_cents: f32,
        slide_duration_samples: usize,
    ) -> i32 {
        let slide_blocks = if slide_duration_samples == 0 {
            0i32
        } else {
            (((slide_duration_samples + 128) / 256).clamp(1, 15) as i32) & 0x0F
        };
        let cg = (color_group & 0x0F) as i32;
        let pan_i8 = (pan.clamp(-1.0, 1.0) * 100.0).round() as i8 as u8 as i32;
        let cents_i16 = (fine_pitch_cents.clamp(-2400.0, 2400.0).round() as i16) as u16 as i32;
        (slide_blocks << 28) | (cg << 24) | (pan_i8 << 16) | cents_i16
    }
}

/// Pre-allocated Piano Roll Pattern targeting a `ChannelGenerator` (`target_channel_id`).
#[derive(Clone, Debug)]
pub struct PianoRollPattern {
    pub pattern_id: u32,
    pub name: String,
    pub target_channel_id: u32,
    pub length_ticks: i64,
    notes: Vec<PianoRollNote>,
    note_count: usize,
}

impl PianoRollPattern {
    pub fn new(
        pattern_id: u32,
        name: impl Into<String>,
        target_channel_id: u32,
        length_ticks: i64,
    ) -> Self {
        Self {
            pattern_id,
            name: name.into(),
            target_channel_id,
            length_ticks: length_ticks.max(24),
            notes: vec![PianoRollNote::default(); MAX_NOTES_PER_PATTERN],
            note_count: 0,
        }
    }

    #[inline]
    pub fn active_notes(&self) -> &[PianoRollNote] {
        &self.notes[..self.note_count]
    }

    #[inline]
    pub fn note_count(&self) -> usize {
        self.note_count
    }

    pub fn clear_notes(&mut self) {
        self.note_count = 0;
    }

    /// Inserts a `PianoRollNote` sorted by `(start_tick, is_slide, id)` without reallocating.
    pub fn add_note(&mut self, mut note: PianoRollNote) -> bool {
        if self.note_count >= self.notes.len() {
            return false;
        }
        note.start_tick = note.start_tick.max(0);
        note.duration_ticks = note.duration_ticks.max(1);
        note.key = note.key.min(127);
        note.velocity = note.velocity.clamp(0.0, 1.0);
        note.pan = note.pan.clamp(-1.0, 1.0);
        note.color_group &= 0x0F;

        let mut insert_idx = self.note_count;
        while insert_idx > 0 {
            let prev = &self.notes[insert_idx - 1];
            if prev.start_tick > note.start_tick
                || (prev.start_tick == note.start_tick && prev.is_slide && !note.is_slide)
            {
                self.notes[insert_idx] = *prev;
                insert_idx -= 1;
            } else {
                break;
            }
        }
        self.notes[insert_idx] = note;
        self.note_count += 1;
        true
    }

    /// Removes a note by `note_id` in-place without heap deallocation.
    pub fn remove_note(&mut self, note_id: u32) -> bool {
        for i in 0..self.note_count {
            if self.notes[i].id == note_id {
                for j in i..(self.note_count - 1) {
                    self.notes[j] = self.notes[j + 1];
                }
                self.note_count -= 1;
                return true;
            }
        }
        false
    }

    /// Updates an existing note (e.g., drag/resize/slide toggle from Svelte 5 Piano Roll UI)
    /// while preserving sorted order and Zero Heap Allocation.
    pub fn update_note(&mut self, updated: PianoRollNote) -> bool {
        if self.remove_note(updated.id) {
            self.add_note(updated)
        } else {
            false
        }
    }
}

// ============================================================================
// PLAYLIST ARRANGEMENT CLIPS & SEQUENCER (`HybridPpqSlicer` SYNCHRONIZED)
// ============================================================================

/// Kind of clip placed on the Playlist Arrangement timeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaylistClipKind {
    /// Triggers a `PianoRollPattern` (`source_id = pattern_id`).
    Pattern,
    /// Triggers an `AudioStemClipPlayerNode` (`source_id = channel_id`).
    AudioStem,
    /// Evaluates an `AutomationClipArticulator` (`source_id = controller_id` on `InternalControllerBus`).
    Automation,
}

/// Clip instance placed on the Playlist Arrangement timeline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaylistClip {
    pub clip_id: u32,
    pub track_row: u16,
    pub kind: PlaylistClipKind,
    pub source_id: u32,
    pub start_tick: i64,
    pub duration_ticks: i64,
    pub clip_offset_ticks: i64,
    pub gain: f32,
    pub muted: bool,
}

impl Default for PlaylistClip {
    fn default() -> Self {
        Self {
            clip_id: 0,
            track_row: 0,
            kind: PlaylistClipKind::Pattern,
            source_id: 0,
            start_tick: 0,
            duration_ticks: 384,
            clip_offset_ticks: 0,
            gain: 1.0,
            muted: false,
        }
    }
}

impl PlaylistClip {
    #[inline]
    pub fn end_tick(&self) -> i64 {
        self.start_tick + self.duration_ticks.max(1)
    }
}

/// Real-time Sequencer & Playlist Arrangement Engine synchronized with `HybridPpqSlicer`.
/// Pre-allocates all Pattern, Playlist Clip, and Automation Articulator storage at initialization
/// so `on_ppq_tick` executes with **Zero Heap Allocation**.
pub struct PlaylistSequencer {
    patterns: Vec<PianoRollPattern>,
    pattern_count: usize,
    clips: Vec<PlaylistClip>,
    clip_count: usize,
    articulators: Vec<AutomationClipArticulator>,
    articulator_count: usize,
    pub song_mode: bool,
    pub active_pattern_id: u32,
    pub loop_length_ticks: i64,
    pub loop_enabled: bool,
    pub auto_evaluate_without_clips: bool,
}

impl Default for PlaylistSequencer {
    fn default() -> Self {
        Self::new()
    }
}

impl PlaylistSequencer {
    pub fn new() -> Self {
        let mut patterns = Vec::with_capacity(MAX_SEQUENCER_PATTERNS);
        for i in 0..MAX_SEQUENCER_PATTERNS {
            patterns.push(PianoRollPattern::new(i as u32, String::new(), 0, 384));
        }
        let clips = vec![PlaylistClip::default(); MAX_PLAYLIST_CLIPS];
        let mut articulators = Vec::with_capacity(MAX_AUTOMATION_ARTICULATORS);
        for i in 0..MAX_AUTOMATION_ARTICULATORS {
            articulators.push(AutomationClipArticulator::new(
                i as u16,
                TensionSpline::default(),
            ));
        }
        Self {
            patterns,
            pattern_count: 0,
            clips,
            clip_count: 0,
            articulators,
            articulator_count: 0,
            song_mode: true,
            active_pattern_id: 0,
            loop_length_ticks: 0,
            loop_enabled: false,
            auto_evaluate_without_clips: true,
        }
    }

    #[inline]
    pub fn active_patterns(&self) -> &[PianoRollPattern] {
        &self.patterns[..self.pattern_count]
    }

    #[inline]
    pub fn active_patterns_mut(&mut self) -> &mut [PianoRollPattern] {
        &mut self.patterns[..self.pattern_count]
    }

    #[inline]
    pub fn active_clips(&self) -> &[PlaylistClip] {
        &self.clips[..self.clip_count]
    }

    #[inline]
    pub fn active_articulators(&self) -> &[AutomationClipArticulator] {
        &self.articulators[..self.articulator_count]
    }

    #[inline]
    pub fn active_articulators_mut(&mut self) -> &mut [AutomationClipArticulator] {
        &mut self.articulators[..self.articulator_count]
    }

    /// Adds or replaces a `PianoRollPattern` in pre-allocated storage.
    pub fn upsert_pattern(&mut self, pattern: PianoRollPattern) -> bool {
        for i in 0..self.pattern_count {
            if self.patterns[i].pattern_id == pattern.pattern_id {
                self.patterns[i] = pattern;
                return true;
            }
        }
        if self.pattern_count >= self.patterns.len() {
            return false;
        }
        self.patterns[self.pattern_count] = pattern;
        self.pattern_count += 1;
        true
    }

    pub fn find_pattern_mut(&mut self, pattern_id: u32) -> Option<&mut PianoRollPattern> {
        self.patterns[..self.pattern_count]
            .iter_mut()
            .find(|p| p.pattern_id == pattern_id)
    }

    pub fn find_pattern(&self, pattern_id: u32) -> Option<&PianoRollPattern> {
        self.patterns[..self.pattern_count]
            .iter()
            .find(|p| p.pattern_id == pattern_id)
    }

    /// Adds or updates a `PlaylistClip` on the timeline without reallocating.
    pub fn upsert_clip(&mut self, mut clip: PlaylistClip) -> bool {
        clip.start_tick = clip.start_tick.max(0);
        clip.duration_ticks = clip.duration_ticks.max(1);
        for i in 0..self.clip_count {
            if self.clips[i].clip_id == clip.clip_id {
                self.clips[i] = clip;
                return true;
            }
        }
        if self.clip_count >= self.clips.len() {
            return false;
        }
        self.clips[self.clip_count] = clip;
        self.clip_count += 1;
        true
    }

    pub fn remove_clip(&mut self, clip_id: u32) -> bool {
        for i in 0..self.clip_count {
            if self.clips[i].clip_id == clip_id {
                for j in i..(self.clip_count - 1) {
                    self.clips[j] = self.clips[j + 1];
                }
                self.clip_count -= 1;
                return true;
            }
        }
        false
    }

    /// Registers or updates an `AutomationClipArticulator` (`TensionSpline` + `LfoModulator`).
    pub fn upsert_automation_articulator(&mut self, articulator: AutomationClipArticulator) -> bool {
        for i in 0..self.articulator_count {
            if self.articulators[i].controller_id == articulator.controller_id {
                self.articulators[i] = articulator;
                return true;
            }
        }
        if self.articulator_count >= self.articulators.len() {
            return false;
        }
        self.articulators[self.articulator_count] = articulator;
        self.articulator_count += 1;
        true
    }

    /// Evaluates all active Playlist Clips, Piano Roll Patterns (including Slide notes), and
    /// `TensionSpline` Automation Articulators at `ppq_tick` (`tick_sample_pos`).
    /// Strictly **Zero Heap Allocation**.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn on_ppq_tick(
        &self,
        ppq_tick: i64,
        tick_sample_pos: i64,
        bpm: f64,
        sample_rate: f64,
        ppq_resolution: u32,
        channels: &mut [crate::mixer_dag::ChannelGenerator],
        controller_bus: &mut InternalControllerBus,
    ) {
        let effective_tick = if self.loop_enabled && self.loop_length_ticks > 0 {
            ppq_tick.rem_euclid(self.loop_length_ticks)
        } else {
            ppq_tick
        };

        let samples_per_tick =
            (sample_rate * 60.0) / (bpm.max(1.0) * (ppq_resolution.max(24) as f64));

        // 1. Evaluate Automation Articulators (`TensionSpline` + `LfoModulator`) -> 30-Bit Controller Bus
        if self.clip_count > 0 || self.pattern_count > 0 || self.auto_evaluate_without_clips {
            for i in 0..self.articulator_count {
                let art = &self.articulators[i];
                if art.spline.points.is_empty() && !art.lfo.enabled {
                    continue;
                }
                // Check if any Automation PlaylistClip gates/offsets this articulator; if no clip references it,
                // evaluate directly in global time.
                let mut has_automation_clip = false;
                let mut active_local_tick = None;
                for c_idx in 0..self.clip_count {
                    let clip = &self.clips[c_idx];
                    if !clip.muted
                        && clip.kind == PlaylistClipKind::Automation
                        && clip.source_id == (art.controller_id as u32)
                    {
                        has_automation_clip = true;
                        if effective_tick >= clip.start_tick && effective_tick < clip.end_tick() {
                            active_local_tick =
                                Some(effective_tick - clip.start_tick + clip.clip_offset_ticks);
                            break;
                        }
                    }
                }

                if has_automation_clip {
                    if let Some(local_tick) = active_local_tick {
                        let fixed30 = art.evaluate_fixed30(local_tick as f64);
                        controller_bus.publish_fixed30(art.controller_id, fixed30);
                    }
                } else {
                    let fixed30 = art.evaluate_fixed30(effective_tick as f64);
                    controller_bus.publish_fixed30(art.controller_id, fixed30);
                }
            }
        }

        // 2. Dispatch Piano Roll Pattern Notes (Song Mode vs Pattern Mode)
        if self.song_mode && self.clip_count > 0 {
            for c_idx in 0..self.clip_count {
                let clip = &self.clips[c_idx];
                if clip.muted || clip.kind != PlaylistClipKind::Pattern {
                    continue;
                }
                if effective_tick < clip.start_tick || effective_tick >= clip.end_tick() {
                    continue;
                }
                let rel_tick = effective_tick - clip.start_tick + clip.clip_offset_ticks;
                if let Some(pattern) = self.find_pattern(clip.source_id) {
                    let pat_tick = if pattern.length_ticks > 0 {
                        rel_tick.rem_euclid(pattern.length_ticks)
                    } else {
                        rel_tick
                    };
                    self.dispatch_pattern_at_tick(
                        pattern,
                        pat_tick,
                        effective_tick == clip.end_tick() - 1,
                        clip.gain,
                        ppq_tick,
                        tick_sample_pos,
                        samples_per_tick,
                        channels,
                    );
                }
            }
        } else if let Some(pattern) = self.find_pattern(self.active_pattern_id) {
            let pat_tick = if pattern.length_ticks > 0 {
                effective_tick.rem_euclid(pattern.length_ticks)
            } else {
                effective_tick
            };
            self.dispatch_pattern_at_tick(
                pattern,
                pat_tick,
                false,
                1.0,
                ppq_tick,
                tick_sample_pos,
                samples_per_tick,
                channels,
            );
        }
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn dispatch_pattern_at_tick(
        &self,
        pattern: &PianoRollPattern,
        pat_tick: i64,
        clip_ending_next_tick: bool,
        clip_gain: f32,
        global_ppq_tick: i64,
        tick_sample_pos: i64,
        samples_per_tick: f64,
        channels: &mut [crate::mixer_dag::ChannelGenerator],
    ) {
        let notes = pattern.active_notes();
        if notes.is_empty() {
            return;
        }

        // First pass: NoteOff events at `pat_tick`
        for note in notes {
            if !note.is_slide && (note.end_tick() == pat_tick || clip_ending_next_tick) {
                let ev = SequencerVoiceEvent {
                    sample_pos: tick_sample_pos,
                    ppq_tick: global_ppq_tick,
                    channel_id: pattern.target_channel_id,
                    note_id: note.id,
                    kind: VoiceEventKind::NoteOff,
                    key: note.key,
                    velocity: 0.0,
                    pan: note.pan,
                    fine_pitch_cents: note.fine_pitch_cents,
                    slide_duration_ticks: 0,
                    slide_duration_samples: 0,
                    color_group: note.color_group,
                };
                Self::deliver_event_to_channels(&ev, channels);
            }
        }

        // Second pass: Normal NoteOn events at `pat_tick` (before Slide notes on the same tick)
        for note in notes {
            if !note.is_slide && note.start_tick == pat_tick {
                let ev = SequencerVoiceEvent {
                    sample_pos: tick_sample_pos,
                    ppq_tick: global_ppq_tick,
                    channel_id: pattern.target_channel_id,
                    note_id: note.id,
                    kind: VoiceEventKind::NoteOn,
                    key: note.key,
                    velocity: (note.velocity * clip_gain).clamp(0.0, 1.0),
                    pan: note.pan,
                    fine_pitch_cents: note.fine_pitch_cents,
                    slide_duration_ticks: note.duration_ticks,
                    slide_duration_samples: ((note.duration_ticks as f64) * samples_per_tick)
                        .round()
                        .max(1.0) as usize,
                    color_group: note.color_group,
                };
                Self::deliver_event_to_channels(&ev, channels);
            }
        }

        // Third pass: FL Studio 26 Slide/Portamento notes starting at `pat_tick`
        for note in notes {
            if note.is_slide && note.start_tick == pat_tick {
                let slide_samples = ((note.duration_ticks.max(1) as f64) * samples_per_tick)
                    .round()
                    .max(1.0) as usize;
                let ev = SequencerVoiceEvent {
                    sample_pos: tick_sample_pos,
                    ppq_tick: global_ppq_tick,
                    channel_id: pattern.target_channel_id,
                    note_id: note.id,
                    kind: VoiceEventKind::SlideNote,
                    key: note.key,
                    velocity: (note.velocity * clip_gain).clamp(0.0, 1.0),
                    pan: note.pan,
                    fine_pitch_cents: note.fine_pitch_cents,
                    slide_duration_ticks: note.duration_ticks,
                    slide_duration_samples: slide_samples,
                    color_group: note.color_group,
                };
                Self::deliver_event_to_channels(&ev, channels);
            }
        }
    }

    #[inline]
    pub fn deliver_event_to_channels(
        ev: &SequencerVoiceEvent,
        channels: &mut [crate::mixer_dag::ChannelGenerator],
    ) {
        let dummy_ctx = ProcessContext {
            current_sample_pos: ev.sample_pos,
            ppq_tick: ev.ppq_tick,
            ..ProcessContext::default()
        };
        for ch in channels.iter_mut() {
            if ch.enabled && ch.id == ev.channel_id {
                ch.generator.on_voice_event(ev, &dummy_ctx);
            }
        }
    }

    /// Helper to schedule all notes of a `PianoRollPattern` within `[start_tick..end_tick)` directly
    /// into a `SampleAccurateAutomationQueue` with Point 1 APDC compensation.
    pub fn schedule_pattern_into_automation_queue(
        &self,
        pattern_id: u32,
        slicer: &crate::slicer_ppq::HybridPpqSlicer,
        apdc_delay_samples: usize,
        queue: &mut SampleAccurateAutomationQueue,
    ) -> usize {
        let Some(pattern) = self.find_pattern(pattern_id) else {
            return 0;
        };
        let mut scheduled = 0usize;
        for note in pattern.active_notes() {
            let on_sample = slicer.tick_to_sample_pos(note.start_tick as f64);
            let off_sample = slicer.tick_to_sample_pos(note.end_tick() as f64);
            let slide_samples = if note.is_slide {
                (off_sample - on_sample).max(1) as usize
            } else {
                0
            };
            let kind = if note.is_slide {
                VoiceEventKind::SlideNote
            } else {
                VoiceEventKind::NoteOn
            };
            let on_ev = DelayedVoiceEvent {
                target_sample_pos: on_sample,
                target_id: pattern.target_channel_id,
                kind,
                note: note.key,
                velocity: (note.velocity.clamp(0.0, 1.0) * 127.0).round() as u8,
                value: SequencerVoiceEvent::pack_delayed_value_with_slide(
                    note.color_group,
                    note.pan,
                    note.fine_pitch_cents,
                    slide_samples,
                ),
            };
            if queue.schedule_voice_event(on_sample, apdc_delay_samples, on_ev) {
                scheduled += 1;
            }
            if !note.is_slide {
                let off_ev = DelayedVoiceEvent {
                    target_sample_pos: off_sample,
                    target_id: pattern.target_channel_id,
                    kind: VoiceEventKind::NoteOff,
                    note: note.key,
                    velocity: 0,
                    value: SequencerVoiceEvent::pack_delayed_value(
                        note.color_group,
                        note.pan,
                        note.fine_pitch_cents,
                    ),
                };
                if queue.schedule_voice_event(off_sample, apdc_delay_samples, off_ev) {
                    scheduled += 1;
                }
            }
        }
        scheduled
    }
}

// ============================================================================
// INTERNAL GENERATOR 1: POLYPHONIC SYNTH & SAMPLER (`PolyphonicSynthGeneratorNode`)
// ============================================================================

/// Oscillator waveform for `PolyphonicSynthGeneratorNode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SynthOscWaveform {
    PolyBlepSaw,
    PolyBlepPulse,
    Sine,
    Triangle,
    SuperSawRich,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdsrStage {
    Idle,
    Attack,
    Decay,
    Sustain,
    Release,
}

#[derive(Clone, Copy, Debug)]
struct SynthVoice {
    active: bool,
    stage: AdsrStage,
    note_id: u32,
    key: u8,
    color_group: u8,
    age_samples: u64,
    phase: f64,
    sub_phase: f64,
    detune_phase_l: f64,
    detune_phase_r: f64,
    sample_cursor: f64,
    env_level: f32,
    current_pitch_semitones: f32,
    current_velocity: f32,
    current_pan: f32,
    // FL Studio 26 Slide / Portamento state
    slide_active: bool,
    slide_remaining_samples: usize,
    slide_pitch_step: f32,
    slide_target_pitch: f32,
    slide_vel_step: f32,
    slide_target_vel: f32,
    slide_pan_step: f32,
    slide_target_pan: f32,
    // Per-voice Cytomic SVF state [Left, Right]
    svf_ic1eq: [f32; 2],
    svf_ic2eq: [f32; 2],
}

impl Default for SynthVoice {
    fn default() -> Self {
        Self {
            active: false,
            stage: AdsrStage::Idle,
            note_id: 0,
            key: 60,
            color_group: 0,
            age_samples: 0,
            phase: 0.0,
            sub_phase: 0.0,
            detune_phase_l: 0.0,
            detune_phase_r: 0.0,
            sample_cursor: 0.0,
            env_level: 0.0,
            current_pitch_semitones: 60.0,
            current_velocity: 0.8,
            current_pan: 0.0,
            slide_active: false,
            slide_remaining_samples: 0,
            slide_pitch_step: 0.0,
            slide_target_pitch: 60.0,
            slide_vel_step: 0.0,
            slide_target_vel: 0.8,
            slide_pan_step: 0.0,
            slide_target_pan: 0.0,
            svf_ic1eq: [0.0; 2],
            svf_ic2eq: [0.0; 2],
        }
    }
}

/// 16-Voice Polyphonic Synthesizer & Sampler Generator (`UnifiedDspNode`) with FL Studio 26
/// `SlideNote` Portamento, Cytomic TPT SVF Filter, ADSR Envelope, Sample-Accurate Event Queue,
/// and configurable Lookahead Latency for **Point 1 APDC** (`ChannelGenerator`).
pub struct PolyphonicSynthGeneratorNode {
    pub channel_id: u32,
    pub waveform: SynthOscWaveform,
    pub attack_sec: f32,
    pub decay_sec: f32,
    pub sustain_level: f32,
    pub release_sec: f32,
    pub cutoff_hz: f32,
    pub resonance_q: f32,
    pub master_gain: f32,
    pub default_slide_samples: usize,
    sample_rate: f64,
    voices: [SynthVoice; MAX_SYNTH_VOICES],
    event_queue: Vec<SequencerVoiceEvent>,
    event_count: usize,
    sampler_pcm: Vec<StereoFrame>,
    sampler_root_key: u8,
    lookahead_latency_samples: usize,
    lookahead_delay: ApdcDelayLine,
    slide_events_processed: u64,
    notes_triggered_count: u64,
}

impl PolyphonicSynthGeneratorNode {
    pub fn new(channel_id: u32, sample_rate: f64) -> Self {
        Self {
            channel_id,
            waveform: SynthOscWaveform::SuperSawRich,
            attack_sec: 0.004,
            decay_sec: 0.18,
            sustain_level: 0.78,
            release_sec: 0.08,
            cutoff_hz: 8500.0,
            resonance_q: 0.85,
            master_gain: 0.35,
            default_slide_samples: (sample_rate * 0.08) as usize,
            sample_rate: sample_rate.max(8000.0),
            voices: [SynthVoice::default(); MAX_SYNTH_VOICES],
            event_queue: vec![SequencerVoiceEvent::default(); MAX_GENERATOR_VOICE_EVENTS],
            event_count: 0,
            sampler_pcm: Vec::new(),
            sampler_root_key: 60,
            lookahead_latency_samples: 0,
            lookahead_delay: ApdcDelayLine::with_capacity(DEFAULT_APDC_CAPACITY),
            slide_events_processed: 0,
            notes_triggered_count: 0,
        }
    }

    /// Configures the generator's intrinsic lookahead latency in samples (compensated by **Point 1 APDC**).
    pub fn with_lookahead_latency(mut self, lookahead_samples: usize) -> Self {
        self.set_lookahead_latency(lookahead_samples);
        self
    }

    pub fn set_lookahead_latency(&mut self, lookahead_samples: usize) {
        self.lookahead_latency_samples = lookahead_samples;
        self.lookahead_delay
            .ensure_capacity(lookahead_samples.max(DEFAULT_APDC_CAPACITY));
        self.lookahead_delay.set_delay_samples(lookahead_samples);
    }

    /// Loads an optional PCM stereo wavetable/sample buffer for hybrid Sampler + Synth playback.
    pub fn load_sampler_pcm(&mut self, pcm: Vec<StereoFrame>, root_key: u8) {
        self.sampler_pcm = pcm;
        self.sampler_root_key = root_key.min(127);
    }

    #[inline]
    pub fn active_voice_count(&self) -> usize {
        self.voices.iter().filter(|v| v.active).count()
    }

    #[inline]
    pub fn slide_events_processed(&self) -> u64 {
        self.slide_events_processed
    }

    #[inline]
    pub fn notes_triggered_count(&self) -> u64 {
        self.notes_triggered_count
    }

    /// Returns the instantaneous pitch in semitones of the first active voice in `color_group` (for testing Slide glide).
    #[inline]
    pub fn active_pitch_semitones_for_group(&self, color_group: u8) -> Option<f32> {
        self.voices
            .iter()
            .find(|v| v.active && v.color_group == color_group)
            .map(|v| v.current_pitch_semitones)
    }

    /// Schedules a sample-accurate `SequencerVoiceEvent` into the pre-allocated internal queue
    /// (Zero Heap Allocation).
    #[inline]
    pub fn schedule_voice_event(&mut self, ev: SequencerVoiceEvent) -> bool {
        if self.event_count >= self.event_queue.len() {
            return false;
        }
        let mut idx = self.event_count;
        while idx > 0 && self.event_queue[idx - 1].sample_pos > ev.sample_pos {
            self.event_queue[idx] = self.event_queue[idx - 1];
            idx -= 1;
        }
        self.event_queue[idx] = ev;
        self.event_count += 1;
        true
    }

    #[inline]
    fn apply_voice_event_immediate(&mut self, ev: &SequencerVoiceEvent) {
        match ev.kind {
            VoiceEventKind::NoteOn => {
                self.notes_triggered_count += 1;
                // Find an idle voice or steal the oldest releasing/active voice
                let mut chosen_idx = 0usize;
                let mut found_idle = false;
                for (i, v) in self.voices.iter().enumerate() {
                    if !v.active {
                        chosen_idx = i;
                        found_idle = true;
                        break;
                    }
                }
                if !found_idle {
                    let mut max_age = 0u64;
                    for (i, v) in self.voices.iter().enumerate() {
                        let score = v.age_samples + if v.stage == AdsrStage::Release { 1_000_000 } else { 0 };
                        if score >= max_age {
                            max_age = score;
                            chosen_idx = i;
                        }
                    }
                }

                let pitch = (ev.key as f32) + ev.fine_pitch_cents * 0.01;
                let voice = &mut self.voices[chosen_idx];
                *voice = SynthVoice {
                    active: true,
                    stage: AdsrStage::Attack,
                    note_id: ev.note_id,
                    key: ev.key,
                    color_group: ev.color_group & 0x0F,
                    age_samples: 0,
                    phase: 0.0,
                    sub_phase: 0.0,
                    detune_phase_l: 0.13,
                    detune_phase_r: 0.67,
                    sample_cursor: 0.0,
                    env_level: 0.0,
                    current_pitch_semitones: pitch,
                    current_velocity: ev.velocity.clamp(0.0, 1.0),
                    current_pan: ev.pan.clamp(-1.0, 1.0),
                    slide_active: false,
                    slide_remaining_samples: 0,
                    slide_pitch_step: 0.0,
                    slide_target_pitch: pitch,
                    slide_vel_step: 0.0,
                    slide_target_vel: ev.velocity.clamp(0.0, 1.0),
                    slide_pan_step: 0.0,
                    slide_target_pan: ev.pan.clamp(-1.0, 1.0),
                    svf_ic1eq: [0.0; 2],
                    svf_ic2eq: [0.0; 2],
                };
            }
            VoiceEventKind::NoteOff => {
                for v in &mut self.voices {
                    if v.active
                        && v.stage != AdsrStage::Release
                        && v.color_group == (ev.color_group & 0x0F)
                        && (v.note_id == ev.note_id || v.key == ev.key)
                    {
                        v.stage = AdsrStage::Release;
                    }
                }
            }
            VoiceEventKind::SlideNote => {
                self.slide_events_processed += 1;
                let target_pitch = (ev.key as f32) + ev.fine_pitch_cents * 0.01;
                let target_vel = ev.velocity.clamp(0.0, 1.0);
                let target_pan = ev.pan.clamp(-1.0, 1.0);
                let slide_samples = if ev.slide_duration_samples > 0 {
                    ev.slide_duration_samples
                } else {
                    self.default_slide_samples.max(1)
                };
                let inv_samples = 1.0 / (slide_samples as f32);
                let target_group = ev.color_group & 0x0F;

                for v in &mut self.voices {
                    if v.active && v.color_group == target_group {
                        v.slide_active = true;
                        v.slide_remaining_samples = slide_samples;
                        v.slide_target_pitch = target_pitch;
                        v.slide_pitch_step =
                            (target_pitch - v.current_pitch_semitones) * inv_samples;
                        v.slide_target_vel = target_vel;
                        v.slide_vel_step = (target_vel - v.current_velocity) * inv_samples;
                        v.slide_target_pan = target_pan;
                        v.slide_pan_step = (target_pan - v.current_pan) * inv_samples;
                    }
                }
            }
            VoiceEventKind::ChannelMute => {
                if ev.velocity <= 0.0 {
                    for v in &mut self.voices {
                        v.active = false;
                        v.stage = AdsrStage::Idle;
                    }
                }
            }
            VoiceEventKind::SlotEnable => {}
        }
    }

    #[inline(always)]
    fn poly_blep(t: f64, dt: f64) -> f64 {
        if t < dt {
            let x = t / dt;
            x + x - x * x - 1.0
        } else if t > 1.0 - dt {
            let x = (t - 1.0) / dt;
            x * x + x + x + 1.0
        } else {
            0.0
        }
    }
}

impl UnifiedDspNode for PolyphonicSynthGeneratorNode {
    fn name(&self) -> &'static str {
        "ToneForge Polyphonic Synth & Sampler (FL26 Slide + Point 1 APDC)"
    }

    fn latency_samples(&self) -> usize {
        self.lookahead_latency_samples
    }

    fn prepare(&mut self, sample_rate: f64, _max_block_size: usize) {
        self.sample_rate = sample_rate.max(8000.0);
        self.lookahead_delay
            .ensure_capacity(self.lookahead_latency_samples.max(DEFAULT_APDC_CAPACITY));
        self.lookahead_delay
            .set_delay_samples(self.lookahead_latency_samples);
    }

    fn reset_state(&mut self) {
        self.voices = [SynthVoice::default(); MAX_SYNTH_VOICES];
        self.event_count = 0;
        self.lookahead_delay.reset();
    }

    fn set_parameter_normalized(&mut self, param_id: u32, norm_val: f32) {
        let v = norm_val.clamp(0.0, 1.0);
        match param_id {
            0 => self.cutoff_hz = 80.0 * (20_000.0f32 / 80.0).powf(v),
            1 => self.resonance_q = 0.5 + v * 6.0,
            2 => self.attack_sec = 0.001 + v * 0.5,
            3 => self.release_sec = 0.005 + v * 1.5,
            4 => self.master_gain = v,
            _ => {}
        }
    }

    fn on_voice_event(&mut self, event: &SequencerVoiceEvent, ctx: &ProcessContext) {
        if event.sample_pos <= ctx.current_sample_pos {
            self.apply_voice_event_immediate(event);
        } else {
            let _ = self.schedule_voice_event(*event);
        }
    }

    fn process_slice(
        &mut self,
        ctx: &ProcessContext,
        main_io: &mut [StereoFrame],
        _sidechain_in: Option<&[StereoFrame]>,
        _aux_sends_out: &mut [&mut [StereoFrame]],
    ) {
        let sr = ctx.sample_rate.max(8000.0);
        let inv_sr = 1.0 / sr;
        let inv_sr_f32 = inv_sr as f32;

        let attack_step = 1.0 / ((self.attack_sec.max(0.0005) as f64 * sr) as f32);
        let decay_step =
            (1.0 - self.sustain_level) / ((self.decay_sec.max(0.001) as f64 * sr) as f32);
        let release_step = 1.0 / ((self.release_sec.max(0.001) as f64 * sr) as f32);

        // Precompute Cytomic TPT SVF coefficients for current cutoff & resonance
        let nyquist = (sr as f32) * 0.48;
        let fc = self.cutoff_hz.clamp(40.0, nyquist);
        let g = (std::f32::consts::PI * fc * inv_sr_f32).tan();
        let k = 1.0 / self.resonance_q.max(0.5);
        let a1 = 1.0 / (1.0 + g * (g + k));
        let a2 = g * a1;
        let a3 = g * a2;

        let mut ev_consumed = 0usize;

        for (frame_idx, frame) in main_io.iter_mut().enumerate() {
            let cur_pos = ctx.current_sample_pos + (frame_idx as i64);

            // Dispatch any queued sample-accurate voice events due at or before `cur_pos`
            while ev_consumed < self.event_count {
                let ev = self.event_queue[ev_consumed];
                if ev.sample_pos <= cur_pos {
                    self.apply_voice_event_immediate(&ev);
                    ev_consumed += 1;
                } else {
                    break;
                }
            }

            let mut sum_l = 0.0f32;
            let mut sum_r = 0.0f32;

            for v in &mut self.voices {
                if !v.active {
                    continue;
                }
                v.age_samples += 1;

                // 1. Advance FL Studio 26 Slide / Portamento glide if active
                if v.slide_active {
                    if v.slide_remaining_samples > 1 {
                        v.current_pitch_semitones += v.slide_pitch_step;
                        v.current_velocity =
                            (v.current_velocity + v.slide_vel_step).clamp(0.0, 1.0);
                        v.current_pan = (v.current_pan + v.slide_pan_step).clamp(-1.0, 1.0);
                        v.slide_remaining_samples -= 1;
                    } else {
                        v.current_pitch_semitones = v.slide_target_pitch;
                        v.current_velocity = v.slide_target_vel;
                        v.current_pan = v.slide_target_pan;
                        v.slide_remaining_samples = 0;
                        v.slide_active = false;
                    }
                }

                // 2. Advance ADSR envelope
                match v.stage {
                    AdsrStage::Idle => {
                        v.active = false;
                        continue;
                    }
                    AdsrStage::Attack => {
                        v.env_level += attack_step;
                        if v.env_level >= 1.0 {
                            v.env_level = 1.0;
                            v.stage = AdsrStage::Decay;
                        }
                    }
                    AdsrStage::Decay => {
                        v.env_level -= decay_step;
                        if v.env_level <= self.sustain_level {
                            v.env_level = self.sustain_level;
                            v.stage = AdsrStage::Sustain;
                        }
                    }
                    AdsrStage::Sustain => {
                        v.env_level = self.sustain_level;
                    }
                    AdsrStage::Release => {
                        v.env_level -= release_step;
                        if v.env_level <= 0.0001 {
                            v.env_level = 0.0;
                            v.active = false;
                            v.stage = AdsrStage::Idle;
                            continue;
                        }
                    }
                }

                // 3. Compute instantaneous oscillator frequency from `current_pitch_semitones`
                let freq_hz =
                    (440.0 * ((v.current_pitch_semitones as f64 - 69.0) / 12.0).exp2()).clamp(20.0, sr * 0.47);
                let dt = freq_hz * inv_sr;

                let (mut sig_l, mut sig_r) = if !self.sampler_pcm.is_empty() {
                    let ratio = ((v.current_pitch_semitones - self.sampler_root_key as f32) / 12.0)
                        .exp2() as f64;
                    let idx0 = v.sample_cursor as usize;
                    if idx0 + 1 < self.sampler_pcm.len() {
                        let frac = (v.sample_cursor - idx0 as f64) as f32;
                        let s0 = self.sampler_pcm[idx0];
                        let s1 = self.sampler_pcm[idx0 + 1];
                        v.sample_cursor += ratio;
                        (
                            s0[0] + frac * (s1[0] - s0[0]),
                            s0[1] + frac * (s1[1] - s0[1]),
                        )
                    } else {
                        v.sample_cursor = 0.0;
                        (0.0, 0.0)
                    }
                } else {
                    match self.waveform {
                        SynthOscWaveform::Sine => {
                            let s = (v.phase * TAU).sin() as f32;
                            (s, s)
                        }
                        SynthOscWaveform::Triangle => {
                            let s = (1.0 - 4.0 * (v.phase - 0.5).abs()) as f32;
                            (s, s)
                        }
                        SynthOscWaveform::PolyBlepSaw => {
                            let saw = 2.0 * v.phase - 1.0 - Self::poly_blep(v.phase, dt);
                            (saw as f32, saw as f32)
                        }
                        SynthOscWaveform::PolyBlepPulse => {
                            let p = if v.phase < 0.5 { 1.0 } else { -1.0 }
                                + Self::poly_blep(v.phase, dt)
                                - Self::poly_blep((v.phase + 0.5).fract(), dt);
                            (p as f32, p as f32)
                        }
                        SynthOscWaveform::SuperSawRich => {
                            let center = 2.0 * v.phase - 1.0 - Self::poly_blep(v.phase, dt);
                            let sub = (v.sub_phase * TAU).sin() * 0.35;
                            let dt_l = (freq_hz * 1.0045 * inv_sr).min(0.48);
                            let dt_r = (freq_hz * 0.9955 * inv_sr).min(0.48);
                            let saw_l =
                                2.0 * v.detune_phase_l - 1.0 - Self::poly_blep(v.detune_phase_l, dt_l);
                            let saw_r =
                                2.0 * v.detune_phase_r - 1.0 - Self::poly_blep(v.detune_phase_r, dt_r);
                            v.detune_phase_l = (v.detune_phase_l + dt_l).fract();
                            v.detune_phase_r = (v.detune_phase_r + dt_r).fract();
                            (
                                ((center * 0.5 + saw_l * 0.35 + sub) as f32),
                                ((center * 0.5 + saw_r * 0.35 + sub) as f32),
                            )
                        }
                    }
                };

                v.phase = (v.phase + dt).fract();
                v.sub_phase = (v.sub_phase + dt * 0.5).fract();

                // 4. Per-voice Cytomic TPT SVF 2-pole lowpass filter
                let raw = [sig_l, sig_r];
                let mut filt = [0.0f32; 2];
                for ch in 0..2 {
                    let v3 = raw[ch] - v.svf_ic2eq[ch];
                    let v1 = a1 * v.svf_ic1eq[ch] + a2 * v3;
                    let v2 = v.svf_ic2eq[ch] + a2 * v.svf_ic1eq[ch] + a3 * v3;
                    v.svf_ic1eq[ch] = 2.0 * v1 - v.svf_ic1eq[ch];
                    v.svf_ic2eq[ch] = 2.0 * v2 - v.svf_ic2eq[ch];
                    filt[ch] = v2;
                }
                sig_l = filt[0];
                sig_r = filt[1];

                // 5. Equal-power stereo pan + velocity + ADSR envelope
                let pan_norm = (v.current_pan + 1.0) * 0.5;
                let angle = pan_norm * std::f32::consts::FRAC_PI_2;
                let gain_v = v.env_level * v.current_velocity * self.master_gain;
                sum_l += sig_l * angle.cos() * gain_v;
                sum_r += sig_r * angle.sin() * gain_v;
            }

            frame[0] += sum_l;
            frame[1] += sum_r;
        }

        if ev_consumed > 0 {
            let rem = self.event_count - ev_consumed;
            for i in 0..rem {
                self.event_queue[i] = self.event_queue[i + ev_consumed];
            }
            self.event_count = rem;
        }

        // Apply intrinsic generator lookahead delay (compensated by Point 1 APDC on `ChannelGenerator`)
        if self.lookahead_latency_samples > 0 {
            self.lookahead_delay.process_in_place(main_io);
        }
    }
}

// ============================================================================
// INTERNAL GENERATOR 2: SAMPLE-ACCURATE AUDIO & STEM CLIP PLAYER
// ============================================================================

/// Timeline region scheduled inside an `AudioStemClipPlayerNode`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StemClipRegion {
    pub clip_id: u32,
    pub timeline_start_sample: i64,
    pub timeline_duration_samples: usize,
    pub source_offset_samples: usize,
    pub fade_in_samples: usize,
    pub fade_out_samples: usize,
    pub gain: f32,
    pub pan: f32,
    pub pitch_semitones: f32,
}

impl Default for StemClipRegion {
    fn default() -> Self {
        Self {
            clip_id: 0,
            timeline_start_sample: 0,
            timeline_duration_samples: 0,
            source_offset_samples: 0,
            fade_in_samples: 64,
            fade_out_samples: 64,
            gain: 1.0,
            pan: 0.0,
            pitch_semitones: 0.0,
        }
    }
}

/// Sample-Accurate Audio Clip & UVR5 Stem Player Generator (`UnifiedDspNode`) with 4-point
/// Hermite cubic pitch/resample interpolation, de-clicked fade-in/out envelopes, and Point 1
/// Lookahead APDC support.
pub struct AudioStemClipPlayerNode {
    pub channel_id: u32,
    pub stem_name: &'static str,
    audio_data: Vec<StereoFrame>,
    regions: Vec<StemClipRegion>,
    region_count: usize,
    pub continuous_mode: bool,
    pub loop_enabled: bool,
    pub master_gain: f32,
    pub pan: f32,
    pub pitch_semitones: f32,
    lookahead_latency_samples: usize,
    lookahead_delay: ApdcDelayLine,
}

impl AudioStemClipPlayerNode {
    pub fn new(channel_id: u32, stem_name: &'static str, audio_data: Vec<StereoFrame>) -> Self {
        let len = audio_data.len();
        let mut regions = vec![StemClipRegion::default(); MAX_STEM_CLIP_REGIONS];
        let mut region_count = 0usize;
        if len > 0 {
            regions[0] = StemClipRegion {
                clip_id: 1,
                timeline_start_sample: 0,
                timeline_duration_samples: len,
                source_offset_samples: 0,
                fade_in_samples: 32.min(len / 4),
                fade_out_samples: 32.min(len / 4),
                gain: 1.0,
                pan: 0.0,
                pitch_semitones: 0.0,
            };
            region_count = 1;
        }
        Self {
            channel_id,
            stem_name,
            audio_data,
            regions,
            region_count,
            continuous_mode: false,
            loop_enabled: false,
            master_gain: 1.0,
            pan: 0.0,
            pitch_semitones: 0.0,
            lookahead_latency_samples: 0,
            lookahead_delay: ApdcDelayLine::with_capacity(DEFAULT_APDC_CAPACITY),
        }
    }

    pub fn with_lookahead_latency(mut self, lookahead_samples: usize) -> Self {
        self.set_lookahead_latency(lookahead_samples);
        self
    }

    pub fn set_lookahead_latency(&mut self, lookahead_samples: usize) {
        self.lookahead_latency_samples = lookahead_samples;
        self.lookahead_delay
            .ensure_capacity(lookahead_samples.max(DEFAULT_APDC_CAPACITY));
        self.lookahead_delay.set_delay_samples(lookahead_samples);
    }

    pub fn set_audio_data(&mut self, audio_data: Vec<StereoFrame>) {
        self.audio_data = audio_data;
    }

    #[inline]
    pub fn audio_data(&self) -> &[StereoFrame] {
        &self.audio_data
    }

    pub fn clear_regions(&mut self) {
        self.region_count = 0;
    }

    pub fn add_region(&mut self, region: StemClipRegion) -> bool {
        if self.region_count >= self.regions.len() {
            return false;
        }
        self.regions[self.region_count] = region;
        self.region_count += 1;
        true
    }

    #[inline(always)]
    fn hermite4_stereo(
        buf: &[StereoFrame],
        pos: f64,
        loop_enabled: bool,
    ) -> StereoFrame {
        let n = buf.len();
        if n == 0 {
            return [0.0, 0.0];
        }
        let idx0 = pos.floor() as isize;
        let frac = (pos - idx0 as f64) as f32;

        let fetch = |i: isize| -> StereoFrame {
            if loop_enabled {
                buf[(i.rem_euclid(n as isize)) as usize]
            } else if i < 0 {
                buf[0]
            } else if (i as usize) >= n {
                [0.0, 0.0]
            } else {
                buf[i as usize]
            }
        };

        let ym1 = fetch(idx0 - 1);
        let y0 = fetch(idx0);
        let y1 = fetch(idx0 + 1);
        let y2 = fetch(idx0 + 2);

        let mut out = [0.0f32; 2];
        for ch in 0..2 {
            let c0 = y0[ch];
            let c1 = 0.5 * (y1[ch] - ym1[ch]);
            let c2 = ym1[ch] - 2.5 * y0[ch] + 2.0 * y1[ch] - 0.5 * y2[ch];
            let c3 = 0.5 * (y2[ch] - ym1[ch]) + 1.5 * (y0[ch] - y1[ch]);
            out[ch] = ((c3 * frac + c2) * frac + c1) * frac + c0;
        }
        out
    }
}

impl UnifiedDspNode for AudioStemClipPlayerNode {
    fn name(&self) -> &'static str {
        self.stem_name
    }

    fn latency_samples(&self) -> usize {
        self.lookahead_latency_samples
    }

    fn prepare(&mut self, _sample_rate: f64, _max_block_size: usize) {
        self.lookahead_delay
            .ensure_capacity(self.lookahead_latency_samples.max(DEFAULT_APDC_CAPACITY));
        self.lookahead_delay
            .set_delay_samples(self.lookahead_latency_samples);
    }

    fn reset_state(&mut self) {
        self.lookahead_delay.reset();
    }

    fn set_parameter_normalized(&mut self, param_id: u32, norm_val: f32) {
        let v = norm_val.clamp(0.0, 1.0);
        match param_id {
            0 => self.master_gain = v * 2.0,
            1 => self.pan = v * 2.0 - 1.0,
            2 => self.pitch_semitones = (v - 0.5) * 24.0,
            _ => {}
        }
    }

    fn process_slice(
        &mut self,
        ctx: &ProcessContext,
        main_io: &mut [StereoFrame],
        _sidechain_in: Option<&[StereoFrame]>,
        _aux_sends_out: &mut [&mut [StereoFrame]],
    ) {
        let src_len = self.audio_data.len();
        if !ctx.is_playing || src_len == 0 {
            if self.lookahead_latency_samples > 0 {
                self.lookahead_delay.process_in_place(main_io);
            }
            return;
        }

        let global_rate = ((self.pitch_semitones / 12.0).exp2()) as f64;

        for (frame_idx, frame) in main_io.iter_mut().enumerate() {
            let cur_pos = ctx.current_sample_pos + (frame_idx as i64);
            let mut acc_l = 0.0f32;
            let mut acc_r = 0.0f32;

            if self.continuous_mode {
                if cur_pos >= 0 {
                    let src_pos = (cur_pos as f64) * global_rate;
                    if self.loop_enabled || (src_pos as usize) < src_len {
                        let s = Self::hermite4_stereo(&self.audio_data, src_pos, self.loop_enabled);
                        acc_l += s[0];
                        acc_r += s[1];
                    }
                }
            } else {
                for r_idx in 0..self.region_count {
                    let reg = &self.regions[r_idx];
                    let reg_end = reg.timeline_start_sample + (reg.timeline_duration_samples as i64);
                    if cur_pos < reg.timeline_start_sample || cur_pos >= reg_end {
                        continue;
                    }
                    let rel = (cur_pos - reg.timeline_start_sample) as usize;
                    let rate =
                        (((self.pitch_semitones + reg.pitch_semitones) / 12.0).exp2()) as f64;
                    let src_pos = (reg.source_offset_samples as f64) + (rel as f64) * rate;
                    if !self.loop_enabled && (src_pos as usize) >= src_len {
                        continue;
                    }
                    let s = Self::hermite4_stereo(&self.audio_data, src_pos, self.loop_enabled);

                    // De-click fade-in and fade-out envelope
                    let mut env = reg.gain;
                    if reg.fade_in_samples > 0 && rel < reg.fade_in_samples {
                        env *= (rel as f32) / (reg.fade_in_samples as f32);
                    }
                    let rem = reg.timeline_duration_samples.saturating_sub(rel);
                    if reg.fade_out_samples > 0 && rem < reg.fade_out_samples {
                        env *= (rem as f32) / (reg.fade_out_samples as f32);
                    }

                    let pan_norm = ((self.pan + reg.pan).clamp(-1.0, 1.0) + 1.0) * 0.5;
                    let angle = pan_norm * std::f32::consts::FRAC_PI_2;
                    acc_l += s[0] * angle.cos() * env;
                    acc_r += s[1] * angle.sin() * env;
                }
            }

            frame[0] += acc_l * self.master_gain;
            frame[1] += acc_r * self.master_gain;
        }

        if self.lookahead_latency_samples > 0 {
            self.lookahead_delay.process_in_place(main_io);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slicer_ppq::HybridPpqSlicer;

    #[test]
    fn test_piano_roll_slide_note_glides_active_voice_pitch() {
        let sr = 48_000.0;
        let mut synth = PolyphonicSynthGeneratorNode::new(1, sr);
        let ctx0 = ProcessContext {
            sample_rate: sr,
            current_sample_pos: 0,
            ppq_tick: 0,
            bpm: 120.0,
            slice_frames: 256,
            is_playing: true,
            is_offline_rendering: false,
        };

        // Trigger C4 (key 60) on color_group 0 at sample 0
        synth.on_voice_event(
            &SequencerVoiceEvent {
                sample_pos: 0,
                ppq_tick: 0,
                channel_id: 1,
                note_id: 1,
                kind: VoiceEventKind::NoteOn,
                key: 60,
                velocity: 0.9,
                pan: 0.0,
                fine_pitch_cents: 0.0,
                slide_duration_ticks: 96,
                slide_duration_samples: 0,
                color_group: 0,
            },
            &ctx0,
        );
        assert_eq!(synth.active_voice_count(), 1);
        assert!((synth.active_pitch_semitones_for_group(0).unwrap() - 60.0).abs() < 1e-5);

        // Trigger FL Studio 26 Slide Note to C5 (key 72) over 256 samples
        synth.on_voice_event(
            &SequencerVoiceEvent {
                sample_pos: 0,
                ppq_tick: 0,
                channel_id: 1,
                note_id: 2,
                kind: VoiceEventKind::SlideNote,
                key: 72,
                velocity: 0.9,
                pan: 0.5,
                fine_pitch_cents: 0.0,
                slide_duration_ticks: 96,
                slide_duration_samples: 256,
                color_group: 0,
            },
            &ctx0,
        );
        // Slide note must NOT allocate a second voice; it glides the existing voice in color_group 0
        assert_eq!(synth.active_voice_count(), 1);

        let mut buf = vec![[0.0f32; 2]; 128];
        synth.process_slice(&ctx0, &mut buf, None, &mut []);
        let mid_pitch = synth.active_pitch_semitones_for_group(0).unwrap();
        assert!(
            mid_pitch > 65.5 && mid_pitch < 66.5,
            "Expected pitch near 66.0 halfway through slide, got {mid_pitch}"
        );

        let ctx1 = ProcessContext {
            current_sample_pos: 128,
            ..ctx0
        };
        synth.process_slice(&ctx1, &mut buf, None, &mut []);
        let end_pitch = synth.active_pitch_semitones_for_group(0).unwrap();
        assert!(
            (end_pitch - 72.0).abs() < 1e-3,
            "Expected pitch at target 72.0 after 256-sample slide, got {end_pitch}"
        );
    }

    #[test]
    fn test_playlist_sequencer_dispatches_notes_and_tension_spline_automation() {
        let sr = 48_000.0;
        let bpm = 120.0;
        let slicer = HybridPpqSlicer::new(sr, bpm, 96);

        let mut seq = PlaylistSequencer::new();
        let mut pat = PianoRollPattern::new(1, "Lead Riff", 10, 384);
        pat.add_note(PianoRollNote {
            id: 1,
            start_tick: 0,
            duration_ticks: 48,
            key: 64,
            velocity: 0.85,
            pan: -0.2,
            fine_pitch_cents: 15.0,
            is_slide: false,
            color_group: 0,
        });
        pat.add_note(PianoRollNote {
            id: 2,
            start_tick: 24,
            duration_ticks: 24,
            key: 76,
            velocity: 1.0,
            pan: 0.4,
            fine_pitch_cents: 0.0,
            is_slide: true,
            color_group: 0,
        });
        seq.upsert_pattern(pat);
        seq.upsert_clip(PlaylistClip {
            clip_id: 100,
            track_row: 0,
            kind: PlaylistClipKind::Pattern,
            source_id: 1,
            start_tick: 0,
            duration_ticks: 384,
            clip_offset_ticks: 0,
            gain: 1.0,
            muted: false,
        });

        let synth = PolyphonicSynthGeneratorNode::new(10, sr);
        let mut channels = vec![crate::mixer_dag::ChannelGenerator::new(
            10,
            1,
            Box::new(synth),
        )];
        let mut bus = InternalControllerBus::new();

        seq.on_ppq_tick(0, 0, bpm, sr, slicer.ppq, &mut channels, &mut bus);
        let tick24_sample = slicer.tick_to_sample_pos(24.0);
        seq.on_ppq_tick(
            24,
            tick24_sample,
            bpm,
            sr,
            slicer.ppq,
            &mut channels,
            &mut bus,
        );

        let ctx = ProcessContext {
            sample_rate: sr,
            current_sample_pos: tick24_sample,
            ppq_tick: 24,
            bpm,
            slice_frames: 64,
            is_playing: true,
            is_offline_rendering: false,
        };
        channels[0].render_slice(&ctx, 0, 64);
        let peak = channels[0].output_buf[..64]
            .iter()
            .fold(0.0f32, |acc, f| acc.max(f[0].abs()).max(f[1].abs()));
        assert!(peak > 1e-3, "Expected active synth output, got peak {peak}");
    }
}
