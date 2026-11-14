//! Global 64-Step Bi-Directional Project Undo/Redo History & Lock-Free RT Diff Synchronizer (`undo_history.rs`).
//!
//! Covers 100% of ToneForge Studio v3 project mutations:
//! - Piano Roll note creation, deletion, dragging, velocity/pan/fine-pitch expression, Quantize, Strum, Arpeggiate
//! - Playlist clip placement, Razor Slice, Slip Trim, Equal-Power Fade In/Out, Clip Gain, Time-Stretch, Pitch, Take selection
//! - FabFilter Pro-Q 4 (`601` params, EQ Sketch, Collision Reference Track)
//! - oeksound Soothe3 (`143` params, External Spectral Sidechain Ducking)
//! - iZotope Ozone 12 (`Low End Focus`, `Codec Preview`, `IRC V Maximizer`, `12AX7 Triode`, `Imager`, `Stabilizer`)
//! - 502-Track Mixer gains, Send/Sidechain DAG routing, Hardware Audio/MIDI Record Arming
//! - External Steinberg VST3 / CLAP plugin insertion, removal, binary state, and parameter automation
//! - Pitcher RT & NewTone Vocal Note Blob edits

use crate::ipc_commands::{
    DawAppState, cmd_update_bsrnn_stem_state, cmd_update_ozone12_master,
};
use crate::project_state::{
    ExternalPluginStateDto, Ozone12MasterConfigDto, PitcherStateDto, PluginConfigDto,
    ProQ4StateDto, Soothe3StateDto, ToneForgeProjectState, decode_hex_bytes,
};
use ai_uvr5_stem_engine::RealTimeBsrnnStemNode;
use daw_core_graph::{
    EmphasisLimiterNode, EngineCommand, GrossBeatTimeBufferNode, LuxeVerbFdnShimmerNode,
    MAX_BLOCK_SIZE, MaximusMultibandNode, MixerSendRoute, PluginRackSlot, RtGraphMutation,
    SparseCompressorNode, UnifiedDspNode,
};
use daw_plugin_host::{ClapHostedPluginNode, Vst3HostedPluginNode};
use dsp_ozone12_mastering::Ozone12MasterChain;
use dsp_pitch_time::{AutoPitchCorrectorNode, MusicalScale};
use dsp_proq4_eq::{PROQ4_TOTAL_PARAMS, ProQ4CollisionBus, ProQ4Processor};
use dsp_soothe3_spectral::{SOOTHE3_TOTAL_PARAMS, Soothe3Processor};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;

pub const DEFAULT_UNDO_HISTORY_LIMIT: usize = 64;

/// UI-facing Undo/Redo stack status returned to `App.svelte` (`Ctrl+Z` / `Ctrl+Y`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UndoRedoStatusDto {
    pub can_undo: bool,
    pub can_redo: bool,
    pub undo_depth: usize,
    pub redo_depth: usize,
    pub next_undo_label: Option<String>,
    pub next_redo_label: Option<String>,
    pub last_applied_label: Option<String>,
}

#[derive(Clone, Debug)]
struct UndoHistoryEntry {
    label: String,
    snapshot: ToneForgeProjectState,
}

/// Complete state transition returned by [`ProjectUndoHistory::undo`] and [`ProjectUndoHistory::redo`].
#[derive(Clone, Debug)]
pub struct UndoRedoTransition {
    pub action_label: String,
    pub previous_state: ToneForgeProjectState,
    pub restored_state: ToneForgeProjectState,
    pub status: UndoRedoStatusDto,
}

/// 64-Step Global Bi-Directional Undo/Redo History Manager.
#[derive(Clone, Debug)]
pub struct ProjectUndoHistory {
    max_entries: usize,
    undo_stack: VecDeque<UndoHistoryEntry>,
    redo_stack: Vec<UndoHistoryEntry>,
    last_applied_label: Option<String>,
}

impl Default for ProjectUndoHistory {
    fn default() -> Self {
        Self::new(DEFAULT_UNDO_HISTORY_LIMIT)
    }
}

impl ProjectUndoHistory {
    pub fn new(max_entries: usize) -> Self {
        let cap = max_entries.clamp(1, 512);
        Self {
            max_entries: cap,
            undo_stack: VecDeque::with_capacity(cap),
            redo_stack: Vec::with_capacity(cap),
            last_applied_label: None,
        }
    }

    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.last_applied_label = None;
    }

    #[inline]
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    #[inline]
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    pub fn status(&self) -> UndoRedoStatusDto {
        UndoRedoStatusDto {
            can_undo: self.can_undo(),
            can_redo: self.can_redo(),
            undo_depth: self.undo_stack.len(),
            redo_depth: self.redo_stack.len(),
            next_undo_label: self.undo_stack.back().map(|e| e.label.clone()),
            next_redo_label: self.redo_stack.last().map(|e| e.label.clone()),
            last_applied_label: self.last_applied_label.clone(),
        }
    }

    /// Captures `current_state` BEFORE applying a user/IPC mutation labeled `label`,
    /// clearing any forward `redo_stack` branch.
    pub fn record_before_action(
        &mut self,
        label: impl Into<String>,
        current_state: &ToneForgeProjectState,
    ) {
        if self.undo_stack.len() >= self.max_entries {
            self.undo_stack.pop_front();
        }
        self.undo_stack.push_back(UndoHistoryEntry {
            label: label.into(),
            snapshot: current_state.clone(),
        });
        self.redo_stack.clear();
    }

    /// Steps backward one state (`Ctrl+Z`), pushing the displaced `current_state` onto `redo_stack`.
    pub fn undo(&mut self, current_state: &mut ToneForgeProjectState) -> Option<UndoRedoTransition> {
        let entry = self.undo_stack.pop_back()?;
        let previous_state = current_state.clone();
        if self.redo_stack.len() >= self.max_entries {
            self.redo_stack.remove(0);
        }
        self.redo_stack.push(UndoHistoryEntry {
            label: entry.label.clone(),
            snapshot: previous_state.clone(),
        });
        *current_state = entry.snapshot.clone();
        self.last_applied_label = Some(format!("Undo: {}", entry.label));
        let status = self.status();
        Some(UndoRedoTransition {
            action_label: entry.label,
            previous_state,
            restored_state: entry.snapshot,
            status,
        })
    }

    /// Steps forward one state (`Ctrl+Y` / `Ctrl+Shift+Z`), pushing the displaced `current_state` onto `undo_stack`.
    pub fn redo(&mut self, current_state: &mut ToneForgeProjectState) -> Option<UndoRedoTransition> {
        let entry = self.redo_stack.pop()?;
        let previous_state = current_state.clone();
        if self.undo_stack.len() >= self.max_entries {
            self.undo_stack.pop_front();
        }
        self.undo_stack.push_back(UndoHistoryEntry {
            label: entry.label.clone(),
            snapshot: previous_state.clone(),
        });
        *current_state = entry.snapshot.clone();
        self.last_applied_label = Some(format!("Redo: {}", entry.label));
        let status = self.status();
        Some(UndoRedoTransition {
            action_label: entry.label,
            previous_state,
            restored_state: entry.snapshot,
            status,
        })
    }
}

/// Instantiates a boxed [`UnifiedDspNode`] on the non-RT Host thread from any [`PluginConfigDto`] variant.
pub fn build_plugin_node_from_dto(
    track_idx: usize,
    sample_rate: f64,
    max_block_size: usize,
    plugin: &PluginConfigDto,
    collision_bus: Option<&Arc<ProQ4CollisionBus>>,
) -> Result<Box<dyn UnifiedDspNode>, String> {
    let sr = sample_rate.max(8_000.0);
    let blk = max_block_size.clamp(16, MAX_BLOCK_SIZE);

    match plugin {
        PluginConfigDto::ProQ4(pq_dto) => Ok(build_proq4_node(track_idx, sr, blk, pq_dto, collision_bus)),
        PluginConfigDto::Soothe3(s3_dto) => Ok(build_soothe3_node(sr, blk, s3_dto)),
        PluginConfigDto::Ozone12Master(oz_dto) => Ok(build_ozone12_node(sr, blk, oz_dto)),
        PluginConfigDto::RealTimeBsrnnStem(bsrnn_dto) => {
            let mut node = Box::new(RealTimeBsrnnStemNode::new(sr));
            node.prepare(sr, blk);
            node.stem_gains_db = bsrnn_dto.stem_gains_db;
            node.stem_solo = bsrnn_dto.stem_solo;
            node.stem_mute = bsrnn_dto.stem_mute;
            Ok(node)
        }
        PluginConfigDto::PatcherModular(patcher_dto) => {
            let mut patcher = patcher_dto.build_patcher_node(sr, blk)?;
            patcher.prepare(sr, blk);
            Ok(Box::new(patcher))
        }
        PluginConfigDto::AutoPitchCorrector(pitcher_dto) => {
            Ok(build_pitcher_node(sr, blk, pitcher_dto))
        }
        PluginConfigDto::ExternalVst3(ext_dto) => build_external_vst3_node(sr, blk, ext_dto),
        PluginConfigDto::ExternalClap(ext_dto) => build_external_clap_node(sr, blk, ext_dto),
        PluginConfigDto::NativeSparseCompressor {
            threshold_linear,
            ratio,
            makeup_gain,
            lookahead_samples,
        } => {
            let mut node = SparseCompressorNode::new(
                *threshold_linear,
                *ratio,
                *makeup_gain,
                *lookahead_samples,
            );
            node.prepare(sr, blk);
            Ok(Box::new(node))
        }
        PluginConfigDto::NativeMaximusMultiband => {
            let mut node = MaximusMultibandNode::new();
            node.prepare(sr, blk);
            Ok(Box::new(node))
        }
        PluginConfigDto::NativeEmphasisLimiter {
            ceiling_linear,
            drive_linear: _,
        } => {
            let ceiling_db = 20.0 * ceiling_linear.max(1e-6).log10();
            let mut node = EmphasisLimiterNode::new(64, ceiling_db);
            node.prepare(sr, blk);
            Ok(Box::new(node))
        }
        PluginConfigDto::NativeGrossBeat => {
            let mut node = GrossBeatTimeBufferNode::new((sr as usize) * 2);
            node.prepare(sr, blk);
            Ok(Box::new(node))
        }
        PluginConfigDto::NativeLuxeVerbFdn => {
            let mut node = LuxeVerbFdnShimmerNode::new();
            node.prepare(sr, blk);
            Ok(Box::new(node))
        }
        PluginConfigDto::Empty => Ok(Box::new(daw_core_graph::EmptySlotNode::new())),
    }
}

pub fn build_proq4_node(
    track_idx: usize,
    sr: f64,
    blk: usize,
    pq_dto: &ProQ4StateDto,
    collision_bus: Option<&Arc<ProQ4CollisionBus>>,
) -> Box<dyn UnifiedDspNode> {
    let bus = collision_bus
        .cloned()
        .unwrap_or_else(|| ProQ4CollisionBus::global().clone());
    bus.set_reference_track(track_idx as u32, pq_dto.collision_reference_track);
    let mut proc = Box::new(ProQ4Processor::with_collision_bus(track_idx as u32, bus));
    proc.prepare(sr, blk);
    let ps = pq_dto.to_parameter_set();
    for pid in 0..PROQ4_TOTAL_PARAMS {
        proc.set_parameter_normalized(pid as u32, ps.get_normalized(pid as u32));
    }
    proc
}

pub fn build_soothe3_node(
    sr: f64,
    blk: usize,
    s3_dto: &Soothe3StateDto,
) -> Box<dyn UnifiedDspNode> {
    let mut proc = Box::new(Soothe3Processor::new(sr as f32));
    proc.prepare(sr, blk);
    let ps = s3_dto.to_parameter_set();
    for pid in 0..SOOTHE3_TOTAL_PARAMS {
        proc.set_parameter_normalized(pid as u32, ps.get_normalized(pid as u32));
    }
    proc
}

pub fn apply_ozone12_dto_to_chain(chain: &mut Ozone12MasterChain, oz_dto: &Ozone12MasterConfigDto) {
    let updates: [(u32, f32); 32] = [
        (0, if oz_dto.maximizer_bypass { 1.0 } else { 0.0 }),
        (1, ((oz_dto.maximizer_gain_db + 20.0) / 40.0).clamp(0.0, 1.0)),
        (
            2,
            ((oz_dto.maximizer_ceiling_dbtp + 20.0) / 20.0).clamp(0.0, 1.0),
        ),
        (5, oz_dto.maximizer_upward_comp_wet.clamp(0.0, 1.0)),
        (6, if oz_dto.maximizer_soft_clip_enabled { 1.0 } else { 0.0 }),
        (7, oz_dto.maximizer_soft_clip_mix.clamp(0.0, 1.0)),
        (8, (oz_dto.maximizer_soft_clip_mode.min(2) as f32) * 0.5),
        (
            9,
            ((oz_dto.maximizer_transient_shaping + 1.0) * 0.5).clamp(0.0, 1.0),
        ),
        (65, (oz_dto.maximizer_irc_mode.min(8) as f32) / 8.0),
        (66, if oz_dto.maximizer_true_peak { 1.0 } else { 0.0 }),
        (10, if oz_dto.exciter_bypass { 1.0 } else { 0.0 }),
        (33, if oz_dto.imager_stereoizer_enabled { 1.0 } else { 0.0 }),
        (34, if oz_dto.imager_stereoizer_mode > 0 { 1.0 } else { 0.0 }),
        (36, (oz_dto.stabilizer_amount_pct * 0.01).clamp(0.0, 1.0)),
        (37, (oz_dto.stabilizer_speed_pct * 0.01).clamp(0.0, 1.0)),
        (
            39,
            (oz_dto.stabilizer_tame_transients_pct * 0.01).clamp(0.0, 1.0),
        ),
        (40, if oz_dto.stabilizer_cut_mode { 1.0 } else { 0.0 }),
        (55, (oz_dto.clarity_amount_pct * 0.01).clamp(0.0, 1.0)),
        (
            56,
            ((oz_dto.clarity_tilt_db_oct + 6.0) / 12.0).clamp(0.0, 1.0),
        ),
        (80, (oz_dto.unlimiter_amount_pct * 0.01).clamp(0.0, 1.0)),
        (
            81,
            ((oz_dto.unlimiter_threshold_db + 24.0) / 24.0).clamp(0.0, 1.0),
        ),
        (83, if oz_dto.unlimiter_enabled { 1.0 } else { 0.0 }),
        (70, if oz_dto.low_end_focus_enabled { 1.0 } else { 0.0 }),
        (71, if oz_dto.low_end_focus_punchy { 0.0 } else { 1.0 }),
        (
            72,
            ((oz_dto.low_end_focus_contrast_pct + 100.0) / 200.0).clamp(0.0, 1.0),
        ),
        (
            73,
            ((oz_dto.low_end_focus_upper_hz.clamp(20.0, 300.0) - 20.0) / 280.0).clamp(0.0, 1.0),
        ),
        (
            74,
            ((oz_dto.low_end_focus_gain_db + 12.0) / 24.0).clamp(0.0, 1.0),
        ),
        (69, if oz_dto.low_end_focus_solo { 1.0 } else { 0.0 }),
        (85, if oz_dto.codec_preview_enabled { 1.0 } else { 0.0 }),
        (
            86,
            ((oz_dto.codec_preview_codec.min(5) as f32) / 5.0).clamp(0.0, 1.0),
        ),
        (87, if oz_dto.codec_preview_artifact_solo { 1.0 } else { 0.0 }),
        (4, 0.5),
    ];
    for (pid, val) in updates {
        chain.set_parameter_normalized(pid, val);
    }
    for b in 0..4 {
        let exc_base = 11 + (b as u32) * 3;
        chain.set_parameter_normalized(
            exc_base,
            (oz_dto.exciter_band_drives_pct[b] * 0.01).clamp(0.0, 1.0),
        );
        chain.set_parameter_normalized(exc_base + 1, oz_dto.exciter_band_mixes[b].clamp(0.0, 1.0));
        chain.set_parameter_normalized(
            exc_base + 2,
            ((oz_dto.exciter_band_modes[b].min(6) as f32) / 6.0).clamp(0.0, 1.0),
        );
        chain.set_parameter_normalized(
            25 + (b as u32),
            ((oz_dto.imager_band_widths_pct[b] + 100.0) / 200.0).clamp(0.0, 1.0),
        );
        chain.set_parameter_normalized(
            29 + (b as u32),
            (oz_dto.imager_band_stereoize_pct[b] * 0.01).clamp(0.0, 1.0),
        );
        chain.set_parameter_normalized(
            45 + (b as u32),
            ((oz_dto.impact_band_amounts[b] + 100.0) / 200.0).clamp(0.0, 1.0),
        );
    }
}

pub fn build_ozone12_chain_boxed(
    sr: f64,
    blk: usize,
    oz_dto: &Ozone12MasterConfigDto,
) -> Box<Ozone12MasterChain> {
    let mut chain = Box::new(Ozone12MasterChain::new(sr));
    if oz_dto.unlimiter_enabled {
        if let Ok(delimit_proc) = ai_uvr5_stem_engine::OzoneDelimitOnnxProcessor::from_default_assets()
        {
            chain.set_unlimiter_neural_backend(Box::new(delimit_proc));
        }
    }
    chain.prepare(sr, blk);
    apply_ozone12_dto_to_chain(&mut chain, oz_dto);
    chain
}

pub fn build_ozone12_node(
    sr: f64,
    blk: usize,
    oz_dto: &Ozone12MasterConfigDto,
) -> Box<dyn UnifiedDspNode> {
    build_ozone12_chain_boxed(sr, blk, oz_dto)
}

pub fn build_pitcher_node(
    sr: f64,
    blk: usize,
    dto: &PitcherStateDto,
) -> Box<dyn UnifiedDspNode> {
    let mut node = Box::new(AutoPitchCorrectorNode::new(sr));
    node.prepare(sr, blk);
    node.root_key = dto.root_key.min(11);
    node.scale = MusicalScale::from_index(dto.scale_index);
    node.retune_speed_ms = dto.retune_speed_ms.clamp(0.0, 400.0);
    node.humanize = dto.humanize.clamp(0.0, 1.0);
    node.formant_preserve = dto.formant_preserve;
    node.formant_shift_semitones = dto.formant_shift_semitones.clamp(-12.0, 12.0);
    node.mix = dto.mix.clamp(0.0, 1.0);
    node.bypass = dto.bypass;
    node
}

pub fn build_external_vst3_node(
    sr: f64,
    blk: usize,
    ext_dto: &ExternalPluginStateDto,
) -> Result<Box<dyn UnifiedDspNode>, String> {
    let cid_opt = if ext_dto.plugin_id.is_empty() {
        None
    } else {
        Some(ext_dto.plugin_id.as_str())
    };
    let mut node = Vst3HostedPluginNode::load(Path::new(&ext_dto.binary_path), cid_opt, sr, blk)?;
    if !ext_dto.state_hex.is_empty() {
        let bytes = decode_hex_bytes(&ext_dto.state_hex)?;
        if !bytes.is_empty() {
            let _ = node.load_state(&bytes);
        }
    }
    for &(pid, norm_val) in &ext_dto.param_overrides {
        node.set_parameter_normalized(pid, norm_val as f32);
    }
    Ok(Box::new(node))
}

pub fn build_external_clap_node(
    sr: f64,
    blk: usize,
    ext_dto: &ExternalPluginStateDto,
) -> Result<Box<dyn UnifiedDspNode>, String> {
    let pid_opt = if ext_dto.plugin_id.is_empty() {
        None
    } else {
        Some(ext_dto.plugin_id.as_str())
    };
    let mut node = ClapHostedPluginNode::load(Path::new(&ext_dto.binary_path), pid_opt, sr, blk)?;
    if !ext_dto.state_hex.is_empty() {
        let bytes = decode_hex_bytes(&ext_dto.state_hex)?;
        if !bytes.is_empty() {
            let _ = node.load_state(&bytes);
        }
    }
    for &(pid, norm_val) in &ext_dto.param_overrides {
        node.set_parameter_normalized(pid, norm_val as f32);
    }
    Ok(Box::new(node))
}

fn same_plugin_variant(a: &PluginConfigDto, b: &PluginConfigDto) -> bool {
    matches!(
        (a, b),
        (PluginConfigDto::ProQ4(_), PluginConfigDto::ProQ4(_))
            | (PluginConfigDto::Soothe3(_), PluginConfigDto::Soothe3(_))
            | (
                PluginConfigDto::Ozone12Master(_),
                PluginConfigDto::Ozone12Master(_)
            )
            | (
                PluginConfigDto::RealTimeBsrnnStem(_),
                PluginConfigDto::RealTimeBsrnnStem(_)
            )
            | (
                PluginConfigDto::AutoPitchCorrector(_),
                PluginConfigDto::AutoPitchCorrector(_)
            )
            | (PluginConfigDto::Empty, PluginConfigDto::Empty)
    )
}

/// Computes the exact delta between `old_state` and `new_state` and dispatches lock-free
/// `EngineCommand` + `RtGraphMutation` updates to the live RT audio thread (`0` Mutex, `0` RT alloc/drop).
pub fn sync_project_state_diff_to_rt(
    old_state: &ToneForgeProjectState,
    new_state: &ToneForgeProjectState,
    app_state: &DawAppState,
    collision_bus: Option<&Arc<ProQ4CollisionBus>>,
) -> Result<(), String> {
    let sr = new_state.sample_rate.max(8_000.0);
    let blk = new_state.max_block_size.clamp(16, MAX_BLOCK_SIZE);

    if (old_state.bpm - new_state.bpm).abs() > 1e-6 {
        let _ = app_state.push_command(EngineCommand::SetBpm(new_state.bpm));
    }
    if old_state.is_playing != new_state.is_playing {
        let _ = app_state.push_command(EngineCommand::SetTransportPlaying(new_state.is_playing));
    }
    if old_state.ppq_resolution != new_state.ppq_resolution {
        let _ = app_state.push_command(EngineCommand::SetPpq(new_state.ppq_resolution));
    }

    if app_state.has_rt_mutations() {
        // 1. Piano Roll Patterns
        for old_pat in &old_state.piano_roll_patterns {
            if !new_state
                .piano_roll_patterns
                .iter()
                .any(|p| p.pattern_id == old_pat.pattern_id)
            {
                let empty = daw_core_graph::PianoRollPattern::new(
                    old_pat.pattern_id,
                    old_pat.name.clone(),
                    old_pat.target_channel_id,
                    old_pat.length_ticks,
                );
                let _ = app_state.push_rt_mutation(RtGraphMutation::UpsertPattern {
                    pattern: Box::new(empty),
                });
            }
        }
        for new_pat in &new_state.piano_roll_patterns {
            let changed = old_state
                .piano_roll_patterns
                .iter()
                .find(|p| p.pattern_id == new_pat.pattern_id)
                .map(|old_pat| old_pat != new_pat)
                .unwrap_or(true);
            if changed {
                let _ = app_state.push_rt_mutation(RtGraphMutation::UpsertPattern {
                    pattern: Box::new(new_pat.to_pattern()),
                });
            }
        }

        // 2. Playlist Clips
        for old_clip in &old_state.playlist_clips {
            if !new_state
                .playlist_clips
                .iter()
                .any(|c| c.clip_id == old_clip.clip_id)
            {
                let _ = app_state.push_rt_mutation(RtGraphMutation::RemovePlaylistClip {
                    clip_id: old_clip.clip_id,
                });
            }
        }
        for new_clip in &new_state.playlist_clips {
            let changed = old_state
                .playlist_clips
                .iter()
                .find(|c| c.clip_id == new_clip.clip_id)
                .map(|old_clip| old_clip != new_clip)
                .unwrap_or(true);
            if changed {
                let _ = app_state.push_rt_mutation(RtGraphMutation::UpsertPlaylistClip {
                    clip: new_clip.to_playlist_clip(),
                });
            }
        }

        // 3. Automation Clips & Controller Links
        for old_auto in &old_state.automation_clips {
            if !new_state
                .automation_clips
                .iter()
                .any(|a| a.controller_id == old_auto.controller_id)
            {
                let empty_art = daw_core_graph::AutomationClipArticulator::new(
                    old_auto.controller_id,
                    daw_core_graph::TensionSpline::default(),
                );
                let _ = app_state.push_rt_mutation(RtGraphMutation::UpsertAutomationArticulator {
                    articulator: Box::new(empty_art),
                });
            }
        }
        for new_auto in &new_state.automation_clips {
            let changed = old_state
                .automation_clips
                .iter()
                .find(|a| a.controller_id == new_auto.controller_id)
                .map(|old_a| old_a != new_auto)
                .unwrap_or(true);
            if changed {
                let _ = app_state.push_rt_mutation(RtGraphMutation::UpsertAutomationArticulator {
                    articulator: Box::new(new_auto.to_articulator()),
                });
            }
        }
        if old_state.controller_links != new_state.controller_links {
            let _ = app_state.push_rt_mutation(RtGraphMutation::ClearControllerLinks);
            for new_link in &new_state.controller_links {
                let _ = app_state.push_rt_mutation(RtGraphMutation::UpsertControllerLink {
                    link: (*new_link).into(),
                });
            }
        }
    }

    // 4. 502 Mixer Tracks: Gains, Hardware Input/Arming, Sends, and Plugin Slots
    let track_count = old_state.mixer_tracks.len().min(new_state.mixer_tracks.len());
    for idx in 0..track_count {
        let old_trk = &old_state.mixer_tracks[idx];
        let new_trk = &new_state.mixer_tracks[idx];
        if old_trk == new_trk {
            continue;
        }
        let t_u16 = idx as u16;

        if (old_trk.gain - new_trk.gain).abs() > 1e-6 {
            let _ = app_state.push_command(EngineCommand::SetTrackGain {
                track_index: t_u16,
                gain_linear: new_trk.gain,
            });
        }
        if (old_trk.pan - new_trk.pan).abs() > 1e-6 {
            let _ = app_state.push_command(EngineCommand::SetTrackPan {
                track_index: t_u16,
                pan: new_trk.pan,
            });
        }
        if old_trk.muted != new_trk.muted {
            let _ = app_state.push_command(EngineCommand::SetTrackMute {
                track_index: t_u16,
                muted: new_trk.muted,
            });
        }
        if old_trk.soloed != new_trk.soloed {
            let _ = app_state.push_command(EngineCommand::SetTrackSolo {
                track_index: t_u16,
                soloed: new_trk.soloed,
            });
        }
        if app_state.has_rt_mutations() && old_trk.channel_instrument != new_trk.channel_instrument {
            let ch_id = 200 + idx as u32;
            match &new_trk.channel_instrument {
                Some(cfg) => {
                    let sr = new_state.sample_rate.max(8_000.0);
                    let node = build_plugin_node_from_dto(idx, sr, MAX_BLOCK_SIZE, cfg, None)?;
                    let mut ch = daw_core_graph::ChannelGenerator::new(ch_id, idx, node);
                    ch.prepare(sr, MAX_BLOCK_SIZE);
                    app_state.push_rt_mutation(RtGraphMutation::UpsertChannel {
                        channel: Box::new(Some(ch)),
                    })?;
                }
                None => {
                    app_state.push_rt_mutation(RtGraphMutation::RemoveChannel {
                        channel_id: ch_id,
                        recycle_channel: Box::new(None),
                    })?;
                }
            }
        }

        if app_state.has_rt_mutations()
            && (old_trk.input_source_bus != new_trk.input_source_bus
                || old_trk.record_armed != new_trk.record_armed
                || old_trk.hw_input_latency_samples != new_trk.hw_input_latency_samples)
        {
            let _ = app_state.push_rt_mutation(RtGraphMutation::SetTrackInputAndRecordArm {
                track_index: t_u16,
                input_source_bus: new_trk.input_source_bus,
                record_armed: new_trk.record_armed,
                hw_input_latency_samples: new_trk.hw_input_latency_samples as u32,
            });
        }

        if app_state.has_rt_mutations() && old_trk.sends != new_trk.sends {
            for old_s in &old_trk.sends {
                if !new_trk.sends.iter().any(|ns| {
                    ns.dest_track_index == old_s.dest_track_index
                        && ns.is_sidechain_only == old_s.is_sidechain_only
                }) {
                    let _ = app_state.push_rt_mutation(RtGraphMutation::RemoveSendRoute {
                        src_track: t_u16,
                        dest_track: old_s.dest_track_index as u16,
                        is_sidechain_only: old_s.is_sidechain_only,
                        recycle_carrier: Box::new(None),
                    });
                }
            }
            for new_s in &new_trk.sends {
                let changed = old_trk
                    .sends
                    .iter()
                    .find(|os| {
                        os.dest_track_index == new_s.dest_track_index
                            && os.is_sidechain_only == new_s.is_sidechain_only
                    })
                    .map(|os| os != new_s)
                    .unwrap_or(true);
                if changed {
                    let route = MixerSendRoute::new(
                        new_s.dest_track_index,
                        new_s.gain,
                        new_s.is_sidechain_only,
                    );
                    let _ = app_state.push_rt_mutation(RtGraphMutation::AddOrUpdateSendRoute {
                        src_track: t_u16,
                        dest_track: new_s.dest_track_index as u16,
                        gain: new_s.gain,
                        is_sidechain_only: new_s.is_sidechain_only,
                        route_carrier: Box::new(Some(route)),
                    });
                }
            }
        }

        // Plugin Slots diff
        if old_trk.slots != new_trk.slots {
            if app_state.has_rt_mutations() && old_trk.slots.len() > new_trk.slots.len() {
                for s_idx in (new_trk.slots.len()..old_trk.slots.len()).rev() {
                    let _ = app_state.push_rt_mutation(RtGraphMutation::RemovePluginSlot {
                        track_index: t_u16,
                        slot_index: s_idx as u8,
                        recycle_slot: Box::new(None),
                    });
                }
            }

            for (s_idx, new_slot) in new_trk.slots.iter().enumerate() {
                let old_slot_opt = old_trk.slots.get(s_idx);
                if old_slot_opt == Some(new_slot) {
                    continue;
                }
                let s_u8 = s_idx as u8;

                let can_update_in_place = old_slot_opt
                    .map(|os| same_plugin_variant(&os.plugin, &new_slot.plugin))
                    .unwrap_or(false);

                if can_update_in_place {
                    let os = old_slot_opt.unwrap();
                    if os.enabled != new_slot.enabled {
                        let _ = app_state.push_command(EngineCommand::SetSlotEnabled {
                            track_index: t_u16,
                            slot_index: s_u8,
                            enabled: new_slot.enabled,
                        });
                    }
                    if (os.wet_mix - new_slot.wet_mix).abs() > 1e-6 {
                        let _ = app_state.push_command(EngineCommand::SetSlotWetMix {
                            track_index: t_u16,
                            slot_index: s_u8,
                            wet_mix: new_slot.wet_mix,
                        });
                    }
                    sync_in_place_plugin_params(
                        t_u16,
                        s_u8,
                        &os.plugin,
                        &new_slot.plugin,
                        app_state,
                        collision_bus,
                    )?;
                } else if app_state.has_rt_mutations() {
                    let node = build_plugin_node_from_dto(
                        idx,
                        sr,
                        blk,
                        &new_slot.plugin,
                        collision_bus,
                    )?;
                    let mut rack_slot = PluginRackSlot::new(node, new_slot.wet_mix);
                    rack_slot.enabled = new_slot.enabled;
                    rack_slot.prepare(sr, blk);
                    let _ = app_state.push_rt_mutation(RtGraphMutation::SetOrReplacePluginSlot {
                        track_index: t_u16,
                        slot_index: s_u8,
                        slot: Box::new(Some(rack_slot)),
                    });
                }
            }
        }
    }

    Ok(())
}

fn sync_in_place_plugin_params(
    track_idx: u16,
    slot_idx: u8,
    old_p: &PluginConfigDto,
    new_p: &PluginConfigDto,
    app_state: &DawAppState,
    collision_bus: Option<&Arc<ProQ4CollisionBus>>,
) -> Result<(), String> {
    match (old_p, new_p) {
        (PluginConfigDto::ProQ4(old_pq), PluginConfigDto::ProQ4(new_pq)) => {
            let bus = collision_bus
                .cloned()
                .unwrap_or_else(|| ProQ4CollisionBus::global().clone());
            bus.set_reference_track(track_idx as u32, new_pq.collision_reference_track);
            let old_ps = old_pq.to_parameter_set();
            let new_ps = new_pq.to_parameter_set();
            for pid in 0..PROQ4_TOTAL_PARAMS {
                let ov = old_ps.get_normalized(pid as u32);
                let nv = new_ps.get_normalized(pid as u32);
                if (ov - nv).abs() > 1e-7 {
                    app_state.push_command(EngineCommand::SetPluginParamNormalized {
                        track_index: track_idx,
                        slot_index: slot_idx,
                        param_id: pid as u32,
                        norm_val: nv,
                        target_sample_pos: None,
                    })?;
                }
            }
            if old_ps.global.processing_mode != new_ps.global.processing_mode
                || old_ps.global.processing_resolution != new_ps.global.processing_resolution
            {
                let _ = app_state.push_command(EngineCommand::RefreshApdc);
            }
        }
        (PluginConfigDto::Soothe3(old_s3), PluginConfigDto::Soothe3(new_s3)) => {
            let old_ps = old_s3.to_parameter_set();
            let new_ps = new_s3.to_parameter_set();
            for pid in 0..SOOTHE3_TOTAL_PARAMS {
                let ov = old_ps.get_normalized(pid as u32);
                let nv = new_ps.get_normalized(pid as u32);
                if (ov - nv).abs() > 1e-7 {
                    app_state.push_command(EngineCommand::SetPluginParamNormalized {
                        track_index: track_idx,
                        slot_index: slot_idx,
                        param_id: pid as u32,
                        norm_val: nv,
                        target_sample_pos: None,
                    })?;
                }
            }
        }
        (PluginConfigDto::Ozone12Master(old_oz), PluginConfigDto::Ozone12Master(new_oz)) => {
            cmd_update_ozone12_master(app_state, track_idx, slot_idx, new_oz)?;
            if old_oz.codec_preview_enabled != new_oz.codec_preview_enabled
                || old_oz.maximizer_bypass != new_oz.maximizer_bypass
                || old_oz.unlimiter_enabled != new_oz.unlimiter_enabled
            {
                let _ = app_state.push_command(EngineCommand::RefreshApdc);
            }
        }
        (
            PluginConfigDto::RealTimeBsrnnStem(_),
            PluginConfigDto::RealTimeBsrnnStem(new_bsrnn),
        ) => {
            cmd_update_bsrnn_stem_state(
                app_state,
                track_idx,
                slot_idx,
                new_bsrnn.stem_gains_db,
                new_bsrnn.stem_solo,
                new_bsrnn.stem_mute,
            )?;
        }
        (
            PluginConfigDto::AutoPitchCorrector(_),
            PluginConfigDto::AutoPitchCorrector(new_pt),
        ) => {
            let params: [(u32, f32); 8] = [
                (0, (new_pt.root_key.min(11) as f32) / 11.0),
                (1, (new_pt.scale_index.min(7) as f32) / 7.0),
                (2, (new_pt.retune_speed_ms / 400.0).clamp(0.0, 1.0)),
                (3, new_pt.humanize.clamp(0.0, 1.0)),
                (4, if new_pt.formant_preserve { 1.0 } else { 0.0 }),
                (
                    5,
                    ((new_pt.formant_shift_semitones + 12.0) / 24.0).clamp(0.0, 1.0),
                ),
                (6, new_pt.mix.clamp(0.0, 1.0)),
                (7, if new_pt.bypass { 1.0 } else { 0.0 }),
            ];
            for (pid, nv) in params {
                app_state.push_command(EngineCommand::SetPluginParamNormalized {
                    track_index: track_idx,
                    slot_index: slot_idx,
                    param_id: pid,
                    norm_val: nv,
                    target_sample_pos: None,
                })?;
            }
            let _ = app_state.push_command(EngineCommand::RefreshApdc);
        }
        _ => {}
    }
    Ok(())
}
