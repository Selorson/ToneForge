//! FL Studio-style fixed 10-position FX insert rack (`slot_rack.rs`).
//!
//! Mixer insert slots are **fixed positions** `0..MAX_INSERT_SLOTS`:
//! - Removing a plug-in turns its slot into [`PluginConfigDto::Empty`] (engine: [`EmptySlotNode`])
//!   **in place** — later slots never shift, so everything keyed by `(track_idx, slot_idx)`
//!   (FL26 `.fst` overlays, Patcher canvases, controller links, automation, editor windows, the
//!   Pro-Q 4 / Soothe3 / Ozone 12 editor tabs) keeps addressing the same plug-in.
//!   Trailing empty slots are trimmed from the end (removing from the end never shifts anything).
//! - Setting a plug-in at `slot_idx > len` first pads `len..slot_idx` with empty slots so the
//!   plug-in lands exactly at `slot_idx` in both the engine and `ToneForgeProjectState`.
//!
//! Every helper keeps the engine (via lock-free [`RtGraphMutation`]s built on the control thread)
//! and the project state in lockstep. Nothing here allocates on the real-time thread.

use crate::ipc_commands::{DawAppState, PatcherCanvasUpdateDto};
use crate::preset_library::Fl26FxPresetDto;
use crate::project_state::{
    Fl26SlotPresetDto, MixerTrackStateDto, Ozone12MasterConfigDto, PatcherGraphStateDto,
    PitcherStateDto, PluginConfigDto, PluginSlotStateDto, ProQ4StateDto, Soothe3StateDto,
    ToneForgeProjectState,
};
use daw_core_graph::{
    EmptySlotNode, MAX_BLOCK_SIZE, MAX_INSERT_SLOTS, MAX_MIXER_TRACKS, PluginRackSlot,
    RtGraphMutation,
};
use dsp_proq4_eq::ProQ4CollisionBus;
use std::sync::Arc;

/// Native plug-in kinds accepted by [`insert_native_plugin_slot`] (`PluginConfigDto` variant names).
pub const NATIVE_INSERT_PLUGIN_KINDS: [&str; 10] = [
    "ProQ4",
    "Soothe3",
    "Ozone12Master",
    "AutoPitchCorrector",
    "PatcherModular",
    "NativeMaximusMultiband",
    "NativeEmphasisLimiter",
    "NativeSparseCompressor",
    "NativeGrossBeat",
    "NativeLuxeVerbFdn",
];

/// Project-state description of an empty insert slot at `slot_index`.
pub fn empty_slot_dto(slot_index: u8) -> PluginSlotStateDto {
    PluginSlotStateDto {
        slot_index,
        enabled: true,
        wet_mix: 1.0,
        plugin: PluginConfigDto::Empty,
    }
}

#[inline]
pub fn is_empty_slot(slot: &PluginSlotStateDto) -> bool {
    matches!(slot.plugin, PluginConfigDto::Empty)
}

/// Builds (on the control thread) a prepared engine slot wrapping an [`EmptySlotNode`].
pub fn build_empty_rack_slot(sample_rate: f64) -> PluginRackSlot {
    let sr = sample_rate.max(8_000.0);
    let mut slot = PluginRackSlot::new(Box::new(EmptySlotNode::new()), 1.0);
    slot.prepare(sr, MAX_BLOCK_SIZE);
    slot
}

fn check_indices(track_idx: u16, slot_idx: u8) -> Result<(), String> {
    if (track_idx as usize) >= MAX_MIXER_TRACKS {
        return Err(format!("Mixer track index {track_idx} out of range (0..502)"));
    }
    if (slot_idx as usize) >= MAX_INSERT_SLOTS {
        return Err(format!("Insert slot index {slot_idx} out of range (0..10)"));
    }
    Ok(())
}

fn track_mut(
    project_state: &mut ToneForgeProjectState,
    track_idx: u16,
) -> Result<&mut MixerTrackStateDto, String> {
    project_state
        .mixer_tracks
        .get_mut(track_idx as usize)
        .ok_or_else(|| format!("Mixer track {track_idx} out of range"))
}

fn renumber(trk: &mut MixerTrackStateDto) {
    for (i, s) in trk.slots.iter_mut().enumerate() {
        s.slot_index = i as u8;
    }
}

/// Pads the track with empty slots for `len..slot_idx` (engine first, in ascending order, then the
/// project state) so that a following set at `slot_idx` lands exactly there. No-op when
/// `slot_idx <= len`.
pub fn pad_slots_up_to(
    project_state: &mut ToneForgeProjectState,
    state: &DawAppState,
    sample_rate: f64,
    track_idx: u16,
    slot_idx: u8,
) -> Result<(), String> {
    check_indices(track_idx, slot_idx)?;
    let trk = track_mut(project_state, track_idx)?;
    renumber(trk);
    while trk.slots.len() < slot_idx as usize {
        let i = trk.slots.len() as u8;
        if state.has_rt_mutations() {
            state.push_rt_mutation(RtGraphMutation::SetOrReplacePluginSlot {
                track_index: track_idx,
                slot_index: i,
                slot: Box::new(Some(build_empty_rack_slot(sample_rate))),
            })?;
        }
        trk.slots.push(empty_slot_dto(i));
    }
    Ok(())
}

/// Writes `plugin` into project-state slot `slot_idx` (appending when `slot_idx == len`).
/// Call [`pad_slots_up_to`] first so `slot_idx <= len`.
pub fn set_project_slot(
    project_state: &mut ToneForgeProjectState,
    track_idx: u16,
    slot_idx: u8,
    plugin: PluginConfigDto,
    enabled: bool,
    wet_mix: f32,
) -> Result<(), String> {
    check_indices(track_idx, slot_idx)?;
    let trk = track_mut(project_state, track_idx)?;
    let s = slot_idx as usize;
    let dto = PluginSlotStateDto {
        slot_index: slot_idx,
        enabled,
        wet_mix: wet_mix.clamp(0.0, 1.0),
        plugin,
    };
    if s < trk.slots.len() {
        trk.slots[s] = dto;
    } else if s == trk.slots.len() {
        trk.slots.push(dto);
    } else {
        return Err(format!(
            "Track {track_idx} slot {slot_idx} is past the end of the rack (pad first)"
        ));
    }
    Ok(())
}

/// Drops the per-slot overlays (FL26 `.fst` preset, Patcher canvas) of the plug-in that occupied
/// `(track_idx, slot_idx)`. Overlays of every other slot are untouched (no index shifting).
pub fn drop_slot_overlays(project_state: &mut ToneForgeProjectState, track_idx: u16, slot_idx: u8) {
    project_state
        .patcher_canvases
        .retain(|c| !(c.track_idx == track_idx && c.slot_idx == slot_idx));
    project_state
        .fl26_fx_presets
        .retain(|p| !(p.track_idx == track_idx && p.slot_idx == slot_idx));
}

/// Pads, hot-swaps the prepared `rack_slot` into `(track_idx, slot_idx)` on the RT thread and mirrors
/// `plugin` into the project state at exactly the same index.
pub fn place_plugin_slot(
    project_state: &mut ToneForgeProjectState,
    state: &DawAppState,
    sample_rate: f64,
    track_idx: u16,
    slot_idx: u8,
    rack_slot: PluginRackSlot,
    plugin: PluginConfigDto,
) -> Result<(), String> {
    check_indices(track_idx, slot_idx)?;
    let enabled = rack_slot.enabled;
    let wet_mix = rack_slot.wet_mix;
    pad_slots_up_to(project_state, state, sample_rate, track_idx, slot_idx)?;
    if state.has_rt_mutations() {
        state.push_rt_mutation(RtGraphMutation::SetOrReplacePluginSlot {
            track_index: track_idx,
            slot_index: slot_idx,
            slot: Box::new(Some(rack_slot)),
        })?;
    }
    set_project_slot(project_state, track_idx, slot_idx, plugin, enabled, wet_mix)
}

/// Removes the plug-in at `(track_idx, slot_idx)`: the slot becomes Empty **in place** (engine +
/// project state), its FL26 / Patcher overlays are dropped, then trailing Empty slots are trimmed
/// from the end. Returns the track's resulting slots. Removing an already-empty / missing slot is a
/// no-op apart from the trim.
pub fn clear_plugin_slot(
    project_state: &mut ToneForgeProjectState,
    state: &DawAppState,
    sample_rate: f64,
    track_idx: u16,
    slot_idx: u8,
) -> Result<Vec<PluginSlotStateDto>, String> {
    check_indices(track_idx, slot_idx)?;
    drop_slot_overlays(project_state, track_idx, slot_idx);
    let trk = track_mut(project_state, track_idx)?;
    renumber(trk);
    let s = slot_idx as usize;
    let mut replace_with_empty = false;
    if s < trk.slots.len() && !is_empty_slot(&trk.slots[s]) {
        trk.slots[s] = empty_slot_dto(slot_idx);
        replace_with_empty = true;
    }
    let old_len = trk.slots.len();
    let mut new_len = old_len;
    while new_len > 0 && is_empty_slot(&trk.slots[new_len - 1]) {
        new_len -= 1;
    }
    if state.has_rt_mutations() {
        // Trailing slots first, highest index first: removing from the end never shifts others.
        for i in (new_len..old_len).rev() {
            state.push_rt_mutation(RtGraphMutation::RemovePluginSlot {
                track_index: track_idx,
                slot_index: i as u8,
                recycle_slot: Box::new(None),
            })?;
        }
        if replace_with_empty && s < new_len {
            state.push_rt_mutation(RtGraphMutation::SetOrReplacePluginSlot {
                track_index: track_idx,
                slot_index: slot_idx,
                slot: Box::new(Some(build_empty_rack_slot(sample_rate))),
            })?;
        }
    }
    trk.slots.truncate(new_len);
    Ok(trk.slots.clone())
}

/// Default (FL-like) configuration for a native insert plug-in `kind`
/// (see [`NATIVE_INSERT_PLUGIN_KINDS`]; a few short aliases are accepted too).
pub fn default_native_plugin_dto(kind: &str) -> Result<PluginConfigDto, String> {
    let dto = match kind.trim() {
        "ProQ4" | "Pro-Q 4" => PluginConfigDto::ProQ4(ProQ4StateDto::default()),
        "Soothe3" => PluginConfigDto::Soothe3(Soothe3StateDto::default()),
        "Ozone12Master" | "Ozone12" => {
            PluginConfigDto::Ozone12Master(Ozone12MasterConfigDto::default())
        }
        "AutoPitchCorrector" | "Pitcher" => {
            PluginConfigDto::AutoPitchCorrector(PitcherStateDto::default())
        }
        "PatcherModular" | "Patcher" => {
            PluginConfigDto::PatcherModular(PatcherGraphStateDto::default())
        }
        "NativeMaximusMultiband" | "Maximus" => PluginConfigDto::NativeMaximusMultiband,
        "NativeEmphasisLimiter" | "Emphasis" => PluginConfigDto::NativeEmphasisLimiter {
            // -0.3 dBFS ceiling, unity drive.
            ceiling_linear: 10f32.powf(-0.3 / 20.0),
            drive_linear: 1.0,
        },
        "NativeSparseCompressor" | "FruityLimiter" => PluginConfigDto::NativeSparseCompressor {
            // -6 dB threshold, 4:1, unity make-up, 64-sample look-ahead (same as the FL26 node).
            threshold_linear: 0.5,
            ratio: 4.0,
            makeup_gain: 1.0,
            lookahead_samples: 64,
        },
        "NativeGrossBeat" | "GrossBeat" => PluginConfigDto::NativeGrossBeat,
        "NativeLuxeVerbFdn" | "LuxeVerb" => PluginConfigDto::NativeLuxeVerbFdn,
        other => {
            return Err(format!(
                "Unknown native plug-in kind '{other}' (expected one of {})",
                NATIVE_INSERT_PLUGIN_KINDS.join(", ")
            ));
        }
    };
    Ok(dto)
}

/// Inserts (or replaces) a native plug-in of `kind` at exactly `(track_idx, slot_idx)`, padding any
/// gap with Empty slots and dropping the previous occupant's FL26 / Patcher overlays. The caller is
/// responsible for closing GUI editors of the old occupant and recording undo. Returns the track's
/// resulting slots.
pub fn insert_native_plugin_slot(
    project_state: &mut ToneForgeProjectState,
    state: &DawAppState,
    sample_rate: f64,
    track_idx: u16,
    slot_idx: u8,
    kind: &str,
    collision_bus: Option<&Arc<ProQ4CollisionBus>>,
) -> Result<Vec<PluginSlotStateDto>, String> {
    check_indices(track_idx, slot_idx)?;
    let dto = default_native_plugin_dto(kind)?;
    let sr = sample_rate.max(8_000.0);
    let node = crate::undo_history::build_plugin_node_from_dto(
        track_idx as usize,
        sr,
        MAX_BLOCK_SIZE,
        &dto,
        collision_bus,
    )?;
    let mut rack_slot = PluginRackSlot::new(node, 1.0);
    rack_slot.prepare(sr, MAX_BLOCK_SIZE);
    place_plugin_slot(project_state, state, sr, track_idx, slot_idx, rack_slot, dto)?;
    drop_slot_overlays(project_state, track_idx, slot_idx);
    Ok(project_state.mixer_tracks[track_idx as usize].slots.clone())
}

/// Project-state mirror of the node an FL26 `.fst` preset instantiates
/// (same mapping as `build_fl26_fx_rack_slot`).
pub fn fl26_preset_plugin_dto(dto: &Fl26FxPresetDto) -> PluginConfigDto {
    let db_to_lin = |db: f32| 10f32.powf(db / 20.0);
    match dto.plugin_name.as_str() {
        "Maximus" => PluginConfigDto::NativeMaximusMultiband,
        "Emphasis" | "Fruity Soft Clipper" => PluginConfigDto::NativeEmphasisLimiter {
            ceiling_linear: db_to_lin(dto.ceiling_db),
            drive_linear: db_to_lin(dto.drive_db),
        },
        "Fruity Limiter" => PluginConfigDto::NativeSparseCompressor {
            threshold_linear: db_to_lin(dto.threshold_db).clamp(0.01, 1.0),
            ratio: 4.0,
            makeup_gain: db_to_lin(dto.drive_db).clamp(0.5, 4.0),
            lookahead_samples: 64,
        },
        "Gross Beat" => PluginConfigDto::NativeGrossBeat,
        "Fruity LuxeVerb" => PluginConfigDto::NativeLuxeVerbFdn,
        _ => PluginConfigDto::ProQ4(ProQ4StateDto::default()),
    }
}

/// Loads an FL26 `.fst` native FX preset into exactly `(track_idx, slot_idx)`: pads the rack, hot-swaps
/// the preset node on the RT thread, mirrors the matching native plug-in config into the project
/// state slot and records the `.fst` overlay (replacing any previous overlay of that slot).
pub fn apply_fl26_fx_preset_slot_rt(
    project_state: &mut ToneForgeProjectState,
    state: &DawAppState,
    sample_rate: f64,
    track_idx: u16,
    slot_idx: u8,
    preset_query: &str,
) -> Result<Fl26FxPresetDto, String> {
    check_indices(track_idx, slot_idx)?;
    pad_slots_up_to(project_state, state, sample_rate, track_idx, slot_idx)?;
    let dto = crate::preset_library::apply_fl26_fx_preset_rt(
        state,
        sample_rate,
        track_idx,
        slot_idx,
        preset_query,
    )?;
    mirror_fl26_preset_into_slot(project_state, track_idx, slot_idx, &dto)?;
    drop_slot_overlays(project_state, track_idx, slot_idx);
    project_state.fl26_fx_presets.push(Fl26SlotPresetDto {
        track_idx,
        slot_idx,
        preset_query: preset_query.to_string(),
    });
    Ok(dto)
}

/// Writes the native plug-in config of an applied FL26 preset into project-state slot
/// `(track_idx, slot_idx)` (padding with Empty slots in the project state only — use when the engine
/// slot was already placed, e.g. when re-applying overlays after opening a project).
pub fn mirror_fl26_preset_into_slot(
    project_state: &mut ToneForgeProjectState,
    track_idx: u16,
    slot_idx: u8,
    dto: &Fl26FxPresetDto,
) -> Result<(), String> {
    check_indices(track_idx, slot_idx)?;
    {
        let trk = track_mut(project_state, track_idx)?;
        while trk.slots.len() < slot_idx as usize {
            let i = trk.slots.len() as u8;
            trk.slots.push(empty_slot_dto(i));
        }
    }
    let wet = (dto.wet_mix_pct / 100.0).clamp(0.0, 1.0);
    set_project_slot(
        project_state,
        track_idx,
        slot_idx,
        fl26_preset_plugin_dto(dto),
        true,
        wet,
    )
}

/// Pads project-state racks so every FL26 overlay / Patcher canvas slot index exists (Empty
/// placeholders), e.g. for projects saved before fixed slot positions. Engine-free: run it on a
/// loaded project before syncing it to the engine so overlays land at their exact index.
pub fn pad_project_slots_for_overlays(project_state: &mut ToneForgeProjectState) {
    let targets: Vec<(u16, u8)> = project_state
        .fl26_fx_presets
        .iter()
        .map(|p| (p.track_idx, p.slot_idx))
        .chain(project_state.patcher_canvases.iter().map(|c| (c.track_idx, c.slot_idx)))
        .collect();
    for (t, s) in targets {
        if (s as usize) >= MAX_INSERT_SLOTS {
            continue;
        }
        if let Some(trk) = project_state.mixer_tracks.get_mut(t as usize) {
            while trk.slots.len() <= s as usize {
                let i = trk.slots.len() as u8;
                trk.slots.push(empty_slot_dto(i));
            }
        }
    }
}

/// Applies an interactive Patcher canvas to exactly `(dto.track_idx, dto.slot_idx)`: pads the rack,
/// hot-swaps the canvas patcher on the RT thread, mirrors a `PatcherModular` config into the project
/// state slot (if it held something else) and records the canvas overlay. Returns the patcher latency.
pub fn apply_patcher_canvas_slot_rt(
    project_state: &mut ToneForgeProjectState,
    state: &DawAppState,
    sample_rate: f64,
    dto: PatcherCanvasUpdateDto,
) -> Result<usize, String> {
    let (t, s) = (dto.track_idx, dto.slot_idx);
    check_indices(t, s)?;
    pad_slots_up_to(project_state, state, sample_rate, t, s)?;
    let latency = crate::ipc_commands::cmd_update_patcher_graph_rt(state, sample_rate, &dto)?;
    let needs_mirror = project_state.mixer_tracks[t as usize]
        .slots
        .get(s as usize)
        .map(|sl| !matches!(sl.plugin, PluginConfigDto::PatcherModular(_)))
        .unwrap_or(true);
    if needs_mirror {
        set_project_slot(
            project_state,
            t,
            s,
            PluginConfigDto::PatcherModular(PatcherGraphStateDto::default()),
            true,
            1.0,
        )?;
    }
    drop_slot_overlays(project_state, t, s);
    project_state.patcher_canvases.push(dto);
    Ok(latency)
}
