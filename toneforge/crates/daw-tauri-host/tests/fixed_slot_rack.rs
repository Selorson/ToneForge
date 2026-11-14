//! FL Studio-style fixed 10-position FX insert rack: removing a slot never shifts later slots,
//! inserting past the end pads with Empty slots, trailing Empty slots are trimmed, undo restores
//! the plug-in at the same index — in BOTH the live engine (via lock-free `RtGraphMutation`s) and
//! `ToneForgeProjectState`.

use daw_tauri_host::{
    AudioStreamController, Fl26SlotPresetDto, PatcherGraphStateDto, PluginConfigDto,
    PluginSlotStateDto, ProjectUndoHistory, ToneForgeProjectState,
    ipc_commands::PatcherCanvasUpdateDto,
};

const SR: f64 = 48_000.0;
const TRACK: u16 = 1;

fn slot(i: u8, plugin: PluginConfigDto) -> PluginSlotStateDto {
    PluginSlotStateDto {
        slot_index: i,
        enabled: true,
        wet_mix: 1.0,
        plugin,
    }
}

fn compressor() -> PluginConfigDto {
    PluginConfigDto::NativeSparseCompressor {
        threshold_linear: 0.5,
        ratio: 4.0,
        makeup_gain: 1.0,
        lookahead_samples: 64,
    }
}

/// Track 1 = [Maximus, Patcher, Compressor] with overlays on slots 0, 1 and 2.
fn three_slot_project() -> ToneForgeProjectState {
    let mut ps = ToneForgeProjectState::new_default(SR, 128.0);
    ps.mixer_tracks[TRACK as usize].slots = vec![
        slot(0, PluginConfigDto::NativeMaximusMultiband),
        slot(1, PluginConfigDto::PatcherModular(PatcherGraphStateDto::default())),
        slot(2, compressor()),
    ];
    ps.fl26_fx_presets = vec![
        Fl26SlotPresetDto {
            track_idx: TRACK,
            slot_idx: 0,
            preset_query: "Maximus: Clear Master.fst".to_string(),
        },
        Fl26SlotPresetDto {
            track_idx: TRACK,
            slot_idx: 2,
            preset_query: "Fruity Limiter: Drum compression 1.fst".to_string(),
        },
    ];
    ps.patcher_canvases = vec![PatcherCanvasUpdateDto {
        track_idx: TRACK,
        slot_idx: 1,
        nodes: Vec::new(),
        edges: Vec::new(),
    }];
    ps
}

fn live_controller(ps: ToneForgeProjectState) -> (AudioStreamController, daw_tauri_host::RtGarbageCollector) {
    let (mut ctrl, gc) = AudioStreamController::new_with_rt_gc(SR, 256, 0);
    ctrl.apply_project_state(ps).expect("apply project state");
    (ctrl, gc)
}

fn engine_names(ctrl: &AudioStreamController, track: u16) -> Vec<&'static str> {
    ctrl.mixer_engine.tracks[track as usize]
        .slots
        .iter()
        .map(|s| s.plugin.name())
        .collect()
}

fn ps_slots(ctrl: &AudioStreamController, track: u16) -> &Vec<PluginSlotStateDto> {
    &ctrl.project_state.mixer_tracks[track as usize].slots
}

fn assert_indices_match_positions(slots: &[PluginSlotStateDto]) {
    for (i, s) in slots.iter().enumerate() {
        assert_eq!(s.slot_index as usize, i, "slot_index must equal its fixed position");
    }
}

#[test]
fn remove_first_slot_keeps_later_slots_at_their_indices() {
    let (mut ctrl, _gc) = live_controller(three_slot_project());
    let before = engine_names(&ctrl, TRACK);
    assert_eq!(before.len(), 3);

    let AudioStreamController { project_state, app_state, mixer_engine, .. } = &mut ctrl;
    let slots = daw_tauri_host::clear_plugin_slot(project_state, app_state, SR, TRACK, 0)
        .expect("remove slot 0");
    mixer_engine.drain_rt_mutations();

    // Engine: slot 0 is an empty pass-through, slots 1 and 2 untouched.
    let after = engine_names(&ctrl, TRACK);
    assert_eq!(after, vec!["Empty", before[1], before[2]]);
    assert_eq!(ctrl.mixer_engine.tracks[TRACK as usize].slots[0].plugin.latency_samples(), 0);

    // Project state mirrors it exactly.
    let ps = ps_slots(&ctrl, TRACK);
    assert_eq!(slots, *ps);
    assert_eq!(ps.len(), 3);
    assert_eq!(ps[0].plugin, PluginConfigDto::Empty);
    assert!(matches!(ps[1].plugin, PluginConfigDto::PatcherModular(_)));
    assert_eq!(ps[2].plugin, compressor());
    assert_indices_match_positions(ps);

    // Overlays: slot 0's preset dropped; slot 1 canvas and slot 2 preset keep their indices.
    let presets = &ctrl.project_state.fl26_fx_presets;
    assert_eq!(presets.len(), 1);
    assert_eq!((presets[0].track_idx, presets[0].slot_idx), (TRACK, 2));
    let canvases = &ctrl.project_state.patcher_canvases;
    assert_eq!(canvases.len(), 1);
    assert_eq!((canvases[0].track_idx, canvases[0].slot_idx), (TRACK, 1));

    // Re-applying overlays by slot index (offline render / project open) still hits the same
    // plug-ins: canvas -> slot 1, `.fst` -> slot 2, slot 0 stays Empty.
    let snapshot = ctrl.project_state.clone();
    let mut offline = AudioStreamController::new(SR, 256, 0);
    offline.apply_project_state(snapshot.clone()).expect("offline apply");
    for canvas in &snapshot.patcher_canvases {
        daw_tauri_host::cmd_update_patcher_graph(&mut offline.mixer_engine, canvas).expect("canvas");
    }
    let patcher_name = offline.mixer_engine.tracks[TRACK as usize].slots[1].plugin.name();
    for p in &snapshot.fl26_fx_presets {
        let AudioStreamController { mixer_engine, app_state, .. } = &mut offline;
        daw_tauri_host::apply_fl26_fx_preset_to_engine(
            mixer_engine,
            app_state,
            p.track_idx,
            p.slot_idx,
            &p.preset_query,
        )
        .expect("fl26 overlay");
    }
    let replay = engine_names(&offline, TRACK);
    assert_eq!(replay.len(), 3);
    assert_eq!(replay[0], "Empty");
    assert_eq!(replay[1], patcher_name);
    assert_eq!(replay[2], "Fruity Sparse Compressor/Limiter");
}

#[test]
fn insert_past_end_pads_with_empty_slots() {
    let mut ps = ToneForgeProjectState::new_default(SR, 128.0);
    ps.mixer_tracks[TRACK as usize].slots = vec![
        slot(0, PluginConfigDto::NativeMaximusMultiband),
        slot(1, PluginConfigDto::NativeGrossBeat),
    ];
    let (mut ctrl, _gc) = live_controller(ps);
    let before = engine_names(&ctrl, TRACK);

    let AudioStreamController { project_state, app_state, mixer_engine, collision_bus, .. } =
        &mut ctrl;
    let slots = daw_tauri_host::insert_native_plugin_slot(
        project_state,
        app_state,
        SR,
        TRACK,
        5,
        "NativeLuxeVerbFdn",
        Some(&*collision_bus),
    )
    .expect("insert at slot 5");
    mixer_engine.drain_rt_mutations();

    let names = engine_names(&ctrl, TRACK);
    assert_eq!(
        names,
        vec![before[0], before[1], "Empty", "Empty", "Empty", "LuxeVerb FDN Shimmer"]
    );
    let ps = ps_slots(&ctrl, TRACK);
    assert_eq!(slots, *ps);
    assert_eq!(ps.len(), 6);
    for i in 2..5 {
        assert_eq!(ps[i].plugin, PluginConfigDto::Empty, "slot {i} must be Empty");
    }
    assert_eq!(ps[5].plugin, PluginConfigDto::NativeLuxeVerbFdn);
    assert_indices_match_positions(ps);

    // Replacing an existing slot keeps the length and every other index.
    let AudioStreamController { project_state, app_state, mixer_engine, collision_bus, .. } =
        &mut ctrl;
    daw_tauri_host::insert_native_plugin_slot(
        project_state,
        app_state,
        SR,
        TRACK,
        3,
        "NativeEmphasisLimiter",
        Some(&*collision_bus),
    )
    .expect("insert at slot 3");
    mixer_engine.drain_rt_mutations();
    assert_eq!(engine_names(&ctrl, TRACK)[3], "Emphasis Mastering Limiter");
    assert_eq!(engine_names(&ctrl, TRACK).len(), 6);
    assert!(matches!(ps_slots(&ctrl, TRACK)[3].plugin, PluginConfigDto::NativeEmphasisLimiter { .. }));

    // Unknown kinds are rejected without touching anything.
    let AudioStreamController { project_state, app_state, collision_bus, .. } = &mut ctrl;
    assert!(daw_tauri_host::insert_native_plugin_slot(
        project_state,
        app_state,
        SR,
        TRACK,
        7,
        "NotAPlugin",
        Some(&*collision_bus),
    )
    .is_err());
    assert_eq!(ps_slots(&ctrl, TRACK).len(), 6);
}

#[test]
fn every_native_kind_builds_and_lands_at_its_slot() {
    let (mut ctrl, _gc) = live_controller(ToneForgeProjectState::new_default(SR, 128.0));
    for (i, kind) in daw_tauri_host::NATIVE_INSERT_PLUGIN_KINDS.iter().enumerate() {
        let AudioStreamController { project_state, app_state, mixer_engine, collision_bus, .. } =
            &mut ctrl;
        daw_tauri_host::insert_native_plugin_slot(
            project_state,
            app_state,
            SR,
            2,
            (9 - i) as u8,
            kind,
            Some(&*collision_bus),
        )
        .unwrap_or_else(|e| panic!("{kind}: {e}"));
        mixer_engine.drain_rt_mutations();
    }
    let names = engine_names(&ctrl, 2);
    assert_eq!(names.len(), 10);
    assert!(names.iter().all(|n| *n != "Empty"), "all 10 slots occupied: {names:?}");
    let ps = ps_slots(&ctrl, 2);
    assert_eq!(ps.len(), 10);
    assert!(ps.iter().all(|s| s.plugin != PluginConfigDto::Empty));
    assert_indices_match_positions(ps);
}

#[test]
fn trailing_empty_slots_are_trimmed() {
    let (mut ctrl, _gc) = live_controller(three_slot_project());

    // Removing the last slot shrinks the rack.
    {
        let AudioStreamController { project_state, app_state, mixer_engine, .. } = &mut ctrl;
        daw_tauri_host::clear_plugin_slot(project_state, app_state, SR, TRACK, 2).unwrap();
        mixer_engine.drain_rt_mutations();
    }
    assert_eq!(engine_names(&ctrl, TRACK).len(), 2);
    assert_eq!(ps_slots(&ctrl, TRACK).len(), 2);

    // Removing slot 0 leaves [Empty, Patcher].
    {
        let AudioStreamController { project_state, app_state, mixer_engine, .. } = &mut ctrl;
        daw_tauri_host::clear_plugin_slot(project_state, app_state, SR, TRACK, 0).unwrap();
        mixer_engine.drain_rt_mutations();
    }
    assert_eq!(engine_names(&ctrl, TRACK)[0], "Empty");
    assert_eq!(engine_names(&ctrl, TRACK).len(), 2);
    assert_eq!(ps_slots(&ctrl, TRACK).len(), 2);

    // Removing slot 1 makes every slot empty -> the whole rack is trimmed away.
    {
        let AudioStreamController { project_state, app_state, mixer_engine, .. } = &mut ctrl;
        let slots = daw_tauri_host::clear_plugin_slot(project_state, app_state, SR, TRACK, 1).unwrap();
        mixer_engine.drain_rt_mutations();
        assert!(slots.is_empty());
    }
    assert!(engine_names(&ctrl, TRACK).is_empty());
    assert!(ps_slots(&ctrl, TRACK).is_empty());

    // Removing a slot past the end is a harmless no-op.
    let AudioStreamController { project_state, app_state, mixer_engine, .. } = &mut ctrl;
    daw_tauri_host::clear_plugin_slot(project_state, app_state, SR, TRACK, 7).unwrap();
    mixer_engine.drain_rt_mutations();
    assert!(engine_names(&ctrl, TRACK).is_empty());
}

#[test]
fn undo_after_remove_restores_plugin_at_same_index() {
    let (mut ctrl, _gc) = live_controller(three_slot_project());
    let original_ps = ctrl.project_state.clone();
    let original_names = engine_names(&ctrl, TRACK);
    let mut history = ProjectUndoHistory::default();

    history.record_before_action("Remove Slot 1", &ctrl.project_state);
    {
        let AudioStreamController { project_state, app_state, mixer_engine, .. } = &mut ctrl;
        daw_tauri_host::clear_plugin_slot(project_state, app_state, SR, TRACK, 0).unwrap();
        mixer_engine.drain_rt_mutations();
    }
    assert_eq!(engine_names(&ctrl, TRACK)[0], "Empty");

    let tr = history.undo(&mut ctrl.project_state).expect("undo available");
    daw_tauri_host::sync_project_state_diff_to_rt(
        &tr.previous_state,
        &tr.restored_state,
        &ctrl.app_state,
        Some(&ctrl.collision_bus),
    )
    .expect("diff sync");
    ctrl.mixer_engine.drain_rt_mutations();

    assert_eq!(engine_names(&ctrl, TRACK), original_names);
    assert_eq!(ctrl.project_state.mixer_tracks[TRACK as usize], original_ps.mixer_tracks[TRACK as usize]);
    assert_eq!(ctrl.project_state.fl26_fx_presets, original_ps.fl26_fx_presets);

    // Redo removes it again, in place.
    let tr = history.redo(&mut ctrl.project_state).expect("redo available");
    daw_tauri_host::sync_project_state_diff_to_rt(
        &tr.previous_state,
        &tr.restored_state,
        &ctrl.app_state,
        Some(&ctrl.collision_bus),
    )
    .expect("diff sync");
    ctrl.mixer_engine.drain_rt_mutations();
    assert_eq!(engine_names(&ctrl, TRACK), vec!["Empty", original_names[1], original_names[2]]);

    // Undo of an insert-past-end trims the padding again (diff removes from the end).
    history.record_before_action("Insert", &ctrl.project_state);
    let before_insert = engine_names(&ctrl, TRACK);
    {
        let AudioStreamController { project_state, app_state, mixer_engine, collision_bus, .. } =
            &mut ctrl;
        daw_tauri_host::insert_native_plugin_slot(
            project_state,
            app_state,
            SR,
            TRACK,
            6,
            "NativeGrossBeat",
            Some(&*collision_bus),
        )
        .unwrap();
        mixer_engine.drain_rt_mutations();
    }
    assert_eq!(engine_names(&ctrl, TRACK).len(), 7);
    let tr = history.undo(&mut ctrl.project_state).expect("undo insert");
    daw_tauri_host::sync_project_state_diff_to_rt(
        &tr.previous_state,
        &tr.restored_state,
        &ctrl.app_state,
        Some(&ctrl.collision_bus),
    )
    .unwrap();
    ctrl.mixer_engine.drain_rt_mutations();
    assert_eq!(engine_names(&ctrl, TRACK), before_insert);
    assert_eq!(ps_slots(&ctrl, TRACK).len(), 3);
}

#[test]
fn fl26_preset_past_end_lands_exactly_and_is_mirrored() {
    let mut ps = ToneForgeProjectState::new_default(SR, 128.0);
    ps.mixer_tracks[TRACK as usize].slots = vec![
        slot(0, PluginConfigDto::NativeMaximusMultiband),
        slot(1, PluginConfigDto::NativeGrossBeat),
    ];
    let (mut ctrl, _gc) = live_controller(ps);
    let AudioStreamController { project_state, app_state, mixer_engine, .. } = &mut ctrl;
    let dto = daw_tauri_host::apply_fl26_fx_preset_slot_rt(
        project_state,
        app_state,
        SR,
        TRACK,
        4,
        "Maximus: Clear Master.fst",
    )
    .expect("fl26 preset");
    mixer_engine.drain_rt_mutations();
    assert_eq!(dto.plugin_name, "Maximus");

    let names = engine_names(&ctrl, TRACK);
    assert_eq!(names.len(), 5);
    assert_eq!(&names[2..4], &["Empty", "Empty"]);
    assert_eq!(names[4], "Maximus 3-Band Dynamics");
    let ps = ps_slots(&ctrl, TRACK);
    assert_eq!(ps.len(), 5);
    assert_eq!(ps[4].plugin, PluginConfigDto::NativeMaximusMultiband);
    assert_indices_match_positions(ps);
    assert_eq!(
        ctrl.project_state.fl26_fx_presets,
        vec![Fl26SlotPresetDto {
            track_idx: TRACK,
            slot_idx: 4,
            preset_query: "Maximus: Clear Master.fst".to_string(),
        }]
    );

    // A legacy project whose overlay points past the end of the rack gets padded on load.
    let mut legacy = ToneForgeProjectState::new_default(SR, 128.0);
    legacy.fl26_fx_presets.push(Fl26SlotPresetDto {
        track_idx: 3,
        slot_idx: 2,
        preset_query: "Gross Beat: 1/2 Gate.fst".to_string(),
    });
    daw_tauri_host::pad_project_slots_for_overlays(&mut legacy);
    assert_eq!(legacy.mixer_tracks[3].slots.len(), 3);
    assert!(legacy.mixer_tracks[3].slots.iter().all(|s| s.plugin == PluginConfigDto::Empty));
}

#[test]
fn empty_slot_round_trips_through_project_json() {
    let mut ps = ToneForgeProjectState::new_default(SR, 128.0);
    ps.mixer_tracks[TRACK as usize].slots = vec![
        daw_tauri_host::empty_slot_dto(0),
        slot(1, PluginConfigDto::NativeLuxeVerbFdn),
    ];
    let json = ps.to_json_pretty().unwrap();
    assert!(json.contains("\"Empty\""));
    let back = ToneForgeProjectState::from_json_str(&json).unwrap();
    assert_eq!(back.mixer_tracks[TRACK as usize].slots, ps.mixer_tracks[TRACK as usize].slots);

    // And it builds a live engine with the plug-in still at slot 1.
    let (ctrl, _gc) = live_controller(back);
    assert_eq!(engine_names(&ctrl, TRACK), vec!["Empty", "LuxeVerb FDN Shimmer"]);
}
