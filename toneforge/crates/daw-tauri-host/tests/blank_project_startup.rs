//! Verifies ToneForge's 100% blank initial project ("trắng tinh"):
//! 1. Fresh launch has empty patterns/clips/automation/controller_links/assets, flat EQ, unity BSRNN
//!    bypass, stopped transport, and outputs exact silence even if the user immediately hits Play.
//! 2. Authoring a Pattern + PlaylistClip + Automation Link from the blank project produces audio
//!    and controller links, and undoing back to the blank snapshot removes links and returns to silence.

use daw_tauri_host::{
    cmd_seek_transport, cmd_set_transport, cmd_update_automation_spline_rt,
    cmd_upsert_piano_roll_pattern_rt, cmd_upsert_playlist_clip_rt, sync_project_state_diff_to_rt,
    AudioStreamController, AutomationClipStateDto, ControllerLinkStateDto, LfoModulatorDto,
    PianoRollNoteDto, PianoRollPatternDto, PlaylistClipDto, PluginConfigDto,
    TensionControlPointDto, ToneForgeProjectState,
};

#[test]
fn blank_startup_has_zero_demo_data_and_outputs_exact_silence_when_playing() {
    let (mut ctrl, _gc) = AudioStreamController::with_blank_project_rt_gc(48_000.0, 256)
        .expect("create blank project controller");
    ctrl.add_polyphonic_synth_channel(1, 1, 64)
        .expect("attach synth channel 1");

    assert!(!ctrl.project_state.is_playing);
    assert!(!ctrl.mixer_engine.is_playing);
    assert!(ctrl.project_state.piano_roll_patterns.is_empty());
    assert!(ctrl.project_state.playlist_clips.is_empty());
    assert!(ctrl.project_state.automation_clips.is_empty());
    assert!(ctrl.project_state.controller_links.is_empty());
    assert!(ctrl.project_state.audio_assets.is_empty());
    assert!(ctrl.mixer_engine.controller_bus.active_links().is_empty());

    // Track 1 Slot 0 Pro-Q 4 must be completely flat (all 24 bands unused, 0.0 dB gain).
    match &ctrl.project_state.mixer_tracks[1].slots[0].plugin {
        PluginConfigDto::ProQ4(eq) => {
            assert!(
                eq.bands.iter().all(|b| !b.used && b.gain_db.abs() < 1e-6),
                "Blank project Pro-Q 4 must start flat with all bands unused"
            );
        }
        other => panic!("Expected ProQ4 in Track 1 Slot 0, got {other:?}"),
    }

    // Master Slot 0 BSRNN must start at unity 0.0 dB across all 4 stems (idle-gated to 0 GPU hops).
    match &ctrl.project_state.mixer_tracks[0].slots[0].plugin {
        PluginConfigDto::RealTimeBsrnnStem(bsrnn) => {
            assert_eq!(bsrnn.stem_gains_db, [0.0; 4]);
            assert_eq!(bsrnn.stem_solo, [false; 4]);
            assert_eq!(bsrnn.stem_mute, [false; 4]);
        }
        other => panic!("Expected RealTimeBsrnnStem in Master Slot 0, got {other:?}"),
    }

    // Pressing Play immediately without creating anything or loading an audio file must output silence.
    cmd_set_transport(&ctrl.app_state, true, 128.0).expect("set transport playing");
    let mut master_out = vec![[0.0f32; 2]; 256];
    let mut peak = 0.0f32;
    for _ in 0..48 {
        ctrl.process_audio_block(&mut master_out);
        for f in &master_out {
            peak = peak.max(f[0].abs()).max(f[1].abs());
        }
    }
    assert!(
        peak < 1e-4,
        "Blank project must remain silent (<-80 dBFS) when Play is pressed before adding notes/audio, got peak={peak}"
    );
}

#[test]
fn blank_project_authoring_and_undo_restores_silence_and_clears_links() {
    let (mut ctrl, _gc) = AudioStreamController::with_blank_project_rt_gc(48_000.0, 256)
        .expect("create blank project controller");
    ctrl.add_polyphonic_synth_channel(1, 1, 64)
        .expect("attach synth channel 1");

    let blank_snapshot: ToneForgeProjectState = ctrl.project_state.clone();
    let mut edited = blank_snapshot.clone();

    let pat = PianoRollPatternDto {
        pattern_id: 1,
        name: "Pattern 1".to_string(),
        target_channel_id: 1,
        length_ticks: 384,
        notes: vec![PianoRollNoteDto {
            id: 1,
            start_tick: 0,
            duration_ticks: 96,
            key: 60,
            velocity: 0.9,
            pan: 0.0,
            fine_pitch_cents: 0.0,
            is_slide: false,
            color_group: 0,
        }],
    };
    cmd_upsert_piano_roll_pattern_rt(&mut edited, &ctrl.app_state, &pat)
        .expect("upsert pattern 1");

    let clip = PlaylistClipDto {
        clip_id: 1,
        track_row: 0,
        kind: "Pattern".to_string(),
        source_id: 1,
        start_tick: 0,
        duration_ticks: 384,
        clip_offset_ticks: 0,
        gain: 1.0,
        muted: false,
        ..Default::default()
    };
    cmd_upsert_playlist_clip_rt(&mut edited, &ctrl.app_state, &clip)
        .expect("upsert playlist clip 1");

    let auto_clip = AutomationClipStateDto {
        controller_id: 0,
        name: "Filter Sweep".to_string(),
        points: vec![
            TensionControlPointDto { x: 0.0, y: 0.2, tension: 0.0 },
            TensionControlPointDto { x: 384.0, y: 0.8, tension: 0.0 },
        ],
        lfo: LfoModulatorDto::default(),
        min_level: 0.0,
        max_level: 1.0,
    };
    cmd_update_automation_spline_rt(&mut edited, &ctrl.app_state, &auto_clip, 0.0)
        .expect("upsert automation spline");
    edited.controller_links.push(ControllerLinkStateDto {
        controller_id: 0,
        target_track: 1,
        slot_index: 0,
        param_id: 2,
        min_out_norm: 0.2,
        max_out_norm: 0.8,
        inverted: false,
    });
    sync_project_state_diff_to_rt(
        &blank_snapshot,
        &edited,
        &ctrl.app_state,
        Some(&ctrl.collision_bus),
    )
    .expect("sync edited state");

    cmd_set_transport(&ctrl.app_state, true, 128.0).expect("play edited project");
    let mut master_out = vec![[0.0f32; 2]; 256];
    let mut active_peak = 0.0f32;
    for _ in 0..48 {
        ctrl.process_audio_block(&mut master_out);
        for f in &master_out {
            active_peak = active_peak.max(f[0].abs()).max(f[1].abs());
        }
    }
    assert!(
        active_peak > 1e-3,
        "Authored pattern + playlist clip must produce audible signal, got peak={active_peak}"
    );
    assert_eq!(ctrl.mixer_engine.controller_bus.active_links().len(), 1);

    // Undo back to the blank snapshot and seek to 0: controller links & pattern notes must clear cleanly.
    sync_project_state_diff_to_rt(
        &edited,
        &blank_snapshot,
        &ctrl.app_state,
        Some(&ctrl.collision_bus),
    )
    .expect("undo back to blank snapshot");
    cmd_seek_transport(&ctrl.app_state, 0, 0).expect("seek to 0");
    cmd_set_transport(&ctrl.app_state, true, 128.0).expect("restart playback");

    // Drain tail from previous note, then verify subsequent blocks are silent and links cleared.
    for _ in 0..80 {
        ctrl.process_audio_block(&mut master_out);
    }
    assert!(ctrl.mixer_engine.controller_bus.active_links().is_empty());

    let mut post_undo_peak = 0.0f32;
    for _ in 0..24 {
        ctrl.process_audio_block(&mut master_out);
        for f in &master_out {
            post_undo_peak = post_undo_peak.max(f[0].abs()).max(f[1].abs());
        }
    }
    assert!(
        post_undo_peak < 1e-4,
        "After undoing back to blank state and draining synth release tail, output must be silent, got {post_undo_peak}"
    );
}
