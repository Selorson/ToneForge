//! Gate 11 — Full-Stack Host E2E Parity (record -> comp -> playback -> save/load, playlist audio
//! clip stretch/pitch/slice/fade without varispeed, live-graph == offline-render mix).
//! Gate 12 — UI <-> Host IPC contract and high-density sequencing.
//!
//! These gates drive the same host building blocks `ToneForge_Studio.exe` uses (`audio_assets`,
//! `AudioStreamController`, `sync_project_state_diff_to_rt`, the RT mutation queues) end to end.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::f64::consts::TAU;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use daw_core_graph::{
    AdaptiveDllDriftResampler, ChannelGenerator, DeterministicGeneratorNode, DiskTrackRecorder,
    EngineCommand, MixerEngine, PianoRollNote, PianoRollPattern, RtGraphMutation, StereoFrame,
    read_wav_f32_stereo,
};
use daw_tauri_host::audio_assets::samples_per_tick;
use daw_tauri_host::project_state::{PlaylistClipDto, ToneForgeProjectState};
use daw_tauri_host::undo_history::{ProjectUndoHistory, sync_project_state_diff_to_rt};
use daw_tauri_host::{
    AudioAssetDto, AudioStreamController, KIND_NEWTONE_COMMIT, KIND_RECORDED_TAKE,
    asset_region_table, build_asset_channel, build_derived_clip_channel, load_asset_frames,
    write_f32_wav_file,
};
use dsp_ozone12_mastering::LufsTruePeakMeter;

// ----------------------------------------------------------------------------
// Thread-local allocation counter (RT zero-allocation verification)
// ----------------------------------------------------------------------------
struct CountingAlloc;
thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.with(|c| c.get()) {
            ALLOCS.with(|a| a.set(a.get() + 1));
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if COUNTING.with(|c| c.get()) {
            ALLOCS.with(|a| a.set(a.get() + 1));
        }
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn count_allocs<R>(f: impl FnOnce() -> R) -> (R, usize) {
    ALLOCS.with(|a| a.set(0));
    COUNTING.with(|c| c.set(true));
    let r = f();
    COUNTING.with(|c| c.set(false));
    (r, ALLOCS.with(|a| a.get()))
}

fn workspace_root() -> PathBuf {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.pop();
    root.pop();
    root
}

fn save_gate_metrics(name: &str, payload: serde_json::Value) {
    let dir = workspace_root().join("target").join("gate_metrics");
    let _ = fs::create_dir_all(&dir);
    if let Ok(s) = serde_json::to_string_pretty(&payload) {
        let _ = fs::write(dir.join(format!("{name}.json")), s);
    }
}

fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("toneforge_gate_{tag}_{}", std::process::id()));
    let _ = fs::create_dir_all(&d);
    d
}

/// Zero-crossing frequency estimate (rising edges, linear interpolation) of a near-pure tone.
fn zero_cross_hz(x: &[f32], sr: f64) -> f64 {
    let mut first = None;
    let mut last = 0.0f64;
    let mut count = 0usize;
    for i in 1..x.len() {
        if x[i - 1] < 0.0 && x[i] >= 0.0 {
            let frac = (-x[i - 1] / (x[i] - x[i - 1])) as f64;
            let t = (i - 1) as f64 + frac;
            if first.is_none() {
                first = Some(t);
            } else {
                count += 1;
            }
            last = t;
        }
    }
    match first {
        Some(f) if count > 0 => sr * (count as f64) / (last - f),
        _ => 0.0,
    }
}

fn rms(x: &[f32]) -> f64 {
    (x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / (x.len().max(1) as f64)).sqrt()
}

fn correlation(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for i in 0..n {
        ab += (a[i] as f64) * (b[i] as f64);
        aa += (a[i] as f64) * (a[i] as f64);
        bb += (b[i] as f64) * (b[i] as f64);
    }
    ab / (aa.sqrt() * bb.sqrt()).max(1e-30)
}

/// Builds a non-RT engine for `ps` exactly as the offline renderer / project loader does.
fn build_offline_graph(ps: &ToneForgeProjectState, sr: f64) -> AudioStreamController {
    let mut ctrl = AudioStreamController::new(sr, 256, 0);
    ctrl.apply_project_state(ps.clone()).expect("apply_project_state");
    ctrl.load_project_media(|a| {
        load_asset_frames(a, sr as u32).map(Arc::new)
    })
    .expect("load_project_media");
    ctrl
}

/// Patcher canvases and FL26 presets are re-applied on top of the project graph, exactly as the
/// offline renderer in `ToneForge_Studio.exe` does.
fn apply_canvases_and_fl26(ctrl: &mut AudioStreamController, ps: &ToneForgeProjectState) {
    for canvas in &ps.patcher_canvases {
        daw_tauri_host::cmd_update_patcher_graph(&mut ctrl.mixer_engine, canvas).expect("patcher");
    }
    for p in &ps.fl26_fx_presets {
        let AudioStreamController { mixer_engine, app_state, .. } = ctrl;
        daw_tauri_host::apply_fl26_fx_preset_to_engine(mixer_engine, app_state, p.track_idx, p.slot_idx, &p.preset_query)
            .expect("fl26 preset");
    }
}

fn locate_test_plugin_dll() -> PathBuf {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.pop();
    root.pop();
    for profile in ["release", "debug"] {
        let p = root.join("target").join(profile).join("test_vst3_clap_plugin.dll");
        if p.is_file() {
            return p;
        }
    }
    panic!("test_vst3_clap_plugin.dll not built (cargo build -p test-vst3-clap-plugin --release)");
}

/// Goertzel magnitude (dBFS of a full-scale sine) of `x` at `f` Hz.
fn tone_db(x: &[f32], f: f64, sr: f64) -> f64 {
    let w = TAU * f / sr;
    let coeff = 2.0 * w.cos();
    let (mut s1, mut s2) = (0.0f64, 0.0f64);
    for &v in x {
        let s0 = v as f64 + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    let power = s1 * s1 + s2 * s2 - coeff * s1 * s2;
    20.0 * (2.0 * power.max(1e-30).sqrt() / x.len() as f64).log10()
}

fn render_left(engine: &mut MixerEngine, frames: usize) -> Vec<f32> {
    engine.is_playing = true;
    let mut out = Vec::with_capacity(frames);
    let mut blk = [[0.0f32; 2]; 256];
    while out.len() < frames {
        blk.fill([0.0; 2]);
        engine.process_audio_block(&mut blk);
        out.extend(blk.iter().map(|f| f[0]));
    }
    out.truncate(frames);
    out
}

#[test]
fn gate11_full_stack_host_e2e_parity() {
    let sr = 48_000.0;
    let dir = temp_dir("g11");

    // ======================================================================
    // 11.1 RECORD (44.1 kHz device -> 48 kHz engine) -> TAKE ASSET -> PLAYBACK -> SAVE/LOAD
    // ======================================================================
    let (mut engine, _cmd, mut muts, _gc, _tel) = MixerEngine::new_with_rt_gc(sr, 120.0, 0);
    engine.route_inserts_to_master(4).unwrap();
    // Track 1 also plays other content: the take must contain the *input* only.
    engine
        .add_channel_generator(ChannelGenerator::new(
            7,
            1,
            Box::new(DeterministicGeneratorNode::new(0, 0.25, false)),
        ))
        .unwrap();
    let (mut cap_tx, cap_rx) = rtrb::RingBuffer::<StereoFrame>::new(1 << 16);
    engine.hw_capture_bridge = Some((cap_rx, AdaptiveDllDriftResampler::new(44_100.0, sr)));
    muts.push(RtGraphMutation::SetTrackInputAndRecordArm {
        track_index: 1,
        input_source_bus: Some(0),
        record_armed: true,
        hw_input_latency_samples: 0,
    })
    .map_err(|_| "mutation ring full")
    .unwrap();
    let (recorder, tap) = DiskTrackRecorder::spawn(dir.clone(), 1, sr as u32, 0, None, 1 << 18);
    muts.push(RtGraphMutation::AttachRecordTap { tap: Box::new(Some(tap)) })
        .map_err(|_| "mutation ring full")
        .unwrap();

    let in_hz = 330.0f64;
    let mut dev_phase = 0.0f64;
    let mut dev_frac = 0.0f64;
    let mut blk = [[0.0f32; 2]; 256];
    for _ in 0..((2.0 * sr) as usize / 256) {
        // Device delivers 256 * 44100/48000 frames per engine block.
        dev_frac += 256.0 * 44_100.0 / sr;
        let n = dev_frac.floor() as usize;
        dev_frac -= n as f64;
        for _ in 0..n {
            let v = (0.5 * dev_phase.sin()) as f32;
            dev_phase += TAU * in_hz / 44_100.0;
            let _ = cap_tx.push([v, v]);
        }
        engine.process_audio_block(&mut blk);
    }
    muts.push(RtGraphMutation::AttachRecordTap { tap: Box::new(None) })
        .map_err(|_| "mutation ring full")
        .unwrap();
    engine.process_audio_block(&mut blk);
    let takes = recorder.finish_and_collect_takes();
    assert_eq!(takes.len(), 1, "one take expected");
    let (take_sr, take_frames) = read_wav_f32_stereo(&takes[0].wav_path).expect("take wav");
    assert_eq!(take_sr, 48_000);
    let settled: Vec<f32> = take_frames[(0.4 * sr) as usize..].iter().map(|f| f[0]).collect();
    let take_hz = zero_cross_hz(&settled, sr);
    let take_rms = rms(&settled);
    assert!((take_hz - in_hz).abs() < 0.5, "take pitch {take_hz:.3} Hz (44.1k capture must not be pitched)");
    assert!(
        (take_rms - 0.5 / 2f64.sqrt()).abs() < 0.02,
        "take must hold the raw input only (rms {take_rms:.4})"
    );

    // Playback through the project asset + Playlist clip.
    let mut ps = ToneForgeProjectState::new_default(sr, 120.0);
    ps.ppq_resolution = 96;
    let asset = AudioAssetDto {
        asset_id: daw_tauri_host::ASSET_ID_TAKE_BASE,
        kind: KIND_RECORDED_TAKE.to_string(),
        name: "Take 1".to_string(),
        file_path: takes[0].wav_path.display().to_string(),
        target_track: 1,
        continuous: false,
        loop_playback: false,
    };
    ps.audio_assets.push(asset.clone());
    let spt = samples_per_tick(&ps, sr);
    let clip_start_tick = 384i64;
    ps.playlist_clips.push(PlaylistClipDto {
        clip_id: 50,
        track_row: 1,
        kind: "AudioStem".to_string(),
        source_id: asset.asset_id,
        start_tick: clip_start_tick,
        duration_ticks: ((take_frames.len() as f64) / spt).floor() as i64,
        ..PlaylistClipDto::default()
    });
    let mut live = build_offline_graph(&ps, sr);
    let total = (clip_start_tick as f64 * spt) as usize + take_frames.len();
    let out = render_left(&mut live.mixer_engine, total);
    let start = (clip_start_tick as f64 * spt).round() as usize;
    assert!(rms(&out[..start - 64]) < 1e-6, "audio before the clip start");
    let seg = &out[start + 9_600..start + 72_000];
    let src: Vec<f32> = take_frames[9_600..72_000].iter().map(|f| f[0]).collect();
    let corr = correlation(seg, &src);
    let play_rms_dbfs = 20.0 * rms(seg).log10();
    assert!(corr >= 0.99, "played take correlation {corr:.5}");
    assert!(play_rms_dbfs > -20.0, "played take level {play_rms_dbfs:.2} dBFS");

    // Save -> load -> identical audio.
    let json = ps.to_json_pretty().unwrap();
    let reloaded = ToneForgeProjectState::from_json_str(&json).unwrap();
    assert_eq!(reloaded.audio_assets, ps.audio_assets);
    let mut again = build_offline_graph(&reloaded, sr);
    let out2 = render_left(&mut again.mixer_engine, total);
    let max_reload_diff = out
        .iter()
        .zip(&out2)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_reload_diff == 0.0, "save/load changed the audio ({max_reload_diff})");

    // ======================================================================
    // 11.2 PLAYLIST CLIP STRETCH 1.35x / PITCH +4 st (NO VARISPEED) + SLICE / FADE
    // ======================================================================
    let src_len = (2.0 * sr) as usize;
    let tone: Vec<StereoFrame> = (0..src_len)
        .map(|n| {
            let v = (0.4 * (TAU * 440.0 * n as f64 / sr).sin()) as f32;
            [v, v]
        })
        .collect();
    let tone_path = dir.join("tone440.wav");
    write_f32_wav_file(&tone_path, &tone, sr as u32).unwrap();
    let mut ps2 = ToneForgeProjectState::new_default(sr, 120.0);
    ps2.ppq_resolution = 96;
    let spt2 = samples_per_tick(&ps2, sr);
    let stretch_clip = PlaylistClipDto {
        clip_id: 61,
        kind: "AudioStem".to_string(),
        source_id: daw_tauri_host::ASSET_ID_TAKE_BASE + 1,
        start_tick: 0,
        duration_ticks: ((src_len as f64 * 1.35) / spt2).floor() as i64,
        stretch_ratio: 1.35,
        ..PlaylistClipDto::default()
    };
    let mut ch = build_derived_clip_channel(&stretch_clip, &tone, 1, &ps2, sr);
    let mut st_out = vec![[0.0f32; 2]; (src_len as f64 * 1.35) as usize];
    let ctx = daw_core_graph::ProcessContext {
        sample_rate: sr,
        current_sample_pos: 0,
        ppq_tick: 0,
        bpm: 120.0,
        slice_frames: st_out.len(),
        is_playing: true,
        is_offline_rendering: true,
        ppq_resolution: daw_core_graph::DEFAULT_PPQ,
    };
    ch.generator.process_slice(&ctx, &mut st_out, None, &mut []);
    let st_mid: Vec<f32> = st_out[24_000..96_000].iter().map(|f| f[0]).collect();
    let stretched_hz = zero_cross_hz(&st_mid, sr);
    assert!(
        (stretched_hz - 440.0).abs() <= 1.5,
        "1.35x stretch changed pitch: {stretched_hz:.2} Hz (varispeed would give 325.9 Hz)"
    );
    let pitch_clip = PlaylistClipDto {
        clip_id: 62,
        stretch_ratio: 1.0,
        pitch_semitones: 4.0,
        duration_ticks: ((src_len as f64) / spt2).floor() as i64,
        ..stretch_clip.clone()
    };
    let mut pch = build_derived_clip_channel(&pitch_clip, &tone, 1, &ps2, sr);
    let mut p_out = vec![[0.0f32; 2]; src_len];
    pch.generator.process_slice(&ctx, &mut p_out, None, &mut []);
    let p_mid: Vec<f32> = p_out[24_000..72_000].iter().map(|f| f[0]).collect();
    let pitched_hz = zero_cross_hz(&p_mid, sr);
    let want_pitch = 440.0 * 2f64.powf(4.0 / 12.0);
    assert!(
        (pitched_hz - want_pitch).abs() <= 2.0,
        "+4 st clip pitch {pitched_hz:.2} Hz, want {want_pitch:.2}"
    );
    // Slice into A [0, 1 s) with a 100 ms fade-out and B [1.5 s, 2.0 s) from source offset 1.0 s.
    let tone_asset = AudioAssetDto {
        asset_id: daw_tauri_host::ASSET_ID_TAKE_BASE + 1,
        kind: KIND_RECORDED_TAKE.to_string(),
        name: "tone".into(),
        file_path: tone_path.display().to_string(),
        target_track: 1,
        continuous: false,
        loop_playback: false,
    };
    ps2.audio_assets.push(tone_asset.clone());
    let one_s_ticks = (sr / spt2).round() as i64;
    ps2.playlist_clips.push(PlaylistClipDto {
        clip_id: 70,
        kind: "AudioStem".into(),
        source_id: tone_asset.asset_id,
        start_tick: 0,
        duration_ticks: one_s_ticks,
        fade_out_samples: 4_800,
        ..PlaylistClipDto::default()
    });
    ps2.playlist_clips.push(PlaylistClipDto {
        clip_id: 71,
        kind: "AudioStem".into(),
        source_id: tone_asset.asset_id,
        start_tick: one_s_ticks * 3 / 2,
        duration_ticks: one_s_ticks / 2,
        source_offset_samples: sr as usize,
        ..PlaylistClipDto::default()
    });
    let (_tbl, region_count) = asset_region_table(&ps2, tone_asset.asset_id, sr);
    assert_eq!(region_count, 2);
    let mut slice_ctrl = build_offline_graph(&ps2, sr);
    let s_out = render_left(&mut slice_ctrl.mixer_engine, (2.0 * sr) as usize);
    let mid_a = rms(&s_out[12_000..36_000]);
    let fade_tail = rms(&s_out[47_760..48_000]);
    let gap = rms(&s_out[49_000..71_000]);
    let b_body = rms(&s_out[74_000..94_000]);
    assert!(fade_tail < 0.1 * mid_a, "fade-out not applied ({fade_tail:.4} vs {mid_a:.4})");
    assert!(gap < 1e-6, "audio in the slice gap ({gap})");
    assert!((b_body - mid_a).abs() < 0.01, "slice B level {b_body:.4} vs A {mid_a:.4}");

    // ======================================================================
    // 11.3 LIVE GRAPH (RT mutation queues) == OFFLINE RENDER GRAPH, UNDO/REDO ROUND TRIP
    // ======================================================================
    let mut proj = ToneForgeProjectState::new_mastering_template(sr, 120.0);
    proj.ppq_resolution = 96;
    proj.audio_assets.push(asset.clone());
    proj.playlist_clips.push(ps.playlist_clips[0].clone());
    // NewTone-committed vocal (+3 st) persisted as a continuous asset on Insert 2.
    let vocal: Vec<StereoFrame> = (0..(1.5 * sr) as usize)
        .map(|n| {
            let t = n as f64 / sr;
            let f0 = 220.0 * (1.0 + 0.01 * (TAU * 5.0 * t).sin());
            let v = (0.3 * (TAU * f0 * t).sin() + 0.1 * (TAU * 2.0 * f0 * t).sin()) as f32;
            [v, v]
        })
        .collect();
    let analysis = daw_tauri_host::analyze_vocal_pitch_blobs_dsp(vocal.clone(), sr as u32);
    let mut blobs = analysis.blobs.clone();
    for b in &mut blobs {
        b.target_midi_note = b.detected_midi_note + 3.0;
    }
    let committed = daw_tauri_host::commit_vocal_pitch_edits_dsp(vocal, sr as u32, blobs);
    let vocal_path = dir.join("newtone_commit.wav");
    write_f32_wav_file(&vocal_path, &committed, sr as u32).unwrap();
    let vocal_asset = AudioAssetDto {
        asset_id: daw_tauri_host::ASSET_ID_NEWTONE_BASE + 2,
        kind: KIND_NEWTONE_COMMIT.to_string(),
        name: "NewTone".into(),
        file_path: vocal_path.display().to_string(),
        target_track: 2,
        continuous: true,
        loop_playback: false,
    };
    proj.audio_assets.push(vocal_asset.clone());
    proj.mixer_tracks[2].pan = -0.3;
    proj.mixer_tracks[3].muted = true;

    // Hosted VST3 instrument (test_vst3_clap_plugin.dll) on Insert 5 playing a D major chord
    // (D5 F#5 A5) from a Pattern clip.
    let dll = locate_test_plugin_dll();
    proj.mixer_tracks[5].channel_instrument = Some(daw_tauri_host::project_state::PluginConfigDto::ExternalVst3(
        daw_tauri_host::project_state::ExternalPluginStateDto {
            binary_path: dll.display().to_string(),
            plugin_id: String::new(),
            plugin_name: "ToneForge Test Synth".into(),
            state_hex: String::new(),
            param_overrides: Vec::new(),
        },
    ));
    proj.mixer_tracks[5].sends.push(daw_tauri_host::project_state::MixerSendRouteDto {
        dest_track_index: 0,
        gain: 1.0,
        is_sidechain_only: false,
    });
    proj.piano_roll_patterns.push(daw_tauri_host::project_state::PianoRollPatternDto {
        pattern_id: 7,
        name: "Chord".into(),
        target_channel_id: 205,
        length_ticks: 384,
        notes: [74u8, 78, 81]
            .iter()
            .enumerate()
            .map(|(i, &key)| daw_tauri_host::project_state::PianoRollNoteDto {
                id: 700 + i as u32,
                start_tick: 0,
                duration_ticks: 288,
                key,
                velocity: 0.8,
                pan: 0.0,
                fine_pitch_cents: 0.0,
                is_slide: false,
                color_group: 0,
            })
            .collect(),
    });
    proj.playlist_clips.push(PlaylistClipDto {
        clip_id: 70,
        track_row: 5,
        kind: "Pattern".to_string(),
        source_id: 7,
        start_tick: 0,
        duration_ticks: 384,
        gain: 1.0,
        ..PlaylistClipDto::default()
    });
    // Custom Patcher canvas (real Pro-Q 4 Linear Phase + Soothe3 + APDC delay) and an FL26 preset.
    proj.patcher_canvases.push(
        serde_json::from_str(
            r#"{ "track_idx": 1, "slot_idx": 2,
                 "nodes": [
                   { "node_index": 0, "plugin_kind": "HostIn" },
                   { "node_index": 1, "plugin_kind": "ProQ4", "latency_samples": 4096 },
                   { "node_index": 2, "plugin_kind": "Soothe3", "latency_samples": 2304 },
                   { "node_index": 3, "plugin_kind": "FixedLatency", "latency_samples": 1024 },
                   { "node_index": 4, "plugin_kind": "Ozone12" } ],
                 "edges": [
                   { "from_node": 0, "to_node": 1, "gain": 1.0 },
                   { "from_node": 0, "to_node": 2, "gain": 1.0 },
                   { "from_node": 1, "to_node": 3, "gain": 1.0 },
                   { "from_node": 3, "to_node": 4, "gain": 0.5 },
                   { "from_node": 2, "to_node": 4, "gain": 0.5 } ] }"#,
        )
        .unwrap(),
    );
    proj.fl26_fx_presets.push(daw_tauri_host::project_state::Fl26SlotPresetDto {
        track_idx: 2,
        slot_idx: 5,
        preset_query: "Fruity Soft Clipper".into(),
    });

    // Undo / Redo round trip returns the exact project.
    let mut hist = ProjectUndoHistory::default();
    let before = proj.clone();
    hist.record_before_action("Gain", &proj);
    proj.mixer_tracks[1].gain = 0.5;
    let after = proj.clone();
    let undo = hist.undo(&mut proj).expect("undo");
    let undo_exact = serde_json::to_value(&undo.restored_state).unwrap() == serde_json::to_value(&before).unwrap();
    let redo = hist.redo(&mut proj).expect("redo");
    let redo_exact = serde_json::to_value(&redo.restored_state).unwrap() == serde_json::to_value(&after).unwrap();
    assert!(undo_exact && redo_exact, "undo/redo must restore the exact project state");

    // Offline graph.
    let render_frames = (3.0 * sr) as usize;
    let mut offline = build_offline_graph(&proj, sr);
    apply_canvases_and_fl26(&mut offline, &proj);
    offline.mixer_engine.is_offline_rendering = true;
    let off_out = render_left(&mut offline.mixer_engine, render_frames);

    // Live graph: the mastering template the app starts with, brought to `proj` through the
    // same lock-free RT mutation path `ToneForge_Studio.exe` uses.
    let (mut live_ctrl, _gc2) = AudioStreamController::with_mastering_template_rt_gc(sr, 256).unwrap();
    let template = live_ctrl.project_state.clone();
    sync_project_state_diff_to_rt(&template, &proj, &live_ctrl.app_state, Some(&live_ctrl.collision_bus))
        .unwrap();
    for a in &proj.audio_assets {
        let frames = load_asset_frames(a, sr as u32).unwrap();
        let ch = build_asset_channel(a, frames, &proj, sr);
        live_ctrl
            .app_state
            .push_rt_mutation(RtGraphMutation::UpsertChannel { channel: Box::new(Some(ch)) })
            .unwrap();
    }
    for canvas in &proj.patcher_canvases {
        daw_tauri_host::cmd_update_patcher_graph_rt(&live_ctrl.app_state, sr, canvas).unwrap();
    }
    for p in &proj.fl26_fx_presets {
        daw_tauri_host::apply_fl26_fx_preset_rt(&live_ctrl.app_state, sr, p.track_idx, p.slot_idx, &p.preset_query)
            .unwrap();
    }
    live_ctrl.app_state.push_command(EngineCommand::SeekPosition { sample_pos: 0, ppq_tick: 0 }).unwrap();
    let live_out = render_left(&mut live_ctrl.mixer_engine, render_frames);

    // The hosted instrument is audible (F#5 of the chord, first 1.5 s) and identical in both graphs.
    let chord = (0.2 * sr) as usize..(1.5 * sr) as usize;
    let fs5_live = tone_db(&live_out[chord.clone()], 739.99, sr);
    let fs5_off = tone_db(&off_out[chord.clone()], 739.99, sr);
    assert!(fs5_live > -50.0, "VST3 instrument chord missing from the live mix ({fs5_live:.1} dB)");
    assert!((fs5_live - fs5_off).abs() < 0.5, "instrument live {fs5_live:.2} dB vs offline {fs5_off:.2} dB");

    let lufs_of = |x: &[f32]| {
        let mut m = LufsTruePeakMeter::new(sr);
        let frames: Vec<StereoFrame> = x.iter().map(|v| [*v, *v]).collect();
        m.push_block(&frames, 0.0);
        m.lufs_integrated as f64
    };
    let lufs_live = lufs_of(&live_out);
    let lufs_off = lufs_of(&off_out);
    assert!(lufs_live > -60.0, "live mix is silent ({lufs_live:.2} LUFS)");
    assert!(
        (lufs_live - lufs_off).abs() <= 0.15,
        "live {lufs_live:.3} LUFS vs offline {lufs_off:.3} LUFS"
    );

    save_gate_metrics(
        "gate11",
        serde_json::json!({
            "gate": "Gate 11 — Full-Stack Host E2E Parity",
            "status": "PASS",
            "record_44k1_capture_to_48k_engine": {
                "take_pitch_hz": take_hz,
                "input_pitch_hz": in_hz,
                "take_rms": take_rms,
                "playback_correlation": corr,
                "playback_rms_dbfs": play_rms_dbfs,
                "save_load_max_abs_diff": max_reload_diff,
            },
            "playlist_clip_dsp": {
                "stretch_1_35x_pitch_hz": stretched_hz,
                "pitch_plus_4st_hz": pitched_hz,
                "pitch_plus_4st_target_hz": want_pitch,
                "slice_fade_tail_rms": fade_tail,
                "slice_gap_rms": gap,
            },
            "live_vs_offline": {
                "lufs_live": lufs_live,
                "lufs_offline": lufs_off,
                "delta_lu": (lufs_live - lufs_off).abs(),
                "undo_restores_exact_state": undo_exact,
                "vst3_instrument_fs5_live_db": fs5_live,
                "vst3_instrument_fs5_offline_db": fs5_off,
                "patcher_canvases": proj.patcher_canvases.len(),
                "fl26_presets": proj.fl26_fx_presets.len(),
                "redo_restores_exact_state": redo_exact,
            }
        }),
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn gate12_ipc_contract_and_high_density_sequencing() {
    // ======================================================================
    // 12.1 Every `invoke('cmd_*')` in the UI is registered in `generate_handler!`, the undo DTO
    //      field names match, and no desktop IPC failure is silently swallowed.
    // ======================================================================
    let root = workspace_root();
    let main_rs = fs::read_to_string(root.join("shell/src-tauri/src/main.rs")).unwrap();
    let handler_start = main_rs.find("generate_handler![").expect("generate_handler!");
    let handler_end = handler_start + main_rs[handler_start..].find("])").unwrap();
    let handler = &main_rs[handler_start..handler_end];
    let mut ui_sources = Vec::new();
    let mut stack = vec![root.join("ui-svelte5-wgpu/src")];
    while let Some(d) = stack.pop() {
        for e in fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if matches!(p.extension().and_then(|x| x.to_str()), Some("ts" | "svelte")) {
                ui_sources.push((p.clone(), fs::read_to_string(&p).unwrap_or_default()));
            }
        }
    }
    let mut invoked = std::collections::BTreeSet::new();
    for (_, src) in &ui_sources {
        let mut rest = src.as_str();
        while let Some(i) = rest.find("'cmd_") {
            let tail = &rest[i + 1..];
            let end = tail.find('\'').unwrap_or(0);
            invoked.insert(tail[..end].to_string());
            rest = &tail[end..];
        }
    }
    let unregistered: Vec<_> = invoked
        .iter()
        .filter(|c| !handler.contains(&format!("{c},")) && !handler.contains(&format!("{c}\n")))
        .cloned()
        .collect();
    assert!(unregistered.is_empty(), "UI invokes unregistered commands: {unregistered:?}");

    let status = daw_tauri_host::UndoRedoStatusDto {
        can_undo: true,
        can_redo: false,
        undo_depth: 3,
        redo_depth: 0,
        next_undo_label: Some("x".into()),
        next_redo_label: None,
        last_applied_label: None,
    };
    let v = serde_json::to_value(&status).unwrap();
    let store = &ui_sources
        .iter()
        .find(|(p, _)| p.ends_with("daw_master_store.svelte.ts"))
        .unwrap()
        .1;
    for key in v.as_object().unwrap().keys() {
        assert!(store.contains(&format!("st.{key}")), "UI store never reads UndoRedoStatusDto.{key}");
    }
    // Silent catch blocks around `invoke` are forbidden in the desktop store.
    let mut silent_catches = 0usize;
    for (i, _) in store.match_indices("} catch {") {
        let window = &store[store[..i].rfind("try {").unwrap_or(0)..i];
        if window.contains("invoke") {
            silent_catches += 1;
        }
    }
    assert_eq!(silent_catches, 0, "silent `catch {{}}` around invoke()");

    // ======================================================================
    // 12.2 5,000 Piano Roll notes: 0 RT allocations, < 50 us per 256-frame block (sequencer + synth)
    // ======================================================================
    let sr = 48_000.0;
    let (mut engine, mut cmd, _tel) = MixerEngine::new(sr, 140.0, 0);
    engine
        .add_channel_generator(ChannelGenerator::new(
            1,
            1,
            Box::new(daw_core_graph::PolyphonicSynthGeneratorNode::new(1, sr)),
        ))
        .unwrap();
    engine.route_inserts_to_master(1).unwrap();
    let mut total_notes = 0usize;
    for pat_id in 0..10u32 {
        let mut pat = PianoRollPattern::new(pat_id + 1, "dense", 1, 96 * 64);
        for n in 0..500u32 {
            let start = ((n as i64) * 12) % (96 * 64);
            if pat.add_note(PianoRollNote::new(n + 1, start, 6, 36 + (n % 48) as u8, 0.7)) {
                total_notes += 1;
            }
        }
        engine.sequencer.upsert_pattern(pat);
    }
    assert_eq!(total_notes, 5_000);
    engine.sequencer.song_mode = false;
    engine.sequencer.active_pattern_id = 1;
    let _ = cmd.push(EngineCommand::SetTransportPlaying(true));
    let mut blk = [[0.0f32; 2]; 256];
    for _ in 0..32 {
        engine.process_audio_block(&mut blk);
    }
    let mut worst_us = 0.0f64;
    let mut total_us = 0.0f64;
    let blocks = 2_000;
    let (_, allocs) = count_allocs(|| {
        for _ in 0..blocks {
            let t = std::time::Instant::now();
            engine.process_audio_block(&mut blk);
            let us = t.elapsed().as_secs_f64() * 1e6;
            worst_us = worst_us.max(us);
            total_us += us;
        }
    });
    let mean_us = total_us / blocks as f64;
    assert_eq!(allocs, 0, "RT allocations while sequencing 5,000 notes");
    assert!(mean_us < 50.0, "mean block time {mean_us:.2} us with 5,000 notes");

    // ======================================================================
    // 12.3 Soprano (f0 = 400 Hz) + plosive onset shifted +10 st with formant preservation:
    //      transient crest retention >= 85 % and no F1 hole: the output harmonics follow the input
    //      formant envelope as sampled by the input harmonics (the resolution a 400 Hz source allows).
    // ======================================================================
    let (crest_retention, env_max_dev_db) = soprano_plosive_shift_metrics();
    assert!(crest_retention >= 0.85, "plosive crest retention {crest_retention:.3}");
    assert!(env_max_dev_db <= 3.0, "formant envelope deviation {env_max_dev_db:.2} dB");

    // ======================================================================
    // 12.4 Codec Preview adaptive long/short window switching + bit reservoir:
    //      PCM bypass null <= -120 dBFS while switching, and lower pre-echo before a drum hit.
    // ======================================================================
    let (pcm_null_dbfs, short_frames, pre_echo_long_db, pre_echo_switch_db) = codec_switching_metrics();
    assert!(short_frames > 0, "transient never triggered short blocks");
    assert!(pcm_null_dbfs <= -120.0, "PCM bypass null {pcm_null_dbfs:.1} dBFS with block switching");
    assert!(
        pre_echo_switch_db <= pre_echo_long_db - 6.0,
        "pre-echo not reduced: long-only {pre_echo_long_db:.1} dB vs switching {pre_echo_switch_db:.1} dB"
    );

    save_gate_metrics(
        "gate12",
        serde_json::json!({
            "gate": "Gate 12 — UI/Host IPC Contract & High-Density Sequencing",
            "status": "PASS",
            "ipc": {
                "ui_invoked_commands": invoked.len(),
                "unregistered_commands": unregistered.len(),
                "undo_status_fields_checked": v.as_object().unwrap().len(),
                "silent_invoke_catch_blocks": silent_catches,
            },
            "sequencer_5000_notes": {
                "notes": total_notes,
                "blocks": blocks,
                "mean_block_us": mean_us,
                "worst_block_us": worst_us,
                "rt_alloc_and_dealloc_calls": allocs,
            },
            "codec_preview_window_switching": {
                "pcm_bypass_null_dbfs": pcm_null_dbfs,
                "short_frames": short_frames,
                "mp3_128_pre_echo_long_only_db": pre_echo_long_db,
                "mp3_128_pre_echo_switching_db": pre_echo_switch_db,
            },
            "soprano_400hz_plus_10st": {
                "plosive_crest_retention": crest_retention,
                "formant_envelope_max_dev_db_600_2500hz": env_max_dev_db,
            }
        }),
    );
}

/// Formant envelope of the synthetic soprano /u/-like vowel (F1 350 Hz, F2 800 Hz, F3 2500 Hz).
fn soprano_envelope(f: f64) -> f64 {
    let pk = |fc: f64, bw: f64, g: f64| g / (1.0 + ((f - fc) / bw).powi(2));
    0.05 + pk(350.0, 90.0, 1.0) + pk(800.0, 120.0, 0.6) + pk(2_500.0, 200.0, 0.25)
}

fn soprano_plosive_shift_metrics() -> (f64, f64) {
    use realfft::RealFftPlanner;
    let sr = 48_000.0f64;
    let n_total = (1.2 * sr) as usize;
    let onset = (0.3 * sr) as usize;
    let mut x = vec![0.0f32; n_total];
    let mut lcg: u32 = 0x1234_5678;
    for (i, v) in x.iter_mut().enumerate() {
        if i < onset {
            continue;
        }
        let t = (i - onset) as f64 / sr;
        // /p/ burst: 4 ms decaying noise, then the voiced vowel.
        lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let noise = ((lcg >> 8) as f64 / (1u32 << 24) as f64) * 2.0 - 1.0;
        let burst = 0.9 * noise * (-t / 0.0015).exp();
        let mut vowel = 0.0;
        if t > 0.004 {
            let a = ((t - 0.004) / 0.01).min(1.0);
            for h in 1..40 {
                let f = 400.0 * h as f64;
                if f > 0.45 * sr {
                    break;
                }
                vowel += soprano_envelope(f) * (TAU * f * t).sin();
            }
            vowel *= 0.12 * a;
        }
        *v = (burst + vowel) as f32;
    }
    let mut shifter = dsp_pitch_time::FormantPitchShifter::new(sr);
    let y = shifter.process_offline_compensated(&x, 400.0, 10.0, 0.0, true);

    // Crest factor (peak / RMS) of the plosive burst, input vs shifted output.
    let crest = |s: &[f32]| {
        let peak = s.iter().fold(0.0f32, |m, v| m.max(v.abs())) as f64;
        peak / rms(s).max(1e-12)
    };
    let win = onset..onset + (0.006 * sr) as usize;
    let crest_retention = crest(&y[win.clone()]) / crest(&x[win]);

    // Envelope fidelity: output harmonic magnitudes (f0' = 400 * 2^(10/12)) must follow the input
    // formant envelope (formant preserved, no F1 hole) between 600 and 2500 Hz. A 400 Hz source only
    // samples the vocal tract every 400 Hz, so the reference is the input's own harmonic levels
    // interpolated in dB; detail narrower than the harmonic spacing (the 90/120 Hz-wide F1-F2 valley)
    // is not recoverable from the signal by any estimator.
    let n = 16_384;
    let seg_start = onset + (0.4 * sr) as usize;
    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let spectrum = |sig: &[f32]| {
        let mut buf: Vec<f64> = (0..n)
            .map(|i| {
                let w = 0.5 - 0.5 * (TAU * i as f64 / n as f64).cos();
                sig[seg_start + i] as f64 * w
            })
            .collect();
        let mut out = fft.make_output_vec();
        fft.process(&mut buf, &mut out).unwrap();
        out.iter().map(|c| c.norm()).collect::<Vec<f64>>()
    };
    let spec_y = spectrum(&y);
    let spec_x = spectrum(&x);
    let f0_out = 400.0 * 2f64.powf(10.0 / 12.0);
    let bin_of = |f: f64| (f * n as f64 / sr).round() as usize;
    let peak_db = |spec: &[f64], f: f64| {
        let b = bin_of(f);
        20.0 * (b.saturating_sub(3)..=b + 3).map(|k| spec[k]).fold(0.0, f64::max).log10()
    };
    let input_env_db = |f: f64| {
        let h = (f / 400.0).floor().max(1.0);
        let (lo, hi) = (peak_db(&spec_x, 400.0 * h), peak_db(&spec_x, 400.0 * (h + 1.0)));
        lo + (f / 400.0 - h) * (hi - lo)
    };
    let mut pairs = Vec::new();
    for h in 1..8 {
        let f = f0_out * h as f64;
        if !(600.0..=2_500.0).contains(&f) {
            continue;
        }
        pairs.push((peak_db(&spec_y, f), input_env_db(f)));
    }
    // Compare shapes (relative to the first harmonic in range) so absolute gain does not matter.
    let (r0_out, r0_env) = pairs[0];
    let env_max_dev_db = pairs
        .iter()
        .map(|(o, e)| ((o - r0_out) - (e - r0_env)).abs())
        .fold(0.0, f64::max);
    (crest_retention, env_max_dev_db)
}


fn codec_switching_metrics() -> (f64, u64, f64, f64) {
    use dsp_ozone12_mastering::{CODEC_PREVIEW_LATENCY_SAMPLES, CodecPreviewModule, CodecTarget};
    let sr = 48_000.0f64;
    let n = 48_000usize;
    // Quiet tonal bed, then sharp drum hits every 250 ms.
    let mut x = vec![[0.0f32; 2]; n];
    let mut lcg: u32 = 0xC0DE_C0DE;
    for (i, f) in x.iter_mut().enumerate() {
        let t = i as f64 / sr;
        let bed = 0.02 * (TAU * 220.0 * t).sin();
        let hit_pos = i % 12_000;
        let hit = if hit_pos >= 6_000 {
            let k = (hit_pos - 6_000) as f64 / sr;
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = ((lcg >> 8) as f64 / (1u32 << 24) as f64) * 2.0 - 1.0;
            0.8 * (-k / 0.004).exp() * (0.6 * noise + 0.4 * (TAU * 90.0 * k).sin())
        } else {
            0.0
        };
        let v = (bed + hit) as f32;
        *f = [v, v];
    }
    let run = |mode: CodecTarget, switching: bool, solo: bool| -> (Vec<StereoFrame>, u64) {
        let mut c = CodecPreviewModule::new(sr);
        c.enabled = true;
        c.mode = mode;
        c.window_switching = switching;
        c.artifact_solo = solo;
        let mut y = x.clone();
        for chunk in y.chunks_mut(256) {
            c.process_frames_in_place(chunk);
        }
        (y, c.short_frames)
    };
    // PCM bypass: the artifact (wet - delayed dry) must null even across LONG/START/SHORT/STOP.
    let (delta, short_frames) = run(CodecTarget::PcmBypass, true, true);
    let tail = &delta[CODEC_PREVIEW_LATENCY_SAMPLES * 2..];
    let peak = tail.iter().fold(0.0f32, |m, f| m.max(f[0].abs()).max(f[1].abs())) as f64;
    let pcm_null_dbfs = 20.0 * peak.max(1e-20).log10();

    // Pre-echo: codec error energy ahead of each hit (aligned to the dry input) over one long MDCT
    // window (512 samples), excluding the last 64 samples (1.33 ms) that backward masking covers;
    // short blocks confine noise to that interval, as in MP3/AAC.
    let pre_echo_db = |switching: bool| {
        let (err, _) = run(CodecTarget::Mp3_128, switching, true);
        let mut e = 0.0f64;
        let mut cnt = 0usize;
        for hit in (6_000..n).step_by(12_000) {
            let start = hit + CODEC_PREVIEW_LATENCY_SAMPLES - 512;
            for i in start..start + 512 - 64 {
                if i < err.len() {
                    e += (err[i][0] as f64).powi(2);
                    cnt += 1;
                }
            }
        }
        10.0 * (e / cnt.max(1) as f64).max(1e-20).log10()
    };
    (pcm_null_dbfs, short_frames, pre_echo_db(false), pre_echo_db(true))
}
