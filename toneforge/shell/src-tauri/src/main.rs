// Hide console window on Windows release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

//! ToneForge Studio — Standalone Tauri v2 Desktop Application (`ToneForge_Studio.exe`).
//!
//! Connects the Svelte 5 + WebGPU 120 FPS UI (`ui-svelte5-wgpu/dist`) directly to the
//! real-time Rust `AudioStreamController` (502-Track Mixer, Pro-Q 4, Soothe3, Ozone 12,
//! 120ms BSRNN Stem Rebalance, UVR5 Ensemble Studio, Hybrid PPQ Piano Roll, and
//! 128 KB Zero-Copy Binary Telemetry Streamer).

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use daw_core_graph::{
    MixerEngine, ProAudioThreadGuard, WasapiAsioBlockAdapter, asio_planar_f32_to_int32_lsb,
    stereo_frames_to_wasapi_interleaved,
};
use daw_tauri_host::{
    AudioStreamController, AutomationClipStateDto, DawAppState, OfflineRenderConfig,
    OfflineRenderResult, Ozone12MasterConfigDto, PianoRollPatternDto, PlaylistClipDto,
    ProQ4BandUpdateDto, Soothe3UpdateDto, ZeroCopyBinaryTelemetryStreamer,
};
use parking_lot::Mutex;
use tauri::{RunEvent, State};

struct SharedDawController {
    app_state: DawAppState,
    telemetry_streamer: Mutex<ZeroCopyBinaryTelemetryStreamer>,
    mixer_engine: Arc<Mutex<Box<MixerEngine>>>,
}

#[tauri::command]
fn cmd_poll_binary_telemetry_frame(state: State<'_, SharedDawController>) -> tauri::ipc::Response {
    let mut streamer = state.telemetry_streamer.lock();
    let bytes = daw_tauri_host::cmd_poll_binary_telemetry_frame(&mut streamer);
    tauri::ipc::Response::new(bytes.to_vec())
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_set_transport(
    state: State<'_, SharedDawController>,
    playing: bool,
    bpm: f64,
) -> Result<(), String> {
    daw_tauri_host::cmd_set_transport(&state.app_state, playing, bpm)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_seek_transport(
    state: State<'_, SharedDawController>,
    sample_pos: i64,
    ppq_tick: i64,
) -> Result<(), String> {
    daw_tauri_host::cmd_seek_transport(&state.app_state, sample_pos, ppq_tick)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_update_proq4_band(
    state: State<'_, SharedDawController>,
    dto: ProQ4BandUpdateDto,
) -> Result<(), String> {
    daw_tauri_host::cmd_update_proq4_band(&state.app_state, dto)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_update_soothe3_params(
    state: State<'_, SharedDawController>,
    dto: Soothe3UpdateDto,
) -> Result<(), String> {
    daw_tauri_host::cmd_update_soothe3_params(&state.app_state, dto)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_update_ozone12_master(
    state: State<'_, SharedDawController>,
    track_idx: u16,
    slot_idx: u8,
    cfg: Ozone12MasterConfigDto,
) -> Result<(), String> {
    daw_tauri_host::cmd_update_ozone12_master(&state.app_state, track_idx, slot_idx, &cfg)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_update_bsrnn_stem_gains(
    state: State<'_, SharedDawController>,
    track_idx: u16,
    slot_idx: u8,
    stem_gains_db: [f32; 4],
) -> Result<(), String> {
    daw_tauri_host::cmd_update_bsrnn_stem_gains(&state.app_state, track_idx, slot_idx, stem_gains_db)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_set_track_gain(
    state: State<'_, SharedDawController>,
    track_idx: u16,
    gain_linear: f32,
) -> Result<(), String> {
    daw_tauri_host::cmd_set_track_gain(&state.app_state, track_idx, gain_linear)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_route_mixer_track(
    state: State<'_, SharedDawController>,
    src_track: u16,
    dst_track: u16,
    gain: f32,
    is_sidechain: bool,
) -> Result<(), String> {
    daw_tauri_host::cmd_route_mixer_track(&state.app_state, src_track, dst_track, gain, is_sidechain)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_set_internal_controller_norm(
    state: State<'_, SharedDawController>,
    controller_id: u16,
    norm_val: f64,
) -> Result<(), String> {
    daw_tauri_host::cmd_set_internal_controller_norm(&state.app_state, controller_id, norm_val)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_run_uvr5_separation(audio_path: String, model_hash: String) -> Result<Vec<String>, String> {
    daw_tauri_host::cmd_run_uvr5_separation(audio_path, model_hash)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_set_ppq_resolution(state: State<'_, SharedDawController>, ppq: u32) -> Result<(), String> {
    daw_tauri_host::cmd_set_ppq_resolution(&state.app_state, ppq)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_upsert_piano_roll_pattern(
    state: State<'_, SharedDawController>,
    pattern: PianoRollPatternDto,
) -> Result<(), String> {
    let mut engine = state.mixer_engine.lock();
    daw_tauri_host::cmd_upsert_piano_roll_pattern(&mut engine, &pattern)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_upsert_playlist_clip(
    state: State<'_, SharedDawController>,
    clip: PlaylistClipDto,
) -> Result<(), String> {
    let mut engine = state.mixer_engine.lock();
    daw_tauri_host::cmd_upsert_playlist_clip(&mut engine, &clip)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_update_automation_spline(
    state: State<'_, SharedDawController>,
    clip: AutomationClipStateDto,
    eval_tick: f64,
) -> Result<i32, String> {
    let mut engine = state.mixer_engine.lock();
    daw_tauri_host::cmd_update_automation_spline(&mut engine, &clip, eval_tick)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_execute_offline_render(
    state: State<'_, SharedDawController>,
    config: OfflineRenderConfig,
) -> Result<OfflineRenderResult, String> {
    let mut engine = state.mixer_engine.lock();
    daw_tauri_host::cmd_execute_offline_render(&mut engine, &config)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_apply_proq4_ffp_preset(
    state: State<'_, SharedDawController>,
    track_idx: u16,
    slot_idx: u8,
    preset_query: String,
) -> Result<daw_tauri_host::ProQ4PresetStateDto, String> {
    daw_tauri_host::cmd_apply_proq4_ffp_preset(&state.app_state, track_idx, slot_idx, &preset_query)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_apply_soothe3_factory_preset(
    state: State<'_, SharedDawController>,
    track_idx: u16,
    slot_idx: u8,
    preset_query: String,
) -> Result<daw_tauri_host::Soothe3PresetStateDto, String> {
    daw_tauri_host::cmd_apply_soothe3_factory_preset(
        &state.app_state,
        track_idx,
        slot_idx,
        &preset_query,
    )
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_load_ozone12_genre_or_xml_preset(
    state: State<'_, SharedDawController>,
    track_idx: u16,
    slot_idx: u8,
    query: String,
    is_xml_preset: bool,
) -> Result<daw_tauri_host::Ozone12AssistantPresetDto, String> {
    daw_tauri_host::cmd_load_ozone12_genre_or_xml_preset(
        &state.app_state,
        track_idx,
        slot_idx,
        &query,
        is_xml_preset,
    )
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_run_ozone12_ai_master_assistant(
    state: State<'_, SharedDawController>,
    track_idx: u16,
    slot_idx: u8,
    mel_80: Option<Vec<f32>>,
) -> Result<daw_tauri_host::Ozone12AssistantPresetDto, String> {
    daw_tauri_host::cmd_run_ozone12_ai_master_assistant(&state.app_state, track_idx, slot_idx, mel_80)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_load_fl26_score(
    state: State<'_, SharedDawController>,
    score_query: String,
    target_ppq: u32,
) -> Result<PianoRollPatternDto, String> {
    let mut engine = state.mixer_engine.lock();
    daw_tauri_host::cmd_load_fl26_score(&mut engine, &score_query, target_ppq)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_load_fl26_sample_kit(
    state: State<'_, SharedDawController>,
    kit_query: String,
) -> Result<daw_tauri_host::Fl26SampleKitDto, String> {
    let mut engine = state.mixer_engine.lock();
    daw_tauri_host::apply_fl26_sample_kit_to_engine(&mut engine, &kit_query)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_load_fl26_fx_preset(
    state: State<'_, SharedDawController>,
    track_idx: u16,
    slot_idx: u8,
    preset_query: String,
) -> Result<daw_tauri_host::Fl26FxPresetDto, String> {
    daw_tauri_host::cmd_load_fl26_fx_preset(&state.app_state, track_idx, slot_idx, &preset_query)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_load_uvr5_model_profile(
    profile_query: String,
) -> Result<daw_tauri_host::Uvr5ModelProfileDto, String> {
    daw_tauri_host::cmd_load_uvr5_model_profile(&profile_query)
}

fn clean_webview_cache() {
    if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
        let eb = PathBuf::from(local_app_data)
            .join("com.toneforge.studio")
            .join("EBWebView");
        for rel in [
            "Crashpad",
            "GrShaderCache",
            "ShaderCache",
            "GraphiteDawnCache",
            "Default\\GPUCache",
            "Default\\Code Cache",
        ] {
            let p = eb.join(rel);
            if p.exists() {
                let _ = std::fs::remove_dir_all(&p);
            }
        }
    }
}

fn main() {
    clean_webview_cache();

    let controller = std::thread::Builder::new()
        .name("toneforge-init".to_string())
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            let mut ctrl = AudioStreamController::with_mastering_template(48_000.0, 256)
                .expect("Failed to initialize ToneForge Mastering Template");
            let _ = ctrl.add_polyphonic_synth_channel(1, 1, 64);

            // Seed rich harmonic test stem on Track 2 so real-time 120 FPS meters & analyzers are live
            let mut demo_stem = Vec::with_capacity(4096);
            for n in 0..4096 {
                let t = (n as f32) / 48_000.0;
                let bass = 0.28 * (std::f32::consts::TAU * 55.0 * t).sin()
                    + 0.16 * (std::f32::consts::TAU * 110.0 * t).sin();
                let lead = 0.22 * (std::f32::consts::TAU * 440.0 * t).sin()
                    + 0.14 * (std::f32::consts::TAU * 880.0 * t).cos();
                let air = 0.08 * (std::f32::consts::TAU * 3520.0 * t).sin();
                demo_stem.push([bass + lead + air, bass + lead * 0.92 - air * 0.85]);
            }
            let _ = ctrl.bounce_stems_to_mixer_tracks(&[demo_stem], &[2]);
            let _ = ctrl.route_mixer_track(2, 0, 0.85, false);
            ctrl
        })
        .expect("Failed to spawn ToneForge init thread")
        .join()
        .expect("ToneForge init thread panicked");

    let AudioStreamController {
        mixer_engine,
        app_state,
        telemetry_streamer,
        ..
    } = controller;

    let shared_engine = Arc::new(Mutex::new(mixer_engine));
    let running = Arc::new(AtomicBool::new(true));

    // Spawn Real-Time Pro Audio processing thread with Windows MMCSS "Pro Audio" registration
    // and WASAPI/ASIO planar Int32LSB hardware bridge.
    let rt_engine = Arc::clone(&shared_engine);
    let rt_running = Arc::clone(&running);
    std::thread::Builder::new()
        .name("toneforge-rt-pro-audio".to_string())
        .stack_size(16 * 1024 * 1024)
        .spawn(move || {
            let _mmcss_guard = ProAudioThreadGuard::enter();
            let mut master_out = [[0.0f32; 2]; 256];
            let mut wasapi_interleaved = [0.0f32; 512];
            let mut asio_left = [0.0f32; 256];
            let mut asio_right = [0.0f32; 256];
            let mut asio_left_i32 = [0i32; 256];
            let mut asio_right_i32 = [0i32; 256];
            let mut bridge = WasapiAsioBlockAdapter::new(256, 1024);

            while rt_running.load(Ordering::Relaxed) {
                {
                    let mut engine = rt_engine.lock();
                    engine.process_audio_block(&mut master_out);
                }
                stereo_frames_to_wasapi_interleaved(&master_out, &mut wasapi_interleaved);
                bridge.push_wasapi_interleaved(&wasapi_interleaved);
                if bridge.pop_asio_planar_block(&mut asio_left, &mut asio_right) {
                    asio_planar_f32_to_int32_lsb(
                        &asio_left,
                        &asio_right,
                        &mut asio_left_i32,
                        &mut asio_right_i32,
                    );
                    std::hint::black_box(&asio_left_i32);
                    std::hint::black_box(&asio_right_i32);
                }
                std::thread::sleep(Duration::from_micros(5_333));
            }
        })
        .expect("Failed to spawn real-time Pro Audio thread");

    let app = tauri::Builder::default()
        .manage(SharedDawController {
            app_state,
            telemetry_streamer: Mutex::new(telemetry_streamer),
            mixer_engine: shared_engine,
        })
        .invoke_handler(tauri::generate_handler![
            cmd_poll_binary_telemetry_frame,
            cmd_set_transport,
            cmd_seek_transport,
            cmd_update_proq4_band,
            cmd_update_soothe3_params,
            cmd_update_ozone12_master,
            cmd_update_bsrnn_stem_gains,
            cmd_set_track_gain,
            cmd_route_mixer_track,
            cmd_set_internal_controller_norm,
            cmd_run_uvr5_separation,
            cmd_set_ppq_resolution,
            cmd_upsert_piano_roll_pattern,
            cmd_upsert_playlist_clip,
            cmd_update_automation_spline,
            cmd_execute_offline_render,
            cmd_apply_proq4_ffp_preset,
            cmd_apply_soothe3_factory_preset,
            cmd_load_ozone12_genre_or_xml_preset,
            cmd_run_ozone12_ai_master_assistant,
            cmd_load_fl26_score,
            cmd_load_fl26_sample_kit,
            cmd_load_fl26_fx_preset,
            cmd_load_uvr5_model_profile,
        ])
        .build(tauri::generate_context!())
        .expect("Failed to build ToneForge Studio Tauri application");

    app.run(move |_app_handle, event| {
        if let RunEvent::Exit = event {
            running.store(false, Ordering::SeqCst);
            clean_webview_cache();
        }
    });
}
