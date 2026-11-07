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
    MixerEngine, ProAudioThreadGuard, StereoFrame, WasapiAsioBlockAdapter,
    asio_planar_f32_to_int32_lsb, stereo_frames_to_wasapi_interleaved,
};
use daw_tauri_host::{
    AudioStreamController, AutomationClipStateDto, DawAppState, GpuAcceleratorStatus,
    NativeLoadedAudioDto, OfflineRenderConfig, OfflineRenderResult, Ozone12MasterConfigDto,
    PatcherCanvasUpdateDto, PianoRollPatternDto, PlaylistClipDto, ProQ4BandUpdateDto,
    ProQ4GlobalUpdateDto, Soothe3ScBandUpdateDto, Soothe3UpdateDto, Uvr5SeparationRequestDto,
    Uvr5SeparationResultDto, ZeroCopyBinaryTelemetryStreamer, decode_audio_file_native,
    encode_exact_sr_pcm16_wav_bytes, pick_audio_file_dialog_native, pick_directory_dialog_native,
    resample_stereo_frames,
};
use parking_lot::Mutex;
use tauri::{RunEvent, State};

struct SharedDawController {
    app_state: DawAppState,
    telemetry_streamer: Mutex<ZeroCopyBinaryTelemetryStreamer>,
    mixer_engine: Arc<Mutex<Box<MixerEngine>>>,
    custom_audio_pcm: Arc<Mutex<Vec<StereoFrame>>>,
    custom_audio_sr: Arc<Mutex<u32>>,
    custom_audio_path: Arc<Mutex<Option<PathBuf>>>,
    custom_audio_wav_cache: Arc<Mutex<Vec<u8>>>,
    custom_audio_active: AtomicBool,
    uvr5_stem_wavs: Arc<Mutex<[Vec<u8>; 4]>>,
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
fn cmd_update_proq4_global(
    state: State<'_, SharedDawController>,
    dto: ProQ4GlobalUpdateDto,
) -> Result<(), String> {
    daw_tauri_host::cmd_update_proq4_global(&state.app_state, dto)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_update_soothe3_params(
    state: State<'_, SharedDawController>,
    dto: Soothe3UpdateDto,
) -> Result<(), String> {
    daw_tauri_host::cmd_update_soothe3_params(&state.app_state, dto)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_update_soothe3_sc_band(
    state: State<'_, SharedDawController>,
    dto: Soothe3ScBandUpdateDto,
) -> Result<(), String> {
    daw_tauri_host::cmd_update_soothe3_sc_band(&state.app_state, dto)
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
    daw_tauri_host::cmd_update_bsrnn_stem_gains(
        &state.app_state,
        track_idx,
        slot_idx,
        stem_gains_db,
    )
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_update_bsrnn_stem_state(
    state: State<'_, SharedDawController>,
    track_idx: u16,
    slot_idx: u8,
    stem_gains_db: [f32; 4],
    stem_solo: [bool; 4],
    stem_mute: [bool; 4],
) -> Result<(), String> {
    daw_tauri_host::cmd_update_bsrnn_stem_state(
        &state.app_state,
        track_idx,
        slot_idx,
        stem_gains_db,
        stem_solo,
        stem_mute,
    )
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
    let mut engine = state.mixer_engine.lock();
    daw_tauri_host::cmd_route_mixer_track_on_engine(
        &mut engine,
        &state.app_state,
        src_track,
        dst_track,
        gain,
        is_sidechain,
    )
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_set_internal_controller_norm(
    state: State<'_, SharedDawController>,
    controller_id: u16,
    norm_val: f64,
) -> Result<(), String> {
    daw_tauri_host::cmd_set_internal_controller_norm(&state.app_state, controller_id, norm_val)
}

fn build_demo_harmonic_stem() -> Vec<StereoFrame> {
    let mut demo_stem = Vec::with_capacity(4096);
    let mut lfsr: u32 = 0xACE1_u32;
    let mut pink_state = 0.0f32;
    for n in 0..4096 {
        let t = (n as f32) / 48_000.0;
        lfsr ^= lfsr << 13;
        lfsr ^= lfsr >> 17;
        lfsr ^= lfsr << 5;
        let white = ((lfsr as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        pink_state = 0.92 * pink_state + 0.08 * white;
        let bass = 0.26 * (std::f32::consts::TAU * 55.0 * t).sin()
            + 0.16 * (std::f32::consts::TAU * 110.0 * t).sin();
        let lead = 0.22 * (std::f32::consts::TAU * 440.0 * t).sin()
            + 0.14 * (std::f32::consts::TAU * 880.0 * t).cos();
        let air = 0.08 * (std::f32::consts::TAU * 3520.0 * t).sin() + 0.12 * pink_state;
        demo_stem.push([bass + lead + air, bass + lead * 0.92 - air * 0.85]);
    }
    demo_stem
}

#[allow(clippy::too_many_arguments)]
fn load_decoded_audio_into_engine(
    path: &std::path::Path,
    mixer_engine: &Arc<Mutex<Box<MixerEngine>>>,
    custom_audio_pcm: &Arc<Mutex<Vec<StereoFrame>>>,
    custom_audio_sr: &Arc<Mutex<u32>>,
    custom_audio_path: &Arc<Mutex<Option<PathBuf>>>,
    custom_audio_wav_cache: &Arc<Mutex<Vec<u8>>>,
    uvr5_stem_wavs: &Arc<Mutex<[Vec<u8>; 4]>>,
) -> Result<NativeLoadedAudioDto, String> {
    let decoded = decode_audio_file_native(path)?;
    let dto = decoded.to_dto();
    let wav_bytes = encode_exact_sr_pcm16_wav_bytes(&decoded.frames, decoded.sample_rate_hz);

    let dst_sr = {
        let engine = mixer_engine.lock();
        engine.slicer.sample_rate.max(8_000.0).round() as u32
    };
    let engine_frames = if decoded.sample_rate_hz == dst_sr {
        decoded.frames.clone()
    } else {
        resample_stereo_frames(&decoded.frames, decoded.sample_rate_hz, dst_sr)
    };

    {
        let mut engine = mixer_engine.lock();
        engine
            .replace_with_custom_audio_stem(engine_frames, true)
            .map_err(|e| e.to_string())?;
    }

    *custom_audio_pcm.lock() = decoded.frames;
    *custom_audio_sr.lock() = decoded.sample_rate_hz.max(8_000);
    *custom_audio_path.lock() = Some(decoded.file_path);
    *custom_audio_wav_cache.lock() = wav_bytes;
    *uvr5_stem_wavs.lock() = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

    Ok(dto)
}

#[tauri::command(rename_all = "camelCase")]
async fn cmd_pick_and_load_audio_native(
    state: State<'_, SharedDawController>,
    initial_dir: Option<String>,
) -> Result<Option<NativeLoadedAudioDto>, String> {
    let start_dir = initial_dir
        .filter(|s| !s.trim().is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            state
                .custom_audio_path
                .lock()
                .as_ref()
                .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        });
    let mixer_engine = Arc::clone(&state.mixer_engine);
    let custom_audio_pcm = Arc::clone(&state.custom_audio_pcm);
    let custom_audio_sr = Arc::clone(&state.custom_audio_sr);
    let custom_audio_path = Arc::clone(&state.custom_audio_path);
    let custom_audio_wav_cache = Arc::clone(&state.custom_audio_wav_cache);
    let uvr5_stem_wavs = Arc::clone(&state.uvr5_stem_wavs);

    let loaded = tauri::async_runtime::spawn_blocking(
        move || -> Result<Option<NativeLoadedAudioDto>, String> {
            let Some(picked_path) = pick_audio_file_dialog_native(start_dir.as_deref()) else {
                return Ok(None);
            };
            let dto = load_decoded_audio_into_engine(
                &picked_path,
                &mixer_engine,
                &custom_audio_pcm,
                &custom_audio_sr,
                &custom_audio_path,
                &custom_audio_wav_cache,
                &uvr5_stem_wavs,
            )?;
            Ok(Some(dto))
        },
    )
    .await
    .map_err(|e| format!("Native audio picker worker join error: {e}"))??;

    if loaded.is_some() {
        state.custom_audio_active.store(true, Ordering::SeqCst);
    }
    Ok(loaded)
}

#[tauri::command(rename_all = "camelCase")]
async fn cmd_load_audio_from_path_native(
    state: State<'_, SharedDawController>,
    file_path: String,
) -> Result<NativeLoadedAudioDto, String> {
    let trimmed = file_path.trim();
    if trimmed.is_empty() {
        return Err("file_path must not be empty".to_string());
    }
    let path = PathBuf::from(trimmed);
    let mixer_engine = Arc::clone(&state.mixer_engine);
    let custom_audio_pcm = Arc::clone(&state.custom_audio_pcm);
    let custom_audio_sr = Arc::clone(&state.custom_audio_sr);
    let custom_audio_path = Arc::clone(&state.custom_audio_path);
    let custom_audio_wav_cache = Arc::clone(&state.custom_audio_wav_cache);
    let uvr5_stem_wavs = Arc::clone(&state.uvr5_stem_wavs);

    let dto = tauri::async_runtime::spawn_blocking(move || {
        load_decoded_audio_into_engine(
            &path,
            &mixer_engine,
            &custom_audio_pcm,
            &custom_audio_sr,
            &custom_audio_path,
            &custom_audio_wav_cache,
            &uvr5_stem_wavs,
        )
    })
    .await
    .map_err(|e| format!("Native audio path loader join error: {e}"))??;

    state.custom_audio_active.store(true, Ordering::SeqCst);
    Ok(dto)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_fetch_loaded_custom_audio_wav(
    state: State<'_, SharedDawController>,
) -> Result<tauri::ipc::Response, String> {
    {
        let cached = state.custom_audio_wav_cache.lock();
        if !cached.is_empty() {
            return Ok(tauri::ipc::Response::new(cached.clone()));
        }
    }
    let pcm = state.custom_audio_pcm.lock();
    if pcm.is_empty() {
        return Err("No custom audio file is currently loaded".to_string());
    }
    let sr = (*state.custom_audio_sr.lock()).max(8_000);
    let encoded = encode_exact_sr_pcm16_wav_bytes(&pcm, sr);
    *state.custom_audio_wav_cache.lock() = encoded.clone();
    Ok(tauri::ipc::Response::new(encoded))
}

#[tauri::command(rename_all = "camelCase")]
async fn cmd_pick_directory_native(
    state: State<'_, SharedDawController>,
    title: Option<String>,
    initial_dir: Option<String>,
) -> Result<Option<String>, String> {
    let start_dir = initial_dir
        .filter(|s| !s.trim().is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            state
                .custom_audio_path
                .lock()
                .as_ref()
                .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        });
    let picked = tauri::async_runtime::spawn_blocking(move || {
        let dialog_title = title
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or("Chọn thư mục xuất file — ToneForge Studio");
        pick_directory_dialog_native(dialog_title, start_dir.as_deref())
    })
    .await
    .map_err(|e| format!("Native directory picker join error: {e}"))?;

    Ok(picked.map(|p| p.display().to_string()))
}

#[tauri::command(rename_all = "camelCase")]
async fn cmd_query_gpu_accelerator_status() -> Result<GpuAcceleratorStatus, String> {
    tauri::async_runtime::spawn_blocking(daw_tauri_host::cmd_query_gpu_accelerator_status)
        .await
        .map_err(|e| format!("GPU status probe join error: {e}"))
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_begin_custom_audio_upload(
    state: State<'_, SharedDawController>,
    sample_rate: u32,
    total_frames: usize,
    _file_name: Option<String>,
) -> Result<(), String> {
    let mut pcm = state.custom_audio_pcm.lock();
    pcm.clear();
    pcm.reserve(total_frames.min(48_000 * 60 * 25)); // reserve up to 25 minutes safely
    *state.custom_audio_sr.lock() = sample_rate.max(8_000);
    *state.custom_audio_path.lock() = None;
    state.custom_audio_wav_cache.lock().clear();
    *state.uvr5_stem_wavs.lock() = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    Ok(())
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_append_custom_audio_chunk(
    state: State<'_, SharedDawController>,
    left_chunk: Vec<f32>,
    right_chunk: Vec<f32>,
) -> Result<usize, String> {
    let n = left_chunk.len().min(right_chunk.len());
    let mut pcm = state.custom_audio_pcm.lock();
    pcm.reserve(n);
    for i in 0..n {
        pcm.push([left_chunk[i], right_chunk[i]]);
    }
    Ok(pcm.len())
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_finalize_custom_audio_upload(state: State<'_, SharedDawController>) -> Result<usize, String> {
    let src_sr = (*state.custom_audio_sr.lock()).max(8_000);
    let raw_frames = {
        let pcm = state.custom_audio_pcm.lock();
        if pcm.is_empty() {
            return Err("Uploaded custom audio buffer is empty".to_string());
        }
        pcm.clone()
    };

    let dst_sr = {
        let engine = state.mixer_engine.lock();
        engine.slicer.sample_rate.max(8_000.0).round() as u32
    };
    let engine_frames: Vec<StereoFrame> = if src_sr == dst_sr {
        raw_frames
    } else {
        let resampled = resample_stereo_frames(&raw_frames, src_sr, dst_sr);
        *state.custom_audio_pcm.lock() = resampled.clone();
        *state.custom_audio_sr.lock() = dst_sr;
        resampled
    };

    let total_len = engine_frames.len();
    {
        let mut engine = state.mixer_engine.lock();
        engine
            .replace_with_custom_audio_stem(engine_frames, true)
            .map_err(|e| e.to_string())?;
    }
    state.custom_audio_active.store(true, Ordering::SeqCst);
    Ok(total_len)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_switch_to_piano_roll_source(state: State<'_, SharedDawController>) -> Result<(), String> {
    state.custom_audio_active.store(false, Ordering::SeqCst);
    let demo_stem = build_demo_harmonic_stem();
    let mut engine = state.mixer_engine.lock();
    let engine_sr = engine.slicer.sample_rate;
    let _ = AudioStreamController::bounce_stems_to_engine(
        &mut engine,
        &[
            demo_stem.clone(),
            demo_stem.clone(),
            demo_stem.clone(),
            demo_stem,
        ],
        &[1, 2, 3, 4],
        engine_sr,
        256,
    )?;
    Ok(())
}

#[tauri::command(rename_all = "camelCase")]
async fn cmd_run_uvr5_separation(
    state: State<'_, SharedDawController>,
    req: Uvr5SeparationRequestDto,
) -> Result<Uvr5SeparationResultDto, String> {
    let loaded_path = state.custom_audio_path.lock().clone();
    let mut req_cloned = req.clone();
    let req_path = PathBuf::from(req_cloned.audio_path.trim());
    let req_is_real_file = !req_cloned.audio_path.trim().is_empty() && req_path.is_file();

    let (mix_frames, sr) = if req_is_real_file {
        let matches_loaded = loaded_path
            .as_ref()
            .map(|lp| lp == &req_path)
            .unwrap_or(false);
        if matches_loaded {
            (
                state.custom_audio_pcm.lock().clone(),
                (*state.custom_audio_sr.lock()).clamp(8_000, 384_000),
            )
        } else {
            // Decode directly from `req_cloned.audio_path` inside `execute_uvr5_separation_dsp`
            (Vec::new(), 48_000)
        }
    } else {
        if let Some(lp) = &loaded_path {
            req_cloned.audio_path = lp.display().to_string();
        }
        (
            state.custom_audio_pcm.lock().clone(),
            (*state.custom_audio_sr.lock()).clamp(8_000, 384_000),
        )
    };

    let (result_dto, stems, wav_buffers) = tauri::async_runtime::spawn_blocking(move || {
        daw_tauri_host::execute_uvr5_separation_dsp(&mix_frames, sr, &req_cloned)
    })
    .await
    .map_err(|e| format!("UVR5 separation worker join error: {e}"))??;

    {
        let mut stored_wavs = state.uvr5_stem_wavs.lock();
        *stored_wavs = wav_buffers;
    }

    if req.bounce_to_mixer {
        let engine_sr = {
            let engine = state.mixer_engine.lock();
            engine.slicer.sample_rate.max(8_000.0)
        };
        let dst_sr_hz = engine_sr.round() as u32;
        let src_sr_hz = result_dto.sample_rate_hz.max(8_000);
        let engine_stems: [Vec<StereoFrame>; 4] = if src_sr_hz == dst_sr_hz {
            stems
        } else {
            std::array::from_fn(|idx| resample_stereo_frames(&stems[idx], src_sr_hz, dst_sr_hz))
        };

        let mut engine = state.mixer_engine.lock();
        let _ = AudioStreamController::bounce_stems_to_engine(
            &mut engine,
            &engine_stems,
            &[1, 2, 3, 4],
            engine_sr,
            256,
        )?;
        if !state.custom_audio_pcm.lock().is_empty() || req_is_real_file {
            state.custom_audio_active.store(true, Ordering::SeqCst);
        }
    }

    Ok(result_dto)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_fetch_uvr5_stem_wav(
    state: State<'_, SharedDawController>,
    stem_idx: usize,
) -> Result<tauri::ipc::Response, String> {
    if stem_idx >= 4 {
        return Err(format!("stem_idx {} out of range (0..4)", stem_idx));
    }
    let wavs = state.uvr5_stem_wavs.lock();
    if wavs[stem_idx].is_empty() {
        return Err(format!("UVR5 stem {} has not been separated yet", stem_idx));
    }
    Ok(tauri::ipc::Response::new(wavs[stem_idx].clone()))
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_fetch_fl26_sample_wav(
    kit_query: Option<String>,
    sample_idx: Option<usize>,
    file_name: Option<String>,
) -> Result<tauri::ipc::Response, String> {
    if let (Some(kq), Some(idx)) = (kit_query, sample_idx) {
        let bytes = daw_tauri_host::read_fl26_sample_kit_wav_bytes(&kq, idx)?;
        return Ok(tauri::ipc::Response::new(bytes));
    }
    if let Some(fname) = file_name {
        let safe_name = std::path::Path::new(&fname)
            .file_name()
            .and_then(|s| s.to_str())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| "Invalid sample file_name".to_string())?;
        let sample_path = daw_tauri_host::resolve_assets_root()
            .join("flstudio26")
            .join("samples")
            .join(safe_name);
        let bytes = std::fs::read(&sample_path).map_err(|e| {
            format!(
                "Failed reading FL26 sample WAV '{}': {e}",
                sample_path.display()
            )
        })?;
        return Ok(tauri::ipc::Response::new(bytes));
    }
    Err("cmd_fetch_fl26_sample_wav requires (kitQuery, sampleIdx) or fileName".to_string())
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
    mut config: OfflineRenderConfig,
) -> Result<OfflineRenderResult, String> {
    if state.custom_audio_active.load(Ordering::SeqCst) {
        let custom_len = state.custom_audio_pcm.lock().len();
        let custom_sr = (*state.custom_audio_sr.lock()).max(8_000) as f64;
        if custom_len > 0 {
            let target_sr = config.sample_rate.as_f64();
            let full_song_frames =
                ((custom_len as f64) * (target_sr / custom_sr)).round() as usize;
            if full_song_frames > config.duration_frames {
                config.duration_frames = full_song_frames;
            }
        }
    }
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
    let mut engine = state.mixer_engine.lock();
    daw_tauri_host::apply_fl26_fx_preset_to_engine(
        &mut engine,
        &state.app_state,
        track_idx,
        slot_idx,
        &preset_query,
    )
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_load_uvr5_model_profile(
    profile_query: String,
) -> Result<daw_tauri_host::Uvr5ModelProfileDto, String> {
    daw_tauri_host::cmd_load_uvr5_model_profile(&profile_query)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_update_patcher_graph(
    state: State<'_, SharedDawController>,
    dto: PatcherCanvasUpdateDto,
) -> Result<usize, String> {
    let mut engine = state.mixer_engine.lock();
    daw_tauri_host::cmd_update_patcher_graph(&mut engine, &dto)
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

            // Activate Tier 1 CUDA 12 (RTX 4080) / Tier 2 DirectML background ONNX worker
            // on Master Track 0 Slot 0 (`RealTimeBsrnnStemNode`) outside the RT audio thread.
            let _ = ctrl.enable_realtime_onnx_gpu_bsrnn_on_master();

            // Seed rich harmonic test stems on Tracks 1..4 so Track 1 (Pro-Q 4 + Soothe3) and Master are live
            let demo_stem = build_demo_harmonic_stem();
            let _ = ctrl.bounce_stems_to_mixer_tracks(
                &[demo_stem.clone(), demo_stem.clone(), demo_stem.clone(), demo_stem],
                &[1, 2, 3, 4],
            );
            for trk in 1..=4 {
                let _ = ctrl.route_mixer_track(trk, 0, 0.85, false);
            }
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
            custom_audio_pcm: Arc::new(Mutex::new(Vec::new())),
            custom_audio_sr: Arc::new(Mutex::new(48_000)),
            custom_audio_path: Arc::new(Mutex::new(None)),
            custom_audio_wav_cache: Arc::new(Mutex::new(Vec::new())),
            custom_audio_active: AtomicBool::new(false),
            uvr5_stem_wavs: Arc::new(Mutex::new([
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ])),
        })
        .invoke_handler(tauri::generate_handler![
            cmd_poll_binary_telemetry_frame,
            cmd_set_transport,
            cmd_seek_transport,
            cmd_update_proq4_band,
            cmd_update_proq4_global,
            cmd_update_soothe3_params,
            cmd_update_soothe3_sc_band,
            cmd_update_ozone12_master,
            cmd_update_bsrnn_stem_gains,
            cmd_update_bsrnn_stem_state,
            cmd_set_track_gain,
            cmd_route_mixer_track,
            cmd_set_internal_controller_norm,
            cmd_pick_and_load_audio_native,
            cmd_load_audio_from_path_native,
            cmd_fetch_loaded_custom_audio_wav,
            cmd_pick_directory_native,
            cmd_query_gpu_accelerator_status,
            cmd_begin_custom_audio_upload,
            cmd_append_custom_audio_chunk,
            cmd_finalize_custom_audio_upload,
            cmd_switch_to_piano_roll_source,
            cmd_run_uvr5_separation,
            cmd_fetch_uvr5_stem_wav,
            cmd_fetch_fl26_sample_wav,
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
            cmd_update_patcher_graph,
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
