// Hide console window on Windows release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

//! ToneForge Studio v3 — Standalone Tauri v2 Desktop Application (`ToneForge_Studio.exe`).
//!
//! Connects the Svelte 5 + WebGPU 120 FPS UI (`ui-svelte5-wgpu/dist`) directly to:
//! 1. **Win32 Core Audio WASAPI + Steinberg ASIO Hardware Drivers** (`WasapiHardwareStream`,
//!    `AsioHardwareStream`, `WasapiCaptureStream` + `AdaptiveDllDriftResampler`, `Win32MidiInputPort`,
//!    and `DiskTrackRecorder` with Loop Take Comping).
//! 2. **100% Lock-Free & Zero-Allocation Real-Time Audio Graph**: Exclusive ownership of `Box<MixerEngine>`
//!    on the RT audio thread (`0` `Mutex` locks, `0` heap allocations, and `0` `drop()`/`HeapFree` calls
//!    on the RT thread via `rtrb::RingBuffer<RtGraphMutation>` and background `RtGarbageCollector`).
//! 3. **Steinberg VST3 3.7 COM & CLAP 1.2 C ABI Host** + STA Win32 `HWND` Floating Plugin Windows
//!    (`ExternalPluginGuiWorker`) + Crash-Isolated Out-of-Process Plugin Scanner.
//! 4. **64-Step Global Undo/Redo** (`ProjectUndoHistory`), **100% Pro-Q 4 / Soothe3 / Ozone 12 Parity**,
//!    and **Real-Time Pitcher + Graphical NewTone Vocal Pitch/Formant Editor** (`dsp-pitch-time`).
//! 5. **128 KB Zero-Copy Binary Telemetry Streamer** (`VisualTelemetryFrame`) feeding 120 FPS UI views.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::time::{Duration, Instant};

use daw_core_graph::{
    AdaptiveDllDriftResampler, AsioDriverInfo, AsioHardwareStream, DiskTrackRecorder,
    RtGraphMutation, StereoFrame, WasapiCaptureStream, Win32MidiInputPort, enumerate_asio_drivers,
};
use daw_plugin_host::{
    PluginScanReport, default_plugin_scan_directories, locate_scanner_executable,
    scan_plugin_directories_isolated,
};
use daw_tauri_host::{
    AudioStreamController, AutomationClipStateDto, BsrnnStemStateDto, DawAppState,
    ExternalPluginGuiWorker, GpuAcceleratorStatus, HardwareAudioMidiInventoryDto,
    HardwareRecordingStatusDto, LoadedExternalPluginInfoDto, NativeLoadedAudioDto,
    OfflineRenderConfig, OfflineRenderResult, Ozone12CodecPreviewDto, Ozone12LowEndFocusDto,
    Ozone12MasterConfigDto, PatcherCanvasUpdateDto, PianoRollPatternDto,
    PianoRollTransformRequestDto, PitcherStateDto, PlaylistClipDto, PlaylistClipEditRequestDto,
    PluginConfigDto, ProQ4BandUpdateDto, ProQ4GlobalUpdateDto, ProjectUndoHistory,
    RecordedTakeUiDto, RtGarbageCollector, Soothe3ScBandUpdateDto, Soothe3UpdateDto,
    ToneForgeProjectState, UndoRedoStatusDto, Uvr5SeparationRequestDto, Uvr5SeparationResultDto,
    VocalPitchAnalysisResultDto, WasapiEndpointInfo, WasapiHardwareStream,
    WasapiStreamStatusSnapshot, ZeroCopyBinaryTelemetryStreamer, analyze_vocal_pitch_blobs_dsp,
    commit_vocal_pitch_edits_dsp, decode_audio_file_native, encode_exact_sr_pcm16_wav_bytes,
    enumerate_wasapi_output_devices, pick_audio_file_dialog_native, pick_directory_dialog_native,
    resample_stereo_frames,
};
use dsp_pitch_time::VocalNoteBlob;
use dsp_proq4_eq::{
    CharacterMode, FilterShape, LinearPhaseResolution, ProQ4CollisionBus, ProcessingMode,
};
use dsp_soothe3_spectral::{SidechainFilterShape, SootheProcessMode, TimeResolutionQuality};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tauri::{Emitter, Manager, RunEvent, State};

mod derived_renders;
use daw_core_graph::StemClipRegion;
use daw_tauri_host::audio_assets as aa;
use derived_renders::{DerivedRenderState, PendingRender, RenderCacheKey, RenderJob};

/// Handle of the running app: background render threads use it to reach the shared controller
/// and to notify the UI (`derived-render-status`).
static APP_HANDLE: std::sync::OnceLock<tauri::AppHandle> = std::sync::OnceLock::new();

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct DerivedRenderStatusDto {
    /// Clips whose stretched / pitch-shifted render is still being computed.
    pending_clips: usize,
}

enum ActiveHardwareAudioDriver {
    Wasapi(WasapiHardwareStream),
    Asio(AsioHardwareStream),
}

impl ActiveHardwareAudioDriver {
    fn set_volume_and_mute(&self, volume_linear: f32, muted: bool) {
        match self {
            Self::Wasapi(s) => s.set_volume_and_mute(volume_linear, muted),
            Self::Asio(s) => s.set_volume_and_mute(volume_linear, muted),
        }
    }

    fn status_snapshot(&self, engine_sample_rate: f64) -> WasapiStreamStatusSnapshot {
        match self {
            Self::Wasapi(s) => s.status_snapshot(),
            Self::Asio(s) => {
                let a = s.status_snapshot();
                // ASIO drivers own their callback thread: MMCSS state is not observable from the
                // host, so it is reported as `false` rather than assumed.
                WasapiStreamStatusSnapshot {
                    stream_running: a.stream_running,
                    using_hardware: a.stream_running,
                    exclusive_mode_requested: true,
                    exclusive_mode_active: a.stream_running,
                    active_device_id: a.clsid,
                    active_device_name: format!("ASIO: {}", a.driver_name),
                    hardware_sample_rate_hz: a.sample_rate_hz,
                    engine_sample_rate_hz: engine_sample_rate.round() as u32,
                    buffer_frames: a.buffer_frames,
                    channels: a.output_channels,
                    hardware_channels: a.output_channels,
                    sample_format: "ASIO Native".to_string(),
                    mmcss_pro_audio: false,
                    mmcss_pro_audio_elevated: false,
                    resampler_active: a.sample_rate_hz != (engine_sample_rate.round() as u32),
                    xrun_count: a.xrun_count,
                    hotplug_recoveries: 0,
                    rendered_blocks: a.callback_blocks,
                    avg_block_render_us: a.avg_block_render_us,
                    speaker_volume_pct: a.speaker_volume_pct,
                    speaker_muted: a.speaker_muted,
                    status_message: a.status_message,
                }
            }
        }
    }

    fn stop_and_take_engine(self) -> Option<Box<daw_core_graph::MixerEngine>> {
        match self {
            Self::Wasapi(s) => s.stop_and_take_engine(),
            Self::Asio(s) => s.stop_and_take_engine(),
        }
    }

    fn shutdown(&mut self) {
        match self {
            Self::Wasapi(s) => s.shutdown(),
            Self::Asio(s) => s.shutdown(),
        }
    }
}

struct RecordingRuntimeMeta {
    driver_mode: String,
    active_input_name: String,
    active_midi_name: String,
    armed_track_index: u16,
    input_monitor: bool,
    hw_input_latency_samples: u32,
    /// Transport tick at which the current disk recording started (takes are placed there).
    record_start_tick: i64,
}

struct SharedDawController {
    app_state: DawAppState,
    telemetry_streamer: Mutex<ZeroCopyBinaryTelemetryStreamer>,
    audio_driver: Arc<Mutex<Option<ActiveHardwareAudioDriver>>>,
    _rt_gc: RtGarbageCollector,
    project_state: Arc<Mutex<ToneForgeProjectState>>,
    undo_history: Mutex<ProjectUndoHistory>,
    collision_bus: Arc<ProQ4CollisionBus>,
    plugin_gui_worker: ExternalPluginGuiWorker,
    capture_stream: Mutex<Option<WasapiCaptureStream>>,
    midi_input_port: Mutex<Option<Win32MidiInputPort>>,
    disk_recorder: Mutex<Option<DiskTrackRecorder>>,
    completed_takes: Mutex<Vec<RecordedTakeUiDto>>,
    recording_meta: Mutex<RecordingRuntimeMeta>,
    vocal_source_pcm: Mutex<Vec<StereoFrame>>,
    vocal_source_sr: Mutex<u32>,
    vocal_blobs: Mutex<Vec<VocalNoteBlob>>,
    engine_sample_rate: f64,
    custom_audio_pcm: Arc<Mutex<Vec<StereoFrame>>>,
    custom_audio_sr: Arc<Mutex<u32>>,
    custom_audio_path: Arc<Mutex<Option<PathBuf>>>,
    custom_audio_wav_cache: Arc<Mutex<Vec<u8>>>,
    custom_audio_active: AtomicBool,
    bounced_stems_pcm: Arc<Mutex<Option<[Vec<StereoFrame>; 4]>>>,
    active_sample_kit_query: Arc<Mutex<Option<String>>>,
    uvr5_stem_wavs: Arc<Mutex<[Vec<u8>; 4]>>,
    /// Decoded audio of every project asset at the engine rate (keyed by asset id).
    asset_cache: Mutex<HashMap<u32, Arc<Vec<StereoFrame>>>>,
    /// Stretched / pitched clip renders: cache, live channels, in-flight background renders.
    derived: Mutex<DerivedRenderState>,
    /// Transport PPQ tick mirrored from the telemetry stream (record start placement).
    transport_tick_clock: Arc<AtomicI64>,
    /// Last undo label + time: continuous gestures (fader drags, knob turns) coalesce into one step.
    undo_coalesce: Mutex<(String, Instant)>,
}

/// Window within which repeated edits with the same undo label form a single undo step.
const UNDO_COALESCE_WINDOW: Duration = Duration::from_millis(1_200);

impl SharedDawController {
    /// Captures a pre-action snapshot into the 64-step global `ProjectUndoHistory` using a fixed
    /// lock order (`project_state` -> `undo_history`) so deadlocks are structurally impossible.
    /// Consecutive edits carrying the same label within `UNDO_COALESCE_WINDOW` (one fader drag,
    /// one knob turn) are coalesced into the snapshot taken at the start of the gesture.
    fn record_undo(&self, action_label: impl Into<String>) {
        let label = action_label.into();
        {
            let mut last = self.undo_coalesce.lock();
            let now = Instant::now();
            if last.0 == label && now.duration_since(last.1) < UNDO_COALESCE_WINDOW {
                last.1 = now;
                return;
            }
            *last = (label.clone(), now);
        }
        let snapshot = self.project_state.lock().clone();
        self.undo_history.lock().record_before_action(label, &snapshot);
    }

    /// Breaks undo coalescing so the next edit always starts a new undo step.
    fn end_undo_gesture(&self) {
        self.undo_coalesce.lock().0.clear();
    }

    fn engine_sr_hz(&self) -> u32 {
        self.engine_sample_rate.round().max(8_000.0) as u32
    }

    /// Audio of `asset` at the engine rate (decoded once, then cached).
    fn asset_frames(&self, asset: &daw_tauri_host::AudioAssetDto) -> Result<Arc<Vec<StereoFrame>>, String> {
        if let Some(f) = self.asset_cache.lock().get(&asset.asset_id) {
            return Ok(Arc::clone(f));
        }
        let frames = Arc::new(daw_tauri_host::load_asset_frames(asset, self.engine_sr_hz())?);
        self.asset_cache
            .lock()
            .insert(asset.asset_id, Arc::clone(&frames));
        Ok(frames)
    }

    /// Re-synchronises every clip-driven audio player with the Playlist: region tables of
    /// recorded-take players, and the derived (stretched / pitched) clip renders.
    ///
    /// Never runs DSP on the caller's thread: an edit that keeps a clip's render key (move, gain,
    /// fades, mute) only swaps its region table; a new render key is served from the render cache
    /// or computed on a background thread (superseded renders are cancelled), while the previous
    /// render keeps playing until its replacement is ready.
    fn sync_playlist_audio(&self, ps: &ToneForgeProjectState) -> Result<(), String> {
        let sr = self.engine_sample_rate;
        for asset in ps.audio_assets.iter().filter(|a| !a.continuous) {
            let (table, count) = daw_tauri_host::asset_region_table(ps, asset.asset_id, sr);
            self.app_state
                .push_rt_mutation(RtGraphMutation::SetStemClipRegions {
                    channel_id: asset.asset_id,
                    regions: Box::new(table),
                    count,
                })?;
        }
        // Derived renders (stretched / pitched clips).
        let mut derived_clips = Vec::new();
        for clip in &ps.playlist_clips {
            if clip.kind != "AudioStem" || !daw_tauri_host::clip_needs_derived_render(clip) {
                continue;
            }
            let Some(asset) = ps
                .audio_assets
                .iter()
                .find(|a| a.asset_id == clip.source_id && !a.continuous)
            else {
                continue;
            };
            let frames = self.asset_frames(asset)?;
            let key = RenderCacheKey::new(aa::derived_render_key(clip, frames.len(), ps, sr), &frames);
            derived_clips.push((clip, frames, asset.target_track, key));
        }
        let mut jobs = Vec::new();
        {
            let mut d = self.derived.lock();
            let wanted: HashSet<u32> = derived_clips.iter().map(|(c, ..)| c.clip_id).collect();
            let stale_live: Vec<u32> = d.live.keys().copied().filter(|id| !wanted.contains(id)).collect();
            for id in stale_live {
                d.live.remove(&id);
                self.app_state.push_rt_mutation(RtGraphMutation::RemoveChannel {
                    channel_id: aa::derived_channel_id(id),
                    recycle_channel: Box::new(None),
                })?;
            }
            let stale_pending: Vec<u32> =
                d.pending.keys().copied().filter(|id| !wanted.contains(id)).collect();
            for id in stale_pending {
                d.cancel(id);
            }
            for (clip, frames, track, key) in derived_clips {
                let id = clip.clip_id;
                let regions = aa::derived_clip_region_table(clip, ps, sr);
                if d.live.get(&id) == Some(&(key, track)) {
                    // Same render: only the region table changes (microseconds, no DSP). Also
                    // cancels a render made obsolete by returning to the loaded settings.
                    d.cancel(id);
                    self.push_derived_regions(id, regions)?;
                    continue;
                }
                if d.live.contains_key(&id) {
                    // The previous render follows the clip until its replacement is ready.
                    self.push_derived_regions(id, regions.clone())?;
                }
                if d.pending.get(&id).is_some_and(|p| p.key == key && p.target_track == track) {
                    continue; // already rendering exactly this
                }
                let (generation, generation_counter) = d.bump_generation(id);
                d.pending.insert(id, PendingRender { key, target_track: track });
                let cached = d.lookup(&key);
                jobs.push(RenderJob {
                    clip_id: id,
                    key,
                    target_track: track,
                    generation,
                    generation_counter,
                    source: frames,
                    cached,
                    fallback_regions: regions,
                });
            }
            d.evict();
            Self::emit_derived_status(d.pending_count());
        }
        for job in jobs {
            self.spawn_derived_render(job);
        }
        Ok(())
    }

    fn push_derived_regions(
        &self,
        clip_id: u32,
        (table, count): (Vec<StemClipRegion>, usize),
    ) -> Result<(), String> {
        self.app_state.push_rt_mutation(RtGraphMutation::SetStemClipRegions {
            channel_id: aa::derived_channel_id(clip_id),
            regions: Box::new(table),
            count,
        })
    }

    fn emit_derived_status(pending_clips: usize) {
        if let Some(app) = APP_HANDLE.get() {
            let _ = app.emit("derived-render-status", DerivedRenderStatusDto { pending_clips });
        }
    }

    /// Phase-vocoder render of `job` (or its cached audio); `None` when superseded meanwhile.
    fn render_derived_job(job: &RenderJob) -> Option<Arc<Vec<StereoFrame>>> {
        if let Some(cached) = &job.cached {
            return Some(Arc::clone(cached));
        }
        let cancelled = || !job.is_current();
        aa::render_derived_clip_audio(&job.key.render, &job.source, &cancelled).map(Arc::new)
    }

    /// Runs `job` on its own background thread (inline only before the event loop exists).
    fn spawn_derived_render(&self, job: RenderJob) {
        let Some(app) = APP_HANDLE.get().cloned() else {
            if let Some(rendered) = Self::render_derived_job(&job) {
                self.commit_derived_render(&job, rendered, false);
            }
            return;
        };
        let clip_id = job.clip_id;
        let spawned = std::thread::Builder::new()
            .name(format!("toneforge-stretch-{clip_id}"))
            .spawn(move || {
                if let Some(rendered) = Self::render_derived_job(&job) {
                    app.state::<SharedDawController>()
                        .commit_derived_render(&job, rendered, true);
                }
            });
        if let Err(e) = spawned {
            eprintln!("[ToneForge] cannot start render thread for clip {clip_id}: {e}");
            let mut d = self.derived.lock();
            d.pending.remove(&clip_id);
            Self::emit_derived_status(d.pending_count());
        }
    }

    /// Loads a finished render into the engine unless a newer render superseded it. With
    /// `use_live_regions` the clip's current region table is read from the project (edits made
    /// while rendering), otherwise the table captured when the job was scheduled is used.
    fn commit_derived_render(&self, job: &RenderJob, rendered: Arc<Vec<StereoFrame>>, use_live_regions: bool) {
        let sr = self.engine_sample_rate;
        let ps = use_live_regions.then(|| self.project_state.lock());
        let mut d = self.derived.lock();
        if !job.is_current() {
            return;
        }
        let (table, count) = ps
            .as_ref()
            .and_then(|ps| {
                ps.playlist_clips
                    .iter()
                    .find(|c| c.clip_id == job.clip_id)
                    .filter(|c| aa::derived_render_key(c, job.source.len(), ps, sr) == job.key.render)
                    .map(|c| aa::derived_clip_region_table(c, ps, sr))
            })
            .unwrap_or_else(|| job.fallback_regions.clone());
        drop(ps);
        if job.cached.is_none() {
            d.insert(job.key, Arc::clone(&rendered), &job.source);
        }
        let ch = aa::build_derived_clip_channel_with_regions(
            job.clip_id,
            rendered,
            job.target_track,
            table,
            count,
            sr,
        );
        match self.app_state.push_rt_mutation(RtGraphMutation::UpsertChannel {
            channel: Box::new(Some(ch)),
        }) {
            Ok(()) => {
                d.live.insert(job.clip_id, (job.key, job.target_track));
            }
            Err(e) => eprintln!("[ToneForge] stretched clip {} upload failed: {e}", job.clip_id),
        }
        d.pending.remove(&job.clip_id);
        d.evict();
        Self::emit_derived_status(d.pending_count());
    }

    /// Brings live asset players in line with `new_ps.audio_assets` (after undo/redo/open project).
    fn sync_asset_players(
        &self,
        old_ps: &ToneForgeProjectState,
        new_ps: &ToneForgeProjectState,
    ) -> Result<(), String> {
        for old in &old_ps.audio_assets {
            if !new_ps.audio_assets.iter().any(|a| a.asset_id == old.asset_id) {
                self.app_state.push_rt_mutation(RtGraphMutation::RemoveChannel {
                    channel_id: old.asset_id,
                    recycle_channel: Box::new(None),
                })?;
            }
        }
        for asset in &new_ps.audio_assets {
            if old_ps.audio_assets.iter().any(|a| a == asset) {
                continue;
            }
            let frames = self.asset_frames(asset)?;
            let ch = aa::build_asset_channel_shared(asset, frames, new_ps, self.engine_sample_rate);
            self.app_state.push_rt_mutation(RtGraphMutation::UpsertChannel {
                channel: Box::new(Some(ch)),
            })?;
        }
        self.sync_playlist_audio(new_ps)
    }
}

/// Result of Undo / Redo: the history status plus the restored project so the UI can re-hydrate.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct UndoProjectSyncDto {
    status: UndoRedoStatusDto,
    project_state: ToneForgeProjectState,
}

#[tauri::command]
fn cmd_poll_binary_telemetry_frame(state: State<'_, SharedDawController>) -> tauri::ipc::Response {
    {
        let mut ps = state.project_state.lock();
        state
            .plugin_gui_worker
            .forward_gui_edits_to_rt(&mut ps, &state.app_state);
    }
    let mut streamer = state.telemetry_streamer.lock();
    let bytes = daw_tauri_host::cmd_poll_binary_telemetry_frame(&mut streamer);
    if bytes.len() >= 24 {
        let tick = i64::from_le_bytes(bytes[16..24].try_into().unwrap_or([0; 8]));
        state.transport_tick_clock.store(tick, Ordering::Relaxed);
    }
    tauri::ipc::Response::new(bytes.to_vec())
}

// ============================================================================
// GLOBAL 64-STEP UNDO / REDO COMMANDS
// ============================================================================

fn apply_undo_redo(
    state: &SharedDawController,
    redo: bool,
) -> Result<UndoProjectSyncDto, String> {
    state.end_undo_gesture();
    // Undo/redo may replace hosted plug-in instances: close their windows first.
    state.plugin_gui_worker.unregister_live_editors_blocking(None);
    let mut ps = state.project_state.lock();
    let mut hist = state.undo_history.lock();
    let transition = if redo { hist.redo(&mut ps) } else { hist.undo(&mut ps) };
    if let Some(tr) = transition {
        daw_tauri_host::sync_project_state_diff_to_rt(
            &tr.previous_state,
            &tr.restored_state,
            &state.app_state,
            Some(&state.collision_bus),
        )?;
        state.sync_asset_players(&tr.previous_state, &tr.restored_state)?;
        state.custom_audio_active.store(
            tr.restored_state.audio_assets.iter().any(|a| {
                a.kind == daw_tauri_host::KIND_CUSTOM_AUDIO
                    || a.kind == daw_tauri_host::KIND_UVR5_STEM
            }),
            Ordering::SeqCst,
        );
        if tr.previous_state.patcher_canvases != tr.restored_state.patcher_canvases {
            for canvas in &tr.restored_state.patcher_canvases {
                daw_tauri_host::cmd_update_patcher_graph_rt(
                    &state.app_state,
                    state.engine_sample_rate,
                    canvas,
                )?;
            }
        }
        if tr.previous_state.fl26_fx_presets != tr.restored_state.fl26_fx_presets {
            for p in &tr.restored_state.fl26_fx_presets {
                if tr.previous_state.fl26_fx_presets.contains(p) {
                    continue;
                }
                daw_tauri_host::apply_fl26_fx_preset_rt(
                    &state.app_state,
                    state.engine_sample_rate,
                    p.track_idx,
                    p.slot_idx,
                    &p.preset_query,
                )?;
            }
        }
        Ok(UndoProjectSyncDto {
            status: tr.status,
            project_state: ps.clone(),
        })
    } else {
        Ok(UndoProjectSyncDto {
            status: hist.status(),
            project_state: ps.clone(),
        })
    }
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_undo_project_action(
    state: State<'_, SharedDawController>,
) -> Result<UndoProjectSyncDto, String> {
    apply_undo_redo(&state, false)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_redo_project_action(
    state: State<'_, SharedDawController>,
) -> Result<UndoProjectSyncDto, String> {
    apply_undo_redo(&state, true)
}

/// Returns the authoritative project state (UI hydration on start-up / after open).
#[tauri::command(rename_all = "camelCase")]
fn cmd_get_project_state(state: State<'_, SharedDawController>) -> ToneForgeProjectState {
    state.project_state.lock().clone()
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_get_undo_redo_status(state: State<'_, SharedDawController>) -> UndoRedoStatusDto {
    state.undo_history.lock().status()
}

// ============================================================================
// WASAPI + STEINBERG ASIO HARDWARE AUDIO I/O & LIVE NOTE AUDITION COMMANDS
// ============================================================================

#[tauri::command(rename_all = "camelCase")]
async fn cmd_list_wasapi_output_devices() -> Result<Vec<WasapiEndpointInfo>, String> {
    tauri::async_runtime::spawn_blocking(enumerate_wasapi_output_devices)
        .await
        .map_err(|e| format!("WASAPI endpoint enumeration join error: {e}"))?
}

#[tauri::command(rename_all = "camelCase")]
async fn cmd_configure_wasapi_output(
    state: State<'_, SharedDawController>,
    device_id: Option<String>,
    exclusive_mode: bool,
    preferred_sample_rate_hz: Option<u32>,
    desired_buffer_frames: Option<u32>,
) -> Result<WasapiStreamStatusSnapshot, String> {
    let drv_arc = Arc::clone(&state.audio_driver);
    let engine_sr = state.engine_sample_rate;
    {
        let mut guard = drv_arc.lock();
        let needs_switch_from_asio = matches!(*guard, Some(ActiveHardwareAudioDriver::Asio(_)));
        if needs_switch_from_asio {
            if let Some(prev) = guard.take() {
                let Some(engine) = prev.stop_and_take_engine() else {
                    return Err(
                        "The audio engine could not be recovered from the ASIO driver".to_string()
                    );
                };
                let ws = WasapiHardwareStream::spawn(engine, device_id.clone(), exclusive_mode);
                ws.set_stream_preferences(preferred_sample_rate_hz, desired_buffer_frames);
                ws.reconfigure_endpoint(device_id, exclusive_mode);
                *guard = Some(ActiveHardwareAudioDriver::Wasapi(ws));
            }
        } else if let Some(ActiveHardwareAudioDriver::Wasapi(ref stream)) = *guard {
            stream.set_stream_preferences(preferred_sample_rate_hz, desired_buffer_frames);
            stream.reconfigure_endpoint(device_id, exclusive_mode);
        } else {
            return Err("Hardware audio stream is shutting down".to_string());
        }
    }
    state.recording_meta.lock().driver_mode = if exclusive_mode {
        "WASAPI Exclusive".to_string()
    } else {
        "WASAPI Shared".to_string()
    };

    tauri::async_runtime::spawn_blocking(move || {
        let deadline = Instant::now() + Duration::from_millis(250);
        while Instant::now() < deadline {
            let pending = {
                let guard = drv_arc.lock();
                match guard.as_ref() {
                    Some(ActiveHardwareAudioDriver::Wasapi(s)) => s.is_reopen_pending(),
                    _ => false,
                }
            };
            if !pending {
                std::thread::sleep(Duration::from_millis(8));
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let guard = drv_arc.lock();
        guard
            .as_ref()
            .map(|s| s.status_snapshot(engine_sr))
            .ok_or_else(|| "Hardware audio stream is shutting down".to_string())
    })
    .await
    .map_err(|e| format!("WASAPI reconfigure worker join error: {e}"))?
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_query_wasapi_status(
    state: State<'_, SharedDawController>,
) -> Result<WasapiStreamStatusSnapshot, String> {
    let guard = state.audio_driver.lock();
    guard
        .as_ref()
        .map(|s| s.status_snapshot(state.engine_sample_rate))
        .ok_or_else(|| "Hardware audio stream is shutting down".to_string())
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_set_wasapi_volume_mute(
    state: State<'_, SharedDawController>,
    volume_linear: f32,
    muted: bool,
) -> Result<(), String> {
    let guard = state.audio_driver.lock();
    if let Some(ref stream) = *guard {
        stream.set_volume_and_mute(volume_linear, muted);
        Ok(())
    } else {
        Err("Hardware audio stream is shutting down".to_string())
    }
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_trigger_audition_note(
    state: State<'_, SharedDawController>,
    key: u8,
    velocity: f32,
    fine_pitch_cents: Option<f32>,
    pan: Option<f32>,
) -> Result<(), String> {
    daw_tauri_host::cmd_trigger_audition_note(
        &state.app_state,
        key,
        velocity,
        fine_pitch_cents.unwrap_or(0.0),
        pan.unwrap_or(0.0),
    )
}

// ============================================================================
// CHẶNG C: HARDWARE ASIO + WASAPI CAPTURE DUPLEX + MIDI IN + DISK RECORDER
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ConfigureHardwareRecordingArgs {
    pub driver_mode: Option<String>, // "WASAPI_Shared" | "WASAPI_Exclusive" | "ASIO"
    pub asio_driver_name: Option<String>,
    pub wasapi_output_id: Option<String>,
    pub wasapi_input_id: Option<String>,
    pub midi_input_device_id: Option<u32>,
    pub armed_track_index: Option<u16>,
    pub input_monitor: Option<bool>,
}

#[tauri::command(rename_all = "camelCase")]
async fn cmd_list_audio_midi_inputs() -> Result<HardwareAudioMidiInventoryDto, String> {
    tauri::async_runtime::spawn_blocking(daw_tauri_host::cmd_list_audio_midi_inputs_sync)
        .await
        .map_err(|e| format!("Audio/MIDI inventory worker join error: {e}"))
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_configure_hardware_recording(
    state: State<'_, SharedDawController>,
    req: ConfigureHardwareRecordingArgs,
) -> Result<HardwareRecordingStatusDto, String> {
    let engine_sr = state.engine_sample_rate;
    let armed_track = req
        .armed_track_index
        .unwrap_or_else(|| state.recording_meta.lock().armed_track_index)
        .min(500);
    let monitor = req
        .input_monitor
        .unwrap_or_else(|| state.recording_meta.lock().input_monitor);

    // 1. Optional Output Driver Switch (WASAPI Shared / Exclusive / Steinberg ASIO)
    if let Some(ref mode) = req.driver_mode {
        let wants_asio = mode.eq_ignore_ascii_case("ASIO");
        let wants_excl = mode.contains("Exclusive") || mode.eq_ignore_ascii_case("exclusive");
        if wants_asio {
            let drivers = enumerate_asio_drivers();
            let chosen: Option<AsioDriverInfo> = if let Some(ref q) = req.asio_driver_name {
                drivers
                    .iter()
                    .find(|d| d.description.eq_ignore_ascii_case(q) || d.clsid.eq_ignore_ascii_case(q))
                    .cloned()
                    .or_else(|| drivers.first().cloned())
            } else {
                drivers.first().cloned()
            };

            if let Some(asio_drv) = chosen {
                let mut guard = state.audio_driver.lock();
                if let Some(prev) = guard.take() {
                    let Some(engine) = prev.stop_and_take_engine() else {
                        return Err(
                            "The audio engine could not be recovered from the previous driver"
                                .to_string(),
                        );
                    };
                    {
                        let asio_stream =
                            AsioHardwareStream::spawn(engine, asio_drv.clone(), engine_sr, false);
                        std::thread::sleep(Duration::from_millis(60));
                        let snap = asio_stream.status_snapshot();
                        if snap.stream_running {
                            let in_label = if snap.input_channels > 0 {
                                format!("ASIO ({} In)", snap.driver_name)
                            } else {
                                "ASIO (No Input Channels)".to_string()
                            };
                            *guard = Some(ActiveHardwareAudioDriver::Asio(asio_stream));
                            let mut m = state.recording_meta.lock();
                            m.driver_mode = format!("ASIO ({})", asio_drv.description);
                            if state.capture_stream.lock().is_none() {
                                m.active_input_name = in_label;
                            }
                        } else if let Some(mut recovered) = asio_stream.stop_and_take_engine() {
                            recovered.hw_capture_bridge = None;
                            // Automatic safe fallback to WASAPI if ASIO hardware has no active jack
                            let ws = WasapiHardwareStream::spawn(
                                recovered,
                                req.wasapi_output_id.clone(),
                                false,
                            );
                            *guard = Some(ActiveHardwareAudioDriver::Wasapi(ws));
                            state.recording_meta.lock().driver_mode = format!(
                                "WASAPI Shared (ASIO '{}' offline: {})",
                                asio_drv.description, snap.status_message
                            );
                        }
                    }
                }
            }
        } else {
            let mut guard = state.audio_driver.lock();
            if matches!(*guard, Some(ActiveHardwareAudioDriver::Asio(_))) {
                if let Some(prev) = guard.take() {
                    let Some(mut engine) = prev.stop_and_take_engine() else {
                        return Err(
                            "The audio engine could not be recovered from the ASIO driver"
                                .to_string(),
                        );
                    };
                    engine.hw_capture_bridge = None;
                    let ws = WasapiHardwareStream::spawn(
                        engine,
                        req.wasapi_output_id.clone(),
                        wants_excl,
                    );
                    *guard = Some(ActiveHardwareAudioDriver::Wasapi(ws));
                }
            } else if let Some(ActiveHardwareAudioDriver::Wasapi(ref ws)) = *guard {
                ws.reconfigure_endpoint(req.wasapi_output_id.clone(), wants_excl);
            }
            state.recording_meta.lock().driver_mode = if wants_excl {
                "WASAPI Exclusive".to_string()
            } else {
                "WASAPI Shared".to_string()
            };
        }
    }

    // 2. Optional WASAPI Capture Input + Adaptive DLL Drift Resampler
    if let Some(ref in_id) = req.wasapi_input_id {
        {
            let mut cap_guard = state.capture_stream.lock();
            *cap_guard = None;
        }
        if !in_id.eq_ignore_ascii_case("none") {
            let dev_opt = if in_id.trim().is_empty() || in_id == "default" {
                None
            } else {
                Some(in_id.clone())
            };
            let (cap_stream, rx) = WasapiCaptureStream::spawn(dev_opt, 16_384);
            // The drift resampler's nominal ratio must use the rate Windows actually negotiated.
            let capture_sr = wait_for_capture_format(&cap_stream);
            let resampler = AdaptiveDllDriftResampler::new(capture_sr as f64, engine_sr);
            state.app_state.push_rt_mutation(RtGraphMutation::AttachHwCaptureBridge {
                bridge: Box::new(Some((rx, resampler))),
            })?;
            *state.capture_stream.lock() = Some(cap_stream);
            state.recording_meta.lock().active_input_name = if in_id.is_empty() {
                "Default Microphone / Line-In".to_string()
            } else {
                in_id.clone()
            };
        } else {
            let asio_input_label = {
                let drv_guard = state.audio_driver.lock();
                match *drv_guard {
                    Some(ActiveHardwareAudioDriver::Asio(ref asio)) => {
                        let snap = asio.status_snapshot();
                        Some(if snap.input_channels > 0 {
                            format!("ASIO ({} In)", snap.driver_name)
                        } else {
                            "ASIO (No Input Channels)".to_string()
                        })
                    }
                    _ => None,
                }
            };
            if let Some(label) = asio_input_label {
                // Keep ASIO clock-locked / native duplex input active when WASAPI capture is "none".
                state.recording_meta.lock().active_input_name = label;
            } else {
                state.app_state.push_rt_mutation(RtGraphMutation::AttachHwCaptureBridge {
                    bridge: Box::new(None),
                })?;
                state.recording_meta.lock().active_input_name = "Disabled".to_string();
            }
        }
    }

    // 3. Arm Track Input & Monitoring
    let (hw_in_lat, _, _) = compute_recording_latencies_engine_frames(&state);
    state.recording_meta.lock().hw_input_latency_samples = hw_in_lat;
    state.app_state.push_rt_mutation(RtGraphMutation::SetTrackInputAndRecordArm {
        track_index: armed_track,
        input_source_bus: if monitor { Some(0) } else { None },
        record_armed: true,
        hw_input_latency_samples: hw_in_lat,
    })?;
    {
        let mut m = state.recording_meta.lock();
        m.armed_track_index = armed_track;
        m.input_monitor = monitor;
    }

    // 4. Optional Win32 MIDI Input Controller
    if let Some(midi_id) = req.midi_input_device_id {
        let mut midi_guard = state.midi_input_port.lock();
        *midi_guard = None;
        if midi_id != u32::MAX {
            let (port, midi_rx) = Win32MidiInputPort::open(
                midi_id,
                1,
                1024,
            )?;
            state.app_state.push_rt_mutation(RtGraphMutation::AttachMidiInputConsumer {
                consumer: Box::new(Some(midi_rx)),
            })?;
            *midi_guard = Some(port);
            state.recording_meta.lock().active_midi_name = format!("MIDI Input #{midi_id}");
        } else {
            state.app_state.push_rt_mutation(RtGraphMutation::AttachMidiInputConsumer {
                consumer: Box::new(None),
            })?;
            state.recording_meta.lock().active_midi_name = "None".to_string();
        }
    }

    Ok(build_hardware_recording_status(&state))
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_start_stop_disk_recording(
    state: State<'_, SharedDawController>,
    start: bool,
    track_index: Option<u16>,
    output_dir: Option<String>,
    loop_length_ticks: Option<i64>,
) -> Result<HardwareRecordingStatusDto, String> {
    let sr_hz = state.engine_sample_rate.round().max(8_000.0) as u32;
    if start {
        // Stop any existing recorder first
        if let Some(prev_rec) = state.disk_recorder.lock().take() {
            let _ = state.app_state.push_rt_mutation(RtGraphMutation::AttachRecordTap {
                tap: Box::new(None),
            });
            for t in prev_rec.finish_and_collect_takes() {
                state.completed_takes.lock().push(RecordedTakeUiDto::from(t));
            }
        }

        let armed_track = track_index
            .unwrap_or_else(|| state.recording_meta.lock().armed_track_index)
            .min(500);
        state.recording_meta.lock().armed_track_index = armed_track;

        let out_folder = output_dir
            .filter(|s| !s.trim().is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| daw_tauri_host::toneforge_user_audio_dir("Recordings"));
        let _ = std::fs::create_dir_all(&out_folder);

        let (bpm, ppq) = {
            let ps = state.project_state.lock();
            (ps.bpm.max(20.0), ps.ppq_resolution.max(96) as f64)
        };
        let loop_frames = loop_length_ticks
            .filter(|&t| t > 0)
            .map(|ticks| (((ticks as f64) / ppq) * (60.0 / bpm) * (sr_hz as f64)).round() as usize);

        let (hw_in_lat, round_trip_latency, _) = compute_recording_latencies_engine_frames(&state);
        state.recording_meta.lock().hw_input_latency_samples = hw_in_lat;

        let (recorder, tap) = DiskTrackRecorder::spawn(
            out_folder,
            armed_track,
            sr_hz,
            round_trip_latency,
            loop_frames,
            65_536,
        );
        state.app_state.push_rt_mutation(RtGraphMutation::AttachRecordTap {
            tap: Box::new(Some(tap)),
        })?;
        *state.disk_recorder.lock() = Some(recorder);

        // Ensure transport is playing while recording; takes are placed at the current tick.
        state.recording_meta.lock().record_start_tick =
            state.transport_tick_clock.load(Ordering::Relaxed).max(0);
        let _ = daw_tauri_host::cmd_set_transport(&state.app_state, true, bpm);
        state.project_state.lock().is_playing = true;
    } else {
        let _ = state.app_state.push_rt_mutation(RtGraphMutation::AttachRecordTap {
            tap: Box::new(None),
        });
        if let Some(recorder) = state.disk_recorder.lock().take() {
            let takes = recorder.finish_and_collect_takes();
            if !takes.is_empty() {
                state.end_undo_gesture();
                state.record_undo("Record Audio Takes");
                let start_tick = state.recording_meta.lock().record_start_tick;
                let mut ps = state.project_state.lock();
                let next_clip_base = ps
                    .playlist_clips
                    .iter()
                    .map(|c| c.clip_id)
                    .max()
                    .unwrap_or(10)
                    + 1;
                for (i, t) in takes.into_iter().enumerate() {
                    let ui_take = RecordedTakeUiDto::from(t);
                    // 1. Register the take WAV as a project audio asset and load it into a player
                    //    on the armed insert (so it plays live, renders offline and survives save/load).
                    let asset = daw_tauri_host::AudioAssetDto {
                        asset_id: daw_tauri_host::next_take_asset_id(&ps),
                        kind: daw_tauri_host::KIND_RECORDED_TAKE.to_string(),
                        name: format!("Take {} (Track {})", ui_take.take_number, ui_take.track_index),
                        file_path: ui_take.wav_path.clone(),
                        target_track: ui_take.track_index,
                        continuous: false,
                        loop_playback: false,
                    };
                    let frames = state.asset_frames(&asset)?;
                    let dur_ticks = ((frames.len() as f64)
                        / daw_tauri_host::audio_assets::samples_per_tick(&ps, state.engine_sample_rate))
                        .round()
                        .max(1.0) as i64;
                    daw_tauri_host::upsert_asset(&mut ps, asset.clone());
                    // 2. Place the take at the tick where recording started; alternate loop takes
                    //    stay muted on their comp lanes.
                    let clip_dto = PlaylistClipDto {
                        clip_id: next_clip_base + (i as u32),
                        track_row: ui_take.track_index,
                        kind: "AudioStem".to_string(),
                        source_id: asset.asset_id,
                        start_tick,
                        duration_ticks: dur_ticks,
                        clip_offset_ticks: 0,
                        gain: 1.0,
                        muted: i > 0,
                        source_offset_samples: 0,
                        fade_in_samples: 96,
                        fade_out_samples: 96,
                        stretch_ratio: 1.0,
                        pitch_semitones: 0.0,
                        take_lane_id: ui_take.take_lane_id,
                    };
                    daw_tauri_host::cmd_upsert_playlist_clip_rt(&mut ps, &state.app_state, &clip_dto)?;
                    let ch = daw_tauri_host::build_asset_channel(
                        &asset,
                        frames.as_ref().clone(),
                        &ps,
                        state.engine_sample_rate,
                    );
                    state.app_state.push_rt_mutation(RtGraphMutation::UpsertChannel {
                        channel: Box::new(Some(ch)),
                    })?;
                    state.completed_takes.lock().push(ui_take);
                }
                let snapshot = ps.clone();
                drop(ps);
                state.sync_playlist_audio(&snapshot)?;
            }
        }
    }

    Ok(build_hardware_recording_status(&state))
}

/// Computes `(hw_input_latency_engine_samples, round_trip_latency_engine_samples, rtt_ms)` in
/// engine samples (`state.engine_sample_rate`), keeping both ASIO (direct & SRC clock-locked) and
/// WASAPI shared capture (10 ms period + Adaptive DLL + render lookahead) unit-consistent.
fn compute_recording_latencies_engine_frames(state: &SharedDawController) -> (u32, usize, f32) {
    let eng_sr = state.engine_sample_rate.max(8_000.0);
    let eng_sr_u32 = eng_sr.round() as u32;

    let (is_asio, hw_sr, hw_buf, asio_in_hw, asio_out_hw) = {
        let guard = state.audio_driver.lock();
        match guard.as_ref() {
            Some(ActiveHardwareAudioDriver::Asio(s)) => {
                let a = s.status_snapshot();
                (
                    true,
                    a.sample_rate_hz.max(8_000),
                    a.buffer_frames.max(1),
                    a.input_latency_samples,
                    a.output_latency_samples.max(a.buffer_frames),
                )
            }
            Some(ActiveHardwareAudioDriver::Wasapi(s)) => {
                let w = s.status_snapshot();
                (
                    false,
                    w.hardware_sample_rate_hz.max(8_000),
                    w.buffer_frames.max(1),
                    0,
                    w.buffer_frames.max(1),
                )
            }
            None => (false, eng_sr_u32, 256, 0, 256),
        }
    };

    let cap_sr_opt = {
        let cap = state.capture_stream.lock();
        cap.as_ref().map(|c| c.snapshot().sample_rate_hz.max(8_000))
    };

    let out_scale = eng_sr / (hw_sr as f64);
    let out_hw_lat_engine = (asio_out_hw as f64) * out_scale
        + AdaptiveDllDriftResampler::output_src_alignment_engine_frames(hw_sr, eng_sr_u32);

    let in_lat_engine_f = if let Some(cap_sr) = cap_sr_opt {
        let wasapi_period = eng_sr * 0.010;
        let dll_delay =
            AdaptiveDllDriftResampler::dll_input_delay_engine_frames(cap_sr as f64, eng_sr);
        let lookahead = if is_asio && hw_sr == eng_sr_u32 {
            0.0
        } else {
            AdaptiveDllDriftResampler::render_lookahead_engine_frames(eng_sr_u32, hw_sr, hw_buf)
        };
        wasapi_period + dll_delay + lookahead
    } else if is_asio {
        let hw_in = (asio_in_hw as f64) * out_scale;
        let bridge_in = if hw_sr != eng_sr_u32 {
            AdaptiveDllDriftResampler::clock_locked_input_delay_engine_frames(
                hw_sr as f64,
                eng_sr,
                hw_buf as usize,
            )
        } else {
            0.0
        };
        hw_in + bridge_in
    } else {
        eng_sr * 0.010
    };

    let in_lat_engine = in_lat_engine_f.round().max(0.0) as u32;
    let rtt_engine = (in_lat_engine_f + out_hw_lat_engine).round().max(0.0) as usize;
    let rtt_ms = (rtt_engine as f32 / (eng_sr as f32)) * 1000.0;
    (in_lat_engine, rtt_engine, rtt_ms)
}

fn build_hardware_recording_status(state: &SharedDawController) -> HardwareRecordingStatusDto {
    let (_, _, rtt_ms) = compute_recording_latencies_engine_frames(state);
    let out_snap = state
        .audio_driver
        .lock()
        .as_ref()
        .map(|d| d.status_snapshot(state.engine_sample_rate));
    let m = state.recording_meta.lock();
    let rec_guard = state.disk_recorder.lock();
    let is_rec = rec_guard.is_some();
    let rec_frames = rec_guard
        .as_ref()
        .map(|r| r.total_written_frames())
        .unwrap_or(0);
    let sr = out_snap
        .as_ref()
        .map(|s| s.hardware_sample_rate_hz.max(8_000))
        .unwrap_or(state.engine_sample_rate.round() as u32);
    let buf = out_snap.as_ref().map(|s| s.buffer_frames).unwrap_or(256);

    HardwareRecordingStatusDto {
        driver_mode: m.driver_mode.clone(),
        active_output_name: out_snap
            .as_ref()
            .map(|s| s.active_device_name.clone())
            .unwrap_or_else(|| "Default Output".to_string()),
        active_input_name: m.active_input_name.clone(),
        active_midi_name: m.active_midi_name.clone(),
        sample_rate_hz: sr,
        buffer_frames: buf,
        round_trip_latency_ms: rtt_ms,
        is_recording: is_rec,
        armed_track_index: m.armed_track_index,
        recorded_frames: rec_frames,
        completed_takes: state.completed_takes.lock().clone(),
    }
}

// ============================================================================
// CHẶNG B: STEINBERG VST3 3.7 COM & CLAP 1.2 C ABI HOST + WIN32 HWND GUI
// ============================================================================

#[tauri::command(rename_all = "camelCase")]
async fn cmd_scan_external_plugins(
    extra_dirs: Option<Vec<String>>,
) -> Result<PluginScanReport, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let mut dirs = default_plugin_scan_directories();
        // Include workspace target/release and target/debug so compiled test_vst3_clap_plugin is discovered immediately
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                dirs.push(parent.to_path_buf());
            }
        }
        let manifest_target = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .map(|root| root.join("target").join("release"));
        if let Some(rel_dir) = manifest_target {
            if rel_dir.exists() {
                dirs.push(rel_dir);
            }
        }
        if let Some(extra) = extra_dirs {
            for d in extra {
                let p = PathBuf::from(d.trim());
                if p.exists() {
                    dirs.push(p);
                }
            }
        }
        let scanner_exe = locate_scanner_executable();
        Ok(scan_plugin_directories_isolated(
            &dirs,
            scanner_exe.as_deref(),
            Duration::from_millis(5_000),
        ))
    })
    .await
    .map_err(|e| format!("Plugin scanner join error: {e}"))?
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_load_external_plugin_slot(
    state: State<'_, SharedDawController>,
    track_idx: u16,
    slot_idx: u8,
    binary_path: String,
    plugin_id: Option<String>,
    is_clap: bool,
    as_channel_instrument: Option<bool>,
) -> Result<LoadedExternalPluginInfoDto, String> {
    state.record_undo(format!(
        "Load {} Plugin on Track {}",
        if is_clap { "CLAP" } else { "VST3" },
        track_idx
    ));
    let mut ps = state.project_state.lock();
    daw_tauri_host::cmd_load_external_plugin_slot_rt(
        &mut ps,
        &state.app_state,
        state.engine_sample_rate,
        track_idx,
        slot_idx,
        &binary_path,
        plugin_id.as_deref(),
        is_clap,
        as_channel_instrument.unwrap_or(false),
        Some(&state.plugin_gui_worker),
    )
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_open_external_plugin_gui(
    state: State<'_, SharedDawController>,
    track_idx: u16,
    slot_idx: u8,
    is_clap: bool,
    binary_path: String,
    plugin_id: Option<String>,
) -> Result<isize, String> {
    // Saved state chunk of the instance (used only when no live editor handle exists, e.g. after
    // undo / project load) so a fallback window opens on the exact sound that is playing.
    let state_hex = {
        let ps = state.project_state.lock();
        ps.mixer_tracks
            .get(track_idx as usize)
            .and_then(|t| {
                if slot_idx == daw_tauri_host::INSTRUMENT_SLOT_INDEX {
                    t.channel_instrument.clone()
                } else {
                    t.slots.get(slot_idx as usize).map(|s| s.plugin.clone())
                }
            })
            .and_then(|cfg| match cfg {
                PluginConfigDto::ExternalVst3(e) | PluginConfigDto::ExternalClap(e) => Some(e.state_hex),
                _ => None,
            })
            .unwrap_or_default()
    };
    state.plugin_gui_worker.open_editor(
        track_idx,
        slot_idx,
        is_clap,
        binary_path.trim().to_string(),
        plugin_id,
        state_hex,
        None,
        true,
        state.engine_sample_rate,
    )
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_close_external_plugin_gui(
    state: State<'_, SharedDawController>,
    track_idx: u16,
    slot_idx: u8,
) -> Result<(), String> {
    state.plugin_gui_worker.close_editor(track_idx, slot_idx);
    Ok(())
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_set_external_plugin_param(
    state: State<'_, SharedDawController>,
    track_idx: u16,
    slot_idx: u8,
    param_id: u32,
    norm_val: f32,
) -> Result<(), String> {
    state.record_undo("Adjust External Plugin Parameter");
    let mut ps = state.project_state.lock();
    daw_tauri_host::cmd_set_external_plugin_param_rt(
        &mut ps,
        &state.app_state,
        track_idx,
        slot_idx,
        param_id,
        norm_val,
    )?;
    state
        .plugin_gui_worker
        .sync_param(track_idx, slot_idx, param_id, norm_val);
    Ok(())
}

// ============================================================================
// CHẶNG A: 100% PRO-Q 4 / SOOTHE3 / OZONE 12 PARITY + PRO PLAYLIST/PIANO ROLL
// ============================================================================

#[tauri::command(rename_all = "camelCase")]
fn cmd_proq4_fit_eq_sketch(
    state: State<'_, SharedDawController>,
    track_idx: u16,
    slot_idx: u8,
    points_hz_db: Vec<(f32, f32)>,
    max_bands: Option<usize>,
) -> Result<Vec<ProQ4BandUpdateDto>, String> {
    state.record_undo("Pro-Q 4 EQ Sketch Curve Fit");
    let mut ps = state.project_state.lock();
    daw_tauri_host::cmd_proq4_fit_eq_sketch(
        &mut ps,
        &state.app_state,
        track_idx,
        slot_idx,
        &points_hz_db,
        max_bands.unwrap_or(6),
    )
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_proq4_set_collision_reference(
    state: State<'_, SharedDawController>,
    track_idx: u16,
    slot_idx: u8,
    reference_track: Option<u16>,
) -> Result<(), String> {
    state.record_undo("Pro-Q 4 Set Collision Reference");
    let mut ps = state.project_state.lock();
    daw_tauri_host::cmd_proq4_set_collision_reference(
        &mut ps,
        &state.app_state,
        &state.collision_bus,
        track_idx,
        slot_idx,
        reference_track,
    )
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_soothe3_set_external_sidechain(
    state: State<'_, SharedDawController>,
    track_idx: u16,
    slot_idx: u8,
    enabled: bool,
    source_track: Option<u16>,
) -> Result<(), String> {
    state.record_undo("Soothe3 External Sidechain");
    let mut ps = state.project_state.lock();
    daw_tauri_host::cmd_soothe3_set_external_sidechain(
        &mut ps,
        &state.app_state,
        track_idx,
        slot_idx,
        enabled,
        source_track,
    )
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_ozone12_set_low_end_focus(
    state: State<'_, SharedDawController>,
    track_idx: u16,
    slot_idx: u8,
    dto: Ozone12LowEndFocusDto,
) -> Result<(), String> {
    state.record_undo("Ozone 12 Low End Focus");
    let mut ps = state.project_state.lock();
    daw_tauri_host::cmd_ozone12_set_low_end_focus(
        &mut ps,
        &state.app_state,
        track_idx,
        slot_idx,
        &dto,
    )
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_ozone12_set_codec_preview(
    state: State<'_, SharedDawController>,
    track_idx: u16,
    slot_idx: u8,
    dto: Ozone12CodecPreviewDto,
) -> Result<(), String> {
    state.record_undo("Ozone 12 Codec Preview");
    let mut ps = state.project_state.lock();
    daw_tauri_host::cmd_ozone12_set_codec_preview(
        &mut ps,
        &state.app_state,
        track_idx,
        slot_idx,
        &dto,
    )
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_piano_roll_transform_notes(
    state: State<'_, SharedDawController>,
    req: PianoRollTransformRequestDto,
) -> Result<PianoRollPatternDto, String> {
    state.record_undo(format!("Piano Roll: {}", req.action));
    let mut ps = state.project_state.lock();
    daw_tauri_host::cmd_piano_roll_transform_notes_rt(&mut ps, &state.app_state, &req)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_playlist_slice_or_edit_clip(
    state: State<'_, SharedDawController>,
    req: PlaylistClipEditRequestDto,
) -> Result<Vec<PlaylistClipDto>, String> {
    state.record_undo(format!("Playlist Clip: {}", req.action));
    let mut ps = state.project_state.lock();
    let clips = daw_tauri_host::cmd_playlist_slice_or_edit_clip_rt(&mut ps, &state.app_state, &req)?;
    let snapshot = ps.clone();
    drop(ps);
    state.sync_playlist_audio(&snapshot)?;
    Ok(clips)
}

// ============================================================================
// CHẶNG D: REAL-TIME PITCHER & GRAPHICAL NEWTONE VOCAL PITCH/FORMANT EDITOR
// ============================================================================

/// Waits (up to 750 ms) for the WASAPI capture thread to report the negotiated mix format and
/// returns its sample rate.
fn wait_for_capture_format(stream: &WasapiCaptureStream) -> u32 {
    let deadline = Instant::now() + Duration::from_millis(750);
    loop {
        let snap = stream.snapshot();
        if snap.running || Instant::now() >= deadline {
            return snap.sample_rate_hz.max(8_000);
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

const NO_AUDIO_FOR_VOCAL_EDIT: &str =
    "Chưa có âm thanh giọng hát: hãy mở một file, tách UVR5 hoặc thu âm trước khi dùng NewTone.";

#[tauri::command(rename_all = "camelCase")]
async fn cmd_vocal_analyze_pitch_blobs(
    state: State<'_, SharedDawController>,
    audio_path: Option<String>,
) -> Result<VocalPitchAnalysisResultDto, String> {
    let engine_sr_hz = state.engine_sample_rate.round().max(8_000.0) as u32;
    let (frames, sr): (Vec<StereoFrame>, u32) = if let Some(ref p) = audio_path {
        let trimmed = p.trim();
        if !trimmed.is_empty() && std::path::Path::new(trimmed).is_file() {
            let decoded = decode_audio_file_native(std::path::Path::new(trimmed))?;
            (decoded.frames, decoded.sample_rate_hz.max(8_000))
        } else if let Some(ref stems) = *state.bounced_stems_pcm.lock() {
            (stems[0].clone(), engine_sr_hz)
        } else if !state.custom_audio_pcm.lock().is_empty() {
            (
                state.custom_audio_pcm.lock().clone(),
                (*state.custom_audio_sr.lock()).max(8_000),
            )
        } else {
            return Err(NO_AUDIO_FOR_VOCAL_EDIT.to_string());
        }
    } else if let Some(ref stems) = *state.bounced_stems_pcm.lock() {
        (stems[0].clone(), engine_sr_hz)
    } else if !state.custom_audio_pcm.lock().is_empty() {
        (
            state.custom_audio_pcm.lock().clone(),
            (*state.custom_audio_sr.lock()).max(8_000),
        )
    } else {
        return Err(NO_AUDIO_FOR_VOCAL_EDIT.to_string());
    };

    // Cap analysis window to 30 seconds for snappy interactive UI response
    let max_len = (sr as usize) * 30;
    let capped_frames = if frames.len() > max_len {
        frames[..max_len].to_vec()
    } else {
        frames
    };

    *state.vocal_source_pcm.lock() = capped_frames.clone();
    *state.vocal_source_sr.lock() = sr;

    let dto = tauri::async_runtime::spawn_blocking(move || {
        analyze_vocal_pitch_blobs_dsp(capped_frames, sr)
    })
    .await
    .map_err(|e| format!("Vocal pYIN worker join error: {e}"))?;

    *state.vocal_blobs.lock() = dto.blobs.clone();
    state.project_state.lock().vocal_note_blobs = dto.blobs.clone();
    Ok(dto)
}

#[tauri::command(rename_all = "camelCase")]
async fn cmd_vocal_commit_pitch_edits(
    state: State<'_, SharedDawController>,
    blobs: Vec<VocalNoteBlob>,
    target_track: Option<u16>,
) -> Result<usize, String> {
    let engine_sr_hz = state.engine_sample_rate.round().max(8_000.0) as u32;
    let (src_pcm, src_sr) = {
        let pcm = state.vocal_source_pcm.lock().clone();
        let sr = (*state.vocal_source_sr.lock()).max(8_000);
        if pcm.is_empty() {
            return Err(NO_AUDIO_FOR_VOCAL_EDIT.to_string());
        }
        (pcm, sr)
    };
    state.end_undo_gesture();
    state.record_undo("Commit NewTone Vocal Pitch Edits");

    let blobs_for_render = blobs.clone();
    let rendered = tauri::async_runtime::spawn_blocking(move || {
        let edited = commit_vocal_pitch_edits_dsp(src_pcm, src_sr, blobs_for_render);
        if src_sr == engine_sr_hz {
            edited
        } else {
            resample_stereo_frames(&edited, src_sr, engine_sr_hz)
        }
    })
    .await
    .map_err(|e| format!("Vocal commit worker join error: {e}"))?;

    let rendered_len = rendered.len();
    let dst_trk = target_track.unwrap_or(1).clamp(1, 500);
    // Persist the committed vocal as a project asset (lossless 32-bit float WAV) and play it from
    // its own player on the destination insert — other players (song, stems, takes) are untouched.
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let wav_path = daw_tauri_host::toneforge_user_audio_dir("Vocals")
        .join(format!("newtone_track{dst_trk}_{stamp}.wav"));
    daw_tauri_host::write_f32_wav_file(&wav_path, &rendered, engine_sr_hz)?;
    let asset = daw_tauri_host::AudioAssetDto {
        asset_id: daw_tauri_host::ASSET_ID_NEWTONE_BASE + dst_trk as u32,
        kind: daw_tauri_host::KIND_NEWTONE_COMMIT.to_string(),
        name: format!("NewTone Vocal (Track {dst_trk})"),
        file_path: wav_path.display().to_string(),
        target_track: dst_trk,
        continuous: true,
        loop_playback: false,
    };
    state
        .asset_cache
        .lock()
        .insert(asset.asset_id, Arc::new(rendered.clone()));
    {
        let mut ps = state.project_state.lock();
        daw_tauri_host::upsert_asset(&mut ps, asset.clone());
        let ch = daw_tauri_host::build_asset_channel(&asset, rendered, &ps, state.engine_sample_rate);
        state.app_state.push_rt_mutation(RtGraphMutation::UpsertChannel {
            channel: Box::new(Some(ch)),
        })?;
        ps.vocal_note_blobs = blobs.clone();
    }
    *state.vocal_blobs.lock() = blobs;
    Ok(rendered_len)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_update_pitcher_rt_params(
    state: State<'_, SharedDawController>,
    track_idx: u16,
    slot_idx: u8,
    cfg: PitcherStateDto,
) -> Result<(), String> {
    state.record_undo("Update Pitcher RT Parameters");
    let mut ps = state.project_state.lock();
    daw_tauri_host::cmd_update_pitcher_rt_params(
        &mut ps,
        &state.app_state,
        state.engine_sample_rate,
        track_idx,
        slot_idx,
        &cfg,
    )
}

// ============================================================================
// TRANSPORT, DSP PARAMETERS & LOCK-FREE RT GRAPH MUTATION COMMANDS
// ============================================================================

#[tauri::command(rename_all = "camelCase")]
fn cmd_set_transport(
    state: State<'_, SharedDawController>,
    playing: bool,
    bpm: f64,
) -> Result<(), String> {
    daw_tauri_host::cmd_set_transport(&state.app_state, playing, bpm)?;
    let mut ps = state.project_state.lock();
    ps.is_playing = playing;
    let tempo_changed = (ps.bpm - bpm).abs() > 1e-9;
    ps.bpm = bpm;
    if tempo_changed {
        // Audio clips are placed in ticks: their sample positions follow the new tempo.
        let snapshot = ps.clone();
        drop(ps);
        state.sync_playlist_audio(&snapshot)?;
    }
    Ok(())
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_seek_transport(
    state: State<'_, SharedDawController>,
    sample_pos: i64,
    ppq_tick: i64,
) -> Result<(), String> {
    daw_tauri_host::cmd_seek_transport(&state.app_state, sample_pos, ppq_tick)?;
    let mut ps = state.project_state.lock();
    ps.sample_pos = sample_pos;
    ps.ppq_tick = ppq_tick;
    Ok(())
}

fn mirror_proq4_band_into_project_state(ps: &mut ToneForgeProjectState, dto: ProQ4BandUpdateDto) {
    if let Some(slot_dto) = ps
        .mixer_tracks
        .get_mut(dto.track_idx as usize)
        .and_then(|t| t.slots.get_mut(dto.slot_idx as usize))
    {
        if let PluginConfigDto::ProQ4(ref mut pq) = slot_dto.plugin {
            let prev_ref = pq.collision_reference_track;
            let mut param_set = pq.to_parameter_set();
            let b_idx = (dto.band_idx as usize).min(23);
            let b = &mut param_set.bands[b_idx];
            b.used = true;
            b.enabled = true;
            b.freq_hz = dto.freq_hz.clamp(10.0, 30_000.0);
            b.gain_db = dto.gain_db.clamp(-30.0, 30.0);
            b.q_ui = dto.q.clamp(0.025, 40.0);
            b.shape = FilterShape::from_norm((dto.shape.min(9) as f32) / 9.0);
            b.slope_db_oct = (dto.slope_db_oct as f32).clamp(6.0, 96.0);
            b.is_brickwall = dto.slope_db_oct >= 96;
            b.dyn_enabled = dto.dynamic_range_db.abs() > 1e-4 || dto.spectral_enabled;
            b.dyn_range_db = dto.dynamic_range_db.clamp(-30.0, 30.0);
            b.spectral_enabled = dto.spectral_enabled;
            b.spectral_density = if dto.spectral_density <= 1.0 {
                (dto.spectral_density * 100.0).clamp(0.0, 100.0)
            } else {
                dto.spectral_density.clamp(0.0, 100.0)
            };
            param_set.sync_mirror_from_structured();
            let mut updated = daw_tauri_host::ProQ4StateDto::from_parameter_set(&param_set);
            updated.collision_reference_track = prev_ref;
            *pq = updated;
        }
    }
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_update_proq4_band(
    state: State<'_, SharedDawController>,
    dto: ProQ4BandUpdateDto,
) -> Result<(), String> {
    state.record_undo(format!("Pro-Q 4 Band {}", dto.band_idx + 1));
    daw_tauri_host::cmd_update_proq4_band(&state.app_state, dto)?;
    let mut ps = state.project_state.lock();
    mirror_proq4_band_into_project_state(&mut ps, dto);
    Ok(())
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_update_proq4_global(
    state: State<'_, SharedDawController>,
    dto: ProQ4GlobalUpdateDto,
) -> Result<(), String> {
    state.record_undo("Pro-Q 4 Global Settings");
    daw_tauri_host::cmd_update_proq4_global(&state.app_state, dto.clone())?;
    let mut ps = state.project_state.lock();
    if let Some(slot_dto) = ps
        .mixer_tracks
        .get_mut(dto.track_idx as usize)
        .and_then(|t| t.slots.get_mut(dto.slot_idx as usize))
    {
        if let PluginConfigDto::ProQ4(ref mut pq) = slot_dto.plugin {
            let prev_ref = pq.collision_reference_track;
            let mut param_set = pq.to_parameter_set();
            let fallback_mode = match dto.processing_mode {
                0 => ProcessingMode::ZeroLatency,
                1 => ProcessingMode::NaturalPhase,
                _ => ProcessingMode::LinearPhase,
            };
            param_set.global.processing_mode = match dto.phase_mode.as_deref().map(str::trim) {
                Some(m)
                    if m.eq_ignore_ascii_case("ZeroLatency")
                        || m.eq_ignore_ascii_case("zero_latency")
                        || m == "0" =>
                {
                    ProcessingMode::ZeroLatency
                }
                Some(m)
                    if m.eq_ignore_ascii_case("NaturalPhase")
                        || m.eq_ignore_ascii_case("natural_phase")
                        || m == "1" =>
                {
                    ProcessingMode::NaturalPhase
                }
                Some(m)
                    if m.eq_ignore_ascii_case("LinearPhase")
                        || m.eq_ignore_ascii_case("linear_phase")
                        || m == "2" =>
                {
                    ProcessingMode::LinearPhase
                }
                _ => fallback_mode,
            };
            param_set.global.processing_resolution = match dto.processing_resolution {
                0 => LinearPhaseResolution::Low,
                1 => LinearPhaseResolution::Medium,
                2 => LinearPhaseResolution::High,
                3 => LinearPhaseResolution::VeryHigh,
                _ => LinearPhaseResolution::Maximum,
            };
            param_set.global.character = match dto.character {
                0 => CharacterMode::Clean,
                1 => CharacterMode::Subtle,
                _ => CharacterMode::Warm,
            };
            param_set.global.gain_scale = (dto.gain_scale_pct * 0.01).clamp(0.0, 2.0);
            param_set.global.output_level_db = dto.output_level_db.clamp(-36.0, 36.0);
            param_set.global.bypass = dto.bypass;
            param_set.global.auto_gain = dto.auto_gain;
            param_set.global.show_collisions = dto.show_collisions;
            param_set.sync_mirror_from_structured();
            let mut updated = daw_tauri_host::ProQ4StateDto::from_parameter_set(&param_set);
            updated.collision_reference_track = prev_ref;
            *pq = updated;
        }
    }
    Ok(())
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_update_soothe3_params(
    state: State<'_, SharedDawController>,
    dto: Soothe3UpdateDto,
) -> Result<(), String> {
    state.record_undo("Soothe3 Parameters");
    daw_tauri_host::cmd_update_soothe3_params(&state.app_state, dto)?;
    let mut ps = state.project_state.lock();
    mirror_soothe3_into_project_state(&mut ps, &dto);
    Ok(())
}

fn mirror_soothe3_into_project_state(ps: &mut ToneForgeProjectState, dto: &Soothe3UpdateDto) {
    if let Some(slot_dto) = ps
        .mixer_tracks
        .get_mut(dto.track_idx as usize)
        .and_then(|t| t.slots.get_mut(dto.slot_idx as usize))
    {
        if let PluginConfigDto::Soothe3(ref mut s3) = slot_dto.plugin {
            let prev_sc = s3.external_sidechain_source_track;
            let mut param_set = s3.to_parameter_set();
            param_set.global.mode = if dto.hard_mode {
                SootheProcessMode::Hard
            } else {
                SootheProcessMode::Soft
            };
            param_set.global.delta_solo = dto.delta_solo;
            param_set.global.depth = dto.depth.clamp(0.0, 10.0);
            param_set.global.detail = dto.detail.clamp(0.0, 10.0);
            param_set.global.attack = dto.attack.clamp(0.0, 10.0);
            param_set.global.release = dto.release.clamp(0.0, 10.0);
            param_set.global.max_cut_db = dto.max_cut_db.clamp(-36.0, 0.0);
            param_set.global.mix_pct = dto.mix_pct.clamp(0.0, 100.0);
            param_set.global.linear_phase = dto.linear_phase;
            param_set.global.low_latency = dto.low_latency;
            param_set.global.quality = match dto.quality {
                0 => TimeResolutionQuality::Normal,
                1 => TimeResolutionQuality::High,
                _ => TimeResolutionQuality::Ultra,
            };
            param_set.sync_mirror_from_structured();
            let mut updated = daw_tauri_host::Soothe3StateDto::from_parameter_set(&param_set);
            updated.external_sidechain_source_track = prev_sc;
            *s3 = updated;
        }
    }
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_update_soothe3_sc_band(
    state: State<'_, SharedDawController>,
    dto: Soothe3ScBandUpdateDto,
) -> Result<(), String> {
    state.record_undo("Soothe3 Sidechain Band");
    daw_tauri_host::cmd_update_soothe3_sc_band(&state.app_state, dto)?;
    let mut ps = state.project_state.lock();
    if let Some(slot_dto) = ps
        .mixer_tracks
        .get_mut(dto.track_idx as usize)
        .and_then(|t| t.slots.get_mut(dto.slot_idx as usize))
    {
        if let PluginConfigDto::Soothe3(ref mut s3) = slot_dto.plugin {
            let prev_sc = s3.external_sidechain_source_track;
            let mut param_set = s3.to_parameter_set();
            let b_idx = (dto.band_idx as usize).min(7);
            let band = &mut param_set.sc_bands[b_idx];
            band.used = true;
            band.enabled = dto.enabled;
            band.shape = SidechainFilterShape::from_u8(dto.shape.min(7));
            band.freq_hz = dto.freq_hz.clamp(20.0, 20_000.0);
            band.depth = (dto.gain_db / 2.4).clamp(-10.0, 10.0);
            band.q = dto.q.clamp(0.1, 16.0);
            param_set.sync_mirror_from_structured();
            let mut updated = daw_tauri_host::Soothe3StateDto::from_parameter_set(&param_set);
            updated.external_sidechain_source_track = prev_sc;
            *s3 = updated;
        }
    }
    Ok(())
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_update_ozone12_master(
    state: State<'_, SharedDawController>,
    track_idx: u16,
    slot_idx: u8,
    cfg: Ozone12MasterConfigDto,
) -> Result<(), String> {
    state.record_undo("Ozone 12 Master Chain");
    daw_tauri_host::cmd_update_ozone12_master(&state.app_state, track_idx, slot_idx, &cfg)?;
    let mut ps = state.project_state.lock();
    if let Some(slot_dto) = ps
        .mixer_tracks
        .get_mut(track_idx as usize)
        .and_then(|t| t.slots.get_mut(slot_idx as usize))
    {
        if !matches!(slot_dto.plugin, PluginConfigDto::Empty) {
            slot_dto.plugin = PluginConfigDto::Ozone12Master(cfg);
        }
    }
    Ok(())
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_update_bsrnn_stem_gains(
    state: State<'_, SharedDawController>,
    track_idx: u16,
    slot_idx: u8,
    stem_gains_db: [f32; 4],
) -> Result<(), String> {
    state.record_undo("BSRNN Stem Rebalance");
    daw_tauri_host::cmd_update_bsrnn_stem_gains(
        &state.app_state,
        track_idx,
        slot_idx,
        stem_gains_db,
    )?;
    let mut ps = state.project_state.lock();
    if let Some(slot_dto) = ps
        .mixer_tracks
        .get_mut(track_idx as usize)
        .and_then(|t| t.slots.get_mut(slot_idx as usize))
    {
        if let PluginConfigDto::RealTimeBsrnnStem(ref mut bsrnn) = slot_dto.plugin {
            bsrnn.stem_gains_db = stem_gains_db;
        }
    }
    Ok(())
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
    state.record_undo("BSRNN Stem Solo/Mute/Gain");
    daw_tauri_host::cmd_update_bsrnn_stem_state(
        &state.app_state,
        track_idx,
        slot_idx,
        stem_gains_db,
        stem_solo,
        stem_mute,
    )?;
    let mut ps = state.project_state.lock();
    if let Some(slot_dto) = ps
        .mixer_tracks
        .get_mut(track_idx as usize)
        .and_then(|t| t.slots.get_mut(slot_idx as usize))
    {
        if !matches!(slot_dto.plugin, PluginConfigDto::Empty) {
            slot_dto.plugin = PluginConfigDto::RealTimeBsrnnStem(BsrnnStemStateDto {
                stem_gains_db,
                stem_solo,
                stem_mute,
            });
        }
    }
    Ok(())
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_set_track_gain(
    state: State<'_, SharedDawController>,
    track_idx: u16,
    gain_linear: f32,
) -> Result<(), String> {
    state.record_undo(format!("Track {} Fader", track_idx));
    daw_tauri_host::cmd_set_track_gain(&state.app_state, track_idx, gain_linear)?;
    let mut ps = state.project_state.lock();
    if let Some(trk) = ps.mixer_tracks.get_mut(track_idx as usize) {
        trk.gain = gain_linear.clamp(0.0, 4.0);
    }
    Ok(())
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_route_mixer_track(
    state: State<'_, SharedDawController>,
    src_track: u16,
    dst_track: u16,
    gain: f32,
    is_sidechain: bool,
) -> Result<(), String> {
    state.record_undo(format!("Route Track {} -> {}", src_track, dst_track));
    let mut ps = state.project_state.lock();
    daw_tauri_host::cmd_route_mixer_track_rt(
        &mut ps,
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

/// Replaces the song / UVR5-stem assets with a single `CustomAudio` asset (mirrors the live
/// `ReplaceStemChannels` mask that clears the stem players on Inserts 1..=4).
fn register_custom_audio_asset(
    state: &SharedDawController,
    file_path: &std::path::Path,
    engine_frames: &[StereoFrame],
) {
    let asset = daw_tauri_host::AudioAssetDto {
        asset_id: daw_tauri_host::ASSET_ID_CUSTOM_AUDIO,
        kind: daw_tauri_host::KIND_CUSTOM_AUDIO.to_string(),
        name: file_path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "Audio".to_string()),
        file_path: file_path.display().to_string(),
        target_track: 1,
        continuous: true,
        loop_playback: true,
    };
    let mut cache = state.asset_cache.lock();
    let mut ps = state.project_state.lock();
    ps.audio_assets.retain(|a| {
        let keep = a.kind != daw_tauri_host::KIND_CUSTOM_AUDIO && a.kind != daw_tauri_host::KIND_UVR5_STEM;
        if !keep {
            cache.remove(&a.asset_id);
        }
        keep
    });
    cache.insert(asset.asset_id, Arc::new(engine_frames.to_vec()));
    ps.audio_assets.push(asset);
}

#[allow(clippy::too_many_arguments)]
fn decode_audio_for_rt_engine(
    path: &std::path::Path,
    engine_sample_rate: f64,
    custom_audio_pcm: &Arc<Mutex<Vec<StereoFrame>>>,
    custom_audio_sr: &Arc<Mutex<u32>>,
    custom_audio_path: &Arc<Mutex<Option<PathBuf>>>,
    custom_audio_wav_cache: &Arc<Mutex<Vec<u8>>>,
    bounced_stems_pcm: &Arc<Mutex<Option<[Vec<StereoFrame>; 4]>>>,
    uvr5_stem_wavs: &Arc<Mutex<[Vec<u8>; 4]>>,
) -> Result<(NativeLoadedAudioDto, Vec<StereoFrame>), String> {
    let decoded = decode_audio_file_native(path)?;
    let dto = decoded.to_dto();
    let wav_bytes = encode_exact_sr_pcm16_wav_bytes(&decoded.frames, decoded.sample_rate_hz);

    let dst_sr = engine_sample_rate.max(8_000.0).round() as u32;
    let engine_frames = if decoded.sample_rate_hz == dst_sr {
        decoded.frames.clone()
    } else {
        resample_stereo_frames(&decoded.frames, decoded.sample_rate_hz, dst_sr)
    };

    *custom_audio_pcm.lock() = decoded.frames;
    *custom_audio_sr.lock() = decoded.sample_rate_hz.max(8_000);
    *custom_audio_path.lock() = Some(decoded.file_path);
    *custom_audio_wav_cache.lock() = wav_bytes;
    *bounced_stems_pcm.lock() = None;
    *uvr5_stem_wavs.lock() = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

    Ok((dto, engine_frames))
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
    let engine_sr = state.engine_sample_rate;
    let custom_audio_pcm = Arc::clone(&state.custom_audio_pcm);
    let custom_audio_sr = Arc::clone(&state.custom_audio_sr);
    let custom_audio_path = Arc::clone(&state.custom_audio_path);
    let custom_audio_wav_cache = Arc::clone(&state.custom_audio_wav_cache);
    let bounced_stems_pcm = Arc::clone(&state.bounced_stems_pcm);
    let uvr5_stem_wavs = Arc::clone(&state.uvr5_stem_wavs);

    let loaded = tauri::async_runtime::spawn_blocking(
        move || -> Result<Option<(NativeLoadedAudioDto, Vec<StereoFrame>)>, String> {
            let Some(picked_path) = pick_audio_file_dialog_native(start_dir.as_deref()) else {
                return Ok(None);
            };
            let pair = decode_audio_for_rt_engine(
                &picked_path,
                engine_sr,
                &custom_audio_pcm,
                &custom_audio_sr,
                &custom_audio_path,
                &custom_audio_wav_cache,
                &bounced_stems_pcm,
                &uvr5_stem_wavs,
            )?;
            Ok(Some(pair))
        },
    )
    .await
    .map_err(|e| format!("Native audio picker worker join error: {e}"))??;

    if let Some((dto, engine_frames)) = loaded {
        state.end_undo_gesture();
        state.record_undo("Load Audio File");
        if let Some(path) = state.custom_audio_path.lock().clone() {
            register_custom_audio_asset(&state, &path, &engine_frames);
        }
        daw_tauri_host::cmd_replace_custom_audio_stem_rt(
            &state.app_state,
            state.engine_sample_rate,
            engine_frames,
            true,
        )?;
        state.custom_audio_active.store(true, Ordering::SeqCst);
        Ok(Some(dto))
    } else {
        Ok(None)
    }
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
    let engine_sr = state.engine_sample_rate;
    let custom_audio_pcm = Arc::clone(&state.custom_audio_pcm);
    let custom_audio_sr = Arc::clone(&state.custom_audio_sr);
    let custom_audio_path = Arc::clone(&state.custom_audio_path);
    let custom_audio_wav_cache = Arc::clone(&state.custom_audio_wav_cache);
    let bounced_stems_pcm = Arc::clone(&state.bounced_stems_pcm);
    let uvr5_stem_wavs = Arc::clone(&state.uvr5_stem_wavs);

    let (dto, engine_frames) = tauri::async_runtime::spawn_blocking(move || {
        decode_audio_for_rt_engine(
            &path,
            engine_sr,
            &custom_audio_pcm,
            &custom_audio_sr,
            &custom_audio_path,
            &custom_audio_wav_cache,
            &bounced_stems_pcm,
            &uvr5_stem_wavs,
        )
    })
    .await
    .map_err(|e| format!("Native audio path loader join error: {e}"))??;

    state.end_undo_gesture();
    state.record_undo("Load Audio File");
    register_custom_audio_asset(&state, std::path::Path::new(trimmed), &engine_frames);
    daw_tauri_host::cmd_replace_custom_audio_stem_rt(
        &state.app_state,
        state.engine_sample_rate,
        engine_frames,
        true,
    )?;
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
    pcm.reserve(total_frames.min(48_000 * 60 * 25));
    *state.custom_audio_sr.lock() = sample_rate.max(8_000);
    *state.custom_audio_path.lock() = None;
    state.custom_audio_wav_cache.lock().clear();
    *state.bounced_stems_pcm.lock() = None;
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

    let dst_sr = state.engine_sample_rate.max(8_000.0).round() as u32;
    let engine_frames: Vec<StereoFrame> = if src_sr == dst_sr {
        raw_frames
    } else {
        let resampled = resample_stereo_frames(&raw_frames, src_sr, dst_sr);
        *state.custom_audio_pcm.lock() = resampled.clone();
        *state.custom_audio_sr.lock() = dst_sr;
        resampled
    };

    let total_len = engine_frames.len();
    *state.bounced_stems_pcm.lock() = None;
    // Browser-decoded uploads have no file on disk: persist a lossless copy so the project can be
    // rendered offline, saved and re-opened with exactly this audio.
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let wav_path = daw_tauri_host::toneforge_user_audio_dir("Imported").join(format!("upload_{stamp}.wav"));
    daw_tauri_host::write_f32_wav_file(&wav_path, &engine_frames, dst_sr)?;
    *state.custom_audio_path.lock() = Some(wav_path.clone());
    state.end_undo_gesture();
    state.record_undo("Load Audio File");
    register_custom_audio_asset(&state, &wav_path, &engine_frames);
    daw_tauri_host::cmd_replace_custom_audio_stem_rt(
        &state.app_state,
        state.engine_sample_rate,
        engine_frames,
        true,
    )?;
    state.custom_audio_active.store(true, Ordering::SeqCst);
    Ok(total_len)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_switch_to_piano_roll_source(state: State<'_, SharedDawController>) -> Result<(), String> {
    state.end_undo_gesture();
    state.record_undo("Switch To Piano Roll Source");
    state.custom_audio_active.store(false, Ordering::SeqCst);
    *state.bounced_stems_pcm.lock() = None;
    state.custom_audio_pcm.lock().clear();
    *state.custom_audio_path.lock() = None;
    state.custom_audio_wav_cache.lock().clear();
    *state.uvr5_stem_wavs.lock() = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    // Silence the song / UVR5 stem players on Inserts 1..=4 so the Piano Roll is heard alone.
    state
        .app_state
        .push_rt_mutation(RtGraphMutation::ReplaceStemChannels {
            target_tracks_mask: 0b1_1110,
            new_channels: Box::new(Vec::new()),
            retired_channels: Box::new(Vec::with_capacity(16)),
            reset_slicer: true,
        })?;
    let _ = daw_tauri_host::cmd_seek_transport(&state.app_state, 0, 0);
    let mut cache = state.asset_cache.lock();
    let mut ps = state.project_state.lock();
    ps.sample_pos = 0;
    ps.ppq_tick = 0;
    ps.audio_assets.retain(|a| {
        let keep = a.kind != daw_tauri_host::KIND_CUSTOM_AUDIO && a.kind != daw_tauri_host::KIND_UVR5_STEM;
        if !keep {
            cache.remove(&a.asset_id);
        }
        keep
    });
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
        let engine_sr = state.engine_sample_rate.max(8_000.0);
        let dst_sr_hz = engine_sr.round() as u32;
        let src_sr_hz = result_dto.sample_rate_hz.max(8_000);
        let engine_stems: [Vec<StereoFrame>; 4] = if src_sr_hz == dst_sr_hz {
            stems
        } else {
            std::array::from_fn(|idx| resample_stereo_frames(&stems[idx], src_sr_hz, dst_sr_hz))
        };

        // Persist the 4 stems (lossless) as project assets on Inserts 1..=4.
        let base_name = PathBuf::from(req.audio_path.trim())
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "mix".to_string());
        let stem_dir = daw_tauri_host::toneforge_user_audio_dir("Stems");
        let stem_names = ["Vocals", "Drums", "Bass", "Other"];
        let mut new_assets = Vec::with_capacity(4);
        for (i, stem) in engine_stems.iter().enumerate() {
            let path = stem_dir.join(format!("{base_name}_{}.wav", stem_names[i]));
            daw_tauri_host::write_f32_wav_file(&path, stem, dst_sr_hz)?;
            new_assets.push(daw_tauri_host::AudioAssetDto {
                asset_id: daw_tauri_host::ASSET_ID_UVR5_BASE + 1 + i as u32,
                kind: daw_tauri_host::KIND_UVR5_STEM.to_string(),
                name: format!("{base_name} — {}", stem_names[i]),
                file_path: path.display().to_string(),
                target_track: 1 + i as u16,
                continuous: true,
                loop_playback: true,
            });
        }
        state.end_undo_gesture();
        state.record_undo("Bounce UVR5 Stems");
        {
            let mut cache = state.asset_cache.lock();
            let mut ps = state.project_state.lock();
            ps.audio_assets.retain(|a| {
                let keep = a.kind != daw_tauri_host::KIND_CUSTOM_AUDIO
                    && a.kind != daw_tauri_host::KIND_UVR5_STEM;
                if !keep {
                    cache.remove(&a.asset_id);
                }
                keep
            });
            for (i, a) in new_assets.into_iter().enumerate() {
                cache.insert(a.asset_id, Arc::new(engine_stems[i].clone()));
                ps.audio_assets.push(a);
            }
        }
        daw_tauri_host::cmd_bounce_stems_to_mixer_rt(
            &state.app_state,
            engine_sr,
            &engine_stems,
            &[1, 2, 3, 4],
        )?;
        *state.bounced_stems_pcm.lock() = Some(engine_stems);
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
    state.record_undo("Set PPQ Resolution");
    daw_tauri_host::cmd_set_ppq_resolution(&state.app_state, ppq)?;
    let mut ps = state.project_state.lock();
    ps.ppq_resolution = ppq.clamp(daw_core_graph::MIN_PPQ, daw_core_graph::MAX_PPQ);
    let snapshot = ps.clone();
    drop(ps);
    state.sync_playlist_audio(&snapshot)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_upsert_piano_roll_pattern(
    state: State<'_, SharedDawController>,
    pattern: PianoRollPatternDto,
) -> Result<(), String> {
    state.record_undo(format!("Edit Pattern {}", pattern.pattern_id));
    let mut ps = state.project_state.lock();
    daw_tauri_host::cmd_upsert_piano_roll_pattern_rt(&mut ps, &state.app_state, &pattern)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_upsert_playlist_clip(
    state: State<'_, SharedDawController>,
    clip: PlaylistClipDto,
) -> Result<(), String> {
    state.record_undo(format!("Edit Clip {}", clip.clip_id));
    let mut ps = state.project_state.lock();
    daw_tauri_host::cmd_upsert_playlist_clip_rt(&mut ps, &state.app_state, &clip)?;
    let snapshot = ps.clone();
    drop(ps);
    state.sync_playlist_audio(&snapshot)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_update_automation_spline(
    state: State<'_, SharedDawController>,
    clip: AutomationClipStateDto,
    eval_tick: f64,
) -> Result<i32, String> {
    state.record_undo("Edit Automation Spline");
    let mut ps = state.project_state.lock();
    daw_tauri_host::cmd_update_automation_spline_rt(&mut ps, &state.app_state, &clip, eval_tick)
}

#[tauri::command(rename_all = "camelCase")]
async fn cmd_execute_offline_render(
    state: State<'_, SharedDawController>,
    mut config: OfflineRenderConfig,
) -> Result<OfflineRenderResult, String> {
    let sample_kit_q = state.active_sample_kit_query.lock().clone();
    let engine_sr_hz = state.engine_sr_hz();
    let mut snap_ps = state.project_state.lock().clone();

    // Every asset the project plays (songs, UVR5 stems, NewTone commits, recorded takes).
    let mut asset_frames: HashMap<u32, Arc<Vec<StereoFrame>>> = HashMap::new();
    for asset in &snap_ps.audio_assets {
        asset_frames.insert(asset.asset_id, state.asset_frames(asset)?);
    }
    // The render covers the whole programme: continuous media and every audio clip end.
    let target_sr_hz = config.sample_rate.as_u32();
    let to_target = |frames_at_engine: usize| -> usize {
        ((frames_at_engine as f64) * (target_sr_hz as f64) / (engine_sr_hz as f64)).round() as usize
    };
    let spt_engine = daw_tauri_host::audio_assets::samples_per_tick(&snap_ps, engine_sr_hz as f64);
    let mut programme_end = 0usize;
    for asset in snap_ps.audio_assets.iter().filter(|a| a.continuous) {
        if let Some(f) = asset_frames.get(&asset.asset_id) {
            programme_end = programme_end.max(to_target(f.len()));
        }
    }
    for clip in snap_ps.playlist_clips.iter().filter(|c| !c.muted) {
        let end_ticks = (clip.start_tick + clip.duration_ticks).max(0) as f64;
        programme_end = programme_end.max(to_target((end_ticks * spt_engine).round() as usize));
    }
    if programme_end > config.duration_frames {
        config.duration_frames = programme_end;
    }

    tauri::async_runtime::spawn_blocking(move || -> Result<OfflineRenderResult, String> {
        std::thread::Builder::new()
            .name("toneforge-offline-render".to_string())
            .stack_size(16 * 1024 * 1024)
            .spawn(move || -> Result<OfflineRenderResult, String> {
                // The offline graph is rebuilt from the same project description as the live graph.
                let target_sr = config.sample_rate.as_f64();
                snap_ps.sample_rate = target_sr;
                snap_ps.is_playing = true;
                snap_ps.sample_pos = 0;
                snap_ps.ppq_tick = 0;
                let canvases = snap_ps.patcher_canvases.clone();
                let fl26_presets = snap_ps.fl26_fx_presets.clone();

                let mut offline_ctrl = AudioStreamController::new(target_sr, 256, 0);
                offline_ctrl.apply_project_state(snap_ps)?;
                let _ = offline_ctrl.add_polyphonic_synth_channel(1, 1, 64);

                if let Some(ref kit) = sample_kit_q {
                    let _ = daw_tauri_host::apply_fl26_sample_kit_to_engine(
                        &mut offline_ctrl.mixer_engine,
                        kit,
                    );
                }
                offline_ctrl.load_project_media(|asset| {
                    let f = asset_frames
                        .get(&asset.asset_id)
                        .cloned()
                        .ok_or_else(|| format!("Audio asset {} missing", asset.asset_id))?;
                    if target_sr_hz == engine_sr_hz {
                        return Ok(f);
                    }
                    // Export at another rate: decode the original file directly at the export rate
                    // so a 96 / 192 kHz source keeps its content above the engine's 24 kHz
                    // Nyquist (the live cache `f` is band-limited to the engine rate). Only if the
                    // file cannot be read is the engine-rate cache upsampled.
                    match daw_tauri_host::load_asset_frames(asset, target_sr_hz) {
                        Ok(native) => Ok(Arc::new(native)),
                        Err(e) => {
                            eprintln!("[ToneForge] export: '{}' re-decode failed ({e}); using the engine-rate cache", asset.name);
                            Ok(Arc::new(resample_stereo_frames(&f, engine_sr_hz, target_sr_hz)))
                        }
                    }
                })?;
                for canvas in &canvases {
                    daw_tauri_host::cmd_update_patcher_graph(&mut offline_ctrl.mixer_engine, canvas)?;
                }
                for p in &fl26_presets {
                    let AudioStreamController { mixer_engine, app_state, .. } = &mut offline_ctrl;
                    daw_tauri_host::apply_fl26_fx_preset_to_engine(
                        mixer_engine,
                        app_state,
                        p.track_idx,
                        p.slot_idx,
                        &p.preset_query,
                    )?;
                }
                offline_ctrl.enable_offline_onnx_on_master()?;
                offline_ctrl.mixer_engine.is_offline_rendering = true;

                daw_tauri_host::cmd_execute_offline_render(&mut offline_ctrl.mixer_engine, &config)
            })
            .map_err(|e| format!("Failed spawning offline render thread: {e}"))?
            .join()
            .map_err(|_| "Offline render worker thread panicked".to_string())?
    })
    .await
    .map_err(|e| format!("Offline render async join error: {e}"))?
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_apply_proq4_ffp_preset(
    state: State<'_, SharedDawController>,
    track_idx: u16,
    slot_idx: u8,
    preset_query: String,
) -> Result<daw_tauri_host::ProQ4PresetStateDto, String> {
    state.record_undo("Load Pro-Q 4 Preset");
    let dto = daw_tauri_host::cmd_apply_proq4_ffp_preset(
        &state.app_state,
        track_idx,
        slot_idx,
        &preset_query,
    )?;
    if let Ok((path, name, cat)) = daw_tauri_host::resolve_proq4_ffp_path(&preset_query) {
        if let Ok(text) = std::fs::read_to_string(&path) {
            if let Ok((param_set, _)) = daw_tauri_host::parse_proq4_ffp(&text, &name, &cat) {
                let mut ps = state.project_state.lock();
                if let Some(slot_dto) = ps
                    .mixer_tracks
                    .get_mut(track_idx as usize)
                    .and_then(|t| t.slots.get_mut(slot_idx as usize))
                {
                    if !matches!(slot_dto.plugin, PluginConfigDto::Empty) {
                        slot_dto.plugin = PluginConfigDto::ProQ4(
                            daw_tauri_host::ProQ4StateDto::from_parameter_set(&param_set),
                        );
                    }
                }
            }
        }
    }
    Ok(dto)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_apply_soothe3_factory_preset(
    state: State<'_, SharedDawController>,
    track_idx: u16,
    slot_idx: u8,
    preset_query: String,
) -> Result<daw_tauri_host::Soothe3PresetStateDto, String> {
    state.record_undo("Load Soothe3 Preset");
    let dto = daw_tauri_host::cmd_apply_soothe3_factory_preset(
        &state.app_state,
        track_idx,
        slot_idx,
        &preset_query,
    )?;
    if let Ok(path) = daw_tauri_host::resolve_soothe3_preset_path(&preset_query) {
        if let Ok(json_str) = std::fs::read_to_string(&path) {
            if let Ok((param_set, _)) = daw_tauri_host::parse_soothe3_preset(&json_str, &preset_query)
            {
                let mut ps = state.project_state.lock();
                if let Some(slot_dto) = ps
                    .mixer_tracks
                    .get_mut(track_idx as usize)
                    .and_then(|t| t.slots.get_mut(slot_idx as usize))
                {
                    if !matches!(slot_dto.plugin, PluginConfigDto::Empty) {
                        slot_dto.plugin = PluginConfigDto::Soothe3(
                            daw_tauri_host::Soothe3StateDto::from_parameter_set(&param_set),
                        );
                    }
                }
            }
        }
    }
    Ok(dto)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_load_ozone12_genre_or_xml_preset(
    state: State<'_, SharedDawController>,
    track_idx: u16,
    slot_idx: u8,
    query: String,
    is_xml_preset: bool,
) -> Result<daw_tauri_host::Ozone12AssistantPresetDto, String> {
    state.record_undo("Load Ozone 12 Preset");
    let dto = daw_tauri_host::cmd_load_ozone12_genre_or_xml_preset(
        &state.app_state,
        track_idx,
        slot_idx,
        &query,
        is_xml_preset,
    )?;
    let mut ps = state.project_state.lock();
    if let Some(trk) = ps.mixer_tracks.get_mut(track_idx as usize) {
        if let Some(slot_dto) = trk.slots.get_mut(slot_idx as usize) {
            if !matches!(slot_dto.plugin, PluginConfigDto::Empty) {
                slot_dto.plugin = PluginConfigDto::Ozone12Master(dto.master_config.clone());
            }
        }
        let bsrnn_slot = if slot_idx == 0 { 1usize } else { 0usize };
        if let Some(PluginConfigDto::RealTimeBsrnnStem(bsrnn)) =
            trk.slots.get_mut(bsrnn_slot).map(|s| &mut s.plugin)
        {
            bsrnn.stem_gains_db = dto.bsrnn_stem_gains_db;
        }
    }
    Ok(dto)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_run_ozone12_ai_master_assistant(
    state: State<'_, SharedDawController>,
    track_idx: u16,
    slot_idx: u8,
    mel_80: Option<Vec<f32>>,
) -> Result<daw_tauri_host::Ozone12AssistantPresetDto, String> {
    let (analysis_audio, analysis_sr): (Vec<StereoFrame>, u32) = {
        let custom_pcm = state.custom_audio_pcm.lock();
        if !custom_pcm.is_empty() {
            (
                custom_pcm.clone(),
                (*state.custom_audio_sr.lock()).max(8_000),
            )
        } else if let Some(ref stems) = *state.bounced_stems_pcm.lock() {
            let n = stems[0].len();
            let mut summed = vec![[0.0f32; 2]; n];
            for s in stems {
                for (i, f) in s.iter().enumerate().take(n) {
                    summed[i][0] += f[0];
                    summed[i][1] += f[1];
                }
            }
            (summed, state.engine_sample_rate.round().max(8_000.0) as u32)
        } else {
            return Err(
                "AI Master Assistant cần âm thanh thật để phân tích: hãy mở một bài nhạc trước."
                    .to_string(),
            );
        }
    };
    state.end_undo_gesture();
    state.record_undo("Ozone 12 AI Master Assistant");

    let dto = daw_tauri_host::run_ozone12_master_assistant_with_audio(
        &state.app_state,
        track_idx,
        slot_idx,
        Some((&analysis_audio, analysis_sr)),
        mel_80,
    )?;

    let mut ps = state.project_state.lock();
    if let Some(trk) = ps.mixer_tracks.get_mut(track_idx as usize) {
        if let Some(slot_dto) = trk.slots.get_mut(slot_idx as usize) {
            if !matches!(slot_dto.plugin, PluginConfigDto::Empty) {
                slot_dto.plugin = PluginConfigDto::Ozone12Master(dto.master_config.clone());
            }
        }
        let bsrnn_slot = if slot_idx == 0 { 1usize } else { 0usize };
        if let Some(PluginConfigDto::RealTimeBsrnnStem(bsrnn)) =
            trk.slots.get_mut(bsrnn_slot).map(|s| &mut s.plugin)
        {
            bsrnn.stem_gains_db = dto.bsrnn_stem_gains_db;
        }
    }
    for b in &dto.corrective_eq_bands {
        mirror_proq4_band_into_project_state(
            &mut ps,
            ProQ4BandUpdateDto {
                track_idx: 1,
                slot_idx: 0,
                band_idx: b.idx,
                freq_hz: b.freq_hz,
                gain_db: b.gain_db,
                q: b.q,
                shape: b.shape,
                slope_db_oct: b.slope_db_oct,
                dynamic_range_db: b.dynamic_range_db,
                spectral_enabled: b.spectral_enabled,
                spectral_density: b.spectral_density,
            },
        );
    }
    if let Some(s3) = dto.applied_soothe3 {
        mirror_soothe3_into_project_state(&mut ps, &s3);
    }
    Ok(dto)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_load_fl26_score(
    state: State<'_, SharedDawController>,
    score_query: String,
    target_ppq: u32,
) -> Result<PianoRollPatternDto, String> {
    state.record_undo("Load FL26 Piano Roll Score");
    let mut ps = state.project_state.lock();
    daw_tauri_host::load_fl26_score_preset_rt(&mut ps, &state.app_state, &score_query, target_ppq)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_load_fl26_sample_kit(
    state: State<'_, SharedDawController>,
    kit_query: String,
) -> Result<daw_tauri_host::Fl26SampleKitDto, String> {
    let dto = daw_tauri_host::apply_fl26_sample_kit_rt(
        &state.app_state,
        state.engine_sample_rate,
        &kit_query,
    )?;
    *state.active_sample_kit_query.lock() = Some(kit_query);
    Ok(dto)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_load_fl26_fx_preset(
    state: State<'_, SharedDawController>,
    track_idx: u16,
    slot_idx: u8,
    preset_query: String,
) -> Result<daw_tauri_host::Fl26FxPresetDto, String> {
    state.end_undo_gesture();
    state.record_undo("Load FL26 FX Preset");
    state
        .plugin_gui_worker
        .unregister_live_editors_blocking(Some((track_idx, slot_idx)));
    let mut ps = state.project_state.lock();
    daw_tauri_host::apply_fl26_fx_preset_slot_rt(
        &mut ps,
        &state.app_state,
        state.engine_sample_rate,
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
    state.record_undo("Update Patcher Graph");
    let mut ps = state.project_state.lock();
    daw_tauri_host::apply_patcher_canvas_slot_rt(&mut ps, &state.app_state, state.engine_sample_rate, dto)
}

// ============================================================================
// MIXER STRIP CONTROLS (MUTE / SOLO / PAN / FX SLOT BYPASS, WET, REMOVE), TELEMETRY FOCUS
// ============================================================================

fn with_track<R>(
    state: &SharedDawController,
    track_idx: u16,
    f: impl FnOnce(&mut daw_tauri_host::MixerTrackStateDto) -> R,
) -> Result<R, String> {
    let mut ps = state.project_state.lock();
    ps.mixer_tracks
        .get_mut(track_idx as usize)
        .map(f)
        .ok_or_else(|| format!("Mixer track {track_idx} out of range"))
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_set_track_mute(state: State<'_, SharedDawController>, track_idx: u16, muted: bool) -> Result<(), String> {
    state.end_undo_gesture();
    state.record_undo(format!("Track {track_idx} Mute"));
    state.app_state.push_command(daw_core_graph::EngineCommand::SetTrackMute { track_index: track_idx, muted })?;
    with_track(&state, track_idx, |t| t.muted = muted)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_set_track_solo(state: State<'_, SharedDawController>, track_idx: u16, soloed: bool) -> Result<(), String> {
    state.end_undo_gesture();
    state.record_undo(format!("Track {track_idx} Solo"));
    state.app_state.push_command(daw_core_graph::EngineCommand::SetTrackSolo { track_index: track_idx, soloed })?;
    with_track(&state, track_idx, |t| t.soloed = soloed && track_idx != 0)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_set_track_pan(state: State<'_, SharedDawController>, track_idx: u16, pan: f32) -> Result<(), String> {
    let pan = if pan.is_finite() { pan.clamp(-1.0, 1.0) } else { 0.0 };
    state.record_undo(format!("Track {track_idx} Pan"));
    state.app_state.push_command(daw_core_graph::EngineCommand::SetTrackPan { track_index: track_idx, pan })?;
    with_track(&state, track_idx, |t| t.pan = pan)
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_set_slot_enabled(
    state: State<'_, SharedDawController>,
    track_idx: u16,
    slot_idx: u8,
    enabled: bool,
) -> Result<(), String> {
    state.end_undo_gesture();
    state.record_undo(format!("Track {track_idx} Slot {} Bypass", slot_idx + 1));
    state.app_state.push_command(daw_core_graph::EngineCommand::SetSlotEnabled {
        track_index: track_idx,
        slot_index: slot_idx,
        enabled,
    })?;
    with_track(&state, track_idx, |t| {
        if let Some(s) = t.slots.get_mut(slot_idx as usize) {
            s.enabled = enabled;
        }
    })
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_set_slot_wet_mix(
    state: State<'_, SharedDawController>,
    track_idx: u16,
    slot_idx: u8,
    wet_mix: f32,
) -> Result<(), String> {
    let wet = wet_mix.clamp(0.0, 1.0);
    state.record_undo(format!("Track {track_idx} Slot {} Mix", slot_idx + 1));
    state.app_state.push_command(daw_core_graph::EngineCommand::SetSlotWetMix {
        track_index: track_idx,
        slot_index: slot_idx,
        wet_mix: wet,
    })?;
    with_track(&state, track_idx, |t| {
        if let Some(s) = t.slots.get_mut(slot_idx as usize) {
            s.wet_mix = wet;
        }
    })
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_remove_plugin_slot(
    state: State<'_, SharedDawController>,
    track_idx: u16,
    slot_idx: u8,
) -> Result<Vec<daw_tauri_host::PluginSlotStateDto>, String> {
    state.end_undo_gesture();
    state.record_undo(format!("Track {track_idx} Remove Slot {}", slot_idx + 1));
    // Close the plug-in window before its instance is retired off the RT thread.
    state
        .plugin_gui_worker
        .unregister_live_editors_blocking(Some((track_idx, slot_idx)));
    // Fixed FL-style slot positions: the slot becomes Empty in place (later slots keep their
    // indices, so presets / Patcher canvases / controller links / editors stay on the right plug-in),
    // then trailing Empty slots are trimmed from the end.
    let mut ps = state.project_state.lock();
    daw_tauri_host::clear_plugin_slot(
        &mut ps,
        &state.app_state,
        state.engine_sample_rate,
        track_idx,
        slot_idx,
    )
}

/// Inserts (or replaces) a native ToneForge plug-in (`kind` = `PluginConfigDto` variant name, e.g.
/// `ProQ4`, `NativeLuxeVerbFdn`) at exactly `slot_idx`, padding any gap with Empty slots.
#[tauri::command(rename_all = "camelCase")]
fn cmd_insert_native_plugin_slot(
    state: State<'_, SharedDawController>,
    track_idx: u16,
    slot_idx: u8,
    kind: String,
) -> Result<Vec<daw_tauri_host::PluginSlotStateDto>, String> {
    daw_tauri_host::default_native_plugin_dto(&kind)?;
    state.end_undo_gesture();
    state.record_undo(format!("Track {track_idx} Insert {kind} in Slot {}", slot_idx + 1));
    // The previous occupant (if any) is retired: close its editor window first.
    state
        .plugin_gui_worker
        .unregister_live_editors_blocking(Some((track_idx, slot_idx)));
    let mut ps = state.project_state.lock();
    daw_tauri_host::insert_native_plugin_slot(
        &mut ps,
        &state.app_state,
        state.engine_sample_rate,
        track_idx,
        slot_idx,
        &kind,
        Some(&state.collision_bus),
    )
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_set_telemetry_focus_track(state: State<'_, SharedDawController>, track_idx: u16) -> Result<(), String> {
    state
        .app_state
        .push_command(daw_core_graph::EngineCommand::SetTelemetryFocusTrack(track_idx))
}

// ============================================================================
// PROJECT SAVE / OPEN (`.tfproj` JSON + referenced audio assets)
// ============================================================================

#[tauri::command(rename_all = "camelCase")]
async fn cmd_save_project(
    state: State<'_, SharedDawController>,
    file_path: Option<String>,
) -> Result<Option<String>, String> {
    let (json, name) = {
        let ps = state.project_state.lock();
        (ps.to_json_pretty()?, ps.project_name.clone())
    };
    let target = match file_path.filter(|p| !p.trim().is_empty()) {
        Some(p) => Some(PathBuf::from(p.trim())),
        None => {
            let default = format!("{}.tfproj", name.replace(['\\', '/', ':', '*', '?', '"', '<', '>', '|'], "_"));
            tauri::async_runtime::spawn_blocking(move || {
                daw_tauri_host::pick_project_save_path_native(&default, None)
            })
            .await
            .map_err(|e| format!("Save dialog join error: {e}"))?
        }
    };
    let Some(path) = target else {
        return Ok(None);
    };
    std::fs::write(&path, json).map_err(|e| format!("Failed to save project {}: {e}", path.display()))?;
    Ok(Some(path.display().to_string()))
}

#[tauri::command(rename_all = "camelCase")]
async fn cmd_open_project(
    state: State<'_, SharedDawController>,
    file_path: Option<String>,
) -> Result<Option<ToneForgeProjectState>, String> {
    let source = match file_path.filter(|p| !p.trim().is_empty()) {
        Some(p) => Some(PathBuf::from(p.trim())),
        None => tauri::async_runtime::spawn_blocking(|| daw_tauri_host::pick_project_open_path_native(None))
            .await
            .map_err(|e| format!("Open dialog join error: {e}"))?,
    };
    let Some(path) = source else {
        return Ok(None);
    };
    let json = std::fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read project {}: {e}", path.display()))?;
    let mut loaded = ToneForgeProjectState::from_json_str(&json)?;
    daw_tauri_host::pad_project_slots_for_overlays(&mut loaded);
    loaded.sample_rate = state.engine_sample_rate;
    loaded.is_playing = false;

    // Verify every referenced audio file before touching the live graph.
    for asset in &loaded.audio_assets {
        if !std::path::Path::new(&asset.file_path).is_file() {
            return Err(format!(
                "Project references missing audio '{}' ({})",
                asset.name, asset.file_path
            ));
        }
    }

    state.end_undo_gesture();
    state.record_undo("Open Project");
    state.plugin_gui_worker.unregister_live_editors_blocking(None);
    let old = state.project_state.lock().clone();
    state.asset_cache.lock().clear();
    daw_tauri_host::cmd_set_transport(&state.app_state, false, loaded.bpm)?;
    daw_tauri_host::sync_project_state_diff_to_rt(&old, &loaded, &state.app_state, Some(&state.collision_bus))?;
    state.sync_asset_players(&ToneForgeProjectState { audio_assets: Vec::new(), ..old.clone() }, &loaded)?;
    for asset in &old.audio_assets {
        if !loaded.audio_assets.iter().any(|a| a.asset_id == asset.asset_id) {
            state.app_state.push_rt_mutation(RtGraphMutation::RemoveChannel {
                channel_id: asset.asset_id,
                recycle_channel: Box::new(None),
            })?;
        }
    }
    for canvas in &loaded.patcher_canvases {
        daw_tauri_host::cmd_update_patcher_graph_rt(&state.app_state, state.engine_sample_rate, canvas)?;
    }
    for p in loaded.fl26_fx_presets.clone() {
        let applied = daw_tauri_host::apply_fl26_fx_preset_rt(
            &state.app_state,
            state.engine_sample_rate,
            p.track_idx,
            p.slot_idx,
            &p.preset_query,
        )?;
        daw_tauri_host::mirror_fl26_preset_into_slot(&mut loaded, p.track_idx, p.slot_idx, &applied)?;
    }
    *state.project_state.lock() = loaded.clone();
    state.custom_audio_active.store(
        loaded.audio_assets.iter().any(|a| a.continuous),
        Ordering::SeqCst,
    );
    Ok(Some(loaded))
}

#[tauri::command(rename_all = "camelCase")]
fn cmd_new_project(state: State<'_, SharedDawController>) -> Result<ToneForgeProjectState, String> {
    let blank = ToneForgeProjectState::new_blank_project(state.engine_sample_rate, 128.0);
    state.end_undo_gesture();
    state.record_undo("New Blank Project");
    state.plugin_gui_worker.unregister_live_editors_blocking(None);
    let old = state.project_state.lock().clone();
    state.asset_cache.lock().clear();
    state.completed_takes.lock().clear();
    state.vocal_source_pcm.lock().clear();
    state.vocal_blobs.lock().clear();
    state.custom_audio_pcm.lock().clear();
    *state.custom_audio_path.lock() = None;
    state.custom_audio_wav_cache.lock().clear();
    *state.bounced_stems_pcm.lock() = None;
    daw_tauri_host::cmd_set_transport(&state.app_state, false, blank.bpm)?;
    daw_tauri_host::cmd_seek_transport(&state.app_state, 0, 0)?;
    daw_tauri_host::sync_project_state_diff_to_rt(
        &old,
        &blank,
        &state.app_state,
        Some(&state.collision_bus),
    )?;
    state.sync_asset_players(&old, &blank)?;
    *state.active_sample_kit_query.lock() = None;
    state.custom_audio_active.store(false, Ordering::SeqCst);
    *state.project_state.lock() = blank.clone();
    Ok(blank)
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

    let (controller, rt_gc) = std::thread::Builder::new()
        .name("toneforge-init".to_string())
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            let (mut ctrl, gc_worker) =
                AudioStreamController::with_blank_project_rt_gc(48_000.0, 256)
                    .expect("Failed to initialize ToneForge Blank Session");
            let _ = ctrl.add_polyphonic_synth_channel(1, 1, 64);

            // Activate Tier 1 CUDA 12 / Tier 2 DirectML background ONNX workers outside RT thread
            let _ = ctrl.enable_realtime_onnx_gpu_bsrnn_on_master();
            let _ = ctrl.enable_realtime_onnx_delimit_on_master();

            // Start with an empty, stopped session: no synthetic audio is ever injected; the user
            // opens a song, records, or plays the Piano Roll.
            ctrl.project_state.is_playing = false;
            ctrl.mixer_engine.is_playing = false;
            (ctrl, gc_worker)
        })
        .expect("Failed to spawn ToneForge init thread")
        .join()
        .expect("ToneForge init thread panicked");

    let AudioStreamController {
        mixer_engine,
        app_state,
        telemetry_streamer,
        project_state,
        undo_history,
        collision_bus,
        sample_rate,
        ..
    } = controller;

    // Transfer 100% exclusive ownership of `Box<MixerEngine>` to the Win32 Core Audio WASAPI
    // Shared Event-Driven hardware thread (`MMCSS "Pro Audio"`, 0 Mutex locks, 0 RT allocs/drops).
    let audio_driver = Arc::new(Mutex::new(Some(ActiveHardwareAudioDriver::Wasapi(
        WasapiHardwareStream::spawn(mixer_engine, None, false),
    ))));
    let audio_driver_exit = Arc::clone(&audio_driver);

    // Plug-in instances retired by the audio thread are destroyed on the plug-in GUI thread.
    let plugin_gui_worker = ExternalPluginGuiWorker::spawn();
    rt_gc.set_plugin_disposer(Some(plugin_gui_worker.plugin_disposer()));

    let app = tauri::Builder::default()
        .manage(SharedDawController {
            app_state,
            telemetry_streamer: Mutex::new(telemetry_streamer),
            audio_driver,
            _rt_gc: rt_gc,
            project_state: Arc::new(Mutex::new(project_state)),
            undo_history: Mutex::new(undo_history),
            collision_bus,
            plugin_gui_worker,
            capture_stream: Mutex::new(None),
            midi_input_port: Mutex::new(None),
            disk_recorder: Mutex::new(None),
            completed_takes: Mutex::new(Vec::new()),
            recording_meta: Mutex::new(RecordingRuntimeMeta {
                driver_mode: "WASAPI Shared".to_string(),
                active_input_name: "None".to_string(),
                active_midi_name: "None".to_string(),
                armed_track_index: 1,
                input_monitor: false,
                hw_input_latency_samples: 0,
                record_start_tick: 0,
            }),
            vocal_source_pcm: Mutex::new(Vec::new()),
            vocal_source_sr: Mutex::new(48_000),
            vocal_blobs: Mutex::new(Vec::new()),
            engine_sample_rate: sample_rate,
            custom_audio_pcm: Arc::new(Mutex::new(Vec::new())),
            custom_audio_sr: Arc::new(Mutex::new(48_000)),
            custom_audio_path: Arc::new(Mutex::new(None)),
            custom_audio_wav_cache: Arc::new(Mutex::new(Vec::new())),
            custom_audio_active: AtomicBool::new(false),
            bounced_stems_pcm: Arc::new(Mutex::new(None)),
            active_sample_kit_query: Arc::new(Mutex::new(None)),
            uvr5_stem_wavs: Arc::new(Mutex::new([
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ])),
            asset_cache: Mutex::new(HashMap::new()),
            undo_coalesce: Mutex::new((String::new(), Instant::now())),
            derived: Mutex::new(DerivedRenderState::default()),
            transport_tick_clock: Arc::new(AtomicI64::new(0)),
        })
        .setup(|app| {
            let _ = APP_HANDLE.set(app.handle().clone());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            cmd_poll_binary_telemetry_frame,
            cmd_undo_project_action,
            cmd_redo_project_action,
            cmd_get_undo_redo_status,
            cmd_list_wasapi_output_devices,
            cmd_configure_wasapi_output,
            cmd_query_wasapi_status,
            cmd_set_wasapi_volume_mute,
            cmd_trigger_audition_note,
            cmd_list_audio_midi_inputs,
            cmd_configure_hardware_recording,
            cmd_start_stop_disk_recording,
            cmd_scan_external_plugins,
            cmd_load_external_plugin_slot,
            cmd_open_external_plugin_gui,
            cmd_close_external_plugin_gui,
            cmd_set_external_plugin_param,
            cmd_proq4_fit_eq_sketch,
            cmd_proq4_set_collision_reference,
            cmd_soothe3_set_external_sidechain,
            cmd_ozone12_set_low_end_focus,
            cmd_ozone12_set_codec_preview,
            cmd_piano_roll_transform_notes,
            cmd_playlist_slice_or_edit_clip,
            cmd_vocal_analyze_pitch_blobs,
            cmd_vocal_commit_pitch_edits,
            cmd_update_pitcher_rt_params,
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
            cmd_get_project_state,
            cmd_set_track_mute,
            cmd_set_track_solo,
            cmd_set_track_pan,
            cmd_set_slot_enabled,
            cmd_set_slot_wet_mix,
            cmd_remove_plugin_slot,
            cmd_insert_native_plugin_slot,
            cmd_set_telemetry_focus_track,
            cmd_save_project,
            cmd_open_project,
            cmd_new_project,
        ])
        .build(tauri::generate_context!())
        .expect("Failed to build ToneForge Studio Tauri application");

    app.run(move |_app_handle, event| {
        if let RunEvent::Exit = event {
            if let Some(mut stream) = audio_driver_exit.lock().take() {
                stream.shutdown();
            }
            clean_webview_cache();
        }
    });
}
