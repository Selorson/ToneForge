//! `daw-tauri-host` — ToneForge Desktop Host Crate.
//!
//! Modules:
//! - [`shm_telemetry_server`]: Zero-Copy Binary Telemetry Streamer (`ZeroCopyBinaryTelemetryStreamer` /
//!   `ShmTelemetryServer`) streaming the 128 KB (`131,072` B) `VisualTelemetryFrame` from `TripleBuffer`
//!   at 120 Hz without JSON serialization.
//! - [`audio_stream_controller`]: Real-time audio stream coordinator (`AudioStreamController`) managing
//!   `MixerEngine`, wait-free `rtrb::Producer<EngineCommand>`, Pro-Q 4 (`601` params), Soothe3 (`143` params),
//!   Ozone 12 Master Chain, `RealTimeBsrnnStemNode` (`120 ms` 4-stem rebalance), and offline UVR5
//!   multi-architecture separation & spectral ensemble (`Mdx23cSeparator`, `HTDemucs4`, `CascadedASPPNet`, `SpectralEnsembler`).
//! - [`project_state`]: Complete DAW project state serialization & reconstruction (`ToneForgeProjectState`)
//!   for 502 Mixer Tracks, Patcher DAGs, Automation Clips with `TensionSpline`, and full plugin chains.
//! - [`ipc_commands`]: Lock-free IPC command handlers and DTOs.
//! - [`wgpu_compositor`]: 120 FPS WebGPU / WGPU uniform and SDF instance packers.

pub mod audio_stream_controller;
pub mod ipc_commands;
pub mod offline_render_engine;
pub mod project_state;
pub mod shm_telemetry_server;
pub mod wgpu_compositor;

pub use audio_stream_controller::{
    AudioStreamController, StemPlaybackGeneratorNode, Uvr5EnsembleStudioOutput,
};
pub use ipc_commands::{
    DawAppState, ProQ4BandUpdateDto, Soothe3UpdateDto, cmd_execute_offline_render,
    cmd_load_proq4_preset, cmd_load_soothe3_preset_json, cmd_poll_binary_telemetry_frame,
    cmd_route_mixer_track, cmd_run_uvr5_separation, cmd_seek_transport,
    cmd_set_internal_controller_norm, cmd_set_ppq_resolution, cmd_set_track_gain,
    cmd_set_transport, cmd_trigger_piano_roll_note, cmd_update_automation_spline,
    cmd_update_bsrnn_stem_gains, cmd_update_ozone12_master, cmd_update_proq4_band,
    cmd_update_soothe3_params, cmd_upsert_piano_roll_pattern, cmd_upsert_playlist_clip,
};
pub use offline_render_engine::{
    DecodedWavFile, ExportedTrackWav, OfflineRenderConfig, OfflineRenderEngine,
    OfflineRenderResult, PostRenderAnalysisReport, RenderMode, RenderSampleRate,
    StreamingLoudnessCompliance, WavBitDepth, analyze_rendered_audio, decode_wav_bytes,
    encode_wav_bytes, read_wav_file, write_wav_file,
};
pub use project_state::{
    AutomationClipStateDto, BsrnnStemStateDto, ControllerLinkStateDto, LfoModulatorDto,
    MixerSendRouteDto, MixerTrackStateDto, Ozone12MasterConfigDto, PatcherAudioEdgeDto,
    PatcherGraphStateDto, PatcherNodeLayoutDto, PatcherNodeStateDto, PatcherParamEdgeDto,
    PianoRollNoteDto, PianoRollPatternDto, PlaylistClipDto, PluginConfigDto, PluginSlotStateDto,
    ProQ4StateDto, Soothe3StateDto, TONEFORGE_PROJECT_SCHEMA_VERSION, TensionControlPointDto,
    ToneForgeProjectState,
};
pub use shm_telemetry_server::{
    OFFSET_AUTOMATION_SPLINE, OFFSET_HEADER, OFFSET_HEADER_ALIGN_PAD, OFFSET_OZONE12_TELEMETRY,
    OFFSET_PROQ4_SPECTRUM, OFFSET_SOOTHE3_REDUCTION, OFFSET_SPECTROGRAM_RING, OFFSET_TRACK_METERS,
    ShmTelemetryServer, TELEMETRY_FRAME_INTERVAL_NS, TELEMETRY_STREAM_TARGET_FPS,
    ZeroCopyBinaryTelemetryStreamer,
};
pub use wgpu_compositor::{
    BiquadCascadeUniforms, MixerMetersUniforms, SdfControlInstance, Soothe3HeatmapUniforms,
};

use daw_core_graph::mixer_dag::MixerEngine;

/// Bundled runtime session connecting the UI host, command dispatcher,
/// telemetry triple-buffer server, and real-time `MixerEngine`.
pub struct ToneForgeHostRuntime {
    pub app_state: DawAppState,
    pub telemetry_server: ShmTelemetryServer,
    pub mixer_engine: Box<MixerEngine>,
}

impl ToneForgeHostRuntime {
    pub fn new(sample_rate: f64, _max_block_size: usize) -> Self {
        let (engine, cmd_prod, telem_out) = MixerEngine::new(sample_rate, 128.0, 0);

        Self {
            app_state: DawAppState::new(cmd_prod),
            telemetry_server: ShmTelemetryServer::new(telem_out),
            mixer_engine: Box::new(engine),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc_commands::{
        ProQ4BandUpdateDto, cmd_route_mixer_track, cmd_set_transport, cmd_update_proq4_band,
    };

    #[test]
    fn test_host_ipc_to_rt_engine_roundtrip() {
        let mut runtime = ToneForgeHostRuntime::new(48000.0, 256);

        cmd_set_transport(&runtime.app_state, true, 140.0).unwrap();
        cmd_route_mixer_track(&runtime.app_state, 1, 2, 0.75, false).unwrap();
        cmd_update_proq4_band(
            &runtime.app_state,
            ProQ4BandUpdateDto {
                track_idx: 1,
                slot_idx: 0,
                band_idx: 0,
                freq_hz: 1000.0,
                gain_db: 3.0,
                q: 1.0,
                shape: 0,
                slope_db_oct: 12,
                dynamic_range_db: 0.0,
                spectral_enabled: false,
                spectral_density: 0.5,
            },
        )
        .unwrap();

        let mut master_out = [[0.0f32; 2]; 64];
        runtime.mixer_engine.process_audio_block(&mut master_out);

        let telem = runtime.telemetry_server.poll_latest_frame();
        assert_eq!(telem.header.frame_seq, 1);
        assert!((telem.header.bpm - 140.0).abs() < 1e-4);
    }
}
