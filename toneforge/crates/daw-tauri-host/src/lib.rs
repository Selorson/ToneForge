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
pub mod preset_library;
pub mod project_state;
pub mod shm_telemetry_server;
pub mod wgpu_compositor;

pub use audio_stream_controller::{
    AudioStreamController, StemPlaybackGeneratorNode, Uvr5EnsembleStudioOutput,
};
pub use ipc_commands::{
    DawAppState, PatcherCanvasEdgeDto, PatcherCanvasNodeDto, PatcherCanvasUpdateDto,
    ProQ4BandUpdateDto, ProQ4GlobalUpdateDto, Soothe3ScBandUpdateDto, Soothe3UpdateDto,
    Uvr5SeparationRequestDto, Uvr5SeparationResultDto, cmd_apply_proq4_ffp_preset,
    cmd_apply_soothe3_factory_preset, cmd_execute_offline_render, cmd_load_fl26_fx_preset,
    cmd_load_fl26_sample_kit, cmd_load_fl26_score, cmd_load_ozone12_genre_or_xml_preset,
    cmd_load_proq4_preset, cmd_load_soothe3_preset_json, cmd_load_uvr5_model_profile,
    cmd_poll_binary_telemetry_frame, cmd_route_mixer_track, cmd_route_mixer_track_on_engine,
    cmd_run_ozone12_ai_master_assistant, cmd_run_uvr5_separation, cmd_seek_transport,
    cmd_set_internal_controller_norm, cmd_set_ppq_resolution, cmd_set_track_gain, cmd_set_transport,
    cmd_trigger_piano_roll_note, cmd_update_automation_spline, cmd_update_bsrnn_stem_gains,
    cmd_update_bsrnn_stem_state, cmd_update_ozone12_master, cmd_update_patcher_graph,
    cmd_update_proq4_band, cmd_update_proq4_global, cmd_update_soothe3_params,
    cmd_update_soothe3_sc_band, cmd_upsert_piano_roll_pattern, cmd_upsert_playlist_clip,
    execute_uvr5_separation_dsp,
};
pub use offline_render_engine::{
    DecodedWavFile, ExportedTrackWav, OfflineRenderConfig, OfflineRenderEngine,
    OfflineRenderResult, PostRenderAnalysisReport, RenderMode, RenderSampleRate,
    StreamingLoudnessCompliance, WavBitDepth, analyze_rendered_audio, decode_wav_bytes,
    encode_wav_bytes, read_wav_file, write_wav_file,
};
pub use preset_library::{
    Fl26FxPresetDto, Fl26SampleKitDto, Fl26SampleVoiceDto, Ozone12AssistantPresetDto,
    ProQ4PresetStateDto, ProQ4UiBandDto, Soothe3PresetStateDto, Soothe3ScBandUiDto,
    Uvr5ModelProfileDto, apply_fl26_fx_preset_to_engine, apply_fl26_sample_kit_to_engine,
    apply_proq4_preset, apply_soothe3_preset, load_fl26_fx_preset, load_fl26_sample_kit,
    load_fl26_score_preset, load_ozone12_genre_preset, load_ozone12_global_xml_preset,
    load_uvr5_model_profile, parse_fl26_fsc_bytes, parse_proq4_ffp, parse_soothe3_preset,
    read_fl26_sample_kit_wav_bytes, resolve_assets_root, resolve_proq4_ffp_path,
    resolve_soothe3_preset_path, run_ozone12_master_assistant,
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
    BiquadCascadeUniforms, CsgOperation, CurveUniforms, HeatmapUniforms, MixerMetersUniforms,
    SdfControlInstance, SdfPrimitive, SdfPrimitiveKind, Soothe3HeatmapUniforms, csg_combine,
    eval_sdf_primitive,
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

    #[test]
    fn test_real_proq4_ffp_and_soothe3_preset_loaders() {
        let mut runtime = ToneForgeHostRuntime::new(48000.0, 256);

        // Verify ALL 12 Pro-Q 4 UI dropdown presets + rapid 10+ preset switching + 120 Hz handle dragging
        let all_proq4_ui_presets = [
            "Default Setting",
            "Vocals / Lead Vocal",
            "Vocals / Female Vocal",
            "Mixbus / Dynamic Smiley Curve ST",
            "Mixbus / Revigorate",
            "Mixbus / Mastering High & Low Cut",
            "Drums / Kick Punch",
            "Drums / Snare Snap",
            "Bass / Sub Tighten",
            "Guitar / Acoustic Sparkle",
            "Synth / Wide Pad",
            "Utility / Brickwall Sub Cut",
        ];

        // Queue 12 full 601-param presets + 240 interactive 120 Hz band updates WITHOUT draining
        // in between (12 * 601 + 240 * 4 = 8,172 commands < 16,384 capacity).
        for (i, preset_name) in all_proq4_ui_presets.iter().enumerate() {
            let proq_dto = cmd_apply_proq4_ffp_preset(&runtime.app_state, 1, 0, preset_name)
                .unwrap_or_else(|e| panic!("Failed Pro-Q 4 preset '{preset_name}': {e}"));
            assert!(
                !proq_dto.bands.is_empty(),
                "Pro-Q 4 preset '{preset_name}' should have active bands"
            );
            assert_eq!(proq_dto.total_catalog_presets, 171);

            for step in 0..20 {
                cmd_update_proq4_band(
                    &runtime.app_state,
                    ProQ4BandUpdateDto {
                        track_idx: 1,
                        slot_idx: 0,
                        band_idx: (i % 8) as u8,
                        freq_hz: 120.0 + (step as f32) * 45.0,
                        gain_db: -2.0 + (step as f32) * 0.25,
                        q: 1.2,
                        shape: 0,
                        slope_db_oct: 12,
                        dynamic_range_db: 0.0,
                        spectral_enabled: false,
                        spectral_density: 0.5,
                    },
                )
                .expect("120 Hz handle drag during rapid preset switching must not overflow");
            }
        }

        // Drain ringbuffer
        let mut master_out = [[0.0f32; 2]; 64];
        runtime.mixer_engine.process_audio_block(&mut master_out);

        // Verify ALL 15 Soothe3 UI dropdown presets (both plain names and Category/Name.preset forms)
        let all_soothe3_ui_presets = [
            "Default Setting",
            "That Grammy balance preset",
            "General purpose de-harsher",
            "SM7 Vocals",
            "Vocals/SM7 Vocals.preset",
            "Lead Vox Bright",
            "De-ess normal",
            "Harsh Cymbals",
            "Dampen ringy snare",
            "Drums/Dampen ringy snare.preset",
            "Clean up muddy bass",
            "Acoustic guitar body",
            "Fizz remover for heavy guitars",
            "Soften synth pad",
            "2bus - glue and air",
            "Master air",
            "Mastering/That Grammy balance preset.preset",
        ];

        for preset_name in &all_soothe3_ui_presets {
            let soothe_dto =
                cmd_apply_soothe3_factory_preset(&runtime.app_state, 1, 1, preset_name)
                    .unwrap_or_else(|e| panic!("Failed Soothe3 preset '{preset_name}': {e}"));
            assert!(soothe_dto.params_loaded_count >= 100);
            assert_eq!(soothe_dto.total_catalog_presets, 68);
            assert_eq!(soothe_dto.sc_bands.len(), 6);
            runtime.mixer_engine.process_audio_block(&mut master_out);
        }
    }

    #[test]
    fn test_real_ozone12_genre_xml_and_ai_master_assistant() {
        let mut runtime = ToneForgeHostRuntime::new(48000.0, 256);
        let mut master_out = [[0.0f32; 2]; 64];

        for genre in [
            "AllPurpose",
            "Pop",
            "EDM",
            "HipHop",
            "Trap",
            "Rock",
            "LoFi",
            "KPopJPop",
            "RnbSoul",
            "Jazz",
            "Orchestral",
            "Modern",
            "Metal",
        ] {
            let dto = cmd_load_ozone12_genre_or_xml_preset(&runtime.app_state, 0, 1, genre, false)
                .unwrap_or_else(|e| panic!("Failed Ozone 12 genre '{genre}': {e}"));
            assert_eq!(dto.genre_id, genre);
            assert_eq!(dto.target_curve_64.len(), 64);
            runtime.mixer_engine.process_audio_block(&mut master_out);
        }

        for xml_preset in [
            "Modern Clarity",
            "Aggro Dance",
            "Clean & Wide",
            "Warm & Open",
            "Punchy Low End",
            "CD Master",
            "Streaming Master (-14 LUFS)",
            "pop-modern-pro.xml",
            "edm-modern-pro.xml",
            "hiphop-modern-pro.xml",
            "rock-modern-pro.xml",
        ] {
            let xml_dto =
                cmd_load_ozone12_genre_or_xml_preset(&runtime.app_state, 0, 1, xml_preset, true)
                    .unwrap_or_else(|e| panic!("Failed Ozone 12 XML preset '{xml_preset}': {e}"));
            assert!(!xml_dto.preset_label.is_empty());
            runtime.mixer_engine.process_audio_block(&mut master_out);
        }

        let ai_dto = cmd_run_ozone12_ai_master_assistant(&runtime.app_state, 0, 1, None)
            .expect("Should run real GenreCnnClassifier + InstrumentCnnClassifier");
        assert!(ai_dto.genre_confidence_pct >= 50.0);
        assert!(!ai_dto.dominant_instrument.is_empty());
    }

    #[test]
    fn test_real_fl26_scores_samples_fst_and_uvr5_profiles() {
        let mut runtime = ToneForgeHostRuntime::new(48000.0, 256);
        let mut master_out = [[0.0f32; 2]; 64];

        for score_name in [
            "ToneForge Lead Hook (Slide)",
            "FL26 Trance Arp (.fsc)",
            "FL26 Trance 2 (.fsc)",
            "FL26 Oxygene Progression (.fsc)",
            "FL26 4-Steps Groove (.fsc)",
            "FL26 DrumRoll (.fsc)",
            "FL26 Porta Slide 3-2-1 (.fsc)",
        ] {
            let pat = cmd_load_fl26_score(&mut runtime.mixer_engine, score_name, 96)
                .unwrap_or_else(|e| panic!("Failed FL26 score '{score_name}': {e}"));
            assert!(
                !pat.notes.is_empty(),
                "Score {score_name} should have notes"
            );
            assert_eq!(pat.target_channel_id, 1);
        }

        for kit_name in [
            "808 Studio Kit (Real .wav)",
            "909 Analog Kit (Real .wav)",
            "MA Skipper Snare Kit",
            "Symphonic Strings (SYM_STR_C_2)",
            "Sytrus / Harmor Lead",
        ] {
            let kit = apply_fl26_sample_kit_to_engine(&mut runtime.mixer_engine, kit_name)
                .unwrap_or_else(|e| panic!("Failed FL26 sample kit '{kit_name}': {e}"));
            assert!(!kit.voices.is_empty());
            assert_eq!(kit.voices[0].sample_rate, 44100);
        }

        for fst_name in [
            "Maximus: Clear Master.fst",
            "Maximus: Maximum loudness.fst",
            "Maximus: New York compression.fst",
            "Maximus: Soundgoodizer A.fst",
            "Emphasis: Pop (heavy).fst",
            "Emphasis: Louder (transient shaping).fst",
            "Emphasis: Soft-clipper (heavy).fst",
            "Emphasis — Clean upward limiter.fst",
            "Fruity Limiter: Drum compression 1.fst",
            "Fruity Limiter: Ducking music behind voice.fst",
            "Fruity Parametric EQ 2: Vocal presence.fst",
            "Fruity Parametric EQ 2: Mastering.fst",
            "Fruity Soft Clipper.fst",
            "Fruity Soft Clipper — Heavy.fst",
            "LuxeVerb: Shimmer Cathedral.fst",
            "LuxeVerb — Bright Hall",
            "Gross Beat: 1/2 Gate.fst",
            "Gross Beat — 1/2 Gate",
        ] {
            let fst_dto = cmd_load_fl26_fx_preset(&runtime.app_state, 1, 2, fst_name)
                .unwrap_or_else(|e| panic!("Failed FL26 .fst preset '{fst_name}': {e}"));
            assert!(fst_dto.fst_byte_size > 100);
            runtime.mixer_engine.process_audio_block(&mut master_out);
        }

        for uvr_profile in [
            "Ensemble: MDX23C + HTDemucs4 + VR-ASPP (3-Model)",
            "MDX23C-8KFFT-InstVoc_HQ (n_fft=8192, dim_f=4096)",
            "Kim_Vocal_2 (MDX-Net Lead Vocal, n_fft=7680)",
            "UVR-MDX-Net-Inst_HQ_3 (n_fft=6144, dim_f=3072)",
            "UVR-MDX-NET Karaoke / Vocals",
            "HTDemucs_ft 4-Stem (Vocals / Drums / Bass / Other)",
            "VR-Arch 4band_v3_sn (Cascaded ASPP ResNet)",
        ] {
            let uvr_dto = cmd_load_uvr5_model_profile(uvr_profile)
                .unwrap_or_else(|e| panic!("Failed UVR5 profile '{uvr_profile}': {e}"));
            assert!(uvr_dto.n_fft >= 1024);
            assert!(uvr_dto.dim_f >= 512);
        }
    }

    #[test]
    fn test_preset_library_edge_cases_and_ui_selectors_verification() {
        let runtime = ToneForgeHostRuntime::new(48000.0, 256);

        // 1. Edge cases: invalid track/slot indices, nonexistent presets, corrupt inputs
        assert!(
            cmd_apply_proq4_ffp_preset(&runtime.app_state, 999, 0, "Default Setting").is_err(),
            "Out-of-range track_idx must fail cleanly"
        );
        assert!(
            cmd_apply_proq4_ffp_preset(&runtime.app_state, 1, 99, "Default Setting").is_err(),
            "Out-of-range slot_idx must fail cleanly"
        );
        assert!(
            cmd_apply_proq4_ffp_preset(&runtime.app_state, 1, 0, "NonExistentPreset12345.ffp")
                .is_err(),
            "Missing .ffp preset must return error"
        );
        assert!(
            parse_proq4_ffp("not a valid ffp file", "broken.ffp", "Root").is_err(),
            "Missing [Preset] header must return error"
        );
        assert!(
            cmd_apply_soothe3_factory_preset(
                &runtime.app_state,
                1,
                1,
                "NonExistentSoothePreset999"
            )
            .is_err(),
            "Missing Soothe3 .preset must return error"
        );
        assert!(
            cmd_load_ozone12_genre_or_xml_preset(
                &runtime.app_state,
                0,
                2,
                "NonExistentXml999",
                true
            )
            .is_err(),
            "Missing Ozone 12 XML preset must return error"
        );
        assert!(
            parse_fl26_fsc_bytes(b"BAD_HEADER", 1, "broken", 1, 96).is_err(),
            "Corrupt .fsc header must return error"
        );

        // Edge case: parse_fl26_fsc_bytes with target_ppq = 0, target_ppq = 1920, and 120-byte polyphonic payload
        let trance_path = resolve_assets_root()
            .join("flstudio26")
            .join("scores")
            .join("Trance 1.fsc");
        let trance_bytes = std::fs::read(&trance_path).expect("Trance 1.fsc");
        let pat_ppq0 = parse_fl26_fsc_bytes(&trance_bytes, 1, "PPQ0", 1, 0)
            .expect("target_ppq=0 must clamp to 24 and never divide by zero");
        assert!(pat_ppq0.length_ticks >= 96);
        let pat_ppq1920 = parse_fl26_fsc_bytes(&trance_bytes, 1, "PPQ1920", 1, 1920)
            .expect("target_ppq=1920 must clamp to 960");
        assert!(pat_ppq1920.length_ticks >= 3840);

        // Construct a synthetic 120-byte (5 * 24-byte notes) .fsc event 0xE0 payload to verify 20 vs 24 byte disambiguation
        let mut synth_fsc = Vec::new();
        synth_fsc.extend_from_slice(b"FLhd\x06\x00\x00\x00\x00\x00\x01\x00\x60\x00FLdt\x7A\x00\x00\x00");
        synth_fsc.push(0xE0);
        synth_fsc.push(120); // 5 * 24 bytes (also divisible by 20!)
        for i in 0..5u32 {
            let mut note24 = [0u8; 24];
            note24[0..4].copy_from_slice(&(i * 24).to_le_bytes()); // pos
            note24[8..12].copy_from_slice(&24u32.to_le_bytes()); // len
            note24[12] = 60 + i as u8; // key <= 127
            note24[13] = 120; // fine pitch
            note24[16] = 64; // pan
            note24[17] = 100; // velocity <= 128
            note24[18..24].copy_from_slice(&[0xFF, 0xFF, 0x40, 0x40, 0x00, 0x00]); // trailing bytes that make 20-byte chunks invalid
            synth_fsc.extend_from_slice(&note24);
        }
        let pat_24 = parse_fl26_fsc_bytes(&synth_fsc, 2, "Poly24", 1, 96)
            .expect("Should disambiguate 24-byte note structs when len % 120 == 0");
        assert_eq!(pat_24.notes.len(), 5);
        assert_eq!(pat_24.notes[0].key, 60);
        assert_eq!(pat_24.notes[4].key, 64);

        // Empty preview slice falls back cleanly in AI Master Assistant
        let ai_empty = cmd_run_ozone12_ai_master_assistant(&runtime.app_state, 0, 1, Some(vec![]))
            .expect("Empty audio preview slice should fall back to reference stem");
        assert!(!ai_empty.detected_genre.is_empty());

        // 2. Verify all 5 UI racks have interactive preset selectors in both UI directories
        let ws = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("workspace root")
            .to_path_buf();
        for ui_dir in ["ui-svelte-webgpu", "ui-svelte5-wgpu"] {
            let comp_dir = ws.join(ui_dir).join("src").join("lib").join("components");
            let pr = std::fs::read_to_string(comp_dir.join("PlaylistPianoRollEditor.svelte"))
                .expect("PlaylistPianoRollEditor.svelte");
            assert!(pr.contains("FL26 Sample & Synth Kit"));
            assert!(pr.contains("FL26 Piano Roll Score (.fsc)"));

            let pq = std::fs::read_to_string(comp_dir.join("ProQ4InteractiveAnalyzer.svelte"))
                .expect("ProQ4InteractiveAnalyzer.svelte");
            assert!(pq.contains("Factory Preset (171 Real .ffp):"));

            let s3 = std::fs::read_to_string(comp_dir.join("Soothe3ResonanceView.svelte"))
                .expect("Soothe3ResonanceView.svelte");
            assert!(s3.contains("Factory Preset (68 Real .preset):"));

            let oz = std::fs::read_to_string(comp_dir.join("Ozone12MasterChain.svelte"))
                .expect("Ozone12MasterChain.svelte");
            assert!(oz.contains("Genre Target Curve (36 Real IZRESEMBED):"));
            assert!(oz.contains("Global Mastering Preset (82 Real XML):"));
            assert!(oz.contains("AI Master Assistant (Auto-Match)"));

            let mx = std::fs::read_to_string(comp_dir.join("MixerRack502.svelte"))
                .expect("MixerRack502.svelte");
            assert!(mx.contains("FL26 Native FX Preset (.fst):"));

            let uv = std::fs::read_to_string(comp_dir.join("Uvr5EnsembleStudio.svelte"))
                .expect("Uvr5EnsembleStudio.svelte");
            assert!(uv.contains("UVR5 Model Profile (143 Real Configs):"));
        }
    }

    #[test]
    fn test_audit_remediation_ipc_routing_fst_uvr5_and_custom_audio() {
        let mut ctrl = AudioStreamController::with_mastering_template(48_000.0, 256)
            .expect("create mastering controller");

        // 1. Seed 4 startup demo_stems on Tracks 1..=4, then replace with custom_audio_stem on Track 1
        let demo = vec![[0.25f32, -0.25f32]; 512];
        ctrl.bounce_stems_to_mixer_tracks(
            &[demo.clone(), demo.clone(), demo.clone(), demo],
            &[1, 2, 3, 4],
        )
        .expect("seed 4 demo stems");
        assert_eq!(ctrl.mixer_engine.channels.len(), 4);

        let custom_song = vec![[0.4f32, 0.35f32]; 1024];
        ctrl.mixer_engine
            .replace_with_custom_audio_stem(custom_song.clone(), true)
            .expect("replace demo stems with custom audio stem on Track 1");
        assert_eq!(
            ctrl.mixer_engine.channels.len(),
            1,
            "Startup demo_stem channels on Tracks 1..4 must be replaced by single Custom Audio Stem on Track 1"
        );
        assert_eq!(ctrl.mixer_engine.channels[0].target_mixer_track, 1);

        // 2. Pro-Q 4 Global update (LinearPhase + RefreshApdc) & Soothe3 Sidechain EQ band update
        cmd_update_proq4_global(
            &ctrl.app_state,
            ProQ4GlobalUpdateDto {
                track_idx: 1,
                slot_idx: 0,
                processing_mode: 2, // LinearPhase
                processing_resolution: 1,
                character: 2, // Warm
                gain_scale_pct: 110.0,
                output_level_db: 1.5,
                bypass: false,
                auto_gain: true,
                ..Default::default()
            },
        )
        .expect("cmd_update_proq4_global");

        cmd_update_soothe3_sc_band(
            &ctrl.app_state,
            Soothe3ScBandUpdateDto {
                track_idx: 1,
                slot_idx: 1,
                band_idx: 2,
                enabled: true,
                freq_hz: 4200.0,
                gain_db: 8.5,
                q: 2.4,
                shape: 2,
            },
        )
        .expect("cmd_update_soothe3_sc_band");

        // 3. BSRNN 4-Stem Solo/Mute state via IPC
        cmd_update_bsrnn_stem_state(
            &ctrl.app_state,
            0,
            0,
            [2.0, -1.5, 1.0, 0.0],
            [true, false, false, false],
            [false, false, false, true],
        )
        .expect("cmd_update_bsrnn_stem_state");

        // 4. Real Mixer DAG routing (`cmd_route_mixer_track_on_engine`) + Cycle Rejection
        cmd_route_mixer_track_on_engine(&mut ctrl.mixer_engine, &ctrl.app_state, 5, 6, 0.85, false)
            .expect("route 5 -> 6");
        cmd_route_mixer_track_on_engine(&mut ctrl.mixer_engine, &ctrl.app_state, 6, 7, 0.90, true)
            .expect("sidechain route 6 -> 7");
        ctrl.mixer_engine.drain_commands();
        assert!(
            (ctrl.mixer_engine.tracks[5].gain - 1.0).abs() < 1e-6,
            "cmd_route_mixer_track_on_engine must not overwrite src_track fader gain"
        );
        assert!(
            cmd_route_mixer_track_on_engine(
                &mut ctrl.mixer_engine,
                &ctrl.app_state,
                7,
                5,
                1.0,
                false
            )
            .is_err(),
            "Cycle 7 -> 5 must be rejected by cmd_route_mixer_track_on_engine"
        );

        // 5. Real FL26 Native FX `.fst` instantiation on `MixerEngine` (`apply_fl26_fx_preset_to_engine`)
        for (slot_idx, fst_query, expected_node_substr) in [
            (0u8, "Maximus: Clear Master.fst", "Maximus"),
            (1u8, "Emphasis: Pop (heavy).fst", "Emphasis"),
            (2u8, "Fruity Limiter: Drum compression 1.fst", "Sparse"),
            (3u8, "Gross Beat: 1/2 Gate.fst", "Gross Beat"),
            (4u8, "LuxeVerb: Shimmer Cathedral.fst", "LuxeVerb"),
        ] {
            let dto = apply_fl26_fx_preset_to_engine(
                &mut ctrl.mixer_engine,
                &ctrl.app_state,
                5,
                slot_idx,
                fst_query,
            )
            .expect("apply_fl26_fx_preset_to_engine");
            assert!(!dto.plugin_name.is_empty());
            let loaded_name = ctrl.mixer_engine.tracks[5].slots[slot_idx as usize]
                .plugin
                .name();
            assert!(
                loaded_name.contains(expected_node_substr),
                "Expected slot {} plugin '{}' to contain '{}'",
                slot_idx,
                loaded_name,
                expected_node_substr
            );
        }

        // 6. Real FL26 Sample Kit discrete voices + raw WAV fetch (`read_fl26_sample_kit_wav_bytes`)
        let kit_dto =
            apply_fl26_sample_kit_to_engine(&mut ctrl.mixer_engine, "808 Studio Kit (Real .wav)")
                .expect("apply 808 Studio Kit");
        assert_eq!(kit_dto.voices.len(), 4);
        let kick_wav = read_fl26_sample_kit_wav_bytes("808 Studio Kit (Real .wav)", 0)
            .expect("read 808 Kick WAV bytes");
        assert!(kick_wav.starts_with(b"RIFF"));

        // 7. Full-length UVR5 3-Model Spectral Ensemble separation + bounce replacement onto Tracks 1..=4
        let uvr_req = Uvr5SeparationRequestDto {
            audio_path: "custom_song.wav".to_string(),
            model_hash: "ensemble_3model".to_string(),
            ensemble_mode: "Weighted".to_string(),
            weights: [0.45, 0.35, 0.20],
            vr_aggressiveness: 0.15,
            ..Default::default()
        };
        let (uvr_res, stems, wav_bufs) =
            execute_uvr5_separation_dsp(&custom_song, 48_000, &uvr_req)
                .expect("execute_uvr5_separation_dsp");
        assert_eq!(uvr_res.separated_frames, custom_song.len());
        assert_eq!(uvr_res.stem_paths.len(), 4);
        for wb in &wav_bufs {
            assert!(wb.starts_with(b"RIFF") && wb.len() > 44);
        }
        AudioStreamController::bounce_stems_to_engine(
            &mut ctrl.mixer_engine,
            &stems,
            &[1, 2, 3, 4],
            48_000.0,
            256,
        )
        .expect("bounce separated stems to tracks 1..4");

        // 8. Verify OfflineRenderEngine drains pending SPSC commands before rendering
        cmd_set_track_gain(&ctrl.app_state, 0, 0.88).expect("queue master gain via SPSC");
        let render_res = OfflineRenderEngine::execute_render(
            &mut ctrl.mixer_engine,
            &OfflineRenderConfig {
                mode: RenderMode::Master,
                sample_rate: RenderSampleRate::Hz48000,
                bit_depth: WavBitDepth::Pcm24MbitPlus,
                enable_mbit_plus_noise_shaping: true,
                trim_apdc_latency: true,
                duration_frames: 512,
                block_size: 256,
                split_track_indices: Vec::new(),
                output_dir: None,
                file_prefix: "AuditVerify".to_string(),
            },
        )
        .expect("offline render after SPSC queue");
        assert!((ctrl.mixer_engine.tracks[0].gain - 0.88).abs() < 1e-4);
        assert_eq!(render_res.master_wav.frames.len(), 512);
    }
}

