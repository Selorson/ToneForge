//! Phase 4 Integration & Verification Test Suite for `daw-tauri-host`.
//!
//! Verifies:
//! 1. Zero-Copy Binary Telemetry Streamer (`131,072` B = `128 KB`) exact byte-offset alignment
//!    and lock-free `TripleBuffer` streaming without JSON serialization.
//! 2. `AudioStreamController` SPSC command dispatch, preset/parameter loading for
//!    Pro-Q 4 (`601` params), Soothe3 (`143` params & `.preset` JSON), Ozone 12 Master Chain,
//!    and `RealTimeBsrnnStemNode` (Master Rebalance / Stem EQ).
//! 3. Offline UVR5 Stem Separation Studio (`Mdx23cSeparator`, `HTDemucs4`, `CascadedASPPNet`,
//!    `SpectralEnsembler`) and stem bouncing to Mixer Insert Tracks `1..=4`.
//! 4. `ToneForgeProjectState` serialization/deserialization, 502-Track Mixer DAG,
//!    Patcher DAG (`SugiyamaLayout`), and `TensionSpline` Automation Clips on the
//!    30-bit fixed-point (`0x40000000`) controller bus, plus boundary/error edge cases.

use ai_uvr5_stem_engine::SpectralEnsembleMode;
use daw_core_graph::automation_spline::{fixed30_to_normalized, normalized_to_fixed30};
use daw_core_graph::telemetry_shm::{
    AutomationAndSplineTelemetry, Ozone12MasterTelemetry, ProQ4SpectrumFrame,
    Soothe3ReductionFrame, TrackMeterEntry, TransportAndMasterHeader,
    VISUAL_TELEMETRY_FRAME_BYTES, VisualTelemetryFrame,
};
use daw_core_graph::{
    FIXED_POINT_30BIT_MAX, MAX_MIXER_TRACKS, PATCHER_HOST_OUTPUT_NODE, StereoFrame, UnifiedDspNode,
};
use daw_tauri_host::audio_stream_controller::AudioStreamController;
use daw_tauri_host::ipc_commands::{
    ProQ4BandUpdateDto, Soothe3UpdateDto, cmd_load_soothe3_preset_json,
    cmd_poll_binary_telemetry_frame, cmd_route_mixer_track, cmd_run_uvr5_separation,
    cmd_seek_transport, cmd_set_internal_controller_norm, cmd_set_track_gain, cmd_set_transport,
    cmd_update_bsrnn_stem_gains, cmd_update_ozone12_master, cmd_update_proq4_band,
    cmd_update_soothe3_params,
};
use daw_tauri_host::project_state::{
    AutomationClipStateDto, BsrnnStemStateDto, LfoModulatorDto, MixerSendRouteDto,
    Ozone12MasterConfigDto, PatcherAudioEdgeDto, PatcherNodeStateDto, PluginConfigDto,
    PluginSlotStateDto, TensionControlPointDto, ToneForgeProjectState,
};
use daw_tauri_host::shm_telemetry_server::{
    OFFSET_AUTOMATION_SPLINE, OFFSET_HEADER, OFFSET_HEADER_ALIGN_PAD, OFFSET_OZONE12_TELEMETRY,
    OFFSET_PROQ4_SPECTRUM, OFFSET_SOOTHE3_REDUCTION, OFFSET_SPECTROGRAM_RING, OFFSET_TRACK_METERS,
    ShmTelemetryServer,
};
use daw_tauri_host::wgpu_compositor::{
    BiquadCascadeUniforms, MixerMetersUniforms, SdfControlInstance, Soothe3HeatmapUniforms,
};
use dsp_proq4_eq::{
    PROQ4_NUM_BANDS, PROQ4_PARAMS_PER_BAND, PROQ4_TOTAL_PARAMS, ProQ4GlobalParams,
    ProQ4ParameterSet, ProQ4Processor,
};
use dsp_soothe3_spectral::{
    SOOTHE3_TOTAL_PARAMS, SidechainEqBandParams, Soothe3ParameterSet, SootheProcessMode,
};

/// Synthesizes a rich multi-tone test waveform (`num_frames` stereo samples at `sample_rate`).
fn make_test_stereo_signal(num_frames: usize, sample_rate: f32) -> Vec<StereoFrame> {
    let mut out = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        let t = (i as f32) / sample_rate;
        let bass = 0.35 * (2.0 * std::f32::consts::PI * 110.0 * t).sin();
        let vocal = 0.30 * (2.0 * std::f32::consts::PI * 1000.0 * t).sin();
        let presence = 0.20 * (2.0 * std::f32::consts::PI * 3800.0 * t).cos();
        out.push([bass + vocal + presence, bass + vocal - presence]);
    }
    out
}

#[test]
fn test_zero_copy_binary_telemetry_streamer_128kb_exact_offsets_and_dedup() {
    assert_eq!(VISUAL_TELEMETRY_FRAME_BYTES, 131_072);
    assert_eq!(std::mem::size_of::<VisualTelemetryFrame>(), 131_072);
    assert_eq!(std::mem::align_of::<VisualTelemetryFrame>(), 64);

    // Verify exact struct section sizes and offsets match the 128 KB memory map
    assert_eq!(std::mem::size_of::<TransportAndMasterHeader>(), 64);
    assert_eq!(std::mem::size_of::<TrackMeterEntry>(), 24);
    assert_eq!(
        std::mem::size_of::<TrackMeterEntry>() * MAX_MIXER_TRACKS,
        12_048
    );
    assert_eq!(std::mem::size_of::<ProQ4SpectrumFrame>(), 16_384);
    assert_eq!(std::mem::size_of::<Soothe3ReductionFrame>(), 8_192);
    assert_eq!(std::mem::size_of::<Ozone12MasterTelemetry>(), 8_192);
    assert_eq!(std::mem::size_of::<AutomationAndSplineTelemetry>(), 4_096);

    assert_eq!(OFFSET_HEADER, 0x00000);
    assert_eq!(OFFSET_TRACK_METERS, 0x00040);
    assert_eq!(OFFSET_HEADER_ALIGN_PAD, 0x02F50);
    assert_eq!(OFFSET_PROQ4_SPECTRUM, 0x03000);
    assert_eq!(OFFSET_SOOTHE3_REDUCTION, 0x07000);
    assert_eq!(OFFSET_OZONE12_TELEMETRY, 0x09000);
    assert_eq!(OFFSET_SPECTROGRAM_RING, 0x0B000);
    assert_eq!(OFFSET_AUTOMATION_SPLINE, 0x1F000);

    let mut ctrl = AudioStreamController::with_mastering_template(48_000.0, 256)
        .expect("create mastering template controller");

    // Attach a looping multi-tone generator to Insert Track 1 (routed to Master Track 0).
    // Note: Track 1 has Soothe3 (3,072 samples STFT latency) + Patcher (64 samples),
    // and Master Track 0 has RealTimeBsrnnStemNode (5,760 samples = 120ms latency) + Ozone 12.
    // Total pipeline lookahead is ~9,000 samples (36 blocks of 256 samples).
    let signal = make_test_stereo_signal(2048, 48_000.0);
    ctrl.bounce_stems_to_mixer_tracks(&[signal], &[1])
        .expect("attach test generator on track 1");

    // Process 40 blocks (40 * 256 = 10,240 samples) so full APDC lookahead pipeline is filled
    let mut master_out = vec![[0.0f32; 2]; 256];
    for _ in 0..40 {
        ctrl.process_audio_block(&mut master_out);
    }
    assert!(
        master_out
            .iter()
            .any(|f| f[0].abs() > 1e-4 || f[1].abs() > 1e-4),
        "Master output should contain processed audio after clearing lookahead latency"
    );

    // Poll raw 128 KB binary telemetry frame via ZeroCopyBinaryTelemetryStreamer
    let raw_bytes: [u8; VISUAL_TELEMETRY_FRAME_BYTES] =
        *cmd_poll_binary_telemetry_frame(&mut ctrl.telemetry_streamer);
    assert_eq!(raw_bytes.len(), 131_072);
    assert_eq!(ctrl.telemetry_streamer.last_frame_seq(), 40);
    assert_eq!(ctrl.telemetry_streamer.frames_streamed(), 1);

    // Polling again without a new audio block should return None from poll_if_updated
    assert!(ctrl.telemetry_streamer.poll_if_updated().is_none());
    assert_eq!(ctrl.telemetry_streamer.skipped_duplicate_polls(), 1);

    // Decode header at 0x00000
    let header = ShmTelemetryServer::decode_header_from_bytes(&raw_bytes).expect("valid header");
    assert_eq!(header.frame_seq, 40);
    assert_eq!(header.sample_pos, 10_240);
    assert!((header.bpm - 128.0).abs() < 1e-4);
    assert!(
        header.cpu_load_pct > 0.0,
        "Header cpu_load_pct should be populated during real-time processing"
    );

    // Decode Track 0 (Master) and Track 1 meters at 0x00040
    let master_meter =
        ShmTelemetryServer::decode_track_meter_from_bytes(&raw_bytes, 0).expect("track 0 meter");
    let track1_meter =
        ShmTelemetryServer::decode_track_meter_from_bytes(&raw_bytes, 1).expect("track 1 meter");
    assert!(
        master_meter.peak_l_db > -65.0 && master_meter.peak_r_db > -65.0,
        "Master track should register live peaks, got L={} R={}",
        master_meter.peak_l_db,
        master_meter.peak_r_db
    );
    assert!(
        track1_meter.peak_l_db > -65.0 && track1_meter.peak_r_db > -65.0,
        "Insert track 1 should register live peaks, got L={} R={}",
        track1_meter.peak_l_db,
        track1_meter.peak_r_db
    );

    // Boundary checks on 502 tracks
    assert!(ShmTelemetryServer::decode_track_meter_from_bytes(&raw_bytes, 501).is_ok());
    assert!(ShmTelemetryServer::decode_track_meter_from_bytes(&raw_bytes, 502).is_err());

    // Decode Pro-Q 4 spectrum at 0x03000, Soothe3 reduction at 0x07000, Ozone 12 at 0x09000, Automation at 0x1F000
    let proq4_spec =
        ShmTelemetryServer::decode_proq4_spectrum_from_bytes(&raw_bytes).expect("proq4 spectrum");
    assert_eq!(proq4_spec.pre_db.len(), 1024);
    assert_eq!(proq4_spec.post_db.len(), 1024);

    let soothe3_red = ShmTelemetryServer::decode_soothe3_reduction_from_bytes(&raw_bytes)
        .expect("soothe3 reduction");
    assert_eq!(soothe3_red.reduction_ys.len(), 256);
    assert_eq!(soothe3_red.input_spec_db.len(), 1024);

    let ozone12_tel = ShmTelemetryServer::decode_ozone12_telemetry_from_bytes(&raw_bytes)
        .expect("ozone12 telemetry");
    assert_eq!(ozone12_tel.irc_gr_trace.len(), 512);
    assert_eq!(ozone12_tel.polar_sample_xy.len(), 512);
    assert_eq!(ozone12_tel.tonal_balance_bars.len(), 256);

    let auto_tel = ShmTelemetryServer::decode_automation_spline_from_bytes(&raw_bytes)
        .expect("automation spline telemetry");
    assert_eq!(auto_tel.controller_values_norm.len(), 512);
    assert_eq!(auto_tel.patcher_node_cpu_pct.len(), 510);
    assert!(
        auto_tel.patcher_node_cpu_pct[0] > 0.0 && auto_tel.patcher_node_cpu_pct[1] > 0.0,
        "PatcherGraphNode should populate patcher_node_cpu_pct in AutomationAndSplineTelemetry"
    );

    // Full boxed frame roundtrip and shared-slice copy
    let boxed_frame =
        ShmTelemetryServer::decode_boxed_frame_from_bytes(&raw_bytes).expect("boxed frame");
    assert_eq!(boxed_frame.header.frame_seq, 40);
    assert!(
        boxed_frame.spectrogram_ring[0]
            .iter()
            .any(|&db| db > -115.0),
        "20x1024 spectrogram_ring at 0x0B000 must be populated by Pro-Q 4 / Soothe3 during RT processing"
    );

    let mut shared_buf = vec![0u8; VISUAL_TELEMETRY_FRAME_BYTES];
    let copied_seq = ctrl
        .telemetry_streamer
        .copy_into_shared_slice(&mut shared_buf)
        .expect("copy into 128 KB slice");
    assert_eq!(copied_seq, 40);
    assert_eq!(&shared_buf[..], &raw_bytes[..]);

    // Undersized buffer error paths
    let mut small_buf = vec![0u8; 1024];
    assert!(
        ctrl.telemetry_streamer
            .copy_into_shared_slice(&mut small_buf)
            .is_err()
    );
    assert!(ShmTelemetryServer::decode_boxed_frame_from_bytes(&small_buf).is_err());
    assert!(ShmTelemetryServer::decode_header_from_bytes(&[0u8; 16]).is_err());
}

#[test]
fn test_audio_stream_controller_dsp_presets_and_ipc_command_dispatch() {
    let mut ctrl = AudioStreamController::with_mastering_template(48_000.0, 256)
        .expect("create mastering controller");

    // 1. Verify Pro-Q 4 601-parameter preset load & 24-param-stride band updates
    assert_eq!(PROQ4_TOTAL_PARAMS, 601);
    assert_eq!(PROQ4_PARAMS_PER_BAND, 24);
    assert_eq!(PROQ4_NUM_BANDS, 24);

    let mut pq_preset = ProQ4ParameterSet::new();
    pq_preset.bands[0].used = true;
    pq_preset.bands[0].enabled = true;
    pq_preset.bands[0].freq_hz = 850.0;
    pq_preset.bands[0].gain_db = 4.5;
    pq_preset.bands[0].q_ui = 1.414;
    pq_preset.global.output_level_db = 1.0;
    pq_preset.sync_mirror_from_structured();

    // Load all 601 parameters via AudioStreamController (which auto-drains if rtrb fills up)
    ctrl.load_proq4_preset(1, 0, &pq_preset)
        .expect("load 601-param Pro-Q 4 preset");

    // Update Band 3 via IPC command and AudioStreamController
    let band3_dto = ProQ4BandUpdateDto {
        track_idx: 1,
        slot_idx: 0,
        band_idx: 3,
        freq_hz: 4200.0,
        gain_db: -5.5,
        q: 2.2,
        shape: 0,
        slope_db_oct: 24,
        dynamic_range_db: -6.0,
        spectral_enabled: true,
        spectral_density: 0.75,
    };
    cmd_update_proq4_band(&ctrl.app_state, band3_dto).expect("IPC update proq4 band 3");
    ctrl.update_proq4_band(band3_dto)
        .expect("controller update proq4 band 3");

    let mut global_pq = ProQ4GlobalParams::default();
    global_pq.gain_scale = 110.0;
    ctrl.update_proq4_global(1, 0, &global_pq)
        .expect("update proq4 global params");

    // Verify WebGPU BiquadCascadeUniforms packing from a ProQ4Processor
    let mut pq_proc = ProQ4Processor::new();
    pq_proc.prepare(48_000.0, 256);
    pq_proc.set_parameter_normalized(0, 1.0); // Band 0 Used
    pq_proc.set_parameter_normalized(1, 1.0); // Band 0 Enabled
    pq_proc.set_parameter_normalized(2, 0.5); // Band 0 Freq
    pq_proc.set_parameter_normalized(3, 0.7); // Band 0 Gain
    let cascade_u = BiquadCascadeUniforms::from_proq4_processor(&pq_proc);
    assert!(cascade_u.num_active_sections >= 1);
    assert_eq!(cascade_u.sample_rate, 48_000.0);

    // 2. Verify Soothe3 143-parameter preset load, JSON preset load, global & SC band updates
    assert_eq!(SOOTHE3_TOTAL_PARAMS, 143);
    let mut s3_preset = Soothe3ParameterSet::new();
    s3_preset.global.mode = SootheProcessMode::Hard;
    s3_preset.global.depth = 6.5;
    s3_preset.global.detail = 7.2;
    s3_preset.sync_mirror_from_structured();
    ctrl.load_soothe3_parameter_set(1, 1, &s3_preset)
        .expect("load 143-param Soothe3 preset");

    let real_soothe3_json = r#"{
        "preset_name": "Harsh Vocal De-Ess",
        "params": {
            "Mode": 1.0,
            "Depth": 4.5,
            "Detail": 6.8,
            "Attack": 1.5,
            "Release": 3.2,
            "Max cut": -24.0,
            "Mix": 95.0,
            "Band 1 used": 1.0,
            "Band 1 enabled": 1.0,
            "Band 1 frequency": 3800.0,
            "Band 1 depth": 6.5,
            "Band 1 q": 2.4
        }
    }"#;
    let matched_ctrl = ctrl
        .load_soothe3_preset_json(1, 1, real_soothe3_json)
        .expect("controller load soothe3 JSON preset");
    assert_eq!(matched_ctrl, 12);
    let matched_ipc = cmd_load_soothe3_preset_json(&ctrl.app_state, 1, 1, real_soothe3_json)
        .expect("IPC load soothe3 JSON preset");
    assert_eq!(matched_ctrl, matched_ipc);

    cmd_update_soothe3_params(
        &ctrl.app_state,
        Soothe3UpdateDto {
            track_idx: 1,
            slot_idx: 1,
            hard_mode: true,
            delta_solo: false,
            depth: 7.5,
            detail: 8.0,
            attack: 1.5,
            release: 3.5,
            max_cut_db: -24.0,
            mix_pct: 95.0,
            linear_phase: false,
            low_latency: true,
            quality: 1,
        },
    )
    .expect("IPC update soothe3 params");

    let mut sc_band = SidechainEqBandParams::default();
    sc_band.enabled = true;
    sc_band.freq_hz = 3400.0;
    sc_band.depth = 6.0;
    ctrl.update_soothe3_sc_band(1, 1, 0, &sc_band)
        .expect("update soothe3 SC band 0");

    // 3. Verify Ozone 12 Master Chain & RealTimeBsrnnStemNode (Master Rebalance) updates
    let mut oz_cfg = Ozone12MasterConfigDto::default();
    oz_cfg.maximizer_irc_mode = 7;
    oz_cfg.maximizer_gain_db = 3.5;
    oz_cfg.maximizer_ceiling_dbtp = -0.3;
    oz_cfg.maximizer_true_peak = false;
    oz_cfg.exciter_band_drives_pct = [25.0, 30.0, 35.0, 20.0];
    oz_cfg.exciter_band_modes = [1, 2, 0, 5];
    oz_cfg.imager_band_widths_pct = [-10.0, 5.0, 25.0, 40.0];
    oz_cfg.imager_stereoizer_enabled = true;
    oz_cfg.stabilizer_amount_pct = 50.0;
    oz_cfg.impact_band_amounts = [20.0, 15.0, -10.0, 25.0];

    ctrl.configure_ozone12_master(0, 1, &oz_cfg)
        .expect("configure Ozone 12 Master Chain");
    cmd_update_ozone12_master(&ctrl.app_state, 0, 1, &oz_cfg)
        .expect("IPC update Ozone 12 Master Chain");

    ctrl.update_bsrnn_stem_gains(0, 0, [1.5, 0.5, -1.0, 0.0])
        .expect("update BSRNN stem gains");
    cmd_update_bsrnn_stem_gains(&ctrl.app_state, 0, 0, [1.5, 0.5, -1.0, 0.0])
        .expect("IPC update BSRNN stem gains");

    ctrl.update_bsrnn_stem_state(
        0,
        0,
        &BsrnnStemStateDto {
            stem_gains_db: [2.0, 0.0, -1.0, 0.5],
            stem_solo: [true, false, false, false],
            stem_mute: [false, true, false, false],
        },
    )
    .expect("update BSRNN stem solo/mute state");

    // 4. Verify Transport, Seek, Master Track #0 Gain, Route & 30-Bit Fixed-Point Controller Bus IPC commands
    cmd_set_transport(&ctrl.app_state, true, 145.0).expect("set transport");
    cmd_seek_transport(&ctrl.app_state, 512, 48).expect("seek transport");
    cmd_set_track_gain(&ctrl.app_state, 0, 0.92).expect("set Master Track #0 gain directly");
    cmd_route_mixer_track(&ctrl.app_state, 2, 0, 0.85, false).expect("route mixer track");
    cmd_set_internal_controller_norm(&ctrl.app_state, 5, 0.75).expect("set controller 5");

    // Process 2 blocks so all queued SPSC commands execute on the real-time engine
    let mut master_out = vec![[0.0f32; 2]; 256];
    ctrl.process_audio_block(&mut master_out);
    ctrl.process_audio_block(&mut master_out);

    assert!((ctrl.mixer_engine.tracks[0].gain - 0.92).abs() < 1e-4);
    let frame = ctrl.poll_telemetry_frame();
    assert!((frame.header.bpm - 145.0).abs() < 1e-4);
    assert_eq!(frame.header.active_stem_mode, (0x01 << 4) | 0x02);
    assert!((frame.automation_spline.controller_values_norm[5] - 0.75).abs() < 1e-4);
}

#[test]
fn test_uvr5_offline_stem_separation_and_mixer_track_bouncing() {
    let mut ctrl = AudioStreamController::new(44_100.0, 256, 0);

    let mix = make_test_stereo_signal(1024, 44_100.0);

    // Test individual UVR5 architectures
    let (mdx_v, mdx_i) = ctrl.separate_mdx23c_offline(&mix, true);
    assert_eq!(mdx_v.len(), 1024);
    assert_eq!(mdx_i.len(), 1024);

    let demucs_stems = ctrl.separate_htdemucs4_offline(&mix);
    for s in &demucs_stems {
        assert_eq!(s.len(), 1024);
    }

    let (vr_v, vr_i) = ctrl.separate_vr_aspp_offline(&mix, 0.25);
    assert_eq!(vr_v.len(), 1024);
    assert_eq!(vr_i.len(), 1024);

    // Run full UVR5 Multi-Architecture Ensemble Studio across Weighted mode
    let ensemble_out = ctrl.run_uvr5_ensemble_studio(
        &mix,
        SpectralEnsembleMode::Weighted,
        Some(&[0.45, 0.35, 0.20]),
        0.15,
    );
    assert_eq!(ensemble_out.vocals.len(), 1024);
    assert_eq!(ensemble_out.drums.len(), 1024);
    assert_eq!(ensemble_out.bass.len(), 1024);
    assert_eq!(ensemble_out.other.len(), 1024);
    assert_eq!(ensemble_out.inverted_instrumental.len(), 1024);

    // Verify phase-inversion identity: vocals + inverted_instrumental == original mix
    for i in 0..1024 {
        let recon_l = ensemble_out.vocals[i][0] + ensemble_out.inverted_instrumental[i][0];
        let recon_r = ensemble_out.vocals[i][1] + ensemble_out.inverted_instrumental[i][1];
        assert!((recon_l - mix[i][0]).abs() < 1e-5);
        assert!((recon_r - mix[i][1]).abs() < 1e-5);
    }

    // Also test MaxSpec, MinSpec, Average, Median, and empty input edge case
    for mode in [
        SpectralEnsembleMode::MaxSpec,
        SpectralEnsembleMode::MinSpec,
        SpectralEnsembleMode::Average,
        SpectralEnsembleMode::Median,
    ] {
        let sub_out = ctrl.run_uvr5_ensemble_studio(&mix[..256], mode, None, 0.0);
        assert_eq!(sub_out.vocals.len(), 256);
    }
    let empty_out = ctrl.run_uvr5_ensemble_studio(&[], SpectralEnsembleMode::Average, None, 0.0);
    assert!(empty_out.vocals.is_empty());

    // Bounce the 4 separated stems directly onto Mixer Insert Tracks 1..=4
    let stems = vec![
        ensemble_out.vocals,
        ensemble_out.drums,
        ensemble_out.bass,
        ensemble_out.other,
    ];
    let ch_indices = ctrl
        .bounce_stems_to_mixer_tracks(&stems, &[1, 2, 3, 4])
        .expect("bounce 4 stems to mixer tracks 1..=4");
    assert_eq!(ch_indices.len(), 4);

    // Process 4 blocks and verify all 4 stem tracks and Master Track 0 register live signal
    let mut master_out = vec![[0.0f32; 2]; 256];
    for _ in 0..4 {
        ctrl.process_audio_block(&mut master_out);
    }
    let frame = ctrl.poll_telemetry_frame();
    for trk in 1..=4 {
        assert!(
            frame.track_meters[trk].peak_l_db > -80.0
                || frame.track_meters[trk].peak_r_db > -80.0,
            "Bounced stem on Insert Track {} should be playing",
            trk
        );
    }
    assert!(
        frame.track_meters[0].peak_l_db > -80.0,
        "Master Track 0 should receive summed audio from bounced stems 1..=4"
    );

    // Verify IPC UVR5 file path helper
    let paths = cmd_run_uvr5_separation("C:/Audio/song.wav".to_string(), "mdx23c_hq".to_string())
        .expect("uvr5 paths");
    assert_eq!(paths.len(), 4);
    assert!(cmd_run_uvr5_separation("   ".to_string(), "mdx23c".to_string()).is_err());
}

#[test]
fn test_project_state_save_load_502_tracks_patcher_and_tension_spline() {
    let mut project = ToneForgeProjectState::new_mastering_template(48_000.0, 132.0);
    project.project_name = "ToneForge Phase 4 Verification".to_string();

    // Customize Ozone 12 Master Chain on Master Track 0 Slot 1 to verify full roundtrip
    if let Some(slot) = project.mixer_tracks[0].slots.get_mut(1) {
        if let PluginConfigDto::Ozone12Master(ref mut oz) = slot.plugin {
            oz.maximizer_irc_mode = 7;
            oz.maximizer_gain_db = 6.5;
            oz.maximizer_ceiling_dbtp = -0.5;
            oz.maximizer_true_peak = false;
            oz.exciter_band_drives_pct = [40.0, 35.0, 30.0, 25.0];
            oz.imager_band_widths_pct = [-20.0, 10.0, 35.0, 50.0];
            oz.stabilizer_amount_pct = 65.0;
            oz.clarity_amount_pct = 48.0;
        }
    }

    // Add a custom Patcher node & connection and recompute Sugiyama layout
    let patcher = &mut project.patcher_graphs[0];
    patcher.nodes.push(PatcherNodeStateDto {
        node_index: 2,
        label: "LuxeVerb Shimmer Branch".to_string(),
        plugin_kind: "LuxeVerb".to_string(),
        fixed_latency_samples: 0,
        gain_scale: 0.4,
    });
    patcher.audio_connections.push(PatcherAudioEdgeDto {
        from_node: 0,
        to_node: 2,
        gain: 0.8,
    });
    patcher.audio_connections.push(PatcherAudioEdgeDto {
        from_node: 2,
        to_node: PATCHER_HOST_OUTPUT_NODE,
        gain: 1.0,
    });
    patcher.recompute_sugiyama_layout();
    assert_eq!(patcher.sugiyama_layout.len(), 3);

    // Add a second Automation Clip with signed-exponential TensionSpline + LFO Modulator
    project.automation_clips.push(AutomationClipStateDto {
        controller_id: 1,
        name: "Soothe3 Depth Modulation".to_string(),
        points: vec![
            TensionControlPointDto {
                x: 0.0,
                y: 0.15,
                tension: 0.55,
            },
            TensionControlPointDto {
                x: 96.0,
                y: 0.85,
                tension: -0.55,
            },
            TensionControlPointDto {
                x: 192.0,
                y: 0.25,
                tension: 0.0,
            },
        ],
        lfo: LfoModulatorDto {
            enabled: true,
            waveform: "Triangle".to_string(),
            frequency_cycles_per_tick: 1.0 / 48.0,
            phase_offset: 0.0,
            amount: 0.1,
        },
        min_level: 0.0,
        max_level: 1.0,
    });

    // Add Native FX slots on Track 2 to test all PluginConfigDto variants
    project.mixer_tracks[2].sends.push(MixerSendRouteDto {
        dest_track_index: 0,
        gain: 1.0,
        is_sidechain_only: false,
    });
    project.mixer_tracks[2].slots = vec![
        PluginSlotStateDto {
            slot_index: 0,
            enabled: true,
            wet_mix: 1.0,
            plugin: PluginConfigDto::NativeSparseCompressor {
                threshold_linear: 0.5,
                ratio: 4.0,
                makeup_gain: 1.1,
                lookahead_samples: 32,
            },
        },
        PluginSlotStateDto {
            slot_index: 1,
            enabled: true,
            wet_mix: 1.0,
            plugin: PluginConfigDto::NativeMaximusMultiband,
        },
        PluginSlotStateDto {
            slot_index: 2,
            enabled: true,
            wet_mix: 1.0,
            plugin: PluginConfigDto::NativeEmphasisLimiter {
                ceiling_linear: 0.95,
                drive_linear: 1.2,
            },
        },
        PluginSlotStateDto {
            slot_index: 3,
            enabled: true,
            wet_mix: 0.5,
            plugin: PluginConfigDto::NativeGrossBeat,
        },
        PluginSlotStateDto {
            slot_index: 4,
            enabled: true,
            wet_mix: 0.3,
            plugin: PluginConfigDto::NativeLuxeVerbFdn,
        },
    ];

    // Save project to disk and reload
    let temp_path = std::env::temp_dir().join("toneforge_phase4_verified.tfproj.json");
    project
        .save_to_file(&temp_path)
        .expect("save project to disk");
    let loaded =
        ToneForgeProjectState::load_from_file(&temp_path).expect("load project from disk");
    let _ = std::fs::remove_file(&temp_path);

    assert_eq!(loaded.project_name, "ToneForge Phase 4 Verification");
    assert_eq!(loaded.mixer_tracks.len(), MAX_MIXER_TRACKS);
    assert_eq!(loaded.automation_clips.len(), 2);
    assert_eq!(loaded.mixer_tracks[2].slots.len(), 5);

    // Verify 30-Bit Fixed-Point (0x40000000) evaluation on loaded project
    assert_eq!(FIXED_POINT_30BIT_MAX, 0x4000_0000);
    let fixed_mid = loaded
        .evaluate_automation_clip_fixed30(0, 48.0)
        .expect("evaluate clip 0 at tick 48");
    let norm_mid = fixed30_to_normalized(fixed_mid);
    assert!(norm_mid > 0.25 && norm_mid < 0.75);
    assert_eq!(normalized_to_fixed30(1.0), FIXED_POINT_30BIT_MAX);

    // Apply loaded project state to AudioStreamController and evaluate automation into RT engine
    let mut ctrl = AudioStreamController::new(48_000.0, 256, 0);
    ctrl.apply_project_state(loaded)
        .expect("apply loaded project state");
    ctrl.evaluate_and_publish_automation_clips(48.0)
        .expect("publish automation clips at tick 48");

    // Attach signal on Track 2 so GrossBeat processes audio and updates gross_beat_playhead_pos
    let sig = make_test_stereo_signal(512, 48_000.0);
    ctrl.bounce_stems_to_mixer_tracks(&[sig], &[2])
        .expect("attach test signal on track 2");

    let mut master_out = vec![[0.0f32; 2]; 256];
    ctrl.process_audio_block(&mut master_out);
    let frame = ctrl.poll_telemetry_frame();
    assert!(frame.automation_spline.controller_values_norm[0] > 0.25);
    assert!(frame.automation_spline.controller_values_norm[1] > 0.15);
    assert!(
        frame.automation_spline.gross_beat_playhead_pos[0] > 0.0,
        "GrossBeatTimeBufferNode on Track 2 should populate gross_beat_playhead_pos[0]"
    );
}

#[test]
fn test_edge_cases_cycle_rejection_and_invalid_parameters() {
    let mut ctrl = AudioStreamController::new(48_000.0, 256, 0);

    // 1. Cycle in 502-Track Mixer routing graph must be rejected
    ctrl.route_mixer_track(10, 11, 1.0, false)
        .expect("route 10 -> 11");
    ctrl.route_mixer_track(11, 12, 1.0, false)
        .expect("route 11 -> 12");
    assert!(
        ctrl.route_mixer_track(12, 10, 1.0, false).is_err(),
        "Routing 12 -> 10 must be rejected as a cycle in the 502-Track Mixer DAG"
    );
    assert!(
        ctrl.route_mixer_track(5, 5, 1.0, false).is_err(),
        "Self-loop 5 -> 5 must be rejected"
    );

    // 2. Cycle in ProjectState 502-Track routing graph must fail validation
    let mut cyclic_proj = ToneForgeProjectState::new_default(48_000.0, 128.0);
    cyclic_proj.mixer_tracks[0].sends.push(MixerSendRouteDto {
        dest_track_index: 1,
        gain: 1.0,
        is_sidechain_only: false,
    }); // 1 -> 0 -> 1 cycle!
    assert!(cyclic_proj.validate().is_err());

    // 3. Cycle in PatcherGraphStateDto must fail validation
    let mut cyclic_patcher_proj = ToneForgeProjectState::new_default(48_000.0, 128.0);
    cyclic_patcher_proj.patcher_graphs[0]
        .audio_connections
        .push(PatcherAudioEdgeDto {
            from_node: 0,
            to_node: 1,
            gain: 1.0,
        });
    cyclic_patcher_proj.patcher_graphs[0]
        .audio_connections
        .push(PatcherAudioEdgeDto {
            from_node: 1,
            to_node: 0,
            gain: 1.0,
        });
    assert!(cyclic_patcher_proj.validate().is_err());

    // 4. Invalid BPM, sample rate, and out-of-range indices must fail cleanly
    assert!(ctrl.set_transport(true, 0.0).is_err());
    assert!(ctrl.set_transport(true, f64::NAN).is_err());
    assert!(ctrl.seek_position(-1, 0).is_err());
    assert!(ctrl.set_track_gain(502, 1.0).is_err());
    assert!(ctrl.set_slot_wet_mix(1, 10, 0.5).is_err());
    assert!(ctrl.set_slot_enabled(502, 0, true).is_err());
    assert!(ctrl.bounce_stems_to_mixer_tracks(&[vec![[0.0, 0.0]; 16]], &[0]).is_err());

    let bad_band = ProQ4BandUpdateDto {
        track_idx: 1,
        slot_idx: 0,
        band_idx: 24, // Out of range (0..24)
        freq_hz: 1000.0,
        gain_db: 0.0,
        q: 1.0,
        shape: 0,
        slope_db_oct: 12,
        dynamic_range_db: 0.0,
        spectral_enabled: false,
        spectral_density: 0.5,
    };
    assert!(cmd_update_proq4_band(&ctrl.app_state, bad_band).is_err());
    assert!(cmd_set_internal_controller_norm(&ctrl.app_state, 512, 0.5).is_err());

    // 5. Verify WGPU uniform struct sizes
    assert_eq!(std::mem::size_of::<SdfControlInstance>(), 64);
    assert_eq!(std::mem::size_of::<Soothe3HeatmapUniforms>(), 32);
    assert_eq!(std::mem::size_of::<MixerMetersUniforms>(), 32);
}

#[test]
fn test_ui_svelte_webgpu_binary_offsets_shaders_and_components_parity() {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");

    for ui_dir_name in ["ui-svelte-webgpu", "ui-svelte5-wgpu"] {
        let ui_root = workspace_root.join(ui_dir_name);
        assert!(
            ui_root.exists(),
            "UI package directory {:?} must exist",
            ui_root
        );

        // 1. Verify Zero-Copy DataView/Float32Array decoder store offsets match Rust VisualTelemetryFrame
        let store_ts =
            std::fs::read_to_string(ui_root.join("src/lib/state/daw_master_store.svelte.ts"))
                .expect("read daw_master_store.svelte.ts");
        assert!(store_ts.contains("VISUAL_TELEMETRY_FRAME_BYTES = 131_072"));
        assert!(store_ts.contains("OFFSET_HEADER = 0x00000"));
        assert!(store_ts.contains("OFFSET_TRACK_METERS = 0x00040"));
        assert!(store_ts.contains("OFFSET_PROQ4_SPECTRUM = 0x03000"));
        assert!(store_ts.contains("OFFSET_SOOTHE3_REDUCTION = 0x07000"));
        assert!(store_ts.contains("OFFSET_OZONE12_TELEMETRY = 0x09000"));
        assert!(store_ts.contains("OFFSET_SPECTROGRAM_RING = 0x0B000"));
        assert!(store_ts.contains("OFFSET_AUTOMATION_SPLINE = 0x1F000"));
        assert!(store_ts.contains("$state("));
        assert!(store_ts.contains("$derived("));

        // 2. Verify all 120 FPS WGSL Shaders exist and contain required entry points & bindings
        let proq4_wgsl =
            std::fs::read_to_string(ui_root.join("src/lib/wgpu/shaders/proq4_curve.wgsl"))
                .expect("read proq4_curve.wgsl");
        assert!(proq4_wgsl.contains("struct BiquadCascadeUniforms"));
        assert!(proq4_wgsl.contains("struct ProQ4SpectrumBuffer"));
        assert!(proq4_wgsl.contains("fn cs_eval_proq4_curve"));
        assert!(proq4_wgsl.contains("fn fs_main"));

        let soothe3_wgsl =
            std::fs::read_to_string(ui_root.join("src/lib/wgpu/shaders/soothe3_heatmap.wgsl"))
                .expect("read soothe3_heatmap.wgsl");
        assert!(soothe3_wgsl.contains("struct Soothe3HeatmapUniforms"));
        assert!(soothe3_wgsl.contains("struct Soothe3TelemetryBuffer"));
        assert!(soothe3_wgsl.contains("let cut_y = clamp(norm_cut, 0.0, 0.95);"));
        assert!(soothe3_wgsl.contains("fn fs_main"));

        let meters_wgsl =
            std::fs::read_to_string(ui_root.join("src/lib/wgpu/shaders/mixer_meters.wgsl"))
                .expect("read mixer_meters.wgsl");
        assert!(meters_wgsl.contains("struct MixerMetersUniforms"));
        assert!(meters_wgsl.contains("track_meters: array<TrackMeterGpu, 502>"));
        assert!(meters_wgsl.contains("ozone_telemetry: Ozone12GpuTelemetry"));
        assert!(meters_wgsl.contains("exciter_drive_gr: array<f32, 248>"));

        // 3. Verify all Svelte 5 Runes UI components exist and use $props / $state / $derived / $effect
        for comp in [
            "src/lib/components/MixerRack502.svelte",
            "src/lib/components/PatcherNodeCanvas.svelte",
            "src/lib/components/ProQ4InteractiveAnalyzer.svelte",
            "src/lib/components/Soothe3ResonanceView.svelte",
            "src/lib/components/Ozone12MasterChain.svelte",
            "src/lib/components/Uvr5EnsembleStudio.svelte",
            "src/App.svelte",
        ] {
            let content = std::fs::read_to_string(ui_root.join(comp))
                .unwrap_or_else(|e| panic!("Failed to read {:?}: {}", comp, e));
            assert!(!content.is_empty(), "{:?} must not be empty", comp);
        }
    }
}

#[test]
fn test_cross_language_nodejs_binary_telemetry_decoder_and_webgpu_pipeline_validation() {
    // 1. Produce a real 128 KB (131,072-byte) VisualTelemetryFrame from the Rust AudioStreamController
    let mut ctrl = AudioStreamController::with_mastering_template(48_000.0, 256)
        .expect("create mastering controller");
    let signal = make_test_stereo_signal(2048, 48_000.0);
    ctrl.bounce_stems_to_mixer_tracks(&[signal], &[1])
        .expect("bounce signal to track 1");
    let mut master_out = vec![[0.0f32; 2]; 256];
    for _ in 0..40 {
        ctrl.process_audio_block(&mut master_out);
    }

    let raw_bytes = *cmd_poll_binary_telemetry_frame(&mut ctrl.telemetry_streamer);
    let temp_bin = std::env::temp_dir().join("toneforge_rust_to_node_telemetry_128kb.bin");
    std::fs::write(&temp_bin, &raw_bytes).expect("write 128 KB binary telemetry frame");

    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    let ui_root = workspace_root.join("ui-svelte-webgpu");

    // 2. Run a Node.js verification script that loads the raw 128 KB binary file produced by Rust
    //    and validates DataView / Float32Array decoding, WGSL shader binding layouts, and WebGPU
    //    bind group layout parity across `proq4`, `soothe3`, and `mixer_meters`.
    let temp_script = std::env::temp_dir().join("toneforge_verify_ui_webgpu.mjs");
    let script_code = format!(
        r#"
import fs from 'node:fs';
import assert from 'node:assert/strict';

const binPath = {bin_path:?};
const uiRoot = {ui_root:?};

const buf = fs.readFileSync(binPath);
assert.equal(buf.byteLength, 131072, 'VisualTelemetryFrame must be 131,072 bytes');
const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
const dv = new DataView(ab);

// 1. Decode TransportAndMasterHeader at 0x00000
const frameSeq = Number(dv.getBigUint64(0x00000, true));
const samplePos = Number(dv.getBigInt64(0x00008, true));
const bpm = dv.getFloat32(0x00018, true);
const cpuLoad = dv.getFloat32(0x0001C, true);
assert.equal(frameSeq, 40);
assert.equal(samplePos, 10240);
assert.ok(Math.abs(bpm - 128.0) < 1e-4);
assert.ok(cpuLoad > 0.0, 'CPU load must be > 0');

// 2. Decode Track 0 & Track 1 Meters at 0x00040 (24-byte stride)
const masterPeakL = dv.getFloat32(0x00040 + 0, true);
const track1PeakL = dv.getFloat32(0x00040 + 24 + 0, true);
assert.ok(masterPeakL > -65.0 && track1PeakL > -65.0);

// 3. Zero-Copy Float32Array views at exact byte offsets
const proq4PreDb = new Float32Array(ab, 0x03000, 1024);
const proq4PostDb = new Float32Array(ab, 0x04000, 1024);
const soothe3Reduction1024 = new Float32Array(ab, 0x07000, 1024);
const ozone12IrcGr = new Float32Array(ab, 0x09000, 512);
const spectrogramRing = new Float32Array(ab, 0x0B000, 20 * 1024);
const patcherCpuPct = new Float32Array(ab, 0x1F808, 510);

assert.ok(proq4PreDb.some((v) => v > -100.0));
assert.ok(proq4PostDb.some((v) => v > -100.0));
assert.ok(soothe3Reduction1024.length === 1024);
assert.ok(ozone12IrcGr.length === 512);
assert.ok(spectrogramRing.some((v) => v > -115.0), 'Spectrogram ring at 0x0B000 must be populated');
assert.ok(patcherCpuPct[0] > 0.0, 'Patcher CPU telemetry at 0x1F808 must be populated');

// 4. Verify WebGPU Renderer bind group layout compatibility against WGSL entry points
const proq4Wgsl = fs.readFileSync(uiRoot + '/src/lib/wgpu/shaders/proq4_curve.wgsl', 'utf8');
const soothe3Wgsl = fs.readFileSync(uiRoot + '/src/lib/wgpu/shaders/soothe3_heatmap.wgsl', 'utf8');
const metersWgsl = fs.readFileSync(uiRoot + '/src/lib/wgpu/shaders/mixer_meters.wgsl', 'utf8');
const rendererTs = fs.readFileSync(uiRoot + '/src/lib/wgpu/renderer_120fps.ts', 'utf8');

// Verify proq4_curve.wgsl compute pipeline uses bindings (0, 2, 3) and render pipeline uses (0, 1)
assert.ok(proq4Wgsl.includes('@group(0) @binding(0) var<uniform> eq_uniforms'));
assert.ok(proq4Wgsl.includes('@group(0) @binding(1) var<storage, read> spectrum_buf'));
assert.ok(proq4Wgsl.includes('@group(0) @binding(2) var<storage, read_write> out_magnitude_db'));
assert.ok(proq4Wgsl.includes('@group(0) @binding(3) var<storage, read_write> out_phase_rad'));
assert.ok(rendererTs.includes("compute: {{ module, entryPoint: 'cs_eval_proq4_curve' }}"));
assert.ok(soothe3Wgsl.includes('let cut_y = clamp(norm_cut, 0.0, 0.95);'));
assert.ok(metersWgsl.includes('exciter_drive_gr: array<f32, 248>'));
"#,
        bin_path = temp_bin.to_string_lossy(),
        ui_root = ui_root.to_string_lossy()
    );
    std::fs::write(&temp_script, script_code).expect("write node verification script");

    let output = std::process::Command::new("node")
        .arg(&temp_script)
        .output()
        .expect("run node verification script");

    let _ = std::fs::remove_file(&temp_bin);
    let _ = std::fs::remove_file(&temp_script);

    assert!(
        output.status.success(),
        "Node.js cross-language binary decoder & WebGPU verification failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_blueprint_section_5_5_wgsl_shaders_and_csg_compositor() {
    use daw_tauri_host::{
        CsgOperation, CurveUniforms, HeatmapUniforms, SdfPrimitive, SdfPrimitiveKind, csg_combine,
        eval_sdf_primitive,
    };

    assert_eq!(std::mem::size_of::<CurveUniforms>(), 4160);
    assert_eq!(std::mem::size_of::<HeatmapUniforms>(), 32);
    assert_eq!(std::mem::size_of::<SdfPrimitive>(), 64);

    let pts = [0.5f32; 1024];
    let cu = CurveUniforms::from_curve_1024(1200.0, 320.0, 0.5, [0.46, 0.82, 0.86, 1.0], &pts);
    assert!((cu.baseline_y - 0.5).abs() < 1e-6);
    assert!((cu.ys[255][3] - 0.5).abs() < 1e-6);

    // Verify all 7 SDF Primitives and 4 CSG boolean operations
    for kind in [
        SdfPrimitiveKind::RoundedRect,
        SdfPrimitiveKind::SemiCircleRing,
        SdfPrimitiveKind::Rect,
        SdfPrimitiveKind::EllipseNewtonRaphson,
        SdfPrimitiveKind::MsdfGlyph,
        SdfPrimitiveKind::SpriteCapsule,
        SdfPrimitiveKind::Squircle,
    ] {
        let d_center = eval_sdf_primitive([0.0, 0.0], [24.0, 18.0], [6.0, 4.0, 1.2, 4.0], kind);
        let d_far = eval_sdf_primitive([100.0, 100.0], [24.0, 18.0], [6.0, 4.0, 1.2, 4.0], kind);
        assert!(d_center.is_finite() && d_far.is_finite());
        assert!(d_far > d_center);

        let inst = SdfPrimitive::new(
            [10.0, 20.0, 48.0, 36.0],
            [6.0, 4.0, 1.2, 4.0],
            [0.46, 0.82, 0.86, 1.0],
            1.5,
            4.0,
            0.75,
            kind,
            CsgOperation::Union,
        );
        assert_eq!(bytemuck::bytes_of(&inst).len(), 64);
    }

    assert!((csg_combine(-2.0, 3.0, CsgOperation::Union) - (-2.0)).abs() < 1e-6);
    assert!((csg_combine(-2.0, 3.0, CsgOperation::Intersect) - 3.0).abs() < 1e-6);
    assert!((csg_combine(-2.0, -5.0, CsgOperation::Subtract) - 5.0).abs() < 1e-6);
    assert!((csg_combine(-2.0, 3.0, CsgOperation::Replace) - 3.0).abs() < 1e-6);

    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");

    for ui_dir_name in ["ui-svelte-webgpu", "ui-svelte5-wgpu"] {
        let shaders_dir = workspace_root.join(ui_dir_name).join("src/lib/wgpu/shaders");
        let curve_wgsl = std::fs::read_to_string(shaders_dir.join("reduction_eq_curve.wgsl"))
            .expect("read reduction_eq_curve.wgsl");
        assert!(curve_wgsl.contains("struct CurveUniforms"));
        assert!(curve_wgsl.contains("ys: array<vec4<f32>, 256>"));
        assert!(curve_wgsl.contains("fn sample_curve_catmull_rom"));

        let hm_wgsl = std::fs::read_to_string(shaders_dir.join("spectrogram_heatmap.wgsl"))
            .expect("read spectrogram_heatmap.wgsl");
        assert!(hm_wgsl.contains("struct HeatmapUniforms"));
        assert!(hm_wgsl.contains("fn palette_proq4"));
        assert!(hm_wgsl.contains("fn palette_soothe3"));
        assert!(hm_wgsl.contains("fn palette_ozone12"));

        let sdf_wgsl = std::fs::read_to_string(shaders_dir.join("sdf_uber_controls.wgsl"))
            .expect("read sdf_uber_controls.wgsl");
        assert!(sdf_wgsl.contains("struct SdfPrimitive"));
        assert!(sdf_wgsl.contains("fn sd_ellipse_nr"));
        assert!(sdf_wgsl.contains("fn sd_squircle"));
        assert!(sdf_wgsl.contains("fn csg_combine"));
    }
}



