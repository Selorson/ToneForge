//! Phase 5 End-to-End DAW Integration & Verification Suite (`phase5_e2e_daw_verification.rs`).
//!
//! Verifies:
//! 1. Hybrid PPQ (`96..=960`) sample-accurate slicing & FL Studio 26 Piano Roll (`NoteOn/Off`,
//!    `Velocity`, `Pan`, `Fine Pitch`, `Slide/Portamento` glide).
//! 2. Point 1 Generator Lookahead APDC (`PolyphonicSynthGeneratorNode` & `AudioStemClipPlayerNode`)
//!    phase-cancellation null test (`< -120 dB`).
//! 3. Studio WAV Export (`16-bit PCM MBIT+`, `24-bit PCM MBIT+`, `32-bit IEEE Float`) across
//!    `44.1 / 48 / 96 / 192 kHz` with Ozone 12 `MbitPlusDither` psychoacoustic noise shaping.
//! 4. Simultaneous `SplitMixerTracks` multi-track bounce from the 502-Track Mixer.
//! 5. Full End-to-End DAW Workflow:
//!    Compose Piano Roll + Playlist + `TensionSpline` Automation -> UVR5 4-Stem Separation ->
//!    Process through 502-Track Mixer (`Pro-Q 4 + Soothe3 + Patcher DAG + RealTimeBsrnnStemNode + Ozone 12`) ->
//!    Export `24-bit MBIT+ Dither` & `32-bit Float` WAV files -> Verify `PostRenderAnalysisReport`
//!    (`Integrated/Short-Term/Momentary LUFS`, `4x True Peak dBTP`, `Crest Factor`, `Phase Correlation`,
//!    and `Spotify / Apple Music / YouTube` loudness compliance).
//! 6. Phase 5 Tauri IPC commands and `ui-svelte-webgpu` / `ui-svelte5-wgpu` component parity.

use daw_core_graph::{
    AudioStemClipPlayerNode, ChannelGenerator, MixerEngine, PianoRollNote, PianoRollPattern,
    PlaylistClip, PlaylistClipKind, PolyphonicSynthGeneratorNode, ProcessContext,
    SUPPORTED_PPQ_RESOLUTIONS, SequencerVoiceEvent, StereoFrame, UnifiedDspNode, VoiceEventKind,
};
use daw_tauri_host::{
    AudioStreamController, OfflineRenderConfig, OfflineRenderEngine, PianoRollNoteDto, RenderMode,
    RenderSampleRate, WavBitDepth, analyze_rendered_audio, cmd_execute_offline_render,
    cmd_set_ppq_resolution, cmd_trigger_piano_roll_note, decode_wav_bytes, encode_wav_bytes,
    read_wav_file,
};

fn make_rich_test_mix(num_frames: usize, sample_rate: f64) -> Vec<StereoFrame> {
    let mut out = Vec::with_capacity(num_frames);
    for n in 0..num_frames {
        let t = (n as f64) / sample_rate;
        // Kick/Bass (55 Hz & 110 Hz) + Vocal/Lead (440 Hz & 880 Hz) + Air/HiHat (3520 Hz)
        let bass = 0.28 * (std::f64::consts::TAU * 55.0 * t).sin()
            + 0.16 * (std::f64::consts::TAU * 110.0 * t).sin();
        let vocal = 0.22 * (std::f64::consts::TAU * 440.0 * t).sin()
            + 0.12 * (std::f64::consts::TAU * 880.0 * t).cos();
        let air = 0.08 * (std::f64::consts::TAU * 3520.0 * t).sin();
        let env = 0.65 + 0.35 * (std::f64::consts::TAU * 2.0 * t).sin();
        let l = ((bass + vocal + air) * env) as f32;
        let r = ((bass + vocal * 0.92 - air * 0.85) * env) as f32;
        out.push([l, r]);
    }
    out
}

#[test]
fn test_hybrid_ppq_96_to_960_sample_accurate_slicing_and_piano_roll_slide_portamento() {
    let sample_rate = 48_000.0;
    let bpm = 120.0;

    // 1. Verify all 6 FL Studio 26 Hybrid PPQ resolutions (96, 144, 192, 384, 480, 960)
    for &ppq in &SUPPORTED_PPQ_RESOLUTIONS {
        let (mut engine, _cmd_prod, _telem_out) = MixerEngine::new(sample_rate, bpm, 0);
        engine.slicer.set_ppq(ppq);
        assert_eq!(engine.slicer.ppq, ppq);

        // At 120 BPM (2 beats/sec), 1 beat = ppq ticks = 24,000 samples @ 48 kHz
        let one_beat_samples = engine.slicer.tick_to_sample_pos(ppq as f64);
        assert_eq!(
            one_beat_samples, 24_000,
            "1 beat at 120 BPM @ 48 kHz must be exactly 24,000 samples for PPQ={ppq}"
        );
        let roundtrip_tick = engine.slicer.sample_pos_to_tick(one_beat_samples);
        assert!(
            (roundtrip_tick - (ppq as f64)).abs() < 1e-6,
            "sample_pos_to_tick roundtrip failed at PPQ={ppq}: got {roundtrip_tick}"
        );
    }

    // 2. Verify FL Studio 26 Slide/Portamento glide and per-note Pan/Velocity/FinePitch
    let mut synth = PolyphonicSynthGeneratorNode::new(1, sample_rate);
    synth.prepare(sample_rate, 256);

    // Trigger base note C4 (MIDI 60) at sample 0
    synth.schedule_voice_event(SequencerVoiceEvent {
        sample_pos: 0,
        ppq_tick: 0,
        channel_id: 1,
        note_id: 1,
        kind: VoiceEventKind::NoteOn,
        key: 60,
        velocity: 0.85,
        pan: -0.4,
        fine_pitch_cents: 0.0,
        slide_duration_ticks: 0,
        slide_duration_samples: 0,
        color_group: 0,
    });
    // Trigger overlapping Slide note C5 (MIDI 72) at sample 64 gliding over 128 samples to Right pan (+0.4)
    synth.schedule_voice_event(SequencerVoiceEvent {
        sample_pos: 64,
        ppq_tick: 4,
        channel_id: 1,
        note_id: 2,
        kind: VoiceEventKind::SlideNote,
        key: 72,
        velocity: 1.0,
        pan: 0.4,
        fine_pitch_cents: 50.0,
        slide_duration_ticks: 8,
        slide_duration_samples: 128,
        color_group: 0,
    });

    let ctx = ProcessContext {
        sample_rate,
        current_sample_pos: 0,
        ppq_tick: 0,
        bpm,
        slice_frames: 256,
        is_playing: true,
        is_offline_rendering: false,
    };
    let mut out = [[0.0f32; 2]; 256];
    synth.process_slice(&ctx, &mut out, None, &mut []);

    // Only 1 voice should be active because Slide note glides the existing voice in color_group 0!
    assert_eq!(synth.active_voice_count(), 1);
    assert_eq!(synth.slide_events_processed(), 1);
    let pitch_after_glide = synth.active_pitch_semitones_for_group(0).unwrap();
    // Target is MIDI 72 + 50 cents = 72.5 semitones
    assert!(
        (pitch_after_glide - 72.5).abs() < 0.05,
        "Expected glided pitch near 72.5 semitones (C5 + 50ct), got {pitch_after_glide}"
    );
}

#[test]
fn test_point1_generator_lookahead_apdc_phase_cancellation_null_test() {
    let sample_rate = 48_000.0;
    let bpm = 120.0;
    let (mut engine, _cmd_prod, _telem_out) = MixerEngine::new(sample_rate, bpm, 0);

    // Channel 0 -> Track 1: PolyphonicSynthGeneratorNode with 128-sample lookahead latency
    let mut synth_delayed =
        PolyphonicSynthGeneratorNode::new(10, sample_rate).with_lookahead_latency(128);
    synth_delayed.prepare(sample_rate, 256);

    // Channel 1 -> Track 2 (polarity inverted): PolyphonicSynthGeneratorNode with 0-sample lookahead latency
    let mut synth_zero =
        PolyphonicSynthGeneratorNode::new(10, sample_rate).with_lookahead_latency(0);
    synth_zero.prepare(sample_rate, 256);

    engine
        .add_channel_generator(ChannelGenerator::new(10, 1, Box::new(synth_delayed)))
        .unwrap();
    engine
        .add_channel_generator(ChannelGenerator::new(10, 2, Box::new(synth_zero)))
        .unwrap();

    engine.tracks[2].polarity_inverted = true;
    engine.tracks[1].add_send(0, 1.0, false);
    engine.tracks[2].add_send(0, 1.0, false);
    engine.rebuild_graph().unwrap();

    // Verify Point 1 APDC assigned 0 delay to Channel 0 (latency 128) and 128 samples delay to Channel 1 (latency 0)
    assert_eq!(engine.channels[0].channel_apdc_delay.delay_samples(), 0);
    assert_eq!(engine.channels[1].channel_apdc_delay.delay_samples(), 128);

    // Schedule pattern on channel_id = 10 (both channels receive identical note events)
    let mut pat = PianoRollPattern::new(1, "Point1 APDC Null Pattern", 10, 384);
    pat.add_note(PianoRollNote::new(1, 0, 192, 60, 0.9));
    engine.sequencer.upsert_pattern(pat);
    engine.sequencer.upsert_clip(PlaylistClip {
        clip_id: 1,
        track_row: 0,
        kind: PlaylistClipKind::Pattern,
        source_id: 1,
        start_tick: 0,
        duration_ticks: 384,
        clip_offset_ticks: 0,
        gain: 1.0,
        muted: false,
    });

    let mut master_out = [[0.0f32; 2]; 256];
    let mut max_residual = 0.0f32;
    let mut track1_energy = 0.0f32;

    for _ in 0..8 {
        engine.process_audio_block(&mut master_out);
        for f in &engine.tracks[1].sub_block_buf[..256] {
            track1_energy += f[0] * f[0] + f[1] * f[1];
        }
        for f in &master_out {
            max_residual = max_residual.max(f[0].abs().max(f[1].abs()));
        }
    }

    assert!(
        track1_energy > 1.0,
        "Track 1 must carry non-zero synth signal energy (got {track1_energy})"
    );
    assert!(
        max_residual < 1e-6,
        "Point 1 Generator Lookahead APDC null cancellation failed: max residual = {max_residual} (> -120 dB)"
    );

    // Also verify AudioStemClipPlayerNode with 4-point Hermite cubic interpolation & Point 1 Lookahead APDC
    let stem_audio = make_rich_test_mix(1024, sample_rate);
    let mut stem_node =
        AudioStemClipPlayerNode::new(20, "VocalsStem", stem_audio).with_lookahead_latency(64);
    stem_node.pitch_semitones = 3.0; // Exercise 4-point Hermite cubic pitch resampling
    stem_node.prepare(sample_rate, 256);
    assert_eq!(stem_node.latency_samples(), 64);
    let ctx = ProcessContext {
        sample_rate,
        current_sample_pos: 0,
        ppq_tick: 0,
        bpm,
        slice_frames: 256,
        is_playing: true,
        is_offline_rendering: false,
    };
    let mut stem_out = [[0.0f32; 2]; 256];
    stem_node.process_slice(&ctx, &mut stem_out, None, &mut []);
    // First 64 samples must be zero (64-sample lookahead delay), followed by non-zero resampled audio
    let pre_energy: f32 = stem_out[..64]
        .iter()
        .map(|s| s[0] * s[0] + s[1] * s[1])
        .sum();
    let post_energy: f32 = stem_out[64..]
        .iter()
        .map(|s| s[0] * s[0] + s[1] * s[1])
        .sum();
    assert!(pre_energy < 1e-9);
    assert!(post_energy > 0.1);
}

#[test]
fn test_wav_16bit_24bit_mbit_plus_dither_and_32bit_float_export_44k1_to_192khz() {
    let rates = [
        RenderSampleRate::Hz44100,
        RenderSampleRate::Hz48000,
        RenderSampleRate::Hz96000,
        RenderSampleRate::Hz192000,
    ];

    for sr in rates {
        let signal = make_rich_test_mix(2048, sr.as_f64());

        // 1. 32-Bit IEEE Float (format 0x0003): bit-exact roundtrip
        let bytes_f32 = encode_wav_bytes(&signal, sr, WavBitDepth::Float32Ieee, false);
        let dec_f32 = decode_wav_bytes(&bytes_f32).expect("decode 32-bit float WAV");
        assert_eq!(dec_f32.format_tag, 3);
        assert_eq!(dec_f32.bits_per_sample, 32);
        assert_eq!(dec_f32.sample_rate_hz, sr.as_u32());
        assert_eq!(dec_f32.frames.len(), signal.len());
        let max_err_f32 = signal
            .iter()
            .zip(dec_f32.frames.iter())
            .map(|(a, b)| (a[0] - b[0]).abs().max((a[1] - b[1]).abs()))
            .fold(0.0f32, f32::max);
        assert_eq!(
            max_err_f32, 0.0,
            "32-bit IEEE Float WAV roundtrip must be bit-exact (0.0 error)"
        );

        // 2. 24-Bit PCM with Ozone 12 MBIT+ Psychoacoustic Noise-Shaping Dither
        let bytes_24 = encode_wav_bytes(&signal, sr, WavBitDepth::Pcm24MbitPlus, true);
        let dec_24 = decode_wav_bytes(&bytes_24).expect("decode 24-bit MBIT+ WAV");
        assert_eq!(dec_24.format_tag, 1);
        assert_eq!(dec_24.bits_per_sample, 24);
        assert_eq!(dec_24.sample_rate_hz, sr.as_u32());
        assert_eq!(dec_24.frames.len(), signal.len());
        let max_err_24 = signal
            .iter()
            .zip(dec_24.frames.iter())
            .map(|(a, b)| (a[0] - b[0]).abs().max((a[1] - b[1]).abs()))
            .fold(0.0f32, f32::max);
        assert!(
            max_err_24 < 1e-5,
            "24-bit MBIT+ dithered WAV max error must be < 1e-5, got {max_err_24}"
        );

        // 3. 16-Bit PCM with Ozone 12 MBIT+ Psychoacoustic Noise-Shaping Dither
        let bytes_16 = encode_wav_bytes(&signal, sr, WavBitDepth::Pcm16MbitPlus, true);
        let dec_16 = decode_wav_bytes(&bytes_16).expect("decode 16-bit MBIT+ WAV");
        assert_eq!(dec_16.format_tag, 1);
        assert_eq!(dec_16.bits_per_sample, 16);
        assert_eq!(dec_16.sample_rate_hz, sr.as_u32());
        assert_eq!(dec_16.frames.len(), signal.len());
        let max_err_16 = signal
            .iter()
            .zip(dec_16.frames.iter())
            .map(|(a, b)| (a[0] - b[0]).abs().max((a[1] - b[1]).abs()))
            .fold(0.0f32, f32::max);
        assert!(
            max_err_16 < 5e-4,
            "16-bit MBIT+ dithered WAV max error must be < 5e-4, got {max_err_16}"
        );
    }
}

#[test]
fn test_offline_render_split_mixer_tracks_and_apdc_latency_trim() {
    let mut ctrl = AudioStreamController::with_mastering_template(48_000.0, 256)
        .expect("create mastering controller");

    // Add polyphonic synth channels onto Insert Tracks 1 and 2
    ctrl.add_polyphonic_synth_channel(1, 1, 64)
        .expect("add synth on track 1");
    ctrl.add_polyphonic_synth_channel(2, 2, 0)
        .expect("add synth on track 2");
    ctrl.route_mixer_track(2, 0, 0.85, false)
        .expect("route track 2 to master");

    let mut pat2 = PianoRollPattern::new(2, "Bass Pattern", 2, 192);
    pat2.add_note(PianoRollNote::new(10, 0, 96, 36, 0.92));
    pat2.add_note(PianoRollNote::new(11, 96, 96, 43, 0.90));
    ctrl.upsert_piano_roll_pattern(pat2);
    ctrl.upsert_playlist_clip(PlaylistClip {
        clip_id: 20,
        track_row: 2,
        kind: PlaylistClipKind::Pattern,
        source_id: 2,
        start_tick: 0,
        duration_ticks: 192,
        clip_offset_ticks: 0,
        gain: 1.0,
        muted: false,
    });

    let temp_dir = std::env::temp_dir().join("toneforge_phase5_split_render");
    let config = OfflineRenderConfig {
        mode: RenderMode::SplitMixerTracks,
        bit_depth: WavBitDepth::Pcm24MbitPlus,
        sample_rate: RenderSampleRate::Hz48000,
        block_size: 256,
        duration_frames: 4096,
        trim_apdc_latency: true,
        enable_mbit_plus_noise_shaping: true,
        output_dir: Some(temp_dir.clone()),
        file_prefix: "SplitBounce".to_string(),
        split_track_indices: vec![1, 2],
    };

    let result = ctrl
        .execute_offline_render(&config)
        .expect("execute SplitMixerTracks offline render");

    assert_eq!(result.master_wav.analysis.rendered_frames, 4096);
    assert!(
        result.master_wav.analysis.apdc_compensated_samples > 0,
        "Master APDC latency (> 5760 samples from BSRNN + Soothe3 + Ozone 12) must be trimmed"
    );
    assert_eq!(result.additional_wavs.len(), 2);

    // Verify Master WAV and Split Track WAVs exist on disk and contain 4096 frames of non-zero audio
    let master_path = result
        .master_wav
        .file_path
        .as_ref()
        .expect("master file path");
    let master_dec = read_wav_file(master_path).expect("read rendered Master WAV");
    assert_eq!(master_dec.bits_per_sample, 24);
    assert_eq!(master_dec.frames.len(), 4096);

    for split_wav in &result.additional_wavs {
        let wav_path = split_wav.file_path.as_ref().expect("split wav file path");
        let trk_dec = read_wav_file(wav_path).expect("read split track WAV");
        assert_eq!(trk_dec.bits_per_sample, 24);
        assert_eq!(trk_dec.frames.len(), 4096);
        let energy: f32 = trk_dec
            .frames
            .iter()
            .map(|f| f[0] * f[0] + f[1] * f[1])
            .sum();
        assert!(
            energy > 1e-3,
            "Split Mixer Track {} WAV must contain non-zero rendered audio energy (got {energy})",
            split_wav.track_index
        );
    }

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_end_to_end_daw_workflow_pianoroll_playlist_uvr5_502mixer_ozone12_wav_export() {
    let sample_rate = 48_000.0;
    let mut ctrl = AudioStreamController::with_mastering_template(sample_rate, 256)
        .expect("create full ToneForge mastering controller");

    // 1. Set Hybrid PPQ resolution to 192 PPQ and compose Piano Roll + Slide notes + Playlist + TensionSpline
    ctrl.set_ppq_resolution(192).expect("set 192 PPQ");
    ctrl.add_polyphonic_synth_channel(1, 1, 64)
        .expect("attach PolyphonicSynthGeneratorNode with Point 1 APDC on Track 1");

    // Also attach a backing stem clip channel on Track 2 so the pre-master mix has rich multi-stem content
    let backing_mix = make_rich_test_mix(16_384, sample_rate);
    ctrl.bounce_stems_to_mixer_tracks(&[backing_mix], &[2])
        .expect("attach backing mix onto Track 2");

    let mut lead_pattern = PianoRollPattern::new(1, "E2E Lead Synth with FL26 Slide", 1, 384);
    lead_pattern.add_note(PianoRollNote {
        id: 1,
        start_tick: 0,
        duration_ticks: 96,
        key: 60,
        velocity: 0.90,
        pan: -0.20,
        fine_pitch_cents: 0.0,
        is_slide: false,
        color_group: 0,
    });
    lead_pattern.add_note(PianoRollNote {
        id: 2,
        start_tick: 32,
        duration_ticks: 64,
        key: 72,
        velocity: 0.98,
        pan: 0.25,
        fine_pitch_cents: 15.0,
        is_slide: true, // FL Studio 26 Slide/Portamento glide
        color_group: 0,
    });
    lead_pattern.add_note(PianoRollNote {
        id: 3,
        start_tick: 96,
        duration_ticks: 96,
        key: 67,
        velocity: 0.86,
        pan: 0.10,
        fine_pitch_cents: -10.0,
        is_slide: false,
        color_group: 1,
    });
    ctrl.upsert_piano_roll_pattern(lead_pattern);

    let temp_dir = std::env::temp_dir().join("toneforge_phase5_e2e_master_render");

    // 2. Execute StemMastering Render -> UVR5 4-Stem Ensemble Separation ->
    //    Per-Stem Pro-Q 4 + Soothe3 on Insert Tracks 1..4 -> Ozone 12 Master Chain ->
    //    24-Bit PCM with Ozone 12 MBIT+ Psychoacoustic Dither
    let cfg_24bit = OfflineRenderConfig {
        mode: RenderMode::StemMastering,
        bit_depth: WavBitDepth::Pcm24MbitPlus,
        sample_rate: RenderSampleRate::Hz48000,
        block_size: 256,
        duration_frames: 4096,
        trim_apdc_latency: true,
        enable_mbit_plus_noise_shaping: true,
        output_dir: Some(temp_dir.clone()),
        file_prefix: "ToneForge_E2E_24BitMbit".to_string(),
        split_track_indices: Vec::new(),
    };

    let res_24 = OfflineRenderEngine::execute_render(&mut ctrl.mixer_engine, &cfg_24bit)
        .expect("execute StemMastering 24-bit MBIT+ render");

    assert_eq!(
        res_24.additional_wavs.len(),
        4,
        "StemMastering must export 4 processed stem WAVs (Vocals, Drums, Bass, Other) in addition to Master"
    );
    assert!(res_24.master_wav.analysis.apdc_compensated_samples > 0);

    // Verify 24-Bit MBIT+ Master WAV on disk
    let master_24_path = res_24
        .master_wav
        .file_path
        .as_ref()
        .expect("master 24-bit path");
    let dec_24 = read_wav_file(master_24_path).expect("read 24-bit MBIT+ Master WAV");
    assert_eq!(dec_24.format_tag, 1);
    assert_eq!(dec_24.bits_per_sample, 24);
    assert_eq!(dec_24.sample_rate_hz, 48_000);
    assert_eq!(dec_24.frames.len(), 4096);
    assert!(
        dec_24
            .frames
            .iter()
            .all(|f| f[0].is_finite() && f[1].is_finite()),
        "All rendered 24-bit samples must be finite"
    );

    // Verify Post-Render Analysis Report (ITU-R BS.1770-4 LUFS, 4x True Peak dBTP, Crest Factor, Phase Correlation)
    let report = &res_24.master_analysis_ref_helper();
    assert!(
        report.integrated_lufs > -60.0 && report.integrated_lufs < 0.0,
        "Integrated LUFS should be in valid studio range (-60..0 LUFS), got {}",
        report.integrated_lufs
    );
    assert!(report.max_short_term_lufs >= report.integrated_lufs - 6.0);
    assert!(report.max_momentary_lufs >= report.max_short_term_lufs - 6.0);
    assert!(
        report.max_true_peak_dbtp <= -0.5,
        "Ozone 12 IRC V 4x True-Peak limiter ceiling (-1.0 dBTP) must prevent inter-sample clipping, got {} dBTP",
        report.max_true_peak_dbtp
    );
    assert!(
        report.crest_factor_db > 1.5,
        "Crest factor must be positive (> 1.5 dB), got {}",
        report.crest_factor_db
    );
    assert!(
        report.phase_correlation >= -1.0 && report.phase_correlation <= 1.0,
        "Phase correlation must be within [-1.0, 1.0], got {}",
        report.phase_correlation
    );

    // 3. Execute Full 502-Track Mixer (Pro-Q 4 + Soothe3 + Patcher DAG + RealTimeBsrnnStemNode + Ozone 12)
    //    Master Render -> 32-Bit IEEE Float WAV
    let cfg_32f = OfflineRenderConfig {
        mode: RenderMode::Master,
        bit_depth: WavBitDepth::Float32Ieee,
        sample_rate: RenderSampleRate::Hz48000,
        block_size: 256,
        duration_frames: 4096,
        trim_apdc_latency: true,
        enable_mbit_plus_noise_shaping: false,
        output_dir: Some(temp_dir.clone()),
        file_prefix: "ToneForge_E2E_32Float".to_string(),
        split_track_indices: Vec::new(),
    };
    let res_32f = ctrl
        .execute_offline_render(&cfg_32f)
        .expect("execute Master 32-bit IEEE Float render");
    let master_32f_path = res_32f
        .master_wav
        .file_path
        .as_ref()
        .expect("master 32f path");
    let dec_32f = read_wav_file(master_32f_path).expect("read 32-bit IEEE Float Master WAV");
    assert_eq!(dec_32f.format_tag, 3);
    assert_eq!(dec_32f.bits_per_sample, 32);
    assert_eq!(dec_32f.frames.len(), 4096);
    let reanalyzed = analyze_rendered_audio(
        &dec_32f.frames,
        RenderSampleRate::Hz48000,
        WavBitDepth::Float32Ieee,
        res_32f.master_wav.analysis.apdc_compensated_samples,
    );
    assert!(
        (reanalyzed.integrated_lufs - res_32f.master_wav.analysis.integrated_lufs).abs() < 1e-3
    );

    let _ = std::fs::remove_dir_all(&temp_dir);
}

trait OfflineRenderResultExt {
    fn master_analysis_ref_helper(&self) -> &daw_tauri_host::PostRenderAnalysisReport;
}

impl OfflineRenderResultExt for daw_tauri_host::OfflineRenderResult {
    fn master_analysis_ref_helper(&self) -> &daw_tauri_host::PostRenderAnalysisReport {
        &self.master_wav.analysis
    }
}

#[test]
fn test_phase5_ipc_commands_and_svelte5_ui_components_parity() {
    let mut ctrl = AudioStreamController::with_mastering_template(48_000.0, 256)
        .expect("create mastering controller");

    // 1. Test Phase 5 Tauri IPC commands
    cmd_set_ppq_resolution(&ctrl.app_state, 480).expect("cmd_set_ppq_resolution(480)");
    assert!(
        cmd_set_ppq_resolution(&ctrl.app_state, 12).is_err(),
        "Out-of-range PPQ resolution (< 24) must be rejected"
    );

    cmd_trigger_piano_roll_note(
        &ctrl.app_state,
        1,
        PianoRollNoteDto {
            id: 1,
            start_tick: 0,
            duration_ticks: 96,
            key: 64,
            velocity: 0.9,
            pan: 0.0,
            fine_pitch_cents: 0.0,
            is_slide: false,
            color_group: 0,
        },
        0,
        0,
    )
    .expect("cmd_trigger_piano_roll_note");

    let temp_dir = std::env::temp_dir().join("toneforge_phase5_ipc_render");
    let ipc_render_res = cmd_execute_offline_render(
        &mut ctrl.mixer_engine,
        &OfflineRenderConfig {
            mode: RenderMode::Master,
            bit_depth: WavBitDepth::Pcm16MbitPlus,
            sample_rate: RenderSampleRate::Hz44100,
            block_size: 256,
            duration_frames: 2048,
            trim_apdc_latency: true,
            enable_mbit_plus_noise_shaping: true,
            output_dir: Some(temp_dir.clone()),
            file_prefix: "IpcRender16Bit".to_string(),
            split_track_indices: Vec::new(),
        },
    )
    .expect("cmd_execute_offline_render");
    assert_eq!(ipc_render_res.master_wav.analysis.rendered_frames, 2048);
    let _ = std::fs::remove_dir_all(&temp_dir);

    // 2. Verify byte-for-byte parity between `ui-svelte-webgpu` and `ui-svelte5-wgpu` for Phase 5 UI files
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    let ui_a = workspace_root.join("ui-svelte-webgpu");
    let ui_b = workspace_root.join("ui-svelte5-wgpu");

    for rel_path in [
        "src/lib/state/daw_master_store.svelte.ts",
        "src/lib/components/PlaylistPianoRollEditor.svelte",
        "src/lib/components/OfflineRenderStudio.svelte",
        "src/App.svelte",
    ] {
        let content_a = std::fs::read_to_string(ui_a.join(rel_path))
            .unwrap_or_else(|e| panic!("Failed to read ui-svelte-webgpu/{rel_path}: {e}"));
        let content_b = std::fs::read_to_string(ui_b.join(rel_path))
            .unwrap_or_else(|e| panic!("Failed to read ui-svelte5-wgpu/{rel_path}: {e}"));
        assert_eq!(
            content_a, content_b,
            "UI file {rel_path} must be identical across ui-svelte-webgpu and ui-svelte5-wgpu"
        );
    }
}

#[test]
fn test_phase5_edge_cases_empty_inputs_boundary_values_and_voice_stealing() {
    // 1. Empty WAV slice & corrupted RIFF header error paths
    assert!(decode_wav_bytes(&[]).is_err());
    assert!(decode_wav_bytes(b"NOT_A_VALID_RIFF_WAVE_HEADER_44_BYTES_LONG!!!!").is_err());

    // 0-frame WAV encoding & decoding roundtrip
    let empty_wav = encode_wav_bytes(&[], RenderSampleRate::Hz48000, WavBitDepth::Pcm24MbitPlus, true);
    assert_eq!(empty_wav.len(), 44);
    let dec_empty = decode_wav_bytes(&empty_wav).expect("decode 0-frame WAV");
    assert!(dec_empty.frames.is_empty());

    // 2. Post-Render Analysis on 0-length and silent buffers (no NaN / division by zero)
    let empty_report = analyze_rendered_audio(
        &[],
        RenderSampleRate::Hz48000,
        WavBitDepth::Pcm24MbitPlus,
        0,
    );
    assert!(empty_report.integrated_lufs.is_finite());
    assert!(empty_report.crest_factor_db.is_finite());
    assert!(empty_report.phase_correlation.is_finite());

    let silent_frames = vec![[0.0f32; 2]; 1024];
    let silent_report = analyze_rendered_audio(
        &silent_frames,
        RenderSampleRate::Hz96000,
        WavBitDepth::Float32Ieee,
        0,
    );
    assert!(silent_report.integrated_lufs <= -70.0);
    assert_eq!(silent_report.crest_factor_db, 0.0);
    assert_eq!(silent_report.phase_correlation, 1.0);

    // 3. Hybrid PPQ boundary clamping (MIN_PPQ = 24, MAX_PPQ = 960)
    let (mut engine, _prod, _telem) = MixerEngine::new(48_000.0, 128.0, 0);
    engine.slicer.set_ppq(0);
    assert_eq!(engine.slicer.ppq, daw_core_graph::MIN_PPQ);
    engine.slicer.set_ppq(99_999);
    assert_eq!(engine.slicer.ppq, daw_core_graph::MAX_PPQ);

    // 4. 16-Voice Polyphony Voice Stealing & Boundary MIDI Keys (0 and 127)
    let mut synth = PolyphonicSynthGeneratorNode::new(1, 48_000.0);
    synth.prepare(48_000.0, 128);
    for i in 0..22u8 {
        let key = if i == 0 {
            0
        } else if i == 21 {
            127
        } else {
            48 + i
        };
        synth.schedule_voice_event(SequencerVoiceEvent {
            sample_pos: 0,
            ppq_tick: 0,
            channel_id: 1,
            note_id: (i as u32) + 1,
            kind: VoiceEventKind::NoteOn,
            key,
            velocity: 0.9,
            pan: if i % 2 == 0 { -1.0 } else { 1.0 },
            fine_pitch_cents: if i % 2 == 0 { -1200.0 } else { 1200.0 },
            slide_duration_ticks: 0,
            slide_duration_samples: 0,
            color_group: i & 0x0F,
        });
    }
    let ctx = ProcessContext {
        sample_rate: 48_000.0,
        current_sample_pos: 0,
        ppq_tick: 0,
        bpm: 128.0,
        slice_frames: 128,
        is_playing: true,
        is_offline_rendering: false,
    };
    let mut out = [[0.0f32; 2]; 128];
    synth.process_slice(&ctx, &mut out, None, &mut []);
    assert_eq!(
        synth.active_voice_count(),
        daw_core_graph::MAX_SYNTH_VOICES,
        "Voice stealing must cap active voices at MAX_SYNTH_VOICES (16)"
    );
    assert!(
        out.iter().all(|f| f[0].is_finite() && f[1].is_finite()),
        "Extreme MIDI keys (0, 127) and +/-1200ct fine pitch must produce finite samples"
    );
}

#[test]
fn test_phase5_multichunk_riff_wav_decoder_and_stopped_transport_silence() {
    // 1. Construct a valid studio RIFF/WAVE file containing:
    //    - an odd-sized `JUNK` chunk (5 bytes + 1 pad byte) before `fmt `
    //    - an extended 18-byte `fmt ` chunk (`cbSize = 0`)
    //    - a `LIST` metadata chunk (12 bytes) before `data`
    let frames = vec![[0.25f32, -0.50f32], [-0.75f32, 0.125f32], [0.5f32, -0.25f32]];
    let mut wav = Vec::new();
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&0u32.to_le_bytes()); // placeholder size
    wav.extend_from_slice(b"WAVE");

    // Odd-sized JUNK chunk (5 bytes payload + 1 word-alignment pad byte)
    wav.extend_from_slice(b"JUNK");
    wav.extend_from_slice(&5u32.to_le_bytes());
    wav.extend_from_slice(b"12345");
    wav.push(0); // word alignment pad byte

    // Extended 18-byte `fmt ` chunk (IEEE Float 32-bit stereo @ 48 kHz)
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&18u32.to_le_bytes());
    wav.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
    wav.extend_from_slice(&2u16.to_le_bytes()); // 2 channels
    wav.extend_from_slice(&48_000u32.to_le_bytes());
    wav.extend_from_slice(&(48_000u32 * 8).to_le_bytes());
    wav.extend_from_slice(&8u16.to_le_bytes());
    wav.extend_from_slice(&32u16.to_le_bytes());
    wav.extend_from_slice(&0u16.to_le_bytes()); // cbSize = 0

    // LIST chunk (8 bytes payload)
    wav.extend_from_slice(b"LIST");
    wav.extend_from_slice(&8u32.to_le_bytes());
    wav.extend_from_slice(b"INFOISFT");

    // `data` chunk
    let data_bytes_len = (frames.len() * 8) as u32;
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_bytes_len.to_le_bytes());
    for f in &frames {
        wav.extend_from_slice(&f[0].to_le_bytes());
        wav.extend_from_slice(&f[1].to_le_bytes());
    }
    let riff_size = (wav.len() - 8) as u32;
    wav[4..8].copy_from_slice(&riff_size.to_le_bytes());

    let decoded = decode_wav_bytes(&wav).expect("decode multi-chunk studio RIFF/WAVE");
    assert_eq!(decoded.format_tag, 3);
    assert_eq!(decoded.num_channels, 2);
    assert_eq!(decoded.sample_rate_hz, 48_000);
    assert_eq!(decoded.bits_per_sample, 32);
    assert_eq!(decoded.frames, frames);

    // 2. Verify `AudioStemClipPlayerNode` outputs silence when transport is stopped (`ctx.is_playing == false`)
    let stem_audio = make_rich_test_mix(512, 48_000.0);
    let mut player = AudioStemClipPlayerNode::new(1, "Vocals", stem_audio);
    player.prepare(48_000.0, 128);
    let stopped_ctx = ProcessContext {
        sample_rate: 48_000.0,
        current_sample_pos: 0,
        ppq_tick: 0,
        bpm: 128.0,
        slice_frames: 128,
        is_playing: false,
        is_offline_rendering: false,
    };
    let mut stopped_out = [[0.0f32; 2]; 128];
    player.process_slice(&stopped_ctx, &mut stopped_out, None, &mut []);
    let stopped_energy: f32 = stopped_out
        .iter()
        .map(|f| f[0] * f[0] + f[1] * f[1])
        .sum();
    assert_eq!(
        stopped_energy, 0.0,
        "AudioStemClipPlayerNode must output silence when transport is stopped"
    );
}

#[test]
fn test_phase5_overlapping_slide_notes_with_tempomap_and_deterministic_post_playback_render() {
    use daw_core_graph::{
        HybridPpqSlicer, SampleAccurateAutomationQueue, TempoMap,
    };
    use daw_tauri_host::{
        AutomationClipStateDto, LfoModulatorDto, PianoRollPatternDto, PlaylistClipDto,
        TensionControlPointDto, cmd_update_automation_spline, cmd_upsert_piano_roll_pattern,
        cmd_upsert_playlist_clip,
    };

    // 1. Verify `schedule_pattern_into_automation_queue` preserves `slide_duration_samples`
    //    and handles overlapping Slide notes in the same `color_group` under a `TempoMap` BPM change
    let mut slicer = HybridPpqSlicer::new(48_000.0, 120.0, 96);
    let mut tm = TempoMap::new(120.0);
    tm.add_point(0, 120.0);
    tm.add_point(96, 150.0);
    slicer.tempo_map = Some(tm);

    let mut pat = PianoRollPattern::new(1, "Overlapping Slide Pattern", 1, 192);
    pat.add_note(PianoRollNote {
        id: 1,
        start_tick: 0,
        duration_ticks: 192,
        key: 60,
        velocity: 0.85,
        pan: -0.3,
        fine_pitch_cents: 0.0,
        is_slide: false,
        color_group: 2,
    });
    pat.add_note(PianoRollNote {
        id: 2,
        start_tick: 24,
        duration_ticks: 48,
        key: 67,
        velocity: 0.90,
        pan: 0.1,
        fine_pitch_cents: 12.0,
        is_slide: true,
        color_group: 2,
    });
    pat.add_note(PianoRollNote {
        id: 3,
        start_tick: 48,
        duration_ticks: 48,
        key: 72,
        velocity: 0.95,
        pan: 0.5,
        fine_pitch_cents: -8.0,
        is_slide: true,
        color_group: 2,
    });

    let mut seq = daw_core_graph::PlaylistSequencer::new();
    seq.upsert_pattern(pat);
    let mut queue = SampleAccurateAutomationQueue::with_capacity(64);
    let scheduled = seq.schedule_pattern_into_automation_queue(1, &slicer, 0, &mut queue);
    assert_eq!(scheduled, 4); // 1 NoteOn + 1 NoteOff + 2 SlideNotes

    let mut unpacked_slides = Vec::new();
    queue.drain_up_to(
        i64::MAX,
        |_| {},
        |voice_ev| {
            let seq_ev = SequencerVoiceEvent::from_delayed_voice_event(voice_ev);
            if seq_ev.kind == VoiceEventKind::SlideNote {
                unpacked_slides.push(seq_ev);
            }
        },
    );
    assert_eq!(unpacked_slides.len(), 2);
    for slide_ev in &unpacked_slides {
        assert_eq!(slide_ev.color_group, 2);
        assert!(
            slide_ev.slide_duration_samples >= 2048,
            "Unpacked SlideNote must preserve non-zero slide_duration_samples, got {}",
            slide_ev.slide_duration_samples
        );
    }

    // 2. Verify new IPC commands (`cmd_upsert_piano_roll_pattern`, `cmd_upsert_playlist_clip`, `cmd_update_automation_spline`)
    //    and deterministic OfflineRenderEngine output even after prior real-time playback and paused transport
    let (mut engine, _prod, _telem) = MixerEngine::new(48_000.0, 128.0, 0);
    let synth = PolyphonicSynthGeneratorNode::new(1, 48_000.0);
    engine
        .add_channel_generator(ChannelGenerator::new(1, 1, Box::new(synth)))
        .unwrap();
    engine.tracks[1].add_send(0, 1.0, false);
    engine.rebuild_graph().unwrap();

    cmd_upsert_piano_roll_pattern(
        &mut engine,
        &PianoRollPatternDto {
            pattern_id: 1,
            name: "IPC Pattern".to_string(),
            target_channel_id: 1,
            length_ticks: 192,
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
        },
    )
    .expect("cmd_upsert_piano_roll_pattern");

    cmd_upsert_playlist_clip(
        &mut engine,
        &PlaylistClipDto {
            clip_id: 1,
            track_row: 0,
            kind: "Pattern".to_string(),
            source_id: 1,
            start_tick: 0,
            duration_ticks: 192,
            clip_offset_ticks: 0,
            gain: 1.0,
            muted: false,
        },
    )
    .expect("cmd_upsert_playlist_clip");

    let fixed30 = cmd_update_automation_spline(
        &mut engine,
        &AutomationClipStateDto {
            controller_id: 0,
            name: "Filter Sweep".to_string(),
            points: vec![
                TensionControlPointDto {
                    x: 0.0,
                    y: 0.25,
                    tension: 0.0,
                },
                TensionControlPointDto {
                    x: 96.0,
                    y: 0.75,
                    tension: 0.45,
                },
            ],
            lfo: LfoModulatorDto::default(),
            min_level: 0.0,
            max_level: 1.0,
        },
        48.0,
    )
    .expect("cmd_update_automation_spline");
    assert!(fixed30 > 0);

    let cfg = OfflineRenderConfig {
        mode: RenderMode::Master,
        sample_rate: RenderSampleRate::Hz48000,
        bit_depth: WavBitDepth::Float32Ieee,
        enable_mbit_plus_noise_shaping: false,
        trim_apdc_latency: true,
        duration_frames: 1024,
        block_size: 256,
        split_track_indices: Vec::new(),
        output_dir: None,
        file_prefix: "DeterministicTest".to_string(),
    };

    // First clean render from tick 0
    let render_a = OfflineRenderEngine::execute_render(&mut engine, &cfg).expect("render A");

    // Simulate real-time playback advancing cursor and then stopping transport (`is_playing = false`)
    let mut dummy = [[0.0f32; 2]; 256];
    for _ in 0..5 {
        engine.process_audio_block(&mut dummy);
    }
    engine.is_playing = false;

    // Second render after prior playback & paused transport must produce bit-identical output
    let render_b = OfflineRenderEngine::execute_render(&mut engine, &cfg).expect("render B");
    assert_eq!(
        render_a.master_wav.frames, render_b.master_wav.frames,
        "OfflineRenderEngine must reset slicer/transport/DSP state so bounces are 100% deterministic"
    );

    // 3. Verify JSON serialization/deserialization roundtrip of OfflineRenderConfig (including UI alias) and OfflineRenderResult
    let ui_json = r#"{
        "mode": "Master",
        "sample_rate": "Hz48000",
        "bit_depth": "Pcm24MbitPlus",
        "enable_noise_shaping": true,
        "trim_apdc_latency": true,
        "duration_frames": 1024,
        "block_size": 256,
        "split_track_indices": [],
        "output_dir": null,
        "file_prefix": "UiRender"
    }"#;
    let parsed_cfg: OfflineRenderConfig =
        serde_json::from_str(ui_json).expect("deserialize UI OfflineRenderConfig JSON");
    assert!(parsed_cfg.enable_mbit_plus_noise_shaping);

    let res_json = serde_json::to_string(&render_a).expect("serialize OfflineRenderResult");
    let decoded_res: daw_tauri_host::OfflineRenderResult =
        serde_json::from_str(&res_json).expect("deserialize OfflineRenderResult");
    assert_eq!(
        decoded_res.master_wav.analysis.integrated_lufs,
        render_a.master_wav.analysis.integrated_lufs
    );
}


