//! Integration tests for Native Multi-Format Audio Decoding (`symphonia`), Real-Folder UVR5
//! GPU Stem Separation (`RTX 4080 CUDAExecutionProvider`), Official UVR5 Model Hashes, and
//! Master Track Real-Time ONNX GPU BSRNN Worker Activation.

use daw_tauri_host::{
    AudioStreamController, GpuExecutionTier, RenderSampleRate, Uvr5SeparationRequestDto,
    WavBitDepth, cmd_load_uvr5_model_profile, cmd_query_gpu_accelerator_status,
    decode_audio_file_native, decode_wav_bytes, encode_exact_sr_pcm16_wav_bytes, encode_wav_bytes,
    execute_uvr5_separation_dsp, resolve_assets_root,
};
use std::path::PathBuf;

fn make_temp_test_workspace(tag: &str) -> PathBuf {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .join(".staging_test_io")
        .join(format!("{}_{}", tag, std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).expect("create test workspace dir");
    base
}

fn write_mono_pcm16_wav(path: &std::path::Path, samples: &[f32], sample_rate_hz: u32) {
    let data_bytes = (samples.len() * 2) as u32;
    let riff_size = 36u32.saturating_add(data_bytes);
    let byte_rate = sample_rate_hz.saturating_mul(2);
    let mut out = Vec::with_capacity(44 + data_bytes as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // Mono
    out.extend_from_slice(&sample_rate_hz.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes()); // block_align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits_per_sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());
    for &s in samples {
        let q = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&q.to_le_bytes());
    }
    std::fs::write(path, out).expect("write mono wav");
}

#[test]
fn test_native_symphonia_decoder_formats_mono_upmix_and_edge_cases() {
    let dir = make_temp_test_workspace("symphonia_decoder");

    // 1. Real FL26 factory WAV asset
    let fl26_kick = resolve_assets_root()
        .join("flstudio26")
        .join("samples")
        .join("808 Kick.wav");
    if fl26_kick.is_file() {
        let dec_kick = decode_audio_file_native(&fl26_kick).expect("decode real 808 Kick.wav");
        assert!(!dec_kick.frames.is_empty());
        assert_eq!(dec_kick.file_name, "808 Kick.wav");
        assert!(dec_kick.sample_rate_hz >= 22_050);
    }

    // 2. 16-bit, 24-bit, and 32-bit float stereo WAV roundtrips
    let synth_frames: Vec<[f32; 2]> = (0..4096)
        .map(|n| {
            let t = (n as f32) / 44_100.0;
            [
                0.45 * (std::f32::consts::TAU * 220.0 * t).sin(),
                0.35 * (std::f32::consts::TAU * 440.0 * t).cos(),
            ]
        })
        .collect();

    for (fname, depth) in [
        ("stereo_16bit.wav", WavBitDepth::Pcm16MbitPlus),
        ("stereo_24bit.wav", WavBitDepth::Pcm24MbitPlus),
        ("stereo_32float.wav", WavBitDepth::Float32Ieee),
    ] {
        let p = dir.join(fname);
        let bytes = encode_wav_bytes(&synth_frames, RenderSampleRate::Hz44100, depth, false);
        std::fs::write(&p, bytes).expect("write test wav");

        let dec = decode_audio_file_native(&p).unwrap_or_else(|e| panic!("decode {fname}: {e}"));
        assert_eq!(dec.sample_rate_hz, 44_100);
        assert_eq!(dec.channels, 2);
        assert_eq!(dec.frames.len(), synth_frames.len());
        let dto = dec.to_dto();
        assert_eq!(dto.total_frames, synth_frames.len());
        assert_eq!(dto.sample_rate_hz, 44_100);

        let max_err = dec
            .frames
            .iter()
            .zip(synth_frames.iter())
            .map(|(a, b)| (a[0] - b[0]).abs().max((a[1] - b[1]).abs()))
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 5e-4,
            "{fname} max_err too large: {max_err}"
        );
    }

    // 3. Mono WAV -> Stereo upmix (`L == R`) + exact-SR binary WAV cache
    let mono_samples: Vec<f32> = (0..2048)
        .map(|n| 0.5 * (std::f32::consts::TAU * 330.0 * (n as f32) / 48_000.0).sin())
        .collect();
    let mono_path = dir.join("mono_48k.wav");
    write_mono_pcm16_wav(&mono_path, &mono_samples, 48_000);
    let dec_mono = decode_audio_file_native(&mono_path).expect("decode mono wav");
    assert_eq!(dec_mono.channels, 1);
    assert_eq!(dec_mono.sample_rate_hz, 48_000);
    assert_eq!(dec_mono.frames.len(), 2048);
    for f in &dec_mono.frames {
        assert!((f[0] - f[1]).abs() < 1e-6, "Mono channel must upmix identically to L and R");
    }

    let exact_wav = encode_exact_sr_pcm16_wav_bytes(&dec_mono.frames, 44_100);
    assert!(exact_wav.starts_with(b"RIFF"));
    let header_sr = u32::from_le_bytes(exact_wav[24..28].try_into().unwrap());
    assert_eq!(header_sr, 44_100);

    // 4. Error & boundary handling: missing file, directory path, zero-byte file, corrupt bytes
    assert!(decode_audio_file_native(&dir.join("does_not_exist.mp3")).is_err());
    assert!(decode_audio_file_native(&dir).is_err());

    let empty_path = dir.join("empty.wav");
    std::fs::write(&empty_path, b"").unwrap();
    assert!(decode_audio_file_native(&empty_path).is_err());

    let corrupt_path = dir.join("corrupt.flac");
    std::fs::write(&corrupt_path, b"NOT_AN_AUDIO_STREAM_AT_ALL_0123456789").unwrap();
    assert!(decode_audio_file_native(&corrupt_path).is_err());

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir(dir.parent().unwrap());
}

#[test]
fn test_uvr5_official_hashes_rtx4080_cuda_and_real_folder_export() {
    // 1. Verify official TRvlvr/model_repo hashes and compensate factors
    let kim = cmd_load_uvr5_model_profile("Kim_Vocal_2 (MDX-Net Lead Vocal, n_fft=7680)")
        .expect("Kim_Vocal_2 profile");
    assert_eq!(kim.model_hash, "970b3f9492014d18fefeedfe4773cb42");
    assert_eq!(kim.n_fft, 7680);
    assert_eq!(kim.dim_f, 3072);
    assert_eq!(kim.dim_t, 8);
    assert_eq!(1usize << kim.dim_t, 256);
    assert!((kim.compensate - 1.009).abs() < 1e-4);

    let inst = cmd_load_uvr5_model_profile("UVR-MDX-Net-Inst_HQ_3 (n_fft=6144, dim_f=3072)")
        .expect("Inst_HQ_3 profile");
    assert_eq!(inst.model_hash, "55657dd70583b0fedfba5f67df11d711");
    assert_eq!(inst.n_fft, 6144);
    assert_eq!(inst.dim_f, 3072);
    assert_eq!(inst.dim_t, 8);
    assert_eq!(1usize << inst.dim_t, 256);
    assert!((inst.compensate - 1.022).abs() < 1e-4);

    // 2. Verify RTX 4080 CUDAExecutionProvider status
    let status = cmd_query_gpu_accelerator_status();
    assert_eq!(
        status.tier,
        GpuExecutionTier::CudaTensorCore,
        "Expected CUDAExecutionProvider on RTX 4080, got {:?}",
        status
    );
    assert_eq!(status.provider_name, "CUDAExecutionProvider");
    assert!(
        status.device_name.contains("4080"),
        "Expected RTX 4080 device_name, got '{}'",
        status.device_name
    );

    // 3. Verify real-folder UVR5 GPU separation WITHOUT `%TEMP%`
    let base_dir = make_temp_test_workspace("uvr5_real_folder");
    let song_folder = base_dir.join("Artist_Album_Session");
    let custom_out_folder = base_dir.join("Custom_Export_Stems");
    std::fs::create_dir_all(&song_folder).unwrap();

    let song_frames: Vec<[f32; 2]> = (0..12_000)
        .map(|n| {
            let t = (n as f32) / 44_100.0;
            let bass = 0.30 * (std::f32::consts::TAU * 65.0 * t).sin();
            let vocal = 0.28 * (std::f32::consts::TAU * 520.0 * t).sin();
            let hat = 0.12 * (std::f32::consts::TAU * 6400.0 * t).cos();
            [bass + vocal + hat, bass + vocal * 0.92 - hat * 0.8]
        })
        .collect();
    let song_path = song_folder.join("My_Studio_Track.wav");
    std::fs::write(
        &song_path,
        encode_exact_sr_pcm16_wav_bytes(&song_frames, 44_100),
    )
    .expect("write test song");

    // Case A: `output_dir = None` -> exports directly alongside `My_Studio_Track.wav` in `song_folder`!
    // Pass `&[]` for `mix_frames` to prove `execute_uvr5_separation_dsp` decodes `song_path` via Symphonia.
    let req_song_dir = Uvr5SeparationRequestDto {
        audio_path: song_path.display().to_string(),
        output_dir: None,
        model_hash: "ensemble_3model".to_string(),
        ensemble_mode: "Weighted".to_string(),
        weights: [0.45, 0.35, 0.20],
        enable_denoise_2pass: false,
        vr_aggressiveness: 0.15,
        phase_denoise_strength: 0.15,
        invert_instrumental: true,
        bounce_to_mixer: false,
    };
    let (res_a, stems_a, _) = execute_uvr5_separation_dsp(&[], 44_100, &req_song_dir)
        .expect("UVR5 GPU separation into song_folder");
    assert_eq!(res_a.gpu_provider, "CUDAExecutionProvider");
    assert_eq!(res_a.separated_frames, song_frames.len());
    assert_eq!(PathBuf::from(&res_a.output_directory), song_folder);
    for sp in &res_a.stem_paths {
        let p = PathBuf::from(sp);
        assert!(p.is_file(), "Expected stem WAV on disk: {}", p.display());
        assert_eq!(
            p.parent().unwrap(),
            song_folder.as_path(),
            "Stem must be exported into real song_folder, NOT %TEMP%: {}",
            p.display()
        );
    }
    assert!(res_a.stem_paths[3].ends_with("_inv_inst.wav"));
    assert_eq!(stems_a[0].len(), song_frames.len());

    // Case B: Explicit `output_dir = Some(custom_out_folder)` + `invert_instrumental = false`
    let req_custom_dir = Uvr5SeparationRequestDto {
        audio_path: song_path.display().to_string(),
        output_dir: Some(custom_out_folder.display().to_string()),
        model_hash: "970b3f9492014d18fefeedfe4773cb42".to_string(), // Kim_Vocal_2 + BSRNN 4-stem
        ensemble_mode: "Weighted".to_string(),
        weights: [0.5, 0.3, 0.2],
        enable_denoise_2pass: true,
        vr_aggressiveness: 0.10,
        phase_denoise_strength: 0.10,
        invert_instrumental: false,
        bounce_to_mixer: false,
    };
    let (res_b, _, _) = execute_uvr5_separation_dsp(&song_frames, 44_100, &req_custom_dir)
        .expect("UVR5 GPU separation into custom_out_folder");
    assert_eq!(res_b.gpu_provider, "CUDAExecutionProvider");
    assert_eq!(PathBuf::from(&res_b.output_directory), custom_out_folder);
    assert!(res_b.stem_paths[3].ends_with("_other.wav"));
    for sp in &res_b.stem_paths {
        let p = PathBuf::from(sp);
        assert!(p.is_file());
        assert_eq!(p.parent().unwrap(), custom_out_folder.as_path());
    }

    // 4. Verify Master Track 0 Slot 0 Real-Time ONNX GPU BSRNN Worker activation
    let mut ctrl = AudioStreamController::with_mastering_template(48_000.0, 256)
        .expect("create mastering controller");
    let tier = ctrl
        .enable_realtime_onnx_gpu_bsrnn_on_master()
        .expect("enable_realtime_onnx_gpu_bsrnn_on_master");
    assert_eq!(tier, GpuExecutionTier::CudaTensorCore);
    let mut out_block = [[0.0f32; 2]; 256];
    for _ in 0..8 {
        ctrl.process_audio_block(&mut out_block);
    }

    let _ = std::fs::remove_dir_all(&base_dir);
    let _ = std::fs::remove_dir(base_dir.parent().unwrap());
}

#[test]
fn test_sub44k_exact_sample_rate_unicode_paths_24bit_stems_and_corrupt_file_error() {
    let base_dir = make_temp_test_workspace("uvr5_sr22k_unicode");
    let unicode_folder = base_dir.join("Phòng Thu Âm — Thử Nghiệm 2026");
    std::fs::create_dir_all(&unicode_folder).expect("create Unicode folder");

    // 1. 22,050 Hz (< 44.1 kHz) stereo song (0.50 s = 11,025 frames) in a Vietnamese Unicode path:
    //    Must NOT be pitch/speed-shifted to 44,100 Hz; exported 24-bit WAVs must carry exact SR=22,050!
    let frames_22k: Vec<[f32; 2]> = (0..11_025)
        .map(|n| {
            let t = (n as f32) / 22_050.0;
            let bass = 0.35 * (std::f32::consts::TAU * 60.0 * t).sin();
            let vox = 0.25 * (std::f32::consts::TAU * 440.0 * t).sin();
            [bass + vox, bass + vox * 0.9]
        })
        .collect();
    let unicode_song_path = unicode_folder.join("Bài Hát Gốc_22050Hz.wav");
    std::fs::write(
        &unicode_song_path,
        encode_exact_sr_pcm16_wav_bytes(&frames_22k, 22_050),
    )
    .expect("write 22,050 Hz Unicode WAV");

    let req_22k = Uvr5SeparationRequestDto {
        audio_path: unicode_song_path.display().to_string(),
        output_dir: None,
        model_hash: "55657dd70583b0fedfba5f67df11d711".to_string(), // Inst_HQ_3 + BSRNN
        ensemble_mode: "Weighted".to_string(),
        weights: [0.5, 0.3, 0.2],
        enable_denoise_2pass: false,
        vr_aggressiveness: 0.1,
        phase_denoise_strength: 0.0,
        invert_instrumental: false,
        bounce_to_mixer: false,
    };
    let (res_22k, stems_22k, wav_bufs_22k) = execute_uvr5_separation_dsp(&[], 22_050, &req_22k)
        .expect("UVR5 GPU separation on 22,050 Hz Unicode file");
    assert_eq!(res_22k.gpu_provider, "CUDAExecutionProvider");
    assert_eq!(
        res_22k.sample_rate_hz, 22_050,
        "22,050 Hz input must preserve exact 22,050 Hz SR (no 2x speed-up!)"
    );
    assert_eq!(res_22k.separated_frames, 11_025);
    assert_eq!(PathBuf::from(&res_22k.output_directory), unicode_folder);

    for s in 0..4 {
        assert_eq!(stems_22k[s].len(), 11_025);
        let hdr = decode_wav_bytes(&wav_bufs_22k[s]).expect("decode exported 24-bit WAV buffer");
        assert_eq!(hdr.sample_rate_hz, 22_050);
        assert_eq!(hdr.bits_per_sample, 24);
        assert_eq!(hdr.frames.len(), 11_025);

        let disk_dec =
            decode_audio_file_native(std::path::Path::new(&res_22k.stem_paths[s]))
                .expect("Symphonia decode exported Unicode stem WAV");
        assert_eq!(disk_dec.sample_rate_hz, 22_050);
        assert_eq!(disk_dec.frames.len(), 11_025);
    }

    // 2. Non-standard 32,000 Hz passed via in-memory PCM (`mix_frames` non-empty)
    let frames_32k: Vec<[f32; 2]> = (0..8_000)
        .map(|n| {
            let t = (n as f32) / 32_000.0;
            let s = 0.3 * (std::f32::consts::TAU * 330.0 * t).sin();
            [s, s]
        })
        .collect();
    let req_32k = Uvr5SeparationRequestDto {
        audio_path: unicode_folder.join("InMemory_32kHz.wav").display().to_string(),
        output_dir: Some(unicode_folder.display().to_string()),
        model_hash: "970b3f9492014d18fefeedfe4773cb42".to_string(),
        ensemble_mode: "Weighted".to_string(),
        weights: [0.5, 0.3, 0.2],
        ..Default::default()
    };
    let (res_32k, _, wav_bufs_32k) = execute_uvr5_separation_dsp(&frames_32k, 32_000, &req_32k)
        .expect("UVR5 GPU separation on 32,000 Hz PCM");
    assert_eq!(res_32k.sample_rate_hz, 32_000);
    let hdr_32k = decode_wav_bytes(&wav_bufs_32k[0]).expect("decode 32kHz stem");
    assert_eq!(hdr_32k.sample_rate_hz, 32_000);
    assert_eq!(hdr_32k.bits_per_sample, 24);
    assert_eq!(hdr_32k.frames.len(), 8_000);

    // 3. Corrupt existing file on disk MUST return Err(...) instead of silently exporting fake sine waves!
    let corrupt_real = unicode_folder.join("Corrupt_Existing_Song.wav");
    std::fs::write(&corrupt_real, b"NOT_A_VALID_RIFF_OR_FLAC_FILE_BYTES_12345").unwrap();
    let req_corrupt = Uvr5SeparationRequestDto {
        audio_path: corrupt_real.display().to_string(),
        output_dir: Some(unicode_folder.display().to_string()),
        model_hash: "ensemble_3model".to_string(),
        ..Default::default()
    };
    assert!(
        execute_uvr5_separation_dsp(&[], 48_000, &req_corrupt).is_err(),
        "Corrupt existing audio file on disk must return Err instead of silently substituting fallback sine wave!"
    );

    let _ = std::fs::remove_dir_all(&base_dir);
    let _ = std::fs::remove_dir(base_dir.parent().unwrap());
}

