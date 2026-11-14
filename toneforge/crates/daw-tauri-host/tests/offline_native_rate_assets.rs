//! Export above the engine rate must keep the source's ultrasonic content.
//!
//! The live engine runs at 48 kHz and caches every asset at that rate. An export at 96 kHz used to
//! upsample that cache (losing everything above 24 kHz); it now re-decodes the original file at the
//! export rate (`load_asset_frames(asset, target_sr)`). This test measures both paths on a real
//! 96 kHz WAV containing a 30 kHz tone.

use daw_tauri_host::{AudioAssetDto, load_asset_frames, resample_stereo_frames, write_f32_wav_file};

/// Goertzel power of `freq` in the left channel, normalised so a full-scale sine reads ~0.25.
fn tone_power(frames: &[[f32; 2]], sr: f64, freq: f64) -> f64 {
    let n = frames.len().min(65_536);
    let w = 2.0 * std::f64::consts::PI * freq / sr;
    let coeff = 2.0 * w.cos();
    let (mut s1, mut s2) = (0.0f64, 0.0f64);
    for f in &frames[frames.len() / 4..frames.len() / 4 + n] {
        let s0 = f[0] as f64 + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    (s1 * s1 + s2 * s2 - coeff * s1 * s2) / ((n * n) as f64)
}

#[test]
fn export_at_96k_keeps_content_above_24khz() {
    let sr = 96_000u32;
    let secs = 2.0;
    let n = (sr as f64 * secs) as usize;
    let frames: Vec<[f32; 2]> = (0..n)
        .map(|i| {
            let t = i as f64 / sr as f64;
            let v = 0.25 * (2.0 * std::f64::consts::PI * 1_000.0 * t).sin()
                + 0.25 * (2.0 * std::f64::consts::PI * 30_000.0 * t).sin();
            [v as f32, v as f32]
        })
        .collect();
    let dir = std::env::temp_dir().join("toneforge_native_rate_test");
    let path = dir.join("tone_1k_30k_96k.wav");
    write_f32_wav_file(&path, &frames, sr).expect("write wav");

    let asset = AudioAssetDto {
        asset_id: 1,
        kind: "CustomAudio".to_string(),
        name: "tone".to_string(),
        file_path: path.display().to_string(),
        target_track: 1,
        continuous: true,
        loop_playback: false,
    };

    // Old export path: engine-rate (48 kHz) cache upsampled to 96 kHz.
    let cache_48k = load_asset_frames(&asset, 48_000).expect("decode @48k");
    let old_path = resample_stereo_frames(&cache_48k, 48_000, sr);
    // New export path: original file decoded at the export rate.
    let new_path = load_asset_frames(&asset, sr).expect("decode @96k");

    let ref_30k = tone_power(&frames, sr as f64, 30_000.0);
    let old_30k = tone_power(&old_path, sr as f64, 30_000.0);
    let new_30k = tone_power(&new_path, sr as f64, 30_000.0);
    let old_1k = tone_power(&old_path, sr as f64, 1_000.0);
    let new_1k = tone_power(&new_path, sr as f64, 1_000.0);
    let db = |p: f64, r: f64| 10.0 * (p.max(1e-30) / r).log10();
    println!(
        "30 kHz tone: old path {:+.1} dB, new path {:+.1} dB (rel. source); 1 kHz: old {:+.2} dB, new {:+.2} dB",
        db(old_30k, ref_30k),
        db(new_30k, ref_30k),
        db(old_1k, ref_30k),
        db(new_1k, ref_30k)
    );
    let _ = std::fs::remove_file(&path);

    assert!(db(new_30k, ref_30k) > -0.5, "native-rate decode must keep the 30 kHz tone");
    assert!(db(old_30k, ref_30k) < -40.0, "the 48 kHz cache cannot carry 30 kHz (sanity)");
    assert!((db(new_1k, ref_30k) - db(old_1k, ref_30k)).abs() < 0.5, "audible band identical");
}
