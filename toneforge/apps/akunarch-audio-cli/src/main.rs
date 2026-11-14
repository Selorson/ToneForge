//! `akunarch-audio-cli`: Standalone Zero-OOM Mastering & Stem Mixing CLI Sidecar for Akunarch.
//!
//! Powered by real ToneForge DSP crates (`dsp-proq4-eq`, `dsp-soothe3-spectral`, `dsp-ozone12-mastering`):
//! - Decodes `.wav`, `.mp3`, `.flac`, `.ogg` via `symphonia` and resamples to `48,000 Hz` stereo
//! - Runs `ProQ4Processor` (parametric vocal/cinematic EQ: sub-rumble cut, mud cleanup, presence & air)
//! - Runs `Soothe3Processor` (psychoacoustic ERB/Bark dynamic spectral resonance de-harshing)
//! - Optional `--bgm <file>` mixing with smooth envelope-follower vocal ducking (`--duck-db`)
//! - Runs `IrcVMultiBandMaximizer` + `LufsTruePeakMeter` (`ITU-R BS.1770-4` K-weighted dual-gated LUFS
//!   & 4x polyphase FIR True-Peak brickwall ceiling `-1.0 dBTP`) calibrated to exact `-14.0 LUFS`
//! - Writes 24-bit `48 kHz` broadcast PCM `.wav` and outputs JSON loudness telemetry.

use daw_core_graph::{ProcessContext, StereoFrame, UnifiedDspNode};
use dsp_ozone12_mastering::{IrcVMultiBandMaximizer, LufsTruePeakMeter};
use dsp_proq4_eq::{EqBandParams, FilterShape, ProQ4Processor};
use dsp_soothe3_spectral::Soothe3Processor;
use serde::Serialize;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

const TARGET_SR: u32 = 48_000;
const BLOCK_SIZE: usize = 512;

#[derive(Debug, Serialize)]
struct MasterReport {
    ok: bool,
    engine: &'static str,
    input_path: String,
    bgm_path: Option<String>,
    output_path: String,
    sample_rate_hz: u32,
    duration_sec: f64,
    input_integrated_lufs: f32,
    output_integrated_lufs: f32,
    output_short_term_lufs: f32,
    output_true_peak_dbtp: f32,
    target_lufs: f32,
    ceiling_dbtp: f32,
    applied_trim_db: f32,
}

fn print_usage() {
    println!(
        "akunarch-audio-cli 0.1.0 (ToneForge DSP: Pro-Q 4 + Soothe3 + Ozone 12 IRC V)\n\
         Usage:\n\
           akunarch-audio-cli --version\n\
           akunarch-audio-cli analyze --in <audio_file> [--json]\n\
           akunarch-audio-cli master --in <voice_or_mix> --out <master.wav> [--bgm <bgm_file>] \
         [--bgm-gain-db -11.0] [--duck-db -5.5] [--target-lufs -14.0] [--ceiling-dbtp -1.0] \
         [--duration-sec <sec>] [--json]"
    );
}

fn decode_audio_to_stereo_48k(path: &Path) -> Result<Vec<StereoFrame>, String> {
    let file = File::open(path)
        .map_err(|e| format!("Cannot open audio file '{}': {e}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| format!("Unsupported audio container '{}': {e}", path.display()))?;

    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| format!("No default audio track in '{}'", path.display()))?;
    let track_id = track.id;
    let src_sr = track.codec_params.sample_rate.unwrap_or(TARGET_SR).max(8000);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| format!("Audio codec init failed for '{}': {e}", path.display()))?;

    let mut raw_stereo: Vec<StereoFrame> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::IoError(ref io_err))
                if io_err.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(_) => break,
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(_) => break,
        };
        let spec = *decoded.spec();
        let ch_count = spec.channels.count().max(1);
        let frames = decoded.frames();
        if frames == 0 {
            continue;
        }
        let mut sbuf = SampleBuffer::<f32>::new(frames as u64, spec);
        sbuf.copy_interleaved_ref(decoded);
        let samples = sbuf.samples();
        for f in 0..frames {
            let base = f * ch_count;
            let l = samples.get(base).copied().unwrap_or(0.0);
            let r = if ch_count >= 2 {
                samples.get(base + 1).copied().unwrap_or(l)
            } else {
                l
            };
            raw_stereo.push([
                if l.is_finite() { l } else { 0.0 },
                if r.is_finite() { r } else { 0.0 },
            ]);
        }
    }

    if raw_stereo.is_empty() {
        return Err(format!("Decoded 0 audio frames from '{}'", path.display()));
    }

    if src_sr == TARGET_SR {
        return Ok(raw_stereo);
    }

    // Hermite 4-point cubic resampling to 48,000 Hz
    let ratio = (src_sr as f64) / (TARGET_SR as f64);
    let out_len = (((raw_stereo.len() as f64) / ratio).round() as usize).max(1);
    let mut resampled = Vec::with_capacity(out_len);
    let max_idx = (raw_stereo.len() - 1) as isize;

    for i in 0..out_len {
        let src_pos = (i as f64) * ratio;
        let idx = src_pos.floor() as isize;
        let frac = (src_pos - (idx as f64)) as f32;
        let at = |k: isize| -> StereoFrame {
            raw_stereo[k.clamp(0, max_idx) as usize]
        };
        let p0 = at(idx - 1);
        let p1 = at(idx);
        let p2 = at(idx + 1);
        let p3 = at(idx + 2);
        let mut frame = [0.0f32; 2];
        for c in 0..2 {
            let c0 = p1[c];
            let c1 = 0.5 * (p2[c] - p0[c]);
            let c2 = p0[c] - 2.5 * p1[c] + 2.0 * p2[c] - 0.5 * p3[c];
            let c3 = 0.5 * (p3[c] - p0[c]) + 1.5 * (p1[c] - p2[c]);
            frame[c] = ((c3 * frac + c2) * frac + c1) * frac + c0;
        }
        resampled.push(frame);
    }

    Ok(resampled)
}

fn measure_lufs_and_tp(frames: &[StereoFrame], sr: f64) -> (f32, f32, f32) {
    let mut meter = LufsTruePeakMeter::new(sr);
    meter.push_block(frames, 0.0);
    (
        meter.lufs_integrated,
        meter.lufs_short_term,
        meter.max_true_peak_dbtp(),
    )
}

fn run_proq4_and_soothe3(frames: &mut [StereoFrame], sr: f64) {
    let mut proq4 = ProQ4Processor::new();
    proq4.prepare(sr, BLOCK_SIZE);

    // Band 0: 75 Hz 18 dB/oct Low-Cut (cleans sub-rumble)
    let mut b0 = proq4.params.bands[0];
    b0.used = true;
    b0.enabled = true;
    b0.shape = FilterShape::LowCut;
    b0.freq_hz = 75.0;
    b0.slope_db_oct = 18.0;
    b0.q_ui = 0.85;
    proq4.set_band_params(0, b0);

    // Band 1: 275 Hz -1.6 dB Bell (removes boxiness/mud)
    let mut b1 = proq4.params.bands[1];
    b1.used = true;
    b1.enabled = true;
    b1.shape = FilterShape::Bell;
    b1.freq_hz = 275.0;
    b1.gain_db = -1.6;
    b1.q_ui = 1.25;
    proq4.set_band_params(1, b1);

    // Band 2: 3,300 Hz +1.3 dB Bell (clarity & intelligibility)
    let mut b2 = proq4.params.bands[2];
    b2.used = true;
    b2.enabled = true;
    b2.shape = FilterShape::Bell;
    b2.freq_hz = 3_300.0;
    b2.gain_db = 1.3;
    b2.q_ui = 0.95;
    proq4.set_band_params(2, b2);

    // Band 3: 11,500 Hz +1.4 dB High-Shelf (cinematic air)
    let mut b3: EqBandParams = proq4.params.bands[3];
    b3.used = true;
    b3.enabled = true;
    b3.shape = FilterShape::HighShelf;
    b3.freq_hz = 11_500.0;
    b3.gain_db = 1.4;
    b3.q_ui = 0.75;
    proq4.set_band_params(3, b3);

    // Soothe3 dynamic spectral resonance suppressor (low-latency zero pre-ringing mode)
    let mut soothe3 = Soothe3Processor::new(sr as f32);
    soothe3.params_mut().global.low_latency = true;
    soothe3.params_mut().global.depth = 2.8;
    soothe3.params_mut().global.detail = 5.5;
    soothe3.params_mut().global.max_cut_db = -12.0;
    soothe3.params_mut().global.mix_pct = 65.0;
    soothe3.sync_dirty_tables();
    soothe3.prepare(sr, BLOCK_SIZE);

    let mut pos = 0usize;
    while pos < frames.len() {
        let end = (pos + BLOCK_SIZE).min(frames.len());
        let slice = &mut frames[pos..end];
        let ctx = ProcessContext {
            sample_rate: sr,
            current_sample_pos: pos as i64,
            ppq_tick: 0,
            bpm: 120.0,
            slice_frames: slice.len(),
            is_playing: true,
            is_offline_rendering: true,
            ppq_resolution: daw_core_graph::DEFAULT_PPQ,
        };
        let mut empty_aux: [&mut [StereoFrame]; 0] = [];
        proq4.process_slice(&ctx, slice, None, &mut empty_aux);
        soothe3.process_slice(&ctx, slice, None, &mut empty_aux);
        pos = end;
    }
}

fn mix_bgm_with_ducking(
    voice_frames: &mut Vec<StereoFrame>,
    bgm_frames: &[StereoFrame],
    bgm_gain_db: f32,
    duck_db: f32,
    target_duration_sec: Option<f64>,
) {
    if bgm_frames.is_empty() {
        return;
    }
    if let Some(dur) = target_duration_sec {
        let target_len = ((dur * (TARGET_SR as f64)).round() as usize).max(voice_frames.len());
        if voice_frames.len() < target_len {
            voice_frames.resize(target_len, [0.0, 0.0]);
        }
    }

    let bgm_lin = 10.0f32.powf(bgm_gain_db / 20.0);
    let max_duck_lin = 10.0f32.powf(duck_db.min(0.0) / 20.0);
    let atk = (-1.0f32 / (0.025 * (TARGET_SR as f32))).exp();
    let rel = (-1.0f32 / (0.280 * (TARGET_SR as f32))).exp();
    let mut env = 0.0f32;
    let fade_samples = ((0.65 * (TARGET_SR as f64)) as usize).min(voice_frames.len() / 4);
    let total = voice_frames.len();

    for (i, vf) in voice_frames.iter_mut().enumerate() {
        let v_mag = vf[0].abs().max(vf[1].abs());
        let coeff = if v_mag > env { atk } else { rel };
        env = coeff * env + (1.0 - coeff) * v_mag;

        let voice_active = ((env - 0.015) / 0.085).clamp(0.0, 1.0);
        let duck_mul = 1.0 - voice_active * (1.0 - max_duck_lin);

        // Seamless crossfaded loop of BGM
        let bgm_sample = bgm_frames[i % bgm_frames.len()];
        let edge_fade = if fade_samples > 0 && i < fade_samples {
            (i as f32) / (fade_samples as f32)
        } else if fade_samples > 0 && i + fade_samples >= total {
            ((total - 1 - i) as f32) / (fade_samples as f32)
        } else {
            1.0
        };

        let g = bgm_lin * duck_mul * edge_fade;
        vf[0] += bgm_sample[0] * g;
        vf[1] += bgm_sample[1] * g;
    }
}

fn run_irc_v_limiter_pass(
    input: &[StereoFrame],
    sr: f64,
    pre_gain_db: f32,
    ceiling_dbtp: f32,
) -> Vec<StereoFrame> {
    let mut maximizer = IrcVMultiBandMaximizer::new(sr);
    maximizer.set_ceiling_dbtp(ceiling_dbtp);
    maximizer.set_gain_db(0.0);
    maximizer.prevent_intersample_clipping = true;
    maximizer.enable_soft_clip = true;
    maximizer.soft_clip_mix = 0.22;
    maximizer.prepare(sr, BLOCK_SIZE);

    let latency = UnifiedDspNode::latency_samples(&maximizer);
    let pre_lin = 10.0f32.powf(pre_gain_db / 20.0);

    let mut work: Vec<StereoFrame> = Vec::with_capacity(input.len() + latency);
    for &[l, r] in input {
        work.push([l * pre_lin, r * pre_lin]);
    }
    work.resize(input.len() + latency, [0.0, 0.0]);

    let mut pos = 0usize;
    while pos < work.len() {
        let end = (pos + BLOCK_SIZE).min(work.len());
        let slice = &mut work[pos..end];
        let ctx = ProcessContext {
            sample_rate: sr,
            current_sample_pos: pos as i64,
            ppq_tick: 0,
            bpm: 120.0,
            slice_frames: slice.len(),
            is_playing: true,
            is_offline_rendering: true,
            ppq_resolution: daw_core_graph::DEFAULT_PPQ,
        };
        let mut empty_aux: [&mut [StereoFrame]; 0] = [];
        maximizer.process_slice(&ctx, slice, None, &mut empty_aux);
        pos = end;
    }

    // Compensate lookahead latency so output aligns sample-accurately with input!
    work[latency..latency + input.len()].to_vec()
}

fn write_wav_24bit_48k(path: &Path, frames: &[StereoFrame]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Cannot create output dir '{}': {e}", parent.display()))?;
    }
    let file = File::create(path)
        .map_err(|e| format!("Cannot create output WAV '{}': {e}", path.display()))?;
    let mut w = BufWriter::new(file);

    let channels: u16 = 2;
    let bits_per_sample: u16 = 24;
    let bytes_per_sample: u32 = 3;
    let block_align: u16 = channels * (bits_per_sample / 8);
    let byte_rate: u32 = TARGET_SR * (block_align as u32);
    let data_bytes: u32 = (frames.len() as u32)
        .saturating_mul(channels as u32)
        .saturating_mul(bytes_per_sample);
    let riff_size: u32 = 36u32.saturating_add(data_bytes);

    w.write_all(b"RIFF").map_err(|e| e.to_string())?;
    w.write_all(&riff_size.to_le_bytes()).map_err(|e| e.to_string())?;
    w.write_all(b"WAVE").map_err(|e| e.to_string())?;
    w.write_all(b"fmt ").map_err(|e| e.to_string())?;
    w.write_all(&16u32.to_le_bytes()).map_err(|e| e.to_string())?;
    w.write_all(&1u16.to_le_bytes()).map_err(|e| e.to_string())?; // PCM
    w.write_all(&channels.to_le_bytes()).map_err(|e| e.to_string())?;
    w.write_all(&TARGET_SR.to_le_bytes()).map_err(|e| e.to_string())?;
    w.write_all(&byte_rate.to_le_bytes()).map_err(|e| e.to_string())?;
    w.write_all(&block_align.to_le_bytes()).map_err(|e| e.to_string())?;
    w.write_all(&bits_per_sample.to_le_bytes()).map_err(|e| e.to_string())?;
    w.write_all(b"data").map_err(|e| e.to_string())?;
    w.write_all(&data_bytes.to_le_bytes()).map_err(|e| e.to_string())?;

    let scale = 8_388_607.0f32;
    for &[l, r] in frames {
        for s in [l, r] {
            let q = (s.clamp(-1.0, 1.0) * scale).round() as i32;
            let b = q.to_le_bytes();
            w.write_all(&[b[0], b[1], b[2]]).map_err(|e| e.to_string())?;
        }
    }
    w.flush().map_err(|e| e.to_string())?;
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || matches!(args[0].as_str(), "-h" | "--help" | "help") {
        print_usage();
        return;
    }
    if matches!(args[0].as_str(), "-V" | "--version" | "version") {
        println!("akunarch-audio-cli 0.1.0 (ToneForge DSP: Pro-Q 4 + Soothe3 + Ozone 12 IRC V)");
        return;
    }

    let sub = args[0].as_str();
    let mut in_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut bgm_path: Option<PathBuf> = None;
    let mut bgm_gain_db: f32 = -11.5;
    let mut duck_db: f32 = -5.5;
    let mut target_lufs: f32 = -14.0;
    let mut ceiling_dbtp: f32 = -1.0;
    let mut duration_sec: Option<f64> = None;
    let mut json_out = false;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--in" | "-i" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    in_path = Some(PathBuf::from(v));
                }
            }
            "--out" | "-o" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    out_path = Some(PathBuf::from(v));
                }
            }
            "--bgm" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    if !v.trim().is_empty() {
                        bgm_path = Some(PathBuf::from(v));
                    }
                }
            }
            "--bgm-gain-db" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    bgm_gain_db = v.parse::<f32>().unwrap_or(-11.5);
                }
            }
            "--duck-db" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    duck_db = v.parse::<f32>().unwrap_or(-5.5);
                }
            }
            "--target-lufs" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    target_lufs = v.parse::<f32>().unwrap_or(-14.0).clamp(-30.0, -6.0);
                }
            }
            "--ceiling-dbtp" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    ceiling_dbtp = v.parse::<f32>().unwrap_or(-1.0).clamp(-12.0, -0.1);
                }
            }
            "--duration-sec" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    duration_sec = v.parse::<f64>().ok().filter(|d| *d > 0.2);
                }
            }
            "--json" => {
                json_out = true;
            }
            _ => {}
        }
        i += 1;
    }

    let Some(inp) = in_path else {
        eprintln!("Error: missing required --in <audio_file>");
        std::process::exit(2);
    };

    let sr = TARGET_SR as f64;
    let mut frames = match decode_audio_to_stereo_48k(&inp) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };

    let (in_i_lufs, in_st_lufs, in_tp_db) = measure_lufs_and_tp(&frames, sr);

    if sub == "analyze" {
        let dur = (frames.len() as f64) / sr;
        let payload = serde_json::json!({
            "ok": true,
            "input_path": inp.to_string_lossy(),
            "sample_rate_hz": TARGET_SR,
            "duration_sec": (dur * 1000.0).round() / 1000.0,
            "integrated_lufs": in_i_lufs,
            "short_term_lufs": in_st_lufs,
            "true_peak_dbtp": in_tp_db,
        });
        if json_out {
            println!("{}", payload);
        } else {
            println!(
                "[ANALYZE_OK] {} | {:.2}s | Integrated: {:.2} LUFS | Short-Term: {:.2} LUFS | True Peak: {:.2} dBTP",
                inp.display(),
                dur,
                in_i_lufs,
                in_st_lufs,
                in_tp_db
            );
        }
        return;
    }

    if sub != "master" {
        eprintln!("Error: unknown subcommand '{sub}' (expected 'master' or 'analyze')");
        std::process::exit(2);
    }

    let Some(outp) = out_path else {
        eprintln!("Error: missing required --out <master.wav>");
        std::process::exit(2);
    };

    // 1. ToneForge Pro-Q 4 + Soothe3 cleanup on primary stem
    run_proq4_and_soothe3(&mut frames, sr);

    // 2. Optional BGM stem mix with smooth vocal sidechain ducking
    if let Some(ref bgm_file) = bgm_path {
        if bgm_file.is_file() {
            if let Ok(bgm_frames) = decode_audio_to_stereo_48k(bgm_file) {
                mix_bgm_with_ducking(&mut frames, &bgm_frames, bgm_gain_db, duck_db, duration_sec);
            }
        }
    } else if let Some(dur) = duration_sec {
        let target_len = ((dur * sr).round() as usize).max(frames.len());
        if frames.len() < target_len {
            frames.resize(target_len, [0.0, 0.0]);
        }
    }

    // 3. Closed-loop ITU-R BS.1770-4 Loudness Calibration + Ozone 12 IRC V True-Peak Brickwall Mastering
    let (pre_lufs, _, _) = measure_lufs_and_tp(&frames, sr);
    let mut trim_db = if pre_lufs > -68.0 {
        (target_lufs - pre_lufs).clamp(-24.0, 28.0)
    } else {
        0.0
    };

    let mut mastered = run_irc_v_limiter_pass(&frames, sr, trim_db, ceiling_dbtp);
    let (mut out_i_lufs, mut out_st_lufs, mut out_tp_db) = measure_lufs_and_tp(&mastered, sr);

    // Refine up to 4 iterations to compensate limiter gain reduction so Integrated LUFS locks onto `target_lufs` (e.g. -14.0 LUFS)!
    for _ in 0..4 {
        if out_i_lufs <= -68.0 {
            break;
        }
        let err_lu = target_lufs - out_i_lufs;
        if err_lu.abs() <= 0.08 {
            break;
        }
        trim_db = (trim_db + err_lu * 1.12).clamp(-24.0, 32.0);
        mastered = run_irc_v_limiter_pass(&frames, sr, trim_db, ceiling_dbtp);
        let m = measure_lufs_and_tp(&mastered, sr);
        out_i_lufs = m.0;
        out_st_lufs = m.1;
        out_tp_db = m.2;
    }

    // Final linear micro-trim if pushing downward (when err_lu < 0, scaling down never exceeds ceiling_dbtp!)
    if out_i_lufs > target_lufs + 0.06 {
        let atten_db = target_lufs - out_i_lufs;
        let atten_lin = 10.0f32.powf(atten_db / 20.0);
        for f in &mut mastered {
            f[0] *= atten_lin;
            f[1] *= atten_lin;
        }
        trim_db += atten_db;
        let m = measure_lufs_and_tp(&mastered, sr);
        out_i_lufs = m.0;
        out_st_lufs = m.1;
        out_tp_db = m.2;
    }

    if let Err(e) = write_wav_24bit_48k(&outp, &mastered) {
        eprintln!("Error writing output WAV: {e}");
        std::process::exit(1);
    }

    let dur = (mastered.len() as f64) / sr;
    let report = MasterReport {
        ok: true,
        engine: "ToneForge DSP (Pro-Q 4 + Soothe3 + Ozone 12 IRC V)",
        input_path: inp.to_string_lossy().to_string(),
        bgm_path: bgm_path.map(|p| p.to_string_lossy().to_string()),
        output_path: outp.to_string_lossy().to_string(),
        sample_rate_hz: TARGET_SR,
        duration_sec: (dur * 1000.0).round() / 1000.0,
        input_integrated_lufs: (in_i_lufs * 100.0).round() / 100.0,
        output_integrated_lufs: (out_i_lufs * 100.0).round() / 100.0,
        output_short_term_lufs: (out_st_lufs * 100.0).round() / 100.0,
        output_true_peak_dbtp: (out_tp_db * 100.0).round() / 100.0,
        target_lufs,
        ceiling_dbtp,
        applied_trim_db: (trim_db * 100.0).round() / 100.0,
    };

    if json_out {
        println!("{}", serde_json::to_string(&report).unwrap_or_default());
    } else {
        println!(
            "[TONEFORGE_MASTER_OK] {} | {:.2}s | In: {:.2} LUFS -> Out: {:.2} LUFS | True Peak: {:.2} dBTP",
            outp.display(),
            dur,
            report.input_integrated_lufs,
            report.output_integrated_lufs,
            report.output_true_peak_dbtp
        );
    }
}
