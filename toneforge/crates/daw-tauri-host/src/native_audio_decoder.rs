//! Native Multi-Format Audio Decoder & Windows File/Folder Dialog Helpers (`native_audio_decoder.rs`).
//!
//! Eliminates browser `<input type="file">` sandboxing and `%TEMP%` path loss by:
//! 1. Opening real Windows Explorer dialogs via `rfd::FileDialog` to obtain true disk `PathBuf`s.
//! 2. Decoding `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`, and `.aiff` directly in Rust via
//!    `symphonia` (`v0.5.5`) with zero JSON IPC chunk serialization overhead.
//! 3. Providing exact-sample-rate 16-bit RIFF/WAVE binary encoding (`encode_exact_sr_pcm16_wav_bytes`)
//!    so the WebView's `RealtimeSpeakerEngine` (`AudioContext.decodeAudioData`) receives a single
//!    zero-JSON `ArrayBuffer` over Tauri binary `Response`.

use daw_core_graph::StereoFrame;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Full decoded stereo audio buffer + filesystem provenance returned by [`decode_audio_file_native`].
#[derive(Clone, Debug)]
pub struct NativeDecodedAudio {
    pub file_path: PathBuf,
    pub file_name: String,
    pub parent_dir: PathBuf,
    pub format_label: String,
    pub sample_rate_hz: u32,
    pub channels: usize,
    pub duration_seconds: f64,
    pub frames: Vec<StereoFrame>,
}

/// Lightweight metadata DTO returned to Svelte 5 UI when a native audio file is picked & loaded.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NativeLoadedAudioDto {
    pub file_path: String,
    pub file_name: String,
    pub parent_dir: String,
    pub format_label: String,
    pub sample_rate_hz: u32,
    pub total_frames: usize,
    pub duration_seconds: f64,
}

impl NativeDecodedAudio {
    pub fn to_dto(&self) -> NativeLoadedAudioDto {
        NativeLoadedAudioDto {
            file_path: self.file_path.display().to_string(),
            file_name: self.file_name.clone(),
            parent_dir: self.parent_dir.display().to_string(),
            format_label: self.format_label.clone(),
            sample_rate_hz: self.sample_rate_hz,
            total_frames: self.frames.len(),
            duration_seconds: self.duration_seconds,
        }
    }
}

fn infer_format_label(path: &Path) -> String {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .as_deref()
    {
        Some("wav") | Some("wave") => "WAV".to_string(),
        Some("mp3") => "MP3".to_string(),
        Some("flac") => "FLAC".to_string(),
        Some("ogg") | Some("oga") => "OGG Vorbis".to_string(),
        Some("m4a") | Some("mp4") => "M4A / AAC".to_string(),
        Some("aac") => "AAC".to_string(),
        Some("aif") | Some("aiff") => "AIFF".to_string(),
        Some(other) => other.to_ascii_uppercase(),
        None => "AUDIO".to_string(),
    }
}

/// Decodes an audio file (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`, `.aiff`) from a real
/// filesystem path into interleaved `[L, R]` `f32` `StereoFrame`s in `[-1.0, 1.0]`.
pub fn decode_audio_file_native(path: impl AsRef<Path>) -> Result<NativeDecodedAudio, String> {
    let raw_path = path.as_ref();
    if !raw_path.is_file() {
        return Err(format!(
            "Audio file does not exist on disk: {}",
            raw_path.display()
        ));
    }
    let canonical = std::fs::canonicalize(raw_path).unwrap_or_else(|_| raw_path.to_path_buf());
    // Strip Windows `\\?\` extended-length prefix for clean user-facing UI paths & Explorer links
    let clean_path = {
        let s = canonical.to_string_lossy();
        if let Some(stripped) = s.strip_prefix(r"\\?\") {
            PathBuf::from(stripped)
        } else {
            canonical
        }
    };

    let file_name = clean_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("audio.wav")
        .to_string();
    let parent_dir = clean_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let format_label = infer_format_label(&clean_path);

    match decode_via_symphonia(&clean_path) {
        Ok((frames, sample_rate_hz, channels)) if !frames.is_empty() => {
            let duration_seconds = (frames.len() as f64) / (sample_rate_hz.max(1) as f64);
            Ok(NativeDecodedAudio {
                file_path: clean_path,
                file_name,
                parent_dir,
                format_label,
                sample_rate_hz,
                channels,
                duration_seconds,
                frames,
            })
        }
        sym_res => {
            // Fallback to ToneForge's native RIFF/WAVE reader for custom studio float/24-bit WAV chunks
            if let Ok(wav) = crate::offline_render_engine::read_wav_file(&clean_path) {
                if !wav.frames.is_empty() {
                    let sr = wav.sample_rate_hz.max(8_000);
                    let dur = (wav.frames.len() as f64) / (sr as f64);
                    return Ok(NativeDecodedAudio {
                        file_path: clean_path,
                        file_name,
                        parent_dir,
                        format_label,
                        sample_rate_hz: sr,
                        channels: wav.num_channels as usize,
                        duration_seconds: dur,
                        frames: wav.frames,
                    });
                }
            }
            match sym_res {
                Err(e) => Err(e),
                Ok(_) => Err(format!(
                    "Decoded audio file '{}' contains 0 audio frames",
                    clean_path.display()
                )),
            }
        }
    }
}

fn decode_via_symphonia(path: &Path) -> Result<(Vec<StereoFrame>, u32, usize), String> {
    let file = File::open(path)
        .map_err(|e| format!("Failed to open '{}': {}", path.display(), e))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let fmt_opts = FormatOptions {
        enable_gapless: true,
        ..Default::default()
    };
    let meta_opts = MetadataOptions::default();
    let dec_opts = DecoderOptions::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &fmt_opts, &meta_opts)
        .map_err(|e| format!("Unsupported or corrupt audio container '{}': {}", path.display(), e))?;

    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| format!("No decodable audio track found in '{}'", path.display()))?;

    let track_id = track.id;
    let mut sample_rate_hz = track.codec_params.sample_rate.unwrap_or(48_000);
    let mut channels = track
        .codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(2)
        .max(1);
    let est_frames = track.codec_params.n_frames.unwrap_or(0) as usize;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &dec_opts)
        .map_err(|e| format!("Unsupported audio codec in '{}': {}", path.display(), e))?;

    let mut frames: Vec<StereoFrame> = Vec::with_capacity(est_frames.min(48_000 * 60 * 20));
    let mut sample_buf: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(pkt) => pkt,
            Err(SymphoniaError::IoError(ref io_err))
                if io_err.kind() == ErrorKind::UnexpectedEof =>
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

        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                let spec = *audio_buf.spec();
                sample_rate_hz = spec.rate.max(8_000);
                let ch_count = spec.channels.count().max(1);
                channels = ch_count;

                let cap = audio_buf.capacity() as u64;
                let needs_realloc = sample_buf
                    .as_ref()
                    .map(|sb| sb.capacity() < (cap as usize) * ch_count)
                    .unwrap_or(true);
                if needs_realloc {
                    sample_buf = Some(SampleBuffer::<f32>::new(cap, spec));
                }

                if let Some(ref mut sb) = sample_buf {
                    sb.copy_interleaved_ref(audio_buf);
                    let interleaved = sb.samples();
                    let pkt_frames = interleaved.len() / ch_count;
                    frames.reserve(pkt_frames);
                    if ch_count == 1 {
                        for &m in &interleaved[..pkt_frames] {
                            let s = if m.is_finite() { m } else { 0.0 };
                            frames.push([s, s]);
                        }
                    } else {
                        for f in 0..pkt_frames {
                            let base = f * ch_count;
                            let l = interleaved[base];
                            let r = interleaved[base + 1];
                            frames.push([
                                if l.is_finite() { l } else { 0.0 },
                                if r.is_finite() { r } else { 0.0 },
                            ]);
                        }
                    }
                }
            }
            Err(SymphoniaError::DecodeError(_)) => {
                // Skip damaged compressed packet and continue decoding seamlessly
                continue;
            }
            Err(SymphoniaError::IoError(ref io_err))
                if io_err.kind() == ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(_) => break,
        }
    }

    Ok((frames, sample_rate_hz.max(8_000), channels))
}

/// Encodes stereo `f32` frames into a standard 16-bit PCM RIFF/WAVE byte buffer preserving the
/// exact `sample_rate_hz` (`44100`, `48000`, `32000`, `22050`, etc.) for zero-JSON WebAudio playback.
pub fn encode_exact_sr_pcm16_wav_bytes(frames: &[StereoFrame], sample_rate_hz: u32) -> Vec<u8> {
    let sr = sample_rate_hz.max(8_000);
    let num_channels: u16 = 2;
    let bits_per_sample: u16 = 16;
    let block_align: u16 = num_channels * (bits_per_sample / 8);
    let byte_rate: u32 = sr * (block_align as u32);
    let data_bytes_len = (frames.len() * (block_align as usize)).min(u32::MAX as usize - 44) as u32;
    let riff_chunk_size = 36u32.saturating_add(data_bytes_len);

    let mut out = Vec::with_capacity(44 + data_bytes_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_chunk_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&num_channels.to_le_bytes());
    out.extend_from_slice(&sr.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes_len.to_le_bytes());

    let max_frames = (data_bytes_len as usize) / (block_align as usize);
    for f in &frames[..max_frames] {
        let l = (f[0].clamp(-1.0, 1.0) * 32767.0).round() as i16;
        let r = (f[1].clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&l.to_le_bytes());
        out.extend_from_slice(&r.to_le_bytes());
    }
    out
}

/// Opens a native Windows file-open dialog (`IFileOpenDialog` via `rfd`) filtered for studio audio formats.
pub fn pick_audio_file_dialog_native(initial_dir: Option<&Path>) -> Option<PathBuf> {
    let mut dlg = rfd::FileDialog::new()
        .set_title("ToneForge Studio — Chọn bài nhạc gốc từ ổ cứng")
        .add_filter(
            "Studio Audio (*.wav, *.mp3, *.flac, *.ogg, *.m4a, *.aac, *.aiff)",
            &["wav", "wave", "mp3", "flac", "ogg", "oga", "m4a", "aac", "aif", "aiff"],
        )
        .add_filter("All Files (*.*)", &["*"]);
    if let Some(dir) = initial_dir {
        if dir.is_dir() {
            dlg = dlg.set_directory(dir);
        }
    }
    dlg.pick_file()
}

/// Opens a native Windows folder-picker dialog (`IFileOpenDialog` `FOS_PICKFOLDERS` via `rfd`).
pub fn pick_directory_dialog_native(title: &str, initial_dir: Option<&Path>) -> Option<PathBuf> {
    let mut dlg = rfd::FileDialog::new().set_title(title);
    if let Some(dir) = initial_dir {
        if dir.is_dir() {
            dlg = dlg.set_directory(dir);
        }
    }
    dlg.pick_folder()
}
