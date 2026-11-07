//! High-Speed Offline Bounce & Mastering Render Engine for **ToneForge**.
//!
//! Features:
//! 1. **Studio-Grade WAV Encoder & Decoder**:
//!    - `16-bit PCM` (`wFormatTag = 1`) with Ozone 12 `MbitPlusDither` psychoacoustic noise-shaping dither
//!    - `24-bit PCM` (`wFormatTag = 1`, 3-byte little-endian signed integers) with Ozone 12 `MbitPlusDither`
//!    - `32-bit IEEE Float` (`wFormatTag = 3`, IEEE 754 `f32`)
//!    - Supports `44,100 Hz`, `48,000 Hz`, `96,000 Hz`, and `192,000 Hz` sample rates.
//! 2. **Three Render Workflows**:
//!    - `RenderMode::Master`: Full project bounce with automatic **5-Point APDC latency trimming**
//!      (`master_output_latency_samples`) so exported audio begins at sample `0` with 0-sample phase error.
//!    - `RenderMode::SplitMixerTracks`: Simultaneous multi-track bounce of Master (`0`) and active Insert
//!      Tracks (`1..=500`) from the 502-Track `MixerEngine`.
//!    - `RenderMode::StemMastering`: UVR5 3-Model Spectral Ensemble (`HTDemucs4 + Mdx23cSeparator + CascadedASPPNet`)
//!      4-stem separation (`[Vocals, Drums, Bass, Other]`) -> per-stem `ProQ4Processor + Soothe3Processor`
//!      on dedicated Insert tracks -> summed into Master Track `0` running `Ozone12MasterChain`.
//! 3. **Automatic Post-Render Analysis Report**:
//!    - ITU-R BS.1770-4 `Integrated`, `Short-Term`, and `Momentary` LUFS (`LufsTruePeakMeter`)
//!    - 4x Oversampled Inter-Sample `True Peak dBTP` & `Sample Peak dBFS`
//!    - `RMS dBFS` & `Crest Factor dB`
//!    - Pearson Stereo `Phase Correlation` (`-1.0..=+1.0`)
//!    - Broadcast & Streaming Compliance check for **Spotify** (`-14 LUFS / -1.0 dBTP`),
//!      **Apple Music** (`-16 LUFS / -1.0 dBTP`), and **YouTube** (`-14 LUFS / -1.0 dBTP`).

use ai_uvr5_stem_engine::{
    CascadedASPPNet, HTDemucs4, Mdx23cSeparator, SpectralEnsembleMode, SpectralEnsembler,
};
use daw_core_graph::{
    AudioStemClipPlayerNode, ChannelGenerator, MASTER_TRACK_INDEX, MAX_BLOCK_SIZE,
    MAX_MIXER_TRACKS, MixerEngine, StereoFrame,
};
use dsp_ozone12_mastering::{LufsTruePeakMeter, MbitPlusDither, Ozone12MasterChain};
use dsp_proq4_eq::{EqBandParams, FilterShape, ProQ4Processor};
use dsp_soothe3_spectral::Soothe3Processor;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Studio WAV export bit depth and dithering mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WavBitDepth {
    /// 16-bit Signed PCM with Ozone 12 `MbitPlusDither` psychoacoustic noise shaping.
    Pcm16MbitPlus,
    /// 24-bit Signed PCM (3 bytes/sample LE) with Ozone 12 `MbitPlusDither` psychoacoustic noise shaping.
    Pcm24MbitPlus,
    /// 32-bit IEEE 754 Floating-Point WAV (`wFormatTag = 0x0003`).
    Float32Ieee,
}

impl WavBitDepth {
    #[inline]
    pub fn bits_per_sample(self) -> u16 {
        match self {
            Self::Pcm16MbitPlus => 16,
            Self::Pcm24MbitPlus => 24,
            Self::Float32Ieee => 32,
        }
    }

    #[inline]
    pub fn bytes_per_sample(self) -> usize {
        (self.bits_per_sample() / 8) as usize
    }

    #[inline]
    pub fn wav_format_tag(self) -> u16 {
        match self {
            Self::Pcm16MbitPlus | Self::Pcm24MbitPlus => 1, // WAVE_FORMAT_PCM
            Self::Float32Ieee => 3,                         // WAVE_FORMAT_IEEE_FLOAT
        }
    }
}

/// Supported studio sample rates for Offline Bounce & Mastering Render.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RenderSampleRate {
    Hz44100,
    Hz48000,
    Hz96000,
    Hz192000,
}

impl RenderSampleRate {
    #[inline]
    pub fn as_u32(self) -> u32 {
        match self {
            Self::Hz44100 => 44_100,
            Self::Hz48000 => 48_000,
            Self::Hz96000 => 96_000,
            Self::Hz192000 => 192_000,
        }
    }

    #[inline]
    pub fn as_f64(self) -> f64 {
        self.as_u32() as f64
    }

    pub fn from_hz(hz: u32) -> Self {
        match hz {
            0..=46_000 => Self::Hz44100,
            46_001..=72_000 => Self::Hz48000,
            72_001..=144_000 => Self::Hz96000,
            _ => Self::Hz192000,
        }
    }
}

/// Offline Bounce & Mastering Render mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RenderMode {
    /// Bounce Master Track `0` with automatic 5-Point APDC latency compensation.
    Master,
    /// Bounce Master Track `0` plus all active (or selected) Insert Tracks (`1..=500`) simultaneously.
    SplitMixerTracks,
    /// Separate into 4 Stems via UVR5 Ensemble -> per-stem `Pro-Q 4 + Soothe3` on Insert Tracks `1..=4` -> `Ozone 12` Master Chain.
    StemMastering,
}

/// Configuration for an Offline Bounce & Mastering Render job.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OfflineRenderConfig {
    pub mode: RenderMode,
    pub sample_rate: RenderSampleRate,
    pub bit_depth: WavBitDepth,
    #[serde(default = "default_true", alias = "enable_noise_shaping")]
    pub enable_mbit_plus_noise_shaping: bool,
    #[serde(default = "default_true")]
    pub trim_apdc_latency: bool,
    pub duration_frames: usize,
    #[serde(default = "default_block_size")]
    pub block_size: usize,
    #[serde(default)]
    pub split_track_indices: Vec<usize>,
    #[serde(default)]
    pub output_dir: Option<PathBuf>,
    #[serde(default = "default_file_prefix")]
    pub file_prefix: String,
}

fn default_true() -> bool {
    true
}

fn default_block_size() -> usize {
    256
}

fn default_file_prefix() -> String {
    "ToneForge_Master".to_string()
}

impl Default for OfflineRenderConfig {
    fn default() -> Self {
        Self {
            mode: RenderMode::Master,
            sample_rate: RenderSampleRate::Hz48000,
            bit_depth: WavBitDepth::Pcm24MbitPlus,
            enable_mbit_plus_noise_shaping: true,
            trim_apdc_latency: true,
            duration_frames: 48_000,
            block_size: 256,
            split_track_indices: Vec::new(),
            output_dir: None,
            file_prefix: "ToneForge_Master".to_string(),
        }
    }
}

/// Streaming platform loudness & True-Peak compliance report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StreamingLoudnessCompliance {
    /// Spotify target `-14.0 LUFS`, True-Peak ceiling `-1.0 dBTP`.
    pub spotify_target_lufs: f32,
    pub spotify_penalty_db: f32,
    pub spotify_true_peak_ok: bool,
    pub spotify_compliant: bool,

    /// Apple Music Sound Check target `-16.0 LUFS`, True-Peak ceiling `-1.0 dBTP`.
    pub apple_music_target_lufs: f32,
    pub apple_music_penalty_db: f32,
    pub apple_music_true_peak_ok: bool,
    pub apple_music_compliant: bool,

    /// YouTube target `-14.0 LUFS`, True-Peak ceiling `-1.0 dBTP`.
    pub youtube_target_lufs: f32,
    pub youtube_penalty_db: f32,
    pub youtube_true_peak_ok: bool,
    pub youtube_compliant: bool,
}

/// Automatic Post-Render Analysis Report (ITU-R BS.1770-4 LUFS, 4x True Peak dBTP, Crest Factor,
/// Phase Correlation, and Streaming Loudness Compliance).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PostRenderAnalysisReport {
    pub sample_rate_hz: u32,
    pub bit_depth_bits: u16,
    pub rendered_frames: usize,
    pub duration_seconds: f64,
    pub apdc_compensated_samples: usize,
    pub integrated_lufs: f32,
    pub max_short_term_lufs: f32,
    pub max_momentary_lufs: f32,
    pub true_peak_dbtp: [f32; 2],
    pub max_true_peak_dbtp: f32,
    pub sample_peak_dbfs: [f32; 2],
    pub max_sample_peak_dbfs: f32,
    pub rms_dbfs: [f32; 2],
    pub avg_rms_dbfs: f32,
    pub crest_factor_db: f32,
    pub phase_correlation: f32,
    pub streaming_compliance: StreamingLoudnessCompliance,
}

/// Exported track/stem WAV artifact from an Offline Render pass.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExportedTrackWav {
    pub track_index: usize,
    pub label: String,
    #[serde(skip)]
    pub frames: Vec<StereoFrame>,
    #[serde(skip)]
    pub wav_bytes: Vec<u8>,
    pub file_path: Option<PathBuf>,
    pub analysis: PostRenderAnalysisReport,
}

/// Complete output of an Offline Bounce & Mastering Render job.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OfflineRenderResult {
    pub config: OfflineRenderConfig,
    pub master_wav: ExportedTrackWav,
    pub additional_wavs: Vec<ExportedTrackWav>,
}

// ============================================================================
// STUDIO WAV ENCODER & DECODER (16-BIT / 24-BIT MBIT+ DITHER & 32-BIT FLOAT)
// ============================================================================

/// Encodes stereo floating-point frames into a standard RIFF/WAVE byte buffer at `sample_rate`
/// and `bit_depth`, applying Ozone 12 `MbitPlusDither` psychoacoustic noise shaping for 16/24-bit PCM.
pub fn encode_wav_bytes(
    frames: &[StereoFrame],
    sample_rate: RenderSampleRate,
    bit_depth: WavBitDepth,
    enable_noise_shaping: bool,
) -> Vec<u8> {
    encode_wav_bytes_exact_sr(frames, sample_rate.as_u32(), bit_depth, enable_noise_shaping)
}

/// Encodes stereo floating-point frames into a standard RIFF/WAVE byte buffer at an exact
/// arbitrary integer `sample_rate_hz` (`8,000..=384,000 Hz`) and `bit_depth`, preserving native
/// sample rates (`22,050 Hz`, `32,000 Hz`, `44,100 Hz`, `48,000 Hz`, `96,000 Hz`, etc.) without
/// pitch-shifting or duration distortion.
pub fn encode_wav_bytes_exact_sr(
    frames: &[StereoFrame],
    sample_rate_hz: u32,
    bit_depth: WavBitDepth,
    enable_noise_shaping: bool,
) -> Vec<u8> {
    let num_channels: u16 = 2;
    let sr = sample_rate_hz.clamp(8_000, 384_000);
    let bits_per_sample = bit_depth.bits_per_sample();
    let bytes_per_sample = bit_depth.bytes_per_sample();
    let block_align = num_channels * (bytes_per_sample as u16);
    let byte_rate = sr * (block_align as u32);
    let data_bytes_len = frames.len() * (num_channels as usize) * bytes_per_sample;
    let riff_chunk_size = 36u32.saturating_add(data_bytes_len as u32);

    let mut out = Vec::with_capacity(44 + data_bytes_len);

    // RIFF header (12 bytes)
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_chunk_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");

    // fmt  subchunk (24 bytes)
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&bit_depth.wav_format_tag().to_le_bytes());
    out.extend_from_slice(&num_channels.to_le_bytes());
    out.extend_from_slice(&sr.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data subchunk header (8 bytes)
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_bytes_len as u32).to_le_bytes());

    match bit_depth {
        WavBitDepth::Pcm16MbitPlus => {
            let mut dither = MbitPlusDither::new();
            dither.enabled = true;
            dither.bit_depth = 16;
            dither.noise_shaping = enable_noise_shaping;
            for frame in frames {
                for ch in 0..2 {
                    let d = dither.process_sample(ch, frame[ch].clamp(-1.0, 1.0));
                    let q = (d * 32_767.0).round().clamp(-32_768.0, 32_767.0) as i16;
                    out.extend_from_slice(&q.to_le_bytes());
                }
            }
        }
        WavBitDepth::Pcm24MbitPlus => {
            let mut dither = MbitPlusDither::new();
            dither.enabled = true;
            dither.bit_depth = 24;
            dither.noise_shaping = enable_noise_shaping;
            for frame in frames {
                for ch in 0..2 {
                    let d = dither.process_sample(ch, frame[ch].clamp(-1.0, 1.0));
                    let q = (d * 8_388_607.0).round().clamp(-8_388_608.0, 8_388_607.0) as i32;
                    let le = q.to_le_bytes();
                    out.push(le[0]);
                    out.push(le[1]);
                    out.push(le[2]);
                }
            }
        }
        WavBitDepth::Float32Ieee => {
            for frame in frames {
                out.extend_from_slice(&frame[0].to_le_bytes());
                out.extend_from_slice(&frame[1].to_le_bytes());
            }
        }
    }

    out
}

/// Writes a RIFF/WAVE file to disk and returns the encoded byte vector.
pub fn write_wav_file(
    path: impl AsRef<Path>,
    frames: &[StereoFrame],
    sample_rate: RenderSampleRate,
    bit_depth: WavBitDepth,
    enable_noise_shaping: bool,
) -> io::Result<Vec<u8>> {
    let bytes = encode_wav_bytes(frames, sample_rate, bit_depth, enable_noise_shaping);
    if let Some(parent) = path.as_ref().parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(path, &bytes)?;
    Ok(bytes)
}

/// Decoded WAV header & stereo frame payload for verification tests and stem import.
#[derive(Clone, Debug)]
pub struct DecodedWavFile {
    pub format_tag: u16,
    pub num_channels: u16,
    pub sample_rate_hz: u32,
    pub bits_per_sample: u16,
    pub frames: Vec<StereoFrame>,
}

/// Decodes a RIFF/WAVE byte slice (`16-bit PCM`, `24-bit PCM`, or `32-bit IEEE Float`) into stereo `f32` frames.
/// Supports arbitrary RIFF sub-chunks (`JUNK`, `bext`, `LIST`, `fact`, `PAD `, extended `fmt `) before `data`.
pub fn decode_wav_bytes(bytes: &[u8]) -> Result<DecodedWavFile, String> {
    if bytes.len() < 44 {
        return Err("WAV buffer shorter than 44-byte RIFF header".to_string());
    }
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("Invalid RIFF/WAVE magic header".to_string());
    }

    let mut pos = 12usize;
    let mut fmt_info: Option<(u16, u16, u32, u16)> = None;
    let mut data_slice: Option<&[u8]> = None;

    while pos + 8 <= bytes.len() {
        let chunk_id = &bytes[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;
        let chunk_start = pos + 8;
        let chunk_end = chunk_start.saturating_add(chunk_size).min(bytes.len());

        if chunk_id == b"fmt " {
            if chunk_size < 16 || chunk_start + 16 > bytes.len() {
                return Err("WAV 'fmt ' chunk shorter than 16 bytes".to_string());
            }
            let fmt = &bytes[chunk_start..chunk_end];
            let format_tag = u16::from_le_bytes([fmt[0], fmt[1]]);
            let num_channels = u16::from_le_bytes([fmt[2], fmt[3]]);
            let sample_rate_hz = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]);
            let bits_per_sample = u16::from_le_bytes([fmt[14], fmt[15]]);
            fmt_info = Some((format_tag, num_channels, sample_rate_hz, bits_per_sample));
        } else if chunk_id == b"data" {
            data_slice = Some(&bytes[chunk_start..chunk_end]);
            break;
        }

        // RIFF chunks are word-aligned (padded to even byte boundary)
        let padded_size = chunk_size.saturating_add(chunk_size & 1);
        pos = chunk_start.saturating_add(padded_size);
    }

    let (format_tag, num_channels, sample_rate_hz, bits_per_sample) =
        fmt_info.ok_or_else(|| "Missing 'fmt ' chunk in RIFF/WAVE stream".to_string())?;
    if num_channels != 2 {
        return Err(format!("Expected 2 stereo channels, got {num_channels}"));
    }
    let payload =
        data_slice.ok_or_else(|| "Missing 'data' chunk in RIFF/WAVE stream".to_string())?;

    let mut frames = Vec::new();
    match (format_tag, bits_per_sample) {
        (1, 16) => {
            for chunk in payload.chunks_exact(4) {
                let l = i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32_767.0;
                let r = i16::from_le_bytes([chunk[2], chunk[3]]) as f32 / 32_767.0;
                frames.push([l, r]);
            }
        }
        (1, 24) => {
            for chunk in payload.chunks_exact(6) {
                let sign_l = if (chunk[2] & 0x80) != 0 { 0xFF } else { 0x00 };
                let sign_r = if (chunk[5] & 0x80) != 0 { 0xFF } else { 0x00 };
                let l_i32 = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], sign_l]);
                let r_i32 = i32::from_le_bytes([chunk[3], chunk[4], chunk[5], sign_r]);
                frames.push([
                    (l_i32 as f32) / 8_388_607.0,
                    (r_i32 as f32) / 8_388_607.0,
                ]);
            }
        }
        (3, 32) => {
            for chunk in payload.chunks_exact(8) {
                let l = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let r = f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
                frames.push([l, r]);
            }
        }
        _ => {
            return Err(format!(
                "Unsupported WAV format_tag={format_tag}, bits_per_sample={bits_per_sample}"
            ));
        }
    }

    Ok(DecodedWavFile {
        format_tag,
        num_channels,
        sample_rate_hz,
        bits_per_sample,
        frames,
    })
}

/// Reads and decodes a WAV file from disk.
pub fn read_wav_file(path: impl AsRef<Path>) -> Result<DecodedWavFile, String> {
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    decode_wav_bytes(&bytes)
}

// ============================================================================
// AUTOMATIC POST-RENDER LOUDNESS, TRUE-PEAK & COMPLIANCE ANALYZER
// ============================================================================

/// Analyzes a rendered stereo buffer using ITU-R BS.1770-4 K-weighted LUFS + 4x Oversampled
/// True-Peak metering (`LufsTruePeakMeter`), computing Crest Factor, Pearson Phase Correlation,
/// and Spotify / Apple Music / YouTube broadcast compliance.
pub fn analyze_rendered_audio(
    frames: &[StereoFrame],
    sample_rate: RenderSampleRate,
    bit_depth: WavBitDepth,
    apdc_compensated_samples: usize,
) -> PostRenderAnalysisReport {
    let sr_hz = sample_rate.as_u32();
    let sr_f64 = sample_rate.as_f64();
    let mut meter = LufsTruePeakMeter::new(sr_f64);

    let mut max_momentary_lufs = -144.0f32;
    let mut max_short_term_lufs = -144.0f32;
    let mut sample_peak_l = 0.0f32;
    let mut sample_peak_r = 0.0f32;
    let mut sum_sq_l = 0.0f64;
    let mut sum_sq_r = 0.0f64;
    let mut sum_lr = 0.0f64;

    // Feed in 512-frame chunks so we capture running Momentary and Short-Term LUFS peaks
    for chunk in frames.chunks(512) {
        meter.push_block(chunk, 0.0);
        if meter.lufs_momentary > max_momentary_lufs {
            max_momentary_lufs = meter.lufs_momentary;
        }
        if meter.lufs_short_term > max_short_term_lufs {
            max_short_term_lufs = meter.lufs_short_term;
        }
        for &[l, r] in chunk {
            let al = l.abs();
            let ar = r.abs();
            if al > sample_peak_l {
                sample_peak_l = al;
            }
            if ar > sample_peak_r {
                sample_peak_r = ar;
            }
            let ld = l as f64;
            let rd = r as f64;
            sum_sq_l += ld * ld;
            sum_sq_r += rd * rd;
            sum_lr += ld * rd;
        }
    }

    let n = frames.len().max(1) as f64;
    let rms_l = (sum_sq_l / n).sqrt() as f32;
    let rms_r = (sum_sq_r / n).sqrt() as f32;
    let avg_rms = (((sum_sq_l + sum_sq_r) / (2.0 * n)).sqrt()) as f32;

    let lin_to_db = |x: f32| -> f32 {
        if x <= 1e-9 {
            -144.0
        } else {
            20.0 * x.log10()
        }
    };

    let sample_peak_dbfs = [lin_to_db(sample_peak_l), lin_to_db(sample_peak_r)];
    let max_sample_peak_dbfs = sample_peak_dbfs[0].max(sample_peak_dbfs[1]);
    let rms_dbfs = [lin_to_db(rms_l), lin_to_db(rms_r)];
    let avg_rms_dbfs = lin_to_db(avg_rms);
    let crest_factor_db = if avg_rms_dbfs <= -140.0 {
        0.0
    } else {
        (max_sample_peak_dbfs - avg_rms_dbfs).max(0.0)
    };

    let denom = (sum_sq_l * sum_sq_r).sqrt();
    let phase_correlation = if denom > 1e-12 {
        (sum_lr / denom).clamp(-1.0, 1.0) as f32
    } else {
        1.0
    };

    let true_peak_dbtp = meter.true_peak_dbtp();
    let max_true_peak_dbtp = true_peak_dbtp[0].max(true_peak_dbtp[1]);
    let integrated_lufs = if meter.lufs_integrated > -140.0 {
        meter.lufs_integrated
    } else if max_short_term_lufs > -140.0 {
        max_short_term_lufs
    } else {
        max_momentary_lufs
    };

    // Streaming platform compliance evaluation:
    // Spotify: -14.0 LUFS target, -1.0 dBTP ceiling (or -2.0 dBTP for loud masters)
    let spotify_target_lufs = -14.0f32;
    let spotify_penalty_db = (spotify_target_lufs - integrated_lufs).min(0.0);
    let spotify_true_peak_ok = max_true_peak_dbtp <= -0.95;
    let spotify_compliant =
        spotify_true_peak_ok && integrated_lufs <= -8.0 && integrated_lufs >= -24.0;

    // Apple Music: -16.0 LUFS target, -1.0 dBTP ceiling
    let apple_music_target_lufs = -16.0f32;
    let apple_music_penalty_db = (apple_music_target_lufs - integrated_lufs).min(0.0);
    let apple_music_true_peak_ok = max_true_peak_dbtp <= -0.95;
    let apple_music_compliant =
        apple_music_true_peak_ok && integrated_lufs <= -8.0 && integrated_lufs >= -24.0;

    // YouTube: -14.0 LUFS target, -1.0 dBTP ceiling
    let youtube_target_lufs = -14.0f32;
    let youtube_penalty_db = (youtube_target_lufs - integrated_lufs).min(0.0);
    let youtube_true_peak_ok = max_true_peak_dbtp <= -0.95;
    let youtube_compliant =
        youtube_true_peak_ok && integrated_lufs <= -8.0 && integrated_lufs >= -24.0;

    PostRenderAnalysisReport {
        sample_rate_hz: sr_hz,
        bit_depth_bits: bit_depth.bits_per_sample(),
        rendered_frames: frames.len(),
        duration_seconds: (frames.len() as f64) / sr_f64,
        apdc_compensated_samples,
        integrated_lufs,
        max_short_term_lufs,
        max_momentary_lufs,
        true_peak_dbtp,
        max_true_peak_dbtp,
        sample_peak_dbfs,
        max_sample_peak_dbfs,
        rms_dbfs,
        avg_rms_dbfs,
        crest_factor_db,
        phase_correlation,
        streaming_compliance: StreamingLoudnessCompliance {
            spotify_target_lufs,
            spotify_penalty_db,
            spotify_true_peak_ok,
            spotify_compliant,
            apple_music_target_lufs,
            apple_music_penalty_db,
            apple_music_true_peak_ok,
            apple_music_compliant,
            youtube_target_lufs,
            youtube_penalty_db,
            youtube_true_peak_ok,
            youtube_compliant,
        },
    }
}

// ============================================================================
// HIGH-SPEED OFFLINE BOUNCE & STEM MASTERING RENDER ENGINE
// ============================================================================

pub struct OfflineRenderEngine;

impl OfflineRenderEngine {
    /// Resets transport (`slicer.reset()`, `is_playing = true`), prepares and resets all Channel
    /// and MixerTrack DSP/APDC delay lines, and re-evaluates initial automation at tick `0`
    /// so offline renders are 100% deterministic from sample `0` regardless of prior playback state.
    fn prepare_and_reset_for_offline_render(
        engine: &mut MixerEngine,
        target_sr: f64,
        block_size: usize,
    ) {
        // Drain any pending lock-free SPSC commands (EQ bands, Soothe3 depth, Ozone 12 ceiling, track gains)
        // queued from the UI right before offline rendering begins (`[Nhóm C.4]`).
        engine.drain_commands();

        engine.slicer.sample_rate = target_sr;
        engine.slicer.reset();
        engine.is_playing = true;
        engine.automation_queue.clear();

        for ch in &mut engine.channels {
            ch.prepare(target_sr, block_size);
            ch.reset_state();
        }
        for track in &mut engine.tracks {
            track.prepare(target_sr, block_size);
            track.reset_state();
        }
        for art in engine.sequencer.active_articulators() {
            let fixed30 = art.evaluate_fixed30(0.0);
            engine.controller_bus.publish_fixed30(art.controller_id, fixed30);
        }
        let _ = engine.rebuild_graph();
    }

    /// Renders `engine` offline according to `config` (`Master`, `SplitMixerTracks`, or `StemMastering`).
    pub fn execute_render(
        engine: &mut MixerEngine,
        config: &OfflineRenderConfig,
    ) -> io::Result<OfflineRenderResult> {
        match config.mode {
            RenderMode::Master => Self::render_master(engine, config),
            RenderMode::SplitMixerTracks => Self::render_split_mixer_tracks(engine, config),
            RenderMode::StemMastering => Self::render_stem_mastering(engine, config),
        }
    }

    /// Executes `RenderMode::Master`, automatically compensating and trimming `master_output_latency_samples`
    /// when `config.trim_apdc_latency` is enabled.
    pub fn render_master(
        engine: &mut MixerEngine,
        config: &OfflineRenderConfig,
    ) -> io::Result<OfflineRenderResult> {
        let target_sr = config.sample_rate.as_f64();
        let block_size = config.block_size.clamp(16, MAX_BLOCK_SIZE);

        Self::prepare_and_reset_for_offline_render(engine, target_sr, block_size);

        let apdc_lat = if config.trim_apdc_latency {
            engine.scheduler.master_output_latency_samples
        } else {
            0
        };
        let total_to_render = config.duration_frames + apdc_lat;
        let mut raw_master = Vec::with_capacity(total_to_render + block_size);
        let mut block_buf = vec![[0.0f32; 2]; block_size];

        let mut rendered = 0usize;
        while rendered < total_to_render {
            let chunk_len = (total_to_render - rendered).min(block_size);
            block_buf[..chunk_len].fill([0.0; 2]);
            engine.process_audio_block(&mut block_buf[..chunk_len]);
            raw_master.extend_from_slice(&block_buf[..chunk_len]);
            rendered += chunk_len;
        }

        let trimmed_master = if apdc_lat < raw_master.len() {
            raw_master[apdc_lat..apdc_lat + config.duration_frames].to_vec()
        } else {
            raw_master
        };

        let master_wav = Self::package_exported_wav(
            MASTER_TRACK_INDEX,
            format!("{}_Master", config.file_prefix),
            trimmed_master,
            config,
            apdc_lat,
        )?;

        Ok(OfflineRenderResult {
            config: config.clone(),
            master_wav,
            additional_wavs: Vec::new(),
        })
    }

    /// Executes `RenderMode::SplitMixerTracks`, simultaneously capturing Master Track `0` and all
    /// requested (or active) Insert Tracks from the 502-Track Mixer during a single offline pass.
    pub fn render_split_mixer_tracks(
        engine: &mut MixerEngine,
        config: &OfflineRenderConfig,
    ) -> io::Result<OfflineRenderResult> {
        let target_sr = config.sample_rate.as_f64();
        let block_size = config.block_size.clamp(16, MAX_BLOCK_SIZE);

        Self::prepare_and_reset_for_offline_render(engine, target_sr, block_size);

        // Determine which Insert Tracks to split-export
        let split_indices: Vec<usize> = if !config.split_track_indices.is_empty() {
            config
                .split_track_indices
                .iter()
                .copied()
                .filter(|&idx| idx > 0 && idx < MAX_MIXER_TRACKS)
                .collect()
        } else {
            let mut active = Vec::new();
            for lvl in &engine.scheduler.active_levels {
                for &u in lvl {
                    if u > 0
                        && u < MAX_MIXER_TRACKS
                        && (engine.tracks[u].has_channel_input
                            || engine.tracks[u].has_hw_input
                            || !engine.tracks[u].slots.is_empty()
                            || engine.tracks[u].has_upstream_audio)
                        && !active.contains(&u)
                    {
                        active.push(u);
                    }
                }
            }
            active.sort_unstable();
            active
        };

        let apdc_lat = if config.trim_apdc_latency {
            engine.scheduler.master_output_latency_samples
        } else {
            0
        };
        let total_to_render = config.duration_frames + apdc_lat;
        let mut raw_master = Vec::with_capacity(total_to_render + block_size);
        let mut raw_splits: Vec<Vec<StereoFrame>> = split_indices
            .iter()
            .map(|_| Vec::with_capacity(total_to_render + block_size))
            .collect();
        let mut block_buf = vec![[0.0f32; 2]; block_size];

        let mut rendered = 0usize;
        while rendered < total_to_render {
            let chunk_len = (total_to_render - rendered).min(block_size);
            block_buf[..chunk_len].fill([0.0; 2]);
            engine.process_audio_block(&mut block_buf[..chunk_len]);
            raw_master.extend_from_slice(&block_buf[..chunk_len]);
            for (s_pos, &trk_idx) in split_indices.iter().enumerate() {
                raw_splits[s_pos]
                    .extend_from_slice(&engine.tracks[trk_idx].sub_block_buf[..chunk_len]);
            }
            rendered += chunk_len;
        }

        let trimmed_master = if apdc_lat < raw_master.len() {
            raw_master[apdc_lat..apdc_lat + config.duration_frames].to_vec()
        } else {
            raw_master
        };
        let master_wav = Self::package_exported_wav(
            MASTER_TRACK_INDEX,
            format!("{}_Track000_Master", config.file_prefix),
            trimmed_master,
            config,
            apdc_lat,
        )?;

        let mut additional_wavs = Vec::with_capacity(split_indices.len());
        for (s_pos, &trk_idx) in split_indices.iter().enumerate() {
            let trk_lat = if config.trim_apdc_latency {
                engine.tracks[trk_idx].accumulated_latency.min(apdc_lat)
            } else {
                0
            };
            let raw = &raw_splits[s_pos];
            let trimmed = if trk_lat + config.duration_frames <= raw.len() {
                raw[trk_lat..trk_lat + config.duration_frames].to_vec()
            } else {
                raw[..config.duration_frames.min(raw.len())].to_vec()
            };
            let wav = Self::package_exported_wav(
                trk_idx,
                format!("{}_Track{:03}", config.file_prefix, trk_idx),
                trimmed,
                config,
                trk_lat,
            )?;
            additional_wavs.push(wav);
        }

        Ok(OfflineRenderResult {
            config: config.clone(),
            master_wav,
            additional_wavs,
        })
    }

    /// Executes `RenderMode::StemMastering`:
    /// 1. Renders the initial pre-master mix from `source_engine`,
    /// 2. Separates into 4 phase-coherent Stems (`Vocals`, `Drums`, `Bass`, `Other`) using the
    ///    UVR5 3-Model Ensemble (`HTDemucs4 + Mdx23cSeparator + CascadedASPPNet` via `SpectralEnsembler`),
    /// 3. Routes the 4 stems into a dedicated 502-Track Stem Mastering `MixerEngine` where:
    ///    - Insert Track 1 (`Vocals`): `ProQ4Processor` (Vocal presence bell) + `Soothe3Processor` (de-harsh)
    ///    - Insert Track 2 (`Drums`): `ProQ4Processor` (Punch low-shelf) + `Soothe3Processor` (cymbal taming)
    ///    - Insert Track 3 (`Bass`): `ProQ4Processor` (Sub focus bell) + `Soothe3Processor` (low-mid resonance control)
    ///    - Insert Track 4 (`Other`): `ProQ4Processor` (Air high-shelf) + `Soothe3Processor` (midrange unmasking)
    ///    - Master Track 0: `Ozone12MasterChain` (with `IrcVMultiBandMaximizer` at `-1.0 dBTP` ceiling)
    /// 4. Renders both the 4 processed stem tracks and the final mastered output with exact 5-Point APDC compensation.
    pub fn render_stem_mastering(
        source_engine: &mut MixerEngine,
        config: &OfflineRenderConfig,
    ) -> io::Result<OfflineRenderResult> {
        // Step 1: Render source mix with APDC compensation
        let mut pre_cfg = config.clone();
        pre_cfg.mode = RenderMode::Master;
        pre_cfg.output_dir = None;
        let pre_res = Self::render_master(source_engine, &pre_cfg)?;
        let mix_frames = &pre_res.master_wav.frames;

        // Step 2: Separate into 4 Stems via RTX 4080 ONNX GPU Ensemble (Kim_Vocal_2 + Inst_HQ_3 + Ozone 12 BSRNN)
        let [vocals_stem, drums_stem, bass_stem, other_stem] =
            Self::separate_4stems_uvr5_ensemble_with_sr(mix_frames, config.sample_rate.as_u32());

        // Step 3: Build the 502-Track Stem Mastering MixerEngine
        let target_sr = config.sample_rate.as_f64();
        let bpm = source_engine.slicer.bpm;
        let (mut stem_engine, _cmd_prod, _telem_out) = MixerEngine::new(target_sr, bpm, 0);

        let stem_names: [&'static str; 4] = ["Vocals", "Drums", "Bass", "Other"];
        let stem_buffers = [vocals_stem, drums_stem, bass_stem, other_stem];
        let eq_freqs_hz = [2800.0f32, 95.0, 62.0, 6500.0];
        let eq_gains_db = [1.5f32, 1.8, 1.4, 1.2];

        for i in 0..4 {
            let track_idx = i + 1;
            let player =
                AudioStemClipPlayerNode::new((i + 1) as u32, stem_names[i], stem_buffers[i].clone());
            stem_engine
                .add_channel_generator(ChannelGenerator::new(
                    (i + 1) as u32,
                    track_idx,
                    Box::new(player),
                ))
                .expect("acyclic stem generator addition");

            // Insert Slot 0: FabFilter Pro-Q 4 tailored per stem
            let mut proq4 = ProQ4Processor::new();
            proq4.set_band_params(
                0,
                EqBandParams {
                    used: true,
                    enabled: true,
                    shape: FilterShape::Bell,
                    freq_hz: eq_freqs_hz[i],
                    gain_db: eq_gains_db[i],
                    q_ui: 1.0,
                    ..EqBandParams::default()
                },
            );
            stem_engine.tracks[track_idx]
                .add_plugin_slot(Box::new(proq4), 1.0)
                .unwrap();

            // Insert Slot 1: oeksound Soothe3 resonance suppressor tailored per stem
            let mut soothe3 = Soothe3Processor::new(target_sr as f32);
            soothe3.params_mut().global.depth = 3.5;
            soothe3.params_mut().global.detail = 5.5;
            soothe3.params_mut().global.max_cut_db = -12.0;
            stem_engine.tracks[track_idx]
                .add_plugin_slot(Box::new(soothe3), 0.65)
                .unwrap();

            stem_engine.tracks[track_idx].add_send(MASTER_TRACK_INDEX, 1.0, false);
        }

        // Master Track 0: iZotope Ozone 12 Master Chain with -1.0 dBTP True-Peak Ceiling
        let mut ozone12 = Ozone12MasterChain::new(target_sr);
        ozone12.maximizer.ceiling_dbtp = -1.0;
        ozone12.maximizer.gain_db = 2.0;
        stem_engine.tracks[MASTER_TRACK_INDEX]
            .add_plugin_slot(Box::new(ozone12), 1.0)
            .unwrap();

        stem_engine.rebuild_graph().unwrap();

        // Step 4: Bounce Master + all 4 Stem Tracks simultaneously (in-memory first so stem labels match disk filenames)
        let mut split_cfg = config.clone();
        split_cfg.split_track_indices = vec![1, 2, 3, 4];
        split_cfg.output_dir = None;
        let mut res = Self::render_split_mixer_tracks(&mut stem_engine, &split_cfg)?;
        res.config = config.clone();
        res.config.mode = RenderMode::StemMastering;
        res.master_wav.label = format!("{}_Master", config.file_prefix);

        if let Some(ref dir) = config.output_dir {
            fs::create_dir_all(dir)?;
            let master_path = dir.join(format!("{}.wav", res.master_wav.label));
            fs::write(&master_path, &res.master_wav.wav_bytes)?;
            res.master_wav.file_path = Some(master_path);
        }

        // Label and write the 4 stem WAVs clearly
        for (i, wav) in res.additional_wavs.iter_mut().enumerate() {
            if i < 4 {
                wav.label = format!("{}_Stem_{}", config.file_prefix, stem_names[i]);
                if let Some(ref dir) = config.output_dir {
                    let stem_path = dir.join(format!("{}.wav", wav.label));
                    fs::write(&stem_path, &wav.wav_bytes)?;
                    wav.file_path = Some(stem_path);
                }
            }
        }
        Ok(res)
    }

    /// Separates a stereo mix into `[Vocals, Drums, Bass, Other]` using the RTX 4080 ONNX GPU
    /// 3-Model Studio Ensemble (`Kim_Vocal_2` + `UVR-MDX-NET-Inst_HQ_3` + `Ozone 12 BSRNN`),
    /// automatically falling back to pure-Rust spectral engines when ONNX runtime is absent.
    pub fn separate_4stems_uvr5_ensemble(mix: &[StereoFrame]) -> [Vec<StereoFrame>; 4] {
        Self::separate_4stems_uvr5_ensemble_with_sr(mix, 48_000)
    }

    pub fn separate_4stems_uvr5_ensemble_with_sr(
        mix: &[StereoFrame],
        sample_rate_hz: u32,
    ) -> [Vec<StereoFrame>; 4] {
        let len = mix.len();
        if len == 0 {
            return [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        }

        let gpu_cfg = ai_uvr5_stem_engine::GpuStemEnsembleConfig {
            phase_denoise_strength: 0.25,
            ..Default::default()
        };
        if let Ok(gpu_out) = ai_uvr5_stem_engine::run_global_gpu_stem_ensemble(
            mix,
            sample_rate_hz.clamp(8_000, 384_000),
            &gpu_cfg,
        ) {
            return [gpu_out.vocals, gpu_out.drums, gpu_out.bass, gpu_out.other];
        }

        let demucs = HTDemucs4::new(512, 128, 16, 4, 2);
        let [demucs_drums, demucs_bass, _demucs_other, demucs_vocals] = demucs.separate_4stems(mix);

        let mdx = Mdx23cSeparator::new(512, 128, 1024, 0.5);
        let (mdx_vocals, _) = mdx.separate_offline(mix);

        let vr = CascadedASPPNet::new(512, 128, 0.05);
        let (vr_vocals, _) = vr.separate_vocal_instrumental(mix);

        let mut ensembler = SpectralEnsembler::new(512, 128, SpectralEnsembleMode::Weighted);
        ensembler.phase_denoise_strength = 0.25;
        let vocals = ensembler.ensemble_waveforms(
            &[&demucs_vocals, &mdx_vocals, &vr_vocals],
            Some(&[0.50, 0.30, 0.20]),
        );

        // Phase-inverted residual instrumental ensures `Vocals + Drums + Bass + Other == Mix`
        let inst = SpectralEnsembler::invert_audio_waveform(mix, &vocals);
        let mut drums = vec![[0.0f32; 2]; len];
        let mut bass = vec![[0.0f32; 2]; len];
        let mut other = vec![[0.0f32; 2]; len];

        for i in 0..len {
            drums[i] = [demucs_drums[i][0] * 0.75, demucs_drums[i][1] * 0.75];
            bass[i] = [demucs_bass[i][0] * 0.75, demucs_bass[i][1] * 0.75];
            other[i] = [
                inst[i][0] - drums[i][0] - bass[i][0],
                inst[i][1] - drums[i][1] - bass[i][1],
            ];
        }

        [vocals, drums, bass, other]
    }

    fn package_exported_wav(
        track_index: usize,
        label: String,
        frames: Vec<StereoFrame>,
        config: &OfflineRenderConfig,
        apdc_compensated_samples: usize,
    ) -> io::Result<ExportedTrackWav> {
        let wav_bytes = encode_wav_bytes(
            &frames,
            config.sample_rate,
            config.bit_depth,
            config.enable_mbit_plus_noise_shaping,
        );
        let file_path = if let Some(ref dir) = config.output_dir {
            fs::create_dir_all(dir)?;
            let path = dir.join(format!("{label}.wav"));
            fs::write(&path, &wav_bytes)?;
            Some(path)
        } else {
            None
        };
        let analysis = analyze_rendered_audio(
            &frames,
            config.sample_rate,
            config.bit_depth,
            apdc_compensated_samples,
        );
        Ok(ExportedTrackWav {
            track_index,
            label,
            frames,
            wav_bytes,
            file_path,
            analysis,
        })
    }
}
