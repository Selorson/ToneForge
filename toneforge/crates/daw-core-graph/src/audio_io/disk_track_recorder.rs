//! `disk_track_recorder`: Lock-Free Real-Time Audio Tap (`rtrb` SPSC `~2.7 s` pre-allocated ring)
//! + Background IEEE Float32 `.wav` Writer with **Exact Round-Trip Latency Subtraction** and
//! **Loop Take Comping (`Take 1..N`)**.
//!
//! Real-time thread invariants (`RtRecordTap::push_rt_slice`):
//! - `0` `HeapAlloc`, `0` `HeapFree`, `0` `Mutex`, `0` syscalls / `File::write` on the audio thread.

use crate::StereoFrame;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Default lock-free SPSC capture ring capacity (`131,072` stereo frames = `2.73 s` @ `48 kHz`).
pub const RECORD_RING_DEFAULT_CAPACITY_FRAMES: usize = 131_072;

/// Metadata for a completed `.wav` take written by [`DiskTrackRecorder`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordedTakeInfo {
    pub take_number: u32,
    pub take_lane_id: u16,
    pub track_index: u16,
    pub wav_path: PathBuf,
    pub num_frames: usize,
    pub sample_rate_hz: u32,
}

/// Packing helper for `TransportAndMasterHeader::reserved[0]` (`128 KB` SHM telemetry):
/// - `bit 0`: `1` when actively recording to disk
/// - `bit 1`: `1` when at least one track is record-armed
/// - `bits 8..15`: first armed track index (`0..=255`)
/// - `bits 16..31`: hardware input peak meter (`0..=65535` mapped linearly from `0.0..=1.0`)
#[inline]
pub fn pack_telemetry_record_status(
    is_recording: bool,
    any_armed: bool,
    armed_track_index: u16,
    input_peak_linear: f32,
) -> u32 {
    let mut word = 0u32;
    if is_recording {
        word |= 1 << 0;
    }
    if any_armed {
        word |= 1 << 1;
    }
    word |= ((armed_track_index.min(255) as u32) & 0xFF) << 8;
    let peak_u16 = (input_peak_linear.clamp(0.0, 1.0) * 65_535.0) as u32;
    word |= (peak_u16 & 0xFFFF) << 16;
    word
}

#[inline]
pub fn unpack_telemetry_record_status(word: u32) -> (bool, bool, u8, f32) {
    let is_rec = (word & 1) != 0;
    let any_armed = (word & 2) != 0;
    let trk = ((word >> 8) & 0xFF) as u8;
    let peak = (((word >> 16) & 0xFFFF) as f32) / 65_535.0;
    (is_rec, any_armed, trk, peak)
}

/// Wait-free Real-Time Record Tap owned exclusively by `MixerEngine` on the RT audio thread.
pub struct RtRecordTap {
    tx: rtrb::Producer<StereoFrame>,
    recording_enabled: Arc<AtomicBool>,
    pub track_index: u16,
    pub pushed_frames: u64,
    pub dropped_frames: u64,
}

impl RtRecordTap {
    /// Pushes `frames` from the RT audio thread into the pre-allocated SPSC ring (`0` alloc, `0` lock).
    #[inline]
    pub fn push_rt_slice(&mut self, frames: &[StereoFrame]) {
        if !self.recording_enabled.load(Ordering::Relaxed) {
            return;
        }
        for &f in frames {
            if self.tx.push(f).is_ok() {
                self.pushed_frames += 1;
            } else {
                self.dropped_frames += 1;
            }
        }
    }

    #[inline]
    pub fn is_recording(&self) -> bool {
        self.recording_enabled.load(Ordering::Relaxed)
    }
}

struct ActiveWavWriter {
    take_number: u32,
    path: PathBuf,
    writer: BufWriter<File>,
    frames_written: usize,
    sample_rate_hz: u32,
}

impl ActiveWavWriter {
    fn create(path: PathBuf, take_number: u32, sample_rate_hz: u32) -> Result<Self, String> {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let file =
            File::create(&path).map_err(|e| format!("Failed to create {}: {e}", path.display()))?;
        let mut writer = BufWriter::with_capacity(64 * 1024, file);
        let header = build_ieee_float32_stereo_wav_header(sample_rate_hz, 0);
        writer
            .write_all(&header)
            .map_err(|e| format!("Failed to write WAV header: {e}"))?;
        Ok(Self {
            take_number,
            path,
            writer,
            frames_written: 0,
            sample_rate_hz,
        })
    }

    #[inline]
    fn write_frame(&mut self, frame: StereoFrame) -> std::io::Result<()> {
        let mut bytes = [0u8; 8];
        bytes[0..4].copy_from_slice(&frame[0].to_le_bytes());
        bytes[4..8].copy_from_slice(&frame[1].to_le_bytes());
        self.writer.write_all(&bytes)?;
        self.frames_written += 1;
        Ok(())
    }

    fn finalize(mut self, track_index: u16) -> Result<RecordedTakeInfo, String> {
        self.writer
            .flush()
            .map_err(|e| format!("WAV flush error: {e}"))?;
        let mut file = self
            .writer
            .into_inner()
            .map_err(|e| format!("WAV into_inner error: {e}"))?;
        let header = build_ieee_float32_stereo_wav_header(self.sample_rate_hz, self.frames_written);
        file.seek(SeekFrom::Start(0))
            .map_err(|e| format!("WAV seek error: {e}"))?;
        file.write_all(&header)
            .map_err(|e| format!("WAV header finalize error: {e}"))?;
        file.sync_data()
            .map_err(|e| format!("WAV sync_data error: {e}"))?;

        Ok(RecordedTakeInfo {
            take_number: self.take_number,
            take_lane_id: self.take_number as u16,
            track_index,
            wav_path: self.path,
            num_frames: self.frames_written,
            sample_rate_hz: self.sample_rate_hz,
        })
    }
}

fn build_ieee_float32_stereo_wav_header(sample_rate_hz: u32, num_frames: usize) -> [u8; 44] {
    let channels: u16 = 2;
    let bits_per_sample: u16 = 32;
    let block_align: u16 = channels * (bits_per_sample / 8); // 8 bytes
    let byte_rate: u32 = sample_rate_hz * (block_align as u32);
    let data_bytes: u32 = (num_frames.saturating_mul(block_align as usize)).min(u32::MAX as usize) as u32;
    let riff_size: u32 = 36u32.saturating_add(data_bytes);

    let mut h = [0u8; 44];
    h[0..4].copy_from_slice(b"RIFF");
    h[4..8].copy_from_slice(&riff_size.to_le_bytes());
    h[8..12].copy_from_slice(b"WAVE");
    h[12..16].copy_from_slice(b"fmt ");
    h[16..20].copy_from_slice(&16u32.to_le_bytes());
    h[20..22].copy_from_slice(&3u16.to_le_bytes()); // WAVE_FORMAT_IEEE_FLOAT (0x0003)
    h[22..24].copy_from_slice(&channels.to_le_bytes());
    h[24..28].copy_from_slice(&sample_rate_hz.to_le_bytes());
    h[28..32].copy_from_slice(&byte_rate.to_le_bytes());
    h[32..34].copy_from_slice(&block_align.to_le_bytes());
    h[34..36].copy_from_slice(&bits_per_sample.to_le_bytes());
    h[36..40].copy_from_slice(b"data");
    h[40..44].copy_from_slice(&data_bytes.to_le_bytes());
    h
}

/// Reads back a RIFF/WAVE `.wav` file (chunk-walked so `JUNK`, `bext`, `LIST`, `fact`, and
/// `WAVEFORMATEXTENSIBLE` chunks never corrupt sample framing) into stereo `f32` frames.
pub fn read_wav_f32_stereo(path: &Path) -> Result<(u32, Vec<StereoFrame>), String> {
    let mut file =
        File::open(path).map_err(|e| format!("Failed to open WAV {}: {e}", path.display()))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|e| format!("Failed to read WAV {}: {e}", path.display()))?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(format!("Invalid RIFF/WAVE header in {}", path.display()));
    }

    let mut fmt_tag = 0u16;
    let mut channels = 0usize;
    let mut sample_rate = 0u32;
    let mut bits_per_sample = 0u16;
    let mut is_ieee_float = false;
    let mut data_range: Option<(usize, usize)> = None;

    let mut cursor = 12usize;
    while cursor + 8 <= bytes.len() {
        let chunk_id = &bytes[cursor..cursor + 4];
        let chunk_size = u32::from_le_bytes([
            bytes[cursor + 4],
            bytes[cursor + 5],
            bytes[cursor + 6],
            bytes[cursor + 7],
        ]) as usize;
        let body_start = cursor + 8;
        let body_end = body_start.saturating_add(chunk_size).min(bytes.len());

        if chunk_id == b"fmt " && body_end >= body_start + 16 {
            let f = &bytes[body_start..body_end];
            fmt_tag = u16::from_le_bytes([f[0], f[1]]);
            channels = u16::from_le_bytes([f[2], f[3]]) as usize;
            sample_rate = u32::from_le_bytes([f[4], f[5], f[6], f[7]]);
            bits_per_sample = u16::from_le_bytes([f[14], f[15]]);
            if fmt_tag == 3 {
                is_ieee_float = true;
            } else if fmt_tag == 0xFFFE && f.len() >= 40 {
                let sub_tag = u16::from_le_bytes([f[24], f[25]]);
                is_ieee_float = sub_tag == 3;
                if sub_tag == 1 {
                    fmt_tag = 1;
                }
            }
        } else if chunk_id == b"data" {
            data_range = Some((body_start, body_end));
            break;
        }

        let padded = chunk_size.saturating_add(chunk_size & 1);
        let next = body_start.saturating_add(padded);
        if next <= cursor {
            break;
        }
        cursor = next;
    }

    let Some((d_start, d_end)) = data_range else {
        return Err(format!("Missing `data` chunk in WAV {}", path.display()));
    };
    if channels == 0 || sample_rate == 0 {
        return Err(format!("Missing or invalid `fmt ` chunk in WAV {}", path.display()));
    }

    let payload = &bytes[d_start..d_end];
    let downmix_frame = |ch_vals: &[f32]| -> StereoFrame {
        let san = |x: f32| if x.is_finite() { x } else { 0.0 };
        match ch_vals.len() {
            0 => [0.0, 0.0],
            1 => {
                let m = san(ch_vals[0]);
                [m, m]
            }
            2 => [san(ch_vals[0]), san(ch_vals[1])],
            3 => {
                let c = 0.707_106_77 * san(ch_vals[2]);
                [san(ch_vals[0]) + c, san(ch_vals[1]) + c]
            }
            4 => [
                san(ch_vals[0]) + 0.707_106_77 * san(ch_vals[2]),
                san(ch_vals[1]) + 0.707_106_77 * san(ch_vals[3]),
            ],
            _ => {
                let c = 0.707_106_77 * san(ch_vals[2]);
                let sl = 0.707_106_77 * san(ch_vals[4])
                    + if ch_vals.len() >= 7 {
                        0.707_106_77 * san(ch_vals[6])
                    } else {
                        0.0
                    };
                let sr = if ch_vals.len() >= 6 {
                    0.707_106_77 * san(ch_vals[5])
                } else {
                    0.0
                } + if ch_vals.len() >= 8 {
                    0.707_106_77 * san(ch_vals[7])
                } else {
                    0.0
                };
                [san(ch_vals[0]) + c + sl, san(ch_vals[1]) + c + sr]
            }
        }
    };

    let bytes_per_sample = (bits_per_sample as usize) / 8;
    if bytes_per_sample == 0 {
        return Err(format!("Unsupported WAV bits_per_sample={bits_per_sample}"));
    }
    let stride = channels * bytes_per_sample;
    let n_frames = payload.len() / stride;
    let mut out = Vec::with_capacity(n_frames);
    let mut ch_scratch = vec![0.0f32; channels.min(16)];

    if is_ieee_float && bits_per_sample == 32 {
        for i in 0..n_frames {
            let base = i * stride;
            let use_ch = ch_scratch.len();
            for c in 0..use_ch {
                let off = base + c * 4;
                ch_scratch[c] = f32::from_le_bytes([
                    payload[off],
                    payload[off + 1],
                    payload[off + 2],
                    payload[off + 3],
                ]);
            }
            out.push(downmix_frame(&ch_scratch));
        }
    } else if (fmt_tag == 1 || fmt_tag == 0xFFFE) && matches!(bits_per_sample, 8 | 16 | 24 | 32) {
        for i in 0..n_frames {
            let base = i * stride;
            let use_ch = ch_scratch.len();
            for c in 0..use_ch {
                let off = base + c * bytes_per_sample;
                ch_scratch[c] = match bits_per_sample {
                    8 => ((payload[off] as f32) - 128.0) / 128.0,
                    16 => (i16::from_le_bytes([payload[off], payload[off + 1]]) as f32) / 32_768.0,
                    24 => {
                        let v = (payload[off] as i32)
                            | ((payload[off + 1] as i32) << 8)
                            | (((payload[off + 2] as i8) as i32) << 16);
                        ((v as f64) / 8_388_608.0) as f32
                    }
                    32 => {
                        let v = i32::from_le_bytes([
                            payload[off],
                            payload[off + 1],
                            payload[off + 2],
                            payload[off + 3],
                        ]);
                        ((v as f64) / 2_147_483_648.0) as f32
                    }
                    _ => 0.0,
                };
            }
            out.push(downmix_frame(&ch_scratch));
        }
    } else {
        return Err(format!(
            "Unsupported WAV format (tag={fmt_tag}, bits={bits_per_sample}, ch={channels})"
        ));
    }

    Ok((sample_rate, out))
}

/// Non-Real-Time Background Disk Recorder & Loop Take Comping Engine.
pub struct DiskTrackRecorder {
    recording_enabled: Arc<AtomicBool>,
    shutdown: Arc<AtomicBool>,
    written_frames: Arc<AtomicU64>,
    completed_takes: Arc<Mutex<Vec<RecordedTakeInfo>>>,
    worker: Option<JoinHandle<()>>,
}

impl DiskTrackRecorder {
    /// Creates a paired `(DiskTrackRecorder, RtRecordTap)` with pre-allocated SPSC ring capacity.
    ///
    /// - `round_trip_latency_samples`: exact hardware + plugin latency discarded from the head of
    ///   the captured stream so musical `sample 0` aligns with `0`-sample offset error on Playlist.
    /// - `loop_length_frames`: when `Some(L)`, automatically splits continuous loop recording into
    ///   `track_{idx}_take_01.wav`, `track_{idx}_take_02.wav`, `track_{idx}_take_03.wav`, etc.
    pub fn spawn(
        output_dir: PathBuf,
        track_index: u16,
        sample_rate_hz: u32,
        round_trip_latency_samples: usize,
        loop_length_frames: Option<usize>,
        ring_capacity_frames: usize,
    ) -> (Self, RtRecordTap) {
        let cap = ring_capacity_frames.max(4096);
        let (tx, mut rx) = rtrb::RingBuffer::<StereoFrame>::new(cap);
        let recording_enabled = Arc::new(AtomicBool::new(true));
        let shutdown = Arc::new(AtomicBool::new(false));
        let written_frames = Arc::new(AtomicU64::new(0));
        let completed_takes: Arc<Mutex<Vec<RecordedTakeInfo>>> = Arc::new(Mutex::new(Vec::new()));

        let shutdown_bg = Arc::clone(&shutdown);
        let written_bg = Arc::clone(&written_frames);
        let takes_bg = Arc::clone(&completed_takes);
        let sr = sample_rate_hz.max(8_000);
        let loop_len = loop_length_frames.filter(|&n| n > 0);

        let worker = thread::Builder::new()
            .name(format!("toneforge-disk-rec-t{track_index}"))
            .spawn(move || {
                let mut latency_to_skip = round_trip_latency_samples;
                let mut next_take_num = 1u32;
                let mut active_wav: Option<ActiveWavWriter> = None;

                let mut drain_one = |frame: StereoFrame| {
                    if latency_to_skip > 0 {
                        latency_to_skip -= 1;
                        return;
                    }
                    if active_wav.is_none() {
                        let fname =
                            format!("track_{track_index:03}_take_{next_take_num:02}.wav");
                        let path = output_dir.join(fname);
                        if let Ok(w) = ActiveWavWriter::create(path, next_take_num, sr) {
                            active_wav = Some(w);
                            next_take_num += 1;
                        }
                    }
                    if let Some(ref mut w) = active_wav {
                        let _ = w.write_frame(frame);
                        written_bg.fetch_add(1, Ordering::Relaxed);
                        if let Some(limit) = loop_len {
                            if w.frames_written >= limit {
                                if let Some(done_writer) = active_wav.take() {
                                    if let Ok(info) = done_writer.finalize(track_index) {
                                        takes_bg.lock().push(info);
                                    }
                                }
                            }
                        }
                    }
                };

                while !shutdown_bg.load(Ordering::Acquire) {
                    let mut popped = 0usize;
                    while let Ok(frame) = rx.pop() {
                        drain_one(frame);
                        popped += 1;
                    }
                    if popped == 0 {
                        thread::sleep(Duration::from_millis(2));
                    }
                }

                // Drain all remaining frames flushed before stop
                while let Ok(frame) = rx.pop() {
                    drain_one(frame);
                }

                if let Some(done_writer) = active_wav.take() {
                    if done_writer.frames_written > 0 {
                        if let Ok(info) = done_writer.finalize(track_index) {
                            takes_bg.lock().push(info);
                        }
                    }
                }
            })
            .expect("failed to spawn toneforge-disk-rec thread");

        let tap = RtRecordTap {
            tx,
            recording_enabled: Arc::clone(&recording_enabled),
            track_index,
            pushed_frames: 0,
            dropped_frames: 0,
        };

        (
            Self {
                recording_enabled,
                shutdown,
                written_frames,
                completed_takes,
                worker: Some(worker),
            },
            tap,
        )
    }

    #[inline]
    pub fn set_recording_enabled(&self, enabled: bool) {
        self.recording_enabled.store(enabled, Ordering::Release);
    }

    #[inline]
    pub fn total_written_frames(&self) -> u64 {
        self.written_frames.load(Ordering::Relaxed)
    }

    pub fn completed_takes_snapshot(&self) -> Vec<RecordedTakeInfo> {
        self.completed_takes.lock().clone()
    }

    /// Stops recording, flushes all pending frames in the `rtrb` ring to disk, finalizes RIFF headers,
    /// and returns all recorded `.wav` takes (`Take 1..N`).
    pub fn finish_and_collect_takes(mut self) -> Vec<RecordedTakeInfo> {
        self.recording_enabled.store(false, Ordering::Release);
        self.shutdown.store(true, Ordering::Release);
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
        self.completed_takes.lock().clone()
    }
}

impl Drop for DiskTrackRecorder {
    fn drop(&mut self) {
        self.recording_enabled.store(false, Ordering::Release);
        self.shutdown.store(true, Ordering::Release);
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

#[cfg(test)]
mod wav_chunk_tests {
    use super::*;

    #[test]
    fn test_read_wav_f32_stereo_skips_junk_and_list_chunks_and_downmixes_surround() {
        let path = std::env::temp_dir().join("toneforge_test_riff_junk_5_1.wav");
        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&0u32.to_le_bytes()); // patched below
        bytes.extend_from_slice(b"WAVE");

        // Odd-sized JUNK chunk (5 bytes + 1 pad byte) BEFORE `fmt `
        bytes.extend_from_slice(b"JUNK");
        bytes.extend_from_slice(&5u32.to_le_bytes());
        bytes.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0x00]);

        // 6-channel IEEE float32 `fmt ` chunk @ 48 kHz
        let channels = 6u16;
        let sr = 48_000u32;
        let bits = 32u16;
        let block_align = channels * (bits / 8);
        let byte_rate = sr * (block_align as u32);
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&sr.to_le_bytes());
        bytes.extend_from_slice(&byte_rate.to_le_bytes());
        bytes.extend_from_slice(&block_align.to_le_bytes());
        bytes.extend_from_slice(&bits.to_le_bytes());

        // `bext` metadata chunk BEFORE `data`
        bytes.extend_from_slice(b"bext");
        bytes.extend_from_slice(&8u32.to_le_bytes());
        bytes.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);

        // `data` chunk: 2 frames of 5.1 [FL, FR, FC, LFE, SL, SR]
        let frame0: [f32; 6] = [0.25, -0.25, 0.50, 0.90, 0.0, 0.0];
        let frame1: [f32; 6] = [0.0, 0.0, 0.0, 0.0, 0.40, -0.40];
        let data_bytes_len = (2 * 6 * 4) as u32;
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_bytes_len.to_le_bytes());
        for s in frame0.iter().chain(frame1.iter()) {
            bytes.extend_from_slice(&s.to_le_bytes());
        }

        let riff_size = (bytes.len() - 8) as u32;
        bytes[4..8].copy_from_slice(&riff_size.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();

        let (decoded_sr, frames) = read_wav_f32_stereo(&path).expect("parse chunked 5.1 WAV");
        let _ = std::fs::remove_file(&path);

        assert_eq!(decoded_sr, 48_000);
        assert_eq!(frames.len(), 2);
        let c_mix = 0.50 * std::f32::consts::FRAC_1_SQRT_2;
        assert!((frames[0][0] - (0.25 + c_mix)).abs() < 1e-5);
        assert!((frames[0][1] - (-0.25 + c_mix)).abs() < 1e-5);
        let s_mix = 0.40 * std::f32::consts::FRAC_1_SQRT_2;
        assert!((frames[1][0] - s_mix).abs() < 1e-5);
        assert!((frames[1][1] + s_mix).abs() < 1e-5);
    }
}
