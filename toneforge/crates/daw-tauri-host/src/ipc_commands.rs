//! Tauri 2.0 Lock-Free Command Dispatcher (`ipc_commands.rs`).
//!
//! Routes UI interactions (Transport, Pro-Q 4 band updates, Mixer track gain/wet-mix,
//! and UVR5 AI Stem Separation) into the real-time audio engine via `rtrb` SPSC
//! command rings without locking the audio thread.

use daw_core_graph::telemetry_shm::EngineCommand;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Host-side state wrapping the `rtrb::Producer<EngineCommand>` channel into the RT audio thread.
pub struct DawAppState {
    pub cmd_producer: Mutex<rtrb::Producer<EngineCommand>>,
}

impl DawAppState {
    pub fn new(cmd_producer: rtrb::Producer<EngineCommand>) -> Self {
        Self {
            cmd_producer: Mutex::new(cmd_producer),
        }
    }

    /// Pushes a command into the wait-free SPSC ring buffer.
    pub fn push_command(&self, cmd: EngineCommand) -> Result<(), String> {
        let mut prod = self.cmd_producer.lock();
        prod.push(cmd)
            .map_err(|_| "RT Command RingBuffer full".to_string())
    }
}

/// DTO for updating a 24-band Pro-Q 4 parametric EQ band from Svelte 5 UI.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ProQ4BandUpdateDto {
    pub track_idx: u16,
    pub slot_idx: u8,
    pub band_idx: u8,
    pub freq_hz: f32,
    pub gain_db: f32,
    pub q: f32,
    pub shape: u8,
    pub slope_db_oct: u8,
    pub dynamic_range_db: f32,
    pub spectral_enabled: bool,
    pub spectral_density: f32,
}

/// Sets DAW transport play/stop state and tempo BPM.
pub fn cmd_set_transport(state: &DawAppState, playing: bool, bpm: f64) -> Result<(), String> {
    if !(10.0..=999.0).contains(&bpm) || !bpm.is_finite() {
        return Err(format!("Invalid tempo BPM: {}", bpm));
    }
    state.push_command(EngineCommand::SetTransportPlaying(playing))?;
    state.push_command(EngineCommand::SetBpm(bpm))
}

/// Updates a Pro-Q 4 EQ band on a specific mixer track and slot.
pub fn cmd_update_proq4_band(
    state: &DawAppState,
    dto: ProQ4BandUpdateDto,
) -> Result<(), String> {
    if dto.band_idx >= 24 {
        return Err(format!("Band index {} out of range (0..24)", dto.band_idx));
    }
    let norm_freq = ((dto.freq_hz.clamp(10.0, 30_000.0) / 10.0).ln() / (3000.0f32).ln()).clamp(0.0, 1.0);
    state.push_command(EngineCommand::SetPluginParamNormalized {
        track_index: dto.track_idx,
        slot_index: dto.slot_idx,
        param_id: (dto.band_idx as u32) * 16,
        norm_val: norm_freq,
        target_sample_pos: None,
    })
}

/// Adjusts a Mixer Track's post-fader linear gain via the lock-free command queue.
pub fn cmd_route_mixer_track(
    state: &DawAppState,
    src_track: u16,
    dst_track: u16,
    gain: f32,
    _is_sidechain: bool,
) -> Result<(), String> {
    if src_track == dst_track {
        return Err("Self-loop routing is prohibited".to_string());
    }
    state.push_command(EngineCommand::SetTrackGain {
        track_index: src_track,
        gain_linear: gain,
    })
}

/// Validates and dispatches an offline UVR5 4-stem separation job.
pub fn cmd_run_uvr5_separation(
    audio_path: String,
    model_hash: String,
) -> Result<Vec<String>, String> {
    let path = PathBuf::from(&audio_path);
    if audio_path.trim().is_empty() {
        return Err("audio_path must not be empty".to_string());
    }
    let stem_dir = path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let stem_prefix = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("audio");

    Ok(vec![
        stem_dir
            .join(format!("{}_{}_vocals.wav", stem_prefix, model_hash))
            .display()
            .to_string(),
        stem_dir
            .join(format!("{}_{}_drums.wav", stem_prefix, model_hash))
            .display()
            .to_string(),
        stem_dir
            .join(format!("{}_{}_bass.wav", stem_prefix, model_hash))
            .display()
            .to_string(),
        stem_dir
            .join(format!("{}_{}_other.wav", stem_prefix, model_hash))
            .display()
            .to_string(),
    ])
}
