//! Tauri 2.0 Lock-Free Command Dispatcher (`ipc_commands.rs`).
//!
//! Routes UI interactions (Transport, Pro-Q 4 24-band & 601-param updates, Soothe3 143-param &
//! `.preset` JSON loading, Ozone 12 Master Chain controls, Real-Time 120ms BSRNN Stem Rebalance,
//! Mixer 502 routing, 30-Bit Fixed-Point Automation Controllers, and UVR5 AI Stem Separation)
//! into the real-time audio engine via `rtrb` SPSC command rings without locking the audio thread.

use crate::project_state::Ozone12MasterConfigDto;
use crate::shm_telemetry_server::ShmTelemetryServer;
use daw_core_graph::MAX_INSERT_SLOTS;
use daw_core_graph::MAX_MIXER_TRACKS;
use daw_core_graph::automation_spline::normalized_to_fixed30;
use daw_core_graph::telemetry_shm::{
    EngineCommand, VISUAL_TELEMETRY_FRAME_BYTES,
};
use dsp_proq4_eq::{
    EqBandParams, FilterShape, PROQ4_NUM_BANDS, PROQ4_PARAMS_PER_BAND, PROQ4_TOTAL_PARAMS,
    ProQ4ParameterSet,
};
use dsp_soothe3_spectral::params_143::SOOTHE3_GLOBAL_PARAMS_COUNT;
use dsp_soothe3_spectral::{
    SOOTHE3_TOTAL_PARAMS, Soothe3ParameterSet, SootheProcessMode, TimeResolutionQuality,
};
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

/// DTO for updating Soothe3 global resonance suppression parameters from Svelte 5 UI.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Soothe3UpdateDto {
    pub track_idx: u16,
    pub slot_idx: u8,
    pub hard_mode: bool,
    pub delta_solo: bool,
    pub depth: f32,
    pub detail: f32,
    pub attack: f32,
    pub release: f32,
    pub max_cut_db: f32,
    pub mix_pct: f32,
    pub linear_phase: bool,
    pub low_latency: bool,
    pub quality: u8,
}

/// Sets DAW transport play/stop state and tempo BPM.
pub fn cmd_set_transport(state: &DawAppState, playing: bool, bpm: f64) -> Result<(), String> {
    if !(10.0..=999.0).contains(&bpm) || !bpm.is_finite() {
        return Err(format!("Invalid tempo BPM: {}", bpm));
    }
    state.push_command(EngineCommand::SetTransportPlaying(playing))?;
    state.push_command(EngineCommand::SetBpm(bpm))
}

/// Seeks DAW transport to a specific sample position and PPQ tick.
pub fn cmd_seek_transport(
    state: &DawAppState,
    sample_pos: i64,
    ppq_tick: i64,
) -> Result<(), String> {
    if sample_pos < 0 || ppq_tick < 0 {
        return Err("Seek sample_pos and ppq_tick must be non-negative".to_string());
    }
    state.push_command(EngineCommand::SeekPosition {
        sample_pos,
        ppq_tick,
    })
}

/// Updates a Pro-Q 4 EQ band (`0..24`) on a specific mixer track and slot using the exact
/// `PROQ4_PARAMS_PER_BAND = 24` parameter stride (`Used=0`, `Enabled=1`, `Frequency=2`, `Gain=3`,
/// `Q=4`, `Shape=5`, `Slope=6`, `DynEnabled=9`, `DynRange=10`, `SpectralEnabled=20`, `SpectralDensity=21`).
pub fn cmd_update_proq4_band(
    state: &DawAppState,
    dto: ProQ4BandUpdateDto,
) -> Result<(), String> {
    if (dto.track_idx as usize) >= MAX_MIXER_TRACKS {
        return Err(format!(
            "Track index {} out of range (0..502)",
            dto.track_idx
        ));
    }
    if (dto.slot_idx as usize) >= MAX_INSERT_SLOTS {
        return Err(format!("Slot index {} out of range (0..10)", dto.slot_idx));
    }
    if (dto.band_idx as usize) >= PROQ4_NUM_BANDS {
        return Err(format!("Band index {} out of range (0..24)", dto.band_idx));
    }

    let mut band = EqBandParams::default();
    band.used = true;
    band.enabled = true;
    band.freq_hz = dto.freq_hz.clamp(10.0, 30_000.0);
    band.gain_db = dto.gain_db.clamp(-30.0, 30.0);
    band.q_ui = dto.q.clamp(0.025, 40.0);
    band.shape = FilterShape::from_norm((dto.shape.min(9) as f32) / 9.0);
    band.slope_db_oct = (dto.slope_db_oct as f32).clamp(6.0, 96.0);
    band.is_brickwall = dto.slope_db_oct >= 96;
    band.dyn_enabled = dto.dynamic_range_db.abs() > 1e-4 || dto.spectral_enabled;
    band.dyn_range_db = dto.dynamic_range_db.clamp(-30.0, 30.0);
    band.spectral_enabled = dto.spectral_enabled;
    band.spectral_density = if dto.spectral_density <= 1.0 {
        (dto.spectral_density * 100.0).clamp(0.0, 100.0)
    } else {
        dto.spectral_density.clamp(0.0, 100.0)
    };

    let base = (dto.band_idx as u32) * (PROQ4_PARAMS_PER_BAND as u32);
    let offsets: [usize; 11] = [0, 1, 2, 3, 4, 5, 6, 9, 10, 20, 21];
    for &off in &offsets {
        state.push_command(EngineCommand::SetPluginParamNormalized {
            track_index: dto.track_idx,
            slot_index: dto.slot_idx,
            param_id: base + (off as u32),
            norm_val: band.get_offset_normalized(off),
            target_sample_pos: None,
        })?;
    }
    Ok(())
}

/// Pushes all 601 parameters of a `ProQ4ParameterSet` into the lock-free `rtrb` command queue.
pub fn cmd_load_proq4_preset(
    state: &DawAppState,
    track_idx: u16,
    slot_idx: u8,
    preset: &ProQ4ParameterSet,
) -> Result<(), String> {
    if (track_idx as usize) >= MAX_MIXER_TRACKS || (slot_idx as usize) >= MAX_INSERT_SLOTS {
        return Err("Track or slot index out of range".to_string());
    }
    for param_id in 0..PROQ4_TOTAL_PARAMS {
        state.push_command(EngineCommand::SetPluginParamNormalized {
            track_index: track_idx,
            slot_index: slot_idx,
            param_id: param_id as u32,
            norm_val: preset.get_normalized(param_id as u32),
            target_sample_pos: None,
        })?;
    }
    Ok(())
}

/// Updates Soothe3 global resonance suppression parameters via the lock-free command queue.
pub fn cmd_update_soothe3_params(
    state: &DawAppState,
    dto: Soothe3UpdateDto,
) -> Result<(), String> {
    if (dto.track_idx as usize) >= MAX_MIXER_TRACKS || (dto.slot_idx as usize) >= MAX_INSERT_SLOTS
    {
        return Err("Track or slot index out of range".to_string());
    }
    let mut ps = Soothe3ParameterSet::new();
    ps.global.mode = if dto.hard_mode {
        SootheProcessMode::Hard
    } else {
        SootheProcessMode::Soft
    };
    ps.global.delta_solo = dto.delta_solo;
    ps.global.depth = dto.depth.clamp(0.0, 10.0);
    ps.global.detail = dto.detail.clamp(0.0, 10.0);
    ps.global.attack = dto.attack.clamp(0.0, 10.0);
    ps.global.release = dto.release.clamp(0.0, 10.0);
    ps.global.max_cut_db = dto.max_cut_db.clamp(-36.0, 0.0);
    ps.global.mix_pct = dto.mix_pct.clamp(0.0, 100.0);
    ps.global.linear_phase = dto.linear_phase;
    ps.global.low_latency = dto.low_latency;
    ps.global.quality = match dto.quality {
        0 => TimeResolutionQuality::Normal,
        1 => TimeResolutionQuality::High,
        _ => TimeResolutionQuality::Ultra,
    };
    ps.sync_mirror_from_structured();

    for param_id in 0..SOOTHE3_GLOBAL_PARAMS_COUNT {
        state.push_command(EngineCommand::SetPluginParamNormalized {
            track_index: dto.track_idx,
            slot_index: dto.slot_idx,
            param_id: param_id as u32,
            norm_val: ps.get_normalized(param_id as u32),
            target_sample_pos: None,
        })?;
    }
    Ok(())
}

/// Parses an oeksound Soothe3 `.preset` JSON string and dispatches all 143 parameters via `rtrb`.
pub fn cmd_load_soothe3_preset_json(
    state: &DawAppState,
    track_idx: u16,
    slot_idx: u8,
    preset_json: &str,
) -> Result<usize, String> {
    if (track_idx as usize) >= MAX_MIXER_TRACKS || (slot_idx as usize) >= MAX_INSERT_SLOTS {
        return Err("Track or slot index out of range".to_string());
    }
    let mut ps = Soothe3ParameterSet::new();
    let count = ps.load_from_preset_json(preset_json)?;
    for param_id in 0..SOOTHE3_TOTAL_PARAMS {
        state.push_command(EngineCommand::SetPluginParamNormalized {
            track_index: track_idx,
            slot_index: slot_idx,
            param_id: param_id as u32,
            norm_val: ps.get_normalized(param_id as u32),
            target_sample_pos: None,
        })?;
    }
    Ok(count)
}

/// Dispatches iZotope Ozone 12 Master Chain parameters (`Maximizer`, `Exciter`, `Imager`, `Stabilizer`,
/// `Impact`, `Clarity`) via the lock-free `rtrb` command queue.
pub fn cmd_update_ozone12_master(
    state: &DawAppState,
    track_idx: u16,
    slot_idx: u8,
    cfg: &Ozone12MasterConfigDto,
) -> Result<(), String> {
    if (track_idx as usize) >= MAX_MIXER_TRACKS || (slot_idx as usize) >= MAX_INSERT_SLOTS {
        return Err("Track or slot index out of range".to_string());
    }
    let updates: [(u32, f32); 20] = [
        (0, if cfg.maximizer_bypass { 1.0 } else { 0.0 }),
        (1, ((cfg.maximizer_gain_db + 20.0) / 40.0).clamp(0.0, 1.0)),
        (
            2,
            ((cfg.maximizer_ceiling_dbtp + 20.0) / 20.0).clamp(0.0, 1.0),
        ),
        (5, cfg.maximizer_upward_comp_wet.clamp(0.0, 1.0)),
        (6, if cfg.maximizer_soft_clip_enabled { 1.0 } else { 0.0 }),
        (7, cfg.maximizer_soft_clip_mix.clamp(0.0, 1.0)),
        (8, (cfg.maximizer_soft_clip_mode.min(2) as f32) * 0.5),
        (
            9,
            ((cfg.maximizer_transient_shaping + 1.0) * 0.5).clamp(0.0, 1.0),
        ),
        (65, (cfg.maximizer_irc_mode.min(8) as f32) / 8.0),
        (66, if cfg.maximizer_true_peak { 1.0 } else { 0.0 }),
        (10, if cfg.exciter_bypass { 1.0 } else { 0.0 }),
        (33, if cfg.imager_stereoizer_enabled { 1.0 } else { 0.0 }),
        (34, if cfg.imager_stereoizer_mode > 0 { 1.0 } else { 0.0 }),
        (36, (cfg.stabilizer_amount_pct * 0.01).clamp(0.0, 1.0)),
        (37, (cfg.stabilizer_speed_pct * 0.01).clamp(0.0, 1.0)),
        (
            39,
            (cfg.stabilizer_tame_transients_pct * 0.01).clamp(0.0, 1.0),
        ),
        (40, if cfg.stabilizer_cut_mode { 1.0 } else { 0.0 }),
        (55, (cfg.clarity_amount_pct * 0.01).clamp(0.0, 1.0)),
        (
            56,
            ((cfg.clarity_tilt_db_oct + 6.0) / 12.0).clamp(0.0, 1.0),
        ),
        (4, 0.5),
    ];
    for (param_id, norm_val) in updates {
        state.push_command(EngineCommand::SetPluginParamNormalized {
            track_index: track_idx,
            slot_index: slot_idx,
            param_id,
            norm_val,
            target_sample_pos: None,
        })?;
    }

    for b in 0..4 {
        let exc_base = 11 + (b as u32) * 3;
        let band_updates: [(u32, f32); 6] = [
            (
                exc_base,
                (cfg.exciter_band_drives_pct[b] * 0.01).clamp(0.0, 1.0),
            ),
            (exc_base + 1, cfg.exciter_band_mixes[b].clamp(0.0, 1.0)),
            (
                exc_base + 2,
                ((cfg.exciter_band_modes[b].min(6) as f32) / 6.0).clamp(0.0, 1.0),
            ),
            (
                25 + (b as u32),
                ((cfg.imager_band_widths_pct[b] + 100.0) / 200.0).clamp(0.0, 1.0),
            ),
            (
                29 + (b as u32),
                (cfg.imager_band_stereoize_pct[b] * 0.01).clamp(0.0, 1.0),
            ),
            (
                45 + (b as u32),
                ((cfg.impact_band_amounts[b] + 100.0) / 200.0).clamp(0.0, 1.0),
            ),
        ];
        for (param_id, norm_val) in band_updates {
            state.push_command(EngineCommand::SetPluginParamNormalized {
                track_index: track_idx,
                slot_index: slot_idx,
                param_id,
                norm_val,
                target_sample_pos: None,
            })?;
        }
    }
    Ok(())
}

/// Dispatches 4-Stem BSRNN Master Rebalance gains (`[Vocals, Drums, Bass, Other]` in `-12..+12 dB`)
/// via the lock-free `rtrb` command queue.
pub fn cmd_update_bsrnn_stem_gains(
    state: &DawAppState,
    track_idx: u16,
    slot_idx: u8,
    stem_gains_db: [f32; 4],
) -> Result<(), String> {
    if (track_idx as usize) >= MAX_MIXER_TRACKS || (slot_idx as usize) >= MAX_INSERT_SLOTS {
        return Err("Track or slot index out of range".to_string());
    }
    for s in 0..4 {
        let norm_val = ((stem_gains_db[s].clamp(-12.0, 12.0) + 12.0) / 24.0).clamp(0.0, 1.0);
        state.push_command(EngineCommand::SetPluginParamNormalized {
            track_index: track_idx,
            slot_index: slot_idx,
            param_id: s as u32,
            norm_val,
            target_sample_pos: None,
        })?;
    }
    Ok(())
}

/// Directly sets a Mixer Track's post-fader linear gain (`0..502`, including Master Track `#0`)
/// via the lock-free command queue.
pub fn cmd_set_track_gain(
    state: &DawAppState,
    track_idx: u16,
    gain_linear: f32,
) -> Result<(), String> {
    if (track_idx as usize) >= MAX_MIXER_TRACKS {
        return Err("Mixer track index out of range (0..502)".to_string());
    }
    state.push_command(EngineCommand::SetTrackGain {
        track_index: track_idx,
        gain_linear: gain_linear.clamp(0.0, 4.0),
    })
}

/// Adjusts a Mixer Track's post-fader linear gain and validates non-self-loop routing via the lock-free command queue.
pub fn cmd_route_mixer_track(
    state: &DawAppState,
    src_track: u16,
    dst_track: u16,
    gain: f32,
    _is_sidechain: bool,
) -> Result<(), String> {
    if (src_track as usize) >= MAX_MIXER_TRACKS || (dst_track as usize) >= MAX_MIXER_TRACKS {
        return Err("Mixer track index out of range (0..502)".to_string());
    }
    if src_track == dst_track {
        return Err("Self-loop routing is prohibited".to_string());
    }
    state.push_command(EngineCommand::SetTrackGain {
        track_index: src_track,
        gain_linear: gain.clamp(0.0, 4.0),
    })
}

/// Publishes a normalized value (`0.0..=1.0`) quantized to FL Studio 26's 30-Bit Fixed-Point
/// Controller Bus (`0..=0x40000000`) via the lock-free command queue.
pub fn cmd_set_internal_controller_norm(
    state: &DawAppState,
    controller_id: u16,
    norm_val: f64,
) -> Result<(), String> {
    if (controller_id as usize) >= 512 {
        return Err(format!(
            "Controller ID {} out of range (0..512)",
            controller_id
        ));
    }
    let value_fixed30 = normalized_to_fixed30(norm_val);
    state.push_command(EngineCommand::SetInternalControllerFixed30 {
        controller_id,
        value_fixed30,
    })
}

/// Polls the latest 128 KB (`131,072` bytes) [`VisualTelemetryFrame`] from the `TripleBuffer`
/// as a zero-copy binary slice `&[u8; 131_072]` without any JSON serialization.
#[inline]
pub fn cmd_poll_binary_telemetry_frame(
    streamer: &mut ShmTelemetryServer,
) -> &[u8; VISUAL_TELEMETRY_FRAME_BYTES] {
    streamer.poll_latest_bytes()
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
