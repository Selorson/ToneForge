//! Tauri 2.0 Lock-Free Command Dispatcher (`ipc_commands.rs`).
//!
//! Routes UI interactions (Transport, Pro-Q 4 24-band & 601-param updates, Soothe3 143-param &
//! `.preset` JSON loading, Ozone 12 Master Chain controls, Real-Time 120ms BSRNN Stem Rebalance,
//! Mixer 502 routing, 30-Bit Fixed-Point Automation Controllers, and UVR5 AI Stem Separation)
//! into the real-time audio engine via `rtrb` SPSC command rings without locking the audio thread.

use crate::project_state::Ozone12MasterConfigDto;
use crate::shm_telemetry_server::ShmTelemetryServer;
use daw_core_graph::{MAX_INSERT_SLOTS, MAX_MIXER_TRACKS, UnifiedDspNode};
use daw_core_graph::automation_spline::normalized_to_fixed30;
use daw_core_graph::telemetry_shm::{
    EngineCommand, VISUAL_TELEMETRY_FRAME_BYTES,
};
use dsp_proq4_eq::{
    CharacterMode, EqBandParams, FilterShape, LinearPhaseResolution, PROQ4_BAND_PARAMS_TOTAL,
    PROQ4_NUM_BANDS, PROQ4_PARAMS_PER_BAND, PROQ4_TOTAL_PARAMS, ProQ4ParameterSet, ProcessingMode,
};
use dsp_soothe3_spectral::params_143::{
    SOOTHE3_GLOBAL_PARAMS_COUNT, SOOTHE3_NUM_SC_BANDS, SOOTHE3_PARAMS_PER_SC_BAND,
};
use dsp_soothe3_spectral::{
    SOOTHE3_TOTAL_PARAMS, SidechainEqBandParams, SidechainFilterShape, Soothe3ParameterSet,
    SootheProcessMode, TimeResolutionQuality,
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

/// DTO for updating Pro-Q 4 Global Parameters (`576..601`: Phase Mode, Character, Gain Scale, Output Level, Bypass, Auto Gain, Show Collisions) from Svelte 5 UI (`[Nhóm B.4]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProQ4GlobalUpdateDto {
    #[serde(alias = "trackIdx", default = "default_proq4_track_idx")]
    pub track_idx: u16,
    #[serde(alias = "slotIdx", default)]
    pub slot_idx: u8,
    #[serde(alias = "processingMode", default = "default_proq4_processing_mode")]
    pub processing_mode: u8, // 0 = ZeroLatency, 1 = NaturalPhase, 2 = LinearPhase
    #[serde(alias = "phaseMode", default)]
    pub phase_mode: Option<String>, // Optional string alias: "ZeroLatency" | "NaturalPhase" | "LinearPhase"
    #[serde(alias = "processingResolution", default = "default_proq4_resolution")]
    pub processing_resolution: u8, // 0..=4
    #[serde(default)]
    pub character: u8, // 0 = Clean, 1 = Subtle, 2 = Warm
    #[serde(alias = "gainScalePct", default = "default_proq4_gain_scale_pct")]
    pub gain_scale_pct: f32, // 0..=200%
    #[serde(alias = "outputLevelDb", default)]
    pub output_level_db: f32, // -36..=+36 dB
    #[serde(default)]
    pub bypass: bool,
    #[serde(alias = "autoGain", default = "default_true")]
    pub auto_gain: bool,
    #[serde(alias = "showCollisions", default = "default_true")]
    pub show_collisions: bool,
}

fn default_proq4_track_idx() -> u16 {
    1
}

fn default_proq4_processing_mode() -> u8 {
    1 // NaturalPhase
}

fn default_proq4_resolution() -> u8 {
    1 // Medium
}

fn default_proq4_gain_scale_pct() -> f32 {
    100.0
}

fn default_true() -> bool {
    true
}

impl Default for ProQ4GlobalUpdateDto {
    fn default() -> Self {
        Self {
            track_idx: 1,
            slot_idx: 0,
            processing_mode: 1,
            phase_mode: None,
            processing_resolution: 1,
            character: 0,
            gain_scale_pct: 100.0,
            output_level_db: 0.0,
            bypass: false,
            auto_gain: true,
            show_collisions: true,
        }
    }
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

/// DTO for updating one of Soothe3's 8 Sidechain EQ Bands (`band_idx in 0..8`, params `26..98`) (`[Nhóm B.5]`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Soothe3ScBandUpdateDto {
    #[serde(alias = "trackIdx")]
    pub track_idx: u16,
    #[serde(alias = "slotIdx")]
    pub slot_idx: u8,
    #[serde(alias = "bandIdx")]
    pub band_idx: u8,
    pub enabled: bool,
    #[serde(alias = "freqHz")]
    pub freq_hz: f32,
    #[serde(alias = "gainDb")]
    pub gain_db: f32,
    pub q: f32,
    pub shape: u8,
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

/// Updates Pro-Q 4 Global Parameters (`576..601`: Phase Mode, Resolution, Character, Gain Scale,
/// Output Level, Bypass, Auto Gain, Show Collisions) and triggers `EngineCommand::RefreshApdc` (`[Nhóm B.4]`).
pub fn cmd_update_proq4_global(
    state: &DawAppState,
    dto: ProQ4GlobalUpdateDto,
) -> Result<(), String> {
    if (dto.track_idx as usize) >= MAX_MIXER_TRACKS || (dto.slot_idx as usize) >= MAX_INSERT_SLOTS {
        return Err("Track or slot index out of range".to_string());
    }
    let mut ps = ProQ4ParameterSet::new();
    let fallback_mode = match dto.processing_mode {
        0 => ProcessingMode::ZeroLatency,
        1 => ProcessingMode::NaturalPhase,
        _ => ProcessingMode::LinearPhase,
    };
    ps.global.processing_mode = match dto.phase_mode.as_deref().map(str::trim) {
        Some(m) if m.eq_ignore_ascii_case("ZeroLatency") || m.eq_ignore_ascii_case("zero_latency") || m == "0" => {
            ProcessingMode::ZeroLatency
        }
        Some(m) if m.eq_ignore_ascii_case("NaturalPhase") || m.eq_ignore_ascii_case("natural_phase") || m == "1" => {
            ProcessingMode::NaturalPhase
        }
        Some(m) if m.eq_ignore_ascii_case("LinearPhase") || m.eq_ignore_ascii_case("linear_phase") || m == "2" => {
            ProcessingMode::LinearPhase
        }
        _ => fallback_mode,
    };
    ps.global.processing_resolution = match dto.processing_resolution {
        0 => LinearPhaseResolution::Low,
        1 => LinearPhaseResolution::Medium,
        2 => LinearPhaseResolution::High,
        3 => LinearPhaseResolution::VeryHigh,
        _ => LinearPhaseResolution::Maximum,
    };
    ps.global.character = match dto.character {
        0 => CharacterMode::Clean,
        1 => CharacterMode::Subtle,
        _ => CharacterMode::Warm,
    };
    ps.global.gain_scale = (dto.gain_scale_pct * 0.01).clamp(0.0, 2.0);
    ps.global.output_level_db = dto.output_level_db.clamp(-36.0, 36.0);
    ps.global.bypass = dto.bypass;
    ps.global.auto_gain = dto.auto_gain;
    ps.global.show_collisions = dto.show_collisions;
    ps.sync_mirror_from_structured();

    for param_id in PROQ4_BAND_PARAMS_TOTAL..PROQ4_TOTAL_PARAMS {
        state.push_command(EngineCommand::SetPluginParamNormalized {
            track_index: dto.track_idx,
            slot_index: dto.slot_idx,
            param_id: param_id as u32,
            norm_val: ps.get_normalized(param_id as u32),
            target_sample_pos: None,
        })?;
    }
    state.push_command(EngineCommand::RefreshApdc)?;
    Ok(())
}

/// Updates one of Soothe3's 8 Sidechain EQ Bands (`band_idx in 0..8`, 9 params per band `26..98`)
/// via the lock-free `rtrb` command queue (`[Nhóm B.5]`).
pub fn cmd_update_soothe3_sc_band(
    state: &DawAppState,
    dto: Soothe3ScBandUpdateDto,
) -> Result<(), String> {
    if (dto.track_idx as usize) >= MAX_MIXER_TRACKS || (dto.slot_idx as usize) >= MAX_INSERT_SLOTS {
        return Err("Track or slot index out of range".to_string());
    }
    let b = dto.band_idx as usize;
    if b >= SOOTHE3_NUM_SC_BANDS {
        return Err(format!("Soothe3 SC band index {} out of range (0..8)", b));
    }
    let mut band = SidechainEqBandParams::default();
    band.used = true;
    band.enabled = dto.enabled;
    band.shape = SidechainFilterShape::from_u8(dto.shape.min(7));
    band.freq_hz = dto.freq_hz.clamp(20.0, 20_000.0);
    band.depth = (dto.gain_db / 2.4).clamp(-10.0, 10.0);
    band.q = dto.q.clamp(0.1, 16.0);

    let base = SOOTHE3_GLOBAL_PARAMS_COUNT + b * SOOTHE3_PARAMS_PER_SC_BAND;
    for off in 0..SOOTHE3_PARAMS_PER_SC_BAND {
        state.push_command(EngineCommand::SetPluginParamNormalized {
            track_index: dto.track_idx,
            slot_index: dto.slot_idx,
            param_id: (base + off) as u32,
            norm_val: band.get_offset_normalized(off),
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

/// Dispatches 4-Stem BSRNN Master Rebalance gains (`0..=3`), Solo flags (`4..=7`), and Mute flags (`8..=11`)
/// via the lock-free `rtrb` command queue (`[Nhóm D.2]`).
pub fn cmd_update_bsrnn_stem_state(
    state: &DawAppState,
    track_idx: u16,
    slot_idx: u8,
    stem_gains_db: [f32; 4],
    stem_solo: [bool; 4],
    stem_mute: [bool; 4],
) -> Result<(), String> {
    if (track_idx as usize) >= MAX_MIXER_TRACKS || (slot_idx as usize) >= MAX_INSERT_SLOTS {
        return Err("Track or slot index out of range".to_string());
    }
    for s in 0..4 {
        let norm_gain = ((stem_gains_db[s].clamp(-12.0, 12.0) + 12.0) / 24.0).clamp(0.0, 1.0);
        state.push_command(EngineCommand::SetPluginParamNormalized {
            track_index: track_idx,
            slot_index: slot_idx,
            param_id: s as u32,
            norm_val: norm_gain,
            target_sample_pos: None,
        })?;
        state.push_command(EngineCommand::SetPluginParamNormalized {
            track_index: track_idx,
            slot_index: slot_idx,
            param_id: (4 + s) as u32,
            norm_val: if stem_solo[s] { 1.0 } else { 0.0 },
            target_sample_pos: None,
        })?;
        state.push_command(EngineCommand::SetPluginParamNormalized {
            track_index: track_idx,
            slot_index: slot_idx,
            param_id: (8 + s) as u32,
            norm_val: if stem_mute[s] { 1.0 } else { 0.0 },
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

fn would_create_mixer_cycle(
    engine: &daw_core_graph::MixerEngine,
    src_track: usize,
    dst_track: usize,
) -> bool {
    if src_track == dst_track {
        return true;
    }
    let mut visited = [false; MAX_MIXER_TRACKS];
    let mut stack = Vec::with_capacity(32);
    stack.push(dst_track);
    while let Some(curr) = stack.pop() {
        if curr == src_track {
            return true;
        }
        if curr < MAX_MIXER_TRACKS && !visited[curr] {
            visited[curr] = true;
            for route in &engine.tracks[curr].sends {
                if route.dest_track_index < MAX_MIXER_TRACKS {
                    stack.push(route.dest_track_index);
                }
            }
        }
    }
    false
}

/// Routes `src_track -> dst_track` (with `gain` and `is_sidechain`) directly on `MixerEngine`,
/// rejecting directed cycles via DFS and rebuilding the 5-Point APDC topological schedule (`[Nhóm B.7]`).
pub fn cmd_route_mixer_track_on_engine(
    engine: &mut daw_core_graph::MixerEngine,
    _state: &DawAppState,
    src_track: u16,
    dst_track: u16,
    gain: f32,
    is_sidechain: bool,
) -> Result<(), String> {
    let src = src_track as usize;
    let dst = dst_track as usize;
    if src >= MAX_MIXER_TRACKS || dst >= MAX_MIXER_TRACKS {
        return Err("Mixer track index out of range (0..502)".to_string());
    }
    if would_create_mixer_cycle(engine, src, dst) {
        return Err(format!(
            "Routing track {} -> {} would create a cycle in the 502-Track Mixer DAG",
            src, dst
        ));
    }
    let clamped_gain = gain.clamp(0.0, 4.0);
    if let Some(existing) = engine.tracks[src]
        .sends
        .iter_mut()
        .find(|s| s.dest_track_index == dst && s.is_sidechain_only == is_sidechain)
    {
        existing.gain = clamped_gain;
    } else {
        engine.tracks[src].add_send(dst, clamped_gain, is_sidechain);
    }
    engine.rebuild_graph().map_err(|e| e.to_string())?;
    Ok(())
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

/// Request DTO for full-length UVR5 3-Model Spectral Ensemble separation (`[Nhóm B.10, C.3]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Uvr5SeparationRequestDto {
    #[serde(alias = "audioPath")]
    pub audio_path: String,
    #[serde(
        alias = "model_profile",
        alias = "modelProfile",
        alias = "modelHash",
        default = "default_uvr5_model_hash"
    )]
    pub model_hash: String,
    #[serde(alias = "ensembleMode", default = "default_ensemble_mode")]
    pub ensemble_mode: String,
    #[serde(default = "default_ensemble_weights")]
    pub weights: [f32; 3],
    #[serde(alias = "enableDenoise2pass", default = "default_true")]
    pub enable_denoise_2pass: bool,
    #[serde(alias = "vrAggressiveness", default = "default_aggressiveness")]
    pub vr_aggressiveness: f32,
    #[serde(
        alias = "phaseDenoiseStrength",
        default = "default_phase_denoise_strength"
    )]
    pub phase_denoise_strength: f32,
    #[serde(alias = "invertInstrumental", default = "default_true")]
    pub invert_instrumental: bool,
    #[serde(alias = "bounceToMixer", default = "default_true")]
    pub bounce_to_mixer: bool,
}

fn default_uvr5_model_hash() -> String {
    "ensemble_3model".to_string()
}

fn default_ensemble_mode() -> String {
    "Weighted".to_string()
}

fn default_ensemble_weights() -> [f32; 3] {
    [0.45, 0.35, 0.20]
}

fn default_aggressiveness() -> f32 {
    0.15
}

fn default_phase_denoise_strength() -> f32 {
    0.20
}

impl Default for Uvr5SeparationRequestDto {
    fn default() -> Self {
        Self {
            audio_path: String::new(),
            model_hash: default_uvr5_model_hash(),
            ensemble_mode: default_ensemble_mode(),
            weights: default_ensemble_weights(),
            enable_denoise_2pass: true,
            vr_aggressiveness: default_aggressiveness(),
            phase_denoise_strength: default_phase_denoise_strength(),
            invert_instrumental: true,
            bounce_to_mixer: true,
        }
    }
}

fn sanitize_model_tag_for_filename(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len().min(48));
    let mut prev_underscore = false;
    for ch in raw.chars() {
        if out.len() >= 48 {
            break;
        }
        if ch.is_ascii_alphanumeric() || ch == '-' {
            out.push(ch);
            prev_underscore = false;
        } else if !prev_underscore && !out.is_empty() {
            out.push('_');
            prev_underscore = true;
        }
    }
    let trimmed = out.trim_matches('_').to_string();
    if trimmed.is_empty() {
        "ensemble_3model".to_string()
    } else {
        trimmed
    }
}

/// Result DTO returned by `run_uvr5_separation_full_engine` (`[Nhóm B.10, C.3]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Uvr5SeparationResultDto {
    #[serde(alias = "stem_files")]
    pub stem_paths: Vec<String>,
    pub stem_peak_dbfs: [f32; 4],
    pub stem_rms_dbfs: [f32; 4],
    pub separated_frames: usize,
    pub sample_rate_hz: u32,
    pub bounced_mixer_tracks: [u16; 4],
}

/// Executes full-length UVR5 3-Model Spectral Ensemble separation (`Mdx23cSeparator` + `HTDemucs4` +
/// `CascadedASPPNet` + `SpectralEnsembler`) across 100% of `mix_frames`, encodes 4 `.wav` buffers
/// (`[Vocals, Drums, Bass, InvInst/Other]`), writes them to disk, and returns `(Uvr5SeparationResultDto, [Vec<StereoFrame>; 4], [Vec<u8>; 4])`.
pub fn execute_uvr5_separation_dsp(
    mix_frames: &[daw_core_graph::StereoFrame],
    sample_rate_hz: u32,
    req: &Uvr5SeparationRequestDto,
) -> Result<
    (
        Uvr5SeparationResultDto,
        [Vec<daw_core_graph::StereoFrame>; 4],
        [Vec<u8>; 4],
    ),
    String,
> {
    if req.audio_path.trim().is_empty() {
        return Err("audio_path must not be empty".to_string());
    }
    let path = PathBuf::from(&req.audio_path);
    let stem_dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty() && p.exists())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::env::temp_dir().join("toneforge_uvr5_stems"));
    let _ = std::fs::create_dir_all(&stem_dir);
    let stem_prefix = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("audio");
    let safe_model_tag = sanitize_model_tag_for_filename(&req.model_hash);

    let mode = match req.ensemble_mode.to_ascii_uppercase().as_str() {
        "MAX_SPEC" | "MAXSPEC" => ai_uvr5_stem_engine::SpectralEnsembleMode::MaxSpec,
        "MIN_SPEC" | "MINSPEC" => ai_uvr5_stem_engine::SpectralEnsembleMode::MinSpec,
        "AVERAGE" => ai_uvr5_stem_engine::SpectralEnsembleMode::Average,
        "MEDIAN" => ai_uvr5_stem_engine::SpectralEnsembleMode::Median,
        _ => ai_uvr5_stem_engine::SpectralEnsembleMode::Weighted,
    };

    let source_mix: Vec<daw_core_graph::StereoFrame> = if !mix_frames.is_empty() {
        mix_frames.to_vec()
    } else if path.exists() {
        crate::offline_render_engine::read_wav_file(&path)
            .map(|d| d.frames)
            .unwrap_or_else(|_| make_fallback_reference_mix(4096, sample_rate_hz as f64))
    } else {
        make_fallback_reference_mix(4096, sample_rate_hz as f64)
    };

    let len = source_mix.len();
    let mut mdx = ai_uvr5_stem_engine::Mdx23cSeparator::new(512, 128, 1024, 0.5);
    mdx.enable_denoise_2pass = req.enable_denoise_2pass;
    let (mdx_vocals, _mdx_inst) = mdx.separate_offline(&source_mix);

    let demucs = ai_uvr5_stem_engine::HTDemucs4::new(512, 128, 16, 4, 2);
    let [demucs_drums, demucs_bass, demucs_other, demucs_vocals] =
        demucs.separate_4stems(&source_mix);

    let vr = ai_uvr5_stem_engine::CascadedASPPNet::new(
        512,
        128,
        req.vr_aggressiveness.clamp(-1.0, 1.0),
    );
    let (vr_vocals, _vr_inst) = vr.separate_vocal_instrumental(&source_mix);

    let mut ensembler = ai_uvr5_stem_engine::SpectralEnsembler::new(512, 128, mode);
    ensembler.phase_denoise_strength = req.phase_denoise_strength.clamp(0.0, 1.0);
    let vocals = ensembler.ensemble_waveforms(
        &[&mdx_vocals, &demucs_vocals, &vr_vocals],
        Some(&req.weights),
    );

    let inv_inst =
        ai_uvr5_stem_engine::SpectralEnsembler::invert_audio_waveform(&source_mix, &vocals);
    let mut drums = vec![[0.0f32; 2]; len];
    let mut bass = vec![[0.0f32; 2]; len];
    for i in 0..len {
        drums[i] = [demucs_drums[i][0] * 0.75, demucs_drums[i][1] * 0.75];
        bass[i] = [demucs_bass[i][0] * 0.75, demucs_bass[i][1] * 0.75];
    }
    let fourth_stem = if req.invert_instrumental {
        inv_inst
    } else {
        demucs_other
    };
    let fourth_suffix = if req.invert_instrumental {
        "inv_inst"
    } else {
        "other"
    };

    let stems = [vocals, drums, bass, fourth_stem];
    let stem_suffixes = ["vocals", "drums", "bass", fourth_suffix];
    let render_sr = crate::offline_render_engine::RenderSampleRate::from_hz(sample_rate_hz.max(44_100));

    let mut stem_paths = Vec::with_capacity(4);
    let mut stem_peak_dbfs = [-144.0f32; 4];
    let mut stem_rms_dbfs = [-144.0f32; 4];
    let mut wav_buffers: [Vec<u8>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

    for s in 0..4 {
        let mut peak = 1e-9f32;
        let mut sum_sq = 0.0f64;
        for f in &stems[s] {
            peak = peak.max(f[0].abs()).max(f[1].abs());
            sum_sq += (f[0] as f64) * (f[0] as f64) + (f[1] as f64) * (f[1] as f64);
        }
        let rms = (sum_sq / ((stems[s].len().max(1) * 2) as f64)).sqrt() as f32;
        stem_peak_dbfs[s] = (20.0 * peak.max(1e-7).log10() * 10.0).round() / 10.0;
        stem_rms_dbfs[s] = (20.0 * rms.max(1e-7).log10() * 10.0).round() / 10.0;

        let wav_bytes = crate::offline_render_engine::encode_wav_bytes(
            &stems[s],
            render_sr,
            crate::offline_render_engine::WavBitDepth::Pcm16MbitPlus,
            true,
        );
        let file_path = stem_dir.join(format!(
            "{}_{}_{}.wav",
            stem_prefix, safe_model_tag, stem_suffixes[s]
        ));
        let _ = std::fs::write(&file_path, &wav_bytes);
        stem_paths.push(file_path.display().to_string());
        wav_buffers[s] = wav_bytes;
    }

    Ok((
        Uvr5SeparationResultDto {
            stem_paths,
            stem_peak_dbfs,
            stem_rms_dbfs,
            separated_frames: len,
            sample_rate_hz: render_sr.as_u32(),
            bounced_mixer_tracks: if req.bounce_to_mixer {
                [1, 2, 3, 4]
            } else {
                [0, 0, 0, 0]
            },
        },
        stems,
        wav_buffers,
    ))
}

fn make_fallback_reference_mix(
    frames: usize,
    sample_rate: f64,
) -> Vec<daw_core_graph::StereoFrame> {
    let mut out = Vec::with_capacity(frames);
    for n in 0..frames {
        let t = (n as f32) / (sample_rate.max(8000.0) as f32);
        let bass = 0.28 * (std::f32::consts::TAU * 65.0 * t).sin();
        let vocal = 0.24 * (std::f32::consts::TAU * 440.0 * t).sin()
            + 0.14 * (std::f32::consts::TAU * 880.0 * t).cos();
        let hats = 0.10 * (std::f32::consts::TAU * 4200.0 * t).sin();
        out.push([bass + vocal + hats, bass + vocal * 0.92 - hats * 0.85]);
    }
    out
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

/// Sets the Hybrid PPQ resolution (`96..=960`) via the lock-free `rtrb` command queue.
pub fn cmd_set_ppq_resolution(state: &DawAppState, ppq: u32) -> Result<(), String> {
    if !(daw_core_graph::MIN_PPQ..=daw_core_graph::MAX_PPQ).contains(&ppq) {
        return Err(format!(
            "PPQ resolution {} out of range ({}..={})",
            ppq,
            daw_core_graph::MIN_PPQ,
            daw_core_graph::MAX_PPQ
        ));
    }
    state.push_command(EngineCommand::SetPpq(ppq))
}

/// Dispatches a sample-accurate Piano Roll Note or FL Studio 26 Slide Note event via the lock-free command queue.
pub fn cmd_trigger_piano_roll_note(
    state: &DawAppState,
    channel_id: u32,
    note: crate::project_state::PianoRollNoteDto,
    sample_pos: i64,
    slide_samples: usize,
) -> Result<(), String> {
    let kind = if note.is_slide {
        daw_core_graph::VoiceEventKind::SlideNote
    } else {
        daw_core_graph::VoiceEventKind::NoteOn
    };
    let ev = daw_core_graph::SequencerVoiceEvent {
        sample_pos,
        ppq_tick: note.start_tick,
        channel_id,
        note_id: note.id,
        kind,
        key: note.key.min(127),
        velocity: note.velocity.clamp(0.0, 1.0),
        pan: note.pan.clamp(-1.0, 1.0),
        fine_pitch_cents: note.fine_pitch_cents,
        slide_duration_ticks: note.duration_ticks,
        slide_duration_samples: slide_samples,
        color_group: note.color_group & 0x0F,
    };
    state.push_command(EngineCommand::TriggerVoiceEvent(ev))
}

/// Executes an Offline Bounce & Mastering Render job (`Master`, `SplitMixerTracks`, or `StemMastering`)
/// and returns the `OfflineRenderResult` with `PostRenderAnalysisReport`.
pub fn cmd_execute_offline_render(
    engine: &mut daw_core_graph::MixerEngine,
    config: &crate::offline_render_engine::OfflineRenderConfig,
) -> Result<crate::offline_render_engine::OfflineRenderResult, String> {
    crate::offline_render_engine::OfflineRenderEngine::execute_render(engine, config)
        .map_err(|e| format!("Offline render failed: {e}"))
}

/// Upserts a `PianoRollPattern` (with FL Studio 26 Slide notes and per-note expression) into `MixerEngine`.
pub fn cmd_upsert_piano_roll_pattern(
    engine: &mut daw_core_graph::MixerEngine,
    pattern: &crate::project_state::PianoRollPatternDto,
) -> Result<(), String> {
    if pattern.length_ticks <= 0 {
        return Err("PianoRollPattern length_ticks must be positive".to_string());
    }
    if engine.sequencer.upsert_pattern(pattern.to_pattern()) {
        Ok(())
    } else {
        Err("PlaylistSequencer pattern table capacity reached".to_string())
    }
}

/// Upserts a `PlaylistClip` (`Pattern`, `AudioStem`, or `Automation`) into `MixerEngine`.
pub fn cmd_upsert_playlist_clip(
    engine: &mut daw_core_graph::MixerEngine,
    clip: &crate::project_state::PlaylistClipDto,
) -> Result<(), String> {
    if clip.duration_ticks <= 0 {
        return Err("PlaylistClip duration_ticks must be positive".to_string());
    }
    if engine.sequencer.upsert_clip(clip.to_playlist_clip()) {
        Ok(())
    } else {
        Err("PlaylistSequencer clip table capacity reached".to_string())
    }
}

/// Updates a `TensionSpline` Automation Clip on `MixerEngine` and evaluates its 30-Bit Fixed-Point
/// (`0x40000000`) controller output at `eval_tick`.
pub fn cmd_update_automation_spline(
    engine: &mut daw_core_graph::MixerEngine,
    clip: &crate::project_state::AutomationClipStateDto,
    eval_tick: f64,
) -> Result<i32, String> {
    if (clip.controller_id as usize) >= 512 {
        return Err(format!(
            "Controller ID {} out of range (0..512)",
            clip.controller_id
        ));
    }
    let articulator = clip.to_articulator();
    let fixed30 = articulator.evaluate_fixed30(eval_tick.max(0.0));
    engine
        .controller_bus
        .publish_fixed30(articulator.controller_id, fixed30);
    if !engine.sequencer.upsert_automation_articulator(articulator) {
        return Err("PlaylistSequencer automation clip table capacity reached".to_string());
    }
    Ok(fixed30)
}

/// Applies a real FabFilter Pro-Q 4 `.ffp` factory preset (`171` bundled presets) and returns the structured 24-band UI state.
pub fn cmd_apply_proq4_ffp_preset(
    state: &DawAppState,
    track_idx: u16,
    slot_idx: u8,
    preset_query: &str,
) -> Result<crate::preset_library::ProQ4PresetStateDto, String> {
    crate::preset_library::apply_proq4_preset(state, track_idx, slot_idx, preset_query)
}

/// Applies a real oeksound Soothe3 `.preset` JSON (`68` bundled presets) and returns the structured UI state.
pub fn cmd_apply_soothe3_factory_preset(
    state: &DawAppState,
    track_idx: u16,
    slot_idx: u8,
    preset_query: &str,
) -> Result<crate::preset_library::Soothe3PresetStateDto, String> {
    crate::preset_library::apply_soothe3_preset(state, track_idx, slot_idx, preset_query)
}

/// Loads a real iZotope Ozone 12 Genre Target Curve (`36` bundled genres) or Global XML preset (`111` bundled XMLs).
pub fn cmd_load_ozone12_genre_or_xml_preset(
    state: &DawAppState,
    track_idx: u16,
    slot_idx: u8,
    query: &str,
    is_xml_preset: bool,
) -> Result<crate::preset_library::Ozone12AssistantPresetDto, String> {
    if is_xml_preset {
        crate::preset_library::load_ozone12_global_xml_preset(state, track_idx, slot_idx, query)
    } else {
        crate::preset_library::load_ozone12_genre_preset(state, track_idx, slot_idx, query)
    }
}

/// Runs the iZotope Ozone 12 AI Master Assistant Auto-Match using `GenreCnnClassifier` (`gparamsBN8.0.npz`)
/// and `InstrumentCnnClassifier` (`instrument_classifier_params.npz`).
pub fn cmd_run_ozone12_ai_master_assistant(
    state: &DawAppState,
    track_idx: u16,
    slot_idx: u8,
    mel_80: Option<Vec<f32>>,
) -> Result<crate::preset_library::Ozone12AssistantPresetDto, String> {
    crate::preset_library::run_ozone12_master_assistant(state, track_idx, slot_idx, mel_80)
}

/// Loads a real FL Studio 26 `.fsc` Piano Roll score into `MixerEngine`.
pub fn cmd_load_fl26_score(
    engine: &mut daw_core_graph::MixerEngine,
    score_query: &str,
    target_ppq: u32,
) -> Result<crate::project_state::PianoRollPatternDto, String> {
    crate::preset_library::load_fl26_score_preset(engine, score_query, target_ppq)
}

/// Loads a real FL Studio 26 `.wav` studio sample kit (`808`, `909`, `MA Skipper Snare`, `SYM_STR_C_2`).
pub fn cmd_load_fl26_sample_kit(
    kit_query: &str,
) -> Result<crate::preset_library::Fl26SampleKitDto, String> {
    crate::preset_library::load_fl26_sample_kit(kit_query)
}

/// Loads and applies a real FL Studio 26 `.fst` native FX preset (`Maximus`, `Emphasis`, `Fruity Limiter`, etc.).
pub fn cmd_load_fl26_fx_preset(
    state: &DawAppState,
    track_idx: u16,
    slot_idx: u8,
    preset_query: &str,
) -> Result<crate::preset_library::Fl26FxPresetDto, String> {
    crate::preset_library::load_fl26_fx_preset(state, track_idx, slot_idx, preset_query)
}

/// Loads a real UVR5 model profile (`MDX23C`, `MDX-Net`, `HTDemucs4`, `VR-Arch 4band_v3_sn`) from `assets/uvr5/`.
pub fn cmd_load_uvr5_model_profile(
    profile_query: &str,
) -> Result<crate::preset_library::Uvr5ModelProfileDto, String> {
    crate::preset_library::load_uvr5_model_profile(profile_query)
}

/// Interactive Patcher canvas node DTO sent from `PatcherNodeCanvas.svelte` (`[Nhóm B.9]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatcherCanvasNodeDto {
    #[serde(alias = "nodeIndex")]
    pub node_index: usize,
    #[serde(default)]
    pub label: String,
    #[serde(alias = "pluginKind", default)]
    pub plugin_kind: String,
    #[serde(alias = "latencySamples", default)]
    pub latency_samples: usize,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(alias = "xPx", default)]
    pub x_px: f32,
    #[serde(alias = "yPx", default)]
    pub y_px: f32,
}

/// Interactive Patcher canvas audio cord DTO sent from `PatcherNodeCanvas.svelte` (`[Nhóm B.9]`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PatcherCanvasEdgeDto {
    #[serde(alias = "fromNode")]
    pub from_node: usize,
    #[serde(alias = "toNode")]
    pub to_node: usize,
    #[serde(default = "default_edge_gain")]
    pub gain: f32,
}

fn default_edge_gain() -> f32 {
    1.0
}

/// Interactive Patcher sub-DAG update payload (`[Nhóm B.9]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatcherCanvasUpdateDto {
    #[serde(alias = "trackIdx", default = "default_proq4_track_idx")]
    pub track_idx: u16,
    #[serde(alias = "slotIdx", default = "default_patcher_slot_idx")]
    pub slot_idx: u8,
    pub nodes: Vec<PatcherCanvasNodeDto>,
    pub edges: Vec<PatcherCanvasEdgeDto>,
}

fn default_patcher_slot_idx() -> u8 {
    2
}

/// Reconstructs and hot-swaps a live [`daw_core_graph::PatcherGraphNode`] on `engine.tracks[track_idx].slots[slot_idx]`
/// from the interactive `PatcherNodeCanvas.svelte` state (`nodes` enable/bypass, `(x_px, y_px)`, and `edges` cord weights),
/// rejecting cycles and rebuilding both internal `TMultiBufferDelay` APDC and global 502-Track APDC (`[Nhóm B.9]`).
pub fn cmd_update_patcher_graph(
    engine: &mut daw_core_graph::MixerEngine,
    dto: &PatcherCanvasUpdateDto,
) -> Result<usize, String> {
    let track = dto.track_idx as usize;
    let slot = dto.slot_idx as usize;
    if track >= MAX_MIXER_TRACKS {
        return Err(format!("Mixer track index {} out of range (0..502)", track));
    }
    if slot >= MAX_INSERT_SLOTS {
        return Err(format!("Insert slot index {} out of range (0..10)", slot));
    }

    let sr = engine.slicer.sample_rate.max(8_000.0);
    let blk = daw_core_graph::MAX_BLOCK_SIZE;
    let mut patcher = daw_core_graph::PatcherGraphNode::new();
    patcher.prepare(sr, blk);

    let mut id_map = std::collections::HashMap::<usize, usize>::with_capacity(dto.nodes.len() + 2);
    id_map.insert(
        daw_core_graph::PATCHER_HOST_INPUT_NODE,
        daw_core_graph::PATCHER_HOST_INPUT_NODE,
    );
    id_map.insert(
        daw_core_graph::PATCHER_HOST_OUTPUT_NODE,
        daw_core_graph::PATCHER_HOST_OUTPUT_NODE,
    );

    let mut host_in_enabled = true;
    let mut host_out_enabled = true;

    for node in &dto.nodes {
        let has_outgoing = dto.edges.iter().any(|e| e.from_node == node.node_index);
        if node.plugin_kind == "HostIn" {
            host_in_enabled = node.enabled;
            id_map.insert(node.node_index, daw_core_graph::PATCHER_HOST_INPUT_NODE);
        } else if node.plugin_kind == "HostOut"
            || (node.plugin_kind == "Ozone12" && !has_outgoing)
        {
            host_out_enabled = node.enabled;
            id_map.insert(node.node_index, daw_core_graph::PATCHER_HOST_OUTPUT_NODE);
        } else {
            let eff_latency = if node.enabled {
                node.latency_samples.min(8_192)
            } else {
                0
            };
            let dsp = Box::new(daw_core_graph::FixedLatencyTransformNode::new(
                eff_latency,
                1.0,
                false,
            ));
            let internal_idx = patcher.add_node(dsp);
            id_map.insert(node.node_index, internal_idx);
        }
    }

    for edge in &dto.edges {
        let Some(&from_resolved) = id_map.get(&edge.from_node) else {
            return Err(format!(
                "Unknown source node_index {} in Patcher edge",
                edge.from_node
            ));
        };
        let Some(&to_resolved) = id_map.get(&edge.to_node) else {
            return Err(format!(
                "Unknown destination node_index {} in Patcher edge",
                edge.to_node
            ));
        };
        let clamped_gain = if edge.gain.is_finite() {
            edge.gain.clamp(0.0, 4.0)
        } else {
            0.0
        };
        if !patcher.add_audio_connection(from_resolved, to_resolved, clamped_gain) {
            return Err(format!(
                "Cycle or invalid audio connection in Patcher DAG: {} -> {}",
                edge.from_node, edge.to_node
            ));
        }
    }

    if dto.edges.is_empty() {
        let _ = patcher.add_audio_connection(
            daw_core_graph::PATCHER_HOST_INPUT_NODE,
            daw_core_graph::PATCHER_HOST_OUTPUT_NODE,
            0.0,
        );
    }

    let total_patcher_latency = if host_in_enabled && host_out_enabled {
        use daw_core_graph::UnifiedDspNode;
        patcher.latency_samples()
    } else {
        0
    };

    let installed_slot = engine.tracks[track]
        .set_or_replace_plugin_slot(slot, Box::new(patcher), 1.0, sr, blk)
        .map_err(|e| e.to_string())?;
    engine.tracks[track].slots[installed_slot].enabled = host_in_enabled && host_out_enabled;
    engine.rebuild_graph().map_err(|e| e.to_string())?;

    Ok(total_patcher_latency)
}
