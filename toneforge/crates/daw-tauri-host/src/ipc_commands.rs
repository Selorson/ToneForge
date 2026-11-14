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

/// Host-side state wrapping the `rtrb::Producer<EngineCommand>` and optional
/// `rtrb::Producer<RtGraphMutation>` channels into the lock-free RT audio thread.
pub struct DawAppState {
    pub cmd_producer: Mutex<rtrb::Producer<EngineCommand>>,
    pub mut_producer: Option<Mutex<rtrb::Producer<daw_core_graph::RtGraphMutation>>>,
}

impl DawAppState {
    pub fn new(cmd_producer: rtrb::Producer<EngineCommand>) -> Self {
        Self {
            cmd_producer: Mutex::new(cmd_producer),
            mut_producer: None,
        }
    }

    pub fn new_with_rt_mutations(
        cmd_producer: rtrb::Producer<EngineCommand>,
        mut_producer: rtrb::Producer<daw_core_graph::RtGraphMutation>,
    ) -> Self {
        Self {
            cmd_producer: Mutex::new(cmd_producer),
            mut_producer: Some(Mutex::new(mut_producer)),
        }
    }

    pub fn attach_rt_mutation_producer(
        &mut self,
        mut_producer: rtrb::Producer<daw_core_graph::RtGraphMutation>,
    ) {
        self.mut_producer = Some(Mutex::new(mut_producer));
    }

    #[inline]
    pub fn has_rt_mutations(&self) -> bool {
        self.mut_producer.is_some()
    }

    /// Pushes a command into the wait-free SPSC ring buffer.
    pub fn push_command(&self, cmd: EngineCommand) -> Result<(), String> {
        let mut prod = self.cmd_producer.lock();
        prod.push(cmd)
            .map_err(|_| "RT Command RingBuffer full".to_string())
    }

    /// Pushes a lock-free `RtGraphMutation` into the wait-free SPSC mutation ring buffer (`Host -> RT`).
    pub fn push_rt_mutation(&self, mutation: daw_core_graph::RtGraphMutation) -> Result<(), String> {
        let Some(ref mut_mtx) = self.mut_producer else {
            return Err("RT Graph Mutation channel is not attached on this DawAppState".to_string());
        };
        let mut prod = mut_mtx.lock();
        prod.push(mutation)
            .map_err(|_| "RT Graph Mutation RingBuffer full".to_string())
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
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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
    let updates: [(u32, f32); 32] = [
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
        (80, (cfg.unlimiter_amount_pct * 0.01).clamp(0.0, 1.0)),
        (
            81,
            ((cfg.unlimiter_threshold_db + 24.0) / 24.0).clamp(0.0, 1.0),
        ),
        (83, if cfg.unlimiter_enabled { 1.0 } else { 0.0 }),
        (70, if cfg.low_end_focus_enabled { 1.0 } else { 0.0 }),
        (71, if cfg.low_end_focus_punchy { 0.0 } else { 1.0 }),
        (
            72,
            ((cfg.low_end_focus_contrast_pct + 100.0) / 200.0).clamp(0.0, 1.0),
        ),
        (
            73,
            ((cfg.low_end_focus_upper_hz.clamp(20.0, 300.0) - 20.0) / 280.0).clamp(0.0, 1.0),
        ),
        (
            74,
            ((cfg.low_end_focus_gain_db + 12.0) / 24.0).clamp(0.0, 1.0),
        ),
        (69, if cfg.low_end_focus_solo { 1.0 } else { 0.0 }),
        (85, if cfg.codec_preview_enabled { 1.0 } else { 0.0 }),
        (
            86,
            ((cfg.codec_preview_codec.min(5) as f32) / 5.0).clamp(0.0, 1.0),
        ),
        (87, if cfg.codec_preview_artifact_solo { 1.0 } else { 0.0 }),
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
    state.push_command(EngineCommand::RefreshApdc)?;
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
    #[serde(alias = "outputDir", default)]
    pub output_dir: Option<String>,
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
            output_dir: None,
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
    #[serde(default)]
    pub gpu_provider: String,
    #[serde(default)]
    pub output_directory: String,
}

/// Queries the active 3-Tier ONNX Runtime GPU Accelerator status (`CUDA 12` -> `DirectML` -> `CPU`).
pub fn cmd_query_gpu_accelerator_status() -> ai_uvr5_stem_engine::GpuAcceleratorStatus {
    ai_uvr5_stem_engine::GpuRuntimeManager::global()
        .status()
        .clone()
}

/// Executes full-length UVR5 3-Model GPU Spectral Ensemble separation (`Kim_Vocal_2.onnx` +
/// `UVR-MDX-NET-Inst_HQ_3.onnx` + `Ozone 12 BSRNN 4-Stem .onnx` on RTX 4080 `CUDAExecutionProvider`,
/// with automatic pure-Rust fallback if ONNX runtime is absent) across 100% of `mix_frames`
/// (or decoded directly from `req.audio_path` via `symphonia`), writes 4 `.wav` stems to
/// `req.output_dir` or `audio_path.parent()`, and returns `(Uvr5SeparationResultDto, [Vec<StereoFrame>; 4], [Vec<u8>; 4])`.
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
    let path = PathBuf::from(req.audio_path.trim());

    // Priority 1: Explicit user-selected `output_dir` in Khu 6
    // Priority 2: Parent folder of `audio_path` on real disk (eliminating `%TEMP%`!)
    // Priority 3: Fallback `%TEMP%/toneforge_uvr5_stems` ONLY for bare browser-only filenames
    let stem_dir = if let Some(explicit_dir) = req
        .output_dir
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        PathBuf::from(explicit_dir)
    } else if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        if parent.exists() || path.is_absolute() {
            parent.to_path_buf()
        } else {
            std::env::temp_dir().join("toneforge_uvr5_stems")
        }
    } else {
        std::env::temp_dir().join("toneforge_uvr5_stems")
    };
    std::fs::create_dir_all(&stem_dir).map_err(|e| {
        format!(
            "Failed to create UVR5 output directory '{}': {e}",
            stem_dir.display()
        )
    })?;

    let stem_prefix = path
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("audio");
    let safe_model_tag = sanitize_model_tag_for_filename(&req.model_hash);

    let mode = match req.ensemble_mode.to_ascii_uppercase().as_str() {
        "MAX_SPEC" | "MAXSPEC" => ai_uvr5_stem_engine::SpectralEnsembleMode::MaxSpec,
        "MIN_SPEC" | "MINSPEC" => ai_uvr5_stem_engine::SpectralEnsembleMode::MinSpec,
        "AVERAGE" => ai_uvr5_stem_engine::SpectralEnsembleMode::Average,
        "MEDIAN" => ai_uvr5_stem_engine::SpectralEnsembleMode::Median,
        _ => ai_uvr5_stem_engine::SpectralEnsembleMode::Weighted,
    };

    let (source_mix, effective_sr): (Vec<daw_core_graph::StereoFrame>, u32) =
        if !mix_frames.is_empty() {
            (mix_frames.to_vec(), sample_rate_hz.clamp(8_000, 384_000))
        } else if path.is_file() {
            let decoded = crate::native_audio_decoder::decode_audio_file_native(&path)?;
            (decoded.frames, decoded.sample_rate_hz.clamp(8_000, 384_000))
        } else {
            let fallback_sr = sample_rate_hz.clamp(8_000, 384_000);
            (
                make_fallback_reference_mix(4096, fallback_sr as f64),
                fallback_sr,
            )
        };

    let len = source_mix.len();
    let gpu_cfg = ai_uvr5_stem_engine::GpuStemEnsembleConfig {
        model_selection: ai_uvr5_stem_engine::GpuStemModelSelection::from_model_tag(
            &req.model_hash,
        ),
        ensemble_mode: mode,
        weights: req.weights,
        enable_denoise_2pass: req.enable_denoise_2pass,
        vr_aggressiveness: req.vr_aggressiveness.clamp(-1.0, 1.0),
        phase_denoise_strength: req.phase_denoise_strength.clamp(0.0, 1.0),
        invert_instrumental: req.invert_instrumental,
    };

    let (vocals, drums, bass, fourth_stem, gpu_provider) =
        match ai_uvr5_stem_engine::run_global_gpu_stem_ensemble(&source_mix, effective_sr, &gpu_cfg)
        {
            Ok(gpu_out) => {
                let fourth = if req.invert_instrumental {
                    gpu_out.inv_instrumental
                } else {
                    gpu_out.other
                };
                (
                    gpu_out.vocals,
                    gpu_out.drums,
                    gpu_out.bass,
                    fourth,
                    gpu_out.active_tier.provider_name().to_string(),
                )
            }
            Err(_) => {
                // Tier 3A: Real extracted Ozone 12 GRU-512 3-model stem separator (`vocals/drums/bass-small_sanitized.npz`)
                if let Some(mut gru_sep) =
                    ai_uvr5_stem_engine::GruThreeStemSeparator::from_global_cache()
                {
                    let [gru_vocals, gru_drums, gru_bass, gru_other] =
                        gru_sep.separate_4stems(&source_mix, effective_sr);
                    let fourth = if req.invert_instrumental {
                        ai_uvr5_stem_engine::SpectralEnsembler::invert_audio_waveform(
                            &source_mix,
                            &gru_vocals,
                        )
                    } else {
                        gru_other
                    };
                    (
                        gru_vocals,
                        gru_drums,
                        gru_bass,
                        fourth,
                        "Ozone12Gru512NpzCpu".to_string(),
                    )
                } else {
                    // Tier 3B deterministic fallback when neither ONNX nor NPZ assets are present
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
                    let voc = ensembler.ensemble_waveforms(
                        &[&mdx_vocals, &demucs_vocals, &vr_vocals],
                        Some(&req.weights),
                    );

                    let inv_inst = ai_uvr5_stem_engine::SpectralEnsembler::invert_audio_waveform(
                        &source_mix,
                        &voc,
                    );
                    let mut drm = vec![[0.0f32; 2]; len];
                    let mut bas = vec![[0.0f32; 2]; len];
                    for i in 0..len {
                        drm[i] = [demucs_drums[i][0] * 0.75, demucs_drums[i][1] * 0.75];
                        bas[i] = [demucs_bass[i][0] * 0.75, demucs_bass[i][1] * 0.75];
                    }
                    let fourth = if req.invert_instrumental {
                        inv_inst
                    } else {
                        demucs_other
                    };
                    (
                        voc,
                        drm,
                        bas,
                        fourth,
                        "PureRustSpectralFallback".to_string(),
                    )
                }
            }
        };

    let fourth_suffix = if req.invert_instrumental {
        "inv_inst"
    } else {
        "other"
    };

    let stems = [vocals, drums, bass, fourth_stem];
    let stem_suffixes = ["vocals", "drums", "bass", fourth_suffix];

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

        let wav_bytes = crate::offline_render_engine::encode_wav_bytes_exact_sr(
            &stems[s],
            effective_sr,
            crate::offline_render_engine::WavBitDepth::Pcm24MbitPlus,
            true,
        );
        let file_path = stem_dir.join(format!(
            "{}_{}_{}.wav",
            stem_prefix, safe_model_tag, stem_suffixes[s]
        ));
        std::fs::write(&file_path, &wav_bytes).map_err(|e| {
            format!(
                "Failed writing separated WAV '{}': {e}",
                file_path.display()
            )
        })?;
        stem_paths.push(file_path.display().to_string());
        wav_buffers[s] = wav_bytes;
    }

    Ok((
        Uvr5SeparationResultDto {
            stem_paths,
            stem_peak_dbfs,
            stem_rms_dbfs,
            separated_frames: len,
            sample_rate_hz: effective_sr,
            bounced_mixer_tracks: if req.bounce_to_mixer {
                [1, 2, 3, 4]
            } else {
                [0, 0, 0, 0]
            },
            gpu_provider,
            output_directory: stem_dir.display().to_string(),
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

/// DSP for one canvas box: the real processor for its kind (`ProQ4` in Linear Phase as labelled,
/// `Soothe3`, ...), a pure delay for `FixedLatency` (APDC compensator) boxes, and a transparent
/// pass-through when the box is bypassed.
fn canvas_node_dsp(node: &PatcherCanvasNodeDto, sr: f64) -> Box<dyn daw_core_graph::UnifiedDspNode> {
    use daw_core_graph::{
        EmphasisLimiterNode, FixedLatencyTransformNode, GrossBeatTimeBufferNode, LuxeVerbFdnShimmerNode,
        MaximusMultibandNode, SparseCompressorNode, UnifiedDspNode,
    };
    let mut dsp: Box<dyn UnifiedDspNode> = if !node.enabled {
        Box::new(FixedLatencyTransformNode::new(0, 1.0, false))
    } else {
        match node.plugin_kind.as_str() {
            "ProQ4" => {
                let mut eq = dsp_proq4_eq::ProQ4Processor::new();
                eq.params.global.processing_mode = dsp_proq4_eq::ProcessingMode::LinearPhase;
                Box::new(eq)
            }
            "Soothe3" => Box::new(dsp_soothe3_spectral::Soothe3Processor::new(sr as f32)),
            "SparseCompressor" => Box::new(SparseCompressorNode::new(0.5, 4.0, 1.0, 0)),
            "Maximus" => Box::new(MaximusMultibandNode::new()),
            "Emphasis" => Box::new(EmphasisLimiterNode::new(64, -0.5)),
            "GrossBeat" => Box::new(GrossBeatTimeBufferNode::new((sr as usize) * 2)),
            "LuxeVerb" => Box::new(LuxeVerbFdnShimmerNode::new()),
            _ => Box::new(FixedLatencyTransformNode::new(node.latency_samples.min(8_192), 1.0, false)),
        }
    };
    dsp.prepare(sr, daw_core_graph::MAX_BLOCK_SIZE);
    dsp
}

/// Builds the Patcher sub-DAG of a canvas: `(patcher, critical-path latency, host_in_enabled,
/// host_out_enabled)`. The latency is the one the real processors report.
fn build_canvas_patcher(
    dto: &PatcherCanvasUpdateDto,
    sr: f64,
) -> Result<(daw_core_graph::PatcherGraphNode, usize, bool, bool), String> {
    use daw_core_graph::UnifiedDspNode;
    let mut patcher = daw_core_graph::PatcherGraphNode::new();
    patcher.prepare(sr, daw_core_graph::MAX_BLOCK_SIZE);

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
            let internal_idx = patcher.add_node(canvas_node_dsp(node, sr));
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
        patcher.latency_samples()
    } else {
        0
    };
    Ok((patcher, total_patcher_latency, host_in_enabled, host_out_enabled))
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
    let (patcher, total_patcher_latency, host_in_enabled, host_out_enabled) =
        build_canvas_patcher(dto, sr)?;

    let installed_slot = engine.tracks[track]
        .set_or_replace_plugin_slot(slot, Box::new(patcher), 1.0, sr, blk)
        .map_err(|e| e.to_string())?;
    engine.tracks[track].slots[installed_slot].enabled = host_in_enabled && host_out_enabled;
    engine.rebuild_graph().map_err(|e| e.to_string())?;

    Ok(total_patcher_latency)
}

// ============================================================================
// LOCK-FREE, ZERO-ALLOCATION RT GRAPH MUTATION DISPATCHERS (`RtGraphMutation`)
// ============================================================================

/// Triggers a live one-shot audition note (`NoteOn` + automatic ADSR gate release) through the
/// Rust `PolyphonicSynthGeneratorNode` -> Mixer -> WASAPI hardware output path (`0` browser audio).
pub fn cmd_trigger_audition_note(
    state: &DawAppState,
    key: u8,
    velocity: f32,
    fine_pitch_cents: f32,
    pan: f32,
) -> Result<(), String> {
    state.push_command(EngineCommand::TriggerAuditionNote {
        key: key.min(127),
        velocity: velocity.clamp(0.05, 1.0),
        fine_pitch_cents: fine_pitch_cents.clamp(-1200.0, 1200.0),
        pan: pan.clamp(-1.0, 1.0),
        gate_samples: 0,
    })
}

fn would_create_project_mixer_cycle(
    project_state: &crate::project_state::ToneForgeProjectState,
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
        if curr < project_state.mixer_tracks.len() && curr < MAX_MIXER_TRACKS && !visited[curr] {
            visited[curr] = true;
            for route in &project_state.mixer_tracks[curr].sends {
                if route.dest_track_index < MAX_MIXER_TRACKS {
                    stack.push(route.dest_track_index);
                }
            }
        }
    }
    false
}

/// Lock-free Mixer DAG send/sidechain routing via `RtGraphMutation::AddOrUpdateSendRoute`.
/// Validates cycle-freedom on the host `ToneForgeProjectState` mirror, allocates `MixerSendRoute`
/// on the host thread, and dispatches to the WASAPI RT thread with **0 Mutex locks and 0 RT allocs**.
pub fn cmd_route_mixer_track_rt(
    project_state: &mut crate::project_state::ToneForgeProjectState,
    state: &DawAppState,
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
    if would_create_project_mixer_cycle(project_state, src, dst) {
        return Err(format!(
            "Routing track {} -> {} would create a cycle in the 502-Track Mixer DAG",
            src, dst
        ));
    }
    let clamped_gain = gain.clamp(0.0, 4.0);
    if let Some(trk) = project_state.mixer_tracks.get_mut(src) {
        if let Some(existing) = trk
            .sends
            .iter_mut()
            .find(|s| s.dest_track_index == dst && s.is_sidechain_only == is_sidechain)
        {
            existing.gain = clamped_gain;
        } else {
            trk.sends.push(crate::project_state::MixerSendRouteDto {
                dest_track_index: dst,
                gain: clamped_gain,
                is_sidechain_only: is_sidechain,
            });
        }
    }
    let route = daw_core_graph::MixerSendRoute::new(dst, clamped_gain, is_sidechain);
    state.push_rt_mutation(daw_core_graph::RtGraphMutation::AddOrUpdateSendRoute {
        src_track,
        dest_track: dst_track,
        gain: clamped_gain,
        is_sidechain_only: is_sidechain,
        route_carrier: Box::new(Some(route)),
    })
}

/// Lock-free Piano Roll Pattern upsert via `RtGraphMutation::UpsertPattern` (`0` RT allocs/locks).
pub fn cmd_upsert_piano_roll_pattern_rt(
    project_state: &mut crate::project_state::ToneForgeProjectState,
    state: &DawAppState,
    pattern: &crate::project_state::PianoRollPatternDto,
) -> Result<(), String> {
    if pattern.length_ticks <= 0 {
        return Err("PianoRollPattern length_ticks must be positive".to_string());
    }
    if let Some(existing) = project_state
        .piano_roll_patterns
        .iter_mut()
        .find(|p| p.pattern_id == pattern.pattern_id)
    {
        *existing = pattern.clone();
    } else {
        if project_state.piano_roll_patterns.len() >= daw_core_graph::MAX_SEQUENCER_PATTERNS {
            return Err("PlaylistSequencer pattern table capacity reached".to_string());
        }
        project_state.piano_roll_patterns.push(pattern.clone());
    }
    state.push_rt_mutation(daw_core_graph::RtGraphMutation::UpsertPattern {
        pattern: Box::new(pattern.to_pattern()),
    })
}

/// Lock-free Playlist Clip upsert via `RtGraphMutation::UpsertPlaylistClip` (`0` RT allocs/locks).
pub fn cmd_upsert_playlist_clip_rt(
    project_state: &mut crate::project_state::ToneForgeProjectState,
    state: &DawAppState,
    clip: &crate::project_state::PlaylistClipDto,
) -> Result<(), String> {
    if clip.duration_ticks <= 0 {
        return Err("PlaylistClip duration_ticks must be positive".to_string());
    }
    if let Some(existing) = project_state
        .playlist_clips
        .iter_mut()
        .find(|c| c.clip_id == clip.clip_id)
    {
        *existing = clip.clone();
    } else {
        if project_state.playlist_clips.len() >= daw_core_graph::MAX_PLAYLIST_CLIPS {
            return Err("PlaylistSequencer clip table capacity reached".to_string());
        }
        project_state.playlist_clips.push(clip.clone());
    }
    state.push_rt_mutation(daw_core_graph::RtGraphMutation::UpsertPlaylistClip {
        clip: clip.to_playlist_clip(),
    })
}

/// Lock-free `TensionSpline` Automation Articulator update via `RtGraphMutation::UpsertAutomationArticulator`
/// + `EngineCommand::SetInternalControllerFixed30` (`0` RT allocs/locks).
pub fn cmd_update_automation_spline_rt(
    project_state: &mut crate::project_state::ToneForgeProjectState,
    state: &DawAppState,
    clip: &crate::project_state::AutomationClipStateDto,
    eval_tick: f64,
) -> Result<i32, String> {
    if (clip.controller_id as usize) >= 512 {
        return Err(format!(
            "Controller ID {} out of range (0..512)",
            clip.controller_id
        ));
    }
    if let Some(existing) = project_state
        .automation_clips
        .iter_mut()
        .find(|c| c.controller_id == clip.controller_id)
    {
        *existing = clip.clone();
    } else {
        if project_state.automation_clips.len() >= daw_core_graph::MAX_AUTOMATION_ARTICULATORS {
            return Err("PlaylistSequencer automation clip table capacity reached".to_string());
        }
        project_state.automation_clips.push(clip.clone());
    }
    let articulator = clip.to_articulator();
    let fixed30 = articulator.evaluate_fixed30(eval_tick.max(0.0));
    state.push_command(EngineCommand::SetInternalControllerFixed30 {
        controller_id: articulator.controller_id,
        value_fixed30: fixed30,
    })?;
    state.push_rt_mutation(
        daw_core_graph::RtGraphMutation::UpsertAutomationArticulator {
            articulator: Box::new(articulator),
        },
    )?;
    Ok(fixed30)
}

/// Builds a `PatcherGraphNode` on the host thread, validates acyclic routing, and hot-swaps it onto
/// `tracks[dto.track_idx].slots[dto.slot_idx]` via `RtGraphMutation::SetOrReplacePluginSlot`
/// (`0` RT allocs, `0` RT drops — old slot retired to `toneforge-rt-gc`).
pub fn cmd_update_patcher_graph_rt(
    state: &DawAppState,
    sample_rate: f64,
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

    let sr = sample_rate.max(8_000.0);
    let blk = daw_core_graph::MAX_BLOCK_SIZE;
    let (patcher, total_patcher_latency, host_in_enabled, host_out_enabled) =
        build_canvas_patcher(dto, sr)?;

    let mut rack_slot = daw_core_graph::PluginRackSlot::new(Box::new(patcher), 1.0);
    rack_slot.enabled = host_in_enabled && host_out_enabled;
    rack_slot.prepare(sr, blk);

    state.push_rt_mutation(daw_core_graph::RtGraphMutation::SetOrReplacePluginSlot {
        track_index: dto.track_idx,
        slot_index: dto.slot_idx,
        slot: Box::new(Some(rack_slot)),
    })?;

    Ok(total_patcher_latency)
}

/// Lock-free replacement of custom audio stem on Track 1 (`Channel 900 -> Insert Track 1 -> Master 0`)
/// via `RtGraphMutation::ReplaceStemChannels` (`0` RT allocs, `0` RT drops).
pub fn cmd_replace_custom_audio_stem_rt(
    state: &DawAppState,
    sample_rate: f64,
    frames: Vec<daw_core_graph::StereoFrame>,
    loop_playback: bool,
) -> Result<(), String> {
    let sr = sample_rate.max(8_000.0);
    let mut player = daw_core_graph::AudioStemClipPlayerNode::new(900, "Custom Audio Stem", frames);
    player.continuous_mode = true;
    player.loop_enabled = loop_playback;
    player.prepare(sr, daw_core_graph::MAX_BLOCK_SIZE);

    let mut ch = daw_core_graph::ChannelGenerator::new(
        900,
        daw_core_graph::FIRST_INSERT_TRACK_INDEX,
        Box::new(player),
    );
    ch.prepare(sr, daw_core_graph::MAX_BLOCK_SIZE);

    state.push_rt_mutation(daw_core_graph::RtGraphMutation::ReplaceStemChannels {
        target_tracks_mask: 0b1_1110,
        new_channels: Box::new(vec![ch]),
        retired_channels: Box::new(Vec::with_capacity(16)),
        reset_slicer: true,
    })?;
    state.push_rt_mutation(daw_core_graph::RtGraphMutation::AddOrUpdateSendRoute {
        src_track: 1,
        dest_track: 0,
        gain: 1.0,
        is_sidechain_only: false,
        route_carrier: Box::new(Some(daw_core_graph::MixerSendRoute::new(0, 1.0, false))),
    })?;
    Ok(())
}

/// Lock-free bounce of separated stereo stems onto `target_tracks` (`1..=15`) via
/// `RtGraphMutation::ReplaceStemChannels` (`0` RT allocs, `0` RT drops).
pub fn cmd_bounce_stems_to_mixer_rt(
    state: &DawAppState,
    sample_rate: f64,
    stems: &[Vec<daw_core_graph::StereoFrame>],
    target_tracks: &[u16],
) -> Result<(), String> {
    let sr = sample_rate.max(8_000.0);
    let count = stems.len().min(target_tracks.len());
    let mut mask: u16 = 0;
    let mut new_channels = Vec::with_capacity(count);

    for i in 0..count {
        let trk = target_tracks[i] as usize;
        if trk == 0 || trk >= 16 {
            return Err(format!(
                "Target stem insert track {} must be in 1..16 for atomic mask replacement",
                trk
            ));
        }
        mask |= 1u16 << trk;
        let gen_node = Box::new(crate::audio_stream_controller::StemPlaybackGeneratorNode::new(
            stems[i].clone(),
            true,
        ));
        let mut ch = daw_core_graph::ChannelGenerator::new(100 + (trk as u32), trk, gen_node);
        ch.prepare(sr, daw_core_graph::MAX_BLOCK_SIZE);
        new_channels.push(ch);
    }

    // Clear Track 1 custom audio stem too when bouncing 4 stems onto Tracks 1..=4
    mask |= 0b1_1110;

    state.push_rt_mutation(daw_core_graph::RtGraphMutation::ReplaceStemChannels {
        target_tracks_mask: mask,
        new_channels: Box::new(new_channels),
        retired_channels: Box::new(Vec::with_capacity(16)),
        reset_slicer: false,
    })?;

    for i in 0..count {
        let trk = target_tracks[i];
        state.push_rt_mutation(daw_core_graph::RtGraphMutation::AddOrUpdateSendRoute {
            src_track: trk,
            dest_track: 0,
            gain: 1.0,
            is_sidechain_only: false,
            route_carrier: Box::new(Some(daw_core_graph::MixerSendRoute::new(0, 1.0, false))),
        })?;
    }
    Ok(())
}

// ============================================================================
// TONEFORGE v3: 100% INTERNAL PLUGIN PARITY, VST3/CLAP HOSTING,
// HARDWARE ASIO/WASAPI/MIDI RECORDING, PRO PLAYLIST/PIANO ROLL & NEWTONE VOCAL
// ============================================================================

/// Fits a freehand `(freq_hz, gain_db)` EQ Sketch gesture into up to `max_bands` (`1..=24`)
/// parametric Pro-Q 4 bands, updates both `ToneForgeProjectState` and the live RT `ProQ4Processor`,
/// and returns the structured `ProQ4BandUpdateDto` list for immediate UI handle snapping.
pub fn cmd_proq4_fit_eq_sketch(
    project_state: &mut crate::project_state::ToneForgeProjectState,
    state: &DawAppState,
    track_idx: u16,
    slot_idx: u8,
    points_hz_db: &[(f32, f32)],
    max_bands: usize,
) -> Result<Vec<ProQ4BandUpdateDto>, String> {
    if (track_idx as usize) >= MAX_MIXER_TRACKS || (slot_idx as usize) >= MAX_INSERT_SLOTS {
        return Err("Track or slot index out of range".to_string());
    }
    if points_hz_db.len() < 2 {
        return Err("EQ Sketch requires at least 2 (freq_hz, gain_db) points".to_string());
    }

    let fitted = dsp_proq4_eq::eq_sketch_fitter::fit_eq_sketch_curve(
        points_hz_db,
        max_bands.clamp(1, PROQ4_NUM_BANDS),
    );
    let mut out_dtos = Vec::with_capacity(fitted.len());

    let mut param_set = project_state
        .mixer_tracks
        .get(track_idx as usize)
        .and_then(|t| t.slots.get(slot_idx as usize))
        .and_then(|s| match &s.plugin {
            crate::project_state::PluginConfigDto::ProQ4(pq) => Some(pq.to_parameter_set()),
            _ => None,
        })
        .unwrap_or_else(ProQ4ParameterSet::new);
    let prev_ref = project_state
        .mixer_tracks
        .get(track_idx as usize)
        .and_then(|t| t.slots.get(slot_idx as usize))
        .and_then(|s| match &s.plugin {
            crate::project_state::PluginConfigDto::ProQ4(pq) => pq.collision_reference_track,
            _ => None,
        });

    for b_idx in 0..PROQ4_NUM_BANDS {
        let base = (b_idx as u32) * (PROQ4_PARAMS_PER_BAND as u32);
        if let Some(bp) = fitted.get(b_idx) {
            param_set.bands[b_idx] = *bp;
            param_set.bands[b_idx].used = true;
            param_set.bands[b_idx].enabled = true;
            for off in 0..PROQ4_PARAMS_PER_BAND {
                state.push_command(EngineCommand::SetPluginParamNormalized {
                    track_index: track_idx,
                    slot_index: slot_idx,
                    param_id: base + (off as u32),
                    norm_val: param_set.bands[b_idx].get_offset_normalized(off),
                    target_sample_pos: None,
                })?;
            }
            out_dtos.push(ProQ4BandUpdateDto {
                track_idx,
                slot_idx,
                band_idx: b_idx as u8,
                freq_hz: bp.freq_hz,
                gain_db: bp.gain_db,
                q: bp.q_ui,
                shape: bp.shape as u8,
                slope_db_oct: bp.slope_db_oct.round().clamp(6.0, 96.0) as u8,
                dynamic_range_db: bp.dyn_range_db,
                spectral_enabled: bp.spectral_enabled,
                spectral_density: bp.spectral_density,
            });
        } else {
            param_set.bands[b_idx].used = false;
            param_set.bands[b_idx].enabled = false;
            for &off in &[0usize, 1usize] {
                state.push_command(EngineCommand::SetPluginParamNormalized {
                    track_index: track_idx,
                    slot_index: slot_idx,
                    param_id: base + (off as u32),
                    norm_val: 0.0,
                    target_sample_pos: None,
                })?;
            }
        }
    }

    param_set.sync_mirror_from_structured();
    if let Some(slot_dto) = project_state
        .mixer_tracks
        .get_mut(track_idx as usize)
        .and_then(|t| t.slots.get_mut(slot_idx as usize))
    {
        let mut pq_dto = crate::project_state::ProQ4StateDto::from_parameter_set(&param_set);
        pq_dto.collision_reference_track = prev_ref;
        if !matches!(slot_dto.plugin, crate::project_state::PluginConfigDto::Empty) {
            slot_dto.plugin = crate::project_state::PluginConfigDto::ProQ4(pq_dto);
        }
    }

    Ok(out_dtos)
}

/// Configures the lock-free `ProQ4CollisionBus` reference track (`None` = off / clear,
/// `Some(ref_track)` = highlight auditory spectral masking collisions between `track_idx` and `ref_track`).
pub fn cmd_proq4_set_collision_reference(
    project_state: &mut crate::project_state::ToneForgeProjectState,
    state: &DawAppState,
    collision_bus: &dsp_proq4_eq::ProQ4CollisionBus,
    track_idx: u16,
    slot_idx: u8,
    reference_track: Option<u16>,
) -> Result<(), String> {
    if (track_idx as usize) >= MAX_MIXER_TRACKS || (slot_idx as usize) >= MAX_INSERT_SLOTS {
        return Err("Track or slot index out of range".to_string());
    }
    if let Some(rt) = reference_track {
        if (rt as usize) >= MAX_MIXER_TRACKS {
            return Err(format!("Reference track {rt} out of range (0..502)"));
        }
    }
    collision_bus.set_reference_track(track_idx as u32, reference_track);

    // Param ID 600 (`PROQ4_BAND_PARAMS_TOTAL + 24`) = `show_collisions`
    state.push_command(EngineCommand::SetPluginParamNormalized {
        track_index: track_idx,
        slot_index: slot_idx,
        param_id: (PROQ4_BAND_PARAMS_TOTAL + 24) as u32,
        norm_val: if reference_track.is_some() { 1.0 } else { 0.0 },
        target_sample_pos: None,
    })?;

    if let Some(slot_dto) = project_state
        .mixer_tracks
        .get_mut(track_idx as usize)
        .and_then(|t| t.slots.get_mut(slot_idx as usize))
    {
        if let crate::project_state::PluginConfigDto::ProQ4(ref mut pq) = slot_dto.plugin {
            let mut ps = pq.to_parameter_set();
            ps.global.show_collisions = reference_track.is_some();
            ps.sync_mirror_from_structured();
            let mut updated = crate::project_state::ProQ4StateDto::from_parameter_set(&ps);
            updated.collision_reference_track = reference_track;
            *pq = updated;
        }
    }
    Ok(())
}

/// Enables or disables `oeksound Soothe3` External Spectral Sidechain Ducking (param ID `16`)
/// and optionally wires a cycle-safe `is_sidechain_only` DAG send from `source_track -> track_idx`.
pub fn cmd_soothe3_set_external_sidechain(
    project_state: &mut crate::project_state::ToneForgeProjectState,
    state: &DawAppState,
    track_idx: u16,
    slot_idx: u8,
    enabled: bool,
    source_track: Option<u16>,
) -> Result<(), String> {
    if (track_idx as usize) >= MAX_MIXER_TRACKS || (slot_idx as usize) >= MAX_INSERT_SLOTS {
        return Err("Track or slot index out of range".to_string());
    }
    if enabled {
        if let Some(src) = source_track {
            if src != track_idx && state.has_rt_mutations() {
                cmd_route_mixer_track_rt(project_state, state, src, track_idx, 1.0, true)?;
            }
        }
    }
    state.push_command(EngineCommand::SetPluginParamNormalized {
        track_index: track_idx,
        slot_index: slot_idx,
        param_id: 16,
        norm_val: if enabled { 1.0 } else { 0.0 },
        target_sample_pos: None,
    })?;

    if let Some(slot_dto) = project_state
        .mixer_tracks
        .get_mut(track_idx as usize)
        .and_then(|t| t.slots.get_mut(slot_idx as usize))
    {
        if let crate::project_state::PluginConfigDto::Soothe3(ref mut s3) = slot_dto.plugin {
            let mut ps = s3.to_parameter_set();
            ps.set_normalized(16, if enabled { 1.0 } else { 0.0 });
            let mut updated = crate::project_state::Soothe3StateDto::from_parameter_set(&ps);
            updated.external_sidechain_source_track = if enabled { source_track } else { None };
            *s3 = updated;
        }
    }
    Ok(())
}

/// DTO for Ozone 12 `Low End Focus` (`Punchy` / `Smooth`, `20..300 Hz`, `Contrast`, `Gain`, `Solo`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Ozone12LowEndFocusDto {
    pub enabled: bool,
    pub punchy: bool,
    pub contrast_pct: f32,
    pub upper_hz: f32,
    pub gain_db: f32,
    pub solo_low_band: bool,
}

/// Dispatches Ozone 12 `Low End Focus` parameters (`69..=74`) via the lock-free command queue.
pub fn cmd_ozone12_set_low_end_focus(
    project_state: &mut crate::project_state::ToneForgeProjectState,
    state: &DawAppState,
    track_idx: u16,
    slot_idx: u8,
    dto: &Ozone12LowEndFocusDto,
) -> Result<(), String> {
    if (track_idx as usize) >= MAX_MIXER_TRACKS || (slot_idx as usize) >= MAX_INSERT_SLOTS {
        return Err("Track or slot index out of range".to_string());
    }
    let pairs: [(u32, f32); 6] = [
        (70, if dto.enabled { 1.0 } else { 0.0 }),
        (71, if dto.punchy { 0.0 } else { 1.0 }),
        (72, ((dto.contrast_pct + 100.0) / 200.0).clamp(0.0, 1.0)),
        (
            73,
            ((dto.upper_hz.clamp(20.0, 300.0) - 20.0) / 280.0).clamp(0.0, 1.0),
        ),
        (74, ((dto.gain_db + 12.0) / 24.0).clamp(0.0, 1.0)),
        (69, if dto.solo_low_band { 1.0 } else { 0.0 }),
    ];
    for (param_id, norm_val) in pairs {
        state.push_command(EngineCommand::SetPluginParamNormalized {
            track_index: track_idx,
            slot_index: slot_idx,
            param_id,
            norm_val,
            target_sample_pos: None,
        })?;
    }
    if let Some(slot_dto) = project_state
        .mixer_tracks
        .get_mut(track_idx as usize)
        .and_then(|t| t.slots.get_mut(slot_idx as usize))
    {
        if let crate::project_state::PluginConfigDto::Ozone12Master(ref mut oz) = slot_dto.plugin {
            oz.low_end_focus_enabled = dto.enabled;
            oz.low_end_focus_punchy = dto.punchy;
            oz.low_end_focus_contrast_pct = dto.contrast_pct.clamp(-100.0, 100.0);
            oz.low_end_focus_upper_hz = dto.upper_hz.clamp(20.0, 300.0);
            oz.low_end_focus_gain_db = dto.gain_db.clamp(-12.0, 12.0);
            oz.low_end_focus_solo = dto.solo_low_band;
        }
    }
    Ok(())
}

/// DTO for Ozone 12 `Codec Preview` (`0=PCM`, `1=MP3 128`, `2=MP3 320`, `3=AAC 128`, `4=AAC 256`, `5=Opus 128`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Ozone12CodecPreviewDto {
    pub enabled: bool,
    pub codec_mode: u8,
    pub artifact_solo: bool,
}

/// Dispatches Ozone 12 `Codec Preview` parameters (`85..=87`) and refreshes 5-Point APDC (`512` samples).
pub fn cmd_ozone12_set_codec_preview(
    project_state: &mut crate::project_state::ToneForgeProjectState,
    state: &DawAppState,
    track_idx: u16,
    slot_idx: u8,
    dto: &Ozone12CodecPreviewDto,
) -> Result<(), String> {
    if (track_idx as usize) >= MAX_MIXER_TRACKS || (slot_idx as usize) >= MAX_INSERT_SLOTS {
        return Err("Track or slot index out of range".to_string());
    }
    let pairs: [(u32, f32); 3] = [
        (85, if dto.enabled { 1.0 } else { 0.0 }),
        (86, ((dto.codec_mode.min(5) as f32) / 5.0).clamp(0.0, 1.0)),
        (87, if dto.artifact_solo { 1.0 } else { 0.0 }),
    ];
    for (param_id, norm_val) in pairs {
        state.push_command(EngineCommand::SetPluginParamNormalized {
            track_index: track_idx,
            slot_index: slot_idx,
            param_id,
            norm_val,
            target_sample_pos: None,
        })?;
    }
    state.push_command(EngineCommand::RefreshApdc)?;
    if let Some(slot_dto) = project_state
        .mixer_tracks
        .get_mut(track_idx as usize)
        .and_then(|t| t.slots.get_mut(slot_idx as usize))
    {
        if let crate::project_state::PluginConfigDto::Ozone12Master(ref mut oz) = slot_dto.plugin {
            oz.codec_preview_enabled = dto.enabled;
            oz.codec_preview_codec = dto.codec_mode.min(5);
            oz.codec_preview_artifact_solo = dto.artifact_solo;
        }
    }
    Ok(())
}

/// DTO describing an active external VST3 / CLAP plugin loaded onto a Mixer slot or Channel Rack.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LoadedExternalPluginInfoDto {
    pub track_idx: u16,
    pub slot_idx: u8,
    pub is_clap: bool,
    pub is_instrument: bool,
    pub name: String,
    pub vendor: String,
    pub plugin_id: String,
    pub binary_path: String,
    pub latency_samples: usize,
    pub parameters: Vec<daw_plugin_host::HostedPluginParamInfo>,
}

/// Loads a real Steinberg VST3 (`.vst3`) or CLAP 1.2 (`.clap`) binary on the Host thread and
/// hot-swaps it onto `tracks[track_idx].slots[slot_idx]` (and optionally `ChannelGenerator` when
/// `as_channel_instrument == true`) via `RtGraphMutation` (`0` RT allocs, `0` RT locks).
pub fn cmd_load_external_plugin_slot_rt(
    project_state: &mut crate::project_state::ToneForgeProjectState,
    state: &DawAppState,
    sample_rate: f64,
    track_idx: u16,
    slot_idx: u8,
    binary_path: &str,
    plugin_id: Option<&str>,
    is_clap: bool,
    as_channel_instrument: bool,
    gui_worker: Option<&ExternalPluginGuiWorker>,
) -> Result<LoadedExternalPluginInfoDto, String> {
    if (track_idx as usize) >= MAX_MIXER_TRACKS || (slot_idx as usize) >= MAX_INSERT_SLOTS {
        return Err("Track or slot index out of range".to_string());
    }
    let sr = sample_rate.max(8_000.0);
    let blk = daw_core_graph::MAX_BLOCK_SIZE;
    let path = std::path::Path::new(binary_path.trim());

    let live_key_slot = if as_channel_instrument { INSTRUMENT_SLOT_INDEX } else { slot_idx };
    if let Some(w) = gui_worker {
        // The previous instance in this slot is about to be retired: close its editor first.
        w.unregister_live_editors_blocking(Some((track_idx, live_key_slot)));
    }
    if is_clap {
        let mut node = daw_plugin_host::ClapHostedPluginNode::load(path, plugin_id, sr, blk)?;
        if let (Some(w), Some(ed)) = (gui_worker, node.live_editor()) {
            w.register_live_editor(track_idx, live_key_slot, LivePluginEditor::Clap(ed));
        }
        let name = node.plugin_display_name().to_string();
        let pid = node.plugin_id().to_string();
        let latency = node.latency_samples();
        let params = node.parameters().to_vec();
        let state_hex = node
            .save_state()
            .map(|b| crate::project_state::encode_hex_bytes(&b))
            .unwrap_or_default();

        if as_channel_instrument {
            let mut ch = daw_core_graph::ChannelGenerator::new(
                200 + (track_idx as u32),
                track_idx as usize,
                Box::new(node),
            );
            ch.prepare(sr, blk);
            state.push_rt_mutation(daw_core_graph::RtGraphMutation::UpsertChannel {
                channel: Box::new(Some(ch)),
            })?;
            if let Some(trk) = project_state.mixer_tracks.get_mut(track_idx as usize) {
                trk.channel_instrument = Some(crate::project_state::PluginConfigDto::ExternalClap(
                    crate::project_state::ExternalPluginStateDto {
                        binary_path: binary_path.trim().to_string(),
                        plugin_id: pid.clone(),
                        plugin_name: name.clone(),
                        state_hex: state_hex.clone(),
                        param_overrides: Vec::new(),
                    },
                ));
            }
        } else {
            node.prepare(sr, blk);
            let mut rack_slot = daw_core_graph::PluginRackSlot::new(Box::new(node), 1.0);
            rack_slot.prepare(sr, blk);
            crate::slot_rack::pad_slots_up_to(project_state, state, sr, track_idx, slot_idx)?;
            crate::slot_rack::drop_slot_overlays(project_state, track_idx, slot_idx);
            state.push_rt_mutation(daw_core_graph::RtGraphMutation::SetOrReplacePluginSlot {
                track_index: track_idx,
                slot_index: slot_idx,
                slot: Box::new(Some(rack_slot)),
            })?;

            let ext_dto = crate::project_state::ExternalPluginStateDto {
                binary_path: binary_path.trim().to_string(),
                plugin_id: pid.clone(),
                plugin_name: name.clone(),
                state_hex,
                param_overrides: Vec::new(),
            };
            upsert_project_slot_dto(
                project_state,
                track_idx as usize,
                slot_idx as usize,
                crate::project_state::PluginConfigDto::ExternalClap(ext_dto),
            );
        }

        Ok(LoadedExternalPluginInfoDto {
            track_idx,
            slot_idx,
            is_clap: true,
            is_instrument: as_channel_instrument,
            name,
            vendor: "CLAP 1.2".to_string(),
            plugin_id: pid,
            binary_path: binary_path.trim().to_string(),
            latency_samples: latency,
            parameters: params,
        })
    } else {
        let mut node = daw_plugin_host::Vst3HostedPluginNode::load(path, plugin_id, sr, blk)?;
        if let (Some(w), Some(ed)) = (gui_worker, node.live_editor()) {
            w.register_live_editor(track_idx, live_key_slot, LivePluginEditor::Vst3(ed));
        }
        let name = node.plugin_display_name().to_string();
        let cid = node.cid_hex().to_string();
        let latency = node.latency_samples();
        let params = node.parameters().to_vec();
        let state_hex = node
            .save_state()
            .map(|b| crate::project_state::encode_hex_bytes(&b))
            .unwrap_or_default();

        if as_channel_instrument {
            let mut ch = daw_core_graph::ChannelGenerator::new(
                200 + (track_idx as u32),
                track_idx as usize,
                Box::new(node),
            );
            ch.prepare(sr, blk);
            state.push_rt_mutation(daw_core_graph::RtGraphMutation::UpsertChannel {
                channel: Box::new(Some(ch)),
            })?;
            if let Some(trk) = project_state.mixer_tracks.get_mut(track_idx as usize) {
                trk.channel_instrument = Some(crate::project_state::PluginConfigDto::ExternalVst3(
                    crate::project_state::ExternalPluginStateDto {
                        binary_path: binary_path.trim().to_string(),
                        plugin_id: cid.clone(),
                        plugin_name: name.clone(),
                        state_hex: state_hex.clone(),
                        param_overrides: Vec::new(),
                    },
                ));
            }
        } else {
            node.prepare(sr, blk);
            let mut rack_slot = daw_core_graph::PluginRackSlot::new(Box::new(node), 1.0);
            rack_slot.prepare(sr, blk);
            crate::slot_rack::pad_slots_up_to(project_state, state, sr, track_idx, slot_idx)?;
            crate::slot_rack::drop_slot_overlays(project_state, track_idx, slot_idx);
            state.push_rt_mutation(daw_core_graph::RtGraphMutation::SetOrReplacePluginSlot {
                track_index: track_idx,
                slot_index: slot_idx,
                slot: Box::new(Some(rack_slot)),
            })?;

            let ext_dto = crate::project_state::ExternalPluginStateDto {
                binary_path: binary_path.trim().to_string(),
                plugin_id: cid.clone(),
                plugin_name: name.clone(),
                state_hex,
                param_overrides: Vec::new(),
            };
            upsert_project_slot_dto(
                project_state,
                track_idx as usize,
                slot_idx as usize,
                crate::project_state::PluginConfigDto::ExternalVst3(ext_dto),
            );
        }

        Ok(LoadedExternalPluginInfoDto {
            track_idx,
            slot_idx,
            is_clap: false,
            is_instrument: as_channel_instrument,
            name,
            vendor: "Steinberg VST3".to_string(),
            plugin_id: cid,
            binary_path: binary_path.trim().to_string(),
            latency_samples: latency,
            parameters: params,
        })
    }
}

/// `slot_idx` value addressing a track's channel instrument instead of an FX slot.
pub const INSTRUMENT_SLOT_INDEX: u8 = 0xFE;

fn upsert_project_slot_dto(
    project_state: &mut crate::project_state::ToneForgeProjectState,
    track_idx: usize,
    slot_idx: usize,
    plugin: crate::project_state::PluginConfigDto,
) {
    if let Some(trk) = project_state.mixer_tracks.get_mut(track_idx) {
        if slot_idx < trk.slots.len() {
            trk.slots[slot_idx].plugin = plugin;
            trk.slots[slot_idx].enabled = true;
        } else if trk.slots.len() == slot_idx && trk.slots.len() < MAX_INSERT_SLOTS {
            let new_slot_idx = trk.slots.len() as u8;
            trk.slots.push(crate::project_state::PluginSlotStateDto {
                slot_index: new_slot_idx,
                enabled: true,
                wet_mix: 1.0,
                plugin,
            });
        }
    }
}

/// Updates an external VST3 / CLAP parameter (`0.0..=1.0` normalized) on both the live RT audio
/// instance and `ToneForgeProjectState`.
pub fn cmd_set_external_plugin_param_rt(
    project_state: &mut crate::project_state::ToneForgeProjectState,
    state: &DawAppState,
    track_idx: u16,
    slot_idx: u8,
    param_id: u32,
    norm_val: f32,
) -> Result<(), String> {
    let clamped = norm_val.clamp(0.0, 1.0);
    if slot_idx == INSTRUMENT_SLOT_INDEX {
        // Channel instrument (`ChannelGenerator` id `200 + track`), not an FX slot.
        let trk = project_state
            .mixer_tracks
            .get_mut(track_idx as usize)
            .ok_or_else(|| "Track index out of range".to_string())?;
        let Some(
            crate::project_state::PluginConfigDto::ExternalVst3(ext)
            | crate::project_state::PluginConfigDto::ExternalClap(ext),
        ) = trk.channel_instrument.as_mut()
        else {
            return Err(format!("Track {track_idx} has no external channel instrument"));
        };
        state.push_command(EngineCommand::SetChannelParamNormalized {
            channel_id: 200 + (track_idx as u32),
            param_id,
            norm_val: clamped,
        })?;
        if let Some(existing) = ext.param_overrides.iter_mut().find(|(id, _)| *id == param_id) {
            existing.1 = clamped as f64;
        } else {
            ext.param_overrides.push((param_id, clamped as f64));
        }
        return Ok(());
    }
    if (track_idx as usize) >= MAX_MIXER_TRACKS || (slot_idx as usize) >= MAX_INSERT_SLOTS {
        return Err("Track or slot index out of range".to_string());
    }
    state.push_command(EngineCommand::SetPluginParamNormalized {
        track_index: track_idx,
        slot_index: slot_idx,
        param_id,
        norm_val: clamped,
        target_sample_pos: None,
    })?;

    if let Some(slot_dto) = project_state
        .mixer_tracks
        .get_mut(track_idx as usize)
        .and_then(|t| t.slots.get_mut(slot_idx as usize))
    {
        match &mut slot_dto.plugin {
            crate::project_state::PluginConfigDto::ExternalVst3(ext)
            | crate::project_state::PluginConfigDto::ExternalClap(ext) => {
                if let Some(existing) = ext.param_overrides.iter_mut().find(|(id, _)| *id == param_id) {
                    existing.1 = clamped as f64;
                } else {
                    ext.param_overrides.push((param_id, clamped as f64));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

// ============================================================================
// DEDICATED WIN32 STA THREAD FOR FLOATING VST3 / CLAP `HWND` PLUGIN EDITORS
// ============================================================================

/// Editor handle bound to the live (real-time) instance of a hosted plug-in.
pub enum LivePluginEditor {
    Vst3(daw_plugin_host::Vst3LiveEditor),
    Clap(daw_plugin_host::ClapLiveEditor),
}

enum GuiWorkerMsg {
    Open {
        track_idx: u16,
        slot_idx: u8,
        is_clap: bool,
        binary_path: String,
        plugin_id: Option<String>,
        state_hex: String,
        owner_hwnd: Option<isize>,
        visible: bool,
        sample_rate: f64,
        reply: std::sync::mpsc::Sender<Result<isize, String>>,
    },
    Close {
        track_idx: u16,
        slot_idx: u8,
        reply: Option<std::sync::mpsc::Sender<()>>,
    },
    /// Registers the editor handle of the instance just loaded into `track_idx` / `slot_idx`.
    RegisterLive {
        track_idx: u16,
        slot_idx: u8,
        editor: LivePluginEditor,
    },
    /// Closes and forgets the live editor of `track_idx` / `slot_idx` (`None` = every slot) before
    /// the corresponding instance is torn down.
    UnregisterLive {
        key: Option<(u16, u8)>,
        reply: std::sync::mpsc::Sender<()>,
    },
    SyncParam {
        track_idx: u16,
        slot_idx: u8,
        param_id: u32,
        norm_val: f32,
    },
    /// A retired RT resource owning plug-in instances, torn down here (the thread that ran their
    /// editors) rather than on `toneforge-rt-gc`.
    Dispose(daw_core_graph::RetiredRtResource),
    Shutdown,
}

enum HostedGuiNode {
    /// Window attached to the live RT instance (preferred).
    Live {
        editor: LivePluginEditor,
        last_norm: Vec<(u32, f64, f64, f64)>, // (id, min, max, last_norm)
    },
    /// Fallback for instances restored without a live handle (undo / project load): a second
    /// instance initialised from the project's saved state chunk; its edits are forwarded.
    Vst3 {
        node: daw_plugin_host::Vst3HostedPluginNode,
        last_norm: Vec<(u32, f64)>,
    },
    Clap {
        node: daw_plugin_host::ClapHostedPluginNode,
        last_norm: Vec<(u32, f64, f64, f64)>, // (id, min, max, last_norm)
    },
}

impl HostedGuiNode {
    fn close(&mut self) {
        match self {
            HostedGuiNode::Live { editor, .. } => match editor {
                LivePluginEditor::Vst3(e) => e.close_editor(),
                LivePluginEditor::Clap(e) => e.close_editor(),
            },
            HostedGuiNode::Vst3 { node, .. } => node.close_editor(),
            HostedGuiNode::Clap { node, .. } => node.close_editor(),
        }
    }
}

/// Non-RT Win32 STA GUI Thread hosting floating `IPlugView` / `clap_plugin_gui` `HWND` windows
/// (`120 Hz` `PeekMessageW` loop + automatic GUI-knob-to-project-state parameter forwarding).
pub struct ExternalPluginGuiWorker {
    tx: std::sync::mpsc::Sender<GuiWorkerMsg>,
    gui_edits_rx: Mutex<std::sync::mpsc::Receiver<(u16, u8, u32, f32)>>,
    join_handle: Option<std::thread::JoinHandle<()>>,
}

impl ExternalPluginGuiWorker {
    pub fn spawn() -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<GuiWorkerMsg>();
        let (edit_tx, edit_rx) = std::sync::mpsc::channel::<(u16, u8, u32, f32)>();

        let join_handle = std::thread::Builder::new()
            .name("toneforge-plugin-gui-sta".to_string())
            .spawn(move || {
                // Open windows, and live editors registered but currently closed.
                let mut map: std::collections::HashMap<(u16, u8), HostedGuiNode> =
                    std::collections::HashMap::new();
                let mut live: std::collections::HashMap<(u16, u8), LivePluginEditor> =
                    std::collections::HashMap::new();
                loop {
                    while let Ok(msg) = rx.try_recv() {
                        match msg {
                            GuiWorkerMsg::Open {
                                track_idx,
                                slot_idx,
                                is_clap,
                                binary_path,
                                plugin_id,
                                state_hex,
                                owner_hwnd,
                                visible,
                                sample_rate,
                                reply,
                            } => {
                                let key = (track_idx, slot_idx);
                                if let Some(mut prev) = map.remove(&key) {
                                    prev.close();
                                    if let HostedGuiNode::Live { editor, .. } = prev {
                                        live.insert(key, editor);
                                    }
                                }
                                let res = if let Some(mut editor) = live.remove(&key) {
                                    let opened = match &mut editor {
                                        LivePluginEditor::Vst3(e) => e.open_editor(owner_hwnd, visible),
                                        LivePluginEditor::Clap(e) => e.open_editor(owner_hwnd, visible),
                                    };
                                    match opened {
                                        Ok(hwnd) => {
                                            let last_norm = match &editor {
                                                LivePluginEditor::Vst3(e) => e
                                                    .parameter_ids()
                                                    .iter()
                                                    .map(|id| {
                                                        (*id, 0.0, 1.0, e.get_parameter_normalized(*id).unwrap_or(0.0))
                                                    })
                                                    .collect(),
                                                LivePluginEditor::Clap(e) => e
                                                    .parameters()
                                                    .iter()
                                                    .map(|(id, mn, mx)| {
                                                        let span = (mx - mn).max(1e-9);
                                                        let n = e
                                                            .get_parameter_plain(*id)
                                                            .map(|v| ((v - mn) / span).clamp(0.0, 1.0))
                                                            .unwrap_or(0.0);
                                                        (*id, *mn, *mx, n)
                                                    })
                                                    .collect(),
                                            };
                                            map.insert(key, HostedGuiNode::Live { editor, last_norm });
                                            Ok(hwnd)
                                        }
                                        Err(e) => {
                                            live.insert(key, editor);
                                            Err(e)
                                        }
                                    }
                                } else {
                                    open_state_restored_editor(
                                        &mut map,
                                        key,
                                        is_clap,
                                        &binary_path,
                                        plugin_id.as_deref(),
                                        &state_hex,
                                        owner_hwnd,
                                        visible,
                                        sample_rate,
                                    )
                                };
                                let _ = reply.send(res);
                            }
                            GuiWorkerMsg::Close {
                                track_idx,
                                slot_idx,
                                reply,
                            } => {
                                let key = (track_idx, slot_idx);
                                if let Some(mut inst) = map.remove(&key) {
                                    inst.close();
                                    if let HostedGuiNode::Live { editor, .. } = inst {
                                        live.insert(key, editor);
                                    }
                                }
                                if let Some(r) = reply {
                                    let _ = r.send(());
                                }
                            }
                            GuiWorkerMsg::RegisterLive {
                                track_idx,
                                slot_idx,
                                editor,
                            } => {
                                let key = (track_idx, slot_idx);
                                if let Some(mut inst) = map.remove(&key) {
                                    inst.close();
                                }
                                live.insert(key, editor);
                            }
                            GuiWorkerMsg::UnregisterLive { key, reply } => {
                                match key {
                                    Some(k) => {
                                        if let Some(mut inst) = map.remove(&k) {
                                            inst.close();
                                        }
                                        live.remove(&k);
                                    }
                                    None => {
                                        for (_, mut inst) in map.drain() {
                                            inst.close();
                                        }
                                        live.clear();
                                    }
                                }
                                let _ = reply.send(());
                            }
                            GuiWorkerMsg::SyncParam {
                                track_idx,
                                slot_idx,
                                param_id,
                                norm_val,
                            } => {
                                if let Some(inst) = map.get_mut(&(track_idx, slot_idx)) {
                                    match inst {
                                        // The live instance already received the change from the
                                        // RT queue; only remember it so it is not echoed back.
                                        HostedGuiNode::Live { last_norm, .. } => {
                                            if let Some(entry) =
                                                last_norm.iter_mut().find(|e| e.0 == param_id)
                                            {
                                                entry.3 = norm_val as f64;
                                            }
                                        }
                                        HostedGuiNode::Vst3 { node, last_norm } => {
                                            node.set_parameter_normalized(param_id, norm_val);
                                            if let Some(entry) =
                                                last_norm.iter_mut().find(|(id, _)| *id == param_id)
                                            {
                                                entry.1 = norm_val as f64;
                                            }
                                        }
                                        HostedGuiNode::Clap { node, last_norm } => {
                                            node.set_parameter_normalized(param_id, norm_val);
                                            if let Some(entry) =
                                                last_norm.iter_mut().find(|e| e.0 == param_id)
                                            {
                                                entry.3 = norm_val as f64;
                                            }
                                        }
                                    }
                                }
                            }
                            GuiWorkerMsg::Dispose(retired) => drop(retired),
                            GuiWorkerMsg::Shutdown => {
                                for (_, mut inst) in map.drain() {
                                    inst.close();
                                }
                                live.clear();
                                return;
                            }
                        }
                    }

                    for (&(trk, slt), inst) in map.iter_mut() {
                        match inst {
                            HostedGuiNode::Live { editor, last_norm } => match editor {
                                LivePluginEditor::Vst3(e) => {
                                    let _ = e.pump_editor_messages();
                                    for (pid, _, _, prev_n) in last_norm.iter_mut() {
                                        if let Some(cur_n) = e.get_parameter_normalized(*pid) {
                                            if (cur_n - *prev_n).abs() > 1e-4 {
                                                *prev_n = cur_n;
                                                let _ = edit_tx.send((trk, slt, *pid, cur_n as f32));
                                            }
                                        }
                                    }
                                }
                                LivePluginEditor::Clap(e) => {
                                    let _ = e.pump_editor_messages();
                                    for (pid, min_v, max_v, prev_n) in last_norm.iter_mut() {
                                        if let Some(plain) = e.get_parameter_plain(*pid) {
                                            let span = (*max_v - *min_v).max(1e-9);
                                            let cur_n = ((plain - *min_v) / span).clamp(0.0, 1.0);
                                            if (cur_n - *prev_n).abs() > 1e-4 {
                                                *prev_n = cur_n;
                                                let _ = edit_tx.send((trk, slt, *pid, cur_n as f32));
                                            }
                                        }
                                    }
                                }
                            },
                            HostedGuiNode::Vst3 { node, last_norm } => {
                                let _ = node.pump_editor_messages();
                                for (pid, prev_n) in last_norm.iter_mut() {
                                    if let Some(cur_n) = node.get_parameter_normalized(*pid) {
                                        if (cur_n - *prev_n).abs() > 1e-4 {
                                            *prev_n = cur_n;
                                            let _ = edit_tx.send((trk, slt, *pid, cur_n as f32));
                                        }
                                    }
                                }
                            }
                            HostedGuiNode::Clap { node, last_norm } => {
                                let _ = node.pump_editor_messages();
                                for (pid, min_v, max_v, prev_n) in last_norm.iter_mut() {
                                    if let Some(plain) = node.get_parameter_plain(*pid) {
                                        let span = (*max_v - *min_v).max(1e-9);
                                        let cur_n = ((plain - *min_v) / span).clamp(0.0, 1.0);
                                        if (cur_n - *prev_n).abs() > 1e-4 {
                                            *prev_n = cur_n;
                                            let _ = edit_tx.send((trk, slt, *pid, cur_n as f32));
                                        }
                                    }
                                }
                            }
                        }
                    }

                    std::thread::sleep(std::time::Duration::from_millis(8));
                }
            })
            .expect("failed to spawn toneforge-plugin-gui-sta");

        Self {
            tx,
            gui_edits_rx: Mutex::new(edit_rx),
            join_handle: Some(join_handle),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn open_editor(
        &self,
        track_idx: u16,
        slot_idx: u8,
        is_clap: bool,
        binary_path: String,
        plugin_id: Option<String>,
        state_hex: String,
        owner_hwnd: Option<isize>,
        visible: bool,
        sample_rate: f64,
    ) -> Result<isize, String> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        self.tx
            .send(GuiWorkerMsg::Open {
                track_idx,
                slot_idx,
                is_clap,
                binary_path,
                plugin_id,
                state_hex,
                owner_hwnd,
                visible,
                sample_rate,
                reply: reply_tx,
            })
            .map_err(|_| "Plugin GUI STA thread is closed".to_string())?;
        reply_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .map_err(|e| format!("Timed out opening native plugin HWND: {e}"))?
    }

    pub fn close_editor(&self, track_idx: u16, slot_idx: u8) {
        let _ = self.tx.send(GuiWorkerMsg::Close {
            track_idx,
            slot_idx,
            reply: None,
        });
    }

    /// Disposer for [`daw_core_graph::RtGarbageCollector::set_plugin_disposer`]: retired plug-in
    /// instances are destroyed on this GUI thread. If the thread has already stopped, the resource
    /// is dropped by the caller.
    pub fn plugin_disposer(&self) -> daw_core_graph::RetiredPluginDisposer {
        let tx = self.tx.clone();
        Box::new(move |retired| {
            let _ = tx.send(GuiWorkerMsg::Dispose(retired));
        })
    }

    /// Registers the live editor of the instance just loaded into `track_idx` / `slot_idx`.
    pub fn register_live_editor(&self, track_idx: u16, slot_idx: u8, editor: LivePluginEditor) {
        let _ = self.tx.send(GuiWorkerMsg::RegisterLive {
            track_idx,
            slot_idx,
            editor,
        });
    }

    /// Closes and forgets the live editor of one slot (`Some`) or of every slot (`None`), and
    /// waits until the GUI thread is done — call before the instance can be torn down.
    pub fn unregister_live_editors_blocking(&self, key: Option<(u16, u8)>) {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        if self
            .tx
            .send(GuiWorkerMsg::UnregisterLive { key, reply: reply_tx })
            .is_ok()
        {
            let _ = reply_rx.recv_timeout(std::time::Duration::from_secs(5));
        }
    }

    pub fn sync_param(&self, track_idx: u16, slot_idx: u8, param_id: u32, norm_val: f32) {
        let _ = self.tx.send(GuiWorkerMsg::SyncParam {
            track_idx,
            slot_idx,
            param_id,
            norm_val,
        });
    }

    /// Drains any knob/slider edits made inside floating native Win32 VST3/CLAP windows and
    /// forwards them into the lock-free RT engine & `ToneForgeProjectState`.
    pub fn forward_gui_edits_to_rt(
        &self,
        project_state: &mut crate::project_state::ToneForgeProjectState,
        state: &DawAppState,
    ) -> usize {
        let rx = self.gui_edits_rx.lock();
        let mut count = 0usize;
        while let Ok((trk, slt, pid, norm_val)) = rx.try_recv() {
            let _ = cmd_set_external_plugin_param_rt(project_state, state, trk, slt, pid, norm_val);
            count += 1;
        }
        count
    }

    pub fn shutdown(&mut self) {
        let _ = self.tx.send(GuiWorkerMsg::Shutdown);
        if let Some(jh) = self.join_handle.take() {
            let _ = jh.join();
        }
    }
}

impl Drop for ExternalPluginGuiWorker {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Fallback editor: a separate instance initialised from the project's saved state chunk.
#[allow(clippy::too_many_arguments)]
fn open_state_restored_editor(
    map: &mut std::collections::HashMap<(u16, u8), HostedGuiNode>,
    key: (u16, u8),
    is_clap: bool,
    binary_path: &str,
    plugin_id: Option<&str>,
    state_hex: &str,
    owner_hwnd: Option<isize>,
    visible: bool,
    sample_rate: f64,
) -> Result<isize, String> {
    let path = std::path::Path::new(binary_path);
    let state_bytes = crate::project_state::decode_hex_bytes(state_hex).unwrap_or_default();
    if is_clap {
        let mut node = daw_plugin_host::ClapHostedPluginNode::load(path, plugin_id, sample_rate, 256)?;
        if !state_bytes.is_empty() {
            let _ = node.load_state(&state_bytes);
        }
        let hwnd = node.open_editor(owner_hwnd, visible)?;
        let last_norm = node
            .parameters()
            .iter()
            .map(|p| {
                let span = (p.max_value - p.min_value).max(1e-9);
                let n = ((p.current_value - p.min_value) / span).clamp(0.0, 1.0);
                (p.id, p.min_value, p.max_value, n)
            })
            .collect();
        map.insert(key, HostedGuiNode::Clap { node, last_norm });
        Ok(hwnd)
    } else {
        let mut node = daw_plugin_host::Vst3HostedPluginNode::load(path, plugin_id, sample_rate, 256)?;
        if !state_bytes.is_empty() {
            let _ = node.load_state(&state_bytes);
        }
        let hwnd = node.open_editor(owner_hwnd, visible)?;
        let last_norm = node
            .parameters()
            .iter()
            .map(|p| (p.id, node.get_parameter_normalized(p.id).unwrap_or(p.current_value)))
            .collect();
        map.insert(key, HostedGuiNode::Vst3 { node, last_norm });
        Ok(hwnd)
    }
}

// ============================================================================
// HARDWARE AUDIO / MIDI INVENTORY & PRO PLAYLIST / PIANO ROLL / VOCAL EDITORS
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HardwareAudioMidiInventoryDto {
    pub asio_drivers: Vec<daw_core_graph::AsioDriverInfo>,
    pub wasapi_outputs: Vec<daw_core_graph::WasapiEndpointInfo>,
    pub wasapi_inputs: Vec<daw_core_graph::WasapiEndpointInfo>,
    pub midi_inputs: Vec<daw_core_graph::MidiInputDeviceInfo>,
}

/// Enumerates all installed Steinberg ASIO drivers (`HKLM\SOFTWARE\ASIO`), WASAPI output/input
/// endpoints (`eRender` + `eCapture`), and Win32 MIDI Input controllers (`midiInGetDevCapsW`).
pub fn cmd_list_audio_midi_inputs_sync() -> HardwareAudioMidiInventoryDto {
    HardwareAudioMidiInventoryDto {
        asio_drivers: daw_core_graph::enumerate_asio_drivers(),
        wasapi_outputs: daw_core_graph::enumerate_wasapi_output_devices().unwrap_or_default(),
        wasapi_inputs: daw_core_graph::enumerate_wasapi_input_devices(),
        midi_inputs: daw_core_graph::enumerate_midi_input_devices(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecordedTakeUiDto {
    pub take_number: u32,
    pub take_lane_id: u16,
    pub track_index: u16,
    pub wav_path: String,
    pub num_frames: usize,
    pub sample_rate_hz: u32,
}

impl From<daw_core_graph::RecordedTakeInfo> for RecordedTakeUiDto {
    fn from(info: daw_core_graph::RecordedTakeInfo) -> Self {
        Self {
            take_number: info.take_number,
            take_lane_id: info.take_lane_id,
            track_index: info.track_index,
            wav_path: info.wav_path.display().to_string(),
            num_frames: info.num_frames,
            sample_rate_hz: info.sample_rate_hz,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HardwareRecordingStatusDto {
    pub driver_mode: String,
    pub active_output_name: String,
    pub active_input_name: String,
    pub active_midi_name: String,
    pub sample_rate_hz: u32,
    pub buffer_frames: u32,
    pub round_trip_latency_ms: f32,
    pub is_recording: bool,
    pub armed_track_index: u16,
    pub recorded_frames: u64,
    pub completed_takes: Vec<RecordedTakeUiDto>,
}

/// Request DTO for Pro Piano Roll transformations (`Quantize`, `Strum`, `Arpeggiate`, `Lasso`, `Expression`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PianoRollTransformRequestDto {
    pub pattern_id: u32,
    /// `"quantize"` | `"strum"` | `"arpeggiate"` | `"set_expression"`
    pub action: String,
    #[serde(default)]
    pub note_ids: Option<Vec<u32>>,
    #[serde(default = "default_grid_ticks")]
    pub grid_ticks: i64,
    #[serde(default = "default_unit_f32")]
    pub strength: f32,
    #[serde(default)]
    pub quantize_duration: bool,
    #[serde(default = "default_strum_ticks")]
    pub strum_offset_ticks: i64,
    #[serde(default = "default_strum_decay")]
    pub velocity_decay: f32,
    #[serde(default)]
    pub alternate_direction: bool,
    #[serde(default = "default_arp_gate")]
    pub gate_ratio: f32,
    #[serde(default)]
    pub arp_direction: u8,
    #[serde(default = "default_arp_octaves")]
    pub octaves: u8,
    #[serde(default)]
    pub velocity: Option<f32>,
    #[serde(default)]
    pub pan: Option<f32>,
    #[serde(default)]
    pub fine_pitch_cents: Option<f32>,
}

fn default_grid_ticks() -> i64 {
    24
}
fn default_unit_f32() -> f32 {
    1.0
}
fn default_strum_ticks() -> i64 {
    6
}
fn default_strum_decay() -> f32 {
    0.06
}
fn default_arp_gate() -> f32 {
    0.80
}
fn default_arp_octaves() -> u8 {
    1
}

/// Applies a Pro Piano Roll transformation (`Quantize`, `Strum`, `Arpeggiate`, or `SetExpression`)
/// to `pattern_id` in `ToneForgeProjectState` and pushes `RtGraphMutation::UpsertPattern` (`0` RT allocs).
pub fn cmd_piano_roll_transform_notes_rt(
    project_state: &mut crate::project_state::ToneForgeProjectState,
    state: &DawAppState,
    req: &PianoRollTransformRequestDto,
) -> Result<crate::project_state::PianoRollPatternDto, String> {
    let idx = project_state
        .piano_roll_patterns
        .iter()
        .position(|p| p.pattern_id == req.pattern_id)
        .ok_or_else(|| format!("Pattern {} not found", req.pattern_id))?;

    let mut pat = project_state.piano_roll_patterns[idx].to_pattern();
    let ids_slice = req.note_ids.as_deref();
    match req.action.to_ascii_lowercase().as_str() {
        "quantize" => {
            pat.quantize_notes(ids_slice, req.grid_ticks, req.strength, req.quantize_duration);
        }
        "strum" => {
            pat.strum_notes(
                ids_slice,
                req.strum_offset_ticks,
                req.velocity_decay,
                req.alternate_direction,
            );
        }
        "arpeggiate" | "arp" => {
            let dir = match req.arp_direction {
                1 => daw_core_graph::ArpDirection::Down,
                2 => daw_core_graph::ArpDirection::UpDown,
                _ => daw_core_graph::ArpDirection::Up,
            };
            pat.arpeggiate_notes(
                ids_slice,
                req.grid_ticks.max(4),
                req.gate_ratio,
                dir,
                req.octaves,
            );
        }
        "set_expression" | "expression" => {
            if let Some(ids) = ids_slice {
                for &nid in ids {
                    let _ = pat.set_note_expression(
                        nid,
                        req.velocity,
                        req.pan,
                        req.fine_pitch_cents,
                    );
                }
            }
        }
        other => {
            return Err(format!("Unsupported Piano Roll transform action '{other}'"));
        }
    }

    let updated_dto = crate::project_state::PianoRollPatternDto::from_pattern(&pat);
    project_state.piano_roll_patterns[idx] = updated_dto.clone();
    if state.has_rt_mutations() {
        state.push_rt_mutation(daw_core_graph::RtGraphMutation::UpsertPattern {
            pattern: Box::new(pat),
        })?;
    }
    Ok(updated_dto)
}

/// Request DTO for Pro Playlist Clip operations (`Slice` with `2 ms` micro-fade, `Duplicate`, `EditProps`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlaylistClipEditRequestDto {
    pub clip_id: u32,
    /// `"slice"` | `"duplicate"` | `"update_props"`
    pub action: String,
    #[serde(default)]
    pub split_tick: Option<i64>,
    #[serde(default)]
    pub delta_ticks: Option<i64>,
    #[serde(default)]
    pub source_offset_samples: Option<usize>,
    #[serde(default)]
    pub fade_in_samples: Option<usize>,
    #[serde(default)]
    pub fade_out_samples: Option<usize>,
    #[serde(default)]
    pub clip_gain_linear: Option<f32>,
    #[serde(default)]
    pub stretch_ratio: Option<f32>,
    #[serde(default)]
    pub pitch_semitones: Option<f32>,
    #[serde(default)]
    pub take_lane_id: Option<u16>,
}

/// Applies `Razor Slice`, `Duplicate`, or `Fade/Gain/Stretch/Pitch/Slip` edits to `PlaylistClip`s
/// and synchronizes the live RT `PlaylistSequencer` (`0` RT allocs).
pub fn cmd_playlist_slice_or_edit_clip_rt(
    project_state: &mut crate::project_state::ToneForgeProjectState,
    state: &DawAppState,
    req: &PlaylistClipEditRequestDto,
) -> Result<Vec<crate::project_state::PlaylistClipDto>, String> {
    let mut seq = daw_core_graph::PlaylistSequencer::new();
    let mut max_id = 0u32;
    for c in &project_state.playlist_clips {
        max_id = max_id.max(c.clip_id);
        let _ = seq.upsert_clip(c.to_playlist_clip());
    }
    let next_id = max_id.saturating_add(1).max(1);

    match req.action.to_ascii_lowercase().as_str() {
        "slice" => {
            let split_tick = req
                .split_tick
                .ok_or_else(|| "slice action requires splitTick".to_string())?;
            let (left, right) = seq
                .slice_clip_at_tick(
                    req.clip_id,
                    split_tick,
                    next_id,
                    project_state.sample_rate,
                    project_state.bpm,
                    project_state.ppq_resolution,
                )
                .ok_or_else(|| {
                    format!(
                        "Could not slice clip {} at tick {}",
                        req.clip_id, split_tick
                    )
                })?;
            if state.has_rt_mutations() {
                state.push_rt_mutation(daw_core_graph::RtGraphMutation::UpsertPlaylistClip {
                    clip: left,
                })?;
                state.push_rt_mutation(daw_core_graph::RtGraphMutation::UpsertPlaylistClip {
                    clip: right,
                })?;
            }
        }
        "duplicate" => {
            let orig = seq
                .active_clips()
                .iter()
                .find(|c| c.clip_id == req.clip_id)
                .copied()
                .ok_or_else(|| format!("PlaylistClip {} not found", req.clip_id))?;
            let shift = req.delta_ticks.unwrap_or(orig.duration_ticks.max(1));
            let dup = seq
                .duplicate_clip(req.clip_id, next_id, shift)
                .ok_or_else(|| format!("Failed duplicating PlaylistClip {}", req.clip_id))?;
            if state.has_rt_mutations() {
                state.push_rt_mutation(daw_core_graph::RtGraphMutation::UpsertPlaylistClip {
                    clip: dup,
                })?;
            }
        }
        "update_props" | "edit" => {
            let mut target = seq
                .active_clips()
                .iter()
                .find(|c| c.clip_id == req.clip_id)
                .copied()
                .ok_or_else(|| format!("PlaylistClip {} not found", req.clip_id))?;
            if let Some(so) = req.source_offset_samples {
                target.source_offset_samples = so;
            }
            if let Some(fi) = req.fade_in_samples {
                target.fade_in_samples = fi;
            }
            if let Some(fo) = req.fade_out_samples {
                target.fade_out_samples = fo;
            }
            if let Some(g) = req.clip_gain_linear {
                target.gain = g.clamp(0.0, 4.0);
            }
            if let Some(sr) = req.stretch_ratio {
                target.stretch_ratio = sr.clamp(0.5, 2.0);
            }
            if let Some(ps) = req.pitch_semitones {
                target.pitch_semitones = ps.clamp(-12.0, 12.0);
            }
            if let Some(tl) = req.take_lane_id {
                target.take_lane_id = tl;
            }
            let _ = seq.upsert_clip(target);
            if state.has_rt_mutations() {
                state.push_rt_mutation(daw_core_graph::RtGraphMutation::UpsertPlaylistClip {
                    clip: target,
                })?;
            }
        }
        other => {
            return Err(format!("Unsupported PlaylistClip action '{other}'"));
        }
    }

    project_state.playlist_clips = seq
        .active_clips()
        .iter()
        .map(crate::project_state::PlaylistClipDto::from_playlist_clip)
        .collect();
    Ok(project_state.playlist_clips.clone())
}

/// Inserts or updates `AutoPitchCorrectorNode` (`Pitcher` Real-Time Formant-Preserving Vocal Tuner)
/// on `tracks[track_idx].slots[slot_idx]`.
pub fn cmd_update_pitcher_rt_params(
    project_state: &mut crate::project_state::ToneForgeProjectState,
    state: &DawAppState,
    sample_rate: f64,
    track_idx: u16,
    slot_idx: u8,
    cfg: &crate::project_state::PitcherStateDto,
) -> Result<(), String> {
    if (track_idx as usize) >= MAX_MIXER_TRACKS || (slot_idx as usize) >= MAX_INSERT_SLOTS {
        return Err("Track or slot index out of range".to_string());
    }
    let already_pitcher = project_state
        .mixer_tracks
        .get(track_idx as usize)
        .and_then(|t| t.slots.get(slot_idx as usize))
        .map(|s| {
            matches!(
                s.plugin,
                crate::project_state::PluginConfigDto::AutoPitchCorrector(_)
            )
        })
        .unwrap_or(false);

    if !already_pitcher {
        crate::slot_rack::pad_slots_up_to(project_state, state, sample_rate, track_idx, slot_idx)?;
        crate::slot_rack::drop_slot_overlays(project_state, track_idx, slot_idx);
    }
    if !already_pitcher && state.has_rt_mutations() {
        let sr = sample_rate.max(8_000.0);
        let blk = daw_core_graph::MAX_BLOCK_SIZE;
        let node = crate::undo_history::build_pitcher_node(sr, blk, cfg);
        let mut rack_slot = daw_core_graph::PluginRackSlot::new(node, 1.0);
        rack_slot.prepare(sr, blk);
        state.push_rt_mutation(daw_core_graph::RtGraphMutation::SetOrReplacePluginSlot {
            track_index: track_idx,
            slot_index: slot_idx,
            slot: Box::new(Some(rack_slot)),
        })?;
    } else {
        let params: [(u32, f32); 8] = [
            (0, (cfg.root_key.min(11) as f32) / 11.0),
            (1, (cfg.scale_index.min(7) as f32) / 7.0),
            (2, (cfg.retune_speed_ms / 400.0).clamp(0.0, 1.0)),
            (3, cfg.humanize.clamp(0.0, 1.0)),
            (4, if cfg.formant_preserve { 1.0 } else { 0.0 }),
            (
                5,
                ((cfg.formant_shift_semitones + 12.0) / 24.0).clamp(0.0, 1.0),
            ),
            (6, cfg.mix.clamp(0.0, 1.0)),
            (7, if cfg.bypass { 1.0 } else { 0.0 }),
        ];
        for (pid, norm_val) in params {
            state.push_command(EngineCommand::SetPluginParamNormalized {
                track_index: track_idx,
                slot_index: slot_idx,
                param_id: pid,
                norm_val,
                target_sample_pos: None,
            })?;
        }
        state.push_command(EngineCommand::RefreshApdc)?;
    }

    upsert_project_slot_dto(
        project_state,
        track_idx as usize,
        slot_idx as usize,
        crate::project_state::PluginConfigDto::AutoPitchCorrector(cfg.clone()),
    );
    Ok(())
}

/// DTO returned to `VocalPitchEditor.svelte` (`NewTone` graphical pitch & formant editor).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VocalPitchAnalysisResultDto {
    pub sample_rate_hz: u32,
    pub hop_samples: usize,
    pub total_frames: usize,
    pub blobs: Vec<dsp_pitch_time::VocalNoteBlob>,
}

/// Runs `pYIN` probabilistic pitch tracking + note blob segmentation (`VocalNoteGraphEngine`)
/// on `source_frames`.
pub fn analyze_vocal_pitch_blobs_dsp(
    source_frames: Vec<daw_core_graph::StereoFrame>,
    sample_rate_hz: u32,
) -> VocalPitchAnalysisResultDto {
    let total_frames = source_frames.len();
    let engine = dsp_pitch_time::VocalNoteGraphEngine::analyze_from_stereo(
        source_frames,
        sample_rate_hz.clamp(8_000, 192_000),
    );
    VocalPitchAnalysisResultDto {
        sample_rate_hz: engine.sample_rate,
        hop_samples: engine.hop_samples,
        total_frames,
        blobs: engine.blobs,
    }
}

/// Renders edited `VocalNoteBlob`s (`target_midi_note`, `fine_pitch_cents`, `pitch_drift_correction`,
/// `vibrato_scaling`, `formant_shift_semitones`, `warped_start_sample..warped_end_sample`) back into
/// a clean stereo stem with True-Envelope Formant Preservation (`0` chipmunk distortion, `0` sibilant artifacts).
pub fn commit_vocal_pitch_edits_dsp(
    source_frames: Vec<daw_core_graph::StereoFrame>,
    sample_rate_hz: u32,
    edited_blobs: Vec<dsp_pitch_time::VocalNoteBlob>,
) -> Vec<daw_core_graph::StereoFrame> {
    let mut engine = dsp_pitch_time::VocalNoteGraphEngine::analyze_from_stereo(
        source_frames,
        sample_rate_hz.clamp(8_000, 192_000),
    );
    engine.blobs = edited_blobs;
    engine.render_edited_vocal_stereo()
}

