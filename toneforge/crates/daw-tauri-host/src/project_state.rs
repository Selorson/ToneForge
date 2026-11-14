//! ToneForge Project State Management (`project_state.rs`).
//!
//! Provides complete serialization, deserialization, validation, and live engine reconstruction
//! (`ToneForgeProjectState`) for:
//! - 502 Mixer Tracks (`0=Master`, `1..=500=Inserts`, `501=Preview`), inter-track Send & Sidechain routing,
//!   and 5-Point APDC topology verification.
//! - Modular `PatcherGraphNode` sub-DAGs with `SugiyamaLayout` 2D coordinates.
//! - `AutomationClipArticulator`s with signed-exponential `TensionSpline`, `LfoModulator`, and
//!   30-bit fixed-point (`0x40000000`) `ControllerLink`s.
//! - Complete plugin chain states for **FabFilter Pro-Q 4** (`601` params), **oeksound Soothe3** (`143` params),
//!   **iZotope Ozone 12 Master Chain**, **Real-Time 120ms BSRNN Stem Rebalance**, and **FL Studio 26 Native FX**.

use daw_core_graph::{
    AutomationClipArticulator, ControllerLink, FIRST_INSERT_TRACK_INDEX, LfoModulator,
    LfoWaveform, MASTER_TRACK_INDEX, MAX_INSERT_SLOTS, MAX_MIXER_TRACKS, ModularParamConnection,
    PATCHER_HOST_INPUT_NODE, PATCHER_HOST_OUTPUT_NODE, PatcherGraphNode, SugiyamaLayout,
    TensionControlPoint, TensionSpline,
};
use dsp_proq4_eq::{
    EqBandParams, PROQ4_TOTAL_PARAMS, ProQ4GlobalParams, ProQ4ParameterSet,
};
use dsp_soothe3_spectral::{
    SOOTHE3_TOTAL_PARAMS, SidechainEqBandParams, Soothe3GlobalParams, Soothe3ParameterSet,
};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const TONEFORGE_PROJECT_SCHEMA_VERSION: u32 = 1;

/// Serializable control point on a signed-exponential [`TensionSpline`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TensionControlPointDto {
    pub x: f64,
    pub y: f64,
    pub tension: f64,
}

impl From<TensionControlPoint> for TensionControlPointDto {
    fn from(p: TensionControlPoint) -> Self {
        Self {
            x: p.x,
            y: p.y,
            tension: p.tension,
        }
    }
}

impl From<TensionControlPointDto> for TensionControlPoint {
    fn from(dto: TensionControlPointDto) -> Self {
        Self::new(dto.x, dto.y, dto.tension)
    }
}

/// Serializable LFO modulator superimposed on a [`TensionSpline`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LfoModulatorDto {
    pub enabled: bool,
    pub waveform: String,
    pub frequency_cycles_per_tick: f64,
    pub phase_offset: f64,
    pub amount: f64,
}

impl Default for LfoModulatorDto {
    fn default() -> Self {
        Self {
            enabled: false,
            waveform: "Sine".to_string(),
            frequency_cycles_per_tick: 1.0 / 96.0,
            phase_offset: 0.0,
            amount: 0.0,
        }
    }
}

impl LfoModulatorDto {
    pub fn to_lfo_modulator(&self) -> LfoModulator {
        let wf = match self.waveform.to_ascii_lowercase().as_str() {
            "triangle" => LfoWaveform::Triangle,
            "square" => LfoWaveform::Square,
            "saw" => LfoWaveform::Saw,
            _ => LfoWaveform::Sine,
        };
        LfoModulator {
            enabled: self.enabled,
            waveform: wf,
            frequency_cycles_per_tick: self.frequency_cycles_per_tick,
            phase_offset: self.phase_offset,
            amount: self.amount,
        }
    }

    pub fn from_lfo_modulator(lfo: &LfoModulator) -> Self {
        let wf = match lfo.waveform {
            LfoWaveform::Sine => "Sine",
            LfoWaveform::Triangle => "Triangle",
            LfoWaveform::Square => "Square",
            LfoWaveform::Saw => "Saw",
        };
        Self {
            enabled: lfo.enabled,
            waveform: wf.to_string(),
            frequency_cycles_per_tick: lfo.frequency_cycles_per_tick,
            phase_offset: lfo.phase_offset,
            amount: lfo.amount,
        }
    }
}

/// Serializable Automation Clip (`TMixVoiceAutomationClip`) with `TensionSpline` & `LfoModulator`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AutomationClipStateDto {
    pub controller_id: u16,
    pub name: String,
    pub points: Vec<TensionControlPointDto>,
    pub lfo: LfoModulatorDto,
    pub min_level: f64,
    pub max_level: f64,
}

impl AutomationClipStateDto {
    pub fn to_articulator(&self) -> AutomationClipArticulator {
        let pts: Vec<TensionControlPoint> = self.points.iter().copied().map(Into::into).collect();
        let mut art = AutomationClipArticulator::new(self.controller_id, TensionSpline::new(pts));
        art.lfo = self.lfo.to_lfo_modulator();
        art.min_level = self.min_level;
        art.max_level = self.max_level;
        art
    }
}

/// Serializable link on the 30-Bit Fixed-Point Internal Controller Bus (`0x40000000`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ControllerLinkStateDto {
    pub controller_id: u16,
    pub target_track: u16,
    pub slot_index: u8,
    pub param_id: u32,
    pub min_out_norm: f32,
    pub max_out_norm: f32,
    pub inverted: bool,
}

impl From<ControllerLink> for ControllerLinkStateDto {
    fn from(link: ControllerLink) -> Self {
        Self {
            controller_id: link.controller_id,
            target_track: link.target_track,
            slot_index: link.slot_index,
            param_id: link.param_id,
            min_out_norm: link.min_out_norm,
            max_out_norm: link.max_out_norm,
            inverted: link.inverted,
        }
    }
}

impl From<ControllerLinkStateDto> for ControllerLink {
    fn from(dto: ControllerLinkStateDto) -> Self {
        Self {
            controller_id: dto.controller_id,
            target_track: dto.target_track,
            slot_index: dto.slot_index,
            param_id: dto.param_id,
            min_out_norm: dto.min_out_norm,
            max_out_norm: dto.max_out_norm,
            inverted: dto.inverted,
        }
    }
}

/// Encodes binary plugin state bytes as lowercase hex for lossless JSON `.tfproj` serialization.
pub fn encode_hex_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0F) as usize] as char);
    }
    out
}

/// Decodes hex-encoded binary plugin state bytes from JSON `.tfproj`.
pub fn decode_hex_bytes(hex_str: &str) -> Result<Vec<u8>, String> {
    let trimmed = hex_str.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    if trimmed.len() % 2 != 0 {
        return Err("Invalid odd-length hex string".to_string());
    }
    let bytes = trimmed.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 2);
    let nibble = |c: u8| -> Result<u8, String> {
        match c {
            b'0'..=b'9' => Ok(c - b'0'),
            b'a'..=b'f' => Ok(c - b'a' + 10),
            b'A'..=b'F' => Ok(c - b'A' + 10),
            _ => Err(format!("Invalid hex character: 0x{c:02X}")),
        }
    };
    for chunk in bytes.chunks_exact(2) {
        out.push((nibble(chunk[0])? << 4) | nibble(chunk[1])?);
    }
    Ok(out)
}

/// Serializable 601-parameter FabFilter Pro-Q 4 state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProQ4StateDto {
    pub bands: Vec<EqBandParams>,
    pub global: ProQ4GlobalParams,
    pub normalized_601: Vec<f32>,
    #[serde(default)]
    pub collision_reference_track: Option<u16>,
}

impl Default for ProQ4StateDto {
    fn default() -> Self {
        Self::from_parameter_set(&ProQ4ParameterSet::new())
    }
}

impl ProQ4StateDto {
    pub fn from_parameter_set(ps: &ProQ4ParameterSet) -> Self {
        let mut norm = Vec::with_capacity(PROQ4_TOTAL_PARAMS);
        for id in 0..PROQ4_TOTAL_PARAMS {
            norm.push(ps.get_normalized(id as u32));
        }
        Self {
            bands: ps.bands.to_vec(),
            global: ps.global,
            normalized_601: norm,
            collision_reference_track: None,
        }
    }

    pub fn to_parameter_set(&self) -> ProQ4ParameterSet {
        let mut ps = ProQ4ParameterSet::new();
        if self.normalized_601.len() == PROQ4_TOTAL_PARAMS {
            for (id, &val) in self.normalized_601.iter().enumerate() {
                ps.set_normalized(id as u32, val);
            }
        }
        for (i, b) in self.bands.iter().copied().enumerate().take(24) {
            ps.bands[i] = b;
        }
        ps.global = self.global;
        ps.sync_mirror_from_structured();
        ps
    }
}

/// Serializable 143-parameter oeksound Soothe3 state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Soothe3StateDto {
    pub global: Soothe3GlobalParams,
    pub sc_bands: Vec<SidechainEqBandParams>,
    pub normalized_143: Vec<f32>,
    #[serde(default)]
    pub external_sidechain_source_track: Option<u16>,
}

impl Default for Soothe3StateDto {
    fn default() -> Self {
        Self::from_parameter_set(&Soothe3ParameterSet::new())
    }
}

impl Soothe3StateDto {
    pub fn from_parameter_set(ps: &Soothe3ParameterSet) -> Self {
        Self {
            global: ps.global,
            sc_bands: ps.sc_bands.to_vec(),
            normalized_143: ps.normalized_mirror.to_vec(),
            external_sidechain_source_track: None,
        }
    }

    pub fn to_parameter_set(&self) -> Soothe3ParameterSet {
        let mut ps = Soothe3ParameterSet::new();
        if self.normalized_143.len() == SOOTHE3_TOTAL_PARAMS {
            for (id, &val) in self.normalized_143.iter().enumerate() {
                ps.set_normalized(id as u32, val);
            }
        }
        ps.global = self.global;
        for (i, b) in self.sc_bands.iter().copied().enumerate().take(8) {
            ps.sc_bands[i] = b;
        }
        ps.sync_mirror_from_structured();
        ps
    }
}

fn default_lef_upper_hz() -> f32 {
    150.0
}

/// Serializable configuration for the iZotope Ozone 12 Mastering Chain.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Ozone12MasterConfigDto {
    pub maximizer_bypass: bool,
    pub maximizer_irc_mode: u8,         // 0=IRC I, 1=IRC II, 2=IRC III, 3=IRC IV, 4=IRC V, 5=IRC LL
    pub maximizer_gain_db: f32,         // -20.0 .. +20.0 dB
    pub maximizer_ceiling_dbtp: f32,    // -20.0 .. 0.0 dBTP
    pub maximizer_true_peak: bool,
    pub maximizer_upward_comp_wet: f32, // 0.0 .. 1.0
    pub maximizer_soft_clip_enabled: bool,
    pub maximizer_soft_clip_mix: f32,   // 0.0 .. 1.0
    pub maximizer_soft_clip_mode: u8,   // 0=Light, 1=Medium, 2=Heavy
    pub maximizer_transient_shaping: f32,
    pub exciter_bypass: bool,
    pub exciter_band_drives_pct: [f32; 4],
    pub exciter_band_mixes: [f32; 4],
    pub exciter_band_modes: [u8; 4],    // 0=Analog, 1=Retro, 2=Tape, 3=Warm, 4=Triode, 5=DualTriode, 6=Tube
    pub imager_band_widths_pct: [f32; 4],
    pub imager_band_stereoize_pct: [f32; 4],
    pub imager_stereoizer_enabled: bool,
    pub imager_stereoizer_mode: u8,     // 0=ModeI, 1=ModeII
    pub stabilizer_amount_pct: f32,
    pub stabilizer_speed_pct: f32,
    pub stabilizer_tame_transients_pct: f32,
    pub stabilizer_cut_mode: bool,
    pub impact_band_amounts: [f32; 4],
    pub clarity_amount_pct: f32,
    pub clarity_tilt_db_oct: f32,
    #[serde(default)]
    pub unlimiter_enabled: bool,
    #[serde(default)]
    pub unlimiter_amount_pct: f32,
    #[serde(default = "default_unlimiter_threshold_db")]
    pub unlimiter_threshold_db: f32,
    #[serde(default)]
    pub low_end_focus_enabled: bool,
    #[serde(default)]
    pub low_end_focus_punchy: bool,
    #[serde(default = "default_lef_upper_hz")]
    pub low_end_focus_upper_hz: f32,
    #[serde(default)]
    pub low_end_focus_contrast_pct: f32,
    #[serde(default)]
    pub low_end_focus_gain_db: f32,
    #[serde(default)]
    pub low_end_focus_solo: bool,
    #[serde(default)]
    pub codec_preview_enabled: bool,
    #[serde(default)]
    pub codec_preview_codec: u8, // 0=PcmBypass, 1=Mp3_128, 2=Mp3_320, 3=Aac_128, 4=Aac_256, 5=Opus_128
    #[serde(default)]
    pub codec_preview_artifact_solo: bool,
}

fn default_unlimiter_threshold_db() -> f32 {
    -6.0
}

impl Default for Ozone12MasterConfigDto {
    fn default() -> Self {
        Self {
            maximizer_bypass: false,
            maximizer_irc_mode: 4, // IRC V
            maximizer_gain_db: 2.0,
            maximizer_ceiling_dbtp: -1.0,
            maximizer_true_peak: true,
            maximizer_upward_comp_wet: 0.15,
            maximizer_soft_clip_enabled: true,
            maximizer_soft_clip_mix: 0.25,
            maximizer_soft_clip_mode: 1, // Medium
            maximizer_transient_shaping: 0.1,
            exciter_bypass: false,
            exciter_band_drives_pct: [15.0, 20.0, 25.0, 20.0],
            exciter_band_mixes: [0.5, 0.5, 0.5, 0.4],
            exciter_band_modes: [4, 4, 5, 3], // Triode / DualTriode / Warm
            imager_band_widths_pct: [0.0, 10.0, 20.0, 30.0],
            imager_band_stereoize_pct: [0.0, 5.0, 15.0, 20.0],
            imager_stereoizer_enabled: true,
            imager_stereoizer_mode: 0,
            stabilizer_amount_pct: 35.0,
            stabilizer_speed_pct: 50.0,
            stabilizer_tame_transients_pct: 25.0,
            stabilizer_cut_mode: false,
            impact_band_amounts: [10.0, 5.0, 5.0, 10.0],
            clarity_amount_pct: 30.0,
            clarity_tilt_db_oct: 0.5,
            unlimiter_enabled: false,
            unlimiter_amount_pct: 0.0,
            unlimiter_threshold_db: -6.0,
            low_end_focus_enabled: false,
            low_end_focus_punchy: true,
            low_end_focus_upper_hz: 150.0,
            low_end_focus_contrast_pct: 0.0,
            low_end_focus_gain_db: 0.0,
            low_end_focus_solo: false,
            codec_preview_enabled: false,
            codec_preview_codec: 0,
            codec_preview_artifact_solo: false,
        }
    }
}

/// Serializable state for `RealTimeBsrnnStemNode` (Master Rebalance / 4-Stem EQ).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BsrnnStemStateDto {
    pub stem_gains_db: [f32; 4], // [Vocals, Drums, Bass, Other]
    pub stem_solo: [bool; 4],
    pub stem_mute: [bool; 4],
}

impl Default for BsrnnStemStateDto {
    fn default() -> Self {
        Self {
            stem_gains_db: [0.0; 4],
            stem_solo: [false; 4],
            stem_mute: [false; 4],
        }
    }
}

/// Serializable node inside a Modular `PatcherGraphNode`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PatcherNodeStateDto {
    pub node_index: usize,
    pub label: String,
    pub plugin_kind: String, // "ProQ4", "Soothe3", "SparseCompressor", "Maximus", "Emphasis", "LuxeVerb", "FixedLatency"
    pub fixed_latency_samples: usize,
    pub gain_scale: f32,
}

/// Serializable audio cord inside `PatcherGraphNode` (`from_node == usize::MAX - 1` = Host Input,
/// `to_node == usize::MAX` = Host Output).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PatcherAudioEdgeDto {
    pub from_node: usize,
    pub to_node: usize,
    pub gain: f32,
}

/// Serializable parameter modulation cord inside `PatcherGraphNode`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PatcherParamEdgeDto {
    pub source_controller: u16,
    pub target_node: usize,
    pub target_param_id: u32,
    pub scale: f32,
    pub offset: f32,
}

/// Serializable 2D layout position computed by `SugiyamaLayout`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PatcherNodeLayoutDto {
    pub node_index: usize,
    pub layer: usize,
    pub rank_in_layer: usize,
    pub x_px: f32,
    pub y_px: f32,
}

/// Serializable Modular `PatcherGraphNode` state with automatic `SugiyamaLayout`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PatcherGraphStateDto {
    pub name: String,
    pub nodes: Vec<PatcherNodeStateDto>,
    pub audio_connections: Vec<PatcherAudioEdgeDto>,
    pub param_connections: Vec<PatcherParamEdgeDto>,
    pub sugiyama_layout: Vec<PatcherNodeLayoutDto>,
}

impl Default for PatcherGraphStateDto {
    fn default() -> Self {
        Self::new_parallel_mastering_patcher()
    }
}

impl PatcherGraphStateDto {
    /// Creates a default parallel multi-branch Patcher DAG (`HostInput -> [Branch A (0), Branch B (1)] -> HostOutput`)
    /// with precomputed Sugiyama 2D layout coordinates.
    pub fn new_parallel_mastering_patcher() -> Self {
        let nodes = vec![
            PatcherNodeStateDto {
                node_index: 0,
                label: "Parallel Dynamic EQ Branch".to_string(),
                plugin_kind: "FixedLatency".to_string(),
                fixed_latency_samples: 64,
                gain_scale: 0.5,
            },
            PatcherNodeStateDto {
                node_index: 1,
                label: "Parallel Harmonic Transient Branch".to_string(),
                plugin_kind: "FixedLatency".to_string(),
                fixed_latency_samples: 0,
                gain_scale: 0.5,
            },
        ];
        let audio_connections = vec![
            PatcherAudioEdgeDto {
                from_node: PATCHER_HOST_INPUT_NODE,
                to_node: 0,
                gain: 1.0,
            },
            PatcherAudioEdgeDto {
                from_node: PATCHER_HOST_INPUT_NODE,
                to_node: 1,
                gain: 1.0,
            },
            PatcherAudioEdgeDto {
                from_node: 0,
                to_node: PATCHER_HOST_OUTPUT_NODE,
                gain: 1.0,
            },
            PatcherAudioEdgeDto {
                from_node: 1,
                to_node: PATCHER_HOST_OUTPUT_NODE,
                gain: 1.0,
            },
        ];
        let mut dto = Self {
            name: "Parallel APDC Mastering Patcher".to_string(),
            nodes,
            audio_connections,
            param_connections: Vec::new(),
            sugiyama_layout: Vec::new(),
        };
        dto.recompute_sugiyama_layout();
        dto
    }

    pub fn recompute_sugiyama_layout(&mut self) {
        let n = self.nodes.len();
        let edges: Vec<(usize, usize)> = self
            .audio_connections
            .iter()
            .filter(|e| e.from_node < n && e.to_node < n)
            .map(|e| (e.from_node, e.to_node))
            .collect();
        self.sugiyama_layout = SugiyamaLayout::compute_layout(n, &edges, 180.0, 96.0)
            .into_iter()
            .map(|p| PatcherNodeLayoutDto {
                node_index: p.node_index,
                layer: p.layer,
                rank_in_layer: p.rank_in_layer,
                x_px: p.x_px,
                y_px: p.y_px,
            })
            .collect();
    }

    /// Reconstructs a live [`PatcherGraphNode`] from this DTO and validates that no cycles exist.
    pub fn build_patcher_node(&self, sample_rate: f64, max_block_size: usize) -> Result<PatcherGraphNode, String> {
        use daw_core_graph::{
            EmphasisLimiterNode, FixedLatencyTransformNode, GrossBeatTimeBufferNode,
            LuxeVerbFdnShimmerNode, MaximusMultibandNode, SparseCompressorNode, UnifiedDspNode,
        };
        use dsp_proq4_eq::ProQ4Processor;
        use dsp_soothe3_spectral::Soothe3Processor;

        let mut patcher = PatcherGraphNode::new();
        patcher.prepare(sample_rate, max_block_size);

        for node_dto in &self.nodes {
            let dsp_box: Box<dyn UnifiedDspNode> = match node_dto.plugin_kind.as_str() {
                "ProQ4" => Box::new(ProQ4Processor::new()),
                "Soothe3" => Box::new(Soothe3Processor::new(sample_rate as f32)),
                "SparseCompressor" => Box::new(SparseCompressorNode::new(0.5, 4.0, 1.0, 0)),
                "Maximus" => Box::new(MaximusMultibandNode::new()),
                "Emphasis" => Box::new(EmphasisLimiterNode::new(64, -0.5)),
                "GrossBeat" => Box::new(GrossBeatTimeBufferNode::new((sample_rate as usize) * 2)),
                "LuxeVerb" => Box::new(LuxeVerbFdnShimmerNode::new()),
                _ => Box::new(FixedLatencyTransformNode::new(
                    node_dto.fixed_latency_samples,
                    node_dto.gain_scale,
                    false,
                )),
            };
            patcher.add_node(dsp_box);
        }

        for edge in &self.audio_connections {
            if !patcher.add_audio_connection(edge.from_node, edge.to_node, edge.gain) {
                return Err(format!(
                    "Cycle or invalid audio connection in Patcher DAG: {} -> {}",
                    edge.from_node, edge.to_node
                ));
            }
        }

        for p_edge in &self.param_connections {
            patcher.add_param_connection(ModularParamConnection {
                source_controller: p_edge.source_controller,
                target_node: p_edge.target_node,
                target_param_id: p_edge.target_param_id,
                scale: p_edge.scale,
                offset: p_edge.offset,
            });
        }

        Ok(patcher)
    }
}

/// Serializable configuration for `AutoPitchCorrectorNode` (`Pitcher` real-time vocal pitch & formant corrector).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PitcherStateDto {
    pub root_key: u8,
    pub scale_index: u8,
    pub retune_speed_ms: f32,
    pub humanize: f32,
    pub formant_preserve: bool,
    pub formant_shift_semitones: f32,
    pub mix: f32,
    pub bypass: bool,
}

impl Default for PitcherStateDto {
    fn default() -> Self {
        Self {
            root_key: 0,
            scale_index: 0,
            retune_speed_ms: 15.0,
            humanize: 0.25,
            formant_preserve: true,
            formant_shift_semitones: 0.0,
            mix: 1.0,
            bypass: false,
        }
    }
}

/// Serializable state for an external hosted Steinberg VST3 3.7 or CLAP 1.2 plugin binary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExternalPluginStateDto {
    pub binary_path: String,
    pub plugin_id: String,
    pub plugin_name: String,
    #[serde(default)]
    pub state_hex: String,
    #[serde(default)]
    pub param_overrides: Vec<(u32, f64)>,
}

/// Union of all supported plugin configurations inside a Mixer Track FX Slot (`0..10`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "plugin_type", content = "config")]
pub enum PluginConfigDto {
    ProQ4(ProQ4StateDto),
    Soothe3(Soothe3StateDto),
    Ozone12Master(Ozone12MasterConfigDto),
    RealTimeBsrnnStem(BsrnnStemStateDto),
    PatcherModular(PatcherGraphStateDto),
    AutoPitchCorrector(PitcherStateDto),
    ExternalVst3(ExternalPluginStateDto),
    ExternalClap(ExternalPluginStateDto),
    NativeSparseCompressor {
        threshold_linear: f32,
        ratio: f32,
        makeup_gain: f32,
        lookahead_samples: usize,
    },
    NativeMaximusMultiband,
    NativeEmphasisLimiter {
        ceiling_linear: f32,
        drive_linear: f32,
    },
    NativeGrossBeat,
    NativeLuxeVerbFdn,
    /// Unoccupied fixed insert position (engine: zero-latency pass-through `EmptySlotNode`).
    /// Removing a plug-in leaves this in place so later slots never shift.
    Empty,
}

/// Serializable state of one of the 10 FX Insert Slots on a `MixerTrack`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PluginSlotStateDto {
    pub slot_index: u8,
    pub enabled: bool,
    pub wet_mix: f32,
    pub plugin: PluginConfigDto,
}

/// Serializable inter-track Send / Sidechain route (`MixerSendRoute`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MixerSendRouteDto {
    pub dest_track_index: usize,
    pub gain: f32,
    pub is_sidechain_only: bool,
}

/// Serializable state of a single `MixerTrack` (`0..502`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MixerTrackStateDto {
    pub index: usize,
    pub name: String,
    pub gain: f32,
    pub polarity_inverted: bool,
    pub has_hw_input: bool,
    pub hw_input_latency_samples: usize,
    #[serde(default)]
    pub input_source_bus: Option<u8>,
    #[serde(default)]
    pub record_armed: bool,
    pub sends: Vec<MixerSendRouteDto>,
    pub slots: Vec<PluginSlotStateDto>,
    /// Stereo balance (`-1.0..=1.0`, unity at centre).
    #[serde(default)]
    pub pan: f32,
    #[serde(default)]
    pub muted: bool,
    #[serde(default)]
    pub soloed: bool,
    /// External VST3/CLAP instrument loaded as this insert's channel generator
    /// (`ChannelGenerator` id `200 + index`): `PluginConfigDto::ExternalVst3` / `ExternalClap`.
    #[serde(default)]
    pub channel_instrument: Option<PluginConfigDto>,
}

impl MixerTrackStateDto {
    pub fn new_default(index: usize) -> Self {
        let name = match index {
            0 => "Master".to_string(),
            501 => "Preview".to_string(),
            i => format!("Insert {}", i),
        };
        let mut sends = Vec::new();
        if (FIRST_INSERT_TRACK_INDEX..=4).contains(&index) {
            sends.push(MixerSendRouteDto {
                dest_track_index: MASTER_TRACK_INDEX,
                gain: 1.0,
                is_sidechain_only: false,
            });
        }
        Self {
            index,
            name,
            gain: 1.0,
            polarity_inverted: false,
            has_hw_input: false,
            hw_input_latency_samples: 0,
            input_source_bus: None,
            record_armed: false,
            sends,
            slots: Vec::new(),
            pan: 0.0,
            muted: false,
            soloed: false,
            channel_instrument: None,
        }
    }
}

/// Serializable Piano Roll Note DTO (with FL Studio 26 Slide/Portamento & per-note expression).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PianoRollNoteDto {
    pub id: u32,
    pub start_tick: i64,
    pub duration_ticks: i64,
    pub key: u8,
    pub velocity: f32,
    pub pan: f32,
    pub fine_pitch_cents: f32,
    pub is_slide: bool,
    pub color_group: u8,
}

impl From<daw_core_graph::PianoRollNote> for PianoRollNoteDto {
    fn from(n: daw_core_graph::PianoRollNote) -> Self {
        Self {
            id: n.id,
            start_tick: n.start_tick,
            duration_ticks: n.duration_ticks,
            key: n.key,
            velocity: n.velocity,
            pan: n.pan,
            fine_pitch_cents: n.fine_pitch_cents,
            is_slide: n.is_slide,
            color_group: n.color_group,
        }
    }
}

impl From<PianoRollNoteDto> for daw_core_graph::PianoRollNote {
    fn from(dto: PianoRollNoteDto) -> Self {
        Self {
            id: dto.id,
            start_tick: dto.start_tick,
            duration_ticks: dto.duration_ticks,
            key: dto.key,
            velocity: dto.velocity,
            pan: dto.pan,
            fine_pitch_cents: dto.fine_pitch_cents,
            is_slide: dto.is_slide,
            color_group: dto.color_group,
        }
    }
}

/// Serializable Piano Roll Pattern DTO.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PianoRollPatternDto {
    pub pattern_id: u32,
    pub name: String,
    pub target_channel_id: u32,
    pub length_ticks: i64,
    pub notes: Vec<PianoRollNoteDto>,
}

impl PianoRollPatternDto {
    pub fn to_pattern(&self) -> daw_core_graph::PianoRollPattern {
        let mut pat = daw_core_graph::PianoRollPattern::new(
            self.pattern_id,
            self.name.clone(),
            self.target_channel_id,
            self.length_ticks,
        );
        for &n in &self.notes {
            let _ = pat.add_note(n.into());
        }
        pat
    }

    pub fn from_pattern(pat: &daw_core_graph::PianoRollPattern) -> Self {
        Self {
            pattern_id: pat.pattern_id,
            name: pat.name.clone(),
            target_channel_id: pat.target_channel_id,
            length_ticks: pat.length_ticks,
            notes: pat.active_notes().iter().copied().map(Into::into).collect(),
        }
    }
}

fn default_clip_stretch_ratio() -> f32 {
    1.0
}

/// Serializable Playlist Arrangement Clip DTO.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlaylistClipDto {
    pub clip_id: u32,
    pub track_row: u16,
    pub kind: String, // "Pattern", "AudioStem", "Automation"
    pub source_id: u32,
    pub start_tick: i64,
    pub duration_ticks: i64,
    pub clip_offset_ticks: i64,
    pub gain: f32,
    pub muted: bool,
    #[serde(default)]
    pub source_offset_samples: usize,
    #[serde(default)]
    pub fade_in_samples: usize,
    #[serde(default)]
    pub fade_out_samples: usize,
    #[serde(default = "default_clip_stretch_ratio")]
    pub stretch_ratio: f32,
    #[serde(default)]
    pub pitch_semitones: f32,
    #[serde(default)]
    pub take_lane_id: u16,
}

impl Default for PlaylistClipDto {
    fn default() -> Self {
        Self {
            clip_id: 0,
            track_row: 0,
            kind: "Pattern".to_string(),
            source_id: 0,
            start_tick: 0,
            duration_ticks: 384,
            clip_offset_ticks: 0,
            gain: 1.0,
            muted: false,
            source_offset_samples: 0,
            fade_in_samples: 0,
            fade_out_samples: 0,
            stretch_ratio: 1.0,
            pitch_semitones: 0.0,
            take_lane_id: 0,
        }
    }
}

impl PlaylistClipDto {
    pub fn to_playlist_clip(&self) -> daw_core_graph::PlaylistClip {
        let kind = match self.kind.to_ascii_lowercase().as_str() {
            "audiostem" | "audio" | "stem" => daw_core_graph::PlaylistClipKind::AudioStem,
            "automation" => daw_core_graph::PlaylistClipKind::Automation,
            _ => daw_core_graph::PlaylistClipKind::Pattern,
        };
        daw_core_graph::PlaylistClip {
            clip_id: self.clip_id,
            track_row: self.track_row,
            kind,
            source_id: self.source_id,
            start_tick: self.start_tick,
            duration_ticks: self.duration_ticks,
            clip_offset_ticks: self.clip_offset_ticks,
            gain: self.gain,
            muted: self.muted,
            source_offset_samples: self.source_offset_samples,
            fade_in_samples: self.fade_in_samples,
            fade_out_samples: self.fade_out_samples,
            stretch_ratio: if self.stretch_ratio > 0.0 {
                self.stretch_ratio.clamp(0.25, 4.0)
            } else {
                1.0
            },
            pitch_semitones: self.pitch_semitones.clamp(-24.0, 24.0),
            take_lane_id: self.take_lane_id,
        }
    }

    pub fn from_playlist_clip(clip: &daw_core_graph::PlaylistClip) -> Self {
        let kind = match clip.kind {
            daw_core_graph::PlaylistClipKind::Pattern => "Pattern",
            daw_core_graph::PlaylistClipKind::AudioStem => "AudioStem",
            daw_core_graph::PlaylistClipKind::Automation => "Automation",
        };
        Self {
            clip_id: clip.clip_id,
            track_row: clip.track_row,
            kind: kind.to_string(),
            source_id: clip.source_id,
            start_tick: clip.start_tick,
            duration_ticks: clip.duration_ticks,
            clip_offset_ticks: clip.clip_offset_ticks,
            gain: clip.gain,
            muted: clip.muted,
            source_offset_samples: clip.source_offset_samples,
            fade_in_samples: clip.fade_in_samples,
            fade_out_samples: clip.fade_out_samples,
            stretch_ratio: clip.stretch_ratio,
            pitch_semitones: clip.pitch_semitones,
            take_lane_id: clip.take_lane_id,
        }
    }
}

fn default_ppq_resolution() -> u32 {
    daw_core_graph::DEFAULT_PPQ
}

/// Complete ToneForge Project State (`ToneForgeProjectState`) managing 502 Mixer Tracks,
/// Patcher DAGs, Automation Clips with `TensionSpline`, 30-Bit Controller Bus links,
/// Piano Roll Patterns, Playlist Clips, and all DSP/AI plugin chains (`Pro-Q 4`, `Soothe3`,
/// `Ozone 12`, `RealTimeBsrnnStemNode`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToneForgeProjectState {
    pub project_name: String,
    pub schema_version: u32,
    pub sample_rate: f64,
    pub max_block_size: usize,
    pub bpm: f64,
    #[serde(default = "default_ppq_resolution")]
    pub ppq_resolution: u32,
    pub is_playing: bool,
    pub sample_pos: i64,
    pub ppq_tick: i64,
    pub process_all_502_tracks: bool,
    pub mixer_tracks: Vec<MixerTrackStateDto>,
    pub patcher_graphs: Vec<PatcherGraphStateDto>,
    pub automation_clips: Vec<AutomationClipStateDto>,
    pub controller_links: Vec<ControllerLinkStateDto>,
    #[serde(default)]
    pub piano_roll_patterns: Vec<PianoRollPatternDto>,
    #[serde(default)]
    pub playlist_clips: Vec<PlaylistClipDto>,
    #[serde(default)]
    pub vocal_note_blobs: Vec<dsp_pitch_time::VocalNoteBlob>,
    /// Audio files the project plays (loaded songs, UVR5 stems, committed vocals, recorded takes).
    #[serde(default)]
    pub audio_assets: Vec<crate::audio_assets::AudioAssetDto>,
    /// Interactive Patcher canvases applied to insert slots (re-applied on undo / render / load).
    #[serde(default)]
    pub patcher_canvases: Vec<crate::ipc_commands::PatcherCanvasUpdateDto>,
    /// FL Studio 26 native FX presets loaded into insert slots (re-applied on render / load).
    #[serde(default)]
    pub fl26_fx_presets: Vec<Fl26SlotPresetDto>,
}

/// An FL Studio 26 `.fst` preset loaded into `track_idx` / `slot_idx`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Fl26SlotPresetDto {
    pub track_idx: u16,
    pub slot_idx: u8,
    pub preset_query: String,
}

impl Default for ToneForgeProjectState {
    fn default() -> Self {
        Self::new_default(48_000.0, 128.0)
    }
}

impl ToneForgeProjectState {
    /// Creates a clean 502-track project with Insert Tracks `1..=4` routed to Master Track `0`.
    pub fn new_default(sample_rate: f64, bpm: f64) -> Self {
        let mut mixer_tracks = Vec::with_capacity(MAX_MIXER_TRACKS);
        for idx in 0..MAX_MIXER_TRACKS {
            mixer_tracks.push(MixerTrackStateDto::new_default(idx));
        }

        Self {
            project_name: "ToneForge Master Session".to_string(),
            schema_version: TONEFORGE_PROJECT_SCHEMA_VERSION,
            sample_rate: sample_rate.max(8000.0),
            max_block_size: 512,
            bpm: bpm.clamp(10.0, 999.0),
            ppq_resolution: daw_core_graph::DEFAULT_PPQ,
            is_playing: true,
            sample_pos: 0,
            ppq_tick: 0,
            process_all_502_tracks: false,
            mixer_tracks,
            patcher_graphs: vec![PatcherGraphStateDto::new_parallel_mastering_patcher()],
            automation_clips: vec![AutomationClipStateDto {
                controller_id: 0,
                name: "Pro-Q 4 Band 1 Sweep Spline".to_string(),
                points: vec![
                    TensionControlPointDto {
                        x: 0.0,
                        y: 0.25,
                        tension: 0.0,
                    },
                    TensionControlPointDto {
                        x: 96.0,
                        y: 0.75,
                        tension: 0.4,
                    },
                    TensionControlPointDto {
                        x: 192.0,
                        y: 0.50,
                        tension: -0.35,
                    },
                ],
                lfo: LfoModulatorDto::default(),
                min_level: 0.0,
                max_level: 1.0,
            }],
            controller_links: Vec::new(),
            piano_roll_patterns: vec![PianoRollPatternDto {
                pattern_id: 1,
                name: "ToneForge Lead Hook (FL26 Slide)".to_string(),
                target_channel_id: 1,
                length_ticks: 384,
                notes: vec![
                    PianoRollNoteDto {
                        id: 1,
                        start_tick: 0,
                        duration_ticks: 96,
                        key: 60,
                        velocity: 0.88,
                        pan: -0.15,
                        fine_pitch_cents: 0.0,
                        is_slide: false,
                        color_group: 0,
                    },
                    PianoRollNoteDto {
                        id: 2,
                        start_tick: 48,
                        duration_ticks: 48,
                        key: 72,
                        velocity: 0.95,
                        pan: 0.25,
                        fine_pitch_cents: 0.0,
                        is_slide: true,
                        color_group: 0,
                    },
                ],
            }],
            playlist_clips: vec![
                PlaylistClipDto {
                    clip_id: 1,
                    track_row: 0,
                    kind: "Pattern".to_string(),
                    source_id: 1,
                    start_tick: 0,
                    duration_ticks: 384,
                    clip_offset_ticks: 0,
                    gain: 1.0,
                    muted: false,
                    ..Default::default()
                },
                PlaylistClipDto {
                    clip_id: 2,
                    track_row: 1,
                    kind: "Automation".to_string(),
                    source_id: 0,
                    start_tick: 0,
                    duration_ticks: 384,
                    clip_offset_ticks: 0,
                    gain: 1.0,
                    muted: false,
                    ..Default::default()
                },
            ],
            vocal_note_blobs: Vec::new(),
            audio_assets: Vec::new(),
            patcher_canvases: Vec::new(),
            fl26_fx_presets: Vec::new(),
        }
    }

    /// Creates a full production mastering template with:
    /// - Insert Track 1: `Pro-Q 4` (Slot 0) + `Soothe3` (Slot 1) + `PatcherModular` (Slot 2)
    /// - Master Track 0: `RealTimeBsrnnStemNode` (Slot 0) + `Ozone12MasterChain` (Slot 1)
    /// - Automation Clip 0 (`TensionSpline`) linked via 30-bit Fixed-Point Controller Bus to Track 1 Slot 0.
    pub fn new_mastering_template(sample_rate: f64, bpm: f64) -> Self {
        let mut proj = Self::new_default(sample_rate, bpm);

        let mut proq4_state = ProQ4ParameterSet::new();
        proq4_state.bands[0].used = true;
        proq4_state.bands[0].enabled = true;
        proq4_state.bands[0].freq_hz = 120.0;
        proq4_state.bands[0].gain_db = 2.5;
        proq4_state.bands[1].used = true;
        proq4_state.bands[1].enabled = true;
        proq4_state.bands[1].freq_hz = 3500.0;
        proq4_state.bands[1].gain_db = -1.8;
        proq4_state.sync_mirror_from_structured();

        let soothe3_state = Soothe3ParameterSet::new();

        proj.mixer_tracks[1].slots = vec![
            PluginSlotStateDto {
                slot_index: 0,
                enabled: true,
                wet_mix: 1.0,
                plugin: PluginConfigDto::ProQ4(ProQ4StateDto::from_parameter_set(&proq4_state)),
            },
            PluginSlotStateDto {
                slot_index: 1,
                enabled: true,
                wet_mix: 1.0,
                plugin: PluginConfigDto::Soothe3(Soothe3StateDto::from_parameter_set(&soothe3_state)),
            },
            PluginSlotStateDto {
                slot_index: 2,
                enabled: true,
                wet_mix: 1.0,
                plugin: PluginConfigDto::PatcherModular(
                    PatcherGraphStateDto::new_parallel_mastering_patcher(),
                ),
            },
        ];

        proj.mixer_tracks[MASTER_TRACK_INDEX].slots = vec![
            PluginSlotStateDto {
                slot_index: 0,
                enabled: true,
                wet_mix: 1.0,
                plugin: PluginConfigDto::RealTimeBsrnnStem(BsrnnStemStateDto::default()),
            },
            PluginSlotStateDto {
                slot_index: 1,
                enabled: true,
                wet_mix: 1.0,
                plugin: PluginConfigDto::Ozone12Master(Ozone12MasterConfigDto::default()),
            },
        ];

        proj.controller_links.push(ControllerLinkStateDto {
            controller_id: 0,
            target_track: 1,
            slot_index: 0,
            param_id: 2, // Band 0 Frequency
            min_out_norm: 0.2,
            max_out_norm: 0.8,
            inverted: false,
        });

        proj
    }

    /// Creates a 100% blank initial DAW session (`Untitled Project`):
    /// - Zero leftover Piano Roll patterns, zero Playlist clips, zero Automation clips, zero controller links,
    ///   zero audio assets, and transport stopped (`is_playing = false`).
    /// - Neutral default Mixer chains so workspace panels & GPU workers attach cleanly with flat `0.0 dB` EQ
    ///   and `[0.0; 4] dB` BSRNN unity bypass (`0` GPU hops / `0` latency when idle).
    pub fn new_blank_project(sample_rate: f64, bpm: f64) -> Self {
        let mut mixer_tracks = Vec::with_capacity(MAX_MIXER_TRACKS);
        for idx in 0..MAX_MIXER_TRACKS {
            mixer_tracks.push(MixerTrackStateDto::new_default(idx));
        }

        let proq4_state = ProQ4StateDto::from_parameter_set(&ProQ4ParameterSet::new());
        let soothe3_state = Soothe3StateDto::from_parameter_set(&Soothe3ParameterSet::new());
        mixer_tracks[1].slots = vec![
            PluginSlotStateDto {
                slot_index: 0,
                enabled: true,
                wet_mix: 1.0,
                plugin: PluginConfigDto::ProQ4(proq4_state),
            },
            PluginSlotStateDto {
                slot_index: 1,
                enabled: true,
                wet_mix: 1.0,
                plugin: PluginConfigDto::Soothe3(soothe3_state),
            },
            PluginSlotStateDto {
                slot_index: 2,
                enabled: true,
                wet_mix: 1.0,
                plugin: PluginConfigDto::PatcherModular(
                    PatcherGraphStateDto::new_parallel_mastering_patcher(),
                ),
            },
        ];

        mixer_tracks[MASTER_TRACK_INDEX].slots = vec![
            PluginSlotStateDto {
                slot_index: 0,
                enabled: true,
                wet_mix: 1.0,
                plugin: PluginConfigDto::RealTimeBsrnnStem(BsrnnStemStateDto::default()),
            },
            PluginSlotStateDto {
                slot_index: 1,
                enabled: true,
                wet_mix: 1.0,
                plugin: PluginConfigDto::Ozone12Master(Ozone12MasterConfigDto::default()),
            },
        ];

        Self {
            project_name: "Untitled Project".to_string(),
            schema_version: TONEFORGE_PROJECT_SCHEMA_VERSION,
            sample_rate: sample_rate.max(8000.0),
            max_block_size: 256,
            bpm: bpm.clamp(20.0, 999.0),
            ppq_resolution: daw_core_graph::DEFAULT_PPQ,
            is_playing: false,
            sample_pos: 0,
            ppq_tick: 0,
            process_all_502_tracks: false,
            mixer_tracks,
            patcher_graphs: Vec::new(),
            automation_clips: Vec::new(),
            controller_links: Vec::new(),
            piano_roll_patterns: Vec::new(),
            playlist_clips: Vec::new(),
            vocal_note_blobs: Vec::new(),
            audio_assets: Vec::new(),
            patcher_canvases: Vec::new(),
            fl26_fx_presets: Vec::new(),
        }
    }

    /// Validates the entire project state:
    /// - Exactly `502` Mixer Tracks (`0..502`)
    /// - Up to `10` plugin slots per track
    /// - Acyclic 502-track Mixer routing DAG
    /// - Acyclic modular `PatcherGraphNode` sub-DAGs
    /// - Valid `TensionSpline` control points (`x`, `y`, `tension` finite).
    pub fn validate(&self) -> Result<(), String> {
        if self.mixer_tracks.len() != MAX_MIXER_TRACKS {
            return Err(format!(
                "Invalid mixer_tracks count: {} (expected {})",
                self.mixer_tracks.len(),
                MAX_MIXER_TRACKS
            ));
        }
        if !(10.0..=999.0).contains(&self.bpm) || !self.bpm.is_finite() {
            return Err(format!("Invalid project BPM: {}", self.bpm));
        }
        if self.sample_rate < 8000.0 || !self.sample_rate.is_finite() {
            return Err(format!("Invalid sample_rate: {}", self.sample_rate));
        }

        // Verify track indices, slot counts, and check for DAG cycles using Kahn's algorithm
        let mut in_deg = vec![0usize; MAX_MIXER_TRACKS];
        let mut adj = vec![Vec::<usize>::new(); MAX_MIXER_TRACKS];

        for (idx, trk) in self.mixer_tracks.iter().enumerate() {
            if trk.index != idx {
                return Err(format!(
                    "Mixer track index mismatch at position {}: found {}",
                    idx, trk.index
                ));
            }
            if trk.slots.len() > MAX_INSERT_SLOTS {
                return Err(format!(
                    "Track {} has {} slots (max {})",
                    idx,
                    trk.slots.len(),
                    MAX_INSERT_SLOTS
                ));
            }
            for route in &trk.sends {
                if route.dest_track_index >= MAX_MIXER_TRACKS {
                    return Err(format!(
                        "Track {} routes to out-of-range destination {}",
                        idx, route.dest_track_index
                    ));
                }
                if route.dest_track_index == idx {
                    return Err(format!("Self-loop detected on track {}", idx));
                }
                if !adj[idx].contains(&route.dest_track_index) {
                    adj[idx].push(route.dest_track_index);
                    in_deg[route.dest_track_index] += 1;
                }
            }
            for slot in &trk.slots {
                if let PluginConfigDto::PatcherModular(ref patcher_dto) = slot.plugin {
                    let _ = patcher_dto.build_patcher_node(self.sample_rate, self.max_block_size)?;
                }
            }
        }

        let mut queue: Vec<usize> = (0..MAX_MIXER_TRACKS).filter(|&i| in_deg[i] == 0).collect();
        let mut visited = 0usize;
        let mut head = 0usize;
        while head < queue.len() {
            let u = queue[head];
            head += 1;
            visited += 1;
            for &v in &adj[u] {
                in_deg[v] -= 1;
                if in_deg[v] == 0 {
                    queue.push(v);
                }
            }
        }
        if visited < MAX_MIXER_TRACKS {
            return Err("Cycle detected in 502-Track Mixer routing graph".to_string());
        }

        for p_dto in &self.patcher_graphs {
            let _ = p_dto.build_patcher_node(self.sample_rate, self.max_block_size)?;
        }

        for clip in &self.automation_clips {
            for pt in &clip.points {
                if !pt.x.is_finite() || !pt.y.is_finite() || !pt.tension.is_finite() {
                    return Err(format!(
                        "Automation clip '{}' contains non-finite control point",
                        clip.name
                    ));
                }
            }
        }

        Ok(())
    }

    /// Evaluates an automation clip at `ppq_tick` and returns its 30-bit Fixed-Point value (`0..=0x40000000`).
    pub fn evaluate_automation_clip_fixed30(
        &self,
        clip_index: usize,
        ppq_tick: f64,
    ) -> Result<i32, String> {
        let clip = self
            .automation_clips
            .get(clip_index)
            .ok_or_else(|| format!("Automation clip index {} out of range", clip_index))?;
        let art = clip.to_articulator();
        Ok(art.evaluate_fixed30(ppq_tick))
    }

    /// Serializes the project state to a pretty-printed JSON string.
    pub fn to_json_pretty(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string_pretty(self).map_err(|e| format!("Failed to serialize project: {}", e))
    }

    /// Deserializes and validates a `ToneForgeProjectState` from a JSON string.
    pub fn from_json_str(json: &str) -> Result<Self, String> {
        let state: Self = serde_json::from_str(json)
            .map_err(|e| format!("Failed to deserialize ToneForgeProjectState: {}", e))?;
        state.validate()?;
        Ok(state)
    }

    /// Saves the project state to a file on disk.
    pub fn save_to_file(&self, path: &Path) -> Result<(), String> {
        let json = self.to_json_pretty()?;
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("Failed to create directory {:?}: {}", parent, e))?;
            }
        }
        std::fs::write(path, json)
            .map_err(|e| format!("Failed to write project file {:?}: {}", path, e))
    }

    /// Loads and validates a `ToneForgeProjectState` from a file on disk.
    pub fn load_from_file(path: &Path) -> Result<Self, String> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read project file {:?}: {}", path, e))?;
        Self::from_json_str(&json)
    }
}
