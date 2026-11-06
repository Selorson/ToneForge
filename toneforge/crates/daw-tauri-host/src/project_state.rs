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

/// Serializable 601-parameter FabFilter Pro-Q 4 state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProQ4StateDto {
    pub bands: Vec<EqBandParams>,
    pub global: ProQ4GlobalParams,
    pub normalized_601: Vec<f32>,
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

/// Union of all supported plugin configurations inside a Mixer Track FX Slot (`0..10`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "plugin_type", content = "config")]
pub enum PluginConfigDto {
    ProQ4(ProQ4StateDto),
    Soothe3(Soothe3StateDto),
    Ozone12Master(Ozone12MasterConfigDto),
    RealTimeBsrnnStem(BsrnnStemStateDto),
    PatcherModular(PatcherGraphStateDto),
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
    pub sends: Vec<MixerSendRouteDto>,
    pub slots: Vec<PluginSlotStateDto>,
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
            sends,
            slots: Vec::new(),
        }
    }
}

/// Complete ToneForge Project State (`ToneForgeProjectState`) managing 502 Mixer Tracks,
/// Patcher DAGs, Automation Clips with `TensionSpline`, 30-Bit Controller Bus links, and
/// all DSP/AI plugin chains (`Pro-Q 4`, `Soothe3`, `Ozone 12`, `RealTimeBsrnnStemNode`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToneForgeProjectState {
    pub project_name: String,
    pub schema_version: u32,
    pub sample_rate: f64,
    pub max_block_size: usize,
    pub bpm: f64,
    pub is_playing: bool,
    pub sample_pos: i64,
    pub ppq_tick: i64,
    pub process_all_502_tracks: bool,
    pub mixer_tracks: Vec<MixerTrackStateDto>,
    pub patcher_graphs: Vec<PatcherGraphStateDto>,
    pub automation_clips: Vec<AutomationClipStateDto>,
    pub controller_links: Vec<ControllerLinkStateDto>,
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
