//! Real-Time Audio Stream Coordinator (`audio_stream_controller.rs`).
//!
//! Manages the lifecycle of the 502-Track [`MixerEngine`], the wait-free SPSC command ring
//! [`rtrb::Producer<EngineCommand>`], the 128 KB [`ZeroCopyBinaryTelemetryStreamer`], preset & parameter
//! provisioning for **FabFilter Pro-Q 4** (`601` params), **oeksound Soothe3** (`143` params),
//! **iZotope Ozone 12 Master Chain**, **RealTimeBsrnnStemNode** (`120 ms` 4-stem rebalance),
//! and the offline **UVR5 Multi-Architecture Stem Separation & Ensemble Studio**
//! (`Mdx23cSeparator`, `HTDemucs4`, `CascadedASPPNet`, `SpectralEnsembler`).

use crate::ipc_commands::{DawAppState, ProQ4BandUpdateDto};
use crate::project_state::{
    BsrnnStemStateDto, MixerSendRouteDto, Ozone12MasterConfigDto, PluginConfigDto,
    PluginSlotStateDto, ProQ4StateDto, Soothe3StateDto, ToneForgeProjectState,
};
use crate::shm_telemetry_server::ZeroCopyBinaryTelemetryStreamer;
use ai_uvr5_stem_engine::{
    CascadedASPPNet, HTDemucs4, Mdx23cSeparator, RealTimeBsrnnStemNode, SpectralEnsembleMode,
    SpectralEnsembler,
};
use daw_core_graph::{
    ChannelGenerator, EmphasisLimiterNode, EngineCommand, GrossBeatTimeBufferNode,
    LuxeVerbFdnShimmerNode, MASTER_TRACK_INDEX, MAX_BLOCK_SIZE, MAX_INSERT_SLOTS,
    MAX_MIXER_TRACKS, MaximusMultibandNode, MixerEngine, ProcessContext, SparseCompressorNode,
    StereoFrame, TopologicalMixerScheduler, UnifiedDspNode, VISUAL_TELEMETRY_FRAME_BYTES,
    VisualTelemetryFrame,
};
use dsp_ozone12_mastering::Ozone12MasterChain;
use dsp_proq4_eq::{
    EqBandParams, FilterShape, PROQ4_BAND_PARAMS_TOTAL, PROQ4_NUM_BANDS, PROQ4_PARAMS_PER_BAND,
    PROQ4_TOTAL_PARAMS, ProQ4GlobalParams, ProQ4ParameterSet, ProQ4Processor,
};
use dsp_soothe3_spectral::params_143::{
    SOOTHE3_GLOBAL_PARAMS_COUNT, SOOTHE3_NUM_SC_BANDS, SOOTHE3_PARAMS_PER_SC_BAND,
};
use dsp_soothe3_spectral::{
    SOOTHE3_TOTAL_PARAMS, SidechainEqBandParams, Soothe3GlobalParams, Soothe3ParameterSet,
    Soothe3Processor,
};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

/// Zero-allocation looping stereo stem playback generator (`UnifiedDspNode`) used when UVR5
/// bounces separated stems directly onto Mixer Insert Tracks `1..=500`.
pub struct StemPlaybackGeneratorNode {
    stem_audio: Vec<StereoFrame>,
    cursor: usize,
    loop_playback: bool,
}

impl StemPlaybackGeneratorNode {
    pub fn new(stem_audio: Vec<StereoFrame>, loop_playback: bool) -> Self {
        Self {
            stem_audio,
            cursor: 0,
            loop_playback,
        }
    }
}

impl UnifiedDspNode for StemPlaybackGeneratorNode {
    fn name(&self) -> &'static str {
        "UVR5 Separated Stem Playback Generator"
    }

    fn latency_samples(&self) -> usize {
        0
    }

    fn prepare(&mut self, _sample_rate: f64, _max_block_size: usize) {}

    fn reset_state(&mut self) {
        self.cursor = 0;
    }

    fn set_parameter_normalized(&mut self, _param_id: u32, _norm_val: f32) {}

    fn process_slice(
        &mut self,
        ctx: &ProcessContext,
        main_io: &mut [StereoFrame],
        _sidechain_in: Option<&[StereoFrame]>,
        _aux_sends_out: &mut [&mut [StereoFrame]],
    ) {
        if !ctx.is_playing || self.stem_audio.is_empty() {
            main_io.fill([0.0, 0.0]);
            return;
        }
        let len = self.stem_audio.len();
        for frame in main_io.iter_mut() {
            if self.cursor < len {
                *frame = self.stem_audio[self.cursor];
                self.cursor += 1;
                if self.cursor >= len && self.loop_playback {
                    self.cursor = 0;
                }
            } else {
                *frame = [0.0, 0.0];
            }
        }
    }
}

/// Complete output of an offline UVR5 Multi-Architecture Ensemble Separation run.
#[derive(Clone, Debug)]
pub struct Uvr5EnsembleStudioOutput {
    /// Phase-coherent ensembled Vocals stem (`MDX23C` + `HTDemucs4` + `CascadedASPPNet`).
    pub vocals: Vec<StereoFrame>,
    /// Separated Drums stem (`HTDemucs4`).
    pub drums: Vec<StereoFrame>,
    /// Separated Bass stem (`HTDemucs4`).
    pub bass: Vec<StereoFrame>,
    /// Separated Other stem (`HTDemucs4`).
    pub other: Vec<StereoFrame>,
    /// Phase-inverted residual instrumental (`mix - vocals` via `SpectralEnsembler::invert_audio_waveform`).
    pub inverted_instrumental: Vec<StereoFrame>,
}

/// Real-Time Audio Stream Coordinator (`AudioStreamController`) managing the `MixerEngine`,
/// wait-free SPSC `rtrb::Producer<EngineCommand>`, 128 KB `ZeroCopyBinaryTelemetryStreamer`,
/// `ToneForgeProjectState`, and all DSP/AI engines.
pub struct AudioStreamController {
    pub app_state: DawAppState,
    pub telemetry_streamer: ZeroCopyBinaryTelemetryStreamer,
    pub mixer_engine: Box<MixerEngine>,
    pub project_state: ToneForgeProjectState,
    pub sample_rate: f64,
    pub max_block_size: usize,
    pub is_streaming: Arc<AtomicBool>,
    pub mdx23c_separator: Mdx23cSeparator,
    pub htdemucs4_separator: HTDemucs4,
    pub vr_aspp_separator: CascadedASPPNet,
    pub spectral_ensembler: SpectralEnsembler,
}

impl AudioStreamController {
    /// Creates a new `AudioStreamController` with a clean 502-Track `MixerEngine`.
    pub fn new(sample_rate: f64, max_block_size: usize, worker_threads: usize) -> Self {
        let sr = sample_rate.max(8000.0);
        let blk = max_block_size.clamp(16, MAX_BLOCK_SIZE);
        let bpm = 128.0;
        let (engine, cmd_prod, telem_out) = MixerEngine::new(sr, bpm, worker_threads);
        let mut project_state = ToneForgeProjectState::new_default(sr, bpm);
        project_state.max_block_size = blk;

        let mut controller = Self {
            app_state: DawAppState::new(cmd_prod),
            telemetry_streamer: ZeroCopyBinaryTelemetryStreamer::new(telem_out),
            mixer_engine: Box::new(engine),
            project_state,
            sample_rate: sr,
            max_block_size: blk,
            is_streaming: Arc::new(AtomicBool::new(true)),
            mdx23c_separator: Mdx23cSeparator::default(),
            htdemucs4_separator: HTDemucs4::default(),
            vr_aspp_separator: CascadedASPPNet::default(),
            spectral_ensembler: SpectralEnsembler::default(),
        };
        let _ = controller.mixer_engine.route_inserts_to_master(4);
        controller
    }

    /// Creates an `AudioStreamController` preloaded with the full ToneForge Mastering Template:
    /// - Track 1: `ProQ4Processor` (Slot 0) + `Soothe3Processor` (Slot 1) + `PatcherGraphNode` (Slot 2)
    /// - Track 0 (Master): `RealTimeBsrnnStemNode` (Slot 0) + `Ozone12MasterChain` (Slot 1)
    /// - Automation Clip 0 (`TensionSpline`) linked to Pro-Q 4 Band 0 Frequency on the 30-Bit Controller Bus.
    pub fn with_mastering_template(sample_rate: f64, max_block_size: usize) -> Result<Self, String> {
        let mut ctrl = Self::new(sample_rate, max_block_size, 0);
        let template = ToneForgeProjectState::new_mastering_template(sample_rate, 128.0);
        ctrl.apply_project_state(template)?;
        Ok(ctrl)
    }

    /// Pushes a command into the wait-free SPSC `rtrb::Producer<EngineCommand>`.
    /// If the ring buffer is temporarily full during a large multi-plugin preset load before an
    /// audio block has ticked, drains pending commands into `mixer_engine` so no command is lost.
    #[inline]
    pub fn push_command(&mut self, cmd: EngineCommand) -> Result<(), String> {
        if self.app_state.push_command(cmd).is_err() {
            self.mixer_engine.drain_commands();
            self.app_state.push_command(cmd)?;
        }
        Ok(())
    }

    /// Processes a single real-time audio block (`master_out.len() <= MAX_BLOCK_SIZE`) and publishes
    /// the updated 128 KB `VisualTelemetryFrame` to the `TripleBuffer`.
    #[inline]
    pub fn process_audio_block(&mut self, master_out: &mut [StereoFrame]) {
        self.mixer_engine.process_audio_block(master_out);
        self.project_state.sample_pos = self.mixer_engine.slicer.current_sample_pos;
        self.project_state.ppq_tick = self.mixer_engine.slicer.global_ppq_tick;
        self.project_state.bpm = self.mixer_engine.slicer.bpm;
        self.project_state.is_playing = self.mixer_engine.is_playing;
    }

    /// Polls the latest 128 KB (`131,072` B) binary telemetry frame as a zero-copy byte slice `&[u8; 131_072]`.
    #[inline]
    pub fn poll_binary_telemetry_bytes(&mut self) -> &[u8; VISUAL_TELEMETRY_FRAME_BYTES] {
        self.telemetry_streamer.poll_latest_bytes()
    }

    /// Polls the latest 128 KB (`131,072` B) structured [`VisualTelemetryFrame`].
    #[inline]
    pub fn poll_telemetry_frame(&mut self) -> &VisualTelemetryFrame {
        self.telemetry_streamer.poll_latest_frame()
    }

    // ========================================================================
    // 1. TRANSPORT, MIXER 502 ROUTING & AUTOMATION SPLINE CONTROL
    // ========================================================================

    pub fn set_transport(&mut self, playing: bool, bpm: f64) -> Result<(), String> {
        if !(10.0..=999.0).contains(&bpm) || !bpm.is_finite() {
            return Err(format!("Invalid tempo BPM: {}", bpm));
        }
        self.project_state.is_playing = playing;
        self.project_state.bpm = bpm;
        self.push_command(EngineCommand::SetTransportPlaying(playing))?;
        self.push_command(EngineCommand::SetBpm(bpm))
    }

    pub fn seek_position(&mut self, sample_pos: i64, ppq_tick: i64) -> Result<(), String> {
        if sample_pos < 0 || ppq_tick < 0 {
            return Err("Seek sample_pos and ppq_tick must be non-negative".to_string());
        }
        self.project_state.sample_pos = sample_pos;
        self.project_state.ppq_tick = ppq_tick;
        self.push_command(EngineCommand::SeekPosition {
            sample_pos,
            ppq_tick,
        })
    }

    /// Routes `src_track -> dst_track` in the 502-Track Mixer DAG with cycle detection and
    /// 5-Point APDC recalculation, plus queues the post-fader gain command.
    pub fn route_mixer_track(
        &mut self,
        src_track: u16,
        dst_track: u16,
        gain: f32,
        is_sidechain_only: bool,
    ) -> Result<(), String> {
        let src = src_track as usize;
        let dst = dst_track as usize;
        if src >= MAX_MIXER_TRACKS || dst >= MAX_MIXER_TRACKS {
            return Err(format!(
                "Track index out of range (0..{}): src={}, dst={}",
                MAX_MIXER_TRACKS, src, dst
            ));
        }
        if src == dst {
            return Err("Self-loop routing is prohibited".to_string());
        }
        let g = gain.clamp(0.0, 4.0);
        if TopologicalMixerScheduler::would_create_cycle(&self.mixer_engine.tracks, src, dst) {
            return Err(format!(
                "Cycle detected in 502-Track Mixer DAG when routing {} -> {}",
                src, dst
            ));
        }
        self.mixer_engine.tracks[src].add_send(dst, g, is_sidechain_only);
        self.mixer_engine
            .rebuild_graph()
            .map_err(|e| e.to_string())?;

        // Update project state mirror
        let sends = &mut self.project_state.mixer_tracks[src].sends;
        if let Some(existing) = sends
            .iter_mut()
            .find(|r| r.dest_track_index == dst && r.is_sidechain_only == is_sidechain_only)
        {
            existing.gain = g;
        } else {
            sends.push(MixerSendRouteDto {
                dest_track_index: dst,
                gain: g,
                is_sidechain_only,
            });
        }

        self.push_command(EngineCommand::SetTrackGain {
            track_index: src_track,
            gain_linear: self.mixer_engine.tracks[src].gain,
        })
    }

    pub fn set_track_gain(&mut self, track_index: u16, gain_linear: f32) -> Result<(), String> {
        let idx = track_index as usize;
        if idx >= MAX_MIXER_TRACKS {
            return Err(format!("Track index {} out of range (0..502)", idx));
        }
        let g = gain_linear.clamp(0.0, 4.0);
        self.project_state.mixer_tracks[idx].gain = g;
        self.push_command(EngineCommand::SetTrackGain {
            track_index,
            gain_linear: g,
        })
    }

    pub fn set_slot_wet_mix(
        &mut self,
        track_index: u16,
        slot_index: u8,
        wet_mix: f32,
    ) -> Result<(), String> {
        let trk = track_index as usize;
        let slt = slot_index as usize;
        if trk >= MAX_MIXER_TRACKS || slt >= MAX_INSERT_SLOTS {
            return Err("Track or slot index out of range".to_string());
        }
        let w = wet_mix.clamp(0.0, 1.0);
        if let Some(slot_dto) = self.project_state.mixer_tracks[trk].slots.get_mut(slt) {
            slot_dto.wet_mix = w;
        }
        self.push_command(EngineCommand::SetSlotWetMix {
            track_index,
            slot_index,
            wet_mix: w,
        })
    }

    pub fn set_slot_enabled(
        &mut self,
        track_index: u16,
        slot_index: u8,
        enabled: bool,
    ) -> Result<(), String> {
        let trk = track_index as usize;
        let slt = slot_index as usize;
        if trk >= MAX_MIXER_TRACKS || slt >= MAX_INSERT_SLOTS {
            return Err("Track or slot index out of range".to_string());
        }
        if let Some(slot_dto) = self.project_state.mixer_tracks[trk].slots.get_mut(slt) {
            slot_dto.enabled = enabled;
        }
        self.push_command(EngineCommand::SetSlotEnabled {
            track_index,
            slot_index,
            enabled,
        })
    }

    /// Evaluates all `AutomationClipStateDto`s (`TensionSpline` + `LfoModulator`) at `ppq_tick`,
    /// quantizes to FL Studio 26's 30-Bit Fixed-Point representation (`0..=0x40000000`), and pushes
    /// `EngineCommand::SetInternalControllerFixed30` into the real-time SPSC ring buffer.
    pub fn evaluate_and_publish_automation_clips(&mut self, ppq_tick: f64) -> Result<(), String> {
        let target_tick = ppq_tick.round() as i64;
        let target_sample_pos = self.mixer_engine.slicer.tick_to_sample_pos(ppq_tick);
        self.project_state.ppq_tick = target_tick;
        self.project_state.sample_pos = target_sample_pos;
        self.mixer_engine.slicer.seek(target_sample_pos, target_tick);

        let clips = self.project_state.automation_clips.clone();
        for clip in &clips {
            let art = clip.to_articulator();
            let fixed30 = art.evaluate_fixed30(ppq_tick);
            self.push_command(EngineCommand::SetInternalControllerFixed30 {
                controller_id: clip.controller_id,
                value_fixed30: fixed30,
            })?;
        }
        Ok(())
    }

    // ========================================================================
    // 2. FABFILTER PRO-Q 4 (601 PARAMETERS) PROVISIONING & CONTROL
    // ========================================================================

    /// Inserts a 24-Band `ProQ4Processor` on `track_index` and rebuilds 5-Point APDC.
    pub fn insert_proq4_on_track(
        &mut self,
        track_index: usize,
        wet_mix: f32,
    ) -> Result<usize, String> {
        if track_index >= MAX_MIXER_TRACKS {
            return Err(format!("Track index {} out of range", track_index));
        }
        let mut proc = ProQ4Processor::new();
        proc.instance_id = (track_index as u32) + 1;
        proc.prepare(self.sample_rate, self.max_block_size);
        let slot_idx = self.mixer_engine.tracks[track_index]
            .add_plugin_slot(Box::new(proc), wet_mix)
            .map_err(|e| e.to_string())?;
        self.mixer_engine.tracks[track_index].slots[slot_idx]
            .prepare(self.sample_rate, self.max_block_size);
        self.mixer_engine
            .rebuild_graph()
            .map_err(|e| e.to_string())?;

        self.project_state.mixer_tracks[track_index]
            .slots
            .push(PluginSlotStateDto {
                slot_index: slot_idx as u8,
                enabled: true,
                wet_mix: wet_mix.clamp(0.0, 1.0),
                plugin: PluginConfigDto::ProQ4(ProQ4StateDto::default()),
            });
        Ok(slot_idx)
    }

    /// Pushes all `601` normalized parameters of a `ProQ4ParameterSet` (`24 Bands × 24 Params + 25 Global`)
    /// into the lock-free `rtrb::Producer<EngineCommand>` queue.
    pub fn load_proq4_preset(
        &mut self,
        track_index: u16,
        slot_index: u8,
        preset: &ProQ4ParameterSet,
    ) -> Result<(), String> {
        let trk = track_index as usize;
        if trk >= MAX_MIXER_TRACKS || (slot_index as usize) >= MAX_INSERT_SLOTS {
            return Err("Track or slot index out of range".to_string());
        }
        for param_id in 0..PROQ4_TOTAL_PARAMS {
            let norm_val = preset.get_normalized(param_id as u32);
            self.push_command(EngineCommand::SetPluginParamNormalized {
                track_index,
                slot_index,
                param_id: param_id as u32,
                norm_val,
                target_sample_pos: None,
            })?;
        }
        if let Some(slot_dto) = self
            .project_state
            .mixer_tracks
            .get_mut(trk)
            .and_then(|t| t.slots.get_mut(slot_index as usize))
        {
            slot_dto.plugin = PluginConfigDto::ProQ4(ProQ4StateDto::from_parameter_set(preset));
        }
        Ok(())
    }

    /// Updates a single Pro-Q 4 band (`0..24`) using the exact `PROQ4_PARAMS_PER_BAND = 24` stride.
    pub fn update_proq4_band(&mut self, dto: ProQ4BandUpdateDto) -> Result<(), String> {
        if (dto.track_idx as usize) >= MAX_MIXER_TRACKS {
            return Err(format!("Track index {} out of range", dto.track_idx));
        }
        if (dto.slot_idx as usize) >= MAX_INSERT_SLOTS {
            return Err(format!("Slot index {} out of range", dto.slot_idx));
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
        let offsets_to_push: [usize; 11] = [0, 1, 2, 3, 4, 5, 6, 9, 10, 20, 21];
        for &off in &offsets_to_push {
            self.push_command(EngineCommand::SetPluginParamNormalized {
                track_index: dto.track_idx,
                slot_index: dto.slot_idx,
                param_id: base + (off as u32),
                norm_val: band.get_offset_normalized(off),
                target_sample_pos: None,
            })?;
        }

        // Mirror into project_state if slot holds ProQ4
        if let Some(slot_dto) = self
            .project_state
            .mixer_tracks
            .get_mut(dto.track_idx as usize)
            .and_then(|t| t.slots.get_mut(dto.slot_idx as usize))
        {
            if let PluginConfigDto::ProQ4(ref mut pq_dto) = slot_dto.plugin {
                if (dto.band_idx as usize) < pq_dto.bands.len() {
                    pq_dto.bands[dto.band_idx as usize] = band;
                }
            }
        }
        Ok(())
    }

    /// Updates the 25 Global Parameters (`576..601`) of Pro-Q 4 via the lock-free command queue.
    pub fn update_proq4_global(
        &mut self,
        track_index: u16,
        slot_index: u8,
        global: &ProQ4GlobalParams,
    ) -> Result<(), String> {
        let mut ps = ProQ4ParameterSet::new();
        ps.global = *global;
        ps.sync_mirror_from_structured();
        for param_id in PROQ4_BAND_PARAMS_TOTAL..PROQ4_TOTAL_PARAMS {
            self.push_command(EngineCommand::SetPluginParamNormalized {
                track_index,
                slot_index,
                param_id: param_id as u32,
                norm_val: ps.get_normalized(param_id as u32),
                target_sample_pos: None,
            })?;
        }
        if let Some(slot_dto) = self
            .project_state
            .mixer_tracks
            .get_mut(track_index as usize)
            .and_then(|t| t.slots.get_mut(slot_index as usize))
        {
            if let PluginConfigDto::ProQ4(ref mut pq_dto) = slot_dto.plugin {
                pq_dto.global = *global;
            }
        }
        Ok(())
    }

    // ========================================================================
    // 3. OEKSOUND SOOTHE3 (143 PARAMETERS) PROVISIONING & CONTROL
    // ========================================================================

    /// Inserts an `oeksound Soothe3Processor` on `track_index` and rebuilds 5-Point APDC.
    pub fn insert_soothe3_on_track(
        &mut self,
        track_index: usize,
        wet_mix: f32,
    ) -> Result<usize, String> {
        if track_index >= MAX_MIXER_TRACKS {
            return Err(format!("Track index {} out of range", track_index));
        }
        let mut proc = Soothe3Processor::new(self.sample_rate as f32);
        proc.prepare(self.sample_rate, self.max_block_size);
        let slot_idx = self.mixer_engine.tracks[track_index]
            .add_plugin_slot(Box::new(proc), wet_mix)
            .map_err(|e| e.to_string())?;
        self.mixer_engine.tracks[track_index].slots[slot_idx]
            .prepare(self.sample_rate, self.max_block_size);
        self.mixer_engine
            .rebuild_graph()
            .map_err(|e| e.to_string())?;

        self.project_state.mixer_tracks[track_index]
            .slots
            .push(PluginSlotStateDto {
                slot_index: slot_idx as u8,
                enabled: true,
                wet_mix: wet_mix.clamp(0.0, 1.0),
                plugin: PluginConfigDto::Soothe3(Soothe3StateDto::default()),
            });
        Ok(slot_idx)
    }

    /// Loads an oeksound Soothe3 `.preset` JSON string and dispatches all `143` normalized
    /// parameters (`26 Global + 72 Sidechain EQ + 45 Atmos`) via the `rtrb` SPSC command ring.
    pub fn load_soothe3_preset_json(
        &mut self,
        track_index: u16,
        slot_index: u8,
        preset_json: &str,
    ) -> Result<usize, String> {
        let mut ps = Soothe3ParameterSet::new();
        let matched_count = ps.load_from_preset_json(preset_json)?;
        self.load_soothe3_parameter_set(track_index, slot_index, &ps)?;
        Ok(matched_count)
    }

    /// Pushes all `143` normalized parameters of a `Soothe3ParameterSet` into the `rtrb` command queue.
    pub fn load_soothe3_parameter_set(
        &mut self,
        track_index: u16,
        slot_index: u8,
        params: &Soothe3ParameterSet,
    ) -> Result<(), String> {
        let trk = track_index as usize;
        if trk >= MAX_MIXER_TRACKS || (slot_index as usize) >= MAX_INSERT_SLOTS {
            return Err("Track or slot index out of range".to_string());
        }
        let mut cloned = params.clone();
        cloned.sync_mirror_from_structured();
        for param_id in 0..SOOTHE3_TOTAL_PARAMS {
            self.push_command(EngineCommand::SetPluginParamNormalized {
                track_index,
                slot_index,
                param_id: param_id as u32,
                norm_val: cloned.get_normalized(param_id as u32),
                target_sample_pos: None,
            })?;
        }
        if let Some(slot_dto) = self
            .project_state
            .mixer_tracks
            .get_mut(trk)
            .and_then(|t| t.slots.get_mut(slot_index as usize))
        {
            slot_dto.plugin = PluginConfigDto::Soothe3(Soothe3StateDto::from_parameter_set(&cloned));
        }
        Ok(())
    }

    /// Updates Soothe3's 26 Global Parameters (`0..26`) via the lock-free SPSC queue.
    pub fn update_soothe3_global(
        &mut self,
        track_index: u16,
        slot_index: u8,
        global: &Soothe3GlobalParams,
    ) -> Result<(), String> {
        let mut ps = Soothe3ParameterSet::new();
        ps.global = *global;
        ps.sync_mirror_from_structured();
        for param_id in 0..SOOTHE3_GLOBAL_PARAMS_COUNT {
            self.push_command(EngineCommand::SetPluginParamNormalized {
                track_index,
                slot_index,
                param_id: param_id as u32,
                norm_val: ps.get_normalized(param_id as u32),
                target_sample_pos: None,
            })?;
        }
        if let Some(slot_dto) = self
            .project_state
            .mixer_tracks
            .get_mut(track_index as usize)
            .and_then(|t| t.slots.get_mut(slot_index as usize))
        {
            if let PluginConfigDto::Soothe3(ref mut s3_dto) = slot_dto.plugin {
                s3_dto.global = *global;
            }
        }
        Ok(())
    }

    /// Updates one of Soothe3's 8 Sidechain EQ Bands (`band_idx in 0..8`, 9 params per band).
    pub fn update_soothe3_sc_band(
        &mut self,
        track_index: u16,
        slot_index: u8,
        band_idx: u8,
        band: &SidechainEqBandParams,
    ) -> Result<(), String> {
        let b = band_idx as usize;
        if b >= SOOTHE3_NUM_SC_BANDS {
            return Err(format!("Soothe3 SC band index {} out of range (0..8)", b));
        }
        let base = SOOTHE3_GLOBAL_PARAMS_COUNT + b * SOOTHE3_PARAMS_PER_SC_BAND;
        for off in 0..SOOTHE3_PARAMS_PER_SC_BAND {
            self.push_command(EngineCommand::SetPluginParamNormalized {
                track_index,
                slot_index,
                param_id: (base + off) as u32,
                norm_val: band.get_offset_normalized(off),
                target_sample_pos: None,
            })?;
        }
        if let Some(slot_dto) = self
            .project_state
            .mixer_tracks
            .get_mut(track_index as usize)
            .and_then(|t| t.slots.get_mut(slot_index as usize))
        {
            if let PluginConfigDto::Soothe3(ref mut s3_dto) = slot_dto.plugin {
                if b < s3_dto.sc_bands.len() {
                    s3_dto.sc_bands[b] = *band;
                }
            }
        }
        Ok(())
    }

    // ========================================================================
    // 4. IZOTOPE OZONE 12 MASTER CHAIN & REALTIME 120MS BSRNN STEM REBALANCE
    // ========================================================================

    /// Inserts an `Ozone12MasterChain` on `track_index` (typically `MASTER_TRACK_INDEX = 0`).
    pub fn insert_ozone12_on_track(
        &mut self,
        track_index: usize,
        wet_mix: f32,
    ) -> Result<usize, String> {
        if track_index >= MAX_MIXER_TRACKS {
            return Err(format!("Track index {} out of range", track_index));
        }
        let mut chain = Ozone12MasterChain::new(self.sample_rate);
        chain.prepare(self.sample_rate, self.max_block_size);
        let slot_idx = self.mixer_engine.tracks[track_index]
            .add_plugin_slot(Box::new(chain), wet_mix)
            .map_err(|e| e.to_string())?;
        self.mixer_engine.tracks[track_index].slots[slot_idx]
            .prepare(self.sample_rate, self.max_block_size);
        self.mixer_engine
            .rebuild_graph()
            .map_err(|e| e.to_string())?;

        self.project_state.mixer_tracks[track_index]
            .slots
            .push(PluginSlotStateDto {
                slot_index: slot_idx as u8,
                enabled: true,
                wet_mix: wet_mix.clamp(0.0, 1.0),
                plugin: PluginConfigDto::Ozone12Master(Ozone12MasterConfigDto::default()),
            });
        Ok(slot_idx)
    }

    /// Updates an `Ozone12MasterChain` slot with `Ozone12MasterConfigDto` via the `rtrb` command queue.
    pub fn configure_ozone12_master(
        &mut self,
        track_index: u16,
        slot_index: u8,
        cfg: &Ozone12MasterConfigDto,
    ) -> Result<(), String> {
        let trk = track_index as usize;
        if trk >= MAX_MIXER_TRACKS || (slot_index as usize) >= MAX_INSERT_SLOTS {
            return Err("Track or slot index out of range".to_string());
        }

        // Maximizer parameters (0..=9, 65, 66)
        let max_params: [(u32, f32); 11] = [
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
            (4, 0.5),
        ];
        for (pid, val) in max_params {
            self.push_command(EngineCommand::SetPluginParamNormalized {
                track_index,
                slot_index,
                param_id: pid,
                norm_val: val,
                target_sample_pos: None,
            })?;
        }

        // 12AX7 DK-Triode Exciter parameters (10..=22)
        self.push_command(EngineCommand::SetPluginParamNormalized {
            track_index,
            slot_index,
            param_id: 10,
            norm_val: if cfg.exciter_bypass { 1.0 } else { 0.0 },
            target_sample_pos: None,
        })?;
        for b in 0..4 {
            let base = 11 + (b as u32) * 3;
            self.push_command(EngineCommand::SetPluginParamNormalized {
                track_index,
                slot_index,
                param_id: base,
                norm_val: (cfg.exciter_band_drives_pct[b] * 0.01).clamp(0.0, 1.0),
                target_sample_pos: None,
            })?;
            self.push_command(EngineCommand::SetPluginParamNormalized {
                track_index,
                slot_index,
                param_id: base + 1,
                norm_val: cfg.exciter_band_mixes[b].clamp(0.0, 1.0),
                target_sample_pos: None,
            })?;
            self.push_command(EngineCommand::SetPluginParamNormalized {
                track_index,
                slot_index,
                param_id: base + 2,
                norm_val: ((cfg.exciter_band_modes[b].min(6) as f32) / 6.0).clamp(0.0, 1.0),
                target_sample_pos: None,
            })?;
        }

        // Multiband Imager parameters (25..=34)
        for b in 0..4 {
            self.push_command(EngineCommand::SetPluginParamNormalized {
                track_index,
                slot_index,
                param_id: 25 + (b as u32),
                norm_val: ((cfg.imager_band_widths_pct[b] + 100.0) / 200.0).clamp(0.0, 1.0),
                target_sample_pos: None,
            })?;
            self.push_command(EngineCommand::SetPluginParamNormalized {
                track_index,
                slot_index,
                param_id: 29 + (b as u32),
                norm_val: (cfg.imager_band_stereoize_pct[b] * 0.01).clamp(0.0, 1.0),
                target_sample_pos: None,
            })?;
        }
        self.push_command(EngineCommand::SetPluginParamNormalized {
            track_index,
            slot_index,
            param_id: 33,
            norm_val: if cfg.imager_stereoizer_enabled { 1.0 } else { 0.0 },
            target_sample_pos: None,
        })?;
        self.push_command(EngineCommand::SetPluginParamNormalized {
            track_index,
            slot_index,
            param_id: 34,
            norm_val: if cfg.imager_stereoizer_mode > 0 { 1.0 } else { 0.0 },
            target_sample_pos: None,
        })?;

        // Stabilizer parameters (36..=40)
        let stab_params: [(u32, f32); 4] = [
            (36, (cfg.stabilizer_amount_pct * 0.01).clamp(0.0, 1.0)),
            (37, (cfg.stabilizer_speed_pct * 0.01).clamp(0.0, 1.0)),
            (
                39,
                (cfg.stabilizer_tame_transients_pct * 0.01).clamp(0.0, 1.0),
            ),
            (40, if cfg.stabilizer_cut_mode { 1.0 } else { 0.0 }),
        ];
        for (pid, val) in stab_params {
            self.push_command(EngineCommand::SetPluginParamNormalized {
                track_index,
                slot_index,
                param_id: pid,
                norm_val: val,
                target_sample_pos: None,
            })?;
        }

        // Impact & Clarity parameters (45..=48, 55..=56)
        for b in 0..4 {
            self.push_command(EngineCommand::SetPluginParamNormalized {
                track_index,
                slot_index,
                param_id: 45 + (b as u32),
                norm_val: ((cfg.impact_band_amounts[b] + 100.0) / 200.0).clamp(0.0, 1.0),
                target_sample_pos: None,
            })?;
        }
        self.push_command(EngineCommand::SetPluginParamNormalized {
            track_index,
            slot_index,
            param_id: 55,
            norm_val: (cfg.clarity_amount_pct * 0.01).clamp(0.0, 1.0),
            target_sample_pos: None,
        })?;
        self.push_command(EngineCommand::SetPluginParamNormalized {
            track_index,
            slot_index,
            param_id: 56,
            norm_val: ((cfg.clarity_tilt_db_oct + 6.0) / 12.0).clamp(0.0, 1.0),
            target_sample_pos: None,
        })?;

        if let Some(slot_dto) = self
            .project_state
            .mixer_tracks
            .get_mut(trk)
            .and_then(|t| t.slots.get_mut(slot_index as usize))
        {
            slot_dto.plugin = PluginConfigDto::Ozone12Master(cfg.clone());
        }
        Ok(())
    }

    /// Inserts a `RealTimeBsrnnStemNode` (`120 ms` = `5,760` samples @ 48 kHz lookahead APDC)
    /// onto `track_index` and rebuilds the 5-Point APDC topological schedule.
    pub fn insert_realtime_bsrnn_on_track(
        &mut self,
        track_index: usize,
        wet_mix: f32,
    ) -> Result<usize, String> {
        if track_index >= MAX_MIXER_TRACKS {
            return Err(format!("Track index {} out of range", track_index));
        }
        let mut node = RealTimeBsrnnStemNode::new(self.sample_rate);
        node.prepare(self.sample_rate, self.max_block_size);
        let slot_idx = self.mixer_engine.tracks[track_index]
            .add_plugin_slot(Box::new(node), wet_mix)
            .map_err(|e| e.to_string())?;
        self.mixer_engine.tracks[track_index].slots[slot_idx]
            .prepare(self.sample_rate, self.max_block_size);
        self.mixer_engine
            .rebuild_graph()
            .map_err(|e| e.to_string())?;

        self.project_state.mixer_tracks[track_index]
            .slots
            .push(PluginSlotStateDto {
                slot_index: slot_idx as u8,
                enabled: true,
                wet_mix: wet_mix.clamp(0.0, 1.0),
                plugin: PluginConfigDto::RealTimeBsrnnStem(BsrnnStemStateDto::default()),
            });
        Ok(slot_idx)
    }

    /// Updates the 4-Stem BSRNN Master Rebalance gains (`[Vocals, Drums, Bass, Other]` in `-12..+12 dB`)
    /// via the lock-free `rtrb` command queue.
    pub fn update_bsrnn_stem_gains(
        &mut self,
        track_index: u16,
        slot_index: u8,
        stem_gains_db: [f32; 4],
    ) -> Result<(), String> {
        let trk = track_index as usize;
        if trk >= MAX_MIXER_TRACKS || (slot_index as usize) >= MAX_INSERT_SLOTS {
            return Err("Track or slot index out of range".to_string());
        }
        for s in 0..4 {
            let norm_val = ((stem_gains_db[s].clamp(-12.0, 12.0) + 12.0) / 24.0).clamp(0.0, 1.0);
            self.push_command(EngineCommand::SetPluginParamNormalized {
                track_index,
                slot_index,
                param_id: s as u32,
                norm_val,
                target_sample_pos: None,
            })?;
        }
        if let Some(slot_dto) = self
            .project_state
            .mixer_tracks
            .get_mut(trk)
            .and_then(|t| t.slots.get_mut(slot_index as usize))
        {
            if let PluginConfigDto::RealTimeBsrnnStem(ref mut bsrnn_dto) = slot_dto.plugin {
                bsrnn_dto.stem_gains_db = stem_gains_db;
            }
        }
        Ok(())
    }

    /// Updates the full `BsrnnStemStateDto` (`stem_gains_db`, `stem_solo`, `stem_mute`) via the
    /// lock-free `rtrb` command queue (`param_id = 0..=11`).
    pub fn update_bsrnn_stem_state(
        &mut self,
        track_index: u16,
        slot_index: u8,
        dto: &BsrnnStemStateDto,
    ) -> Result<(), String> {
        self.update_bsrnn_stem_gains(track_index, slot_index, dto.stem_gains_db)?;
        for s in 0..4 {
            self.push_command(EngineCommand::SetPluginParamNormalized {
                track_index,
                slot_index,
                param_id: (4 + s) as u32,
                norm_val: if dto.stem_solo[s] { 1.0 } else { 0.0 },
                target_sample_pos: None,
            })?;
            self.push_command(EngineCommand::SetPluginParamNormalized {
                track_index,
                slot_index,
                param_id: (8 + s) as u32,
                norm_val: if dto.stem_mute[s] { 1.0 } else { 0.0 },
                target_sample_pos: None,
            })?;
        }
        if let Some(slot_dto) = self
            .project_state
            .mixer_tracks
            .get_mut(track_index as usize)
            .and_then(|t| t.slots.get_mut(slot_index as usize))
        {
            if let PluginConfigDto::RealTimeBsrnnStem(ref mut bsrnn_dto) = slot_dto.plugin {
                *bsrnn_dto = dto.clone();
            }
        }
        Ok(())
    }

    // ========================================================================
    // 5. OFFLINE UVR5 MULTI-ARCHITECTURE STEM SEPARATION & ENSEMBLE STUDIO
    // ========================================================================

    /// Runs offline MDX23C (`TFC-TDF v3`) separation returning `(primary_stem, complement_stem)`.
    pub fn separate_mdx23c_offline(
        &mut self,
        mix: &[StereoFrame],
        enable_denoise_2pass: bool,
    ) -> (Vec<StereoFrame>, Vec<StereoFrame>) {
        self.mdx23c_separator.enable_denoise_2pass = enable_denoise_2pass;
        self.mdx23c_separator.separate_offline(mix)
    }

    /// Runs offline Hybrid Transformer Demucs v4 (`HTDemucs4`) 4-stem separation
    /// returning `[Drums, Bass, Other, Vocals]`.
    pub fn separate_htdemucs4_offline(&self, mix: &[StereoFrame]) -> [Vec<StereoFrame>; 4] {
        self.htdemucs4_separator.separate_4stems(mix)
    }

    /// Runs offline VR `CascadedASPPNet` (`dilations = [1, 2, 4, 8, 16]`) separation
    /// returning `(vocal_stem, instrumental_stem)`.
    pub fn separate_vr_aspp_offline(
        &mut self,
        mix: &[StereoFrame],
        aggressiveness: f32,
    ) -> (Vec<StereoFrame>, Vec<StereoFrame>) {
        self.vr_aspp_separator.aggressiveness = aggressiveness.clamp(-1.0, 1.0);
        self.vr_aspp_separator.separate_vocal_instrumental(mix)
    }

    /// Executes the full UVR5 Multi-Model Ensemble Studio pipeline across `Mdx23cSeparator`,
    /// `HTDemucs4`, and `CascadedASPPNet`, combining the vocal stems in the complex STFT domain
    /// via `SpectralEnsembler` (`MaxSpec`, `MinSpec`, `Average`, `Median`, or `Weighted`) and
    /// computing the phase-inverted instrumental residual (`invert_audio_waveform`).
    pub fn run_uvr5_ensemble_studio(
        &mut self,
        mix: &[StereoFrame],
        mode: SpectralEnsembleMode,
        weights: Option<&[f32]>,
        phase_denoise_strength: f32,
    ) -> Uvr5EnsembleStudioOutput {
        if mix.is_empty() {
            return Uvr5EnsembleStudioOutput {
                vocals: Vec::new(),
                drums: Vec::new(),
                bass: Vec::new(),
                other: Vec::new(),
                inverted_instrumental: Vec::new(),
            };
        }

        let (mdx_vocals, _mdx_inst) = self.mdx23c_separator.separate_offline(mix);
        let [demucs_drums, demucs_bass, demucs_other, demucs_vocals] =
            self.htdemucs4_separator.separate_4stems(mix);
        let (vr_vocals, _vr_inst) = self.vr_aspp_separator.separate_vocal_instrumental(mix);

        self.spectral_ensembler.mode = mode;
        self.spectral_ensembler.phase_denoise_strength = phase_denoise_strength.clamp(0.0, 1.0);

        let vocal_models: [&[StereoFrame]; 3] = [
            mdx_vocals.as_slice(),
            demucs_vocals.as_slice(),
            vr_vocals.as_slice(),
        ];
        let ensembled_vocals = self
            .spectral_ensembler
            .ensemble_waveforms(&vocal_models, weights);
        let inverted_instrumental =
            SpectralEnsembler::invert_audio_waveform(mix, &ensembled_vocals);

        Uvr5EnsembleStudioOutput {
            vocals: ensembled_vocals,
            drums: demucs_drums,
            bass: demucs_bass,
            other: demucs_other,
            inverted_instrumental,
        }
    }

    /// Bounces separated stereo stems directly onto Mixer Insert Tracks (`target_tracks`) as
    /// zero-allocation `StemPlaybackGeneratorNode` channels (replacing any prior stem channels on those tracks),
    /// routes those tracks to Master (`0`), and rebuilds the 5-Point APDC topological schedule.
    pub fn bounce_stems_to_mixer_tracks(
        &mut self,
        stems: &[Vec<StereoFrame>],
        target_tracks: &[usize],
    ) -> Result<Vec<usize>, String> {
        Self::bounce_stems_to_engine(
            &mut self.mixer_engine,
            stems,
            target_tracks,
            self.sample_rate,
            self.max_block_size,
        )
    }

    /// Replaces any existing stem channels on `target_tracks` (and clears any prior custom audio stem on Track 1 when
    /// bouncing to Tracks 1..4) and attaches new `StemPlaybackGeneratorNode` channels onto `engine`.
    pub fn bounce_stems_to_engine(
        engine: &mut MixerEngine,
        stems: &[Vec<StereoFrame>],
        target_tracks: &[usize],
        sample_rate: f64,
        max_block_size: usize,
    ) -> Result<Vec<usize>, String> {
        let count = stems.len().min(target_tracks.len());
        for i in 0..count {
            let trk = target_tracks[i];
            if trk == 0 || trk >= MAX_MIXER_TRACKS {
                return Err(format!(
                    "Target stem insert track {} must be in 1..{}",
                    trk, MAX_MIXER_TRACKS
                ));
            }
        }
        engine.remove_stem_channels_on_tracks(&target_tracks[..count]);

        let mut channel_indices = Vec::with_capacity(count);
        for i in 0..count {
            let trk = target_tracks[i];
            let gen_node = Box::new(StemPlaybackGeneratorNode::new(stems[i].clone(), true));
            let ch_id = 100 + (trk as u32);
            let mut ch = ChannelGenerator::new(ch_id, trk, gen_node);
            ch.prepare(sample_rate, max_block_size);
            let ch_idx = engine
                .add_channel_generator(ch)
                .map_err(|e| e.to_string())?;
            channel_indices.push(ch_idx);

            // Ensure target insert track routes to Master Track 0 if no route exists yet
            if engine.tracks[trk].sends.is_empty() {
                engine.tracks[trk].add_send(MASTER_TRACK_INDEX, 1.0, false);
            }
        }
        engine.rebuild_graph().map_err(|e| e.to_string())?;

        Ok(channel_indices)
    }

    // ========================================================================
    // ========================================================================
    // 6. PROJECT STATE RECONSTRUCTION (`ToneForgeProjectState` <-> `MixerEngine`)
    // ========================================================================

    #[inline(never)]
    fn build_proq4_boxed(
        idx: usize,
        sr: f64,
        blk: usize,
        pq_dto: &ProQ4StateDto,
    ) -> Box<dyn UnifiedDspNode> {
        let mut proc = Box::new(ProQ4Processor::new());
        proc.instance_id = (idx as u32) + 1;
        proc.prepare(sr, blk);
        let ps = pq_dto.to_parameter_set();
        for pid in 0..PROQ4_TOTAL_PARAMS {
            proc.set_parameter_normalized(pid as u32, ps.get_normalized(pid as u32));
        }
        proc
    }

    #[inline(never)]
    fn build_soothe3_boxed(
        sr: f64,
        blk: usize,
        s3_dto: &Soothe3StateDto,
    ) -> Box<dyn UnifiedDspNode> {
        let mut proc = Box::new(Soothe3Processor::new(sr as f32));
        proc.prepare(sr, blk);
        let ps = s3_dto.to_parameter_set();
        for pid in 0..SOOTHE3_TOTAL_PARAMS {
            proc.set_parameter_normalized(pid as u32, ps.get_normalized(pid as u32));
        }
        proc
    }

    #[inline(never)]
    fn build_ozone12_boxed(
        sr: f64,
        blk: usize,
        oz_dto: &Ozone12MasterConfigDto,
    ) -> Box<dyn UnifiedDspNode> {
        let mut chain = Box::new(Ozone12MasterChain::new(sr));
        chain.prepare(sr, blk);
        let updates: [(u32, f32); 20] = [
            (0, if oz_dto.maximizer_bypass { 1.0 } else { 0.0 }),
            (
                1,
                ((oz_dto.maximizer_gain_db + 20.0) / 40.0).clamp(0.0, 1.0),
            ),
            (
                2,
                ((oz_dto.maximizer_ceiling_dbtp + 20.0) / 20.0).clamp(0.0, 1.0),
            ),
            (5, oz_dto.maximizer_upward_comp_wet.clamp(0.0, 1.0)),
            (
                6,
                if oz_dto.maximizer_soft_clip_enabled {
                    1.0
                } else {
                    0.0
                },
            ),
            (7, oz_dto.maximizer_soft_clip_mix.clamp(0.0, 1.0)),
            (8, (oz_dto.maximizer_soft_clip_mode.min(2) as f32) * 0.5),
            (
                9,
                ((oz_dto.maximizer_transient_shaping + 1.0) * 0.5).clamp(0.0, 1.0),
            ),
            (65, (oz_dto.maximizer_irc_mode.min(8) as f32) / 8.0),
            (66, if oz_dto.maximizer_true_peak { 1.0 } else { 0.0 }),
            (10, if oz_dto.exciter_bypass { 1.0 } else { 0.0 }),
            (33, if oz_dto.imager_stereoizer_enabled { 1.0 } else { 0.0 }),
            (
                34,
                if oz_dto.imager_stereoizer_mode > 0 {
                    1.0
                } else {
                    0.0
                },
            ),
            (36, (oz_dto.stabilizer_amount_pct * 0.01).clamp(0.0, 1.0)),
            (37, (oz_dto.stabilizer_speed_pct * 0.01).clamp(0.0, 1.0)),
            (
                39,
                (oz_dto.stabilizer_tame_transients_pct * 0.01).clamp(0.0, 1.0),
            ),
            (40, if oz_dto.stabilizer_cut_mode { 1.0 } else { 0.0 }),
            (55, (oz_dto.clarity_amount_pct * 0.01).clamp(0.0, 1.0)),
            (
                56,
                ((oz_dto.clarity_tilt_db_oct + 6.0) / 12.0).clamp(0.0, 1.0),
            ),
            (4, 0.5),
        ];
        for (pid, val) in updates {
            chain.set_parameter_normalized(pid, val);
        }
        for b in 0..4 {
            let exc_base = 11 + (b as u32) * 3;
            chain.set_parameter_normalized(
                exc_base,
                (oz_dto.exciter_band_drives_pct[b] * 0.01).clamp(0.0, 1.0),
            );
            chain.set_parameter_normalized(exc_base + 1, oz_dto.exciter_band_mixes[b].clamp(0.0, 1.0));
            chain.set_parameter_normalized(
                exc_base + 2,
                ((oz_dto.exciter_band_modes[b].min(6) as f32) / 6.0).clamp(0.0, 1.0),
            );
            chain.set_parameter_normalized(
                25 + (b as u32),
                ((oz_dto.imager_band_widths_pct[b] + 100.0) / 200.0).clamp(0.0, 1.0),
            );
            chain.set_parameter_normalized(
                29 + (b as u32),
                (oz_dto.imager_band_stereoize_pct[b] * 0.01).clamp(0.0, 1.0),
            );
            chain.set_parameter_normalized(
                45 + (b as u32),
                ((oz_dto.impact_band_amounts[b] + 100.0) / 200.0).clamp(0.0, 1.0),
            );
        }
        chain
    }

    #[inline(never)]
    fn build_bsrnn_boxed(
        sr: f64,
        blk: usize,
        bsrnn_dto: &BsrnnStemStateDto,
    ) -> Box<dyn UnifiedDspNode> {
        let mut node = Box::new(RealTimeBsrnnStemNode::new(sr));
        node.prepare(sr, blk);
        node.stem_gains_db = bsrnn_dto.stem_gains_db;
        node.stem_solo = bsrnn_dto.stem_solo;
        node.stem_mute = bsrnn_dto.stem_mute;
        node
    }

    /// Applies a complete [`ToneForgeProjectState`] to this `AudioStreamController`, rebuilding
    /// all 502 Mixer Tracks, routing edges, FX plugin chains (`ProQ4`, `Soothe3`, `Ozone12MasterChain`,
    /// `RealTimeBsrnnStemNode`, `PatcherGraphNode`, Native FX), `ControllerLink`s on the 30-Bit
    /// Fixed-Point Controller Bus, and the 5-Point APDC topological schedule in-place.
    pub fn apply_project_state(&mut self, state: ToneForgeProjectState) -> Result<(), String> {
        state.validate()?;

        let sr = state.sample_rate.max(8000.0);
        let blk = state.max_block_size.clamp(16, MAX_BLOCK_SIZE);

        self.mixer_engine.drain_commands();
        self.mixer_engine.is_playing = state.is_playing;
        self.mixer_engine.process_all_502_tracks = state.process_all_502_tracks;
        self.mixer_engine.slicer.set_sample_rate(sr);
        self.mixer_engine.slicer.set_bpm(state.bpm);
        self.mixer_engine.slicer.set_ppq(state.ppq_resolution.max(24));
        self.mixer_engine.slicer.seek(state.sample_pos, state.ppq_tick);
        self.mixer_engine.channels.clear();
        self.mixer_engine.controller_bus = daw_core_graph::InternalControllerBus::new();
        self.mixer_engine.sequencer = daw_core_graph::PlaylistSequencer::new();
        self.mixer_engine.sequencer.auto_evaluate_without_clips =
            !state.playlist_clips.is_empty() || !state.piano_roll_patterns.is_empty();

        for pat_dto in &state.piano_roll_patterns {
            let _ = self.mixer_engine.sequencer.upsert_pattern(pat_dto.to_pattern());
        }
        for clip_dto in &state.playlist_clips {
            let _ = self
                .mixer_engine
                .sequencer
                .upsert_clip(clip_dto.to_playlist_clip());
        }
        for auto_dto in &state.automation_clips {
            let _ = self
                .mixer_engine
                .sequencer
                .upsert_automation_articulator(auto_dto.to_articulator());
        }

        // Configure all 502 Mixer Tracks in-place
        for trk_dto in &state.mixer_tracks {
            let idx = trk_dto.index;
            let track = &mut self.mixer_engine.tracks[idx];
            track.gain = trk_dto.gain;
            track.polarity_inverted = trk_dto.polarity_inverted;
            track.has_hw_input = trk_dto.has_hw_input;
            track.hw_input_latency_samples = trk_dto.hw_input_latency_samples;
            track.sends.clear();
            for send_dto in &trk_dto.sends {
                track.add_send(
                    send_dto.dest_track_index,
                    send_dto.gain,
                    send_dto.is_sidechain_only,
                );
            }
            track.slots.clear();
            for slot_dto in &trk_dto.slots {
                let plugin_box: Box<dyn UnifiedDspNode> = match &slot_dto.plugin {
                    PluginConfigDto::ProQ4(pq_dto) => Self::build_proq4_boxed(idx, sr, blk, pq_dto),
                    PluginConfigDto::Soothe3(s3_dto) => Self::build_soothe3_boxed(sr, blk, s3_dto),
                    PluginConfigDto::Ozone12Master(oz_dto) => {
                        Self::build_ozone12_boxed(sr, blk, oz_dto)
                    }
                    PluginConfigDto::RealTimeBsrnnStem(bsrnn_dto) => {
                        Self::build_bsrnn_boxed(sr, blk, bsrnn_dto)
                    }
                    PluginConfigDto::PatcherModular(patcher_dto) => {
                        let mut patcher = patcher_dto.build_patcher_node(sr, blk)?;
                        patcher.prepare(sr, blk);
                        Box::new(patcher)
                    }
                    PluginConfigDto::NativeSparseCompressor {
                        threshold_linear,
                        ratio,
                        makeup_gain,
                        lookahead_samples,
                    } => {
                        let mut node = SparseCompressorNode::new(
                            *threshold_linear,
                            *ratio,
                            *makeup_gain,
                            *lookahead_samples,
                        );
                        node.prepare(sr, blk);
                        Box::new(node)
                    }
                    PluginConfigDto::NativeMaximusMultiband => {
                        let mut node = MaximusMultibandNode::new();
                        node.prepare(sr, blk);
                        Box::new(node)
                    }
                    PluginConfigDto::NativeEmphasisLimiter {
                        ceiling_linear,
                        drive_linear: _,
                    } => {
                        let ceiling_db = 20.0 * ceiling_linear.max(1e-6).log10();
                        let mut node = EmphasisLimiterNode::new(64, ceiling_db);
                        node.prepare(sr, blk);
                        Box::new(node)
                    }
                    PluginConfigDto::NativeGrossBeat => {
                        let mut node = GrossBeatTimeBufferNode::new((sr as usize) * 2);
                        node.prepare(sr, blk);
                        Box::new(node)
                    }
                    PluginConfigDto::NativeLuxeVerbFdn => {
                        let mut node = LuxeVerbFdnShimmerNode::new();
                        node.prepare(sr, blk);
                        Box::new(node)
                    }
                };

                let s_idx = track
                    .add_plugin_slot(plugin_box, slot_dto.wet_mix)
                    .map_err(|e| e.to_string())?;
                track.slots[s_idx].enabled = slot_dto.enabled;
                track.slots[s_idx].prepare(sr, blk);
            }
        }

        // Register all 30-Bit Fixed-Point Internal Controller Links
        for link_dto in &state.controller_links {
            let _ = self.mixer_engine.controller_bus.add_link((*link_dto).into());
        }

        // Rebuild topological schedule and 5-Point APDC across all 502 tracks
        self.mixer_engine
            .rebuild_graph()
            .map_err(|e| e.to_string())?;

        self.sample_rate = sr;
        self.max_block_size = blk;
        self.project_state = state;

        // Evaluate initial automation clips at current `ppq_tick`
        let cur_tick = self.project_state.ppq_tick as f64;
        self.evaluate_and_publish_automation_clips(cur_tick)?;
        self.mixer_engine.drain_commands();

        Ok(())
    }

    /// Saves the current `ToneForgeProjectState` to a file on disk.
    pub fn save_project_to_file(&self, path: &Path) -> Result<(), String> {
        self.project_state.save_to_file(path)
    }

    /// Loads a `ToneForgeProjectState` from a file on disk and reconstructs the live `MixerEngine`.
    pub fn load_project_from_file(&mut self, path: &Path) -> Result<(), String> {
        let state = ToneForgeProjectState::load_from_file(path)?;
        self.apply_project_state(state)
    }

    // ========================================================================
    // 7. PHASE 5: SEQUENCER, PIANO ROLL, GENERATOR APDC & OFFLINE RENDER
    // ========================================================================

    /// Updates the Hybrid PPQ resolution (`96..=960`) on both the project state and the real-time slicer.
    pub fn set_ppq_resolution(&mut self, ppq: u32) -> Result<(), String> {
        let clamped = ppq.clamp(daw_core_graph::MIN_PPQ, daw_core_graph::MAX_PPQ);
        self.project_state.ppq_resolution = clamped;
        self.push_command(EngineCommand::SetPpq(clamped))?;
        self.mixer_engine.drain_commands();
        Ok(())
    }

    /// Adds a `PolyphonicSynthGeneratorNode` channel connected directly to **Point 1 APDC** (`ChannelGenerator`)
    /// targeting `target_mixer_track`.
    pub fn add_polyphonic_synth_channel(
        &mut self,
        channel_id: u32,
        target_mixer_track: usize,
        lookahead_latency_samples: usize,
    ) -> Result<usize, String> {
        if target_mixer_track >= MAX_MIXER_TRACKS {
            return Err(format!("Target mixer track {target_mixer_track} out of range"));
        }
        let synth = daw_core_graph::PolyphonicSynthGeneratorNode::new(channel_id, self.sample_rate)
            .with_lookahead_latency(lookahead_latency_samples);
        let mut ch = ChannelGenerator::new(channel_id, target_mixer_track, Box::new(synth));
        ch.prepare(self.sample_rate, self.max_block_size);
        self.mixer_engine
            .add_channel_generator(ch)
            .map_err(|e| e.to_string())
    }

    /// Upserts a `PianoRollPattern` into both the live `PlaylistSequencer` and `ToneForgeProjectState`.
    pub fn upsert_piano_roll_pattern(&mut self, pattern: daw_core_graph::PianoRollPattern) -> bool {
        let dto = crate::project_state::PianoRollPatternDto::from_pattern(&pattern);
        if let Some(existing) = self
            .project_state
            .piano_roll_patterns
            .iter_mut()
            .find(|p| p.pattern_id == dto.pattern_id)
        {
            *existing = dto;
        } else {
            self.project_state.piano_roll_patterns.push(dto);
        }
        self.mixer_engine.sequencer.upsert_pattern(pattern)
    }

    /// Upserts a `PlaylistClip` into both the live `PlaylistSequencer` and `ToneForgeProjectState`.
    pub fn upsert_playlist_clip(&mut self, clip: daw_core_graph::PlaylistClip) -> bool {
        let dto = crate::project_state::PlaylistClipDto::from_playlist_clip(&clip);
        if let Some(existing) = self
            .project_state
            .playlist_clips
            .iter_mut()
            .find(|c| c.clip_id == dto.clip_id)
        {
            *existing = dto;
        } else {
            self.project_state.playlist_clips.push(dto);
        }
        self.mixer_engine.sequencer.upsert_clip(clip)
    }

    /// Executes an Offline Bounce & Mastering Render job (`Master`, `SplitMixerTracks`, or `StemMastering`).
    pub fn execute_offline_render(
        &mut self,
        config: &crate::offline_render_engine::OfflineRenderConfig,
    ) -> std::io::Result<crate::offline_render_engine::OfflineRenderResult> {
        crate::offline_render_engine::OfflineRenderEngine::execute_render(
            &mut self.mixer_engine,
            config,
        )
    }
}
