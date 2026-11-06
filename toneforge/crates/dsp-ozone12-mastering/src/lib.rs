//! `dsp-ozone12-mastering` — Complete Clean-Room Implementation of iZotope Ozone 12 Mastering Suite:
//!
//! 1. **DK-Method 12AX7 Koren Triode Exciter** (`DkTriodeLutExciter` / `DkTriodeExciterModule`):
//!    - Exact binary parser & 2D bilinear state-space solver for `exciter-triode-48000.000000.dat`
//!      (`0x8BAD43FC`, `DSP::Shared::KState<3,2,2,KorenFunction>` at `0x181DB8BA0`).
//!    - 4-Band Linkwitz-Riley crossover + 7 saturation modes (`Analog`, `Retro`, `Tape`,
//!      `Warm`, `Triode`, `DualTriode`, `Tube`).
//! 2. **IRC I–V Multi-Band True-Peak Maximizer** (`IrcVMultiBandMaximizer` / `IrcMaximizerModule`):
//!    - 4-Band Linkwitz-Riley crossover + `SlidingMaxHeap` (`0x183F81EE4`) lookahead envelope.
//!    - Adaptive program-dependent release (`IRC I–IV` `Classic`/`Modern`/`Transient`/`Pumping`) + `IRC V`
//!      Multi-Band Envelope Limiter (`0x183F69CE4`).
//!    - 4x Oversampling Polyphase FIR True-Peak Detector (`ITU-R BS.1770-4` Table 3, 48 taps)
//!      with predictive inter-sample peak brickwall enforcement.
//!    - Upward Compression (`EnableLowLevelBoost`), Soft-Clip (`Light`/`Medium`/`Heavy`), and `MbitPlusDither`.
//! 3. **Core Mastering Modules**:
//!    - `MultibandImagerModule` (`OzoneImager`): 4-Band M/S Width, Phase-Safe Mono-Compatible
//!      Stereoizer (Mode I Velvet-Noise & Mode II All-Pass), `Recover Sides`, and Pearson correlation.
//!    - `StabilizerModule` (`OzoneStabilizer`): Adaptive Tonal Balance (`Shape` & `Cut` + `Tame Transients`).
//!    - `ImpactModule` (`OzoneImpact`): 4-Band Micro-Dynamics Shaper (`env_fast / env_slow`).
//!    - `ClarityModule` (`OzoneClarity`): Psychoacoustic ISO 226 Spectral Tilt & Unmasking.
//!    - `UnlimiterModule` (`OzoneUnlimiter`): Brickwall Transient Recovery.
//!    - `BassControlModule` (`OzoneBassControl`): Low-End Sustain Power & Peak Punch.
//!    - `LowEndFocusModule` & `SpectralShaperModule`.
//!    - `StemEqModule` (`0x1857397B0`) & `MasterRebalanceModule` (`0x18572A8F0`).
//! 4. **Broadcast Loudness & True-Peak Metering** (`LufsTruePeakMeter`):
//!    - `ITU-R BS.1770-4` 2-stage K-Weighting IIR + Momentary (`400 ms`), Short-Term (`3.0 s`),
//!      Integrated (dual-gated `-70 LUFS` / `-10 LU`) LUFS + 4x True Peak (`dBTP`) + Gain Reduction trace
//!      exporting directly to `Ozone12MeterFrame` inside `VisualTelemetryFrame`.
//!
//! All real-time audio processing methods are guaranteed **Zero-Allocation** on the audio thread.

pub mod chain_element;
pub mod dk_triode_exciter;
pub mod dynamics_impact_bass;
pub mod irc_maximizer;
pub mod lufs_true_peak_meter;
pub mod multiband_imager;
pub mod spectral_modules;
pub mod stem_eq_rebalance;

pub use chain_element::{
    DEFAULT_CROSSOVER_FREQS_HZ, LinkwitzRiley4BandCrossover, LinkwitzRiley4Splitter2Way,
    OZONE_MULTIBAND_COUNT, OzoneDspModule, StereoBiquadF64,
};
pub use dk_triode_exciter::{
    DEFAULT_TRIODE_LUT_PATHS, DkTriodeChannelState, DkTriodeExciterModule, DkTriodeLut,
    DkTriodeLutExciter, ExciterBandConfig, ExciterMode, IZOTOPE_DK_LUT_MAGIC,
    IZOTOPE_DK_LUT_VERSION, IZOTOPE_TRIODE_DAT_SIZE, koren_12ax7_currents,
};
pub use dynamics_impact_bass::{
    BassControlModule, ImpactBandConfig, ImpactModule, OzoneBassControl, OzoneImpact,
    OzoneUnlimiter, UnlimiterModule,
};
pub use irc_maximizer::{
    IrcCharacter, IrcMaximizerModule, IrcMode, IrcVMaximizerCore, IrcVMultiBandMaximizer,
    MbitPlusDither, SlidingMaxHeap, SoftClipMode,
};
pub use lufs_true_peak_meter::{
    ITU_BS1770_4_POLYPHASE_TAPS, KWeightingStereoFilter, LufsTruePeakMeter, Ozone12MeterFrame,
    TruePeakDetector4x,
};
pub use multiband_imager::{
    ImagerBandConfig, MultibandImagerModule, OzoneImager, StereoizerMode,
};
pub use spectral_modules::{
    ClarityModule, LowEndFocusMode, LowEndFocusModule, OzoneClarity, OzoneLowEndFocus,
    OzoneSpectralShaper, OzoneStabilizer, SpectralShaperModule, StabilizerMode, StabilizerModule,
    StabilizerTarget,
};
pub use stem_eq_rebalance::{
    MasterRebalanceModule, OzoneMasterRebalance, OzoneStemEq, OzoneStemIndex, StemEqBandSet,
    StemEqModule,
};

use daw_core_graph::{ProcessContext, StereoFrame, UnifiedDspNode, VisualTelemetryFrame};
use std::path::Path;

/// Complete iZotope Ozone 12 Master Chain (`Ozone12MasterChain`) integrating all mastering
/// modules and the broadcast `LufsTruePeakMeter` into a single Zero-Allocation `UnifiedDspNode`.
#[derive(Clone, Debug)]
pub struct Ozone12MasterChain {
    pub unlimiter: UnlimiterModule,
    pub stabilizer: StabilizerModule,
    pub clarity: ClarityModule,
    pub low_end_focus: LowEndFocusModule,
    pub spectral_shaper: SpectralShaperModule,
    pub impact: ImpactModule,
    pub bass_control: BassControlModule,
    pub exciter: DkTriodeLutExciter,
    pub imager: MultibandImagerModule,
    pub maximizer: IrcVMultiBandMaximizer,
    pub meter: LufsTruePeakMeter,
    sample_rate: f64,
}

impl Default for Ozone12MasterChain {
    fn default() -> Self {
        Self::new(48_000.0)
    }
}

impl Ozone12MasterChain {
    pub fn new(sample_rate: f64) -> Self {
        let sr = sample_rate.max(8000.0);
        let mut chain = Self {
            unlimiter: UnlimiterModule::new(sr),
            stabilizer: StabilizerModule::new(sr),
            clarity: ClarityModule::new(sr),
            low_end_focus: LowEndFocusModule::new(sr),
            spectral_shaper: SpectralShaperModule::new(sr),
            impact: ImpactModule::new(sr),
            bass_control: BassControlModule::new(sr),
            exciter: DkTriodeLutExciter::new(sr),
            imager: MultibandImagerModule::new(sr),
            maximizer: IrcVMultiBandMaximizer::new(sr),
            meter: LufsTruePeakMeter::new(sr),
            sample_rate: sr,
        };
        // Default gentle mastering configuration with IRC V True-Peak Maximizer at -1.0 dBTP.
        chain.unlimiter.enabled = false;
        chain.low_end_focus.enabled = false;
        chain.spectral_shaper.enabled = false;
        chain.maximizer.set_ceiling_dbtp(-1.0);
        chain.maximizer.prevent_intersample_clipping = true;
        chain
    }

    /// Loads the real 12AX7 Koren Triode LUT `.dat` file into the Exciter module.
    pub fn load_exciter_lut_from_path(&mut self, path: &Path) -> Result<(), String> {
        self.exciter = DkTriodeLutExciter::from_dat_file(path, self.sample_rate)?;
        Ok(())
    }

    /// Enables or bypasses the IRC V True-Peak Maximizer stage (dynamically updating chain latency).
    #[inline]
    pub fn set_irc_maximizer_enabled(&mut self, enabled: bool) {
        self.maximizer.bypass = !enabled;
    }

    /// Exports the complete `Ozone12MeterFrame` combining `LufsTruePeakMeter`, `DkTriodeLutExciter`,
    /// `StabilizerModule`, and `MultibandImagerModule` telemetry. Zero Heap Allocation.
    #[inline]
    pub fn export_ozone_meter_frame(&self, out: &mut Ozone12MeterFrame) {
        self.meter.export_to_meter_frame(out);
        self.exciter.export_telemetry(&mut out.exciter_drive_gr);
        out.tonal_balance_bars = self.stabilizer.tonal_balance_bars;
        out.imager_corr_bands = self.imager.band_correlations;
    }

    /// Exports directly into the shared `VisualTelemetryFrame`. Zero Heap Allocation.
    #[inline]
    pub fn export_to_visual_telemetry(&self, telemetry: &mut VisualTelemetryFrame) {
        self.meter.export_to_visual_telemetry(telemetry);
        self.exciter
            .export_telemetry(&mut telemetry.ozone12_telemetry.exciter_drive_gr);
        telemetry.ozone12_telemetry.tonal_balance_bars = self.stabilizer.tonal_balance_bars;
        telemetry.ozone12_telemetry.imager_corr_bands = self.imager.band_correlations;
    }
}

impl UnifiedDspNode for Ozone12MasterChain {
    fn name(&self) -> &'static str {
        "iZotope Ozone 12 Complete Mastering Chain"
    }

    fn latency_samples(&self) -> usize {
        UnifiedDspNode::latency_samples(&self.maximizer)
    }

    fn prepare(&mut self, sample_rate: f64, max_block_size: usize) {
        let sr = sample_rate.max(8000.0);
        self.sample_rate = sr;
        UnifiedDspNode::prepare(&mut self.unlimiter, sr, max_block_size);
        UnifiedDspNode::prepare(&mut self.stabilizer, sr, max_block_size);
        UnifiedDspNode::prepare(&mut self.clarity, sr, max_block_size);
        OzoneDspModule::reset(&mut self.low_end_focus, sr, max_block_size);
        OzoneDspModule::reset(&mut self.spectral_shaper, sr, max_block_size);
        UnifiedDspNode::prepare(&mut self.impact, sr, max_block_size);
        UnifiedDspNode::prepare(&mut self.bass_control, sr, max_block_size);
        UnifiedDspNode::prepare(&mut self.exciter, sr, max_block_size);
        UnifiedDspNode::prepare(&mut self.imager, sr, max_block_size);
        UnifiedDspNode::prepare(&mut self.maximizer, sr, max_block_size);
        self.meter.set_sample_rate(sr);
    }

    fn reset_state(&mut self) {
        UnifiedDspNode::reset_state(&mut self.unlimiter);
        UnifiedDspNode::reset_state(&mut self.stabilizer);
        UnifiedDspNode::reset_state(&mut self.clarity);
        self.low_end_focus.reset_internal();
        self.spectral_shaper.reset_internal();
        UnifiedDspNode::reset_state(&mut self.impact);
        UnifiedDspNode::reset_state(&mut self.bass_control);
        UnifiedDspNode::reset_state(&mut self.exciter);
        UnifiedDspNode::reset_state(&mut self.imager);
        UnifiedDspNode::reset_state(&mut self.maximizer);
        self.meter.reset();
    }

    fn set_parameter_normalized(&mut self, param_id: u32, norm_val: f32) {
        match param_id {
            0..=9 => self.maximizer.set_parameter_normalized(param_id, norm_val),
            10..=24 => self.exciter.set_parameter_normalized(param_id - 10, norm_val),
            25..=35 => self.imager.set_parameter_normalized(param_id - 25, norm_val),
            36..=44 => self.stabilizer.set_parameter_normalized(param_id - 36, norm_val),
            45..=54 => self.impact.set_parameter_normalized(param_id - 45, norm_val),
            55..=64 => self.clarity.set_parameter_normalized(param_id - 55, norm_val),
            _ => {}
        }
    }

    fn current_gain_reduction_db(&self) -> f32 {
        self.maximizer.last_gr_db
    }

    fn export_visual_telemetry(&self, frame: &mut VisualTelemetryFrame) {
        self.export_to_visual_telemetry(frame);
    }

    fn process_slice(
        &mut self,
        ctx: &ProcessContext,
        main_io: &mut [StereoFrame],
        sidechain_in: Option<&[StereoFrame]>,
        aux_sends_out: &mut [&mut [StereoFrame]],
    ) {
        self.unlimiter.process_frames_in_place(main_io);
        self.stabilizer.process_frames_in_place(main_io);
        self.clarity.process_frames_in_place(main_io);
        self.low_end_focus.process_frames_in_place(main_io);
        self.spectral_shaper.process_frames_in_place(main_io);
        self.impact.process_frames_in_place(main_io);
        self.bass_control.process_frames_in_place(main_io);
        self.exciter
            .process_slice(ctx, main_io, sidechain_in, aux_sends_out);
        self.imager
            .process_slice(ctx, main_io, sidechain_in, aux_sends_out);

        for frame in main_io.iter_mut() {
            let out = self.maximizer.step_frame(*frame);
            *frame = out;
            self.meter
                .push_frame(out[0], out[1], self.maximizer.last_gr_db);
        }
    }
}
