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
pub mod codec_preview;
pub mod dk_triode_exciter;
pub mod dynamics_impact_bass;
pub mod irc_maximizer;
pub mod low_end_focus;
pub mod lufs_true_peak_meter;
pub mod multiband_imager;
pub mod oversampler;
pub mod spectral_modules;
pub mod stem_eq_rebalance;

pub use codec_preview::{
    CODEC_MDCT_BINS, CODEC_MDCT_WINDOW_LEN, CODEC_PREVIEW_LATENCY_SAMPLES, CODEC_SHORT_BINS,
    CODEC_SHORT_WINDOW_LEN, CodecPreviewModule, CodecWindowSequence,
    CodecTarget,
};

pub use oversampler::{
    KAISER_BETA, KAISER_FIR_TAPS, KaiserPolyphaseFirCoeffs, OversampleFactor,
    OversamplerFilterMode, PolyphaseHalfBandOversampler, STAGE1_ALLPASS_POLES,
    STAGE2_ALLPASS_POLES, design_elliptic_halfband_poles,
};

pub use chain_element::{
    DEFAULT_CROSSOVER_FREQS_HZ, LinkwitzRiley4BandCrossover, LinkwitzRiley4Splitter2Way,
    OZONE_MULTIBAND_COUNT, OzoneDspModule, StereoBiquadF64,
};
pub use dk_triode_exciter::{
    DEFAULT_TRIODE_LUT_PATHS, DiscretizedDkMatrices, DkDualTriodeChannelState, DkDualTriodeLut,
    DkKStateBlock, DkTriodeChannelState, DkTriodeExciterModule, DkTriodeLut, DkTriodeLutExciter,
    DkTriodeMultibandExciter, ExciterBandConfig, ExciterMode, IZOTOPE_DK_LUT_MAGIC,
    IZOTOPE_DK_LUT_VERSION, IZOTOPE_DUALTRIODE_DAT_SIZE, IZOTOPE_DUALTRIODE_LUT_MAGIC,
    IZOTOPE_KSTATE_BLOCK_SIZE, IZOTOPE_TRIODE_DAT_SIZE, STANDARD_EXCITER_LUT_RATES,
    koren_12ax7_currents, koren_12ax7_currents_and_jacobian, koren_12ax7_newton_solve,
    load_dualtriode_lut_cached, load_triode_lut_cached, rediscretize_dk_matrices,
};
pub use dynamics_impact_bass::{
    BassControlModule, ImpactBandConfig, ImpactModule, MAX_UNLIMITER_DELAY,
    NeuralUnlimiterBackend, OzoneBassControl, OzoneImpact, OzoneUnlimiter, UnlimiterModule,
};
pub use irc_maximizer::{
    IRC5_CROSSOVER_FREQS_HZ, IrcCharacter, IrcMaximizerModule, IrcMode, IrcVMaximizerCore,
    IrcVMultiBandMaximizer, MbitPlusDither, OZONE12_ISP_SAFETY_FACTOR, SlidingMaxHeap,
    SoftClipMode, catmull_rom_cubic_max_abs, eval_maxflat_c5_poly_p4, eval_ozone12_soft_clip_curve,
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
    OzoneSpectralShaper, OzoneStabilizer, SpectralShaperModule, SpectralShaperToneMode,
    StabilizerMode, StabilizerModule, StabilizerTarget,
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
    pub codec_preview: CodecPreviewModule,
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
            codec_preview: CodecPreviewModule::new(sr),
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

    /// Attaches a real-time or offline `NeuralUnlimiterBackend` (`delimit_exp009_ep080_fixed2.onnx`)
    /// to the Unlimiter stage inside `Ozone12MasterChain`.
    pub fn set_unlimiter_neural_backend(&mut self, backend: Box<dyn NeuralUnlimiterBackend>) {
        self.unlimiter.set_neural_backend(Some(backend));
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
        self.write_extended_ozone_telemetry(&mut out.exciter_drive_gr);
        out.tonal_balance_bars = self.stabilizer.tonal_balance_bars;
        out.imager_corr_bands = self.imager.band_correlations;
    }

    /// Exports directly into the shared `VisualTelemetryFrame`. Zero Heap Allocation.
    #[inline]
    pub fn export_to_visual_telemetry(&self, telemetry: &mut VisualTelemetryFrame) {
        self.meter.export_to_visual_telemetry(telemetry);
        self.exciter
            .export_telemetry(&mut telemetry.ozone12_telemetry.exciter_drive_gr);
        self.write_extended_ozone_telemetry(&mut telemetry.ozone12_telemetry.exciter_drive_gr);
        telemetry.ozone12_telemetry.tonal_balance_bars = self.stabilizer.tonal_balance_bars;
        telemetry.ozone12_telemetry.imager_corr_bands = self.imager.band_correlations;
    }

    #[inline]
    fn write_extended_ozone_telemetry(&self, exciter_drive_gr: &mut [f32; 248]) {
        exciter_drive_gr[240] = self.low_end_focus.last_delta_db();
        exciter_drive_gr[241] = if self.low_end_focus.enabled {
            self.low_end_focus.contrast_percent
        } else {
            0.0
        };
        exciter_drive_gr[242] = self.codec_preview.post_codec_true_peak_dbtp[0];
        exciter_drive_gr[243] = self.codec_preview.post_codec_true_peak_dbtp[1];
        exciter_drive_gr[244] = self.codec_preview.artifact_rms_db;
        exciter_drive_gr[245] = (self.codec_preview.mode as u8) as f32;
    }
}

impl UnifiedDspNode for Ozone12MasterChain {
    fn name(&self) -> &'static str {
        "iZotope Ozone 12 Complete Mastering Chain"
    }

    fn latency_samples(&self) -> usize {
        UnifiedDspNode::latency_samples(&self.unlimiter)
            + UnifiedDspNode::latency_samples(&self.maximizer)
            + UnifiedDspNode::latency_samples(&self.codec_preview)
    }

    fn prepare(&mut self, sample_rate: f64, max_block_size: usize) {
        let sr = sample_rate.max(8000.0);
        self.sample_rate = sr;
        UnifiedDspNode::prepare(&mut self.unlimiter, sr, max_block_size);
        UnifiedDspNode::prepare(&mut self.stabilizer, sr, max_block_size);
        UnifiedDspNode::prepare(&mut self.clarity, sr, max_block_size);
        UnifiedDspNode::prepare(&mut self.low_end_focus, sr, max_block_size);
        UnifiedDspNode::prepare(&mut self.spectral_shaper, sr, max_block_size);
        UnifiedDspNode::prepare(&mut self.impact, sr, max_block_size);
        UnifiedDspNode::prepare(&mut self.bass_control, sr, max_block_size);
        UnifiedDspNode::prepare(&mut self.exciter, sr, max_block_size);
        UnifiedDspNode::prepare(&mut self.imager, sr, max_block_size);
        UnifiedDspNode::prepare(&mut self.maximizer, sr, max_block_size);
        UnifiedDspNode::prepare(&mut self.codec_preview, sr, max_block_size);
        self.meter.set_sample_rate(sr);
    }

    fn reset_state(&mut self) {
        UnifiedDspNode::reset_state(&mut self.unlimiter);
        UnifiedDspNode::reset_state(&mut self.stabilizer);
        UnifiedDspNode::reset_state(&mut self.clarity);
        UnifiedDspNode::reset_state(&mut self.low_end_focus);
        UnifiedDspNode::reset_state(&mut self.spectral_shaper);
        UnifiedDspNode::reset_state(&mut self.impact);
        UnifiedDspNode::reset_state(&mut self.bass_control);
        UnifiedDspNode::reset_state(&mut self.exciter);
        UnifiedDspNode::reset_state(&mut self.imager);
        UnifiedDspNode::reset_state(&mut self.maximizer);
        UnifiedDspNode::reset_state(&mut self.codec_preview);
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
            65 => self.maximizer.set_parameter_normalized(12, norm_val),
            66 => self.maximizer.set_parameter_normalized(13, norm_val),
            69 => self.low_end_focus.set_parameter_normalized(5, norm_val),
            70..=74 => self
                .low_end_focus
                .set_parameter_normalized(param_id - 70, norm_val),
            75..=79 => self
                .spectral_shaper
                .set_parameter_normalized(param_id - 75, norm_val),
            80..=84 => self
                .unlimiter
                .set_parameter_normalized(param_id - 80, norm_val),
            85..=89 => self
                .codec_preview
                .set_parameter_normalized(param_id - 85, norm_val),
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
        }
        if self.codec_preview.enabled {
            self.codec_preview.process_frames_in_place(main_io);
        }
        for frame in main_io.iter() {
            self.meter
                .push_frame(frame[0], frame[1], self.maximizer.last_gr_db);
        }
    }
}

#[cfg(test)]
mod v3_ozone_parity_tests {
    use super::*;

    fn crest_factor_db(samples: &[f32]) -> f64 {
        let mut peak = 0.0f64;
        let mut sum_sq = 0.0f64;
        for &s in samples {
            let a = (s as f64).abs();
            if a > peak {
                peak = a;
            }
            sum_sq += (s as f64) * (s as f64);
        }
        let rms = (sum_sq / (samples.len().max(1) as f64)).sqrt().max(1e-12);
        20.0 * (peak.max(1e-12) / rms).log10()
    }

    #[test]
    fn test_low_end_focus_punchy_crest_boost_and_high_band_invariance() {
        let sr = 48_000.0;
        let n = 48_000usize; // 1.0 s @ 48 kHz
        let mut dry = Vec::with_capacity(n);
        for i in 0..n {
            let t = (i as f64) / sr;
            // 2 Hz kick/bass pattern: sharp attack + warm 65 Hz sustain + 1.2 kHz upper synth tone
            let beat_pos = (t * 2.0).fract();
            let env_low = 0.35 + 0.55 * (-beat_pos * 9.0).exp();
            let low_sig = env_low * (std::f64::consts::TAU * 65.0 * t).sin();
            let high_sig = 0.25 * (std::f64::consts::TAU * 1200.0 * t).sin();
            let s = (low_sig + high_sig) as f32;
            dry.push([s, s]);
        }

        // Reference pass (`contrast = 0%`) to eliminate pure LR4 all-pass group delay difference
        let mut lef_ref = LowEndFocusModule::new(sr);
        lef_ref.enabled = true;
        lef_ref.mode = LowEndFocusMode::Punchy;
        lef_ref.contrast_percent = 0.0;
        lef_ref.set_frequency_range_hz(20.0, 150.0);
        let mut ref_out = dry.clone();
        lef_ref.process_frames_in_place(&mut ref_out);

        // Punchy +75% pass (`20-150 Hz`)
        let mut lef_punch = LowEndFocusModule::new(sr);
        lef_punch.enabled = true;
        lef_punch.mode = LowEndFocusMode::Punchy;
        lef_punch.contrast_percent = 75.0;
        lef_punch.set_frequency_range_hz(20.0, 150.0);
        let mut punch_out = dry.clone();
        lef_punch.process_frames_in_place(&mut punch_out);

        // Measure < 150 Hz Crest Factor and > 500 Hz RMS invariance over steady region [4096..n]
        let mut lp_ref = StereoBiquadF64::new_unity();
        let mut lp_punch = StereoBiquadF64::new_unity();
        lp_ref.set_butterworth_lp(150.0, sr);
        lp_punch.set_butterworth_lp(150.0, sr);

        let mut hp_dry = StereoBiquadF64::new_unity();
        let mut hp_punch = StereoBiquadF64::new_unity();
        hp_dry.set_butterworth_hp(650.0, sr);
        hp_punch.set_butterworth_hp(650.0, sr);

        let mut low_ref_vec = Vec::with_capacity(n);
        let mut low_punch_vec = Vec::with_capacity(n);
        let mut hi_dry_sq = 0.0f64;
        let mut hi_punch_sq = 0.0f64;

        for i in 0..n {
            let lr = lp_ref.step_channel(0, ref_out[i][0] as f64) as f32;
            let lp = lp_punch.step_channel(0, punch_out[i][0] as f64) as f32;
            let hd = hp_dry.step_channel(0, dry[i][0] as f64);
            let hp = hp_punch.step_channel(0, punch_out[i][0] as f64);
            if i >= 4096 {
                low_ref_vec.push(lr);
                low_punch_vec.push(lp);
                hi_dry_sq += hd * hd;
                hi_punch_sq += hp * hp;
            }
        }

        let cf_ref = crest_factor_db(&low_ref_vec);
        let cf_punch = crest_factor_db(&low_punch_vec);
        let cf_boost_db = cf_punch - cf_ref;
        let hi_diff_db = (10.0 * (hi_punch_sq / hi_dry_sq).log10()).abs();

        assert!(
            cf_boost_db >= 1.5,
            "LowEndFocus Punchy (+75%) must increase < 150 Hz Crest Factor by >= 1.5 dB, got +{cf_boost_db:.2} dB"
        );
        assert!(
            hi_diff_db < 0.1,
            "LowEndFocus must leave > 500 Hz band invariant (< 0.1 dB), got {hi_diff_db:.4} dB"
        );
    }

    #[test]
    fn test_codec_preview_pcm_bypass_null_and_mp3_128_vs_320_noise_separation() {
        let sr = 48_000.0;
        let n = 16_384usize;
        let mut src = Vec::with_capacity(n);
        for i in 0..n {
            let t = (i as f64) / sr;
            let l = (0.35 * (std::f64::consts::TAU * 220.0 * t).sin()
                + 0.22 * (std::f64::consts::TAU * 1350.0 * t).sin()
                + 0.14 * (std::f64::consts::TAU * 6200.0 * t).sin()
                + 0.08 * (std::f64::consts::TAU * 13500.0 * t).sin()) as f32;
            let r = (0.35 * (std::f64::consts::TAU * 220.0 * t).sin()
                + 0.20 * (std::f64::consts::TAU * 1350.0 * t + 0.4).sin()
                + 0.15 * (std::f64::consts::TAU * 6200.0 * t - 0.5).sin()
                + 0.08 * (std::f64::consts::TAU * 13500.0 * t + 0.8).sin()) as f32;
            src.push([l, r]);
        }

        // 1. PCM Bypass + Artifact Solo -> residual RMS must be < -110 dBFS
        let mut pcm = CodecPreviewModule::new(sr);
        pcm.enabled = true;
        pcm.mode = CodecTarget::PcmBypass;
        pcm.artifact_solo = true;
        let mut buf_pcm = src.clone();
        pcm.process_frames_in_place(&mut buf_pcm);
        let pcm_delta_rms_db = {
            let sq: f64 = buf_pcm[2048..]
                .iter()
                .map(|f| 0.5 * ((f[0] as f64).powi(2) + (f[1] as f64).powi(2)))
                .sum::<f64>()
                / ((n - 2048) as f64);
            10.0 * sq.max(1e-18).log10()
        };
        assert!(
            pcm_delta_rms_db < -110.0,
            "CodecPreview PCM Bypass Artifact Solo must null < -110 dBFS, got {pcm_delta_rms_db:.2} dBFS"
        );

        // 2. MP3 128 kbps vs MP3 320 kbps Artifact Solo RMS separation >= 6.0 dB
        let mut mp3_128 = CodecPreviewModule::new(sr);
        mp3_128.enabled = true;
        mp3_128.mode = CodecTarget::Mp3_128;
        mp3_128.artifact_solo = true;
        let mut buf_128 = src.clone();
        mp3_128.process_frames_in_place(&mut buf_128);

        let mut mp3_320 = CodecPreviewModule::new(sr);
        mp3_320.enabled = true;
        mp3_320.mode = CodecTarget::Mp3_320;
        mp3_320.artifact_solo = true;
        let mut buf_320 = src;
        mp3_320.process_frames_in_place(&mut buf_320);

        let rms_128_db = {
            let sq: f64 = buf_128[2048..]
                .iter()
                .map(|f| 0.5 * ((f[0] as f64).powi(2) + (f[1] as f64).powi(2)))
                .sum::<f64>()
                / ((n - 2048) as f64);
            10.0 * sq.max(1e-18).log10()
        };
        let rms_320_db = {
            let sq: f64 = buf_320[2048..]
                .iter()
                .map(|f| 0.5 * ((f[0] as f64).powi(2) + (f[1] as f64).powi(2)))
                .sum::<f64>()
                / ((n - 2048) as f64);
            10.0 * sq.max(1e-18).log10()
        };
        let sep_db = rms_128_db - rms_320_db;
        assert!(
            sep_db >= 6.0,
            "MP3 128 kbps quantization artifact RMS ({rms_128_db:.2} dBFS) must exceed MP3 320 kbps ({rms_320_db:.2} dBFS) by >= 6.0 dB (got {sep_db:.2} dB)"
        );
    }
}
