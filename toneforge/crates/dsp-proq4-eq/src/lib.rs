//! `dsp-proq4-eq`: 24-Band Dynamic/Spectral/Linear-Phase Equalizer (`FabFilter Pro-Q 4.clap` `v4.13`).
//!
//! Modules:
//! - [`params_601`]: 601 Parameter System (`24 Bands * 24 Params + 25 Global` from `DigiPageTable.xml`)
//! - [`biquad_frac_slope`]: Fractional-Slope Dual Biquad Cascade (`6..96 dB/oct`, bitmask `0x10F6`)
//!   & Analog Gain-Q Interaction (`1.0905077326652577`, bitmask `0x1074`) across all 9 EQ shapes + AllPass
//! - [`natural_phase`]: Natural Phase complex magnitude + phase matching (`0..12` samples latency)
//! - [`linear_phase_conv`]: Modified Bessel `I0(x)` (`0x180102A10`), Kaiser-60dB Window (`0x180232D00`),
//!   and Recursive Non-Uniform Partitioned Overlap-Save Convolver (`64 -> 512 -> 4096 -> 32768`) with Dual-IR Crossfading
//! - [`dynamic_rc`]: Soft-Knee Detector + Double-Precision `ProgramDependentCapacitor` (`f64`)
//! - [`spectral_dynamics`]: Per-Band `2048 * scale` STFT Spectral Dynamics with `f32x4` SIMD & Spectral Density
//! - [`character_waveshaper`]: Analog Harmonic Saturation (`Clean` / `Subtle` / `Warm`)
//! - [`auto_gain_512`]: 512-Bin 8-Knot Perceptual Loudness Auto-Gain & 16 KB `ProQ4SpectrumFrame` Exporter

pub mod auto_gain_512;
pub mod biquad_frac_slope;
pub mod character_waveshaper;
pub mod collision_bus;
pub mod dynamic_rc;
pub mod eq_sketch_fitter;
pub mod linear_phase_conv;
pub mod natural_phase;
pub mod params_601;
pub mod spectral_dynamics;

pub use auto_gain_512::{
    AUTO_GAIN_BINS, PerceptualAutoGain, ProQ4SpectrumAnalyzer, SPECTRUM_FRAME_BINS,
    SharedInstanceCollisionBus,
};
pub use biquad_frac_slope::{AnalogSection2, CytomicSvfCoeffs, FractionalSlopeBiquadBand};
pub use character_waveshaper::{CharacterWaveShaper, CharacterWaveshaper};
pub use collision_bus::{MAX_COLLISION_SLOTS, NO_REFERENCE_TRACK, ProQ4CollisionBus};
pub use dynamic_rc::{DynamicEqBandProcessor, ProgramDependentCapacitor, SoftKneeDetector};
pub use eq_sketch_fitter::{eval_bands_combined_db, fit_eq_sketch_curve};
pub use linear_phase_conv::{
    DEFAULT_MAX_LINEAR_PHASE_TAPS, KAISER_ATTENUATION_DB, KAISER_BETA_60DB,
    NON_UNIFORM_PARTITION_SIZES, NonUniformPartitionedConvolver, bessel_i0,
    compute_kaiser_60db_window,
};
pub use natural_phase::{
    AllpassHalfbandStage, BRICKWALL_EXTRA_LATENCY_SAMPLES, JacobiEllipticHalfbandSpec,
    JacobiEllipticOversampler4x, NATURAL_PHASE_BASE_LATENCY_SAMPLES, NaturalPhaseBank,
    NaturalPhaseOversampler, NaturalPhaseOversampler2x, PROQ4_STAGE1_OMEGA_P, PROQ4_STAGE2_OMEGA_P,
    analog_prototype_mag_db, design_jacobi_elliptic_halfband, eval_analog_s_plane_response,
    eval_complex_polynomial_ratio, eval_natural_phase_band_response,
};
pub use params_601::{
    AnalogFilterPrototype, ChannelMode, CharacterMode, EqBandParams, FilterShape,
    LinearPhaseResolution, PROQ4_BAND_PARAMS_TOTAL, PROQ4_CONTINUOUS_PARAM_COUNT,
    PROQ4_DISCRETE_PARAM_COUNT, PROQ4_GAIN_DEPENDENT_MASK, PROQ4_GAIN_Q_INTERACTION_BASE,
    PROQ4_GAIN_Q_INTERACTION_MASK, PROQ4_GLOBAL_PARAMS, PROQ4_MAX_FREQ_HZ, PROQ4_MIN_FREQ_HZ,
    PROQ4_NUM_BANDS, PROQ4_PARAMS_PER_BAND, PROQ4_TOTAL_PARAMS, ProQ4GlobalParams,
    ProQ4ParameterSet, ProcessingMode, StereoPlacement, TAP_TABLE_1X, freq_hz_to_norm,
    norm_to_freq_hz, norm_to_q_ui, norm_to_slope_db_oct, q_ui_to_norm, slope_db_oct_to_norm,
};
pub use spectral_dynamics::{
    SPECTRAL_BASE_BINS, SPECTRAL_BASE_FFT_SIZE, SPECTRAL_BASE_HOP_SIZE, SPECTRAL_BASE_LATENCY,
    SpectralDynamicsCore,
};

use daw_core_graph::{
    ProQ4SpectrumFrame, ProcessContext, StereoFrame, UnifiedDspNode, VisualTelemetryFrame,
};
use num_complex::Complex64;
use std::sync::Arc;

/// 24-Band FabFilter Pro-Q 4 Processor implementing `UnifiedDspNode` (Zero-Allocation in `process_slice`).
pub struct ProQ4Processor {
    pub instance_id: u32,
    pub collision_bus: Arc<ProQ4CollisionBus>,
    pub params: ProQ4ParameterSet,
    pub bands: [FractionalSlopeBiquadBand; PROQ4_NUM_BANDS],
    pub band_enabled: [bool; PROQ4_NUM_BANDS],
    pub natural_phase_bank: NaturalPhaseBank,
    pub linear_phase_convolver: NonUniformPartitionedConvolver,
    pub dynamic_bands: [DynamicEqBandProcessor; PROQ4_NUM_BANDS],
    pub spectral_core: SpectralDynamicsCore,
    pub character_shaper: CharacterWaveShaper,
    pub auto_gain_calc: PerceptualAutoGain,
    pub auto_gain_mag_sq_buf: [f32; AUTO_GAIN_BINS],
    pub auto_gain_linear: f32,
    pub spectrum_analyzer: ProQ4SpectrumAnalyzer,
    pub sample_rate: f64,
    pub current_gr_db: f32,
    has_active_spectral_band: bool,
    has_active_brickwall_band: bool,
    linear_phase_ir_dirty: bool,
    /// Band snapshot used by the in-flight (block-amortised) linear-phase IR design.
    lp_design_bands: [FractionalSlopeBiquadBand; PROQ4_NUM_BANDS],
    lp_design_enabled: [bool; PROQ4_NUM_BANDS],
    brickwall_delay_ring: [[f64; BRICKWALL_EXTRA_LATENCY_SAMPLES]; 2],
    brickwall_cursor: usize,
}

/// Magnitude evaluations spent per `process_slice` on an in-flight linear-phase IR design.
const LP_IR_DESIGN_EVALS_PER_SLICE: usize = 512;

impl Default for ProQ4Processor {
    fn default() -> Self {
        Self::new()
    }
}

impl ProQ4Processor {
    pub fn new() -> Self {
        Self::with_collision_bus(1, ProQ4CollisionBus::global().clone())
    }

    pub fn with_collision_bus(instance_id: u32, collision_bus: Arc<ProQ4CollisionBus>) -> Self {
        Self {
            instance_id,
            collision_bus,
            params: ProQ4ParameterSet::new(),
            bands: [FractionalSlopeBiquadBand::new_unity(); PROQ4_NUM_BANDS],
            band_enabled: [false; PROQ4_NUM_BANDS],
            natural_phase_bank: NaturalPhaseBank::new(),
            linear_phase_convolver: NonUniformPartitionedConvolver::new(
                DEFAULT_MAX_LINEAR_PHASE_TAPS,
            ),
            dynamic_bands: [DynamicEqBandProcessor::new(); PROQ4_NUM_BANDS],
            spectral_core: SpectralDynamicsCore::new(),
            character_shaper: CharacterWaveShaper::new(48_000.0),
            auto_gain_calc: PerceptualAutoGain::new(),
            auto_gain_mag_sq_buf: [1.0; AUTO_GAIN_BINS],
            auto_gain_linear: 1.0,
            spectrum_analyzer: ProQ4SpectrumAnalyzer::new(),
            sample_rate: 48_000.0,
            current_gr_db: 0.0,
            has_active_spectral_band: false,
            has_active_brickwall_band: false,
            linear_phase_ir_dirty: false,
            lp_design_bands: [FractionalSlopeBiquadBand::new_unity(); PROQ4_NUM_BANDS],
            lp_design_enabled: [false; PROQ4_NUM_BANDS],
            brickwall_delay_ring: [[0.0; BRICKWALL_EXTRA_LATENCY_SAMPLES]; 2],
            brickwall_cursor: 0,
        }
    }

    /// Configures a single band's full parameter struct and updates all internal DSP engines
    /// (Zero Latency IIR, Natural Phase 2x IIR, Dynamic EQ RC, Spectral Mask, Auto-Gain, and Linear Phase FIR).
    pub fn set_band_params(&mut self, band_idx: usize, band_params: EqBandParams) {
        if band_idx >= PROQ4_NUM_BANDS {
            return;
        }
        self.params.bands[band_idx] = band_params;
        self.params.sync_mirror_from_structured();
        self.refresh_band_dsp(band_idx, true);
        self.linear_phase_ir_dirty = true;
        self.refresh_global_aggregates(true);
    }

    /// Updates global parameters and refreshes dependent DSP stages.
    pub fn set_global_params(&mut self, global_params: ProQ4GlobalParams) {
        self.params.global = global_params;
        self.params.sync_mirror_from_structured();
        for b in 0..PROQ4_NUM_BANDS {
            if self.params.bands[b].used {
                self.refresh_band_dsp(b, false);
            }
        }
        self.linear_phase_ir_dirty = true;
        self.refresh_global_aggregates(true);
    }

    pub fn refresh_band_dsp(&mut self, band_idx: usize, _smooth_ir: bool) {
        self.linear_phase_ir_dirty = true;
        let bp = &self.params.bands[band_idx];
        let active = bp.used && bp.enabled;
        self.band_enabled[band_idx] = active;
        if !active {
            return;
        }

        let drift_norm = (self.params.global.drift_pct * 0.01).clamp(0.0, 1.0) as f64;
        // Deterministic per-band analog component tolerance offset when Drift > 0%
        let band_sign = if (band_idx & 1) == 0 { 1.0 } else { -1.0 };
        let freq_drift_mul = 1.0 + drift_norm * band_sign * 0.003 * (((band_idx % 5) + 1) as f64);
        let eff_freq_hz = ((bp.freq_hz as f64) * freq_drift_mul).clamp(10.0, 30_000.0);
        let eff_gain_db = bp.effective_static_gain_db(self.params.global.gain_scale) as f64;
        let eff_slope_db_oct = if bp.is_brickwall {
            96.0
        } else {
            bp.slope_db_oct as f64
        };
        let gq = self.params.global.gain_q_interaction && bp.shape.has_gain_q_interaction();

        self.bands[band_idx].configure_shape(
            bp.shape,
            eff_freq_hz,
            eff_gain_db,
            bp.q_ui as f64,
            eff_slope_db_oct,
            self.sample_rate,
            gq,
        );

        self.natural_phase_bank.bands_2x[band_idx].configure_shape_natural_phase(
            bp.shape,
            eff_freq_hz,
            eff_gain_db,
            bp.q_ui as f64,
            eff_slope_db_oct,
            self.sample_rate * 2.0,
            gq,
        );

        self.dynamic_bands[band_idx]
            .detector
            .update_sidechain_filter(bp, self.sample_rate);
        self.dynamic_bands[band_idx]
            .capacitor
            .set_ballistics_pct(bp.dyn_attack_pct, bp.dyn_release_pct);
    }

    /// Starts a linear-phase IR design from the current band state. Parameter changes arriving on the
    /// real-time thread (`smooth_ir_crossfade == true` on a primed convolver) are designed
    /// incrementally inside `process_slice` (bounded work per block, then a dual-bank crossfade);
    /// the first IR, or a hard reload, is designed synchronously.
    fn rebuild_linear_phase_ir_now(&mut self, smooth_ir_crossfade: bool) {
        let fir_len = self
            .params
            .global
            .processing_resolution
            .fir_length(self.sample_rate as f32)
            .min(self.linear_phase_convolver.max_fir_len());
        self.lp_design_bands = self.bands;
        self.lp_design_enabled = self.band_enabled;
        self.linear_phase_convolver
            .begin_ir_design(fir_len, self.sample_rate, smooth_ir_crossfade);
        if !smooth_ir_crossfade || !self.linear_phase_convolver.is_primed() {
            self.continue_linear_phase_ir_design(usize::MAX);
        }
        self.linear_phase_ir_dirty = false;
    }

    /// Advances the pending linear-phase IR design by at most `max_evals` magnitude evaluations.
    fn continue_linear_phase_ir_design(&mut self, max_evals: usize) {
        let fs = self.sample_rate;
        let bands = &self.lp_design_bands;
        let enabled = &self.lp_design_enabled;
        self.linear_phase_convolver
            .continue_ir_design(max_evals, |freq_hz| {
                let mut h = Complex64::new(1.0, 0.0);
                for b in 0..PROQ4_NUM_BANDS {
                    if enabled[b] {
                        h *= bands[b].eval_complex_response(freq_hz.max(1.0), fs);
                    }
                }
                h.norm() as f32
            });
    }

    pub fn refresh_global_aggregates(&mut self, smooth_ir_crossfade: bool) {
        let mut any_spectral = false;
        let mut any_brickwall = false;

        for b in 0..PROQ4_NUM_BANDS {
            let bp = &self.params.bands[b];
            if bp.used && bp.enabled {
                if bp.dyn_enabled && bp.spectral_enabled && bp.shape.supports_spectral_dynamics() {
                    any_spectral = true;
                }
                if bp.is_brickwall {
                    any_brickwall = true;
                }
            }
        }
        self.has_active_spectral_band = any_spectral;
        self.has_active_brickwall_band = any_brickwall;

        if any_spectral {
            self.spectral_core.update_spectral_masks(
                &self.params.bands,
                self.params.global.gain_scale,
                self.sample_rate,
            );
        }

        // Recompute 512-bin perceptual Auto-Gain
        if self.params.global.auto_gain {
            for i in 0..AUTO_GAIN_BINS {
                let f = self.auto_gain_calc.freq_hz_table[i];
                let h = self.eval_combined_static_response(f);
                self.auto_gain_mag_sq_buf[i] = (h.norm_sqr() as f32).max(1e-12);
            }
            self.auto_gain_linear = self
                .auto_gain_calc
                .compute_compensation_linear(&self.auto_gain_mag_sq_buf);
        } else {
            self.auto_gain_linear = 1.0;
        }

        // If Linear Phase mode is active and not currently mid-crossfade, rebuild Kaiser-60dB FIR now.
        // If a crossfade is currently in flight, `linear_phase_ir_dirty` remains true and will
        // trigger as soon as the current crossfade completes in `process_slice`.
        if matches!(
            self.params.global.processing_mode,
            ProcessingMode::LinearPhase
        ) && (!smooth_ir_crossfade
            || (self.linear_phase_ir_dirty && !self.linear_phase_convolver.is_crossfading()))
        {
            self.rebuild_linear_phase_ir_now(smooth_ir_crossfade);
        }
    }

    /// Evaluates the combined complex frequency response across all active bands at `freq_hz`.
    #[inline]
    pub fn eval_combined_static_response(&self, freq_hz: f64) -> Complex64 {
        let mut total = Complex64::new(1.0, 0.0);
        for b in 0..PROQ4_NUM_BANDS {
            if self.band_enabled[b] {
                let h = match self.params.global.processing_mode {
                    ProcessingMode::NaturalPhase => eval_natural_phase_band_response(
                        &self.natural_phase_bank.bands_2x[b],
                        freq_hz,
                        self.sample_rate,
                    ),
                    _ => self.bands[b].eval_complex_response(freq_hz, self.sample_rate),
                };
                total *= h;
            }
        }
        total
    }

    /// Applies freehand `EQ Sketch` `(freq_hz, gain_db)` points by fitting up to `max_bands`
    /// parametric bands and activating them on this processor.
    pub fn apply_eq_sketch(&mut self, points_hz_db: &[(f32, f32)], max_bands: usize) -> Vec<EqBandParams> {
        let fitted = fit_eq_sketch_curve(points_hz_db, max_bands);
        for b in 0..PROQ4_NUM_BANDS {
            if b < fitted.len() {
                self.params.bands[b] = fitted[b];
            } else {
                self.params.bands[b] = EqBandParams::default();
            }
            self.band_enabled[b] = self.params.bands[b].used && self.params.bands[b].enabled;
            if self.band_enabled[b] {
                self.refresh_band_dsp(b, false);
            }
        }
        self.params.sync_mirror_from_structured();
        self.linear_phase_ir_dirty = true;
        self.refresh_global_aggregates(false);
        fitted
    }

    /// Exports the latest 16 KB `ProQ4SpectrumFrame` directly into `dst` (Zero Heap Allocation).
    #[inline]
    pub fn export_spectrum_frame(&self, dst: &mut ProQ4SpectrumFrame) {
        self.spectrum_analyzer.export_to_frame(dst);
        if self.params.global.show_collisions {
            self.collision_bus.publish_and_compute_collisions(
                self.instance_id,
                &dst.post_db,
                &mut dst.collision_mask,
            );
        } else {
            self.collision_bus
                .publish_track_spectrum(self.instance_id, &dst.post_db);
            dst.collision_mask.fill(0.0);
        }
    }

    #[inline(always)]
    fn process_band_with_placement(
        band: &mut FractionalSlopeBiquadBand,
        placement: StereoPlacement,
        channel_mode: ChannelMode,
        l: &mut f64,
        r: &mut f64,
    ) {
        match placement {
            StereoPlacement::Stereo => {
                *l = band.process_sample(0, *l);
                *r = band.process_sample(1, *r);
            }
            StereoPlacement::Left => {
                if matches!(channel_mode, ChannelMode::MidSide) {
                    let mid = 0.5 * (*l + *r);
                    let side = 0.5 * (*l - *r);
                    let mid_filt = band.process_sample(0, mid);
                    *l = mid_filt + side;
                    *r = mid_filt - side;
                } else {
                    *l = band.process_sample(0, *l);
                }
            }
            StereoPlacement::Right => {
                if matches!(channel_mode, ChannelMode::MidSide) {
                    let mid = 0.5 * (*l + *r);
                    let side = 0.5 * (*l - *r);
                    let side_filt = band.process_sample(1, side);
                    *l = mid + side_filt;
                    *r = mid - side_filt;
                } else {
                    *r = band.process_sample(1, *r);
                }
            }
            StereoPlacement::Mid => {
                let mid = 0.5 * (*l + *r);
                let side = 0.5 * (*l - *r);
                let mid_filt = band.process_sample(0, mid);
                *l = mid_filt + side;
                *r = mid_filt - side;
            }
            StereoPlacement::Side => {
                let mid = 0.5 * (*l + *r);
                let side = 0.5 * (*l - *r);
                let side_filt = band.process_sample(1, side);
                *l = mid + side_filt;
                *r = mid - side_filt;
            }
        }
    }

    #[inline(always)]
    fn process_band_with_placement_svf(
        band: &mut FractionalSlopeBiquadBand,
        placement: StereoPlacement,
        channel_mode: ChannelMode,
        l: &mut f64,
        r: &mut f64,
    ) {
        match placement {
            StereoPlacement::Stereo => {
                *l = band.process_sample_svf(0, *l);
                *r = band.process_sample_svf(1, *r);
            }
            StereoPlacement::Left => {
                if matches!(channel_mode, ChannelMode::MidSide) {
                    let mid = 0.5 * (*l + *r);
                    let side = 0.5 * (*l - *r);
                    let mid_filt = band.process_sample_svf(0, mid);
                    *l = mid_filt + side;
                    *r = mid_filt - side;
                } else {
                    *l = band.process_sample_svf(0, *l);
                }
            }
            StereoPlacement::Right => {
                if matches!(channel_mode, ChannelMode::MidSide) {
                    let mid = 0.5 * (*l + *r);
                    let side = 0.5 * (*l - *r);
                    let side_filt = band.process_sample_svf(1, side);
                    *l = mid + side_filt;
                    *r = mid - side_filt;
                } else {
                    *r = band.process_sample_svf(1, *r);
                }
            }
            StereoPlacement::Mid => {
                let mid = 0.5 * (*l + *r);
                let side = 0.5 * (*l - *r);
                let mid_filt = band.process_sample_svf(0, mid);
                *l = mid_filt + side;
                *r = mid_filt - side;
            }
            StereoPlacement::Side => {
                let mid = 0.5 * (*l + *r);
                let side = 0.5 * (*l - *r);
                let side_filt = band.process_sample_svf(1, side);
                *l = mid + side_filt;
                *r = mid - side_filt;
            }
        }
    }
}

impl UnifiedDspNode for ProQ4Processor {
    fn name(&self) -> &'static str {
        "FabFilter Pro-Q 4 (24-Band)"
    }

    fn latency_samples(&self) -> usize {
        if self.params.global.bypass {
            return 0;
        }
        let mut lat = match self.params.global.processing_mode {
            ProcessingMode::ZeroLatency => 0,
            ProcessingMode::NaturalPhase => NATURAL_PHASE_BASE_LATENCY_SAMPLES,
            ProcessingMode::LinearPhase => self
                .params
                .global
                .processing_resolution
                .latency_samples(self.sample_rate as f32)
                .min(self.linear_phase_convolver.max_fir_len() / 2),
        };
        if self.has_active_brickwall_band
            && !matches!(
                self.params.global.processing_mode,
                ProcessingMode::LinearPhase
            )
        {
            lat += BRICKWALL_EXTRA_LATENCY_SAMPLES;
        }
        if self.has_active_spectral_band {
            lat += self.spectral_core.latency_samples();
        }
        lat
    }

    fn prepare(&mut self, sample_rate: f64, _max_block_size: usize) {
        self.sample_rate = sample_rate.max(8000.0);
        self.natural_phase_bank.set_sample_rate(self.sample_rate);
        self.spectral_core.set_sample_rate(self.sample_rate);
        self.character_shaper.set_sample_rate(self.sample_rate);
        for b in 0..PROQ4_NUM_BANDS {
            if self.params.bands[b].used {
                self.refresh_band_dsp(b, false);
            }
        }
        self.linear_phase_ir_dirty = true;
        self.refresh_global_aggregates(false);
    }

    fn reset_state(&mut self) {
        for b in &mut self.bands {
            b.reset_state();
        }
        self.natural_phase_bank.reset();
        self.linear_phase_convolver.reset_state();
        for d in &mut self.dynamic_bands {
            d.reset();
        }
        self.spectral_core.reset();
        self.character_shaper.reset();
        self.spectrum_analyzer.reset();
        self.brickwall_delay_ring = [[0.0; BRICKWALL_EXTRA_LATENCY_SAMPLES]; 2];
        self.brickwall_cursor = 0;
        self.current_gr_db = 0.0;
    }

    fn set_parameter_normalized(&mut self, param_id: u32, norm_val: f32) {
        let idx = param_id as usize;
        if idx >= PROQ4_TOTAL_PARAMS {
            return;
        }
        self.params.set_normalized(param_id, norm_val);
        if idx < PROQ4_BAND_PARAMS_TOTAL {
            let band_idx = idx / PROQ4_PARAMS_PER_BAND;
            let offset = idx % PROQ4_PARAMS_PER_BAND;
            // Automatically mark band as `used` when `Enabled` is toggled on
            if offset == 1 && norm_val >= 0.5 {
                self.params.bands[band_idx].used = true;
            }
            self.refresh_band_dsp(band_idx, true);
            // Mark Linear Phase IR dirty when static EQ band parameters change
            if matches!(offset, 0..=8) {
                self.linear_phase_ir_dirty = true;
            }
        } else {
            let g_offset = idx - PROQ4_BAND_PARAMS_TOTAL;
            if matches!(g_offset, 0 | 1 | 5 | 21 | 22) {
                for b in 0..PROQ4_NUM_BANDS {
                    if self.params.bands[b].used || self.band_enabled[b] {
                        self.refresh_band_dsp(b, true);
                    }
                }
                self.linear_phase_ir_dirty = true;
            }
        }
        self.refresh_global_aggregates(true);
    }

    fn current_gain_reduction_db(&self) -> f32 {
        self.current_gr_db
    }

    fn export_visual_telemetry(&self, frame: &mut VisualTelemetryFrame) {
        self.export_spectrum_frame(&mut frame.proq4_spectrum);
        let row = (frame.header.frame_seq as usize) % 20;
        frame.spectrogram_ring[row] = frame.proq4_spectrum.post_db;
    }

    fn process_slice(
        &mut self,
        _ctx: &ProcessContext,
        main_io: &mut [StereoFrame],
        sidechain_in: Option<&[StereoFrame]>,
        _aux_sends_out: &mut [&mut [StereoFrame]],
    ) {
        if self.params.global.bypass {
            return;
        }

        // Block-amortised linear-phase IR design (bounded RT cost per slice).
        if self.linear_phase_convolver.ir_design_pending() {
            self.continue_linear_phase_ir_design(LP_IR_DESIGN_EVALS_PER_SLICE);
        }

        let mode = self.params.global.processing_mode;
        let char_mode = self.params.global.character;
        let channel_mode = self.params.global.channel_mode;
        let gain_scale = self.params.global.gain_scale;
        let gq = self.params.global.gain_q_interaction;
        let out_level_lin = 10.0f64.powf((self.params.global.output_level_db as f64) / 20.0)
            * (self.auto_gain_linear as f64)
            * if self.params.global.output_invert_phase {
                -1.0
            } else {
                1.0
            };
        let pan = (self.params.global.output_pan as f64).clamp(-1.0, 1.0);
        let pan_l = (1.0 - pan).min(1.0);
        let pan_r = (1.0 + pan).min(1.0);
        let pan_mid_side = matches!(self.params.global.output_pan_mode, ChannelMode::MidSide);
        let tilt = self.params.global.analyzer_tilt_db_oct;
        let freeze = self.params.global.analyzer_freeze;

        // Check if any band has Solo or Sidechain Audition enabled
        let mut solo_band: Option<usize> = None;
        let mut sc_audition_band: Option<usize> = None;
        for b in 0..PROQ4_NUM_BANDS {
            if self.band_enabled[b] {
                if self.params.bands[b].sc_audition && sc_audition_band.is_none() {
                    sc_audition_band = Some(b);
                }
                if self.params.bands[b].solo && solo_band.is_none() {
                    solo_band = Some(b);
                }
            }
        }

        let mut min_gr_db = 0.0f64;

        for (i, frame) in main_io.iter_mut().enumerate() {
            // If Linear Phase IR was dirtied during a previous crossfade that has now finished,
            // trigger the coalesced IR rebuild before stepping the sample.
            if matches!(mode, ProcessingMode::LinearPhase)
                && self.linear_phase_ir_dirty
                && !self.linear_phase_convolver.is_crossfading()
            {
                self.rebuild_linear_phase_ir_now(true);
            }

            let in_l = frame[0] as f64;
            let in_r = frame[1] as f64;
            let (ext_sc_l, ext_sc_r) = match sidechain_in {
                Some(sc) if i < sc.len() => (sc[i][0] as f64, sc[i][1] as f64),
                _ => (in_l, in_r),
            };

            if let Some(sc_b) = sc_audition_band {
                let (trig_l, trig_r) = if self.params.bands[sc_b].ext_sidechain {
                    (ext_sc_l, ext_sc_r)
                } else {
                    (in_l, in_r)
                };
                let sc_l = self.dynamic_bands[sc_b]
                    .detector
                    .filter_sidechain_sample(0, trig_l);
                let sc_r = self.dynamic_bands[sc_b]
                    .detector
                    .filter_sidechain_sample(1, trig_r);
                frame[0] = (sc_l * out_level_lin) as f32;
                frame[1] = (sc_r * out_level_lin) as f32;
                continue;
            }

            // Step 1: Update Wideband Dynamic EQ (`Detector` + `ProgramDependentCapacitor`)
            for b in 0..PROQ4_NUM_BANDS {
                if self.band_enabled[b]
                    && self.params.bands[b].dyn_enabled
                    && !self.params.bands[b].spectral_enabled
                {
                    let (trig_l, trig_r) = if self.params.bands[b].ext_sidechain {
                        (ext_sc_l, ext_sc_r)
                    } else {
                        (in_l, in_r)
                    };
                    let dyn_db = self.dynamic_bands[b].step_dynamic_band_full(
                        &self.params.bands[b],
                        &mut self.bands[b],
                        if matches!(mode, ProcessingMode::NaturalPhase) {
                            Some(&mut self.natural_phase_bank.bands_2x[b])
                        } else {
                            None
                        },
                        trig_l,
                        trig_r,
                        gain_scale,
                        gq,
                        self.sample_rate,
                    );
                    if dyn_db < min_gr_db {
                        min_gr_db = dyn_db;
                    }
                }
            }

            // Step 2: Core EQ Processing (`ZeroLatency`, `NaturalPhase`, or `LinearPhase`)
            let (mut l, mut r) = if let Some(sb) = solo_band {
                let mut yl = in_l;
                let mut yr = in_r;
                let placement = self.params.bands[sb].placement;
                Self::process_band_with_placement(
                    &mut self.bands[sb],
                    placement,
                    channel_mode,
                    &mut yl,
                    &mut yr,
                );
                let solo_gain =
                    10.0f64.powf((self.params.global.solo_gain_db as f64) / 20.0);
                (yl * solo_gain, yr * solo_gain)
            } else {
                match mode {
                    ProcessingMode::ZeroLatency => {
                        let mut yl = in_l;
                        let mut yr = in_r;
                        for b in 0..PROQ4_NUM_BANDS {
                            if self.band_enabled[b] {
                                let placement = self.params.bands[b].placement;
                                Self::process_band_with_placement(
                                    &mut self.bands[b],
                                    placement,
                                    channel_mode,
                                    &mut yl,
                                    &mut yr,
                                );
                            }
                        }
                        (yl, yr)
                    }
                    ProcessingMode::NaturalPhase => {
                        let [u_l0, u_l1] =
                            self.natural_phase_bank.oversampler.upsample_step(0, in_l);
                        let [u_r0, u_r1] =
                            self.natural_phase_bank.oversampler.upsample_step(1, in_r);
                        let mut l0 = u_l0;
                        let mut r0 = u_r0;
                        let mut l1 = u_l1;
                        let mut r1 = u_r1;
                        for b in 0..PROQ4_NUM_BANDS {
                            if self.band_enabled[b] {
                                let placement = self.params.bands[b].placement;
                                Self::process_band_with_placement_svf(
                                    &mut self.natural_phase_bank.bands_2x[b],
                                    placement,
                                    channel_mode,
                                    &mut l0,
                                    &mut r0,
                                );
                                Self::process_band_with_placement_svf(
                                    &mut self.natural_phase_bank.bands_2x[b],
                                    placement,
                                    channel_mode,
                                    &mut l1,
                                    &mut r1,
                                );
                            }
                        }
                        let dl = self
                            .natural_phase_bank
                            .oversampler
                            .downsample_step(0, l0, l1);
                        let dr = self
                            .natural_phase_bank
                            .oversampler
                            .downsample_step(1, r0, r1);
                        (dl, dr)
                    }
                    ProcessingMode::LinearPhase => {
                        let conv_out = self
                            .linear_phase_convolver
                            .process_stereo_sample(in_l as f32, in_r as f32);
                        (conv_out[0] as f64, conv_out[1] as f64)
                    }
                }
            };

            // Step 2b: Brickwall Extra Latency Alignment (`320` samples when active in IIR modes)
            if self.has_active_brickwall_band && !matches!(mode, ProcessingMode::LinearPhase) {
                let bc = self.brickwall_cursor;
                let delayed_l = self.brickwall_delay_ring[0][bc];
                let delayed_r = self.brickwall_delay_ring[1][bc];
                self.brickwall_delay_ring[0][bc] = l;
                self.brickwall_delay_ring[1][bc] = r;
                self.brickwall_cursor = (bc + 1) % BRICKWALL_EXTRA_LATENCY_SAMPLES;
                l = delayed_l;
                r = delayed_r;
            }

            // Step 3: Per-Band Spectral Dynamics STFT (if any band has `spectral_enabled`)
            if self.has_active_spectral_band {
                let spec_out = self
                    .spectral_core
                    .process_stereo_sample(l as f32, r as f32);
                l = spec_out[0] as f64;
                r = spec_out[1] as f64;
            }

            // Step 4: Analog Harmonic Saturation (`Character`: `Clean` / `Subtle` / `Warm`)
            if !matches!(char_mode, CharacterMode::Clean) {
                l = self.character_shaper.process_sample(char_mode, 0, l);
                r = self.character_shaper.process_sample(char_mode, 1, r);
            }

            // Step 5: Output Gain, Auto-Gain, Polarity & Stereo/Mid-Side Pan
            if pan_mid_side {
                let mid = 0.5 * (l + r) * pan_l;
                let side = 0.5 * (l - r) * pan_r;
                l = (mid + side) * out_level_lin;
                r = (mid - side) * out_level_lin;
            } else {
                l *= out_level_lin * pan_l;
                r *= out_level_lin * pan_r;
            }

            frame[0] = l as f32;
            frame[1] = r as f32;

            // Step 6: Feed 1024-bin Spectrum Analyzer (`pre_db`, `post_db`, `sc_db`) & publish lock-free collision slot
            let pre_mono = 0.5 * (in_l + in_r) as f32;
            let post_mono = 0.5 * (l + r) as f32;
            let sc_mono = 0.5 * (ext_sc_l + ext_sc_r) as f32;
            if self.spectrum_analyzer.push_sample(
                pre_mono,
                post_mono,
                sc_mono,
                self.sample_rate,
                tilt,
                freeze,
            ) {
                self.collision_bus.publish_track_spectrum(
                    self.instance_id,
                    &self.spectrum_analyzer.latest_frame.post_db,
                );
            }
        }

        self.current_gr_db = min_gr_db as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fractional_slope_bell_and_proq4_processor() {
        let mut band = FractionalSlopeBiquadBand::new_unity();
        band.configure_bell(1000.0, 6.0, 1.0, 15.0, 48_000.0, true);
        // 15 dB/oct / 6 = 2.5 -> n1 = 2, n2 = 3, alpha_frac = 0.5
        assert_eq!(band.sections_n1, 2);
        assert_eq!(band.sections_n2, 3);
        assert!((band.alpha_frac - 0.5).abs() < 1e-9);

        let mut proc = ProQ4Processor::new();
        proc.band_enabled[0] = true;
        proc.bands[0].configure_bell(1000.0, 0.0, 1.0, 12.0, 48_000.0, true);
        let mut io = [[0.5f32, -0.5f32]; 32];
        proc.process_slice(&ProcessContext::default(), &mut io, None, &mut []);
        assert!((io[31][0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_eq_sketch_fitter_and_lockfree_cross_track_collision_65hz() {
        // 1. Target curve: Low-Cut 80 Hz (12 dB/oct), Bell -4.5 dB @ 350 Hz (Q=1.2), High-Shelf +3.0 dB @ 8 kHz (Q=0.8)
        let mut ref_bands = Vec::new();
        let mut lc = EqBandParams::default();
        lc.used = true;
        lc.enabled = true;
        lc.shape = FilterShape::LowCut;
        lc.freq_hz = 80.0;
        lc.q_ui = 1.0;
        lc.slope_db_oct = 12.0;
        ref_bands.push(lc);

        let mut bell = EqBandParams::default();
        bell.used = true;
        bell.enabled = true;
        bell.shape = FilterShape::Bell;
        bell.freq_hz = 350.0;
        bell.gain_db = -4.5;
        bell.q_ui = 1.2;
        bell.slope_db_oct = 12.0;
        ref_bands.push(bell);

        let mut hs = EqBandParams::default();
        hs.used = true;
        hs.enabled = true;
        hs.shape = FilterShape::HighShelf;
        hs.freq_hz = 8_000.0;
        hs.gain_db = 3.0;
        hs.q_ui = 0.85;
        hs.slope_db_oct = 12.0;
        ref_bands.push(hs);

        let mut sketch_pts = Vec::with_capacity(96);
        for i in 0..96 {
            let t = i as f64 / 95.0;
            let f = (25.0f64.log2() + t * (18_000.0f64.log2() - 25.0f64.log2())).exp2();
            let db = eval_bands_combined_db(&ref_bands, f, 48_000.0);
            sketch_pts.push((f as f32, db));
        }

        let fitted = fit_eq_sketch_curve(&sketch_pts, 6);
        assert!(!fitted.is_empty());

        let mut sq_sum = 0.0f64;
        let mut n_eval = 0usize;
        for i in 0..128 {
            let t = i as f64 / 127.0;
            let f = (80.0f64.log2() + t * (15_000.0f64.log2() - 80.0f64.log2())).exp2();
            let target = eval_bands_combined_db(&ref_bands, f, 48_000.0) as f64;
            let got = eval_bands_combined_db(&fitted, f, 48_000.0) as f64;
            let d = target - got;
            sq_sum += d * d;
            n_eval += 1;
        }
        let rms_err_db = (sq_sum / (n_eval as f64)).sqrt();
        assert!(
            rms_err_db < 0.85,
            "EQ Sketch RMS error must be < 0.85 dB over 80 Hz..15 kHz, got {rms_err_db:.4} dB"
        );

        // 2. Lock-free Cross-Track Collision Bus: Track 1 (Kick 65 Hz) vs Track 2 (Bass 65 Hz + 130 Hz)
        let bus = ProQ4CollisionBus::new_arc();
        let mut kick_eq = ProQ4Processor::with_collision_bus(1, bus.clone());
        let mut bass_eq = ProQ4Processor::with_collision_bus(2, bus.clone());
        kick_eq.params.global.show_collisions = true;
        bass_eq.params.global.show_collisions = true;
        kick_eq.prepare(48_000.0, 512);
        bass_eq.prepare(48_000.0, 512);

        let sr = 48_000.0f64;
        let ctx = ProcessContext {
            sample_rate: sr,
            slice_frames: 512,
            ..ProcessContext::default()
        };
        let mut kick_buf = [[0.0f32; 2]; 512];
        let mut bass_buf = [[0.0f32; 2]; 512];
        for blk in 0..12 {
            let base = blk * 512;
            for i in 0..512 {
                let t = (base + i) as f64 / sr;
                let k = (0.5 * (std::f64::consts::TAU * 65.0 * t).sin()) as f32;
                let b = (0.4 * (std::f64::consts::TAU * 65.0 * t).sin()
                    + 0.45 * (std::f64::consts::TAU * 130.0 * t).sin())
                    as f32;
                kick_buf[i] = [k, k];
                bass_buf[i] = [b, b];
            }
            kick_eq.process_slice(&ctx, &mut kick_buf, None, &mut []);
            bass_eq.process_slice(&ctx, &mut bass_buf, None, &mut []);
        }

        let mut frame = ProQ4SpectrumFrame::default();
        kick_eq.export_spectrum_frame(&mut frame);
        let (peak_bin, &peak_val) = frame
            .collision_mask
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let peak_hz = ProQ4CollisionBus::bin_to_freq_hz(peak_bin);
        assert!(peak_val > 0.35, "Expected strong collision mask > 0.35, got {peak_val}");
        assert!(
            (peak_hz - 65.0).abs() <= 5.0,
            "Collision peak must land within 65 Hz ± 5 Hz, got {peak_hz:.2} Hz (bin {peak_bin})"
        );
    }
}
