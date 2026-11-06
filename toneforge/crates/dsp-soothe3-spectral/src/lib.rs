//! `dsp-soothe3-spectral`: Clean-Room Rust Implementation of **oeksound Soothe3** (`v1.0.5`)
//! Dynamic Resonance Suppressor for **ToneForge**.
//!
//! Architecture (`05_master_rust_svelte_blueprint.md` Section 5.3 & `02_proq4_and_soothe3_dsp.md` Section 2 & 3.2):
//! - `params_143`: Complete 143-parameter system (`26 Global + 8×9 Sidechain EQ + 45 Multichannel/Atmos`)
//!   matching `DigiPageTable.xml` and `.preset` JSON files.
//! - `polyphase_resampler`: `BaseRateResampler` (`16-tap` polyphase FIR) normalizing 2x/4x sample rates to
//!   `f_base` (`44.1 / 48 kHz`) while preserving the ultrasonic band (`> f_base / 2`) on the dry/wet merge branch.
//! - `sidechain_eq_8band`: 8-Band Hybrid Series / Parallel Sidechain EQ (`shape == 3` Bandpass routed in parallel
//!   and multiplied by series cascade) with per-band `focus` and Mid/Side balance.
//! - `tilt_curves`: Frequency Tilt Curves around `500 Hz` and `2000 Hz` for `Detail`, `Attack`, and `Release`.
//! - `resonance_detector`: Psychoacoustic ERB / Bark local resonance detector (`Soft` vs `Hard` modes,
//!   `Depth`, `Detail`, `Max cut`, per-bin `Attack` / `Release` ballistics, `Stereo link`).
//! - `cepstral_min_phase`: `CepstralMinPhaseReconstructor` (`2,304` samples latency @ 1x Regular,
//!   `3,456` samples @ 1x Linear-Phase, `0` samples @ 1x Low-Latency) with zero pre-ringing on transients
//!   and direct telemetry export to `Soothe3ReductionFrame` (`8 KB`).

pub mod cepstral_min_phase;
pub mod params_143;
pub mod polyphase_resampler;
pub mod resonance_detector;
pub mod sidechain_eq_8band;
pub mod tilt_curves;

pub use cepstral_min_phase::{
    CepstralMinPhaseReconstructor, Soothe3SpectralPipeline, SOOTHE3_BINS, SOOTHE3_FFT_SIZE,
    SOOTHE3_LINEAR_PHASE_LATENCY_1X, SOOTHE3_REGULAR_LATENCY_1X,
};
pub use params_143::{
    SidechainEqBandParams, SidechainFilterShape, Soothe3GlobalParams, Soothe3ParameterSet,
    Soothe3PresetJson, SootheProcessMode, SpeakerGroupParams, TimeResolutionQuality,
    SOOTHE3_TOTAL_PARAMS,
};
pub use polyphase_resampler::BaseRateResampler;
pub use resonance_detector::ResonanceDetector;
pub use sidechain_eq_8band::SidechainEq8Band;
pub use tilt_curves::TiltPair;

use daw_core_graph::{
    ProcessContext, Soothe3ReductionFrame, StereoFrame, UnifiedDspNode, VisualTelemetryFrame,
};

/// Top-level **oeksound Soothe3** processor implementing `UnifiedDspNode` with strict
/// real-time zero-allocation guarantees.
pub struct Soothe3Processor {
    pub params: Soothe3ParameterSet,
    params_dirty: bool,
    resampler: BaseRateResampler,
    pipeline: Soothe3SpectralPipeline,
}

impl Default for Soothe3Processor {
    fn default() -> Self {
        Self::new(48000.0)
    }
}

impl Soothe3Processor {
    pub fn new(sample_rate: f32) -> Self {
        let resampler = BaseRateResampler::new(sample_rate);
        let base_sr = resampler.base_sample_rate();
        let params = Soothe3ParameterSet::new();
        let mut pipeline = Soothe3SpectralPipeline::new(base_sr);
        pipeline.sidechain_eq.update_weights(
            &params.sc_bands,
            params.global.stereo_focus,
            SOOTHE3_FFT_SIZE,
            base_sr,
        );
        pipeline.detector.update_tilt_tables(
            &params.global,
            params.global.quality.hop_size_1x(),
            base_sr,
        );

        Self {
            params,
            params_dirty: false,
            resampler,
            pipeline,
        }
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.resampler.set_sample_rate(sample_rate);
        let base_sr = self.resampler.base_sample_rate();
        self.pipeline.update_sample_rate(base_sr);
        self.params_dirty = true;
        self.sync_dirty_tables();
    }

    #[inline]
    pub fn params(&self) -> &Soothe3ParameterSet {
        &self.params
    }

    #[inline]
    pub fn params_mut(&mut self) -> &mut Soothe3ParameterSet {
        self.params_dirty = true;
        &mut self.params
    }

    #[inline]
    pub fn resampler(&self) -> &BaseRateResampler {
        &self.resampler
    }

    #[inline]
    pub fn pipeline(&self) -> &Soothe3SpectralPipeline {
        &self.pipeline
    }

    #[inline]
    pub fn pipeline_mut(&mut self) -> &mut Soothe3SpectralPipeline {
        &mut self.pipeline
    }

    /// Loads a `.preset` JSON string into the processor and updates all filter/tilt tables.
    pub fn load_preset_json(&mut self, json_str: &str) -> Result<usize, String> {
        let count = self.params.load_from_preset_json(json_str)?;
        self.params_dirty = true;
        self.sync_dirty_tables();
        Ok(count)
    }

    #[inline]
    pub fn sync_dirty_tables(&mut self) {
        if !self.params_dirty {
            return;
        }
        self.params_dirty = false;
        let base_sr = self.resampler.base_sample_rate();
        self.pipeline.sidechain_eq.update_weights(
            &self.params.sc_bands,
            self.params.global.stereo_focus,
            SOOTHE3_FFT_SIZE,
            base_sr,
        );
        self.pipeline.detector.update_tilt_tables(
            &self.params.global,
            self.params.global.quality.hop_size_1x(),
            base_sr,
        );
    }

    /// Copies the latest 1,024-point reduction curve and spectrum into `Soothe3ReductionFrame` (`8 KB`).
    pub fn export_reduction_frame(&self, out_frame: &mut Soothe3ReductionFrame) {
        self.pipeline.export_telemetry(out_frame);
    }
}

impl UnifiedDspNode for Soothe3Processor {
    fn name(&self) -> &'static str {
        "oeksound Soothe3 Dynamic Resonance Suppressor"
    }

    fn latency_samples(&self) -> usize {
        self.resampler.total_latency_samples(
            self.params.global.linear_phase,
            self.params.global.low_latency,
        ) as usize
    }

    fn prepare(&mut self, sample_rate: f64, _max_block_size: usize) {
        self.set_sample_rate(sample_rate as f32);
        self.reset_state();
    }

    fn reset_state(&mut self) {
        self.resampler.reset();
        self.pipeline.reset();
    }

    fn set_parameter_normalized(&mut self, param_id: u32, norm_val: f32) {
        if (param_id as usize) < SOOTHE3_TOTAL_PARAMS {
            self.params.set_normalized(param_id, norm_val);
            self.params_dirty = true;
            self.sync_dirty_tables();
        }
    }

    fn current_gain_reduction_db(&self) -> f32 {
        self.pipeline.peak_gr_db
    }

    fn export_visual_telemetry(&self, frame: &mut VisualTelemetryFrame) {
        self.export_reduction_frame(&mut frame.soothe3_reduction);
        let row = (frame.header.frame_seq as usize) % 20;
        frame.spectrogram_ring[row] = frame.soothe3_reduction.input_spec_db;
    }

    fn process_slice(
        &mut self,
        ctx: &ProcessContext,
        main_io: &mut [StereoFrame],
        sidechain_in: Option<&[StereoFrame]>,
        _aux_sends_out: &mut [&mut [StereoFrame]],
    ) {
        if main_io.is_empty() {
            return;
        }

        if (ctx.sample_rate as f32 - self.resampler.sample_rate()).abs() > 1.0 {
            self.resampler.set_sample_rate(ctx.sample_rate as f32);
            let base_sr = self.resampler.base_sample_rate();
            self.pipeline.update_sample_rate(base_sr);
            self.params_dirty = true;
        }

        self.sync_dirty_tables();

        if self.params.global.bypass {
            return;
        }

        let mix = (self.params.global.mix_pct * 0.01).clamp(0.0, 1.0);
        let trim_lin = 10.0f32.powf(self.params.global.out_trim_db / 20.0);
        let delta_solo = self.params.global.delta_solo;
        let sc_listen = self.params.global.sidechain_listen;
        let use_ext_sc = self.params.global.sidechain;
        let factor = self.resampler.oversample_factor();
        let total_lat = self.resampler.total_latency_samples(
            self.params.global.linear_phase,
            self.params.global.low_latency,
        ) as usize;

        for (i, frame) in main_io.iter_mut().enumerate() {
            let in_l = frame[0];
            let in_r = frame[1];

            let (sc_l, sc_r) = match sidechain_in {
                Some(sc) if use_ext_sc && i < sc.len() => (sc[i][0], sc[i][1]),
                _ => (in_l, in_r),
            };

            let (out_l, out_r) = if factor == 1 {
                let (wet, dry) =
                    self.pipeline
                        .step_base_rate([in_l, in_r], [sc_l, sc_r], &self.params);
                if sc_listen {
                    (wet[0] * trim_lin, wet[1] * trim_lin)
                } else if delta_solo {
                    ((dry[0] - wet[0]) * trim_lin, (dry[1] - wet[1]) * trim_lin)
                } else {
                    (
                        ((1.0 - mix) * dry[0] + mix * wet[0]) * trim_lin,
                        ((1.0 - mix) * dry[1] + mix * wet[1]) * trim_lin,
                    )
                }
            } else {
                // High sample rate (2x / 4x): preserve full-bandwidth ultrasonic spectrum (> f_base / 2)
                // via zero-phase ultrasonic isolation at `total_lat` combined with `wet_base_up`!
                let (delayed_high_dry, ultra_high) = self
                    .resampler
                    .push_and_extract_ultrasonic_at_delay(in_l, in_r, total_lat);
                if let Some((base_main, base_sc)) = self
                    .resampler
                    .push_high_rate_frame([in_l, in_r], [sc_l, sc_r])
                {
                    let (wet_base, dry_base) =
                        self.pipeline
                            .step_base_rate(base_main, base_sc, &self.params);
                    self.resampler.push_base_rate_wet_and_dry(wet_base, dry_base);
                }
                let (wet_base_up, dry_base_up) = self.resampler.next_upsampled_wet_and_dry();
                if sc_listen {
                    (wet_base_up[0] * trim_lin, wet_base_up[1] * trim_lin)
                } else if delta_solo {
                    (
                        (dry_base_up[0] - wet_base_up[0]) * trim_lin,
                        (dry_base_up[1] - wet_base_up[1]) * trim_lin,
                    )
                } else {
                    let wet_full_l = wet_base_up[0] + ultra_high[0];
                    let wet_full_r = wet_base_up[1] + ultra_high[1];
                    (
                        ((1.0 - mix) * delayed_high_dry[0] + mix * wet_full_l) * trim_lin,
                        ((1.0 - mix) * delayed_high_dry[1] + mix * wet_full_r) * trim_lin,
                    )
                }
            };

            frame[0] = out_l;
            frame[1] = out_r;
        }
    }
}
