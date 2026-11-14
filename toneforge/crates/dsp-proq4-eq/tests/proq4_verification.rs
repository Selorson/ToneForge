//! Comprehensive Verification Suite for `dsp-proq4-eq` (FabFilter Pro-Q 4 Clean-Room DSP).
//!
//! Covers:
//! 1. Full 601-parameter system (`24 Bands × 24 Params + 25 Global Params`) & Stereo/L/R/M/S routing.
//! 2. Magnitude and phase responses of all 9 EQ filter shapes (`Bell`, `LowShelf`, `HighShelf`,
//!    `LowCut`, `HighCut`, `Notch`, `BandPass`, `TiltShelf`, `FlatTilt`) + `AllPass`.
//! 3. Fractional-Slope Dual Biquad Cascade (`6..96 dB/oct`, bitmask `0x10F6`) & Gain-Q interaction
//!    (`1.0905077326652577`, bitmask `0x1074`).
//! 4. `NaturalPhase` mode (analog complex magnitude & phase matching near Nyquist).
//! 5. `LinearPhase` FIR mode with Kaiser 60 dB window (`bessel_i0`) and Recursive Non-Uniform
//!    Partitioned Overlap-Save Convolver (`64 -> 512 -> 4096 -> 32768`) + Dual-IR Crossfading.
//! 6. `ProgramDependentCapacitor` (`f64`) Dynamic EQ, per-band `SpectralDynamicsCore` STFT,
//!    `Character` saturation (`Clean` / `Subtle` / `Warm`), 512-bin 8-knot `PerceptualAutoGain`,
//!    and `ProQ4SpectrumFrame` (`16 KB`) telemetry export.

use daw_core_graph::{ProcessContext, ProQ4SpectrumFrame, UnifiedDspNode};
use dsp_proq4_eq::{
    analog_prototype_mag_db, bessel_i0, compute_kaiser_60db_window, eval_analog_s_plane_response,
    AnalogFilterPrototype, CharacterMode, CharacterWaveshaper, FilterShape,
    FractionalSlopeBiquadBand, LinearPhaseResolution, NaturalPhaseOversampler2x,
    NonUniformPartitionedConvolver, PerceptualAutoGain, ProQ4ParameterSet, ProQ4Processor,
    ProcessingMode, ProgramDependentCapacitor, StereoPlacement, PROQ4_GAIN_DEPENDENT_MASK,
    PROQ4_GAIN_Q_INTERACTION_BASE, PROQ4_GAIN_Q_INTERACTION_MASK, PROQ4_TOTAL_PARAMS,
};
use rustfft::{num_complex::Complex64, FftPlanner};
use std::f64::consts::PI;

#[test]
fn test_601_parameter_system_and_placement_routing() {
    let mut params = ProQ4ParameterSet::new();
    assert_eq!(PROQ4_TOTAL_PARAMS, 601);

    // Verify band and global parameter name resolution from DigiPageTable.xml
    assert_eq!(ProQ4ParameterSet::find_param_id_by_name("Band 1 Used"), Some(0));
    assert_eq!(
        ProQ4ParameterSet::find_param_id_by_name("Band 1 Enabled"),
        Some(1)
    );
    assert_eq!(
        ProQ4ParameterSet::find_param_id_by_name("Band 1 Frequency"),
        Some(2)
    );
    assert_eq!(
        ProQ4ParameterSet::find_param_id_by_name("Band 24 Invert Gain"),
        Some(575)
    );
    assert_eq!(
        ProQ4ParameterSet::find_param_id_by_name("Processing Mode"),
        Some(576)
    );
    assert_eq!(
        ProQ4ParameterSet::find_param_id_by_name("Character"),
        Some(578)
    );
    assert_eq!(ProQ4ParameterSet::find_param_id_by_name("Bypass"), Some(583));

    // Verify roundtrip normalized <-> structured mapping for all 601 parameters
    for id in 0..(PROQ4_TOTAL_PARAMS as u32) {
        params.set_normalized(id, 0.65);
        let readback = params.get_normalized(id);
        assert!(
            (readback - 0.65).abs() < 1e-5,
            "Param {} roundtrip mismatch",
            id
        );
    }

    // Test Left-only vs Right-only vs Mid-only vs Side-only placement routing
    let mut proc = ProQ4Processor::new();
    proc.prepare(48_000.0, 256);

    // Configure Band 0 as a -18 dB Bell at 1000 Hz routed to LEFT only
    proc.params.bands[0].used = true;
    proc.params.bands[0].enabled = true;
    proc.params.bands[0].shape = FilterShape::Bell;
    proc.params.bands[0].freq_hz = 1000.0;
    proc.params.bands[0].gain_db = -18.0;
    proc.params.bands[0].q_ui = 1.0;
    proc.params.bands[0].placement = StereoPlacement::Left;
    proc.refresh_band_dsp(0, false);

    let ctx = ProcessContext::default();
    let mut frames = [[0.0f32; 2]; 1024];
    for (n, frame) in frames.iter_mut().enumerate() {
        let s = (2.0 * PI * 1000.0 * (n as f64) / 48_000.0).sin() as f32;
        *frame = [s, s];
    }
    proc.process_slice(&ctx, &mut frames, None, &mut []);

    let mut rms_l = 0.0f64;
    let mut rms_r = 0.0f64;
    for frame in &frames[512..] {
        rms_l += (frame[0] as f64).powi(2);
        rms_r += (frame[1] as f64).powi(2);
    }
    rms_l = (rms_l / 512.0).sqrt();
    rms_r = (rms_r / 512.0).sqrt();

    // Left channel should be attenuated by ~18 dB (~0.126x), while Right channel stays at unity (~0.707)
    assert!(
        rms_l < 0.15 && (rms_r - 0.7071).abs() < 0.02,
        "Left placement failed: rms_l={}, rms_r={}",
        rms_l,
        rms_r
    );

    // Now switch Band 0 to MID only and feed a pure SIDE signal (L = +s, R = -s) -> should pass untouched!
    proc.reset_state();
    proc.params.bands[0].placement = StereoPlacement::Mid;
    proc.refresh_band_dsp(0, false);
    for (n, frame) in frames.iter_mut().enumerate() {
        let s = (2.0 * PI * 1000.0 * (n as f64) / 48_000.0).sin() as f32;
        *frame = [s, -s];
    }
    proc.process_slice(&ctx, &mut frames, None, &mut []);
    let mut rms_side = 0.0f64;
    for frame in &frames[512..] {
        rms_side += (frame[0] as f64).powi(2);
    }
    rms_side = (rms_side / 512.0).sqrt();
    assert!(
        (rms_side - 0.7071).abs() < 0.02,
        "Mid placement affected pure Side signal: rms_side={}",
        rms_side
    );
}

#[test]
fn test_all_9_filter_shapes_plus_allpass_and_fractional_slope() {
    let fs = 48_000.0;
    let mut band = FractionalSlopeBiquadBand::new_unity();

    assert_eq!(PROQ4_GAIN_DEPENDENT_MASK, 0x10F6);
    assert_eq!(PROQ4_GAIN_Q_INTERACTION_MASK, 0x1074);
    assert!(AnalogFilterPrototype::Bell.is_gain_dependent());
    assert!(!AnalogFilterPrototype::LowCut.is_gain_dependent());

    // 1. Bell (+12 dB at 1 kHz)
    band.configure_shape(FilterShape::Bell, 1000.0, 12.0, 1.0, 12.0, fs, true);
    let mag_center_db = 20.0 * band.eval_complex_response(1000.0, fs).norm().log10();
    let mag_dc_db = 20.0 * band.eval_complex_response(20.0, fs).norm().log10();
    assert!((mag_center_db - 12.0).abs() < 0.25);
    assert!(mag_dc_db.abs() < 0.25);

    // 2. LowShelf (+12 dB below 500 Hz)
    band.configure_shape(FilterShape::LowShelf, 500.0, 12.0, 1.0, 12.0, fs, true);
    let low_db = 20.0 * band.eval_complex_response(30.0, fs).norm().log10();
    let high_db = 20.0 * band.eval_complex_response(10_000.0, fs).norm().log10();
    assert!((low_db - 12.0).abs() < 0.5);
    assert!(high_db.abs() < 0.3);

    // 3. HighShelf (+12 dB above 2000 Hz)
    band.configure_shape(FilterShape::HighShelf, 2000.0, 12.0, 1.0, 12.0, fs, true);
    let low_db = 20.0 * band.eval_complex_response(100.0, fs).norm().log10();
    let high_db = 20.0 * band.eval_complex_response(16_000.0, fs).norm().log10();
    assert!(low_db.abs() < 0.3);
    assert!((high_db - 12.0).abs() < 1.0);

    // 4. LowCut (High-pass at 1000 Hz) with Fractional Slope interpolation (12.0 -> 15.0 -> 18.0 -> 24.0 -> 96.0 dB/oct)
    let mut prev_atten_db = 0.0;
    for &slope in &[12.0, 15.0, 18.0, 21.0, 24.0, 48.0, 96.0] {
        band.configure_shape(FilterShape::LowCut, 1000.0, 0.0, 1.0, slope, fs, false);
        let mag_500_db = 20.0 * band.eval_complex_response(500.0, fs).norm().log10();
        assert!(
            mag_500_db < prev_atten_db - 1.5,
            "Fractional slope {} dB/oct not monotonic: {} dB vs prev {} dB",
            slope,
            mag_500_db,
            prev_atten_db
        );
        prev_atten_db = mag_500_db;
    }

    // 5. HighCut (Low-pass at 2000 Hz, 24 dB/oct)
    band.configure_shape(FilterShape::HighCut, 2000.0, 0.0, 1.0, 24.0, fs, false);
    let pass_db = 20.0 * band.eval_complex_response(200.0, fs).norm().log10();
    let stop_db = 20.0 * band.eval_complex_response(8000.0, fs).norm().log10();
    assert!(pass_db.abs() < 0.2);
    assert!(stop_db < -40.0);

    // 6. Notch (1000 Hz)
    band.configure_shape(FilterShape::Notch, 1000.0, 0.0, 2.0, 12.0, fs, false);
    let notch_db = 20.0 * band.eval_complex_response(1000.0, fs).norm().log10();
    assert!(notch_db < -50.0);

    // 7. BandPass (1000 Hz)
    band.configure_shape(FilterShape::BandPass, 1000.0, 0.0, 2.0, 12.0, fs, false);
    let bp_center_db = 20.0 * band.eval_complex_response(1000.0, fs).norm().log10();
    let bp_skirt_db = 20.0 * band.eval_complex_response(100.0, fs).norm().log10();
    assert!(bp_center_db.abs() < 0.5);
    assert!(bp_skirt_db < -20.0);

    // 8. TiltShelf (+12 dB at 1000 Hz: -6 dB low, +6 dB high)
    band.configure_shape(FilterShape::TiltShelf, 1000.0, 12.0, 1.0, 12.0, fs, true);
    let tilt_low_db = 20.0 * band.eval_complex_response(50.0, fs).norm().log10();
    let tilt_high_db = 20.0 * band.eval_complex_response(15_000.0, fs).norm().log10();
    assert!(tilt_low_db < -4.0 && tilt_high_db > 4.0);

    // 9. FlatTilt (+6 dB tilt across spectrum)
    band.configure_shape(FilterShape::FlatTilt, 1000.0, 6.0, 1.0, 12.0, fs, false);
    let ft_low_db = 20.0 * band.eval_complex_response(50.0, fs).norm().log10();
    let ft_high_db = 20.0 * band.eval_complex_response(15_000.0, fs).norm().log10();
    assert!(ft_low_db < -1.0 && ft_high_db > 1.0);

    // 10. AllPass (unity magnitude 0 dB everywhere)
    band.configure_shape(FilterShape::AllPass, 1000.0, 0.0, 1.0, 12.0, fs, false);
    for &f in &[50.0, 500.0, 1000.0, 5000.0, 15_000.0] {
        let mag_db = 20.0 * band.eval_complex_response(f, fs).norm().log10();
        assert!(mag_db.abs() < 1e-4, "AllPass magnitude not unity at {} Hz", f);
    }

    // Verify Gain-Q interaction constant 1.0905077326652577 = 2^(1/8) and bitmask 0x1074
    assert!((PROQ4_GAIN_Q_INTERACTION_BASE - 2.0f64.powf(0.125)).abs() < 1e-15);
    assert!(FilterShape::Bell.has_gain_q_interaction());
    assert!(FilterShape::LowShelf.has_gain_q_interaction());
    assert!(!FilterShape::FlatTilt.has_gain_q_interaction());
    assert!(!FilterShape::LowCut.has_gain_q_interaction());
}

#[test]
fn test_natural_phase_and_linear_phase_partitioned_convolver() {
    // 1. Verify Kaiser 60 dB window symmetry & Bessel I0 accuracy
    assert!((bessel_i0(0.0) - 1.0).abs() < 1e-12);
    let mut win = vec![0.0f32; 256];
    compute_kaiser_60db_window(&mut win);
    assert!((win[128] - 1.0).abs() < 1e-3);
    assert!((win[0] - win[255]).abs() < 1e-6);
    assert!(win[0] < 0.025);

    // 2. Verify NonUniformPartitionedConvolver (64 -> 512 -> 4096 -> 32768) exact impulse reproduction
    // across all partition boundaries (tap 10 in direct head, tap 100 in Tier 0, tap 1000 in Tier 1, tap 5000 in Tier 2)
    let mut conv = NonUniformPartitionedConvolver::new(16384);
    let mut test_ir = vec![0.0f32; 8192];
    test_ir[10] = 0.5;
    test_ir[100] = -0.25;
    test_ir[1000] = 0.125;
    test_ir[5000] = 0.0625;
    conv.set_impulse_response(&test_ir, false);

    let mut impulse_out = Vec::with_capacity(6000);
    for n in 0..6000 {
        let x = if n == 0 { 1.0 } else { 0.0 };
        let y = conv.process_stereo_sample(x, x);
        impulse_out.push(y[0]);
    }

    assert!(
        (impulse_out[10] - 0.5).abs() < 1e-4,
        "Direct head tap 10 mismatch: {}",
        impulse_out[10]
    );
    assert!(
        (impulse_out[100] - (-0.25)).abs() < 1e-4,
        "Tier 0 (64) tap 100 mismatch: {}",
        impulse_out[100]
    );
    assert!(
        (impulse_out[1000] - 0.125).abs() < 1e-4,
        "Tier 1 (512) tap 1000 mismatch: {}",
        impulse_out[1000]
    );
    assert!(
        (impulse_out[5000] - 0.0625).abs() < 1e-4,
        "Tier 2 (4096) tap 5000 mismatch: {}",
        impulse_out[5000]
    );

    // 3. Test Dual-IR Crossfading on parameter change during steady-state signal (no discontinuities)
    for _ in 0..6000 {
        conv.process_stereo_sample(1.0, 1.0);
    }
    let mut prev_y = conv.process_stereo_sample(1.0, 1.0)[0];
    let mut new_ir = vec![0.0f32; 8192];
    new_ir[0] = 0.25;
    conv.set_impulse_response(&new_ir, true);
    assert!(conv.is_crossfading());
    let mut max_step = 0.0f32;
    for _ in 0..3000 {
        let y = conv.process_stereo_sample(1.0, 1.0)[0];
        max_step = max_step.max((y - prev_y).abs());
        prev_y = y;
    }
    assert!(!conv.is_crossfading());
    assert!((prev_y - 0.25).abs() < 1e-4);
    assert!(
        max_step < 0.005,
        "Dual-IR crossfade produced step {}",
        max_step
    );

    // 4. Test Linear Phase & Natural Phase inside ProQ4Processor
    let mut proc = ProQ4Processor::new();
    proc.prepare(48_000.0, 256);
    proc.params.global.processing_mode = ProcessingMode::LinearPhase;
    proc.params.global.processing_resolution = LinearPhaseResolution::Low; // 4096 taps -> 2048 latency
    proc.params.bands[0].used = true;
    proc.params.bands[0].enabled = true;
    proc.params.bands[0].shape = FilterShape::Bell;
    proc.params.bands[0].freq_hz = 1000.0;
    proc.params.bands[0].gain_db = -12.0;
    proc.refresh_band_dsp(0, false);
    proc.refresh_global_aggregates(false);

    assert_eq!(proc.latency_samples(), 2048);
}

#[test]
fn test_dynamic_rc_spectral_dynamics_character_autogain_and_telemetry() {
    // 1. ProgramDependentCapacitor (f64) attack/release verification
    let mut cap = ProgramDependentCapacitor::new();
    cap.set_ballistics_pct(20.0, 50.0);
    for _ in 0..2400 {
        cap.step(-12.0, 48_000.0);
    }
    assert!(
        cap.env_state_db < -8.0,
        "Capacitor did not charge toward -12 dB: {}",
        cap.env_state_db
    );
    for _ in 0..4800 {
        cap.step(0.0, 48_000.0);
    }
    assert!(
        cap.env_state_db > -6.0,
        "Capacitor did not release toward 0 dB: {}",
        cap.env_state_db
    );

    // 2. PerceptualAutoGain 512-bin 8-knot table verification
    let ag = PerceptualAutoGain::new();
    let flat_plus_6db = [4.0f32; 512]; // |H|^2 = 4.0 -> |H| = 2.0 (+6 dB)
    let comp = ag.compute_compensation_linear(&flat_plus_6db);
    assert!(
        (comp - 0.5).abs() < 1e-4,
        "PerceptualAutoGain +6dB compensation should be 0.5, got {}",
        comp
    );

    // 3. Full ProQ4Processor with Dynamic EQ + Spectral Dynamics + Warm Character + Spectrum Telemetry
    let mut proc = ProQ4Processor::new();
    proc.prepare(48_000.0, 512);
    proc.params.global.character = CharacterMode::Warm;
    proc.params.global.auto_gain = true;

    // Band 0: Wideband Dynamic EQ compressing 1 kHz
    proc.params.bands[0].used = true;
    proc.params.bands[0].enabled = true;
    proc.params.bands[0].shape = FilterShape::Bell;
    proc.params.bands[0].freq_hz = 1000.0;
    proc.params.bands[0].gain_db = 0.0;
    proc.params.bands[0].dyn_enabled = true;
    proc.params.bands[0].dyn_range_db = -12.0;
    proc.params.bands[0].dyn_threshold_db = -24.0;
    proc.refresh_band_dsp(0, false);

    // Band 1: Spectral Dynamics Bell at 3 kHz
    proc.params.bands[1].used = true;
    proc.params.bands[1].enabled = true;
    proc.params.bands[1].shape = FilterShape::Bell;
    proc.params.bands[1].freq_hz = 3000.0;
    proc.params.bands[1].q_ui = 1.0;
    proc.params.bands[1].dyn_enabled = true;
    proc.params.bands[1].dyn_range_db = -15.0;
    proc.params.bands[1].dyn_threshold_db = -30.0;
    proc.params.bands[1].spectral_enabled = true;
    proc.params.bands[1].spectral_density = 75.0;
    proc.refresh_band_dsp(1, false);
    proc.refresh_global_aggregates(false);

    let ctx = ProcessContext::default();
    let mut block = [[0.0f32; 2]; 512];
    for b in 0..16 {
        for (i, frame) in block.iter_mut().enumerate() {
            let n = b * 512 + i;
            let s1 = 0.5 * (2.0 * PI * 1000.0 * (n as f64) / 48_000.0).sin() as f32;
            let s2 = 0.5 * (2.0 * PI * 3000.0 * (n as f64) / 48_000.0).sin() as f32;
            *frame = [s1 + s2, s1 + s2];
        }
        proc.process_slice(&ctx, &mut block, None, &mut []);
    }

    assert!(
        proc.current_gain_reduction_db() < -1.0,
        "Dynamic EQ did not trigger gain reduction: {}",
        proc.current_gain_reduction_db()
    );

    let mut spec_frame = ProQ4SpectrumFrame::default();
    assert_eq!(std::mem::size_of::<ProQ4SpectrumFrame>(), 16384);
    proc.export_spectrum_frame(&mut spec_frame);
    let max_pre_db = spec_frame
        .pre_db
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max_pre_db > -40.0,
        "ProQ4SpectrumFrame pre_db was not populated: max_pre_db={}",
        max_pre_db
    );
}

#[test]
fn test_linear_phase_tier3_131072_taps_and_tier2_dual_ir_crossfade_continuity() {
    // 1. Verify Tier 3 (P = 32768) impulse reproduction at tap 40,000 and Maximum 131,072-tap resolution
    let mut conv = NonUniformPartitionedConvolver::new(131_072);
    let mut ir_large = vec![0.0f32; 65_536];
    ir_large[40_000] = 0.75;
    conv.set_impulse_response(&ir_large, false);

    let mut tap_40k_out = 0.0f32;
    for n in 0..=40_005 {
        let x = if n == 0 { 1.0 } else { 0.0 };
        let y = conv.process_stereo_sample(x, x);
        if n == 40_000 {
            tap_40k_out = y[0];
        }
    }
    assert!(
        (tap_40k_out - 0.75).abs() < 1e-4,
        "Tier 3 (32768) tap 40000 mismatch: got {}",
        tap_40k_out
    );

    // 2. Verify LinearPhaseResolution::Maximum (131,072 taps -> 65,536 latency samples) APDC alignment
    let mut proc = ProQ4Processor::new();
    proc.prepare(48_000.0, 512);
    proc.params.global.processing_mode = ProcessingMode::LinearPhase;
    proc.params.global.processing_resolution = LinearPhaseResolution::Maximum;
    proc.refresh_global_aggregates(false);
    assert_eq!(proc.latency_samples(), 65_536);
    assert_eq!(proc.linear_phase_convolver.current_fir_len(), 131_072);
    assert_eq!(
        proc.linear_phase_convolver.current_latency_samples(),
        65_536
    );

    // 3. Verify Dual-IR Crossfading on Tier 2 (High = 16,384 taps, tau = 8,192 in P = 4096 tier)
    // during continuous sine playback with rapid parameter updates (no audio dropout!)
    let mut proc_lp = ProQ4Processor::new();
    proc_lp.prepare(48_000.0, 256);
    proc_lp.params.global.processing_mode = ProcessingMode::LinearPhase;
    proc_lp.params.global.processing_resolution = LinearPhaseResolution::High; // 16384 taps -> tau = 8192 (Tier 2)
    proc_lp.params.bands[0].used = true;
    proc_lp.params.bands[0].enabled = true;
    proc_lp.params.bands[0].shape = FilterShape::Bell;
    proc_lp.params.bands[0].freq_hz = 1000.0;
    proc_lp.params.bands[0].gain_db = 0.0;
    proc_lp.params.bands[0].q_ui = 1.0;
    proc_lp.refresh_band_dsp(0, false);
    proc_lp.refresh_global_aggregates(false);
    assert_eq!(proc_lp.latency_samples(), 8192);

    let ctx = ProcessContext::default();
    let mut block = [[0.0f32; 2]; 256];
    let mut sample_idx = 0usize;

    // Warm up the 8,192-sample Linear Phase pipeline (40 blocks = 10,240 samples)
    for _ in 0..40 {
        for frame in block.iter_mut() {
            let s = (2.0 * PI * 1000.0 * (sample_idx as f64) / 48_000.0).sin() as f32;
            *frame = [s, s];
            sample_idx += 1;
        }
        proc_lp.process_slice(&ctx, &mut block, None, &mut []);
    }

    // Trigger multiple parameter automation updates (+6 dB boost at 1 kHz) across blocks
    // and verify every 256-sample block maintains non-zero continuous RMS without dropping out!
    let mut min_block_rms = f64::INFINITY;
    for b in 0..24 {
        if b == 2 {
            // Set Band 1 Gain to +6 dB (norm = 0.6)
            proc_lp.set_parameter_normalized(3, 0.60);
        } else if b == 3 {
            // Rapid follow-up automation while first crossfade is still in flight
            proc_lp.set_parameter_normalized(3, 0.65);
        }
        for frame in block.iter_mut() {
            let s = (2.0 * PI * 1000.0 * (sample_idx as f64) / 48_000.0).sin() as f32;
            *frame = [s, s];
            sample_idx += 1;
        }
        proc_lp.process_slice(&ctx, &mut block, None, &mut []);
        let rms = (block.iter().map(|f| (f[0] as f64).powi(2)).sum::<f64>() / 256.0).sqrt();
        min_block_rms = min_block_rms.min(rms);
    }

    assert!(
        min_block_rms > 0.65,
        "LinearPhase Tier-2 Dual-IR crossfade suffered audio dropout: min_block_rms = {}",
        min_block_rms
    );
}

#[test]
fn test_spectral_dynamics_exact_1024_latency_unity_cola_and_brickwall_delay() {
    // 1. Verify SpectralDynamicsCore exact 1,024-sample impulse arrival and unity COLA gain (1.0x)
    let mut core = dsp_proq4_eq::SpectralDynamicsCore::new();
    core.set_sample_rate(48_000.0);
    assert_eq!(core.latency_samples(), 1024);

    let mut peak_idx = 0usize;
    let mut peak_val = 0.0f32;
    for n in 0..2500 {
        let x = if n == 100 { 1.0 } else { 0.0 };
        let y = core.process_stereo_sample(x, x)[0];
        if y.abs() > peak_val {
            peak_val = y.abs();
            peak_idx = n;
        }
    }
    assert_eq!(
        peak_idx,
        100 + 1024,
        "SpectralDynamicsCore impulse arrived at {} instead of {}",
        peak_idx,
        100 + 1024
    );
    assert!(
        (peak_val - 1.0).abs() < 1e-4,
        "SpectralDynamicsCore COLA passband gain error: got {} (expected 1.0)",
        peak_val
    );

    // 2. Verify Brickwall extra latency (320 samples) matches actual audio delay in ZeroLatency mode
    let mut proc = ProQ4Processor::new();
    proc.prepare(48_000.0, 256);
    proc.params.bands[0].used = true;
    proc.params.bands[0].enabled = true;
    proc.params.bands[0].shape = FilterShape::HighCut;
    proc.params.bands[0].freq_hz = 20_000.0;
    proc.params.bands[0].is_brickwall = true;
    proc.refresh_band_dsp(0, false);
    proc.refresh_global_aggregates(false);
    assert_eq!(proc.latency_samples(), 320);

    let ctx = ProcessContext::default();
    let mut io = [[0.0f32; 2]; 512];
    io[10] = [1.0, 1.0];
    proc.process_slice(&ctx, &mut io, None, &mut []);

    let pre_330_max = io[..330]
        .iter()
        .map(|f| f[0].abs())
        .fold(0.0f32, f32::max);
    assert_eq!(
        pre_330_max, 0.0,
        "Brickwall band output arrived before 320-sample latency!"
    );
    assert!(
        io[330][0].abs() > 0.05 && io[331][0].abs() > 0.20,
        "Brickwall band output did not arrive at sample 10 + 320 = 330: got {}, {}",
        io[330][0],
        io[331][0]
    );
}

#[test]
fn test_gate1_character_waveshaper_4x_adaa_alias_rejection_unity_gain_and_harmonics() {
    let sr = 48_000.0;
    let f_test = 11_300.0;
    let amp_minus_1db = 10.0_f64.powf(-1.0 / 20.0); // -1.0 dBFS
    let warmup = 9_600usize;
    let fft_n = 24_000usize; // Exact 2.0 Hz bins -> 11,300 Hz is exact integer bin 5,650

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(fft_n);

    // 4-term Blackman-Harris window (-92 dB sidelobes) normalized to unity coherent gain
    let mut win = vec![0.0f64; fft_n];
    let mut win_sum = 0.0f64;
    for (n, w) in win.iter_mut().enumerate() {
        let p = 2.0 * PI * (n as f64) / (fft_n as f64);
        let val = 0.35875 - 0.48829 * p.cos() + 0.14128 * (2.0 * p).cos() - 0.01168 * (3.0 * p).cos();
        *w = val;
        win_sum += val;
    }

    for &mode in &[CharacterMode::Subtle, CharacterMode::Warm] {
        let mut shaper = CharacterWaveshaper::new(sr);
        let mut buf_oversampled = vec![Complex64::new(0.0, 0.0); fft_n];
        let mut buf_naive = vec![Complex64::new(0.0, 0.0); fft_n];

        for n in 0..(warmup + fft_n) {
            let phase = 2.0 * PI * f_test * (n as f64) / sr;
            let x = amp_minus_1db * phase.sin();
            let y_os = shaper.process_sample(mode, 0, x);
            let y_naive = CharacterWaveshaper::eval_transfer_curve(mode, x);
            if n >= warmup {
                let idx = n - warmup;
                let w = win[idx];
                buf_oversampled[idx] = Complex64::new(y_os * w, 0.0);
                buf_naive[idx] = Complex64::new(y_naive * w, 0.0);
            }
        }

        fft.process(&mut buf_oversampled);
        fft.process(&mut buf_naive);

        let scale = 2.0 / win_sum;
        // All bins in 60 Hz .. 10,000 Hz are strictly folded alias spurs for a 11,300 Hz fundamental
        let bin_start = (60.0 / 2.0) as usize;
        let bin_end = (10_000.0 / 2.0) as usize;

        let mut peak_os_lin = 0.0f64;
        let mut peak_naive_lin = 0.0f64;
        let mut pow_os = 0.0f64;
        let mut pow_naive = 0.0f64;

        for k in bin_start..bin_end {
            let m_os = buf_oversampled[k].norm() * scale;
            let m_na = buf_naive[k].norm() * scale;
            peak_os_lin = peak_os_lin.max(m_os);
            peak_naive_lin = peak_naive_lin.max(m_na);
            pow_os += m_os * m_os;
            pow_naive += m_na * m_na;
        }

        let peak_os_db = 20.0 * peak_os_lin.max(1e-18).log10();
        let peak_naive_db = 20.0 * peak_naive_lin.max(1e-18).log10();
        let peak_reduction_db = peak_naive_db - peak_os_db;
        let power_reduction_db = 10.0 * (pow_naive.max(1e-30) / pow_os.max(1e-30)).log10();

        println!(
            "[Gate 1] {:?} @ 11.3 kHz -1 dBFS: naive_peak={:.2} dBFS, 4x_ADAA_peak={:.2} dBFS, peak_spur_reduction={:.2} dB, integrated_alias_reduction={:.2} dB",
            mode, peak_naive_db, peak_os_db, peak_reduction_db, power_reduction_db
        );

        assert!(
            peak_reduction_db >= 30.0 && power_reduction_db >= 30.0,
            "{:?} failed >= 30 dB folded alias spur reduction: peak_red={:.2} dB, pow_red={:.2} dB",
            mode,
            peak_reduction_db,
            power_reduction_db
        );

        // Verify small-signal (-40 dBFS = 0.01) unity gain (< 0.02 dB) at 1 kHz and 11.3 kHz
        for &f_small in &[1_000.0, 11_300.0] {
            let mut shaper_sm = CharacterWaveshaper::new(sr);
            let amp_sm = 0.01_f64;
            let mut in_sq = 0.0f64;
            let mut out_sq = 0.0f64;
            for n in 0..(4_800 + 9_600) {
                let x = amp_sm * (2.0 * PI * f_small * (n as f64) / sr).sin();
                let y = shaper_sm.process_sample(mode, 0, x);
                if n >= 4_800 {
                    in_sq += x * x;
                    out_sq += y * y;
                }
            }
            let gain_err_db = (10.0 * (out_sq / in_sq).log10()).abs();
            println!(
                "[Gate 1] {:?} small-signal (-40 dBFS @ {:.0} Hz) gain error = {:.6} dB",
                mode, f_small, gain_err_db
            );
            assert!(
                gain_err_db < 0.02,
                "{:?} small-signal gain error {:.5} dB >= 0.02 dB at {} Hz",
                mode,
                gain_err_db,
                f_small
            );
        }
    }

    // Verify authentic 2nd-order even harmonic generation + DC blocking in Warm mode (1 kHz @ -1 dBFS)
    let mut warm_shaper = CharacterWaveshaper::new(sr);
    let mut h1_re = 0.0f64;
    let mut h1_im = 0.0f64;
    let mut h2_re = 0.0f64;
    let mut h2_im = 0.0f64;
    let mut dc_acc = 0.0f64;
    let meas_len = 24_000usize;
    for n in 0..(24_000 + meas_len) {
        let x = amp_minus_1db * (2.0 * PI * 1_000.0 * (n as f64) / sr).sin();
        let y = warm_shaper.process_sample(CharacterMode::Warm, 0, x);
        if n >= 24_000 {
            let t = (n - 24_000) as f64;
            let p1 = 2.0 * PI * 1_000.0 * t / sr;
            let p2 = 2.0 * PI * 2_000.0 * t / sr;
            h1_re += y * p1.cos();
            h1_im -= y * p1.sin();
            h2_re += y * p2.cos();
            h2_im -= y * p2.sin();
            dc_acc += y;
        }
    }
    let h1_amp = 2.0 * (h1_re * h1_re + h1_im * h1_im).sqrt() / (meas_len as f64);
    let h2_amp = 2.0 * (h2_re * h2_re + h2_im * h2_im).sqrt() / (meas_len as f64);
    let h2_dbfs = 20.0 * h2_amp.log10();
    let dc_mean = (dc_acc / (meas_len as f64)).abs();
    println!(
        "[Gate 1] Warm 1 kHz @ -1 dBFS: H1={:.2} dBFS, H2 (even)={:.2} dBFS, residual_DC={:.2e}",
        20.0 * h1_amp.log10(),
        h2_dbfs,
        dc_mean
    );
    assert!(
        h2_dbfs > -25.0,
        "Warm mode missing strong 2nd-order even harmonic: H2 = {:.2} dBFS",
        h2_dbfs
    );
    assert!(
        dc_mean < 1e-3,
        "Warm mode 5 Hz f64 DC blocker failed to suppress even-harmonic DC offset: {}",
        dc_mean
    );
}

#[test]
fn test_gate2_natural_phase_jacobi_elliptic_oversampler_and_analog_matching() {
    let sr = 48_000.0;
    let ctx = ProcessContext {
        sample_rate: sr,
        ..ProcessContext::default()
    };

    // 1. Verify Jacobi-Elliptic Half-Band Oversampler stopband attenuation > 105 dB
    let mut os = NaturalPhaseOversampler2x::new();
    os.set_sample_rate(sr);
    let omega_s_start = PI - os.spec().omega_p; // 0.56 * pi (26.88 kHz in 96 kHz 2x domain)
    let mut worst_stopband_atten_db = f64::INFINITY;
    for i in 0..=2000 {
        let w = omega_s_start + (PI - omega_s_start) * (i as f64) / 2000.0;
        let mag = os.eval_one_way_response(w).norm().max(1e-20);
        let atten_db = -20.0 * mag.log10();
        worst_stopband_atten_db = worst_stopband_atten_db.min(atten_db);
    }

    // Also verify time-domain stopband rejection by injecting a 34,000 Hz sine (in 96 kHz 2x stopband)
    // into `downsample_step` and measuring the folded 14,000 Hz alias in the 48 kHz 1x output!
    os.reset();
    let mut alias_sq = 0.0f64;
    let td_warmup = 2_000usize;
    let td_len = 8_000usize;
    for n in 0..(td_warmup + td_len) {
        let p0 = 2.0 * PI * 34_000.0 * ((2 * n) as f64) / (2.0 * sr);
        let p1 = 2.0 * PI * 34_000.0 * ((2 * n + 1) as f64) / (2.0 * sr);
        let y = os.downsample_step(0, p0.sin(), p1.sin());
        if n >= td_warmup {
            alias_sq += y * y;
        }
    }
    let td_alias_rms = (alias_sq / (td_len as f64)).sqrt();
    let td_stopband_atten_db = -20.0 * (td_alias_rms * 2.0_f64.sqrt()).max(1e-20).log10();
    println!(
        "[Gate 2] Oversampler stopband attenuation: analytical_min={:.2} dB, time_domain_34kHz={:.2} dB",
        worst_stopband_atten_db, td_stopband_atten_db
    );
    assert!(
        worst_stopband_atten_db > 105.0 && td_stopband_atten_db > 105.0,
        "Oversampler stopband attenuation <= 105 dB: analytical={:.2} dB, td={:.2} dB",
        worst_stopband_atten_db,
        td_stopband_atten_db
    );

    // Helper to measure steady-state sine magnitude through full ProQ4Processor::process_slice
    let measure_sine_mag_db = |proc: &mut ProQ4Processor, freq_hz: f64| -> f64 {
        proc.reset_state();
        let warmup_frames = 2_400usize;
        let meas_frames = 4_800usize; // Exact 10 Hz bin resolution at 48 kHz
        let total = warmup_frames + meas_frames;
        let mut block = [[0.0f32; 2]; 256];
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        let mut win_sum = 0.0f64;
        let mut idx = 0usize;
        while idx < total {
            let chunk = (total - idx).min(256);
            for i in 0..chunk {
                let n = idx + i;
                let s = (0.5 * (2.0 * PI * freq_hz * (n as f64) / sr).sin()) as f32;
                block[i] = [s, s];
            }
            proc.process_slice(&ctx, &mut block[..chunk], None, &mut []);
            for i in 0..chunk {
                let n = idx + i;
                if n >= warmup_frames {
                    let k = (n - warmup_frames) as f64;
                    let w = 0.5 * (1.0 - (2.0 * PI * k / (meas_frames as f64)).cos());
                    let y = block[i][0] as f64;
                    let ph = 2.0 * PI * freq_hz * (n as f64) / sr;
                    re += y * w * ph.cos();
                    im -= y * w * ph.sin();
                    win_sum += w;
                }
            }
            idx += chunk;
        }
        let out_amp = 2.0 * (re * re + im * im).sqrt() / win_sum;
        20.0 * (out_amp / 0.5).max(1e-15).log10()
    };

    // 2. Flat NaturalPhase passband ripple across 20 Hz .. 20,000 Hz < 0.002 dB
    let mut proc_flat = ProQ4Processor::new();
    proc_flat.prepare(sr, 256);
    proc_flat.params.global.processing_mode = ProcessingMode::NaturalPhase;
    proc_flat.params.bands[0].used = true;
    proc_flat.params.bands[0].enabled = true;
    proc_flat.params.bands[0].shape = FilterShape::Bell;
    proc_flat.params.bands[0].freq_hz = 10_000.0;
    proc_flat.params.bands[0].gain_db = 0.0;
    proc_flat.params.bands[0].q_ui = 1.0;
    proc_flat.refresh_band_dsp(0, false);
    proc_flat.refresh_global_aggregates(false);

    let sweep_freqs: [f64; 22] = [
        20.0, 35.0, 60.0, 100.0, 200.0, 400.0, 800.0, 1_200.0, 2_000.0, 3_500.0, 5_000.0,
        7_000.0, 9_000.0, 10_000.0, 12_000.0, 14_000.0, 15_500.0, 17_000.0, 18_000.0, 19_000.0,
        19_500.0, 20_000.0,
    ];
    let mut max_flat_ripple_db = 0.0f64;
    for &f in &sweep_freqs {
        let db = measure_sine_mag_db(&mut proc_flat, f).abs();
        max_flat_ripple_db = max_flat_ripple_db.max(db);
    }
    println!(
        "[Gate 2] Flat NaturalPhase passband max ripple (20 Hz..20 kHz, process_slice) = {:.6} dB",
        max_flat_ripple_db
    );
    assert!(
        max_flat_ripple_db < 0.002,
        "Flat NaturalPhase passband ripple {:.6} dB >= 0.002 dB",
        max_flat_ripple_db
    );

    // 3. Analog prototype magnitude matching (< 0.05 dB across 20 Hz .. 20,000 Hz)
    // Test Bell (12 kHz, +9 dB, Q=2.0), HighShelf (10 kHz, +6 dB, Q=0.707), and all remaining shapes!
    let shape_cases: &[(FilterShape, f64, f64, f64, f64, &str)] = &[
        (FilterShape::Bell, 12_000.0, 9.0, 2.0, 12.0, "Bell 12kHz +9dB Q=2.0"),
        (FilterShape::HighShelf, 10_000.0, 6.0, 0.707, 12.0, "HighShelf 10kHz +6dB Q=0.707"),
        (FilterShape::LowShelf, 250.0, 6.0, 0.707, 12.0, "LowShelf 250Hz +6dB Q=0.707"),
        (FilterShape::LowCut, 120.0, 0.0, 0.707, 12.0, "LowCut 120Hz 12dB/oct"),
        (FilterShape::HighCut, 9_000.0, 0.0, 0.707, 24.0, "HighCut 9kHz 24dB/oct"),
        (FilterShape::Notch, 5_000.0, 0.0, 2.0, 12.0, "Notch 5kHz Q=2.0"),
        (FilterShape::BandPass, 3_000.0, 0.0, 1.5, 12.0, "BandPass 3kHz Q=1.5"),
        (FilterShape::TiltShelf, 1_500.0, 6.0, 0.707, 12.0, "TiltShelf 1.5kHz +6dB Q=0.707"),
        (FilterShape::AllPass, 6_000.0, 0.0, 1.2, 12.0, "AllPass 6kHz Q=1.2"),
    ];

    for &(shape, fc, gain, q, slope, label) in shape_cases {
        let mut proc = ProQ4Processor::new();
        proc.prepare(sr, 256);
        proc.params.global.processing_mode = ProcessingMode::NaturalPhase;
        proc.params.global.gain_q_interaction = false;
        proc.params.bands[0].used = true;
        proc.params.bands[0].enabled = true;
        proc.params.bands[0].shape = shape;
        proc.params.bands[0].freq_hz = fc as f32;
        proc.params.bands[0].gain_db = gain as f32;
        proc.params.bands[0].q_ui = q as f32;
        proc.params.bands[0].slope_db_oct = slope as f32;
        proc.refresh_band_dsp(0, false);
        proc.refresh_global_aggregates(false);

        // Dense 300-point logarithmic check (20 Hz .. 20,000 Hz) on configured NaturalPhase 2x band
        let mut max_static_err_db = 0.0f64;
        for i in 0..=300 {
            let t = (i as f64) / 300.0;
            let f = 20.0 * (1_000.0_f64).powf(t);
            let target_db = analog_prototype_mag_db(shape, fc, gain, q, slope, f, false);
            // Skip exact -inf Notch singularity below -50 dB or extreme stopband below -60 dB
            if target_db < -50.0 {
                continue;
            }
            let actual_c = proc.natural_phase_bank.bands_2x[0].eval_complex_response(f, 2.0 * sr);
            let actual_db = 20.0 * actual_c.norm().max(1e-15).log10();
            let err = (actual_db - target_db).abs();
            max_static_err_db = max_static_err_db.max(err);
        }

        // Also verify full time-domain `process_slice` (including oversampler + Cytomic SVF + downsampler)
        // on the swept sine grid for Bell 12kHz and HighShelf 10kHz!
        let mut max_td_err_db = 0.0f64;
        if matches!(shape, FilterShape::Bell | FilterShape::HighShelf) {
            for &f in &sweep_freqs {
                let target_db = analog_prototype_mag_db(shape, fc, gain, q, slope, f, false);
                let td_db = measure_sine_mag_db(&mut proc, f);
                max_td_err_db = max_td_err_db.max((td_db - target_db).abs());
            }
        }

        // For AllPass, also verify complex phase match against continuous-time Laplace prototype!
        if shape == FilterShape::AllPass {
            let h_an = eval_analog_s_plane_response(shape, fc, gain, q, slope, fc, false);
            let h_np = proc.natural_phase_bank.bands_2x[0].eval_complex_response(fc, 2.0 * sr);
            let phase_diff_rad = (h_np * h_an.conj()).arg().abs();
            assert!(
                phase_diff_rad < 1e-3,
                "AllPass NaturalPhase center phase mismatch: {} rad",
                phase_diff_rad
            );
        }

        println!(
            "[Gate 2] {}: max_2x_matched_err = {:.5} dB, max_process_slice_td_err = {:.5} dB",
            label, max_static_err_db, max_td_err_db
        );
        assert!(
            max_static_err_db < 0.05 && max_td_err_db < 0.05,
            "{} exceeded 0.05 dB analog matching tolerance: static={:.5} dB, td={:.5} dB",
            label,
            max_static_err_db,
            max_td_err_db
        );
    }
}



/// Linear-phase IR redesign triggered from the real-time thread must be block-amortised: no single
/// `set_parameter_normalized` / `process_slice` call may carry the full 131,072-tap design, and once
/// the design completes the output must equal a processor whose IR was designed synchronously.
#[test]
fn test_linear_phase_ir_redesign_is_amortised_off_the_rt_critical_path() {
    let setup = |gain_db: f32| {
        let mut p = ProQ4Processor::new();
        p.prepare(48_000.0, 256);
        p.params.global.processing_mode = ProcessingMode::LinearPhase;
        p.params.global.processing_resolution = LinearPhaseResolution::Maximum;
        for b in 0..8 {
            p.params.bands[b].used = true;
            p.params.bands[b].enabled = true;
            p.params.bands[b].shape = FilterShape::Bell;
            p.params.bands[b].freq_hz = 60.0 * (2.0f32).powi(b as i32);
            p.params.bands[b].gain_db = if b == 3 { gain_db } else { 2.0 };
            p.refresh_band_dsp(b, false);
        }
        p.refresh_global_aggregates(false);
        p
    };
    let ctx = ProcessContext {
        sample_rate: 48_000.0,
        current_sample_pos: 0,
        ppq_tick: 0,
        bpm: 120.0,
        slice_frames: 256,
        is_playing: true,
        is_offline_rendering: false,
        ppq_resolution: daw_core_graph::DEFAULT_PPQ,
    };
    let mut live = setup(-6.0);
    let mut reference = setup(9.0);

    // Retune band 3 on the "RT thread": the call itself must not run the full design.
    let mut bp = live.params.bands[3];
    bp.gain_db = 9.0;
    let t0 = std::time::Instant::now();
    live.params.bands[3] = bp;
    live.refresh_band_dsp(3, true);
    live.refresh_global_aggregates(true);
    let change_cost = t0.elapsed();

    let mut worst_block = std::time::Duration::ZERO;
    let mut buf = vec![[0.0f32; 2]; 256];
    for _ in 0..64 {
        buf.fill([0.0; 2]);
        let t = std::time::Instant::now();
        live.process_slice(&ctx, &mut buf, None, &mut []);
        worst_block = worst_block.max(t.elapsed());
    }
    // Reference cost in the same environment: one synchronous full 131,072-tap redesign.
    let mut sync_ref = setup(9.0);
    let t_sync = std::time::Instant::now();
    sync_ref.params.global.processing_mode = ProcessingMode::LinearPhase;
    sync_ref.refresh_global_aggregates(false);
    let sync_cost = t_sync.elapsed();
    assert!(
        change_cost < sync_cost / 4,
        "parameter change ran the IR design synchronously: {change_cost:?} (full design {sync_cost:?})"
    );
    assert!(
        worst_block < sync_cost / 2,
        "worst 256-sample block {worst_block:?} vs synchronous design {sync_cost:?}"
    );

    // After the design + crossfade, both processors must render identically (impulse response).
    let mut imp_live = vec![[0.0f32; 2]; 256];
    let mut imp_ref = vec![[0.0f32; 2]; 256];
    reference.reset_state();
    live.reset_state();
    let mut max_diff = 0.0f32;
    for blk in 0..600 {
        imp_live.fill([0.0; 2]);
        imp_ref.fill([0.0; 2]);
        if blk == 0 {
            imp_live[0] = [1.0, 1.0];
            imp_ref[0] = [1.0, 1.0];
        }
        live.process_slice(&ctx, &mut imp_live, None, &mut []);
        reference.process_slice(&ctx, &mut imp_ref, None, &mut []);
        for i in 0..256 {
            max_diff = max_diff.max((imp_live[i][0] - imp_ref[i][0]).abs());
        }
    }
    assert!(max_diff < 1e-5, "amortised IR differs from synchronous IR by {max_diff}");
}
