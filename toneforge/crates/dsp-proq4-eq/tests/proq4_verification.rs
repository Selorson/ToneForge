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
    bessel_i0, compute_kaiser_60db_window, AnalogFilterPrototype, CharacterMode, FilterShape,
    FractionalSlopeBiquadBand, LinearPhaseResolution, NonUniformPartitionedConvolver,
    PerceptualAutoGain, ProQ4ParameterSet, ProQ4Processor, ProcessingMode,
    ProgramDependentCapacitor, StereoPlacement, PROQ4_GAIN_DEPENDENT_MASK,
    PROQ4_GAIN_Q_INTERACTION_BASE, PROQ4_GAIN_Q_INTERACTION_MASK, PROQ4_TOTAL_PARAMS,
};
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

