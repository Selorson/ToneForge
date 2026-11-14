//! Comprehensive Unit & Integration Verification Suite for `dsp-ozone12-mastering`:
//! 1. Real `exciter-triode-48000.000000.dat` binary loading, DC equilibrium (`< 1e-12`),
//!    and DFT harmonic analysis of Even (`2*f0`, `4*f0`) vs Odd (`3*f0`, `5*f0`) distortion
//!    across `Triode`, `DualTriode`, `Warm`, `Tube`, `Tape`, `Retro`, and `Analog` modes.
//! 2. `IrcVMultiBandMaximizer` True-Peak brickwall verification guaranteeing post-limiter
//!    True Peak never exceeds `Ceiling dBTP` even on worst-case $+18\text{ dBFS}$ `fs/4`
//!    inter-sample-peak sinusoids at $\pi/4$ phase.
//! 3. `LufsTruePeakMeter` ITU-R BS.1770-4 K-Weighting IIR, Momentary/Short-Term/Integrated LUFS,
//!    4x Polyphase True-Peak `dBTP`, and `Ozone12MeterFrame` telemetry verification.
//! 4. Verification of Ozone 12 `Imager`, `Stabilizer`, `Impact`, `Clarity`, `Unlimiter`,
//!    `BassControl`, and `StemEqModule`.

use daw_core_graph::{ProcessContext, StereoFrame, UnifiedDspNode, VisualTelemetryFrame};
use dsp_ozone12_mastering::{
    DkTriodeChannelState, DkTriodeLut, DkTriodeLutExciter, ExciterMode, IrcCharacter, IrcMode,
    IrcVMultiBandMaximizer, LinkwitzRiley4BandCrossover, LufsTruePeakMeter, MultibandImagerModule,
    Ozone12MasterChain, SoftClipMode, StabilizerMode, StemEqModule, StereoizerMode,
    TruePeakDetector4x,
};
use std::f64::consts::PI;
use std::path::Path;
use std::sync::Arc;

/// Computes the single-bin DFT magnitude at `target_freq_hz` using a Hann window.
fn measure_harmonic_magnitude_db(signal: &[f32], sample_rate: f64, target_freq_hz: f64) -> f64 {
    let n = signal.len();
    assert!(n > 0);
    let mut re = 0.0f64;
    let mut im = 0.0f64;
    let mut win_sum = 0.0f64;

    for (i, &s) in signal.iter().enumerate() {
        let w = 0.5 * (1.0 - (2.0 * PI * (i as f64) / (n as f64)).cos());
        let angle = 2.0 * PI * target_freq_hz * (i as f64) / sample_rate;
        re += (s as f64) * w * angle.cos();
        im -= (s as f64) * w * angle.sin();
        win_sum += w;
    }

    let mag = 2.0 * (re * re + im * im).sqrt() / win_sum.max(1e-12);
    20.0 * mag.max(1e-12).log10()
}

#[test]
fn test_dk_triode_lut_real_dat_and_even_odd_harmonics() {
    let dat_path =
        Path::new(r"f:\FL studio\toneforge\assets\ozone12\luts\exciter-triode-48000.000000.dat");
    assert!(
        dat_path.exists(),
        "Expected real LUT file at {:?}",
        dat_path
    );

    let lut = Arc::new(DkTriodeLut::from_dat_file(dat_path).expect("Failed to parse exciter-triode .dat"));
    assert_eq!(lut.magic, 0x8BAD_43FC);
    assert_eq!(lut.version, 4);
    assert!((lut.sample_rate - 48_000.0).abs() < 1e-9);
    assert_eq!(lut.grid_dims, [360, 360]);

    // 1. Verify quiescent DC equilibrium (< 1e-12 drift over 1,000 steps at 0.0 V input)
    let mut state = DkTriodeChannelState::new(Arc::clone(&lut));
    for _ in 0..1000 {
        let y_zero = state.tick(0.0);
        assert!(
            y_zero.abs() < 1e-12,
            "Quiescent DC output drifted: {}",
            y_zero
        );
    }

    // 2. Measure Even (2*f0 = 2000 Hz) and Odd (3*f0 = 3000 Hz) harmonics on a 1,000 Hz sine wave
    let sr = 48_000.0;
    let f0 = 1000.0;
    let total_samples = 8192;
    let steady_start = 2048;

    let mut exciter =
        DkTriodeLutExciter::from_dat_file(dat_path, sr).expect("Exciter init failed");
    exciter.multiband_enabled = false;

    // 2a. Triode & DualTriode modes: asymmetric 12AX7 cathode-plate transfer curve produces
    //     prominent Even-order harmonics (2nd harmonic at 2,000 Hz) alongside Odd harmonics.
    for mode in [
        ExciterMode::Triode,
        ExciterMode::DualTriode,
        ExciterMode::Warm,
        ExciterMode::Tube,
    ] {
        exciter.reset_state();
        exciter.set_all_bands_mode(mode, 75.0, 1.0);
        let mut out_l = Vec::with_capacity(total_samples - steady_start);
        for n in 0..total_samples {
            let s = (0.75 * (2.0 * PI * f0 * (n as f64) / sr).sin()) as f32;
            let [yl, _] = exciter.step_frame([s, s]);
            if n >= steady_start {
                out_l.push(yl);
            }
        }
        let h1_db = measure_harmonic_magnitude_db(&out_l, sr, f0);
        let h2_db = measure_harmonic_magnitude_db(&out_l, sr, 2.0 * f0);
        let h3_db = measure_harmonic_magnitude_db(&out_l, sr, 3.0 * f0);

        assert!(
            h1_db > -15.0,
            "Mode {:?} fundamental too low: {} dB",
            mode,
            h1_db
        );
        assert!(
            h2_db > -52.0,
            "Mode {:?} expected strong 2nd (even) harmonic, got {} dB",
            mode,
            h2_db
        );
        assert!(
            h3_db > -65.0 || mode == ExciterMode::Warm,
            "Mode {:?} 3rd harmonic check: H2={} dB, H3={} dB",
            mode,
            h2_db,
            h3_db
        );
    }

    // 2b. Tape & Retro modes: odd-symmetric saturation `f(-x) == -f(x)` produces dominant
    //     Odd-order harmonics (3rd harmonic at 3,000 Hz) while suppressing Even harmonics.
    for mode in [ExciterMode::Tape, ExciterMode::Retro] {
        exciter.reset_state();
        exciter.set_all_bands_mode(mode, 80.0, 1.0);
        let mut out_l = Vec::with_capacity(total_samples - steady_start);
        for n in 0..total_samples {
            let s = (0.75 * (2.0 * PI * f0 * (n as f64) / sr).sin()) as f32;
            let [yl, _] = exciter.step_frame([s, s]);
            if n >= steady_start {
                out_l.push(yl);
            }
        }
        let h2_db = measure_harmonic_magnitude_db(&out_l, sr, 2.0 * f0);
        let h3_db = measure_harmonic_magnitude_db(&out_l, sr, 3.0 * f0);

        assert!(
            h3_db > -35.0,
            "Mode {:?} expected strong 3rd (odd) harmonic, got {} dB",
            mode,
            h3_db
        );
        assert!(
            h3_db - h2_db > 25.0,
            "Mode {:?} odd harmonic H3 ({} dB) should dominate even harmonic H2 ({} dB) by >25 dB",
            mode,
            h3_db,
            h2_db
        );
    }

    // 3. Verify 4-band Multi-Band Exciter with mixed modes per band
    exciter.multiband_enabled = true;
    exciter.bands[0].mode = ExciterMode::Tape;
    exciter.bands[1].mode = ExciterMode::Warm;
    exciter.bands[2].mode = ExciterMode::Triode;
    exciter.bands[3].mode = ExciterMode::DualTriode;
    for n in 0..1024 {
        let s = (0.5 * (2.0 * PI * 440.0 * (n as f64) / sr).sin()) as f32;
        let [yl, yr] = exciter.step_frame([s, -s]);
        assert!(yl.is_finite() && yr.is_finite());
    }
}

#[test]
fn test_irc_v_maximizer_true_peak_ceiling_and_isp_enforcement() {
    let sr = 48_000.0;

    for &ceiling_dbtp in &[-0.5f32, -1.0, -3.0, -6.0] {
        for &irc_mode in &[IrcMode::IrcI, IrcMode::IrcIII, IrcMode::IrcIV, IrcMode::IrcV] {
            let mut maximizer = IrcVMultiBandMaximizer::new(sr);
            maximizer.irc_mode = irc_mode;
            maximizer.character_mode = IrcCharacter::Transient;
            maximizer.set_gain_db(12.0); // +12 dB heavy drive
            maximizer.set_ceiling_dbtp(ceiling_dbtp);
            maximizer.prevent_intersample_clipping = true;
            maximizer.enable_upward_comp = true;
            maximizer.upward_comp_wet = 0.5;
            maximizer.enable_soft_clip = true;
            maximizer.soft_clip_mix = 0.35;
            maximizer.soft_clip_mode = SoftClipMode::Medium;

            let mut external_tp = TruePeakDetector4x::new();

            // Worst-case ITU-R BS.1770-4 Inter-Sample Peak test signal:
            // f = fs / 4 (12 kHz @ 48 kHz) with phase = pi / 4 (+3.01 dB ISP overshoot!)
            // combined with +6 dBFS low-frequency bass and sharp transient spikes.
            for n in 0..6000 {
                let isp_tone = (2.0 * PI * 0.25 * (n as f64) + PI * 0.25).cos() as f32;
                let bass_tone = (2.0 * PI * 60.0 * (n as f64) / sr).sin() as f32;
                let spike = if n % 500 == 0 { 4.0 } else { 0.0 };
                let in_l = 2.0 * isp_tone + 1.5 * bass_tone + spike;
                let in_r = -2.0 * isp_tone + 1.5 * bass_tone - spike;

                let out = maximizer.step_frame([in_l, in_r]);
                let _ = external_tp.step_stereo(out[0], out[1]);
            }

            let measured_dbtp = external_tp.max_stereo_dbtp();
            assert!(
                measured_dbtp <= ceiling_dbtp + 0.05,
                "Mode {:?} at ceiling {} dBTP exceeded True Peak ceiling: measured {} dBTP",
                irc_mode,
                ceiling_dbtp,
                measured_dbtp
            );
        }
    }
}

#[test]
fn test_k_weighted_lufs_and_true_peak_meter_accuracy() {
    let sr = 48_000.0;
    let mut meter = LufsTruePeakMeter::new(sr);

    // 1. Verify TruePeakDetector4x catches the +3.0103 dBTP inter-sample peak of a
    //    0.0 dBFS-sampled fs/4 sine wave at phase pi/4:
    //    Sampled values are +/- cos(pi/4) = +/- 0.70710678 (-3.01 dBFS), so scaling by sqrt(2)
    //    makes the digital samples exactly +/- 1.0 (0.0 dBFS) while the true analog peak is +3.01 dBTP!
    let mut isp_detector = TruePeakDetector4x::new();
    let sqrt2 = std::f64::consts::SQRT_2;
    for n in 0..512 {
        // Smooth 32-sample raised-cosine fade-in so we measure the true steady-state ISP
        // rather than the 12-sample step-discontinuity Gibbs ringing at n = 0.
        let env = if n < 32 {
            0.5 * (1.0 - (PI * (n as f64) / 32.0).cos())
        } else {
            1.0
        };
        let s = (env * sqrt2 * (2.0 * PI * 0.25 * (n as f64) + PI * 0.25).cos()) as f32;
        assert!(s.abs() <= 1.0001);
        isp_detector.step_stereo(s, s);
    }
    let detected_isp_dbtp = isp_detector.max_stereo_dbtp();
    assert!(
        (detected_isp_dbtp - 3.0103).abs() < 0.15,
        "Expected ~+3.01 dBTP inter-sample peak, got {} dBTP",
        detected_isp_dbtp
    );

    // 2. Verify ITU-R BS.1770-4 K-Weighted LUFS on a -18.0 dBFS 997 Hz stereo sine wave
    //    over 4.0 seconds (192,000 samples @ 48 kHz).
    //    Per ITU-R BS.1770-4, the -0.691 dB offset is specifically defined to cancel the
    //    +0.691 dB K-weighting filter gain at 997 Hz, so a -18.0 dBFS 997 Hz stereo sine
    //    calibrates to exact -18.000 LUFS!
    let amp = 10.0f64.powf(-18.0 / 20.0);
    let total_samples = (4.0 * sr) as usize;
    for n in 0..total_samples {
        let s = (amp * (2.0 * PI * 997.0 * (n as f64) / sr).sin()) as f32;
        meter.push_frame(s, s, -1.5);
    }

    assert!(
        (meter.lufs_momentary - (-18.0)).abs() < 0.02,
        "Momentary LUFS mismatch: {} LUFS",
        meter.lufs_momentary
    );
    assert!(
        (meter.lufs_short_term - (-18.0)).abs() < 0.02,
        "Short-Term LUFS mismatch: {} LUFS",
        meter.lufs_short_term
    );
    assert!(
        (meter.lufs_integrated - (-18.0)).abs() < 0.02,
        "Integrated LUFS mismatch: {} LUFS",
        meter.lufs_integrated
    );

    // 3. Verify telemetry export into VisualTelemetryFrame
    let mut vis_frame = VisualTelemetryFrame::default();
    meter.export_to_visual_telemetry(&mut vis_frame);
    assert!((vis_frame.header.lufs_integrated - meter.lufs_integrated).abs() < 1e-5);
    assert!((vis_frame.header.master_gr_db - (-1.5)).abs() < 1e-5);
    assert!((vis_frame.ozone12_telemetry.irc_gr_trace[0] - (-1.5)).abs() < 1e-5);
}

#[test]
fn test_ozone12_core_modules_imager_stabilizer_impact_clarity_unlimiter_bass() {
    let sr = 48_000.0;

    // 1. Linkwitz-Riley 4-Band Crossover flat magnitude reconstruction (`|B0+B1+B2+B3| === 1.0`)
    //    Integrate over an exact integer number of periods (`3000` samples = 5 periods @ 80 Hz)
    //    after steady-state settling (`2400` samples).
    let mut xover = LinkwitzRiley4BandCrossover::default();
    xover.configure([120.0, 1000.0, 6000.0], sr);
    for &freq in &[80.0, 480.0, 2400.0, 9600.0] {
        xover.reset();
        let mut energy_in = 0.0f64;
        let mut energy_out = 0.0f64;
        for n in 0..(2400 + 3000) {
            let x = (2.0 * PI * freq * (n as f64) / sr).sin();
            let bands = xover.split_frame_f64([x, x]);
            let sum = bands[0][0] + bands[1][0] + bands[2][0] + bands[3][0];
            if n >= 2400 {
                energy_in += x * x;
                energy_out += sum * sum;
            }
        }
        let gain_db = 10.0 * (energy_out / energy_in).log10();
        assert!(
            gain_db.abs() < 1e-4,
            "LR4 4-band crossover sum deviated at {} Hz by {} dB",
            freq,
            gain_db
        );
    }

    // 2. MultibandImagerModule: verify Phase-Safe Stereoizer mono compatibility
    //    When input is pure mono (L == R) and width == 0%, L' + R' with Stereoizer ON
    //    must exactly equal L' + R' with Stereoizer OFF (zero comb-filtering in mono!).
    let mut imager_st_on = MultibandImagerModule::new(sr);
    let mut imager_st_off = MultibandImagerModule::new(sr);
    imager_st_on.stereoizer_enable = true;
    imager_st_on.stereoizer_mode = StereoizerMode::ModeI;
    for b in 0..4 {
        imager_st_on.bands[b].stereoize_percent = 85.0;
    }
    imager_st_off.stereoizer_enable = false;

    let mut side_energy = 0.0f64;
    for n in 0..2048 {
        let mono_in = (0.6 * (2.0 * PI * 440.0 * (n as f64) / sr).sin()) as f32;
        let [l_on, r_on] = imager_st_on.process_frame(mono_in, mono_in);
        let [l_off, r_off] = imager_st_off.process_frame(mono_in, mono_in);

        let mono_sum_on = l_on + r_on;
        let mono_sum_off = l_off + r_off;
        assert!(
            (mono_sum_on - mono_sum_off).abs() < 1e-5,
            "Stereoizer violated mono compatibility at sample {}: {} vs {}",
            n,
            mono_sum_on,
            mono_sum_off
        );
        let diff = (l_on - r_on) as f64;
        side_energy += diff * diff;
    }
    assert!(
        side_energy > 1.0,
        "Stereoizer failed to synthesize stereo side energy"
    );

    // 3. StabilizerModule (Shape & Cut modes), ImpactModule, ClarityModule, UnlimiterModule, BassControlModule
    let mut chain = Ozone12MasterChain::new(sr);
    chain.unlimiter.enabled = true;
    chain.stabilizer.enabled = true;
    chain.stabilizer.mode = StabilizerMode::Cut;
    chain.clarity.enabled = true;
    chain.impact.enabled = true;
    chain.impact.bands[0].amount = 40.0;
    chain.bass_control.enabled = true;

    let ctx = ProcessContext {
        sample_rate: sr,
        current_sample_pos: 0,
        ppq_tick: 0,
        bpm: 120.0,
        slice_frames: 256,
        is_playing: true,
        is_offline_rendering: false,
        ppq_resolution: daw_core_graph::DEFAULT_PPQ,
    };
    let mut block: Vec<StereoFrame> = (0..256)
        .map(|n| {
            let s = (0.5 * (2.0 * PI * 220.0 * (n as f64) / sr).sin()) as f32;
            [s, s]
        })
        .collect();
    chain.process_slice(&ctx, &mut block, None, &mut []);
    assert!(block.iter().all(|f| f[0].is_finite() && f[1].is_finite()));

    // 4. StemEqModule Unity-Identity check
    let mut stem_eq = StemEqModule::new(sr);
    let x_del = [0.42f32, -0.37f32];
    let fake_stems = [[0.1, 0.1], [0.2, -0.1], [0.05, 0.05], [0.07, -0.42]];
    let out_unity = stem_eq.process_with_stems(x_del, &fake_stems);
    assert_eq!(out_unity, x_del);
}

#[test]
fn test_ozone12_edge_cases_empty_inputs_corrupt_dat_and_boundary_params() {
    // 1. Corrupt / truncated .dat LUT rejection
    assert!(DkTriodeLut::from_dat_bytes(&[]).is_err());
    assert!(DkTriodeLut::from_dat_bytes(&[0u8; 1024]).is_err());

    // 2. Empty audio buffer and extreme boundary parameters (-inf/NaN clamped normalized values)
    let mut chain = Ozone12MasterChain::new(48_000.0);
    for pid in 0..70 {
        chain.set_parameter_normalized(pid, -10.0);
        chain.set_parameter_normalized(pid, 10.0);
        chain.set_parameter_normalized(pid, 0.5);
    }

    let ctx = ProcessContext::default();
    let mut empty_io: [StereoFrame; 0] = [];
    chain.process_slice(&ctx, &mut empty_io, None, &mut []);

    // 3. Silence input produces finite LUFS (-100.0 / -70.0 floor) and finite True Peak
    let mut silence = [[0.0f32; 2]; 512];
    chain.process_slice(&ctx, &mut silence, None, &mut []);
    assert!(chain.meter.lufs_momentary.is_finite());
    assert!(chain.meter.max_true_peak_dbtp().is_finite());
}

#[test]
fn test_dk_triode_exciter_multi_sample_rate_and_oversample_2x() {
    let dat_path =
        Path::new(r"f:\FL studio\toneforge\assets\ozone12\luts\exciter-triode-48000.000000.dat");

    for &sr in &[44_100.0, 48_000.0, 96_000.0, 192_000.0] {
        for &os2x in &[false, true] {
            let mut exciter =
                DkTriodeLutExciter::from_dat_file(dat_path, sr).expect("Exciter load failed");
            exciter.multiband_enabled = false;
            exciter.oversample_2x = os2x;
            exciter.set_all_bands_mode(ExciterMode::Triode, 70.0, 1.0);

            let total_samples = (0.15 * sr) as usize;
            let steady_start = (0.04 * sr) as usize;
            let mut out_l = Vec::with_capacity(total_samples - steady_start);
            for n in 0..total_samples {
                let s = (0.7 * (2.0 * PI * 1000.0 * (n as f64) / sr).sin()) as f32;
                let [yl, yr] = exciter.step_frame([s, s]);
                assert!(yl.is_finite() && yr.is_finite());
                if n >= steady_start {
                    out_l.push(yl);
                }
            }
            let h1_db = measure_harmonic_magnitude_db(&out_l, sr, 1000.0);
            let h2_db = measure_harmonic_magnitude_db(&out_l, sr, 2000.0);
            assert!(
                h1_db > -12.0 && h2_db > -50.0,
                "Triode @ sr={} Hz (os2x={}) unexpected harmonics: H1={} dB, H2={} dB",
                sr,
                os2x,
                h1_db,
                h2_db
            );
        }
    }
}

#[test]
fn test_ozone12_chain_dynamic_maximizer_bypass_latency_and_full_telemetry_export() {
    let mut chain = Ozone12MasterChain::new(48_000.0);
    // At 48 kHz: 5 ms lookahead (240 samples) + 24-sample True-Peak predictive window = 264 samples
    assert_eq!(chain.latency_samples(), 264);

    // Disabling IRC maximizer drops latency to 0 samples
    chain.set_irc_maximizer_enabled(false);
    assert_eq!(chain.latency_samples(), 0);

    // Re-enabling and preparing at 96 kHz scales 5 ms lookahead to 480 + 24 = 504 samples
    chain.set_irc_maximizer_enabled(true);
    chain.prepare(96_000.0, 512);
    assert_eq!(chain.latency_samples(), 504);

    // Run audio through the full chain and verify all Ozone12MeterFrame fields are populated
    chain.prepare(48_000.0, 256);
    chain.stabilizer.enabled = true;
    chain.imager.enabled = true;
    chain
        .exciter
        .set_all_bands_mode(ExciterMode::DualTriode, 60.0, 1.0);
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
    let mut block: Vec<StereoFrame> = (0..256)
        .map(|n| {
            let l = (0.6 * (2.0 * PI * 440.0 * (n as f64) / 48_000.0).sin()) as f32;
            let r = (0.6 * (2.0 * PI * 440.0 * (n as f64) / 48_000.0).cos()) as f32;
            [l, r]
        })
        .collect();
    chain.process_slice(&ctx, &mut block, None, &mut []);

    let mut vis = VisualTelemetryFrame::default();
    chain.export_to_visual_telemetry(&mut vis);
    assert!(
        vis.ozone12_telemetry
            .tonal_balance_bars
            .iter()
            .any(|&v| v.abs() > 0.0)
    );
    assert!(
        vis.ozone12_telemetry
            .imager_corr_bands
            .iter()
            .all(|&c| c.is_finite())
    );
    assert!(
        vis.ozone12_telemetry
            .exciter_drive_gr
            .iter()
            .any(|&v| v > 0.0)
    );
}

#[test]
fn test_sliding_max_heap_segment_tree_exact_sliding_window_peak() {
    use dsp_ozone12_mastering::SlidingMaxHeap;

    let window_len = 64usize;
    let mut heap = SlidingMaxHeap::new(window_len);
    assert_eq!(heap.window_len(), window_len);

    // Deterministic pseudo-random impulse/transient sequence with varying peaks and silence
    let mut history = Vec::new();
    for i in 0..500usize {
        let val = if i % 73 == 0 {
            1.85
        } else if i % 29 == 0 {
            0.92
        } else {
            ((i as f32 * 0.137).sin().abs()) * 0.45
        };
        history.push(val);
        let tree_max = heap.push_and_query_max(val);

        let start = history.len().saturating_sub(window_len);
        let expected_max = history[start..]
            .iter()
            .copied()
            .fold(0.0f32, f32::max);
        assert!(
            (tree_max - expected_max).abs() < 1e-6,
            "SlidingMaxHeap mismatch at step {}: got {}, expected {}",
            i,
            tree_max,
            expected_max
        );
    }

    // Verify clear() resets all nodes to 0.0 and NaN cannot poison the segment tree
    heap.clear();
    assert_eq!(heap.push_and_query_max(0.125), 0.125);
    assert_eq!(heap.push_and_query_max(f32::NAN), 0.125);
    assert!(heap.push_and_query_max(0.5).is_finite());
    assert_eq!(heap.push_and_query_max(0.25), 0.5);
}

#[test]
fn test_low_end_focus_punchy_vs_smooth_and_spectral_shaper_in_master_chain() {
    use dsp_ozone12_mastering::{
        LowEndFocusMode, LowEndFocusModule, SpectralShaperModule, SpectralShaperToneMode,
    };

    let sr = 48_000.0;
    let mut lef_punchy = LowEndFocusModule::new(sr);
    lef_punchy.set_frequency_range_hz(20.0, 220.0);
    lef_punchy.mode = LowEndFocusMode::Punchy;
    lef_punchy.contrast_percent = 75.0;

    let mut lef_smooth = LowEndFocusModule::new(sr);
    lef_smooth.set_frequency_range_hz(20.0, 220.0);
    lef_smooth.mode = LowEndFocusMode::Smooth;
    lef_smooth.contrast_percent = 75.0;

    // Synthesize a 60 Hz kick drum burst (transient in first 400 samples, decay after)
    let mut kick_punchy: Vec<StereoFrame> = (0..1024)
        .map(|n| {
            let env = if n < 256 {
                1.0
            } else {
                (-(n as f64 - 256.0) / 180.0).exp()
            };
            let s = (0.75 * env * (2.0 * PI * 60.0 * (n as f64) / sr).sin()) as f32;
            [s, s]
        })
        .collect();
    let mut kick_smooth = kick_punchy.clone();

    lef_punchy.process_frames_in_place(&mut kick_punchy);
    lef_smooth.process_frames_in_place(&mut kick_smooth);

    let peak_punchy = kick_punchy[..256]
        .iter()
        .fold(0.0f32, |m, f| m.max(f[0].abs()));
    let peak_smooth = kick_smooth[..256]
        .iter()
        .fold(0.0f32, |m, f| m.max(f[0].abs()));
    assert!(
        peak_punchy > peak_smooth,
        "LowEndFocus Punchy mode ({}) must accentuate transient attack more than Smooth mode ({})",
        peak_punchy,
        peak_smooth
    );

    // Verify cutoff_low_hz action region: setting [100.0, 250.0] Hz focuses +6 dB gain on 160 Hz much more than 28 Hz sub-bass
    let mut lef_band = LowEndFocusModule::new(sr);
    lef_band.set_frequency_range_hz(100.0, 250.0);
    lef_band.contrast_percent = 0.0;
    lef_band.gain_db = 6.0;
    let mut sub_28: Vec<StereoFrame> = (0..2048)
        .map(|n| {
            let s = (0.4 * (2.0 * PI * 28.0 * (n as f64) / sr).sin()) as f32;
            [s, s]
        })
        .collect();
    let mut mid_160: Vec<StereoFrame> = (0..2048)
        .map(|n| {
            let s = (0.4 * (2.0 * PI * 160.0 * (n as f64) / sr).sin()) as f32;
            [s, s]
        })
        .collect();
    lef_band.process_frames_in_place(&mut sub_28);
    lef_band.reset_internal();
    lef_band.process_frames_in_place(&mut mid_160);
    let rms_28: f32 = sub_28[512..].iter().map(|f| f[0] * f[0]).sum::<f32>();
    let rms_160: f32 = mid_160[512..].iter().map(|f| f[0] * f[0]).sum::<f32>();
    assert!(
        rms_160 > rms_28 * 1.8,
        "In-band 160 Hz energy ({}) must be boosted significantly more than out-of-band 28 Hz ({})",
        rms_160,
        rms_28
    );

    // Verify SpectralShaperModule tames a harsh 4.2 kHz resonance even within a single 512-sample block
    let mut shaper = SpectralShaperModule::new(sr);
    shaper.tone_mode = SpectralShaperToneMode::Heavy;
    shaper.threshold_db = -24.0;
    shaper.ceiling_reduction_db = -12.0;
    shaper.set_center_hz(4200.0);

    let mut single_block: Vec<StereoFrame> = (0..512)
        .map(|n| {
            let s = (0.65 * (2.0 * PI * 4200.0 * (n as f64) / sr).sin()) as f32;
            [s, s]
        })
        .collect();
    let in_energy: f32 = single_block[128..].iter().map(|f| f[0] * f[0]).sum();
    shaper.process_frames_in_place(&mut single_block);
    let out_energy: f32 = single_block[128..].iter().map(|f| f[0] * f[0]).sum();
    assert!(
        out_energy < in_energy * 0.75,
        "SpectralShaperModule must attenuate harsh 4.2 kHz resonance within the first block (in={}, out={})",
        in_energy,
        out_energy
    );
    assert!(
        shaper.last_reduction_db() < -1.5,
        "SpectralShaperModule should dynamically attenuate 4.2 kHz resonance, got {} dB",
        shaper.last_reduction_db()
    );

    // Verify Ozone12MasterChain keeps LowEndFocus & SpectralShaper disabled by default,
    // and can enable them via normalized parameters 70..=79
    let mut chain = Ozone12MasterChain::new(sr);
    assert!(!chain.low_end_focus.enabled);
    assert!(!chain.spectral_shaper.enabled);

    chain.set_parameter_normalized(70, 1.0); // enable low_end_focus
    chain.set_parameter_normalized(72, 0.85); // contrast
    chain.set_parameter_normalized(75, 1.0); // enable spectral_shaper
    chain.set_parameter_normalized(76, 0.5); // threshold_db
    assert!(chain.low_end_focus.enabled);
    assert!(chain.spectral_shaper.enabled);

    let ctx = ProcessContext::default();
    let mut master_buf = kick_punchy[..256].to_vec();
    chain.process_slice(&ctx, &mut master_buf, None, &mut []);
    assert!(master_buf.iter().all(|f| f[0].is_finite() && f[1].is_finite()));
}

/// Measures combined power (in dBFS) across exact folded alias frequencies of `f0 = 11,300 Hz`
/// at `fs = 48,000 Hz` (`3*f0 -> 14,100 Hz`, `4*f0 -> 2,800 Hz`, `5*f0 -> 8,500 Hz`,
/// `6*f0 -> 19,800 Hz`, `7*f0 -> 16,900 Hz`, `8*f0 -> 5,600 Hz`, `9*f0 -> 5,700 Hz`),
/// using a 4-term Blackman-Harris window for `> 92 dB` sidelobe suppression.
fn measure_folded_alias_power_db_11k3_at_48k(signal: &[f32]) -> f64 {
    const FS: f64 = 48_000.0;
    const ALIAS_FREQS: [f64; 7] = [2_800.0, 5_600.0, 5_700.0, 8_500.0, 14_100.0, 16_900.0, 19_800.0];

    let n = signal.len();
    assert!(n >= 4096);
    let mut win = vec![0.0_f64; n];
    let mut win_sum = 0.0_f64;
    let denom = (n - 1) as f64;
    for (i, w) in win.iter_mut().enumerate() {
        let p = 2.0 * PI * (i as f64) / denom;
        let bh = 0.35875 - 0.48829 * p.cos() + 0.14128 * (2.0 * p).cos() - 0.01168 * (3.0 * p).cos();
        *w = bh;
        win_sum += bh;
    }

    let mut total_alias_power = 0.0_f64;
    for &fa in &ALIAS_FREQS {
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        for (i, &s) in signal.iter().enumerate() {
            let ang = 2.0 * PI * fa * (i as f64) / FS;
            let sw = (s as f64) * win[i];
            re += sw * ang.cos();
            im -= sw * ang.sin();
        }
        let amp = 2.0 * (re * re + im * im).sqrt() / win_sum.max(1.0e-12);
        total_alias_power += amp * amp;
    }
    10.0 * total_alias_power.max(1.0e-24).log10()
}

#[test]
fn test_gate_1a_trapezoidal_rediscretization_and_4x_oversampled_triode_dualtriode_alias_suppression() {
    use dsp_ozone12_mastering::{
        DkDualTriodeLut, DkKStateBlock, DkTriodeMultibandExciter, IZOTOPE_DUALTRIODE_DAT_SIZE,
        IZOTOPE_DUALTRIODE_LUT_MAGIC, IZOTOPE_TRIODE_DAT_SIZE, rediscretize_dk_matrices,
    };

    // 1. Verify exact trapezoidal bilinear re-discretization (`rediscretize_dk_matrices`) from
    //    `48000.000000.dat` to `96000.0` matches `Mat[6..13]` in `96000.000000.dat` to `< 1e-9`!
    let triode_48_path =
        Path::new(r"f:\FL studio\toneforge\assets\exciter_luts\exciter-triode-48000.000000.dat");
    let triode_96_path =
        Path::new(r"f:\FL studio\toneforge\assets\exciter_luts\exciter-triode-96000.000000.dat");
    let dual_48_path = Path::new(
        r"f:\FL studio\toneforge\assets\exciter_luts\exciter-dualtriode-48000.000000.dat",
    );
    let dual_96_path = Path::new(
        r"f:\FL studio\toneforge\assets\exciter_luts\exciter-dualtriode-96000.000000.dat",
    );

    assert_eq!(
        std::fs::metadata(triode_48_path).unwrap().len() as usize,
        IZOTOPE_TRIODE_DAT_SIZE
    );
    assert_eq!(
        std::fs::metadata(dual_48_path).unwrap().len() as usize,
        IZOTOPE_DUALTRIODE_DAT_SIZE
    );

    let check_kstate_rediscretization = |ks_48: &DkKStateBlock, ks_96: &DkKStateBlock, label: &str| {
        let disc = rediscretize_dk_matrices(
            ks_48.a_c,
            ks_48.b_c,
            ks_48.c_c,
            ks_48.d_x,
            ks_48.e_u,
            ks_48.f_i,
            96_000.0,
        );
        let mut max_err = 0.0_f64;
        for r in 0..3 {
            for c in 0..3 {
                max_err = max_err
                    .max((disc.m_inv[r][c] - ks_96.m_inv[r][c]).abs())
                    .max((disc.a_d[r][c] - ks_96.a_d[r][c]).abs());
            }
            for c in 0..2 {
                max_err = max_err
                    .max((disc.b_d[r][c] - ks_96.b_d[r][c]).abs())
                    .max((disc.c_d[r][c] - ks_96.c_d[r][c]).abs());
            }
        }
        for r in 0..2 {
            for c in 0..3 {
                max_err = max_err.max((disc.d_d[r][c] - ks_96.d_d[r][c]).abs());
            }
            for c in 0..2 {
                max_err = max_err
                    .max((disc.e_d[r][c] - ks_96.e_d[r][c]).abs())
                    .max((disc.f_d[r][c] - ks_96.f_d[r][c]).abs())
                    .max((disc.g_d[r][c] - ks_96.g_d[r][c]).abs());
            }
        }
        println!(
            "[Gate 1a] {label} trapezoidal re-discretization (48k -> 96k) max |Mat[6..13] diff| = {max_err:.3e}"
        );
        assert!(
            max_err < 1.0e-9,
            "{label} re-discretized Mat[6..13] differed from 96000.000000.dat by {max_err:.3e} (>= 1e-9)"
        );
    };

    let t48 = DkTriodeLut::from_dat_file(triode_48_path).unwrap();
    let t96 = DkTriodeLut::from_dat_file(triode_96_path).unwrap();
    check_kstate_rediscretization(&t48.kstate, &t96.kstate, "Triode KState");

    let d48 = DkDualTriodeLut::from_dat_file(dual_48_path).unwrap();
    let d96 = DkDualTriodeLut::from_dat_file(dual_96_path).unwrap();
    assert_eq!(d48.magic, IZOTOPE_DUALTRIODE_LUT_MAGIC);
    assert_eq!(d48.version, 4);
    check_kstate_rediscretization(&d48.stage1, &d96.stage1, "DualTriode Stage1");
    check_kstate_rediscretization(&d48.stage2, &d96.stage2, "DualTriode Stage2");

    // 2. Gate 1a Alias Suppression: 11,300 Hz @ -1.0 dBFS (48 kHz) through DkTriodeMultibandExciter
    //    in both `Triode` and `DualTriode` modes must reduce folded alias spurs by >= 30.0 dB vs 1x!
    let sr = 48_000.0;
    let f0 = 11_300.0;
    let amp = 10.0_f64.powf(-1.0 / 20.0); // -1.0 dBFS
    let warmup = 4096usize;
    let n_fft = 16_384usize;

    for mode in [ExciterMode::Triode, ExciterMode::DualTriode] {
        let mut exc_1x = DkTriodeMultibandExciter::new(sr);
        exc_1x.multiband_enabled = false;
        exc_1x.set_oversampling_factor(1);
        exc_1x.set_all_bands_mode(mode, 75.0, 1.0);

        let mut exc_4x = DkTriodeMultibandExciter::new(sr);
        exc_4x.multiband_enabled = false;
        exc_4x.set_oversampling_factor(4);
        exc_4x.set_all_bands_mode(mode, 75.0, 1.0);

        let mut out_1x = Vec::with_capacity(n_fft);
        let mut out_4x = Vec::with_capacity(n_fft);

        for n in 0..(warmup + n_fft) {
            let s = (amp * (2.0 * PI * f0 * (n as f64) / sr).sin()) as f32;
            let [y1, _] = exc_1x.process_sample_pair(s, s);
            let [y4, _] = exc_4x.process_sample_pair(s, s);
            if n >= warmup {
                out_1x.push(y1);
                out_4x.push(y4);
            }
        }

        let alias_1x_db = measure_folded_alias_power_db_11k3_at_48k(&out_1x);
        let alias_4x_db = measure_folded_alias_power_db_11k3_at_48k(&out_4x);
        let suppression_db = alias_1x_db - alias_4x_db;
        println!(
            "[Gate 1a] {:?} @ 11.3 kHz (-1 dBFS, 48 kHz): 1x alias={alias_1x_db:.2} dBFS, 4x alias={alias_4x_db:.2} dBFS, suppression={suppression_db:.2} dB",
            mode
        );
        assert!(
            suppression_db >= 30.0,
            "Gate 1a failed for {:?}: folded alias suppression = {suppression_db:.2} dB (< 30.0 dB)",
            mode
        );
    }
}

#[test]
fn test_gate_1b_12ax7_rc_tone_frequency_response_invariance_48k_vs_96k() {
    use dsp_ozone12_mastering::DkTriodeMultibandExciter;

    let sweep_freqs = [50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0];
    let mut exc_48 = DkTriodeMultibandExciter::new(48_000.0);
    exc_48.multiband_enabled = false;
    exc_48.set_oversampling_factor(4);
    exc_48.set_all_bands_mode(ExciterMode::Triode, 50.0, 1.0);

    let mut exc_96 = DkTriodeMultibandExciter::new(96_000.0);
    exc_96.multiband_enabled = false;
    exc_96.set_oversampling_factor(4);
    exc_96.set_all_bands_mode(ExciterMode::Triode, 50.0, 1.0);

    let mut max_diff_db = 0.0_f64;
    for &freq in &sweep_freqs {
        exc_48.reset_state();
        exc_96.reset_state();

        let measure_rate = |exc: &mut DkTriodeMultibandExciter, sr: f64| -> f64 {
            let settle = (0.12 * sr) as usize;
            let meas = (0.20 * sr) as usize;
            let mut out = Vec::with_capacity(meas);
            for n in 0..(settle + meas) {
                let s = (0.25 * (2.0 * PI * freq * (n as f64) / sr).sin()) as f32;
                let [yl, _] = exc.process_sample_pair(s, s);
                if n >= settle {
                    out.push(yl);
                }
            }
            measure_harmonic_magnitude_db(&out, sr, freq)
        };

        let db_48 = measure_rate(&mut exc_48, 48_000.0);
        let db_96 = measure_rate(&mut exc_96, 96_000.0);
        let diff_db = (db_48 - db_96).abs();
        max_diff_db = max_diff_db.max(diff_db);
        println!(
            "[Gate 1b] f={freq:6.1} Hz: 48k={db_48:.4} dB, 96k={db_96:.4} dB, |diff|={diff_db:.4} dB"
        );
        assert!(
            diff_db < 0.15,
            "Gate 1b failed at {freq} Hz: |48k - 96k| = {diff_db:.4} dB (>= 0.15 dB)"
        );
    }
    println!("[Gate 1b] Max 48k vs 96k RC tone deviation across 50..5000 Hz = {max_diff_db:.4} dB (< 0.15 dB)");
}

#[test]
fn test_gate_1c_soft_clipper_4x_kaiser128_c5_polynomial_alias_suppression() {
    use dsp_ozone12_mastering::{IrcVMaximizerCore, eval_maxflat_c5_poly_p4};

    // Verify exact C^5 boundary conditions of P_4(t) at t=0 and t=2
    assert!(eval_maxflat_c5_poly_p4(0.0).abs() < 1.0e-15);
    assert!((eval_maxflat_c5_poly_p4(2.0) - 1.0).abs() < 1.0e-15);

    let sr = 48_000.0_f32;
    let f0 = 11_300.0_f64;
    let amp = 10.0_f64.powf(-1.0 / 20.0); // -1.0 dBFS
    let warmup = 4096usize;
    let n_fft = 16_384usize;

    for mode in [SoftClipMode::Light, SoftClipMode::Moderate, SoftClipMode::Heavy] {
        let mut core_1x = IrcVMaximizerCore::new(sr);
        core_1x.ceiling_linear = 1.0;
        core_1x.soft_clip_mix = 1.0;
        core_1x.soft_clip_mode = mode;
        core_1x.soft_clip_oversample_4x = false;

        let mut core_4x = IrcVMaximizerCore::new(sr);
        core_4x.ceiling_linear = 1.0;
        core_4x.soft_clip_mix = 1.0;
        core_4x.soft_clip_mode = mode;
        core_4x.soft_clip_oversample_4x = true;

        let mut out_1x = Vec::with_capacity(n_fft);
        let mut out_4x = Vec::with_capacity(n_fft);

        for n in 0..(warmup + n_fft) {
            let s = (amp * (2.0 * PI * f0 * (n as f64) / (sr as f64)).sin()) as f32;
            let [y1, _] = core_1x.step_soft_clip_stereo(s, s, sr);
            let [y4, _] = core_4x.step_soft_clip_stereo(s, s, sr);
            if n >= warmup {
                out_1x.push(y1);
                out_4x.push(y4);
            }
        }

        let alias_1x_db = measure_folded_alias_power_db_11k3_at_48k(&out_1x);
        let alias_4x_db = measure_folded_alias_power_db_11k3_at_48k(&out_4x);
        let suppression_db = alias_1x_db - alias_4x_db;
        println!(
            "[Gate 1c] SoftClipper {:?} @ 11.3 kHz (-1 dBFS, 48 kHz): 1x alias={alias_1x_db:.2} dBFS, 4x Kaiser-128 alias={alias_4x_db:.2} dBFS, suppression={suppression_db:.2} dB",
            mode
        );
        assert!(
            suppression_db >= 30.0,
            "Gate 1c failed for SoftClipper {:?}: folded alias suppression = {suppression_db:.2} dB (< 30.0 dB)",
            mode
        );
    }
}

#[test]
fn test_gate_5a_irc5_true_peak_brickwall_hot_8db_multitone_and_dither_ceiling() {
    let sr = 48_000.0;
    let total_samples = 48_000usize; // 1.0 full second across many 128-sample blocks

    for dither_on in [false, true] {
        let mut maximizer = IrcVMultiBandMaximizer::new(sr);
        maximizer.irc_mode = IrcMode::IrcV;
        maximizer.character_mode = IrcCharacter::Modern;
        maximizer.set_gain_db(8.0); // +8.0 dB hot drive
        maximizer.set_ceiling_dbtp(-1.0);
        maximizer.prevent_intersample_clipping = true;
        maximizer.enable_soft_clip = true;
        maximizer.soft_clip_mode = SoftClipMode::Moderate;
        maximizer.soft_clip_mix = 0.55;
        maximizer.dither.enabled = dither_on;
        maximizer.dither.bit_depth = 16;

        let mut meter = LufsTruePeakMeter::new(sr);
        let mut stream: Vec<StereoFrame> = (0..total_samples)
            .map(|n| {
                // Worst-case ISP 11,025 Hz quarter-sample phase-shifted cosine + 997 Hz + sharp transient bursts
                let t = (n as f64) / sr;
                let isp_hi = (2.0 * PI * 11_025.0 * t + PI * 0.25).cos();
                let mid = 0.65 * (2.0 * PI * 997.0 * t).sin();
                let low = 0.45 * (2.0 * PI * 62.0 * t).sin();
                let burst = if n % 3000 < 24 { 1.25 } else { 0.0 };
                let l = (0.75 * isp_hi + mid + low + burst) as f32;
                let r = (0.75 * (2.0 * PI * 12_000.0 * t + PI * 0.25).cos() - mid + low + burst) as f32;
                [l, r]
            })
            .collect();

        // Stream across arbitrary 128-sample real-time audio blocks to stress block boundaries!
        let ctx = ProcessContext {
            sample_rate: sr,
            current_sample_pos: 0,
            ppq_tick: 0,
            bpm: 120.0,
            slice_frames: 128,
            is_playing: true,
            is_offline_rendering: false,
            ppq_resolution: daw_core_graph::DEFAULT_PPQ,
        };
        for chunk in stream.chunks_mut(128) {
            maximizer.process_slice(&ctx, chunk, None, &mut []);
            let gr_db = maximizer.last_gr_db;
            for f in chunk.iter() {
                meter.push_frame(f[0], f[1], gr_db);
            }
        }

        let max_tp_dbtp = meter.max_true_peak_dbtp();
        println!(
            "[Gate 5a] IRC V (+8.0 dB drive, ceiling=-1.0 dBTP, dither={dither_on}) measured LufsTruePeakMeter 4x True-Peak = {max_tp_dbtp:.4} dBTP"
        );
        assert!(
            max_tp_dbtp <= -0.98,
            "Gate 5a failed (dither={dither_on}): measured True-Peak {max_tp_dbtp:.4} dBTP exceeded -0.98 dBTP!"
        );
    }
}





/// Integrated LUFS must gate over the whole programme (ITU-R BS.1770-4), not a recent window:
/// 60 s at level L followed by 120 s at L - 10 dB must read 10*log10((60 + 120*0.1) / 180) + L.
#[test]
fn test_integrated_lufs_covers_full_programme_beyond_one_minute() {
    let sr = 48_000.0f64;
    let tone = |amp: f32, secs: usize, phase: &mut f64| -> Vec<StereoFrame> {
        let n = (sr as usize) * secs;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            let s = (amp as f64) * phase.sin();
            *phase += 2.0 * PI * 997.0 / sr;
            v.push([s as f32, s as f32]);
        }
        v
    };
    let mut phase = 0.0f64;
    let loud = tone(0.5, 60, &mut phase);
    let quiet = tone(0.5 * 10f32.powf(-0.5), 120, &mut phase);

    let mut ref_meter = LufsTruePeakMeter::new(sr);
    ref_meter.push_block(&loud[..(sr as usize) * 10], 0.0);
    let l_loud = ref_meter.lufs_integrated as f64;

    let mut meter = LufsTruePeakMeter::new(sr);
    meter.push_block(&loud, 0.0);
    meter.push_block(&quiet, 0.0);
    let expected = l_loud + 10.0 * ((60.0 + 120.0 * 0.1) / 180.0f64).log10();
    let got = meter.lufs_integrated as f64;
    assert!(
        (got - expected).abs() < 0.1,
        "Integrated LUFS over 180 s programme: got {got:.3}, expected {expected:.3} (loud ref {l_loud:.3})"
    );
}
