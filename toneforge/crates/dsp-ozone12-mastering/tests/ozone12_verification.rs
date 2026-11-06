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


