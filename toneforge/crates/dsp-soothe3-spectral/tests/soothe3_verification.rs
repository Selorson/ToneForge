//! Comprehensive Verification Suite for `dsp-soothe3-spectral` (oeksound Soothe3 Clean-Room DSP).
//!
//! Covers:
//! 1. Full 143-parameter system (`26 Global + 8×9 Sidechain EQ + 45 Multichannel/Atmos`) & real `.preset` JSON loading.
//! 2. `BaseRateResampler` (`16-tap` polyphase FIR) exact latency table across 1x/2x/4x and ultrasonic band preservation (`> f_base / 2`).
//! 3. 8-Band Hybrid Series / Parallel Sidechain EQ (`shape == 3` Bandpass parallel sum × series cascade) & `TiltPair` (`500 Hz` / `2000 Hz`).
//! 4. Automatic dynamic resonance suppression on signals with harshness spikes (`Soft` & `Hard` modes, `Delta` solo) & `Soothe3ReductionFrame` (`8 KB`) export.
//! 5. `CepstralMinPhaseReconstructor` minimum-phase causality and **strict zero pre-ringing** on transients (`2,304` samples @ 1x Regular, `0` samples @ 1x Low-Latency).

use daw_core_graph::{ProcessContext, Soothe3ReductionFrame, UnifiedDspNode};
use dsp_soothe3_spectral::{
    apply_min_phase_tail_taper, compute_bin_ballistic_taus_sec, smooth_gains_constant_q_filtfilt,
    BaseRateResampler, CepstralMinPhaseReconstructor, SidechainEq8Band, SidechainFilterShape,
    Soothe3ParameterSet, Soothe3Processor, Soothe3SpectralPipeline, SootheProcessMode, TiltPair,
    TimeResolutionQuality, ZeroDelayPartitionedConvolver4096, MIN_PHASE_CAUSAL_TAPS, SOOTHE3_BINS,
    SOOTHE3_FFT_SIZE, SOOTHE3_LINEAR_PHASE_LATENCY_1X, SOOTHE3_REGULAR_LATENCY_1X,
    SOOTHE3_TOTAL_PARAMS,
};
use rustfft::num_complex::Complex32;
use std::f32::consts::PI;

#[test]
fn test_143_parameter_system_and_real_preset_loading() {
    assert_eq!(SOOTHE3_TOTAL_PARAMS, 143);
    let mut params = Soothe3ParameterSet::new();

    // Verify DigiPageTable.xml parameter name resolution across Global, 8-Band SC EQ, and Atmos
    assert_eq!(Soothe3ParameterSet::find_param_id_by_name("Mode"), Some(0));
    assert_eq!(Soothe3ParameterSet::find_param_id_by_name("Delta"), Some(1));
    assert_eq!(Soothe3ParameterSet::find_param_id_by_name("Depth"), Some(2));
    assert_eq!(Soothe3ParameterSet::find_param_id_by_name("Detail"), Some(3));
    assert_eq!(
        Soothe3ParameterSet::find_param_id_by_name("Linear phase"),
        Some(18)
    );
    assert_eq!(
        Soothe3ParameterSet::find_param_id_by_name("Low latency"),
        Some(19)
    );
    assert_eq!(
        Soothe3ParameterSet::find_param_id_by_name("Band 1 used"),
        Some(26)
    );
    assert_eq!(
        Soothe3ParameterSet::find_param_id_by_name("Band 1 shape"),
        Some(28)
    );
    assert_eq!(
        Soothe3ParameterSet::find_param_id_by_name("Band 8 channels"),
        Some(97)
    );
    assert_eq!(
        Soothe3ParameterSet::find_param_id_by_name("Channel 1 enabled"),
        Some(98)
    );
    assert_eq!(
        Soothe3ParameterSet::find_param_id_by_name("Group link ceiling"),
        Some(142)
    );

    // Verify all 143 normalized parameters can be set and read back
    for id in 0..(SOOTHE3_TOTAL_PARAMS as u32) {
        params.set_normalized(id, 0.6);
        assert!((params.get_normalized(id) - 0.6).abs() < 1e-5);
    }

    // Load real `.preset` JSON files from `extracted_targets/soothe3/Presets/` if present
    let flat_preset_path =
        std::path::Path::new(r"f:\FL studio\extracted_targets\soothe3\Presets\Flat start.preset");
    if flat_preset_path.exists() {
        let json_str = std::fs::read_to_string(flat_preset_path).unwrap();
        let mut proc = Soothe3Processor::new(48_000.0);
        let count = proc.load_preset_json(&json_str).unwrap();
        assert_eq!(count, 143);
        assert_eq!(proc.params().global.mode, SootheProcessMode::Soft);
        assert!((proc.params().global.depth - 0.0).abs() < 1e-4);
        assert!((proc.params().sc_bands[1].depth - 2.5).abs() < 1e-4);
        assert!((proc.params().global.detail - 5.0).abs() < 1e-4);
    }

    let bp_preset_path =
        std::path::Path::new(r"f:\FL studio\extracted_targets\soothe3\Presets\Bandpass.preset");
    if bp_preset_path.exists() {
        let json_str = std::fs::read_to_string(bp_preset_path).unwrap();
        let mut proc = Soothe3Processor::new(48_000.0);
        let count = proc.load_preset_json(&json_str).unwrap();
        assert_eq!(count, 143);
        assert!(proc.params().sc_bands[0].enabled);
        assert_eq!(
            proc.params().sc_bands[0].shape,
            SidechainFilterShape::Bandpass
        );
    }
}

#[test]
fn test_base_rate_resampler_latency_table_and_ultrasonic_preservation() {
    // 1. Verify exact latency table (`soothe3_manual.pdf` p. 100 / `02_proq4_and_soothe3_dsp.md` Section 2.3.A)
    let res_1x = BaseRateResampler::new(48_000.0);
    let res_2x = BaseRateResampler::new(96_000.0);
    let res_4x = BaseRateResampler::new(192_000.0);

    // Regular (Minimum-Phase): 2304 @ 1x, 4704 @ 2x, 9408 @ 4x
    assert_eq!(res_1x.total_latency_samples(false, false), 2304);
    assert_eq!(res_2x.total_latency_samples(false, false), 4704);
    assert_eq!(res_4x.total_latency_samples(false, false), 9408);

    // Low-Latency (Minimum-Phase): 0 @ 1x, 96 @ 2x, 192 @ 4x
    assert_eq!(res_1x.total_latency_samples(false, true), 0);
    assert_eq!(res_2x.total_latency_samples(false, true), 96);
    assert_eq!(res_4x.total_latency_samples(false, true), 192);

    // Linear-Phase: 3456 @ 1x, 7008 @ 2x, 14016 @ 4x
    assert_eq!(res_1x.total_latency_samples(true, false), 3456);
    assert_eq!(res_2x.total_latency_samples(true, false), 7008);
    assert_eq!(res_4x.total_latency_samples(true, false), 14016);

    // 2. Verify ultrasonic band (> 24 kHz) preservation at 96 kHz (2x)
    // Feed a 32 kHz ultrasonic sine wave + a harsh 3.5 kHz resonance spike at 96 kHz sample rate
    let mut proc = Soothe3Processor::new(96_000.0);
    proc.params_mut().global.depth = 8.0;
    proc.params_mut().global.detail = 7.0;
    proc.params_mut().global.low_latency = true; // 96 samples latency @ 96 kHz
    proc.params_mut().sc_bands[1].enabled = true;
    proc.params_mut().sc_bands[1].shape = SidechainFilterShape::Bell;
    proc.params_mut().sc_bands[1].freq_hz = 3500.0;
    proc.params_mut().sc_bands[1].depth = 6.0;
    assert_eq!(proc.latency_samples(), 96);

    let ctx = ProcessContext {
        sample_rate: 96_000.0,
        ..ProcessContext::default()
    };

    let total_samples = 16384usize;
    let mut io = vec![[0.0f32; 2]; total_samples];
    for (n, frame) in io.iter_mut().enumerate() {
        let t = (n as f32) / 96_000.0;
        let ultra_32k = 0.4 * (2.0 * PI * 32_000.0 * t).sin();
        let harsh_3k5 = 0.6 * (2.0 * PI * 3_500.0 * t).sin();
        *frame = [ultra_32k + harsh_3k5, ultra_32k + harsh_3k5];
    }

    for chunk in io.chunks_mut(512) {
        proc.process_slice(&ctx, chunk, None, &mut []);
    }

    // Measure DFT magnitude at 32 kHz (ultrasonic) and 3.5 kHz (baseband resonance) in the steady tail
    let tail = &io[12000..16000];
    let measure_tone_amp = |freq_hz: f32| -> f32 {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (i, frame) in tail.iter().enumerate() {
            let phase = 2.0 * std::f64::consts::PI * (freq_hz as f64) * (i as f64) / 96_000.0;
            re += (frame[0] as f64) * phase.cos();
            im -= (frame[0] as f64) * phase.sin();
        }
        (2.0 * (re * re + im * im).sqrt() / (tail.len() as f64)) as f32
    };

    let amp_ultra_32k = measure_tone_amp(32_000.0);
    let amp_harsh_3k5 = measure_tone_amp(3_500.0);

    // Ultrasonic 32 kHz must be preserved at 0.40 within 0.01, while 3.5 kHz resonance is suppressed!
    assert!(
        (amp_ultra_32k - 0.40).abs() < 0.01,
        "Ultrasonic 32 kHz tone was not preserved: got {} (expected 0.40)",
        amp_ultra_32k
    );
    assert!(
        amp_harsh_3k5 < 0.35,
        "Baseband 3.5 kHz resonance was not suppressed at 96 kHz: got {}",
        amp_harsh_3k5
    );
}

#[test]
fn test_8band_hybrid_series_parallel_sidechain_eq_and_tilt_curves() {
    // 1. Verify TiltPair around 500 Hz and 2000 Hz
    let default_tilt = TiltPair::default();
    assert!((default_tilt.evaluate(100.0) - 1.0).abs() < 1e-6);
    assert!((default_tilt.evaluate(10_000.0) - 1.0).abs() < 1e-6);

    let custom_tilt = TiltPair::new(0.25, 4.0);
    assert!(custom_tilt.evaluate(50.0) < 0.30);
    assert!(custom_tilt.evaluate(15_000.0) > 3.50);

    // 2. Verify Hybrid Series / Parallel Bandpass routing in SidechainEq8Band
    let mut eq = SidechainEq8Band::new();
    let mut bands = [dsp_soothe3_spectral::SidechainEqBandParams::default(); 8];

    // Enable TWO parallel Bandpass bands at 1000 Hz and 5000 Hz (`shape == 3`)
    bands[0].enabled = true;
    bands[0].shape = SidechainFilterShape::Bandpass;
    bands[0].freq_hz = 1000.0;
    bands[0].q = 4.0;
    bands[0].depth = 0.0; // 0 dB = 1.0 peak

    bands[1].enabled = true;
    bands[1].shape = SidechainFilterShape::Bandpass;
    bands[1].freq_hz = 5000.0;
    bands[1].q = 4.0;
    bands[1].depth = 0.0; // 0 dB = 1.0 peak

    // If Bandpass bands were cascaded in series, both 1 kHz and 5 kHz would be attenuated!
    // Because they are summed in PARALLEL, both 1 kHz and 5 kHz pass near unity (>= 0.95)!
    eq.update_weights(&bands, 0.5, SOOTHE3_FFT_SIZE, 48_000.0);
    let bin_1k = (1000.0 * (SOOTHE3_FFT_SIZE as f32) / 48_000.0).round() as usize;
    let bin_5k = (5000.0 * (SOOTHE3_FFT_SIZE as f32) / 48_000.0).round() as usize;
    let bin_200 = (200.0 * (SOOTHE3_FFT_SIZE as f32) / 48_000.0).round() as usize;

    assert!(
        eq.combined_mag_response[bin_1k] > 0.95,
        "Parallel Bandpass 1 kHz peak missing: {}",
        eq.combined_mag_response[bin_1k]
    );
    assert!(
        eq.combined_mag_response[bin_5k] > 0.95,
        "Parallel Bandpass 5 kHz peak missing: {}",
        eq.combined_mag_response[bin_5k]
    );
    assert!(
        eq.combined_mag_response[bin_200] < 0.25,
        "Out-of-band 200 Hz not rejected: {}",
        eq.combined_mag_response[bin_200]
    );

    // Now add a Series Bell boost (+12 dB -> depth = +5.0) at 5000 Hz (`shape == 2`):
    // It multiplies the parallel bandpass sum (`W_EQ = H_series * \sum H_bp`)!
    bands[2].enabled = true;
    bands[2].shape = SidechainFilterShape::Bell;
    bands[2].freq_hz = 5000.0;
    bands[2].q = 2.0;
    bands[2].depth = 5.0; // +12 dB
    eq.update_weights(&bands, 0.5, SOOTHE3_FFT_SIZE, 48_000.0);

    assert!(
        eq.combined_mag_response[bin_5k] > 3.5,
        "Series × Parallel multiplication at 5 kHz failed: {}",
        eq.combined_mag_response[bin_5k]
    );
}

#[test]
fn test_automatic_resonance_suppression_on_harshness_spikes_and_telemetry() {
    let fs = 48_000.0;
    let mut proc = Soothe3Processor::new(fs);
    proc.params_mut().global.mode = SootheProcessMode::Soft;
    proc.params_mut().global.depth = 7.5;
    proc.params_mut().global.detail = 8.0;
    proc.params_mut().global.attack = 8.0;
    proc.params_mut().global.release = 5.0;
    proc.params_mut().global.quality = TimeResolutionQuality::High;

    // Emphasize harshness region around 3.5 kHz
    proc.params_mut().sc_bands[1].enabled = true;
    proc.params_mut().sc_bands[1].shape = SidechainFilterShape::Bell;
    proc.params_mut().sc_bands[1].freq_hz = 3500.0;
    proc.params_mut().sc_bands[1].depth = 5.0;

    let ctx = ProcessContext::default();
    let total_samples = 16384usize;
    let mut io = vec![[0.0f32; 2]; total_samples];

    // Synthesize a broadband multi-tone musical bed + a loud harsh 3.5 kHz resonance spike
    for (n, frame) in io.iter_mut().enumerate() {
        let t = (n as f32) / fs;
        let bed = 0.08 * (2.0 * PI * 220.0 * t).sin()
            + 0.08 * (2.0 * PI * 440.0 * t).sin()
            + 0.08 * (2.0 * PI * 880.0 * t).sin()
            + 0.08 * (2.0 * PI * 1760.0 * t).sin()
            + 0.08 * (2.0 * PI * 7000.0 * t).sin();
        let harsh_spike = 0.65 * (2.0 * PI * 3500.0 * t).sin();
        let s = bed + harsh_spike;
        *frame = [s, s];
    }

    for chunk in io.chunks_mut(512) {
        proc.process_slice(&ctx, chunk, None, &mut []);
    }

    // Measure amplitude of the 440 Hz musical fundamental vs the 3.5 kHz harsh resonance spike in the output tail
    let tail = &io[10000..15000];
    let measure_amp = |freq_hz: f32| -> f32 {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (i, frame) in tail.iter().enumerate() {
            let phase = 2.0 * std::f64::consts::PI * (freq_hz as f64) * (i as f64) / (fs as f64);
            re += (frame[0] as f64) * phase.cos();
            im -= (frame[0] as f64) * phase.sin();
        }
        (2.0 * (re * re + im * im).sqrt() / (tail.len() as f64)) as f32
    };

    let out_bed_440 = measure_amp(440.0);
    let out_spike_3500 = measure_amp(3500.0);
    let spike_atten_db = 20.0 * (out_spike_3500 / 0.65).log10();

    // The 3.5 kHz harshness spike must be attenuated by at least 8 dB, while the 440 Hz fundamental stays intact!
    assert!(
        spike_atten_db < -8.0,
        "Harsh 3.5 kHz spike was not sufficiently suppressed: spike_atten_db = {} dB",
        spike_atten_db
    );
    assert!(
        (out_bed_440 - 0.08).abs() < 0.015,
        "Musical bed at 440 Hz was degraded: got {} (expected 0.08)",
        out_bed_440
    );

    // Verify Soothe3ReductionFrame (8 KB = 8,192 bytes) telemetry export
    let mut red_frame = Soothe3ReductionFrame::default();
    assert_eq!(std::mem::size_of::<Soothe3ReductionFrame>(), 8192);
    proc.export_reduction_frame(&mut red_frame);

    let max_red_y = red_frame
        .reduction_ys
        .iter()
        .flatten()
        .copied()
        .fold(0.0f32, f32::max);
    assert!(
        max_red_y > 0.20,
        "Soothe3ReductionFrame reduction_ys was not populated: max_red_y={}",
        max_red_y
    );
}

#[test]
fn test_cepstral_min_phase_reconstructor_zero_pre_ringing_on_transients() {
    // 1. Direct unit test of CepstralMinPhaseReconstructor:
    // Create a sharp -18 dB notch at bin 300 (~3.5 kHz) and reconstruct its minimum-phase impulse response
    let mut recon = CepstralMinPhaseReconstructor::new();
    let mut gains_lin = [1.0f32; SOOTHE3_BINS];
    for k in 285..=315 {
        let dist = ((k as f32) - 300.0) / 6.0;
        let db = -18.0 * (-0.5 * dist * dist).exp();
        gains_lin[k] = 10.0f32.powf(db / 20.0);
    }

    let mut h_min = [Complex32::new(0.0, 0.0); SOOTHE3_BINS];
    let mut ir_time = [0.0f32; SOOTHE3_FFT_SIZE];
    recon.reconstruct_minimum_phase_ir(&gains_lin, &mut h_min, &mut ir_time);

    // Verify frequency magnitude accuracy: |H_min[k]| == gains_lin[k]
    for k in 0..SOOTHE3_BINS {
        assert!(
            (h_min[k].norm() - gains_lin[k]).abs() < 1e-4,
            "Cepstral magnitude mismatch at bin {}",
            k
        );
    }

    // Verify minimum-phase causality: > 99.9% of the impulse energy is concentrated in the causal half (0..N/2),
    // and sample 0 (`ir_time[0]`) is the dominant instantaneous peak!
    let causal_energy: f32 = ir_time[..256].iter().map(|x| x * x).sum();
    let anticausal_energy: f32 = ir_time[SOOTHE3_FFT_SIZE - 256..]
        .iter()
        .map(|x| x * x)
        .sum();
    assert!(
        ir_time[0] > 0.90 && anticausal_energy < 1e-6 * causal_energy,
        "Cepstral IR is not causal: ir[0]={}, causal_e={}, anticausal_e={}",
        ir_time[0],
        causal_energy,
        anticausal_energy
    );

    // 2. Full Soothe3Processor transient test in Regular Minimum-Phase Mode (2,304 samples latency @ 1x):
    // First prime the resonance detector with a 3.5 kHz resonance in the sidechain so an active
    // minimum-phase notch filter is engaged, then send a sharp transient impulse at `t_0 = 1000` on main_io!
    let mut proc = Soothe3Processor::new(48_000.0);
    proc.params_mut().global.depth = 8.0;
    proc.params_mut().global.detail = 7.0;
    proc.params_mut().global.sidechain = true;
    proc.params_mut().global.linear_phase = false;
    proc.params_mut().global.low_latency = false;
    assert_eq!(proc.latency_samples(), SOOTHE3_REGULAR_LATENCY_1X);

    let ctx = ProcessContext::default();
    let total_len = 6000usize;
    let transient_t0 = 2000usize;
    let expected_arrival = transient_t0 + SOOTHE3_REGULAR_LATENCY_1X; // 2000 + 2304 = 4304

    let mut main_io = vec![[0.0f32; 2]; total_len];
    main_io[transient_t0] = [1.0, 1.0]; // Unit Dirac impulse at t_0 = 2000

    let mut sc_in = vec![[0.0f32; 2]; total_len];
    for (n, frame) in sc_in.iter_mut().enumerate() {
        let s = 0.8 * (2.0 * PI * 3500.0 * (n as f32) / 48_000.0).sin();
        *frame = [s, s];
    }

    for (m_chunk, s_chunk) in main_io.chunks_mut(256).zip(sc_in.chunks(256)) {
        proc.process_slice(&ctx, m_chunk, Some(s_chunk), &mut []);
    }

    // Check EVERY sample before `expected_arrival` (`0 .. 4304`):
    // Because the minimum-phase filter is strictly causal, pre-ringing before `t_0 + 2304` must be EXACTLY 0.0!
    let max_pre_ring = main_io[..expected_arrival]
        .iter()
        .map(|f| f[0].abs())
        .fold(0.0f32, f32::max);
    let peak_at_arrival = main_io[expected_arrival][0].abs();
    let post_ring_energy: f32 = main_io[expected_arrival + 1..expected_arrival + 64]
        .iter()
        .map(|f| f[0] * f[0])
        .sum();

    assert_eq!(
        max_pre_ring, 0.0,
        "Pre-ringing detected before transient arrival at sample {}: max_pre_ring = {}",
        expected_arrival, max_pre_ring
    );
    assert!(
        peak_at_arrival > 0.80 && post_ring_energy > 1e-5,
        "Transient did not arrive or was not minimum-phase filtered at sample {}: peak={}, post_e={}",
        expected_arrival,
        peak_at_arrival,
        post_ring_energy
    );

    // 3. Verify Zero-Latency Mode (0 samples latency @ 1x): transient at t_0 = 1000 arrives at sample 1000 with 0 pre-ringing!
    proc.reset_state();
    proc.params_mut().global.low_latency = true;
    assert_eq!(proc.latency_samples(), 0);
    let mut main_zero_lat = vec![[0.0f32; 2]; 2048];
    main_zero_lat[1000] = [1.0, 1.0];
    proc.process_slice(&ctx, &mut main_zero_lat, Some(&sc_in[..2048]), &mut []);
    let max_pre_zero_lat = main_zero_lat[..1000]
        .iter()
        .map(|f| f[0].abs())
        .fold(0.0f32, f32::max);
    assert_eq!(max_pre_zero_lat, 0.0);
    assert!(main_zero_lat[1000][0].abs() > 0.80);

    // 4. Verify Linear-Phase Mode latency (3,456 samples @ 1x)
    proc.reset_state();
    proc.params_mut().global.low_latency = false;
    proc.params_mut().global.linear_phase = true;
    assert_eq!(proc.latency_samples(), SOOTHE3_LINEAR_PHASE_LATENCY_1X);
}

#[test]
fn test_high_rate_2x_4x_exact_impulse_alignment_and_midstream_sr_switching() {
    // 1. Verify exact 96-sample impulse peak alignment at 96 kHz (2x Low-Latency)
    let mut proc_2x = Soothe3Processor::new(96_000.0);
    proc_2x.params_mut().global.low_latency = true;
    proc_2x.params_mut().global.depth = 0.0; // Unity passband through wet path
    assert_eq!(proc_2x.latency_samples(), 96);

    let ctx_2x = ProcessContext {
        sample_rate: 96_000.0,
        ..ProcessContext::default()
    };
    let mut io_2x = vec![[0.0f32; 2]; 256];
    io_2x[10] = [1.0, 1.0]; // Dirac impulse at n = 10 -> must peak at n = 10 + 96 = 106
    proc_2x.process_slice(&ctx_2x, &mut io_2x, None, &mut []);

    let (peak_idx_2x, peak_val_2x) = io_2x
        .iter()
        .enumerate()
        .fold((0usize, 0.0f32), |(bi, bv), (i, f)| {
            if f[0].abs() > bv {
                (i, f[0].abs())
            } else {
                (bi, bv)
            }
        });
    assert_eq!(
        peak_idx_2x,
        10 + 96,
        "96 kHz (2x) Low-Latency impulse peaked at sample {} instead of 106 (val={})",
        peak_idx_2x,
        peak_val_2x
    );

    // 2. Verify exact 192-sample impulse peak alignment at 192 kHz (4x Low-Latency)
    let mut proc_4x = Soothe3Processor::new(192_000.0);
    proc_4x.params_mut().global.low_latency = true;
    proc_4x.params_mut().global.depth = 0.0;
    assert_eq!(proc_4x.latency_samples(), 192);

    let ctx_4x = ProcessContext {
        sample_rate: 192_000.0,
        ..ProcessContext::default()
    };
    let mut io_4x = vec![[0.0f32; 2]; 512];
    io_4x[20] = [1.0, 1.0]; // Dirac impulse at n = 20 -> must peak at n = 20 + 192 = 212
    proc_4x.process_slice(&ctx_4x, &mut io_4x, None, &mut []);

    let (peak_idx_4x, peak_val_4x) = io_4x
        .iter()
        .enumerate()
        .fold((0usize, 0.0f32), |(bi, bv), (i, f)| {
            if f[0].abs() > bv {
                (i, f[0].abs())
            } else {
                (bi, bv)
            }
        });
    assert_eq!(
        peak_idx_4x,
        20 + 192,
        "192 kHz (4x) Low-Latency impulse peaked at sample {} instead of 212 (val={})",
        peak_idx_4x,
        peak_val_4x
    );

    // 3. Verify mid-stream sample rate switching (48k -> 96k -> 192k -> 44.1k) and sidechain_listen
    let mut proc_sw = Soothe3Processor::new(48_000.0);
    proc_sw.params_mut().global.sidechain_listen = true;
    proc_sw.params_mut().sc_bands[0].enabled = true;
    proc_sw.params_mut().sc_bands[0].shape = SidechainFilterShape::Bandpass;
    proc_sw.params_mut().sc_bands[0].freq_hz = 2000.0;
    for &sr in &[48_000.0, 96_000.0, 192_000.0, 44_100.0] {
        let ctx = ProcessContext {
            sample_rate: sr,
            ..ProcessContext::default()
        };
        let mut buf = [[0.25f32, -0.25f32]; 256];
        proc_sw.process_slice(&ctx, &mut buf, None, &mut []);
        assert!(buf.iter().all(|f| f[0].is_finite() && f[1].is_finite()));
    }
}

#[test]
fn test_gate3a_4096_tap_partitioned_overlap_save_simd_convolver_snr() {
    assert_eq!(MIN_PHASE_CAUSAL_TAPS, 4096);

    // Build a realistic multi-resonance suppression mask including a narrow 80 Hz sub-bass notch
    // so the causal minimum-phase impulse response rings significantly across taps 0..4096.
    let bin_hz = 48_000.0_f32 / (SOOTHE3_FFT_SIZE as f32);
    let mut raw_gains = [1.0f32; SOOTHE3_BINS];
    for k in 1..SOOTHE3_BINS {
        let f = (k as f32) * bin_hz;
        let g80 = 1.0 - 0.80 * (-((f - 80.0) / 14.0).powi(2)).exp();
        let g440 = 1.0 - 0.72 * (-((f - 440.0) / 35.0).powi(2)).exp();
        let g3200 = 1.0 - 0.85 * (-((f - 3200.0) / 180.0).powi(2)).exp();
        let g8500 = 1.0 - 0.65 * (-((f - 8500.0) / 500.0).powi(2)).exp();
        raw_gains[k] = (g80 * g440 * g3200 * g8500).clamp(0.05, 1.0);
    }

    let mut smoothed_gains = [1.0f32; SOOTHE3_BINS];
    smooth_gains_constant_q_filtfilt(&raw_gains, &mut smoothed_gains);

    let mut recon = CepstralMinPhaseReconstructor::new();
    let mut h_min_bins = [Complex32::new(0.0, 0.0); SOOTHE3_BINS];
    let mut ir_4096 = [0.0f32; SOOTHE3_FFT_SIZE];
    recon.reconstruct_minimum_phase_ir(&smoothed_gains, &mut h_min_bins, &mut ir_4096);
    apply_min_phase_tail_taper(&mut ir_4096);

    // Confirm genuine tail energy past tap 512 (proving no 512-tap truncation!)
    let tail_energy_post_512: f64 = ir_4096[512..4096]
        .iter()
        .map(|&v| (v as f64) * (v as f64))
        .sum();
    assert!(
        tail_energy_post_512 > 1e-7,
        "Expected active sub-bass minimum-phase tail past tap 512, got energy={:.3e}",
        tail_energy_post_512
    );

    let mut conv = ZeroDelayPartitionedConvolver4096::new();
    conv.set_ir(&ir_4096, 0);

    // Deterministic broadband multi-sine + LCG PRNG excitation (16,384 samples)
    let n_samples = 16_384usize;
    let mut x_in = vec![0.0f32; n_samples];
    let mut rng_state = 0x9E37_79B9_7F4A_7C15_u64;
    for (n, s) in x_in.iter_mut().enumerate() {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let white = (((rng_state >> 33) as f32) / ((1u64 << 31) as f32)) - 1.0;
        let t = (n as f32) / 48_000.0;
        *s = 0.25 * (2.0 * PI * 80.0 * t).sin()
            + 0.20 * (2.0 * PI * 440.0 * t).sin()
            + 0.20 * (2.0 * PI * 3_200.0 * t).sin()
            + 0.15 * white;
    }

    let mut sig_pow = 0.0f64;
    let mut err_pow = 0.0f64;
    for n in 0..n_samples {
        let y_fast = conv.process_sample(x_in[n]) as f64;
        let max_k = (n + 1).min(SOOTHE3_FFT_SIZE);
        let mut y_ref = 0.0f64;
        for k in 0..max_k {
            y_ref += (ir_4096[k] as f64) * (x_in[n - k] as f64);
        }
        let diff = y_fast - y_ref;
        sig_pow += y_ref * y_ref;
        err_pow += diff * diff;
    }

    let snr_db = 10.0 * (sig_pow / err_pow.max(1e-30)).log10();
    println!(
        "[Gate 3a] ZeroDelayPartitionedConvolver4096 (64-tap f32x8 SIMD + 63x128-pt Overlap-Save) vs f64 4096-tap direct SNR = {:.2} dB (post-512 tail energy = {:.3e})",
        snr_db, tail_energy_post_512
    );
    assert!(
        snr_db >= 115.0,
        "4096-tap partitioned convolver SNR {:.2} dB < 115.0 dB",
        snr_db
    );
}

#[test]
fn test_gate3b_sub_bass_80hz_resonance_isolation_and_tilted_ballistics() {
    // 1. Verify sub-150 Hz tilted temporal ballistics: >= 25 ms attack, >= 80 ms release across 50..100 Hz
    for &f_low in &[50.0f32, 65.0, 80.0, 100.0] {
        let (tau_att, tau_rel) = compute_bin_ballistic_taus_sec(f_low, 5.0, 3.0);
        assert!(
            tau_att >= 0.025 && tau_rel >= 0.080,
            "Sub-bass ballistics too fast at {} Hz: tau_att={:.4}s, tau_rel={:.4}s",
            f_low,
            tau_att,
            tau_rel
        );
    }
    let (tau_att_hi, _) = compute_bin_ballistic_taus_sec(6_000.0, 6.0, 3.0);
    assert!(
        tau_att_hi < 0.012,
        "High-frequency transient attack should stay fast at 6 kHz, got {:.4}s",
        tau_att_hi
    );

    // 2. Multi-sine sub-bass resonance isolation test at 48 kHz:
    //    55 Hz @ 0.10 + 80 Hz @ +15 dB (0.56234) + 110 Hz @ 0.10
    let sr = 48_000.0_f64;
    let mut proc = Soothe3Processor::new(sr as f32);
    {
        let p = proc.params_mut();
        p.global.mode = SootheProcessMode::Hard;
        p.global.depth = 8.5;
        p.global.detail = 8.5;
        p.global.max_cut_db = -24.0;
        for b in &mut p.sc_bands {
            b.enabled = false; // Unweighted flat sidechain across sub-bass
        }
    }

    let ctx = ProcessContext {
        sample_rate: sr,
        ..ProcessContext::default()
    };

    let amp_55 = 0.10_f64;
    let amp_80 = 0.10_f64 * 10.0_f64.powf(15.0 / 20.0); // 0.5623413 (+15 dB resonance)
    let amp_110 = 0.10_f64;

    let warmup = 24_000usize; // 0.5 s warm-up (includes 2,304 lookahead + slow sub-bass envelope settling)
    let meas_n = 48_000usize; // Exact 1.0 s -> 1.0 Hz coherent integer DFT bins for 55, 80, 110, 160, 240 Hz!
    let total = warmup + meas_n;

    let mut out_meas = Vec::with_capacity(meas_n);
    let mut block = [[0.0f32; 2]; 256];
    let mut idx = 0usize;
    while idx < total {
        let chunk = (total - idx).min(256);
        for i in 0..chunk {
            let t = ((idx + i) as f64) / sr;
            let s = (amp_55 * (2.0 * std::f64::consts::PI * 55.0 * t).sin()
                + amp_80 * (2.0 * std::f64::consts::PI * 80.0 * t).sin()
                + amp_110 * (2.0 * std::f64::consts::PI * 110.0 * t).sin())
                as f32;
            block[i] = [s, s];
        }
        proc.process_slice(&ctx, &mut block[..chunk], None, &mut []);
        for i in 0..chunk {
            if idx + i >= warmup {
                out_meas.push(block[i][0] as f64);
            }
        }
        idx += chunk;
    }

    let coherent_amp = |freq_hz: f64| -> f64 {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (n, &y) in out_meas.iter().enumerate() {
            let ph = 2.0 * std::f64::consts::PI * freq_hz * (n as f64) / sr;
            re += y * ph.cos();
            im -= y * ph.sin();
        }
        2.0 * (re * re + im * im).sqrt() / (meas_n as f64)
    };

    let out_55 = coherent_amp(55.0);
    let out_80 = coherent_amp(80.0);
    let out_110 = coherent_amp(110.0);
    let out_160 = coherent_amp(160.0);
    let out_240 = coherent_amp(240.0);

    let atten_55_db = 20.0 * (amp_55 / out_55.max(1e-12)).log10();
    let atten_80_db = 20.0 * (amp_80 / out_80.max(1e-12)).log10();
    let atten_110_db = 20.0 * (amp_110 / out_110.max(1e-12)).log10();
    let selectivity_vs_55_db = atten_80_db - atten_55_db;
    let selectivity_vs_110_db = atten_80_db - atten_110_db;
    let thd_80_db = 20.0
        * ((out_160 * out_160 + out_240 * out_240).sqrt() / out_80.max(1e-12))
            .max(1e-12)
            .log10();

    println!(
        "[Gate 3b] Sub-bass 4096-tap selectivity: atten_55Hz={:.2} dB, atten_80Hz={:.2} dB, atten_110Hz={:.2} dB (delta_55={:.2} dB, delta_110={:.2} dB, 80Hz_THD={:.2} dB)",
        atten_55_db, atten_80_db, atten_110_db, selectivity_vs_55_db, selectivity_vs_110_db, thd_80_db
    );

    assert!(
        selectivity_vs_55_db >= 6.0 && selectivity_vs_110_db >= 6.0,
        "80 Hz sub-bass resonance not selectively suppressed by >= 6.0 dB over 55/110 Hz neighbors: atten_55={:.2} dB, atten_80={:.2} dB, atten_110={:.2} dB",
        atten_55_db,
        atten_80_db,
        atten_110_db
    );
    assert!(
        thd_80_db < -35.0,
        "Sub-bass waveform tearing / harmonic distortion too high at 80 Hz: THD = {:.2} dB",
        thd_80_db
    );
}

#[test]
fn test_gate3c_dual_slot_convolver_abrupt_18db_step_click_free_crossfade() {
    let sr = 48_000.0_f32;
    let mut pipe_xfade = Soothe3SpectralPipeline::new(sr);
    let mut pipe_hard = Soothe3SpectralPipeline::new(sr);

    let unity_mask = [1.0f32; SOOTHE3_BINS];
    let mut step_mask = [1.0f32; SOOTHE3_BINS];
    let bin_hz = sr / (SOOTHE3_FFT_SIZE as f32);
    let gain_minus_18db = 10.0_f32.powf(-18.0 / 20.0); // 0.12589254
    for k in 1..SOOTHE3_BINS {
        let f = (k as f32) * bin_hz;
        if (40.0..=160.0).contains(&f) {
            step_mask[k] = gain_minus_18db;
        }
    }

    pipe_xfade.apply_direct_gain_mask(&unity_mask, 0);
    pipe_hard.apply_direct_gain_mask(&unity_mask, 0);

    // Choose step_idx so sin(2*pi*80*step_idx/48000) is near maximum (+0.8) where an un-crossfaded
    // -18 dB step produces a severe ~0.70 amplitude cliff! (48000 / 80 = 600 samples/period, quarter = 150)
    let step_idx = 6_000 + 150;
    let total_samples = step_idx + 1_024;

    let mut y_xfade = Vec::with_capacity(total_samples);
    let mut y_hard = Vec::with_capacity(total_samples);

    for n in 0..total_samples {
        if n == step_idx {
            pipe_xfade.apply_direct_gain_mask(&step_mask, 256);
            pipe_hard.apply_direct_gain_mask(&step_mask, 0);
        }
        let x = 0.8 * (2.0 * PI * 80.0 * (n as f32) / sr).sin();
        y_xfade.push(pipe_xfade.convolvers[0].process_sample(x));
        y_hard.push(pipe_hard.convolvers[0].process_sample(x));
    }

    let mut max_d2_xfade = 0.0f32;
    let mut max_d2_hard = 0.0f32;
    for n in (step_idx - 4)..(step_idx + 512) {
        let d2_x = (y_xfade[n] - 2.0 * y_xfade[n - 1] + y_xfade[n - 2]).abs();
        let d2_h = (y_hard[n] - 2.0 * y_hard[n - 1] + y_hard[n - 2]).abs();
        max_d2_xfade = max_d2_xfade.max(d2_x);
        max_d2_hard = max_d2_hard.max(d2_h);
    }

    let click_reduction_db = 20.0 * (max_d2_hard.max(1e-12) / max_d2_xfade.max(1e-12)).log10();
    println!(
        "[Gate 3c] Abrupt 0 -> -18 dB mask step @ 80 Hz peak: uncrossfaded_max_d2={:.5e}, dual_slot_xfade_max_d2={:.5e} (transient reduction = {:.2} dB)",
        max_d2_hard, max_d2_xfade, click_reduction_db
    );

    assert!(
        max_d2_xfade < 5.0e-4 && click_reduction_db >= 25.0,
        "Dual-slot crossfade failed click-free C^1 transition: max_d2_xfade={:.5e}, max_d2_hard={:.5e}, reduction={:.2} dB",
        max_d2_xfade,
        max_d2_hard,
        click_reduction_db
    );
}


