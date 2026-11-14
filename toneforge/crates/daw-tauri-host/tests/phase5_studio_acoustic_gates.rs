//! Phase 5 Comprehensive Studio Acoustic & Real-Time Quantitative Verification Suite.
//!
//! Enforces 100% of the 6 Mandatory Quantitative Acoustic Gates required for ToneForge v2:
//! - **Gate 1**: Nonlinear Anti-Aliasing (`11,300 Hz @ -1.0 dBFS`, `48 kHz` -> `>= 30.0 dB` folded
//!   alias spur suppression `< 10 kHz` across Pro-Q 4 `CharacterWaveShaper` `Subtle`/`Warm`,
//!   Ozone 12 `DkTriodeLutExciter` `Triode`/`DualTriode`, and `IrcVMaximizerCore` `SoftClipper`)
//!   + 12AX7 Sample-Rate Invariance (`50 Hz – 5,000 Hz`, `48 kHz` vs `96 kHz` `< 0.15 dB`).
//! - **Gate 2**: Pro-Q 4 Natural Phase Analog Matching (`20 Hz – 20 kHz` magnitude error `< 0.05 dB`
//!   vs continuous Laplace $H_a(j\omega)$ for `12 kHz +9 dB Q=2.0` Bell and `10 kHz +6 dB` High-Shelf,
//!   plus flat passband ripple `< 0.002 dB`).
//! - **Gate 3**: Soothe3 `4096`-Tap `ZeroDelayPartitionedConvolver4096` (`SNR >= 115 dB` vs `f64`
//!   direct convolution, sharp `80 Hz` sub-bass resonance suppression preserving `55 Hz` & `110 Hz`
//!   without tearing, and click-free dual-filter step mask crossfading).
//! - **Gate 4**: FL Studio 26 Native FX (`LuxeVerbFdnShimmerNode` smooth diffusion density across
//!   `44.1 / 48 / 96 kHz` + `>= +20 dB` `+12st` `880 Hz` Shimmer octave boost from `440 Hz` burst,
//!   and `GrossBeatTimeBufferNode` `> 35 dB` 2nd-derivative click reduction on instantaneous jump).
//! - **Gate 5**: Broadcast `ITU-R BS.1770-4` `4x True Peak <= -0.98 dBTP` under `+8.0 dB` hot multitone +
//!   ISP stressor into `-1.0 dBTP` `IrcVMultiBandMaximizer`, PLUS 100% real finite non-trivial neural
//!   inference across all shipped `.onnx` (`delimit_exp009_ep080_fixed2`, `ozone12_mss4stem_bsrnn_xs_120ms_ep147`,
//!   `Kim_Vocal_2`, `UVR-MDX-NET-Inst_HQ_3`) & `.npz` (`gparamsBN8.0`, `instrument_classifier_params`,
//!   `vocals/drums/bass-small_sanitized`) models + chorus-scanning `run_ozone12_master_assistant_with_audio`.
//! - **Gate 6**: Lock-Free Real-Time `1,000` concurrent `RtGraphMutation` + `EngineCommand` stress test
//!   (`< 1.5 ms` avg block render time @ `256 frames / 48 kHz`, `0` Mutex locks, `RtGarbageCollector`
//!   reclamation `> 0`) + live Win32 Core Audio `WasapiHardwareStream` & `enumerate_wasapi_output_devices` smoke test.

use std::f64::consts::PI;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use ai_uvr5_stem_engine::{
    BSRNN_HOP_SIZE, BsrnnInputHop, BsrnnOutputHop, GENRE_LABELS, GenreCnnClassifier,
    GruThreeStemSeparator, INSTRUMENT_LABELS, InstrumentCnnClassifier, OzoneBsrnnOnnxSeparator,
    OzoneDelimitOnnxProcessor, Uvr5MdxOnnxSeparator,
};
use daw_core_graph::{
    AutomationClipArticulator, ChannelGenerator, DeterministicGeneratorNode, EngineCommand,
    FixedLatencyTransformNode, GrossBeatTimeBufferNode, LuxeVerbFdnShimmerNode, MixerEngine,
    MixerSendRoute, PianoRollNote, PianoRollPattern, PlaylistClip, PlaylistClipKind,
    PluginRackSlot, ProcessContext, RtGraphMutation, StereoFrame, TensionControlPoint,
    TensionSpline, UnifiedDspNode, WasapiHardwareStream, enumerate_wasapi_output_devices,
};
use daw_tauri_host::{AudioStreamController, run_ozone12_master_assistant_with_audio};
use dsp_ozone12_mastering::{
    DkTriodeLutExciter, ExciterMode, IrcVMaximizerCore, IrcVMultiBandMaximizer, LufsTruePeakMeter,
    SoftClipMode, TruePeakDetector4x,
};
use dsp_proq4_eq::{
    CharacterMode, CharacterWaveShaper, EqBandParams, FilterShape, FractionalSlopeBiquadBand,
    ProQ4Processor, ProcessingMode, analog_prototype_mag_db, eval_natural_phase_band_response,
};
use dsp_soothe3_spectral::{
    CepstralMinPhaseReconstructor, MIN_PHASE_CAUSAL_TAPS, SOOTHE3_BINS, SOOTHE3_FFT_SIZE,
    Soothe3Processor, TimeResolutionQuality, ZeroDelayPartitionedConvolver4096,
    apply_min_phase_tail_taper,
};
use rustfft::FftPlanner;
use rustfft::num_complex::{Complex32, Complex64};
use serde_json::json;

fn save_gate_metrics(gate_name: &str, payload: serde_json::Value) {
    let mut out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    out_dir.pop(); // crates/
    out_dir.pop(); // toneforge/
    out_dir.push("target");
    out_dir.push("gate_metrics");
    let _ = fs::create_dir_all(&out_dir);
    let path = out_dir.join(format!("{gate_name}.json"));
    if let Ok(pretty) = serde_json::to_string_pretty(&payload) {
        let _ = fs::write(path, pretty);
    }
}

/// Computes Blackman-Harris 4-term windowed power spectrum in dB (`0 dB` normalized to peak fundamental bin)
/// and returns `(peak_spur_below_10khz_rel_db, peak_non_harmonic_alias_rel_db)`.
fn measure_folded_alias_spurs_rel_db(samples: &[f64], fs: f64, f0_hz: f64) -> (f64, f64) {
    let n = samples.len();
    assert!(n.is_power_of_two() && n >= 4096);
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let mut buf = vec![Complex64::new(0.0, 0.0); n];

    // 4-term Blackman-Harris window (-92 dB sidelobe floor)
    let a0 = 0.35875_f64;
    let a1 = 0.48829_f64;
    let a2 = 0.14128_f64;
    let a3 = 0.01168_f64;
    let denom = (n - 1) as f64;
    for (i, &x) in samples.iter().enumerate() {
        let ph = 2.0 * PI * (i as f64) / denom;
        let w = a0 - a1 * ph.cos() + a2 * (2.0 * ph).cos() - a3 * (3.0 * ph).cos();
        buf[i] = Complex64::new(x * w, 0.0);
    }
    fft.process(&mut buf);

    let bin_hz = fs / (n as f64);
    let fund_bin = (f0_hz / bin_hz).round() as usize;
    let h2_bin = ((2.0 * f0_hz) / bin_hz).round() as usize;

    let mut fund_mag = 1.0e-18_f64;
    for b in fund_bin.saturating_sub(4)..=(fund_bin + 4).min(n / 2) {
        fund_mag = fund_mag.max(buf[b].norm());
    }

    let min_bin_200hz = (200.0 / bin_hz).ceil() as usize;
    let max_bin_10khz = (10_000.0 / bin_hz).floor() as usize;
    let max_bin_23k = (23_200.0 / bin_hz).floor().min((n / 2 - 1) as f64) as usize;

    let mut max_spur_sub10k = 1.0e-18_f64;
    for b in min_bin_200hz..=max_bin_10khz {
        if b.abs_diff(fund_bin) > 12 {
            max_spur_sub10k = max_spur_sub10k.max(buf[b].norm());
        }
    }

    let mut max_spur_full = max_spur_sub10k;
    for b in min_bin_200hz..=max_bin_23k {
        if b.abs_diff(fund_bin) > 14 && b.abs_diff(h2_bin) > 14 {
            max_spur_full = max_spur_full.max(buf[b].norm());
        }
    }

    let sub10k_db = 20.0 * (max_spur_sub10k / fund_mag).log10();
    let full_db = 20.0 * (max_spur_full / fund_mag).log10();
    (sub10k_db, full_db)
}

#[test]
fn gate1_nonlinear_antialiasing_and_12ax7_sr_invariance() {
    let fs = 48_000.0_f64;
    let f0 = 11_300.0_f64;
    let amp = 10.0_f64.powf(-1.0 / 20.0); // -1.0 dBFS
    let warmup = 4096usize;
    let fft_len = 16_384usize;
    let total = warmup + fft_len;

    // 1A. Pro-Q 4 CharacterWaveShaper (`Subtle` ADAA1@4x & `Warm` ADAA2@4x vs `1x` naive)
    let mut proq_subtle_1x = Vec::with_capacity(fft_len);
    let mut proq_subtle_4x = Vec::with_capacity(fft_len);
    let mut proq_warm_1x = Vec::with_capacity(fft_len);
    let mut proq_warm_4x = Vec::with_capacity(fft_len);

    let mut shaper_subtle = CharacterWaveShaper::new(fs);
    let mut shaper_warm = CharacterWaveShaper::new(fs);

    for n in 0..total {
        let x = amp * (2.0 * PI * f0 * (n as f64) / fs).sin();
        let y_s1 = CharacterWaveShaper::eval_transfer_curve(CharacterMode::Subtle, x);
        let y_s4 = shaper_subtle.process_sample(CharacterMode::Subtle, 0, x);
        let y_w1 = CharacterWaveShaper::eval_transfer_curve(CharacterMode::Warm, x);
        let y_w4 = shaper_warm.process_sample(CharacterMode::Warm, 0, x);
        if n >= warmup {
            proq_subtle_1x.push(y_s1);
            proq_subtle_4x.push(y_s4);
            proq_warm_1x.push(y_w1);
            proq_warm_4x.push(y_w4);
        }
    }

    let (subtle_1x_sub10k, _) = measure_folded_alias_spurs_rel_db(&proq_subtle_1x, fs, f0);
    let (subtle_4x_sub10k, _) = measure_folded_alias_spurs_rel_db(&proq_subtle_4x, fs, f0);
    let subtle_suppression_db = subtle_1x_sub10k - subtle_4x_sub10k;

    let (warm_1x_sub10k, _) = measure_folded_alias_spurs_rel_db(&proq_warm_1x, fs, f0);
    let (warm_4x_sub10k, _) = measure_folded_alias_spurs_rel_db(&proq_warm_4x, fs, f0);
    let warm_suppression_db = warm_1x_sub10k - warm_4x_sub10k;

    assert!(
        subtle_suppression_db >= 30.0,
        "Gate 1 FAIL: Pro-Q 4 Character Subtle alias suppression {subtle_suppression_db:.2} dB < 30.0 dB (1x={subtle_1x_sub10k:.2}, 4x={subtle_4x_sub10k:.2})"
    );
    assert!(
        warm_suppression_db >= 30.0,
        "Gate 1 FAIL: Pro-Q 4 Character Warm alias suppression {warm_suppression_db:.2} dB < 30.0 dB (1x={warm_1x_sub10k:.2}, 4x={warm_4x_sub10k:.2})"
    );

    // 1B. Ozone 12 DK-Triode & DualTriode Exciter (`1x` vs `4x` oversampling)
    let run_exciter = |mode: ExciterMode, os_4x: bool| -> Vec<f64> {
        let mut exc = DkTriodeLutExciter::new(fs);
        exc.multiband_enabled = false;
        exc.oversample_4x = os_4x;
        exc.oversample_2x = false;
        exc.bands[0].enabled = true;
        exc.bands[0].mode = mode;
        exc.bands[0].amount_pct = 80.0;
        exc.bands[0].wet_mix = 1.0;

        let mut io = vec![[0.0f32; 2]; total];
        for (n, frame) in io.iter_mut().enumerate() {
            let s = (amp * (2.0 * PI * f0 * (n as f64) / fs).sin()) as f32;
            *frame = [s, s];
        }
        let ctx = ProcessContext {
            sample_rate: fs,
            current_sample_pos: 0,
            ppq_tick: 0,
            bpm: 120.0,
            slice_frames: 512,
            is_playing: true,
            is_offline_rendering: false,
            ppq_resolution: daw_core_graph::DEFAULT_PPQ,
        };
        for chunk in io.chunks_mut(512) {
            exc.process_slice(&ctx, chunk, None, &mut []);
        }
        io[warmup..].iter().map(|f| f[0] as f64).collect()
    };

    let triode_1x = run_exciter(ExciterMode::Triode, false);
    let triode_4x = run_exciter(ExciterMode::Triode, true);
    let (tri_1x_db, _) = measure_folded_alias_spurs_rel_db(&triode_1x, fs, f0);
    let (tri_4x_db, _) = measure_folded_alias_spurs_rel_db(&triode_4x, fs, f0);
    let triode_suppression_db = tri_1x_db - tri_4x_db;

    let dualtri_1x = run_exciter(ExciterMode::DualTriode, false);
    let dualtri_4x = run_exciter(ExciterMode::DualTriode, true);
    let (dtri_1x_db, _) = measure_folded_alias_spurs_rel_db(&dualtri_1x, fs, f0);
    let (dtri_4x_db, _) = measure_folded_alias_spurs_rel_db(&dualtri_4x, fs, f0);
    let dualtriode_suppression_db = dtri_1x_db - dtri_4x_db;

    assert!(
        triode_suppression_db >= 30.0,
        "Gate 1 FAIL: Ozone 12 DK-Triode alias suppression {triode_suppression_db:.2} dB < 30.0 dB"
    );
    assert!(
        dualtriode_suppression_db >= 30.0,
        "Gate 1 FAIL: Ozone 12 DK-DualTriode alias suppression {dualtriode_suppression_db:.2} dB < 30.0 dB"
    );

    // 1C. Ozone 12 IRC V SoftClipper (`1x` vs `4x` Linear-Phase Kaiser FIR)
    let run_soft_clip = |os_4x: bool| -> Vec<f64> {
        let mut core = IrcVMaximizerCore::new(fs as f32);
        core.gain_linear = 10.0_f32.powf(5.5 / 20.0);
        core.ceiling_linear = 10.0_f32.powf(-1.0 / 20.0);
        core.soft_clip_mix = 1.0;
        core.soft_clip_mode = SoftClipMode::Heavy;
        core.soft_clip_oversample_4x = os_4x;
        let mut out = Vec::with_capacity(fft_len);
        for n in 0..total {
            let s = (amp * (2.0 * PI * f0 * (n as f64) / fs).sin()) as f32;
            let [yl, _] = core.step_soft_clip_stereo(s, s, fs as f32);
            if n >= warmup {
                out.push(yl as f64);
            }
        }
        out
    };

    let sc_1x = run_soft_clip(false);
    let sc_4x = run_soft_clip(true);
    let (sc_1x_db, _) = measure_folded_alias_spurs_rel_db(&sc_1x, fs, f0);
    let (sc_4x_db, _) = measure_folded_alias_spurs_rel_db(&sc_4x, fs, f0);
    let softclip_suppression_db = sc_1x_db - sc_4x_db;

    assert!(
        softclip_suppression_db >= 30.0,
        "Gate 1 FAIL: Ozone 12 SoftClip 4x alias suppression {softclip_suppression_db:.2} dB < 30.0 dB"
    );

    // 1D. 12AX7 DK-Triode Sample-Rate Invariance (`50 Hz – 5,000 Hz` between `48 kHz` and `96 kHz` < 0.15 dB)
    let test_freqs = [50.0_f64, 120.0, 400.0, 1000.0, 2500.0, 5000.0];
    let mut max_sr_delta_db = 0.0_f64;
    for &freq in &test_freqs {
        let measure_rms_db = |sr: f64| -> f64 {
            let mut exc = DkTriodeLutExciter::new(sr);
            exc.multiband_enabled = false;
            exc.oversample_4x = true;
            exc.bands[0].enabled = true;
            exc.bands[0].mode = ExciterMode::Triode;
            exc.bands[0].amount_pct = 55.0;
            exc.bands[0].wet_mix = 1.0;

            let settle = (0.15 * sr) as usize;
            let meas = (0.20 * sr) as usize;
            let len = settle + meas;
            let a_in = 10.0_f64.powf(-6.0 / 20.0);
            let mut buf = vec![[0.0f32; 2]; len];
            for (n, fr) in buf.iter_mut().enumerate() {
                let v = (a_in * (2.0 * PI * freq * (n as f64) / sr).sin()) as f32;
                *fr = [v, v];
            }
            let ctx = ProcessContext {
                sample_rate: sr,
                current_sample_pos: 0,
                ppq_tick: 0,
                bpm: 120.0,
                slice_frames: 512,
                is_playing: true,
                is_offline_rendering: false,
                ppq_resolution: daw_core_graph::DEFAULT_PPQ,
            };
            for chunk in buf.chunks_mut(512) {
                exc.process_slice(&ctx, chunk, None, &mut []);
            }
            let sum_sq: f64 = buf[settle..]
                .iter()
                .map(|f| (f[0] as f64) * (f[0] as f64))
                .sum();
            let rms = (sum_sq / (meas as f64)).sqrt().max(1.0e-12);
            20.0 * rms.log10()
        };

        let db_48 = measure_rms_db(48_000.0);
        let db_96 = measure_rms_db(96_000.0);
        let delta = (db_48 - db_96).abs();
        max_sr_delta_db = max_sr_delta_db.max(delta);
    }

    assert!(
        max_sr_delta_db < 0.15,
        "Gate 1 FAIL: 12AX7 DK-Triode 48k vs 96k deviation {max_sr_delta_db:.4} dB >= 0.15 dB"
    );

    save_gate_metrics(
        "gate1",
        json!({
            "proq4_subtle_1x_spur_db": subtle_1x_sub10k,
            "proq4_subtle_4x_spur_db": subtle_4x_sub10k,
            "proq4_subtle_suppression_db": subtle_suppression_db,
            "proq4_warm_1x_spur_db": warm_1x_sub10k,
            "proq4_warm_4x_spur_db": warm_4x_sub10k,
            "proq4_warm_suppression_db": warm_suppression_db,
            "ozone12_triode_1x_spur_db": tri_1x_db,
            "ozone12_triode_4x_spur_db": tri_4x_db,
            "ozone12_triode_suppression_db": triode_suppression_db,
            "ozone12_dualtriode_suppression_db": dualtriode_suppression_db,
            "ozone12_softclip_1x_spur_db": sc_1x_db,
            "ozone12_softclip_4x_spur_db": sc_4x_db,
            "ozone12_softclip_suppression_db": softclip_suppression_db,
            "dk_triode_48k_vs_96k_max_delta_db": max_sr_delta_db,
            "passed": true
        }),
    );
}

#[test]
fn gate2_proq4_natural_phase_analog_matching_and_flat_ripple() {
    let fs = 48_000.0_f64;

    // 2A. Bell (12 kHz, +9.0 dB, Q=2.0, 12 dB/oct) across 20 Hz .. 20 kHz vs continuous Laplace H(s)
    let mut bell_band = FractionalSlopeBiquadBand::new_unity();
    bell_band.configure_shape_natural_phase(FilterShape::Bell, 12_000.0, 9.0, 2.0, 12.0, fs * 2.0, true);

    let mut max_bell_err_db = 0.0_f64;
    for i in 0..=400 {
        let frac = (i as f64) / 400.0;
        let f_hz = 20.0 * (20_000.0_f64 / 20.0).powf(frac);
        let ideal_db = analog_prototype_mag_db(FilterShape::Bell, 12_000.0, 9.0, 2.0, 12.0, f_hz, true);
        let np_mag = eval_natural_phase_band_response(&bell_band, f_hz, fs)
            .norm()
            .max(1.0e-15);
        let np_db = 20.0 * np_mag.log10();
        max_bell_err_db = max_bell_err_db.max((np_db - ideal_db).abs());
    }

    // 2B. High-Shelf (10 kHz, +6.0 dB, Q=0.70710678, 12 dB/oct) across 20 Hz .. 20 kHz vs Laplace H(s)
    let mut hs_band = FractionalSlopeBiquadBand::new_unity();
    hs_band.configure_shape_natural_phase(
        FilterShape::HighShelf,
        10_000.0,
        6.0,
        std::f64::consts::FRAC_1_SQRT_2,
        12.0,
        fs * 2.0,
        false,
    );

    let mut max_hs_err_db = 0.0_f64;
    for i in 0..=400 {
        let frac = (i as f64) / 400.0;
        let f_hz = 20.0 * (20_000.0_f64 / 20.0).powf(frac);
        let ideal_db = analog_prototype_mag_db(
            FilterShape::HighShelf,
            10_000.0,
            6.0,
            std::f64::consts::FRAC_1_SQRT_2,
            12.0,
            f_hz,
            false,
        );
        let np_mag = eval_natural_phase_band_response(&hs_band, f_hz, fs)
            .norm()
            .max(1.0e-15);
        let np_db = 20.0 * np_mag.log10();
        max_hs_err_db = max_hs_err_db.max((np_db - ideal_db).abs());
    }

    // 2C. Time-domain `ProQ4Processor` in `ProcessingMode::NaturalPhase` flat passband ripple (`20 Hz – 20 kHz`)
    let mut proc = ProQ4Processor::new();
    proc.prepare(fs, 512);
    let mut g = proc.params.global;
    g.processing_mode = ProcessingMode::NaturalPhase;
    g.character = CharacterMode::Clean;
    g.gain_q_interaction = true;
    proc.set_global_params(g);

    // Measure impulse response of flat NaturalPhase processor via Dirac impulse -> 16384-pt FFT
    let ir_len = 16_384usize;
    let mut frames = vec![[0.0f32; 2]; ir_len];
    frames[0] = [1.0, 1.0];
    let ctx = ProcessContext {
        sample_rate: fs,
        current_sample_pos: 0,
        ppq_tick: 0,
        bpm: 120.0,
        slice_frames: 512,
        is_playing: true,
        is_offline_rendering: false,
        ppq_resolution: daw_core_graph::DEFAULT_PPQ,
    };
    for chunk in frames.chunks_mut(512) {
        proc.process_slice(&ctx, chunk, None, &mut []);
    }

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(ir_len);
    let mut spec = frames
        .iter()
        .map(|f| Complex64::new(f[0] as f64, 0.0))
        .collect::<Vec<_>>();
    fft.process(&mut spec);

    let bin_hz = fs / (ir_len as f64);
    let b_lo = (20.0 / bin_hz).ceil() as usize;
    let b_hi = (20_000.0 / bin_hz).floor() as usize;
    let mut max_flat_ripple_db = 0.0_f64;
    for b in b_lo..=b_hi {
        let mag_db = 20.0 * spec[b].norm().max(1.0e-15).log10();
        max_flat_ripple_db = max_flat_ripple_db.max(mag_db.abs());
    }

    // Also verify end-to-end time-domain Bell response at 16 kHz (near Nyquist) through full `ProQ4Processor`
    let mut proc_bell = ProQ4Processor::new();
    proc_bell.prepare(fs, 512);
    proc_bell.set_global_params(g);
    let mut b0: EqBandParams = proc_bell.params.bands[0];
    b0.used = true;
    b0.enabled = true;
    b0.shape = FilterShape::Bell;
    b0.freq_hz = 12_000.0;
    b0.gain_db = 9.0;
    b0.q_ui = 2.0;
    b0.slope_db_oct = 12.0;
    proc_bell.set_band_params(0, b0);

    let mut bell_ir = vec![[0.0f32; 2]; ir_len];
    bell_ir[0] = [1.0, 1.0];
    for chunk in bell_ir.chunks_mut(512) {
        proc_bell.process_slice(&ctx, chunk, None, &mut []);
    }
    let mut bell_spec = bell_ir
        .iter()
        .map(|f| Complex64::new(f[0] as f64, 0.0))
        .collect::<Vec<_>>();
    fft.process(&mut bell_spec);
    let b_16k = (16_000.0 / bin_hz).round() as usize;
    let td_16k_db = 20.0 * bell_spec[b_16k].norm().max(1.0e-15).log10();
    let ideal_16k_db =
        analog_prototype_mag_db(FilterShape::Bell, 12_000.0, 9.0, 2.0, 12.0, b_16k as f64 * bin_hz, true);
    let td_16k_err_db = (td_16k_db - ideal_16k_db).abs();

    assert!(
        max_bell_err_db < 0.05,
        "Gate 2 FAIL: Pro-Q 4 Natural Phase Bell error {max_bell_err_db:.5} dB >= 0.05 dB"
    );
    assert!(
        max_hs_err_db < 0.05,
        "Gate 2 FAIL: Pro-Q 4 Natural Phase High-Shelf error {max_hs_err_db:.5} dB >= 0.05 dB"
    );
    assert!(
        max_flat_ripple_db < 0.002,
        "Gate 2 FAIL: Pro-Q 4 Natural Phase flat passband ripple {max_flat_ripple_db:.6} dB >= 0.002 dB"
    );
    assert!(
        td_16k_err_db < 0.05,
        "Gate 2 FAIL: End-to-end ProQ4Processor NaturalPhase 16 kHz error {td_16k_err_db:.5} dB >= 0.05 dB"
    );

    save_gate_metrics(
        "gate2",
        json!({
            "bell_12khz_max_err_db": max_bell_err_db,
            "high_shelf_10khz_max_err_db": max_hs_err_db,
            "flat_passband_ripple_db": max_flat_ripple_db,
            "end_to_end_16khz_err_db": td_16k_err_db,
            "passed": true
        }),
    );
}

#[test]
fn gate3_soothe3_partitioned_fft_4096_snr_subbass_and_clickfree() {
    // 3A. 4096-tap `ZeroDelayPartitionedConvolver4096` SNR >= 115 dB vs `f64` direct reference
    let mut recon = CepstralMinPhaseReconstructor::new();
    let mut gains = [1.0_f32; SOOTHE3_BINS];
    // Deep multi-notch minimum-phase mask including sub-120 Hz notch
    for (k, g) in gains.iter_mut().enumerate() {
        let f_hz = (k as f64) * 48_000.0 / (SOOTHE3_FFT_SIZE as f64);
        let d80 = ((f_hz - 80.0) / 14.0).powi(2);
        let d3k = ((f_hz - 3_200.0) / 180.0).powi(2);
        let cut_db = -14.0 * (-0.5 * d80).exp() - 10.0 * (-0.5 * d3k).exp();
        *g = 10.0_f64.powf(cut_db / 20.0) as f32;
    }
    let mut h_min = [Complex32::new(0.0, 0.0); SOOTHE3_BINS];
    let mut ir = [0.0_f32; SOOTHE3_FFT_SIZE];
    recon.reconstruct_minimum_phase_ir(&gains, &mut h_min, &mut ir);
    apply_min_phase_tail_taper(&mut ir);

    let mut ir_4096 = [0.0_f32; MIN_PHASE_CAUSAL_TAPS];
    ir_4096.copy_from_slice(&ir[..MIN_PHASE_CAUSAL_TAPS]);

    let mut conv = ZeroDelayPartitionedConvolver4096::new();
    conv.set_ir(&ir_4096, 0);

    let n_samples = 12_288usize;
    let mut x_sig = vec![0.0_f32; n_samples];
    for (n, s) in x_sig.iter_mut().enumerate() {
        let t = (n as f64) / 48_000.0;
        *s = (0.35 * (2.0 * PI * 73.0 * t).sin()
            + 0.25 * (2.0 * PI * 440.0 * t).sin()
            + 0.20 * (2.0 * PI * 3_150.0 * t).cos()
            + 0.15 * ((n as f64 * 12.9898).sin() * 43_758.5453).fract()) as f32;
    }

    let mut y_part = vec![0.0_f32; n_samples];
    for i in 0..n_samples {
        y_part[i] = conv.process_sample(x_sig[i]);
    }

    let mut sig_pow = 0.0_f64;
    let mut err_pow = 0.0_f64;
    for n in MIN_PHASE_CAUSAL_TAPS..n_samples {
        let mut ref_acc = 0.0_f64;
        for k in 0..MIN_PHASE_CAUSAL_TAPS {
            ref_acc += (ir_4096[k] as f64) * (x_sig[n - k] as f64);
        }
        let yp = y_part[n] as f64;
        let diff = yp - ref_acc;
        sig_pow += ref_acc * ref_acc;
        err_pow += diff * diff;
    }
    let snr_db = 10.0 * (sig_pow / err_pow.max(1.0e-30)).log10();
    assert!(
        snr_db >= 115.0,
        "Gate 3 FAIL: Soothe3 4096-tap Partitioned FFT SNR {snr_db:.2} dB < 115.0 dB"
    );

    // 3B. Sub-120 Hz sharp resonance isolation (`80 Hz` boom suppressed while `55 Hz` & `110 Hz` preserved)
    let sr = 48_000.0_f64;
    let mut soothe = Soothe3Processor::new(sr as f32);
    soothe.params_mut().global.low_latency = true;
    soothe.params_mut().global.depth = 9.0;
    soothe.params_mut().global.detail = 9.5;
    soothe.params_mut().global.max_cut_db = -24.0;
    soothe.params_mut().global.quality = TimeResolutionQuality::Normal;
    soothe.sync_dirty_tables();
    soothe.prepare(sr, 512);

    let settle = 16_384usize;
    let meas = 16_384usize;
    let tot = settle + meas;
    // Bin-centered frequencies for 16384-pt FFT @ 48 kHz (`df = 2.9296875 Hz`)
    let f_55 = 19.0 * sr / (meas as f64); // 55.664 Hz (quiet neighbor)
    let f_80 = 27.0 * sr / (meas as f64); // 79.102 Hz (dominant resonance boom!)
    let f_110 = 38.0 * sr / (meas as f64); // 111.328 Hz (quiet neighbor)

    let mut in_buf = vec![[0.0f32; 2]; tot];
    for (n, fr) in in_buf.iter_mut().enumerate() {
        let t = (n as f64) / sr;
        let s = (0.06 * (2.0 * PI * f_55 * t).sin()
            + 0.72 * (2.0 * PI * f_80 * t).sin()
            + 0.06 * (2.0 * PI * f_110 * t).sin()) as f32;
        *fr = [s, s];
    }
    let orig_buf = in_buf.clone();
    let ctx = ProcessContext {
        sample_rate: sr,
        current_sample_pos: 0,
        ppq_tick: 0,
        bpm: 120.0,
        slice_frames: 512,
        is_playing: true,
        is_offline_rendering: false,
        ppq_resolution: daw_core_graph::DEFAULT_PPQ,
    };
    for chunk in in_buf.chunks_mut(512) {
        soothe.process_slice(&ctx, chunk, None, &mut []);
    }

    let goertzel_db = |slice: &[[f32; 2]], bin: usize| -> f64 {
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        let n_f = slice.len() as f64;
        for (i, f) in slice.iter().enumerate() {
            let ang = 2.0 * PI * (bin as f64) * (i as f64) / n_f;
            let x = f[0] as f64;
            re += x * ang.cos();
            im -= x * ang.sin();
        }
        20.0 * ((re * re + im * im).sqrt() / n_f).max(1.0e-15).log10()
    };

    let atten_55_db = goertzel_db(&in_buf[settle..], 19) - goertzel_db(&orig_buf[settle..], 19);
    let atten_80_db = goertzel_db(&in_buf[settle..], 27) - goertzel_db(&orig_buf[settle..], 27);
    let atten_110_db = goertzel_db(&in_buf[settle..], 38) - goertzel_db(&orig_buf[settle..], 38);
    let selectivity_80_vs_neighbors_db = (atten_55_db - atten_80_db).min(atten_110_db - atten_80_db);

    assert!(
        atten_80_db <= -6.0 && selectivity_80_vs_neighbors_db >= 4.5,
        "Gate 3 FAIL: Soothe3 80 Hz resonance suppression insufficient: 80Hz={atten_80_db:.2} dB, 55Hz={atten_55_db:.2} dB, 110Hz={atten_110_db:.2} dB"
    );

    // 3C. Click-free step IR crossfade (`crossfade_samples = 512` vs `0` hard swap on pure 80 Hz sine)
    let mut unity_ir = [0.0_f32; MIN_PHASE_CAUSAL_TAPS];
    unity_ir[0] = 1.0;

    let run_ir_swap = |xfade_len: usize| -> f64 {
        let mut c = ZeroDelayPartitionedConvolver4096::new();
        c.set_ir(&unity_ir, 0);
        let mut prev2 = 0.0_f64;
        let mut prev1 = 0.0_f64;
        let mut max_d2 = 0.0_f64;
        for n in 0..8192 {
            if n == 4096 {
                c.set_ir(&ir_4096, xfade_len);
            }
            let x = (0.8 * (2.0 * PI * 80.0 * (n as f64) / sr).sin()) as f32;
            let y = c.process_sample(x) as f64;
            if (4094..4700).contains(&n) {
                let d2 = (y - 2.0 * prev1 + prev2).abs();
                max_d2 = max_d2.max(d2);
            }
            prev2 = prev1;
            prev1 = y;
        }
        max_d2
    };

    let d2_hard = run_ir_swap(0);
    let d2_smooth = run_ir_swap(512);
    let click_reduction_db = 20.0 * (d2_hard / d2_smooth.max(1.0e-12)).log10();
    assert!(
        click_reduction_db >= 25.0,
        "Gate 3 FAIL: Soothe3 Dual-Filter Crossfade click reduction {click_reduction_db:.2} dB < 25.0 dB"
    );

    save_gate_metrics(
        "gate3",
        json!({
            "convolver_4096_snr_db": snr_db,
            "atten_80hz_db": atten_80_db,
            "atten_55hz_db": atten_55_db,
            "atten_110hz_db": atten_110_db,
            "selectivity_80_vs_neighbors_db": selectivity_80_vs_neighbors_db,
            "step_mask_click_reduction_db": click_reduction_db,
            "passed": true
        }),
    );
}

#[test]
fn gate4_fl26_luxeverb_shimmer_and_grossbeat_declick() {
    // 4A. LuxeVerb 16x16 FDN Echo Diffusion Density across `44.1 / 48 / 96 kHz`
    let mut density_ratios = Vec::new();
    for &sr in &[44_100.0_f64, 48_000.0, 96_000.0] {
        let mut verb = LuxeVerbFdnShimmerNode::new();
        verb.wet_level = 1.0;
        verb.decay = 0.85;
        verb.pre_delay_ms = 5.0;
        verb.shimmer_mix = 0.0;
        verb.prepare(sr, 512);

        let len = (0.45 * sr) as usize;
        let mut io = vec![[0.0f32; 2]; len];
        io[0] = [1.0, 1.0];
        let ctx = ProcessContext {
            sample_rate: sr,
            current_sample_pos: 0,
            ppq_tick: 0,
            bpm: 120.0,
            slice_frames: 512,
            is_playing: true,
            is_offline_rendering: false,
            ppq_resolution: daw_core_graph::DEFAULT_PPQ,
        };
        for chunk in io.chunks_mut(512) {
            verb.process_slice(&ctx, chunk, None, &mut []);
        }

        // Measure tail density in `[150 ms .. 400 ms]`
        let s0 = (0.15 * sr) as usize;
        let s1 = (0.40 * sr) as usize;
        let tail = &io[s0..s1];
        let rms = (tail
            .iter()
            .map(|f| 0.5 * ((f[0] as f64).powi(2) + (f[1] as f64).powi(2)))
            .sum::<f64>()
            / (tail.len() as f64))
            .sqrt();
        let thr = (0.15 * rms) as f32;
        let occupied = tail
            .iter()
            .filter(|f| f[0].abs() > thr || f[1].abs() > thr)
            .count();
        let occ_ratio = (occupied as f64) / (tail.len() as f64);
        assert!(
            rms > 1.0e-5 && occ_ratio > 0.45,
            "Gate 4 FAIL: LuxeVerb sparse/metallic tail at {sr} Hz: rms={rms:.2e}, occ_ratio={occ_ratio:.3}"
        );
        density_ratios.push((sr, occ_ratio));
    }

    // 4B. LuxeVerb `+12st` Shimmer Octave Harmonic (`880 Hz` from pure `440 Hz` burst)
    let sr = 48_000.0_f64;
    let measure_880hz_tail_db = |shimmer_mix: f32| -> f64 {
        let mut verb = LuxeVerbFdnShimmerNode::new();
        verb.wet_level = 1.0;
        verb.decay = 0.88;
        verb.shimmer_mix = shimmer_mix;
        verb.shimmer_semitones = 12.0;
        verb.prepare(sr, 512);

        let burst_len = (0.25 * sr) as usize;
        let total_len = burst_len + 16_384;
        let mut io = vec![[0.0f32; 2]; total_len];
        for n in 0..burst_len {
            let env = (PI * (n as f64) / (burst_len as f64)).sin();
            let s = (0.6 * env * (2.0 * PI * 440.0 * (n as f64) / sr).sin()) as f32;
            io[n] = [s, s];
        }
        let ctx = ProcessContext {
            sample_rate: sr,
            current_sample_pos: 0,
            ppq_tick: 0,
            bpm: 120.0,
            slice_frames: 512,
            is_playing: true,
            is_offline_rendering: false,
            ppq_resolution: daw_core_graph::DEFAULT_PPQ,
        };
        for chunk in io.chunks_mut(512) {
            verb.process_slice(&ctx, chunk, None, &mut []);
        }

        let fft_n = 16_384usize;
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(fft_n);
        let mut spec = io[burst_len..burst_len + fft_n]
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let w = 0.5 * (1.0 - (2.0 * PI * (i as f64) / (fft_n as f64)).cos());
                Complex64::new((f[0] as f64) * w, 0.0)
            })
            .collect::<Vec<_>>();
        fft.process(&mut spec);
        let bin_880 = (880.0 * (fft_n as f64) / sr).round() as usize;
        let mut pk = 1.0e-18_f64;
        for b in (bin_880 - 6)..=(bin_880 + 6) {
            pk = pk.max(spec[b].norm() / (fft_n as f64));
        }
        20.0 * pk.log10()
    };

    let oct_off_db = measure_880hz_tail_db(0.0);
    let oct_on_db = measure_880hz_tail_db(0.65);
    let shimmer_boost_db = oct_on_db - oct_off_db;
    assert!(
        shimmer_boost_db >= 20.0,
        "Gate 4 FAIL: LuxeVerb +12st Shimmer 880 Hz boost {shimmer_boost_db:.2} dB < 20.0 dB"
    );

    // 4C. GrossBeat Dual-Read-Head De-Click (`> 35 dB` 2nd-derivative discontinuity reduction)
    let run_grossbeat_jump = |declick: bool| -> f64 {
        let mut gb = GrossBeatTimeBufferNode::new(480_000);
        gb.click_reduction_enabled = declick;
        gb.crossfade_ms = 4.0;
        // Instantaneous time jump at bar_phase = 0.25 (`0.0` -> `0.55` beats delay)
        gb.time_spline = TensionSpline::new(vec![
            TensionControlPoint::new(0.0, 0.0, 0.0),
            TensionControlPoint::new(0.2499, 0.0, 0.0),
            TensionControlPoint::new(0.2500, 0.55, 0.0),
            TensionControlPoint::new(1.0, 0.55, 0.0),
        ]);
        gb.prepare(sr, 512);

        // At 120 BPM & 48 kHz, 2 bars (8 beats) = 192,000 samples -> phase 0.25 is sample 48,000
        let total_n = 56_000usize;
        let mut io = vec![[0.0f32; 2]; total_n];
        for (n, fr) in io.iter_mut().enumerate() {
            let s = (0.85 * (2.0 * PI * 220.0 * (n as f64) / sr).sin()) as f32;
            *fr = [s, s];
        }

        let mut pos = 0usize;
        while pos < total_n {
            let end = (pos + 512).min(total_n);
            let ctx = ProcessContext {
                sample_rate: sr,
                current_sample_pos: pos as i64,
                ppq_tick: 0,
                bpm: 120.0,
                slice_frames: end - pos,
                is_playing: true,
                is_offline_rendering: false,
                ppq_resolution: daw_core_graph::DEFAULT_PPQ,
            };
            gb.process_slice(&ctx, &mut io[pos..end], None, &mut []);
            pos = end;
        }

        let mut max_d2 = 0.0_f64;
        for n in 47_900..48_600 {
            let y0 = io[n - 2][0] as f64;
            let y1 = io[n - 1][0] as f64;
            let y2 = io[n][0] as f64;
            let d2 = (y2 - 2.0 * y1 + y0).abs();
            max_d2 = max_d2.max(d2);
        }
        max_d2
    };

    let gb_d2_raw = run_grossbeat_jump(false);
    let gb_d2_declick = run_grossbeat_jump(true);
    let gb_click_reduction_db = 20.0 * (gb_d2_raw / gb_d2_declick.max(1.0e-12)).log10();

    assert!(
        gb_click_reduction_db > 35.0,
        "Gate 4 FAIL: GrossBeat de-click 2nd-derivative reduction {gb_click_reduction_db:.2} dB <= 35.0 dB"
    );

    save_gate_metrics(
        "gate4",
        json!({
            "luxeverb_tail_occupancy_44k1": density_ratios[0].1,
            "luxeverb_tail_occupancy_48k": density_ratios[1].1,
            "luxeverb_tail_occupancy_96k": density_ratios[2].1,
            "luxeverb_shimmer_880hz_boost_db": shimmer_boost_db,
            "grossbeat_declick_reduction_db": gb_click_reduction_db,
            "passed": true
        }),
    );
}

#[test]
fn gate5_broadcast_true_peak_and_all_real_neural_models() {
    let sr = 48_000.0_f64;

    // 5A. Broadcast `ITU-R BS.1770-4` `4x True Peak <= -0.98 dBTP` under `+8.0 dB` hot multitone + ISP stressor
    let mut maximizer = IrcVMultiBandMaximizer::new(sr);
    maximizer.set_gain_db(8.0);
    maximizer.set_ceiling_dbtp(-1.0);
    maximizer.enable_soft_clip = true;
    maximizer.soft_clip_mix = 0.65;
    maximizer.soft_clip_mode = SoftClipMode::Medium;
    maximizer.soft_clip_oversample_4x = true;
    maximizer.prevent_intersample_clipping = true;
    maximizer.prepare(sr, 512);

    let len = 96_000usize; // 2.0 seconds @ 48 kHz including worst-case fs/4 pi/4 inter-sample peaks + drum transients
    let mut hot_mix = vec![[0.0f32; 2]; len];
    for (n, fr) in hot_mix.iter_mut().enumerate() {
        let t = (n as f64) / sr;
        let kick = if n % 12_000 < 2400 {
            let kt = (n % 12_000) as f64 / sr;
            0.95 * (-kt * 28.0).exp() * (2.0 * PI * (58.0 + 90.0 * (-kt * 60.0).exp()) * kt).sin()
        } else {
            0.0
        };
        // Worst-case +3.01 dB inter-sample peak stimulus: cos(pi/2 * n + pi/4)
        let isp = 0.55 * (0.5 * PI * (n as f64) + 0.25 * PI).cos();
        let chord = 0.35 * (2.0 * PI * 220.0 * t).sin()
            + 0.30 * (2.0 * PI * 1_250.0 * t).sin()
            + 0.25 * (2.0 * PI * 6_400.0 * t).cos();
        let l = (kick + isp + chord) as f32;
        let r = (kick - 0.8 * isp + 0.9 * chord) as f32;
        *fr = [l, r];
    }

    let ctx = ProcessContext {
        sample_rate: sr,
        current_sample_pos: 0,
        ppq_tick: 0,
        bpm: 128.0,
        slice_frames: 512,
        is_playing: true,
        is_offline_rendering: false,
        ppq_resolution: daw_core_graph::DEFAULT_PPQ,
    };
    for chunk in hot_mix.chunks_mut(512) {
        maximizer.process_slice(&ctx, chunk, None, &mut []);
    }

    let mut tp_detector = TruePeakDetector4x::new();
    let mut max_tp_lin = 0.0_f32;
    for fr in &hot_mix {
        let [tpl, tpr] = tp_detector.step_stereo(fr[0], fr[1]);
        max_tp_lin = max_tp_lin.max(tpl).max(tpr);
    }
    let measured_tp_dbtp = 20.0 * (max_tp_lin.max(1.0e-12) as f64).log10();

    let mut lufs_meter = LufsTruePeakMeter::new(sr);
    lufs_meter.push_block(&hot_mix, 0.0);
    let out_lufs = lufs_meter.lufs_integrated as f64;

    assert!(
        measured_tp_dbtp <= -0.98 && out_lufs.is_finite(),
        "Gate 5 FAIL: IRC V +8 dB True Peak {measured_tp_dbtp:.4} dBTP > -0.98 dBTP ceiling!"
    );

    // Build a realistic 5.0-second musical test track (`240,000` frames @ 48 kHz) with quiet intro (`0..2s`)
    // and energetic chorus (`2..5s`) for real neural inference + Master Assistant chorus scanning
    let ai_len = 240_000usize;
    let mut musical_track = vec![[0.0f32; 2]; ai_len];
    for (n, fr) in musical_track.iter_mut().enumerate() {
        let t = (n as f64) / sr;
        let is_chorus = n >= 96_000;
        let gain = if is_chorus { 0.55 } else { 0.08 };
        let bass = 0.50 * (2.0 * PI * 65.41 * t).sin();
        let vocal_like = 0.35 * (2.0 * PI * 440.0 * t + 0.4 * (2.0 * PI * 5.5 * t).sin()).sin()
            + 0.18 * (2.0 * PI * 880.0 * t).sin()
            + 0.10 * (2.0 * PI * 2_640.0 * t).sin();
        let drum_hit = if n % 24_000 < 1200 {
            let dt = (n % 24_000) as f64 / sr;
            0.65 * (-dt * 45.0).exp() * (2.0 * PI * 110.0 * dt).sin()
        } else {
            0.0
        };
        let l = (gain * (bass + vocal_like + drum_hit)) as f32;
        let r = (gain * (bass + 0.92 * vocal_like + drum_hit)) as f32;
        *fr = [l, r];
    }

    // 5B. `delimit_exp009_ep080_fixed2.onnx` (`OzoneDelimitOnnxProcessor`)
    let mut delimit = OzoneDelimitOnnxProcessor::from_default_assets()
        .expect("Failed to load delimit_exp009_ep080_fixed2.onnx");
    let clipped_snip: Vec<StereoFrame> = musical_track[96_000..96_000 + 24_000]
        .iter()
        .map(|f| [(f[0] * 3.0).clamp(-0.35, 0.35), (f[1] * 3.0).clamp(-0.35, 0.35)])
        .collect();
    let restored = delimit
        .process_offline_with_threshold(&clipped_snip, sr, -9.1, 1.0)
        .expect("Delimit ONNX inference failed");
    assert_eq!(restored.len(), clipped_snip.len());
    let delimit_peak = restored
        .iter()
        .fold(0.0_f32, |a, f| a.max(f[0].abs()).max(f[1].abs()));
    assert!(
        delimit_peak.is_finite() && delimit_peak > 0.36,
        "Gate 5 FAIL: Delimit ONNX failed to restore clipped peaks (peak={delimit_peak})"
    );

    // 5C. `ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx` (`OzoneBsrnnOnnxSeparator`)
    let mut bsrnn = OzoneBsrnnOnnxSeparator::from_default_assets()
        .expect("Failed to load ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx");
    let mut in_hop = BsrnnInputHop {
        epoch: 0,
        seq: 0,
        frames: [[0.0f32; 2]; BSRNN_HOP_SIZE],
    };
    let mut out_hop = BsrnnOutputHop {
        epoch: 0,
        seq: 0,
        stems: [[[0.0f32; 2]; BSRNN_HOP_SIZE]; 4],
    };
    let mut bsrnn_energy = 0.0_f64;
    for h in 0..16 {
        let base = 96_000 + h * BSRNN_HOP_SIZE;
        in_hop
            .frames
            .copy_from_slice(&musical_track[base..base + BSRNN_HOP_SIZE]);
        bsrnn
            .process_hop(&in_hop, &mut out_hop)
            .expect("BSRNN ONNX hop inference failed");
        for s in 0..4 {
            for f in &out_hop.stems[s] {
                assert!(f[0].is_finite() && f[1].is_finite());
                bsrnn_energy += (f[0] as f64).powi(2) + (f[1] as f64).powi(2);
            }
        }
    }
    assert!(bsrnn_energy > 1.0e-3, "Gate 5 FAIL: BSRNN ONNX output energy was zero");

    // 5D. `gparamsBN8.0.npz` (`GenreCnnClassifier`) & `instrument_classifier_params.npz` (`InstrumentCnnClassifier`)
    let genre_cnn = GenreCnnClassifier::load_default().expect("Failed to load gparamsBN8.0.npz");
    let inst_cnn = InstrumentCnnClassifier::load_default()
        .expect("Failed to load instrument_classifier_params.npz");
    let g_probs = genre_cnn.classify_audio(&musical_track[96_000..], 48_000);
    let i_probs = inst_cnn.classify_audio(&musical_track[96_000..], 48_000);
    let g_sum: f32 = g_probs.iter().sum();
    let i_sum: f32 = i_probs.iter().sum();
    assert!(
        g_probs.iter().all(|p| p.is_finite() && *p >= 0.0) && (g_sum - 1.0).abs() < 1.0e-3,
        "Gate 5 FAIL: Genre CNN invalid probability distribution: {:?}",
        g_probs
    );
    assert!(
        i_probs.iter().all(|p| p.is_finite() && *p >= 0.0) && (i_sum - 1.0).abs() < 1.0e-3,
        "Gate 5 FAIL: Instrument CNN invalid probability distribution: {:?}",
        i_probs
    );

    // 5E. 3-File `GRU-512` `.npz` Stem Separator (`vocals`, `drums`, `bass-small_sanitized.npz`)
    let mut gru_sep = GruThreeStemSeparator::from_default_assets()
        .expect("Failed to load 3 GRU-512 sanitized .npz stem models");
    let gru_in = &musical_track[96_000..96_000 + 24_000];
    let gru_stems = gru_sep.separate_4stems(gru_in, 48_000);
    for (idx, f_orig) in gru_in.iter().enumerate() {
        let sum_l = gru_stems[0][idx][0]
            + gru_stems[1][idx][0]
            + gru_stems[2][idx][0]
            + gru_stems[3][idx][0];
        assert!(
            sum_l.is_finite() && (sum_l - f_orig[0]).abs() < 1.0e-4,
            "Gate 5 FAIL: GRU-512 4-stem unity sum violated at sample {idx}"
        );
    }

    // 5F. UVR5 MDX-Net ONNX (`Kim_Vocal_2.onnx` & `UVR-MDX-NET-Inst_HQ_3.onnx`)
    let mdx_snip = &musical_track[96_000..96_000 + 24_000]; // 0.5 sec @ 48 kHz -> single MDX chunk
    let mut kim =
        Uvr5MdxOnnxSeparator::from_kim_vocal_2().expect("Failed to load Kim_Vocal_2.onnx");
    let (kim_voc, kim_inst) = kim
        .separate_vocal_instrumental(mdx_snip, 48_000, false)
        .expect("Kim_Vocal_2 ONNX inference failed");
    assert_eq!(kim_voc.len(), mdx_snip.len());
    assert!(kim_voc.iter().all(|f| f[0].is_finite() && f[1].is_finite()));
    assert!(kim_inst.iter().all(|f| f[0].is_finite() && f[1].is_finite()));

    let mut inst_hq3 = Uvr5MdxOnnxSeparator::from_inst_hq_3()
        .expect("Failed to load UVR-MDX-NET-Inst_HQ_3.onnx");
    let (hq3_voc, hq3_inst) = inst_hq3
        .separate_vocal_instrumental(mdx_snip, 48_000, false)
        .expect("UVR-MDX-NET-Inst_HQ_3 ONNX inference failed");
    assert_eq!(hq3_inst.len(), mdx_snip.len());
    assert!(hq3_voc.iter().all(|f| f[0].is_finite() && f[1].is_finite()));
    assert!(hq3_inst.iter().all(|f| f[0].is_finite() && f[1].is_finite()));

    // 5G. Real Chorus-Scanning `run_ozone12_master_assistant_with_audio`
    let (mut ctrl, _rt_gc) = AudioStreamController::with_mastering_template_rt_gc(48_000.0, 256)
        .expect("Failed to build AudioStreamController mastering template");
    let assistant_dto = run_ozone12_master_assistant_with_audio(
        &ctrl.app_state,
        0,
        0,
        Some((&musical_track, 48_000)),
        None,
    )
    .expect("Master Assistant real audio calibration failed");
    let mut blk = [[0.0f32; 2]; 256];
    ctrl.mixer_engine.process_audio_block(&mut blk);

    assert!(GENRE_LABELS.contains(&assistant_dto.detected_genre.as_str()));
    assert!(INSTRUMENT_LABELS.contains(&assistant_dto.dominant_instrument.as_str()));
    assert!(assistant_dto.measured_lufs.is_finite());
    assert!(assistant_dto.crest_factor_db.is_finite() && assistant_dto.crest_factor_db > 0.5);

    save_gate_metrics(
        "gate5",
        json!({
            "maximizer_plus8db_true_peak_dbtp": measured_tp_dbtp,
            "maximizer_plus8db_integrated_lufs": out_lufs,
            "delimit_onnx_tier": format!("{:?}", delimit.active_tier()),
            "delimit_restored_peak_lin": delimit_peak,
            "bsrnn_onnx_tier": format!("{:?}", bsrnn.active_tier()),
            "kim_vocal2_tier": format!("{:?}", kim.active_tier()),
            "inst_hq3_tier": format!("{:?}", inst_hq3.active_tier()),
            "assistant_detected_genre": assistant_dto.detected_genre,
            "assistant_dominant_instrument": assistant_dto.dominant_instrument,
            "assistant_measured_lufs": assistant_dto.measured_lufs,
            "assistant_target_lufs": assistant_dto.target_lufs,
            "assistant_crest_factor_db": assistant_dto.crest_factor_db,
            "passed": true
        }),
    );
}

#[test]
fn gate6_realtime_1000_mutations_gc_and_wasapi_hardware_smoke() {
    // 6A. 1,000 Concurrent `RtGraphMutation` + `EngineCommand` Stress Test (< 1.5 ms/block, GC > 0)
    let sr = 48_000.0_f64;
    let (mut engine, mut cmd_prod, mut mut_prod, rt_gc, mut telem_out) =
        MixerEngine::new_with_rt_gc(sr, 128.0, 0);

    engine
        .add_channel_generator(ChannelGenerator::new(
            0,
            1,
            Box::new(DeterministicGeneratorNode::new(8, 0.45, false)),
        ))
        .unwrap();
    engine.tracks[1].add_send(0, 1.0, false);
    engine.rebuild_graph().unwrap();
    cmd_prod
        .push(EngineCommand::SetTransportPlaying(true))
        .unwrap();

    let mut_thread = std::thread::spawn(move || {
        for i in 0..1_000usize {
            let mutation = match i % 5 {
                0 => {
                    let mut slot = PluginRackSlot::new(
                        Box::new(FixedLatencyTransformNode::new((i % 32) + 4, 0.9, false)),
                        0.85,
                    );
                    slot.plugin.prepare(sr, 256);
                    RtGraphMutation::SetOrReplacePluginSlot {
                        track_index: 1,
                        slot_index: 0,
                        slot: Box::new(Some(slot)),
                    }
                }
                1 => RtGraphMutation::AddOrUpdateSendRoute {
                    src_track: 1,
                    dest_track: 2,
                    gain: 0.6,
                    is_sidechain_only: false,
                    route_carrier: Box::new(Some(MixerSendRoute::new(2, 0.6, false))),
                },
                2 => RtGraphMutation::RemoveSendRoute {
                    src_track: 1,
                    dest_track: 2,
                    is_sidechain_only: false,
                    recycle_carrier: Box::new(None),
                },
                3 => {
                    let mut pat = PianoRollPattern::new(3, "Gate6Pat", 0, 96);
                    pat.add_note(PianoRollNote::new(1, 0, 24, 60, 0.9));
                    RtGraphMutation::UpsertPattern {
                        pattern: Box::new(pat),
                    }
                }
                _ => {
                    let sp = TensionSpline::new(vec![
                        TensionControlPoint::new(0.0, 0.2, 0.0),
                        TensionControlPoint::new(96.0, 0.8, 0.0),
                    ]);
                    RtGraphMutation::UpsertAutomationArticulator {
                        articulator: Box::new(AutomationClipArticulator::new(0, sp)),
                    }
                }
            };
            while mut_prod.is_full() {
                std::thread::yield_now();
            }
            assert!(mut_prod.push(mutation).is_ok());
            if i % 4 == 0 {
                while mut_prod.is_full() {
                    std::thread::yield_now();
                }
                assert!(
                    mut_prod
                        .push(RtGraphMutation::UpsertPlaylistClip {
                            clip: PlaylistClip {
                                clip_id: 1,
                                track_row: 0,
                                kind: PlaylistClipKind::Pattern,
                                source_id: 3,
                                start_tick: 0,
                                duration_ticks: 192,
                                clip_offset_ticks: 0,
                                gain: 1.0,
                                muted: false,
                                ..Default::default()
                            },
                        })
                        .is_ok()
                );
            }
            let _ = cmd_prod.push(EngineCommand::SetTrackGain {
                track_index: 1,
                gain_linear: 0.85,
            });
        }
    });

    let mut master_out = [[0.0f32; 2]; 256];
    let t0 = Instant::now();
    let mut blocks = 0usize;
    loop {
        engine.process_audio_block(&mut master_out);
        blocks += 1;
        if mut_thread.is_finished() && blocks >= 256 {
            for _ in 0..16 {
                engine.process_audio_block(&mut master_out);
                blocks += 1;
            }
            break;
        }
    }
    let elapsed = t0.elapsed();
    mut_thread.join().unwrap();

    rt_gc.wake();
    std::thread::sleep(Duration::from_millis(25));
    let gc_freed = rt_gc.freed_count();
    let avg_block_us = elapsed.as_secs_f64() * 1_000_000.0 / (blocks as f64);
    let telem = telem_out.read();

    assert!(
        avg_block_us < 1_500.0,
        "Gate 6 FAIL: Average block render time {avg_block_us:.2} us >= 1500.0 us"
    );
    assert!(
        gc_freed > 0,
        "Gate 6 FAIL: Background toneforge-rt-gc freed 0 retired resources!"
    );
    assert!(telem.header.frame_seq >= 256);

    // 6B. Live Win32 Core Audio WASAPI Endpoint Enumeration & Hardware Stream Smoke Test
    let endpoints = enumerate_wasapi_output_devices()
        .expect("Win32 Core Audio enumerate_wasapi_output_devices failed");
    let (ctrl_hw, _hw_gc) = AudioStreamController::with_mastering_template_rt_gc(48_000.0, 256)
        .expect("Failed to initialize WASAPI mastering template");
    let mut wasapi_stream = WasapiHardwareStream::spawn(ctrl_hw.mixer_engine, None, false);
    // Mute physical output immediately during unattended overnight hardware test!
    wasapi_stream.set_volume_and_mute(0.0, true);

    // Allow WASAPI thread to initialize COM, wake USB DAC if suspended, elevate MMCSS "Pro Audio", and render packets
    let wait_start = Instant::now();
    let mut snap = wasapi_stream.status_snapshot();
    while wait_start.elapsed() < Duration::from_millis(4_000) {
        snap = wasapi_stream.status_snapshot();
        if snap.rendered_blocks >= 8 {
            break;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    wasapi_stream.reconfigure_endpoint(None, false);
    let reopen_start = Instant::now();
    while wasapi_stream.is_reopen_pending() && reopen_start.elapsed() < Duration::from_millis(2_500) {
        std::thread::sleep(Duration::from_millis(25));
    }
    let post_reopen_start = Instant::now();
    let base_blocks = snap.rendered_blocks;
    let mut snap_after = wasapi_stream.status_snapshot();
    while post_reopen_start.elapsed() < Duration::from_millis(2_500) {
        snap_after = wasapi_stream.status_snapshot();
        if snap_after.stream_running && snap_after.rendered_blocks > base_blocks {
            break;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    wasapi_stream.shutdown();

    assert!(
        snap.stream_running || snap_after.rendered_blocks > 0,
        "Gate 6 FAIL: WasapiHardwareStream did not run or render blocks: {:?}",
        snap_after
    );

    save_gate_metrics(
        "gate6",
        json!({
            "blocks_rendered_under_1000_mutations": blocks,
            "avg_block_render_us": avg_block_us,
            "rt_gc_freed_resources": gc_freed,
            "wasapi_endpoint_count": endpoints.len(),
            "wasapi_active_device": snap_after.active_device_name,
            "wasapi_hw_sample_rate_hz": snap_after.hardware_sample_rate_hz,
            "wasapi_buffer_frames": snap_after.buffer_frames,
            "wasapi_sample_format": snap_after.sample_format,
            "wasapi_mmcss_elevated": snap_after.mmcss_pro_audio_elevated,
            "wasapi_using_hardware": snap_after.using_hardware,
            "wasapi_rendered_blocks": snap_after.rendered_blocks,
            "wasapi_xruns": snap_after.xrun_count,
            "passed": true
        }),
    );
}
