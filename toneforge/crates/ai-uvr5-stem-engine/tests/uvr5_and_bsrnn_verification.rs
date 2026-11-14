//! Comprehensive Unit, Integration & Zero-Allocation Verification Suite for `ai-uvr5-stem-engine`
//! and Full-Chain `Pro-Q 4 + Soothe3 + Ozone 12 + RealTimeBsrnnStemNode` inside `MixerEngine`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::f64::consts::PI;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use ai_uvr5_stem_engine::{
    BSRNN_LATENCY_SAMPLES_48K, CascadedASPPNet, GenreCnnClassifier, HTDemucs4, Mdx23cSeparator,
    OnnxModelDescriptor, RealTimeBsrnnStemNode, SpectralEnsembleMode, SpectralEnsembler,
    VR_ASPP_DILATIONS, combine_complex_spectrogram_frame,
};
use daw_core_graph::{
    ChannelGenerator, DeterministicGeneratorNode, EngineCommand, MixerEngine, StereoFrame,
    UnifiedDspNode,
};
use dsp_ozone12_mastering::{ExciterMode, IrcMode, Ozone12MasterChain};
use dsp_proq4_eq::{CharacterMode, FilterShape, ProQ4Processor, ProcessingMode};
use dsp_soothe3_spectral::{
    SidechainFilterShape, Soothe3Processor, SootheProcessMode, TimeResolutionQuality,
};
use rustfft::num_complex::Complex32;

struct RtTrackingAllocator;

static GLOBAL_ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static GLOBAL_ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
static GLOBAL_DEALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static THREAD_RT_GUARD_ACTIVE: Cell<bool> = const { Cell::new(false) };
    static THREAD_ALLOC_COUNT: Cell<usize> = const { Cell::new(0) };
    static THREAD_ALLOC_BYTES: Cell<usize> = const { Cell::new(0) };
    static THREAD_DEALLOC_COUNT: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for RtTrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        THREAD_RT_GUARD_ACTIVE.with(|active| {
            if active.get() {
                THREAD_ALLOC_COUNT.with(|c| c.set(c.get() + 1));
                THREAD_ALLOC_BYTES.with(|b| b.set(b.get() + layout.size()));
                GLOBAL_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
                GLOBAL_ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            }
        });
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        THREAD_RT_GUARD_ACTIVE.with(|active| {
            if active.get() {
                THREAD_DEALLOC_COUNT.with(|c| c.set(c.get() + 1));
                GLOBAL_DEALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            }
        });
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        THREAD_RT_GUARD_ACTIVE.with(|active| {
            if active.get() {
                THREAD_ALLOC_COUNT.with(|c| c.set(c.get() + 1));
                THREAD_ALLOC_BYTES.with(|b| b.set(b.get() + new_size));
                GLOBAL_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
                GLOBAL_ALLOC_BYTES.fetch_add(new_size, Ordering::Relaxed);
            }
        });
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: RtTrackingAllocator = RtTrackingAllocator;

fn begin_rt_no_alloc_region() {
    GLOBAL_ALLOC_COUNT.store(0, Ordering::SeqCst);
    GLOBAL_ALLOC_BYTES.store(0, Ordering::SeqCst);
    GLOBAL_DEALLOC_COUNT.store(0, Ordering::SeqCst);
    THREAD_ALLOC_COUNT.with(|c| c.set(0));
    THREAD_ALLOC_BYTES.with(|b| b.set(0));
    THREAD_DEALLOC_COUNT.with(|c| c.set(0));
    THREAD_RT_GUARD_ACTIVE.with(|a| a.set(true));
}

fn end_rt_no_alloc_region() -> (usize, usize, usize) {
    THREAD_RT_GUARD_ACTIVE.with(|a| a.set(false));
    let allocs = THREAD_ALLOC_COUNT.with(|c| c.get()) + GLOBAL_ALLOC_COUNT.load(Ordering::SeqCst);
    let bytes = THREAD_ALLOC_BYTES.with(|b| b.get()) + GLOBAL_ALLOC_BYTES.load(Ordering::SeqCst);
    let deallocs =
        THREAD_DEALLOC_COUNT.with(|c| c.get()) + GLOBAL_DEALLOC_COUNT.load(Ordering::SeqCst);
    (allocs, bytes, deallocs)
}

#[test]
fn test_realtime_bsrnn_5760_latency_and_apdc_null_cancellation() {
    let sample_rate = 48_000.0;
    let bsrnn = RealTimeBsrnnStemNode::new(sample_rate);
    assert_eq!(bsrnn.latency_samples(), BSRNN_LATENCY_SAMPLES_48K);
    assert_eq!(bsrnn.latency_samples(), 5760);

    // Build a 2-track parallel phase-cancellation setup in MixerEngine:
    // - Track 1: Deterministic generator (+1.0) -> RealTimeBsrnnStemNode (5,760 samples latency @ 0.0 dB unity) -> Master
    // - Track 2: Deterministic generator (-1.0 phase-inverted) -> 0 plugin latency -> Master
    // MixerEngine's 5-Point APDC must delay Track 2 by exact 5,760 samples so Track 1 + Track 2 === 0.0!
    let (mut engine, mut cmd_prod, _telem_out) = MixerEngine::new(sample_rate, 120.0, 0);

    engine
        .add_channel_generator(ChannelGenerator::new(
            0,
            1,
            Box::new(DeterministicGeneratorNode::new(0, 0.8, false)),
        ))
        .unwrap();
    engine
        .add_channel_generator(ChannelGenerator::new(
            1,
            2,
            Box::new(DeterministicGeneratorNode::new(0, 0.8, true)),
        ))
        .unwrap();

    engine.tracks[1]
        .add_plugin_slot(Box::new(bsrnn), 1.0)
        .unwrap();
    engine.tracks[1].add_send(0, 1.0, false);
    engine.tracks[2].add_send(0, 1.0, false);
    engine.rebuild_graph().unwrap();

    assert_eq!(engine.tracks[1].rack_output_latency, 5760);
    assert_eq!(engine.tracks[2].rack_output_latency, 0);
    assert_eq!(engine.tracks[0].input_arrival_latency, 5760);

    cmd_prod
        .push(EngineCommand::SetTransportPlaying(true))
        .unwrap();

    let mut master_out = [[0.0f32; 2]; 256];
    let mut max_err_after_warmup = 0.0f32;

    // Run 40 blocks (10,240 samples > 5,760-sample APDC warmup)
    for block_idx in 0..40 {
        engine.process_audio_block(&mut master_out);
        if block_idx >= 25 {
            for frame in &master_out {
                max_err_after_warmup = max_err_after_warmup
                    .max(frame[0].abs())
                    .max(frame[1].abs());
            }
        }
    }

    assert!(
        max_err_after_warmup < 1e-6,
        "5,760-sample APDC parallel null cancellation failed: max residual = {}",
        max_err_after_warmup
    );
}

#[test]
fn test_spectral_ensembler_max_min_avg_median_weighted_and_invert() {
    let m1 = [
        Complex32::new(3.0, 4.0), // |z| = 5.0
        Complex32::new(1.0, 0.0), // |z| = 1.0
        Complex32::new(0.0, 2.0), // |z| = 2.0
    ];
    let m2 = [
        Complex32::new(1.0, 1.0), // |z| = sqrt(2)
        Complex32::new(0.0, 5.0), // |z| = 5.0
        Complex32::new(0.0, 6.0), // |z| = 6.0
    ];
    let m3 = [
        Complex32::new(2.0, 0.0), // |z| = 2.0
        Complex32::new(0.0, 3.0), // |z| = 3.0
        Complex32::new(0.0, 4.0), // |z| = 4.0
    ];
    let frames: [&[Complex32]; 3] = [&m1, &m2, &m3];
    let mut out = [Complex32::new(0.0, 0.0); 3];

    // 1. MAX_SPEC preserves complex value & phase of maximum-magnitude bin
    combine_complex_spectrogram_frame(SpectralEnsembleMode::MaxSpec, &frames, &mut out);
    assert_eq!(out[0], Complex32::new(3.0, 4.0));
    assert_eq!(out[1], Complex32::new(0.0, 5.0));
    assert_eq!(out[2], Complex32::new(0.0, 6.0));

    // 2. MIN_SPEC preserves complex value & phase of minimum-magnitude bin
    combine_complex_spectrogram_frame(SpectralEnsembleMode::MinSpec, &frames, &mut out);
    assert_eq!(out[0], Complex32::new(1.0, 1.0));
    assert_eq!(out[1], Complex32::new(1.0, 0.0));
    assert_eq!(out[2], Complex32::new(0.0, 2.0));

    // 3. AVERAGE computes exact complex average
    combine_complex_spectrogram_frame(SpectralEnsembleMode::Average, &frames, &mut out);
    assert!((out[0].re - 2.0).abs() < 1e-6 && (out[0].im - 5.0 / 3.0).abs() < 1e-6);
    assert!((out[2].re - 0.0).abs() < 1e-6 && (out[2].im - 4.0).abs() < 1e-6);

    // 4. MEDIAN selects median-magnitude bin
    combine_complex_spectrogram_frame(SpectralEnsembleMode::Median, &frames, &mut out);
    assert_eq!(out[0], Complex32::new(2.0, 0.0));
    assert_eq!(out[1], Complex32::new(0.0, 3.0));
    assert_eq!(out[2], Complex32::new(0.0, 4.0));

    // 5. Full STFT waveform ensemble + phase-inverted subtraction
    let len = 4096;
    let wf1: Vec<StereoFrame> = (0..len)
        .map(|n| {
            let s = (0.5 * (2.0 * PI * 440.0 * (n as f64) / 48_000.0).sin()) as f32;
            [s, -s]
        })
        .collect();
    let wf2: Vec<StereoFrame> = (0..len)
        .map(|n| {
            let s = (0.3 * (2.0 * PI * 440.0 * (n as f64) / 48_000.0).sin()) as f32;
            [s, -s]
        })
        .collect();

    let mut ensembler = SpectralEnsembler::new(1024, 256, SpectralEnsembleMode::Weighted);
    ensembler.phase_denoise_strength = 0.5;
    let ensembled = ensembler.ensemble_waveforms(&[&wf1, &wf2], Some(&[0.7, 0.3]));
    assert_eq!(ensembled.len(), len);

    let inverted = SpectralEnsembler::invert_audio_waveform(&wf1, &ensembled);
    for i in 0..len {
        let recon_l = ensembled[i][0] + inverted[i][0];
        let recon_r = ensembled[i][1] + inverted[i][1];
        assert!((recon_l - wf1[i][0]).abs() < 1e-6);
        assert!((recon_r - wf1[i][1]).abs() < 1e-6);
    }
}

#[test]
fn test_uvr5_offline_architectures_mdx_tfctdf_htdemucs4_vr_aspp_and_assets() {
    // 1. Verify real extracted ONNX & NPZ assets on disk
    let bsrnn_onnx_path = Path::new(
        r"f:\FL studio\toneforge\assets\ozone12\models\ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx",
    );
    let delimit_onnx_path =
        Path::new(r"f:\FL studio\toneforge\assets\ozone12\models\delimit_exp009_ep080_fixed2.onnx");
    let npz_path = Path::new(r"f:\FL studio\toneforge\assets\ozone12\models\gparamsBN8.0.npz");

    assert!(bsrnn_onnx_path.exists());
    assert!(delimit_onnx_path.exists());
    assert!(npz_path.exists());

    let bsrnn_desc = OnnxModelDescriptor::from_onnx_file(bsrnn_onnx_path).unwrap();
    assert_eq!(bsrnn_desc.file_size_bytes, 7_230_143);
    assert!(bsrnn_desc.has_input_tensor);

    let delimit_desc = OnnxModelDescriptor::from_onnx_file(delimit_onnx_path).unwrap();
    assert_eq!(delimit_desc.file_size_bytes, 5_580_442);

    let cnn = GenreCnnClassifier::from_npz_file(npz_path).unwrap();
    assert!(cnn.tensor_count() > 0);

    // Synthesize a 2,048-sample test mixture containing bass (80 Hz), vocals (660 Hz), and transient clicks
    let mix: Vec<StereoFrame> = (0..2048)
        .map(|n| {
            let bass = (0.4 * (2.0 * PI * 80.0 * (n as f64) / 48_000.0).sin()) as f32;
            let vocal = (0.4 * (2.0 * PI * 660.0 * (n as f64) / 48_000.0).sin()) as f32;
            let click = if n % 256 == 0 { 0.5 } else { 0.0 };
            [bass + vocal + click, bass - vocal + click]
        })
        .collect();

    // 2. Test MDX-Net / MDX23C TFC-TDF v3 with Hann window chunk anti-aliasing & 2-pass denoise
    let mut mdx = Mdx23cSeparator::new(512, 128, 1024, 0.5);
    mdx.enable_denoise_2pass = true;
    let (mdx_primary, mdx_comp) = mdx.separate_offline(&mix);
    assert_eq!(mdx_primary.len(), mix.len());
    for i in 0..mix.len() {
        assert!((mdx_primary[i][0] + mdx_comp[i][0] - mix[i][0]).abs() < 1e-5);
        assert!((mdx_primary[i][1] + mdx_comp[i][1] - mix[i][1]).abs() < 1e-5);
    }

    // 3. Test Hybrid Transformer Demucs v4 (HTDemucs4) with CrossTransformerEncoder
    let htdemucs = HTDemucs4::new(512, 128, 16, 4, 2);
    let stems4 = htdemucs.separate_4stems(&mix);
    for s in 0..4 {
        assert_eq!(stems4[s].len(), mix.len());
        let e: f32 = stems4[s].iter().map(|f| f[0] * f[0] + f[1] * f[1]).sum();
        assert!(e > 1e-3, "HTDemucs4 stem {} had zero energy", s);
    }

    // 4. Test VR Architecture CascadedASPPNet with dilations [1, 2, 4, 8, 16]
    assert_eq!(VR_ASPP_DILATIONS, [1, 2, 4, 8, 16]);
    let vr_net = CascadedASPPNet::new(512, 128, 0.25);
    let (vr_vocals, vr_inst) = vr_net.separate_vocal_instrumental(&mix);
    assert_eq!(vr_vocals.len(), mix.len());
    for i in 0..mix.len() {
        assert!((vr_vocals[i][0] + vr_inst[i][0] - mix[i][0]).abs() < 1e-5);
    }
}

#[test]
fn test_zero_heap_allocation_full_chain_proq4_soothe3_ozone12_bsrnn_in_mixer_engine() {
    let sample_rate = 48_000.0;
    let (mut engine, mut cmd_prod, mut telem_out) = MixerEngine::new(sample_rate, 128.0, 0);

    // 1. Add deterministic generator on Track 1
    engine
        .add_channel_generator(ChannelGenerator::new(
            0,
            1,
            Box::new(DeterministicGeneratorNode::new(0, 0.75, false)),
        ))
        .unwrap();

    // 2. Slot 0: Pro-Q 4 (NaturalPhase + Character Warm + Dynamic EQ + Fractional Slope)
    let mut proq4 = Box::new(ProQ4Processor::new());
    proq4.params.global.processing_mode = ProcessingMode::NaturalPhase;
    proq4.params.global.character = CharacterMode::Warm;
    proq4.params.bands[0].used = true;
    proq4.params.bands[0].enabled = true;
    proq4.params.bands[0].shape = FilterShape::Bell;
    proq4.params.bands[0].freq_hz = 1200.0;
    proq4.params.bands[0].gain_db = 4.5;
    proq4.params.bands[0].dyn_enabled = true;
    proq4.params.bands[0].dyn_range_db = -6.0;
    proq4.refresh_band_dsp(0, false);
    proq4.refresh_global_aggregates(false);

    // 3. Slot 1: Soothe3 (8-Band Hybrid Sidechain + STFT Minimum-Phase Suppressor)
    let mut soothe3 = Box::new(Soothe3Processor::new(sample_rate as f32));
    soothe3.params_mut().global.mode = SootheProcessMode::Soft;
    soothe3.params_mut().global.depth = 5.0;
    soothe3.params_mut().global.quality = TimeResolutionQuality::Normal;
    soothe3.params_mut().sc_bands[0].enabled = true;
    soothe3.params_mut().sc_bands[0].shape = SidechainFilterShape::Bandpass;
    soothe3.params_mut().sc_bands[0].freq_hz = 3200.0;
    soothe3.sync_dirty_tables();

    // 4. Slot 2: RealTimeBsrnnStemNode (5,760-sample 120ms BSRNN with active stem rebalance gains
    //    exercising the lock-free rtrb::RingBuffer worker + 2048-pt STFT + 13-subband masks!)
    let mut bsrnn = Box::new(RealTimeBsrnnStemNode::new(sample_rate));
    bsrnn.stem_gains_db = [2.0, -1.5, 1.5, -1.0];

    // 5. Slot 3: Complete Ozone 12 Master Chain (Stabilizer + Clarity + Impact + BassControl +
    //    12AX7 DK-Triode Exciter + Multiband Imager + IRC V True-Peak Maximizer + LufsTruePeakMeter)
    let mut ozone12 = Box::new(Ozone12MasterChain::new(sample_rate));
    ozone12.unlimiter.enabled = true;
    ozone12.stabilizer.enabled = true;
    ozone12.clarity.enabled = true;
    ozone12.impact.enabled = true;
    ozone12.bass_control.enabled = true;
    ozone12
        .exciter
        .set_all_bands_mode(ExciterMode::Triode, 45.0, 0.8);
    ozone12.imager.stereoizer_enable = true;
    ozone12.imager.bands[2].stereoize_percent = 35.0;
    ozone12.maximizer.irc_mode = IrcMode::IrcV;
    ozone12.maximizer.set_gain_db(4.0);
    ozone12.maximizer.set_ceiling_dbtp(-1.0);

    engine.tracks[1].add_plugin_slot(proq4, 1.0).unwrap();
    engine.tracks[1].add_plugin_slot(soothe3, 1.0).unwrap();
    engine.tracks[1].add_plugin_slot(bsrnn, 1.0).unwrap();
    engine.tracks[1].add_plugin_slot(ozone12, 1.0).unwrap();
    engine.tracks[1].add_send(0, 1.0, false);
    engine.rebuild_graph().unwrap();

    cmd_prod
        .push(EngineCommand::SetTransportPlaying(true))
        .unwrap();

    let mut master_out = [[0.0f32; 2]; 256];

    // =========================================================================
    // ENTER STRICT REAL-TIME ZERO-ALLOCATION GUARD REGION
    // =========================================================================
    begin_rt_no_alloc_region();

    for _block in 0..35 {
        engine.process_audio_block(&mut master_out);
    }

    let (alloc_count, alloc_bytes, dealloc_count) = end_rt_no_alloc_region();
    // =========================================================================

    assert_eq!(
        alloc_count, 0,
        "VIOLATION: Full-Chain (ProQ4 + Soothe3 + BSRNN + Ozone12) allocated {} times ({} bytes) on RT thread!",
        alloc_count, alloc_bytes
    );
    assert_eq!(alloc_bytes, 0);
    assert_eq!(
        dealloc_count, 0,
        "VIOLATION: Full-Chain (ProQ4 + Soothe3 + BSRNN + Ozone12) deallocated {} times on RT thread!",
        dealloc_count
    );

    let telem = telem_out.read();
    assert_eq!(telem.header.frame_seq, 35);
    assert!(telem.header.true_peak_db[0].is_finite());
}

#[test]
fn test_uvr5_edge_cases_empty_inputs_and_corrupt_binaries() {
    // 1. Corrupt ONNX / NPZ rejection
    assert!(OnnxModelDescriptor::from_onnx_bytes(&[]).is_err());
    assert!(OnnxModelDescriptor::from_onnx_bytes(&[0xFF; 128]).is_err());
    assert!(ai_uvr5_stem_engine::load_npz_archive_bytes(&[0, 1, 2, 3]).is_err());

    // 2. Empty spectrogram and waveform slices
    let mut out_spec = [Complex32::new(1.0, 1.0); 4];
    combine_complex_spectrogram_frame(SpectralEnsembleMode::MaxSpec, &[], &mut out_spec);
    assert!(out_spec.iter().all(|c| c.norm_sqr() == 0.0));

    let ensembler = SpectralEnsembler::default();
    assert!(ensembler.ensemble_waveforms(&[], None).is_empty());

    let mdx = Mdx23cSeparator::default();
    let (p, c) = mdx.separate_offline(&[]);
    assert!(p.is_empty() && c.is_empty());

    let htdemucs = HTDemucs4::default();
    let stems = htdemucs.separate_4stems(&[]);
    assert!(stems.iter().all(|s| s.is_empty()));

    let vr = CascadedASPPNet::default();
    let (v, i) = vr.separate_vocal_instrumental(&[]);
    assert!(v.is_empty() && i.is_empty());
}

#[test]
fn test_realtime_bsrnn_active_stem_alignment_mute_cancellation_and_midstream_automation() {
    use daw_core_graph::ProcessContext;

    let sr = 48_000.0;
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

    // 1. All 4 stems muted (`stem_mute = [true; 4]`):
    //    Since `sum_{s=0..3} M_s(k) == 1.0`, `Y[n] = X_delayed[n] - sum_s S_s[n]` must cancel
    //    to 0 (< 1e-5 = -100 dB) once the 5,760-sample latency + 2,048-sample STFT window warm up!
    // Real ONNX BSRNN separator serviced inline (lockstep) — the stems come from the trained model.
    let mut node_mute = RealTimeBsrnnStemNode::new_with_onnx_lockstep(sr)
        .expect("ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx must load for lockstep verification");
    assert!(node_mute.separation_available());
    node_mute.stem_mute = [true; 4];

    let total_samples = 12_288;
    let warmup_samples = 5760 + 2048;
    let mut max_mute_residual = 0.0f32;

    for block_start in (0..total_samples).step_by(256) {
        let mut block = [[0.0f32; 2]; 256];
        for i in 0..256 {
            let n = block_start + i;
            let l = (0.45 * (2.0 * PI * 110.0 * (n as f64) / sr).sin()
                + 0.35 * (2.0 * PI * 880.0 * (n as f64) / sr).cos()) as f32;
            let r = (0.45 * (2.0 * PI * 110.0 * (n as f64) / sr).sin()
                - 0.30 * (2.0 * PI * 1320.0 * (n as f64) / sr).sin()) as f32;
            block[i] = [l, r];
        }
        node_mute.process_slice(&ctx, &mut block, None, &mut []);
        for i in 0..256 {
            let n = block_start + i;
            if n >= warmup_samples {
                max_mute_residual = max_mute_residual
                    .max(block[i][0].abs())
                    .max(block[i][1].abs());
            }
        }
    }
    // Float32 STFT/iSTFT reconstruction floor of the real model path (stems sum to the mixture
    // through 4 independent overlap-add syntheses): <= 5e-5 (-86 dBFS) on a 0.8-peak signal.
    assert!(
        max_mute_residual < 5e-5,
        "Active 4-stem mute subtraction failed to cancel dry signal (comb/delay mismatch): max_residual = {}",
        max_mute_residual
    );

    // 2. Mid-stream automation from 0.0 dB to +6.0 dB across all 4 stems: the worker queue stays
    //    warm during unity, the move is de-zippered (~10 ms ramp, never overshooting the target) and
    //    then settles exactly on `G * x[n - 5760]`.
    let mut node_auto = RealTimeBsrnnStemNode::new_with_onnx_lockstep(sr)
        .expect("BSRNN ONNX lockstep separator");
    let total_samples = 16_384;
    let gain_db = 6.0f32;
    let gain_lin = 10.0f32.powf(gain_db / 20.0);
    let mut input_history = Vec::with_capacity(total_samples);
    let mut max_boost_err = 0.0f32;

    for block_start in (0..total_samples).step_by(256) {
        if block_start == 8192 {
            // Switch from unity 0.0 dB to +6.0 dB mid-stream!
            node_auto.stem_gains_db = [gain_db; 4];
        }
        let mut block = [[0.0f32; 2]; 256];
        for i in 0..256 {
            let n = block_start + i;
            let l = (0.4 * (2.0 * PI * 220.0 * (n as f64) / sr).sin()) as f32;
            let r = (0.4 * (2.0 * PI * 660.0 * (n as f64) / sr).cos()) as f32;
            block[i] = [l, r];
            input_history.push([l, r]);
        }
        node_auto.process_slice(&ctx, &mut block, None, &mut []);
        if block_start >= 8192 {
            for i in 0..256 {
                let n = block_start + i;
                let expected = input_history[n - 5760];
                // Ramp region: bounded by the target gain (no overshoot / zipper spikes).
                assert!(block[i][0].abs() <= expected[0].abs() * gain_lin + 1e-4);
                assert!(block[i][1].abs() <= expected[1].abs() * gain_lin + 1e-4);
                if n >= 8192 + 6000 {
                    let err_l = (block[i][0] - expected[0] * gain_lin).abs();
                    let err_r = (block[i][1] - expected[1] * gain_lin).abs();
                    max_boost_err = max_boost_err.max(err_l).max(err_r);
                }
            }
        }
    }
    assert!(
        max_boost_err < 5e-5,
        "Mid-stream 0 dB -> +6 dB stem automation desynchronized: max_boost_err = {}",
        max_boost_err
    );
}

#[test]
fn test_npz_archive_reader_and_ozone_npz_gru_engine_real_models() {
    use ai_uvr5_stem_engine::{InstrumentCnnClassifier, NpzArchiveReader, OzoneNpzGruEngine};

    let vocals_npz_path =
        Path::new(r"f:\FL studio\toneforge\assets\ozone12\models\vocals-small_sanitized.npz");
    let gparams_npz_path =
        Path::new(r"f:\FL studio\toneforge\assets\ozone12\models\gparamsBN8.0.npz");
    let inst_npz_path = Path::new(
        r"f:\FL studio\toneforge\assets\ozone12\models\instrument_classifier_params.npz",
    );

    assert!(vocals_npz_path.exists());
    assert!(gparams_npz_path.exists());
    assert!(inst_npz_path.exists());

    assert_eq!(
        std::fs::metadata(vocals_npz_path).unwrap().len(),
        26_261_516
    );
    assert_eq!(std::fs::metadata(gparams_npz_path).unwrap().len(), 6_522_880);
    assert_eq!(std::fs::metadata(inst_npz_path).unwrap().len(), 19_853_709);

    // 1. Parse vocals-small_sanitized.npz via NpzArchiveReader & OzoneNpzGruEngine
    let vocals_reader = NpzArchiveReader::from_file(vocals_npz_path).unwrap();
    assert_eq!(vocals_reader.tensor_count(), 27);

    let mut gru_engine = OzoneNpzGruEngine::from_reader(&vocals_reader).unwrap();
    assert_eq!(gru_engine.fft_size, 2048);
    assert_eq!(gru_engine.fft_stride, 512);
    assert_eq!(gru_engine.sampling_rate, 44_100);
    assert_eq!(gru_engine.weight_tensor_count, 22);
    assert_eq!(gru_engine.total_npy_entries, 27);

    // Run 1 frame of 2050-bin stereo STFT magnitude through 2-layer GRU-512 + Dense-2050
    let mut x_mag = [0.0f32; 2050];
    for (k, val) in x_mag.iter_mut().enumerate() {
        *val = ((k as f32 * 0.031).sin().abs()) * 0.25;
    }
    let mut out_mask = [0.0f32; 2050];
    gru_engine.forward_stft_frame(&x_mag, &mut out_mask);
    assert!(
        out_mask
            .iter()
            .all(|&m| m.is_finite() && m > 0.0 && m < 1.0),
        "OzoneNpzGruEngine mask values must be finite and in (0, 1)"
    );

    gru_engine.reset_hidden_state();
    assert_eq!(gru_engine.h1_state, gru_engine.gru1.h_0);
    assert_eq!(gru_engine.h2_state, gru_engine.gru2.h_0);

    // 2. Parse gparamsBN8.0.npz and instrument_classifier_params.npz and run 10-class Mel-80 inference
    let genre_cnn = GenreCnnClassifier::from_npz_file(gparams_npz_path).unwrap();
    assert_eq!(genre_cnn.tensor_count(), 34);
    let mut mel_80 = [0.0f32; 80];
    for (b, m) in mel_80.iter_mut().enumerate() {
        *m = -25.0 + (b as f32) * 0.22;
    }
    let genre_probs = genre_cnn.classify_mel_80(&mel_80);
    assert!(genre_probs.iter().all(|&p| p.is_finite() && p >= 0.0));
    assert!((genre_probs.iter().sum::<f32>() - 1.0).abs() < 1e-5);

    let inst_cnn = InstrumentCnnClassifier::from_npz_file(inst_npz_path).unwrap();
    assert_eq!(inst_cnn.tensor_count(), 12);
    let inst_probs = inst_cnn.classify_mel_80(&mel_80);
    assert!(inst_probs.iter().all(|&p| p.is_finite() && p >= 0.0));
    assert!((inst_probs.iter().sum::<f32>() - 1.0).abs() < 1e-5);
}

#[test]
fn test_gate_5b_all_4_onnx_and_all_5_npz_models_real_inference() {
    use ai_uvr5_stem_engine::{
        DELIMIT_LATENCY_SAMPLES_48K, GENRE_LABELS, GruThreeStemSeparator, INSTRUMENT_LABELS,
        InstrumentCnnClassifier, OzoneBsrnnOnnxSeparator, OzoneDelimitOnnxProcessor,
        Uvr5MdxOnnxSeparator,
    };
    use daw_core_graph::ProcessContext;

    let sr = 48_000.0_f64;
    let sr_u32 = 48_000_u32;

    // Synthesize a rich 16,384-sample (341 ms @ 48 kHz) multi-stem test mixture:
    // Kick/snare drum transients + 55 Hz bassline + 440/880 Hz vocal harmonic + stereo chord pad
    let mix_len = 16_384usize;
    let mix: Vec<StereoFrame> = (0..mix_len)
        .map(|n| {
            let t = (n as f64) / sr;
            let hit = (n % 4096) as f64;
            let kick = ((-hit / 220.0).exp() * (2.0 * PI * 58.0 * t).sin()) as f32;
            let snare_noise = (((n as u64).wrapping_mul(6364136223846793005).wrapping_add(1) >> 32)
                as u32 as i32 as f32
                / (i32::MAX as f32))
                * ((-(((n + 2048) % 4096) as f64) / 140.0).exp() as f32);
            let bass = (0.32 * (2.0 * PI * 55.0 * t).sin() + 0.14 * (2.0 * PI * 110.0 * t).sin()) as f32;
            let vocal = (0.30 * (2.0 * PI * 523.25 * t).sin()
                + 0.16 * (2.0 * PI * 1046.5 * t).sin()
                + 0.08 * (2.0 * PI * 1569.75 * t).sin()) as f32;
            let pad_l = (0.18 * (2.0 * PI * 329.63 * t).sin()) as f32;
            let pad_r = (0.18 * (2.0 * PI * 392.00 * t).cos()) as f32;
            [
                0.45 * kick + 0.25 * snare_noise + bass + vocal + pad_l,
                0.45 * kick + 0.25 * snare_noise + bass + vocal + pad_r,
            ]
        })
        .collect();

    // =========================================================================
    // PART A: ALL 4 REAL `.onnx` MODELS (`CUDA -> DirectML -> CPU`)
    // =========================================================================

    // ONNX #1: `delimit_exp009_ep080_fixed2.onnx` wired into `Ozone12MasterChain::unlimiter`
    let delimit = OzoneDelimitOnnxProcessor::from_default_assets()
        .expect("delimit_exp009_ep080_fixed2.onnx must load via ONNX Runtime");
    let delimit_tier = delimit.active_tier();

    let mut chain = Ozone12MasterChain::new(sr);
    chain.unlimiter.enabled = true;
    chain.unlimiter.threshold_db = -8.0;
    chain.unlimiter.amount_percent = 90.0;
    chain.set_unlimiter_neural_backend(Box::new(delimit));
    // Total chain latency = 2048 (Neural Unlimiter @ 48 kHz) + 264 (IRC V Maximizer @ 48 kHz) = 2312 samples
    assert_eq!(chain.latency_samples(), DELIMIT_LATENCY_SAMPLES_48K + 264);

    // Disable Maximizer to measure pure Neural Unlimiter transient crest restoration on brickwall-clipped audio
    chain.set_irc_maximizer_enabled(false);
    assert_eq!(chain.latency_samples(), DELIMIT_LATENCY_SAMPLES_48K);

    // Brickwall-clip the transient mix at +/- 0.22 (-13.15 dBFS) to crush drum peaks
    let clip_ceil = 0.22_f32;
    let clipped_in: Vec<StereoFrame> = mix
        .iter()
        .map(|f| [f[0].clamp(-clip_ceil, clip_ceil), f[1].clamp(-clip_ceil, clip_ceil)])
        .collect();

    let crest_db = |buf: &[StereoFrame]| -> f64 {
        let mut peak = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        for f in buf {
            let l = f[0] as f64;
            let r = f[1] as f64;
            peak = peak.max(l.abs()).max(r.abs());
            sum_sq += 0.5 * (l * l + r * r);
        }
        let rms = (sum_sq / (buf.len().max(1) as f64)).sqrt().max(1.0e-12);
        20.0 * (peak / rms).max(1.0e-12).log10()
    };

    let mut unlim_stream = clipped_in.clone();
    // Append 2048 zero frames to flush lookahead delay
    unlim_stream.extend(std::iter::repeat_n([0.0_f32; 2], DELIMIT_LATENCY_SAMPLES_48K));
    let ctx = ProcessContext {
        sample_rate: sr,
        current_sample_pos: 0,
        ppq_tick: 0,
        bpm: 120.0,
        slice_frames: 256,
        is_playing: true,
        is_offline_rendering: true,
        ppq_resolution: daw_core_graph::DEFAULT_PPQ,
    };
    for chunk in unlim_stream.chunks_mut(256) {
        chain.process_slice(&ctx, chunk, None, &mut []);
    }
    let aligned_restored = &unlim_stream[DELIMIT_LATENCY_SAMPLES_48K..DELIMIT_LATENCY_SAMPLES_48K + mix_len];
    let in_crest = crest_db(&clipped_in[2048..]);
    let out_crest = crest_db(&aligned_restored[2048..]);
    println!(
        "[Gate 5b][ONNX 1/4] delimit_exp009_ep080_fixed2.onnx tier={:?}, clipped crest={in_crest:.2} dB -> restored crest={out_crest:.2} dB (+{:.2} dB)",
        delimit_tier,
        out_crest - in_crest
    );
    assert!(
        aligned_restored.iter().all(|f| f[0].is_finite() && f[1].is_finite()),
        "Delimit ONNX output contained non-finite samples"
    );
    assert!(
        out_crest > in_crest + 0.25,
        "Neural Unlimiter (delimit_exp009_ep080_fixed2.onnx) failed to expand crushed transient crest factor ({in_crest:.2} dB -> {out_crest:.2} dB)"
    );

    // ONNX #2: `ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx`
    let mut bsrnn = OzoneBsrnnOnnxSeparator::from_default_assets()
        .expect("ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx must load");
    let bsrnn_stems = bsrnn
        .separate_4stems_offline(&mix)
        .expect("BSRNN 4-stem separation failed");
    let mut bsrnn_max_recon_err = 0.0_f32;
    let mut bsrnn_energies = [0.0_f64; 4];
    for i in 0..mix_len {
        for s in 0..4 {
            bsrnn_energies[s] += (bsrnn_stems[s][i][0] as f64).powi(2) + (bsrnn_stems[s][i][1] as f64).powi(2);
        }
        for ch in 0..2 {
            let sum = bsrnn_stems[0][i][ch]
                + bsrnn_stems[1][i][ch]
                + bsrnn_stems[2][i][ch]
                + bsrnn_stems[3][i][ch];
            bsrnn_max_recon_err = bsrnn_max_recon_err.max((sum - mix[i][ch]).abs());
        }
    }
    println!(
        "[Gate 5b][ONNX 2/4] ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx tier={:?}, energies={:?}, max_recon_err={bsrnn_max_recon_err:.3e}",
        bsrnn.active_tier(),
        bsrnn_energies
    );
    assert!(bsrnn_energies.iter().all(|&e| e > 1.0e-3));
    assert!(bsrnn_max_recon_err < 1.0e-5);

    // ONNX #3: `Kim_Vocal_2.onnx`
    let mut kim = Uvr5MdxOnnxSeparator::from_kim_vocal_2().expect("Kim_Vocal_2.onnx must load");
    let (kim_voc, kim_inst) = kim
        .separate_vocal_instrumental(&mix, sr_u32, false)
        .expect("Kim_Vocal_2 inference failed");
    let kim_voc_e: f64 = kim_voc.iter().map(|f| (f[0] as f64).powi(2)).sum();
    let kim_inst_e: f64 = kim_inst.iter().map(|f| (f[0] as f64).powi(2)).sum();
    println!(
        "[Gate 5b][ONNX 3/4] Kim_Vocal_2.onnx tier={:?}, voc_energy={kim_voc_e:.3}, inst_energy={kim_inst_e:.3}",
        kim.active_tier()
    );
    assert!(kim_voc_e.is_finite() && kim_inst_e.is_finite() && (kim_voc_e + kim_inst_e) > 1.0);

    // ONNX #4: `UVR-MDX-NET-Inst_HQ_3.onnx`
    let mut inst_hq3 =
        Uvr5MdxOnnxSeparator::from_inst_hq_3().expect("UVR-MDX-NET-Inst_HQ_3.onnx must load");
    let (hq3_voc, hq3_inst) = inst_hq3
        .separate_vocal_instrumental(&mix, sr_u32, false)
        .expect("UVR-MDX-NET-Inst_HQ_3 inference failed");
    let hq3_voc_e: f64 = hq3_voc.iter().map(|f| (f[0] as f64).powi(2)).sum();
    let hq3_inst_e: f64 = hq3_inst.iter().map(|f| (f[0] as f64).powi(2)).sum();
    println!(
        "[Gate 5b][ONNX 4/4] UVR-MDX-NET-Inst_HQ_3.onnx tier={:?}, voc_energy={hq3_voc_e:.3}, inst_energy={hq3_inst_e:.3}",
        inst_hq3.active_tier()
    );
    assert!(hq3_voc_e.is_finite() && hq3_inst_e.is_finite() && (hq3_voc_e + hq3_inst_e) > 1.0);

    // =========================================================================
    // PART B: ALL 5 REAL `.npz` MODELS (Slaney Mel + 2D CNNs + 3x GRU-512)
    // =========================================================================

    // NPZ #1: `gparamsBN8.0.npz` (`GenreCnnClassifier`)
    let genre_clf = GenreCnnClassifier::load_default().expect("gparamsBN8.0.npz must load");
    let g_probs = genre_clf.classify_audio(&mix, sr_u32);
    let (top_g_idx, &top_g_prob) = g_probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    println!(
        "[Gate 5b][NPZ 1/5] gparamsBN8.0.npz top_genre={} ({:.1}%)",
        GENRE_LABELS[top_g_idx],
        top_g_prob * 100.0
    );
    assert!((g_probs.iter().sum::<f32>() - 1.0).abs() < 1.0e-5);

    // NPZ #2: `instrument_classifier_params.npz` (`InstrumentCnnClassifier`)
    let inst_clf =
        InstrumentCnnClassifier::load_default().expect("instrument_classifier_params.npz must load");
    let i_probs = inst_clf.classify_audio(&mix, sr_u32);
    let (top_i_idx, &top_i_prob) = i_probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    println!(
        "[Gate 5b][NPZ 2/5] instrument_classifier_params.npz top_inst={} ({:.1}%)",
        INSTRUMENT_LABELS[top_i_idx],
        top_i_prob * 100.0
    );
    assert!((i_probs.iter().sum::<f32>() - 1.0).abs() < 1.0e-5);

    // NPZ #3, #4, #5: `vocals-small_sanitized.npz`, `drums-small_sanitized.npz`, `bass-small_sanitized.npz`
    let mut gru3 = GruThreeStemSeparator::from_default_assets()
        .expect("All 3 GRU-512 .npz stem models must load");
    let [gru_v, gru_d, gru_b, gru_o] = gru3.separate_4stems(&mix, sr_u32);
    let gru_energies = [
        gru_v.iter().map(|f| (f[0] as f64).powi(2)).sum::<f64>(),
        gru_d.iter().map(|f| (f[0] as f64).powi(2)).sum::<f64>(),
        gru_b.iter().map(|f| (f[0] as f64).powi(2)).sum::<f64>(),
        gru_o.iter().map(|f| (f[0] as f64).powi(2)).sum::<f64>(),
    ];
    let mut gru_max_recon_err = 0.0_f32;
    for i in 0..mix_len {
        for ch in 0..2 {
            let s = gru_v[i][ch] + gru_d[i][ch] + gru_b[i][ch] + gru_o[i][ch];
            gru_max_recon_err = gru_max_recon_err.max((s - mix[i][ch]).abs());
        }
    }
    println!(
        "[Gate 5b][NPZ 3..5/5] GRU-512 3-model stems energies (V,D,B,O)={gru_energies:?}, max_recon_err={gru_max_recon_err:.3e}"
    );
    assert!(gru_energies[0] > 1.0e-3 && gru_energies[1] > 1.0e-3 && gru_energies[2] > 1.0e-3);
    assert!(
        (gru_energies[0] - gru_energies[1]).abs() > 1.0e-3
            && (gru_energies[1] - gru_energies[2]).abs() > 1.0e-3,
        "Vocals, Drums, and Bass .npz GRU models must produce distinct trained stem separations!"
    );
    assert!(gru_max_recon_err < 1.0e-5);
}

#[test]
fn test_realtime_delimit_onnx_worker_zero_rt_heap_allocations_in_ozone12_chain() {
    use ai_uvr5_stem_engine::{
        DELIMIT_LATENCY_SAMPLES_48K, DELIMIT_RT_CUSHION_SAMPLES, RealtimeDelimitOnnxWorker,
    };
    use daw_core_graph::ProcessContext;

    let sr = 48_000.0;
    let worker = RealtimeDelimitOnnxWorker::from_default_assets(sr)
        .expect("RealtimeDelimitOnnxWorker must initialize");
    let mut chain = Ozone12MasterChain::new(sr);
    chain.unlimiter.enabled = true;
    chain.unlimiter.threshold_db = -6.0;
    chain.unlimiter.amount_percent = 75.0;
    chain.set_unlimiter_neural_backend(Box::new(worker));

    // The RT worker adds its fixed alignment cushion to the reported latency.
    assert_eq!(
        chain.latency_samples(),
        DELIMIT_LATENCY_SAMPLES_48K + DELIMIT_RT_CUSHION_SAMPLES + 264
    );

    let ctx = ProcessContext {
        sample_rate: sr,
        current_sample_pos: 0,
        ppq_tick: 0,
        bpm: 128.0,
        slice_frames: 256,
        is_playing: true,
        is_offline_rendering: false,
        ppq_resolution: daw_core_graph::DEFAULT_PPQ,
    };

    let mut block = [[0.0_f32; 2]; 256];
    // Warm up 4 blocks
    for b in 0..4 {
        for (i, f) in block.iter_mut().enumerate() {
            let n = b * 256 + i;
            let s = (0.75 * (2.0 * PI * 220.0 * (n as f64) / sr).sin()).clamp(-0.35, 0.35) as f32;
            *f = [s, s];
        }
        chain.process_slice(&ctx, &mut block, None, &mut []);
    }

    // Strict Zero-Heap-Allocation Real-Time Audio Callback Region (32 x 256 = 8,192 frames)
    begin_rt_no_alloc_region();
    for b in 0..32 {
        for (i, f) in block.iter_mut().enumerate() {
            let n = (b + 4) * 256 + i;
            let s = (0.85 * (2.0 * PI * 110.0 * (n as f64) / sr).sin()).clamp(-0.35, 0.35) as f32;
            *f = [s, -s];
        }
        chain.process_slice(&ctx, &mut block, None, &mut []);
    }
    let (allocs, bytes, deallocs) = end_rt_no_alloc_region();

    println!(
        "[Gate 5b][RT Zero-Alloc] Ozone12MasterChain + RealtimeDelimitOnnxWorker over 8,192 frames: allocs={allocs}, bytes={bytes}, deallocs={deallocs}"
    );
    assert_eq!(
        (allocs, bytes, deallocs),
        (0, 0, 0),
        "RealtimeDelimitOnnxWorker + Ozone12MasterChain allocated on the RT audio thread!"
    );
}






/// The real-time DeLimit worker must return exactly the inline (lockstep) ONNX residual delayed by
/// `DELIMIT_RT_CUSHION_SAMPLES` — a constant, reported offset regardless of worker timing.
#[test]
fn test_realtime_delimit_worker_residual_is_sample_aligned_with_inline_processor() {
    use ai_uvr5_stem_engine::{
        DELIMIT_RT_CUSHION_SAMPLES, OzoneDelimitOnnxProcessor, RealtimeDelimitOnnxWorker,
    };
    use dsp_ozone12_mastering::NeuralUnlimiterBackend;

    let sr = 48_000.0;
    let n_total = 48_000usize;
    let input: Vec<f32> = (0..n_total)
        .map(|n| {
            let t = n as f64 / sr;
            ((0.9 * (2.0 * PI * 140.0 * t).sin() + 0.4 * (2.0 * PI * 2300.0 * t).sin()).clamp(-0.4, 0.4))
                as f32
        })
        .collect();

    let mut inline = OzoneDelimitOnnxProcessor::from_default_assets().expect("DeLimit ONNX");
    let mut reference = vec![0.0f32; n_total];
    for (inp, out) in input.chunks(256).zip(reference.chunks_mut(256)) {
        inline.process_delimit_envelope(inp, sr, out);
    }

    let mut rt = RealtimeDelimitOnnxWorker::from_default_assets(sr).expect("DeLimit RT worker");
    let mut got = vec![0.0f32; n_total];
    for (inp, out) in input.chunks(256).zip(got.chunks_mut(256)) {
        rt.process_delimit_envelope(inp, sr, out);
        rt.wait_for_worker_drain(std::time::Duration::from_millis(500));
    }
    let mut max_err = 0.0f32;
    let mut energy = 0.0f64;
    for i in DELIMIT_RT_CUSHION_SAMPLES..n_total {
        let r = reference[i - DELIMIT_RT_CUSHION_SAMPLES];
        max_err = max_err.max((got[i] - r).abs());
        energy += (r as f64) * (r as f64);
    }
    assert!(energy > 1e-6, "inline DeLimit produced no residual to compare");
    assert!(max_err < 1e-5, "RT residual misaligned vs inline ONNX: max_err = {max_err}");
}
