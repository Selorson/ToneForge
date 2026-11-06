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
    };

    // 1. All 4 stems muted (`stem_mute = [true; 4]`):
    //    Since `sum_{s=0..3} M_s(k) == 1.0`, `Y[n] = X_delayed[n] - sum_s S_s[n]` must cancel
    //    to 0 (< 1e-5 = -100 dB) once the 5,760-sample latency + 2,048-sample STFT window warm up!
    let mut node_mute = RealTimeBsrnnStemNode::new(sr);
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
    assert!(
        max_mute_residual < 1e-5,
        "Active 4-stem mute subtraction failed to cancel dry signal (comb/delay mismatch): max_residual = {}",
        max_mute_residual
    );

    // 2. Mid-stream automation from 0.0 dB (unity fast path) to +6.0 dB across all 4 stems:
    //    Verifies worker queue stays warm during unity bypass and matches `G * x[n - 5760]` immediately!
    let mut node_auto = RealTimeBsrnnStemNode::new(sr);
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
                let err_l = (block[i][0] - expected[0] * gain_lin).abs();
                let err_r = (block[i][1] - expected[1] * gain_lin).abs();
                max_boost_err = max_boost_err.max(err_l).max(err_r);
            }
        }
    }
    assert!(
        max_boost_err < 1e-5,
        "Mid-stream 0 dB -> +6 dB stem automation desynchronized: max_boost_err = {}",
        max_boost_err
    );
}


