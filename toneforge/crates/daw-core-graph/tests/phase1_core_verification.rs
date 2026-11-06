//! Phase 1 Comprehensive Verification Suite for `daw-core-graph`:
//! - 5-Point APDC Null-Test Phase Cancellation (`< -140 dBFS`)
//! - `SampleAccurateAutomationQueue` (`TAutomationDelay`) sample accuracy
//! - `HybridPpqSlicer` (`f64` tick accumulator + 8-sample SIMD alignment)
//! - `TensionSpline` (FL Studio 26 signed exponential formula) + 30-Bit Fixed-Point Bus (`0x40000000`)
//! - `TopologicalMixerScheduler` (502 Tracks, Kahn topological sort, cycle detection, parallel thread pool)
//! - `SparseCompressorNode` (Gather/Scatter `log2`/`exp2`) & `PatcherGraphNode`
//! - `VisualTelemetryFrame` (128 KB / `131,072` bytes) via `TripleBuffer` & `rtrb`

use daw_core_graph::{
    AutomationClipArticulator, ChannelGenerator, ControllerLink, DEFAULT_PPQ,
    DeterministicGeneratorNode, EngineCommand, FIXED_POINT_30BIT_MAX, FIXED_POINT_30BIT_SCALE,
    FixedLatencyTransformNode, HybridPpqSlicer, InternalControllerBus, LfoModulator, LfoWaveform,
    MASTER_TRACK_INDEX, MAX_MIXER_TRACKS, MixerEngine, MixerTrack, ModularParamConnection,
    PATCHER_HOST_INPUT_NODE, PATCHER_HOST_OUTPUT_NODE, PREVIEW_TRACK_INDEX, PatcherGraphNode,
    ProcessContext, SampleAccurateAutomationQueue, SparseCompressorNode, StereoFrame, TempoMap,
    TensionControlPoint, TensionSpline, TopologicalMixerScheduler, UnifiedDspNode,
    VISUAL_TELEMETRY_FRAME_BYTES, VisualTelemetryFrame, create_telemetry_triple_buffer,
    fixed30_to_normalized, normalized_to_fixed30,
};

fn rms_dbfs(frames: &[StereoFrame]) -> (f64, f64) {
    if frames.is_empty() {
        return (-400.0, -400.0);
    }
    let mut sum_l = 0.0f64;
    let mut sum_r = 0.0f64;
    for f in frames {
        sum_l += (f[0] as f64) * (f[0] as f64);
        sum_r += (f[1] as f64) * (f[1] as f64);
    }
    let rms_l = (sum_l / (frames.len() as f64)).sqrt();
    let rms_r = (sum_r / (frames.len() as f64)).sqrt();
    let db_l = if rms_l <= 1e-20 {
        -400.0
    } else {
        20.0 * rms_l.log10()
    };
    let db_r = if rms_r <= 1e-20 {
        -400.0
    } else {
        20.0 * rms_r.log10()
    };
    (db_l, db_r)
}

// =============================================================================
// 1. 5-POINT APDC NULL-TEST SUITE (< -140 dBFS PHASE CANCELLATION)
// =============================================================================

#[test]
fn test_5point_apdc_null_cancellation_all_points_simultaneously() {
    let sample_rate = 48_000.0;
    let (mut engine, _cmd_prod, _telem_out) = MixerEngine::new(sample_rate, 128.0, 0);

    // -------------------------------------------------------------------------
    // Setup a topology that exercises ALL 5 APDC points simultaneously:
    //
    // - Point 1 (Channel Output APDC):
    //   * Channel 0 -> Track 1: latency = 13 samples, amplitude = 1.0, normal polarity (+1.0)
    //   * Channel 1 -> Track 2: latency = 41 samples, amplitude = 0.5, inverted polarity (-0.5)
    //   * Channel 2 -> Track 4: latency =  7 samples, amplitude = 0.5, inverted polarity (-0.5)
    //   (Max channel latency = 41 samples; Ch 0 is delayed by 28, Ch 2 by 34)
    //
    // - Point 3 (FX Slot Parallel Dry/Wet APDC):
    //   * Track 1 FX Slot 0 has latency = 23 samples, wet_mix = 0.35 (35% wet + 65% dry).
    //     Point 3 delays the 65% Dry branch by 23 samples so Dry + Wet sum coherently
    //     to 1.00 * signal at latency 41 + 23 = 64 samples!
    //   * Track 2 FX Slot 0 has latency = 11 samples, wet_mix = 0.60 (60% wet + 40% dry).
    //
    // - Point 2 (Hardware/Direct Input APDC):
    //   * Track 4 receives Channel 2 (-0.5, aligned to 41 samples), passes through FX Slot 0
    //     with latency = 19 samples (rack out = 41 + 19 = 60 samples), and routes to Track 2 (gain = 1.0).
    //   * Track 2 ALSO receives direct input from Channel 1 (-0.5, aligned to 41 samples).
    //   * Point 2 (`Track 2::input_apdc_delay`) delays Track 2's direct input by `60 - 41 = 19`
    //     samples so Channel 1 (-0.5) and Track 4 send (-0.5) arrive in phase at Track 2's
    //     FX rack entrance at sample 60 and sum to `-1.0`!
    //
    // - Point 4 (Post-Rack Shared APDC) & Point 5 (Inter-Track Differential Send APDC):
    //   * Track 1 (rack out = 64 samples, gain = +1.0) splits to TWO destinations:
    //     (a) Track 1 -> Track 10 (gain = 0.5), where Track 10 has FX Slot 0 with latency = 37
    //         samples (rack out = 64 + 37 = 101 samples) and routes to Master Track 0 (gain = 1.0).
    //     (b) Track 1 -> Master Track 0 (gain = 0.5) directly.
    //   * Point 4 (`post_rack_apdc_delay`) + Point 5 (`send_apdc_delay` on route `1 -> 0`)
    //     compensate the direct branch `1 -> 0` by 37 samples so both halves of Track 1
    //     arrive at Master Track 0 at sample 101 with combined gain `+0.5 + 0.5 = +1.0`!
    //   * Meanwhile, Track 2 (rack out = 60 + 11 = 71 samples, gain = -1.0) routes to
    //     Master Track 0 (gain = 1.0), compensated by `101 - 71 = 30` samples via Point 4/5!
    //   * At Master Track 0: `+1.0 + (-1.0) = 0.0` (<-140 dBFS absolute phase cancellation)!
    // -------------------------------------------------------------------------

    engine.channels.push(ChannelGenerator::new(
        0,
        1,
        Box::new(DeterministicGeneratorNode::new(13, 1.0, false)),
    ));
    engine.channels.push(ChannelGenerator::new(
        1,
        2,
        Box::new(DeterministicGeneratorNode::new(41, 0.5, true)),
    ));
    engine.channels.push(ChannelGenerator::new(
        2,
        4,
        Box::new(DeterministicGeneratorNode::new(7, 0.5, true)),
    ));

    // Track 1 FX Slot 0: latency = 23 samples, parallel wet_mix = 0.35 (Point 3 APDC!)
    engine.tracks[1]
        .add_plugin_slot(Box::new(FixedLatencyTransformNode::new(23, 1.0, false)), 0.35)
        .unwrap();

    // Track 10 FX Slot 0: latency = 37 samples, wet_mix = 1.0
    engine.tracks[10]
        .add_plugin_slot(Box::new(FixedLatencyTransformNode::new(37, 1.0, false)), 1.0)
        .unwrap();

    // Track 4 FX Slot 0: latency = 19 samples, wet_mix = 1.0
    engine.tracks[4]
        .add_plugin_slot(Box::new(FixedLatencyTransformNode::new(19, 1.0, false)), 1.0)
        .unwrap();

    // Track 2 FX Slot 0: latency = 11 samples, parallel wet_mix = 0.60 (Point 3 APDC!)
    engine.tracks[2]
        .add_plugin_slot(Box::new(FixedLatencyTransformNode::new(11, 1.0, false)), 0.60)
        .unwrap();

    // Configure Mixer DAG routes
    engine.tracks[1].add_send(10, 0.5, false);
    engine.tracks[1].add_send(MASTER_TRACK_INDEX, 0.5, false);
    engine.tracks[10].add_send(MASTER_TRACK_INDEX, 1.0, false);
    engine.tracks[4].add_send(2, 1.0, false);
    engine.tracks[2].add_send(MASTER_TRACK_INDEX, 1.0, false);

    engine.rebuild_graph().unwrap();

    // Verify all 5 APDC points have non-zero calculated compensation delays:
    assert_eq!(
        engine.channels[0].channel_apdc_delay.delay_samples(),
        41 - 13,
        "Point 1 Channel 0 APDC mismatch"
    );
    assert_eq!(
        engine.channels[2].channel_apdc_delay.delay_samples(),
        41 - 7,
        "Point 1 Channel 2 APDC mismatch"
    );
    assert_eq!(
        engine.tracks[2].input_apdc_delay.delay_samples(),
        19,
        "Point 2 Track 2 Input APDC should delay direct input by 19 samples"
    );
    assert_eq!(
        engine.tracks[1].slots[0].dry_apdc_delay.delay_samples(),
        23,
        "Point 3 Track 1 FX Slot 0 Dry APDC should equal plugin latency (23)"
    );
    assert_eq!(
        engine.tracks[2].post_rack_apdc_delay.delay_samples(),
        30,
        "Point 4 Track 2 Post-Rack APDC should equal 101 - 71 = 30 samples"
    );
    let route_1_to_0_send_apdc = engine.tracks[1]
        .sends
        .iter()
        .find(|s| s.dest_track_index == MASTER_TRACK_INDEX)
        .unwrap()
        .send_apdc_delay
        .delay_samples();
    assert_eq!(
        route_1_to_0_send_apdc, 37,
        "Point 5 Send Route (1 -> 0) APDC should delay direct send by 37 samples"
    );

    // Warmup delay lines (total latency is 101 samples; 3 blocks of 128 = 384 samples > 101)
    let mut master_out = [[0.0f32; 2]; 128];
    for _ in 0..3 {
        engine.process_audio_block(&mut master_out);
    }

    // Measure steady-state null cancellation across 8 consecutive blocks (1024 frames)
    let mut max_abs_err = 0.0f32;
    let mut steady_frames = Vec::with_capacity(1024);

    for _ in 0..8 {
        engine.process_audio_block(&mut master_out);
        for &frame in &master_out {
            max_abs_err = max_abs_err.max(frame[0].abs()).max(frame[1].abs());
            steady_frames.push(frame);
        }
    }

    let (dbfs_l, dbfs_r) = rms_dbfs(&steady_frames);
    assert!(
        dbfs_l < -140.0 && dbfs_r < -140.0,
        "5-Point APDC Null-Test FAILED: residual L = {:.2} dBFS, R = {:.2} dBFS (max_abs = {:e}), expected < -140 dBFS",
        dbfs_l,
        dbfs_r,
        max_abs_err
    );
}

#[test]
fn test_patcher_modular_internal_apdc_null_cancellation() {
    let mut patcher = PatcherGraphNode::new();
    patcher.prepare(48_000.0, 256);

    // Branch A: latency = 31 samples, gain = +1.0, normal polarity
    let idx_a = patcher.add_node(Box::new(FixedLatencyTransformNode::new(31, 1.0, false)));
    // Branch B: latency = 9 samples, gain = 1.0, inverted polarity (-1.0)
    let idx_b = patcher.add_node(Box::new(FixedLatencyTransformNode::new(9, 1.0, true)));

    assert!(patcher.add_audio_connection(PATCHER_HOST_INPUT_NODE, idx_a, 1.0));
    assert!(patcher.add_audio_connection(PATCHER_HOST_INPUT_NODE, idx_b, 1.0));
    assert!(patcher.add_audio_connection(idx_a, PATCHER_HOST_OUTPUT_NODE, 1.0));
    assert!(patcher.add_audio_connection(idx_b, PATCHER_HOST_OUTPUT_NODE, 1.0));

    assert_eq!(patcher.latency_samples(), 31);

    let mut buf = [[0.0f32; 2]; 128];
    let mut ctx = ProcessContext::default();
    ctx.slice_frames = 128;

    for block in 0..6 {
        ctx.current_sample_pos = (block * 128) as i64;
        for (i, frame) in buf.iter_mut().enumerate() {
            *frame = DeterministicGeneratorNode::reference_waveform_at(ctx.current_sample_pos + i as i64);
        }
        patcher.process_slice(&ctx, &mut buf, None, &mut []);
    }

    let (dbfs_l, dbfs_r) = rms_dbfs(&buf);
    assert!(
        dbfs_l < -140.0 && dbfs_r < -140.0,
        "PatcherGraphNode internal APDC null-test failed: L={:.2} dBFS, R={:.2} dBFS",
        dbfs_l,
        dbfs_r
    );
}

#[test]
fn test_tautomation_delay_sample_accurate_synchronization() {
    let mut queue = SampleAccurateAutomationQueue::with_capacity(64);

    // Schedule parameter change at raw sample 20 with upstream APDC delay = 150 samples
    // -> target compensated sample position = 170
    assert!(queue.schedule_param_normalized(20, 150, 1, 0, 100, 3, 0.85));
    assert_eq!(queue.pending_param_count(), 1);

    // Block 0 (samples 0..128): event at 170 should NOT fire yet
    let mut fired_b0 = Vec::new();
    queue.drain_up_to(128, |ev| fired_b0.push(*ev), |_| {});
    assert!(fired_b0.is_empty());
    assert_eq!(queue.pending_param_count(), 1);

    // Block 1 (samples 128..256): event at 170 MUST fire!
    let mut fired_b1 = Vec::new();
    queue.drain_up_to(256, |ev| fired_b1.push(*ev), |_| {});
    assert_eq!(fired_b1.len(), 1);
    assert_eq!(fired_b1[0].target_sample_pos, 170);
    assert_eq!(fired_b1[0].target_track, 1);
    assert_eq!(fired_b1[0].param_index, 3);
    assert!((fired_b1[0].norm_value - 0.85).abs() < 1e-6);
    assert_eq!(queue.pending_param_count(), 0);
}

// =============================================================================
// 2. HYBRID PPQ SLICER & TENSION SPLINE & 30-BIT FIXED-POINT TESTS
// =============================================================================

#[test]
fn test_hybrid_ppq_slicer_simd_alignment_and_zero_drift() {
    // At 48000 Hz, 125 BPM, 96 PPQ:
    // samples_per_tick = (60 * 48000) / (125 * 96) = 240.0 samples/tick
    let mut slicer = HybridPpqSlicer::new(48_000.0, 125.0, DEFAULT_PPQ);
    assert!((slicer.samples_per_tick_f64 - 240.0).abs() < 1e-12);

    let mut total_frames = 0usize;
    let mut tick_count = 0usize;

    for _ in 0..100 {
        let mut block_accum = 0usize;
        slicer.process_buffer(
            512,
            |ppq_tick, sample_pos| {
                assert_eq!(ppq_tick as usize, tick_count);
                assert_eq!(
                    sample_pos % 8,
                    0,
                    "Tick sample_pos {} must be 8-sample SIMD aligned",
                    sample_pos
                );
                tick_count += 1;
            },
            |offset, len| {
                assert_eq!(offset, block_accum);
                block_accum += len;
            },
        );
        assert_eq!(block_accum, 512);
        total_frames += 512;
    }

    assert_eq!(total_frames, 51_200);
    // 51,200 / 240 = 213.3333... => ticks 0..=213 (214 ticks) fired!
    assert_eq!(tick_count, 214);
    assert_eq!(slicer.current_sample_pos, 51_200);

    // Dynamic TempoMap transition test
    let mut map = TempoMap::new(120.0);
    map.add_point(0, 120.0);
    map.add_point(10, 240.0);
    assert!((map.bpm_at_tick(5) - 120.0).abs() < 1e-9);
    assert!((map.bpm_at_tick(15) - 240.0).abs() < 1e-9);
}

#[test]
fn test_tension_spline_flstudio26_signed_exponential_and_30bit_bus() {
    let spline = TensionSpline::new(vec![
        TensionControlPoint::new(0.0, 0.0, 0.0),
        TensionControlPoint::new(96.0, 1.0, 0.6),  // k = 6.0
        TensionControlPoint::new(192.0, 0.0, -0.6), // k = -6.0
    ]);

    // Verify boundary clamping
    assert!((spline.eval(-10.0) - 0.0).abs() < 1e-9);
    assert!((spline.eval(0.0) - 0.0).abs() < 1e-9);
    assert!((spline.eval(96.0) - 1.0).abs() < 1e-9);
    assert!((spline.eval(192.0) - 0.0).abs() < 1e-9);
    assert!((spline.eval(500.0) - 0.0).abs() < 1e-9);

    // Verify exact formula ((e^(k*t) - 1) / (e^k - 1)) at t = 0.5 (x = 48.0, tension = 0.6 -> k = 6.0)
    let mid_pos = spline.eval(48.0);
    let expected_pos = ((3.0f64).exp() - 1.0) / ((6.0f64).exp() - 1.0);
    assert!(
        (mid_pos - expected_pos).abs() < 1e-9,
        "TensionSpline k=+6.0 mismatch: got {}, expected {}",
        mid_pos,
        expected_pos
    );

    // Verify 30-Bit Fixed-Point InternalControllerBus (0x40000000 = 1,073,741,824)
    assert_eq!(normalized_to_fixed30(1.0), FIXED_POINT_30BIT_MAX);
    assert_eq!(normalized_to_fixed30(0.0), 0);
    assert!((FIXED_POINT_30BIT_SCALE - 1_073_741_824.0).abs() < 1e-9);

    let mut bus = InternalControllerBus::new();
    bus.publish_normalized(7, 0.33333334);
    assert!((bus.get_normalized(7) - 0.33333334).abs() < 1e-8);
    assert!((fixed30_to_normalized(bus.get_fixed30(7)) - 0.33333334).abs() < 1e-8);

    let link = ControllerLink {
        controller_id: 7,
        target_track: 1,
        slot_index: 0,
        param_id: 2,
        min_out_norm: 0.2,
        max_out_norm: 0.8,
        inverted: true,
    };
    assert!(bus.add_link(link));
    let mapped = link.map_fixed30_to_norm(bus.get_fixed30(7));
    assert!((mapped - (0.2 + (1.0 - 0.33333334) * 0.6)).abs() < 1e-6);

    // Verify LfoModulator & AutomationClipArticulator
    let mut articulator = AutomationClipArticulator::new(0, spline);
    articulator.lfo = LfoModulator {
        enabled: true,
        waveform: LfoWaveform::Sine,
        frequency_cycles_per_tick: 1.0 / 48.0,
        phase_offset: 0.0,
        amount: 0.25,
    };
    for tick in 0..192 {
        let fixed = articulator.evaluate_fixed30(tick as f64);
        assert!((0..=FIXED_POINT_30BIT_MAX).contains(&fixed));
    }
}

// =============================================================================
// 3. SPARSE COMPRESSOR NODE (GATHER/SCATTER LOG2/EXP2)
// =============================================================================

#[test]
fn test_sparse_compressor_gather_scatter_math_and_lookahead() {
    let mut comp = SparseCompressorNode::new(-12.0, 4.0, 0.0, 0);
    comp.bypass_envelope_ballistics = true; // Instantaneous envelope for exact transfer curve verification
    comp.prepare(48_000.0, 256);

    let ctx = ProcessContext::default();

    // 1. Below-threshold fast path (-26 dBFS = 0.05) -> 0 active sparse samples, unity gain
    let mut below_buf = [[0.05f32; 2]; 64];
    comp.process_slice(&ctx, &mut below_buf, None, &mut []);
    assert_eq!(comp.last_active_sparse_count(), 0);
    for frame in &below_buf {
        assert!((frame[0] - 0.05).abs() < 1e-6);
    }

    // 2. Above-threshold signal (0 dBFS = 1.0):
    // Threshold = -12 dB, Ratio = 4:1 => overshoot = 12 dB => gain reduction = -12 * (1 - 0.25) = -9.0 dB!
    // Linear gain = 10^(-9 / 20) = 0.35481338923357547
    let mut above_buf = [[1.0f32; 2]; 64];
    comp.process_slice(&ctx, &mut above_buf, None, &mut []);
    assert_eq!(comp.last_active_sparse_count(), 64);
    let expected_linear_gain = 10.0f32.powf(-9.0 / 20.0);
    for (i, frame) in above_buf.iter().enumerate() {
        assert!(
            (frame[0] - expected_linear_gain).abs() < 1e-5,
            "SparseCompressor output mismatch at sample {}: got {}, expected {}",
            i,
            frame[0],
            expected_linear_gain
        );
    }
    assert!((comp.last_gr_db() - (-9.0)).abs() < 1e-2);
}

// =============================================================================
// 4. TOPOLOGICAL MIXER SCHEDULER (502 TRACKS, KAHN DAG, CYCLE DETECTION, POOL)
// =============================================================================

#[test]
fn test_topological_mixer_scheduler_502_tracks_and_cycle_rejection() {
    let mut tracks: Vec<MixerTrack> = (0..MAX_MIXER_TRACKS).map(MixerTrack::new).collect();
    let mut channels: Vec<ChannelGenerator> = Vec::new();
    assert_eq!(tracks.len(), 502);
    assert_eq!(MASTER_TRACK_INDEX, 0);
    assert_eq!(PREVIEW_TRACK_INDEX, 501);

    // Build a 4-level diamond + bus DAG:
    // Track 1 -> Track 10, Track 1 -> Track 20
    // Track 10 -> Track 30, Track 20 -> Track 30
    // Track 30 -> Track 0 (Master)
    tracks[1].add_send(10, 0.5, false);
    tracks[1].add_send(20, 0.5, false);
    tracks[10].add_send(30, 1.0, false);
    tracks[20].add_send(30, 1.0, false);
    tracks[30].add_send(MASTER_TRACK_INDEX, 1.0, false);

    let sched =
        TopologicalMixerScheduler::rebuild_levels_and_apdc_checked(&mut tracks, &mut channels)
            .expect("DAG should be acyclic");

    let level_of = |trk: usize| -> usize {
        for (lvl_idx, lvl) in sched.levels.iter().enumerate() {
            if lvl.contains(&trk) {
                return lvl_idx;
            }
        }
        panic!("Track {} not found in any topological level", trk);
    };

    assert!(level_of(1) < level_of(10));
    assert!(level_of(1) < level_of(20));
    assert!(level_of(10) < level_of(30));
    assert!(level_of(20) < level_of(30));
    assert!(level_of(30) < level_of(MASTER_TRACK_INDEX));

    // Introduce a 3-hop cycle (30 -> 1) and verify Kahn cycle detector rejects it
    tracks[30].add_send(1, 1.0, false);
    let err =
        TopologicalMixerScheduler::rebuild_levels_and_apdc_checked(&mut tracks, &mut channels)
            .unwrap_err();
    assert!(err.cyclic_tracks.contains(&1));
    assert!(err.cyclic_tracks.contains(&10));
    assert!(err.cyclic_tracks.contains(&30));
}

#[test]
fn test_multithreaded_mixer_engine_parity_with_single_threaded() {
    let build_and_run = |workers: usize| -> Vec<StereoFrame> {
        let (mut engine, _cmd, _telem) = MixerEngine::new(48_000.0, 128.0, workers);
        for trk in 1..=8 {
            engine.channels.push(ChannelGenerator::new(
                trk as u32,
                trk,
                Box::new(DeterministicGeneratorNode::new(trk * 4, 0.25, false)),
            ));
            engine.tracks[trk].add_send(MASTER_TRACK_INDEX, 0.5, false);
        }
        engine.rebuild_graph().unwrap();
        let mut out = [[0.0f32; 2]; 128];
        engine.process_audio_block(&mut out);
        engine.process_audio_block(&mut out);
        out.to_vec()
    };

    let single = build_and_run(0);
    let multi = build_and_run(3);
    for i in 0..128 {
        assert!((single[i][0] - multi[i][0]).abs() < 1e-6);
        assert!((single[i][1] - multi[i][1]).abs() < 1e-6);
    }
}

// =============================================================================
// 5. VISUAL TELEMETRY FRAME (128 KB / 131,072 BYTES) & TRIPLE BUFFER
// =============================================================================

#[test]
fn test_visual_telemetry_frame_128kb_layout_and_triple_buffer() {
    assert_eq!(std::mem::size_of::<VisualTelemetryFrame>(), 131_072);
    assert_eq!(VISUAL_TELEMETRY_FRAME_BYTES, 131_072);
    assert_eq!(std::mem::align_of::<VisualTelemetryFrame>(), 64);

    let (mut input, mut output) = create_telemetry_triple_buffer();
    {
        let buf = input.input_buffer_mut();
        buf.header.frame_seq = 42;
        buf.header.ppq_tick = 384;
        buf.header.bpm = 150.0;
        buf.track_meters[1].peak_l_db = -3.01;
        buf.proq4_spectrum.post_db[512] = -18.5;
        buf.soothe3_reduction.reduction_ys[10][0] = -6.25;
        input.publish();
    }

    let snap = output.read();
    assert_eq!(snap.header.frame_seq, 42);
    assert_eq!(snap.header.ppq_tick, 384);
    assert!((snap.header.bpm - 150.0).abs() < 1e-6);
    assert!((snap.track_meters[1].peak_l_db - (-3.01)).abs() < 1e-5);
    assert!((snap.proq4_spectrum.post_db[512] - (-18.5)).abs() < 1e-5);
    assert!((snap.soothe3_reduction.reduction_ys[10][0] - (-6.25)).abs() < 1e-5);
    assert_eq!(snap.as_bytes().len(), 131_072);
}

// =============================================================================
// 6. REGRESSION & EDGE-CASE ATTACK TESTS (REVIEWER VERIFICATION SUITE)
// =============================================================================

#[test]
fn test_combined_point1_generator_and_point2_hardware_input_diamond_null_cancellation() {
    let sample_rate = 48_000.0;
    let (mut engine, _cmd_prod, _telem_out) = MixerEngine::new(sample_rate, 128.0, 0);

    // Track 1 has BOTH a ChannelGenerator (Point 1, latency = 12, gain = +0.5)
    // AND direct Hardware Input (`has_hw_input = true`, Point 2, latency = 35, gain = +0.5),
    // PLUS it receives an upstream send from Track 3 (`has_hw_input = true`, latency = 19,
    // FX slot latency = 27 => rack_out = 35 + 27 = 62, gain = +0.5).
    // Track 2 receives ChannelGenerator 1 (latency = 44, gain = -1.5, inverted) and routes to Master 0.
    // All 3 positive branches (+0.5 + 0.5 + 0.5 = +1.5) must align with Track 2 (-1.5) to < -140 dBFS!
    engine.channels.push(ChannelGenerator::new(
        0,
        1,
        Box::new(DeterministicGeneratorNode::new(12, 0.5, false)),
    ));
    engine.channels.push(ChannelGenerator::new(
        1,
        2,
        Box::new(DeterministicGeneratorNode::new(44, 1.5, true)),
    ));

    engine.tracks[1].has_hw_input = true;
    engine.tracks[1].hw_input_latency_samples = 35;
    engine.tracks[1]
        .add_plugin_slot(Box::new(FixedLatencyTransformNode::new(14, 1.0, false)), 0.5)
        .unwrap();

    engine.tracks[3].has_hw_input = true;
    engine.tracks[3].hw_input_latency_samples = 19;
    engine.tracks[3]
        .add_plugin_slot(Box::new(FixedLatencyTransformNode::new(27, 1.0, false)), 1.0)
        .unwrap();

    engine.tracks[3].add_send(1, 1.0, false);
    engine.tracks[1].add_send(MASTER_TRACK_INDEX, 1.0, false);
    engine.tracks[2].add_send(MASTER_TRACK_INDEX, 1.0, false);
    engine.rebuild_graph().unwrap();

    // Helper generators to synthesize the simulated hardware input streams at their declared hardware latencies
    let mut hw_gen_trk1 = DeterministicGeneratorNode::new(35, 0.5, false);
    let mut hw_gen_trk3 = DeterministicGeneratorNode::new(19, 0.5, false);

    let block_size = 256;
    let mut master_out = vec![[0.0f32; 2]; block_size];
    for block_idx in 0..8 {
        let ctx = ProcessContext {
            sample_rate,
            current_sample_pos: (block_idx * block_size) as i64,
            ppq_tick: 0,
            bpm: 128.0,
            slice_frames: block_size,
            is_playing: true,
            is_offline_rendering: false,
        };
        hw_gen_trk1.process_slice(
            &ctx,
            &mut engine.tracks[1].hw_input_buf[..block_size],
            None,
            &mut [],
        );
        hw_gen_trk3.process_slice(
            &ctx,
            &mut engine.tracks[3].hw_input_buf[..block_size],
            None,
            &mut [],
        );
        engine.process_audio_block(&mut master_out);
    }

    let (db_l, db_r) = rms_dbfs(&master_out);
    assert!(
        db_l < -140.0 && db_r < -140.0,
        "Combined Point 1 + Point 2 + Diamond Send null test failed: L = {:.2} dBFS, R = {:.2} dBFS",
        db_l,
        db_r
    );
}

#[test]
fn test_midstream_slot_bypass_and_dynamic_latency_recompute_apdc_null_cancellation() {
    let sample_rate = 48_000.0;
    let (mut engine, mut cmd_prod, _telem_out) = MixerEngine::new(sample_rate, 128.0, 0);

    engine.channels.push(ChannelGenerator::new(
        0,
        1,
        Box::new(DeterministicGeneratorNode::new(10, 1.0, false)),
    ));
    engine.channels.push(ChannelGenerator::new(
        1,
        2,
        Box::new(DeterministicGeneratorNode::new(10, 1.0, true)),
    ));

    // Track 1 has two FX slots: Slot 0 (latency = 31) and Slot 1 (latency = 47)
    engine.tracks[1]
        .add_plugin_slot(Box::new(FixedLatencyTransformNode::new(31, 1.0, false)), 1.0)
        .unwrap();
    engine.tracks[1]
        .add_plugin_slot(Box::new(FixedLatencyTransformNode::new(47, 1.0, false)), 0.6)
        .unwrap();

    // Track 2 has one FX slot: Slot 0 (latency = 18)
    engine.tracks[2]
        .add_plugin_slot(Box::new(FixedLatencyTransformNode::new(18, 1.0, false)), 1.0)
        .unwrap();

    engine.tracks[1].add_send(MASTER_TRACK_INDEX, 1.0, false);
    engine.tracks[2].add_send(MASTER_TRACK_INDEX, 1.0, false);
    engine.rebuild_graph().unwrap();

    let mut master_out = [[0.0f32; 2]; 256];
    for _ in 0..4 {
        engine.process_audio_block(&mut master_out);
    }
    let (db_l0, db_r0) = rms_dbfs(&master_out);
    assert!(db_l0 < -140.0 && db_r0 < -140.0);

    // 1. Mid-stream bypass of Track 1 Slot 1 (latency drops by 47 samples on Track 1!)
    cmd_prod
        .push(EngineCommand::SetSlotEnabled {
            track_index: 1,
            slot_index: 1,
            enabled: false,
        })
        .unwrap();
    for _ in 0..4 {
        engine.process_audio_block(&mut master_out);
    }
    let (db_l1, db_r1) = rms_dbfs(&master_out);
    assert!(
        db_l1 < -140.0 && db_r1 < -140.0,
        "Mid-stream SetSlotEnabled(false) broke APDC phase cancellation: L = {:.2} dBFS, R = {:.2} dBFS",
        db_l1,
        db_r1
    );

    // 2. Mid-stream dynamic latency change on Track 2 Slot 0 via param_id = 2 (latency -> 96 samples)
    cmd_prod
        .push(EngineCommand::SetPluginParamNormalized {
            track_index: 2,
            slot_index: 0,
            param_id: 2,
            norm_val: 96.0 / 256.0,
            target_sample_pos: None,
        })
        .unwrap();
    for _ in 0..4 {
        engine.process_audio_block(&mut master_out);
    }
    let (db_l2, db_r2) = rms_dbfs(&master_out);
    assert!(
        db_l2 < -140.0 && db_r2 < -140.0,
        "Dynamic plugin latency change broke APDC phase cancellation: L = {:.2} dBFS, R = {:.2} dBFS",
        db_l2,
        db_r2
    );
}

#[test]
fn test_seek_position_backward_and_forward_resynchronizes_ppq_slicer() {
    let sample_rate = 48_000.0;
    let (mut engine, mut cmd_prod, _telem_out) = MixerEngine::new(sample_rate, 120.0, 0);
    // At 120 BPM, 48 kHz, 96 PPQ => 1 tick = 250.0 samples (aligned to 248 / 256)

    let mut master_out = [[0.0f32; 2]; 512];
    for _ in 0..20 {
        engine.process_audio_block(&mut master_out);
    }
    assert_eq!(engine.slicer.current_sample_pos, 10_240);
    assert!(engine.slicer.global_ppq_tick >= 40);

    // 1. Seek BACKWARD to sample 0, tick 0
    cmd_prod
        .push(EngineCommand::SeekPosition {
            sample_pos: 0,
            ppq_tick: 0,
        })
        .unwrap();
    engine.process_audio_block(&mut master_out);
    assert_eq!(engine.slicer.current_sample_pos, 512);
    assert!(
        engine.slicer.global_ppq_tick >= 2 && engine.slicer.global_ppq_tick <= 3,
        "After backward seek to 0, 512 samples at 250 samples/tick should advance 2..3 ticks, got {}",
        engine.slicer.global_ppq_tick
    );

    // 2. Seek FORWARD to sample 96_000, tick 384
    cmd_prod
        .push(EngineCommand::SeekPosition {
            sample_pos: 96_000,
            ppq_tick: 384,
        })
        .unwrap();
    engine.process_audio_block(&mut master_out);
    assert_eq!(engine.slicer.current_sample_pos, 96_512);
    assert!(
        engine.slicer.global_ppq_tick >= 386 && engine.slicer.global_ppq_tick <= 387,
        "After forward seek to 96_000 (tick 384), 512 samples should advance 2..3 ticks, got {}",
        engine.slicer.global_ppq_tick
    );
}

#[test]
fn test_subslice_sample_accurate_automation_with_preceding_slot_latency() {
    let sample_rate = 48_000.0;
    let (mut engine, mut cmd_prod, _telem_out) = MixerEngine::new(sample_rate, 60.0, 0);

    // Channel 0 outputs constant +1.0 with 0 latency into Track 1
    struct DcOneNode;
    impl UnifiedDspNode for DcOneNode {
        fn name(&self) -> &'static str {
            "DcOne"
        }
        fn latency_samples(&self) -> usize {
            0
        }
        fn prepare(&mut self, _sample_rate: f64, _max_block_size: usize) {}
        fn reset_state(&mut self) {}
        fn set_parameter_normalized(&mut self, _param_id: u32, _norm_val: f32) {}
        fn process_slice(
            &mut self,
            _ctx: &ProcessContext,
            audio: &mut [StereoFrame],
            _sc: Option<&[StereoFrame]>,
            _aux: &mut [&mut [StereoFrame]],
        ) {
            for f in audio.iter_mut() {
                *f = [1.0, 1.0];
            }
        }
    }

    engine
        .channels
        .push(ChannelGenerator::new(0, 1, Box::new(DcOneNode)));

    // Track 1 Slot 0 has latency = 20 samples (gain = 1.0)
    // Track 1 Slot 1 has latency = 0 samples (initial gain = 0.25, automated via param_id = 0)
    engine.tracks[1]
        .add_plugin_slot(Box::new(FixedLatencyTransformNode::new(20, 1.0, false)), 1.0)
        .unwrap();
    engine.tracks[1]
        .add_plugin_slot(Box::new(FixedLatencyTransformNode::new(0, 0.25, false)), 1.0)
        .unwrap();
    engine.tracks[1].add_send(MASTER_TRACK_INDEX, 1.0, false);
    engine.rebuild_graph().unwrap();

    // Warm up delay line in Slot 0 for 1 block (128 samples: 0..128)
    let mut master_out = [[0.0f32; 2]; 128];
    engine.process_audio_block(&mut master_out);
    assert!((master_out[127][0] - 0.25).abs() < 1e-6);

    // Block 1 covers samples 128..256.
    // Schedule a parameter change on Track 1 Slot 1 at raw musical sample 145 (offset 17 into Block 1).
    // Because Slot 0 precedes Slot 1 with latency = 20 samples, `slot_input_arrival_latency(1)` = 20,
    // so the automation event must fire at exact engine sample `145 + 20 = 165` (offset `165 - 128 = 37` in Block 1)
    // setting Slot 1's gain to `norm_val = 0.75`!
    cmd_prod
        .push(EngineCommand::SetPluginParamNormalized {
            track_index: 1,
            slot_index: 1,
            param_id: 0,
            norm_val: 0.75,
            target_sample_pos: Some(145),
        })
        .unwrap();

    engine.process_audio_block(&mut master_out);

    for i in 0..37 {
        assert!(
            (master_out[i][0] - 0.25).abs() < 1e-6,
            "Sample {} before compensated target 37 changed early: got {}",
            i,
            master_out[i][0]
        );
    }
    for i in 37..128 {
        assert!(
            (master_out[i][0] - 0.75).abs() < 1e-6,
            "Sample {} at/after compensated target 37 did not update: got {}",
            i,
            master_out[i][0]
        );
    }
}

#[test]
fn test_multislice_peak_meter_retention_and_patcher_param_connection() {
    let sample_rate = 48_000.0;
    // At 150 BPM, 96 PPQ, 1 tick = 200 samples -> a 256-sample block is split at sample 200 into [0..200] and [200..256]
    let (mut engine, _cmd_prod, mut telem_out) = MixerEngine::new(sample_rate, 150.0, 0);

    // Inject a transient burst at samples 10..40 (inside first slice 0..200) and 0.0 in second slice 200..256
    engine.tracks[1].has_hw_input = true;
    for i in 10..40 {
        engine.tracks[1].hw_input_buf[i] = [0.8, 0.8];
    }
    let mut comp = SparseCompressorNode::new(-12.0, 4.0, 0.0, 0);
    comp.bypass_envelope_ballistics = true;
    engine.tracks[1]
        .add_plugin_slot(Box::new(comp), 1.0)
        .unwrap();
    engine.tracks[1].add_send(MASTER_TRACK_INDEX, 1.0, false);
    engine.rebuild_graph().unwrap();

    let mut master_out = [[0.0f32; 2]; 256];
    engine.process_audio_block(&mut master_out);

    let snap = telem_out.read();
    assert!(
        snap.track_meters[1].peak_l_db > -20.0,
        "Track 1 peak meter lost transient from first PPQ sub-slice: got {} dB",
        snap.track_meters[1].peak_l_db
    );
    assert!(
        snap.track_meters[1].gr_db < -4.0,
        "Track 1 gain reduction meter lost SparseCompressorNode GR across PPQ sub-slice: got {} dB",
        snap.track_meters[1].gr_db
    );

    // Also verify PatcherGraphNode ModularParamConnection forwards parameter changes to child nodes
    let mut patcher = PatcherGraphNode::new();
    let child_idx = patcher.add_node(Box::new(FixedLatencyTransformNode::new(0, 0.0, false)));
    assert!(patcher.add_audio_connection(PATCHER_HOST_INPUT_NODE, child_idx, 1.0));
    assert!(patcher.add_audio_connection(child_idx, PATCHER_HOST_OUTPUT_NODE, 1.0));
    patcher.add_param_connection(ModularParamConnection {
        source_controller: 7,
        target_node: child_idx,
        target_param_id: 0, // gain = norm_val
        scale: 0.5,
        offset: 0.25,
    });
    // Sending param_id = 7 with norm_val = 0.5 => mapped = 0.5 * 0.5 + 0.25 = 0.5 => gain = 0.5
    patcher.set_parameter_normalized(7, 0.5);
    let ctx = ProcessContext::default();
    let mut buf = [[1.0f32; 2]; 16];
    patcher.process_slice(&ctx, &mut buf, None, &mut []);
    assert!((buf[0][0] - 0.5).abs() < 1e-6);
}

