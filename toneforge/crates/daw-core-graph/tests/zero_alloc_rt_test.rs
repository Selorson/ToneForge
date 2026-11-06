//! Strict Zero-Allocation Real-Time Audio Thread Verification Test.
//!
//! Installs a custom `#[global_allocator]` that intercepts every heap allocation,
//! deallocation, and reallocation on the calling thread and globally, proving that
//! `MixerEngine::process_audio_block` performs **0 heap allocations** across an entire
//! multi-block real-time render cycle (including 5-Point APDC, HybridPpqSlicer,
//! TensionSpline, InternalControllerBus, SparseCompressorNode, PatcherGraphNode,
//! rtrb command ingestion, and TripleBuffer 128 KB VisualTelemetryFrame publication).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use daw_core_graph::{
    AudioStemClipPlayerNode, AutomationClipArticulator, ChannelGenerator, ControllerLink,
    DeterministicGeneratorNode, EngineCommand, FixedLatencyTransformNode, MixerEngine,
    PATCHER_HOST_INPUT_NODE, PATCHER_HOST_OUTPUT_NODE, PatcherGraphNode, PianoRollNote,
    PianoRollPattern, PlaylistClip, PlaylistClipKind, PolyphonicSynthGeneratorNode,
    SparseCompressorNode, TempoMap, TensionControlPoint, TensionSpline, UnifiedDspNode,
    normalized_to_fixed30,
};

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
fn test_strict_zero_heap_allocation_in_realtime_render_cycle() {
    let sample_rate = 48_000.0;
    let (mut engine, mut cmd_prod, mut telem_out) = MixerEngine::new(sample_rate, 128.0, 0);

    // 1. Configure Channel Generators with distinct latencies (Point 1 APDC), including Phase 5
    //    PolyphonicSynthGeneratorNode (with 32-sample lookahead) and AudioStemClipPlayerNode
    engine
        .add_channel_generator(ChannelGenerator::new(
            0,
            1,
            Box::new(DeterministicGeneratorNode::new(16, 0.8, false)),
        ))
        .unwrap();
    engine
        .add_channel_generator(ChannelGenerator::new(
            1,
            2,
            Box::new(DeterministicGeneratorNode::new(48, 0.6, false)),
        ))
        .unwrap();

    let mut poly_synth = PolyphonicSynthGeneratorNode::new(2, sample_rate).with_lookahead_latency(32);
    poly_synth.prepare(sample_rate, 256);
    engine
        .add_channel_generator(ChannelGenerator::new(2, 1, Box::new(poly_synth)))
        .unwrap();

    let stem_samples = vec![[0.25f32, -0.25f32]; 4096];
    let mut stem_player =
        AudioStemClipPlayerNode::new(10, "ZeroAllocStem", stem_samples).with_lookahead_latency(24);
    stem_player.prepare(sample_rate, 256);
    engine
        .add_channel_generator(ChannelGenerator::new(3, 2, Box::new(stem_player)))
        .unwrap();

    // Configure Phase 5 Piano Roll Pattern (with FL Studio 26 Slide note), Playlist Clips, and TensionSpline Articulator
    let mut pattern = PianoRollPattern::new(1, "RT Zero Alloc Pattern", 2, 96);
    pattern.add_note(PianoRollNote::new(1, 0, 24, 60, 0.9));
    pattern.add_note(PianoRollNote::new_slide(2, 4, 16, 72, 1.0));
    engine.sequencer.upsert_pattern(pattern);
    engine.sequencer.upsert_clip(PlaylistClip {
        clip_id: 1,
        track_row: 0,
        kind: PlaylistClipKind::Pattern,
        source_id: 1,
        start_tick: 0,
        duration_ticks: 384,
        clip_offset_ticks: 0,
        gain: 1.0,
        muted: false,
    });
    let spline = TensionSpline::new(vec![
        TensionControlPoint::new(0.0, 0.2, 0.35),
        TensionControlPoint::new(64.0, 0.85, -0.35),
    ]);
    engine
        .sequencer
        .upsert_automation_articulator(AutomationClipArticulator::new(0, spline));

    // 2. Attach SparseCompressorNode (with lookahead) and PatcherGraphNode onto Mixer Tracks (Point 3 APDC)
    let comp = SparseCompressorNode::new(-18.0, 4.0, 2.0, 32);
    engine.tracks[1]
        .add_plugin_slot(Box::new(comp), 0.65)
        .unwrap(); // Exercises parallel Dry/Wet APDC Point 3

    let mut patcher = PatcherGraphNode::new();
    let idx_a = patcher.add_node(Box::new(FixedLatencyTransformNode::new(24, 1.0, false)));
    let idx_b = patcher.add_node(Box::new(SparseCompressorNode::new(-12.0, 3.0, 0.0, 12)));
    assert!(patcher.add_audio_connection(PATCHER_HOST_INPUT_NODE, idx_a, 0.5));
    assert!(patcher.add_audio_connection(PATCHER_HOST_INPUT_NODE, idx_b, 0.5));
    assert!(patcher.add_audio_connection(idx_a, PATCHER_HOST_OUTPUT_NODE, 1.0));
    assert!(patcher.add_audio_connection(idx_b, PATCHER_HOST_OUTPUT_NODE, 1.0));
    engine.tracks[2]
        .add_plugin_slot(Box::new(patcher), 1.0)
        .unwrap();

    // 3. Configure multi-hop Mixer DAG routing (Points 2, 4, 5 APDC)
    engine.tracks[1].add_send(3, 0.8, false);
    engine.tracks[1].add_send(0, 0.2, false);
    engine.tracks[2].add_send(3, 0.8, false);
    engine.tracks[3].add_send(0, 1.0, false);
    engine.rebuild_graph().unwrap();

    // 4. Configure TempoMap and 30-Bit Fixed-Point InternalControllerBus link (pre-allocated before RT loop)
    let mut tempo_map = TempoMap::new(128.0);
    tempo_map.add_point(0, 128.0);
    tempo_map.add_point(48, 140.0);
    engine.slicer.tempo_map = Some(tempo_map);

    engine.controller_bus.add_link(ControllerLink {
        controller_id: 0,
        target_track: 1,
        slot_index: 0,
        param_id: 0,
        min_out_norm: 0.2,
        max_out_norm: 0.9,
        inverted: false,
    });

    // Pre-enqueue lock-free commands into the SPSC ring buffer before entering RT section
    cmd_prod
        .push(EngineCommand::SetTransportPlaying(true))
        .unwrap();
    cmd_prod.push(EngineCommand::SetBpm(132.0)).unwrap();
    cmd_prod
        .push(EngineCommand::SetTrackGain {
            track_index: 1,
            gain_linear: 0.95,
        })
        .unwrap();
    cmd_prod
        .push(EngineCommand::SetPluginParamNormalized {
            track_index: 1,
            slot_index: 0,
            param_id: 0,
            norm_val: 0.75,
            target_sample_pos: Some(64),
        })
        .unwrap();
    cmd_prod
        .push(EngineCommand::SetInternalControllerFixed30 {
            controller_id: 0,
            value_fixed30: normalized_to_fixed30(0.625),
        })
        .unwrap();
    cmd_prod
        .push(EngineCommand::SetSlotEnabled {
            track_index: 1,
            slot_index: 0,
            enabled: false,
        })
        .unwrap();
    cmd_prod
        .push(EngineCommand::SetSlotEnabled {
            track_index: 1,
            slot_index: 0,
            enabled: true,
        })
        .unwrap();
    cmd_prod.push(EngineCommand::RefreshApdc).unwrap();

    let mut master_out = [[0.0f32; 2]; 256];

    // =========================================================================
    // ENTER STRICT REAL-TIME ZERO-ALLOCATION GUARD REGION
    // =========================================================================
    begin_rt_no_alloc_region();

    for block in 0..64 {
        if block == 32 {
            // Also test dynamic mid-stream slot bypass + seek + APDC recompute inside the RT loop
            let _ = cmd_prod.push(EngineCommand::SetSlotEnabled {
                track_index: 1,
                slot_index: 0,
                enabled: false,
            });
            let _ = cmd_prod.push(EngineCommand::SeekPosition {
                sample_pos: 4096,
                ppq_tick: 16,
            });
            let _ = cmd_prod.push(EngineCommand::SetSlotEnabled {
                track_index: 1,
                slot_index: 0,
                enabled: true,
            });
        }
        engine.process_audio_block(&mut master_out);
    }

    let (alloc_count, alloc_bytes, dealloc_count) = end_rt_no_alloc_region();
    // =========================================================================

    assert_eq!(
        alloc_count, 0,
        "VIOLATION: Real-time audio render performed {} heap allocations ({} bytes)!",
        alloc_count, alloc_bytes
    );
    assert_eq!(
        alloc_bytes, 0,
        "VIOLATION: Real-time audio render allocated {} bytes on the heap!",
        alloc_bytes
    );
    assert_eq!(
        dealloc_count, 0,
        "VIOLATION: Real-time audio render performed {} heap deallocations!",
        dealloc_count
    );

    // Verify that audio was actually synthesized, processed, and published to Telemetry
    let energy: f32 = master_out
        .iter()
        .map(|frame| frame[0] * frame[0] + frame[1] * frame[1])
        .sum();
    assert!(
        energy > 1e-4,
        "Master output should contain non-zero synthesized audio energy, got {}",
        energy
    );

    let frame = telem_out.read();
    assert_eq!(frame.header.frame_seq, 64);
    assert!(frame.header.ppq_tick > 0);
}
