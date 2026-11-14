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
        ..Default::default()
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

#[test]
fn gate6_1000_live_rt_graph_mutations_zero_alloc_zero_dealloc() {
    use daw_core_graph::{
        EmphasisLimiterNode, GrossBeatTimeBufferNode, LuxeVerbFdnShimmerNode, MixerSendRoute,
        PluginRackSlot, RtGraphMutation,
    };

    let sample_rate = 48_000.0;
    let (mut engine, mut cmd_prod, mut mut_prod, _rt_gc, _telem_out) =
        MixerEngine::new_with_rt_gc(sample_rate, 128.0, 0);

    // Seed initial channels, native FX (LuxeVerb, GrossBeat, Emphasis), and routing before RT loop
    let mut verb = LuxeVerbFdnShimmerNode::new();
    verb.shimmer_mix = 0.35;
    verb.prepare(sample_rate, 256);

    let mut gb = GrossBeatTimeBufferNode::new(240_000);
    gb.prepare(sample_rate, 256);

    let mut emph = EmphasisLimiterNode::new(64, -1.0);
    emph.prepare(sample_rate, 256);

    engine
        .add_channel_generator(ChannelGenerator::new(
            0,
            1,
            Box::new(DeterministicGeneratorNode::new(8, 0.5, false)),
        ))
        .unwrap();
    engine.tracks[1].add_plugin_slot(Box::new(verb), 0.35).unwrap();
    engine.tracks[1].add_plugin_slot(Box::new(gb), 1.0).unwrap();
    engine.tracks[0].add_plugin_slot(Box::new(emph), 1.0).unwrap();
    engine.tracks[1].add_send(2, 0.75, false);
    engine.tracks[2].add_send(0, 1.0, false);
    engine.rebuild_graph().unwrap();

    cmd_prod
        .push(EngineCommand::SetTransportPlaying(true))
        .unwrap();

    // Spawn a host mutation producer thread that fires 1,000 live RtGraphMutation ops
    // (plugin insert/replace/remove, send add/update/remove, channel upsert/remove,
    // pattern/note/clip/articulator mutations) concurrently while the RT thread renders!
    let producer_handle = std::thread::spawn(move || {
        let mut pushed = 0usize;
        for i in 0..1_000usize {
            let mutation = match i % 10 {
                0 => {
                    let mut slot = PluginRackSlot::new(
                        Box::new(FixedLatencyTransformNode::new((i % 64) + 4, 0.9, false)),
                        0.8,
                    );
                    slot.plugin.prepare(sample_rate, 256);
                    RtGraphMutation::SetOrReplacePluginSlot {
                        track_index: 2,
                        slot_index: 0,
                        slot: Box::new(Some(slot)),
                    }
                }
                1 => RtGraphMutation::RemovePluginSlot {
                    track_index: 2,
                    slot_index: 0,
                    recycle_slot: Box::new(None),
                },
                2 => RtGraphMutation::AddOrUpdateSendRoute {
                    src_track: 1,
                    dest_track: 3,
                    gain: 0.5,
                    is_sidechain_only: false,
                    route_carrier: Box::new(Some(MixerSendRoute::new(3, 0.5, false))),
                },
                3 => RtGraphMutation::RemoveSendRoute {
                    src_track: 1,
                    dest_track: 3,
                    is_sidechain_only: false,
                    recycle_carrier: Box::new(None),
                },
                4 => {
                    let mut ch = ChannelGenerator::new(
                        10,
                        2,
                        Box::new(DeterministicGeneratorNode::new(12, 0.3, false)),
                    );
                    ch.generator.prepare(sample_rate, 256);
                    RtGraphMutation::UpsertChannel {
                        channel: Box::new(Some(ch)),
                    }
                }
                5 => RtGraphMutation::RemoveChannel {
                    channel_id: 10,
                    recycle_channel: Box::new(None),
                },
                6 => {
                    let mut pat = PianoRollPattern::new(2, "LiveMutPat", 0, 96);
                    pat.add_note(PianoRollNote::new(1, 0, 24, 64, 0.85));
                    RtGraphMutation::UpsertPattern {
                        pattern: Box::new(pat),
                    }
                }
                7 => RtGraphMutation::AddOrUpdateNote {
                    pattern_id: 2,
                    note: PianoRollNote::new(2, 24, 24, 67, 0.80),
                },
                8 => RtGraphMutation::UpsertPlaylistClip {
                    clip: PlaylistClip {
                        clip_id: 50,
                        track_row: 1,
                        kind: PlaylistClipKind::Pattern,
                        source_id: 2,
                        start_tick: 0,
                        duration_ticks: 192,
                        clip_offset_ticks: 0,
                        gain: 0.9,
                        muted: false,
                        ..Default::default()
                    },
                },
                _ => {
                    let sp = TensionSpline::new(vec![
                        TensionControlPoint::new(0.0, 0.25, 0.0),
                        TensionControlPoint::new(96.0, 0.75, 0.0),
                    ]);
                    RtGraphMutation::UpsertAutomationArticulator {
                        articulator: Box::new(AutomationClipArticulator::new(1, sp)),
                    }
                }
            };

            while mut_prod.is_full() {
                std::thread::yield_now();
            }
            assert!(mut_prod.push(mutation).is_ok());
            pushed += 1;
            if pushed % 32 == 0 {
                std::thread::yield_now();
            }
        }
        pushed
    });

    let mut master_out = [[0.0f32; 2]; 256];
    let t0 = std::time::Instant::now();
    let mut blocks = 0usize;

    begin_rt_no_alloc_region();
    loop {
        engine.process_audio_block(&mut master_out);
        blocks += 1;
        if producer_handle.is_finished() && blocks >= 128 {
            // Run 8 extra drain blocks to ensure 100% of queued mutations were ingested inside the guard
            for _ in 0..8 {
                engine.process_audio_block(&mut master_out);
                blocks += 1;
            }
            break;
        }
    }
    let (alloc_count, alloc_bytes, dealloc_count) = end_rt_no_alloc_region();

    let total_elapsed = t0.elapsed();
    let pushed = producer_handle.join().unwrap();
    assert_eq!(pushed, 1_000);

    assert_eq!(
        alloc_count, 0,
        "Gate 6 VIOLATION: RT thread performed {alloc_count} heap allocations ({alloc_bytes} bytes) during 1,000 live graph mutations!"
    );
    assert_eq!(
        dealloc_count, 0,
        "Gate 6 VIOLATION: RT thread performed {dealloc_count} heap deallocations (drop/HeapFree) during 1,000 live graph mutations!"
    );

    let avg_us_per_block = total_elapsed.as_secs_f64() * 1_000_000.0 / (blocks.max(1) as f64);
    assert!(
        avg_us_per_block < 1_500.0,
        "Gate 6 latency budget exceeded: avg block time {avg_us_per_block:.1} us >= 1500 us"
    );
}

/// Retired resources that own plug-in instances go to the host's disposer (its plug-in thread);
/// everything else is still freed on `toneforge-rt-gc`.
#[test]
fn test_rt_gc_routes_retired_plugins_to_host_disposer() {
    use daw_core_graph::{RetiredRtResource, create_rt_mutation_channels};
    use std::sync::{Arc, Mutex};
    let (_mp, _mc, mut gc_prod, gc) = create_rt_mutation_channels(64, 128);
    let routed: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
    let routed_bg = Arc::clone(&routed);
    gc.set_plugin_disposer(Some(Box::new(move |r| {
        let kind = match r {
            RetiredRtResource::PluginSlot(_) => "slot",
            RetiredRtResource::Channel(_) => "channel",
            _ => "other",
        };
        routed_bg.lock().unwrap().push(kind);
    })));
    assert!(gc_prod.push(RetiredRtResource::PluginSlot(Box::new(None))).is_ok());
    assert!(gc_prod.push(RetiredRtResource::AudioFrames(Box::new(vec![[0.0; 2]; 16]))).is_ok());
    assert!(gc_prod.push(RetiredRtResource::Channel(Box::new(None))).is_ok());
    gc.wake();
    let t0 = std::time::Instant::now();
    while gc.freed_count() < 3 && t0.elapsed() < std::time::Duration::from_secs(5) {
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    assert_eq!(gc.freed_count(), 3);
    assert_eq!(*routed.lock().unwrap(), vec!["slot", "channel"]);
}
