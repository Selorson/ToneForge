//! Strict Zero-Allocation Real-Time Audio Thread Verification for `ProQ4Processor` + `Soothe3Processor`
//! running together inside `MixerEngine` (`daw-core-graph`).
//!
//! Verifies:
//! 1. **0 heap allocations (`alloc_count == 0`)** and **0 heap deallocations (`dealloc_count == 0`)**
//!    across 32 blocks (`512` frames/block = `16,384` frames) of real-time processing with:
//!    - `ProQ4Processor` (Dynamic EQ + Spectral Dynamics + Fractional-Slope Biquads + Character Warm + 1024-bin Spectrum Analyzer)
//!    - `Soothe3Processor` (8-Band Hybrid Series/Parallel Sidechain EQ + 4096-point STFT + ERB/Bark Resonance Detector + CepstralMinPhaseReconstructor)
//!    - Sample-accurate parameter automation events and 5-Point APDC alignment.
//! 2. Non-zero visual telemetry publication for both `frame.proq4_spectrum` (`16 KB`) and
//!    `frame.soothe3_reduction` (`8 KB`) over the lock-free `triple_buffer`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use daw_core_graph::{
    ChannelGenerator, DeterministicGeneratorNode, EngineCommand, MixerEngine,
};
use dsp_proq4_eq::{CharacterMode, FilterShape, ProQ4Processor, ProcessingMode};
use dsp_soothe3_spectral::{
    SidechainFilterShape, Soothe3Processor, SootheProcessMode, TimeResolutionQuality,
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
fn test_zero_heap_allocation_proq4_and_soothe3_in_mixer_engine() {
    let sample_rate = 48_000.0;
    let (mut engine, mut cmd_prod, mut telem_out) = MixerEngine::new(sample_rate, 128.0, 0);

    // 1. Add deterministic audio generator routed to Track 1
    engine
        .add_channel_generator(ChannelGenerator::new(
            0,
            1,
            Box::new(DeterministicGeneratorNode::new(0, 0.85, false)),
        ))
        .unwrap();

    // 2. Configure ProQ4Processor with active Fractional-Slope EQ, Dynamic EQ, Spectral Dynamics & Warm Character
    let mut proq4 = Box::new(ProQ4Processor::new());
    proq4.params.global.processing_mode = ProcessingMode::NaturalPhase;
    proq4.params.global.character = CharacterMode::Warm;
    proq4.params.global.auto_gain = true;

    proq4.params.bands[0].used = true;
    proq4.params.bands[0].enabled = true;
    proq4.params.bands[0].shape = FilterShape::Bell;
    proq4.params.bands[0].freq_hz = 1000.0;
    proq4.params.bands[0].gain_db = 6.0;
    proq4.params.bands[0].slope_db_oct = 18.0;
    proq4.params.bands[0].dyn_enabled = true;
    proq4.params.bands[0].dyn_range_db = -9.0;
    proq4.params.bands[0].dyn_threshold_db = -24.0;
    proq4.refresh_band_dsp(0, false);

    proq4.params.bands[1].used = true;
    proq4.params.bands[1].enabled = true;
    proq4.params.bands[1].shape = FilterShape::LowCut;
    proq4.params.bands[1].freq_hz = 80.0;
    proq4.params.bands[1].slope_db_oct = 21.0; // Fractional slope
    proq4.refresh_band_dsp(1, false);
    proq4.refresh_global_aggregates(false);

    // 3. Configure Soothe3Processor with active 8-Band Hybrid Series/Parallel Sidechain EQ & Minimum-Phase Suppressor
    let mut soothe3 = Box::new(Soothe3Processor::new(sample_rate as f32));
    soothe3.params_mut().global.mode = SootheProcessMode::Soft;
    soothe3.params_mut().global.depth = 6.5;
    soothe3.params_mut().global.detail = 7.0;
    soothe3.params_mut().global.quality = TimeResolutionQuality::Normal;
    soothe3.params_mut().sc_bands[0].enabled = true;
    soothe3.params_mut().sc_bands[0].shape = SidechainFilterShape::Bandpass;
    soothe3.params_mut().sc_bands[0].freq_hz = 2500.0;
    soothe3.params_mut().sc_bands[0].depth = 4.0;
    soothe3.sync_dirty_tables();

    // Attach both ProQ4Processor (slot 0) and Soothe3Processor (slot 1) onto Track 1
    engine.tracks[1].add_plugin_slot(proq4, 1.0).unwrap();
    engine.tracks[1].add_plugin_slot(soothe3, 1.0).unwrap();
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

    for block in 0..40 {
        if block == 10 {
            // Test sample-accurate automation of Pro-Q 4 Band 1 Gain and Soothe3 Depth inside the RT loop
            let _ = cmd_prod.push(EngineCommand::SetPluginParamNormalized {
                track_index: 1,
                slot_index: 0,
                param_id: 3, // Band 1 Gain
                norm_val: 0.65,
                target_sample_pos: Some(block * 256 + 64),
            });
            let _ = cmd_prod.push(EngineCommand::SetPluginParamNormalized {
                track_index: 1,
                slot_index: 1,
                param_id: 2, // Soothe3 Depth
                norm_val: 0.80,
                target_sample_pos: Some(block * 256 + 128),
            });
        }
        engine.process_audio_block(&mut master_out);
    }

    let (alloc_count, alloc_bytes, dealloc_count) = end_rt_no_alloc_region();
    // =========================================================================

    assert_eq!(
        alloc_count, 0,
        "VIOLATION: ProQ4 + Soothe3 real-time render performed {} heap allocations ({} bytes)!",
        alloc_count, alloc_bytes
    );
    assert_eq!(alloc_bytes, 0);
    assert_eq!(
        dealloc_count, 0,
        "VIOLATION: ProQ4 + Soothe3 real-time render performed {} heap deallocations!",
        dealloc_count
    );

    // Verify Master output contains processed audio and VisualTelemetryFrame contains both Pro-Q 4 and Soothe3 frames
    let out_energy: f32 = master_out
        .iter()
        .map(|f| f[0] * f[0] + f[1] * f[1])
        .sum();
    assert!(out_energy > 1e-4, "Master output energy was zero");

    let telem = telem_out.read();
    assert_eq!(telem.header.frame_seq, 40);
    let max_proq4_pre_db = telem
        .proq4_spectrum
        .pre_db
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max_proq4_pre_db > -50.0,
        "VisualTelemetryFrame.proq4_spectrum was not updated by MixerEngine"
    );
    let max_soothe3_spec_db = telem
        .soothe3_reduction
        .input_spec_db
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max_soothe3_spec_db > -50.0,
        "VisualTelemetryFrame.soothe3_reduction was not updated by MixerEngine"
    );
}
