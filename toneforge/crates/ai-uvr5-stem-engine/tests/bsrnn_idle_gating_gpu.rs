//! Idle gating of `RealTimeBsrnnStemNode` with the REAL async ONNX GPU worker.
//!
//! Skips (passes with a message) when no ONNX GPU worker can be created on this machine.
//! The hardware-independent proofs live in `bsrnn_streaming::tests` (inline mock separator).

use ai_uvr5_stem_engine::RealTimeBsrnnStemNode;
use daw_core_graph::{ProcessContext, StereoFrame, UnifiedDspNode};
use std::f32::consts::TAU;

const SR: f64 = 48_000.0;
const BLOCK: usize = 256;

fn tone(n: usize) -> StereoFrame {
    let t = n as f32 / SR as f32;
    let x = 0.35 * (TAU * 60.0 * t).sin() + 0.20 * (TAU * 520.0 * t).sin();
    let y = 0.30 * (TAU * 85.0 * t).sin() + 0.22 * (TAU * 440.0 * t).sin();
    [x, y]
}

/// Processes `blocks` real-time-paced blocks starting at absolute sample `start`, recording the
/// dry input and output into the history vectors. `silent` feeds digital silence.
fn run_blocks(
    node: &mut RealTimeBsrnnStemNode,
    ctx: &ProcessContext,
    start: usize,
    blocks: usize,
    silent: bool,
    input_hist: &mut Vec<StereoFrame>,
    output_hist: &mut Vec<StereoFrame>,
) -> usize {
    let mut block = [[0.0f32; 2]; BLOCK];
    for b in 0..blocks {
        for i in 0..BLOCK {
            block[i] = if silent { [0.0; 2] } else { tone(start + b * BLOCK + i) };
        }
        input_hist.extend_from_slice(&block);
        node.process_slice(ctx, &mut block, None, &mut []);
        output_hist.extend_from_slice(&block);
        // ~real-time cadence (256 frames = 5.33 ms) so the async worker sees live timing.
        std::thread::sleep(std::time::Duration::from_micros(5_000));
    }
    start + blocks * BLOCK
}

fn rms(frames: &[StereoFrame]) -> f64 {
    let s: f64 = frames
        .iter()
        .map(|f| (f[0] as f64).powi(2) + (f[1] as f64).powi(2))
        .sum();
    (s / (2 * frames.len().max(1)) as f64).sqrt()
}

#[test]
fn async_gpu_worker_idle_gating_and_primed_resume() {
    let mut node = match RealTimeBsrnnStemNode::new_with_onnx_gpu(SR) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("SKIP: ONNX GPU worker unavailable: {e}");
            return;
        }
    };
    assert!(node.is_onnx_gpu_worker_active());
    assert!(node.unity_idle_gating());
    let lat = node.latency_samples();
    let ctx = ProcessContext {
        sample_rate: SR,
        ..Default::default()
    };
    let mut inp = Vec::new();
    let mut out = Vec::new();

    // Phase 1: unity bypass on real audio -> zero hops submitted to the GPU.
    let mut pos = run_blocks(&mut node, &ctx, 0, 188, false, &mut inp, &mut out); // ~1 s
    assert!(node.is_idle_gated());
    assert_eq!(node.pushed_hops(), 0, "unity bypass must not submit GPU hops");
    assert_eq!(node.latency_samples(), lat);
    for i in lat..pos {
        assert_eq!(out[i], inp[i - lat], "unity output must be the exact dry delay");
    }

    // Phase 2: mute all stems -> primed restart; once stems fade in, Y = X_delayed - sum(S) ~ 0.
    node.stem_mute = [true; 4];
    let flip = pos;
    let underflow_before = node.underflow_hops;
    pos = run_blocks(&mut node, &ctx, pos, 188, false, &mut inp, &mut out); // ~1 s
    let underflows_first_second = node.underflow_hops - underflow_before;
    pos = run_blocks(&mut node, &ctx, pos, 94, false, &mut inp, &mut out); // ~0.5 s
    assert!(!node.is_idle_gated());
    let hops_phase2 = node.pushed_hops();
    let underflows_phase2 = node.underflow_hops - underflow_before;
    for i in flip..flip + lat - 2560 {
        assert_eq!(out[i], inp[i - lat], "stems invalid during warm-up: exact dry output");
    }
    let settle = flip + (lat - 2560) + 9_600;
    let null_after_resume = rms(&out[settle..pos]);
    let null_last_half = rms(&out[pos - 24_064..pos]);
    eprintln!(
        "phase2: pushed_hops={hops_phase2} underflow_hops={underflows_phase2} \
         (first 1 s: {underflows_first_second}) mute-null rms after resume = \
         {null_after_resume:.3e}, last 0.5 s = {null_last_half:.3e}"
    );
    // The pipeline must self-heal in every build profile.
    assert!(null_last_half < 1e-3, "primed restart never re-locked: {null_last_half}");
    // In unoptimised debug builds the worker's Rust STFT/packing code is barely faster than real
    // time, so the 8-hop priming burst takes ~1 s to catch up (underflows -> stems fade in late).
    // Release builds (what ships) have 0 underflows and must be aligned right after the warm-up.
    if !cfg!(debug_assertions) {
        assert_eq!(underflows_phase2, 0, "primed restart underflowed in release");
        assert!(
            null_after_resume < 1e-3,
            "primed restart misaligned: mute-null rms = {null_after_resume}"
        );
    }

    // Phase 3: back to unity -> gated again once the deltas settle (~100 ms); GPU gets no hops.
    node.stem_mute = [false; 4];
    pos = run_blocks(&mut node, &ctx, pos, 40, false, &mut inp, &mut out);
    assert!(node.is_idle_gated());
    let hops_after_unity = node.pushed_hops();
    pos = run_blocks(&mut node, &ctx, pos, 94, false, &mut inp, &mut out);
    assert_eq!(node.pushed_hops(), hops_after_unity);

    // Phase 4: muted stems + long digital silence -> gated; first onset resumes seamlessly.
    node.stem_mute = [true; 4];
    pos = run_blocks(&mut node, &ctx, pos, 188, true, &mut inp, &mut out);
    assert!(node.is_idle_gated());
    let hops_silence = node.pushed_hops();
    pos = run_blocks(&mut node, &ctx, pos, 94, true, &mut inp, &mut out);
    assert_eq!(node.pushed_hops(), hops_silence, "silence must not submit GPU hops");
    let onset = pos;
    let underflow_before = node.underflow_hops;
    pos = run_blocks(&mut node, &ctx, pos, 188, false, &mut inp, &mut out);
    let underflows_phase4 = node.underflow_hops - underflow_before;
    // Delayed onset arrives at `onset + lat`; the stems are already fully faded in by then.
    let null_onset = rms(&out[onset + lat..onset + lat + 4096]);
    let null_late = rms(&out[onset + lat + 4096..pos]);
    let dry_ref = rms(&inp[onset..onset + 4096]);
    eprintln!(
        "phase4: underflow_hops={underflows_phase4} mute-null rms first 4096 after delayed onset = \
         {null_onset:.3e} (dry {dry_ref:.3e}), later = {null_late:.3e}"
    );
    assert!(null_late < 1e-3, "silence resume misaligned: {null_late}");
    assert!(null_onset < 0.1 * dry_ref, "onset not rebalanced: {null_onset} vs dry {dry_ref}");
}
