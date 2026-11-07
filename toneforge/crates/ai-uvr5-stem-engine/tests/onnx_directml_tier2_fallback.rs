//! Hardware-in-the-Loop Verification of Tier 2 DirectX 12 (`DirectMLExecutionProvider`) Fallback
//! (`onnx_directml_tier2_fallback.rs`).
//!
//! Runs in an isolated integration test binary so `TONEFORGE_FORCE_GPU_TIER=directml` initializes
//! `GpuRuntimeManager::global()` onto `<exe_dir>/gpu_runtime/directml/onnxruntime.dll` + `DirectML.dll`,
//! verifying that when Tier 1 CUDA is unavailable or skipped, Tier 2 DirectML DX12 attaches cleanly
//! and executes real ONNX neural inference on GPU.

use ai_uvr5_stem_engine::{
    BSRNN_HOP_SIZE, BsrnnInputHop, BsrnnOutputHop, GpuExecutionTier, GpuRuntimeManager,
    OzoneBsrnnOnnxSeparator, OzoneDelimitOnnxProcessor, Uvr5MdxOnnxSeparator,
};
use std::f32::consts::TAU;

#[test]
fn test_tier2_directml_dx12_fallback_and_real_onnx_inference() {
    // Must be set before `GpuRuntimeManager::global()` is touched in this process.
    // SAFETY: Single test function in this dedicated integration test binary before any worker threads spawn.
    unsafe {
        std::env::set_var("TONEFORGE_FORCE_GPU_TIER", "directml");
    }

    let mgr = GpuRuntimeManager::global();
    let status = mgr.status();
    println!(
        "[Tier 2 DirectML Telemetry] Tier={} ({}) | Device={} | DLL={} | Note={}",
        status.tier_number,
        status.provider_name,
        status.device_name,
        status.runtime_dll_path,
        status.diagnostic_note
    );

    assert_eq!(
        status.tier,
        GpuExecutionTier::DirectMlDx12,
        "Expected Tier 2 DirectMlDx12 when Tier 1 CUDA is bypassed, got {:?} ({})",
        status.tier,
        status.diagnostic_note
    );
    assert_eq!(status.tier_number, 2);
    assert_eq!(status.provider_name, "DmlExecutionProvider");
    assert!(
        status.runtime_dll_path.to_ascii_lowercase().contains("directml"),
        "Expected directml/onnxruntime.dll path, got {}",
        status.runtime_dll_path
    );

    // 1. Verify Ozone 12 Delimit (`delimit_exp009_ep080_fixed2.onnx`) runs on DirectML
    let mut delimit = OzoneDelimitOnnxProcessor::from_default_assets()
        .expect("OzoneDelimitOnnxProcessor must load under Tier 2 DirectML");
    assert_eq!(delimit.active_tier(), GpuExecutionTier::DirectMlDx12);
    let test_clip: Vec<[f32; 2]> = (0..4096)
        .map(|n| {
            let t = (n as f32) / 48_000.0;
            let s = (0.9 * (TAU * 220.0 * t).sin()).clamp(-0.6, 0.6);
            [s, s]
        })
        .collect();
    let restored = delimit
        .process_offline(&test_clip, 0.8)
        .expect("Delimit DirectML inference");
    assert_eq!(restored.len(), test_clip.len());
    assert!(restored.iter().all(|f| f[0].is_finite() && f[1].is_finite()));

    // 2. Verify Ozone 12 BSRNN (`ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx`) runs on DirectML
    let mut bsrnn = OzoneBsrnnOnnxSeparator::from_default_assets()
        .expect("OzoneBsrnnOnnxSeparator must load under Tier 2 DirectML");
    assert!(
        matches!(
            bsrnn.active_tier(),
            GpuExecutionTier::DirectMlDx12 | GpuExecutionTier::CpuOrt
        ),
        "Unexpected BSRNN tier under DirectML runtime: {:?}",
        bsrnn.active_tier()
    );
    let mut hop_in = BsrnnInputHop::default();
    let mut hop_out = BsrnnOutputHop::default();
    for i in 0..BSRNN_HOP_SIZE {
        let t = (i as f32) / 48_000.0;
        let x = 0.4 * (TAU * 110.0 * t).sin();
        hop_in.frames[i] = [x, x];
    }
    bsrnn
        .process_hop(&hop_in, &mut hop_out)
        .expect("BSRNN hop inference under DirectML runtime");
    assert!(
        hop_out
            .stems
            .iter()
            .all(|s| s.iter().all(|f| f[0].is_finite() && f[1].is_finite()))
    );

    // 3. Verify UVR5 MDX-Net (`UVR-MDX-NET-Inst_HQ_3.onnx`) runs on DirectML
    let mut inst_hq3 = Uvr5MdxOnnxSeparator::from_inst_hq_3()
        .expect("UVR-MDX-NET-Inst_HQ_3 must load under Tier 2 DirectML");
    assert_eq!(inst_hq3.active_tier(), GpuExecutionTier::DirectMlDx12);
    let short_mix: Vec<[f32; 2]> = (0..8192)
        .map(|n| {
            let t = (n as f32) / 44_100.0;
            let x = 0.35 * (TAU * 65.0 * t).sin();
            [x, x]
        })
        .collect();
    let (voc, inst) = inst_hq3
        .separate_vocal_instrumental(&short_mix, 44_100, false)
        .expect("UVR-MDX-NET-Inst_HQ_3 DirectML inference");
    assert_eq!(voc.len(), short_mix.len());
    assert_eq!(inst.len(), short_mix.len());
    assert!(inst.iter().all(|f| f[0].is_finite() && f[1].is_finite()));
}
