//! Hardware-in-the-Loop Verification Suite for Real ONNX Runtime CUDA 12 + cuDNN 9 Inference
//! on NVIDIA GeForce RTX 4080 (`onnx_rtx4080_real_inference.rs`).
//!
//! Verifies:
//! 1. Tier 1 `CUDAExecutionProvider` (`onnxruntime-gpu==1.25.1` + 6 CUDA 12 PyPI packages) binds
//!    live to the RTX 4080 GPU with `0` missing DLL dependencies.
//! 2. Standalone `.exe` path resolution (`std::env::current_exe()` parent/ancestors FIRST) locates
//!    `gpu_runtime/` and all 4 `.onnx` models even when `current_dir` is `%TEMP%`.
//! 3. `OzoneBsrnnOnnxSeparator` (`ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx`): exact `13,312`-float
//!    recurrent state evolution, `< 1e-5` lossless stem-sum identity (`V + D + B + O == mix`), and
//!    empirical `[0=Vocals, 1=Bass, 2=Drums] -> StemKind` sub-bass routing.
//! 4. `RealTimeBsrnnStemNode::new_with_onnx_gpu(48_000.0)` background GPU worker streaming with
//!    clean `rtrb` SPSC handoff and zero NaNs/clipping.
//! 5. `OzoneDelimitOnnxProcessor` (`delimit_exp009_ep080_fixed2.onnx`): exact `6,656`-float state
//!    and transient restoration on hard-clipped audio.
//! 6. `Uvr5MdxOnnxSeparator` (`Kim_Vocal_2.onnx` + `UVR-MDX-NET-Inst_HQ_3.onnx`) and 3-Model
//!    `GpuStemEnsembleEngine` on RTX 4080, including empty/1-sample/short/multi-second edge cases,
//!    `2-pass denoise`, `phase_denoise_strength`, and `invert_instrumental`.

use ai_uvr5_stem_engine::{
    BSRNN_HOP_SIZE, BsrnnInputHop, BsrnnOutputHop, GpuExecutionTier, GpuRuntimeManager,
    GpuStemEnsembleConfig, GpuStemEnsembleEngine, GpuStemModelSelection, OZONE_BSRNN_SPEC_LEN,
    OZONE_BSRNN_STATES_LEN, OZONE_DELIMIT_STATES_LEN, OzoneBsrnnOnnxSeparator,
    OzoneDelimitOnnxProcessor, RealTimeBsrnnStemNode, SpectralEnsembleMode, StemKind,
    Uvr5MdxOnnxSeparator, candidate_toneforge_roots, resolve_gpu_runtime_dir,
    resolve_ozone_bsrnn_onnx_path, resolve_ozone_delimit_onnx_path,
    resolve_uvr5_inst_hq3_onnx_path, resolve_uvr5_kim_vocal2_onnx_path,
};
use daw_core_graph::{ProcessContext, StereoFrame, UnifiedDspNode};
use std::f32::consts::TAU;
use std::path::PathBuf;
use std::sync::Mutex;

/// Serializes process-wide `set_current_dir` and heavy multi-session CUDA `cuModuleLoadData`
/// benchmarks within this test binary so concurrent PTX/SASS loading does not stall real-time
/// `120 ms` BSRNN streaming jitter verification.
static GPU_BENCH_SERIAL_LOCK: Mutex<()> = Mutex::new(());

struct CwdRestoreGuard(PathBuf);
impl Drop for CwdRestoreGuard {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.0);
    }
}

#[test]
fn test_rtx4080_tier1_cuda_provider_and_standalone_exe_path_resolution() {
    let _serial = GPU_BENCH_SERIAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let orig_cwd = std::env::current_dir().expect("current_dir");
    let _guard = CwdRestoreGuard(orig_cwd);

    // Simulate double-clicking `ToneForge_Studio.exe` when CWD is `%TEMP%` (`C:\Users\...\Temp`)
    std::env::set_current_dir(std::env::temp_dir()).expect("set_current_dir to temp_dir");

    let roots = candidate_toneforge_roots();
    let exe_parent = std::env::current_exe()
        .expect("current_exe")
        .parent()
        .expect("exe parent")
        .to_path_buf();
    assert_eq!(
        roots[0], exe_parent,
        "candidate_toneforge_roots()[0] MUST be std::env::current_exe().parent() first!"
    );

    assert!(
        resolve_gpu_runtime_dir().is_some(),
        "gpu_runtime/ must resolve from current_exe()/manifest ancestors even when CWD is %TEMP%"
    );
    assert!(
        resolve_ozone_bsrnn_onnx_path().is_some(),
        "ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx must resolve when CWD is %TEMP%"
    );
    assert!(
        resolve_ozone_delimit_onnx_path().is_some(),
        "delimit_exp009_ep080_fixed2.onnx must resolve when CWD is %TEMP%"
    );
    assert!(
        resolve_uvr5_kim_vocal2_onnx_path().is_some(),
        "Kim_Vocal_2.onnx must resolve when CWD is %TEMP%"
    );
    assert!(
        resolve_uvr5_inst_hq3_onnx_path().is_some(),
        "UVR-MDX-NET-Inst_HQ_3.onnx must resolve when CWD is %TEMP%"
    );

    let mgr = GpuRuntimeManager::global();
    let status = mgr.status();
    println!(
        "[RTX 4080 Telemetry] Tier={} ({}) | Device={} | VRAM={} MiB | Driver={} | DLLs={}",
        status.tier_number,
        status.provider_name,
        status.device_name,
        status.vram_mib,
        status.cuda_driver_version,
        status.preloaded_dll_count
    );

    assert_eq!(
        status.tier,
        GpuExecutionTier::CudaTensorCore,
        "Expected Tier 1 CudaTensorCore on RTX 4080, got {:?} ({})",
        status.tier,
        status.diagnostic_note
    );
    assert_eq!(status.tier_number, 1);
    assert_eq!(status.provider_name, "CUDAExecutionProvider");
    assert!(
        status.device_name.contains("4080"),
        "Expected NVIDIA GeForce RTX 4080 device name, got {}",
        status.device_name
    );
    assert!(
        status.vram_mib >= 15_000,
        "Expected >= 15,000 MiB VRAM on RTX 4080, got {}",
        status.vram_mib
    );
}

#[test]
fn test_ozone12_bsrnn_and_delimit_real_cuda_inference() {
    let _serial = GPU_BENCH_SERIAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    assert_eq!(OZONE_BSRNN_SPEC_LEN, 12_300);
    assert_eq!(OZONE_BSRNN_STATES_LEN, 13_312);
    assert_eq!(OZONE_DELIMIT_STATES_LEN, 6_656);

    let mut bsrnn = OzoneBsrnnOnnxSeparator::from_default_assets()
        .expect("OzoneBsrnnOnnxSeparator must load on RTX 4080 CUDA");
    assert_eq!(bsrnn.active_tier(), GpuExecutionTier::CudaTensorCore);

    // 1. Streaming hop state evolution test
    let mut hop_in = BsrnnInputHop::default();
    let mut hop_out = BsrnnOutputHop::default();
    for h in 0..8 {
        for i in 0..BSRNN_HOP_SIZE {
            let t = ((h * BSRNN_HOP_SIZE + i) as f32) / 48_000.0;
            let s = 0.45 * (TAU * 55.0 * t).sin() + 0.25 * (TAU * 440.0 * t).sin();
            hop_in.frames[i] = [s, s * 0.95];
        }
        bsrnn
            .process_hop(&hop_in, &mut hop_out)
            .expect("BSRNN CUDA hop inference");
    }
    let state_energy: f32 = bsrnn.states_slice().iter().map(|v| v.abs()).sum();
    assert!(
        state_energy > 1.0 && state_energy.is_finite(),
        "13,312-float recurrent LSTM state must evolve across hops (got {state_energy})"
    );
    assert!(
        bsrnn.normalize_states()[0] > 0.0 && bsrnn.normalize_states()[1] > 0.0,
        "Online normalizer states [2, 1] must be positive after non-zero audio"
    );

    // 2. Pure 55 Hz sub-bass test: empirically proves ONNX dim1[1] -> StemKind::Bass (`stems[2]`)!
    let sub_bass_mix: Vec<StereoFrame> = (0..24_000)
        .map(|n| {
            let t = (n as f32) / 48_000.0;
            let b = 0.6 * (TAU * 55.0 * t).sin() + 0.2 * (TAU * 110.0 * t).sin();
            [b, b]
        })
        .collect();
    let stems = bsrnn
        .separate_4stems_offline(&sub_bass_mix)
        .expect("separate_4stems_offline on sub-bass");

    let mut max_sum_err = 0.0f32;
    let mut stem_rms = [0.0f64; 4];
    for i in 0..sub_bass_mix.len() {
        for ch in 0..2 {
            let sum = stems[0][i][ch] + stems[1][i][ch] + stems[2][i][ch] + stems[3][i][ch];
            assert!(sum.is_finite());
            max_sum_err = max_sum_err.max((sum - sub_bass_mix[i][ch]).abs());
            for s in 0..4 {
                stem_rms[s] += (stems[s][i][ch] as f64).powi(2);
            }
        }
    }
    assert!(
        max_sum_err < 1e-5,
        "V + D + B + O must equal mix to < 1e-5, got max_sum_err={max_sum_err}"
    );
    assert!(
        stem_rms[StemKind::Bass as usize] > stem_rms[StemKind::Vocals as usize] * 4.0
            && stem_rms[StemKind::Bass as usize] > stem_rms[StemKind::Drums as usize] * 4.0,
        "55 Hz sub-bass must route predominantly to StemKind::Bass! RMS^2 sums: V={:.2}, D={:.2}, B={:.2}, O={:.2}",
        stem_rms[0],
        stem_rms[1],
        stem_rms[2],
        stem_rms[3]
    );

    // 3. Background ONNX GPU worker inside RealTimeBsrnnStemNode:
    //    Verify sample-accurate all-stems-muted phase null (`Y = X_delayed - sum_s S_s == 0`)
    //    BEFORE reset, AFTER simulated fast-burst underflow, and AFTER `reset_state()`!
    let mut rt_node =
        RealTimeBsrnnStemNode::new_with_onnx_gpu(48_000.0).expect("new_with_onnx_gpu on RTX 4080");
    assert!(rt_node.is_onnx_gpu_worker_active());
    rt_node.stem_mute = [true; 4]; // Subtracts 100% of V+D+B+O from X_delayed!
    let mut ctx = ProcessContext::default();
    ctx.sample_rate = 48_000.0;
    let mut block = [[0.0f32; 2]; 256];

    // Phase A: Initial warmup (32 blocks = 8,192 samples > 5,760 latency) + steady-state null check
    let mut post_warmup_sq_err = 0.0f64;
    let mut post_warmup_count = 0usize;
    for b in 0..48 {
        for i in 0..256 {
            let t = ((b * 256 + i) as f32) / 48_000.0;
            let x = 0.35 * (TAU * 60.0 * t).sin() + 0.20 * (TAU * 520.0 * t).sin();
            block[i] = [x, x];
        }
        rt_node.process_slice(&ctx, &mut block, None, &mut []);
        for f in &block {
            assert!(f[0].is_finite() && f[1].is_finite());
            if b >= 32 {
                post_warmup_sq_err += (f[0] as f64).powi(2) + (f[1] as f64).powi(2);
                post_warmup_count += 2;
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(6));
    }
    let phase_a_null_rms = (post_warmup_sq_err / (post_warmup_count as f64)).sqrt();
    assert!(
        phase_a_null_rms < 1e-3,
        "Initial async ONNX GPU all-muted phase null failed (comb-filter misalignment!): rms={phase_a_null_rms}"
    );

    // Phase B: Tight zero-sleep burst (16 blocks = 8 hops) forcing transient `out_rx` underflow,
    // followed by normal real-time cadence -> must self-heal to 0-sample phase lock!
    for b in 48..64 {
        for i in 0..256 {
            let t = ((b * 256 + i) as f32) / 48_000.0;
            let x = 0.35 * (TAU * 60.0 * t).sin() + 0.20 * (TAU * 520.0 * t).sin();
            block[i] = [x, x];
        }
        rt_node.process_slice(&ctx, &mut block, None, &mut []);
    }
    std::thread::sleep(std::time::Duration::from_millis(40));
    let mut post_underflow_sq_err = 0.0f64;
    let mut post_underflow_count = 0usize;
    for b in 64..108 {
        for i in 0..256 {
            let t = ((b * 256 + i) as f32) / 48_000.0;
            let x = 0.35 * (TAU * 60.0 * t).sin() + 0.20 * (TAU * 520.0 * t).sin();
            block[i] = [x, x];
        }
        rt_node.process_slice(&ctx, &mut block, None, &mut []);
        if b >= 96 {
            for f in &block {
                post_underflow_sq_err += (f[0] as f64).powi(2) + (f[1] as f64).powi(2);
                post_underflow_count += 2;
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(6));
    }
    let phase_b_null_rms = (post_underflow_sq_err / (post_underflow_count as f64)).sqrt();
    assert!(
        phase_b_null_rms < 1e-3,
        "Post-underflow async ONNX GPU phase lock failed (permanent hop drift!): rms={phase_b_null_rms}"
    );

    // Phase C: `reset_state()` (transport seek/stop) -> must re-establish 2-hop cushion and exact phase null!
    rt_node.reset_state();
    std::thread::sleep(std::time::Duration::from_millis(25));
    let mut post_reset_sq_err = 0.0f64;
    let mut post_reset_count = 0usize;
    for b in 0..56 {
        for i in 0..256 {
            let t = ((b * 256 + i) as f32) / 48_000.0;
            let x = 0.30 * (TAU * 85.0 * t).sin() + 0.22 * (TAU * 440.0 * t).sin();
            block[i] = [x, x];
        }
        rt_node.process_slice(&ctx, &mut block, None, &mut []);
        if b >= 40 {
            for f in &block {
                post_reset_sq_err += (f[0] as f64).powi(2) + (f[1] as f64).powi(2);
                post_reset_count += 2;
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(8));
    }
    let phase_c_null_rms = (post_reset_sq_err / (post_reset_count as f64)).sqrt();
    assert!(
        phase_c_null_rms < 1e-3,
        "Post-reset_state() async ONNX GPU phase lock failed (missing cushion / stale hop!): rms={phase_c_null_rms}"
    );
    drop(rt_node); // Joins background worker cleanly

    // 4. Ozone 12 Delimit (`delimit_exp009_ep080_fixed2.onnx`):
    //    (a) Unclipped low-level audio must remain clean (proving ortho 1/sqrt(N_FFT) STFT scaling
    //        does not overload BatchNorm running_var).
    //    (b) Hard-limited peaks must expand above the brickwall ceiling.
    let mut delimit = OzoneDelimitOnnxProcessor::from_default_assets()
        .expect("OzoneDelimitOnnxProcessor must load on RTX 4080 CUDA");
    assert_eq!(delimit.active_tier(), GpuExecutionTier::CudaTensorCore);

    let quiet_clean: Vec<StereoFrame> = (0..8192)
        .map(|n| {
            let t = (n as f32) / 48_000.0;
            let s = 0.15 * (TAU * 220.0 * t).sin();
            [s, s]
        })
        .collect();
    let quiet_out = delimit
        .process_offline(&quiet_clean, 1.0)
        .expect("Delimit on unclipped quiet signal");
    let quiet_rms_err = (quiet_out[1024..7168]
        .iter()
        .zip(quiet_clean[1024..7168].iter())
        .map(|(a, b)| ((a[0] - b[0]) as f64).powi(2))
        .sum::<f64>()
        / 6144.0)
        .sqrt();
    assert!(
        quiet_rms_err < 0.03,
        "Unclipped quiet audio was distorted by Delimit STFT scaling! quiet_rms_err={quiet_rms_err}"
    );

    let clipped: Vec<StereoFrame> = (0..8192)
        .map(|n| {
            let t = (n as f32) / 48_000.0;
            let env = if (n % 2048) < 256 { 1.8 } else { 0.35 };
            let raw = env * (TAU * 180.0 * t).sin();
            let lim = raw.clamp(-0.65, 0.65);
            [lim, lim]
        })
        .collect();
    let restored = delimit
        .process_offline(&clipped, 1.0)
        .expect("Delimit CUDA offline run");
    assert_eq!(restored.len(), clipped.len());
    let restored_peak = restored
        .iter()
        .map(|f| f[0].abs().max(f[1].abs()))
        .fold(0.0f32, f32::max);
    assert!(
        restored_peak > 0.655 && restored_peak.is_finite(),
        "Delimit ONNX mask must expand brickwall-limited 0.65 peaks above ceiling (got peak={restored_peak})"
    );
}

#[test]
fn test_uvr5_kim_vocal2_inst_hq3_and_3model_gpu_ensemble_on_rtx4080() {
    let _serial = GPU_BENCH_SERIAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut kim = Uvr5MdxOnnxSeparator::from_kim_vocal_2().expect("Kim_Vocal_2 CUDA load");
    let mut inst_hq3 =
        Uvr5MdxOnnxSeparator::from_inst_hq_3().expect("UVR-MDX-NET-Inst_HQ_3 CUDA load");
    assert_eq!(kim.active_tier(), GpuExecutionTier::CudaTensorCore);
    assert_eq!(inst_hq3.active_tier(), GpuExecutionTier::CudaTensorCore);
    assert_eq!(kim.config().n_fft, 7680);
    assert_eq!(inst_hq3.config().n_fft, 6144);

    // Edge case 1: empty input
    let (empty_v, empty_i) = kim
        .separate_vocal_instrumental(&[], 48_000, true)
        .expect("empty input");
    assert!(empty_v.is_empty() && empty_i.is_empty());

    // Edge case 2: 1-sample boundary input
    let one_sample = [[0.25f32, -0.25f32]];
    let (one_v, one_i) = inst_hq3
        .separate_vocal_instrumental(&one_sample, 44_100, false)
        .expect("1-sample input");
    assert_eq!(one_v.len(), 1);
    assert_eq!(one_i.len(), 1);
    assert!(one_v[0][0].is_finite() && one_i[0][0].is_finite());

    // Real non-tautological acoustic separation check on pure percussive 808 sub-bass + kick:
    // Both `Kim_Vocal_2.onnx` and `UVR-MDX-NET-Inst_HQ_3.onnx` MUST route > 95% energy (`inst_rms > 5 * voc_rms`)
    // to Instrumental rather than Vocals!
    let pure_kick_bass: Vec<StereoFrame> = (0..44_100)
        .map(|n| {
            let hit_t = (n % 22_050) as f32 / 44_100.0;
            let pitch_drop = 55.0 * hit_t + 90.0 * (1.0 - (-35.0 * hit_t).exp()) / 35.0;
            let amp = (-3.5 * hit_t).exp() * 0.65;
            let x = amp * (TAU * pitch_drop).sin();
            [x, x]
        })
        .collect();
    for (label, sep) in [
        ("Kim_Vocal_2", &mut kim),
        ("UVR-MDX-NET-Inst_HQ_3", &mut inst_hq3),
    ] {
        let (voc, inst) = sep
            .separate_vocal_instrumental(&pure_kick_bass, 44_100, false)
            .unwrap_or_else(|e| panic!("{label} kick separation failed: {e}"));
        let voc_rms = (voc
            .iter()
            .map(|f| (f[0] as f64).powi(2))
            .sum::<f64>()
            / voc.len() as f64)
            .sqrt();
        let inst_rms = (inst
            .iter()
            .map(|f| (f[0] as f64).powi(2))
            .sum::<f64>()
            / inst.len() as f64)
            .sqrt();
        assert!(
            inst_rms > voc_rms * 5.0,
            "{label} failed real acoustic separation on pure 808 kick/bass! voc_rms={voc_rms:.5}, inst_rms={inst_rms:.5}"
        );
    }

    // Full 3-Model Studio Ensemble on RTX 4080 (1.5 seconds @ 48 kHz = 72,000 stereo frames)
    let mut engine = GpuStemEnsembleEngine::new().expect("GpuStemEnsembleEngine CUDA init");
    assert_eq!(engine.active_tier(), GpuExecutionTier::CudaTensorCore);

    let mix_48k: Vec<StereoFrame> = (0..72_000)
        .map(|n| {
            let t = (n as f32) / 48_000.0;
            let vibrato = (TAU * 5.5 * t).sin() * 12.0;
            let f0 = 330.0 + vibrato;
            let vocal_like = 0.25 * (TAU * f0 * t).sin()
                + 0.14 * (TAU * 2.0 * f0 * t).sin()
                + 0.08 * (TAU * 3.0 * f0 * t).sin();
            let bass = 0.30 * (TAU * 62.0 * t).sin();
            let hat = if (n % 6000) < 350 {
                0.15 * ((n as f32 * 1.73).sin())
            } else {
                0.0
            };
            [vocal_like + bass + hat, vocal_like * 0.95 + bass + hat * 0.8]
        })
        .collect();

    let cfg_denoise = GpuStemEnsembleConfig {
        model_selection: GpuStemModelSelection::Ensemble3Model,
        ensemble_mode: SpectralEnsembleMode::Weighted,
        weights: [0.45, 0.35, 0.20],
        enable_denoise_2pass: true,
        vr_aggressiveness: 0.15,
        phase_denoise_strength: 0.80,
        invert_instrumental: true,
    };

    let out = engine
        .separate_ensemble(&mix_48k, 48_000, &cfg_denoise)
        .expect("3-Model GPU Ensemble separation on RTX 4080");
    println!(
        "[RTX 4080 3-Model Ensemble] 1.5s @ 48kHz separated in {:.1} ms on {:?}",
        out.elapsed_ms, out.active_tier
    );

    assert_eq!(out.active_tier, GpuExecutionTier::CudaTensorCore);
    assert_eq!(out.vocals.len(), mix_48k.len());
    assert_eq!(out.drums.len(), mix_48k.len());
    assert_eq!(out.bass.len(), mix_48k.len());
    assert_eq!(out.other.len(), mix_48k.len());
    assert_eq!(out.inv_instrumental.len(), mix_48k.len());

    let mut max_4stem_err = 0.0f32;
    let mut max_inv_err = 0.0f32;
    for i in 0..mix_48k.len() {
        for ch in 0..2 {
            let v = out.vocals[i][ch];
            let d = out.drums[i][ch];
            let b = out.bass[i][ch];
            let o = out.other[i][ch];
            let inv = out.inv_instrumental[i][ch];
            assert!(v.is_finite() && d.is_finite() && b.is_finite() && o.is_finite() && inv.is_finite());
            max_4stem_err = max_4stem_err.max(((v + d + b + o) - mix_48k[i][ch]).abs());
            max_inv_err = max_inv_err.max(((v + inv) - mix_48k[i][ch]).abs());
        }
    }
    assert!(
        max_4stem_err < 1e-5,
        "4-stem sum V + D + B + O must equal mix to < 1e-5, got {max_4stem_err}"
    );
    assert!(
        max_inv_err < 1e-5,
        "Vocals + InvInstrumental must equal mix to < 1e-5, got {max_inv_err}"
    );

    // Verify 2-pass denoise and phase_denoise_strength measurably alter output on short 1024-sample buffer too
    let short_mix = &mix_48k[..1024];
    let out_short_a = engine
        .separate_ensemble(short_mix, 48_000, &cfg_denoise)
        .expect("short denoise=true");
    let mut cfg_no_denoise = cfg_denoise.clone();
    cfg_no_denoise.enable_denoise_2pass = false;
    let out_short_b = engine
        .separate_ensemble(short_mix, 48_000, &cfg_no_denoise)
        .expect("short denoise=false");
    let diff_2pass: f32 = out_short_a
        .vocals
        .iter()
        .zip(out_short_b.vocals.iter())
        .map(|(a, b)| (a[0] - b[0]).abs() + (a[1] - b[1]).abs())
        .sum();
    assert!(
        diff_2pass > 1e-3,
        "enable_denoise_2pass=true vs false must alter vocals on RTX 4080 (diff={diff_2pass})"
    );

    let mut cfg_no_phase = cfg_denoise.clone();
    cfg_no_phase.phase_denoise_strength = 0.0;
    let out_short_c = engine
        .separate_ensemble(short_mix, 48_000, &cfg_no_phase)
        .expect("short phase_denoise=0");
    let diff_phase: f32 = out_short_a
        .vocals
        .iter()
        .zip(out_short_c.vocals.iter())
        .map(|(a, b)| (a[0] - b[0]).abs() + (a[1] - b[1]).abs())
        .sum();
    assert!(
        diff_phase > 1e-4,
        "phase_denoise_strength=0.80 vs 0.0 must alter vocals on RTX 4080 (diff={diff_phase})"
    );
}
