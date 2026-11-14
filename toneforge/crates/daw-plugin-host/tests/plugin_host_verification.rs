//! Integration verification for `daw-plugin-host` against the real compiled native DLL
//! `test_vst3_clap_plugin.dll` (`LoadLibraryW` CLAP 1.2 C ABI + Steinberg VST3 3.7 COM ABI +
//! Win32 `HWND` GUI + 5-Point APDC `64`-sample parallel null cancellation + crash-isolated scanner).

use daw_core_graph::{
    MixerEngine, ProcessContext, SequencerVoiceEvent, StereoFrame, UnifiedDspNode, VoiceEventKind,
};
use daw_plugin_host::{
    ClapHostedPluginNode, Vst3HostedPluginNode, locate_scanner_executable, probe_plugin_isolated,
    scan_plugin_directories_isolated,
};
use std::path::{Path, PathBuf};
use std::time::Duration;

fn locate_test_plugin_dll() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let ws_root = manifest_dir.parent().unwrap().parent().unwrap();
    for profile in ["release", "debug"] {
        let p = ws_root
            .join("target")
            .join(profile)
            .join("test_vst3_clap_plugin.dll");
        if p.is_file() {
            return p;
        }
    }
    panic!("test_vst3_clap_plugin.dll not found in target/release or target/debug; build `-p test-vst3-clap-plugin` first");
}

fn measure_fundamental_hz_zero_crossing(samples: &[f32], sample_rate: f64) -> f64 {
    let mut crossings = Vec::new();
    for i in 1..samples.len() {
        let s0 = samples[i - 1] as f64;
        let s1 = samples[i] as f64;
        if s0 <= 0.0 && s1 > 0.0 {
            let frac = (-s0) / (s1 - s0).max(1e-15);
            crossings.push((i - 1) as f64 + frac);
        }
    }
    if crossings.len() < 3 {
        return 0.0;
    }
    let periods = (crossings.len() - 1) as f64;
    let span = crossings[crossings.len() - 1] - crossings[0];
    sample_rate * periods / span
}

struct PolarityInvertGainNode {
    gain: f32,
}

impl UnifiedDspNode for PolarityInvertGainNode {
    fn name(&self) -> &'static str {
        "PolarityInvertGainNode"
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
        main_io: &mut [StereoFrame],
        _sidechain_in: Option<&[StereoFrame]>,
        _aux_sends_out: &mut [&mut [StereoFrame]],
    ) {
        for f in main_io.iter_mut() {
            f[0] *= self.gain;
            f[1] *= self.gain;
        }
    }
}

#[test]
fn test_clap_host_real_dll_synth_a4_state_and_win32_hwnd() {
    let dll = locate_test_plugin_dll();
    let sr = 48_000.0;
    let mut clap_node = ClapHostedPluginNode::load(&dll, None, sr, 512)
        .expect("Failed to load CLAP 1.2 plugin via LoadLibraryW");

    assert_eq!(clap_node.latency_samples(), 64);
    assert_eq!(clap_node.parameters().len(), 4);

    let ctx = ProcessContext {
        sample_rate: sr,
        current_sample_pos: 0,
        ppq_tick: 0,
        bpm: 128.0,
        slice_frames: 4096,
        is_playing: true,
        is_offline_rendering: false,
        ppq_resolution: daw_core_graph::DEFAULT_PPQ,
    };

    // Trigger MIDI Note 69 = A4 = 440.0 Hz
    clap_node.on_voice_event(
        &SequencerVoiceEvent {
            sample_pos: 0,
            key: 69,
            velocity: 0.9,
            kind: VoiceEventKind::NoteOn,
            note_id: 69,
            ..SequencerVoiceEvent::default()
        },
        &ctx,
    );

    let mut rendered = Vec::with_capacity(12_288);
    let mut empty_aux: [&mut [StereoFrame]; 0] = [];
    for blk in 0..24 {
        let mut buf = [[0.0f32; 2]; 512];
        let bctx = ProcessContext {
            current_sample_pos: (blk * 512) as i64,
            slice_frames: 512,
            ..ctx
        };
        clap_node.process_slice(&bctx, &mut buf, None, &mut empty_aux);
        for f in &buf {
            rendered.push(f[0]);
        }
    }

    // Analyze steady-state region [2048..11264]
    let f0 = measure_fundamental_hz_zero_crossing(&rendered[2048..11264], sr);
    assert!(
        (f0 - 440.0).abs() <= 0.5,
        "CLAP A4 fundamental must be 440.0 Hz +- 0.5 Hz, got {f0:.3} Hz"
    );

    // Parameter automation + binary state save/restore
    clap_node.queue_parameter_plain(0, 1.25, 0);
    clap_node.queue_parameter_plain(1, 3200.0, 0);
    let mut step_buf = [[0.0f32; 2]; 64];
    clap_node.process_slice(&ctx, &mut step_buf, None, &mut empty_aux);
    let saved = clap_node.save_state().expect("CLAP save_state failed");
    assert_eq!(saved.len(), 40);

    let mut restored = ClapHostedPluginNode::load(&dll, None, sr, 512).unwrap();
    restored.load_state(&saved).expect("CLAP load_state failed");
    assert!((restored.get_parameter_plain(0).unwrap() - 1.25).abs() < 1e-9);
    assert!((restored.get_parameter_plain(1).unwrap() - 3200.0).abs() < 1e-9);
    assert_eq!(restored.save_state().unwrap(), saved);

    // Hidden Win32 HWND attach/pump/detach test
    let hwnd = restored
        .open_editor(None, false)
        .expect("CLAP Win32 HWND open_editor failed");
    assert_ne!(hwnd, 0);
    let _ = restored.pump_editor_messages();
    restored.close_editor();
}

#[test]
fn test_vst3_host_real_dll_com_synth_a4_state_hwnd_and_apdc_null() {
    let dll = locate_test_plugin_dll();
    let sr = 48_000.0;
    let mut vst3_node = Vst3HostedPluginNode::load(&dll, None, sr, 512)
        .expect("Failed to load Steinberg VST3 3.7 plugin via LoadLibraryW");

    assert_eq!(vst3_node.latency_samples(), 64);
    assert_eq!(vst3_node.parameters().len(), 4);

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

    vst3_node.on_voice_event(
        &SequencerVoiceEvent {
            sample_pos: 0,
            key: 69,
            velocity: 0.9,
            kind: VoiceEventKind::NoteOn,
            note_id: 69,
            ..SequencerVoiceEvent::default()
        },
        &ctx,
    );

    let mut rendered = Vec::with_capacity(12_288);
    let mut empty_aux: [&mut [StereoFrame]; 0] = [];
    for blk in 0..24 {
        let mut buf = [[0.0f32; 2]; 512];
        let bctx = ProcessContext {
            current_sample_pos: (blk * 512) as i64,
            ..ctx
        };
        vst3_node.process_slice(&bctx, &mut buf, None, &mut empty_aux);
        for f in &buf {
            rendered.push(f[0]);
        }
    }

    let f0 = measure_fundamental_hz_zero_crossing(&rendered[2048..11264], sr);
    assert!(
        (f0 - 440.0).abs() <= 0.5,
        "VST3 A4 fundamental must be 440.0 Hz +- 0.5 Hz, got {f0:.3} Hz"
    );

    // State save & restore via COM IBStream
    vst3_node.queue_parameter_point(0, 0.77, 0);
    vst3_node.queue_parameter_point(2, 0.42, 0);
    let mut flush_buf = [[0.0f32; 2]; 64];
    vst3_node.process_slice(&ctx, &mut flush_buf, None, &mut empty_aux);
    let saved = vst3_node.save_state().expect("VST3 save_state failed");

    let mut vst3_restored = Vst3HostedPluginNode::load(&dll, None, sr, 512).unwrap();
    vst3_restored
        .load_state(&saved)
        .expect("VST3 load_state failed");
    assert!((vst3_restored.get_parameter_normalized(0).unwrap() - 0.77).abs() < 1e-6);
    assert!((vst3_restored.get_parameter_normalized(2).unwrap() - 0.42).abs() < 1e-6);
    assert_eq!(vst3_restored.save_state().unwrap(), saved);

    // Hidden Win32 HWND IPlugView::attached test
    let hwnd = vst3_restored
        .open_editor(None, false)
        .expect("VST3 IPlugView Win32 HWND open_editor failed");
    assert_ne!(hwnd, 0);
    let _ = vst3_restored.pump_editor_messages();
    vst3_restored.close_editor();

    // 5-Point APDC Parallel Null Test (`64` samples VST3 latency on Track 1 vs `0` latency inverted Track 2)
    struct BroadbandTestGen;
    impl UnifiedDspNode for BroadbandTestGen {
        fn name(&self) -> &'static str {
            "BroadbandTestGen"
        }
        fn latency_samples(&self) -> usize {
            0
        }
        fn prepare(&mut self, _sample_rate: f64, _max_block_size: usize) {}
        fn reset_state(&mut self) {}
        fn set_parameter_normalized(&mut self, _param_id: u32, _norm_val: f32) {}
        fn process_slice(
            &mut self,
            ctx: &ProcessContext,
            main_io: &mut [StereoFrame],
            _sidechain_in: Option<&[StereoFrame]>,
            _aux_sends_out: &mut [&mut [StereoFrame]],
        ) {
            for (i, f) in main_io.iter_mut().enumerate() {
                let n = (ctx.current_sample_pos as usize + i) as f64;
                let s = (0.30 * (0.041 * n).sin()
                    + 0.20 * (0.173 * n).cos()
                    + 0.15 * (0.517 * n).sin()) as f32;
                *f = [s, -0.8 * s];
            }
        }
    }

    let (mut engine, _cmd_tx, _tel_rx) = MixerEngine::new(sr, 128.0, 0);
    let vst3_apdc = Vst3HostedPluginNode::load(&dll, None, sr, 256).unwrap();
    engine.tracks[1]
        .add_plugin_slot(Box::new(vst3_apdc), 1.0)
        .unwrap();
    engine.tracks[2]
        .add_plugin_slot(Box::new(PolarityInvertGainNode { gain: -1.0 }), 1.0)
        .unwrap();
    // Route Track 3 -> Track 1 and Track 3 -> Track 2, then Track 1 -> Master(0) and Track 2 -> Master(0)
    engine.tracks[3].add_send(1, 1.0, false);
    engine.tracks[3].add_send(2, 1.0, false);
    engine.tracks[1].add_send(0, 1.0, false);
    engine.tracks[2].add_send(0, 1.0, false);
    engine
        .add_channel_generator(daw_core_graph::ChannelGenerator::new(
            1,
            3,
            Box::new(BroadbandTestGen),
        ))
        .unwrap();
    engine.rebuild_graph().unwrap();

    let mut master_out = vec![[0.0f32; 2]; 4096];
    engine.process_audio_block(&mut master_out);

    // Steady region after initial 64-sample APDC fill (`[256..4096]`) must null <= -90 dBFS
    let mut max_err = 0.0f64;
    for f in &master_out[256..] {
        max_err = max_err.max((f[0] as f64).abs()).max((f[1] as f64).abs());
    }
    let null_db = 20.0 * max_err.max(1e-15).log10();
    assert!(
        null_db <= -90.0,
        "5-Point APDC parallel VST3 (64 samples) null test must reach <= -90 dBFS, got {null_db:.2} dBFS"
    );
}

#[test]
fn test_out_of_process_scanner_discovers_plugin_and_quarantines_crash() {
    let dll = locate_test_plugin_dll();
    let scanner_exe = locate_scanner_executable();
    assert!(
        scanner_exe.is_some(),
        "toneforge-plugin-scanner.exe should be built before running scanner test"
    );

    // 1. Valid dual-ABI plugin discovery in isolated process
    let report = scan_plugin_directories_isolated(&[dll.clone()], None, Duration::from_secs(4));
    assert!(
        report.plugins.len() >= 2,
        "Dual-ABI DLL must yield both CLAP and VST3 descriptors, got {:?}",
        report.plugins
    );

    // 2. Simulated 0xC0000005 access violation in child process must NOT crash host and must quarantine!
    let crash_res = probe_plugin_isolated(
        scanner_exe.as_deref(),
        &dll,
        Duration::from_secs(4),
        &["--simulate-crash"],
    );
    let q = crash_res.expect_err("Simulated child crash must return PluginQuarantineEntry");
    assert_eq!(q.exit_code, Some(-1073741819));
    assert!(q.reason.contains("0xC0000005"));
}
