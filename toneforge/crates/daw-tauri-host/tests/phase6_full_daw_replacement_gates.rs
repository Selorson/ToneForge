//! Phase 6 Full DAW Replacement Quantitative Acceptance Suite (`Gates 7–10`).
//!
//! Enforces 100% of the ToneForge Studio v3 Quantitative Acceptance Contract
//! (`09_toneforge_v3_full_daw_master_plan.md` Section 5, Tier 1):
//! - **Gate 7**: 20-Step Bi-Directional `Undo -> Redo` bit-exact state & audio restoration,
//!   `Pro-Q 4 EQ Sketch` (`< 0.85 dB` RMS) + `ProQ4CollisionBus` (`65 Hz ± 5 Hz`),
//!   `Soothe3 External Sidechain` (`>= 8.0 dB` @ `2.5 kHz`, `< 1.0 dB` @ `200 Hz` & `8 kHz`, `0` clicks),
//!   and `Ozone 12 Low End Focus` (`>= 1.5 dB` `< 150 Hz` Crest Factor boost, `< 0.1 dB` `> 500 Hz` invariance)
//!   + `Codec Preview` (`MP3 128` vs `MP3 320` `>= 6.0 dB` noise separation, `PcmBypass` `Artifact Solo <= -110 dBFS`).
//! - **Gate 8**: Real Win32 `LoadLibraryW` hosting of compiled `test_vst3_clap_plugin.dll` via BOTH
//!   `ClapHostedPluginNode` (`CLAP 1.2`) and `Vst3HostedPluginNode` (`Steinberg VST3 3.7 COM`),
//!   crash-isolated scanner + rogue plugin quarantine, `NoteOn A4=440 Hz` (`±1 Hz`), parameter automation,
//!   100% binary state round-trip, 5-Point APDC `64`-sample parallel null cancellation (`<= -90 dBFS`),
//!   and `0` RT heap `alloc` / `dealloc`.
//! - **Gate 9**: Hardware ASIO / WASAPI Capture / Win32 MIDI In enumeration, `60s` `+150 ppm` clock
//!   drift stress test through `AdaptiveDllDriftResampler` (`0` xruns, `>= 95 dB` SNR), and lock-free
//!   `DiskTrackRecorder` round-trip latency compensation (`0`-sample impulse alignment) + `3`-Take Loop Comping
//!   with `0` RT heap `alloc` / `dealloc`.
//! - **Gate 10**: `PyinPitchDetector` (`< 2.0 cents` RMS on `5.5 Hz ±35 cents` vibrato, `100%` unvoiced
//!   sibilant rejection), `+5.0 st` `FormantPitchShifter` (`+500 ± 5 cents` `f0`, `< 8.0%` `F1`/`F2` formant drift),
//!   and `TransientLockedTimeStretcher` (`1.35x` & `0.75x`, `±4` sample length accuracy, `>= 85%` Kick
//!   transient Crest Factor preservation, `>= 0.95` stereo chord correlation).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::f64::consts::TAU;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use daw_core_graph::{
    AdaptiveDllDriftResampler, ChannelGenerator, DeterministicGeneratorNode, DiskTrackRecorder,
    MixerEngine, PianoRollNote, PianoRollPattern, PlaylistClip, PlaylistClipKind, PluginRackSlot,
    PolyphonicSynthGeneratorNode, ProQ4SpectrumFrame, ProcessContext, StereoFrame, UnifiedDspNode,
    enumerate_asio_drivers, enumerate_midi_input_devices, enumerate_wasapi_input_devices,
    read_wav_f32_stereo,
};
use daw_plugin_host::{
    ClapHostedPluginNode, Vst3HostedPluginNode, scan_plugin_directories_isolated,
};
use daw_tauri_host::project_state::{
    MixerSendRouteDto, PianoRollNoteDto, PluginConfigDto, PluginSlotStateDto, ProQ4StateDto,
    ToneForgeProjectState,
};
use daw_tauri_host::undo_history::{
    ProjectUndoHistory, build_plugin_node_from_dto, sync_project_state_diff_to_rt,
};
use daw_tauri_host::DawAppState;
use dsp_ozone12_mastering::{
    CodecPreviewModule, CodecTarget, LinkwitzRiley4Splitter2Way, LowEndFocusMode,
    LowEndFocusModule,
};
use dsp_pitch_time::{
    FormantPitchShifter, PyinPitchDetector, TransientLockedTimeStretcher,
    estimate_vocal_formant_envelope_db, find_formant_peak_in_band,
};
use dsp_proq4_eq::{
    EqBandParams, FilterShape, ProQ4CollisionBus, ProQ4ParameterSet, ProQ4Processor,
    SPECTRUM_FRAME_BINS, eval_bands_combined_db, fit_eq_sketch_curve,
};
use dsp_soothe3_spectral::Soothe3Processor;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;
use serde_json::json;

// ============================================================================
// THREAD-LOCAL COUNTING GLOBAL ALLOCATOR FOR ZERO-ALLOC RT VERIFICATION
// ============================================================================

struct CountingSystemAllocator;

thread_local! {
    static COUNTING_ACTIVE: Cell<bool> = const { Cell::new(false) };
    static ALLOC_CALLS: Cell<usize> = const { Cell::new(0) };
    static DEALLOC_CALLS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingSystemAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = COUNTING_ACTIVE.try_with(|active| {
            if active.get() {
                let _ = ALLOC_CALLS.try_with(|c| c.set(c.get().saturating_add(1)));
            }
        });
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let _ = COUNTING_ACTIVE.try_with(|active| {
            if active.get() {
                let _ = ALLOC_CALLS.try_with(|c| c.set(c.get().saturating_add(1)));
            }
        });
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = COUNTING_ACTIVE.try_with(|active| {
            if active.get() {
                let _ = ALLOC_CALLS.try_with(|c| c.set(c.get().saturating_add(1)));
            }
        });
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let _ = COUNTING_ACTIVE.try_with(|active| {
            if active.get() {
                let _ = DEALLOC_CALLS.try_with(|c| c.set(c.get().saturating_add(1)));
            }
        });
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL_RT_AUDIT_ALLOCATOR: CountingSystemAllocator = CountingSystemAllocator;

fn measure_rt_allocs<F, R>(f: F) -> (R, usize, usize)
where
    F: FnOnce() -> R,
{
    ALLOC_CALLS.with(|c| c.set(0));
    DEALLOC_CALLS.with(|c| c.set(0));
    COUNTING_ACTIVE.with(|a| a.set(true));
    let out = f();
    COUNTING_ACTIVE.with(|a| a.set(false));
    let allocs = ALLOC_CALLS.with(|c| c.get());
    let deallocs = DEALLOC_CALLS.with(|c| c.get());
    (out, allocs, deallocs)
}

fn workspace_root_dir() -> PathBuf {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.pop(); // crates/
    root.pop(); // toneforge/
    root
}

fn save_gate_metrics(gate_name: &str, payload: serde_json::Value) {
    let out_dir = workspace_root_dir().join("target").join("gate_metrics");
    let _ = fs::create_dir_all(&out_dir);
    let path = out_dir.join(format!("{gate_name}.json"));
    if let Ok(pretty) = serde_json::to_string_pretty(&payload) {
        let _ = fs::write(path, pretty);
    }
}

fn locate_compiled_test_plugin_dll() -> PathBuf {
    let root = workspace_root_dir();
    let release_dll = root
        .join("target")
        .join("release")
        .join("test_vst3_clap_plugin.dll");
    if release_dll.is_file() {
        return release_dll;
    }
    let debug_dll = root
        .join("target")
        .join("debug")
        .join("test_vst3_clap_plugin.dll");
    if debug_dll.is_file() {
        return debug_dll;
    }
    panic!(
        "Compiled test_vst3_clap_plugin.dll not found at {} or {}. Run `cargo build -p test-vst3-clap-plugin --release` first.",
        release_dll.display(),
        debug_dll.display()
    );
}

fn goertzel_mag_db_f32(samples: &[f32], target_hz: f64, sr: f64) -> f64 {
    let n = samples.len();
    assert!(n > 0);
    let k = ((n as f64) * target_hz / sr).round();
    let omega = TAU * k / (n as f64);
    let coeff = 2.0 * omega.cos();
    let mut s0 = 0.0_f64;
    let mut s1 = 0.0_f64;
    let mut s2;
    for &x in samples {
        s2 = s1;
        s1 = s0;
        s0 = (x as f64) + coeff * s1 - s2;
    }
    let re = s0 - s1 * omega.cos();
    let im = s1 * omega.sin();
    let mag = (2.0 * (re * re + im * im).sqrt()) / (n as f64);
    20.0 * mag.max(1e-15).log10()
}

fn measure_fft_peak_freq_hz(samples: &[f32], sr: f64) -> f64 {
    let n = samples.len().next_power_of_two();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let mut buf = vec![Complex64::new(0.0, 0.0); n];
    let len = samples.len().min(n);
    for i in 0..len {
        let w = 0.5 * (1.0 - (TAU * (i as f64) / ((len - 1).max(1) as f64)).cos());
        buf[i] = Complex64::new((samples[i] as f64) * w, 0.0);
    }
    fft.process(&mut buf);

    let mut best_k = 1usize;
    let mut best_mag = 0.0_f64;
    for k in 2..(n / 2 - 2) {
        let m = buf[k].norm();
        if m > best_mag {
            best_mag = m;
            best_k = k;
        }
    }
    let m_prev = buf[best_k - 1].norm().max(1e-15).ln();
    let m_curr = buf[best_k].norm().max(1e-15).ln();
    let m_next = buf[best_k + 1].norm().max(1e-15).ln();
    let denom = m_prev - 2.0 * m_curr + m_next;
    let delta = if denom.abs() > 1e-12 {
        0.5 * (m_prev - m_next) / denom
    } else {
        0.0
    };
    ((best_k as f64) + delta.clamp(-0.5, 0.5)) * sr / (n as f64)
}

/// Deterministically reconstructs a standalone `MixerEngine` from `ToneForgeProjectState`
/// and renders `num_blocks` of `256` stereo frames.
fn render_deterministic_project(state: &ToneForgeProjectState, num_blocks: usize) -> Vec<StereoFrame> {
    let sr = state.sample_rate.max(8_000.0);
    let (mut engine, _, _) = MixerEngine::new(sr, state.bpm, 0);

    // Attach deterministic polyphonic synth on Channel 1 routed to Mixer Track 1 so Note, Clip,
    // Pro-Q 4 EQ, and Mixer Send Routing mutations directly shape the rendered Master output.
    let synth = PolyphonicSynthGeneratorNode::new(1, sr);
    engine
        .add_channel_generator(ChannelGenerator::new(1, 1, Box::new(synth)))
        .expect("attach Channel 1 generator");

    for p_dto in &state.piano_roll_patterns {
        let mut pat = PianoRollPattern::new(
            p_dto.pattern_id,
            p_dto.name.clone(),
            p_dto.target_channel_id,
            p_dto.length_ticks,
        );
        for n in &p_dto.notes {
            pat.add_note(PianoRollNote {
                id: n.id,
                start_tick: n.start_tick,
                duration_ticks: n.duration_ticks,
                key: n.key,
                velocity: n.velocity,
                pan: n.pan,
                fine_pitch_cents: n.fine_pitch_cents,
                is_slide: n.is_slide,
                color_group: n.color_group,
            });
        }
        let _ = engine.sequencer.upsert_pattern(pat);
    }

    for c_dto in &state.playlist_clips {
        let kind = match c_dto.kind.as_str() {
            "AudioStem" => PlaylistClipKind::AudioStem,
            "Automation" => PlaylistClipKind::Automation,
            _ => PlaylistClipKind::Pattern,
        };
        let _ = engine.sequencer.upsert_clip(PlaylistClip {
            clip_id: c_dto.clip_id,
            track_row: c_dto.track_row,
            kind,
            source_id: c_dto.source_id,
            start_tick: c_dto.start_tick,
            duration_ticks: c_dto.duration_ticks,
            clip_offset_ticks: c_dto.clip_offset_ticks,
            gain: c_dto.gain,
            muted: c_dto.muted,
            source_offset_samples: c_dto.source_offset_samples,
            fade_in_samples: c_dto.fade_in_samples,
            fade_out_samples: c_dto.fade_out_samples,
            stretch_ratio: c_dto.stretch_ratio,
            pitch_semitones: c_dto.pitch_semitones,
            take_lane_id: c_dto.take_lane_id,
        });
    }
    engine.sequencer.song_mode = true;

    for (idx, trk_dto) in state.mixer_tracks.iter().enumerate() {
        engine.tracks[idx].gain = trk_dto.gain;
        engine.tracks[idx].polarity_inverted = trk_dto.polarity_inverted;
        engine.tracks[idx].sends.clear();
        for s in &trk_dto.sends {
            engine.tracks[idx].add_send(s.dest_track_index, s.gain, s.is_sidechain_only);
        }
        engine.tracks[idx].slots.clear();
        for slot_dto in &trk_dto.slots {
            let node = build_plugin_node_from_dto(idx, sr, 256, &slot_dto.plugin, None)
                .expect("build plugin node from DTO");
            let mut rack_slot = PluginRackSlot::new(node, slot_dto.wet_mix);
            rack_slot.enabled = slot_dto.enabled;
            rack_slot.prepare(sr, 256);
            engine.tracks[idx].slots.push(rack_slot);
        }
    }

    engine.is_playing = state.is_playing;
    engine.rebuild_graph().expect("valid acyclic 502-track DAG");

    let mut out = Vec::with_capacity(num_blocks * 256);
    let mut block = [[0.0f32; 2]; 256];
    for _ in 0..num_blocks {
        engine.process_audio_block(&mut block);
        out.extend_from_slice(&block);
    }
    out
}

// ============================================================================
// GATE 7: WORKFLOW (20-STEP UNDO/REDO) & 100% INTERNAL PLUGIN PARITY
// ============================================================================

#[test]
fn gate7_workflow_and_100_percent_plugin_parity() {
    // ------------------------------------------------------------------------
    // 7.1: 20-Step Bi-Directional Undo -> Redo across Note, Clip, EQ, Routing
    // ------------------------------------------------------------------------
    let sr = 48_000.0;
    let mut state_0 = ToneForgeProjectState::new_default(sr, 128.0);
    let pq_init = ProQ4ParameterSet::new();
    state_0.mixer_tracks[1].slots = vec![PluginSlotStateDto {
        slot_index: 0,
        enabled: true,
        wet_mix: 1.0,
        plugin: PluginConfigDto::ProQ4(ProQ4StateDto::from_parameter_set(&pq_init)),
    }];
    // Ensure auxiliary bus tracks 2..=8 route to Master 0 so Send mutations sum audibly into Master
    for bus in 2..=8 {
        if state_0.mixer_tracks[bus].sends.is_empty() {
            state_0.mixer_tracks[bus].sends.push(MixerSendRouteDto {
                dest_track_index: 0,
                gain: 1.0,
                is_sidechain_only: false,
            });
        }
    }

    let json_0 = state_0.to_json_pretty().expect("serialize baseline state_0");
    let audio_0 = render_deterministic_project(&state_0, 24);

    let mut cur_state = state_0.clone();
    let mut undo_hist = ProjectUndoHistory::new(64);

    // Apply 20 distinct mutations (5 Notes, 5 Clips, 5 Pro-Q 4 EQ params, 5 Mixer Send Routes)
    for step in 0..20usize {
        undo_hist.record_before_action(format!("Gate7 Step #{}", step + 1), &cur_state);
        match step % 4 {
            0 => {
                // Piano Roll Note mutation
                let next_id = (100 + step) as u32;
                cur_state.piano_roll_patterns[0].notes.push(PianoRollNoteDto {
                    id: next_id,
                    start_tick: ((step as i64) * 8) % 192,
                    duration_ticks: 48,
                    key: (62 + (step % 10)) as u8,
                    velocity: 0.82,
                    pan: 0.1,
                    fine_pitch_cents: 0.0,
                    is_slide: false,
                    color_group: 0,
                });
            }
            1 => {
                // Playlist Clip Gain / Fade mutation
                cur_state.playlist_clips[0].gain = 0.62 + (step as f32) * 0.015;
                cur_state.playlist_clips[0].fade_in_samples = step * 16;
            }
            2 => {
                // Pro-Q 4 Band #0 Bell Boost mutation
                if let Some(slot) = cur_state.mixer_tracks[1]
                    .slots
                    .iter_mut()
                    .find(|s| s.slot_index == 0)
                {
                    if let PluginConfigDto::ProQ4(ref mut pq) = slot.plugin {
                        let mut ps = pq.to_parameter_set();
                        ps.bands[0].used = true;
                        ps.bands[0].enabled = true;
                        ps.bands[0].freq_hz = 950.0;
                        ps.bands[0].gain_db = 2.0 + 0.35 * ((step + 1) as f32);
                        ps.sync_mirror_from_structured();
                        *pq = ProQ4StateDto::from_parameter_set(&ps);
                    }
                }
            }
            _ => {
                // Mixer Send Routing mutation (Track 1 -> Aux Bus `2 + step/4` -> Master 0)
                let aux_bus = 2 + (step / 4);
                if !cur_state.mixer_tracks[1]
                    .sends
                    .iter()
                    .any(|s| s.dest_track_index == aux_bus)
                {
                    cur_state.mixer_tracks[1].sends.push(MixerSendRouteDto {
                        dest_track_index: aux_bus,
                        gain: 0.35,
                        is_sidechain_only: false,
                    });
                }
            }
        }
    }

    assert_eq!(undo_hist.status().undo_depth, 20);
    let state_20 = cur_state.clone();
    let json_20 = state_20.to_json_pretty().expect("serialize state_20");
    let audio_20 = render_deterministic_project(&state_20, 24);
    assert_ne!(
        audio_0, audio_20,
        "20 mutations must measurably change rendered output audio"
    );

    // Also verify lock-free RT delta sync over `DawAppState`
    let (cmd_tx, mut cmd_rx) = rtrb::RingBuffer::new(1024);
    let (mut_tx, mut mut_rx) = rtrb::RingBuffer::new(1024);
    let app_state = DawAppState::new_with_rt_mutations(cmd_tx, mut_tx);
    sync_project_state_diff_to_rt(&state_20, &state_0, &app_state, None)
        .expect("RT diff dispatch must succeed");
    let mut dispatched_rt_msgs = 0usize;
    while cmd_rx.pop().is_ok() {
        dispatched_rt_msgs += 1;
    }
    while mut_rx.pop().is_ok() {
        dispatched_rt_msgs += 1;
    }
    assert!(dispatched_rt_msgs > 0);

    // Undo all 20 steps -> bit-exact restoration of state_0 and audio_0
    for _ in 0..20 {
        assert!(undo_hist.undo(&mut cur_state).is_some());
    }
    assert_eq!(undo_hist.status().undo_depth, 0);
    assert_eq!(undo_hist.status().redo_depth, 20);
    assert_eq!(cur_state, state_0, "State after 20 Undos must equal state_0");
    assert_eq!(
        cur_state.to_json_pretty().unwrap(),
        json_0,
        "JSON after 20 Undos must be 100% byte-identical"
    );

    let audio_after_undo = render_deterministic_project(&cur_state, 24);
    let mut max_undo_bit_diff = 0u32;
    for (a, b) in audio_0.iter().zip(audio_after_undo.iter()) {
        max_undo_bit_diff = max_undo_bit_diff
            .max(a[0].to_bits().abs_diff(b[0].to_bits()))
            .max(a[1].to_bits().abs_diff(b[1].to_bits()));
    }
    assert_eq!(
        max_undo_bit_diff, 0,
        "Rendered audio after 20 Undos must be 100% bit-exact (0 ULP diff)"
    );

    // Redo all 20 steps -> bit-exact restoration of state_20 and audio_20
    for _ in 0..20 {
        assert!(undo_hist.redo(&mut cur_state).is_some());
    }
    assert_eq!(cur_state, state_20, "State after 20 Redos must equal state_20");
    assert_eq!(cur_state.to_json_pretty().unwrap(), json_20);
    let audio_after_redo = render_deterministic_project(&cur_state, 24);
    let mut max_redo_bit_diff = 0u32;
    for (a, b) in audio_20.iter().zip(audio_after_redo.iter()) {
        max_redo_bit_diff = max_redo_bit_diff
            .max(a[0].to_bits().abs_diff(b[0].to_bits()))
            .max(a[1].to_bits().abs_diff(b[1].to_bits()));
    }
    assert_eq!(
        max_redo_bit_diff, 0,
        "Rendered audio after 20 Redos must be 100% bit-exact (0 ULP diff)"
    );

    // ------------------------------------------------------------------------
    // 7.2: Pro-Q 4 EQ Sketch (< 0.85 dB RMS) & Collision Bus (65 Hz ± 5 Hz)
    // ------------------------------------------------------------------------
    let mut ref_bands = vec![EqBandParams::default(); 3];
    // LowCut @ 60 Hz (12 dB/oct)
    ref_bands[0].used = true;
    ref_bands[0].enabled = true;
    ref_bands[0].shape = FilterShape::LowCut;
    ref_bands[0].freq_hz = 60.0;
    ref_bands[0].q_ui = 0.707;
    ref_bands[0].slope_db_oct = 12.0;
    // Bell +6 dB @ 1 kHz
    ref_bands[1].used = true;
    ref_bands[1].enabled = true;
    ref_bands[1].shape = FilterShape::Bell;
    ref_bands[1].freq_hz = 1_000.0;
    ref_bands[1].gain_db = 6.0;
    ref_bands[1].q_ui = 1.4;
    ref_bands[1].slope_db_oct = 12.0;
    // HighShelf -4 dB @ 8 kHz
    ref_bands[2].used = true;
    ref_bands[2].enabled = true;
    ref_bands[2].shape = FilterShape::HighShelf;
    ref_bands[2].freq_hz = 8_000.0;
    ref_bands[2].gain_db = -4.0;
    ref_bands[2].q_ui = 0.85;
    ref_bands[2].slope_db_oct = 12.0;

    let mut sketch_pts = Vec::with_capacity(96);
    for i in 0..96 {
        let u = (i as f64) / 95.0;
        let f = 20.0 * (20_000.0_f64 / 20.0).powf(u);
        let g = eval_bands_combined_db(&ref_bands, f, sr);
        sketch_pts.push((f as f32, g));
    }
    let fitted_bands = fit_eq_sketch_curve(&sketch_pts, 6);
    assert!(!fitted_bands.is_empty());

    let eval_count = 128usize;
    let mut sq_db_err = 0.0_f64;
    for i in 0..eval_count {
        let u = (i as f64) / ((eval_count - 1) as f64);
        let f = 80.0 * (15_000.0_f64 / 80.0).powf(u);
        let target_db = eval_bands_combined_db(&ref_bands, f, sr) as f64;
        let fit_db = eval_bands_combined_db(&fitted_bands, f, sr) as f64;
        let diff = fit_db - target_db;
        sq_db_err += diff * diff;
    }
    let sketch_rms_err_db = (sq_db_err / (eval_count as f64)).sqrt();
    assert!(
        sketch_rms_err_db < 0.85,
        "Gate 7.2 EQ Sketch RMS error must be < 0.85 dB across 80 Hz..15 kHz, got {sketch_rms_err_db:.4} dB"
    );

    // Cross-track Kick (65 Hz) vs Bass (65 Hz) Collision Bus test
    let collision_bus = ProQ4CollisionBus::new_arc();
    collision_bus.set_reference_track(2, Some(1));
    let mut kick_eq = ProQ4Processor::with_collision_bus(1, collision_bus.clone());
    let mut bass_eq = ProQ4Processor::with_collision_bus(2, collision_bus.clone());
    kick_eq.params.global.show_collisions = true;
    bass_eq.params.global.show_collisions = true;
    kick_eq.prepare(sr, 512);
    bass_eq.prepare(sr, 512);

    let ctx_512 = ProcessContext {
        sample_rate: sr,
        slice_frames: 512,
        ..ProcessContext::default()
    };
    for blk in 0..24usize {
        let mut kick_buf = [[0.0f32; 2]; 512];
        let mut bass_buf = [[0.0f32; 2]; 512];
        for i in 0..512 {
            let t = ((blk * 512 + i) as f64) / sr;
            let s65 = (0.65 * (TAU * 65.0 * t).sin()) as f32;
            kick_buf[i] = [s65, s65];
            bass_buf[i] = [s65, s65];
        }
        kick_eq.process_slice(&ctx_512, &mut kick_buf, None, &mut []);
        bass_eq.process_slice(&ctx_512, &mut bass_buf, None, &mut []);
        let mut dummy_frame = ProQ4SpectrumFrame::default();
        kick_eq.export_spectrum_frame(&mut dummy_frame);
    }
    let mut bass_spec_frame = ProQ4SpectrumFrame::default();
    bass_eq.export_spectrum_frame(&mut bass_spec_frame);

    let mut peak_bin = 0usize;
    let mut peak_collision = 0.0f32;
    for b in 0..SPECTRUM_FRAME_BINS {
        if bass_spec_frame.collision_mask[b] > peak_collision {
            peak_collision = bass_spec_frame.collision_mask[b];
            peak_bin = b;
        }
    }
    let collision_peak_hz =
        20.0 * (20_000.0_f64 / 20.0).powf((peak_bin as f64) / ((SPECTRUM_FRAME_BINS - 1) as f64));
    assert!(
        peak_collision > 0.35 && (collision_peak_hz - 65.0).abs() <= 5.0,
        "Gate 7.2 Collision peak must occur at 65 Hz ± 5 Hz, got {collision_peak_hz:.2} Hz (mask={peak_collision:.3})"
    );

    // ------------------------------------------------------------------------
    // 7.3: Soothe3 External Sidechain (Pad ducked by 2.5 kHz Vocal SC)
    // ------------------------------------------------------------------------
    let mut soothe = Soothe3Processor::new(sr as f32);
    soothe.set_parameter_normalized(16, 1.0); // Param ID 16 = external_sidechain_enabled
    assert!(soothe.params().global.external_sidechain_enabled);
    soothe.params_mut().global.depth = 5.5;
    soothe.params_mut().global.detail = 6.0;
    soothe.params_mut().global.max_cut_db = -24.0;
    soothe.prepare(sr, 512);

    let total_frames = 16_384usize;
    let mut pad_no_sc = Vec::with_capacity(total_frames);
    let sc_silent = [[0.0f32; 2]; 512];
    for blk in 0..(total_frames / 512) {
        let mut buf = [[0.0f32; 2]; 512];
        for i in 0..512 {
            let t = ((blk * 512 + i) as f64) / sr;
            let p = (0.25 * (TAU * 200.0 * t).sin()
                + 0.25 * (TAU * 2500.0 * t).sin()
                + 0.25 * (TAU * 8000.0 * t).sin()) as f32;
            buf[i] = [p, p];
        }
        soothe.process_slice(&ctx_512, &mut buf, Some(&sc_silent), &mut []);
        for f in &buf {
            pad_no_sc.push(f[0]);
        }
    }

    let mut pad_with_sc = Vec::with_capacity(total_frames);
    let mut prev_s = *pad_no_sc.last().unwrap_or(&0.0);
    let mut soothe_max_step = 0.0f32;
    for blk in 0..(total_frames / 512) {
        let mut buf = [[0.0f32; 2]; 512];
        let mut sc_buf = [[0.0f32; 2]; 512];
        for i in 0..512 {
            let n = blk * 512 + i;
            let t = ((total_frames + n) as f64) / sr;
            let p = (0.25 * (TAU * 200.0 * t).sin()
                + 0.25 * (TAU * 2500.0 * t).sin()
                + 0.25 * (TAU * 8000.0 * t).sin()) as f32;
            let env = ((n as f32) / 256.0).min(1.0);
            let v = env * (0.55 * (TAU * 2500.0 * t).sin()) as f32;
            buf[i] = [p, p];
            sc_buf[i] = [v, v];
        }
        soothe.process_slice(&ctx_512, &mut buf, Some(&sc_buf), &mut []);
        for f in &buf {
            soothe_max_step = soothe_max_step.max((f[0] - prev_s).abs());
            prev_s = f[0];
            pad_with_sc.push(f[0]);
        }
    }

    let steady = 8_192..total_frames;
    let soothe_cut_2k5_db = goertzel_mag_db_f32(&pad_no_sc[steady.clone()], 2500.0, sr)
        - goertzel_mag_db_f32(&pad_with_sc[steady.clone()], 2500.0, sr);
    let soothe_diff_200_db = (goertzel_mag_db_f32(&pad_no_sc[steady.clone()], 200.0, sr)
        - goertzel_mag_db_f32(&pad_with_sc[steady.clone()], 200.0, sr))
    .abs();
    let soothe_diff_8k_db = (goertzel_mag_db_f32(&pad_no_sc[steady.clone()], 8000.0, sr)
        - goertzel_mag_db_f32(&pad_with_sc[steady], 8000.0, sr))
    .abs();

    assert!(
        soothe_cut_2k5_db >= 8.0,
        "Gate 7.3 Soothe3 Ext SC cut at 2.5 kHz must be >= 8.0 dB, got {soothe_cut_2k5_db:.2} dB"
    );
    assert!(
        soothe_diff_200_db < 1.0 && soothe_diff_8k_db < 1.0,
        "Gate 7.3 Soothe3 Ext SC must keep 200 Hz ({soothe_diff_200_db:.3} dB) & 8 kHz ({soothe_diff_8k_db:.3} dB) < 1.0 dB"
    );
    assert!(soothe_max_step < 0.45, "Gate 7.3 click-free check failed");

    // ------------------------------------------------------------------------
    // 7.4: Ozone 12 Low End Focus & Psychoacoustic Codec Preview
    // ------------------------------------------------------------------------
    let lef_frames = 48_000usize;
    let mut dry_mix = Vec::with_capacity(lef_frames);
    let period = 12_000usize; // 4 kicks per second
    for n in 0..lef_frames {
        let t = (n as f64) / sr;
        let pos_in_beat = (n % period) as f64 / sr;
        let kick_punch = (-pos_in_beat / 0.018).exp() * (TAU * 62.0 * pos_in_beat).sin();
        let bass_sustain = 0.28 * (TAU * 55.0 * t).sin();
        let hi_ref = 0.18 * (TAU * 1200.0 * t).sin() + 0.14 * (TAU * 4200.0 * t).sin();
        let s = (0.55 * kick_punch + bass_sustain + hi_ref) as f32;
        dry_mix.push([s, s]);
    }

    // Bypassed LR4 all-pass reference (`contrast = 0%`) vs `Punchy +90%` (`cutoff = 150 Hz`)
    let mut lef_ref = LowEndFocusModule::new(sr);
    lef_ref.set_cutoff_hz(150.0);
    lef_ref.mode = LowEndFocusMode::Punchy;
    lef_ref.contrast_percent = 0.0;

    let mut lef_punchy = LowEndFocusModule::new(sr);
    lef_punchy.set_cutoff_hz(150.0);
    lef_punchy.mode = LowEndFocusMode::Punchy;
    lef_punchy.contrast_percent = 90.0;

    let mut buf_ref = dry_mix.clone();
    let mut buf_punchy = dry_mix.clone();
    for chunk in buf_ref.chunks_mut(512) {
        lef_ref.process_frames_in_place(chunk);
    }
    for chunk in buf_punchy.chunks_mut(512) {
        lef_punchy.process_frames_in_place(chunk);
    }

    // Measure < 150 Hz Crest Factor on steady beats [12_000 .. 48_000]
    let mut lp_meas_ref = LinkwitzRiley4Splitter2Way::new(150.0, sr);
    let mut lp_meas_pun = LinkwitzRiley4Splitter2Way::new(150.0, sr);
    let mut peak_sq_ref = 0.0_f64;
    let mut sum_sq_ref = 0.0_f64;
    let mut peak_sq_pun = 0.0_f64;
    let mut sum_sq_pun = 0.0_f64;
    let mut hi_ref_samples = Vec::new();
    let mut hi_pun_samples = Vec::new();

    for i in 0..lef_frames {
        let ([lo_r, _], [hi_r, _]) =
            lp_meas_ref.split_stereo([buf_ref[i][0] as f64, buf_ref[i][1] as f64]);
        let ([lo_p, _], [hi_p, _]) =
            lp_meas_pun.split_stereo([buf_punchy[i][0] as f64, buf_punchy[i][1] as f64]);
        if i >= 12_000 {
            peak_sq_ref = peak_sq_ref.max(lo_r * lo_r);
            sum_sq_ref += lo_r * lo_r;
            peak_sq_pun = peak_sq_pun.max(lo_p * lo_p);
            sum_sq_pun += lo_p * lo_p;
            hi_ref_samples.push(hi_r as f32);
            hi_pun_samples.push(hi_p as f32);
        }
    }
    let n_eval = (lef_frames - 12_000) as f64;
    let crest_ref_db = 10.0 * (peak_sq_ref / (sum_sq_ref / n_eval).max(1e-18)).log10();
    let crest_pun_db = 10.0 * (peak_sq_pun / (sum_sq_pun / n_eval).max(1e-18)).log10();
    let lef_crest_boost_db = crest_pun_db - crest_ref_db;

    let hi_1200_diff_db = (goertzel_mag_db_f32(&hi_ref_samples, 1200.0, sr)
        - goertzel_mag_db_f32(&hi_pun_samples, 1200.0, sr))
    .abs();
    let hi_4200_diff_db = (goertzel_mag_db_f32(&hi_ref_samples, 4200.0, sr)
        - goertzel_mag_db_f32(&hi_pun_samples, 4200.0, sr))
    .abs();
    let lef_high_band_max_diff_db = hi_1200_diff_db.max(hi_4200_diff_db);

    assert!(
        lef_crest_boost_db >= 1.5,
        "Gate 7.4 Low End Focus Punchy must boost < 150 Hz Crest Factor >= 1.5 dB, got {lef_crest_boost_db:.2} dB"
    );
    assert!(
        lef_high_band_max_diff_db < 0.1,
        "Gate 7.4 Low End Focus > 500 Hz invariance must be < 0.1 dB, got {lef_high_band_max_diff_db:.4} dB"
    );

    // Codec Preview: PCM Bypass Artifact Solo null <= -110 dBFS & MP3 128 vs MP3 320 >= 6.0 dB
    let mut codec_pcm = CodecPreviewModule::new(sr);
    codec_pcm.enabled = true;
    codec_pcm.mode = CodecTarget::PcmBypass;
    codec_pcm.artifact_solo = true;

    let mut codec_128 = CodecPreviewModule::new(sr);
    codec_128.enabled = true;
    codec_128.mode = CodecTarget::Mp3_128;
    codec_128.artifact_solo = true;

    let mut codec_320 = CodecPreviewModule::new(sr);
    codec_320.enabled = true;
    codec_320.mode = CodecTarget::Mp3_320;
    codec_320.artifact_solo = true;

    let codec_len = 24_576usize;
    let mut codec_src = Vec::with_capacity(codec_len);
    for n in 0..codec_len {
        let t = (n as f64) / sr;
        let l = (0.28 * (TAU * 220.0 * t).sin()
            + 0.22 * (TAU * 1350.0 * t).sin()
            + 0.16 * (TAU * 5400.0 * t).sin()
            + 0.10 * (TAU * 11200.0 * t).sin()) as f32;
        let r = (0.28 * (TAU * 220.0 * t).sin()
            + 0.22 * (TAU * 1355.0 * t).cos()
            + 0.16 * (TAU * 5420.0 * t).sin()
            + 0.10 * (TAU * 11250.0 * t).cos()) as f32;
        codec_src.push([l, r]);
    }

    let mut pcm_out = codec_src.clone();
    let mut mp3_128_out = codec_src.clone();
    let mut mp3_320_out = codec_src.clone();
    for chunk in pcm_out.chunks_mut(512) {
        codec_pcm.process_frames_in_place(chunk);
    }
    for chunk in mp3_128_out.chunks_mut(512) {
        codec_128.process_frames_in_place(chunk);
    }
    for chunk in mp3_320_out.chunks_mut(512) {
        codec_320.process_frames_in_place(chunk);
    }

    let rms_db_slice = |frames: &[StereoFrame]| -> f64 {
        let mut acc = 0.0_f64;
        for f in frames {
            acc += 0.5 * ((f[0] as f64).powi(2) + (f[1] as f64).powi(2));
        }
        10.0 * (acc / (frames.len() as f64)).max(1e-24).log10()
    };

    let pcm_null_db = rms_db_slice(&pcm_out[4096..]);
    let mp3_128_art_db = rms_db_slice(&mp3_128_out[4096..]);
    let mp3_320_art_db = rms_db_slice(&mp3_320_out[4096..]);
    let mp3_noise_sep_db = mp3_128_art_db - mp3_320_art_db;

    assert!(
        pcm_null_db <= -110.0,
        "Gate 7.4 Codec Preview PCM Bypass Artifact Solo must null <= -110 dBFS, got {pcm_null_db:.2} dBFS"
    );
    assert!(
        mp3_noise_sep_db >= 6.0,
        "Gate 7.4 MP3 128 artifact noise ({mp3_128_art_db:.2} dB) must exceed MP3 320 ({mp3_320_art_db:.2} dB) by >= 6.0 dB (got {mp3_noise_sep_db:.2} dB)"
    );

    save_gate_metrics(
        "gate7",
        json!({
            "gate": "Gate 7 — Workflow (20-Step Undo/Redo) & 100% Internal Plugin Parity",
            "status": "PASS",
            "undo_redo_20_steps": {
                "steps_tested": 20,
                "state_bit_exact_after_20_undo": true,
                "state_bit_exact_after_20_redo": true,
                "max_audio_ulp_diff_undo": max_undo_bit_diff,
                "max_audio_ulp_diff_redo": max_redo_bit_diff,
            },
            "proq4_parity": {
                "eq_sketch_rms_error_db_80hz_15khz": sketch_rms_err_db,
                "collision_peak_hz": collision_peak_hz,
                "collision_peak_mask": peak_collision,
            },
            "soothe3_external_sidechain": {
                "cut_2500hz_db": soothe_cut_2k5_db,
                "delta_200hz_db": soothe_diff_200_db,
                "delta_8000hz_db": soothe_diff_8k_db,
                "max_sample_step": soothe_max_step,
            },
            "ozone12_parity": {
                "low_end_focus_crest_boost_db": lef_crest_boost_db,
                "low_end_focus_above_500hz_diff_db": lef_high_band_max_diff_db,
                "codec_pcm_bypass_artifact_solo_dbfs": pcm_null_db,
                "codec_mp3_128_artifact_rms_dbfs": mp3_128_art_db,
                "codec_mp3_320_artifact_rms_dbfs": mp3_320_art_db,
                "codec_mp3_128_vs_320_separation_db": mp3_noise_sep_db,
            }
        }),
    );
}

// ============================================================================
// GATE 8: REAL VST3 3.7 COM & CLAP 1.2 C ABI HOSTING + 5-POINT APDC + 0 ALLOC
// ============================================================================

#[test]
fn gate8_vst3_clap_real_binary_hosting_and_apdc_zero_alloc() {
    let sr = 48_000.0;
    let dll_path = locate_compiled_test_plugin_dll();

    // Stage a clean temporary scan folder containing `.clap`, `.vst3`, AND a corrupt binary to verify
    // both out-of-process discovery AND crash-isolated quarantine.
    let scan_dir = std::env::temp_dir().join("toneforge_gate8_plugin_scan");
    let _ = fs::remove_dir_all(&scan_dir);
    fs::create_dir_all(&scan_dir).expect("create temp plugin scan dir");
    let staged_clap = scan_dir.join("toneforge_ref_plugin.clap");
    let staged_vst3 = scan_dir.join("toneforge_ref_plugin.vst3");
    let staged_corrupt = scan_dir.join("corrupt_rogue_plugin.vst3");
    fs::copy(&dll_path, &staged_clap).expect("copy .clap binary");
    fs::copy(&dll_path, &staged_vst3).expect("copy .vst3 binary");
    fs::write(&staged_corrupt, b"NOT_A_VALID_PE_DLL_BINARY").expect("write corrupt binary");
    let quarantine_file = scan_dir.join("quarantine.json");

    let scan_report = scan_plugin_directories_isolated(
        std::slice::from_ref(&scan_dir),
        Some(&quarantine_file),
        Duration::from_millis(5_000),
    );
    assert!(
        scan_report.plugins.len() >= 2 && !scan_report.quarantined.is_empty(),
        "Crash-isolated scanner must discover both CLAP & VST3 plugins ({}) and quarantine rogue binary ({})",
        scan_report.plugins.len(),
        scan_report.quarantined.len()
    );

    // ------------------------------------------------------------------------
    // 8.1: Real LoadLibraryW Hosting, MIDI NoteOn A4=440 Hz, Automation & State
    // ------------------------------------------------------------------------
    let mut clap_node =
        ClapHostedPluginNode::load(&staged_clap, None, sr, 256).expect("Load CLAP 1.2 plugin");
    let mut vst3_node =
        Vst3HostedPluginNode::load(&staged_vst3, None, sr, 256).expect("Load VST3 3.7 COM plugin");

    assert_eq!(clap_node.latency_samples(), 64);
    assert_eq!(vst3_node.latency_samples(), 64);

    // Trigger A4 = MIDI 69 = 440.0 Hz on both CLAP and VST3
    clap_node.queue_note_on(69, 0.90, 0, 0);
    vst3_node.queue_note_on(69, 0.90, 0, 0);

    let ctx_256 = ProcessContext {
        sample_rate: sr,
        slice_frames: 256,
        ..ProcessContext::default()
    };

    let render_plugin_note = |node: &mut dyn UnifiedDspNode| -> Vec<f32> {
        let mut samples = Vec::with_capacity(16_384);
        for _ in 0..(16_384 / 256) {
            let mut block = [[0.0f32; 2]; 256];
            node.process_slice(&ctx_256, &mut block, None, &mut []);
            for f in &block {
                samples.push(f[0]);
            }
        }
        samples
    };

    let clap_wave = render_plugin_note(&mut clap_node);
    let vst3_wave = render_plugin_note(&mut vst3_node);

    let clap_f0_hz = measure_fft_peak_freq_hz(&clap_wave[2048..], sr);
    let vst3_f0_hz = measure_fft_peak_freq_hz(&vst3_wave[2048..], sr);
    assert!(
        (clap_f0_hz - 440.0).abs() <= 1.0,
        "Gate 8.1 CLAP NoteOn A4 must produce 440 ± 1 Hz, got {clap_f0_hz:.3} Hz"
    );
    assert!(
        (vst3_f0_hz - 440.0).abs() <= 1.0,
        "Gate 8.1 VST3 NoteOn A4 must produce 440 ± 1 Hz, got {vst3_f0_hz:.3} Hz"
    );

    // Parameter automation + Binary state save/load 100% roundtrip
    clap_node.set_parameter_normalized(1, 0.25); // SVF Cutoff
    clap_node.set_parameter_normalized(2, 0.62); // SVF Resonance
    let mut dummy = [[0.0f32; 2]; 256];
    clap_node.process_slice(&ctx_256, &mut dummy, None, &mut []);
    let clap_saved = clap_node.save_state().expect("CLAP save_state");

    let mut clap_restored =
        ClapHostedPluginNode::load(&staged_clap, None, sr, 256).expect("Load fresh CLAP");
    clap_restored
        .load_state(&clap_saved)
        .expect("CLAP load_state");
    let clap_resaved = clap_restored.save_state().expect("CLAP re-save");
    assert_eq!(
        clap_saved, clap_resaved,
        "Gate 8.1 CLAP binary state save/load roundtrip must match 100%"
    );

    vst3_node.set_parameter_normalized(1, 0.33);
    vst3_node.set_parameter_normalized(2, 0.71);
    vst3_node.process_slice(&ctx_256, &mut dummy, None, &mut []);
    let vst3_saved = vst3_node.save_state().expect("VST3 save_state");

    let mut vst3_restored =
        Vst3HostedPluginNode::load(&staged_vst3, None, sr, 256).expect("Load fresh VST3");
    vst3_restored
        .load_state(&vst3_saved)
        .expect("VST3 load_state");
    let vst3_resaved = vst3_restored.save_state().expect("VST3 re-save");
    assert_eq!(
        vst3_saved, vst3_resaved,
        "Gate 8.1 VST3 IBStream binary state save/load roundtrip must match 100%"
    );

    // ------------------------------------------------------------------------
    // 8.2: 5-Point APDC Parallel Null Test (64-sample latency -> <= -90 dBFS)
    //      & #[global_allocator] Zero-Alloc / Zero-Dealloc Verification
    // ------------------------------------------------------------------------
    let mut apdc_clap =
        ClapHostedPluginNode::load(&staged_clap, None, sr, 256).expect("Load APDC CLAP");
    // Synth gain = 0.0, Dry Passthrough Mix = 1.0 (pure 64-sample delayed passthrough)
    apdc_clap.set_parameter_normalized(0, 0.0);
    apdc_clap.set_parameter_normalized(3, 1.0);

    let (mut mixer, _, _) = MixerEngine::new(sr, 128.0, 0);
    // Feed identical deterministic multitone into Track 1 (64-sample CLAP plugin)
    // and phase-inverted deterministic multitone into Track 2 (0 latency)
    mixer
        .add_channel_generator(ChannelGenerator::new(
            1,
            1,
            Box::new(DeterministicGeneratorNode::new(0, 0.5, false)),
        ))
        .unwrap();
    mixer
        .add_channel_generator(ChannelGenerator::new(
            2,
            2,
            Box::new(DeterministicGeneratorNode::new(0, 0.5, true)),
        ))
        .unwrap();

    mixer.tracks[1]
        .set_or_replace_plugin_slot(0, Box::new(apdc_clap), 1.0, sr, 256)
        .unwrap();
    mixer.tracks[1].sends.clear();
    mixer.tracks[2].sends.clear();
    mixer.tracks[1].add_send(0, 1.0, false);
    mixer.tracks[2].add_send(0, 1.0, false);
    mixer.is_playing = true;
    mixer.rebuild_graph().expect("compile APDC DAG");

    // Warm up 8 blocks (2,048 frames) so the 64-sample lookahead ring & APDC delay lines fill
    let mut out_blk = [[0.0f32; 2]; 256];
    for _ in 0..8 {
        mixer.process_audio_block(&mut out_blk);
    }

    // Measure null residual AND strict Zero-Alloc / Zero-Dealloc across 16 real-time blocks (4,096 frames)
    let (null_rms_dbfs, mixer_rt_allocs, mixer_rt_deallocs) = measure_rt_allocs(|| {
        let mut sq_sum = 0.0_f64;
        let mut count = 0usize;
        for _ in 0..16 {
            mixer.process_audio_block(&mut out_blk);
            for f in &out_blk {
                sq_sum += 0.5 * ((f[0] as f64).powi(2) + (f[1] as f64).powi(2));
                count += 1;
            }
        }
        10.0 * (sq_sum / (count as f64)).max(1e-24).log10()
    });

    // Also audit VST3 process_slice directly under CountingSystemAllocator
    let (_, vst3_rt_allocs, vst3_rt_deallocs) = measure_rt_allocs(|| {
        for i in 0..16u32 {
            vst3_node.set_parameter_normalized(1, 0.30 + (i as f32) * 0.02);
            vst3_node.process_slice(&ctx_256, &mut dummy, None, &mut []);
        }
    });

    assert!(
        null_rms_dbfs <= -90.0,
        "Gate 8.2 5-Point APDC parallel null cancellation must be <= -90 dBFS, got {null_rms_dbfs:.2} dBFS"
    );
    assert_eq!(
        mixer_rt_allocs + mixer_rt_deallocs + vst3_rt_allocs + vst3_rt_deallocs,
        0,
        "Gate 8.2 RT process_slice must perform 0 allocs and 0 deallocs (mixer={mixer_rt_allocs}/{mixer_rt_deallocs}, vst3={vst3_rt_allocs}/{vst3_rt_deallocs})"
    );

    let _ = fs::remove_dir_all(&scan_dir);

    save_gate_metrics(
        "gate8",
        json!({
            "gate": "Gate 8 — Real VST3 3.7 COM & CLAP 1.2 C ABI Hosting + 5-Point APDC + Zero-Alloc",
            "status": "PASS",
            "scanner_discovered_count": scan_report.plugins.len(),
            "scanner_quarantined_rogue_count": scan_report.quarantined.len(),
            "clap_a4_freq_hz": clap_f0_hz,
            "vst3_a4_freq_hz": vst3_f0_hz,
            "clap_state_roundtrip_exact": true,
            "vst3_state_roundtrip_exact": true,
            "apdc_64_sample_null_dbfs": null_rms_dbfs,
            "rt_alloc_calls": mixer_rt_allocs + vst3_rt_allocs,
            "rt_dealloc_calls": mixer_rt_deallocs + vst3_rt_deallocs,
        }),
    );
}

// ============================================================================
// GATE 9: ASIO / WASAPI DUPLEX + 60s +150 PPM DLL DRIFT + DISK LOOP COMPING
// ============================================================================

#[test]
fn gate9_asio_wasapi_duplex_drift_and_lockfree_disk_recorder() {
    // ------------------------------------------------------------------------
    // 9.1: Hardware ASIO / WASAPI Capture / MIDI In Enumeration + 60s +150 ppm Drift
    // ------------------------------------------------------------------------
    let asio_list = enumerate_asio_drivers();
    let wasapi_inputs = enumerate_wasapi_input_devices();
    let midi_inputs = enumerate_midi_input_devices();

    let engine_sr = 48_000.0_f64;
    let drift_ppm = 150.0_f64;
    let capture_sr = engine_sr * (1.0 + drift_ppm * 1e-6); // 48,007.2 Hz (+150 ppm fast clock)
    let tone_hz = 997.0_f64;

    // Blind nominal 48 kHz / 48 kHz initialization: DLL must autonomously lock onto +150 ppm hardware clock drift!
    let mut dll_resampler = AdaptiveDllDriftResampler::new(engine_sr, engine_sr);

    // Prime ring buffer to nominal anchor (~2,048 frames) before starting 60s duplex clock run
    let mut cap_sample_idx = 0u64;
    let mut prime_buf = Vec::with_capacity(2048);
    for _ in 0..2048 {
        let t = (cap_sample_idx as f64) / capture_sr;
        let s = (0.75 * (TAU * tone_hz * t).sin()) as f32;
        prime_buf.push([s, -s]);
        cap_sample_idx += 1;
    }
    dll_resampler.push_captured_frames(&prime_buf);

    let total_blocks_60s = (60 * 48_000) / 256; // 11,250 blocks = 2,880,000 frames (60.0s)
    let mut cap_frac_accum = 0.0_f64;
    let cap_per_block = 256.0 * (capture_sr / engine_sr);
    let mut capture_scratch = [[0.0f32; 2]; 300];
    let mut engine_out = [[0.0f32; 2]; 256];
    let eval_len = 48_000usize; // Exact 1.000s steady-state window (integer 997 cycles @ 48,000 Hz)
    let tail_start_blk = total_blocks_60s - (eval_len / 256) - 2;
    let mut tail_48k = Vec::with_capacity(eval_len + 512);

    let (_, dll_rt_allocs, dll_rt_deallocs) = measure_rt_allocs(|| {
        for blk in 0..total_blocks_60s {
            cap_frac_accum += cap_per_block;
            let n_cap = cap_frac_accum.floor() as usize;
            cap_frac_accum -= n_cap as f64;

            for slot in capture_scratch.iter_mut().take(n_cap) {
                let t = (cap_sample_idx as f64) / capture_sr;
                let s = (0.75 * (TAU * tone_hz * t).sin()) as f32;
                *slot = [s, -s];
                cap_sample_idx += 1;
            }
            dll_resampler.push_captured_frames(&capture_scratch[..n_cap]);
            dll_resampler.process_output_block(&mut engine_out);

            if blk >= tail_start_blk {
                for f in &engine_out {
                    if tail_48k.len() < eval_len {
                        tail_48k.push(f[0]);
                    }
                }
            }
        }
    });

    assert_eq!(
        dll_resampler.total_xruns(),
        0,
        "Gate 9.1 60s +150 ppm drift run must produce 0 xruns"
    );
    assert_eq!(
        dll_rt_allocs + dll_rt_deallocs,
        0,
        "Gate 9.1 AdaptiveDllDriftResampler must perform 0 RT heap allocations"
    );
    let locked_ppm = dll_resampler.current_ppm_offset();
    assert!(
        (locked_ppm - drift_ppm).abs() < 2.0,
        "Gate 9.1 Adaptive DLL must lock onto +150 ppm clock drift, got {locked_ppm:.3} ppm"
    );

    // Measure coherent full-band SINAD / SNR across all 48,000 steady-state output samples (1.0s exact)
    let mut sum_cc = 0.0_f64;
    let mut sum_ss = 0.0_f64;
    let mut sum_yc = 0.0_f64;
    let mut sum_ys = 0.0_f64;
    for (i, &x) in tail_48k.iter().enumerate() {
        let ph = TAU * tone_hz * (i as f64) / engine_sr;
        let c = ph.cos();
        let s = ph.sin();
        let y = x as f64;
        sum_cc += c * c;
        sum_ss += s * s;
        sum_yc += y * c;
        sum_ys += y * s;
    }
    let a_cos = sum_yc / sum_cc;
    let b_sin = sum_ys / sum_ss;
    let mut sig_pow = 0.0_f64;
    let mut noise_pow = 0.0_f64;
    for (i, &x) in tail_48k.iter().enumerate() {
        let ph = TAU * tone_hz * (i as f64) / engine_sr;
        let ideal = a_cos * ph.cos() + b_sin * ph.sin();
        let err = (x as f64) - ideal;
        sig_pow += ideal * ideal;
        noise_pow += err * err;
    }
    let dll_snr_db = 10.0 * (sig_pow / noise_pow.max(1e-30)).log10();
    let measured_out_hz = measure_fft_peak_freq_hz(&tail_48k[..32_768], engine_sr);

    assert!(
        dll_snr_db >= 95.0,
        "Gate 9.1 Adaptive DLL Resampler SNR must be >= 95.0 dB, got {dll_snr_db:.2} dB"
    );

    // ------------------------------------------------------------------------
    // 9.2: Lock-Free DiskTrackRecorder + Round-Trip Latency + 3-Take Loop Comping
    // ------------------------------------------------------------------------
    let rec_dir = std::env::temp_dir().join("toneforge_gate9_disk_rec");
    let _ = fs::remove_dir_all(&rec_dir);
    fs::create_dir_all(&rec_dir).expect("create temp disk rec dir");

    let rtl_samples = 312usize;
    let loop_frames = 4_800usize;
    let (recorder, mut rt_tap) = DiskTrackRecorder::spawn(
        rec_dir.clone(),
        1,
        48_000,
        rtl_samples,
        Some(loop_frames),
        65_536,
    );

    let total_stream_frames = rtl_samples + 3 * loop_frames;
    let mut stream_data = vec![[0.0f32; 2]; total_stream_frames];
    for take_idx in 0..3usize {
        let onset = rtl_samples + take_idx * loop_frames;
        stream_data[onset] = [0.92, -0.92];
    }

    let (_, rec_rt_allocs, rec_rt_deallocs) = measure_rt_allocs(|| {
        for slice in stream_data.chunks(128) {
            rt_tap.push_rt_slice(slice);
        }
    });

    assert_eq!(rt_tap.dropped_frames, 0);
    assert_eq!(
        rec_rt_allocs + rec_rt_deallocs,
        0,
        "Gate 9.2 RtRecordTap::push_rt_slice must perform 0 RT alloc/free"
    );

    let takes = recorder.finish_and_collect_takes();
    assert_eq!(takes.len(), 3, "Gate 9.2 must finalize all 3 loop takes");

    let mut impulse_alignment_errors = Vec::new();
    for (idx, take) in takes.iter().enumerate() {
        assert_eq!(take.num_frames, loop_frames);
        let (wav_sr, frames) =
            read_wav_f32_stereo(&take.wav_path).expect("read recorded take WAV");
        assert_eq!(wav_sr, 48_000);
        assert_eq!(frames.len(), loop_frames);
        let first_onset = frames
            .iter()
            .position(|f| f[0].abs() > 0.5)
            .expect("find recorded impulse");
        impulse_alignment_errors.push(first_onset);
        assert_eq!(
            first_onset,
            0,
            "Gate 9.2 Take #{} impulse must align at sample 0 (got {})",
            idx + 1,
            first_onset
        );
    }

    let _ = fs::remove_dir_all(&rec_dir);

    save_gate_metrics(
        "gate9",
        json!({
            "gate": "Gate 9 — Hardware ASIO / WASAPI Duplex Drift Resampler & Lock-Free Disk Recorder",
            "status": "PASS",
            "enumerated_asio_drivers": asio_list.iter().map(|d| &d.description).collect::<Vec<_>>(),
            "enumerated_wasapi_inputs_count": wasapi_inputs.len(),
            "enumerated_midi_inputs_count": midi_inputs.len(),
            "dll_60s_150ppm": {
                "xruns": dll_resampler.total_xruns(),
                "snr_db": dll_snr_db,
                "measured_output_tone_hz": measured_out_hz,
                "locked_ppm_offset": dll_resampler.current_ppm_offset(),
            },
            "disk_recorder_loop_comping": {
                "takes_recorded": takes.len(),
                "impulse_sample_errors": impulse_alignment_errors,
                "rt_alloc_calls": rec_rt_allocs,
                "rt_dealloc_calls": rec_rt_deallocs,
            }
        }),
    );
}

// ============================================================================
// GATE 10: STUDIO VOCAL PITCH (pYIN + FORMANT PRESERVE) & TRANSIENT TIME-STRETCH
// ============================================================================

fn synth_formant_vowel_signal(
    len: usize,
    sr: f64,
    base_f0_hz: f64,
    vib_rate_hz: f64,
    vib_depth_cents: f64,
    formants_hz: [f64; 3],
) -> (Vec<f32>, Vec<f64>) {
    let mut out = Vec::with_capacity(len);
    let mut inst_f0 = Vec::with_capacity(len);
    let mut phase = 0.0_f64;

    let env_gain = |f: f64| -> f64 {
        let r1 = 1.0 / (1.0 + ((f - formants_hz[0]) / 65.0).powi(2));
        let r2 = 0.85 / (1.0 + ((f - formants_hz[1]) / 85.0).powi(2));
        let r3 = 0.50 / (1.0 + ((f - formants_hz[2]) / 120.0).powi(2));
        r1 + r2 + r3 + 0.03
    };

    for n in 0..len {
        let t = (n as f64) / sr;
        let cents = vib_depth_cents * (TAU * vib_rate_hz * t).sin();
        let f0 = base_f0_hz * 2.0_f64.powf(cents / 1200.0);
        inst_f0.push(f0);
        phase += TAU * f0 / sr;

        let mut acc = 0.0_f64;
        for h in 1..=22usize {
            let hf = (h as f64) * f0;
            if hf < 0.46 * sr {
                acc += env_gain(hf) * ((h as f64) * phase).sin();
            }
        }
        out.push((acc * 0.15) as f32);
    }
    (out, inst_f0)
}

#[test]
fn gate10_vocal_pyin_formant_shifter_and_polyphonic_time_stretch() {
    let sr = 48_000.0;

    // ------------------------------------------------------------------------
    // 10.1: pYIN < 2.0 cents RMS on 5.5 Hz ±35 cents vibrato + 100% sibilant guard
    // ------------------------------------------------------------------------
    let voiced_len = 57_600usize; // 1.2s voiced vibrato @ 220 Hz
    let (mut pyin_sig, true_f0) =
        synth_formant_vowel_signal(voiced_len, sr, 220.0, 5.5, 35.0, [730.0, 1090.0, 2440.0]);

    // Append 0.5s (24,000 samples) unvoiced sibilant noise (5..10 kHz)
    let sib_len = 24_000usize;
    let mut lcg = 0xCAFE_BABE_1234_5678_u64;
    let (mut z1, mut z2) = (0.0_f64, 0.0_f64);
    for _ in 0..sib_len {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
        let w = ((lcg >> 32) as i32 as f64) / (i32::MAX as f64);
        let hp = w - 1.45 * z1 + 0.65 * z2;
        z2 = z1;
        z1 = w;
        pyin_sig.push((hp * 0.12) as f32);
    }

    let mut detector = PyinPitchDetector::new(sr);
    let hop = 256usize;
    let req_len = detector.required_frame_len();

    let mut cents_sq_sum = 0.0_f64;
    let mut voiced_eval_frames = 0usize;
    let mut center = 4_800usize;
    while center <= voiced_len - 4_800 {
        let res = detector.analyze_centered_at(&pyin_sig[..voiced_len], center);
        assert!(res.is_voiced, "Interior vibrato frame at {center} must be voiced");
        let cents_err = res.cents_error_from(true_f0[center]);
        cents_sq_sum += cents_err * cents_err;
        voiced_eval_frames += 1;
        center += hop;
    }
    let pyin_rms_cents = (cents_sq_sum / (voiced_eval_frames.max(1) as f64)).sqrt();
    assert!(
        pyin_rms_cents < 2.0,
        "Gate 10.1 pYIN vibrato RMS error must be < 2.0 cents, got {pyin_rms_cents:.3} cents"
    );

    detector.reset();
    let mut sib_frames = 0usize;
    let mut sib_unvoiced = 0usize;
    let mut s_pos = voiced_len + 2_400;
    while s_pos + req_len <= pyin_sig.len() - 1_200 {
        let res = detector.analyze_window(&pyin_sig[s_pos..s_pos + req_len]);
        sib_frames += 1;
        if (res.voiced_prob as f64) < 0.20 && !res.is_voiced {
            sib_unvoiced += 1;
        }
        s_pos += hop;
    }
    let sibilant_rejection_pct = 100.0 * (sib_unvoiced as f64) / (sib_frames.max(1) as f64);
    assert_eq!(
        sib_unvoiced, sib_frames,
        "Gate 10.1 100% of sibilant frames must have voiced_prob < 0.2"
    );

    // ------------------------------------------------------------------------
    // 10.2: +5.0 Semitone Formant-Preserving Pitch Shift (F1=730 Hz, F2=1090 Hz)
    // ------------------------------------------------------------------------
    let vowel_len = 48_000usize;
    let base_f0 = 165.0_f64;
    let (vowel_in, _) =
        synth_formant_vowel_signal(vowel_len, sr, base_f0, 0.0, 0.0, [730.0, 1090.0, 2440.0]);

    let mut shifter = FormantPitchShifter::new(sr);
    let vowel_out =
        shifter.process_offline_compensated(&vowel_in, base_f0 as f32, 5.0, 0.0, true);

    // Measure shifted f0 on steady interior
    let mut det_out = PyinPitchDetector::new(sr);
    let mut f0_sum = 0.0_f64;
    let mut f0_cnt = 0usize;
    let mut p = 12_000usize;
    while p + req_len <= 40_000 {
        let r = det_out.analyze_window(&vowel_out[p..p + req_len]);
        if r.is_voiced {
            f0_sum += r.f0_hz as f64;
            f0_cnt += 1;
        }
        p += hop;
    }
    let shifted_f0_hz = f0_sum / (f0_cnt.max(1) as f64);
    let shifted_cents = 1200.0 * (shifted_f0_hz / base_f0).log2();
    assert!(
        (shifted_cents - 500.0).abs() <= 5.0,
        "Gate 10.2 Shifted f0 must be +500 ± 5 cents, got {shifted_cents:.2} cents"
    );

    // Measure F1 and F2 spectral envelope peaks before and after +5 st shift
    let (freqs_in, env_in) =
        estimate_vocal_formant_envelope_db(&vowel_in[12_000..36_000], sr, base_f0 as f32, 4096);
    let (freqs_out, env_out) = estimate_vocal_formant_envelope_db(
        &vowel_out[12_000..36_000],
        sr,
        shifted_f0_hz as f32,
        4096,
    );

    let f1_in = find_formant_peak_in_band(&freqs_in, &env_in, 450.0, 920.0) as f64;
    let f2_in = find_formant_peak_in_band(&freqs_in, &env_in, 930.0, 1450.0) as f64;
    let f1_out = find_formant_peak_in_band(&freqs_out, &env_out, 450.0, 920.0) as f64;
    let f2_out = find_formant_peak_in_band(&freqs_out, &env_out, 930.0, 1450.0) as f64;

    let f1_drift_pct = ((f1_out - f1_in).abs() / f1_in) * 100.0;
    let f2_drift_pct = ((f2_out - f2_in).abs() / f2_in) * 100.0;
    assert!(
        f1_drift_pct < 8.0 && f2_drift_pct < 8.0,
        "Gate 10.2 Formant F1 ({f1_in:.1}->{f1_out:.1} Hz, {f1_drift_pct:.2}%) & F2 ({f2_in:.1}->{f2_out:.1} Hz, {f2_drift_pct:.2}%) must shift < 8.0%"
    );

    // ------------------------------------------------------------------------
    // 10.3: TransientLockedTimeStretcher (1.35x & 0.75x Kick + Stereo Chord)
    // ------------------------------------------------------------------------
    let mix_len = 48_000usize;
    let kick_onset = 9_600usize; // 0.20s sharp Kick onset
    let mut poly_src = Vec::with_capacity(mix_len);
    for n in 0..mix_len {
        let t = (n as f64) / sr;
        // Stereo sustained chord (gentle level so Kick transient stands out clearly)
        let chord_l = 0.08 * (TAU * 261.63 * t).sin()
            + 0.07 * (TAU * 329.63 * t).sin()
            + 0.07 * (TAU * 392.00 * t).sin();
        let chord_r = 0.08 * (TAU * 261.63 * t + 0.18).sin()
            + 0.07 * (TAU * 329.63 * t - 0.15).sin()
            + 0.07 * (TAU * 392.00 * t + 0.12).sin();

        let mut kick = 0.0_f64;
        if n >= kick_onset && n < kick_onset + 4_800 {
            let kt = ((n - kick_onset) as f64) / sr;
            let body_env = (-kt / 0.022).exp();
            let click_env = (-kt / 0.0018).exp();
            let sweep_phase = TAU * (55.0 * kt + 2.8 * (1.0 - (-kt / 0.008).exp()));
            kick =
                0.82 * body_env * sweep_phase.sin() + 0.55 * click_env * (TAU * 1800.0 * kt).sin();
        }
        poly_src.push([(chord_l + kick) as f32, (chord_r + kick) as f32]);
    }

    let measure_kick_crest = |sig: &[StereoFrame], center_idx: usize| -> f64 {
        let half_w = 2_400usize;
        let lo = center_idx.saturating_sub(600);
        let hi = (center_idx + half_w).min(sig.len());
        let mut peak = 0.0_f64;
        let mut sq = 0.0_f64;
        for f in &sig[lo..hi] {
            let v = (f[0] as f64).abs();
            peak = peak.max(v);
            sq += v * v;
        }
        let rms = (sq / ((hi - lo).max(1) as f64)).sqrt().max(1e-12);
        peak / rms
    };

    let measure_stereo_corr = |sig: &[StereoFrame], start: usize, end: usize| -> f64 {
        let slice = &sig[start.min(sig.len())..end.min(sig.len())];
        let mut sum_lr = 0.0_f64;
        let mut sum_ll = 0.0_f64;
        let mut sum_rr = 0.0_f64;
        for f in slice {
            let l = f[0] as f64;
            let r = f[1] as f64;
            sum_lr += l * r;
            sum_ll += l * l;
            sum_rr += r * r;
        }
        sum_lr / (sum_ll * sum_rr).sqrt().max(1e-18)
    };

    let orig_crest = measure_kick_crest(&poly_src, kick_onset);

    let mut stretcher = TransientLockedTimeStretcher::new(sr);
    let stretched_135 = stretcher.stretch_stereo_offline(&poly_src, 1.35, 0.0);
    let stretched_075 = stretcher.stretch_stereo_offline(&poly_src, 0.75, 0.0);

    let expected_135 = ((mix_len as f64) * 1.35).round() as isize;
    let expected_075 = ((mix_len as f64) * 0.75).round() as isize;
    let len_err_135 = ((stretched_135.len() as isize) - expected_135).abs();
    let len_err_075 = ((stretched_075.len() as isize) - expected_075).abs();
    assert!(
        len_err_135 <= 4 && len_err_075 <= 4,
        "Gate 10.3 Output length error must be <= 4 samples (1.35x err={len_err_135}, 0.75x err={len_err_075})"
    );

    let crest_135 = measure_kick_crest(&stretched_135, ((kick_onset as f64) * 1.35).round() as usize);
    let crest_075 = measure_kick_crest(&stretched_075, ((kick_onset as f64) * 0.75).round() as usize);
    let crest_ratio_135 = crest_135 / orig_crest;
    let crest_ratio_075 = crest_075 / orig_crest;
    assert!(
        crest_ratio_135 >= 0.85 && crest_ratio_075 >= 0.85,
        "Gate 10.3 Kick Transient Crest Factor retention must be >= 85% (1.35x={:.1}%, 0.75x={:.1}%)",
        crest_ratio_135 * 100.0,
        crest_ratio_075 * 100.0
    );

    let corr_135 = measure_stereo_corr(&stretched_135, 32_000, 58_000);
    let corr_075 = measure_stereo_corr(&stretched_075, 18_000, 33_000);
    assert!(
        corr_135 >= 0.95 && corr_075 >= 0.95,
        "Gate 10.3 Stereo chord correlation must be >= 0.95 (1.35x={corr_135:.4}, 0.75x={corr_075:.4})"
    );

    save_gate_metrics(
        "gate10",
        json!({
            "gate": "Gate 10 — Studio Vocal Pitch (pYIN + True-Envelope Formant) & Polyphonic Time-Stretch",
            "status": "PASS",
            "pyin": {
                "vibrato_rms_error_cents": pyin_rms_cents,
                "sibilant_unvoiced_rejection_pct": sibilant_rejection_pct,
            },
            "formant_pitch_shifter_plus_5st": {
                "shifted_f0_cents": shifted_cents,
                "f1_in_hz": f1_in,
                "f1_out_hz": f1_out,
                "f1_drift_pct": f1_drift_pct,
                "f2_in_hz": f2_in,
                "f2_out_hz": f2_out,
                "f2_drift_pct": f2_drift_pct,
            },
            "transient_locked_time_stretcher": {
                "len_error_samples_135x": len_err_135,
                "len_error_samples_075x": len_err_075,
                "kick_crest_retention_pct_135x": crest_ratio_135 * 100.0,
                "kick_crest_retention_pct_075x": crest_ratio_075 * 100.0,
                "stereo_chord_correlation_135x": corr_135,
                "stereo_chord_correlation_075x": corr_075,
            }
        }),
    );
}
