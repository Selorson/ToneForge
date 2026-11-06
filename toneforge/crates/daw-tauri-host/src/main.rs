//! ToneForge Host Entrypoint (`toneforge-host`).

use daw_tauri_host::ToneForgeHostRuntime;

fn main() {
    let mut runtime = ToneForgeHostRuntime::new(48000.0, 512);
    let mut master_out = [[0.0f32; 2]; 256];
    runtime.mixer_engine.process_audio_block(&mut master_out);
    let frame = runtime.telemetry_server.poll_latest_frame();
    println!(
        "ToneForge Host Initialized — Frame Seq: {}, BPM: {:.1}, Active Tracks: 502",
        frame.header.frame_seq, frame.header.bpm
    );
}
