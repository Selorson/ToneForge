//! ToneForge Host Entrypoint (`toneforge-host`).

use daw_tauri_host::AudioStreamController;

fn main() {
    // Run on an 8 MB stack thread so Windows debug builds (which default to a 1 MB main-thread stack)
    // have ample headroom for 502-Track Mixer + 128 KB TripleBuffer + Multi-Plugin initialization.
    let handle = std::thread::Builder::new()
        .name("toneforge-host-main".to_string())
        .stack_size(8 * 1024 * 1024)
        .spawn(|| {
            let mut controller = AudioStreamController::with_mastering_template(48000.0, 512)
                .expect("Failed to initialize ToneForge Mastering Template");
            let mut master_out = [[0.0f32; 2]; 256];
            controller.process_audio_block(&mut master_out);
            let raw_bytes = controller.poll_binary_telemetry_bytes();
            let hdr = daw_tauri_host::ShmTelemetryServer::decode_header_from_bytes(raw_bytes)
                .expect("Failed to decode binary telemetry header");
            println!(
                "ToneForge Host Initialized — Frame Seq: {}, BPM: {:.1}, Binary Frame Size: {} B (128 KB), Active Tracks: 502",
                hdr.frame_seq,
                hdr.bpm,
                raw_bytes.len()
            );
        })
        .expect("Failed to spawn toneforge-host-main thread");

    handle.join().expect("toneforge-host-main panicked");
}
