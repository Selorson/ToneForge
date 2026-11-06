//! SharedArrayBuffer / TripleBuffer Telemetry Server (`shm_telemetry_server.rs`).
//!
//! Exposes the 128 KB (`131,072` bytes) [`VisualTelemetryFrame`] read from the
//! real-time audio thread's [`triple_buffer::Output`] to the UI WebView / WGPU compositor
//! at 60–120 Hz without locking or copying onto the audio thread.

use daw_core_graph::telemetry_shm::{VISUAL_TELEMETRY_FRAME_BYTES, VisualTelemetryFrame};
use triple_buffer::Output;

/// Server wrapper around the consumer side of the `VisualTelemetryFrame` triple buffer.
pub struct ShmTelemetryServer {
    telemetry_output: Output<VisualTelemetryFrame>,
    last_frame_seq: u64,
}

impl ShmTelemetryServer {
    pub fn new(telemetry_output: Output<VisualTelemetryFrame>) -> Self {
        Self {
            telemetry_output,
            last_frame_seq: 0,
        }
    }

    /// Polls the triple buffer for the latest 128 KB frame and returns a zero-copy byte slice.
    #[inline]
    pub fn poll_latest_bytes(&mut self) -> &[u8; VISUAL_TELEMETRY_FRAME_BYTES] {
        let frame = self.telemetry_output.read();
        self.last_frame_seq = frame.header.frame_seq;
        frame.as_bytes()
    }

    /// Polls the triple buffer and returns a reference to the structured [`VisualTelemetryFrame`].
    #[inline]
    pub fn poll_latest_frame(&mut self) -> &VisualTelemetryFrame {
        let frame = self.telemetry_output.read();
        self.last_frame_seq = frame.header.frame_seq;
        frame
    }

    #[inline]
    pub fn last_frame_seq(&self) -> u64 {
        self.last_frame_seq
    }
}
