//! Zero-Copy Binary Telemetry Streamer (`shm_telemetry_server.rs`).
//!
//! Exposes the 128 KB (`131,072` bytes = `0x20000`) [`VisualTelemetryFrame`] read from the
//! real-time audio thread's [`triple_buffer::Output`] to the UI WebView / WebGPU compositor
//! at 120 Hz (`8.333 ms` frame budget) without locking the audio thread and without JSON serialization.

use bytemuck::Zeroable;
use daw_core_graph::telemetry_shm::{
    AutomationAndSplineTelemetry, Ozone12MasterTelemetry, ProQ4SpectrumFrame,
    Soothe3ReductionFrame, TrackMeterEntry, TransportAndMasterHeader,
    VISUAL_TELEMETRY_FRAME_BYTES, VisualTelemetryFrame,
};
use std::mem::{offset_of, size_of};
use triple_buffer::Output;

/// Target UI refresh rate in Hz (`120 FPS`).
pub const TELEMETRY_STREAM_TARGET_FPS: u32 = 120;

/// Target frame interval in nanoseconds (`8,333,333 ns` = `8.333 ms`).
pub const TELEMETRY_FRAME_INTERVAL_NS: u64 = 1_000_000_000 / (TELEMETRY_STREAM_TARGET_FPS as u64);

/// Exact byte offsets of each region inside the 128 KB (`131,072` B) [`VisualTelemetryFrame`].
pub const OFFSET_HEADER: usize = offset_of!(VisualTelemetryFrame, header); // 0x00000 (64 B)
pub const OFFSET_TRACK_METERS: usize = offset_of!(VisualTelemetryFrame, track_meters); // 0x00040 (12,048 B)
pub const OFFSET_HEADER_ALIGN_PAD: usize = offset_of!(VisualTelemetryFrame, header_align_pad); // 0x02F50 (176 B)
pub const OFFSET_PROQ4_SPECTRUM: usize = offset_of!(VisualTelemetryFrame, proq4_spectrum); // 0x03000 (16,384 B)
pub const OFFSET_SOOTHE3_REDUCTION: usize = offset_of!(VisualTelemetryFrame, soothe3_reduction); // 0x07000 (8,192 B)
pub const OFFSET_OZONE12_TELEMETRY: usize = offset_of!(VisualTelemetryFrame, ozone12_telemetry); // 0x09000 (8,192 B)
pub const OFFSET_SPECTROGRAM_RING: usize = offset_of!(VisualTelemetryFrame, spectrogram_ring); // 0x0B000 (81,920 B)
pub const OFFSET_AUTOMATION_SPLINE: usize = offset_of!(VisualTelemetryFrame, automation_spline); // 0x1F000 (4,096 B)

/// Lock-free Zero-Copy Binary Telemetry Streamer reading `VisualTelemetryFrame` (`128 KB`)
/// from the real-time audio engine's `TripleBuffer` and streaming raw binary slices to WebView / WebGPU
/// at 120 Hz without JSON serialization.
pub struct ShmTelemetryServer {
    telemetry_output: Output<VisualTelemetryFrame>,
    last_frame_seq: u64,
    frames_streamed: u64,
    skipped_duplicate_polls: u64,
}

/// Alias matching the Phase 4 specification name `ZeroCopyBinaryTelemetryStreamer`.
pub type ZeroCopyBinaryTelemetryStreamer = ShmTelemetryServer;

impl ShmTelemetryServer {
    pub fn new(telemetry_output: Output<VisualTelemetryFrame>) -> Self {
        Self {
            telemetry_output,
            last_frame_seq: 0,
            frames_streamed: 0,
            skipped_duplicate_polls: 0,
        }
    }

    /// Polls the `TripleBuffer` for the latest 128 KB (`131,072` bytes) frame and returns
    /// a zero-copy byte array reference `&[u8; 131_072]` suitable for raw binary IPC (`tauri::ipc::Response`)
    /// or `WebView2` `PostSharedBufferToScript`.
    #[inline]
    pub fn poll_latest_bytes(&mut self) -> &[u8; VISUAL_TELEMETRY_FRAME_BYTES] {
        let frame = self.telemetry_output.read();
        let seq = frame.header.frame_seq;
        if seq != self.last_frame_seq {
            self.last_frame_seq = seq;
            self.frames_streamed = self.frames_streamed.wrapping_add(1);
        } else {
            self.skipped_duplicate_polls = self.skipped_duplicate_polls.wrapping_add(1);
        }
        frame.as_bytes()
    }

    /// Polls the `TripleBuffer` only if a new `frame_seq` has been published since the last call.
    /// Returns `Some(&[u8; 131_072])` when a new frame is available, or `None` if unchanged.
    #[inline]
    pub fn poll_if_updated(&mut self) -> Option<&[u8; VISUAL_TELEMETRY_FRAME_BYTES]> {
        let frame = self.telemetry_output.read();
        let seq = frame.header.frame_seq;
        if seq != self.last_frame_seq {
            self.last_frame_seq = seq;
            self.frames_streamed = self.frames_streamed.wrapping_add(1);
            Some(frame.as_bytes())
        } else {
            self.skipped_duplicate_polls = self.skipped_duplicate_polls.wrapping_add(1);
            None
        }
    }

    /// Polls the `TripleBuffer` and returns a structured reference to [`VisualTelemetryFrame`].
    #[inline]
    pub fn poll_latest_frame(&mut self) -> &VisualTelemetryFrame {
        let frame = self.telemetry_output.read();
        let seq = frame.header.frame_seq;
        if seq != self.last_frame_seq {
            self.last_frame_seq = seq;
            self.frames_streamed = self.frames_streamed.wrapping_add(1);
        }
        frame
    }

    /// Copies the latest 128 KB binary telemetry frame directly into a pre-mapped destination
    /// buffer (e.g. a `SharedArrayBuffer` / OS shared memory view) with zero intermediate allocations.
    #[inline]
    pub fn copy_into_shared_slice(&mut self, dst: &mut [u8]) -> Result<u64, String> {
        if dst.len() < VISUAL_TELEMETRY_FRAME_BYTES {
            return Err(format!(
                "Destination buffer too small: {} bytes (expected {} bytes)",
                dst.len(),
                VISUAL_TELEMETRY_FRAME_BYTES
            ));
        }
        let src = self.poll_latest_bytes();
        dst[..VISUAL_TELEMETRY_FRAME_BYTES].copy_from_slice(src);
        Ok(self.last_frame_seq)
    }

    /// Returns the sequence number (`frame_seq`) of the most recently observed telemetry frame.
    #[inline]
    pub fn last_frame_seq(&self) -> u64 {
        self.last_frame_seq
    }

    /// Total number of distinct frames streamed to the UI consumer.
    #[inline]
    pub fn frames_streamed(&self) -> u64 {
        self.frames_streamed
    }

    /// Number of 120 Hz polls where the audio engine had not yet published a newer frame.
    #[inline]
    pub fn skipped_duplicate_polls(&self) -> u64 {
        self.skipped_duplicate_polls
    }

    /// Decodes a raw 128 KB (`131,072` bytes) binary buffer back into a boxed [`VisualTelemetryFrame`]
    /// without JSON parsing (handles arbitrary byte alignment safely).
    pub fn decode_boxed_frame_from_bytes(
        raw_bytes: &[u8],
    ) -> Result<Box<VisualTelemetryFrame>, String> {
        if raw_bytes.len() != VISUAL_TELEMETRY_FRAME_BYTES {
            return Err(format!(
                "Invalid VisualTelemetryFrame byte length: {} (expected {})",
                raw_bytes.len(),
                VISUAL_TELEMETRY_FRAME_BYTES
            ));
        }
        let mut boxed = VisualTelemetryFrame::new_boxed();
        boxed.as_bytes_mut().copy_from_slice(raw_bytes);
        Ok(boxed)
    }

    /// Extracts the 64-byte [`TransportAndMasterHeader`] (`0x00000..0x00040`) from a raw binary frame.
    pub fn decode_header_from_bytes(raw_bytes: &[u8]) -> Result<TransportAndMasterHeader, String> {
        if raw_bytes.len() < size_of::<TransportAndMasterHeader>() {
            return Err("Buffer too small for TransportAndMasterHeader".to_string());
        }
        let mut hdr = TransportAndMasterHeader::zeroed();
        bytemuck::bytes_of_mut(&mut hdr)
            .copy_from_slice(&raw_bytes[OFFSET_HEADER..OFFSET_HEADER + 64]);
        Ok(hdr)
    }

    /// Extracts a single track's [`TrackMeterEntry`] (`24 B`) at `track_index` (`0..502`) from a raw binary frame.
    pub fn decode_track_meter_from_bytes(
        raw_bytes: &[u8],
        track_index: usize,
    ) -> Result<TrackMeterEntry, String> {
        if track_index >= 502 {
            return Err(format!("Track index {} out of range (0..502)", track_index));
        }
        if raw_bytes.len() < OFFSET_HEADER_ALIGN_PAD {
            return Err("Buffer too small for TrackMeterEntry array".to_string());
        }
        let off = OFFSET_TRACK_METERS + track_index * size_of::<TrackMeterEntry>();
        let mut entry = TrackMeterEntry::zeroed();
        bytemuck::bytes_of_mut(&mut entry)
            .copy_from_slice(&raw_bytes[off..off + size_of::<TrackMeterEntry>()]);
        Ok(entry)
    }

    /// Extracts the 16 KB [`ProQ4SpectrumFrame`] (`0x03000..0x07000`) from a raw binary frame.
    pub fn decode_proq4_spectrum_from_bytes(
        raw_bytes: &[u8],
    ) -> Result<ProQ4SpectrumFrame, String> {
        let end = OFFSET_PROQ4_SPECTRUM + size_of::<ProQ4SpectrumFrame>();
        if raw_bytes.len() < end {
            return Err("Buffer too small for ProQ4SpectrumFrame".to_string());
        }
        let mut spec = ProQ4SpectrumFrame::zeroed();
        bytemuck::bytes_of_mut(&mut spec)
            .copy_from_slice(&raw_bytes[OFFSET_PROQ4_SPECTRUM..end]);
        Ok(spec)
    }

    /// Extracts the 8 KB [`Soothe3ReductionFrame`] (`0x07000..0x09000`) from a raw binary frame.
    pub fn decode_soothe3_reduction_from_bytes(
        raw_bytes: &[u8],
    ) -> Result<Soothe3ReductionFrame, String> {
        let end = OFFSET_SOOTHE3_REDUCTION + size_of::<Soothe3ReductionFrame>();
        if raw_bytes.len() < end {
            return Err("Buffer too small for Soothe3ReductionFrame".to_string());
        }
        let mut red = Soothe3ReductionFrame::zeroed();
        bytemuck::bytes_of_mut(&mut red)
            .copy_from_slice(&raw_bytes[OFFSET_SOOTHE3_REDUCTION..end]);
        Ok(red)
    }

    /// Extracts the 8 KB [`Ozone12MasterTelemetry`] (`0x09000..0x0B000`) from a raw binary frame.
    pub fn decode_ozone12_telemetry_from_bytes(
        raw_bytes: &[u8],
    ) -> Result<Ozone12MasterTelemetry, String> {
        let end = OFFSET_OZONE12_TELEMETRY + size_of::<Ozone12MasterTelemetry>();
        if raw_bytes.len() < end {
            return Err("Buffer too small for Ozone12MasterTelemetry".to_string());
        }
        let mut oz = Ozone12MasterTelemetry::zeroed();
        bytemuck::bytes_of_mut(&mut oz)
            .copy_from_slice(&raw_bytes[OFFSET_OZONE12_TELEMETRY..end]);
        Ok(oz)
    }

    /// Extracts the 4 KB [`AutomationAndSplineTelemetry`] (`0x1F000..0x20000`) from a raw binary frame.
    pub fn decode_automation_spline_from_bytes(
        raw_bytes: &[u8],
    ) -> Result<AutomationAndSplineTelemetry, String> {
        let end = OFFSET_AUTOMATION_SPLINE + size_of::<AutomationAndSplineTelemetry>();
        if raw_bytes.len() < end {
            return Err("Buffer too small for AutomationAndSplineTelemetry".to_string());
        }
        let mut auto = AutomationAndSplineTelemetry::zeroed();
        bytemuck::bytes_of_mut(&mut auto)
            .copy_from_slice(&raw_bytes[OFFSET_AUTOMATION_SPLINE..end]);
        Ok(auto)
    }
}
