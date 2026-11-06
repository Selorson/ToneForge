//! Lock-Free 120 FPS Shared-Memory Telemetry (`VisualTelemetryFrame` = `131,072` bytes = `128 KB`)
//! and SPSC Command Channel (`rtrb::RingBuffer<EngineCommand>`).
//!
//! Verified at compile time via `size_of`, `align_of`, and `offset_of!` to match every byte offset
//! specified in `05_master_rust_svelte_blueprint.md` Section 3.2 & Section 5.1.

use bytemuck::{Pod, Zeroable};
use std::mem::{align_of, offset_of, size_of};

/// Exact byte size of `VisualTelemetryFrame` (`131,072` bytes = `0x20000` = `128 KB`).
pub const VISUAL_TELEMETRY_FRAME_BYTES: usize = 131_072;

/// Default capacity of the lock-free `rtrb` SPSC command channel (`4,096` slots).
pub const DEFAULT_COMMAND_RING_CAPACITY: usize = 4096;

/// Offset `0x00000..0x0003F` (`64 B` = `0x0040`): Transport & Master Bus Metering Header.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct TransportAndMasterHeader {
    pub frame_seq: u64,         // 0x0000
    pub sample_pos: i64,        // 0x0008
    pub ppq_tick: i64,          // 0x0010
    pub bpm: f32,               // 0x0018
    pub cpu_load_pct: f32,      // 0x001C
    pub lufs_momentary: f32,    // 0x0020
    pub lufs_short_term: f32,   // 0x0024
    pub lufs_integrated: f32,   // 0x0028
    pub true_peak_db: [f32; 2], // 0x002C..0x0033
    pub master_gr_db: f32,      // 0x0034
    pub active_stem_mode: u32,  // 0x0038
    pub reserved: [u32; 1],     // 0x003C..0x003F (Total = 64 B)
}

impl Default for TransportAndMasterHeader {
    fn default() -> Self {
        Self::zeroed()
    }
}

/// Offset `0x00040..0x02F4F` (`502 * 24 B = 12,048 B` = `0x2F10`): Per-Track Metering at 120 FPS.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct TrackMeterEntry {
    pub peak_l_db: f32,      // +0x00
    pub peak_r_db: f32,      // +0x04
    pub rms_l_db: f32,       // +0x08
    pub rms_r_db: f32,       // +0x0C
    pub gr_db: f32,          // +0x10
    pub activity_flags: u32, // +0x14 (Total = 24 B)
}

impl Default for TrackMeterEntry {
    fn default() -> Self {
        Self {
            peak_l_db: -144.0,
            peak_r_db: -144.0,
            rms_l_db: -144.0,
            rms_r_db: -144.0,
            gr_db: 0.0,
            activity_flags: 0,
        }
    }
}

/// Offset `0x03000..0x06FFF` (`16,384 B` = `0x4000`): FabFilter Pro-Q 4 Spectrum & Collision Frame.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct ProQ4SpectrumFrame {
    pub pre_db: [f32; 1024],         // 4,096 B
    pub post_db: [f32; 1024],        // 4,096 B
    pub sc_db: [f32; 1024],          // 4,096 B
    pub collision_mask: [f32; 1024], // 4,096 B (Total = 16,384 B)
}

impl Default for ProQ4SpectrumFrame {
    fn default() -> Self {
        Self::zeroed()
    }
}

/// Offset `0x07000..0x08FFF` (`8,192 B` = `0x2000`): oeksound Soothe3 Reduction Curve & Input Spec.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Soothe3ReductionFrame {
    pub reduction_ys: [[f32; 4]; 256], // 4,096 B (1,024 bins packed as vec4<f32>[256])
    pub input_spec_db: [f32; 1024],    // 4,096 B (Total = 8,192 B)
}

impl Default for Soothe3ReductionFrame {
    fn default() -> Self {
        Self::zeroed()
    }
}

/// Offset `0x09000..0x0AFFF` (`8,192 B` = `0x2000`): iZotope Ozone 12 Master Telemetry.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Ozone12MasterTelemetry {
    pub irc_gr_trace: [f32; 512],         // 2,048 B
    pub polar_sample_xy: [[f32; 2]; 512], // 4,096 B
    pub tonal_balance_bars: [f32; 256],   // 1,024 B
    pub stem_rms_db: [f32; 4],            // 16 B ([Vocals, Drums, Bass, Other])
    pub imager_corr_bands: [f32; 4],      // 16 B
    pub exciter_drive_gr: [f32; 248],     // 992 B (Total = 8,192 B)
}

impl Default for Ozone12MasterTelemetry {
    fn default() -> Self {
        Self::zeroed()
    }
}

/// Offset `0x1F000..0x1FFFF` (`4,096 B` = `0x1000`): Automation, Gross Beat & Patcher Telemetry.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct AutomationAndSplineTelemetry {
    pub controller_values_norm: [f32; 512], // 2,048 B
    pub gross_beat_playhead_pos: [f32; 2],  // 8 B ([TimeSlotPos, VolSlotPos])
    pub patcher_node_cpu_pct: [f32; 510],   // 2,040 B (Total = 4,096 B)
}

impl Default for AutomationAndSplineTelemetry {
    fn default() -> Self {
        Self::zeroed()
    }
}

/// Complete 120 FPS Shared-Memory Telemetry Frame (`131,072` bytes = `128 KB`, 64-byte cache-aligned).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct VisualTelemetryFrame {
    pub header: TransportAndMasterHeader,                // 0x00000..0x0003F (64 B)
    pub track_meters: [TrackMeterEntry; 502],            // 0x00040..0x02F4F (12,048 B)
    pub header_align_pad: [u8; 176],                     // 0x02F50..0x02FFF (176 B)
    pub proq4_spectrum: ProQ4SpectrumFrame,              // 0x03000..0x06FFF (16,384 B)
    pub soothe3_reduction: Soothe3ReductionFrame,        // 0x07000..0x08FFF (8,192 B)
    pub ozone12_telemetry: Ozone12MasterTelemetry,       // 0x09000..0x0AFFF (8,192 B)
    pub spectrogram_ring: [[f32; 1024]; 20],             // 0x0B000..0x1EFFF (81,920 B)
    pub automation_spline: AutomationAndSplineTelemetry, // 0x1F000..0x1FFFF (4,096 B)
}

impl Default for VisualTelemetryFrame {
    fn default() -> Self {
        Self::zeroed()
    }
}

impl VisualTelemetryFrame {
    /// Allocates a zeroed `Box<VisualTelemetryFrame>` directly on the heap without placing 128 KB on the stack.
    #[inline]
    pub fn new_boxed() -> Box<Self> {
        bytemuck::allocation::zeroed_box()
    }

    /// Views the entire 131,072-byte structure as a raw byte slice for zero-copy IPC / WebGPU upload.
    #[inline]
    pub fn as_bytes(&self) -> &[u8; VISUAL_TELEMETRY_FRAME_BYTES] {
        bytemuck::cast_ref(self)
    }

    /// Mutable byte view of the 131,072-byte structure.
    #[inline]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; VISUAL_TELEMETRY_FRAME_BYTES] {
        bytemuck::cast_mut(self)
    }
}

// ============================================================================
// COMPILE-TIME SIZE, ALIGNMENT, AND EXACT HEX OFFSET ASSERTIONS
// ============================================================================
const _: () = assert!(size_of::<TransportAndMasterHeader>() == 64);
const _: () = assert!(size_of::<TrackMeterEntry>() == 24);
const _: () = assert!(size_of::<[TrackMeterEntry; 502]>() == 12_048);
const _: () = assert!(size_of::<ProQ4SpectrumFrame>() == 16_384);
const _: () = assert!(size_of::<Soothe3ReductionFrame>() == 8_192);
const _: () = assert!(size_of::<Ozone12MasterTelemetry>() == 8_192);
const _: () = assert!(size_of::<[[f32; 1024]; 20]>() == 81_920);
const _: () = assert!(size_of::<AutomationAndSplineTelemetry>() == 4_096);
const _: () = assert!(size_of::<VisualTelemetryFrame>() == VISUAL_TELEMETRY_FRAME_BYTES);
const _: () = assert!(align_of::<VisualTelemetryFrame>() == 64);

const _: () = assert!(offset_of!(VisualTelemetryFrame, header) == 0x00000);
const _: () = assert!(offset_of!(VisualTelemetryFrame, track_meters) == 0x00040);
const _: () = assert!(offset_of!(VisualTelemetryFrame, header_align_pad) == 0x02F50);
const _: () = assert!(offset_of!(VisualTelemetryFrame, proq4_spectrum) == 0x03000);
const _: () = assert!(offset_of!(VisualTelemetryFrame, soothe3_reduction) == 0x07000);
const _: () = assert!(offset_of!(VisualTelemetryFrame, ozone12_telemetry) == 0x09000);
const _: () = assert!(offset_of!(VisualTelemetryFrame, spectrogram_ring) == 0x0B000);
const _: () = assert!(offset_of!(VisualTelemetryFrame, automation_spline) == 0x1F000);

/// Real-time lock-free commands sent from the UI / Host thread to the Audio Engine via `rtrb::RingBuffer`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EngineCommand {
    /// Start or stop transport playback (`is_playing`).
    SetTransportPlaying(bool),
    /// Change project tempo in BPM.
    SetBpm(f64),
    /// Seek transport to sample position and PPQ tick.
    SeekPosition { sample_pos: i64, ppq_tick: i64 },
    /// Adjust post-fader linear gain on a Mixer Track (`0..502`).
    SetTrackGain { track_index: u16, gain_linear: f32 },
    /// Enable/disable polarity inversion on a Mixer Track (`0..502`).
    SetTrackPolarityInverted { track_index: u16, inverted: bool },
    /// Adjust parallel Dry/Wet mix (`0.0..=1.0`) on a Mixer Track FX Slot (`0..10`).
    SetSlotWetMix {
        track_index: u16,
        slot_index: u8,
        wet_mix: f32,
    },
    /// Enable or bypass a Mixer Track FX Slot (`0..10`).
    SetSlotEnabled {
        track_index: u16,
        slot_index: u8,
        enabled: bool,
    },
    /// Immediate or sample-delayed normalized parameter update for a plugin slot.
    SetPluginParamNormalized {
        track_index: u16,
        slot_index: u8,
        param_id: u32,
        norm_val: f32,
        target_sample_pos: Option<i64>,
    },
    /// Update a 30-bit fixed-point value (`0..=0x40000000`) on the Internal Controller Bus.
    SetInternalControllerFixed30 {
        controller_id: u16,
        value_fixed30: i32,
    },
    /// Change PPQ resolution dynamically (`96..=960`).
    SetPpq(u32),
    /// Trigger a sample-accurate MIDI / Piano Roll / Slide note event on a ChannelGenerator.
    TriggerVoiceEvent(crate::sequencer_pianoroll::SequencerVoiceEvent),
    /// Recomputes 5-Point APDC delays across the existing topological schedule in-place
    /// with Zero Heap Allocation (e.g. after a plugin dynamically changes its reported latency).
    RefreshApdc,
}

/// Allocates a wait-free `TripleBuffer<VisualTelemetryFrame>` directly on the heap
/// without creating temporary 128 KB frames on the calling thread's stack.
pub fn create_telemetry_triple_buffer() -> (
    triple_buffer::Input<VisualTelemetryFrame>,
    triple_buffer::Output<VisualTelemetryFrame>,
) {
    triple_buffer::TripleBuffer::from_boxes([
        VisualTelemetryFrame::new_boxed(),
        VisualTelemetryFrame::new_boxed(),
        VisualTelemetryFrame::new_boxed(),
    ])
}

/// Allocates a wait-free SPSC `rtrb::RingBuffer<EngineCommand>` with `capacity` slots.
pub fn create_command_ring_buffer(
    capacity: usize,
) -> (rtrb::Producer<EngineCommand>, rtrb::Consumer<EngineCommand>) {
    rtrb::RingBuffer::new(capacity.max(16))
}
