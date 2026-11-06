//! Windows WASAPI & Steinberg ASIO Real-Time Audio I/O Bridge (`wasapi_asio_bridge.rs`).
//!
//! Implements clean-room equivalents of `FL Studio ASIO.dll` (`sub_39b8ff0` / `0x039B9B80`):
//! - Zero-allocation conversions between Interleaved WASAPI (`&[f32]` / `&mut [f32]`),
//!   Planar ASIO (`&[f32], &[f32]` / `&mut [f32], &mut [f32]`), and `StereoFrame` (`[f32; 2]`).
//! - Hardware sample format conversions (`ASIOSTInt16LSB`, `ASIOSTInt24LSB`, `ASIOSTInt32LSB`,
//!   `ASIOSTFloat32LSB`, `ASIOSTFloat64LSB`), including the exact `0x4F000000` (`2_147_483_520.0_f32`)
//!   headroom-clamped `f32 <-> i32` converter from `0x039B9B80`.
//! - `WasapiAsioBlockAdapter`: Zero-allocation fixed-capacity ring adapter bridging variable-sized
//!   WASAPI shared/exclusive packets into deterministic ASIO power-of-two blocks.

use crate::StereoFrame;

/// Exact IEEE-754 float32 constant `2,147,483,520.0` (`0x4EFF_FFFF`, largest `f32` `< 2^31 = 0x4F00_0000`)
/// used at `0x039B9B80` in `FL Studio ASIO.dll` to prevent integer wraparound when converting `+1.0f32` to `i32`.
pub const ASIO_INT32_SCALE_F32: f32 = 2_147_483_520.0_f32; // f32::from_bits(0x4EFF_FFFF)

const _: () = assert!(ASIO_INT32_SCALE_F32.to_bits() == 0x4EFF_FFFF);

/// ASIO sample format codes matching `ASIOSampleType` in `FL Studio ASIO.dll`.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AsioSampleFormat {
    Int16Lsb = 16,
    Int24Lsb = 17,
    Int32Lsb = 18,
    Float32Lsb = 19,
    Float64Lsb = 20,
}

///// Converts normalized `f32` (`[-1.0, +1.0]`) to `ASIOSTInt32LSB` (`i32`) using the exact
/// `2,147,483,520.0` (`0x4F000000`) multiplier from `0x039B9B80` to prevent `i32` overflow at `+1.0`.
#[inline(always)]
pub fn convert_f32_to_int32_lsb(sample: f32) -> i32 {
    if sample.is_nan() {
        return 0;
    }
    let clamped = sample.clamp(-1.0, 1.0);
    (clamped * ASIO_INT32_SCALE_F32) as i32
}

/// Converts `ASIOSTInt32LSB` (`i32`) back to normalized `f32` (`[-1.0, +1.0]`).
#[inline(always)]
pub fn convert_int32_lsb_to_f32(sample: i32) -> f32 {
    ((sample as f64) / (2_147_483_648.0_f64)).clamp(-1.0, 1.0) as f32
}

/// Converts normalized `f32` (`[-1.0, +1.0]`) to `ASIOSTInt16LSB` (`i16`).
#[inline(always)]
pub fn convert_f32_to_int16_lsb(sample: f32) -> i16 {
    if sample.is_nan() {
        return 0;
    }
    let clamped = sample.clamp(-1.0, 1.0);
    (clamped * 32_767.0_f32).round() as i16
}

/// Converts normalized `f32` (`[-1.0, +1.0]`) to 3-byte packed little-endian `ASIOSTInt24LSB`.
#[inline(always)]
pub fn convert_f32_to_int24_lsb(sample: f32) -> [u8; 3] {
    if sample.is_nan() {
        return [0, 0, 0];
    }
    let clamped = sample.clamp(-1.0, 1.0);
    let val = (clamped * 8_388_607.0_f32).round() as i32;
    let le = val.to_le_bytes();
    [le[0], le[1], le[2]]
}

/// De-interleaves a WASAPI stereo buffer (`[L0, R0, L1, R1, ...]`) into `StereoFrame` slices.
/// Zero heap allocation. Returns the number of stereo frames converted.
#[inline]
pub fn wasapi_interleaved_to_stereo_frames(
    interleaved_in: &[f32],
    frames_out: &mut [StereoFrame],
) -> usize {
    let n = (interleaved_in.len() / 2).min(frames_out.len());
    for i in 0..n {
        frames_out[i] = [interleaved_in[i * 2], interleaved_in[i * 2 + 1]];
    }
    n
}

/// Interleaves `StereoFrame` slices into a WASAPI stereo buffer (`[L0, R0, L1, R1, ...]`).
/// Zero heap allocation. Returns the number of stereo frames written.
#[inline]
pub fn stereo_frames_to_wasapi_interleaved(
    frames_in: &[StereoFrame],
    interleaved_out: &mut [f32],
) -> usize {
    let n = frames_in.len().min(interleaved_out.len() / 2);
    for i in 0..n {
        interleaved_out[i * 2] = frames_in[i][0];
        interleaved_out[i * 2 + 1] = frames_in[i][1];
    }
    n
}

/// Converts planar ASIO input buffers (`left_in`, `right_in`) into `StereoFrame` slices.
/// Zero heap allocation. Returns the number of stereo frames converted.
#[inline]
pub fn asio_planar_to_stereo_frames(
    left_in: &[f32],
    right_in: &[f32],
    frames_out: &mut [StereoFrame],
) -> usize {
    let n = left_in.len().min(right_in.len()).min(frames_out.len());
    for i in 0..n {
        frames_out[i] = [left_in[i], right_in[i]];
    }
    n
}

/// Splits `StereoFrame` slices into planar ASIO output buffers (`left_out`, `right_out`).
/// Zero heap allocation. Returns the number of stereo frames written.
#[inline]
pub fn stereo_frames_to_asio_planar(
    frames_in: &[StereoFrame],
    left_out: &mut [f32],
    right_out: &mut [f32],
) -> usize {
    let n = frames_in.len().min(left_out.len()).min(right_out.len());
    for i in 0..n {
        left_out[i] = frames_in[i][0];
        right_out[i] = frames_in[i][1];
    }
    n
}

/// Converts planar `f32` ASIO output channels into interleaved/planar `ASIOSTInt32LSB` (`i32`)
/// using `convert_f32_to_int32_lsb` (`0x4F000000`). Zero heap allocation.
#[inline]
pub fn asio_planar_f32_to_int32_lsb(
    left_in: &[f32],
    right_in: &[f32],
    left_i32_out: &mut [i32],
    right_i32_out: &mut [i32],
) -> usize {
    let n = left_in
        .len()
        .min(right_in.len())
        .min(left_i32_out.len())
        .min(right_i32_out.len());
    for i in 0..n {
        left_i32_out[i] = convert_f32_to_int32_lsb(left_in[i]);
        right_i32_out[i] = convert_f32_to_int32_lsb(right_in[i]);
    }
    n
}

/// Zero-allocation block adapter converting arbitrary WASAPI packet sizes (e.g., 480 samples @ 10 ms)
/// into fixed-size ASIO blocks (e.g., 64, 128, 256, or 512 frames).
#[derive(Clone, Debug)]
pub struct WasapiAsioBlockAdapter {
    ring: Vec<StereoFrame>,
    capacity: usize,
    read_pos: usize,
    write_pos: usize,
    available: usize,
    asio_block_size: usize,
}

impl WasapiAsioBlockAdapter {
    pub fn new(asio_block_size: usize, max_ring_frames: usize) -> Self {
        let cap = max_ring_frames.max(asio_block_size * 4).max(256);
        Self {
            ring: vec![[0.0, 0.0]; cap],
            capacity: cap,
            read_pos: 0,
            write_pos: 0,
            available: 0,
            asio_block_size: asio_block_size.max(16),
        }
    }

    #[inline]
    pub fn asio_block_size(&self) -> usize {
        self.asio_block_size
    }

    #[inline]
    pub fn available_frames(&self) -> usize {
        self.available
    }

    pub fn clear(&mut self) {
        self.read_pos = 0;
        self.write_pos = 0;
        self.available = 0;
    }

    /// Pushes interleaved WASAPI stereo samples (`[L0, R0, L1, R1, ...]`) into the adapter ring.
    pub fn push_wasapi_interleaved(&mut self, interleaved: &[f32]) -> usize {
        let frames = interleaved.len() / 2;
        let can_write = frames.min(self.capacity - self.available);
        for i in 0..can_write {
            self.ring[self.write_pos] = [interleaved[i * 2], interleaved[i * 2 + 1]];
            self.write_pos += 1;
            if self.write_pos == self.capacity {
                self.write_pos = 0;
            }
        }
        self.available += can_write;
        can_write
    }

    /// Pushes `StereoFrame` samples into the adapter ring.
    pub fn push_wasapi_frames(&mut self, frames: &[StereoFrame]) -> usize {
        let can_write = frames.len().min(self.capacity - self.available);
        for &f in &frames[..can_write] {
            self.ring[self.write_pos] = f;
            self.write_pos += 1;
            if self.write_pos == self.capacity {
                self.write_pos = 0;
            }
        }
        self.available += can_write;
        can_write
    }

    /// Pops one full `asio_block_size` block of `StereoFrame`s if enough frames are buffered.
    pub fn pop_asio_block(&mut self, out_block: &mut [StereoFrame]) -> bool {
        let needed = self.asio_block_size;
        if out_block.len() < needed || self.available < needed {
            return false;
        }
        for slot in &mut out_block[..needed] {
            *slot = self.ring[self.read_pos];
            self.read_pos += 1;
            if self.read_pos == self.capacity {
                self.read_pos = 0;
            }
        }
        self.available -= needed;
        true
    }

    /// Pops one full `asio_block_size` block directly into planar ASIO buffers (`left_out`, `right_out`).
    pub fn pop_asio_planar_block(&mut self, left_out: &mut [f32], right_out: &mut [f32]) -> bool {
        let needed = self.asio_block_size;
        if left_out.len() < needed || right_out.len() < needed || self.available < needed {
            return false;
        }
        for i in 0..needed {
            let f = self.ring[self.read_pos];
            left_out[i] = f[0];
            right_out[i] = f[1];
            self.read_pos += 1;
            if self.read_pos == self.capacity {
                self.read_pos = 0;
            }
        }
        self.available -= needed;
        true
    }
}
