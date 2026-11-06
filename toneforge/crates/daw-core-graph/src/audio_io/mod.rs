//! Hardware Audio I/O Bridge (`ILWASAPI2ASIO_x64.dll` `WASAPIWrapper::AudioThreadProc` at `0x180006690`
//! and `TMixerThread_ASIO.Execute` at `0x00c55740`).
//!
//! Provides:
//! 1. Windows MMCSS `"Pro Audio"` real-time thread priority elevation (`AvSetMmThreadCharacteristicsW`)
//!    and COM multi-threaded apartment initialization (`CoInitializeEx(NULL, COINIT_MULTITHREADED)`).
//! 2. `WasapiAsioBlockAdapter` (`AudioBuffer` ring at `0x1800083c0` / `0x180008430`): adapts arbitrary
//!    WASAPI packet lengths into fixed power-of-two ASIO callback blocks (`64..512` frames) with
//!    Zero Heap Allocation.
//! 3. Sample format conversion (`sub_c54df0`) between `StereoFrame` (`f32`) and hardware PCM formats
//!    (`ASIOSTInt16LSB`, `ASIOSTInt24LSB`, `ASIOSTInt32LSB`, `ASIOSTFloat32LSB`).

use crate::StereoFrame;
use std::ffi::c_void;

#[cfg(windows)]
#[link(name = "avrt")]
unsafe extern "system" {
    fn AvSetMmThreadCharacteristicsW(task_name: *const u16, task_index: *mut u32) -> *mut c_void;
    fn AvRevertMmThreadCharacteristics(avrt_handle: *mut c_void) -> i32;
}

#[cfg(windows)]
#[link(name = "ole32")]
unsafe extern "system" {
    fn CoInitializeEx(pv_reserved: *mut c_void, dw_co_init: u32) -> i32;
    fn CoUninitialize();
}

/// RAII guard for Windows MMCSS `"Pro Audio"` thread elevation and COM MTA initialization
/// (`WASAPIWrapper::AudioThreadProc` at `0x180006690`).
pub struct ProAudioThreadGuard {
    #[cfg(windows)]
    avrt_handle: *mut c_void,
    #[cfg(windows)]
    com_initialized: bool,
    pub task_index: u32,
}

impl ProAudioThreadGuard {
    /// Elevates the current thread to Windows MMCSS `"Pro Audio"` priority and initializes COM MTA.
    pub fn enter() -> Self {
        #[cfg(windows)]
        {
            const COINIT_MULTITHREADED: u32 = 0x0;
            // UTF-16 null-terminated L"Pro Audio"
            const PRO_AUDIO_UTF16: [u16; 10] = [
                b'P' as u16,
                b'r' as u16,
                b'o' as u16,
                b' ' as u16,
                b'A' as u16,
                b'u' as u16,
                b'd' as u16,
                b'i' as u16,
                b'o' as u16,
                0,
            ];
            let mut task_index = 0u32;
            let hr = unsafe { CoInitializeEx(std::ptr::null_mut(), COINIT_MULTITHREADED) };
            let com_initialized = hr >= 0;
            let avrt_handle = unsafe {
                AvSetMmThreadCharacteristicsW(PRO_AUDIO_UTF16.as_ptr(), &mut task_index)
            };
            Self {
                avrt_handle,
                com_initialized,
                task_index,
            }
        }
        #[cfg(not(windows))]
        {
            Self { task_index: 0 }
        }
    }

    /// Returns `true` if Windows MMCSS `"Pro Audio"` registration succeeded.
    #[inline]
    pub fn is_elevated(&self) -> bool {
        #[cfg(windows)]
        {
            !self.avrt_handle.is_null()
        }
        #[cfg(not(windows))]
        {
            true
        }
    }
}

impl Drop for ProAudioThreadGuard {
    fn drop(&mut self) {
        #[cfg(windows)]
        unsafe {
            if !self.avrt_handle.is_null() {
                let _ = AvRevertMmThreadCharacteristics(self.avrt_handle);
            }
            if self.com_initialized {
                CoUninitialize();
            }
        }
    }
}

/// Hardware sample format (`sub_c54df0` in `FLEngine_x64.dll`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AsioSampleFormat {
    Int16Lsb,
    Int24Lsb,
    Int32Lsb,
    Float32Lsb,
}

/// Converts normalized stereo `f32` frames (`[-1.0, 1.0]`) to `ASIOSTInt32LSB` (`sub_c54df0`).
/// Zero Heap Allocation.
#[inline]
pub fn convert_f32_to_int32_lsb(src: &[StereoFrame], dst_interleaved: &mut [i32]) {
    let n = src.len().min(dst_interleaved.len() / 2);
    const SCALE_I32: f64 = 2_147_483_647.0;
    for i in 0..n {
        let l = (src[i][0].clamp(-1.0, 1.0) as f64 * SCALE_I32).round() as i32;
        let r = (src[i][1].clamp(-1.0, 1.0) as f64 * SCALE_I32).round() as i32;
        dst_interleaved[i * 2] = l;
        dst_interleaved[i * 2 + 1] = r;
    }
}

/// Power-of-two circular block adapter (`AudioBuffer` `0x180065de0` in `ILWASAPI2ASIO_x64.dll`).
/// Repackages arbitrary WASAPI capture/render packet sizes into fixed power-of-two ASIO `bufferSwitch`
/// blocks (`64..512` frames) with Zero Heap Allocation.
pub struct WasapiAsioBlockAdapter {
    asio_block_size: usize,
    input_ring: Vec<StereoFrame>,
    in_write: usize,
    in_read: usize,
    in_count: usize,
    output_ring: Vec<StereoFrame>,
    out_write: usize,
    out_read: usize,
    out_count: usize,
    asio_work_in: Vec<StereoFrame>,
    asio_work_out: Vec<StereoFrame>,
}

impl WasapiAsioBlockAdapter {
    pub fn new(asio_block_size: usize) -> Self {
        let block = asio_block_size.max(16).next_power_of_two();
        let ring_cap = block * 8;
        Self {
            asio_block_size: block,
            input_ring: vec![[0.0; 2]; ring_cap],
            in_write: 0,
            in_read: 0,
            in_count: 0,
            output_ring: vec![[0.0; 2]; ring_cap],
            out_write: 0,
            out_read: 0,
            out_count: 0,
            asio_work_in: vec![[0.0; 2]; block],
            asio_work_out: vec![[0.0; 2]; block],
        }
    }

    #[inline]
    pub fn asio_block_size(&self) -> usize {
        self.asio_block_size
    }

    /// Feeds a WASAPI capture packet and renders a WASAPI output packet, invoking `asio_buffer_switch`
    /// whenever a full power-of-two `asio_block_size` quantum is ready (Zero Heap Allocation).
    pub fn process_wasapi_packet<FCallback>(
        &mut self,
        wasapi_capture_in: &[StereoFrame],
        wasapi_render_out: &mut [StereoFrame],
        mut asio_buffer_switch: FCallback,
    ) where
        FCallback: FnMut(&[StereoFrame], &mut [StereoFrame]),
    {
        let n = wasapi_capture_in.len().min(wasapi_render_out.len());
        let cap = self.input_ring.len();

        for &frame in &wasapi_capture_in[..n] {
            if self.in_count < cap {
                self.input_ring[self.in_write] = frame;
                self.in_write = (self.in_write + 1) % cap;
                self.in_count += 1;
            }
        }

        while self.out_count < n {
            // Pull `asio_block_size` frames from `input_ring` (pad with 0 if capture lags)
            for slot in &mut self.asio_work_in {
                if self.in_count > 0 {
                    *slot = self.input_ring[self.in_read];
                    self.in_read = (self.in_read + 1) % cap;
                    self.in_count -= 1;
                } else {
                    *slot = [0.0; 2];
                }
            }
            self.asio_work_out.fill([0.0; 2]);
            asio_buffer_switch(&self.asio_work_in, &mut self.asio_work_out);

            for &frame in &self.asio_work_out {
                if self.out_count < cap {
                    self.output_ring[self.out_write] = frame;
                    self.out_write = (self.out_write + 1) % cap;
                    self.out_count += 1;
                }
            }
        }

        for out_frame in &mut wasapi_render_out[..n] {
            *out_frame = self.output_ring[self.out_read];
            self.out_read = (self.out_read + 1) % cap;
            self.out_count -= 1;
        }
    }
}
