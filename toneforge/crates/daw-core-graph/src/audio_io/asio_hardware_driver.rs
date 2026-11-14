//! `asio_hardware_driver`: Native Windows x86_64 **Steinberg ASIO 2.3** Hardware Driver Host
//! (`HKLM\SOFTWARE\ASIO` Registry Scanner + Full-Duplex `IASIO` COM Planar Audio Bridge).
//!
//! - Discovers installed ASIO drivers (e.g., `Realtek ASIO`, `Focusrite USB ASIO`, `FL Studio ASIO`)
//!   directly from `HKEY_LOCAL_MACHINE\SOFTWARE\ASIO`.
//! - Hosts COM `IASIO` on a dedicated STA + MMCSS `"Pro Audio"` thread with a valid Win32 `sysHandle`
//!   `HWND`, negotiating planar input + output double buffers (`Int32LSB`, `Int24LSB`, `Int16LSB`,
//!   `Float32LSB`, `Float64LSB`).
//! - Feeds hardware Mic/Line-In directly into `MixerEngine`'s `hw_input_buf` while writing Master
//!   output in the same `bufferSwitch` interrupt (`0` RT heap allocations, `0` RT locks).
//! - Supports atomic `set_volume_and_mute(0.0, true)` for silent automated hardware verification.

use crate::StereoFrame;
use crate::audio_io::{
    ProAudioThreadGuard, convert_f32_to_int16_lsb, convert_f32_to_int24_lsb,
    convert_f32_to_int32_lsb, convert_int32_lsb_to_f32,
};
use crate::mixer_dag::MixerEngine;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::ffi::{CStr, c_char, c_void};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use windows_sys::Win32::Foundation::ERROR_SUCCESS;
use windows_sys::Win32::System::Com::{
    CLSCTX_INPROC_SERVER, COINIT_APARTMENTTHREADED, CoCreateInstance, CoInitializeEx,
    CoUninitialize,
};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::System::Registry::{
    HKEY, HKEY_LOCAL_MACHINE, KEY_READ, KEY_WOW64_64KEY, REG_SZ, RegCloseKey, RegEnumKeyExW,
    RegOpenKeyExW, RegQueryValueExW,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW, MSG, PM_REMOVE, PeekMessageW,
    RegisterClassExW, TranslateMessage, WNDCLASSEXW, WS_POPUP,
};
use windows_sys::core::GUID;

// Steinberg ASIO Constants (`asio.h`)
pub const ASE_OK: i32 = 0;
pub const ASE_SUCCESS: i32 = 0x3F48_47A0;
pub const ASIO_TRUE: i32 = 1;
pub const ASIO_FALSE: i32 = 0;

pub const ASIOST_INT16_LSB: i32 = 16;
pub const ASIOST_INT24_LSB: i32 = 17;
pub const ASIOST_INT32_LSB: i32 = 18;
pub const ASIOST_FLOAT32_LSB: i32 = 19;
pub const ASIOST_FLOAT64_LSB: i32 = 20;

/// Metadata for an installed Windows ASIO driver discovered in `HKLM\SOFTWARE\ASIO`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AsioDriverInfo {
    pub key_name: String,
    pub description: String,
    pub clsid: String,
}

/// Live status snapshot of an active or probed Steinberg ASIO stream.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AsioStreamStatusSnapshot {
    pub driver_name: String,
    pub clsid: String,
    pub stream_running: bool,
    pub sample_rate_hz: u32,
    pub buffer_frames: u32,
    pub input_channels: u16,
    pub output_channels: u16,
    pub input_latency_samples: u32,
    pub output_latency_samples: u32,
    pub callback_blocks: u64,
    pub speaker_volume_pct: f32,
    pub speaker_muted: bool,
    pub status_message: String,
    /// Measured engine render time per callback (EMA, microseconds).
    #[serde(default)]
    pub avg_block_render_us: u32,
    /// Driver-reported resync / overload events.
    #[serde(default)]
    pub xrun_count: u64,
}

impl Default for AsioStreamStatusSnapshot {
    fn default() -> Self {
        Self {
            driver_name: String::new(),
            clsid: String::new(),
            stream_running: false,
            sample_rate_hz: 48_000,
            buffer_frames: 256,
            input_channels: 0,
            output_channels: 2,
            input_latency_samples: 0,
            output_latency_samples: 0,
            callback_blocks: 0,
            speaker_volume_pct: 85.0,
            speaker_muted: false,
            status_message: "Idle".to_string(),
            avg_block_render_us: 0,
            xrun_count: 0,
        }
    }
}

/// Parses `{XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}` into a Win32 `GUID`.
pub fn parse_win32_guid(text: &str) -> Option<GUID> {
    let trimmed = text.trim().trim_matches('{').trim_matches('}');
    let parts: Vec<&str> = trimmed.split('-').collect();
    if parts.len() != 5 {
        return None;
    }
    let data1 = u32::from_str_radix(parts[0], 16).ok()?;
    let data2 = u16::from_str_radix(parts[1], 16).ok()?;
    let data3 = u16::from_str_radix(parts[2], 16).ok()?;
    if parts[3].len() != 4 || parts[4].len() != 12 {
        return None;
    }
    let mut data4 = [0u8; 8];
    data4[0] = u8::from_str_radix(&parts[3][0..2], 16).ok()?;
    data4[1] = u8::from_str_radix(&parts[3][2..4], 16).ok()?;
    for i in 0..6 {
        data4[2 + i] = u8::from_str_radix(&parts[4][i * 2..i * 2 + 2], 16).ok()?;
    }
    Some(GUID {
        data1,
        data2,
        data3,
        data4,
    })
}

fn wide_z(s: &str) -> Vec<u16> {
    let mut v: Vec<u16> = s.encode_utf16().collect();
    v.push(0);
    v
}

unsafe fn query_reg_sz(hkey: HKEY, value_name: &str) -> Option<String> {
    let name_w = wide_z(value_name);
    let mut reg_type = 0u32;
    let mut byte_len = 512u32;
    let mut buf = vec![0u16; 256];
    let rc = unsafe {
        RegQueryValueExW(
            hkey,
            name_w.as_ptr(),
            std::ptr::null(),
            &mut reg_type,
            buf.as_mut_ptr() as *mut u8,
            &mut byte_len,
        )
    };
    if rc != ERROR_SUCCESS || reg_type != REG_SZ {
        return None;
    }
    let u16_len = (byte_len as usize / 2).min(buf.len());
    let nul = buf[..u16_len]
        .iter()
        .position(|&c| c == 0)
        .unwrap_or(u16_len);
    let s = String::from_utf16_lossy(&buf[..nul]).trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

/// Scans `HKEY_LOCAL_MACHINE\SOFTWARE\ASIO` for installed x64 ASIO drivers (`Realtek ASIO`, etc.).
pub fn enumerate_asio_drivers() -> Vec<AsioDriverInfo> {
    let mut drivers = Vec::new();
    let subkey_path = wide_z("SOFTWARE\\ASIO");
    let mut asio_root: HKEY = std::ptr::null_mut();
    let rc = unsafe {
        RegOpenKeyExW(
            HKEY_LOCAL_MACHINE,
            subkey_path.as_ptr(),
            0,
            KEY_READ | KEY_WOW64_64KEY,
            &mut asio_root,
        )
    };
    if rc != ERROR_SUCCESS || asio_root.is_null() {
        return drivers;
    }

    let mut idx = 0u32;
    loop {
        let mut key_buf = [0u16; 256];
        let mut key_len = key_buf.len() as u32;
        let enum_rc = unsafe {
            RegEnumKeyExW(
                asio_root,
                idx,
                key_buf.as_mut_ptr(),
                &mut key_len,
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        if enum_rc != ERROR_SUCCESS {
            break;
        }
        let key_name = String::from_utf16_lossy(&key_buf[..key_len as usize]);
        let mut child_key: HKEY = std::ptr::null_mut();
        let child_w = wide_z(&key_name);
        let open_rc = unsafe {
            RegOpenKeyExW(
                asio_root,
                child_w.as_ptr(),
                0,
                KEY_READ | KEY_WOW64_64KEY,
                &mut child_key,
            )
        };
        if open_rc == ERROR_SUCCESS && !child_key.is_null() {
            if let Some(clsid) = unsafe { query_reg_sz(child_key, "CLSID") } {
                if parse_win32_guid(&clsid).is_some() {
                    let description = unsafe { query_reg_sz(child_key, "Description") }
                        .unwrap_or_else(|| key_name.clone());
                    drivers.push(AsioDriverInfo {
                        key_name: key_name.clone(),
                        description,
                        clsid,
                    });
                }
            }
            unsafe {
                let _ = RegCloseKey(child_key);
            }
        }
        idx += 1;
    }

    unsafe {
        let _ = RegCloseKey(asio_root);
    }
    drivers
}

// ============================================================================
// STEINBERG `IASIO` COM VTABLE & STRUCTS (`asio.h` x64 Windows ABI)
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy)]
struct ASIOChannelInfo {
    channel: i32,
    is_input: i32,
    is_active: i32,
    channel_group: i32,
    sample_type: i32,
    name: [c_char; 32],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ASIOBufferInfo {
    is_input: i32,
    channel_num: i32,
    buffers: [*mut c_void; 2],
}

#[repr(C)]
struct ASIOCallbacks {
    buffer_switch: unsafe extern "C" fn(double_buffer_index: i32, direct_process: i32),
    sample_rate_did_change: unsafe extern "C" fn(s_rate: f64),
    asio_message:
        unsafe extern "C" fn(selector: i32, value: i32, message: *mut c_void, opt: *mut f64) -> i32,
    buffer_switch_time_info:
        unsafe extern "C" fn(params: *mut c_void, double_buffer_index: i32, direct_process: i32) -> *mut c_void,
}

#[repr(C)]
struct IASIOVtbl {
    query_interface:
        unsafe extern "system" fn(this: *mut c_void, riid: *const GUID, ppv: *mut *mut c_void) -> i32,
    add_ref: unsafe extern "system" fn(this: *mut c_void) -> u32,
    release: unsafe extern "system" fn(this: *mut c_void) -> u32,
    init: unsafe extern "system" fn(this: *mut c_void, sys_handle: *mut c_void) -> i32,
    get_driver_name: unsafe extern "system" fn(this: *mut c_void, name: *mut c_char),
    get_driver_version: unsafe extern "system" fn(this: *mut c_void) -> i32,
    get_error_message: unsafe extern "system" fn(this: *mut c_void, string: *mut c_char),
    start: unsafe extern "system" fn(this: *mut c_void) -> i32,
    stop: unsafe extern "system" fn(this: *mut c_void) -> i32,
    get_channels:
        unsafe extern "system" fn(this: *mut c_void, num_in: *mut i32, num_out: *mut i32) -> i32,
    get_latencies:
        unsafe extern "system" fn(this: *mut c_void, lat_in: *mut i32, lat_out: *mut i32) -> i32,
    get_buffer_size: unsafe extern "system" fn(
        this: *mut c_void,
        min_sz: *mut i32,
        max_sz: *mut i32,
        pref_sz: *mut i32,
        gran: *mut i32,
    ) -> i32,
    can_sample_rate: unsafe extern "system" fn(this: *mut c_void, s_rate: f64) -> i32,
    get_sample_rate: unsafe extern "system" fn(this: *mut c_void, s_rate: *mut f64) -> i32,
    set_sample_rate: unsafe extern "system" fn(this: *mut c_void, s_rate: f64) -> i32,
    get_clock_sources:
        unsafe extern "system" fn(this: *mut c_void, clocks: *mut c_void, num_sources: *mut i32) -> i32,
    set_clock_source: unsafe extern "system" fn(this: *mut c_void, reference: i32) -> i32,
    get_sample_position:
        unsafe extern "system" fn(this: *mut c_void, s_pos: *mut i64, t_stamp: *mut i64) -> i32,
    get_channel_info:
        unsafe extern "system" fn(this: *mut c_void, info: *mut ASIOChannelInfo) -> i32,
    create_buffers: unsafe extern "system" fn(
        this: *mut c_void,
        buffer_infos: *mut ASIOBufferInfo,
        num_channels: i32,
        buffer_size: i32,
        callbacks: *const ASIOCallbacks,
    ) -> i32,
    dispose_buffers: unsafe extern "system" fn(this: *mut c_void) -> i32,
    control_panel: unsafe extern "system" fn(this: *mut c_void) -> i32,
    future: unsafe extern "system" fn(this: *mut c_void, selector: i32, opt: *mut c_void) -> i32,
    output_ready: unsafe extern "system" fn(this: *mut c_void) -> i32,
}

// ============================================================================
// PLANAR ASIO SAMPLE CONVERSION HELPERS (ZERO HEAP ALLOCATION)
// ============================================================================

/// Reads `frames` samples from a single planar ASIO hardware input buffer into `dst_ch` (`0` = L, `1` = R).
#[inline]
pub unsafe fn unpack_asio_planar_channel(
    src_ptr: *const u8,
    sample_type: i32,
    frames: usize,
    dst: &mut [StereoFrame],
    dst_ch: usize,
) {
    if src_ptr.is_null() || frames == 0 {
        return;
    }
    let n = frames.min(dst.len());
    match sample_type {
        ASIOST_INT32_LSB => {
            let s = unsafe { std::slice::from_raw_parts(src_ptr as *const i32, n) };
            for i in 0..n {
                dst[i][dst_ch] = convert_int32_lsb_to_f32(s[i]);
            }
        }
        ASIOST_FLOAT32_LSB => {
            let s = unsafe { std::slice::from_raw_parts(src_ptr as *const f32, n) };
            for i in 0..n {
                let v = s[i];
                dst[i][dst_ch] = if v.is_finite() { v.clamp(-1.0, 1.0) } else { 0.0 };
            }
        }
        ASIOST_INT24_LSB => {
            let s = unsafe { std::slice::from_raw_parts(src_ptr, n * 3) };
            for i in 0..n {
                let b0 = s[i * 3] as i32;
                let b1 = (s[i * 3 + 1] as i32) << 8;
                let b2 = ((s[i * 3 + 2] as i8) as i32) << 16;
                let v24 = b0 | b1 | b2;
                dst[i][dst_ch] = ((v24 as f64) / 8_388_608.0).clamp(-1.0, 1.0) as f32;
            }
        }
        ASIOST_INT16_LSB => {
            let s = unsafe { std::slice::from_raw_parts(src_ptr as *const i16, n) };
            for i in 0..n {
                dst[i][dst_ch] = ((s[i] as f32) / 32_768.0).clamp(-1.0, 1.0);
            }
        }
        ASIOST_FLOAT64_LSB => {
            let s = unsafe { std::slice::from_raw_parts(src_ptr as *const f64, n) };
            for i in 0..n {
                let v = s[i];
                dst[i][dst_ch] = if v.is_finite() {
                    v.clamp(-1.0, 1.0) as f32
                } else {
                    0.0
                };
            }
        }
        _ => {
            for i in 0..n {
                dst[i][dst_ch] = 0.0;
            }
        }
    }
}

/// Writes `frames` samples from `src` (`src_ch`: `0` = L, `1` = R) scaled by `gain` into a planar ASIO output buffer.
#[inline]
pub unsafe fn pack_asio_planar_channel(
    src: &[StereoFrame],
    src_ch: usize,
    gain: f32,
    dst_ptr: *mut u8,
    sample_type: i32,
    frames: usize,
) {
    if dst_ptr.is_null() || frames == 0 {
        return;
    }
    let n = frames.min(src.len());
    match sample_type {
        ASIOST_INT32_LSB => {
            let d = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut i32, n) };
            for i in 0..n {
                d[i] = convert_f32_to_int32_lsb(src[i][src_ch] * gain);
            }
        }
        ASIOST_FLOAT32_LSB => {
            let d = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut f32, n) };
            for i in 0..n {
                let v = src[i][src_ch] * gain;
                d[i] = if v.is_finite() { v.clamp(-1.0, 1.0) } else { 0.0 };
            }
        }
        ASIOST_INT24_LSB => {
            let d = unsafe { std::slice::from_raw_parts_mut(dst_ptr, n * 3) };
            for i in 0..n {
                let b = convert_f32_to_int24_lsb(src[i][src_ch] * gain);
                d[i * 3] = b[0];
                d[i * 3 + 1] = b[1];
                d[i * 3 + 2] = b[2];
            }
        }
        ASIOST_INT16_LSB => {
            let d = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut i16, n) };
            for i in 0..n {
                d[i] = convert_f32_to_int16_lsb(src[i][src_ch] * gain);
            }
        }
        ASIOST_FLOAT64_LSB => {
            let d = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut f64, n) };
            for i in 0..n {
                let v = (src[i][src_ch] * gain) as f64;
                d[i] = if v.is_finite() { v.clamp(-1.0, 1.0) } else { 0.0 };
            }
        }
        _ => {}
    }
}

// ============================================================================
// STEINBERG ASIO CALLBACK SINGLETON CONTEXT (`ASIOCallbacks` C function pointers)
// ============================================================================

struct AsioRtContext {
    engine: Box<MixerEngine>,
    iasio_ptr: *mut c_void,
    buf_infos: [ASIOBufferInfo; 4], // Up to 2 inputs + 2 outputs
    sample_types: [i32; 4],
    num_in_ch: usize,
    num_out_ch: usize,
    buffer_frames: usize,
    scratch_in: Vec<StereoFrame>,
    scratch_out: Vec<StereoFrame>,
    control: Arc<AsioSharedControl>,
    /// Present when the driver runs at a rate other than the engine rate: the engine keeps its own
    /// sample rate and the 128-tap polyphase SRC converts engine -> hardware for the output.
    out_resampler: Option<Box<crate::audio_io::WasapiStreamingResampler>>,
    /// Hardware input frames at the driver rate, consumed by the engine's clock-locked input bridge.
    input_tx: Option<rtrb::Producer<StereoFrame>>,
}

// SAFETY: Accessed exclusively inside the single active ASIO driver callback while running,
// and reclaimed only after `IASIO::stop()` + `disposeBuffers()` completes.
unsafe impl Send for AsioRtContext {}

static ACTIVE_ASIO_CTX: AtomicU64 = AtomicU64::new(0);
/// Set by `asioMessage(kAsioResetRequest / kAsioBufferSizeChange)`; serviced on the STA thread.
static ASIO_RESET_REQUESTED: AtomicBool = AtomicBool::new(false);
/// Driver-reported resync/overload events (`kAsioResyncRequest`, `kAsioOverload`).
static ASIO_XRUN_COUNT: AtomicU64 = AtomicU64::new(0);

unsafe extern "C" fn asio_cb_buffer_switch(double_buffer_index: i32, _direct_process: i32) {
    let raw_ptr = ACTIVE_ASIO_CTX.load(Ordering::Acquire) as *mut AsioRtContext;
    if raw_ptr.is_null() {
        return;
    }
    let idx = (double_buffer_index & 1) as usize;
    unsafe {
        let ctx = &mut *raw_ptr;
        let frames = ctx.buffer_frames;

        // 1. Capture planar ASIO inputs into `scratch_in` -> feed armed/input-enabled tracks (`Point 2 APDC`)
        ctx.scratch_in[..frames].fill([0.0, 0.0]);
        for ch in 0..ctx.num_in_ch.min(2) {
            let ptr = ctx.buf_infos[ch].buffers[idx] as *const u8;
            unpack_asio_planar_channel(
                ptr,
                ctx.sample_types[ch],
                frames,
                &mut ctx.scratch_in[..frames],
                ch,
            );
        }
        if ctx.num_in_ch == 1 {
            // Duplicate mono mic onto right channel
            for i in 0..frames {
                ctx.scratch_in[i][1] = ctx.scratch_in[i][0];
            }
        }
        if ctx.num_in_ch > 0 {
            if let Some(tx) = &mut ctx.input_tx {
                for &f in &ctx.scratch_in[..frames] {
                    let _ = tx.push(f);
                }
            } else {
                ctx.engine.inject_hardware_input_block(&ctx.scratch_in[..frames]);
            }
        }

        // 2. Render full 502-Track Mixer DAG + 5-Point APDC (directly, or through the SRC when the
        // driver rate differs from the engine rate).
        ctx.scratch_out[..frames].fill([0.0, 0.0]);
        if let Some(rs) = &mut ctx.out_resampler {
            rs.render_hardware_frames(
                &mut ctx.engine,
                &mut ctx.scratch_out[..frames],
                1.0,
                &ctx.control.avg_block_render_us,
                &ctx.control.callback_blocks,
            );
        } else {
            let t0 = std::time::Instant::now();
            ctx.engine.process_audio_block(&mut ctx.scratch_out[..frames]);
            let us = t0.elapsed().as_micros() as u32;
            let prev = ctx.control.avg_block_render_us.load(Ordering::Relaxed);
            let ema = if prev == 0 { us } else { ((prev as u64 * 7 + us as u64) / 8) as u32 };
            ctx.control.avg_block_render_us.store(ema, Ordering::Relaxed);
        }

        // 3. Apply atomic speaker volume & mute (`set_volume_and_mute(0.0, true)` -> exact 0.0 silence)
        let gain = if ctx.control.speaker_muted.load(Ordering::Relaxed) {
            0.0
        } else {
            f32::from_bits(ctx.control.speaker_volume_bits.load(Ordering::Relaxed))
                .clamp(0.0, 1.5)
        };

        // 4. Pack planar ASIO outputs + notify `outputReady()`
        for ch in 0..ctx.num_out_ch.min(2) {
            let slot_idx = ctx.num_in_ch + ch;
            let dst = ctx.buf_infos[slot_idx].buffers[idx] as *mut u8;
            pack_asio_planar_channel(
                &ctx.scratch_out[..frames],
                ch,
                gain,
                dst,
                ctx.sample_types[slot_idx],
                frames,
            );
        }
        if !ctx.iasio_ptr.is_null() && ctx.control.output_ready_supported.load(Ordering::Relaxed) {
            let vtbl = *(ctx.iasio_ptr as *const *const IASIOVtbl);
            let _ = ((*vtbl).output_ready)(ctx.iasio_ptr);
        }
        if ctx.out_resampler.is_none() {
            ctx.control.callback_blocks.fetch_add(1, Ordering::Relaxed);
        }
    }
}

unsafe extern "C" fn asio_cb_sample_rate_changed(_s_rate: f64) {}

unsafe extern "C" fn asio_cb_message(
    selector: i32,
    _value: i32,
    _message: *mut c_void,
    _opt: *mut f64,
) -> i32 {
    const K_ASIO_SELECTOR_SUPPORTED: i32 = 1;
    const K_ASIO_ENGINE_VERSION: i32 = 2;
    const K_ASIO_RESET_REQUEST: i32 = 3;
    const K_ASIO_BUFFER_SIZE_CHANGE: i32 = 4;
    const K_ASIO_RESYNC_REQUEST: i32 = 5;
    const K_ASIO_LATENCIES_CHANGED: i32 = 6;
    const K_ASIO_OVERLOAD: i32 = 13;
    match selector {
        K_ASIO_SELECTOR_SUPPORTED => match _value {
            K_ASIO_RESET_REQUEST
            | K_ASIO_BUFFER_SIZE_CHANGE
            | K_ASIO_RESYNC_REQUEST
            | K_ASIO_LATENCIES_CHANGED
            | K_ASIO_OVERLOAD => 1,
            _ => 0,
        },
        K_ASIO_ENGINE_VERSION => 2,
        K_ASIO_RESET_REQUEST | K_ASIO_BUFFER_SIZE_CHANGE => {
            // The driver configuration changed: the STA thread tears the stream down and reopens it.
            ASIO_RESET_REQUESTED.store(true, Ordering::Release);
            1
        }
        K_ASIO_RESYNC_REQUEST | K_ASIO_OVERLOAD => {
            ASIO_XRUN_COUNT.fetch_add(1, Ordering::Relaxed);
            1
        }
        K_ASIO_LATENCIES_CHANGED => 1,
        _ => 0,
    }
}

unsafe extern "C" fn asio_cb_buffer_switch_time_info(
    params: *mut c_void,
    double_buffer_index: i32,
    direct_process: i32,
) -> *mut c_void {
    unsafe {
        asio_cb_buffer_switch(double_buffer_index, direct_process);
    }
    params
}

static ASIO_CALLBACKS_STATIC: ASIOCallbacks = ASIOCallbacks {
    buffer_switch: asio_cb_buffer_switch,
    sample_rate_did_change: asio_cb_sample_rate_changed,
    asio_message: asio_cb_message,
    buffer_switch_time_info: asio_cb_buffer_switch_time_info,
};

struct AsioSharedControl {
    shutdown: AtomicBool,
    stream_running: AtomicBool,
    sample_rate_hz: AtomicU32,
    buffer_frames: AtomicU32,
    input_channels: AtomicU32,
    output_channels: AtomicU32,
    input_latency: AtomicU32,
    output_latency: AtomicU32,
    callback_blocks: AtomicU64,
    speaker_volume_bits: AtomicU32,
    speaker_muted: AtomicBool,
    /// Measured engine render time per callback (EMA, microseconds).
    avg_block_render_us: AtomicU32,
    /// `IASIO::outputReady()` returned `ASE_OK` once at start-up (ASIO 2.0 contract).
    output_ready_supported: AtomicBool,
    meta: RwLock<(String, String, String)>, // (driver_name, clsid, status_msg)
}

/// Full-Duplex Steinberg ASIO Hardware Stream (`IASIO` COM).
pub struct AsioHardwareStream {
    control: Arc<AsioSharedControl>,
    recovered_engine: Arc<Mutex<Option<Box<MixerEngine>>>>,
    worker: Option<JoinHandle<()>>,
}

impl AsioHardwareStream {
    /// Spawns a dedicated STA + MMCSS `"Pro Audio"` thread opening the ASIO driver identified by `driver`.
    /// Call `set_volume_and_mute(0.0, true)` immediately (or pass `start_muted = true`) in automated tests!
    pub fn spawn(
        engine: Box<MixerEngine>,
        driver: AsioDriverInfo,
        preferred_sample_rate: f64,
        start_muted: bool,
    ) -> Self {
        let sr_hz = preferred_sample_rate.round().max(8_000.0) as u32;
        let control = Arc::new(AsioSharedControl {
            shutdown: AtomicBool::new(false),
            stream_running: AtomicBool::new(false),
            sample_rate_hz: AtomicU32::new(sr_hz),
            buffer_frames: AtomicU32::new(256),
            input_channels: AtomicU32::new(0),
            output_channels: AtomicU32::new(2),
            input_latency: AtomicU32::new(0),
            output_latency: AtomicU32::new(0),
            callback_blocks: AtomicU64::new(0),
            speaker_volume_bits: AtomicU32::new(if start_muted {
                0.0_f32.to_bits()
            } else {
                0.85_f32.to_bits()
            }),
            speaker_muted: AtomicBool::new(start_muted),
            avg_block_render_us: AtomicU32::new(0),
            output_ready_supported: AtomicBool::new(false),
            meta: RwLock::new((
                driver.description.clone(),
                driver.clsid.clone(),
                "Initializing IASIO COM...".to_string(),
            )),
        });
        let recovered_engine = Arc::new(Mutex::new(None));
        let ctrl_bg = Arc::clone(&control);
        let rec_bg = Arc::clone(&recovered_engine);

        let worker = thread::Builder::new()
            .name("toneforge-asio-rt".to_string())
            .stack_size(16 * 1024 * 1024)
            .spawn(move || {
                run_asio_sta_thread(engine, driver, preferred_sample_rate, ctrl_bg, rec_bg);
            })
            .expect("failed to spawn toneforge-asio-rt thread");

        Self {
            control,
            recovered_engine,
            worker: Some(worker),
        }
    }

    /// Atomically sets hardware speaker output gain (`0.0..=1.5`) and mute state (`0` RT locks).
    #[inline]
    pub fn set_volume_and_mute(&self, volume_linear: f32, muted: bool) {
        self.control
            .speaker_volume_bits
            .store(volume_linear.clamp(0.0, 1.5).to_bits(), Ordering::Relaxed);
        self.control.speaker_muted.store(muted, Ordering::Relaxed);
    }

    pub fn status_snapshot(&self) -> AsioStreamStatusSnapshot {
        let (driver_name, clsid, status_message) = self.control.meta.read().clone();
        let vol = f32::from_bits(self.control.speaker_volume_bits.load(Ordering::Relaxed));
        AsioStreamStatusSnapshot {
            driver_name,
            clsid,
            stream_running: self.control.stream_running.load(Ordering::Relaxed),
            sample_rate_hz: self.control.sample_rate_hz.load(Ordering::Relaxed),
            buffer_frames: self.control.buffer_frames.load(Ordering::Relaxed),
            input_channels: self.control.input_channels.load(Ordering::Relaxed) as u16,
            output_channels: self.control.output_channels.load(Ordering::Relaxed) as u16,
            input_latency_samples: self.control.input_latency.load(Ordering::Relaxed),
            output_latency_samples: self.control.output_latency.load(Ordering::Relaxed),
            callback_blocks: self.control.callback_blocks.load(Ordering::Relaxed),
            speaker_volume_pct: (vol * 100.0).clamp(0.0, 150.0),
            speaker_muted: self.control.speaker_muted.load(Ordering::Relaxed),
            status_message,
            avg_block_render_us: self.control.avg_block_render_us.load(Ordering::Relaxed),
            xrun_count: ASIO_XRUN_COUNT.load(Ordering::Relaxed),
        }
    }

    /// Stops the ASIO stream (`IASIO::stop` + `disposeBuffers` + `release`) and joins the STA thread.
    pub fn shutdown(&mut self) {
        self.control.shutdown.store(true, Ordering::Release);
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }

    /// Stops the ASIO stream cleanly and returns ownership of `Box<MixerEngine>` (for seamless driver switching).
    pub fn stop_and_take_engine(mut self) -> Option<Box<MixerEngine>> {
        self.control.shutdown.store(true, Ordering::Release);
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
        self.recovered_engine.lock().take()
    }
}

impl Drop for AsioHardwareStream {
    fn drop(&mut self) {
        self.control.shutdown.store(true, Ordering::Release);
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

fn create_hidden_asio_sys_hwnd() -> *mut c_void {
    const CLS: &[u16] = &[
        b'T' as u16,
        b'F' as u16,
        b'A' as u16,
        b's' as u16,
        b'i' as u16,
        b'o' as u16,
        b'H' as u16,
        b'w' as u16,
        0,
    ];
    unsafe {
        let hinst = GetModuleHandleW(std::ptr::null());
        let wc = WNDCLASSEXW {
            cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
            style: 0,
            lpfnWndProc: Some(DefWindowProcW),
            cbClsExtra: 0,
            cbWndExtra: 0,
            hInstance: hinst,
            hIcon: std::ptr::null_mut(),
            hCursor: std::ptr::null_mut(),
            hbrBackground: std::ptr::null_mut(),
            lpszMenuName: std::ptr::null(),
            lpszClassName: CLS.as_ptr(),
            hIconSm: std::ptr::null_mut(),
        };
        let _ = RegisterClassExW(&wc);
        let empty = [0u16; 1];
        // Message-only window (`HWND_MESSAGE` parent) owned by this ASIO STA thread.
        const HWND_MESSAGE: *mut c_void = -3isize as *mut c_void;
        CreateWindowExW(
            0,
            CLS.as_ptr(),
            empty.as_ptr(),
            WS_POPUP,
            0,
            0,
            1,
            1,
            HWND_MESSAGE,
            std::ptr::null_mut(),
            hinst,
            std::ptr::null(),
        )
    }
}

fn run_asio_sta_thread(
    mut engine: Box<MixerEngine>,
    driver: AsioDriverInfo,
    preferred_sr: f64,
    control: Arc<AsioSharedControl>,
    recovered_slot: Arc<Mutex<Option<Box<MixerEngine>>>>,
) {
    let _mmcss = ProAudioThreadGuard::enter();
    let hr_com = unsafe { CoInitializeEx(std::ptr::null(), COINIT_APARTMENTTHREADED as u32) };
    let com_inited = hr_com >= 0;

    let Some(clsid_guid) = parse_win32_guid(&driver.clsid) else {
        control.meta.write().2 = format!("Invalid ASIO CLSID {}", driver.clsid);
        *recovered_slot.lock() = Some(engine);
        if com_inited {
            unsafe { CoUninitialize() };
        }
        return;
    };

    let sys_hwnd = create_hidden_asio_sys_hwnd();
    let mut iasio_ptr: *mut c_void = std::ptr::null_mut();
    let create_hr = unsafe {
        CoCreateInstance(
            &clsid_guid,
            std::ptr::null_mut(),
            CLSCTX_INPROC_SERVER,
            &clsid_guid,
            &mut iasio_ptr,
        )
    };

    if create_hr < 0 || iasio_ptr.is_null() {
        control.meta.write().2 = format!("CoCreateInstance(IASIO) returned 0x{create_hr:08X}");
        if !sys_hwnd.is_null() {
            unsafe {
                let _ = DestroyWindow(sys_hwnd);
            }
        }
        *recovered_slot.lock() = Some(engine);
        if com_inited {
            unsafe { CoUninitialize() };
        }
        return;
    }

    unsafe {
        let vtbl = *(iasio_ptr as *const *const IASIOVtbl);
        let init_ok = ((*vtbl).init)(iasio_ptr, sys_hwnd);
        if init_ok == ASIO_FALSE {
            let mut err_buf = [0 as c_char; 128];
            ((*vtbl).get_error_message)(iasio_ptr, err_buf.as_mut_ptr());
            let msg = CStr::from_ptr(err_buf.as_ptr()).to_string_lossy().into_owned();
            control.meta.write().2 = format!("IASIO::init failed: {msg}");
            let _ = ((*vtbl).release)(iasio_ptr);
            if !sys_hwnd.is_null() {
                let _ = DestroyWindow(sys_hwnd);
            }
            *recovered_slot.lock() = Some(engine);
            if com_inited {
                CoUninitialize();
            }
            return;
        }

        let mut num_in = 0i32;
        let mut num_out = 0i32;
        let _ = ((*vtbl).get_channels)(iasio_ptr, &mut num_in, &mut num_out);

        let mut min_sz = 64i32;
        let mut max_sz = 2048i32;
        let mut pref_sz = 256i32;
        let mut gran = 0i32;
        let _ = ((*vtbl).get_buffer_size)(
            iasio_ptr,
            &mut min_sz,
            &mut max_sz,
            &mut pref_sz,
            &mut gran,
        );
        let chosen_buf = pref_sz.clamp(32, crate::MAX_BLOCK_SIZE as i32) as usize;

        if ((*vtbl).can_sample_rate)(iasio_ptr, preferred_sr) == ASE_OK {
            let _ = ((*vtbl).set_sample_rate)(iasio_ptr, preferred_sr);
        }
        let mut actual_sr = preferred_sr;
        let _ = ((*vtbl).get_sample_rate)(iasio_ptr, &mut actual_sr);
        if !(actual_sr > 8_000.0) {
            actual_sr = preferred_sr;
        }
        // The engine keeps its own sample rate (all plugins, assets and tempo maths are prepared for
        // it); a driver running at another rate is bridged by high-quality SRC in both directions.
        let engine_sr = engine.slicer.sample_rate;
        let needs_src = (actual_sr - engine_sr).abs() > 0.5;

        let use_in = (num_in.max(0) as usize).min(2);
        let use_out = (num_out.max(0) as usize).min(2);
        let total_ch = use_in + use_out;

        let mut buf_infos = [ASIOBufferInfo {
            is_input: 0,
            channel_num: 0,
            buffers: [std::ptr::null_mut(); 2],
        }; 4];
        let mut sample_types = [ASIOST_INT32_LSB; 4];

        for c in 0..use_in {
            buf_infos[c] = ASIOBufferInfo {
                is_input: ASIO_TRUE,
                channel_num: c as i32,
                buffers: [std::ptr::null_mut(); 2],
            };
            let mut ch_info: ASIOChannelInfo = std::mem::zeroed();
            ch_info.channel = c as i32;
            ch_info.is_input = ASIO_TRUE;
            if ((*vtbl).get_channel_info)(iasio_ptr, &mut ch_info) == ASE_OK {
                sample_types[c] = ch_info.sample_type;
            }
        }
        for c in 0..use_out {
            let slot = use_in + c;
            buf_infos[slot] = ASIOBufferInfo {
                is_input: ASIO_FALSE,
                channel_num: c as i32,
                buffers: [std::ptr::null_mut(); 2],
            };
            let mut ch_info: ASIOChannelInfo = std::mem::zeroed();
            ch_info.channel = c as i32;
            ch_info.is_input = ASIO_FALSE;
            if ((*vtbl).get_channel_info)(iasio_ptr, &mut ch_info) == ASE_OK {
                sample_types[slot] = ch_info.sample_type;
            }
        }

        if total_ch == 0 {
            control.meta.write().2 = "ASIO driver reported 0 audio channels".to_string();
            let _ = ((*vtbl).release)(iasio_ptr);
            if !sys_hwnd.is_null() {
                let _ = DestroyWindow(sys_hwnd);
            }
            *recovered_slot.lock() = Some(engine);
            if com_inited {
                CoUninitialize();
            }
            return;
        }

        let cb_rc = ((*vtbl).create_buffers)(
            iasio_ptr,
            buf_infos.as_mut_ptr(),
            total_ch as i32,
            chosen_buf as i32,
            &ASIO_CALLBACKS_STATIC,
        );
        if cb_rc != ASE_OK {
            control.meta.write().2 = format!("IASIO::createBuffers failed ({cb_rc})");
            let _ = ((*vtbl).release)(iasio_ptr);
            if !sys_hwnd.is_null() {
                let _ = DestroyWindow(sys_hwnd);
            }
            *recovered_slot.lock() = Some(engine);
            if com_inited {
                CoUninitialize();
            }
            return;
        }

        let mut lat_in = 0i32;
        let mut lat_out = 0i32;
        let _ = ((*vtbl).get_latencies)(iasio_ptr, &mut lat_in, &mut lat_out);

        control
            .sample_rate_hz
            .store(actual_sr.round() as u32, Ordering::Relaxed);
        control
            .buffer_frames
            .store(chosen_buf as u32, Ordering::Relaxed);
        control
            .input_channels
            .store(use_in as u32, Ordering::Relaxed);
        control
            .output_channels
            .store(use_out as u32, Ordering::Relaxed);
        control
            .input_latency
            .store(lat_in.max(0) as u32, Ordering::Relaxed);
        control
            .output_latency
            .store(lat_out.max(0) as u32, Ordering::Relaxed);

        let (out_resampler, input_tx) = if needs_src {
            let rs = Box::new(crate::audio_io::WasapiStreamingResampler::new(
                engine_sr.round() as u32,
                actual_sr.round() as u32,
            ));
            let tx = if use_in > 0 {
                let (tx, rx) = rtrb::RingBuffer::<StereoFrame>::new(32_768);
                // Input and output share the ASIO sample clock (same bufferSwitch), so the input
                // uses the fixed-ratio clock-locked bridge (minimum cushion, constant latency)
                // instead of a drift-tracking DLL with a 2048-frame queue.
                engine.hw_capture_bridge = Some((
                    rx,
                    crate::audio_io::AdaptiveDllDriftResampler::new_clock_locked(
                        actual_sr,
                        engine_sr,
                        chosen_buf,
                    ),
                ));
                Some(tx)
            } else {
                None
            };
            (Some(rs), tx)
        } else {
            (None, None)
        };
        let mut rt_ctx = Box::new(AsioRtContext {
            engine,
            iasio_ptr,
            buf_infos,
            sample_types,
            num_in_ch: use_in,
            num_out_ch: use_out,
            buffer_frames: chosen_buf,
            scratch_in: vec![[0.0; 2]; chosen_buf.max(256)],
            scratch_out: vec![[0.0; 2]; chosen_buf.max(256)],
            control: Arc::clone(&control),
            out_resampler,
            input_tx,
        });
        control
            .output_ready_supported
            .store(((*vtbl).output_ready)(iasio_ptr) == ASE_OK, Ordering::Relaxed);
        ASIO_RESET_REQUESTED.store(false, Ordering::Release);
        let ctx_raw = (&mut *rt_ctx) as *mut AsioRtContext;
        ACTIVE_ASIO_CTX.store(ctx_raw as u64, Ordering::Release);

        let start_rc = ((*vtbl).start)(iasio_ptr);
        if start_rc == ASE_OK {
            control.stream_running.store(true, Ordering::Release);
            control.meta.write().2 = format!(
                "ASIO Active ({} Hz, {} smp, {} in / {} out){}",
                actual_sr.round() as u32,
                chosen_buf,
                use_in,
                use_out,
                if needs_src {
                    format!(" — SRC {} Hz ↔ engine {} Hz", actual_sr.round(), engine_sr.round())
                } else {
                    String::new()
                }
            );

            while !control.shutdown.load(Ordering::Acquire) {
                if ASIO_RESET_REQUESTED.load(Ordering::Acquire) {
                    control.meta.write().2 =
                        "ASIO driver requested a reset (buffer size / format changed) — reselect the driver to reopen it"
                            .to_string();
                    break;
                }
                let mut msg: MSG = std::mem::zeroed();
                while PeekMessageW(&mut msg, std::ptr::null_mut(), 0, 0, PM_REMOVE) != 0 {
                    let _ = TranslateMessage(&msg);
                    DispatchMessageW(&msg);
                }
                thread::sleep(Duration::from_millis(8));
            }
            let _ = ((*vtbl).stop)(iasio_ptr);
        } else {
            control.meta.write().2 = format!("IASIO::start returned {start_rc}");
        }

        control.stream_running.store(false, Ordering::Release);
        ACTIVE_ASIO_CTX.store(0, Ordering::Release);
        let _ = ((*vtbl).dispose_buffers)(iasio_ptr);
        let _ = ((*vtbl).release)(iasio_ptr);

        if !sys_hwnd.is_null() {
            let _ = DestroyWindow(sys_hwnd);
        }
        let AsioRtContext { mut engine, input_tx, .. } = *rt_ctx;
        if input_tx.is_some() {
            // Detach the driver-rate input bridge that only existed for this ASIO session.
            engine.hw_capture_bridge = None;
        }
        *recovered_slot.lock() = Some(engine);
        if com_inited {
            CoUninitialize();
        }
    }
}
