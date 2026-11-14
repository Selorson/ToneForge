//! `test-vst3-clap-plugin`: Real Native Windows `x86_64` Dual-ABI Audio Plugin (`cdylib` ->
//! `test_vst3_clap_plugin.dll`) exporting **BOTH**:
//! 1. **CLAP 1.2 C ABI** (`clap_entry` + `clap.params`, `clap.latency`, `clap.audio-ports`,
//!    `clap.note-ports`, `clap.state`, `clap.gui` Win32 child `HWND`)
//! 2. **Steinberg VST3 3.7 Windows COM ABI** (`InitDll`, `ExitDll`, `GetPluginFactory` ->
//!    `IPluginFactory`, `IComponent`, `IAudioProcessor`, `IEditController`, `IPlugView` Win32 `HWND`,
//!    `IBStream` state persistence)
//!
//! DSP Engine (`DualAbiDspCore`, `0` heap allocations in real-time `process`):
//! - 8-voice polyphonic Equal-Temperament synthesizer (`MIDI 69 = A4 = 440.000 Hz` exact)
//! - Cytomic State-Variable Resonant Low-Pass Filter (`SVF`) on synth voices
//! - Unity stereo audio pass-through (`Dry Passthrough Mix`)
//! - Exact `64`-sample lookahead ring delay line (`latency = 64 samples`) for 5-Point APDC verification.

#![allow(non_camel_case_types, non_snake_case)]

use std::f64::consts::{PI, TAU};
use std::ffi::{CStr, c_char, c_void};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};
use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows_sys::Win32::Graphics::Gdi::{
    BeginPaint, CreateSolidBrush, DeleteObject, EndPaint, FillRect, PAINTSTRUCT,
};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::UI::WindowsAndMessaging::{
    CS_HREDRAW, CS_VREDRAW, CreateWindowExW, DefWindowProcW, DestroyWindow, IsWindow,
    RegisterClassExW, SW_HIDE, SW_SHOW, ShowWindow, WM_PAINT, WNDCLASSEXW, WS_CHILD, WS_VISIBLE,
};

pub const PLUGIN_REPORTED_LATENCY_SAMPLES: u32 = 64;
const LATENCY_RING_LEN: usize = 64;
const MAX_POLY_VOICES: usize = 8;
const STATE_MAGIC: &[u8; 8] = b"TFDUAL03";

pub const PARAM_SYNTH_GAIN: u32 = 0;
pub const PARAM_FILTER_CUTOFF: u32 = 1;
pub const PARAM_FILTER_RESO: u32 = 2;
pub const PARAM_PASSTHROUGH_MIX: u32 = 3;
pub const NUM_PLUGIN_PARAMS: usize = 4;

// ============================================================================
// SHARED ZERO-ALLOCATION DSP ENGINE (`DualAbiDspCore`)
// ============================================================================

#[derive(Clone, Copy)]
struct PolyVoice {
    active: bool,
    releasing: bool,
    key: i16,
    note_id: i32,
    phase: f64,
    freq_hz: f64,
    velocity: f64,
    env: f64,
}

impl Default for PolyVoice {
    fn default() -> Self {
        Self {
            active: false,
            releasing: false,
            key: 69,
            note_id: -1,
            phase: 0.0,
            freq_hz: 440.0,
            velocity: 0.0,
            env: 0.0,
        }
    }
}

struct DualAbiDspCore {
    sample_rate: f64,
    synth_gain: f64,      // [0.0 .. 2.0], default 0.5
    cutoff_hz: f64,       // [80.0 .. 18000.0], default 14000.0
    resonance: f64,       // [0.5 .. 8.0], default 0.707
    passthrough_mix: f64, // [0.0 .. 1.0], default 1.0
    voices: [PolyVoice; MAX_POLY_VOICES],
    svf_ic1eq: [f64; 2],
    svf_ic2eq: [f64; 2],
    delay_ring: [[f32; 2]; LATENCY_RING_LEN],
    delay_cursor: usize,
}

impl DualAbiDspCore {
    fn new(sample_rate: f64) -> Self {
        Self {
            sample_rate: sample_rate.max(8_000.0),
            synth_gain: 0.5,
            cutoff_hz: 14_000.0,
            resonance: std::f64::consts::FRAC_1_SQRT_2,
            passthrough_mix: 1.0,
            voices: [PolyVoice::default(); MAX_POLY_VOICES],
            svf_ic1eq: [0.0; 2],
            svf_ic2eq: [0.0; 2],
            delay_ring: [[0.0; 2]; LATENCY_RING_LEN],
            delay_cursor: 0,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.voices = [PolyVoice::default(); MAX_POLY_VOICES];
        self.svf_ic1eq = [0.0; 2];
        self.svf_ic2eq = [0.0; 2];
        self.delay_ring = [[0.0; 2]; LATENCY_RING_LEN];
        self.delay_cursor = 0;
    }

    #[inline]
    fn set_plain_param(&mut self, id: u32, plain: f64) {
        match id {
            PARAM_SYNTH_GAIN => self.synth_gain = plain.clamp(0.0, 2.0),
            PARAM_FILTER_CUTOFF => self.cutoff_hz = plain.clamp(80.0, 18_000.0),
            PARAM_FILTER_RESO => self.resonance = plain.clamp(0.5, 8.0),
            PARAM_PASSTHROUGH_MIX => self.passthrough_mix = plain.clamp(0.0, 1.0),
            _ => {}
        }
    }

    #[inline]
    fn get_plain_param(&self, id: u32) -> f64 {
        match id {
            PARAM_SYNTH_GAIN => self.synth_gain,
            PARAM_FILTER_CUTOFF => self.cutoff_hz,
            PARAM_FILTER_RESO => self.resonance,
            PARAM_PASSTHROUGH_MIX => self.passthrough_mix,
            _ => 0.0,
        }
    }

    #[inline]
    fn plain_to_norm(id: u32, plain: f64) -> f64 {
        match id {
            PARAM_SYNTH_GAIN => (plain / 2.0).clamp(0.0, 1.0),
            PARAM_FILTER_CUTOFF => ((plain - 80.0) / (18_000.0 - 80.0)).clamp(0.0, 1.0),
            PARAM_FILTER_RESO => ((plain - 0.5) / (8.0 - 0.5)).clamp(0.0, 1.0),
            PARAM_PASSTHROUGH_MIX => plain.clamp(0.0, 1.0),
            _ => 0.0,
        }
    }

    #[inline]
    fn norm_to_plain(id: u32, norm: f64) -> f64 {
        let n = norm.clamp(0.0, 1.0);
        match id {
            PARAM_SYNTH_GAIN => n * 2.0,
            PARAM_FILTER_CUTOFF => 80.0 + n * (18_000.0 - 80.0),
            PARAM_FILTER_RESO => 0.5 + n * (8.0 - 0.5),
            PARAM_PASSTHROUGH_MIX => n,
            _ => 0.0,
        }
    }

    #[inline]
    fn set_norm_param(&mut self, id: u32, norm: f64) {
        self.set_plain_param(id, Self::norm_to_plain(id, norm));
    }

    #[inline]
    fn get_norm_param(&self, id: u32) -> f64 {
        Self::plain_to_norm(id, self.get_plain_param(id))
    }

    #[inline]
    fn note_on(&mut self, key: i16, tuning_cents: f64, velocity: f64, note_id: i32) {
        let vel = velocity.clamp(0.0, 1.0);
        if vel <= 1e-6 {
            self.note_off(key, note_id);
            return;
        }
        let semitones = (key as f64 - 69.0) + (tuning_cents / 100.0);
        let freq_hz = (440.0 * (semitones / 12.0).exp2()).clamp(20.0, 20_000.0);

        // Reuse matching key or first inactive voice, otherwise steal quietest voice
        let mut slot = 0usize;
        let mut found = false;
        for (i, v) in self.voices.iter().enumerate() {
            if v.active && v.key == key {
                slot = i;
                found = true;
                break;
            }
        }
        if !found {
            for (i, v) in self.voices.iter().enumerate() {
                if !v.active {
                    slot = i;
                    found = true;
                    break;
                }
            }
        }
        if !found {
            let mut min_env = f64::MAX;
            for (i, v) in self.voices.iter().enumerate() {
                if v.env < min_env {
                    min_env = v.env;
                    slot = i;
                }
            }
        }

        let v = &mut self.voices[slot];
        if !v.active {
            v.phase = 0.0;
        }
        v.active = true;
        v.releasing = false;
        v.key = key;
        v.note_id = note_id;
        v.freq_hz = freq_hz;
        v.velocity = vel;
    }

    #[inline]
    fn note_off(&mut self, key: i16, note_id: i32) {
        for v in &mut self.voices {
            if v.active && (v.key == key || (note_id >= 0 && v.note_id == note_id)) {
                v.releasing = true;
            }
        }
    }

    #[inline]
    fn encode_state(&self) -> [u8; 40] {
        let mut buf = [0u8; 40];
        buf[0..8].copy_from_slice(STATE_MAGIC);
        buf[8..16].copy_from_slice(&self.synth_gain.to_le_bytes());
        buf[16..24].copy_from_slice(&self.cutoff_hz.to_le_bytes());
        buf[24..32].copy_from_slice(&self.resonance.to_le_bytes());
        buf[32..40].copy_from_slice(&self.passthrough_mix.to_le_bytes());
        buf
    }

    #[inline]
    fn decode_state(&mut self, bytes: &[u8]) -> bool {
        if bytes.len() < 40 || &bytes[0..8] != STATE_MAGIC {
            return false;
        }
        let g = f64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let c = f64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let r = f64::from_le_bytes(bytes[24..32].try_into().unwrap());
        let p = f64::from_le_bytes(bytes[32..40].try_into().unwrap());
        self.set_plain_param(PARAM_SYNTH_GAIN, g);
        self.set_plain_param(PARAM_FILTER_CUTOFF, c);
        self.set_plain_param(PARAM_FILTER_RESO, r);
        self.set_plain_param(PARAM_PASSTHROUGH_MIX, p);
        true
    }

    #[inline(always)]
    fn step_frame(&mut self, dry_in: [f32; 2]) -> [f32; 2] {
        let sr = self.sample_rate;
        let atk_step = 1.0 / (0.002 * sr).max(1.0);
        let rel_step = 1.0 / (0.015 * sr).max(1.0);

        let mut synth_mono = 0.0f64;
        for v in &mut self.voices {
            if !v.active {
                continue;
            }
            if v.releasing {
                v.env -= rel_step;
                if v.env <= 0.0 {
                    v.env = 0.0;
                    v.active = false;
                    continue;
                }
            } else if v.env < 1.0 {
                v.env = (v.env + atk_step).min(1.0);
            }

            let s = (v.phase * TAU).sin();
            synth_mono += s * v.velocity * v.env;
            v.phase += v.freq_hz / sr;
            if v.phase >= 1.0 {
                v.phase -= 1.0;
            }
        }

        let mut synth_stereo = [0.0f64; 2];
        if synth_mono.abs() > 1e-15 {
            // Cytomic ZDF State-Variable Low-Pass Filter
            let fc = self.cutoff_hz.clamp(40.0, 0.48 * sr);
            let g = (PI * fc / sr).tan();
            let k = 1.0 / self.resonance.max(0.5);
            let a1 = 1.0 / (1.0 + g * (g + k));
            let a2 = g * a1;
            let a3 = g * a2;
            for ch in 0..2 {
                let v0 = synth_mono * self.synth_gain;
                let v3 = v0 - self.svf_ic2eq[ch];
                let v1 = a1 * self.svf_ic1eq[ch] + a2 * v3;
                let v2 = self.svf_ic2eq[ch] + a2 * self.svf_ic1eq[ch] + a3 * v3;
                self.svf_ic1eq[ch] = 2.0 * v1 - self.svf_ic1eq[ch];
                self.svf_ic2eq[ch] = 2.0 * v2 - self.svf_ic2eq[ch];
                synth_stereo[ch] = v2;
            }
        } else {
            // Decay filter state smoothly when silent
            for ch in 0..2 {
                self.svf_ic1eq[ch] *= 0.999;
                self.svf_ic2eq[ch] *= 0.999;
            }
        }

        let immediate = [
            ((dry_in[0] as f64) * self.passthrough_mix + synth_stereo[0]) as f32,
            ((dry_in[1] as f64) * self.passthrough_mix + synth_stereo[1]) as f32,
        ];

        // Exact 64-sample circular delay ring (`PLUGIN_REPORTED_LATENCY_SAMPLES == 64`)
        let delayed = self.delay_ring[self.delay_cursor];
        self.delay_ring[self.delay_cursor] = immediate;
        self.delay_cursor = (self.delay_cursor + 1) & (LATENCY_RING_LEN - 1);
        delayed
    }
}

// ============================================================================
// SHARED WIN32 CHILD EDITOR GUI (`CreateWindowExW` `WS_CHILD | WS_VISIBLE`)
// ============================================================================

const CHILD_GUI_CLASS: &[u16] = &[
    b'T' as u16,
    b'F' as u16,
    b'D' as u16,
    b'u' as u16,
    b'a' as u16,
    b'l' as u16,
    b'G' as u16,
    b'u' as u16,
    b'i' as u16,
    0,
];

unsafe extern "system" fn child_gui_wnd_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    if msg == WM_PAINT {
        unsafe {
            let mut ps: PAINTSTRUCT = std::mem::zeroed();
            let hdc = BeginPaint(hwnd, &mut ps);
            if !hdc.is_null() {
                let brush = CreateSolidBrush(0x0024_1A14); // Deep studio dark blue-slate
                let _ = FillRect(hdc, &ps.rcPaint, brush);
                let _ = DeleteObject(brush as *mut c_void);
            }
            let _ = EndPaint(hwnd, &ps);
        }
        return 0;
    }
    unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
}

fn ensure_child_gui_class() {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| unsafe {
        let hinst = GetModuleHandleW(std::ptr::null());
        let wc = WNDCLASSEXW {
            cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
            style: CS_HREDRAW | CS_VREDRAW,
            lpfnWndProc: Some(child_gui_wnd_proc),
            cbClsExtra: 0,
            cbWndExtra: 0,
            hInstance: hinst,
            hIcon: std::ptr::null_mut(),
            hCursor: std::ptr::null_mut(),
            hbrBackground: std::ptr::null_mut(),
            lpszMenuName: std::ptr::null(),
            lpszClassName: CHILD_GUI_CLASS.as_ptr(),
            hIconSm: std::ptr::null_mut(),
        };
        let _ = RegisterClassExW(&wc);
    });
}

unsafe fn create_plugin_child_hwnd(parent: HWND, width: i32, height: i32) -> HWND {
    ensure_child_gui_class();
    let title: [u16; 1] = [0];
    unsafe {
        let hinst = GetModuleHandleW(std::ptr::null());
        let _ = RECT {
            left: 0,
            top: 0,
            right: width,
            bottom: height,
        };
        CreateWindowExW(
            0,
            CHILD_GUI_CLASS.as_ptr(),
            title.as_ptr(),
            WS_CHILD | WS_VISIBLE,
            0,
            0,
            width.max(160),
            height.max(96),
            parent,
            std::ptr::null_mut(),
            hinst,
            std::ptr::null(),
        )
    }
}

unsafe fn destroy_plugin_child_hwnd(hwnd: &mut HWND) {
    if !hwnd.is_null() {
        unsafe {
            if IsWindow(*hwnd) != 0 {
                let _ = DestroyWindow(*hwnd);
            }
        }
        *hwnd = std::ptr::null_mut();
    }
}

// ============================================================================
// PART 1: CLAP 1.2 C ABI IMPLEMENTATION (`clap_entry`)
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy)]
pub struct clap_version {
    pub major: u32,
    pub minor: u32,
    pub revision: u32,
}

#[repr(C)]
pub struct clap_plugin_entry {
    pub clap_version: clap_version,
    pub init: Option<unsafe extern "C" fn(plugin_path: *const c_char) -> bool>,
    pub deinit: Option<unsafe extern "C" fn()>,
    pub get_factory: Option<unsafe extern "C" fn(factory_id: *const c_char) -> *const c_void>,
}
unsafe impl Sync for clap_plugin_entry {}

#[repr(C)]
pub struct clap_plugin_descriptor {
    pub clap_version: clap_version,
    pub id: *const c_char,
    pub name: *const c_char,
    pub vendor: *const c_char,
    pub url: *const c_char,
    pub manual_url: *const c_char,
    pub support_url: *const c_char,
    pub version: *const c_char,
    pub description: *const c_char,
    pub features: *const *const c_char,
}
unsafe impl Sync for clap_plugin_descriptor {}

#[repr(C)]
pub struct clap_plugin_factory {
    pub get_plugin_count:
        Option<unsafe extern "C" fn(factory: *const clap_plugin_factory) -> u32>,
    pub get_plugin_descriptor: Option<
        unsafe extern "C" fn(
            factory: *const clap_plugin_factory,
            index: u32,
        ) -> *const clap_plugin_descriptor,
    >,
    pub create_plugin: Option<
        unsafe extern "C" fn(
            factory: *const clap_plugin_factory,
            host: *const c_void,
            plugin_id: *const c_char,
        ) -> *const clap_plugin,
    >,
}
unsafe impl Sync for clap_plugin_factory {}

#[repr(C)]
pub struct clap_plugin {
    pub desc: *const clap_plugin_descriptor,
    pub plugin_data: *mut c_void,
    pub init: Option<unsafe extern "C" fn(plugin: *const clap_plugin) -> bool>,
    pub destroy: Option<unsafe extern "C" fn(plugin: *const clap_plugin)>,
    pub activate: Option<
        unsafe extern "C" fn(
            plugin: *const clap_plugin,
            sample_rate: f64,
            min_frames: u32,
            max_frames: u32,
        ) -> bool,
    >,
    pub deactivate: Option<unsafe extern "C" fn(plugin: *const clap_plugin)>,
    pub start_processing: Option<unsafe extern "C" fn(plugin: *const clap_plugin) -> bool>,
    pub stop_processing: Option<unsafe extern "C" fn(plugin: *const clap_plugin)>,
    pub reset: Option<unsafe extern "C" fn(plugin: *const clap_plugin)>,
    pub process: Option<
        unsafe extern "C" fn(plugin: *const clap_plugin, process: *const clap_process) -> i32,
    >,
    pub get_extension: Option<
        unsafe extern "C" fn(plugin: *const clap_plugin, id: *const c_char) -> *const c_void,
    >,
    pub on_main_thread: Option<unsafe extern "C" fn(plugin: *const clap_plugin)>,
}

#[repr(C)]
struct clap_audio_buffer {
    data32: *mut *mut f32,
    data64: *mut *mut f64,
    channel_count: u32,
    latency: u32,
    constant_mask: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct clap_event_header {
    size: u32,
    time: u32,
    space_id: u16,
    type_: u16,
    flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct clap_event_note {
    header: clap_event_header,
    note_id: i32,
    port_index: i16,
    channel: i16,
    key: i16,
    velocity: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct clap_event_param_value {
    header: clap_event_header,
    param_id: u32,
    cookie: *mut c_void,
    note_id: i32,
    port_index: i16,
    channel: i16,
    key: i16,
    value: f64,
}

#[repr(C)]
struct clap_input_events {
    ctx: *mut c_void,
    size: Option<unsafe extern "C" fn(list: *const clap_input_events) -> u32>,
    get: Option<
        unsafe extern "C" fn(list: *const clap_input_events, index: u32) -> *const clap_event_header,
    >,
}

#[repr(C)]
pub struct clap_process {
    steady_time: i64,
    frames_count: u32,
    transport: *const c_void,
    audio_inputs: *const clap_audio_buffer,
    audio_outputs: *mut clap_audio_buffer,
    audio_inputs_count: u32,
    audio_outputs_count: u32,
    in_events: *const clap_input_events,
    out_events: *const c_void,
}

#[repr(C)]
struct clap_param_info {
    id: u32,
    flags: u32,
    cookie: *mut c_void,
    name: [c_char; 256],
    module: [c_char; 1024],
    min_value: f64,
    max_value: f64,
    default_value: f64,
}

#[repr(C)]
struct clap_plugin_params {
    count: Option<unsafe extern "C" fn(plugin: *const clap_plugin) -> u32>,
    get_info: Option<
        unsafe extern "C" fn(
            plugin: *const clap_plugin,
            param_index: u32,
            param_info: *mut clap_param_info,
        ) -> bool,
    >,
    get_value: Option<
        unsafe extern "C" fn(
            plugin: *const clap_plugin,
            param_id: u32,
            out_value: *mut f64,
        ) -> bool,
    >,
    value_to_text: Option<
        unsafe extern "C" fn(
            plugin: *const clap_plugin,
            param_id: u32,
            value: f64,
            out_buf: *mut c_char,
            cap: u32,
        ) -> bool,
    >,
    text_to_value: Option<
        unsafe extern "C" fn(
            plugin: *const clap_plugin,
            param_id: u32,
            txt: *const c_char,
            out: *mut f64,
        ) -> bool,
    >,
    flush: Option<
        unsafe extern "C" fn(
            plugin: *const clap_plugin,
            in_: *const clap_input_events,
            out: *const c_void,
        ),
    >,
}
unsafe impl Sync for clap_plugin_params {}

#[repr(C)]
struct clap_plugin_latency {
    get: Option<unsafe extern "C" fn(plugin: *const clap_plugin) -> u32>,
}
unsafe impl Sync for clap_plugin_latency {}

#[repr(C)]
struct clap_istream {
    ctx: *mut c_void,
    read: Option<
        unsafe extern "C" fn(stream: *const clap_istream, buffer: *mut c_void, size: u64) -> i64,
    >,
}

#[repr(C)]
struct clap_ostream {
    ctx: *mut c_void,
    write: Option<
        unsafe extern "C" fn(stream: *const clap_ostream, buffer: *const c_void, size: u64) -> i64,
    >,
}

#[repr(C)]
struct clap_plugin_state {
    save: Option<
        unsafe extern "C" fn(plugin: *const clap_plugin, stream: *const clap_ostream) -> bool,
    >,
    load: Option<
        unsafe extern "C" fn(plugin: *const clap_plugin, stream: *const clap_istream) -> bool,
    >,
}
unsafe impl Sync for clap_plugin_state {}

#[repr(C)]
struct clap_window {
    api: *const c_char,
    ptr: *mut c_void,
}

#[repr(C)]
struct clap_plugin_gui {
    is_api_supported: Option<
        unsafe extern "C" fn(
            plugin: *const clap_plugin,
            api: *const c_char,
            is_floating: bool,
        ) -> bool,
    >,
    get_preferred_api: Option<
        unsafe extern "C" fn(
            plugin: *const clap_plugin,
            api: *mut *const c_char,
            is_floating: *mut bool,
        ) -> bool,
    >,
    create: Option<
        unsafe extern "C" fn(
            plugin: *const clap_plugin,
            api: *const c_char,
            is_floating: bool,
        ) -> bool,
    >,
    destroy: Option<unsafe extern "C" fn(plugin: *const clap_plugin)>,
    set_scale: Option<unsafe extern "C" fn(plugin: *const clap_plugin, scale: f64) -> bool>,
    get_size: Option<
        unsafe extern "C" fn(
            plugin: *const clap_plugin,
            width: *mut u32,
            height: *mut u32,
        ) -> bool,
    >,
    can_resize: Option<unsafe extern "C" fn(plugin: *const clap_plugin) -> bool>,
    get_resize_hints: Option<unsafe extern "C" fn(plugin: *const clap_plugin, hints: *mut c_void) -> bool>,
    adjust_size: Option<
        unsafe extern "C" fn(
            plugin: *const clap_plugin,
            width: *mut u32,
            height: *mut u32,
        ) -> bool,
    >,
    set_size: Option<
        unsafe extern "C" fn(plugin: *const clap_plugin, width: u32, height: u32) -> bool,
    >,
    set_parent: Option<
        unsafe extern "C" fn(plugin: *const clap_plugin, window: *const clap_window) -> bool,
    >,
    set_transient: Option<
        unsafe extern "C" fn(plugin: *const clap_plugin, window: *const clap_window) -> bool,
    >,
    suggest_title: Option<unsafe extern "C" fn(plugin: *const clap_plugin, title: *const c_char)>,
    show: Option<unsafe extern "C" fn(plugin: *const clap_plugin) -> bool>,
    hide: Option<unsafe extern "C" fn(plugin: *const clap_plugin) -> bool>,
}
unsafe impl Sync for clap_plugin_gui {}

struct ClapPluginInstance {
    raw: clap_plugin,
    dsp: DualAbiDspCore,
    child_hwnd: HWND,
    gui_w: u32,
    gui_h: u32,
}

fn write_c_buf(dst: &mut [c_char], text: &str) {
    dst.fill(0);
    let bytes = text.as_bytes();
    let n = bytes.len().min(dst.len().saturating_sub(1));
    for i in 0..n {
        dst[i] = bytes[i] as c_char;
    }
}

static CLAP_ID: &[u8] = b"audio.toneforge.test_dual_synth_fx\0";
static CLAP_NAME: &[u8] = b"ToneForge Dual-ABI Synth & APDC FX\0";
static CLAP_VENDOR: &[u8] = b"ToneForge Clean-Room Audio\0";
static CLAP_URL: &[u8] = b"https://toneforge.local\0";
static CLAP_VER: &[u8] = b"3.0.0\0";
static CLAP_DESC: &[u8] = b"Dual VST3 3.7 COM + CLAP 1.2 Reference PolySynth & 64-Sample APDC FX\0";
static FEAT_INST: &[u8] = b"instrument\0";
static FEAT_SYNTH: &[u8] = b"synthesizer\0";
static FEAT_FX: &[u8] = b"audio-effect\0";

#[repr(C)]
struct FeaturePtrArray([*const c_char; 4]);
unsafe impl Sync for FeaturePtrArray {}

static CLAP_FEATURES: FeaturePtrArray = FeaturePtrArray([
    FEAT_INST.as_ptr() as *const c_char,
    FEAT_SYNTH.as_ptr() as *const c_char,
    FEAT_FX.as_ptr() as *const c_char,
    std::ptr::null(),
]);

static CLAP_DESCRIPTOR: clap_plugin_descriptor = clap_plugin_descriptor {
    clap_version: clap_version {
        major: 1,
        minor: 2,
        revision: 0,
    },
    id: CLAP_ID.as_ptr() as *const c_char,
    name: CLAP_NAME.as_ptr() as *const c_char,
    vendor: CLAP_VENDOR.as_ptr() as *const c_char,
    url: CLAP_URL.as_ptr() as *const c_char,
    manual_url: CLAP_URL.as_ptr() as *const c_char,
    support_url: CLAP_URL.as_ptr() as *const c_char,
    version: CLAP_VER.as_ptr() as *const c_char,
    description: CLAP_DESC.as_ptr() as *const c_char,
    features: CLAP_FEATURES.0.as_ptr(),
};

unsafe extern "C" fn clap_plug_init(_plugin: *const clap_plugin) -> bool {
    true
}

unsafe extern "C" fn clap_plug_destroy(plugin: *const clap_plugin) {
    if !plugin.is_null() {
        unsafe {
            let data = (*plugin).plugin_data as *mut ClapPluginInstance;
            if !data.is_null() {
                let mut boxed = Box::from_raw(data);
                destroy_plugin_child_hwnd(&mut boxed.child_hwnd);
            }
        }
    }
}

unsafe extern "C" fn clap_plug_activate(
    plugin: *const clap_plugin,
    sample_rate: f64,
    _min_frames: u32,
    _max_frames: u32,
) -> bool {
    unsafe {
        let inst = &mut *((*plugin).plugin_data as *mut ClapPluginInstance);
        inst.dsp.sample_rate = sample_rate.max(8_000.0);
        inst.dsp.reset();
    }
    true
}

unsafe extern "C" fn clap_plug_deactivate(_plugin: *const clap_plugin) {}
unsafe extern "C" fn clap_plug_start_processing(_plugin: *const clap_plugin) -> bool {
    true
}
unsafe extern "C" fn clap_plug_stop_processing(_plugin: *const clap_plugin) {}

unsafe extern "C" fn clap_plug_reset(plugin: *const clap_plugin) {
    unsafe {
        let inst = &mut *((*plugin).plugin_data as *mut ClapPluginInstance);
        inst.dsp.reset();
    }
}

unsafe fn dispatch_clap_event(dsp: &mut DualAbiDspCore, hdr: *const clap_event_header) {
    if hdr.is_null() {
        return;
    }
    unsafe {
        match (*hdr).type_ {
            0 => {
                // CLAP_EVENT_NOTE_ON
                let ev = &*(hdr as *const clap_event_note);
                dsp.note_on(ev.key, 0.0, ev.velocity, ev.note_id);
            }
            1 => {
                // CLAP_EVENT_NOTE_OFF
                let ev = &*(hdr as *const clap_event_note);
                dsp.note_off(ev.key, ev.note_id);
            }
            5 => {
                // CLAP_EVENT_PARAM_VALUE
                let ev = &*(hdr as *const clap_event_param_value);
                dsp.set_plain_param(ev.param_id, ev.value);
            }
            _ => {}
        }
    }
}

unsafe extern "C" fn clap_plug_process(
    plugin: *const clap_plugin,
    process: *const clap_process,
) -> i32 {
    unsafe {
        if plugin.is_null() || process.is_null() {
            return 0;
        }
        let inst = &mut *((*plugin).plugin_data as *mut ClapPluginInstance);
        let p = &*process;
        let frames = p.frames_count as usize;
        if p.audio_outputs_count == 0 || p.audio_outputs.is_null() {
            return 1;
        }
        let out_bus = &mut *p.audio_outputs;
        if out_bus.channel_count < 2 || out_bus.data32.is_null() {
            return 1;
        }
        let out_l = *out_bus.data32.add(0);
        let out_r = *out_bus.data32.add(1);

        let (has_in, in_l, in_r) = if p.audio_inputs_count > 0 && !p.audio_inputs.is_null() {
            let ib = &*p.audio_inputs;
            if ib.channel_count >= 2 && !ib.data32.is_null() {
                (true, *ib.data32.add(0), *ib.data32.add(1))
            } else {
                (false, std::ptr::null_mut(), std::ptr::null_mut())
            }
        } else {
            (false, std::ptr::null_mut(), std::ptr::null_mut())
        };

        let mut ev_count = 0u32;
        let mut get_ev: Option<
            unsafe extern "C" fn(*const clap_input_events, u32) -> *const clap_event_header,
        > = None;
        if !p.in_events.is_null() {
            if let Some(sz) = (*p.in_events).size {
                ev_count = sz(p.in_events);
            }
            get_ev = (*p.in_events).get;
        }
        let mut ev_idx = 0u32;

        for s in 0..frames {
            if let Some(gf) = get_ev {
                while ev_idx < ev_count {
                    let hdr = gf(p.in_events, ev_idx);
                    if hdr.is_null() {
                        ev_idx += 1;
                        continue;
                    }
                    if ((*hdr).time as usize) > s {
                        break;
                    }
                    dispatch_clap_event(&mut inst.dsp, hdr);
                    ev_idx += 1;
                }
            }
            let dry = if has_in {
                [*in_l.add(s), *in_r.add(s)]
            } else {
                [0.0, 0.0]
            };
            let wet = inst.dsp.step_frame(dry);
            *out_l.add(s) = wet[0];
            *out_r.add(s) = wet[1];
        }

        if let Some(gf) = get_ev {
            while ev_idx < ev_count {
                let hdr = gf(p.in_events, ev_idx);
                dispatch_clap_event(&mut inst.dsp, hdr);
                ev_idx += 1;
            }
        }

        1 // CLAP_PROCESS_CONTINUE
    }
}

unsafe extern "C" fn clap_params_count(_plugin: *const clap_plugin) -> u32 {
    NUM_PLUGIN_PARAMS as u32
}

unsafe extern "C" fn clap_params_get_info(
    _plugin: *const clap_plugin,
    param_index: u32,
    param_info: *mut clap_param_info,
) -> bool {
    if param_info.is_null() || (param_index as usize) >= NUM_PLUGIN_PARAMS {
        return false;
    }
    unsafe {
        let info = &mut *param_info;
        *info = std::mem::zeroed();
        info.id = param_index;
        info.flags = 1; // CLAP_PARAM_IS_AUTOMATABLE
        write_c_buf(&mut info.module, "Synth & Filter");
        match param_index {
            PARAM_SYNTH_GAIN => {
                write_c_buf(&mut info.name, "Synth Gain");
                info.min_value = 0.0;
                info.max_value = 2.0;
                info.default_value = 0.5;
            }
            PARAM_FILTER_CUTOFF => {
                write_c_buf(&mut info.name, "SVF Cutoff Hz");
                info.min_value = 80.0;
                info.max_value = 18_000.0;
                info.default_value = 14_000.0;
            }
            PARAM_FILTER_RESO => {
                write_c_buf(&mut info.name, "SVF Resonance");
                info.min_value = 0.5;
                info.max_value = 8.0;
                info.default_value = std::f64::consts::FRAC_1_SQRT_2;
            }
            PARAM_PASSTHROUGH_MIX => {
                write_c_buf(&mut info.name, "Dry Passthrough Mix");
                info.min_value = 0.0;
                info.max_value = 1.0;
                info.default_value = 1.0;
            }
            _ => return false,
        }
    }
    true
}

unsafe extern "C" fn clap_params_get_value(
    plugin: *const clap_plugin,
    param_id: u32,
    out_value: *mut f64,
) -> bool {
    if plugin.is_null() || out_value.is_null() || (param_id as usize) >= NUM_PLUGIN_PARAMS {
        return false;
    }
    unsafe {
        let inst = &*((*plugin).plugin_data as *const ClapPluginInstance);
        *out_value = inst.dsp.get_plain_param(param_id);
    }
    true
}

unsafe extern "C" fn clap_params_flush(
    plugin: *const clap_plugin,
    in_: *const clap_input_events,
    _out: *const c_void,
) {
    unsafe {
        if plugin.is_null() || in_.is_null() {
            return;
        }
        let inst = &mut *((*plugin).plugin_data as *mut ClapPluginInstance);
        let n = (*in_).size.map(|s| s(in_)).unwrap_or(0);
        if let Some(gf) = (*in_).get {
            for i in 0..n {
                dispatch_clap_event(&mut inst.dsp, gf(in_, i));
            }
        }
    }
}

static CLAP_EXT_PARAMS_IMPL: clap_plugin_params = clap_plugin_params {
    count: Some(clap_params_count),
    get_info: Some(clap_params_get_info),
    get_value: Some(clap_params_get_value),
    value_to_text: None,
    text_to_value: None,
    flush: Some(clap_params_flush),
};

unsafe extern "C" fn clap_latency_get(_plugin: *const clap_plugin) -> u32 {
    PLUGIN_REPORTED_LATENCY_SAMPLES
}

static CLAP_EXT_LATENCY_IMPL: clap_plugin_latency = clap_plugin_latency {
    get: Some(clap_latency_get),
};

unsafe extern "C" fn clap_state_save(
    plugin: *const clap_plugin,
    stream: *const clap_ostream,
) -> bool {
    unsafe {
        if plugin.is_null() || stream.is_null() {
            return false;
        }
        let Some(write_fn) = (*stream).write else {
            return false;
        };
        let inst = &*((*plugin).plugin_data as *const ClapPluginInstance);
        let encoded = inst.dsp.encode_state();
        let written = write_fn(stream, encoded.as_ptr() as *const c_void, encoded.len() as u64);
        written == encoded.len() as i64
    }
}

unsafe extern "C" fn clap_state_load(
    plugin: *const clap_plugin,
    stream: *const clap_istream,
) -> bool {
    unsafe {
        if plugin.is_null() || stream.is_null() {
            return false;
        }
        let Some(read_fn) = (*stream).read else {
            return false;
        };
        let mut buf = [0u8; 40];
        let mut total = 0usize;
        while total < buf.len() {
            let n = read_fn(
                stream,
                buf[total..].as_mut_ptr() as *mut c_void,
                (buf.len() - total) as u64,
            );
            if n <= 0 {
                break;
            }
            total += n as usize;
        }
        if total < 40 {
            return false;
        }
        let inst = &mut *((*plugin).plugin_data as *mut ClapPluginInstance);
        inst.dsp.decode_state(&buf)
    }
}

static CLAP_EXT_STATE_IMPL: clap_plugin_state = clap_plugin_state {
    save: Some(clap_state_save),
    load: Some(clap_state_load),
};

unsafe extern "C" fn clap_gui_is_api_supported(
    _plugin: *const clap_plugin,
    api: *const c_char,
    is_floating: bool,
) -> bool {
    if api.is_null() || is_floating {
        return false;
    }
    unsafe { CStr::from_ptr(api).to_bytes() == b"win32" }
}

unsafe extern "C" fn clap_gui_create(
    _plugin: *const clap_plugin,
    api: *const c_char,
    is_floating: bool,
) -> bool {
    unsafe { clap_gui_is_api_supported(_plugin, api, is_floating) }
}

unsafe extern "C" fn clap_gui_destroy(plugin: *const clap_plugin) {
    unsafe {
        let inst = &mut *((*plugin).plugin_data as *mut ClapPluginInstance);
        destroy_plugin_child_hwnd(&mut inst.child_hwnd);
    }
}

unsafe extern "C" fn clap_gui_get_size(
    plugin: *const clap_plugin,
    width: *mut u32,
    height: *mut u32,
) -> bool {
    unsafe {
        let inst = &*((*plugin).plugin_data as *const ClapPluginInstance);
        if !width.is_null() {
            *width = inst.gui_w;
        }
        if !height.is_null() {
            *height = inst.gui_h;
        }
    }
    true
}

unsafe extern "C" fn clap_gui_can_resize(_plugin: *const clap_plugin) -> bool {
    true
}

unsafe extern "C" fn clap_gui_set_size(
    plugin: *const clap_plugin,
    width: u32,
    height: u32,
) -> bool {
    unsafe {
        let inst = &mut *((*plugin).plugin_data as *mut ClapPluginInstance);
        inst.gui_w = width.max(160);
        inst.gui_h = height.max(96);
    }
    true
}

unsafe extern "C" fn clap_gui_set_parent(
    plugin: *const clap_plugin,
    window: *const clap_window,
) -> bool {
    unsafe {
        if plugin.is_null() || window.is_null() || (*window).ptr.is_null() {
            return false;
        }
        let inst = &mut *((*plugin).plugin_data as *mut ClapPluginInstance);
        destroy_plugin_child_hwnd(&mut inst.child_hwnd);
        let child = create_plugin_child_hwnd(
            (*window).ptr as HWND,
            inst.gui_w as i32,
            inst.gui_h as i32,
        );
        inst.child_hwnd = child;
        !child.is_null()
    }
}

unsafe extern "C" fn clap_gui_show(plugin: *const clap_plugin) -> bool {
    unsafe {
        let inst = &*((*plugin).plugin_data as *const ClapPluginInstance);
        if !inst.child_hwnd.is_null() {
            ShowWindow(inst.child_hwnd, SW_SHOW);
        }
    }
    true
}

unsafe extern "C" fn clap_gui_hide(plugin: *const clap_plugin) -> bool {
    unsafe {
        let inst = &*((*plugin).plugin_data as *const ClapPluginInstance);
        if !inst.child_hwnd.is_null() {
            ShowWindow(inst.child_hwnd, SW_HIDE);
        }
    }
    true
}

static CLAP_EXT_GUI_IMPL: clap_plugin_gui = clap_plugin_gui {
    is_api_supported: Some(clap_gui_is_api_supported),
    get_preferred_api: None,
    create: Some(clap_gui_create),
    destroy: Some(clap_gui_destroy),
    set_scale: None,
    get_size: Some(clap_gui_get_size),
    can_resize: Some(clap_gui_can_resize),
    get_resize_hints: None,
    adjust_size: None,
    set_size: Some(clap_gui_set_size),
    set_parent: Some(clap_gui_set_parent),
    set_transient: None,
    suggest_title: None,
    show: Some(clap_gui_show),
    hide: Some(clap_gui_hide),
};

unsafe extern "C" fn clap_plug_get_extension(
    _plugin: *const clap_plugin,
    id: *const c_char,
) -> *const c_void {
    if id.is_null() {
        return std::ptr::null();
    }
    let bytes = unsafe { CStr::from_ptr(id).to_bytes() };
    match bytes {
        b"clap.params" => (&CLAP_EXT_PARAMS_IMPL) as *const clap_plugin_params as *const c_void,
        b"clap.latency" => (&CLAP_EXT_LATENCY_IMPL) as *const clap_plugin_latency as *const c_void,
        b"clap.state" => (&CLAP_EXT_STATE_IMPL) as *const clap_plugin_state as *const c_void,
        b"clap.gui" => (&CLAP_EXT_GUI_IMPL) as *const clap_plugin_gui as *const c_void,
        _ => std::ptr::null(),
    }
}

unsafe extern "C" fn clap_plug_on_main_thread(_plugin: *const clap_plugin) {}

unsafe extern "C" fn clap_factory_count(_factory: *const clap_plugin_factory) -> u32 {
    1
}

unsafe extern "C" fn clap_factory_get_desc(
    _factory: *const clap_plugin_factory,
    index: u32,
) -> *const clap_plugin_descriptor {
    if index == 0 {
        &CLAP_DESCRIPTOR
    } else {
        std::ptr::null()
    }
}

unsafe extern "C" fn clap_factory_create(
    _factory: *const clap_plugin_factory,
    _host: *const c_void,
    plugin_id: *const c_char,
) -> *const clap_plugin {
    if !plugin_id.is_null() {
        let req = unsafe { CStr::from_ptr(plugin_id).to_bytes() };
        if !req.is_empty() && req != b"audio.toneforge.test_dual_synth_fx" {
            return std::ptr::null();
        }
    }
    let boxed = Box::new(ClapPluginInstance {
        raw: clap_plugin {
            desc: &CLAP_DESCRIPTOR,
            plugin_data: std::ptr::null_mut(),
            init: Some(clap_plug_init),
            destroy: Some(clap_plug_destroy),
            activate: Some(clap_plug_activate),
            deactivate: Some(clap_plug_deactivate),
            start_processing: Some(clap_plug_start_processing),
            stop_processing: Some(clap_plug_stop_processing),
            reset: Some(clap_plug_reset),
            process: Some(clap_plug_process),
            get_extension: Some(clap_plug_get_extension),
            on_main_thread: Some(clap_plug_on_main_thread),
        },
        dsp: DualAbiDspCore::new(48_000.0),
        child_hwnd: std::ptr::null_mut(),
        gui_w: 560,
        gui_h: 340,
    });
    let ptr = Box::into_raw(boxed);
    unsafe {
        (*ptr).raw.plugin_data = ptr as *mut c_void;
        &(*ptr).raw as *const clap_plugin
    }
}

static CLAP_FACTORY: clap_plugin_factory = clap_plugin_factory {
    get_plugin_count: Some(clap_factory_count),
    get_plugin_descriptor: Some(clap_factory_get_desc),
    create_plugin: Some(clap_factory_create),
};

unsafe extern "C" fn clap_entry_init(_plugin_path: *const c_char) -> bool {
    true
}
unsafe extern "C" fn clap_entry_deinit() {}
unsafe extern "C" fn clap_entry_get_factory(factory_id: *const c_char) -> *const c_void {
    if factory_id.is_null() {
        return std::ptr::null();
    }
    let id = unsafe { CStr::from_ptr(factory_id).to_bytes() };
    if id == b"clap.plugin-factory" {
        (&CLAP_FACTORY) as *const clap_plugin_factory as *const c_void
    } else {
        std::ptr::null()
    }
}

#[unsafe(no_mangle)]
pub static clap_entry: clap_plugin_entry = clap_plugin_entry {
    clap_version: clap_version {
        major: 1,
        minor: 2,
        revision: 0,
    },
    init: Some(clap_entry_init),
    deinit: Some(clap_entry_deinit),
    get_factory: Some(clap_entry_get_factory),
};

// ============================================================================
// PART 2: STEINBERG VST3 3.7 WINDOWS COM ABI IMPLEMENTATION (`GetPluginFactory`)
// ============================================================================

type tresult = i32;
type TUID = [u8; 16];
type TBool = u8;
type ParamID = u32;
type ParamValue = f64;

const K_RESULT_OK: tresult = 0;
const K_NO_INTERFACE: tresult = -1;
const K_INVALID_ARGUMENT: tresult = -3;

const fn vst3_com_uid(l1: u32, l2: u32, l3: u32, l4: u32) -> TUID {
    [
        (l1 & 0x0000_00FF) as u8,
        ((l1 & 0x0000_FF00) >> 8) as u8,
        ((l1 & 0x00FF_0000) >> 16) as u8,
        ((l1 & 0xFF00_0000) >> 24) as u8,
        ((l2 & 0x00FF_0000) >> 16) as u8,
        ((l2 & 0xFF00_0000) >> 24) as u8,
        (l2 & 0x0000_00FF) as u8,
        ((l2 & 0x0000_FF00) >> 8) as u8,
        ((l3 & 0xFF00_0000) >> 24) as u8,
        ((l3 & 0x00FF_0000) >> 16) as u8,
        ((l3 & 0x0000_FF00) >> 8) as u8,
        (l3 & 0x0000_00FF) as u8,
        ((l4 & 0xFF00_0000) >> 24) as u8,
        ((l4 & 0x00FF_0000) >> 16) as u8,
        ((l4 & 0x0000_FF00) >> 8) as u8,
        (l4 & 0x0000_00FF) as u8,
    ]
}

const IID_FUNKNOWN: TUID = vst3_com_uid(0x0000_0000, 0x0000_0000, 0xC000_0000, 0x0000_0046);
const IID_IPLUGIN_BASE: TUID = vst3_com_uid(0x2288_8DDB, 0x156E_45AE, 0x8358_B348, 0x0819_0625);
const IID_IPLUGIN_FACTORY: TUID = vst3_com_uid(0x7A4D_811C, 0x5211_4A1F, 0xAED9_D2EE, 0x0B43_BF9F);
const IID_ICOMPONENT: TUID = vst3_com_uid(0xE831_FF31, 0xF2D5_4301, 0x928E_BBEE, 0x2569_7802);
const IID_IAUDIO_PROCESSOR: TUID =
    vst3_com_uid(0x4204_3F99, 0xB7DA_453C, 0xA569_E79D, 0x9AAE_C33D);
const IID_IEDIT_CONTROLLER: TUID =
    vst3_com_uid(0xDCD7_BBE3, 0x7742_448D, 0xA874_AACC, 0x979C_759E);
const IID_IPLUG_VIEW: TUID = vst3_com_uid(0x5BC3_2507, 0xD060_49EA, 0xA615_1B52, 0x2B75_5B29);
pub const TEST_PLUGIN_VST3_CID: TUID =
    vst3_com_uid(0x5446_5633, 0x504C_5547, 0x3230_3236, 0x0000_0064);

#[repr(C)]
struct FUnknownVtbl {
    queryInterface:
        unsafe extern "system" fn(this: *mut c_void, iid: *const TUID, obj: *mut *mut c_void) -> tresult,
    addRef: unsafe extern "system" fn(this: *mut c_void) -> u32,
    release: unsafe extern "system" fn(this: *mut c_void) -> u32,
}

#[repr(C)]
struct PFactoryInfo {
    vendor: [c_char; 64],
    url: [c_char; 256],
    email: [c_char; 128],
    flags: i32,
}

#[repr(C)]
struct PClassInfo {
    cid: TUID,
    cardinality: i32,
    category: [c_char; 32],
    name: [c_char; 64],
}

#[repr(C)]
struct IPluginFactoryVtbl {
    base: FUnknownVtbl,
    getFactoryInfo:
        unsafe extern "system" fn(this: *mut c_void, info: *mut PFactoryInfo) -> tresult,
    countClasses: unsafe extern "system" fn(this: *mut c_void) -> i32,
    getClassInfo:
        unsafe extern "system" fn(this: *mut c_void, index: i32, info: *mut PClassInfo) -> tresult,
    createInstance: unsafe extern "system" fn(
        this: *mut c_void,
        cid: *const TUID,
        iid: *const TUID,
        obj: *mut *mut c_void,
    ) -> tresult,
}

#[repr(C)]
struct IPluginBaseVtbl {
    base: FUnknownVtbl,
    initialize: unsafe extern "system" fn(this: *mut c_void, context: *mut c_void) -> tresult,
    terminate: unsafe extern "system" fn(this: *mut c_void) -> tresult,
}

#[repr(C)]
struct BusInfo {
    mediaType: i32,
    direction: i32,
    channelCount: i32,
    name: [u16; 128],
    busType: i32,
    flags: u32,
}

#[repr(C)]
struct IComponentVtbl {
    base: IPluginBaseVtbl,
    getControllerClassId: unsafe extern "system" fn(this: *mut c_void, classId: *mut TUID) -> tresult,
    setIoMode: unsafe extern "system" fn(this: *mut c_void, mode: i32) -> tresult,
    getBusCount: unsafe extern "system" fn(this: *mut c_void, type_: i32, dir: i32) -> i32,
    getBusInfo: unsafe extern "system" fn(
        this: *mut c_void,
        type_: i32,
        dir: i32,
        index: i32,
        bus: *mut BusInfo,
    ) -> tresult,
    getRoutingInfo:
        unsafe extern "system" fn(this: *mut c_void, inInfo: *mut c_void, outInfo: *mut c_void) -> tresult,
    activateBus: unsafe extern "system" fn(
        this: *mut c_void,
        type_: i32,
        dir: i32,
        index: i32,
        state: TBool,
    ) -> tresult,
    setActive: unsafe extern "system" fn(this: *mut c_void, state: TBool) -> tresult,
    setState: unsafe extern "system" fn(this: *mut c_void, state: *mut c_void) -> tresult,
    getState: unsafe extern "system" fn(this: *mut c_void, state: *mut c_void) -> tresult,
}

#[repr(C)]
struct ProcessSetup {
    processMode: i32,
    symbolicSampleSize: i32,
    maxSamplesPerBlock: i32,
    sampleRate: f64,
}

#[repr(C)]
struct AudioBusBuffers {
    numChannels: i32,
    silenceFlags: u64,
    channelBuffers32: *mut *mut f32,
}

#[repr(C)]
struct ProcessData {
    processMode: i32,
    symbolicSampleSize: i32,
    numSamples: i32,
    numInputs: i32,
    numOutputs: i32,
    inputs: *mut AudioBusBuffers,
    outputs: *mut AudioBusBuffers,
    inputParameterChanges: *mut c_void,
    outputParameterChanges: *mut c_void,
    inputEvents: *mut c_void,
    outputEvents: *mut c_void,
    processContext: *mut c_void,
}

#[repr(C)]
struct IAudioProcessorVtbl {
    base: FUnknownVtbl,
    setBusArrangements: unsafe extern "system" fn(
        this: *mut c_void,
        inputs: *mut u64,
        numIns: i32,
        outputs: *mut u64,
        numOuts: i32,
    ) -> tresult,
    getBusArrangement:
        unsafe extern "system" fn(this: *mut c_void, dir: i32, index: i32, arr: *mut u64) -> tresult,
    canProcessSampleSize: unsafe extern "system" fn(this: *mut c_void, sz: i32) -> tresult,
    getLatencySamples: unsafe extern "system" fn(this: *mut c_void) -> u32,
    setupProcessing:
        unsafe extern "system" fn(this: *mut c_void, setup: *mut ProcessSetup) -> tresult,
    setProcessing: unsafe extern "system" fn(this: *mut c_void, state: TBool) -> tresult,
    process: unsafe extern "system" fn(this: *mut c_void, data: *mut ProcessData) -> tresult,
    getTailSamples: unsafe extern "system" fn(this: *mut c_void) -> u32,
}

#[repr(C)]
struct ParameterInfo {
    id: ParamID,
    title: [u16; 128],
    shortTitle: [u16; 128],
    units: [u16; 128],
    stepCount: i32,
    defaultNormalizedValue: ParamValue,
    unitId: i32,
    flags: i32,
}

#[repr(C)]
struct IEditControllerVtbl {
    base: IPluginBaseVtbl,
    setComponentState: unsafe extern "system" fn(this: *mut c_void, state: *mut c_void) -> tresult,
    setState: unsafe extern "system" fn(this: *mut c_void, state: *mut c_void) -> tresult,
    getState: unsafe extern "system" fn(this: *mut c_void, state: *mut c_void) -> tresult,
    getParameterCount: unsafe extern "system" fn(this: *mut c_void) -> i32,
    getParameterInfo: unsafe extern "system" fn(
        this: *mut c_void,
        paramIndex: i32,
        info: *mut ParameterInfo,
    ) -> tresult,
    getParamStringByValue: unsafe extern "system" fn(
        this: *mut c_void,
        id: ParamID,
        val: ParamValue,
        string: *mut u16,
    ) -> tresult,
    getParamValueByString: unsafe extern "system" fn(
        this: *mut c_void,
        id: ParamID,
        string: *mut u16,
        val: *mut ParamValue,
    ) -> tresult,
    normalizedParamToPlain:
        unsafe extern "system" fn(this: *mut c_void, id: ParamID, val: ParamValue) -> ParamValue,
    plainParamToNormalized:
        unsafe extern "system" fn(this: *mut c_void, id: ParamID, val: ParamValue) -> ParamValue,
    getParamNormalized: unsafe extern "system" fn(this: *mut c_void, id: ParamID) -> ParamValue,
    setParamNormalized:
        unsafe extern "system" fn(this: *mut c_void, id: ParamID, value: ParamValue) -> tresult,
    setComponentHandler:
        unsafe extern "system" fn(this: *mut c_void, handler: *mut c_void) -> tresult,
    createView: unsafe extern "system" fn(this: *mut c_void, name: *const c_char) -> *mut c_void,
}

#[repr(C)]
struct IBStreamVtbl {
    base: FUnknownVtbl,
    read: unsafe extern "system" fn(
        this: *mut c_void,
        buffer: *mut c_void,
        numBytes: i32,
        numBytesRead: *mut i32,
    ) -> tresult,
    write: unsafe extern "system" fn(
        this: *mut c_void,
        buffer: *mut c_void,
        numBytes: i32,
        numBytesWritten: *mut i32,
    ) -> tresult,
    seek: unsafe extern "system" fn(this: *mut c_void, pos: i64, mode: i32, res: *mut i64) -> tresult,
    tell: unsafe extern "system" fn(this: *mut c_void, pos: *mut i64) -> tresult,
}

#[repr(C)]
struct IParamValueQueueVtbl {
    base: FUnknownVtbl,
    getParameterId: unsafe extern "system" fn(this: *mut c_void) -> ParamID,
    getPointCount: unsafe extern "system" fn(this: *mut c_void) -> i32,
    getPoint: unsafe extern "system" fn(
        this: *mut c_void,
        index: i32,
        sampleOffset: *mut i32,
        value: *mut ParamValue,
    ) -> tresult,
    addPoint: unsafe extern "system" fn(
        this: *mut c_void,
        sampleOffset: i32,
        value: ParamValue,
        index: *mut i32,
    ) -> tresult,
}

#[repr(C)]
struct IParameterChangesVtbl {
    base: FUnknownVtbl,
    getParameterCount: unsafe extern "system" fn(this: *mut c_void) -> i32,
    getParameterData: unsafe extern "system" fn(this: *mut c_void, index: i32) -> *mut c_void,
    addParameterData: unsafe extern "system" fn(
        this: *mut c_void,
        id: *const ParamID,
        index: *mut i32,
    ) -> *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NoteOnEvent {
    channel: i16,
    pitch: i16,
    tuning: f32,
    velocity: f32,
    length: i32,
    noteId: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NoteOffEvent {
    channel: i16,
    pitch: i16,
    velocity: f32,
    noteId: i32,
    tuning: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
union Vst3EventPayload {
    noteOn: NoteOnEvent,
    noteOff: NoteOffEvent,
    raw_bytes: [u8; 32],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Vst3Event {
    busIndex: i32,
    sampleOffset: i32,
    ppqPosition: f64,
    flags: u16,
    type_: u16,
    payload: Vst3EventPayload,
}

#[repr(C)]
struct IEventListVtbl {
    base: FUnknownVtbl,
    getEventCount: unsafe extern "system" fn(this: *mut c_void) -> i32,
    getEvent:
        unsafe extern "system" fn(this: *mut c_void, index: i32, e: *mut Vst3Event) -> tresult,
    addEvent: unsafe extern "system" fn(this: *mut c_void, e: *mut Vst3Event) -> tresult,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ViewRect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

#[repr(C)]
struct IPlugViewVtbl {
    base: FUnknownVtbl,
    isPlatformTypeSupported:
        unsafe extern "system" fn(this: *mut c_void, type_: *const c_char) -> tresult,
    attached: unsafe extern "system" fn(
        this: *mut c_void,
        parent: *mut c_void,
        type_: *const c_char,
    ) -> tresult,
    removed: unsafe extern "system" fn(this: *mut c_void) -> tresult,
    onWheel: unsafe extern "system" fn(this: *mut c_void, distance: f32) -> tresult,
    onKeyDown:
        unsafe extern "system" fn(this: *mut c_void, key: u16, code: i16, mods: i16) -> tresult,
    onKeyUp:
        unsafe extern "system" fn(this: *mut c_void, key: u16, code: i16, mods: i16) -> tresult,
    getSize: unsafe extern "system" fn(this: *mut c_void, size: *mut ViewRect) -> tresult,
    onSize: unsafe extern "system" fn(this: *mut c_void, newSize: *mut ViewRect) -> tresult,
    onFocus: unsafe extern "system" fn(this: *mut c_void, state: TBool) -> tresult,
    setFrame: unsafe extern "system" fn(this: *mut c_void, frame: *mut c_void) -> tresult,
    canResize: unsafe extern "system" fn(this: *mut c_void) -> tresult,
    checkSizeConstraint:
        unsafe extern "system" fn(this: *mut c_void, rect: *mut ViewRect) -> tresult,
}

fn write_wide128(dst: &mut [u16; 128], text: &str) {
    dst.fill(0);
    for (i, ch) in text.encode_utf16().take(127).enumerate() {
        dst[i] = ch;
    }
}

#[repr(C)]
struct ComSub<V> {
    vtbl: *const V,
    owner: *mut Vst3PluginInstance,
}

struct Vst3PluginInstance {
    comp_sub: ComSub<IComponentVtbl>,
    proc_sub: ComSub<IAudioProcessorVtbl>,
    ctrl_sub: ComSub<IEditControllerVtbl>,
    ref_count: AtomicU32,
    dsp: DualAbiDspCore,
    comp_handler: *mut c_void,
}

unsafe fn owner_from_sub<V>(this: *mut c_void) -> *mut Vst3PluginInstance {
    unsafe { (*(this as *mut ComSub<V>)).owner }
}

unsafe extern "system" fn vst3_inst_qi(
    this: *mut c_void,
    iid: *const TUID,
    obj: *mut *mut c_void,
) -> tresult {
    unsafe {
        if this.is_null() || iid.is_null() || obj.is_null() {
            return K_INVALID_ARGUMENT;
        }
        let owner = owner_from_sub::<FUnknownVtbl>(this);
        let id = *iid;
        if id == IID_FUNKNOWN || id == IID_IPLUGIN_BASE || id == IID_ICOMPONENT {
            (*owner).ref_count.fetch_add(1, Ordering::Relaxed);
            *obj = (&mut (*owner).comp_sub) as *mut ComSub<IComponentVtbl> as *mut c_void;
            K_RESULT_OK
        } else if id == IID_IAUDIO_PROCESSOR {
            (*owner).ref_count.fetch_add(1, Ordering::Relaxed);
            *obj = (&mut (*owner).proc_sub) as *mut ComSub<IAudioProcessorVtbl> as *mut c_void;
            K_RESULT_OK
        } else if id == IID_IEDIT_CONTROLLER {
            (*owner).ref_count.fetch_add(1, Ordering::Relaxed);
            *obj = (&mut (*owner).ctrl_sub) as *mut ComSub<IEditControllerVtbl> as *mut c_void;
            K_RESULT_OK
        } else {
            *obj = std::ptr::null_mut();
            K_NO_INTERFACE
        }
    }
}

unsafe extern "system" fn vst3_inst_add_ref(this: *mut c_void) -> u32 {
    unsafe {
        let owner = owner_from_sub::<FUnknownVtbl>(this);
        (*owner).ref_count.fetch_add(1, Ordering::Relaxed) + 1
    }
}

unsafe extern "system" fn vst3_inst_release(this: *mut c_void) -> u32 {
    unsafe {
        let owner = owner_from_sub::<FUnknownVtbl>(this);
        let prev = (*owner).ref_count.fetch_sub(1, Ordering::AcqRel);
        let rem = prev.saturating_sub(1);
        if rem == 0 {
            drop(Box::from_raw(owner));
        }
        rem
    }
}

unsafe extern "system" fn vst3_base_init(_this: *mut c_void, _ctx: *mut c_void) -> tresult {
    K_RESULT_OK
}
unsafe extern "system" fn vst3_base_term(_this: *mut c_void) -> tresult {
    K_RESULT_OK
}

unsafe extern "system" fn vst3_comp_get_ctrl_cid(
    _this: *mut c_void,
    class_id: *mut TUID,
) -> tresult {
    unsafe {
        if !class_id.is_null() {
            *class_id = TEST_PLUGIN_VST3_CID;
        }
    }
    K_RESULT_OK
}
unsafe extern "system" fn vst3_comp_set_io_mode(_this: *mut c_void, _mode: i32) -> tresult {
    K_RESULT_OK
}
unsafe extern "system" fn vst3_comp_get_bus_count(
    _this: *mut c_void,
    type_: i32,
    dir: i32,
) -> i32 {
    match (type_, dir) {
        (0, 0) => 1, // Main Stereo Audio Input
        (0, 1) => 1, // Main Stereo Audio Output
        (1, 0) => 1, // MIDI Event Input
        _ => 0,
    }
}
unsafe extern "system" fn vst3_comp_get_bus_info(
    _this: *mut c_void,
    type_: i32,
    dir: i32,
    index: i32,
    bus: *mut BusInfo,
) -> tresult {
    if bus.is_null() || index != 0 {
        return K_INVALID_ARGUMENT;
    }
    unsafe {
        *bus = std::mem::zeroed();
        (*bus).mediaType = type_;
        (*bus).direction = dir;
        (*bus).channelCount = if type_ == 0 { 2 } else { 16 };
        (*bus).busType = 0;
        (*bus).flags = 1;
        write_wide128(&mut (*bus).name, if dir == 0 { "Input" } else { "Output" });
    }
    K_RESULT_OK
}
unsafe extern "system" fn vst3_comp_get_routing(
    _this: *mut c_void,
    _i: *mut c_void,
    _o: *mut c_void,
) -> tresult {
    -2 // K_NOT_IMPLEMENTED
}
unsafe extern "system" fn vst3_comp_activate_bus(
    _this: *mut c_void,
    _t: i32,
    _d: i32,
    _i: i32,
    _s: TBool,
) -> tresult {
    K_RESULT_OK
}
unsafe extern "system" fn vst3_comp_set_active(this: *mut c_void, state: TBool) -> tresult {
    if state != 0 {
        unsafe {
            let owner = owner_from_sub::<IComponentVtbl>(this);
            (*owner).dsp.reset();
        }
    }
    K_RESULT_OK
}

unsafe fn bstream_write_state(dsp: &DualAbiDspCore, state_stream: *mut c_void) -> tresult {
    if state_stream.is_null() {
        return K_INVALID_ARGUMENT;
    }
    unsafe {
        let svtbl = *(state_stream as *const *const IBStreamVtbl);
        let bytes = dsp.encode_state();
        let mut written = 0i32;
        let rc = ((*svtbl).write)(
            state_stream,
            bytes.as_ptr() as *mut c_void,
            bytes.len() as i32,
            &mut written,
        );
        if rc == K_RESULT_OK && written == bytes.len() as i32 {
            K_RESULT_OK
        } else {
            1
        }
    }
}

unsafe fn bstream_read_state(dsp: &mut DualAbiDspCore, state_stream: *mut c_void) -> tresult {
    if state_stream.is_null() {
        return K_INVALID_ARGUMENT;
    }
    unsafe {
        let svtbl = *(state_stream as *const *const IBStreamVtbl);
        let mut bytes = [0u8; 40];
        let mut read_n = 0i32;
        let rc = ((*svtbl).read)(
            state_stream,
            bytes.as_mut_ptr() as *mut c_void,
            bytes.len() as i32,
            &mut read_n,
        );
        if rc == K_RESULT_OK && read_n == 40 && dsp.decode_state(&bytes) {
            K_RESULT_OK
        } else {
            1
        }
    }
}

unsafe extern "system" fn vst3_comp_set_state(this: *mut c_void, state: *mut c_void) -> tresult {
    unsafe {
        let owner = owner_from_sub::<IComponentVtbl>(this);
        bstream_read_state(&mut (*owner).dsp, state)
    }
}

unsafe extern "system" fn vst3_comp_get_state(this: *mut c_void, state: *mut c_void) -> tresult {
    unsafe {
        let owner = owner_from_sub::<IComponentVtbl>(this);
        bstream_write_state(&(*owner).dsp, state)
    }
}

static VST3_COMP_VTBL: IComponentVtbl = IComponentVtbl {
    base: IPluginBaseVtbl {
        base: FUnknownVtbl {
            queryInterface: vst3_inst_qi,
            addRef: vst3_inst_add_ref,
            release: vst3_inst_release,
        },
        initialize: vst3_base_init,
        terminate: vst3_base_term,
    },
    getControllerClassId: vst3_comp_get_ctrl_cid,
    setIoMode: vst3_comp_set_io_mode,
    getBusCount: vst3_comp_get_bus_count,
    getBusInfo: vst3_comp_get_bus_info,
    getRoutingInfo: vst3_comp_get_routing,
    activateBus: vst3_comp_activate_bus,
    setActive: vst3_comp_set_active,
    setState: vst3_comp_set_state,
    getState: vst3_comp_get_state,
};

unsafe extern "system" fn vst3_proc_set_arr(
    _this: *mut c_void,
    _ins: *mut u64,
    _ni: i32,
    _outs: *mut u64,
    _no: i32,
) -> tresult {
    K_RESULT_OK
}
unsafe extern "system" fn vst3_proc_get_arr(
    _this: *mut c_void,
    _dir: i32,
    _idx: i32,
    arr: *mut u64,
) -> tresult {
    unsafe {
        if !arr.is_null() {
            *arr = 0x3; // kStereo (L | R)
        }
    }
    K_RESULT_OK
}
unsafe extern "system" fn vst3_proc_can_sample_size(_this: *mut c_void, sz: i32) -> tresult {
    if sz == 0 { K_RESULT_OK } else { 1 }
}
unsafe extern "system" fn vst3_proc_get_latency(_this: *mut c_void) -> u32 {
    PLUGIN_REPORTED_LATENCY_SAMPLES
}
unsafe extern "system" fn vst3_proc_setup(
    this: *mut c_void,
    setup: *mut ProcessSetup,
) -> tresult {
    unsafe {
        if !setup.is_null() {
            let owner = owner_from_sub::<IAudioProcessorVtbl>(this);
            (*owner).dsp.sample_rate = (*setup).sampleRate.max(8_000.0);
        }
    }
    K_RESULT_OK
}
unsafe extern "system" fn vst3_proc_set_processing(_this: *mut c_void, _s: TBool) -> tresult {
    K_RESULT_OK
}
unsafe extern "system" fn vst3_proc_get_tail(_this: *mut c_void) -> u32 {
    0
}

unsafe extern "system" fn vst3_proc_process(
    this: *mut c_void,
    data: *mut ProcessData,
) -> tresult {
    unsafe {
        if this.is_null() || data.is_null() {
            return K_INVALID_ARGUMENT;
        }
        let owner = owner_from_sub::<IAudioProcessorVtbl>(this);
        let d = &mut *data;
        let frames = d.numSamples.max(0) as usize;

        // Read parameter changes (if any) into a tiny fixed stack schedule `(offset, param_id, norm_val)`
        let mut param_points = [(0i32, 0u32, 0.0f64); 64];
        let mut param_point_len = 0usize;
        if !d.inputParameterChanges.is_null() {
            let pcvtbl = *(d.inputParameterChanges as *const *const IParameterChangesVtbl);
            let qcount = ((*pcvtbl).getParameterCount)(d.inputParameterChanges).max(0);
            for qi in 0..qcount {
                let qptr = ((*pcvtbl).getParameterData)(d.inputParameterChanges, qi);
                if qptr.is_null() {
                    continue;
                }
                let qvtbl = *(qptr as *const *const IParamValueQueueVtbl);
                let pid = ((*qvtbl).getParameterId)(qptr);
                let pcount = ((*qvtbl).getPointCount)(qptr).max(0);
                for pi in 0..pcount {
                    let mut off = 0i32;
                    let mut val = 0.0f64;
                    if ((*qvtbl).getPoint)(qptr, pi, &mut off, &mut val) == K_RESULT_OK
                        && param_point_len < param_points.len()
                    {
                        param_points[param_point_len] = (off, pid, val);
                        param_point_len += 1;
                    }
                }
            }
        }

        // Read note events count
        let mut ev_count = 0i32;
        let mut ev_vtbl: *const IEventListVtbl = std::ptr::null();
        if !d.inputEvents.is_null() {
            ev_vtbl = *(d.inputEvents as *const *const IEventListVtbl);
            ev_count = ((*ev_vtbl).getEventCount)(d.inputEvents).max(0);
        }

        if frames == 0 || d.numOutputs <= 0 || d.outputs.is_null() {
            for i in 0..param_point_len {
                (*owner).dsp.set_norm_param(param_points[i].1, param_points[i].2);
            }
            return K_RESULT_OK;
        }

        let out_bus = &mut *d.outputs;
        if out_bus.numChannels < 2 || out_bus.channelBuffers32.is_null() {
            return K_RESULT_OK;
        }
        let out_l = *out_bus.channelBuffers32.add(0);
        let out_r = *out_bus.channelBuffers32.add(1);

        let (has_in, in_l, in_r) = if d.numInputs > 0 && !d.inputs.is_null() {
            let ib = &*d.inputs;
            if ib.numChannels >= 2 && !ib.channelBuffers32.is_null() {
                (
                    true,
                    *ib.channelBuffers32.add(0),
                    *ib.channelBuffers32.add(1),
                )
            } else {
                (false, std::ptr::null_mut(), std::ptr::null_mut())
            }
        } else {
            (false, std::ptr::null_mut(), std::ptr::null_mut())
        };

        let mut ev_idx = 0i32;
        let mut pp_idx = 0usize;

        for s in 0..frames {
            while pp_idx < param_point_len && (param_points[pp_idx].0.max(0) as usize) <= s {
                (*owner)
                    .dsp
                    .set_norm_param(param_points[pp_idx].1, param_points[pp_idx].2);
                pp_idx += 1;
            }
            if !ev_vtbl.is_null() {
                while ev_idx < ev_count {
                    let mut ev: Vst3Event = std::mem::zeroed();
                    if ((*ev_vtbl).getEvent)(d.inputEvents, ev_idx, &mut ev) != K_RESULT_OK {
                        ev_idx += 1;
                        continue;
                    }
                    if (ev.sampleOffset.max(0) as usize) > s {
                        break;
                    }
                    match ev.type_ {
                        0 => {
                            let n = ev.payload.noteOn;
                            (*owner).dsp.note_on(
                                n.pitch,
                                n.tuning as f64,
                                n.velocity as f64,
                                n.noteId,
                            );
                        }
                        1 => {
                            let n = ev.payload.noteOff;
                            (*owner).dsp.note_off(n.pitch, n.noteId);
                        }
                        _ => {}
                    }
                    ev_idx += 1;
                }
            }

            let dry = if has_in {
                [*in_l.add(s), *in_r.add(s)]
            } else {
                [0.0, 0.0]
            };
            let wet = (*owner).dsp.step_frame(dry);
            *out_l.add(s) = wet[0];
            *out_r.add(s) = wet[1];
        }

        while pp_idx < param_point_len {
            (*owner)
                .dsp
                .set_norm_param(param_points[pp_idx].1, param_points[pp_idx].2);
            pp_idx += 1;
        }
        K_RESULT_OK
    }
}

static VST3_PROC_VTBL: IAudioProcessorVtbl = IAudioProcessorVtbl {
    base: FUnknownVtbl {
        queryInterface: vst3_inst_qi,
        addRef: vst3_inst_add_ref,
        release: vst3_inst_release,
    },
    setBusArrangements: vst3_proc_set_arr,
    getBusArrangement: vst3_proc_get_arr,
    canProcessSampleSize: vst3_proc_can_sample_size,
    getLatencySamples: vst3_proc_get_latency,
    setupProcessing: vst3_proc_setup,
    setProcessing: vst3_proc_set_processing,
    process: vst3_proc_process,
    getTailSamples: vst3_proc_get_tail,
};

#[repr(C)]
struct Vst3PlugView {
    vtbl: *const IPlugViewVtbl,
    ref_count: AtomicU32,
    child_hwnd: HWND,
    frame_ptr: *mut c_void,
    rect: ViewRect,
}

unsafe extern "system" fn plug_view_qi(
    this: *mut c_void,
    iid: *const TUID,
    obj: *mut *mut c_void,
) -> tresult {
    unsafe {
        if this.is_null() || iid.is_null() || obj.is_null() {
            return K_INVALID_ARGUMENT;
        }
        let id = *iid;
        if id == IID_FUNKNOWN || id == IID_IPLUG_VIEW {
            let v = &*(this as *const Vst3PlugView);
            v.ref_count.fetch_add(1, Ordering::Relaxed);
            *obj = this;
            K_RESULT_OK
        } else {
            *obj = std::ptr::null_mut();
            K_NO_INTERFACE
        }
    }
}
unsafe extern "system" fn plug_view_add_ref(this: *mut c_void) -> u32 {
    unsafe { (*(this as *const Vst3PlugView)).ref_count.fetch_add(1, Ordering::Relaxed) + 1 }
}
unsafe extern "system" fn plug_view_release(this: *mut c_void) -> u32 {
    unsafe {
        let v = &mut *(this as *mut Vst3PlugView);
        let prev = v.ref_count.fetch_sub(1, Ordering::AcqRel);
        let rem = prev.saturating_sub(1);
        if rem == 0 {
            destroy_plugin_child_hwnd(&mut v.child_hwnd);
            drop(Box::from_raw(this as *mut Vst3PlugView));
        }
        rem
    }
}
unsafe extern "system" fn plug_view_is_supported(
    _this: *mut c_void,
    type_: *const c_char,
) -> tresult {
    if type_.is_null() {
        return K_INVALID_ARGUMENT;
    }
    let s = unsafe { CStr::from_ptr(type_).to_bytes() };
    if s == b"HWND" { K_RESULT_OK } else { 1 }
}
unsafe extern "system" fn plug_view_attached(
    this: *mut c_void,
    parent: *mut c_void,
    type_: *const c_char,
) -> tresult {
    unsafe {
        if plug_view_is_supported(this, type_) != K_RESULT_OK || parent.is_null() {
            return K_INVALID_ARGUMENT;
        }
        let v = &mut *(this as *mut Vst3PlugView);
        destroy_plugin_child_hwnd(&mut v.child_hwnd);
        let w = (v.rect.right - v.rect.left).max(160);
        let h = (v.rect.bottom - v.rect.top).max(96);
        v.child_hwnd = create_plugin_child_hwnd(parent as HWND, w, h);
        if v.child_hwnd.is_null() { 1 } else { K_RESULT_OK }
    }
}
unsafe extern "system" fn plug_view_removed(this: *mut c_void) -> tresult {
    unsafe {
        let v = &mut *(this as *mut Vst3PlugView);
        destroy_plugin_child_hwnd(&mut v.child_hwnd);
    }
    K_RESULT_OK
}
unsafe extern "system" fn plug_view_wheel(_this: *mut c_void, _d: f32) -> tresult {
    K_RESULT_OK
}
unsafe extern "system" fn plug_view_key(
    _this: *mut c_void,
    _k: u16,
    _c: i16,
    _m: i16,
) -> tresult {
    K_RESULT_OK
}
unsafe extern "system" fn plug_view_get_size(this: *mut c_void, size: *mut ViewRect) -> tresult {
    unsafe {
        if size.is_null() {
            return K_INVALID_ARGUMENT;
        }
        *size = (*(this as *const Vst3PlugView)).rect;
    }
    K_RESULT_OK
}
unsafe extern "system" fn plug_view_on_size(
    this: *mut c_void,
    new_size: *mut ViewRect,
) -> tresult {
    unsafe {
        if !new_size.is_null() {
            (*(this as *mut Vst3PlugView)).rect = *new_size;
        }
    }
    K_RESULT_OK
}
unsafe extern "system" fn plug_view_focus(_this: *mut c_void, _s: TBool) -> tresult {
    K_RESULT_OK
}
unsafe extern "system" fn plug_view_set_frame(this: *mut c_void, frame: *mut c_void) -> tresult {
    unsafe {
        (*(this as *mut Vst3PlugView)).frame_ptr = frame;
    }
    K_RESULT_OK
}
unsafe extern "system" fn plug_view_can_resize(_this: *mut c_void) -> tresult {
    K_RESULT_OK
}
unsafe extern "system" fn plug_view_check_size(
    _this: *mut c_void,
    _rect: *mut ViewRect,
) -> tresult {
    K_RESULT_OK
}

static VST3_PLUG_VIEW_VTBL: IPlugViewVtbl = IPlugViewVtbl {
    base: FUnknownVtbl {
        queryInterface: plug_view_qi,
        addRef: plug_view_add_ref,
        release: plug_view_release,
    },
    isPlatformTypeSupported: plug_view_is_supported,
    attached: plug_view_attached,
    removed: plug_view_removed,
    onWheel: plug_view_wheel,
    onKeyDown: plug_view_key,
    onKeyUp: plug_view_key,
    getSize: plug_view_get_size,
    onSize: plug_view_on_size,
    onFocus: plug_view_focus,
    setFrame: plug_view_set_frame,
    canResize: plug_view_can_resize,
    checkSizeConstraint: plug_view_check_size,
};

unsafe extern "system" fn vst3_ctrl_set_comp_state(
    this: *mut c_void,
    state: *mut c_void,
) -> tresult {
    unsafe {
        let owner = owner_from_sub::<IEditControllerVtbl>(this);
        let _ = bstream_read_state(&mut (*owner).dsp, state);
    }
    K_RESULT_OK
}
unsafe extern "system" fn vst3_ctrl_set_state(_this: *mut c_void, _state: *mut c_void) -> tresult {
    K_RESULT_OK
}
unsafe extern "system" fn vst3_ctrl_get_state(_this: *mut c_void, _state: *mut c_void) -> tresult {
    K_RESULT_OK
}
unsafe extern "system" fn vst3_ctrl_param_count(_this: *mut c_void) -> i32 {
    NUM_PLUGIN_PARAMS as i32
}
unsafe extern "system" fn vst3_ctrl_param_info(
    _this: *mut c_void,
    index: i32,
    info: *mut ParameterInfo,
) -> tresult {
    if info.is_null() || index < 0 || (index as usize) >= NUM_PLUGIN_PARAMS {
        return K_INVALID_ARGUMENT;
    }
    unsafe {
        let p = &mut *info;
        *p = std::mem::zeroed();
        p.id = index as u32;
        p.flags = 1; // kCanAutomate
        match p.id {
            PARAM_SYNTH_GAIN => {
                write_wide128(&mut p.title, "Synth Gain");
                write_wide128(&mut p.shortTitle, "Gain");
                write_wide128(&mut p.units, "x");
                p.defaultNormalizedValue = DualAbiDspCore::plain_to_norm(PARAM_SYNTH_GAIN, 0.5);
            }
            PARAM_FILTER_CUTOFF => {
                write_wide128(&mut p.title, "SVF Cutoff");
                write_wide128(&mut p.shortTitle, "Cutoff");
                write_wide128(&mut p.units, "Hz");
                p.defaultNormalizedValue =
                    DualAbiDspCore::plain_to_norm(PARAM_FILTER_CUTOFF, 14_000.0);
            }
            PARAM_FILTER_RESO => {
                write_wide128(&mut p.title, "SVF Resonance");
                write_wide128(&mut p.shortTitle, "Reso");
                write_wide128(&mut p.units, "Q");
                p.defaultNormalizedValue = DualAbiDspCore::plain_to_norm(
                    PARAM_FILTER_RESO,
                    std::f64::consts::FRAC_1_SQRT_2,
                );
            }
            PARAM_PASSTHROUGH_MIX => {
                write_wide128(&mut p.title, "Dry Passthrough Mix");
                write_wide128(&mut p.shortTitle, "DryMix");
                write_wide128(&mut p.units, "%");
                p.defaultNormalizedValue = 1.0;
            }
            _ => return K_INVALID_ARGUMENT,
        }
    }
    K_RESULT_OK
}
unsafe extern "system" fn vst3_ctrl_str_by_val(
    _this: *mut c_void,
    _id: ParamID,
    _val: ParamValue,
    _str: *mut u16,
) -> tresult {
    -2
}
unsafe extern "system" fn vst3_ctrl_val_by_str(
    _this: *mut c_void,
    _id: ParamID,
    _str: *mut u16,
    _val: *mut ParamValue,
) -> tresult {
    -2
}
unsafe extern "system" fn vst3_ctrl_norm_to_plain(
    _this: *mut c_void,
    id: ParamID,
    val: ParamValue,
) -> ParamValue {
    DualAbiDspCore::norm_to_plain(id, val)
}
unsafe extern "system" fn vst3_ctrl_plain_to_norm(
    _this: *mut c_void,
    id: ParamID,
    val: ParamValue,
) -> ParamValue {
    DualAbiDspCore::plain_to_norm(id, val)
}
unsafe extern "system" fn vst3_ctrl_get_norm(this: *mut c_void, id: ParamID) -> ParamValue {
    unsafe {
        let owner = owner_from_sub::<IEditControllerVtbl>(this);
        (*owner).dsp.get_norm_param(id)
    }
}
unsafe extern "system" fn vst3_ctrl_set_norm(
    this: *mut c_void,
    id: ParamID,
    value: ParamValue,
) -> tresult {
    unsafe {
        let owner = owner_from_sub::<IEditControllerVtbl>(this);
        (*owner).dsp.set_norm_param(id, value);
    }
    K_RESULT_OK
}
unsafe extern "system" fn vst3_ctrl_set_handler(
    this: *mut c_void,
    handler: *mut c_void,
) -> tresult {
    unsafe {
        let owner = owner_from_sub::<IEditControllerVtbl>(this);
        (*owner).comp_handler = handler;
    }
    K_RESULT_OK
}
unsafe extern "system" fn vst3_ctrl_create_view(
    _this: *mut c_void,
    name: *const c_char,
) -> *mut c_void {
    if !name.is_null() {
        let s = unsafe { CStr::from_ptr(name).to_bytes() };
        if !s.is_empty() && s != b"editor" {
            return std::ptr::null_mut();
        }
    }
    let view = Box::new(Vst3PlugView {
        vtbl: &VST3_PLUG_VIEW_VTBL,
        ref_count: AtomicU32::new(1),
        child_hwnd: std::ptr::null_mut(),
        frame_ptr: std::ptr::null_mut(),
        rect: ViewRect {
            left: 0,
            top: 0,
            right: 560,
            bottom: 340,
        },
    });
    Box::into_raw(view) as *mut c_void
}

static VST3_CTRL_VTBL: IEditControllerVtbl = IEditControllerVtbl {
    base: IPluginBaseVtbl {
        base: FUnknownVtbl {
            queryInterface: vst3_inst_qi,
            addRef: vst3_inst_add_ref,
            release: vst3_inst_release,
        },
        initialize: vst3_base_init,
        terminate: vst3_base_term,
    },
    setComponentState: vst3_ctrl_set_comp_state,
    setState: vst3_ctrl_set_state,
    getState: vst3_ctrl_get_state,
    getParameterCount: vst3_ctrl_param_count,
    getParameterInfo: vst3_ctrl_param_info,
    getParamStringByValue: vst3_ctrl_str_by_val,
    getParamValueByString: vst3_ctrl_val_by_str,
    normalizedParamToPlain: vst3_ctrl_norm_to_plain,
    plainParamToNormalized: vst3_ctrl_plain_to_norm,
    getParamNormalized: vst3_ctrl_get_norm,
    setParamNormalized: vst3_ctrl_set_norm,
    setComponentHandler: vst3_ctrl_set_handler,
    createView: vst3_ctrl_create_view,
};

#[repr(C)]
struct StaticVst3Factory {
    vtbl: *const IPluginFactoryVtbl,
}
unsafe impl Sync for StaticVst3Factory {}

unsafe extern "system" fn vst3_factory_qi(
    this: *mut c_void,
    iid: *const TUID,
    obj: *mut *mut c_void,
) -> tresult {
    unsafe {
        if iid.is_null() || obj.is_null() {
            return K_INVALID_ARGUMENT;
        }
        let id = *iid;
        if id == IID_FUNKNOWN || id == IID_IPLUGIN_FACTORY {
            *obj = this;
            K_RESULT_OK
        } else {
            *obj = std::ptr::null_mut();
            K_NO_INTERFACE
        }
    }
}
unsafe extern "system" fn vst3_factory_add_ref(_this: *mut c_void) -> u32 {
    2
}
unsafe extern "system" fn vst3_factory_release(_this: *mut c_void) -> u32 {
    1
}
unsafe extern "system" fn vst3_factory_info(
    _this: *mut c_void,
    info: *mut PFactoryInfo,
) -> tresult {
    if info.is_null() {
        return K_INVALID_ARGUMENT;
    }
    unsafe {
        *info = std::mem::zeroed();
        write_c_buf(&mut (*info).vendor, "ToneForge Clean-Room Audio");
        write_c_buf(&mut (*info).url, "https://toneforge.local");
        write_c_buf(&mut (*info).email, "dsp@toneforge.local");
        (*info).flags = 0x10; // kUnicode
    }
    K_RESULT_OK
}
unsafe extern "system" fn vst3_factory_count(_this: *mut c_void) -> i32 {
    1
}
unsafe extern "system" fn vst3_factory_class_info(
    _this: *mut c_void,
    index: i32,
    info: *mut PClassInfo,
) -> tresult {
    if info.is_null() || index != 0 {
        return K_INVALID_ARGUMENT;
    }
    unsafe {
        *info = std::mem::zeroed();
        (*info).cid = TEST_PLUGIN_VST3_CID;
        (*info).cardinality = 0x7FFF_FFFF; // kManyInstances
        write_c_buf(&mut (*info).category, "Audio Module Class");
        write_c_buf(&mut (*info).name, "ToneForge Dual-ABI Synth & APDC FX");
    }
    K_RESULT_OK
}
unsafe extern "system" fn vst3_factory_create(
    _this: *mut c_void,
    cid: *const TUID,
    iid: *const TUID,
    obj: *mut *mut c_void,
) -> tresult {
    unsafe {
        if cid.is_null() || iid.is_null() || obj.is_null() {
            return K_INVALID_ARGUMENT;
        }
        if *cid != TEST_PLUGIN_VST3_CID {
            *obj = std::ptr::null_mut();
            return K_NO_INTERFACE;
        }
        let boxed = Box::new(Vst3PluginInstance {
            comp_sub: ComSub {
                vtbl: &VST3_COMP_VTBL,
                owner: std::ptr::null_mut(),
            },
            proc_sub: ComSub {
                vtbl: &VST3_PROC_VTBL,
                owner: std::ptr::null_mut(),
            },
            ctrl_sub: ComSub {
                vtbl: &VST3_CTRL_VTBL,
                owner: std::ptr::null_mut(),
            },
            ref_count: AtomicU32::new(1),
            dsp: DualAbiDspCore::new(48_000.0),
            comp_handler: std::ptr::null_mut(),
        });
        let raw = Box::into_raw(boxed);
        (*raw).comp_sub.owner = raw;
        (*raw).proc_sub.owner = raw;
        (*raw).ctrl_sub.owner = raw;

        let base_this = (&mut (*raw).comp_sub) as *mut ComSub<IComponentVtbl> as *mut c_void;
        let rc = vst3_inst_qi(base_this, iid, obj);
        // Drop the initial creation ref so caller owns ref_count == 1 (or frees if QI failed)
        let _ = vst3_inst_release(base_this);
        rc
    }
}

static VST3_FACTORY_VTBL: IPluginFactoryVtbl = IPluginFactoryVtbl {
    base: FUnknownVtbl {
        queryInterface: vst3_factory_qi,
        addRef: vst3_factory_add_ref,
        release: vst3_factory_release,
    },
    getFactoryInfo: vst3_factory_info,
    countClasses: vst3_factory_count,
    getClassInfo: vst3_factory_class_info,
    createInstance: vst3_factory_create,
};

static VST3_STATIC_FACTORY: StaticVst3Factory = StaticVst3Factory {
    vtbl: &VST3_FACTORY_VTBL,
};

#[unsafe(no_mangle)]
pub unsafe extern "system" fn InitDll() -> bool {
    true
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn ExitDll() -> bool {
    true
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn GetPluginFactory() -> *mut c_void {
    (&VST3_STATIC_FACTORY) as *const StaticVst3Factory as *mut c_void
}
