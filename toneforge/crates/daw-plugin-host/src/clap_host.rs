//! `clap_host`: Production Real-Time **CLAP 1.2** Host (`ClapHostedPluginNode` implementing
//! `UnifiedDspNode` with `0` RT heap allocations, `0` RT locks, sample-accurate MIDI/Note/Param
//! dispatch, binary state save/restore, and Win32 `HWND` `clap.gui` editor embedding).

use crate::clap_abi::{
    CLAP_BEATTIME_FACTOR, CLAP_CORE_EVENT_SPACE_ID, CLAP_EVENT_MIDI, CLAP_EVENT_NOTE_OFF,
    CLAP_EVENT_NOTE_ON, CLAP_EVENT_TRANSPORT, CLAP_SECTIME_FACTOR, CLAP_TRANSPORT_HAS_BEATS_TIMELINE,
    CLAP_TRANSPORT_HAS_SECONDS_TIMELINE, CLAP_TRANSPORT_HAS_TEMPO, CLAP_TRANSPORT_HAS_TIME_SIGNATURE,
    CLAP_TRANSPORT_IS_PLAYING, clap_event_transport,
    CLAP_EVENT_PARAM_VALUE, CLAP_EXT_AUDIO_PORTS, CLAP_EXT_GUI, CLAP_EXT_LATENCY,
    CLAP_EXT_NOTE_PORTS, CLAP_EXT_PARAMS, CLAP_EXT_STATE, CLAP_PLUGIN_FACTORY_ID,
    CLAP_VERSION_1_2_0, CLAP_WINDOW_API_WIN32, clap_audio_buffer, clap_event_header,
    clap_event_midi, clap_event_note, clap_event_param_value, clap_host, clap_id,
    clap_input_events, clap_istream, clap_ostream, clap_output_events, clap_param_info,
    clap_plugin, clap_plugin_audio_ports, clap_plugin_descriptor, clap_plugin_entry,
    clap_plugin_factory, clap_plugin_gui, clap_plugin_latency, clap_plugin_note_ports,
    clap_plugin_params, clap_plugin_state, clap_process, clap_window,
};
use crate::hwnd_window::NativePluginWindow;
use daw_core_graph::{
    MAX_BLOCK_SIZE, ProcessContext, SequencerVoiceEvent, StereoFrame, UnifiedDspNode,
    VoiceEventKind,
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_void};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use windows_sys::Win32::Foundation::{FreeLibrary, HMODULE};
use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};

pub const MAX_CLAP_RT_EVENTS: usize = 512;
pub const MAX_CLAP_RT_PARAMS: usize = 128;

/// Parameter descriptor shared across CLAP 1.2 and VST3 3.7 hosted plugins.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HostedPluginParamInfo {
    pub id: u32,
    pub name: String,
    pub module: String,
    pub min_value: f64,
    pub max_value: f64,
    pub default_value: f64,
    pub current_value: f64,
}

/// Summary of a CLAP plugin descriptor discovered inside a `.clap` binary.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClapDescriptorSummary {
    pub id: String,
    pub name: String,
    pub vendor: String,
    pub version: String,
    pub description: String,
    pub is_instrument: bool,
    pub latency_samples: u32,
    pub num_params: u32,
}

#[derive(Clone, Copy)]
struct RtParamSlot {
    id: clap_id,
    min_value: f64,
    max_value: f64,
    current_plain: f64,
}

impl Default for RtParamSlot {
    fn default() -> Self {
        Self {
            id: 0,
            min_value: 0.0,
            max_value: 1.0,
            current_plain: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
union ClapEventSlot {
    header: clap_event_header,
    note: clap_event_note,
    param: clap_event_param_value,
    midi: clap_event_midi,
}

impl Default for ClapEventSlot {
    fn default() -> Self {
        Self {
            header: clap_event_header::default(),
        }
    }
}

#[repr(C)]
struct ClapRtArena {
    in_left: [f32; MAX_BLOCK_SIZE],
    in_right: [f32; MAX_BLOCK_SIZE],
    out_left: [f32; MAX_BLOCK_SIZE],
    out_right: [f32; MAX_BLOCK_SIZE],
    in_ptrs: [*mut f32; 2],
    out_ptrs: [*mut f32; 2],
    events: [ClapEventSlot; MAX_CLAP_RT_EVENTS],
    event_count: usize,
    out_param_edits: [clap_event_param_value; 64],
    out_param_count: usize,
    params: [RtParamSlot; MAX_CLAP_RT_PARAMS],
    param_count: usize,
}

impl ClapRtArena {
    fn new_boxed() -> Box<Self> {
        let mut boxed = Box::new(Self {
            in_left: [0.0; MAX_BLOCK_SIZE],
            in_right: [0.0; MAX_BLOCK_SIZE],
            out_left: [0.0; MAX_BLOCK_SIZE],
            out_right: [0.0; MAX_BLOCK_SIZE],
            in_ptrs: [std::ptr::null_mut(); 2],
            out_ptrs: [std::ptr::null_mut(); 2],
            events: [ClapEventSlot::default(); MAX_CLAP_RT_EVENTS],
            event_count: 0,
            out_param_edits: [clap_event_param_value::default(); 64],
            out_param_count: 0,
            params: [RtParamSlot::default(); MAX_CLAP_RT_PARAMS],
            param_count: 0,
        });
        boxed.in_ptrs[0] = boxed.in_left.as_mut_ptr();
        boxed.in_ptrs[1] = boxed.in_right.as_mut_ptr();
        boxed.out_ptrs[0] = boxed.out_left.as_mut_ptr();
        boxed.out_ptrs[1] = boxed.out_right.as_mut_ptr();
        boxed
    }

    #[inline]
    fn push_note_event(&mut self, type_: u16, sample_offset: u32, note_id: i32, channel: i16, key: i16, velocity: f64) {
        if self.event_count >= MAX_CLAP_RT_EVENTS {
            return;
        }
        self.events[self.event_count] = ClapEventSlot {
            note: clap_event_note {
                header: clap_event_header {
                    size: std::mem::size_of::<clap_event_note>() as u32,
                    time: sample_offset,
                    space_id: CLAP_CORE_EVENT_SPACE_ID,
                    type_,
                    flags: 0,
                },
                note_id,
                port_index: 0,
                channel,
                key,
                velocity,
            },
        };
        self.event_count += 1;
    }

    #[inline]
    fn push_param_event(&mut self, sample_offset: u32, param_id: clap_id, plain_value: f64) {
        if self.event_count >= MAX_CLAP_RT_EVENTS {
            return;
        }
        self.events[self.event_count] = ClapEventSlot {
            param: clap_event_param_value {
                header: clap_event_header {
                    size: std::mem::size_of::<clap_event_param_value>() as u32,
                    time: sample_offset,
                    space_id: CLAP_CORE_EVENT_SPACE_ID,
                    type_: CLAP_EVENT_PARAM_VALUE,
                    flags: 0,
                },
                param_id,
                cookie: std::ptr::null_mut(),
                note_id: -1,
                port_index: -1,
                channel: -1,
                key: -1,
                value: plain_value,
            },
        };
        self.event_count += 1;
    }

    #[inline]
    fn push_midi_event(&mut self, sample_offset: u32, status: u8, d1: u8, d2: u8) {
        if self.event_count >= MAX_CLAP_RT_EVENTS {
            return;
        }
        self.events[self.event_count] = ClapEventSlot {
            midi: clap_event_midi {
                header: clap_event_header {
                    size: std::mem::size_of::<clap_event_midi>() as u32,
                    time: sample_offset,
                    space_id: CLAP_CORE_EVENT_SPACE_ID,
                    type_: CLAP_EVENT_MIDI,
                    flags: 0,
                },
                port_index: 0,
                data: [status, d1, d2],
            },
        };
        self.event_count += 1;
    }

    #[inline]
    fn sort_events_by_time(&mut self) {
        // Insertion sort on pre-allocated slice (`<= 512`, typically `0..8` events, `0` alloc).
        let n = self.event_count;
        for i in 1..n {
            let cur = self.events[i];
            let cur_time = unsafe { cur.header.time };
            let mut j = i;
            while j > 0 {
                let prev_time = unsafe { self.events[j - 1].header.time };
                if prev_time <= cur_time {
                    break;
                }
                self.events[j] = self.events[j - 1];
                j -= 1;
            }
            self.events[j] = cur;
        }
    }
}

unsafe extern "C" fn clap_in_events_size(list: *const clap_input_events) -> u32 {
    unsafe {
        if list.is_null() || (*list).ctx.is_null() {
            return 0;
        }
        let arena = &*((*list).ctx as *const ClapRtArena);
        arena.event_count as u32
    }
}

unsafe extern "C" fn clap_in_events_get(
    list: *const clap_input_events,
    index: u32,
) -> *const clap_event_header {
    unsafe {
        if list.is_null() || (*list).ctx.is_null() {
            return std::ptr::null();
        }
        let arena = &*((*list).ctx as *const ClapRtArena);
        let idx = index as usize;
        if idx >= arena.event_count {
            return std::ptr::null();
        }
        &arena.events[idx].header as *const clap_event_header
    }
}

unsafe extern "C" fn clap_out_events_try_push(
    list: *const clap_output_events,
    event: *const clap_event_header,
) -> bool {
    unsafe {
        if list.is_null() || (*list).ctx.is_null() || event.is_null() {
            return false;
        }
        let arena = &mut *((*list).ctx as *mut ClapRtArena);
        if (*event).type_ == CLAP_EVENT_PARAM_VALUE
            && ((*event).size as usize) >= std::mem::size_of::<clap_event_param_value>()
            && arena.out_param_count < arena.out_param_edits.len()
        {
            let pev = *(event as *const clap_event_param_value);
            arena.out_param_edits[arena.out_param_count] = pev;
            arena.out_param_count += 1;
        }
        true
    }
}

static HOST_NAME: &[u8] = b"ToneForge Studio v3\0";
static HOST_VENDOR: &[u8] = b"ToneForge Clean-Room Audio\0";
static HOST_URL: &[u8] = b"https://toneforge.local\0";
static HOST_VERSION: &[u8] = b"3.0.0\0";

unsafe extern "C" fn host_get_extension(
    _host: *const clap_host,
    _extension_id: *const c_char,
) -> *const c_void {
    std::ptr::null()
}
unsafe extern "C" fn host_request_restart(_host: *const clap_host) {}
unsafe extern "C" fn host_request_process(_host: *const clap_host) {}
unsafe extern "C" fn host_request_callback(_host: *const clap_host) {}

#[repr(C)]
struct PinnedClapHost {
    raw: clap_host,
}

impl PinnedClapHost {
    fn new_boxed() -> Box<Self> {
        let mut b = Box::new(Self {
            raw: clap_host {
                clap_version: CLAP_VERSION_1_2_0,
                host_data: std::ptr::null_mut(),
                name: HOST_NAME.as_ptr() as *const c_char,
                vendor: HOST_VENDOR.as_ptr() as *const c_char,
                url: HOST_URL.as_ptr() as *const c_char,
                version: HOST_VERSION.as_ptr() as *const c_char,
                get_extension: Some(host_get_extension),
                request_restart: Some(host_request_restart),
                request_process: Some(host_request_process),
                request_callback: Some(host_request_callback),
            },
        });
        let ptr = (&mut *b) as *mut Self as *mut c_void;
        b.raw.host_data = ptr;
        b
    }
}

/// Non-RT global refcount table for `clap_entry.init` / `clap_entry.deinit` per canonical DLL path.
fn clap_module_refcounts() -> &'static Mutex<HashMap<PathBuf, usize>> {
    static MAP: OnceLock<Mutex<HashMap<PathBuf, usize>>> = OnceLock::new();
    MAP.get_or_init(|| Mutex::new(HashMap::new()))
}

unsafe fn cstr_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
}

fn wide_null_path(path: &Path) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    let mut v: Vec<u16> = path.as_os_str().encode_wide().collect();
    v.push(0);
    v
}

struct WriteStreamCtx {
    buf: Vec<u8>,
}

unsafe extern "C" fn clap_ostream_write(
    stream: *const clap_ostream,
    buffer: *const c_void,
    size: u64,
) -> i64 {
    unsafe {
        if stream.is_null() || (*stream).ctx.is_null() || (buffer.is_null() && size > 0) {
            return -1;
        }
        let ctx = &mut *((*stream).ctx as *mut WriteStreamCtx);
        let slice = std::slice::from_raw_parts(buffer as *const u8, size as usize);
        ctx.buf.extend_from_slice(slice);
        size as i64
    }
}

struct ReadStreamCtx<'a> {
    data: &'a [u8],
    cursor: usize,
}

unsafe extern "C" fn clap_istream_read(
    stream: *const clap_istream,
    buffer: *mut c_void,
    size: u64,
) -> i64 {
    unsafe {
        if stream.is_null() || (*stream).ctx.is_null() || (buffer.is_null() && size > 0) {
            return -1;
        }
        let ctx = &mut *((*stream).ctx as *mut ReadStreamCtx<'_>);
        let rem = ctx.data.len().saturating_sub(ctx.cursor);
        let to_copy = rem.min(size as usize);
        if to_copy > 0 {
            std::ptr::copy_nonoverlapping(
                ctx.data[ctx.cursor..].as_ptr(),
                buffer as *mut u8,
                to_copy,
            );
            ctx.cursor += to_copy;
        }
        to_copy as i64
    }
}

/// Enumerates all CLAP 1.2 plugin descriptors inside `binary_path` via `LoadLibraryW` (`clap_entry`).
pub fn enumerate_clap_plugins_in_library(
    binary_path: &Path,
) -> Result<Vec<ClapDescriptorSummary>, String> {
    let canon = std::fs::canonicalize(binary_path).unwrap_or_else(|_| binary_path.to_path_buf());
    let wide = wide_null_path(&canon);
    let hmodule = unsafe { LoadLibraryW(wide.as_ptr()) };
    if hmodule.is_null() {
        return Err(format!(
            "LoadLibraryW failed for CLAP binary: {}",
            canon.display()
        ));
    }

    let entry_ptr = unsafe {
        GetProcAddress(hmodule, b"clap_entry\0".as_ptr())
            .map(|f| f as *const clap_plugin_entry)
            .unwrap_or(std::ptr::null())
    };
    if entry_ptr.is_null() {
        unsafe {
            let _ = FreeLibrary(hmodule);
        }
        return Err(format!(
            "Symbol `clap_entry` not found in {}",
            canon.display()
        ));
    }

    let c_path = CString::new(canon.to_string_lossy().as_bytes())
        .unwrap_or_else(|_| CString::new("plugin.clap").unwrap());

    {
        let mut counts = clap_module_refcounts().lock();
        let entry_count = counts.entry(canon.clone()).or_insert(0);
        if *entry_count == 0 {
            let ok = unsafe {
                (*entry_ptr)
                    .init
                    .map(|init_fn| init_fn(c_path.as_ptr()))
                    .unwrap_or(false)
            };
            if !ok {
                counts.remove(&canon);
                unsafe {
                    let _ = FreeLibrary(hmodule);
                }
                return Err(format!("clap_entry.init() returned false for {}", canon.display()));
            }
        }
        *entry_count += 1;
    }

    let result = (|| -> Result<Vec<ClapDescriptorSummary>, String> {
        let factory_ptr = unsafe {
            (*entry_ptr)
                .get_factory
                .map(|gf| {
                    gf(CLAP_PLUGIN_FACTORY_ID.as_ptr() as *const c_char)
                        as *const clap_plugin_factory
                })
                .unwrap_or(std::ptr::null())
        };
        if factory_ptr.is_null() {
            return Err("clap_entry.get_factory(CLAP_PLUGIN_FACTORY_ID) returned NULL".to_string());
        }

        let count = unsafe {
            (*factory_ptr)
                .get_plugin_count
                .map(|c| c(factory_ptr))
                .unwrap_or(0)
        };
        let mut out = Vec::with_capacity(count as usize);
        let host = PinnedClapHost::new_boxed();

        for idx in 0..count {
            let desc_ptr: *const clap_plugin_descriptor = unsafe {
                (*factory_ptr)
                    .get_plugin_descriptor
                    .map(|gd| gd(factory_ptr, idx))
                    .unwrap_or(std::ptr::null())
            };
            if desc_ptr.is_null() {
                continue;
            }
            let id = unsafe { cstr_to_string((*desc_ptr).id) };
            let name = unsafe { cstr_to_string((*desc_ptr).name) };
            let vendor = unsafe { cstr_to_string((*desc_ptr).vendor) };
            let version = unsafe { cstr_to_string((*desc_ptr).version) };
            let description = unsafe { cstr_to_string((*desc_ptr).description) };

            let mut is_instrument = false;
            unsafe {
                let mut feat_cursor = (*desc_ptr).features;
                let mut guard = 0usize;
                while !feat_cursor.is_null() && guard < 32 {
                    let item = *feat_cursor;
                    if item.is_null() {
                        break;
                    }
                    let fstr = cstr_to_string(item).to_ascii_lowercase();
                    if fstr.contains("instrument") || fstr.contains("synthesizer") {
                        is_instrument = true;
                    }
                    feat_cursor = feat_cursor.add(1);
                    guard += 1;
                }
            }

            // Briefly instantiate to inspect latency & parameter count accurately
            let mut latency_samples = 0u32;
            let mut num_params = 0u32;
            if let Some(create_fn) = unsafe { (*factory_ptr).create_plugin } {
                let plug_ptr = unsafe { create_fn(factory_ptr, &host.raw, (*desc_ptr).id) };
                if !plug_ptr.is_null() {
                    let inited = unsafe { (*plug_ptr).init.map(|f| f(plug_ptr)).unwrap_or(false) };
                    if inited {
                        if let Some(get_ext) = unsafe { (*plug_ptr).get_extension } {
                            let lat_ext = unsafe {
                                get_ext(plug_ptr, CLAP_EXT_LATENCY.as_ptr() as *const c_char)
                                    as *const clap_plugin_latency
                            };
                            if !lat_ext.is_null() {
                                latency_samples = unsafe {
                                    (*lat_ext).get.map(|g| g(plug_ptr)).unwrap_or(0)
                                };
                            }
                            let par_ext = unsafe {
                                get_ext(plug_ptr, CLAP_EXT_PARAMS.as_ptr() as *const c_char)
                                    as *const clap_plugin_params
                            };
                            if !par_ext.is_null() {
                                num_params = unsafe {
                                    (*par_ext).count.map(|c| c(plug_ptr)).unwrap_or(0)
                                };
                            }
                        }
                    }
                    unsafe {
                        if let Some(destroy_fn) = (*plug_ptr).destroy {
                            destroy_fn(plug_ptr);
                        }
                    }
                }
            }

            out.push(ClapDescriptorSummary {
                id,
                name,
                vendor,
                version,
                description,
                is_instrument,
                latency_samples,
                num_params,
            });
        }
        Ok(out)
    })();

    {
        let mut counts = clap_module_refcounts().lock();
        if let Some(c) = counts.get_mut(&canon) {
            *c = c.saturating_sub(1);
            if *c == 0 {
                counts.remove(&canon);
                unsafe {
                    if let Some(deinit_fn) = (*entry_ptr).deinit {
                        deinit_fn();
                    }
                }
            }
        }
    }
    unsafe {
        let _ = FreeLibrary(hmodule);
    }
    result
}

/// Real-Time Hosted CLAP 1.2 Plugin Node (`UnifiedDspNode`, `0` RT heap allocations).
pub struct ClapHostedPluginNode {
    canon_path: PathBuf,
    plugin_id: String,
    plugin_name: String,
    hmodule: HMODULE,
    entry_ptr: *const clap_plugin_entry,
    plugin_ptr: *const clap_plugin,
    ext_params: *const clap_plugin_params,
    ext_latency: *const clap_plugin_latency,
    ext_state: *const clap_plugin_state,
    ext_gui: *const clap_plugin_gui,
    _ext_audio_ports: *const clap_plugin_audio_ports,
    _ext_note_ports: *const clap_plugin_note_ports,
    _host: Box<PinnedClapHost>,
    rt_arena: Box<ClapRtArena>,
    param_metadata: Vec<HostedPluginParamInfo>,
    editor_window: Option<NativePluginWindow>,
    gui_created: bool,
    /// Shared with [`ClapLiveEditor`]s on this instance; cleared (under lock) before destroy.
    live_guard: std::sync::Arc<parking_lot::Mutex<bool>>,
    activated: bool,
    processing_started: bool,
    sample_rate: f64,
    max_block_size: usize,
    cached_latency_samples: usize,
    steady_time: i64,
}

// SAFETY: `ClapHostedPluginNode` moves cleanly onto the real-time audio thread (via `RtGraphMutation`)
// and back off to `toneforge-rt-gc` on retirement without concurrent cross-thread mutation.
unsafe impl Send for ClapHostedPluginNode {}

impl ClapHostedPluginNode {
    /// Loads a `.clap` (or dual-ABI `.dll`) from `binary_path` via `LoadLibraryW` and activates
    /// `requested_plugin_id` (if `None`, loads descriptor index `0`).
    pub fn load(
        binary_path: &Path,
        requested_plugin_id: Option<&str>,
        sample_rate: f64,
        max_block_size: usize,
    ) -> Result<Self, String> {
        let canon =
            std::fs::canonicalize(binary_path).unwrap_or_else(|_| binary_path.to_path_buf());
        let wide = wide_null_path(&canon);
        let hmodule = unsafe { LoadLibraryW(wide.as_ptr()) };
        if hmodule.is_null() {
            return Err(format!(
                "LoadLibraryW failed for CLAP plugin: {}",
                canon.display()
            ));
        }

        let entry_ptr = unsafe {
            GetProcAddress(hmodule, b"clap_entry\0".as_ptr())
                .map(|f| f as *const clap_plugin_entry)
                .unwrap_or(std::ptr::null())
        };
        if entry_ptr.is_null() {
            unsafe {
                let _ = FreeLibrary(hmodule);
            }
            return Err(format!(
                "Symbol `clap_entry` missing in {}",
                canon.display()
            ));
        }

        let c_path = CString::new(canon.to_string_lossy().as_bytes())
            .unwrap_or_else(|_| CString::new("plugin.clap").unwrap());

        {
            let mut counts = clap_module_refcounts().lock();
            let entry_count = counts.entry(canon.clone()).or_insert(0);
            if *entry_count == 0 {
                let ok = unsafe {
                    (*entry_ptr)
                        .init
                        .map(|init_fn| init_fn(c_path.as_ptr()))
                        .unwrap_or(false)
                };
                if !ok {
                    counts.remove(&canon);
                    unsafe {
                        let _ = FreeLibrary(hmodule);
                    }
                    return Err(format!("clap_entry.init() failed for {}", canon.display()));
                }
            }
            *entry_count += 1;
        }

        let factory_ptr = unsafe {
            (*entry_ptr)
                .get_factory
                .map(|gf| {
                    gf(CLAP_PLUGIN_FACTORY_ID.as_ptr() as *const c_char)
                        as *const clap_plugin_factory
                })
                .unwrap_or(std::ptr::null())
        };
        if factory_ptr.is_null() {
            Self::decrement_entry_and_free(&canon, entry_ptr, hmodule);
            return Err("CLAP factory NULL".to_string());
        }

        let count = unsafe {
            (*factory_ptr)
                .get_plugin_count
                .map(|c| c(factory_ptr))
                .unwrap_or(0)
        };
        if count == 0 {
            Self::decrement_entry_and_free(&canon, entry_ptr, hmodule);
            return Err("CLAP binary contains 0 plugins".to_string());
        }

        let mut chosen_desc: *const clap_plugin_descriptor = std::ptr::null();
        for idx in 0..count {
            let d = unsafe {
                (*factory_ptr)
                    .get_plugin_descriptor
                    .map(|gd| gd(factory_ptr, idx))
                    .unwrap_or(std::ptr::null())
            };
            if d.is_null() {
                continue;
            }
            let id_str = unsafe { cstr_to_string((*d).id) };
            if let Some(req) = requested_plugin_id {
                if !req.is_empty() && id_str == req {
                    chosen_desc = d;
                    break;
                }
            } else if chosen_desc.is_null() {
                chosen_desc = d;
                break;
            }
            if chosen_desc.is_null() {
                chosen_desc = d;
            }
        }

        if chosen_desc.is_null() {
            Self::decrement_entry_and_free(&canon, entry_ptr, hmodule);
            return Err("No matching CLAP descriptor found".to_string());
        }

        let plugin_id = unsafe { cstr_to_string((*chosen_desc).id) };
        let plugin_name = unsafe { cstr_to_string((*chosen_desc).name) };
        let host = PinnedClapHost::new_boxed();

        let plugin_ptr = unsafe {
            (*factory_ptr)
                .create_plugin
                .map(|cf| cf(factory_ptr, &host.raw, (*chosen_desc).id))
                .unwrap_or(std::ptr::null())
        };
        if plugin_ptr.is_null() {
            Self::decrement_entry_and_free(&canon, entry_ptr, hmodule);
            return Err(format!("create_plugin failed for `{plugin_id}`"));
        }

        let init_ok = unsafe { (*plugin_ptr).init.map(|f| f(plugin_ptr)).unwrap_or(false) };
        if !init_ok {
            unsafe {
                if let Some(destroy_fn) = (*plugin_ptr).destroy {
                    destroy_fn(plugin_ptr);
                }
            }
            Self::decrement_entry_and_free(&canon, entry_ptr, hmodule);
            return Err(format!("clap_plugin.init() failed for `{plugin_id}`"));
        }

        let get_ext = unsafe { (*plugin_ptr).get_extension };
        let query_ext = |ext_id: &[u8]| -> *const c_void {
            unsafe {
                get_ext
                    .map(|f| f(plugin_ptr, ext_id.as_ptr() as *const c_char))
                    .unwrap_or(std::ptr::null())
            }
        };

        let ext_params = query_ext(CLAP_EXT_PARAMS) as *const clap_plugin_params;
        let ext_latency = query_ext(CLAP_EXT_LATENCY) as *const clap_plugin_latency;
        let ext_state = query_ext(CLAP_EXT_STATE) as *const clap_plugin_state;
        let ext_gui = query_ext(CLAP_EXT_GUI) as *const clap_plugin_gui;
        let ext_audio_ports = query_ext(CLAP_EXT_AUDIO_PORTS) as *const clap_plugin_audio_ports;
        let ext_note_ports = query_ext(CLAP_EXT_NOTE_PORTS) as *const clap_plugin_note_ports;

        let mut rt_arena = ClapRtArena::new_boxed();
        let mut param_metadata = Vec::new();

        if !ext_params.is_null() {
            let pcount = unsafe { (*ext_params).count.map(|c| c(plugin_ptr)).unwrap_or(0) };
            for i in 0..pcount {
                let mut info: clap_param_info = unsafe { std::mem::zeroed() };
                let ok = unsafe {
                    (*ext_params)
                        .get_info
                        .map(|gi| gi(plugin_ptr, i, &mut info))
                        .unwrap_or(false)
                };
                if ok {
                    let mut cur_val = info.default_value;
                    unsafe {
                        if let Some(gv) = (*ext_params).get_value {
                            let _ = gv(plugin_ptr, info.id, &mut cur_val);
                        }
                    }
                    let pname = unsafe { cstr_to_string(info.name.as_ptr()) };
                    let pmod = unsafe { cstr_to_string(info.module.as_ptr()) };
                    param_metadata.push(HostedPluginParamInfo {
                        id: info.id,
                        name: pname,
                        module: pmod,
                        min_value: info.min_value,
                        max_value: info.max_value,
                        default_value: info.default_value,
                        current_value: cur_val,
                    });
                    if rt_arena.param_count < MAX_CLAP_RT_PARAMS {
                        rt_arena.params[rt_arena.param_count] = RtParamSlot {
                            id: info.id,
                            min_value: info.min_value,
                            max_value: info.max_value,
                            current_plain: cur_val,
                        };
                        rt_arena.param_count += 1;
                    }
                }
            }
        }

        let sr = sample_rate.max(8_000.0);
        let mbs = max_block_size.clamp(1, MAX_BLOCK_SIZE);
        let mut node = Self {
            canon_path: canon,
            plugin_id,
            plugin_name,
            hmodule,
            entry_ptr,
            plugin_ptr,
            ext_params,
            ext_latency,
            ext_state,
            ext_gui,
            _ext_audio_ports: ext_audio_ports,
            _ext_note_ports: ext_note_ports,
            _host: host,
            rt_arena,
            param_metadata,
            editor_window: None,
            gui_created: false,
            live_guard: std::sync::Arc::new(parking_lot::Mutex::new(true)),
            activated: false,
            processing_started: false,
            sample_rate: sr,
            max_block_size: mbs,
            cached_latency_samples: 0,
            steady_time: 0,
        };
        node.activate_and_start(sr, mbs)?;
        Ok(node)
    }

    fn decrement_entry_and_free(
        canon: &Path,
        entry_ptr: *const clap_plugin_entry,
        hmodule: HMODULE,
    ) {
        {
            let mut counts = clap_module_refcounts().lock();
            if let Some(c) = counts.get_mut(canon) {
                *c = c.saturating_sub(1);
                if *c == 0 {
                    counts.remove(canon);
                    unsafe {
                        if !entry_ptr.is_null() {
                            if let Some(deinit_fn) = (*entry_ptr).deinit {
                                deinit_fn();
                            }
                        }
                    }
                }
            }
        }
        if !hmodule.is_null() {
            unsafe {
                let _ = FreeLibrary(hmodule);
            }
        }
    }

    fn activate_and_start(&mut self, sample_rate: f64, max_block_size: usize) -> Result<(), String> {
        let sr = sample_rate.max(8_000.0);
        let mbs = max_block_size.clamp(1, MAX_BLOCK_SIZE);
        if self.processing_started {
            unsafe {
                if let Some(stop_fn) = (*self.plugin_ptr).stop_processing {
                    stop_fn(self.plugin_ptr);
                }
            }
            self.processing_started = false;
        }
        if self.activated {
            unsafe {
                if let Some(deact_fn) = (*self.plugin_ptr).deactivate {
                    deact_fn(self.plugin_ptr);
                }
            }
            self.activated = false;
        }

        let act_ok = unsafe {
            (*self.plugin_ptr)
                .activate
                .map(|a| a(self.plugin_ptr, sr, 1, mbs as u32))
                .unwrap_or(false)
        };
        if !act_ok {
            return Err(format!(
                "clap_plugin.activate() failed for `{}`",
                self.plugin_name
            ));
        }
        self.activated = true;

        let start_ok = unsafe {
            (*self.plugin_ptr)
                .start_processing
                .map(|s| s(self.plugin_ptr))
                .unwrap_or(true)
        };
        self.processing_started = start_ok;
        self.sample_rate = sr;
        self.max_block_size = mbs;

        if !self.ext_latency.is_null() {
            self.cached_latency_samples = unsafe {
                (*self.ext_latency)
                    .get
                    .map(|g| g(self.plugin_ptr) as usize)
                    .unwrap_or(0)
            };
        }
        Ok(())
    }

    #[inline]
    pub fn plugin_id(&self) -> &str {
        &self.plugin_id
    }

    #[inline]
    pub fn plugin_display_name(&self) -> &str {
        &self.plugin_name
    }

    #[inline]
    pub fn parameters(&self) -> &[HostedPluginParamInfo] {
        &self.param_metadata
    }

    /// Reads the live plain parameter value directly from `clap.params.get_value`.
    pub fn get_parameter_plain(&self, param_id: u32) -> Option<f64> {
        if self.ext_params.is_null() {
            return None;
        }
        let mut val = 0.0f64;
        let ok = unsafe {
            (*self.ext_params)
                .get_value
                .map(|gv| gv(self.plugin_ptr, param_id, &mut val))
                .unwrap_or(false)
        };
        if ok { Some(val) } else { None }
    }

    /// Schedules a sample-accurate plain parameter change (`0` heap allocations).
    #[inline]
    pub fn queue_parameter_plain(&mut self, param_id: u32, plain_value: f64, sample_offset: u32) {
        for i in 0..self.rt_arena.param_count {
            if self.rt_arena.params[i].id == param_id {
                self.rt_arena.params[i].current_plain = plain_value.clamp(
                    self.rt_arena.params[i].min_value,
                    self.rt_arena.params[i].max_value,
                );
                break;
            }
        }
        self.rt_arena
            .push_param_event(sample_offset, param_id, plain_value);
    }

    /// Schedules a sample-accurate CLAP NoteOn event (`0` heap allocations).
    #[inline]
    pub fn queue_note_on(&mut self, key: u8, velocity: f32, channel: u8, sample_offset: u32) {
        self.rt_arena.push_note_event(
            CLAP_EVENT_NOTE_ON,
            sample_offset,
            key as i32,
            (channel & 0x0F) as i16,
            (key & 0x7F) as i16,
            (velocity as f64).clamp(0.0, 1.0),
        );
    }

    /// Schedules a sample-accurate CLAP NoteOff event (`0` heap allocations).
    #[inline]
    pub fn queue_note_off(&mut self, key: u8, velocity: f32, channel: u8, sample_offset: u32) {
        self.rt_arena.push_note_event(
            CLAP_EVENT_NOTE_OFF,
            sample_offset,
            key as i32,
            (channel & 0x0F) as i16,
            (key & 0x7F) as i16,
            (velocity as f64).clamp(0.0, 1.0),
        );
    }

    /// Saves the binary plugin state via `clap.state.save`.
    pub fn save_state(&self) -> Result<Vec<u8>, String> {
        if self.ext_state.is_null() {
            return Err("Plugin does not implement clap.state".to_string());
        }
        let save_fn = unsafe { (*self.ext_state).save }
            .ok_or_else(|| "clap.state.save is NULL".to_string())?;
        let mut ctx = WriteStreamCtx {
            buf: Vec::with_capacity(256),
        };
        let ostream = clap_ostream {
            ctx: (&mut ctx) as *mut WriteStreamCtx as *mut c_void,
            write: Some(clap_ostream_write),
        };
        let ok = unsafe { save_fn(self.plugin_ptr, &ostream) };
        if !ok {
            return Err("clap.state.save returned false".to_string());
        }
        Ok(ctx.buf)
    }

    /// Restores the binary plugin state via `clap.state.load` and refreshes cached parameter values.
    pub fn load_state(&mut self, state_bytes: &[u8]) -> Result<(), String> {
        if self.ext_state.is_null() {
            return Err("Plugin does not implement clap.state".to_string());
        }
        let load_fn = unsafe { (*self.ext_state).load }
            .ok_or_else(|| "clap.state.load is NULL".to_string())?;
        let mut ctx = ReadStreamCtx {
            data: state_bytes,
            cursor: 0,
        };
        let istream = clap_istream {
            ctx: (&mut ctx) as *mut ReadStreamCtx<'_> as *mut c_void,
            read: Some(clap_istream_read),
        };
        let ok = unsafe { load_fn(self.plugin_ptr, &istream) };
        if !ok {
            return Err("clap.state.load returned false".to_string());
        }
        self.refresh_cached_params();
        Ok(())
    }

    fn refresh_cached_params(&mut self) {
        if self.ext_params.is_null() {
            return;
        }
        for meta in &mut self.param_metadata {
            let mut v = meta.current_value;
            let ok = unsafe {
                (*self.ext_params)
                    .get_value
                    .map(|gv| gv(self.plugin_ptr, meta.id, &mut v))
                    .unwrap_or(false)
            };
            if ok {
                meta.current_value = v;
                for i in 0..self.rt_arena.param_count {
                    if self.rt_arena.params[i].id == meta.id {
                        self.rt_arena.params[i].current_plain = v;
                    }
                }
            }
        }
    }

    /// Opens the native Win32 `HWND` plugin GUI (`clap.gui` with `CLAP_WINDOW_API_WIN32`).
    pub fn open_editor(&mut self, owner_hwnd: Option<isize>, visible: bool) -> Result<isize, String> {
        if self.ext_gui.is_null() {
            return Err("Plugin does not implement clap.gui".to_string());
        }
        if let Some(ref win) = self.editor_window {
            if win.is_valid() {
                if visible {
                    win.show();
                }
                return Ok(win.hwnd_isize());
            }
        }
        self.close_editor();

        let api_ptr = CLAP_WINDOW_API_WIN32.as_ptr() as *const c_char;
        let supported = unsafe {
            (*self.ext_gui)
                .is_api_supported
                .map(|f| f(self.plugin_ptr, api_ptr, false))
                .unwrap_or(false)
        };
        if !supported {
            return Err("Plugin clap.gui does not support win32 embedded HWND".to_string());
        }

        let created = unsafe {
            (*self.ext_gui)
                .create
                .map(|f| f(self.plugin_ptr, api_ptr, false))
                .unwrap_or(false)
        };
        if !created {
            return Err("clap.gui.create(win32) returned false".to_string());
        }
        self.gui_created = true;

        let mut width = 640u32;
        let mut height = 400u32;
        unsafe {
            if let Some(gs) = (*self.ext_gui).get_size {
                let _ = gs(self.plugin_ptr, &mut width, &mut height);
            }
        }
        let resizable = unsafe {
            (*self.ext_gui)
                .can_resize
                .map(|f| f(self.plugin_ptr))
                .unwrap_or(false)
        };

        let title = format!("{} (CLAP)", self.plugin_name);
        let win = NativePluginWindow::open(&title, width, height, owner_hwnd, resizable, visible)?;
        let cwin = clap_window {
            api: api_ptr,
            ptr: win.hwnd_raw(),
        };
        let parent_ok = unsafe {
            (*self.ext_gui)
                .set_parent
                .map(|f| f(self.plugin_ptr, &cwin))
                .unwrap_or(false)
        };
        if !parent_ok {
            self.close_editor();
            return Err("clap.gui.set_parent failed".to_string());
        }
        unsafe {
            if let Some(show_fn) = (*self.ext_gui).show {
                let _ = show_fn(self.plugin_ptr);
            }
        }
        let hwnd_val = win.hwnd_isize();
        self.editor_window = Some(win);
        Ok(hwnd_val)
    }

    /// Hands out an editor bound to *this* plug-in instance (CLAP `clap.gui` is main-thread and
    /// may run concurrently with `process()` on the audio thread). Call before the node moves to
    /// the real-time graph.
    pub fn live_editor(&self) -> Option<ClapLiveEditor> {
        if self.ext_gui.is_null() || self.plugin_ptr.is_null() {
            return None;
        }
        Some(ClapLiveEditor {
            plugin_ptr: self.plugin_ptr,
            ext_gui: self.ext_gui,
            ext_params: self.ext_params,
            plugin_name: self.plugin_name.clone(),
            params: self
                .param_metadata
                .iter()
                .map(|p| (p.id, p.min_value, p.max_value))
                .collect(),
            live_guard: std::sync::Arc::clone(&self.live_guard),
            editor_window: None,
            gui_created: false,
        })
    }

    pub fn pump_editor_messages(&self) -> usize {
        if let Some(ref win) = self.editor_window {
            win.pump_messages()
        } else {
            0
        }
    }

    pub fn close_editor(&mut self) {
        if self.gui_created && !self.ext_gui.is_null() {
            unsafe {
                if let Some(hide_fn) = (*self.ext_gui).hide {
                    let _ = hide_fn(self.plugin_ptr);
                }
                if let Some(destroy_fn) = (*self.ext_gui).destroy {
                    destroy_fn(self.plugin_ptr);
                }
            }
            self.gui_created = false;
        }
        if let Some(mut win) = self.editor_window.take() {
            win.destroy_now();
        }
    }
}

impl UnifiedDspNode for ClapHostedPluginNode {
    fn name(&self) -> &'static str {
        "CLAP 1.2 Hosted Plugin"
    }

    fn latency_samples(&self) -> usize {
        self.cached_latency_samples
    }

    fn prepare(&mut self, sample_rate: f64, max_block_size: usize) {
        let _ = self.activate_and_start(sample_rate, max_block_size);
    }

    fn reset_state(&mut self) {
        self.rt_arena.event_count = 0;
        self.rt_arena.out_param_count = 0;
        unsafe {
            if let Some(reset_fn) = (*self.plugin_ptr).reset {
                reset_fn(self.plugin_ptr);
            }
        }
    }

    fn set_parameter_normalized(&mut self, param_id: u32, norm_val: f32) {
        let n = (norm_val as f64).clamp(0.0, 1.0);
        let mut resolved_id = param_id;
        let mut plain = n;
        let mut matched = false;
        for i in 0..self.rt_arena.param_count {
            if self.rt_arena.params[i].id == param_id {
                let slot = &mut self.rt_arena.params[i];
                plain = slot.min_value + (slot.max_value - slot.min_value) * n;
                slot.current_plain = plain;
                matched = true;
                break;
            }
        }
        if !matched && (param_id as usize) < self.rt_arena.param_count {
            let slot = &mut self.rt_arena.params[param_id as usize];
            resolved_id = slot.id;
            plain = slot.min_value + (slot.max_value - slot.min_value) * n;
            slot.current_plain = plain;
        }
        self.rt_arena.push_param_event(0, resolved_id, plain);
    }

    fn on_voice_event(&mut self, event: &SequencerVoiceEvent, ctx: &ProcessContext) {
        let max_off = ctx.slice_frames.saturating_sub(1) as i64;
        let offset = (event.sample_pos - ctx.current_sample_pos).clamp(0, max_off) as u32;
        let ch = (event.color_group & 0x0F) as i16;
        let key = (event.key & 0x7F) as i16;
        let vel = (event.velocity as f64).clamp(0.0, 1.0);
        match event.kind {
            VoiceEventKind::NoteOn | VoiceEventKind::SlideNote => {
                self.rt_arena.push_note_event(
                    CLAP_EVENT_NOTE_ON,
                    offset,
                    event.note_id as i32,
                    ch,
                    key,
                    vel,
                );
            }
            VoiceEventKind::NoteOff => {
                self.rt_arena.push_note_event(
                    CLAP_EVENT_NOTE_OFF,
                    offset,
                    event.note_id as i32,
                    ch,
                    key,
                    vel,
                );
            }
            VoiceEventKind::ChannelMute => {
                // All-Notes-Off MIDI CC 123 on channel mute
                self.rt_arena
                    .push_midi_event(offset, 0xB0 | (ch as u8 & 0x0F), 123, 0);
            }
            VoiceEventKind::SlotEnable => {}
        }
    }

    fn process_slice(
        &mut self,
        ctx: &ProcessContext,
        main_io: &mut [StereoFrame],
        _sidechain_in: Option<&[StereoFrame]>,
        _aux_sends_out: &mut [&mut [StereoFrame]],
    ) {
        let total = main_io.len();
        if total == 0 {
            self.rt_arena.event_count = 0;
            return;
        }

        let Some(process_fn) = (unsafe { (*self.plugin_ptr).process }) else {
            self.rt_arena.event_count = 0;
            return;
        };

        self.rt_arena.sort_events_by_time();
        let arena_ptr: *mut ClapRtArena = &mut *self.rt_arena;

        // Chunk safely in <= MAX_BLOCK_SIZE blocks (`0` heap allocations).
        let mut cursor = 0usize;
        while cursor < total {
            let chunk = (total - cursor).min(MAX_BLOCK_SIZE);
            for i in 0..chunk {
                let f = main_io[cursor + i];
                self.rt_arena.in_left[i] = f[0];
                self.rt_arena.in_right[i] = f[1];
                self.rt_arena.out_left[i] = 0.0;
                self.rt_arena.out_right[i] = 0.0;
            }
            self.rt_arena.in_ptrs[0] = self.rt_arena.in_left.as_mut_ptr();
            self.rt_arena.in_ptrs[1] = self.rt_arena.in_right.as_mut_ptr();
            self.rt_arena.out_ptrs[0] = self.rt_arena.out_left.as_mut_ptr();
            self.rt_arena.out_ptrs[1] = self.rt_arena.out_right.as_mut_ptr();
            self.rt_arena.out_param_count = 0;

            let in_buf = clap_audio_buffer {
                data32: self.rt_arena.in_ptrs.as_mut_ptr(),
                data64: std::ptr::null_mut(),
                channel_count: 2,
                latency: 0,
                constant_mask: 0,
            };
            let mut out_buf = clap_audio_buffer {
                data32: self.rt_arena.out_ptrs.as_mut_ptr(),
                data64: std::ptr::null_mut(),
                channel_count: 2,
                latency: 0,
                constant_mask: 0,
            };
            let in_events = clap_input_events {
                ctx: arena_ptr as *mut c_void,
                size: Some(clap_in_events_size),
                get: Some(clap_in_events_get),
            };
            let out_events = clap_output_events {
                ctx: arena_ptr as *mut c_void,
                try_push: Some(clap_out_events_try_push),
            };
            // Transport at this chunk: active PPQ tick plus the chunk offset at the current tempo.
            let beats = (ctx.ppq_tick as f64) / (ctx.ppq_resolution.max(1) as f64)
                + (cursor as f64) * ctx.bpm / (60.0 * ctx.sample_rate.max(1.0));
            let seconds = ((ctx.current_sample_pos + cursor as i64) as f64) / ctx.sample_rate.max(1.0);
            let bar = (beats / 4.0).floor();
            let transport = clap_event_transport {
                header: clap_event_header {
                    size: std::mem::size_of::<clap_event_transport>() as u32,
                    time: 0,
                    space_id: CLAP_CORE_EVENT_SPACE_ID,
                    type_: CLAP_EVENT_TRANSPORT,
                    flags: 0,
                },
                flags: CLAP_TRANSPORT_HAS_TEMPO
                    | CLAP_TRANSPORT_HAS_BEATS_TIMELINE
                    | CLAP_TRANSPORT_HAS_SECONDS_TIMELINE
                    | CLAP_TRANSPORT_HAS_TIME_SIGNATURE
                    | if ctx.is_playing { CLAP_TRANSPORT_IS_PLAYING } else { 0 },
                song_pos_beats: (beats * CLAP_BEATTIME_FACTOR as f64).round() as i64,
                song_pos_seconds: (seconds * CLAP_SECTIME_FACTOR as f64).round() as i64,
                tempo: ctx.bpm,
                tempo_inc: 0.0,
                loop_start_beats: 0,
                loop_end_beats: 0,
                loop_start_seconds: 0,
                loop_end_seconds: 0,
                bar_start: ((bar * 4.0) * CLAP_BEATTIME_FACTOR as f64).round() as i64,
                bar_number: bar as i32,
                tsig_num: 4,
                tsig_denom: 4,
            };
            let proc_data = clap_process {
                steady_time: self.steady_time,
                frames_count: chunk as u32,
                transport: &transport as *const clap_event_transport as *const c_void,
                audio_inputs: &in_buf,
                audio_outputs: &mut out_buf,
                audio_inputs_count: 1,
                audio_outputs_count: 1,
                in_events: &in_events,
                out_events: &out_events,
            };

            unsafe {
                let _ = process_fn(self.plugin_ptr, &proc_data);
            }

            for i in 0..chunk {
                main_io[cursor + i] = [self.rt_arena.out_left[i], self.rt_arena.out_right[i]];
            }
            self.steady_time = self.steady_time.wrapping_add(chunk as i64);
            self.rt_arena.event_count = 0;
            cursor += chunk;
        }
    }
}

impl Drop for ClapHostedPluginNode {
    fn drop(&mut self) {
        *self.live_guard.lock() = false;
        self.close_editor();
        if !self.plugin_ptr.is_null() {
            unsafe {
                if self.processing_started {
                    if let Some(stop_fn) = (*self.plugin_ptr).stop_processing {
                        stop_fn(self.plugin_ptr);
                    }
                    self.processing_started = false;
                }
                if self.activated {
                    if let Some(deact_fn) = (*self.plugin_ptr).deactivate {
                        deact_fn(self.plugin_ptr);
                    }
                    self.activated = false;
                }
                if let Some(destroy_fn) = (*self.plugin_ptr).destroy {
                    destroy_fn(self.plugin_ptr);
                }
            }
            self.plugin_ptr = std::ptr::null();
        }
        Self::decrement_entry_and_free(&self.canon_path, self.entry_ptr, self.hmodule);
        self.hmodule = std::ptr::null_mut();
    }
}


/// Editor attached to the *live* (real-time) CLAP instance: the GUI thread calls `clap.gui` and
/// `clap.params.get_value` on the same plug-in object the audio thread processes, so any preset,
/// sample or parameter edited in the window is what plays.
pub struct ClapLiveEditor {
    plugin_ptr: *const clap_plugin,
    ext_gui: *const clap_plugin_gui,
    ext_params: *const clap_plugin_params,
    plugin_name: String,
    params: Vec<(u32, f64, f64)>,
    live_guard: std::sync::Arc<parking_lot::Mutex<bool>>,
    editor_window: Option<NativePluginWindow>,
    gui_created: bool,
}

// SAFETY: handed once to the single GUI thread; every plug-in call happens under `live_guard`.
unsafe impl Send for ClapLiveEditor {}

impl ClapLiveEditor {
    pub fn open_editor(&mut self, owner_hwnd: Option<isize>, visible: bool) -> Result<isize, String> {
        let alive = self.live_guard.lock();
        if !*alive {
            return Err("Plug-in instance was removed".to_string());
        }
        if let Some(ref win) = self.editor_window {
            if win.is_valid() {
                if visible {
                    win.show();
                }
                return Ok(win.hwnd_isize());
            }
        }
        let api_ptr = CLAP_WINDOW_API_WIN32.as_ptr() as *const c_char;
        unsafe {
            let gui = &*self.ext_gui;
            if !gui.is_api_supported.map(|f| f(self.plugin_ptr, api_ptr, false)).unwrap_or(false) {
                return Err("Plugin clap.gui does not support win32 embedded HWND".to_string());
            }
            if !gui.create.map(|f| f(self.plugin_ptr, api_ptr, false)).unwrap_or(false) {
                return Err("clap.gui.create(win32) returned false".to_string());
            }
            self.gui_created = true;
            let (mut width, mut height) = (640u32, 400u32);
            if let Some(gs) = gui.get_size {
                let _ = gs(self.plugin_ptr, &mut width, &mut height);
            }
            let resizable = gui.can_resize.map(|f| f(self.plugin_ptr)).unwrap_or(false);
            let title = format!("{} (CLAP)", self.plugin_name);
            let win = NativePluginWindow::open(&title, width, height, owner_hwnd, resizable, visible)?;
            let cwin = clap_window {
                api: api_ptr,
                ptr: win.hwnd_raw(),
            };
            if !gui.set_parent.map(|f| f(self.plugin_ptr, &cwin)).unwrap_or(false) {
                if let Some(destroy_fn) = gui.destroy {
                    destroy_fn(self.plugin_ptr);
                }
                self.gui_created = false;
                let mut w = win;
                w.destroy_now();
                return Err("clap.gui.set_parent failed".to_string());
            }
            if let Some(show_fn) = gui.show {
                let _ = show_fn(self.plugin_ptr);
            }
            let hwnd_val = win.hwnd_isize();
            self.editor_window = Some(win);
            Ok(hwnd_val)
        }
    }

    pub fn close_editor(&mut self) {
        let alive = self.live_guard.lock();
        if *alive && self.gui_created {
            unsafe {
                let gui = &*self.ext_gui;
                if let Some(hide_fn) = gui.hide {
                    let _ = hide_fn(self.plugin_ptr);
                }
                if let Some(destroy_fn) = gui.destroy {
                    destroy_fn(self.plugin_ptr);
                }
            }
        }
        self.gui_created = false;
        if let Some(mut win) = self.editor_window.take() {
            win.destroy_now();
        }
    }

    pub fn pump_editor_messages(&mut self) -> usize {
        if !*self.live_guard.lock() {
            self.close_editor();
            return 0;
        }
        self.editor_window.as_ref().map(|w| w.pump_messages()).unwrap_or(0)
    }

    /// `(param_id, min, max)` of every exposed parameter.
    pub fn parameters(&self) -> &[(u32, f64, f64)] {
        &self.params
    }

    pub fn get_parameter_plain(&self, param_id: u32) -> Option<f64> {
        let alive = self.live_guard.lock();
        if !*alive || self.ext_params.is_null() {
            return None;
        }
        let mut val = 0.0f64;
        let ok = unsafe {
            (*self.ext_params)
                .get_value
                .map(|gv| gv(self.plugin_ptr, param_id, &mut val))
                .unwrap_or(false)
        };
        if ok { Some(val) } else { None }
    }

    pub fn is_alive(&self) -> bool {
        *self.live_guard.lock()
    }
}

impl Drop for ClapLiveEditor {
    fn drop(&mut self) {
        self.close_editor();
    }
}
