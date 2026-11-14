//! `vst3_host`: Production Real-Time **Steinberg VST3 3.7** Windows COM Host
//! (`Vst3HostedPluginNode` implementing `UnifiedDspNode` with `0` RT heap allocations, `0` RT
//! locks, single & dual `IComponent`/`IEditController` + `IConnectionPoint`, `IComponentHandler`
//! GUI parameter feedback via lock-free `rtrb`, `IBStream` binary state save/restore, and Win32
//! `HWND` `IPlugView` + `IPlugFrame` editor hosting).

use crate::clap_host::HostedPluginParamInfo;
use crate::hwnd_window::NativePluginWindow;
use crate::vst3_abi::{
    BusInfo,
    IID_IPLUG_VIEW_CONTENT_SCALE_SUPPORT, IPlugViewContentScaleSupportVtbl,
    AudioBusBuffers, FUnknownVtbl, IAudioProcessorVtbl, IBStreamVtbl, IComponentHandlerVtbl,
    IComponentVtbl, IConnectionPointVtbl, IEditControllerVtbl, IEventListVtbl,
    IHostApplicationVtbl, IID_FUNKNOWN, IID_IAUDIO_PROCESSOR, IID_IBSTREAM, IID_ICOMPONENT,
    IID_ICOMPONENT_HANDLER, IID_ICONNECTION_POINT, IID_IEDIT_CONTROLLER, IID_IEVENT_LIST,
    IID_IHOST_APPLICATION, IID_IPARAM_VALUE_QUEUE, IID_IPARAMETER_CHANGES, IID_IPLUG_FRAME,
    IParamValueQueueVtbl, IParameterChangesVtbl, IPlugFrameVtbl, IPlugViewVtbl,
    IPluginFactoryVtbl, K_AUDIO, K_EVENT, K_INPUT, K_INVALID_ARGUMENT, K_NO_INTERFACE,
    K_NOTE_OFF_EVENT, K_NOTE_ON_EVENT, K_NOT_IMPLEMENTED, K_OUTPUT, K_REALTIME, K_RESULT_FALSE,
    K_RESULT_OK, K_SAMPLE_32, NoteOffEvent, NoteOnEvent, PClassInfo, PFactoryInfo, ParamID,
    ParamValue, ParameterInfo, ProcessContextVst3, ProcessData, ProcessSetup, TUID,
    VST3_PLATFORM_TYPE_HWND, VST3_VIEW_TYPE_EDITOR, ViewRect, Vst3Event, Vst3EventPayload, tresult,
};
use daw_core_graph::{
    MAX_BLOCK_SIZE, ProcessContext, SequencerVoiceEvent, StereoFrame, UnifiedDspNode,
    VoiceEventKind,
};
use serde::{Deserialize, Serialize};
use std::ffi::{c_char, c_void};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use windows_sys::Win32::Foundation::{FreeLibrary, HMODULE};
use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};

pub const MAX_VST3_RT_EVENTS: usize = 512;
pub const MAX_VST3_PARAM_QUEUES: usize = 64;
pub const MAX_POINTS_PER_QUEUE: usize = 16;

/// Summary of a Steinberg VST3 `"Audio Module Class"` discovered inside a `.vst3` binary/bundle.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Vst3DescriptorSummary {
    pub cid_hex: String,
    pub name: String,
    pub vendor: String,
    pub category: String,
    pub is_instrument: bool,
    pub latency_samples: u32,
    pub num_params: u32,
}

pub fn tuid_to_hex(uid: &TUID) -> String {
    let mut out = String::with_capacity(32);
    for b in uid {
        use std::fmt::Write;
        let _ = write!(&mut out, "{b:02X}");
    }
    out
}

pub fn hex_to_tuid(hex: &str) -> Option<TUID> {
    let clean: String = hex.chars().filter(|c| c.is_ascii_hexdigit()).collect();
    if clean.len() != 32 {
        return None;
    }
    let mut uid = [0u8; 16];
    for i in 0..16 {
        uid[i] = u8::from_str_radix(&clean[i * 2..i * 2 + 2], 16).ok()?;
    }
    Some(uid)
}

/// Resolves either a flat `.vst3` / `.dll` file or a Steinberg VST3 3.7 bundle directory
/// (`MyPlugin.vst3/Contents/x86_64-win/MyPlugin.vst3`).
pub fn resolve_vst3_binary_path(path: &Path) -> Result<PathBuf, String> {
    if path.is_file() {
        return Ok(std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf()));
    }
    if path.is_dir() {
        let win64_dir = path.join("Contents").join("x86_64-win");
        if win64_dir.is_dir() {
            if let Some(stem) = path.file_stem() {
                let candidate = win64_dir.join(format!("{}.vst3", stem.to_string_lossy()));
                if candidate.is_file() {
                    return Ok(std::fs::canonicalize(&candidate).unwrap_or(candidate));
                }
            }
            if let Ok(entries) = std::fs::read_dir(&win64_dir) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.is_file() {
                        if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
                            if ext.eq_ignore_ascii_case("vst3") || ext.eq_ignore_ascii_case("dll") {
                                return Ok(std::fs::canonicalize(&p).unwrap_or(p));
                            }
                        }
                    }
                }
            }
        }
    }
    Err(format!(
        "Could not resolve loadable x86_64 VST3 binary from {}",
        path.display()
    ))
}

fn wide_null_path(path: &Path) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    let mut v: Vec<u16> = path.as_os_str().encode_wide().collect();
    v.push(0);
    v
}

unsafe fn fixed_cchar_to_string(buf: &[c_char]) -> String {
    let nul_pos = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, nul_pos) };
    String::from_utf8_lossy(bytes).trim().to_string()
}

fn wide128_to_string(buf: &[u16; 128]) -> String {
    let end = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
    String::from_utf16_lossy(&buf[..end]).trim().to_string()
}

// ============================================================================
// HOST COM OBJECTS (`IHostApplication`, `IComponentHandler`, `IBStream`, `IPlugFrame`)
// ============================================================================

#[repr(C)]
struct HostApplicationCom {
    vtbl: *const IHostApplicationVtbl,
    ref_count: AtomicU32,
}

static HOST_APP_VTBL: IHostApplicationVtbl = IHostApplicationVtbl {
    base: FUnknownVtbl {
        queryInterface: host_app_qi,
        addRef: host_app_add_ref,
        release: host_app_release,
    },
    getName: host_app_get_name,
    createInstance: host_app_create_instance,
};

unsafe extern "system" fn host_app_qi(
    this: *mut c_void,
    iid: *const TUID,
    obj: *mut *mut c_void,
) -> tresult {
    unsafe {
        if obj.is_null() || iid.is_null() {
            return K_INVALID_ARGUMENT;
        }
        let id = *iid;
        if id == IID_FUNKNOWN || id == IID_IHOST_APPLICATION {
            let _ = host_app_add_ref(this);
            *obj = this;
            K_RESULT_OK
        } else {
            *obj = std::ptr::null_mut();
            K_NO_INTERFACE
        }
    }
}

unsafe extern "system" fn host_app_add_ref(this: *mut c_void) -> u32 {
    unsafe {
        let me = &*(this as *const HostApplicationCom);
        me.ref_count.fetch_add(1, Ordering::Relaxed) + 1
    }
}

unsafe extern "system" fn host_app_release(this: *mut c_void) -> u32 {
    unsafe {
        let me = &*(this as *const HostApplicationCom);
        me.ref_count.fetch_sub(1, Ordering::Relaxed).saturating_sub(1)
    }
}

unsafe extern "system" fn host_app_get_name(_this: *mut c_void, name: *mut u16) -> tresult {
    if name.is_null() {
        return K_INVALID_ARGUMENT;
    }
    let label = "ToneForge Studio v3";
    let utf16: Vec<u16> = label.encode_utf16().collect();
    let copy_len = utf16.len().min(127);
    unsafe {
        std::ptr::copy_nonoverlapping(utf16.as_ptr(), name, copy_len);
        *name.add(copy_len) = 0;
    }
    K_RESULT_OK
}

unsafe extern "system" fn host_app_create_instance(
    _this: *mut c_void,
    _cid: *const TUID,
    _iid: *const TUID,
    obj: *mut *mut c_void,
) -> tresult {
    unsafe {
        if !obj.is_null() {
            *obj = std::ptr::null_mut();
        }
    }
    K_NO_INTERFACE
}

#[repr(C)]
struct ComponentHandlerCom {
    vtbl: *const IComponentHandlerVtbl,
    ref_count: AtomicU32,
    gui_edit_tx: parking_lot::Mutex<rtrb::Producer<(ParamID, ParamValue)>>,
}

static COMP_HANDLER_VTBL: IComponentHandlerVtbl = IComponentHandlerVtbl {
    base: FUnknownVtbl {
        queryInterface: comp_handler_qi,
        addRef: comp_handler_add_ref,
        release: comp_handler_release,
    },
    beginEdit: comp_handler_begin_edit,
    performEdit: comp_handler_perform_edit,
    endEdit: comp_handler_end_edit,
    restartComponent: comp_handler_restart,
};

unsafe extern "system" fn comp_handler_qi(
    this: *mut c_void,
    iid: *const TUID,
    obj: *mut *mut c_void,
) -> tresult {
    unsafe {
        if obj.is_null() || iid.is_null() {
            return K_INVALID_ARGUMENT;
        }
        let id = *iid;
        if id == IID_FUNKNOWN || id == IID_ICOMPONENT_HANDLER {
            let _ = comp_handler_add_ref(this);
            *obj = this;
            K_RESULT_OK
        } else {
            *obj = std::ptr::null_mut();
            K_NO_INTERFACE
        }
    }
}

unsafe extern "system" fn comp_handler_add_ref(this: *mut c_void) -> u32 {
    unsafe {
        let me = &*(this as *const ComponentHandlerCom);
        me.ref_count.fetch_add(1, Ordering::Relaxed) + 1
    }
}

unsafe extern "system" fn comp_handler_release(this: *mut c_void) -> u32 {
    unsafe {
        let me = &*(this as *const ComponentHandlerCom);
        me.ref_count.fetch_sub(1, Ordering::Relaxed).saturating_sub(1)
    }
}

unsafe extern "system" fn comp_handler_begin_edit(_this: *mut c_void, _id: ParamID) -> tresult {
    K_RESULT_OK
}

unsafe extern "system" fn comp_handler_perform_edit(
    this: *mut c_void,
    id: ParamID,
    value_normalized: ParamValue,
) -> tresult {
    // Called on the Win32 UI thread when the user turns a knob in the native plugin GUI;
    // forwards lock-free via `rtrb` to the RT `IAudioProcessor::process` queue.
    unsafe {
        let me = &*(this as *const ComponentHandlerCom);
        if let Some(mut tx) = me.gui_edit_tx.try_lock() {
            let _ = tx.push((id, value_normalized.clamp(0.0, 1.0)));
        }
    }
    K_RESULT_OK
}

unsafe extern "system" fn comp_handler_end_edit(_this: *mut c_void, _id: ParamID) -> tresult {
    K_RESULT_OK
}

unsafe extern "system" fn comp_handler_restart(_this: *mut c_void, _flags: i32) -> tresult {
    K_RESULT_OK
}

/// Stack/Heap `IBStream` COM adapter backed by `Vec<u8>` for `IComponent::getState`/`setState`.
#[repr(C)]
struct MemoryBStream {
    vtbl: *const IBStreamVtbl,
    data: Vec<u8>,
    cursor: usize,
}

static MEM_BSTREAM_VTBL: IBStreamVtbl = IBStreamVtbl {
    base: FUnknownVtbl {
        queryInterface: mem_bstream_qi,
        addRef: mem_bstream_add_ref,
        release: mem_bstream_release,
    },
    read: mem_bstream_read,
    write: mem_bstream_write,
    seek: mem_bstream_seek,
    tell: mem_bstream_tell,
};

impl MemoryBStream {
    fn new_empty() -> Self {
        Self {
            vtbl: &MEM_BSTREAM_VTBL,
            data: Vec::with_capacity(256),
            cursor: 0,
        }
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            vtbl: &MEM_BSTREAM_VTBL,
            data: bytes.to_vec(),
            cursor: 0,
        }
    }
}

unsafe extern "system" fn mem_bstream_qi(
    this: *mut c_void,
    iid: *const TUID,
    obj: *mut *mut c_void,
) -> tresult {
    unsafe {
        if obj.is_null() || iid.is_null() {
            return K_INVALID_ARGUMENT;
        }
        let id = *iid;
        if id == IID_FUNKNOWN || id == IID_IBSTREAM {
            *obj = this;
            K_RESULT_OK
        } else {
            *obj = std::ptr::null_mut();
            K_NO_INTERFACE
        }
    }
}

unsafe extern "system" fn mem_bstream_add_ref(_this: *mut c_void) -> u32 {
    2
}

unsafe extern "system" fn mem_bstream_release(_this: *mut c_void) -> u32 {
    1
}

unsafe extern "system" fn mem_bstream_read(
    this: *mut c_void,
    buffer: *mut c_void,
    num_bytes: i32,
    num_bytes_read: *mut i32,
) -> tresult {
    unsafe {
        if this.is_null() || (buffer.is_null() && num_bytes > 0) || num_bytes < 0 {
            return K_INVALID_ARGUMENT;
        }
        let me = &mut *(this as *mut MemoryBStream);
        let avail = me.data.len().saturating_sub(me.cursor);
        let to_read = avail.min(num_bytes as usize);
        if to_read > 0 {
            std::ptr::copy_nonoverlapping(
                me.data[me.cursor..].as_ptr(),
                buffer as *mut u8,
                to_read,
            );
            me.cursor += to_read;
        }
        if !num_bytes_read.is_null() {
            *num_bytes_read = to_read as i32;
        }
        K_RESULT_OK
    }
}

unsafe extern "system" fn mem_bstream_write(
    this: *mut c_void,
    buffer: *mut c_void,
    num_bytes: i32,
    num_bytes_written: *mut i32,
) -> tresult {
    unsafe {
        if this.is_null() || (buffer.is_null() && num_bytes > 0) || num_bytes < 0 {
            return K_INVALID_ARGUMENT;
        }
        let me = &mut *(this as *mut MemoryBStream);
        let n = num_bytes as usize;
        if n > 0 {
            let slice = std::slice::from_raw_parts(buffer as *const u8, n);
            if me.cursor == me.data.len() {
                me.data.extend_from_slice(slice);
            } else {
                let end = me.cursor + n;
                if end > me.data.len() {
                    me.data.resize(end, 0);
                }
                me.data[me.cursor..end].copy_from_slice(slice);
            }
            me.cursor += n;
        }
        if !num_bytes_written.is_null() {
            *num_bytes_written = n as i32;
        }
        K_RESULT_OK
    }
}

unsafe extern "system" fn mem_bstream_seek(
    this: *mut c_void,
    pos: i64,
    mode: i32,
    result: *mut i64,
) -> tresult {
    unsafe {
        let me = &mut *(this as *mut MemoryBStream);
        let base: i64 = match mode {
            0 => 0,
            1 => me.cursor as i64,
            2 => me.data.len() as i64,
            _ => return K_INVALID_ARGUMENT,
        };
        let next = (base + pos).clamp(0, me.data.len() as i64) as usize;
        me.cursor = next;
        if !result.is_null() {
            *result = next as i64;
        }
        K_RESULT_OK
    }
}

unsafe extern "system" fn mem_bstream_tell(this: *mut c_void, pos: *mut i64) -> tresult {
    unsafe {
        if pos.is_null() {
            return K_INVALID_ARGUMENT;
        }
        let me = &*(this as *const MemoryBStream);
        *pos = me.cursor as i64;
        K_RESULT_OK
    }
}

/// Host `IPlugFrame` COM implementation allowing VST3 plugin editors to request window resize.
#[repr(C)]
struct HostPlugFrameCom {
    vtbl: *const IPlugFrameVtbl,
    hwnd_ptr: *mut NativePluginWindow,
}

static HOST_PLUG_FRAME_VTBL: IPlugFrameVtbl = IPlugFrameVtbl {
    base: FUnknownVtbl {
        queryInterface: plug_frame_qi,
        addRef: mem_bstream_add_ref,
        release: mem_bstream_release,
    },
    resizeView: plug_frame_resize_view,
};

unsafe extern "system" fn plug_frame_qi(
    this: *mut c_void,
    iid: *const TUID,
    obj: *mut *mut c_void,
) -> tresult {
    unsafe {
        if obj.is_null() || iid.is_null() {
            return K_INVALID_ARGUMENT;
        }
        let id = *iid;
        if id == IID_FUNKNOWN || id == IID_IPLUG_FRAME {
            *obj = this;
            K_RESULT_OK
        } else {
            *obj = std::ptr::null_mut();
            K_NO_INTERFACE
        }
    }
}

unsafe extern "system" fn plug_frame_resize_view(
    this: *mut c_void,
    view: *mut c_void,
    new_size: *mut ViewRect,
) -> tresult {
    unsafe {
        if this.is_null() || new_size.is_null() {
            return K_INVALID_ARGUMENT;
        }
        let me = &mut *(this as *mut HostPlugFrameCom);
        let rect = *new_size;
        let w = (rect.right - rect.left).max(160) as u32;
        let h = (rect.bottom - rect.top).max(96) as u32;
        if !me.hwnd_ptr.is_null() {
            (*me.hwnd_ptr).resize_client(w, h, true);
        }
        if !view.is_null() {
            let vtbl = *(view as *const *const IPlugViewVtbl);
            let _ = ((*vtbl).onSize)(view, new_size);
        }
        K_RESULT_OK
    }
}

// ============================================================================
// PRE-ALLOCATED ZERO-ALLOCATION RT PROCESS ARENA (`IParameterChanges` & `IEventList`)
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy)]
struct PreallocParamQueue {
    vtbl: *const IParamValueQueueVtbl,
    param_id: ParamID,
    point_count: usize,
    sample_offsets: [i32; MAX_POINTS_PER_QUEUE],
    values: [ParamValue; MAX_POINTS_PER_QUEUE],
}

static PARAM_QUEUE_VTBL: IParamValueQueueVtbl = IParamValueQueueVtbl {
    base: FUnknownVtbl {
        queryInterface: param_queue_qi,
        addRef: mem_bstream_add_ref,
        release: mem_bstream_release,
    },
    getParameterId: param_queue_get_id,
    getPointCount: param_queue_get_count,
    getPoint: param_queue_get_point,
    addPoint: param_queue_add_point,
};

impl Default for PreallocParamQueue {
    fn default() -> Self {
        Self {
            vtbl: &PARAM_QUEUE_VTBL,
            param_id: 0,
            point_count: 0,
            sample_offsets: [0; MAX_POINTS_PER_QUEUE],
            values: [0.0; MAX_POINTS_PER_QUEUE],
        }
    }
}

unsafe extern "system" fn param_queue_qi(
    this: *mut c_void,
    iid: *const TUID,
    obj: *mut *mut c_void,
) -> tresult {
    unsafe {
        if obj.is_null() || iid.is_null() {
            return K_INVALID_ARGUMENT;
        }
        let id = *iid;
        if id == IID_FUNKNOWN || id == IID_IPARAM_VALUE_QUEUE {
            *obj = this;
            K_RESULT_OK
        } else {
            *obj = std::ptr::null_mut();
            K_NO_INTERFACE
        }
    }
}

unsafe extern "system" fn param_queue_get_id(this: *mut c_void) -> ParamID {
    unsafe { (*(this as *const PreallocParamQueue)).param_id }
}

unsafe extern "system" fn param_queue_get_count(this: *mut c_void) -> i32 {
    unsafe { (*(this as *const PreallocParamQueue)).point_count as i32 }
}

unsafe extern "system" fn param_queue_get_point(
    this: *mut c_void,
    index: i32,
    sample_offset: *mut i32,
    value: *mut ParamValue,
) -> tresult {
    unsafe {
        let q = &*(this as *const PreallocParamQueue);
        if index < 0 || (index as usize) >= q.point_count {
            return K_INVALID_ARGUMENT;
        }
        let i = index as usize;
        if !sample_offset.is_null() {
            *sample_offset = q.sample_offsets[i];
        }
        if !value.is_null() {
            *value = q.values[i];
        }
        K_RESULT_OK
    }
}

unsafe extern "system" fn param_queue_add_point(
    this: *mut c_void,
    sample_offset: i32,
    value: ParamValue,
    index: *mut i32,
) -> tresult {
    unsafe {
        let q = &mut *(this as *mut PreallocParamQueue);
        if q.point_count >= MAX_POINTS_PER_QUEUE {
            return K_RESULT_FALSE;
        }
        let idx = q.point_count;
        q.sample_offsets[idx] = sample_offset;
        q.values[idx] = value;
        q.point_count += 1;
        if !index.is_null() {
            *index = idx as i32;
        }
        K_RESULT_OK
    }
}

#[repr(C)]
struct PreallocParameterChanges {
    vtbl: *const IParameterChangesVtbl,
    queue_count: usize,
    queues: [PreallocParamQueue; MAX_VST3_PARAM_QUEUES],
}

static PARAM_CHANGES_VTBL: IParameterChangesVtbl = IParameterChangesVtbl {
    base: FUnknownVtbl {
        queryInterface: param_changes_qi,
        addRef: mem_bstream_add_ref,
        release: mem_bstream_release,
    },
    getParameterCount: param_changes_get_count,
    getParameterData: param_changes_get_data,
    addParameterData: param_changes_add_data,
};

impl PreallocParameterChanges {
    fn new() -> Self {
        Self {
            vtbl: &PARAM_CHANGES_VTBL,
            queue_count: 0,
            queues: [PreallocParamQueue::default(); MAX_VST3_PARAM_QUEUES],
        }
    }

    #[inline]
    fn clear(&mut self) {
        for i in 0..self.queue_count {
            self.queues[i].point_count = 0;
        }
        self.queue_count = 0;
    }

    #[inline]
    fn push_point(&mut self, param_id: ParamID, sample_offset: i32, norm_val: ParamValue) {
        for i in 0..self.queue_count {
            if self.queues[i].param_id == param_id {
                let q = &mut self.queues[i];
                if q.point_count < MAX_POINTS_PER_QUEUE {
                    q.sample_offsets[q.point_count] = sample_offset;
                    q.values[q.point_count] = norm_val;
                    q.point_count += 1;
                }
                return;
            }
        }
        if self.queue_count < MAX_VST3_PARAM_QUEUES {
            let q = &mut self.queues[self.queue_count];
            q.vtbl = &PARAM_QUEUE_VTBL;
            q.param_id = param_id;
            q.sample_offsets[0] = sample_offset;
            q.values[0] = norm_val;
            q.point_count = 1;
            self.queue_count += 1;
        }
    }
}

unsafe extern "system" fn param_changes_qi(
    this: *mut c_void,
    iid: *const TUID,
    obj: *mut *mut c_void,
) -> tresult {
    unsafe {
        if obj.is_null() || iid.is_null() {
            return K_INVALID_ARGUMENT;
        }
        let id = *iid;
        if id == IID_FUNKNOWN || id == IID_IPARAMETER_CHANGES {
            *obj = this;
            K_RESULT_OK
        } else {
            *obj = std::ptr::null_mut();
            K_NO_INTERFACE
        }
    }
}

unsafe extern "system" fn param_changes_get_count(this: *mut c_void) -> i32 {
    unsafe { (*(this as *const PreallocParameterChanges)).queue_count as i32 }
}

unsafe extern "system" fn param_changes_get_data(this: *mut c_void, index: i32) -> *mut c_void {
    unsafe {
        let pc = &mut *(this as *mut PreallocParameterChanges);
        if index < 0 || (index as usize) >= pc.queue_count {
            return std::ptr::null_mut();
        }
        (&mut pc.queues[index as usize]) as *mut PreallocParamQueue as *mut c_void
    }
}

unsafe extern "system" fn param_changes_add_data(
    this: *mut c_void,
    id: *const ParamID,
    index: *mut i32,
) -> *mut c_void {
    unsafe {
        if id.is_null() {
            return std::ptr::null_mut();
        }
        let pid = *id;
        let pc = &mut *(this as *mut PreallocParameterChanges);
        for i in 0..pc.queue_count {
            if pc.queues[i].param_id == pid {
                if !index.is_null() {
                    *index = i as i32;
                }
                return (&mut pc.queues[i]) as *mut PreallocParamQueue as *mut c_void;
            }
        }
        if pc.queue_count >= MAX_VST3_PARAM_QUEUES {
            return std::ptr::null_mut();
        }
        let idx = pc.queue_count;
        pc.queues[idx].vtbl = &PARAM_QUEUE_VTBL;
        pc.queues[idx].param_id = pid;
        pc.queues[idx].point_count = 0;
        pc.queue_count += 1;
        if !index.is_null() {
            *index = idx as i32;
        }
        (&mut pc.queues[idx]) as *mut PreallocParamQueue as *mut c_void
    }
}

#[repr(C)]
struct PreallocEventList {
    vtbl: *const IEventListVtbl,
    event_count: usize,
    events: [Vst3Event; MAX_VST3_RT_EVENTS],
}

static EVENT_LIST_VTBL: IEventListVtbl = IEventListVtbl {
    base: FUnknownVtbl {
        queryInterface: event_list_qi,
        addRef: mem_bstream_add_ref,
        release: mem_bstream_release,
    },
    getEventCount: event_list_get_count,
    getEvent: event_list_get_event,
    addEvent: event_list_add_event,
};

impl PreallocEventList {
    fn new() -> Self {
        Self {
            vtbl: &EVENT_LIST_VTBL,
            event_count: 0,
            events: [Vst3Event::default(); MAX_VST3_RT_EVENTS],
        }
    }

    #[inline]
    fn clear(&mut self) {
        self.event_count = 0;
    }

    #[inline]
    fn push(&mut self, ev: Vst3Event) {
        if self.event_count < MAX_VST3_RT_EVENTS {
            self.events[self.event_count] = ev;
            self.event_count += 1;
        }
    }
}

unsafe extern "system" fn event_list_qi(
    this: *mut c_void,
    iid: *const TUID,
    obj: *mut *mut c_void,
) -> tresult {
    unsafe {
        if obj.is_null() || iid.is_null() {
            return K_INVALID_ARGUMENT;
        }
        let id = *iid;
        if id == IID_FUNKNOWN || id == IID_IEVENT_LIST {
            *obj = this;
            K_RESULT_OK
        } else {
            *obj = std::ptr::null_mut();
            K_NO_INTERFACE
        }
    }
}

unsafe extern "system" fn event_list_get_count(this: *mut c_void) -> i32 {
    unsafe { (*(this as *const PreallocEventList)).event_count as i32 }
}

unsafe extern "system" fn event_list_get_event(
    this: *mut c_void,
    index: i32,
    e: *mut Vst3Event,
) -> tresult {
    unsafe {
        let el = &*(this as *const PreallocEventList);
        if e.is_null() || index < 0 || (index as usize) >= el.event_count {
            return K_INVALID_ARGUMENT;
        }
        *e = el.events[index as usize];
        K_RESULT_OK
    }
}

unsafe extern "system" fn event_list_add_event(
    this: *mut c_void,
    e: *mut Vst3Event,
) -> tresult {
    unsafe {
        if e.is_null() {
            return K_INVALID_ARGUMENT;
        }
        let el = &mut *(this as *mut PreallocEventList);
        if el.event_count >= MAX_VST3_RT_EVENTS {
            return K_RESULT_FALSE;
        }
        el.events[el.event_count] = *e;
        el.event_count += 1;
        K_RESULT_OK
    }
}

#[repr(C)]
struct Vst3RtArena {
    in_left: [f32; MAX_BLOCK_SIZE],
    in_right: [f32; MAX_BLOCK_SIZE],
    out_left: [f32; MAX_BLOCK_SIZE],
    out_right: [f32; MAX_BLOCK_SIZE],
    in_ptrs: [*mut f32; 2],
    out_ptrs: [*mut f32; 2],
    /// Aux (sidechain) stereo input bus, fed from the mixer's sidechain buffer.
    aux_left: [f32; MAX_BLOCK_SIZE],
    aux_right: [f32; MAX_BLOCK_SIZE],
    aux_ptrs: [*mut f32; 2],
    in_param_changes: PreallocParameterChanges,
    out_param_changes: PreallocParameterChanges,
    in_events: PreallocEventList,
    out_events: PreallocEventList,
    param_ids: [ParamID; 128],
    param_count: usize,
}

impl Vst3RtArena {
    fn new_boxed() -> Box<Self> {
        let mut b = Box::new(Self {
            in_left: [0.0; MAX_BLOCK_SIZE],
            in_right: [0.0; MAX_BLOCK_SIZE],
            out_left: [0.0; MAX_BLOCK_SIZE],
            out_right: [0.0; MAX_BLOCK_SIZE],
            in_ptrs: [std::ptr::null_mut(); 2],
            out_ptrs: [std::ptr::null_mut(); 2],
            aux_left: [0.0; MAX_BLOCK_SIZE],
            aux_right: [0.0; MAX_BLOCK_SIZE],
            aux_ptrs: [std::ptr::null_mut(); 2],
            in_param_changes: PreallocParameterChanges::new(),
            out_param_changes: PreallocParameterChanges::new(),
            in_events: PreallocEventList::new(),
            out_events: PreallocEventList::new(),
            param_ids: [0; 128],
            param_count: 0,
        });
        b.in_ptrs[0] = b.in_left.as_mut_ptr();
        b.in_ptrs[1] = b.in_right.as_mut_ptr();
        b.out_ptrs[0] = b.out_left.as_mut_ptr();
        b.out_ptrs[1] = b.out_right.as_mut_ptr();
        b.aux_ptrs[0] = b.aux_left.as_mut_ptr();
        b.aux_ptrs[1] = b.aux_right.as_mut_ptr();
        b
    }
}

/// Pre-allocated buffers for every audio output bus beyond the main stereo bus (multi-out
/// instruments): the plug-in renders into them, the host keeps the main bus (`0` RT allocs).
struct ExtraOutputBuses {
    /// One `MAX_BLOCK_SIZE` buffer per channel of every extra bus (owns the memory behind
    /// `channel_ptrs`).
    _channel_bufs: Vec<Vec<f32>>,
    /// Channel pointer table (`channel_bufs[i].as_mut_ptr()`), sliced per bus.
    channel_ptrs: Vec<*mut f32>,
    /// `(first pointer index, channel count)` per extra bus.
    layout: Vec<(usize, i32)>,
    /// Bus descriptor table passed as `ProcessData::outputs` (`[main, extra...]`).
    buses: Vec<AudioBusBuffers>,
}

impl ExtraOutputBuses {
    fn new(extra_channel_counts: &[i32]) -> Self {
        let total: usize = extra_channel_counts.iter().map(|c| (*c).max(0) as usize).sum();
        let mut channel_bufs: Vec<Vec<f32>> = (0..total).map(|_| vec![0.0f32; MAX_BLOCK_SIZE]).collect();
        let channel_ptrs: Vec<*mut f32> = channel_bufs.iter_mut().map(|b| b.as_mut_ptr()).collect();
        let mut layout = Vec::with_capacity(extra_channel_counts.len());
        let mut cursor = 0usize;
        for &c in extra_channel_counts {
            layout.push((cursor, c.max(0)));
            cursor += c.max(0) as usize;
        }
        let buses = Vec::with_capacity(1 + extra_channel_counts.len());
        Self {
            _channel_bufs: channel_bufs,
            channel_ptrs,
            layout,
            buses,
        }
    }
}

type GetPluginFactoryFn = unsafe extern "system" fn() -> *mut c_void;
type InitExitDllFn = unsafe extern "system" fn() -> bool;

unsafe fn com_release(ptr: *mut c_void) {
    if !ptr.is_null() {
        unsafe {
            let unk = *(ptr as *const *const FUnknownVtbl);
            let _ = ((*unk).release)(ptr);
        }
    }
}

unsafe fn com_query(ptr: *mut c_void, iid: &TUID) -> *mut c_void {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    let mut out: *mut c_void = std::ptr::null_mut();
    unsafe {
        let unk = *(ptr as *const *const FUnknownVtbl);
        let rc = ((*unk).queryInterface)(ptr, iid as *const TUID, &mut out);
        if rc == K_RESULT_OK && !out.is_null() {
            out
        } else {
            std::ptr::null_mut()
        }
    }
}

/// Enumerates all Steinberg VST3 `"Audio Module Class"` plugins inside `vst3_path` via `LoadLibraryW`.
pub fn enumerate_vst3_plugins_in_library(
    vst3_path: &Path,
) -> Result<Vec<Vst3DescriptorSummary>, String> {
    let dll_path = resolve_vst3_binary_path(vst3_path)?;
    let wide = wide_null_path(&dll_path);
    let hmodule = unsafe { LoadLibraryW(wide.as_ptr()) };
    if hmodule.is_null() {
        return Err(format!(
            "LoadLibraryW failed for VST3 binary: {}",
            dll_path.display()
        ));
    }

    let init_dll: Option<InitExitDllFn> = unsafe {
        GetProcAddress(hmodule, b"InitDll\0".as_ptr()).map(|f| std::mem::transmute(f))
    };
    let exit_dll: Option<InitExitDllFn> = unsafe {
        GetProcAddress(hmodule, b"ExitDll\0".as_ptr()).map(|f| std::mem::transmute(f))
    };
    if let Some(init_fn) = init_dll {
        let ok = unsafe { init_fn() };
        if !ok {
            unsafe {
                let _ = FreeLibrary(hmodule);
            }
            return Err(format!("VST3 InitDll() returned false for {}", dll_path.display()));
        }
    }

    let get_factory: Option<GetPluginFactoryFn> = unsafe {
        GetProcAddress(hmodule, b"GetPluginFactory\0".as_ptr()).map(|f| std::mem::transmute(f))
    };
    let Some(get_factory_fn) = get_factory else {
        if let Some(ex) = exit_dll {
            unsafe {
                let _ = ex();
            }
        }
        unsafe {
            let _ = FreeLibrary(hmodule);
        }
        return Err(format!(
            "GetPluginFactory export not found in {}",
            dll_path.display()
        ));
    };

    let factory_ptr = unsafe { get_factory_fn() };
    if factory_ptr.is_null() {
        if let Some(ex) = exit_dll {
            unsafe {
                let _ = ex();
            }
        }
        unsafe {
            let _ = FreeLibrary(hmodule);
        }
        return Err("GetPluginFactory() returned NULL".to_string());
    }

    let mut host_app = Box::new(HostApplicationCom {
        vtbl: &HOST_APP_VTBL,
        ref_count: AtomicU32::new(1),
    });
    let host_app_ptr = (&mut *host_app) as *mut HostApplicationCom as *mut c_void;

    let mut out = Vec::new();
    unsafe {
        let fvtbl = *(factory_ptr as *const *const IPluginFactoryVtbl);
        let mut finfo: PFactoryInfo = std::mem::zeroed();
        let _ = ((*fvtbl).getFactoryInfo)(factory_ptr, &mut finfo);
        let vendor = fixed_cchar_to_string(&finfo.vendor);

        let n_classes = ((*fvtbl).countClasses)(factory_ptr).max(0);
        for i in 0..n_classes {
            let mut cinfo: PClassInfo = std::mem::zeroed();
            if ((*fvtbl).getClassInfo)(factory_ptr, i, &mut cinfo) != K_RESULT_OK {
                continue;
            }
            let category = fixed_cchar_to_string(&cinfo.category);
            if !category.eq_ignore_ascii_case("Audio Module Class") {
                continue;
            }
            let name = fixed_cchar_to_string(&cinfo.name);
            let cid_hex = tuid_to_hex(&cinfo.cid);

            let mut latency_samples = 0u32;
            let mut num_params = 0u32;
            let mut is_instrument = false;

            let mut comp_ptr: *mut c_void = std::ptr::null_mut();
            if ((*fvtbl).createInstance)(
                factory_ptr,
                &cinfo.cid,
                &IID_ICOMPONENT,
                &mut comp_ptr,
            ) == K_RESULT_OK
                && !comp_ptr.is_null()
            {
                let cvtbl = *(comp_ptr as *const *const IComponentVtbl);
                if ((*cvtbl).base.initialize)(comp_ptr, host_app_ptr) == K_RESULT_OK {
                    let ev_in_buses = ((*cvtbl).getBusCount)(comp_ptr, K_EVENT, K_INPUT);
                    if ev_in_buses > 0 {
                        is_instrument = true;
                    }
                    let proc_ptr = com_query(comp_ptr, &IID_IAUDIO_PROCESSOR);
                    if !proc_ptr.is_null() {
                        let pvtbl = *(proc_ptr as *const *const IAudioProcessorVtbl);
                        let mut setup = ProcessSetup {
                            processMode: K_REALTIME,
                            symbolicSampleSize: K_SAMPLE_32,
                            maxSamplesPerBlock: 512,
                            sampleRate: 48_000.0,
                        };
                        let _ = ((*pvtbl).setupProcessing)(proc_ptr, &mut setup);
                        latency_samples = ((*pvtbl).getLatencySamples)(proc_ptr);
                        com_release(proc_ptr);
                    }
                    let ctrl_ptr = com_query(comp_ptr, &IID_IEDIT_CONTROLLER);
                    if !ctrl_ptr.is_null() {
                        let evtbl = *(ctrl_ptr as *const *const IEditControllerVtbl);
                        num_params = ((*evtbl).getParameterCount)(ctrl_ptr).max(0) as u32;
                        com_release(ctrl_ptr);
                    }
                    let _ = ((*cvtbl).base.terminate)(comp_ptr);
                }
                com_release(comp_ptr);
            }

            out.push(Vst3DescriptorSummary {
                cid_hex,
                name,
                vendor: vendor.clone(),
                category,
                is_instrument,
                latency_samples,
                num_params,
            });
        }
        com_release(factory_ptr);
        if let Some(ex) = exit_dll {
            let _ = ex();
        }
        let _ = FreeLibrary(hmodule);
    }

    Ok(out)
}

const TF_VST3_STATE_MAGIC: &[u8; 8] = b"TFVST3ST";

/// Native editor window state (`IPlugView` + `IPlugFrame` + HWND) for one controller.
#[derive(Default)]
pub struct Vst3EditorSession {
    plug_view_ptr: *mut c_void,
    plug_frame: Option<Box<HostPlugFrameCom>>,
    editor_window: Option<NativePluginWindow>,
}

impl Vst3EditorSession {
    /// Creates (or re-shows) the `"editor"` view of `controller_ptr` inside a native window.
    /// Must run on the GUI (STA) thread.
    pub fn open(
        &mut self,
        controller_ptr: *mut c_void,
        plugin_name: &str,
        owner_hwnd: Option<isize>,
        visible: bool,
    ) -> Result<isize, String> {
        if controller_ptr.is_null() {
            return Err("Plugin does not expose IEditController".to_string());
        }
        if let Some(ref win) = self.editor_window {
            if win.is_valid() {
                if visible {
                    win.show();
                }
                return Ok(win.hwnd_isize());
            }
        }
        self.close();

        let view_ptr = unsafe {
            let evtbl = *(controller_ptr as *const *const IEditControllerVtbl);
            ((*evtbl).createView)(
                controller_ptr,
                VST3_VIEW_TYPE_EDITOR.as_ptr() as *const c_char,
            )
        };
        if view_ptr.is_null() {
            return Err("IEditController::createView(\"editor\") returned NULL".to_string());
        }

        let platform_ptr = VST3_PLATFORM_TYPE_HWND.as_ptr() as *const c_char;
        let sup_rc = unsafe {
            let vvtbl = *(view_ptr as *const *const IPlugViewVtbl);
            ((*vvtbl).isPlatformTypeSupported)(view_ptr, platform_ptr)
        };
        if sup_rc != K_RESULT_OK {
            unsafe { com_release(view_ptr) };
            return Err("VST3 IPlugView does not support Win32 HWND".to_string());
        }

        // HiDPI: tell the editor the monitor scale *before* asking for its size so the window is
        // created at the right physical size (IPlugViewContentScaleSupport, VST 3.6.6+).
        let initial_scale = dpi_scale_for(owner_hwnd.map(|h| h as *mut c_void));
        unsafe { set_view_content_scale(view_ptr, initial_scale) };

        let mut rect = ViewRect {
            left: 0,
            top: 0,
            right: 640,
            bottom: 400,
        };
        let resizable = unsafe {
            let vvtbl = *(view_ptr as *const *const IPlugViewVtbl);
            let _ = ((*vvtbl).getSize)(view_ptr, &mut rect);
            ((*vvtbl).canResize)(view_ptr) == K_RESULT_OK
        };

        let width = (rect.right - rect.left).max(160) as u32;
        let height = (rect.bottom - rect.top).max(96) as u32;
        let title = format!("{} (VST3)", plugin_name);
        let mut win =
            NativePluginWindow::open(&title, width, height, owner_hwnd, resizable, visible)?;

        let mut frame_box = Box::new(HostPlugFrameCom {
            vtbl: &HOST_PLUG_FRAME_VTBL,
            hwnd_ptr: (&mut win) as *mut NativePluginWindow,
        });
        unsafe {
            let vvtbl = *(view_ptr as *const *const IPlugViewVtbl);
            let _ = ((*vvtbl).setFrame)(
                view_ptr,
                (&mut *frame_box) as *mut HostPlugFrameCom as *mut c_void,
            );
            let att_rc = ((*vvtbl).attached)(view_ptr, win.hwnd_raw(), platform_ptr);
            if att_rc != K_RESULT_OK {
                let _ = ((*vvtbl).setFrame)(view_ptr, std::ptr::null_mut());
                com_release(view_ptr);
                win.destroy_now();
                return Err(format!("IPlugView::attached(HWND) failed ({att_rc})"));
            }
        }

        // The editor window may land on a monitor with a different scale than its owner.
        let window_scale = dpi_scale_for(Some(win.hwnd_raw()));
        if (window_scale - initial_scale).abs() > 0.01 {
            unsafe {
                if set_view_content_scale(view_ptr, window_scale) {
                    let vvtbl = *(view_ptr as *const *const IPlugViewVtbl);
                    let mut r = ViewRect::default();
                    if ((*vvtbl).getSize)(view_ptr, &mut r) == K_RESULT_OK {
                        win.resize_client(
                            (r.right - r.left).max(160) as u32,
                            (r.bottom - r.top).max(96) as u32,
                            resizable,
                        );
                    }
                }
            }
        }

        let hwnd_val = win.hwnd_isize();
        self.editor_window = Some(win);
        frame_box.hwnd_ptr = self
            .editor_window
            .as_mut()
            .map(|w| w as *mut NativePluginWindow)
            .unwrap_or(std::ptr::null_mut());
        self.plug_frame = Some(frame_box);
        self.plug_view_ptr = view_ptr;
        Ok(hwnd_val)
        }

    pub fn pump(&self) -> usize {
        if let Some(ref win) = self.editor_window {
            win.pump_messages()
        } else {
            0
        }
    }

    pub fn close(&mut self) {
        if !self.plug_view_ptr.is_null() {
            unsafe {
                let vvtbl = *(self.plug_view_ptr as *const *const IPlugViewVtbl);
                let _ = ((*vvtbl).removed)(self.plug_view_ptr);
                let _ = ((*vvtbl).setFrame)(self.plug_view_ptr, std::ptr::null_mut());
                com_release(self.plug_view_ptr);
            }
            self.plug_view_ptr = std::ptr::null_mut();
        }
        self.plug_frame = None;
        if let Some(mut win) = self.editor_window.take() {
            win.destroy_now();
        }
    }
}

/// Editor attached to the *live* (real-time) instance of a hosted VST3: the GUI thread talks to the
/// same `IEditController` the audio thread's processor is connected to, so presets, sample loads and
/// any state edited in the plug-in window act on the sound that is playing. GUI parameter edits
/// reach the processor through the instance's lock-free `IComponentHandler` queue.
pub struct Vst3LiveEditor {
    controller_ptr: *mut c_void,
    module_pin: HMODULE,
    plugin_name: String,
    param_ids: Vec<u32>,
    live_guard: std::sync::Arc<parking_lot::Mutex<bool>>,
    session: Vst3EditorSession,
}

// SAFETY: used only from the single GUI STA thread after being handed over once.
unsafe impl Send for Vst3LiveEditor {}

impl Vst3LiveEditor {
    pub fn open_editor(&mut self, owner_hwnd: Option<isize>, visible: bool) -> Result<isize, String> {
        let alive = self.live_guard.lock();
        if !*alive {
            return Err("Plug-in instance was removed".to_string());
        }
        self.session
            .open(self.controller_ptr, &self.plugin_name, owner_hwnd, visible)
    }

    pub fn close_editor(&mut self) {
        let alive = self.live_guard.lock();
        if *alive {
            self.session.close();
        } else if let Some(mut win) = self.session.editor_window.take() {
            // Instance already torn down: only drop our host window, never touch the plug-in.
            self.session.plug_view_ptr = std::ptr::null_mut();
            self.session.plug_frame = None;
            win.destroy_now();
        }
    }

    pub fn pump_editor_messages(&mut self) -> usize {
        if !*self.live_guard.lock() {
            self.close_editor();
            return 0;
        }
        self.session.pump()
    }

    pub fn parameter_ids(&self) -> &[u32] {
        &self.param_ids
    }

    pub fn get_parameter_normalized(&self, param_id: u32) -> Option<f64> {
        let alive = self.live_guard.lock();
        if !*alive || self.controller_ptr.is_null() {
            return None;
        }
        unsafe {
            let evtbl = *(self.controller_ptr as *const *const IEditControllerVtbl);
            Some(((*evtbl).getParamNormalized)(self.controller_ptr, param_id))
        }
    }

    pub fn is_alive(&self) -> bool {
        *self.live_guard.lock()
    }
}

impl Drop for Vst3LiveEditor {
    fn drop(&mut self) {
        self.close_editor();
        let alive = *self.live_guard.lock();
        if alive && !self.controller_ptr.is_null() {
            unsafe { com_release(self.controller_ptr) };
        }
        if !self.module_pin.is_null() {
            unsafe {
                let _ = FreeLibrary(self.module_pin);
            }
        }
    }
}

#[cfg(target_os = "windows")]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetModuleHandleExW(flags: u32, module_name: *const u16, module: *mut HMODULE) -> i32;
}

/// Pins (adds a reference to) the module containing `addr` so it stays loaded.
fn pin_module_containing(addr: *const c_void) -> HMODULE {
    const GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS: u32 = 0x4;
    let mut h: HMODULE = std::ptr::null_mut();
    unsafe {
        if GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, addr as *const u16, &mut h) == 0 {
            return std::ptr::null_mut();
        }
    }
    h
}

/// Real-Time Hosted Steinberg VST3 3.7 Plugin Node (`UnifiedDspNode`, `0` RT heap allocations).
pub struct Vst3HostedPluginNode {
    _dll_path: PathBuf,
    cid_hex: String,
    plugin_name: String,
    hmodule: HMODULE,
    exit_dll: Option<InitExitDllFn>,
    factory_ptr: *mut c_void,
    component_ptr: *mut c_void,
    processor_ptr: *mut c_void,
    controller_ptr: *mut c_void,
    separate_controller: bool,
    comp_conn_ptr: *mut c_void,
    ctrl_conn_ptr: *mut c_void,
    _host_app: Box<HostApplicationCom>,
    _comp_handler: Box<ComponentHandlerCom>,
    gui_edit_rx: rtrb::Consumer<(ParamID, ParamValue)>,
    rt_arena: Box<Vst3RtArena>,
    param_metadata: Vec<HostedPluginParamInfo>,
    editor: Vst3EditorSession,
    /// Shared with live editors attached to this instance: set to `false` before teardown so a
    /// GUI thread never calls into a terminated controller.
    live_guard: std::sync::Arc<parking_lot::Mutex<bool>>,
    has_audio_input: bool,
    /// The plug-in exposes an aux (sidechain) audio input bus that is active.
    has_sidechain_bus: bool,
    /// Number of audio input buses passed in `ProcessData` (main + optional aux).
    num_input_buses: i32,
    extra_outputs: ExtraOutputBuses,
    active_flag: bool,
    processing_flag: bool,
    sample_rate: f64,
    max_block_size: usize,
    cached_latency_samples: usize,
}

unsafe impl Send for Vst3HostedPluginNode {}

impl Vst3HostedPluginNode {
    /// Loads a `.vst3` binary or bundle directory via `LoadLibraryW` and activates `requested_cid_hex`
    /// (if `None`, loads the first `"Audio Module Class"`).
    pub fn load(
        vst3_path: &Path,
        requested_cid_hex: Option<&str>,
        sample_rate: f64,
        max_block_size: usize,
    ) -> Result<Self, String> {
        let dll_path = resolve_vst3_binary_path(vst3_path)?;
        let wide = wide_null_path(&dll_path);
        let hmodule = unsafe { LoadLibraryW(wide.as_ptr()) };
        if hmodule.is_null() {
            return Err(format!(
                "LoadLibraryW failed for VST3 plugin: {}",
                dll_path.display()
            ));
        }

        let init_dll: Option<InitExitDllFn> = unsafe {
            GetProcAddress(hmodule, b"InitDll\0".as_ptr()).map(|f| std::mem::transmute(f))
        };
        let exit_dll: Option<InitExitDllFn> = unsafe {
            GetProcAddress(hmodule, b"ExitDll\0".as_ptr()).map(|f| std::mem::transmute(f))
        };
        if let Some(init_fn) = init_dll {
            if !unsafe { init_fn() } {
                unsafe {
                    let _ = FreeLibrary(hmodule);
                }
                return Err("VST3 InitDll() returned false".to_string());
            }
        }

        let get_factory: Option<GetPluginFactoryFn> = unsafe {
            GetProcAddress(hmodule, b"GetPluginFactory\0".as_ptr()).map(|f| std::mem::transmute(f))
        };
        let Some(get_factory_fn) = get_factory else {
            if let Some(ex) = exit_dll {
                unsafe {
                    let _ = ex();
                }
            }
            unsafe {
                let _ = FreeLibrary(hmodule);
            }
            return Err("GetPluginFactory missing".to_string());
        };

        let factory_ptr = unsafe { get_factory_fn() };
        if factory_ptr.is_null() {
            if let Some(ex) = exit_dll {
                unsafe {
                    let _ = ex();
                }
            }
            unsafe {
                let _ = FreeLibrary(hmodule);
            }
            return Err("GetPluginFactory returned NULL".to_string());
        }

        let target_tuid = requested_cid_hex.and_then(hex_to_tuid);
        let mut chosen_cid: Option<TUID> = None;
        let mut chosen_name = String::from("VST3 Plugin");

        unsafe {
            let fvtbl = *(factory_ptr as *const *const IPluginFactoryVtbl);
            let n_classes = ((*fvtbl).countClasses)(factory_ptr).max(0);
            for i in 0..n_classes {
                let mut cinfo: PClassInfo = std::mem::zeroed();
                if ((*fvtbl).getClassInfo)(factory_ptr, i, &mut cinfo) != K_RESULT_OK {
                    continue;
                }
                let cat = fixed_cchar_to_string(&cinfo.category);
                if !cat.eq_ignore_ascii_case("Audio Module Class") {
                    continue;
                }
                if let Some(req) = target_tuid {
                    if cinfo.cid == req {
                        chosen_cid = Some(cinfo.cid);
                        chosen_name = fixed_cchar_to_string(&cinfo.name);
                        break;
                    }
                } else if chosen_cid.is_none() {
                    chosen_cid = Some(cinfo.cid);
                    chosen_name = fixed_cchar_to_string(&cinfo.name);
                }
            }
        }

        let Some(cid) = chosen_cid else {
            unsafe {
                com_release(factory_ptr);
                if let Some(ex) = exit_dll {
                    let _ = ex();
                }
                let _ = FreeLibrary(hmodule);
            }
            return Err("No Audio Module Class found in VST3 factory".to_string());
        };

        let mut host_app = Box::new(HostApplicationCom {
            vtbl: &HOST_APP_VTBL,
            ref_count: AtomicU32::new(1),
        });
        let host_app_ptr = (&mut *host_app) as *mut HostApplicationCom as *mut c_void;

        let (gui_tx, gui_rx) = rtrb::RingBuffer::new(256);
        let mut comp_handler = Box::new(ComponentHandlerCom {
            vtbl: &COMP_HANDLER_VTBL,
            ref_count: AtomicU32::new(1),
            gui_edit_tx: parking_lot::Mutex::new(gui_tx),
        });
        let comp_handler_ptr = (&mut *comp_handler) as *mut ComponentHandlerCom as *mut c_void;

        let mut component_ptr: *mut c_void = std::ptr::null_mut();
        let create_rc = unsafe {
            let fvtbl = *(factory_ptr as *const *const IPluginFactoryVtbl);
            ((*fvtbl).createInstance)(factory_ptr, &cid, &IID_ICOMPONENT, &mut component_ptr)
        };
        if create_rc != K_RESULT_OK || component_ptr.is_null() {
            unsafe {
                com_release(factory_ptr);
                if let Some(ex) = exit_dll {
                    let _ = ex();
                }
                let _ = FreeLibrary(hmodule);
            }
            return Err("VST3 createInstance(IID_ICOMPONENT) failed".to_string());
        }

        let init_rc = unsafe {
            let cvtbl = *(component_ptr as *const *const IComponentVtbl);
            ((*cvtbl).base.initialize)(component_ptr, host_app_ptr)
        };
        if init_rc != K_RESULT_OK {
            unsafe {
                com_release(component_ptr);
                com_release(factory_ptr);
                if let Some(ex) = exit_dll {
                    let _ = ex();
                }
                let _ = FreeLibrary(hmodule);
            }
            return Err("IComponent::initialize failed".to_string());
        }

        let processor_ptr = unsafe { com_query(component_ptr, &IID_IAUDIO_PROCESSOR) };
        if processor_ptr.is_null() {
            unsafe {
                let cvtbl = *(component_ptr as *const *const IComponentVtbl);
                let _ = ((*cvtbl).base.terminate)(component_ptr);
                com_release(component_ptr);
                com_release(factory_ptr);
                if let Some(ex) = exit_dll {
                    let _ = ex();
                }
                let _ = FreeLibrary(hmodule);
            }
            return Err("IComponent does not expose IAudioProcessor".to_string());
        }

        // Single-component vs separate controller architecture
        let mut separate_controller = false;
        let mut controller_ptr = unsafe { com_query(component_ptr, &IID_IEDIT_CONTROLLER) };
        if controller_ptr.is_null() {
            let mut ctrl_cid: TUID = [0u8; 16];
            let cid_rc = unsafe {
                let cvtbl = *(component_ptr as *const *const IComponentVtbl);
                ((*cvtbl).getControllerClassId)(component_ptr, &mut ctrl_cid)
            };
            if cid_rc == K_RESULT_OK && ctrl_cid != [0u8; 16] {
                let mut separate_ptr: *mut c_void = std::ptr::null_mut();
                let crc = unsafe {
                    let fvtbl = *(factory_ptr as *const *const IPluginFactoryVtbl);
                    ((*fvtbl).createInstance)(
                        factory_ptr,
                        &ctrl_cid,
                        &IID_IEDIT_CONTROLLER,
                        &mut separate_ptr,
                    )
                };
                if crc == K_RESULT_OK && !separate_ptr.is_null() {
                    let ctrl_init = unsafe {
                        let evtbl = *(separate_ptr as *const *const IEditControllerVtbl);
                        ((*evtbl).base.initialize)(separate_ptr, host_app_ptr)
                    };
                    if ctrl_init == K_RESULT_OK {
                        controller_ptr = separate_ptr;
                        separate_controller = true;
                    } else {
                        unsafe { com_release(separate_ptr) };
                    }
                }
            }
        }

        let mut comp_conn_ptr: *mut c_void = std::ptr::null_mut();
        let mut ctrl_conn_ptr: *mut c_void = std::ptr::null_mut();
        if !controller_ptr.is_null() {
            unsafe {
                let evtbl = *(controller_ptr as *const *const IEditControllerVtbl);
                let _ = ((*evtbl).setComponentHandler)(controller_ptr, comp_handler_ptr);
            }
            if separate_controller {
                unsafe {
                    comp_conn_ptr = com_query(component_ptr, &IID_ICONNECTION_POINT);
                    ctrl_conn_ptr = com_query(controller_ptr, &IID_ICONNECTION_POINT);
                    if !comp_conn_ptr.is_null() && !ctrl_conn_ptr.is_null() {
                        let c1 = *(comp_conn_ptr as *const *const IConnectionPointVtbl);
                        let c2 = *(ctrl_conn_ptr as *const *const IConnectionPointVtbl);
                        let _ = ((*c1).connect)(comp_conn_ptr, ctrl_conn_ptr);
                        let _ = ((*c2).connect)(ctrl_conn_ptr, comp_conn_ptr);
                    }
                    // Synchronize component state to separate controller
                    let mut stream = MemoryBStream::new_empty();
                    let cvtbl = *(component_ptr as *const *const IComponentVtbl);
                    if ((*cvtbl).getState)(component_ptr, (&mut stream) as *mut _ as *mut c_void)
                        == K_RESULT_OK
                        && !stream.data.is_empty()
                    {
                        stream.cursor = 0;
                        let evtbl = *(controller_ptr as *const *const IEditControllerVtbl);
                        let _ = ((*evtbl).setComponentState)(
                            controller_ptr,
                            (&mut stream) as *mut _ as *mut c_void,
                        );
                    }
                }
            }
        }

        // Negotiate audio buses: main stereo in/out, an optional aux (sidechain) stereo input, and
        // every additional output bus (multi-out instruments) rendered into host scratch buffers.
        const K_BUS_TYPE_AUX: i32 = 1;
        const K_SPEAKER_STEREO: u64 = 0x3;
        let mut has_audio_input = false;
        let mut has_sidechain_bus = false;
        let mut num_input_buses = 0i32;
        let mut extra_out_channels: Vec<i32> = Vec::new();
        unsafe {
            let cvtbl = *(component_ptr as *const *const IComponentVtbl);
            let audio_ins = ((*cvtbl).getBusCount)(component_ptr, K_AUDIO, K_INPUT);
            if audio_ins > 0 {
                has_audio_input = true;
                num_input_buses = 1;
                let _ = ((*cvtbl).activateBus)(component_ptr, K_AUDIO, K_INPUT, 0, 1);
                if audio_ins > 1 {
                    let mut info: BusInfo = std::mem::zeroed();
                    if ((*cvtbl).getBusInfo)(component_ptr, K_AUDIO, K_INPUT, 1, &mut info) == K_RESULT_OK
                        && info.busType == K_BUS_TYPE_AUX
                    {
                        has_sidechain_bus = true;
                        num_input_buses = 2;
                        let _ = ((*cvtbl).activateBus)(component_ptr, K_AUDIO, K_INPUT, 1, 1);
                    }
                }
            }
            let audio_outs = ((*cvtbl).getBusCount)(component_ptr, K_AUDIO, K_OUTPUT);
            if audio_outs > 0 {
                let _ = ((*cvtbl).activateBus)(component_ptr, K_AUDIO, K_OUTPUT, 0, 1);
                for bus in 1..audio_outs {
                    let mut info: BusInfo = std::mem::zeroed();
                    let ch = if ((*cvtbl).getBusInfo)(component_ptr, K_AUDIO, K_OUTPUT, bus, &mut info)
                        == K_RESULT_OK
                    {
                        info.channelCount.clamp(1, 8)
                    } else {
                        2
                    };
                    extra_out_channels.push(ch);
                }
            }
            let event_ins = ((*cvtbl).getBusCount)(component_ptr, K_EVENT, K_INPUT);
            if event_ins > 0 {
                let _ = ((*cvtbl).activateBus)(component_ptr, K_EVENT, K_INPUT, 0, 1);
            }

            // Stereo main/aux arrangement; extra outputs keep the plug-in's own arrangement.
            let pvtbl_neg = *(processor_ptr as *const *const IAudioProcessorVtbl);
            let mut in_arr = [K_SPEAKER_STEREO; 2];
            let mut out_arr: Vec<u64> = Vec::with_capacity(1 + extra_out_channels.len());
            if audio_outs > 0 {
                out_arr.push(K_SPEAKER_STEREO);
                for bus in 1..audio_outs {
                    let mut arr: u64 = 0;
                    let _ = ((*pvtbl_neg).getBusArrangement)(processor_ptr, K_OUTPUT, bus, &mut arr);
                    out_arr.push(arr);
                }
            }
            let rc = ((*pvtbl_neg).setBusArrangements)(
                processor_ptr,
                in_arr.as_mut_ptr(),
                audio_ins.min(2).max(0),
                out_arr.as_mut_ptr(),
                out_arr.len() as i32,
            );
            if rc != K_RESULT_OK && has_sidechain_bus {
                // Keep the plug-in default layout; it still processes main + aux.
                let _ = rc;
            }
        }
        let mut extra_outputs = ExtraOutputBuses::new(&extra_out_channels);
        extra_outputs.buses.reserve(1 + extra_out_channels.len());

        let mut rt_arena = Vst3RtArena::new_boxed();
        let mut param_metadata = Vec::new();
        if !controller_ptr.is_null() {
            unsafe {
                let evtbl = *(controller_ptr as *const *const IEditControllerVtbl);
                let pcount = ((*evtbl).getParameterCount)(controller_ptr).max(0);
                for i in 0..pcount {
                    let mut pinfo: ParameterInfo = std::mem::zeroed();
                    if ((*evtbl).getParameterInfo)(controller_ptr, i, &mut pinfo) == K_RESULT_OK {
                        let cur_norm = ((*evtbl).getParamNormalized)(controller_ptr, pinfo.id);
                        param_metadata.push(HostedPluginParamInfo {
                            id: pinfo.id,
                            name: wide128_to_string(&pinfo.title),
                            module: wide128_to_string(&pinfo.units),
                            min_value: 0.0,
                            max_value: 1.0,
                            default_value: pinfo.defaultNormalizedValue,
                            current_value: cur_norm,
                        });
                        if rt_arena.param_count < rt_arena.param_ids.len() {
                            rt_arena.param_ids[rt_arena.param_count] = pinfo.id;
                            rt_arena.param_count += 1;
                        }
                    }
                }
            }
        }

        let sr = sample_rate.max(8_000.0);
        let mbs = max_block_size.clamp(1, MAX_BLOCK_SIZE);
        let mut node = Self {
            _dll_path: dll_path,
            cid_hex: tuid_to_hex(&cid),
            plugin_name: chosen_name,
            hmodule,
            exit_dll,
            factory_ptr,
            component_ptr,
            processor_ptr,
            controller_ptr,
            separate_controller,
            comp_conn_ptr,
            ctrl_conn_ptr,
            _host_app: host_app,
            _comp_handler: comp_handler,
            gui_edit_rx: gui_rx,
            rt_arena,
            param_metadata,
            editor: Vst3EditorSession::default(),
            live_guard: std::sync::Arc::new(parking_lot::Mutex::new(true)),
            has_audio_input,
            has_sidechain_bus,
            num_input_buses,
            extra_outputs,
            active_flag: false,
            processing_flag: false,
            sample_rate: sr,
            max_block_size: mbs,
            cached_latency_samples: 0,
        };
        node.setup_and_activate(sr, mbs)?;
        Ok(node)
    }

    fn setup_and_activate(&mut self, sample_rate: f64, max_block_size: usize) -> Result<(), String> {
        let sr = sample_rate.max(8_000.0);
        let mbs = max_block_size.clamp(1, MAX_BLOCK_SIZE);
        unsafe {
            let pvtbl = *(self.processor_ptr as *const *const IAudioProcessorVtbl);
            let cvtbl = *(self.component_ptr as *const *const IComponentVtbl);
            if self.processing_flag {
                let _ = ((*pvtbl).setProcessing)(self.processor_ptr, 0);
                self.processing_flag = false;
            }
            if self.active_flag {
                let _ = ((*cvtbl).setActive)(self.component_ptr, 0);
                self.active_flag = false;
            }
            let mut setup = ProcessSetup {
                processMode: K_REALTIME,
                symbolicSampleSize: K_SAMPLE_32,
                maxSamplesPerBlock: mbs as i32,
                sampleRate: sr,
            };
            let rc = ((*pvtbl).setupProcessing)(self.processor_ptr, &mut setup);
            if rc != K_RESULT_OK && rc != K_NOT_IMPLEMENTED {
                return Err(format!("VST3 setupProcessing failed with code {rc}"));
            }
            let _ = ((*cvtbl).setActive)(self.component_ptr, 1);
            self.active_flag = true;
            let _ = ((*pvtbl).setProcessing)(self.processor_ptr, 1);
            self.processing_flag = true;
            self.cached_latency_samples = ((*pvtbl).getLatencySamples)(self.processor_ptr) as usize;
        }
        self.sample_rate = sr;
        self.max_block_size = mbs;
        Ok(())
    }

    #[inline]
    pub fn cid_hex(&self) -> &str {
        &self.cid_hex
    }

    #[inline]
    pub fn plugin_display_name(&self) -> &str {
        &self.plugin_name
    }

    #[inline]
    pub fn parameters(&self) -> &[HostedPluginParamInfo] {
        &self.param_metadata
    }

    /// Reads the current normalized parameter `[0.0..=1.0]` from `IEditController::getParamNormalized`.
    pub fn get_parameter_normalized(&self, param_id: u32) -> Option<f64> {
        if self.controller_ptr.is_null() {
            return None;
        }
        unsafe {
            let evtbl = *(self.controller_ptr as *const *const IEditControllerVtbl);
            Some(((*evtbl).getParamNormalized)(self.controller_ptr, param_id))
        }
    }

    /// Queues a sample-accurate normalized parameter change `[0.0..=1.0]` (`0` heap allocations).
    #[inline]
    pub fn queue_parameter_point(&mut self, param_id: u32, norm_val: f64, sample_offset: i32) {
        let clamped = norm_val.clamp(0.0, 1.0);
        if !self.controller_ptr.is_null() {
            unsafe {
                let evtbl = *(self.controller_ptr as *const *const IEditControllerVtbl);
                let _ = ((*evtbl).setParamNormalized)(self.controller_ptr, param_id, clamped);
            }
        }
        self.rt_arena
            .in_param_changes
            .push_point(param_id, sample_offset, clamped);
    }

    /// Queues a sample-accurate VST3 NoteOn event (`0` heap allocations).
    #[inline]
    pub fn queue_note_on(&mut self, key: u8, velocity: f32, channel: u8, sample_offset: i32) {
        self.rt_arena.in_events.push(Vst3Event {
            busIndex: 0,
            sampleOffset: sample_offset.max(0),
            ppqPosition: 0.0,
            flags: 1,
            type_: K_NOTE_ON_EVENT,
            payload: Vst3EventPayload {
                noteOn: NoteOnEvent {
                    channel: (channel & 0x0F) as i16,
                    pitch: (key & 0x7F) as i16,
                    tuning: 0.0,
                    velocity: velocity.clamp(0.0, 1.0),
                    length: 0,
                    noteId: key as i32,
                },
            },
        });
    }

    /// Queues a sample-accurate VST3 NoteOff event (`0` heap allocations).
    #[inline]
    pub fn queue_note_off(&mut self, key: u8, velocity: f32, channel: u8, sample_offset: i32) {
        self.rt_arena.in_events.push(Vst3Event {
            busIndex: 0,
            sampleOffset: sample_offset.max(0),
            ppqPosition: 0.0,
            flags: 1,
            type_: K_NOTE_OFF_EVENT,
            payload: Vst3EventPayload {
                noteOff: NoteOffEvent {
                    channel: (channel & 0x0F) as i16,
                    pitch: (key & 0x7F) as i16,
                    velocity: velocity.clamp(0.0, 1.0),
                    noteId: key as i32,
                    tuning: 0.0,
                },
            },
        });
    }

    /// Saves `IComponent` + `IEditController` binary state via `IBStream`.
    pub fn save_state(&self) -> Result<Vec<u8>, String> {
        let mut comp_stream = MemoryBStream::new_empty();
        let comp_rc = unsafe {
            let cvtbl = *(self.component_ptr as *const *const IComponentVtbl);
            ((*cvtbl).getState)(
                self.component_ptr,
                (&mut comp_stream) as *mut MemoryBStream as *mut c_void,
            )
        };
        if comp_rc != K_RESULT_OK && comp_rc != K_NOT_IMPLEMENTED {
            return Err(format!("IComponent::getState failed ({comp_rc})"));
        }

        let mut ctrl_stream = MemoryBStream::new_empty();
        if !self.controller_ptr.is_null() {
            unsafe {
                let evtbl = *(self.controller_ptr as *const *const IEditControllerVtbl);
                let _ = ((*evtbl).getState)(
                    self.controller_ptr,
                    (&mut ctrl_stream) as *mut MemoryBStream as *mut c_void,
                );
            }
        }

        let mut out =
            Vec::with_capacity(16 + comp_stream.data.len() + ctrl_stream.data.len());
        out.extend_from_slice(TF_VST3_STATE_MAGIC);
        out.extend_from_slice(&(comp_stream.data.len() as u32).to_le_bytes());
        out.extend_from_slice(&comp_stream.data);
        out.extend_from_slice(&(ctrl_stream.data.len() as u32).to_le_bytes());
        out.extend_from_slice(&ctrl_stream.data);
        Ok(out)
    }

    /// Restores `IComponent` + `IEditController` binary state via `IBStream`.
    pub fn load_state(&mut self, bytes: &[u8]) -> Result<(), String> {
        let (comp_bytes, ctrl_bytes): (&[u8], &[u8]) =
            if bytes.len() >= 16 && &bytes[0..8] == TF_VST3_STATE_MAGIC {
                let comp_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
                if 16 + comp_len > bytes.len() {
                    return Err("Corrupted TFVST3ST component chunk length".to_string());
                }
                let c_end = 12 + comp_len;
                let ctrl_len =
                    u32::from_le_bytes(bytes[c_end..c_end + 4].try_into().unwrap()) as usize;
                if c_end + 4 + ctrl_len > bytes.len() {
                    return Err("Corrupted TFVST3ST controller chunk length".to_string());
                }
                (&bytes[12..c_end], &bytes[c_end + 4..c_end + 4 + ctrl_len])
            } else {
                (bytes, &[])
            };

        if !comp_bytes.is_empty() {
            let mut cs = MemoryBStream::from_bytes(comp_bytes);
            let rc = unsafe {
                let cvtbl = *(self.component_ptr as *const *const IComponentVtbl);
                ((*cvtbl).setState)(
                    self.component_ptr,
                    (&mut cs) as *mut MemoryBStream as *mut c_void,
                )
            };
            if rc != K_RESULT_OK && rc != K_NOT_IMPLEMENTED {
                return Err(format!("IComponent::setState failed ({rc})"));
            }
            if !self.controller_ptr.is_null() {
                cs.cursor = 0;
                unsafe {
                    let evtbl = *(self.controller_ptr as *const *const IEditControllerVtbl);
                    let _ = ((*evtbl).setComponentState)(
                        self.controller_ptr,
                        (&mut cs) as *mut MemoryBStream as *mut c_void,
                    );
                }
            }
        }

        if !ctrl_bytes.is_empty() && !self.controller_ptr.is_null() {
            let mut es = MemoryBStream::from_bytes(ctrl_bytes);
            unsafe {
                let evtbl = *(self.controller_ptr as *const *const IEditControllerVtbl);
                let _ = ((*evtbl).setState)(
                    self.controller_ptr,
                    (&mut es) as *mut MemoryBStream as *mut c_void,
                );
            }
        }

        if !self.controller_ptr.is_null() {
            unsafe {
                let evtbl = *(self.controller_ptr as *const *const IEditControllerVtbl);
                for meta in &mut self.param_metadata {
                    meta.current_value =
                        ((*evtbl).getParamNormalized)(self.controller_ptr, meta.id);
                }
            }
        }
        Ok(())
    }

    /// Opens the native Win32 `HWND` editor via `IEditController::createView("editor")` + `IPlugView::attached`.
    pub fn open_editor(
        &mut self,
        owner_hwnd: Option<isize>,
        visible: bool,
    ) -> Result<isize, String> {
        let controller_ptr = self.controller_ptr;
        let name = self.plugin_name.clone();
        self.editor.open(controller_ptr, &name, owner_hwnd, visible)
    }

    /// Hands out an editor bound to *this* instance's controller (see [`Vst3LiveEditor`]). Call
    /// on the loading thread before the node moves to the real-time graph.
    pub fn live_editor(&self) -> Option<Vst3LiveEditor> {
        if self.controller_ptr.is_null() {
            return None;
        }
        unsafe {
            let evtbl = *(self.controller_ptr as *const *const IEditControllerVtbl);
            ((*evtbl).base.base.addRef)(self.controller_ptr);
            let vtbl_addr = *(self.controller_ptr as *const *const c_void);
            Some(Vst3LiveEditor {
                controller_ptr: self.controller_ptr,
                module_pin: pin_module_containing(vtbl_addr),
                plugin_name: self.plugin_name.clone(),
                param_ids: self.param_metadata.iter().map(|p| p.id).collect(),
                live_guard: std::sync::Arc::clone(&self.live_guard),
                session: Vst3EditorSession::default(),
            })
        }
    }

    /// `true` when the plug-in exposes an active aux (sidechain) audio input bus.
    pub fn has_sidechain_bus(&self) -> bool {
        self.has_sidechain_bus
    }

    pub fn pump_editor_messages(&self) -> usize {
        self.editor.pump()
    }

    pub fn close_editor(&mut self) {
        self.editor.close();
    }
}

impl UnifiedDspNode for Vst3HostedPluginNode {
    fn name(&self) -> &'static str {
        "Steinberg VST3 3.7 Hosted Plugin"
    }

    fn latency_samples(&self) -> usize {
        self.cached_latency_samples
    }

    fn prepare(&mut self, sample_rate: f64, max_block_size: usize) {
        let _ = self.setup_and_activate(sample_rate, max_block_size);
    }

    fn reset_state(&mut self) {
        self.rt_arena.in_param_changes.clear();
        self.rt_arena.out_param_changes.clear();
        self.rt_arena.in_events.clear();
        self.rt_arena.out_events.clear();
        if self.processing_flag && !self.processor_ptr.is_null() {
            unsafe {
                let pvtbl = *(self.processor_ptr as *const *const IAudioProcessorVtbl);
                let _ = ((*pvtbl).setProcessing)(self.processor_ptr, 0);
                let _ = ((*pvtbl).setProcessing)(self.processor_ptr, 1);
            }
        }
    }

    fn set_parameter_normalized(&mut self, param_id: u32, norm_val: f32) {
        let n = (norm_val as f64).clamp(0.0, 1.0);
        let mut resolved_id = param_id;
        let mut matched = false;
        for i in 0..self.rt_arena.param_count {
            if self.rt_arena.param_ids[i] == param_id {
                matched = true;
                break;
            }
        }
        if !matched && (param_id as usize) < self.rt_arena.param_count {
            resolved_id = self.rt_arena.param_ids[param_id as usize];
        }
        self.rt_arena.in_param_changes.push_point(resolved_id, 0, n);
    }

    fn on_voice_event(&mut self, event: &SequencerVoiceEvent, ctx: &ProcessContext) {
        let max_off = ctx.slice_frames.saturating_sub(1) as i64;
        let offset = (event.sample_pos - ctx.current_sample_pos).clamp(0, max_off) as i32;
        let ch = (event.color_group & 0x0F) as i16;
        let pitch = (event.key & 0x7F) as i16;
        let vel = event.velocity.clamp(0.0, 1.0);
        match event.kind {
            VoiceEventKind::NoteOn | VoiceEventKind::SlideNote => {
                self.rt_arena.in_events.push(Vst3Event {
                    busIndex: 0,
                    sampleOffset: offset,
                    ppqPosition: 0.0,
                    flags: 1,
                    type_: K_NOTE_ON_EVENT,
                    payload: Vst3EventPayload {
                        noteOn: NoteOnEvent {
                            channel: ch,
                            pitch,
                            tuning: event.fine_pitch_cents,
                            velocity: vel,
                            length: 0,
                            noteId: event.note_id as i32,
                        },
                    },
                });
            }
            VoiceEventKind::NoteOff => {
                self.rt_arena.in_events.push(Vst3Event {
                    busIndex: 0,
                    sampleOffset: offset,
                    ppqPosition: 0.0,
                    flags: 1,
                    type_: K_NOTE_OFF_EVENT,
                    payload: Vst3EventPayload {
                        noteOff: NoteOffEvent {
                            channel: ch,
                            pitch,
                            velocity: vel,
                            noteId: event.note_id as i32,
                            tuning: event.fine_pitch_cents,
                        },
                    },
                });
            }
            VoiceEventKind::ChannelMute => {
                // All-notes-off: VST3 has no CC 123 event, so release every key on this channel.
                for key in 0..128i16 {
                    self.rt_arena.in_events.push(Vst3Event {
                        busIndex: 0,
                        sampleOffset: offset,
                        ppqPosition: 0.0,
                        flags: 1,
                        type_: K_NOTE_OFF_EVENT,
                        payload: Vst3EventPayload {
                            noteOff: NoteOffEvent {
                                channel: ch,
                                pitch: key,
                                velocity: 0.0,
                                noteId: -1,
                                tuning: 0.0,
                            },
                        },
                    });
                }
            }
            VoiceEventKind::SlotEnable => {}
        }
    }

    fn process_slice(
        &mut self,
        ctx: &ProcessContext,
        main_io: &mut [StereoFrame],
        sidechain_in: Option<&[StereoFrame]>,
        _aux_sends_out: &mut [&mut [StereoFrame]],
    ) {
        let total = main_io.len();
        if total == 0 || self.processor_ptr.is_null() {
            self.rt_arena.in_param_changes.clear();
            self.rt_arena.in_events.clear();
            return;
        }

        // Drain any lock-free GUI parameter changes from `IComponentHandler::performEdit`
        while let Ok((pid, nval)) = self.gui_edit_rx.pop() {
            self.rt_arena.in_param_changes.push_point(pid, 0, nval);
        }

        let pvtbl = unsafe { *(self.processor_ptr as *const *const IAudioProcessorVtbl) };
        let mut cursor = 0usize;

        while cursor < total {
            let chunk = (total - cursor).min(MAX_BLOCK_SIZE);
            for i in 0..chunk {
                let f = main_io[cursor + i];
                self.rt_arena.in_left[i] = f[0];
                self.rt_arena.in_right[i] = f[1];
                self.rt_arena.out_left[i] = 0.0;
                self.rt_arena.out_right[i] = 0.0;
                let sc = sidechain_in
                    .and_then(|s| s.get(cursor + i))
                    .copied()
                    .unwrap_or([0.0, 0.0]);
                self.rt_arena.aux_left[i] = sc[0];
                self.rt_arena.aux_right[i] = sc[1];
            }
            self.rt_arena.aux_ptrs[0] = self.rt_arena.aux_left.as_mut_ptr();
            self.rt_arena.aux_ptrs[1] = self.rt_arena.aux_right.as_mut_ptr();
            self.rt_arena.in_ptrs[0] = self.rt_arena.in_left.as_mut_ptr();
            self.rt_arena.in_ptrs[1] = self.rt_arena.in_right.as_mut_ptr();
            self.rt_arena.out_ptrs[0] = self.rt_arena.out_left.as_mut_ptr();
            self.rt_arena.out_ptrs[1] = self.rt_arena.out_right.as_mut_ptr();
            self.rt_arena.out_param_changes.clear();
            self.rt_arena.out_events.clear();

            let out_bus = AudioBusBuffers {
                numChannels: 2,
                silenceFlags: 0,
                channelBuffers32: self.rt_arena.out_ptrs.as_mut_ptr(),
            };
            let mut in_buses = [
                AudioBusBuffers {
                    numChannels: 2,
                    silenceFlags: 0,
                    channelBuffers32: self.rt_arena.in_ptrs.as_mut_ptr(),
                },
                AudioBusBuffers {
                    numChannels: 2,
                    silenceFlags: if sidechain_in.is_some() { 0 } else { 0x3 },
                    channelBuffers32: self.rt_arena.aux_ptrs.as_mut_ptr(),
                },
            ];
            // `buses` has capacity for main + extras (reserved at load): clear/push never reallocates.
            let extra = &mut self.extra_outputs;
            extra.buses.clear();
            extra.buses.push(out_bus);
            for &(first, ch) in &extra.layout {
                extra.buses.push(AudioBusBuffers {
                    numChannels: ch,
                    silenceFlags: 0,
                    channelBuffers32: extra.channel_ptrs[first..].as_mut_ptr(),
                });
            }
            let num_outputs = extra.buses.len() as i32;
            let outputs_ptr = extra.buses.as_mut_ptr();
            // Musical position (quarter notes) at this chunk: the active PPQ tick plus the chunk
            // offset at the current tempo. 4/4 bars.
            let quarters = (ctx.ppq_tick as f64) / (ctx.ppq_resolution.max(1) as f64)
                + (cursor as f64) * ctx.bpm / (60.0 * ctx.sample_rate.max(1.0));
            const K_PLAYING: u32 = 1 << 1;
            const K_PROJECT_TIME_MUSIC_VALID: u32 = 1 << 9;
            const K_TEMPO_VALID: u32 = 1 << 10;
            const K_BAR_POSITION_VALID: u32 = 1 << 11;
            const K_TIME_SIG_VALID: u32 = 1 << 13;
            const K_CONT_TIME_VALID: u32 = 1 << 17;
            let mut proc_ctx = ProcessContextVst3 {
                state: if ctx.is_playing { K_PLAYING } else { 0 }
                    | K_PROJECT_TIME_MUSIC_VALID
                    | K_TEMPO_VALID
                    | K_BAR_POSITION_VALID
                    | K_TIME_SIG_VALID
                    | K_CONT_TIME_VALID,
                sampleRate: ctx.sample_rate,
                projectTimeSamples: ctx.current_sample_pos + (cursor as i64),
                continousTimeSamples: ctx.current_sample_pos + (cursor as i64),
                projectTimeMusic: quarters,
                barPositionMusic: (quarters / 4.0).floor() * 4.0,
                tempo: ctx.bpm,
                timeSigNumerator: 4,
                timeSigDenominator: 4,
                ..ProcessContextVst3::default()
            };

            let mut data = ProcessData {
                processMode: if ctx.is_offline_rendering {
                    crate::vst3_abi::K_OFFLINE
                } else {
                    K_REALTIME
                },
                symbolicSampleSize: K_SAMPLE_32,
                numSamples: chunk as i32,
                numInputs: if self.has_audio_input { self.num_input_buses } else { 0 },
                numOutputs: num_outputs,
                inputs: if self.has_audio_input {
                    in_buses.as_mut_ptr()
                } else {
                    std::ptr::null_mut()
                },
                outputs: outputs_ptr,
                inputParameterChanges: (&mut self.rt_arena.in_param_changes)
                    as *mut PreallocParameterChanges
                    as *mut c_void,
                outputParameterChanges: (&mut self.rt_arena.out_param_changes)
                    as *mut PreallocParameterChanges
                    as *mut c_void,
                inputEvents: (&mut self.rt_arena.in_events) as *mut PreallocEventList
                    as *mut c_void,
                outputEvents: (&mut self.rt_arena.out_events) as *mut PreallocEventList
                    as *mut c_void,
                processContext: &mut proc_ctx,
            };

            unsafe {
                let _ = ((*pvtbl).process)(self.processor_ptr, &mut data);
            }

            for i in 0..chunk {
                main_io[cursor + i] = [self.rt_arena.out_left[i], self.rt_arena.out_right[i]];
            }
            self.rt_arena.in_param_changes.clear();
            self.rt_arena.in_events.clear();
            cursor += chunk;
        }
    }
}

impl Drop for Vst3HostedPluginNode {
    fn drop(&mut self) {
        *self.live_guard.lock() = false;
        self.close_editor();
        unsafe {
            if !self.processor_ptr.is_null() && self.processing_flag {
                let pvtbl = *(self.processor_ptr as *const *const IAudioProcessorVtbl);
                let _ = ((*pvtbl).setProcessing)(self.processor_ptr, 0);
                self.processing_flag = false;
            }
            if !self.component_ptr.is_null() && self.active_flag {
                let cvtbl = *(self.component_ptr as *const *const IComponentVtbl);
                let _ = ((*cvtbl).setActive)(self.component_ptr, 0);
                self.active_flag = false;
            }
            if !self.comp_conn_ptr.is_null() && !self.ctrl_conn_ptr.is_null() {
                let c1 = *(self.comp_conn_ptr as *const *const IConnectionPointVtbl);
                let c2 = *(self.ctrl_conn_ptr as *const *const IConnectionPointVtbl);
                let _ = ((*c1).disconnect)(self.comp_conn_ptr, self.ctrl_conn_ptr);
                let _ = ((*c2).disconnect)(self.ctrl_conn_ptr, self.comp_conn_ptr);
            }
            com_release(self.comp_conn_ptr);
            com_release(self.ctrl_conn_ptr);
            self.comp_conn_ptr = std::ptr::null_mut();
            self.ctrl_conn_ptr = std::ptr::null_mut();

            if !self.controller_ptr.is_null() {
                let evtbl = *(self.controller_ptr as *const *const IEditControllerVtbl);
                let _ = ((*evtbl).setComponentHandler)(self.controller_ptr, std::ptr::null_mut());
                if self.separate_controller {
                    let _ = ((*evtbl).base.terminate)(self.controller_ptr);
                }
                com_release(self.controller_ptr);
                self.controller_ptr = std::ptr::null_mut();
            }
            com_release(self.processor_ptr);
            self.processor_ptr = std::ptr::null_mut();

            if !self.component_ptr.is_null() {
                let cvtbl = *(self.component_ptr as *const *const IComponentVtbl);
                let _ = ((*cvtbl).base.terminate)(self.component_ptr);
                com_release(self.component_ptr);
                self.component_ptr = std::ptr::null_mut();
            }
            com_release(self.factory_ptr);
            self.factory_ptr = std::ptr::null_mut();

            if let Some(ex) = self.exit_dll.take() {
                let _ = ex();
            }
            if !self.hmodule.is_null() {
                let _ = FreeLibrary(self.hmodule);
                self.hmodule = std::ptr::null_mut();
            }
        }
    }
}


#[cfg(target_os = "windows")]
#[link(name = "user32")]
unsafe extern "system" {
    fn GetDpiForWindow(hwnd: *mut c_void) -> u32;
    fn GetDpiForSystem() -> u32;
}

/// Monitor scale factor (`DPI / 96`) of `hwnd`, or of the system when `hwnd` is `None`.
fn dpi_scale_for(hwnd: Option<*mut c_void>) -> f32 {
    #[cfg(target_os = "windows")]
    unsafe {
        let dpi = match hwnd {
            Some(h) if !h.is_null() => GetDpiForWindow(h),
            _ => GetDpiForSystem(),
        };
        if dpi == 0 { 1.0 } else { dpi as f32 / 96.0 }
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = hwnd;
        1.0
    }
}

/// Calls `IPlugViewContentScaleSupport::setContentScaleFactor` when the view implements it.
/// Returns `true` when the plug-in accepted the factor.
unsafe fn set_view_content_scale(view_ptr: *mut c_void, factor: f32) -> bool {
    if view_ptr.is_null() {
        return false;
    }
    unsafe {
        let vvtbl = *(view_ptr as *const *const IPlugViewVtbl);
        let mut scale_ptr: *mut c_void = std::ptr::null_mut();
        let rc = ((*vvtbl).base.queryInterface)(
            view_ptr,
            &IID_IPLUG_VIEW_CONTENT_SCALE_SUPPORT,
            &mut scale_ptr,
        );
        if rc != K_RESULT_OK || scale_ptr.is_null() {
            return false;
        }
        let svtbl = *(scale_ptr as *const *const IPlugViewContentScaleSupportVtbl);
        let ok = ((*svtbl).setContentScaleFactor)(scale_ptr, factor) == K_RESULT_OK;
        com_release(scale_ptr);
        ok
    }
}
