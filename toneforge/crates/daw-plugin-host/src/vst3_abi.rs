//! `vst3_abi`: Clean-room `#[repr(C)]` Windows COM (`extern "system"`) Interface & VTable
//! definitions for **Steinberg VST3 3.7** (`FUnknown`, `IPluginFactory`, `IComponent`,
//! `IAudioProcessor`, `IEditController`, `IComponentHandler`, `IHostApplication`, `IBStream`,
//! `IPlugView`, `IParameterChanges`, `IParamValueQueue`, `IEventList`).

#![allow(non_camel_case_types, non_snake_case)]

use std::ffi::{c_char, c_void};

pub type tresult = i32;
pub type TUID = [u8; 16];
pub type TBool = u8;
pub type ParamID = u32;
pub type ParamValue = f64;
pub type MediaType = i32;
pub type BusDirection = i32;

pub const K_RESULT_OK: tresult = 0;
pub const K_RESULT_FALSE: tresult = 1;
pub const K_NO_INTERFACE: tresult = -1;
pub const K_NOT_IMPLEMENTED: tresult = -2;
pub const K_INVALID_ARGUMENT: tresult = -3;

pub const K_AUDIO: MediaType = 0;
pub const K_EVENT: MediaType = 1;
pub const K_INPUT: BusDirection = 0;
pub const K_OUTPUT: BusDirection = 1;

pub const K_REALTIME: i32 = 0;
pub const K_OFFLINE: i32 = 2;
pub const K_SAMPLE_32: i32 = 0;
pub const K_SAMPLE_64: i32 = 1;

pub const K_NOTE_ON_EVENT: u16 = 0;
pub const K_NOTE_OFF_EVENT: u16 = 1;

pub const VST3_PLATFORM_TYPE_HWND: &[u8] = b"HWND\0";
pub const VST3_VIEW_TYPE_EDITOR: &[u8] = b"editor\0";

/// Windows `COM_COMPATIBLE == 1` `INLINE_UID(l1, l2, l3, l4)` byte layout (`funknown.h`).
pub const fn vst3_com_uid(l1: u32, l2: u32, l3: u32, l4: u32) -> TUID {
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

pub const IID_FUNKNOWN: TUID = vst3_com_uid(0x0000_0000, 0x0000_0000, 0xC000_0000, 0x0000_0046);
pub const IID_IPLUGIN_BASE: TUID = vst3_com_uid(0x2288_8DDB, 0x156E_45AE, 0x8358_B348, 0x0819_0625);
pub const IID_IPLUGIN_FACTORY: TUID =
    vst3_com_uid(0x7A4D_811C, 0x5211_4A1F, 0xAED9_D2EE, 0x0B43_BF9F);
pub const IID_ICOMPONENT: TUID = vst3_com_uid(0xE831_FF31, 0xF2D5_4301, 0x928E_BBEE, 0x2569_7802);
pub const IID_IAUDIO_PROCESSOR: TUID =
    vst3_com_uid(0x4204_3F99, 0xB7DA_453C, 0xA569_E79D, 0x9AAE_C33D);
pub const IID_IEDIT_CONTROLLER: TUID =
    vst3_com_uid(0xDCD7_BBE3, 0x7742_448D, 0xA874_AACC, 0x979C_759E);
pub const IID_ICONNECTION_POINT: TUID =
    vst3_com_uid(0x70A4_156F, 0x6E6E_4026, 0x9891_48BF, 0xAA60_D8D1);
pub const IID_ICOMPONENT_HANDLER: TUID =
    vst3_com_uid(0x93A0_BEA3, 0x0BD0_45DB, 0x8E89_0B0C, 0xC1E4_6AC6);
pub const IID_IHOST_APPLICATION: TUID =
    vst3_com_uid(0x58E5_95CC, 0xDB2D_4969, 0x8B6A_AF8C, 0x36A6_64E5);
pub const IID_IBSTREAM: TUID = vst3_com_uid(0xC3BF_6EA2, 0x3099_4752, 0x9B6B_F990, 0x1EE3_3E9B);
pub const IID_IPLUG_VIEW: TUID = vst3_com_uid(0x5BC3_2507, 0xD060_49EA, 0xA615_1B52, 0x2B75_5B29);
pub const IID_IPARAMETER_CHANGES: TUID =
    vst3_com_uid(0xA477_9663, 0x0BB6_4A56, 0xB443_84A8, 0x466F_EB9D);
pub const IID_IPARAM_VALUE_QUEUE: TUID =
    vst3_com_uid(0x0126_3A18, 0xED07_4F6F, 0x98C9_D356, 0x4686_F9BA);
pub const IID_IEVENT_LIST: TUID = vst3_com_uid(0x3A2C_4214, 0x3463_49FE, 0xB2C4_F397, 0xB969_5A44);
pub const IID_IPLUG_FRAME: TUID = vst3_com_uid(0x367F_AF28, 0xC5C1_4D93, 0xA437_AD29, 0x3857_D8A0);
/// `Steinberg::IPlugViewContentScaleSupport` (`65ED9690-8AC4-4525-8AAD-EF7A72EA703F`).
pub const IID_IPLUG_VIEW_CONTENT_SCALE_SUPPORT: TUID =
    vst3_com_uid(0x65ED_9690, 0x8AC4_4525, 0x8AAD_EF7A, 0x72EA_703F);

#[repr(C)]
pub struct IPlugViewContentScaleSupportVtbl {
    pub base: FUnknownVtbl,
    pub setContentScaleFactor: unsafe extern "system" fn(this: *mut c_void, factor: f32) -> tresult,
}

#[repr(C)]
pub struct FUnknown {
    pub vtbl: *const FUnknownVtbl,
}

#[repr(C)]
pub struct FUnknownVtbl {
    pub queryInterface:
        unsafe extern "system" fn(this: *mut c_void, iid: *const TUID, obj: *mut *mut c_void) -> tresult,
    pub addRef: unsafe extern "system" fn(this: *mut c_void) -> u32,
    pub release: unsafe extern "system" fn(this: *mut c_void) -> u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PFactoryInfo {
    pub vendor: [c_char; 64],
    pub url: [c_char; 256],
    pub email: [c_char; 128],
    pub flags: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PClassInfo {
    pub cid: TUID,
    pub cardinality: i32,
    pub category: [c_char; 32],
    pub name: [c_char; 64],
}

#[repr(C)]
pub struct IPluginFactoryVtbl {
    pub base: FUnknownVtbl,
    pub getFactoryInfo:
        unsafe extern "system" fn(this: *mut c_void, info: *mut PFactoryInfo) -> tresult,
    pub countClasses: unsafe extern "system" fn(this: *mut c_void) -> i32,
    pub getClassInfo:
        unsafe extern "system" fn(this: *mut c_void, index: i32, info: *mut PClassInfo) -> tresult,
    pub createInstance: unsafe extern "system" fn(
        this: *mut c_void,
        cid: *const TUID,
        iid: *const TUID,
        obj: *mut *mut c_void,
    ) -> tresult,
}

#[repr(C)]
pub struct IPluginBaseVtbl {
    pub base: FUnknownVtbl,
    pub initialize: unsafe extern "system" fn(this: *mut c_void, context: *mut c_void) -> tresult,
    pub terminate: unsafe extern "system" fn(this: *mut c_void) -> tresult,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BusInfo {
    pub mediaType: MediaType,
    pub direction: BusDirection,
    pub channelCount: i32,
    pub name: [u16; 128],
    pub busType: i32,
    pub flags: u32,
}

#[repr(C)]
pub struct IComponentVtbl {
    pub base: IPluginBaseVtbl,
    pub getControllerClassId:
        unsafe extern "system" fn(this: *mut c_void, classId: *mut TUID) -> tresult,
    pub setIoMode: unsafe extern "system" fn(this: *mut c_void, mode: i32) -> tresult,
    pub getBusCount:
        unsafe extern "system" fn(this: *mut c_void, type_: MediaType, dir: BusDirection) -> i32,
    pub getBusInfo: unsafe extern "system" fn(
        this: *mut c_void,
        type_: MediaType,
        dir: BusDirection,
        index: i32,
        bus: *mut BusInfo,
    ) -> tresult,
    pub getRoutingInfo:
        unsafe extern "system" fn(this: *mut c_void, inInfo: *mut c_void, outInfo: *mut c_void) -> tresult,
    pub activateBus: unsafe extern "system" fn(
        this: *mut c_void,
        type_: MediaType,
        dir: BusDirection,
        index: i32,
        state: TBool,
    ) -> tresult,
    pub setActive: unsafe extern "system" fn(this: *mut c_void, state: TBool) -> tresult,
    pub setState: unsafe extern "system" fn(this: *mut c_void, state: *mut c_void) -> tresult,
    pub getState: unsafe extern "system" fn(this: *mut c_void, state: *mut c_void) -> tresult,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ProcessSetup {
    pub processMode: i32,
    pub symbolicSampleSize: i32,
    pub maxSamplesPerBlock: i32,
    pub sampleRate: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AudioBusBuffers {
    pub numChannels: i32,
    pub silenceFlags: u64,
    pub channelBuffers32: *mut *mut f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ProcessContextVst3 {
    pub state: u32,
    pub sampleRate: f64,
    pub projectTimeSamples: i64,
    pub systemTime: i64,
    pub continousTimeSamples: i64,
    pub projectTimeMusic: f64,
    pub barPositionMusic: f64,
    pub cycleStartMusic: f64,
    pub cycleEndMusic: f64,
    pub tempo: f64,
    pub timeSigNumerator: i32,
    pub timeSigDenominator: i32,
    pub chord: i32,
    pub smpteOffsetSubframes: i32,
    pub frameRate: u64,
    pub samplesToNextClock: i32,
}

#[repr(C)]
pub struct ProcessData {
    pub processMode: i32,
    pub symbolicSampleSize: i32,
    pub numSamples: i32,
    pub numInputs: i32,
    pub numOutputs: i32,
    pub inputs: *mut AudioBusBuffers,
    pub outputs: *mut AudioBusBuffers,
    pub inputParameterChanges: *mut c_void,
    pub outputParameterChanges: *mut c_void,
    pub inputEvents: *mut c_void,
    pub outputEvents: *mut c_void,
    pub processContext: *mut ProcessContextVst3,
}

#[repr(C)]
pub struct IAudioProcessorVtbl {
    pub base: FUnknownVtbl,
    pub setBusArrangements: unsafe extern "system" fn(
        this: *mut c_void,
        inputs: *mut u64,
        numIns: i32,
        outputs: *mut u64,
        numOuts: i32,
    ) -> tresult,
    pub getBusArrangement: unsafe extern "system" fn(
        this: *mut c_void,
        dir: BusDirection,
        index: i32,
        arr: *mut u64,
    ) -> tresult,
    pub canProcessSampleSize:
        unsafe extern "system" fn(this: *mut c_void, symbolicSampleSize: i32) -> tresult,
    pub getLatencySamples: unsafe extern "system" fn(this: *mut c_void) -> u32,
    pub setupProcessing:
        unsafe extern "system" fn(this: *mut c_void, setup: *mut ProcessSetup) -> tresult,
    pub setProcessing: unsafe extern "system" fn(this: *mut c_void, state: TBool) -> tresult,
    pub process: unsafe extern "system" fn(this: *mut c_void, data: *mut ProcessData) -> tresult,
    pub getTailSamples: unsafe extern "system" fn(this: *mut c_void) -> u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ParameterInfo {
    pub id: ParamID,
    pub title: [u16; 128],
    pub shortTitle: [u16; 128],
    pub units: [u16; 128],
    pub stepCount: i32,
    pub defaultNormalizedValue: ParamValue,
    pub unitId: i32,
    pub flags: i32,
}

#[repr(C)]
pub struct IEditControllerVtbl {
    pub base: IPluginBaseVtbl,
    pub setComponentState:
        unsafe extern "system" fn(this: *mut c_void, state: *mut c_void) -> tresult,
    pub setState: unsafe extern "system" fn(this: *mut c_void, state: *mut c_void) -> tresult,
    pub getState: unsafe extern "system" fn(this: *mut c_void, state: *mut c_void) -> tresult,
    pub getParameterCount: unsafe extern "system" fn(this: *mut c_void) -> i32,
    pub getParameterInfo: unsafe extern "system" fn(
        this: *mut c_void,
        paramIndex: i32,
        info: *mut ParameterInfo,
    ) -> tresult,
    pub getParamStringByValue: unsafe extern "system" fn(
        this: *mut c_void,
        id: ParamID,
        valueNormalized: ParamValue,
        string: *mut u16,
    ) -> tresult,
    pub getParamValueByString: unsafe extern "system" fn(
        this: *mut c_void,
        id: ParamID,
        string: *mut u16,
        valueNormalized: *mut ParamValue,
    ) -> tresult,
    pub normalizedParamToPlain: unsafe extern "system" fn(
        this: *mut c_void,
        id: ParamID,
        valueNormalized: ParamValue,
    ) -> ParamValue,
    pub plainParamToNormalized: unsafe extern "system" fn(
        this: *mut c_void,
        id: ParamID,
        plainValue: ParamValue,
    ) -> ParamValue,
    pub getParamNormalized: unsafe extern "system" fn(this: *mut c_void, id: ParamID) -> ParamValue,
    pub setParamNormalized:
        unsafe extern "system" fn(this: *mut c_void, id: ParamID, value: ParamValue) -> tresult,
    pub setComponentHandler:
        unsafe extern "system" fn(this: *mut c_void, handler: *mut c_void) -> tresult,
    pub createView:
        unsafe extern "system" fn(this: *mut c_void, name: *const c_char) -> *mut c_void,
}

#[repr(C)]
pub struct IComponentHandlerVtbl {
    pub base: FUnknownVtbl,
    pub beginEdit: unsafe extern "system" fn(this: *mut c_void, id: ParamID) -> tresult,
    pub performEdit: unsafe extern "system" fn(
        this: *mut c_void,
        id: ParamID,
        valueNormalized: ParamValue,
    ) -> tresult,
    pub endEdit: unsafe extern "system" fn(this: *mut c_void, id: ParamID) -> tresult,
    pub restartComponent: unsafe extern "system" fn(this: *mut c_void, flags: i32) -> tresult,
}

#[repr(C)]
pub struct IHostApplicationVtbl {
    pub base: FUnknownVtbl,
    pub getName: unsafe extern "system" fn(this: *mut c_void, name: *mut u16) -> tresult,
    pub createInstance: unsafe extern "system" fn(
        this: *mut c_void,
        cid: *const TUID,
        iid: *const TUID,
        obj: *mut *mut c_void,
    ) -> tresult,
}

#[repr(C)]
pub struct IBStreamVtbl {
    pub base: FUnknownVtbl,
    pub read: unsafe extern "system" fn(
        this: *mut c_void,
        buffer: *mut c_void,
        numBytes: i32,
        numBytesRead: *mut i32,
    ) -> tresult,
    pub write: unsafe extern "system" fn(
        this: *mut c_void,
        buffer: *mut c_void,
        numBytes: i32,
        numBytesWritten: *mut i32,
    ) -> tresult,
    pub seek: unsafe extern "system" fn(
        this: *mut c_void,
        pos: i64,
        mode: i32,
        result: *mut i64,
    ) -> tresult,
    pub tell: unsafe extern "system" fn(this: *mut c_void, pos: *mut i64) -> tresult,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ViewRect {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

#[repr(C)]
pub struct IPlugViewVtbl {
    pub base: FUnknownVtbl,
    pub isPlatformTypeSupported:
        unsafe extern "system" fn(this: *mut c_void, type_: *const c_char) -> tresult,
    pub attached: unsafe extern "system" fn(
        this: *mut c_void,
        parent: *mut c_void,
        type_: *const c_char,
    ) -> tresult,
    pub removed: unsafe extern "system" fn(this: *mut c_void) -> tresult,
    pub onWheel: unsafe extern "system" fn(this: *mut c_void, distance: f32) -> tresult,
    pub onKeyDown: unsafe extern "system" fn(
        this: *mut c_void,
        key: u16,
        keyCode: i16,
        modifiers: i16,
    ) -> tresult,
    pub onKeyUp: unsafe extern "system" fn(
        this: *mut c_void,
        key: u16,
        keyCode: i16,
        modifiers: i16,
    ) -> tresult,
    pub getSize: unsafe extern "system" fn(this: *mut c_void, size: *mut ViewRect) -> tresult,
    pub onSize: unsafe extern "system" fn(this: *mut c_void, newSize: *mut ViewRect) -> tresult,
    pub onFocus: unsafe extern "system" fn(this: *mut c_void, state: TBool) -> tresult,
    pub setFrame: unsafe extern "system" fn(this: *mut c_void, frame: *mut c_void) -> tresult,
    pub canResize: unsafe extern "system" fn(this: *mut c_void) -> tresult,
    pub checkSizeConstraint:
        unsafe extern "system" fn(this: *mut c_void, rect: *mut ViewRect) -> tresult,
}

#[repr(C)]
pub struct IParamValueQueueVtbl {
    pub base: FUnknownVtbl,
    pub getParameterId: unsafe extern "system" fn(this: *mut c_void) -> ParamID,
    pub getPointCount: unsafe extern "system" fn(this: *mut c_void) -> i32,
    pub getPoint: unsafe extern "system" fn(
        this: *mut c_void,
        index: i32,
        sampleOffset: *mut i32,
        value: *mut ParamValue,
    ) -> tresult,
    pub addPoint: unsafe extern "system" fn(
        this: *mut c_void,
        sampleOffset: i32,
        value: ParamValue,
        index: *mut i32,
    ) -> tresult,
}

#[repr(C)]
pub struct IParameterChangesVtbl {
    pub base: FUnknownVtbl,
    pub getParameterCount: unsafe extern "system" fn(this: *mut c_void) -> i32,
    pub getParameterData: unsafe extern "system" fn(this: *mut c_void, index: i32) -> *mut c_void,
    pub addParameterData: unsafe extern "system" fn(
        this: *mut c_void,
        id: *const ParamID,
        index: *mut i32,
    ) -> *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct NoteOnEvent {
    pub channel: i16,
    pub pitch: i16,
    pub tuning: f32,
    pub velocity: f32,
    pub length: i32,
    pub noteId: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct NoteOffEvent {
    pub channel: i16,
    pub pitch: i16,
    pub velocity: f32,
    pub noteId: i32,
    pub tuning: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union Vst3EventPayload {
    pub noteOn: NoteOnEvent,
    pub noteOff: NoteOffEvent,
    pub raw_bytes: [u8; 32],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vst3Event {
    pub busIndex: i32,
    pub sampleOffset: i32,
    pub ppqPosition: f64,
    pub flags: u16,
    pub type_: u16,
    pub payload: Vst3EventPayload,
}

impl Default for Vst3Event {
    fn default() -> Self {
        Self {
            busIndex: 0,
            sampleOffset: 0,
            ppqPosition: 0.0,
            flags: 1,
            type_: K_NOTE_ON_EVENT,
            payload: Vst3EventPayload { raw_bytes: [0; 32] },
        }
    }
}

#[repr(C)]
pub struct IEventListVtbl {
    pub base: FUnknownVtbl,
    pub getEventCount: unsafe extern "system" fn(this: *mut c_void) -> i32,
    pub getEvent:
        unsafe extern "system" fn(this: *mut c_void, index: i32, e: *mut Vst3Event) -> tresult,
    pub addEvent: unsafe extern "system" fn(this: *mut c_void, e: *mut Vst3Event) -> tresult,
}

#[repr(C)]
pub struct IConnectionPointVtbl {
    pub base: FUnknownVtbl,
    pub connect: unsafe extern "system" fn(this: *mut c_void, other: *mut c_void) -> tresult,
    pub disconnect: unsafe extern "system" fn(this: *mut c_void, other: *mut c_void) -> tresult,
    pub notify: unsafe extern "system" fn(this: *mut c_void, message: *mut c_void) -> tresult,
}

#[repr(C)]
pub struct IPlugFrameVtbl {
    pub base: FUnknownVtbl,
    pub resizeView: unsafe extern "system" fn(
        this: *mut c_void,
        view: *mut c_void,
        newSize: *mut ViewRect,
    ) -> tresult,
}
