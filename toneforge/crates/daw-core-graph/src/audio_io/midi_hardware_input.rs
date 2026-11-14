//! `midi_hardware_input`: Native Win32 Multimedia **MIDI Input** (`winmm.dll` `midiInGetNumDevs`,
//! `midiInGetDevCapsW`, `midiInOpen`, `midiInStart` `MIM_DATA` callback) pushing sample-accurate
//! [`SequencerVoiceEvent`]s into a lock-free `rtrb` SPSC queue (`0` heap allocations, `0` locks).

use crate::apdc_engine::VoiceEventKind;
use crate::sequencer_pianoroll::SequencerVoiceEvent;
use serde::{Deserialize, Serialize};
use std::ffi::c_void;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

/// Monotonic clock (ns) shared by the MIDI input callback and the audio engine: incoming events are
/// stamped with their arrival time, and the engine places them at the matching offset of the next
/// block (constant one-block latency, no block-quantisation jitter). Allocation-free.
#[inline]
pub fn midi_arrival_clock_ns() -> i64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    EPOCH.get_or_init(Instant::now).elapsed().as_nanos() as i64
}

const MMSYSERR_NOERROR: u32 = 0;
const CALLBACK_FUNCTION: u32 = 0x0003_0000;
const MIM_DATA: u32 = 0x03C3;

#[repr(C)]
struct MidiInCapsW {
    w_mid: u16,
    w_pid: u16,
    v_driver_version: u32,
    sz_pname: [u16; 32],
    dw_support: u32,
}

const _: () = assert!(std::mem::size_of::<MidiInCapsW>() == 76);

#[cfg(target_os = "windows")]
#[link(name = "winmm")]
unsafe extern "system" {
    fn midiInGetNumDevs() -> u32;
    fn midiInGetDevCapsW(u_device_id: usize, pmic: *mut MidiInCapsW, cbmic: u32) -> u32;
    fn midiInOpen(
        phmi: *mut *mut c_void,
        u_device_id: u32,
        dw_callback: usize,
        dw_instance: usize,
        fdw_open: u32,
    ) -> u32;
    fn midiInStart(hmi: *mut c_void) -> u32;
    fn midiInStop(hmi: *mut c_void) -> u32;
    fn midiInReset(hmi: *mut c_void) -> u32;
    fn midiInClose(hmi: *mut c_void) -> u32;
}

/// Hardware MIDI Input device descriptor (`midiInGetDevCapsW`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MidiInputDeviceInfo {
    pub device_id: u32,
    pub name: String,
    pub manufacturer_id: u16,
    pub product_id: u16,
}

/// Enumerates all connected Win32 MIDI Input controllers / keyboards (`midiInGetNumDevs`).
pub fn enumerate_midi_input_devices() -> Vec<MidiInputDeviceInfo> {
    let mut list = Vec::new();
    #[cfg(target_os = "windows")]
    unsafe {
        let count = midiInGetNumDevs();
        for id in 0..count {
            let mut caps: MidiInCapsW = std::mem::zeroed();
            if midiInGetDevCapsW(
                id as usize,
                &mut caps,
                std::mem::size_of::<MidiInCapsW>() as u32,
            ) == MMSYSERR_NOERROR
            {
                let nul = caps
                    .sz_pname
                    .iter()
                    .position(|&c| c == 0)
                    .unwrap_or(caps.sz_pname.len());
                let name = String::from_utf16_lossy(&caps.sz_pname[..nul])
                    .trim()
                    .to_string();
                list.push(MidiInputDeviceInfo {
                    device_id: id,
                    name,
                    manufacturer_id: caps.w_mid,
                    product_id: caps.w_pid,
                });
            }
        }
    }
    list
}

/// Pure zero-allocation decoder converting a raw 24-bit Win32 `MIM_DATA` short MIDI message
/// (`status = msg & 0xFF`, `data1 = (msg >> 8) & 0x7F`, `data2 = (msg >> 16) & 0x7F`) into a
/// sample-accurate [`SequencerVoiceEvent`].
#[inline]
pub fn decode_raw_midi_short_message(
    raw_msg: u32,
    sample_pos: i64,
    target_channel_id: u32,
    pitch_bend_cents: f32,
) -> Option<SequencerVoiceEvent> {
    let status = (raw_msg & 0xFF) as u8;
    let cmd = status & 0xF0;
    let midi_ch = status & 0x0F;
    let d1 = ((raw_msg >> 8) & 0x7F) as u8;
    let d2 = ((raw_msg >> 16) & 0x7F) as u8;

    match cmd {
        0x90 if d2 > 0 => Some(SequencerVoiceEvent {
            sample_pos,
            ppq_tick: 0,
            channel_id: target_channel_id,
            note_id: d1 as u32,
            kind: VoiceEventKind::NoteOn,
            key: d1,
            velocity: (d2 as f32 / 127.0).clamp(0.0, 1.0),
            pan: 0.0,
            fine_pitch_cents: pitch_bend_cents,
            slide_duration_ticks: 0,
            slide_duration_samples: 0,
            color_group: midi_ch,
        }),
        0x80 | 0x90 => Some(SequencerVoiceEvent {
            sample_pos,
            ppq_tick: 0,
            channel_id: target_channel_id,
            note_id: d1 as u32,
            kind: VoiceEventKind::NoteOff,
            key: d1,
            velocity: (d2 as f32 / 127.0).clamp(0.0, 1.0),
            pan: 0.0,
            fine_pitch_cents: pitch_bend_cents,
            slide_duration_ticks: 0,
            slide_duration_samples: 0,
            color_group: midi_ch,
        }),
        0xB0 if d1 == 120 || d1 == 123 => Some(SequencerVoiceEvent {
            sample_pos,
            ppq_tick: 0,
            channel_id: target_channel_id,
            note_id: 0,
            kind: VoiceEventKind::ChannelMute,
            key: 0,
            velocity: 0.0,
            pan: 0.0,
            fine_pitch_cents: 0.0,
            slide_duration_ticks: 0,
            slide_duration_samples: 0,
            color_group: midi_ch,
        }),
        _ => None,
    }
}

struct MidiCallbackContext {
    tx: rtrb::Producer<SequencerVoiceEvent>,
    active: AtomicBool,
    target_channel_id: AtomicU32,
    pitch_bend_cents_bits: AtomicU32,
    received_messages: AtomicU64,
}

#[cfg(target_os = "windows")]
unsafe extern "system" fn win32_midi_in_proc(
    _h_midi_in: *mut c_void,
    w_msg: u32,
    dw_instance: usize,
    dw_param1: usize,
    _dw_param2: usize,
) {
    if w_msg != MIM_DATA || dw_instance == 0 {
        return;
    }
    // SAFETY: `dw_instance` points to a pinned `MidiCallbackContext` kept alive until AFTER
    // `midiInStop` + `midiInReset` + `midiInClose` return in `Win32MidiInputPort::drop`.
    let ctx = unsafe { &mut *(dw_instance as *mut MidiCallbackContext) };
    if !ctx.active.load(Ordering::Acquire) {
        return;
    }
    ctx.received_messages.fetch_add(1, Ordering::Relaxed);
    let raw_msg = dw_param1 as u32;
    let status_cmd = (raw_msg & 0xF0) as u8;
    if status_cmd == 0xE0 {
        // 14-bit PitchBend (`8192` = center, `±200 cents` standard ±2 semitone range)
        let lsb = ((raw_msg >> 8) & 0x7F) as i32;
        let msb = ((raw_msg >> 16) & 0x7F) as i32;
        let bend14 = (msb << 7) | lsb;
        let cents = (((bend14 - 8192) as f32) / 8192.0) * 200.0;
        ctx.pitch_bend_cents_bits
            .store(cents.to_bits(), Ordering::Relaxed);
        return;
    }

    let bend_cents = f32::from_bits(ctx.pitch_bend_cents_bits.load(Ordering::Relaxed));
    // `sample_pos` carries the arrival time until the engine converts it to a block offset.
    let arrival_ns = midi_arrival_clock_ns();
    let target_ch = ctx.target_channel_id.load(Ordering::Relaxed);
    if let Some(ev) = decode_raw_midi_short_message(raw_msg, arrival_ns, target_ch, bend_cents) {
        let _ = ctx.tx.push(ev);
    }
}

/// Active Win32 Hardware MIDI Input Port (`midiInOpen` -> `rtrb::Consumer<SequencerVoiceEvent>`).
pub struct Win32MidiInputPort {
    device_id: u32,
    h_midi: *mut c_void,
    cb_ctx: *mut MidiCallbackContext,
}

unsafe impl Send for Win32MidiInputPort {}

impl Win32MidiInputPort {
    /// Opens hardware MIDI input `device_id` and returns `(Win32MidiInputPort, rtrb::Consumer<SequencerVoiceEvent>)`.
    pub fn open(
        device_id: u32,
        initial_target_channel_id: u32,
        ring_capacity: usize,
    ) -> Result<(Self, rtrb::Consumer<SequencerVoiceEvent>), String> {
        let (tx, rx) = rtrb::RingBuffer::new(ring_capacity.max(256));
        let boxed_ctx = Box::new(MidiCallbackContext {
            tx,
            active: AtomicBool::new(true),
            target_channel_id: AtomicU32::new(initial_target_channel_id),
            pitch_bend_cents_bits: AtomicU32::new(0.0_f32.to_bits()),
            received_messages: AtomicU64::new(0),
        });
        let ctx_ptr = Box::into_raw(boxed_ctx);

        #[cfg(target_os = "windows")]
        {
            let mut h_midi: *mut c_void = std::ptr::null_mut();
            let rc = unsafe {
                midiInOpen(
                    &mut h_midi,
                    device_id,
                    win32_midi_in_proc as *const () as usize,
                    ctx_ptr as usize,
                    CALLBACK_FUNCTION,
                )
            };
            if rc != MMSYSERR_NOERROR || h_midi.is_null() {
                unsafe {
                    drop(Box::from_raw(ctx_ptr));
                }
                return Err(format!(
                    "midiInOpen(device_id={device_id}) failed with MMRESULT={rc}"
                ));
            }
            let start_rc = unsafe { midiInStart(h_midi) };
            if start_rc != MMSYSERR_NOERROR {
                unsafe {
                    let _ = midiInClose(h_midi);
                    drop(Box::from_raw(ctx_ptr));
                }
                return Err(format!(
                    "midiInStart(device_id={device_id}) failed with MMRESULT={start_rc}"
                ));
            }
            Ok((
                Self {
                    device_id,
                    h_midi,
                    cb_ctx: ctx_ptr,
                },
                rx,
            ))
        }
        #[cfg(not(target_os = "windows"))]
        {
            unsafe {
                drop(Box::from_raw(ctx_ptr));
            }
            let _ = rx;
            Err("Win32 MIDI In is only supported on Windows".to_string())
        }
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    #[inline]
    pub fn set_target_channel_id(&self, channel_id: u32) {
        if !self.cb_ctx.is_null() {
            unsafe {
                (*self.cb_ctx)
                    .target_channel_id
                    .store(channel_id, Ordering::Relaxed);
            }
        }
    }

    #[inline]
    pub fn received_message_count(&self) -> u64 {
        if self.cb_ctx.is_null() {
            0
        } else {
            unsafe { (*self.cb_ctx).received_messages.load(Ordering::Relaxed) }
        }
    }
}

impl Drop for Win32MidiInputPort {
    fn drop(&mut self) {
        if !self.cb_ctx.is_null() {
            unsafe {
                (*self.cb_ctx).active.store(false, Ordering::Release);
            }
        }
        #[cfg(target_os = "windows")]
        if !self.h_midi.is_null() {
            unsafe {
                let _ = midiInStop(self.h_midi);
                let _ = midiInReset(self.h_midi);
                let _ = midiInClose(self.h_midi);
            }
            self.h_midi = std::ptr::null_mut();
        }
        if !self.cb_ctx.is_null() {
            unsafe {
                drop(Box::from_raw(self.cb_ctx));
            }
            self.cb_ctx = std::ptr::null_mut();
        }
    }
}
