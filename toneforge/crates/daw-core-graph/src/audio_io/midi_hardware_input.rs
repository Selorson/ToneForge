//! `midi_hardware_input`: Native Win32 Multimedia **MIDI Input** (`winmm.dll` `midiInGetNumDevs`,
//! `midiInGetDevCapsW`, `midiInOpen`, `midiInStart` `MIM_DATA` callback) pushing sample-accurate
//! [`SequencerVoiceEvent`]s into a lock-free `rtrb` SPSC queue (`0` heap allocations, `0` locks).

use crate::apdc_engine::VoiceEventKind;
use crate::sequencer_pianoroll::SequencerVoiceEvent;
use serde::{Deserialize, Serialize};
use std::cell::UnsafeCell;
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
        0xE0 => {
            let bend14 = ((d2 as i32) << 7) | (d1 as i32);
            let cents = (((bend14 - 8192) as f32) / 8192.0) * 200.0;
            Some(SequencerVoiceEvent {
                sample_pos,
                ppq_tick: 0,
                channel_id: target_channel_id,
                note_id: 0,
                kind: VoiceEventKind::PitchBend,
                key: 0,
                velocity: 0.0,
                pan: 0.0,
                fine_pitch_cents: cents,
                slide_duration_ticks: 0,
                slide_duration_samples: 0,
                color_group: midi_ch,
            })
        }
        _ => None,
    }
}

struct MidiCallbackContext {
    tx: UnsafeCell<rtrb::Producer<SequencerVoiceEvent>>,
    active: AtomicBool,
    target_channel_id: AtomicU32,
    pitch_bend_cents_bits: [AtomicU32; 16],
    received_messages: AtomicU64,
}

// SAFETY: `tx` is accessed exclusively inside `win32_midi_in_proc` (which WinMM serializes per
// `HMIDIIN` handle while `active == true`) and is never touched from any other thread until after
// `midiInStop` + `midiInReset` + `midiInClose` return in `Win32MidiInputPort::drop`. All other
// fields are atomic, and `win32_midi_in_proc` only ever creates a shared `&MidiCallbackContext`.
unsafe impl Sync for MidiCallbackContext {}

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
    // SAFETY: Shared `&MidiCallbackContext` only — never `&mut MidiCallbackContext`, preserving
    // LLVM `noalias` soundness while the UI/Host thread concurrently accesses atomic fields.
    let ctx = unsafe { &*(dw_instance as *const MidiCallbackContext) };
    if !ctx.active.load(Ordering::Acquire) {
        return;
    }
    ctx.received_messages.fetch_add(1, Ordering::Relaxed);
    let raw_msg = dw_param1 as u32;
    let status = (raw_msg & 0xFF) as u8;
    let status_cmd = status & 0xF0;
    let midi_ch = (status & 0x0F) as usize;
    let arrival_ns = midi_arrival_clock_ns();
    let target_ch = ctx.target_channel_id.load(Ordering::Relaxed);

    if status_cmd == 0xE0 {
        if let Some(ev) = decode_raw_midi_short_message(raw_msg, arrival_ns, target_ch, 0.0) {
            ctx.pitch_bend_cents_bits[midi_ch]
                .store(ev.fine_pitch_cents.to_bits(), Ordering::Relaxed);
            // SAFETY: WinMM serializes `MIM_DATA` callbacks for a single `HMIDIIN`.
            let tx = unsafe { &mut *ctx.tx.get() };
            let _ = tx.push(ev);
        }
        return;
    }

    let bend_cents = f32::from_bits(ctx.pitch_bend_cents_bits[midi_ch].load(Ordering::Relaxed));
    if let Some(ev) = decode_raw_midi_short_message(raw_msg, arrival_ns, target_ch, bend_cents) {
        // SAFETY: WinMM serializes `MIM_DATA` callbacks for a single `HMIDIIN`.
        let tx = unsafe { &mut *ctx.tx.get() };
        let _ = tx.push(ev);
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
        let zero_bits = 0.0_f32.to_bits();
        let boxed_ctx = Box::new(MidiCallbackContext {
            tx: UnsafeCell::new(tx),
            active: AtomicBool::new(true),
            target_channel_id: AtomicU32::new(initial_target_channel_id),
            pitch_bend_cents_bits: std::array::from_fn(|_| AtomicU32::new(zero_bits)),
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

#[cfg(test)]
mod midi_decode_tests {
    use super::*;
    use crate::apdc_engine::VoiceEventKind;

    #[test]
    fn test_decode_win32_midi_pitch_bend_14bit_and_note_on_zero_velocity() {
        // Center pitch bend: LSB = 0x00, MSB = 0x40 -> 0x2000 -> 0.0 cents
        let center_word = 0xE2u32 | (0x00u32 << 8) | (0x40u32 << 16);
        let ev_center = decode_raw_midi_short_message(center_word, 7, 1, 0.0).unwrap();
        assert_eq!(ev_center.kind, VoiceEventKind::PitchBend);
        assert_eq!(ev_center.color_group, 2);
        assert!(ev_center.fine_pitch_cents.abs() < 1e-4);

        // Max upward pitch bend: LSB = 0x7F, MSB = 0x7F -> 16383 -> +200.0 cents
        let up_word = 0xE0u32 | (0x7Fu32 << 8) | (0x7Fu32 << 16);
        let ev_up = decode_raw_midi_short_message(up_word, 7, 1, 0.0).unwrap();
        assert_eq!(ev_up.kind, VoiceEventKind::PitchBend);
        assert!((ev_up.fine_pitch_cents - 200.0).abs() < 0.05);

        // Max downward pitch bend: LSB = 0x00, MSB = 0x00 -> 0 -> -200.0 cents
        let down_word = 0xE0u32;
        let ev_down = decode_raw_midi_short_message(down_word, 7, 1, 0.0).unwrap();
        assert_eq!(ev_down.kind, VoiceEventKind::PitchBend);
        assert!((ev_down.fine_pitch_cents + 200.0).abs() < 1e-3);

        // NoteOn carries current channel pitch bend cents; velocity 0 -> NoteOff
        let note_on_word = 0x90u32 | (60u32 << 8) | (100u32 << 16);
        let ev_on = decode_raw_midi_short_message(note_on_word, 1, 1, -75.0).unwrap();
        assert_eq!(ev_on.kind, VoiceEventKind::NoteOn);
        assert!((ev_on.fine_pitch_cents + 75.0).abs() < 1e-5);

        let note_off_word = 0x90u32 | (60u32 << 8);
        let ev_off = decode_raw_midi_short_message(note_off_word, 1, 1, 0.0).unwrap();
        assert_eq!(ev_off.kind, VoiceEventKind::NoteOff);
        assert_eq!(ev_off.key, 60);
    }
}
