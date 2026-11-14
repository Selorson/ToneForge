//! Windows MMCSS `"Pro Audio"` Thread Elevation (`AvSetMmThreadCharacteristicsW`, `0x0044b6d0` in `FLEngine_x64.dll`),
//! WASAPI-to-ASIO Real-Time Ring Adapter (`FL Studio ASIO.dll` `sub_39b8ff0` / `0x039B9B80`),
//! and Production Win32 Core Audio **WASAPI Hardware Driver** (`Shared Event-Driven` + `Exclusive Low-Latency`).

pub mod asio_hardware_driver;
pub mod disk_track_recorder;
pub mod midi_hardware_input;
pub mod mmcss_pro_audio;
pub mod wasapi_asio_bridge;
pub mod wasapi_capture_duplex;
pub mod wasapi_hardware_driver;

pub use asio_hardware_driver::{
    ASE_OK, ASE_SUCCESS, ASIO_FALSE, ASIO_TRUE, ASIOST_FLOAT32_LSB, ASIOST_FLOAT64_LSB,
    ASIOST_INT16_LSB, ASIOST_INT24_LSB, ASIOST_INT32_LSB, AsioDriverInfo, AsioHardwareStream,
    AsioStreamStatusSnapshot, enumerate_asio_drivers,
};
pub use disk_track_recorder::{
    DiskTrackRecorder, RECORD_RING_DEFAULT_CAPACITY_FRAMES, RecordedTakeInfo, RtRecordTap,
    pack_telemetry_record_status, read_wav_f32_stereo, unpack_telemetry_record_status,
};
pub use midi_hardware_input::{
    MidiInputDeviceInfo, Win32MidiInputPort, decode_raw_midi_short_message,
    enumerate_midi_input_devices, midi_arrival_clock_ns,
};
pub use mmcss_pro_audio::{MmcssThreadPriority, ProAudioThreadGuard};
pub use wasapi_asio_bridge::{
    ASIO_INT32_SCALE_F32, AsioSampleFormat, WasapiAsioBlockAdapter, asio_planar_f32_to_int32_lsb,
    asio_planar_to_stereo_frames, convert_f32_to_int16_lsb, convert_f32_to_int24_lsb,
    convert_f32_to_int32_lsb, convert_int32_lsb_to_f32, stereo_frames_to_asio_planar,
    stereo_frames_to_wasapi_interleaved, wasapi_interleaved_to_stereo_frames,
};
pub use wasapi_capture_duplex::{
    AdaptiveDllDriftResampler, CLOCK_LOCKED_GUARD_FRAMES, CaptureBridgeMode, DLL_MAX_PPM_ADJUST,
    DLL_RING_CAPACITY, DLL_TARGET_FILL_FRAMES, DLL_TARGET_FILL_SECONDS,
    OUTPUT_SRC_RENDER_LEAD_ENGINE_FRAMES, WasapiCaptureStatusSnapshot, WasapiCaptureStream,
    enumerate_wasapi_input_devices, unpack_hardware_capture_to_stereo,
};
pub use wasapi_hardware_driver::{
    WASAPI_ENGINE_RENDER_QUANTUM, WasapiEndpointInfo, WasapiHardwareStream, WasapiSampleFormat,
    WasapiSharedControl, WasapiStreamStatusSnapshot, WasapiStreamingResampler,
    enumerate_wasapi_output_devices, pack_stereo_frames_to_hardware_buffer,
};
