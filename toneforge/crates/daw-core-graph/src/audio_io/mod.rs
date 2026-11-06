//! Windows MMCSS `"Pro Audio"` Thread Elevation (`AvSetMmThreadCharacteristicsW`, `0x0044b6d0` in `FLEngine_x64.dll`)
//! and WASAPI-to-ASIO Real-Time Ring Adapter (`FL Studio ASIO.dll` `sub_39b8ff0` / `0x039B9B80`).

pub mod mmcss_pro_audio;
pub mod wasapi_asio_bridge;

pub use mmcss_pro_audio::{MmcssThreadPriority, ProAudioThreadGuard};
pub use wasapi_asio_bridge::{
    ASIO_INT32_SCALE_F32, AsioSampleFormat, WasapiAsioBlockAdapter, asio_planar_f32_to_int32_lsb,
    asio_planar_to_stereo_frames, convert_f32_to_int16_lsb, convert_f32_to_int24_lsb,
    convert_f32_to_int32_lsb, convert_int32_lsb_to_f32, stereo_frames_to_asio_planar,
    stereo_frames_to_wasapi_interleaved, wasapi_interleaved_to_stereo_frames,
};
