//! `ai-uvr5-stem-engine` — Complete Clean-Room Implementation of Ozone 12 Real-Time BSRNN
//! Stem Engine & UVR5 Multi-Architecture Offline Stem Separation + Spectral Ensemble:
//!
//! 1. **Real-Time Low-Latency (`120 ms` = `5,760` samples @ 48 kHz) 4-Stem BSRNN Pipeline**
//!    (`RealTimeBsrnnStemNode` / `RealtimeBsrnnStemNode`):
//!    - Lock-Free `rtrb::RingBuffer` SPSC worker queue (`0` heap allocations on the audio thread).
//!    - 13-subband BSRNN STFT (`N_fft = 2048`, `hop = 512`) + recurrent state `[2, 2, 2, 1, 13, 128]`.
//!    - Unity-Identity Additive stem rebalance & EQ (`Y = X_delayed + sum_s (Delta H_s - 1) * S_s`),
//!      strictly bit-transparent when stem gains are `0.0 dB`.
//!    - ONNX model descriptor & verifier for `ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx` and
//!      `delimit_exp009_ep080_fixed2.onnx`.
//! 2. **Offline Multi-Model UVR5 Stem Separation Framework**:
//!    - **MDX-Net / MDX23C `TFC-TDF v3`** (`TfcTdfBlock`, `Mdx23cSeparator`, Complex-as-Channels `CaC`,
//!      Nyquist bin trim/pad, STFT overlap-add with Hann window chunk boundary anti-aliasing, and
//!      2-pass phase-inverted bias denoising).
//!    - **Hybrid Transformer Demucs v4 (`HTDemucs4`)** (`CrossTransformerEncoder` bidirectional
//!      Time <-> Frequency cross-attention + dual-domain waveform/STFT U-Net synthesis).
//!    - **VR Architecture `CascadedASPPNet`** (`AsppModule` with dilations `[1, 2, 4, 8, 16]` +
//!      3-stage cascaded multi-band U-Net + UVR5 aggressiveness tuning).
//! 3. **Multi-Model `SpectralEnsembler`**:
//!    - Supports `MaxSpec` (`MAX_SPEC`), `MinSpec` (`MIN_SPEC`), `Average` (`AVERAGE`),
//!      `Median`, and `Weighted` ensemble modes with complex phase preservation, inter-model
//!      circular phase coherence denoising, and `invert_audio` phase-inverted stem subtraction.

pub mod bsrnn_onnx_gpu;
pub mod bsrnn_streaming;
pub mod mdx_onnx_gpu;
pub mod npz_gru_legacy;
pub mod onnx_gpu_runtime;
pub mod spectral_ensemble;
pub mod uvr5_htdemucs;
pub mod uvr5_mdx_tfctdf;
pub mod uvr5_vr_aspp;

pub use bsrnn_onnx_gpu::{
    OZONE_BSRNN_LOOKAHEAD_HOPS, OZONE_BSRNN_LOOKAHEAD_SAMPLES, OZONE_BSRNN_SPEC_LEN,
    OZONE_BSRNN_STATES_LEN, OZONE_DELIMIT_SPEC_IN_LEN, OZONE_DELIMIT_SPEC_OUT_LEN,
    OZONE_DELIMIT_STATES_LEN, OzoneBsrnnOnnxSeparator, OzoneDelimitOnnxProcessor,
};
pub use bsrnn_streaming::{
    BSRNN_FREQ_BINS, BSRNN_HIDDEN_STATE_LEN, BSRNN_HIDDEN_STATE_SHAPE, BSRNN_HOP_SIZE,
    BSRNN_LATENCY_SAMPLES_48K, BSRNN_N_FFT, BSRNN_REALTIME_LATENCY_SEC, BSRNN_STEM_COUNT,
    BSRNN_SUBBAND_BINS, BsrnnInputHop, BsrnnOutputHop, BsrnnStreamingWorker,
    DEFAULT_BSRNN_ONNX_PATHS, DEFAULT_DELIMIT_ONNX_PATHS, OnnxModelDescriptor,
    RealTimeBsrnnStemNode, RealtimeBsrnnStemNode, StemKind,
};
pub use mdx_onnx_gpu::{
    GpuStemEnsembleConfig, GpuStemEnsembleEngine, GpuStemEnsembleOutput, GpuStemModelSelection,
    MdxOnnxModelConfig, MdxPrimaryStem, UVR5_MDX_DIM_F, UVR5_MDX_DIM_T, UVR5_MDX_HOP_LENGTH,
    UVR5_MDX_SAMPLE_RATE_HZ, Uvr5MdxOnnxSeparator, resample_stereo_frames,
    run_global_gpu_stem_ensemble,
};
pub use npz_gru_legacy::{
    DEFAULT_GPARAMS_NPZ_PATHS, DEFAULT_INSTRUMENT_NPZ_PATHS, DEFAULT_VOCALS_GRU_NPZ_PATHS,
    DenseLayerWeights, GenreCnnClassifier, GruLayerWeights, InstrumentCnnClassifier,
    NpzArchiveReader, NumpyArrayF32, OzoneNpzGruEngine, load_npz_archive_bytes,
    load_npz_archive_file,
};
pub use onnx_gpu_runtime::{
    GpuAcceleratorStatus, GpuExecutionTier, GpuRuntimeManager, NvidiaCudaDeviceProbe,
    candidate_toneforge_roots, probe_nvidia_cuda_driver, resolve_asset_file, resolve_assets_dir,
    resolve_gpu_runtime_dir, resolve_ozone_bsrnn_onnx_path, resolve_ozone_delimit_onnx_path,
    resolve_uvr5_inst_hq3_onnx_path, resolve_uvr5_kim_vocal2_onnx_path,
};
pub use spectral_ensemble::{
    SpectralEnsembleMode, SpectralEnsembler, combine_complex_spectrogram_frame,
    combine_complex_spectrogram_frame_weighted,
};
pub use uvr5_htdemucs::{CrossTransformerEncoder, DemucsStem, HTDemucs4};
pub use uvr5_mdx_tfctdf::{Mdx23cSeparator, TfcTdfBlock};
pub use uvr5_vr_aspp::{AsppModule, CascadedASPPNet, VR_ASPP_DILATIONS};
