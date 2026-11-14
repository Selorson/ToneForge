//! Self-Contained 3-Tier ONNX Runtime GPU Accelerator Manager (`onnx_gpu_runtime.rs`).
//!
//! Hardware Acceleration Hierarchy:
//! - **Tier 1 (`CudaTensorCore`)**: NVIDIA CUDA 12 + cuDNN 9 + NVRTC (`onnxruntime-gpu==1.25.1`)
//!   dynamically loaded from `<exe_dir>/gpu_runtime/cuda/` with TensorFloat-32 (`TF32 = true`)
//!   and `ConvAlgorithmSearch::Heuristic` for Ada Lovelace (`RTX 4080`) Tensor Cores.
//! - **Tier 2 (`DirectMlDx12`)**: Universal Windows DirectX 12 GPU (`onnxruntime-directml==1.24.4`)
//!   dynamically loaded from `<exe_dir>/gpu_runtime/directml/`.
//! - **Tier 3A (`CpuOrt`)**: Multi-threaded AVX2 `CPUExecutionProvider` via ONNX Runtime.
//! - **Tier 3B (`PureRustFallback`)**: Deterministic pure-Rust spectral engine when no ORT DLLs exist.
//!
//! Path Resolution Contract:
//! Every lookup for `gpu_runtime/` and `assets/` strictly prioritizes `std::env::current_exe()`
//! parent and ancestor directories **first**, before falling back to `CARGO_MANIFEST_DIR` or
//! `std::env::current_dir()`. This guarantees that launching `ToneForge_Studio.exe` standalone
//! from any working directory always activates Tier 1 CUDA and locates all `.onnx` models.

use ort::{
    environment::Environment,
    ep::{self, ArenaExtendStrategy, ExecutionProviderDispatch},
    session::{Session, builder::GraphOptimizationLevel},
};
use serde::{Deserialize, Serialize};
use std::ffi::CStr;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Active hardware execution tier for ONNX neural inference.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuExecutionTier {
    /// Tier 1: NVIDIA CUDA 12 + cuDNN 9 (`CUDAExecutionProvider`, Tensor Cores / TF32).
    CudaTensorCore,
    /// Tier 2: Microsoft DirectX 12 DirectML (`DmlExecutionProvider`).
    DirectMlDx12,
    /// Tier 3A: ONNX Runtime AVX2 Multi-Threaded CPU (`CPUExecutionProvider`).
    CpuOrt,
    /// Tier 3B: Pure-Rust deterministic DSP fallback (no ONNX Runtime DLLs present).
    PureRustFallback,
}

impl GpuExecutionTier {
    pub fn provider_name(self) -> &'static str {
        match self {
            Self::CudaTensorCore => "CUDAExecutionProvider",
            Self::DirectMlDx12 => "DmlExecutionProvider",
            Self::CpuOrt => "CPUExecutionProvider",
            Self::PureRustFallback => "PureRustSpectralFallback",
        }
    }

    pub fn tier_number(self) -> u8 {
        match self {
            Self::CudaTensorCore => 1,
            Self::DirectMlDx12 => 2,
            Self::CpuOrt | Self::PureRustFallback => 3,
        }
    }
}

/// Hardware telemetry & runtime status surfaced to the UI and verification tests.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuAcceleratorStatus {
    pub tier: GpuExecutionTier,
    pub tier_number: u8,
    pub provider_name: String,
    pub device_name: String,
    pub vram_mib: u64,
    pub cuda_driver_version: i32,
    pub runtime_dll_path: String,
    pub preloaded_dll_count: usize,
    pub badge_label: String,
    pub diagnostic_note: String,
}

/// Probed NVIDIA GPU metadata from `nvcuda.dll` CUDA Driver API.
#[derive(Clone, Debug)]
pub struct NvidiaCudaDeviceProbe {
    pub device_name: String,
    pub total_vram_bytes: usize,
    pub driver_version: i32,
    pub device_count: i32,
}

/// Ordered candidate root directories:
/// 1. `std::env::current_exe()` parent + ancestors (MANDATORY #1 priority for standalone `.exe`)
/// 2. `CARGO_MANIFEST_DIR` + ancestors (compile-time workspace fallback for dev/tests)
/// 3. `std::env::current_dir()` + ancestors (last-resort working directory fallback)
pub fn candidate_toneforge_roots() -> Vec<PathBuf> {
    let mut roots = Vec::with_capacity(16);
    let mut push_unique = |p: PathBuf| {
        if !roots.iter().any(|existing| existing == &p) {
            roots.push(p);
        }
    };

    // Priority 1: std::env::current_exe() parent directory and ancestors
    if let Ok(exe_path) = std::env::current_exe() {
        let mut cur = exe_path.parent();
        let mut depth = 0usize;
        while let Some(dir) = cur {
            push_unique(dir.to_path_buf());
            push_unique(dir.join("toneforge"));
            depth += 1;
            if depth >= 6 {
                break;
            }
            cur = dir.parent();
        }
    }

    // Priority 2: CARGO_MANIFEST_DIR and ancestors
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    {
        let mut cur = Some(manifest_dir.as_path());
        let mut depth = 0usize;
        while let Some(dir) = cur {
            push_unique(dir.to_path_buf());
            push_unique(dir.join("toneforge"));
            depth += 1;
            if depth >= 5 {
                break;
            }
            cur = dir.parent();
        }
    }

    // Priority 3: current_dir() and ancestors
    if let Ok(cwd) = std::env::current_dir() {
        let mut cur = Some(cwd.as_path());
        let mut depth = 0usize;
        while let Some(dir) = cur {
            push_unique(dir.to_path_buf());
            push_unique(dir.join("toneforge"));
            depth += 1;
            if depth >= 4 {
                break;
            }
            cur = dir.parent();
        }
    }

    roots
}

/// Resolves `<toneforge_root>/gpu_runtime` prioritizing `current_exe()` parent first.
pub fn resolve_gpu_runtime_dir() -> Option<PathBuf> {
    for root in candidate_toneforge_roots() {
        let cand = root.join("gpu_runtime");
        if cand.is_dir() {
            return Some(cand);
        }
    }
    None
}

/// Resolves `<toneforge_root>/assets` prioritizing `current_exe()` parent first.
pub fn resolve_assets_dir() -> Option<PathBuf> {
    for root in candidate_toneforge_roots() {
        let cand = root.join("assets");
        if cand.is_dir() {
            return Some(cand);
        }
    }
    None
}

/// Resolves an asset or model file by checking `current_exe()`-anchored `assets/` roots first,
/// then relative/explicit fallback paths.
pub fn resolve_asset_file(relative_under_toneforge: &[&str]) -> Option<PathBuf> {
    let roots = candidate_toneforge_roots();
    for root in &roots {
        for rel in relative_under_toneforge {
            let p = root.join(rel);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    for rel in relative_under_toneforge {
        let p = PathBuf::from(rel);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

/// Resolves `ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx` prioritizing `current_exe()` parent first.
pub fn resolve_ozone_bsrnn_onnx_path() -> Option<PathBuf> {
    resolve_asset_file(&[
        "assets/ozone12/models/ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx",
        "assets/models/bsrnn_realtime/ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx",
        "assets/models/ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx",
    ])
}

/// Resolves `delimit_exp009_ep080_fixed2.onnx` prioritizing `current_exe()` parent first.
pub fn resolve_ozone_delimit_onnx_path() -> Option<PathBuf> {
    resolve_asset_file(&[
        "assets/ozone12/models/delimit_exp009_ep080_fixed2.onnx",
        "assets/models/delimit_exp009_ep080_fixed2.onnx",
    ])
}

/// Resolves `Kim_Vocal_2.onnx` prioritizing `current_exe()` parent first.
pub fn resolve_uvr5_kim_vocal2_onnx_path() -> Option<PathBuf> {
    resolve_asset_file(&[
        "assets/uvr5/models/MDX_Net_Models/Kim_Vocal_2.onnx",
        "assets/models/Kim_Vocal_2.onnx",
    ])
}

/// Resolves `UVR-MDX-NET-Inst_HQ_3.onnx` prioritizing `current_exe()` parent first.
pub fn resolve_uvr5_inst_hq3_onnx_path() -> Option<PathBuf> {
    resolve_asset_file(&[
        "assets/uvr5/models/MDX_Net_Models/UVR-MDX-NET-Inst_HQ_3.onnx",
        "assets/models/UVR-MDX-NET-Inst_HQ_3.onnx",
    ])
}

#[cfg(windows)]
fn wide_null(s: &std::ffi::OsStr) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    s.encode_wide().chain(std::iter::once(0)).collect()
}

/// Probes NVIDIA Driver (`nvcuda.dll`) without requiring CUDA Toolkit installation on `C:\`.
#[cfg(windows)]
pub fn probe_nvidia_cuda_driver() -> Option<NvidiaCudaDeviceProbe> {
    use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};

    unsafe {
        let dll_name = wide_null(std::ffi::OsStr::new("nvcuda.dll"));
        let h_mod = LoadLibraryW(dll_name.as_ptr());
        if h_mod.is_null() {
            return None;
        }

        type CuInitFn = unsafe extern "system" fn(u32) -> i32;
        type CuDriverGetVersionFn = unsafe extern "system" fn(*mut i32) -> i32;
        type CuDeviceGetCountFn = unsafe extern "system" fn(*mut i32) -> i32;
        type CuDeviceGetFn = unsafe extern "system" fn(*mut i32, i32) -> i32;
        type CuDeviceGetNameFn = unsafe extern "system" fn(*mut u8, i32, i32) -> i32;
        type CuDeviceTotalMemV2Fn = unsafe extern "system" fn(*mut usize, i32) -> i32;

        let p_init = GetProcAddress(h_mod, b"cuInit\0".as_ptr())?;
        let p_cnt = GetProcAddress(h_mod, b"cuDeviceGetCount\0".as_ptr())?;
        let p_get = GetProcAddress(h_mod, b"cuDeviceGet\0".as_ptr())?;
        let p_name = GetProcAddress(h_mod, b"cuDeviceGetName\0".as_ptr())?;

        let cu_init: CuInitFn = std::mem::transmute(p_init);
        let cu_cnt: CuDeviceGetCountFn = std::mem::transmute(p_cnt);
        let cu_get: CuDeviceGetFn = std::mem::transmute(p_get);
        let cu_name: CuDeviceGetNameFn = std::mem::transmute(p_name);

        if cu_init(0) != 0 {
            return None;
        }

        let mut count = 0i32;
        if cu_cnt(&mut count) != 0 || count <= 0 {
            return None;
        }

        let mut dev = 0i32;
        if cu_get(&mut dev, 0) != 0 {
            return None;
        }

        let mut name_buf = [0u8; 256];
        let device_name = if cu_name(name_buf.as_mut_ptr(), 255, dev) == 0 {
            CStr::from_bytes_until_nul(&name_buf)
                .map(|c| c.to_string_lossy().into_owned())
                .unwrap_or_else(|_| "NVIDIA CUDA GPU".to_string())
        } else {
            "NVIDIA CUDA GPU".to_string()
        };

        let mut driver_version = 0i32;
        if let Some(p_ver) = GetProcAddress(h_mod, b"cuDriverGetVersion\0".as_ptr()) {
            let cu_ver: CuDriverGetVersionFn = std::mem::transmute(p_ver);
            let _ = cu_ver(&mut driver_version);
        }

        let mut total_vram_bytes = 0usize;
        if let Some(p_mem) = GetProcAddress(h_mod, b"cuDeviceTotalMem_v2\0".as_ptr())
            .or_else(|| GetProcAddress(h_mod, b"cuDeviceTotalMem\0".as_ptr()))
        {
            let cu_mem: CuDeviceTotalMemV2Fn = std::mem::transmute(p_mem);
            let _ = cu_mem(&mut total_vram_bytes, dev);
        }

        Some(NvidiaCudaDeviceProbe {
            device_name,
            total_vram_bytes,
            driver_version,
            device_count: count,
        })
    }
}

#[cfg(not(windows))]
pub fn probe_nvidia_cuda_driver() -> Option<NvidiaCudaDeviceProbe> {
    None
}

/// Preloads all CUDA 12 + NVRTC + cuBLAS + cuFFT + cuRAND + cuDNN 9 + ORT shared bridge DLLs
/// from `cuda_dir` in dependency order using Win32 `SetDllDirectoryW` + `AddDllDirectory` + `LoadLibraryExW`.
///
/// Note: `nvblas64_12.dll` (CPU BLAS interceptor) and `onnxruntime_providers_cuda.dll` (which requires
/// `onnxruntime.dll` to initialize `Provider_SetHost()` first) are intentionally excluded from direct
/// `DllMain` preloading; `onnxruntime.dll` loads `onnxruntime_providers_cuda.dll` automatically when
/// `SessionOptionsAppendExecutionProvider_CUDA_V2` runs.
#[cfg(windows)]
fn preload_cuda12_runtime_dlls(cuda_dir: &Path) -> Result<usize, String> {
    use windows_sys::Win32::System::LibraryLoader::{
        AddDllDirectory, LOAD_WITH_ALTERED_SEARCH_PATH, LoadLibraryExW, SetDllDirectoryW,
    };

    const REQUIRED_FILES: &[&str] = &[
        "cudart64_12.dll",
        "cublasLt64_12.dll",
        "cublas64_12.dll",
        "cufft64_11.dll",
        "curand64_10.dll",
        "nvrtc64_120_0.dll",
        "cudnn64_9.dll",
        "onnxruntime_providers_shared.dll",
        "onnxruntime_providers_cuda.dll",
        "onnxruntime.dll",
    ];
    for req in REQUIRED_FILES {
        let p = cuda_dir.join(req);
        if !p.is_file() {
            return Err(format!("Missing required CUDA 12 runtime DLL: {}", p.display()));
        }
    }

    let entries: Vec<PathBuf> = std::fs::read_dir(cuda_dir)
        .map_err(|e| format!("Failed to read {:?}: {}", cuda_dir, e))?
        .filter_map(|e| e.ok().map(|x| x.path()))
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("dll"))
                .unwrap_or(false)
        })
        .collect();

    // Ensure `cuda_dir` is registered with the Win32 loader for transitive delay-loads (`onnxruntime_providers_cuda.dll`)
    let wide_dir = wide_null(cuda_dir.as_os_str());
    unsafe {
        let _ = SetDllDirectoryW(wide_dir.as_ptr());
        let _ = AddDllDirectory(wide_dir.as_ptr());
    }

    const ORDERED_PREFIXES: &[&str] = &[
        "cudart64_",
        "nvrtc-builtins64_",
        "nvrtc64_",
        "cublaslt64_",
        "cublas64_",
        "cufft64_",
        "cufftw64_",
        "curand64_",
        "cudnn64_",
        "cudnn_ops",
        "cudnn_cnn",
        "cudnn_adv",
        "cudnn_graph",
        "cudnn_heuristic",
        "cudnn_engines_",
        "cudnn_ext",
        "onnxruntime_providers_shared",
    ];

    let skip_direct = ["nvblas64_12.dll", "onnxruntime_providers_cuda.dll", "onnxruntime.dll"];
    let mut loaded_names: Vec<String> = Vec::with_capacity(entries.len());

    let try_load_one = |dll_path: &Path| -> Result<(), String> {
        let w = wide_null(dll_path.as_os_str());
        let handle = unsafe { LoadLibraryExW(w.as_ptr(), std::ptr::null_mut(), LOAD_WITH_ALTERED_SEARCH_PATH) };
        if handle.is_null() {
            return Err(format!("Win32 LoadLibraryExW failed for {}", dll_path.display()));
        }
        Ok(())
    };

    for prefix in ORDERED_PREFIXES {
        for path in &entries {
            let Some(fname) = path.file_name().and_then(|s| s.to_str()) else {
                continue;
            };
            let low = fname.to_ascii_lowercase();
            if skip_direct.iter().any(|&s| s == low) || loaded_names.contains(&low) {
                continue;
            }
            if low.starts_with(prefix) {
                try_load_one(path)?;
                loaded_names.push(low);
            }
        }
    }

    for path in &entries {
        let Some(fname) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        let low = fname.to_ascii_lowercase();
        if skip_direct.iter().any(|&s| s == low) || loaded_names.contains(&low) {
            continue;
        }
        try_load_one(path)?;
        loaded_names.push(low);
    }

    Ok(loaded_names.len())
}

#[cfg(not(windows))]
fn preload_cuda12_runtime_dlls(_cuda_dir: &Path) -> Result<usize, String> {
    Err("Windows CUDA DLL preloader only supported on Windows".to_string())
}

#[cfg(windows)]
fn preload_directml_dlls(dml_dir: &Path) -> Result<usize, String> {
    use windows_sys::Win32::System::LibraryLoader::{
        AddDllDirectory, LOAD_WITH_ALTERED_SEARCH_PATH, LoadLibraryExW, SetDllDirectoryW,
    };

    let dml_dll = dml_dir.join("DirectML.dll");
    let ort_dll = dml_dir.join("onnxruntime.dll");
    if !dml_dll.is_file() || !ort_dll.is_file() {
        return Err(format!(
            "Missing DirectML.dll or onnxruntime.dll in {}",
            dml_dir.display()
        ));
    }

    let wide_dir = wide_null(dml_dir.as_os_str());
    unsafe {
        let _ = SetDllDirectoryW(wide_dir.as_ptr());
        let _ = AddDllDirectory(wide_dir.as_ptr());
        let w_dml = wide_null(dml_dll.as_os_str());
        let h = LoadLibraryExW(w_dml.as_ptr(), std::ptr::null_mut(), LOAD_WITH_ALTERED_SEARCH_PATH);
        if h.is_null() {
            return Err(format!("Failed to load {}", dml_dll.display()));
        }
    }
    Ok(1)
}

#[cfg(not(windows))]
fn preload_directml_dlls(_dml_dir: &Path) -> Result<usize, String> {
    Err("DirectML is only available on Windows".to_string())
}

#[cfg(windows)]
fn preflight_probe_cuda_ort_dll(ort_dll: &Path) -> Result<(), String> {
    use windows_sys::Win32::Foundation::FreeLibrary;
    use windows_sys::Win32::System::LibraryLoader::{
        GetProcAddress, LOAD_WITH_ALTERED_SEARCH_PATH, LoadLibraryExW,
    };

    unsafe {
        let w_ort = wide_null(ort_dll.as_os_str());
        let h_ort = LoadLibraryExW(
            w_ort.as_ptr(),
            std::ptr::null_mut(),
            LOAD_WITH_ALTERED_SEARCH_PATH,
        );
        if h_ort.is_null() {
            return Err(format!(
                "Win32 LoadLibraryExW failed for {}",
                ort_dll.display()
            ));
        }

        type OrtGetApiBaseFn = unsafe extern "system" fn() -> *const ort::sys::OrtApiBase;
        let Some(p_base) = GetProcAddress(h_ort, b"OrtGetApiBase\0".as_ptr()) else {
            let _ = FreeLibrary(h_ort);
            return Err("Missing OrtGetApiBase export in cuda/onnxruntime.dll".to_string());
        };
        let get_api_base: OrtGetApiBaseFn = std::mem::transmute(p_base);
        let api_base = get_api_base();
        if api_base.is_null() {
            let _ = FreeLibrary(h_ort);
            return Err("OrtGetApiBase returned NULL".to_string());
        }
        let get_api = (*api_base).GetApi;
        let api = get_api(ort::sys::ORT_API_VERSION);
        if api.is_null() {
            let _ = FreeLibrary(h_ort);
            return Err(format!(
                "OrtApiBase::GetApi({}) returned NULL",
                ort::sys::ORT_API_VERSION
            ));
        }

        let extract_err = |st: ort::sys::OrtStatusPtr| -> String {
            let msg_ptr = ((*api).GetErrorMessage)(st.0);
            let msg = if msg_ptr.is_null() {
                "Unknown ORT C-API status error".to_string()
            } else {
                CStr::from_ptr(msg_ptr).to_string_lossy().into_owned()
            };
            ((*api).ReleaseStatus)(st.0);
            msg
        };

        let mut sess_opts: *mut ort::sys::OrtSessionOptions = std::ptr::null_mut();
        let st_so = ((*api).CreateSessionOptions)(&mut sess_opts);
        if !st_so.0.is_null() {
            let err = extract_err(st_so);
            let _ = FreeLibrary(h_ort);
            return Err(format!("CreateSessionOptions failed: {err}"));
        }

        let mut cuda_opts: *mut ort::sys::OrtCUDAProviderOptionsV2 = std::ptr::null_mut();
        let st_co = ((*api).CreateCUDAProviderOptions)(&mut cuda_opts);
        if !st_co.0.is_null() {
            ((*api).ReleaseSessionOptions)(sess_opts);
            let err = extract_err(st_co);
            let _ = FreeLibrary(h_ort);
            return Err(format!("CreateCUDAProviderOptions failed: {err}"));
        }

        let st_ep = ((*api).SessionOptionsAppendExecutionProvider_CUDA_V2)(sess_opts, cuda_opts);
        ((*api).ReleaseCUDAProviderOptions)(cuda_opts);
        ((*api).ReleaseSessionOptions)(sess_opts);

        if !st_ep.0.is_null() {
            let err = extract_err(st_ep);
            let _ = FreeLibrary(h_ort);
            return Err(format!(
                "SessionOptionsAppendExecutionProvider_CUDA_V2 failed: {err}"
            ));
        }

        Ok(())
    }
}

#[cfg(not(windows))]
fn preflight_probe_cuda_ort_dll(_ort_dll: &Path) -> Result<(), String> {
    Err("CUDA DLL preflight probe is only implemented on Windows".to_string())
}

/// Process-global ONNX Runtime GPU Accelerator Manager.
///
/// Must NEVER be initialized on the Real-Time audio callback thread; always initialize via
/// `GpuRuntimeManager::global()` during startup, `new_with_onnx_gpu()`, or offline separation commands.
pub struct GpuRuntimeManager {
    environment: Option<Environment>,
    status: GpuAcceleratorStatus,
}

static GLOBAL_GPU_RUNTIME: OnceLock<GpuRuntimeManager> = OnceLock::new();

impl GpuRuntimeManager {
    /// Returns (or initializes once outside RT thread) the singleton `GpuRuntimeManager`.
    pub fn global() -> &'static Self {
        GLOBAL_GPU_RUNTIME.get_or_init(Self::initialize_three_tier)
    }

    pub fn status(&self) -> &GpuAcceleratorStatus {
        &self.status
    }

    pub fn tier(&self) -> GpuExecutionTier {
        self.status.tier
    }

    pub fn environment(&self) -> Option<&Environment> {
        self.environment.as_ref()
    }

    /// Builds the optimal CUDA Execution Provider dispatch for NVIDIA Ada Lovelace (`RTX 4080`):
    /// - `TF32 = true` (4th-gen Tensor Core acceleration for FP32 MatMul/Conv/LSTM)
    /// - `ConvAlgorithmSearch::Heuristic` (instant startup, no 10-second exhaustive stall)
    /// - `ArenaExtendStrategy::NextPowerOfTwo`
    pub fn cuda_ep_dispatch(error_on_failure: bool) -> ExecutionProviderDispatch {
        let dispatch = ep::CUDA::default()
            .with_device_id(0)
            .with_tf32(true)
            .with_conv_algorithm_search(ep::cuda::ConvAlgorithmSearch::Heuristic)
            // Nới vùng nhớ vừa đúng yêu cầu: NextPowerOfTwo nhân đôi mỗi lần thiếu, một lần tách 30 s
            // đã để lại 11,1 GB VRAM (đo 2026-11-14).
            .with_arena_extend_strategy(ArenaExtendStrategy::SameAsRequested)
            .build();
        if error_on_failure {
            dispatch.error_on_failure()
        } else {
            dispatch
        }
    }

    /// Builds the DirectML Execution Provider dispatch (`HighPerformance` GPU filter).
    pub fn directml_ep_dispatch(error_on_failure: bool) -> ExecutionProviderDispatch {
        let dispatch = ep::DirectML::default()
            .with_performance_preference(ep::directml::PerformancePreference::HighPerformance)
            .with_device_filter(ep::directml::DeviceFilter::Gpu)
            .build();
        if error_on_failure {
            dispatch.error_on_failure()
        } else {
            dispatch
        }
    }

    fn initialize_three_tier() -> Self {
        let forced_tier = std::env::var("TONEFORGE_FORCE_GPU_TIER")
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .unwrap_or_default();
        let skip_tier1_cuda = matches!(
            forced_tier.as_str(),
            "directml" | "tier2" | "cpu_ort" | "tier3a" | "pure_rust" | "tier3b"
        );
        let skip_tier2_dml_ep = matches!(
            forced_tier.as_str(),
            "cpu_ort" | "tier3a" | "pure_rust" | "tier3b"
        );
        let skip_all_ort_dlls = matches!(forced_tier.as_str(), "pure_rust" | "tier3b");

        let gpu_rt_root = if skip_all_ort_dlls {
            None
        } else {
            resolve_gpu_runtime_dir()
        };
        let nv_probe = probe_nvidia_cuda_driver();
        let mut tier1_fallback_note: Option<String> = None;

        // =====================================================================
        // TIER 1: NVIDIA CUDA 12 + cuDNN 9 (`gpu_runtime/cuda/onnxruntime.dll`)
        // =====================================================================
        if !skip_tier1_cuda {
            if let (Some(rt_root), Some(probe)) = (&gpu_rt_root, &nv_probe) {
                let cuda_dir = rt_root.join("cuda");
                let ort_dll = cuda_dir.join("onnxruntime.dll");
                if ort_dll.is_file() {
                    match preload_cuda12_runtime_dlls(&cuda_dir)
                        .and_then(|cnt| preflight_probe_cuda_ort_dll(&ort_dll).map(|()| cnt))
                    {
                        Ok(preloaded_cnt) => {
                            if let Ok(builder) = ort::init_from(&ort_dll) {
                                if let Ok(env) = builder
                                    .with_name("ToneForgeStudioCuda")
                                    .with_telemetry(false)
                                    .build()
                                {
                                    // Confirm CUDAExecutionProvider registration on GPU 0
                                    let cuda_ep_ok = Session::builder(&env).and_then(|mut sb| {
                                        Self::cuda_ep_dispatch(true).register(&mut sb)
                                    });

                                    let vram_mib = (probe.total_vram_bytes as u64) / (1024 * 1024);
                                    let vram_gb = ((vram_mib as f64) / 1024.0).round() as u64;
                                    let short_gpu = probe
                                        .device_name
                                        .trim_start_matches("NVIDIA ")
                                        .trim_start_matches("GeForce ")
                                        .to_string();

                                    match cuda_ep_ok {
                                        Ok(()) => {
                                            return Self {
                                                environment: Some(env),
                                                status: GpuAcceleratorStatus {
                                                    tier: GpuExecutionTier::CudaTensorCore,
                                                    tier_number: 1,
                                                    provider_name: "CUDAExecutionProvider"
                                                        .to_string(),
                                                    device_name: probe.device_name.clone(),
                                                    vram_mib,
                                                    cuda_driver_version: probe.driver_version,
                                                    runtime_dll_path: ort_dll.display().to_string(),
                                                    preloaded_dll_count: preloaded_cnt + 2,
                                                    badge_label: format!(
                                                        "CUDA 12 • {} {}GB",
                                                        short_gpu, vram_gb
                                                    ),
                                                    diagnostic_note: format!(
                                                        "Tier 1 Active: CUDA 12 + cuDNN 9 Tensor Cores (TF32=ON) on {} ({} MiB VRAM, Driver CUDA {})",
                                                        probe.device_name,
                                                        vram_mib,
                                                        probe.driver_version
                                                    ),
                                                },
                                            };
                                        }
                                        Err(cuda_err) => {
                                            return Self {
                                                environment: Some(env),
                                                status: GpuAcceleratorStatus {
                                                    tier: GpuExecutionTier::CpuOrt,
                                                    tier_number: 3,
                                                    provider_name: "CPUExecutionProvider"
                                                        .to_string(),
                                                    device_name: "AVX2 Multi-Core CPU".to_string(),
                                                    vram_mib,
                                                    cuda_driver_version: probe.driver_version,
                                                    runtime_dll_path: ort_dll.display().to_string(),
                                                    preloaded_dll_count: preloaded_cnt + 1,
                                                    badge_label: "ORT CPU (CUDA Fallback)"
                                                        .to_string(),
                                                    diagnostic_note: format!(
                                                        "CUDA EP registration failed ({}); using CPUExecutionProvider",
                                                        cuda_err
                                                    ),
                                                },
                                            };
                                        }
                                    }
                                }
                            }
                        }
                        Err(preflight_err) => {
                            // `ort::init_from` has NOT been called yet — fall through cleanly to Tier 2 DirectML!
                            tier1_fallback_note = Some(preflight_err);
                        }
                    }
                }
            }
        } else if !forced_tier.is_empty() {
            tier1_fallback_note = Some(format!("TONEFORGE_FORCE_GPU_TIER={forced_tier}"));
        }

        // =====================================================================
        // TIER 2: Universal Windows DirectX 12 (`gpu_runtime/directml/onnxruntime.dll`)
        // =====================================================================
        if let Some(rt_root) = &gpu_rt_root {
            let dml_dir = rt_root.join("directml");
            let ort_dll = dml_dir.join("onnxruntime.dll");
            if ort_dll.is_file() {
                let preloaded_cnt = preload_directml_dlls(&dml_dir).unwrap_or(0);
                if let Ok(builder) = ort::init_from(&ort_dll) {
                    if let Ok(env) = builder
                        .with_name("ToneForgeStudioDml")
                        .with_telemetry(false)
                        .build()
                    {
                        let dml_ok = if skip_tier2_dml_ep {
                            Err("Skipped DirectML via TONEFORGE_FORCE_GPU_TIER".to_string())
                        } else {
                            Session::builder(&env)
                                .and_then(|mut sb| {
                                    Self::directml_ep_dispatch(true).register(&mut sb)
                                })
                                .map_err(|e| e.to_string())
                        };
                        let dev_name = nv_probe
                            .as_ref()
                            .map(|p| p.device_name.clone())
                            .unwrap_or_else(|| "Windows DirectX 12 GPU".to_string());
                        if dml_ok.is_ok() {
                            let suffix = tier1_fallback_note
                                .as_deref()
                                .map(|r| format!(" (Tier 1 skipped/fallback: {r})"))
                                .unwrap_or_default();
                            return Self {
                                environment: Some(env),
                                status: GpuAcceleratorStatus {
                                    tier: GpuExecutionTier::DirectMlDx12,
                                    tier_number: 2,
                                    provider_name: "DmlExecutionProvider".to_string(),
                                    device_name: dev_name.clone(),
                                    vram_mib: nv_probe
                                        .as_ref()
                                        .map(|p| (p.total_vram_bytes as u64) / (1024 * 1024))
                                        .unwrap_or(0),
                                    cuda_driver_version: nv_probe
                                        .as_ref()
                                        .map(|p| p.driver_version)
                                        .unwrap_or(0),
                                    runtime_dll_path: ort_dll.display().to_string(),
                                    preloaded_dll_count: preloaded_cnt + 1,
                                    badge_label: format!("DirectML DX12 • {}", dev_name),
                                    diagnostic_note: format!(
                                        "Tier 2 Active: DirectMLExecutionProvider on {}{}",
                                        dev_name, suffix
                                    ),
                                },
                            };
                        } else {
                            return Self {
                                environment: Some(env),
                                status: GpuAcceleratorStatus {
                                    tier: GpuExecutionTier::CpuOrt,
                                    tier_number: 3,
                                    provider_name: "CPUExecutionProvider".to_string(),
                                    device_name: "AVX2 Multi-Core CPU".to_string(),
                                    vram_mib: 0,
                                    cuda_driver_version: 0,
                                    runtime_dll_path: ort_dll.display().to_string(),
                                    preloaded_dll_count: preloaded_cnt + 1,
                                    badge_label: "ORT CPU (AVX2)".to_string(),
                                    diagnostic_note:
                                        "Tier 3A Active: ONNX Runtime CPUExecutionProvider"
                                            .to_string(),
                                },
                            };
                        }
                    }
                }
            }
        }

        // =====================================================================
        // TIER 3B: Pure-Rust Deterministic Spectral Fallback
        // =====================================================================
        Self {
            environment: None,
            status: GpuAcceleratorStatus {
                tier: GpuExecutionTier::PureRustFallback,
                tier_number: 3,
                provider_name: "PureRustSpectralFallback".to_string(),
                device_name: "Pure-Rust SIMD CPU".to_string(),
                vram_mib: 0,
                cuda_driver_version: 0,
                runtime_dll_path: String::new(),
                preloaded_dll_count: 0,
                badge_label: "CPU • Pure-Rust DSP".to_string(),
                diagnostic_note:
                    "Tier 3B Active: gpu_runtime/ not found; running deterministic pure-Rust DSP"
                        .to_string(),
            },
        }
    }

    /// Creates an ONNX `Session` from `model_path` bound to the highest available GPU Execution Provider,
    /// automatically falling back to `CPUExecutionProvider` if a model-specific graph issue occurs.
    pub fn create_session(&self, model_path: &Path) -> Result<(Session, GpuExecutionTier), String> {
        let env = self.environment.as_ref().ok_or_else(|| {
            "ONNX Runtime DLLs are not loaded (run scripts/setup_gpu_and_models.ps1)".to_string()
        })?;

        if !model_path.is_file() {
            return Err(format!("ONNX model file does not exist: {}", model_path.display()));
        }

        match self.status.tier {
            GpuExecutionTier::CudaTensorCore => {
                // Attempt strict CUDAExecutionProvider first
                let cuda_attempt = Session::builder(env)
                    .map_err(|e| e.to_string())
                    .and_then(|sb| {
                        sb.with_optimization_level(GraphOptimizationLevel::All)
                            .map_err(|e| e.to_string())
                    })
                    .and_then(|sb| {
                        sb.with_execution_providers([Self::cuda_ep_dispatch(true)])
                            .map_err(|e| e.to_string())
                    })
                    .and_then(|mut sb| sb.commit_from_file(model_path).map_err(|e| e.to_string()));

                if let Ok(session) = cuda_attempt {
                    return Ok((session, GpuExecutionTier::CudaTensorCore));
                }
            }
            GpuExecutionTier::DirectMlDx12 => {
                // DirectML requires sequential execution (`with_parallel_execution(false)`) and
                // disabled memory pattern (`with_memory_pattern(false)`).
                let dml_attempt = Session::builder(env)
                    .map_err(|e| e.to_string())
                    .and_then(|sb| sb.with_memory_pattern(false).map_err(|e| e.to_string()))
                    .and_then(|sb| sb.with_parallel_execution(false).map_err(|e| e.to_string()))
                    .and_then(|sb| {
                        sb.with_optimization_level(GraphOptimizationLevel::All)
                            .map_err(|e| e.to_string())
                    })
                    .and_then(|sb| {
                        sb.with_execution_providers([Self::directml_ep_dispatch(true)])
                            .map_err(|e| e.to_string())
                    })
                    .and_then(|mut sb| sb.commit_from_file(model_path).map_err(|e| e.to_string()));

                if let Ok(session) = dml_attempt {
                    return Ok((session, GpuExecutionTier::DirectMlDx12));
                }
            }
            _ => {}
        }

        // Fallback to multi-threaded CPUExecutionProvider
        let mut cpu_sb = Session::builder(env)
            .map_err(|e| format!("Failed to create ORT SessionBuilder: {}", e))?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(|e| format!("Failed to set ORT optimization level: {}", e))?;
        let session = cpu_sb
            .commit_from_file(model_path)
            .map_err(|e| format!("Failed to commit ONNX session {:?}: {}", model_path, e))?;
        Ok((session, GpuExecutionTier::CpuOrt))
    }
}
