//! Optional Downloadable Add-on Packs Manager (`addon_packs.rs`).
//!
//! Keeps the base ToneForge Studio installer lightweight (~12 MB) while letting users
//! inspect, download, verify (SHA-256 / byte-length), hot-activate, and uninstall optional
//! GPU runtimes and studio neural models directly inside the About & Updates panel:
//! - `directml_runtime`: Windows DirectX 12 DirectML + CPU AVX2 ONNX Runtime (for standard PCs without NVIDIA).
//! - `cuda12_runtime`: NVIDIA CUDA 12 + cuDNN 9 Tensor Cores ONNX Runtime (for GTX / RTX GPUs).
//! - `uvr5_models`: Studio UVR5 MDX-Net weights (`Kim_Vocal_2.onnx` + `UVR-MDX-NET-Inst_HQ_3.onnx`).
//! - `ozone12_ai_models`: Ozone 12 AI Master Assistant CNN + 4-Stem BSRNN + DeLimit weights.
//!
//! All upstream URLs remain strictly internal to compiled Rust code and are stripped from any error
//! strings so no Git/repository/CDN details are exposed in the UI.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use ai_uvr5_stem_engine::{
    GpuExecutionTier, GpuRuntimeManager, has_complete_cuda12_runtime_dlls,
    has_complete_directml_runtime_dlls, notify_addon_packs_changed, probe_nvidia_cuda_driver,
    resolve_asset_file, resolve_cuda_runtime_dir, resolve_directml_runtime_dir,
    resolve_ozone_bsrnn_onnx_path, resolve_ozone_delimit_onnx_path, resolve_uvr5_inst_hq3_onnx_path,
    resolve_uvr5_kim_vocal2_onnx_path, toneforge_user_addons_dir,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Emitter};

pub const PACK_DIRECTML: &str = "directml_runtime";
pub const PACK_CUDA12: &str = "cuda12_runtime";
pub const PACK_UVR5_MODELS: &str = "uvr5_models";
pub const PACK_OZONE12_AI: &str = "ozone12_ai_models";

const UVR5_KIM_VOCAL2_URL: &str =
    "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Kim_Vocal_2.onnx";
const UVR5_KIM_VOCAL2_SIZE: u64 = 66_759_214;
const UVR5_KIM_VOCAL2_SHA256: &str =
    "ce74ef3b6a6024ce44211a07be9cf8bc6d87728cc852a68ab34eb8e58cde9c8b";

const UVR5_INST_HQ3_URL: &str =
    "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_3.onnx";
const UVR5_INST_HQ3_SIZE: u64 = 66_759_214;
const UVR5_INST_HQ3_SHA256: &str =
    "317554b07fe1ea5279a77f2b1520a41ea4b93432560c4ffd08792c30fddf9adc";

const OZONE12_AI_PACK_URL: &str =
    "https://github.com/Selorson/ToneForge-Release/releases/latest/download/ToneForge_Ozone12_AI_Pack.zip";

const CUDA12_WHEEL_SPECS: &[(&str, Option<&str>)] = &[
    ("onnxruntime-gpu", Some("1.25.1")),
    ("nvidia-cuda-runtime-cu12", None),
    ("nvidia-cublas-cu12", None),
    ("nvidia-cufft-cu12", None),
    ("nvidia-curand-cu12", None),
    ("nvidia-cuda-nvrtc-cu12", None),
    ("nvidia-cudnn-cu12", None),
];

static DOWNLOAD_ACTIVE: AtomicBool = AtomicBool::new(false);
static CANCEL_REQUESTED: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AddonPackInfoDto {
    pub id: String,
    pub title: String,
    pub subtitle: String,
    /// `"standard_pc"` | `"nvidia_gpu"` | `"all_pcs"`
    pub target_audience: String,
    pub audience_badge: String,
    pub recommended: bool,
    pub estimated_size_label: String,
    pub installed: bool,
    /// `"not_installed"` | `"addon_pack"` | `"bundled_local"`
    pub installed_source: String,
    pub installed_bytes: u64,
    pub can_uninstall: bool,
    pub status_detail: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AddonSystemOverviewDto {
    pub addons_root_path: String,
    pub has_nvidia_gpu: bool,
    pub nvidia_device_name: Option<String>,
    pub nvidia_vram_mib: u64,
    pub active_tier: GpuExecutionTier,
    pub active_badge: String,
    pub active_provider: String,
    pub restart_recommended: bool,
    pub restart_reason: Option<String>,
    pub download_in_progress: bool,
    pub packs: Vec<AddonPackInfoDto>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AddonDownloadProgressDto {
    pub pack_id: String,
    /// `"starting"` | `"downloading"` | `"extracting"` | `"verifying"` | `"completed"` | `"cancelled"` | `"error"`
    pub phase: String,
    pub step_current: usize,
    pub step_total: usize,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub percent: u8,
    pub message: String,
}

#[derive(Deserialize)]
struct PypiReleaseResponse {
    urls: Vec<PypiWheelEntry>,
}

#[derive(Deserialize)]
struct PypiWheelEntry {
    filename: String,
    packagetype: String,
    url: String,
    size: u64,
    digests: PypiDigests,
}

#[derive(Deserialize)]
struct PypiDigests {
    sha256: String,
}

/// Sweeps any deferred `.trash_*` or `.staging_*` directories inside `%LOCALAPPDATA%\ToneForgeStudio\addons`
/// at startup before any ONNX Runtime DLLs are mapped into the process.
pub fn sweep_pending_addon_trash() {
    let Some(addons_root) = toneforge_user_addons_dir() else {
        return;
    };
    let Ok(entries) = fs::read_dir(&addons_root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
            if name.starts_with(".trash_") || name.starts_with(".staging_") {
                let _ = if path.is_dir() {
                    fs::remove_dir_all(&path)
                } else {
                    fs::remove_file(&path)
                };
            }
        }
    }
}

/// Strips any raw URLs or repository/host identifiers from error messages before surfacing to UI.
pub fn sanitize_error_message(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for token in raw.split_whitespace() {
        let low = token.to_ascii_lowercase();
        if low.starts_with("http://")
            || low.starts_with("https://")
            || low.contains("github")
            || low.contains("selorson")
            || low.contains("trvlvr")
            || low.contains("pypi.org")
            || low.contains("pythonhosted")
        {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str("[máy-chủ-tải-gói]");
        } else {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(token);
        }
    }
    out
}

fn dir_size_bytes(dir: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(dir) else {
        return 0;
    };
    let mut sum = 0u64;
    for entry in entries.flatten() {
        let p = entry.path();
        if let Ok(meta) = entry.metadata() {
            if meta.is_file() {
                sum = sum.saturating_add(meta.len());
            } else if meta.is_dir() {
                sum = sum.saturating_add(dir_size_bytes(&p));
            }
        }
    }
    sum
}

fn is_under_addons_root(path: &Path, addons_root: &Path) -> bool {
    let norm_p = path.to_string_lossy().replace('/', "\\").to_ascii_lowercase();
    let norm_r = addons_root
        .to_string_lossy()
        .replace('/', "\\")
        .trim_end_matches('\\')
        .to_ascii_lowercase();
    norm_p == norm_r || norm_p.starts_with(&format!("{norm_r}\\"))
}

fn addons_root_or_err() -> Result<PathBuf, String> {
    let dir = toneforge_user_addons_dir()
        .ok_or_else(|| "Không xác định được thư mục dữ liệu người dùng (LOCALAPPDATA).".to_string())?;
    fs::create_dir_all(&dir)
        .map_err(|e| format!("Không tạo được thư mục gói mở rộng: {}", sanitize_error_message(&e.to_string())))?;
    Ok(dir)
}

pub fn inspect_addon_system() -> AddonSystemOverviewDto {
    let addons_root = toneforge_user_addons_dir().unwrap_or_else(|| PathBuf::from("addons"));
    let nv_probe = probe_nvidia_cuda_driver();
    let has_nvidia_gpu = nv_probe.is_some();
    let nvidia_device_name = nv_probe.as_ref().map(|p| p.device_name.clone());
    let nvidia_vram_mib = nv_probe
        .as_ref()
        .map(|p| (p.total_vram_bytes as u64) / (1024 * 1024))
        .unwrap_or(0);

    let rt = GpuRuntimeManager::global();
    let rt_status = rt.status();

    // 1. DirectML Pack inspection
    let dml_dir = resolve_directml_runtime_dir();
    let dml_in_addons = addons_root.join("gpu_runtime").join("directml");
    let dml_installed = dml_dir.is_some();
    let dml_bytes = dml_dir.as_deref().map(dir_size_bytes).unwrap_or(0);
    let dml_can_uninstall = has_complete_directml_runtime_dlls(&dml_in_addons)
        || dml_dir
            .as_deref()
            .map(|d| is_under_addons_root(d, &addons_root))
            .unwrap_or(false);
    let dml_source = if !dml_installed {
        "not_installed"
    } else if dml_can_uninstall {
        "addon_pack"
    } else {
        "bundled_local"
    };

    let pack_dml = AddonPackInfoDto {
        id: PACK_DIRECTML.to_string(),
        title: "Gói 1 · Nhân AI Máy thường (DirectML DX12 & CPU AVX2)".to_string(),
        subtitle: "Chạy mô hình AI bằng card onboard Intel (UHD / Iris Xe), AMD Radeon hoặc CPU đa nhân trên Windows 10/11. Giữ app cực nhẹ mà không cần card NVIDIA.".to_string(),
        target_audience: "standard_pc".to_string(),
        audience_badge: if has_nvidia_gpu {
            "Dành cho Máy thường / Dự phòng DX12".to_string()
        } else {
            "Khuyên dùng cho máy của bạn".to_string()
        },
        recommended: !has_nvidia_gpu,
        estimated_size_label: "~17 MB tải về (~39 MB trên đĩa)".to_string(),
        installed: dml_installed,
        installed_source: dml_source.to_string(),
        installed_bytes: dml_bytes,
        can_uninstall: dml_can_uninstall,
        status_detail: if dml_installed {
            format!(
                "Đã cài đặt đầy đủ (DirectML.dll + ONNX Runtime • {:.1} MB)",
                (dml_bytes as f64) / (1024.0 * 1024.0)
            )
        } else {
            "Chưa cài đặt — Máy không có NVIDIA nên cài gói này để mở khóa AI.".to_string()
        },
    };

    // 2. NVIDIA CUDA 12 Pack inspection
    let cuda_dir = resolve_cuda_runtime_dir();
    let cuda_in_addons = addons_root.join("gpu_runtime").join("cuda");
    let cuda_installed = cuda_dir.is_some();
    let cuda_bytes = cuda_dir.as_deref().map(dir_size_bytes).unwrap_or(0);
    let cuda_can_uninstall = has_complete_cuda12_runtime_dlls(&cuda_in_addons)
        || cuda_dir
            .as_deref()
            .map(|d| is_under_addons_root(d, &addons_root))
            .unwrap_or(false);
    let cuda_source = if !cuda_installed {
        "not_installed"
    } else if cuda_can_uninstall {
        "addon_pack"
    } else {
        "bundled_local"
    };

    let pack_cuda = AddonPackInfoDto {
        id: PACK_CUDA12.to_string(),
        title: "Gói 2 · Nhân Tăng tốc Card rời NVIDIA (CUDA 12 + cuDNN 9)".to_string(),
        subtitle: "Mở khóa toàn bộ sức mạnh nhân Tensor Cores (TF32) trên card rời NVIDIA GTX / RTX, tách nhạc 4-stem nhanh gấp 10–30 lần. Máy thường không có NVIDIA KHÔNG cần tải.".to_string(),
        target_audience: "nvidia_gpu".to_string(),
        audience_badge: if has_nvidia_gpu {
            "Khuyên dùng cho Card NVIDIA của bạn".to_string()
        } else {
            "Chỉ dành cho máy có Card NVIDIA".to_string()
        },
        recommended: has_nvidia_gpu,
        estimated_size_label: "~1.1 GB tải nén (~2.5 GB trên đĩa)".to_string(),
        installed: cuda_installed,
        installed_source: cuda_source.to_string(),
        installed_bytes: cuda_bytes,
        can_uninstall: cuda_can_uninstall,
        status_detail: if cuda_installed {
            format!(
                "Đã cài đặt đầy đủ bộ thư viện CUDA 12 + cuDNN 9 ({:.2} GB)",
                (cuda_bytes as f64) / (1024.0 * 1024.0 * 1024.0)
            )
        } else if has_nvidia_gpu {
            "Chưa cài đặt — Tải gói này để chạy tối đa tốc độ trên card NVIDIA.".to_string()
        } else {
            "Không cần cài — Máy hiện tại không dùng card rời NVIDIA.".to_string()
        },
    };

    // 3. UVR5 Studio Models Pack inspection
    let kim_path = resolve_uvr5_kim_vocal2_onnx_path()
        .filter(|p| p.metadata().map(|m| m.len() == UVR5_KIM_VOCAL2_SIZE).unwrap_or(false));
    let inst_path = resolve_uvr5_inst_hq3_onnx_path()
        .filter(|p| p.metadata().map(|m| m.len() == UVR5_INST_HQ3_SIZE).unwrap_or(false));
    let uvr5_installed = kim_path.is_some() && inst_path.is_some();
    let uvr5_bytes = kim_path
        .as_ref()
        .and_then(|p| p.metadata().ok().map(|m| m.len()))
        .unwrap_or(0)
        + inst_path
            .as_ref()
            .and_then(|p| p.metadata().ok().map(|m| m.len()))
            .unwrap_or(0);
    let uvr5_addons_dir = addons_root
        .join("assets")
        .join("uvr5")
        .join("models")
        .join("MDX_Net_Models");
    let uvr5_can_uninstall = uvr5_addons_dir.join("Kim_Vocal_2.onnx").is_file()
        || uvr5_addons_dir.join("UVR-MDX-NET-Inst_HQ_3.onnx").is_file();
    let uvr5_source = if !uvr5_installed {
        "not_installed"
    } else if uvr5_can_uninstall {
        "addon_pack"
    } else {
        "bundled_local"
    };
    let present_uvr_cnt = usize::from(kim_path.is_some()) + usize::from(inst_path.is_some());

    let pack_uvr5 = AddonPackInfoDto {
        id: PACK_UVR5_MODELS.to_string(),
        title: "Gói 3 · Mô hình AI Tách nhạc UVR5 Phòng thu (Kim_Vocal_2 & Inst_HQ_3)".to_string(),
        subtitle: "Bộ đôi trọng số mạng nơ-ron MDX-Net chuẩn phòng thu để tách riêng Giọng hát (Vocals) và Nhạc nền (Instrumental) sạch chi tiết. Dùng chung cho cả Máy thường lẫn Máy NVIDIA.".to_string(),
        target_audience: "all_pcs".to_string(),
        audience_badge: "Cần cho Tách nhạc AI (Mọi dòng máy)".to_string(),
        recommended: true,
        estimated_size_label: "~128 MB (2 mô hình ONNX)".to_string(),
        installed: uvr5_installed,
        installed_source: uvr5_source.to_string(),
        installed_bytes: uvr5_bytes,
        can_uninstall: uvr5_can_uninstall,
        status_detail: if uvr5_installed {
            format!(
                "Đã cài đủ 2/2 mô hình (Kim_Vocal_2 + Inst_HQ_3 • {:.1} MB)",
                (uvr5_bytes as f64) / (1024.0 * 1024.0)
            )
        } else {
            format!("Chưa đủ mô hình ({present_uvr_cnt}/2 tệp) — Cần tải kèm Gói 1 hoặc Gói 2.")
        },
    };

    // 4. Ozone 12 AI Models Pack inspection
    let bsrnn_p = resolve_ozone_bsrnn_onnx_path();
    let delimit_p = resolve_ozone_delimit_onnx_path();
    let genre_p = resolve_asset_file(&["assets/ozone12/models/gparamsBN8.0.npz"]);
    let inst_clf_p = resolve_asset_file(&["assets/ozone12/models/instrument_classifier_params.npz"]);
    let ozone_core_files = [&bsrnn_p, &delimit_p, &genre_p, &inst_clf_p];
    let ozone_present_cnt = ozone_core_files.iter().filter(|p| p.is_some()).count();
    let ozone_installed = ozone_present_cnt == 4;
    let mut ozone_bytes = 0u64;
    for rel in [
        "assets/ozone12/models/ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx",
        "assets/ozone12/models/delimit_exp009_ep080_fixed2.onnx",
        "assets/ozone12/models/gparamsBN8.0.npz",
        "assets/ozone12/models/instrument_classifier_params.npz",
        "assets/ozone12/models/vocals-small_sanitized.npz",
        "assets/ozone12/models/drums-small_sanitized.npz",
        "assets/ozone12/models/bass-small_sanitized.npz",
    ] {
        if let Some(p) = resolve_asset_file(&[rel]) {
            if let Ok(m) = p.metadata() {
                ozone_bytes = ozone_bytes.saturating_add(m.len());
            }
        }
    }
    let ozone_addons_dir = addons_root.join("assets").join("ozone12").join("models");
    let ozone_can_uninstall = ozone_addons_dir
        .read_dir()
        .map(|mut it| it.next().is_some())
        .unwrap_or(false);
    let ozone_source = if !ozone_installed {
        "not_installed"
    } else if ozone_can_uninstall {
        "addon_pack"
    } else {
        "bundled_local"
    };

    let pack_ozone = AddonPackInfoDto {
        id: PACK_OZONE12_AI.to_string(),
        title: "Gói 4 · Mô hình AI Mastering Ozone 12 (Master Assistant & 4-Stem BSRNN)".to_string(),
        subtitle: "Cung cấp AI tự động nhận diện thể loại nhạc (Master Assistant), cân chỉnh 4-stem trực tiếp (BSRNN) và khử méo đỉnh (DeLimit). Tùy chọn cho cả Máy thường lẫn Máy NVIDIA.".to_string(),
        target_audience: "all_pcs".to_string(),
        audience_badge: "Tùy chọn AI Mastering (Mọi dòng máy)".to_string(),
        recommended: true,
        estimated_size_label: "~85 MB nén (~112 MB trên đĩa)".to_string(),
        installed: ozone_installed,
        installed_source: ozone_source.to_string(),
        installed_bytes: ozone_bytes,
        can_uninstall: ozone_can_uninstall,
        status_detail: if ozone_installed {
            format!(
                "Đã cài đủ bộ trọng số Ozone 12 AI ({:.1} MB)",
                (ozone_bytes as f64) / (1024.0 * 1024.0)
            )
        } else {
            format!("Chưa đủ trọng số ({ozone_present_cnt}/4 tệp chính) — Tải khi cần dùng AI Master Assistant.")
        },
    };

    // Detect if a higher tier was installed after an earlier tier was already locked in memory
    let (restart_recommended, restart_reason) = if has_nvidia_gpu
        && cuda_installed
        && rt_status.tier != GpuExecutionTier::CudaTensorCore
    {
        (
            true,
            Some(
                "Đã cài đặt Nhân NVIDIA CUDA 12! Hãy bấm Khởi động lại ứng dụng để chuyển sang chạy trên nhân CUDA 12."
                    .to_string(),
            ),
        )
    } else if !has_nvidia_gpu
        && dml_installed
        && rt_status.tier == GpuExecutionTier::PureRustFallback
    {
        (
            true,
            Some(
                "Đã cài đặt Nhân DirectML! Hãy khởi động lại ứng dụng để kích hoạt tăng tốc DirectX 12."
                    .to_string(),
            ),
        )
    } else {
        (false, None)
    };

    AddonSystemOverviewDto {
        addons_root_path: addons_root.display().to_string(),
        has_nvidia_gpu,
        nvidia_device_name,
        nvidia_vram_mib,
        active_tier: rt_status.tier,
        active_badge: rt_status.badge_label.clone(),
        active_provider: rt_status.provider_name.clone(),
        restart_recommended,
        restart_reason,
        download_in_progress: DOWNLOAD_ACTIVE.load(Ordering::Acquire),
        packs: vec![pack_dml, pack_cuda, pack_uvr5, pack_ozone],
    }
}

pub fn request_cancel_addon_download() {
    CANCEL_REQUESTED.store(true, Ordering::Release);
}

fn check_cancelled() -> Result<(), String> {
    if CANCEL_REQUESTED.load(Ordering::Acquire) {
        Err("Đã hủy tải gói theo yêu cầu người dùng.".to_string())
    } else {
        Ok(())
    }
}

fn emit_progress(app: &AppHandle, evt: AddonDownloadProgressDto) {
    let _ = app.emit("addon-pack-progress", evt);
}

fn build_http_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .user_agent("ToneForgeStudio-AddonManager/1.0")
        .connect_timeout(std::time::Duration::from_secs(20))
        .build()
        .map_err(|e| format!("Không khởi tạo được kết nối mạng: {}", sanitize_error_message(&e.to_string())))
}

async fn resolve_pypi_win_amd64_wheel(
    client: &reqwest::Client,
    pkg_name: &str,
    pinned_version: Option<&str>,
) -> Result<PypiWheelEntry, String> {
    let meta_url = match pinned_version {
        Some(ver) => format!("https://pypi.org/pypi/{pkg_name}/{ver}/json"),
        None => format!("https://pypi.org/pypi/{pkg_name}/json"),
    };
    let resp = client
        .get(&meta_url)
        .send()
        .await
        .map_err(|e| format!("Không kết nối được máy chủ gói phụ trợ: {}", sanitize_error_message(&e.to_string())))?;

    if !resp.status().is_success() {
        if pinned_version.is_some() {
            // Fallback to latest version on PyPI if pinned version metadata is unavailable
            return Box::pin(resolve_pypi_win_amd64_wheel(client, pkg_name, None)).await;
        }
        return Err(format!(
            "Máy chủ gói phụ trợ trả mã trạng thái HTTP {} cho thành phần {}",
            resp.status().as_u16(),
            pkg_name
        ));
    }

    let bytes = resp
        .bytes()
        .await
        .map_err(|e| format!("Lỗi đọc danh mục gói: {}", sanitize_error_message(&e.to_string())))?;
    let parsed: PypiReleaseResponse = serde_json::from_slice(&bytes)
        .map_err(|_| "Dữ liệu danh mục gói phụ trợ không hợp lệ.".to_string())?;

    parsed
        .urls
        .into_iter()
        .find(|u| {
            u.packagetype == "bdist_wheel"
                && u.filename.to_ascii_lowercase().ends_with("win_amd64.whl")
        })
        .ok_or_else(|| format!("Không tìm thấy bản đóng gói Windows x64 cho thành phần {pkg_name}."))
}

struct StreamDownloadSpec<'a> {
    pack_id: &'a str,
    url: &'a str,
    dest_file: &'a Path,
    expected_size: Option<u64>,
    expected_sha256: Option<&'a str>,
    step_current: usize,
    step_total: usize,
    label: &'a str,
    prior_bytes: u64,
    overall_total_bytes: u64,
}

async fn stream_download_verified(
    app: &AppHandle,
    client: &reqwest::Client,
    spec: StreamDownloadSpec<'_>,
) -> Result<u64, String> {
    check_cancelled()?;
    if let Some(parent) = spec.dest_file.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Không tạo được thư mục đích: {}", sanitize_error_message(&e.to_string())))?;
    }

    let part_path = spec.dest_file.with_extension("part_dl");
    let _ = fs::remove_file(&part_path);

    let mut resp = client
        .get(spec.url)
        .send()
        .await
        .map_err(|e| format!("Lỗi kết nối khi tải {}: {}", spec.label, sanitize_error_message(&e.to_string())))?;

    let status = resp.status();
    if status.as_u16() == 404 {
        return Err(format!(
            "Gói '{}' hiện chưa được đồng bộ lên máy chủ phát hành chính thức. Vui lòng thử lại sau.",
            spec.label
        ));
    }
    if !status.is_success() {
        return Err(format!(
            "Máy chủ tải gói trả lỗi HTTP {} khi tải {}.",
            status.as_u16(),
            spec.label
        ));
    }

    let file_total = spec
        .expected_size
        .or_else(|| resp.content_length())
        .unwrap_or(0);
    let grand_total = if spec.overall_total_bytes > 0 {
        spec.overall_total_bytes
    } else {
        file_total
    };

    let mut out_file = File::create(&part_path)
        .map_err(|e| format!("Không thể ghi tệp tạm: {}", sanitize_error_message(&e.to_string())))?;
    let mut hasher = Sha256::new();
    let mut file_downloaded = 0u64;
    let mut last_emit_bytes = 0u64;

    while let Some(chunk) = resp
        .chunk()
        .await
        .map_err(|e| {
            let _ = fs::remove_file(&part_path);
            format!("Ngắt kết nối khi đang tải {}: {}", spec.label, sanitize_error_message(&e.to_string()))
        })?
    {
        if let Err(cancel_err) = check_cancelled() {
            drop(out_file);
            let _ = fs::remove_file(&part_path);
            return Err(cancel_err);
        }

        out_file.write_all(&chunk).map_err(|e| {
            let _ = fs::remove_file(&part_path);
            format!("Lỗi ghi đĩa (kiểm tra dung lượng ổ cứng trống): {}", sanitize_error_message(&e.to_string()))
        })?;
        hasher.update(&chunk);
        file_downloaded += chunk.len() as u64;

        if file_downloaded.saturating_sub(last_emit_bytes) >= 256 * 1024 || file_downloaded == file_total {
            last_emit_bytes = file_downloaded;
            let cum_done = spec.prior_bytes.saturating_add(file_downloaded);
            let pct = if grand_total > 0 {
                ((cum_done as f64 / grand_total as f64) * 100.0)
                    .round()
                    .clamp(0.0, 99.0) as u8
            } else {
                0
            };
            let done_mb = (cum_done as f64) / (1024.0 * 1024.0);
            let msg = if grand_total > 0 {
                let tot_mb = (grand_total as f64) / (1024.0 * 1024.0);
                format!(
                    "Bước {}/{}: Đang tải {} ({:.1} / {:.1} MB • {}%)",
                    spec.step_current, spec.step_total, spec.label, done_mb, tot_mb, pct
                )
            } else {
                format!(
                    "Bước {}/{}: Đang tải {} ({:.1} MB)",
                    spec.step_current, spec.step_total, spec.label, done_mb
                )
            };
            emit_progress(
                app,
                AddonDownloadProgressDto {
                    pack_id: spec.pack_id.to_string(),
                    phase: "downloading".to_string(),
                    step_current: spec.step_current,
                    step_total: spec.step_total,
                    downloaded_bytes: cum_done,
                    total_bytes: grand_total,
                    percent: pct,
                    message: msg,
                },
            );
        }
    }

    out_file
        .flush()
        .map_err(|e| format!("Lỗi đồng bộ tệp xuống đĩa: {}", sanitize_error_message(&e.to_string())))?;
    drop(out_file);

    if let Some(exp_sz) = spec.expected_size {
        if file_downloaded != exp_sz {
            let _ = fs::remove_file(&part_path);
            return Err(format!(
                "Kích thước tệp {} không khớp ({} / {} byte). Vui lòng bấm tải lại.",
                spec.label, file_downloaded, exp_sz
            ));
        }
    }

    if let Some(exp_sha) = spec.expected_sha256 {
        let got_sha = hex::encode(hasher.finalize());
        if !got_sha.eq_ignore_ascii_case(exp_sha) {
            let _ = fs::remove_file(&part_path);
            return Err(format!(
                "Kiểm tra toàn vẹn SHA-256 thất bại cho {}. Tệp tải lỗi đã được dọn sạch.",
                spec.label
            ));
        }
    }

    if spec.dest_file.exists() {
        let _ = fs::remove_file(spec.dest_file);
    }
    fs::rename(&part_path, spec.dest_file)
        .map_err(|e| format!("Không thể hoàn tất lưu tệp {}: {}", spec.label, sanitize_error_message(&e.to_string())))?;

    Ok(file_downloaded)
}

fn extract_dlls_from_wheel_zip(wheel_path: &Path, target_dir: &Path) -> Result<usize, String> {
    fs::create_dir_all(target_dir)
        .map_err(|e| format!("Không tạo được thư mục runtime: {}", sanitize_error_message(&e.to_string())))?;
    let file = File::open(wheel_path)
        .map_err(|e| format!("Không mở được gói nén vừa tải: {}", sanitize_error_message(&e.to_string())))?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|_| "Định dạng gói nén tải về bị hỏng.".to_string())?;

    let mut extracted_count = 0usize;
    for idx in 0..archive.len() {
        check_cancelled()?;
        let mut entry = archive
            .by_index(idx)
            .map_err(|_| "Lỗi đọc mục bên trong gói nén.".to_string())?;
        if entry.is_dir() {
            continue;
        }
        let Some(fname) = Path::new(entry.name())
            .file_name()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string())
        else {
            continue;
        };
        if !fname.to_ascii_lowercase().ends_with(".dll") {
            continue;
        }
        if fname.eq_ignore_ascii_case("onnxruntime_providers_tensorrt.dll") {
            continue;
        }

        let tmp_dest = target_dir.join(format!("{fname}.extract_tmp"));
        let final_dest = target_dir.join(&fname);
        let mut out = File::create(&tmp_dest)
            .map_err(|e| format!("Không ghi được thư viện {fname}: {}", sanitize_error_message(&e.to_string())))?;
        let mut buf = [0u8; 64 * 1024];
        loop {
            let n = entry
                .read(&mut buf)
                .map_err(|_| format!("Lỗi giải nén thư viện {fname}."))?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n])
                .map_err(|e| format!("Lỗi ghi đĩa khi giải nén {fname}: {}", sanitize_error_message(&e.to_string())))?;
        }
        out.flush().ok();
        drop(out);

        if final_dest.exists() {
            let _ = move_to_deferred_trash(&final_dest);
        }
        fs::rename(&tmp_dest, &final_dest).map_err(|e| {
            let _ = fs::remove_file(&tmp_dest);
            format!("Không thể cập nhật thư viện {fname}: {}", sanitize_error_message(&e.to_string()))
        })?;
        extracted_count += 1;
    }

    Ok(extracted_count)
}

fn extract_ozone12_models_zip(zip_path: &Path, models_dir: &Path) -> Result<usize, String> {
    fs::create_dir_all(models_dir)
        .map_err(|e| format!("Không tạo được thư mục Ozone 12: {}", sanitize_error_message(&e.to_string())))?;
    let file = File::open(zip_path)
        .map_err(|e| format!("Không mở được gói Ozone 12: {}", sanitize_error_message(&e.to_string())))?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|_| "Gói nén Ozone 12 AI không hợp lệ.".to_string())?;

    let mut extracted = 0usize;
    for idx in 0..archive.len() {
        check_cancelled()?;
        let mut entry = archive
            .by_index(idx)
            .map_err(|_| "Lỗi đọc tệp trong gói Ozone 12 AI.".to_string())?;
        if entry.is_dir() {
            continue;
        }
        let Some(fname) = Path::new(entry.name())
            .file_name()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string())
        else {
            continue;
        };
        let low = fname.to_ascii_lowercase();
        if !low.ends_with(".onnx") && !low.ends_with(".npz") {
            continue;
        }
        let tmp_dest = models_dir.join(format!("{fname}.extract_tmp"));
        let final_dest = models_dir.join(&fname);
        let mut out = File::create(&tmp_dest)
            .map_err(|e| format!("Không ghi được mô hình {fname}: {}", sanitize_error_message(&e.to_string())))?;
        std::io::copy(&mut entry, &mut out)
            .map_err(|e| format!("Lỗi giải nén mô hình {fname}: {}", sanitize_error_message(&e.to_string())))?;
        out.flush().ok();
        drop(out);
        if final_dest.exists() {
            let _ = fs::remove_file(&final_dest);
        }
        fs::rename(&tmp_dest, &final_dest)
            .map_err(|e| format!("Không lưu được mô hình {fname}: {}", sanitize_error_message(&e.to_string())))?;
        extracted += 1;
    }
    Ok(extracted)
}

async fn install_wheel_pack(
    app: &AppHandle,
    pack_id: &str,
    specs: &[(&str, Option<&str>)],
    target_dir: &Path,
    staging_dir: &Path,
) -> Result<(), String> {
    let client = build_http_client()?;
    emit_progress(
        app,
        AddonDownloadProgressDto {
            pack_id: pack_id.to_string(),
            phase: "starting".to_string(),
            step_current: 0,
            step_total: specs.len(),
            downloaded_bytes: 0,
            total_bytes: 0,
            percent: 0,
            message: "Đang kiểm tra danh mục thành phần trên máy chủ…".to_string(),
        },
    );

    let mut wheels = Vec::with_capacity(specs.len());
    let mut grand_total = 0u64;
    for &(pkg, ver) in specs {
        check_cancelled()?;
        let w = resolve_pypi_win_amd64_wheel(&client, pkg, ver).await?;
        grand_total = grand_total.saturating_add(w.size);
        wheels.push((pkg, w));
    }

    fs::create_dir_all(staging_dir)
        .map_err(|e| format!("Không tạo được thư mục tạm: {}", sanitize_error_message(&e.to_string())))?;

    let mut cumulative_bytes = 0u64;
    let total_steps = wheels.len();
    for (idx, (pkg_label, wheel)) in wheels.into_iter().enumerate() {
        check_cancelled()?;
        let step_num = idx + 1;
        let wheel_file = staging_dir.join(&wheel.filename);
        let dl_bytes = stream_download_verified(
            app,
            &client,
            StreamDownloadSpec {
                pack_id,
                url: &wheel.url,
                dest_file: &wheel_file,
                expected_size: Some(wheel.size),
                expected_sha256: Some(&wheel.digests.sha256),
                step_current: step_num,
                step_total: total_steps,
                label: pkg_label,
                prior_bytes: cumulative_bytes,
                overall_total_bytes: grand_total,
            },
        )
        .await?;
        cumulative_bytes = cumulative_bytes.saturating_add(dl_bytes);

        let pct = if grand_total > 0 {
            ((cumulative_bytes as f64 / grand_total as f64) * 100.0)
                .round()
                .clamp(1.0, 99.0) as u8
        } else {
            95
        };
        emit_progress(
            app,
            AddonDownloadProgressDto {
                pack_id: pack_id.to_string(),
                phase: "extracting".to_string(),
                step_current: step_num,
                step_total: total_steps,
                downloaded_bytes: cumulative_bytes,
                total_bytes: grand_total,
                percent: pct,
                message: format!("Bước {step_num}/{total_steps}: Đang giải nén thư viện {pkg_label}…"),
            },
        );

        let wheel_for_extract = wheel_file.clone();
        let target_for_extract = target_dir.to_path_buf();
        tauri::async_runtime::spawn_blocking(move || {
            let res = extract_dlls_from_wheel_zip(&wheel_for_extract, &target_for_extract);
            let _ = fs::remove_file(&wheel_for_extract);
            res
        })
        .await
        .map_err(|_| "Luồng giải nén gặp sự cố bất ngờ.".to_string())??;
    }

    let _ = fs::remove_dir_all(staging_dir);
    Ok(())
}

pub async fn install_addon_pack(
    app: AppHandle,
    pack_id: String,
) -> Result<AddonSystemOverviewDto, String> {
    if DOWNLOAD_ACTIVE
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return Err("Đang có một gói bổ sung khác trong quá trình tải. Vui lòng đợi hoàn tất.".to_string());
    }
    CANCEL_REQUESTED.store(false, Ordering::Release);

    let result = install_addon_pack_inner(&app, &pack_id).await;
    DOWNLOAD_ACTIVE.store(false, Ordering::Release);

    match result {
        Ok(()) => {
            notify_addon_packs_changed();
            let overview = tauri::async_runtime::spawn_blocking(inspect_addon_system)
                .await
                .map_err(|_| "Lỗi đọc trạng thái gói sau cài đặt.".to_string())?;
            emit_progress(
                &app,
                AddonDownloadProgressDto {
                    pack_id,
                    phase: "completed".to_string(),
                    step_current: 1,
                    step_total: 1,
                    downloaded_bytes: 100,
                    total_bytes: 100,
                    percent: 100,
                    message: "Đã tải, xác thực chữ ký băm và kích hoạt gói thành công!".to_string(),
                },
            );
            Ok(overview)
        }
        Err(err_msg) => {
            let cancelled = CANCEL_REQUESTED.load(Ordering::Acquire);
            emit_progress(
                &app,
                AddonDownloadProgressDto {
                    pack_id,
                    phase: if cancelled { "cancelled" } else { "error" }.to_string(),
                    step_current: 0,
                    step_total: 0,
                    downloaded_bytes: 0,
                    total_bytes: 0,
                    percent: 0,
                    message: err_msg.clone(),
                },
            );
            Err(err_msg)
        }
    }
}

async fn install_addon_pack_inner(app: &AppHandle, pack_id: &str) -> Result<(), String> {
    let addons_root = addons_root_or_err()?;
    let staging_dir = addons_root.join(format!(".staging_{pack_id}"));
    let _ = fs::remove_dir_all(&staging_dir);

    let res = match pack_id {
        PACK_DIRECTML => {
            let dml_dir = addons_root.join("gpu_runtime").join("directml");
            install_wheel_pack(
                app,
                pack_id,
                &[("onnxruntime-directml", Some("1.24.4"))],
                &dml_dir,
                &staging_dir,
            )
            .await?;
            if !has_complete_directml_runtime_dlls(&dml_dir) {
                return Err("Giải nén xong nhưng thiếu DirectML.dll hoặc onnxruntime.dll.".to_string());
            }
            Ok(())
        }
        PACK_CUDA12 => {
            let cuda_dir = addons_root.join("gpu_runtime").join("cuda");
            install_wheel_pack(app, pack_id, CUDA12_WHEEL_SPECS, &cuda_dir, &staging_dir).await?;
            if !has_complete_cuda12_runtime_dlls(&cuda_dir) {
                return Err("Giải nén xong nhưng thiếu một số thư viện CUDA 12 / cuDNN 9 bắt buộc.".to_string());
            }
            Ok(())
        }
        PACK_UVR5_MODELS => {
            let client = build_http_client()?;
            let mdx_dir = addons_root
                .join("assets")
                .join("uvr5")
                .join("models")
                .join("MDX_Net_Models");
            fs::create_dir_all(&mdx_dir)
                .map_err(|e| format!("Không tạo được thư mục UVR5: {}", sanitize_error_message(&e.to_string())))?;

            let models = [
                (
                    "Kim_Vocal_2.onnx",
                    UVR5_KIM_VOCAL2_URL,
                    UVR5_KIM_VOCAL2_SIZE,
                    UVR5_KIM_VOCAL2_SHA256,
                ),
                (
                    "UVR-MDX-NET-Inst_HQ_3.onnx",
                    UVR5_INST_HQ3_URL,
                    UVR5_INST_HQ3_SIZE,
                    UVR5_INST_HQ3_SHA256,
                ),
            ];
            let grand_total = UVR5_KIM_VOCAL2_SIZE + UVR5_INST_HQ3_SIZE;
            let mut prior = 0u64;
            for (i, (fname, url, sz, sha)) in models.iter().enumerate() {
                let dest = mdx_dir.join(fname);
                let already_ok = dest.metadata().map(|m| m.len() == *sz).unwrap_or(false);
                if already_ok {
                    prior += *sz;
                    continue;
                }
                let got = stream_download_verified(
                    app,
                    &client,
                    StreamDownloadSpec {
                        pack_id,
                        url,
                        dest_file: &dest,
                        expected_size: Some(*sz),
                        expected_sha256: Some(sha),
                        step_current: i + 1,
                        step_total: models.len(),
                        label: fname,
                        prior_bytes: prior,
                        overall_total_bytes: grand_total,
                    },
                )
                .await?;
                prior += got;
            }
            Ok(())
        }
        PACK_OZONE12_AI => {
            let client = build_http_client()?;
            fs::create_dir_all(&staging_dir)
                .map_err(|e| format!("Không tạo được thư mục tạm: {}", sanitize_error_message(&e.to_string())))?;
            let zip_file = staging_dir.join("ToneForge_Ozone12_AI_Pack.zip");
            stream_download_verified(
                app,
                &client,
                StreamDownloadSpec {
                    pack_id,
                    url: OZONE12_AI_PACK_URL,
                    dest_file: &zip_file,
                    expected_size: None,
                    expected_sha256: None,
                    step_current: 1,
                    step_total: 2,
                    label: "Ozone 12 AI Pack",
                    prior_bytes: 0,
                    overall_total_bytes: 0,
                },
            )
            .await?;

            emit_progress(
                app,
                AddonDownloadProgressDto {
                    pack_id: pack_id.to_string(),
                    phase: "extracting".to_string(),
                    step_current: 2,
                    step_total: 2,
                    downloaded_bytes: 95,
                    total_bytes: 100,
                    percent: 96,
                    message: "Bước 2/2: Đang giải nén trọng số AI Mastering Ozone 12…".to_string(),
                },
            );

            let models_dir = addons_root.join("assets").join("ozone12").join("models");
            let zip_clone = zip_file.clone();
            let count = tauri::async_runtime::spawn_blocking(move || {
                let r = extract_ozone12_models_zip(&zip_clone, &models_dir);
                let _ = fs::remove_file(&zip_clone);
                r
            })
            .await
            .map_err(|_| "Luồng giải nén Ozone 12 AI gặp sự cố.".to_string())??;

            if count == 0 {
                return Err("Gói Ozone 12 AI không chứa tệp trọng số hợp lệ.".to_string());
            }
            Ok(())
        }
        other => Err(format!("Mã gói bổ sung không hợp lệ: {other}")),
    };

    let _ = fs::remove_dir_all(&staging_dir);
    res
}

/// Moves a file or directory inside `addons/` into `.trash_<ts>` when Win32 locks an active DLL,
/// allowing instant logical uninstall and clean deletion on next launch.
fn move_to_deferred_trash(target: &Path) -> Result<(), String> {
    if !target.exists() {
        return Ok(());
    }
    let rm_res = if target.is_dir() {
        fs::remove_dir_all(target)
    } else {
        fs::remove_file(target)
    };
    if rm_res.is_ok() {
        return Ok(());
    }

    let addons_root = addons_root_or_err()?;
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let trash_dir = addons_root.join(format!(".trash_{ts}"));
    fs::create_dir_all(&trash_dir)
        .map_err(|e| format!("Không thể tạo thư mục dọn tạm: {}", sanitize_error_message(&e.to_string())))?;

    if target.is_file() {
        let fname = target
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("locked.dll");
        fs::rename(target, trash_dir.join(fname))
            .map_err(|e| format!("Không thể gỡ tệp đang khóa: {}", sanitize_error_message(&e.to_string())))?;
    } else if let Ok(entries) = fs::read_dir(target) {
        for (idx, entry) in entries.flatten().enumerate() {
            let p = entry.path();
            if fs::remove_file(&p).is_err() {
                let fname = p
                    .file_name()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("locked_{idx}.dll"));
                let _ = fs::rename(&p, trash_dir.join(fname));
            }
        }
        let _ = fs::remove_dir_all(target);
    }
    Ok(())
}

pub fn uninstall_addon_pack(pack_id: &str) -> Result<AddonSystemOverviewDto, String> {
    if DOWNLOAD_ACTIVE.load(Ordering::Acquire) {
        return Err("Đang có gói trong quá trình tải — vui lòng hủy hoặc đợi tải xong trước khi gỡ.".to_string());
    }
    let addons_root = addons_root_or_err()?;
    let target = match pack_id {
        PACK_DIRECTML => addons_root.join("gpu_runtime").join("directml"),
        PACK_CUDA12 => addons_root.join("gpu_runtime").join("cuda"),
        PACK_UVR5_MODELS => addons_root
            .join("assets")
            .join("uvr5")
            .join("models")
            .join("MDX_Net_Models"),
        PACK_OZONE12_AI => addons_root.join("assets").join("ozone12").join("models"),
        other => return Err(format!("Mã gói không hợp lệ: {other}")),
    };

    if !is_under_addons_root(&target, &addons_root) {
        return Err("Chỉ cho phép gỡ các gói mở rộng nằm trong thư mục người dùng.".to_string());
    }

    move_to_deferred_trash(&target)?;
    notify_addon_packs_changed();
    Ok(inspect_addon_system())
}

pub fn open_addons_folder_in_explorer() -> Result<String, String> {
    let dir = addons_root_or_err()?;
    #[cfg(windows)]
    {
        std::process::Command::new("explorer.exe")
            .arg(&dir)
            .spawn()
            .map_err(|e| format!("Không mở được File Explorer: {e}"))?;
    }
    Ok(dir.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_error_strips_git_and_cdn_urls() {
        let dirty = "error sending request for url (https://github.com/Selorson/ToneForge-Release/releases/latest/download/a.zip): connection reset";
        let clean = sanitize_error_message(dirty);
        assert!(!clean.to_ascii_lowercase().contains("github"));
        assert!(!clean.to_ascii_lowercase().contains("selorson"));
        assert!(!clean.contains("https://"));
    }
}
