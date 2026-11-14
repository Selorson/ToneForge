//! `scanner`: Crash-Isolated Out-of-Process Plugin Scanner & Quarantine Blacklist Manager
//! (`toneforge-plugin-scanner --probe <path>`).
//!
//! Scans standard Windows VST3 & CLAP directories (`C:\Program Files\Common Files\VST3`,
//! `C:\Program Files\Common Files\CLAP`, plus user-selected folders) in isolated child processes
//! with strict per-binary timeouts so faulty/unlicensed third-party DLLs causing
//! `STATUS_ACCESS_VIOLATION` (`0xC0000005`) or deadlocks never crash ToneForge Studio.

use crate::clap_host::enumerate_clap_plugins_in_library;
use crate::vst3_host::enumerate_vst3_plugins_in_library;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PluginBinaryFormat {
    Vst3,
    Clap,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScannedPluginDescriptor {
    pub id: String,
    pub name: String,
    pub vendor: String,
    pub version: String,
    pub category: String,
    pub format: PluginBinaryFormat,
    pub path: String,
    pub is_instrument: bool,
    pub latency_samples: u32,
    pub num_params: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PluginQuarantineEntry {
    pub path: String,
    pub reason: String,
    pub exit_code: Option<i32>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PluginScanReport {
    pub plugins: Vec<ScannedPluginDescriptor>,
    pub quarantined: Vec<PluginQuarantineEntry>,
    pub scanned_binaries: usize,
}

/// Probes a single binary inside the current process (called by `toneforge-plugin-scanner --probe`).
pub fn probe_single_binary_in_process(path: &Path) -> Result<Vec<ScannedPluginDescriptor>, String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let path_str = path.to_string_lossy().into_owned();
    let mut found = Vec::new();
    let mut errors = Vec::new();

    if ext == "clap" || ext == "dll" {
        match enumerate_clap_plugins_in_library(path) {
            Ok(list) => {
                for d in list {
                    found.push(ScannedPluginDescriptor {
                        id: d.id,
                        name: d.name,
                        vendor: d.vendor,
                        version: d.version,
                        category: if d.is_instrument {
                            "Instrument".to_string()
                        } else {
                            "Audio Effect".to_string()
                        },
                        format: PluginBinaryFormat::Clap,
                        path: path_str.clone(),
                        is_instrument: d.is_instrument,
                        latency_samples: d.latency_samples,
                        num_params: d.num_params,
                    });
                }
            }
            Err(e) => errors.push(e),
        }
    }

    if ext == "vst3" || ext == "dll" || path.is_dir() {
        match enumerate_vst3_plugins_in_library(path) {
            Ok(list) => {
                for d in list {
                    found.push(ScannedPluginDescriptor {
                        id: d.cid_hex,
                        name: d.name,
                        vendor: d.vendor,
                        version: "3.7".to_string(),
                        category: d.category,
                        format: PluginBinaryFormat::Vst3,
                        path: path_str.clone(),
                        is_instrument: d.is_instrument,
                        latency_samples: d.latency_samples,
                        num_params: d.num_params,
                    });
                }
            }
            Err(e) => errors.push(e),
        }
    }

    if !found.is_empty() {
        Ok(found)
    } else if !errors.is_empty() {
        Err(errors.join(" | "))
    } else {
        Err(format!("Unsupported plugin extension for {}", path.display()))
    }
}

/// Locates `toneforge-plugin-scanner.exe` next to the current executable or in workspace `target/` dirs.
pub fn locate_scanner_executable() -> Option<PathBuf> {
    if let Ok(cur_exe) = std::env::current_exe() {
        if let Some(parent) = cur_exe.parent() {
            let sibling = parent.join("toneforge-plugin-scanner.exe");
            if sibling.is_file() {
                return Some(sibling);
            }
            if let Some(grand) = parent.parent() {
                let up_sibling = grand.join("toneforge-plugin-scanner.exe");
                if up_sibling.is_file() {
                    return Some(up_sibling);
                }
            }
        }
    }
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    if let Some(ws_root) = manifest_dir.parent().and_then(|p| p.parent()) {
        for sub in ["release", "debug"] {
            let candidate = ws_root
                .join("target")
                .join(sub)
                .join("toneforge-plugin-scanner.exe");
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

/// Collects candidate `.vst3` files/bundles and `.clap` files recursively up to `max_depth`.
pub fn collect_candidate_plugin_paths(root: &Path, max_depth: usize) -> Vec<PathBuf> {
    let mut out = Vec::new();
    fn visit(dir: &Path, depth: usize, max_depth: usize, out: &mut Vec<PathBuf>) {
        if depth > max_depth || !dir.exists() {
            return;
        }
        if dir.is_file() {
            out.push(dir.to_path_buf());
            return;
        }
        if let Some(ext) = dir.extension().and_then(|e| e.to_str()) {
            if ext.eq_ignore_ascii_case("vst3") && dir.join("Contents").is_dir() {
                out.push(dir.to_path_buf());
                return;
            }
        }
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let p = entry.path();
            let ext = p
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if p.is_dir() {
                if ext == "vst3" {
                    out.push(p);
                } else {
                    visit(&p, depth + 1, max_depth, out);
                }
            } else if ext == "vst3" || ext == "clap" {
                out.push(p);
            }
        }
    }
    visit(root, 0, max_depth, &mut out);
    out
}

/// Default standard Windows x64 VST3 & CLAP directories (plus built workspace test plugin if present).
pub fn default_plugin_scan_directories() -> Vec<PathBuf> {
    let mut dirs = vec![
        PathBuf::from(r"C:\Program Files\Common Files\VST3"),
        PathBuf::from(r"C:\Program Files\Common Files\CLAP"),
    ];
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    if let Some(ws_root) = manifest_dir.parent().and_then(|p| p.parent()) {
        for sub in ["release", "debug"] {
            let test_dll = ws_root
                .join("target")
                .join(sub)
                .join("test_vst3_clap_plugin.dll");
            if test_dll.is_file() {
                dirs.push(test_dll);
                break;
            }
        }
    }
    dirs
}

/// Probes a single plugin path inside an isolated child process (`toneforge-plugin-scanner.exe`)
/// with `timeout`. If `scanner_exe` is `None`, falls back to in-process probing.
pub fn probe_plugin_isolated(
    scanner_exe: Option<&Path>,
    target_path: &Path,
    timeout: Duration,
    extra_args: &[&str],
) -> Result<Vec<ScannedPluginDescriptor>, PluginQuarantineEntry> {
    let path_str = target_path.to_string_lossy().into_owned();
    let Some(exe) = scanner_exe else {
        return probe_single_binary_in_process(target_path).map_err(|reason| {
            PluginQuarantineEntry {
                path: path_str,
                reason,
                exit_code: None,
            }
        });
    };

    let mut cmd = Command::new(exe);
    cmd.arg("--probe").arg(target_path);
    for a in extra_args {
        cmd.arg(a);
    }
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| PluginQuarantineEntry {
        path: path_str.clone(),
        reason: format!("Failed to spawn toneforge-plugin-scanner: {e}"),
        exit_code: None,
    })?;

    let start = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(st)) => break st,
            Ok(None) => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(PluginQuarantineEntry {
                        path: path_str,
                        reason: format!(
                            "Plugin scan timed out after {} ms (hung DllMain/Factory)",
                            timeout.as_millis()
                        ),
                        exit_code: None,
                    });
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(e) => {
                let _ = child.kill();
                return Err(PluginQuarantineEntry {
                    path: path_str,
                    reason: format!("Scanner wait error: {e}"),
                    exit_code: None,
                });
            }
        }
    };

    let mut stdout_buf = String::new();
    let mut stderr_buf = String::new();
    if let Some(mut out) = child.stdout.take() {
        let _ = out.read_to_string(&mut stdout_buf);
    }
    if let Some(mut err) = child.stderr.take() {
        let _ = err.read_to_string(&mut stderr_buf);
    }

    if !status.success() {
        let code = status.code();
        return Err(PluginQuarantineEntry {
            path: path_str,
            reason: if stderr_buf.trim().is_empty() {
                format!("Isolated scanner exited abnormally (code={code:?})")
            } else {
                stderr_buf.trim().to_string()
            },
            exit_code: code,
        });
    }

    serde_json::from_str::<Vec<ScannedPluginDescriptor>>(&stdout_buf).map_err(|e| {
        PluginQuarantineEntry {
            path: path_str,
            reason: format!("Malformed scanner JSON output: {e}"),
            exit_code: status.code(),
        }
    })
}

/// Scans a list of directories/files with crash isolation and optional persistent quarantine JSON file.
pub fn scan_plugin_directories_isolated(
    roots: &[PathBuf],
    quarantine_json_path: Option<&Path>,
    per_plugin_timeout: Duration,
) -> PluginScanReport {
    let scanner_exe = locate_scanner_executable();
    let mut blacklisted_paths: HashSet<String> = HashSet::new();
    let mut report = PluginScanReport::default();

    if let Some(qpath) = quarantine_json_path {
        if let Ok(txt) = std::fs::read_to_string(qpath) {
            if let Ok(existing) = serde_json::from_str::<Vec<PluginQuarantineEntry>>(&txt) {
                for item in existing {
                    blacklisted_paths.insert(item.path.clone());
                    report.quarantined.push(item);
                }
            }
        }
    }

    let mut seen_candidates = HashSet::new();
    let mut candidates = Vec::new();
    for r in roots {
        for c in collect_candidate_plugin_paths(r, 4) {
            let key = c.to_string_lossy().to_ascii_lowercase();
            if seen_candidates.insert(key) {
                candidates.push(c);
            }
        }
    }

    for candidate in candidates {
        let c_str = candidate.to_string_lossy().into_owned();
        if blacklisted_paths.contains(&c_str) {
            continue;
        }
        report.scanned_binaries += 1;
        match probe_plugin_isolated(scanner_exe.as_deref(), &candidate, per_plugin_timeout, &[]) {
            Ok(mut list) => report.plugins.append(&mut list),
            Err(qentry) => {
                blacklisted_paths.insert(qentry.path.clone());
                report.quarantined.push(qentry);
            }
        }
    }

    if let Some(qpath) = quarantine_json_path {
        if let Some(parent) = qpath.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(&report.quarantined) {
            let _ = std::fs::write(qpath, json);
        }
    }

    report
}
