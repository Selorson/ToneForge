//! CLI Utility `extract_izresembed` (`05_master_rust_svelte_blueprint.md` Section 7.1):
//! Reads the `IZRESEMBED` TOC at `0x068E5B84..0x0690B624` in `iZOzone12Core.dll` and extracts
//! Clean-Room `.onnx` / `.npz` models and parameter tables to `assets/`.

use extract_cleanroom_assets::run_cleanroom_extraction;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));

    let default_extracted = workspace_root
        .parent()
        .map(|p| p.join("extracted_targets"))
        .unwrap_or_else(|| PathBuf::from("../extracted_targets"));
    let default_assets = workspace_root.join("assets");

    let extracted_dir = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or(default_extracted);
    let assets_dir = args.get(2).map(PathBuf::from).unwrap_or(default_assets);
    let metadata_only = args.iter().any(|a| a == "--metadata-only");

    match run_cleanroom_extraction(&extracted_dir, &assets_dir, !metadata_only) {
        Ok(manifest) => {
            println!(
                "[extract_izresembed] Extracted {} files from {} IZRESEMBED entries.",
                manifest.extracted_files.len(),
                manifest.izresembed_total_entries
            );
        }
        Err(err) => {
            eprintln!("[extract_izresembed] Error: {err}");
            std::process::exit(1);
        }
    }
}
