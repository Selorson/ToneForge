//! CLI Entrypoint for ToneForge Clean-Room Asset Extractor (`tools/extract_cleanroom_assets`).

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
                "[ToneForge Clean-Room Extractor] Verified {} IZRESEMBED entries ({} .onnx, {} .npz, {} param tables), {} multi-rate 12AX7 LUTs (48k ref={} Hz), {} GRU-512 stem .npz models, {} ONNX tensor signatures, {} NPZ tensor archives, extracted {} files.",
                manifest.izresembed_total_entries,
                manifest.onnx_models_count,
                manifest.npz_models_count,
                manifest.param_lists_count,
                manifest.multi_rate_lut_count,
                manifest.triode_lut_sample_rate,
                manifest.gru_stem_npz_count,
                manifest.onnx_model_signatures.len(),
                manifest.npz_archive_signatures.len(),
                manifest.extracted_files.len()
            );
        }
        Err(err) => {
            eprintln!("[ToneForge Clean-Room Extractor] Error: {err}");
            std::process::exit(1);
        }
    }
}
