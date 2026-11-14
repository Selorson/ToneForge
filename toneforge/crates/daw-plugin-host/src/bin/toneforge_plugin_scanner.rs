//! `toneforge-plugin-scanner`: Out-of-process crash-isolated VST3 3.7 & CLAP 1.2 binary probe.
//!
//! Usage:
//! - `toneforge-plugin-scanner --probe <path_to_vst3_or_clap>`
//! - `toneforge-plugin-scanner --probe <path> --simulate-crash` (used by automated crash-isolation gate)

use daw_plugin_host::scanner::probe_single_binary_in_process;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--simulate-crash") {
        eprintln!("Simulated faulty third-party plugin access violation (0xC0000005)");
        std::process::exit(-1073741819); // 0xC0000005 STATUS_ACCESS_VIOLATION
    }

    let mut probe_target: Option<PathBuf> = None;
    let mut i = 1usize;
    while i < args.len() {
        if args[i] == "--probe" && i + 1 < args.len() {
            probe_target = Some(PathBuf::from(&args[i + 1]));
            i += 2;
        } else {
            i += 1;
        }
    }

    let Some(target) = probe_target else {
        eprintln!("Usage: toneforge-plugin-scanner --probe <plugin_path>");
        std::process::exit(2);
    };

    match probe_single_binary_in_process(&target) {
        Ok(descriptors) => match serde_json::to_string(&descriptors) {
            Ok(json) => {
                println!("{json}");
                std::process::exit(0);
            }
            Err(e) => {
                eprintln!("Serialization error: {e}");
                std::process::exit(3);
            }
        },
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    }
}
