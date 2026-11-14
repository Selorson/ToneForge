//! `daw-plugin-host`: Native Windows x86_64 **Steinberg VST3 3.7 COM** & **CLAP 1.2 C ABI**
//! Real-Time Plugin Host, Win32 `HWND` Floating GUI Manager, and Crash-Isolated Out-of-Process
//! Plugin Scanner for **ToneForge Studio v3**.
//!
//! Real-time audio thread invariants:
//! - `0` `Mutex` locks inside `process_slice`, `on_voice_event`, `set_parameter_normalized`
//! - `0` `HeapAlloc` / `0` `HeapFree` (`drop()`) inside `process_slice`
//! - Exact plugin latency reporting (`IAudioProcessor::getLatencySamples` & `clap_plugin_latency::get`)
//!   wired directly into `MixerEngine`'s 5-Point Automatic Plugin Delay Compensation (`APDC`).

pub mod clap_abi;
pub mod clap_host;
pub mod hwnd_window;
pub mod scanner;
pub mod vst3_abi;
pub mod vst3_host;

pub use clap_host::{
    ClapLiveEditor,
    ClapDescriptorSummary, ClapHostedPluginNode, HostedPluginParamInfo,
    enumerate_clap_plugins_in_library,
};
pub use hwnd_window::NativePluginWindow;
pub use scanner::{
    PluginBinaryFormat, PluginQuarantineEntry, PluginScanReport, ScannedPluginDescriptor,
    collect_candidate_plugin_paths, default_plugin_scan_directories, locate_scanner_executable,
    probe_plugin_isolated, probe_single_binary_in_process, scan_plugin_directories_isolated,
};
pub use vst3_host::{
    Vst3EditorSession, Vst3LiveEditor,
    Vst3DescriptorSummary, Vst3HostedPluginNode, enumerate_vst3_plugins_in_library,
    resolve_vst3_binary_path,
};
