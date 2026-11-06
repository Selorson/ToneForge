//! `daw-tauri-host` — ToneForge Desktop Host Crate.

pub mod ipc_commands;
pub mod shm_telemetry_server;
pub mod wgpu_compositor;

use daw_core_graph::mixer_dag::MixerEngine;
use ipc_commands::DawAppState;
use shm_telemetry_server::ShmTelemetryServer;

/// Bundled runtime session connecting the UI host, command dispatcher,
/// telemetry triple-buffer server, and real-time `MixerEngine`.
pub struct ToneForgeHostRuntime {
    pub app_state: DawAppState,
    pub telemetry_server: ShmTelemetryServer,
    pub mixer_engine: Box<MixerEngine>,
}

impl ToneForgeHostRuntime {
    pub fn new(sample_rate: f64, _max_block_size: usize) -> Self {
        let (engine, cmd_prod, telem_out) = MixerEngine::new(sample_rate, 128.0, 0);

        Self {
            app_state: DawAppState::new(cmd_prod),
            telemetry_server: ShmTelemetryServer::new(telem_out),
            mixer_engine: Box::new(engine),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc_commands::{
        ProQ4BandUpdateDto, cmd_route_mixer_track, cmd_set_transport, cmd_update_proq4_band,
    };

    #[test]
    fn test_host_ipc_to_rt_engine_roundtrip() {
        let mut runtime = ToneForgeHostRuntime::new(48000.0, 256);

        cmd_set_transport(&runtime.app_state, true, 140.0).unwrap();
        cmd_route_mixer_track(&runtime.app_state, 1, 2, 0.75, false).unwrap();
        cmd_update_proq4_band(
            &runtime.app_state,
            ProQ4BandUpdateDto {
                track_idx: 1,
                slot_idx: 0,
                band_idx: 0,
                freq_hz: 1000.0,
                gain_db: 3.0,
                q: 1.0,
                shape: 0,
                slope_db_oct: 12,
                dynamic_range_db: 0.0,
                spectral_enabled: false,
                spectral_density: 0.5,
            },
        )
        .unwrap();

        let mut master_out = [[0.0f32; 2]; 64];
        runtime.mixer_engine.process_audio_block(&mut master_out);

        let telem = runtime.telemetry_server.poll_latest_frame();
        assert_eq!(telem.header.frame_seq, 1);
        assert!((telem.header.bpm - 140.0).abs() < 1e-4);
    }
}
