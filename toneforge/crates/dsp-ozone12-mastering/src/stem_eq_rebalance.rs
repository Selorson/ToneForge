//! iZotope Ozone 12 Unity-Identity Additive Stem EQ (`DSP::OzoneStemEQ` at `0x1857397B0`)
//! and Master Rebalance (`DSP::MasterRebalance` at `0x18572A8F0`).
//!
//! Mathematical Identity:
//! `Y[n] = X_delayed[n] + sum_{s in {Vocals, Drums, Bass, Other}} (H_s(z) - 1) * S_s[n]`
//! When all stem gains/EQ bands are at `0.0 dB`, `H_s(z) - 1 == 0`, guaranteeing 100%
//! bit-exact transparency (`Y[n] === X_delayed[n]`) free of neural separation artifacts!

use crate::chain_element::{OzoneDspModule, StereoBiquadF64};
use daw_core_graph::StereoFrame;

/// 4-Stem indices matching `ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum OzoneStemIndex {
    Vocals = 0,
    Drums = 1,
    Bass = 2,
    Other = 3,
}

/// Per-stem 3-band EQ configuration for `StemEqModule`.
#[derive(Clone, Copy, Debug)]
pub struct StemEqBandSet {
    pub low_shelf_hz: f32,
    pub low_shelf_db: f32,
    pub mid_peak_hz: f32,
    pub mid_peak_q: f32,
    pub mid_peak_db: f32,
    pub high_shelf_hz: f32,
    pub high_shelf_db: f32,
    pub mute: bool,
    pub solo: bool,
}

impl Default for StemEqBandSet {
    fn default() -> Self {
        Self {
            low_shelf_hz: 150.0,
            low_shelf_db: 0.0,
            mid_peak_hz: 1500.0,
            mid_peak_q: 1.0,
            mid_peak_db: 0.0,
            high_shelf_hz: 8000.0,
            high_shelf_db: 0.0,
            mute: false,
            solo: false,
        }
    }
}

impl StemEqBandSet {
    #[inline]
    pub fn is_flat(&self) -> bool {
        !self.mute
            && !self.solo
            && self.low_shelf_db.abs() < 1e-5
            && self.mid_peak_db.abs() < 1e-5
            && self.high_shelf_db.abs() < 1e-5
    }
}

/// Unity-Identity Additive 4-Stem EQ (`StemEqModule` / `OzoneStemEq`).
#[derive(Clone, Debug)]
pub struct StemEqModule {
    pub enabled: bool,
    pub stems: [StemEqBandSet; 4],
    filters: [[StereoBiquadF64; 3]; 4],
    sample_rate: f64,
}

pub type OzoneStemEq = StemEqModule;

impl Default for StemEqModule {
    fn default() -> Self {
        Self::new(48_000.0)
    }
}

impl StemEqModule {
    pub fn new(sample_rate: f64) -> Self {
        let sr = sample_rate.max(8000.0);
        let mut m = Self {
            enabled: true,
            stems: [StemEqBandSet::default(); 4],
            filters: [[StereoBiquadF64::default(); 3]; 4],
            sample_rate: sr,
        };
        m.update_coefficients();
        m
    }

    pub fn update_coefficients(&mut self) {
        for s in 0..4 {
            let cfg = &self.stems[s];
            self.filters[s][0].set_low_shelf(
                cfg.low_shelf_hz as f64,
                0.7071,
                cfg.low_shelf_db as f64,
                self.sample_rate,
            );
            self.filters[s][1].set_peaking_eq(
                cfg.mid_peak_hz as f64,
                cfg.mid_peak_q as f64,
                cfg.mid_peak_db as f64,
                self.sample_rate,
            );
            self.filters[s][2].set_high_shelf(
                cfg.high_shelf_hz as f64,
                0.7071,
                cfg.high_shelf_db as f64,
                self.sample_rate,
            );
        }
    }

    pub fn reset_internal(&mut self) {
        for s in 0..4 {
            for bq in &mut self.filters[s] {
                bq.reset();
            }
        }
    }

    /// Processes a single stereo sample using Unity-Identity Additive stem deltas:
    /// `y = x_delayed + sum_s (H_s(s_hat) - s_hat)`.
    #[inline]
    pub fn process_with_stems(
        &mut self,
        x_delayed: StereoFrame,
        separated_stems: &[StereoFrame; 4],
    ) -> StereoFrame {
        if !self.enabled {
            return x_delayed;
        }
        let any_solo = self.stems.iter().any(|s| s.solo);
        if !any_solo && self.stems.iter().all(|s| s.is_flat()) {
            return x_delayed;
        }

        let mut out_l = x_delayed[0] as f64;
        let mut out_r = x_delayed[1] as f64;

        for s in 0..4 {
            let cfg = &self.stems[s];
            let sl = separated_stems[s][0] as f64;
            let sr = separated_stems[s][1] as f64;

            if cfg.mute || (any_solo && !cfg.solo) {
                out_l -= sl;
                out_r -= sr;
                continue;
            }

            if cfg.is_flat() {
                continue;
            }

            let mut fl = sl;
            let mut fr = sr;
            for bq in &mut self.filters[s] {
                let (nl, nr) = bq.tick(fl, fr);
                fl = nl;
                fr = nr;
            }
            out_l += fl - sl;
            out_r += fr - sr;
        }

        [out_l as f32, out_r as f32]
    }
}

impl OzoneDspModule for StemEqModule {
    fn name(&self) -> &'static str {
        "iZotope Ozone 12 Stem EQ"
    }

    fn reset(&mut self, sample_rate: f64, _max_block_size: usize) {
        self.sample_rate = sample_rate.max(8000.0);
        self.update_coefficients();
        self.reset_internal();
    }

    fn process_block(&mut self, _left: &mut [f32], _right: &mut [f32]) {}
}

/// Unity-Identity Additive 4-Stem Master Rebalance (`MasterRebalanceModule` / `OzoneMasterRebalance`).
#[derive(Clone, Debug)]
pub struct MasterRebalanceModule {
    pub enabled: bool,
    /// Per-stem gain in dB (`[-12.0, +12.0]` dB) for `[Vocals, Drums, Bass, Other]`.
    pub stem_gains_db: [f32; 4],
}

pub type OzoneMasterRebalance = MasterRebalanceModule;

impl Default for MasterRebalanceModule {
    fn default() -> Self {
        Self {
            enabled: true,
            stem_gains_db: [0.0; 4],
        }
    }
}

impl MasterRebalanceModule {
    #[inline]
    pub fn process_with_stems(
        &self,
        x_delayed: StereoFrame,
        separated_stems: &[StereoFrame; 4],
    ) -> StereoFrame {
        if !self.enabled || self.stem_gains_db.iter().all(|g| g.abs() < 1e-5) {
            return x_delayed;
        }
        let mut l = x_delayed[0];
        let mut r = x_delayed[1];
        for s in 0..4 {
            let g_db = self.stem_gains_db[s].clamp(-24.0, 12.0);
            if g_db.abs() >= 1e-5 {
                let delta_lin = 10.0f32.powf(g_db / 20.0) - 1.0;
                l += delta_lin * separated_stems[s][0];
                r += delta_lin * separated_stems[s][1];
            }
        }
        [l, r]
    }
}
