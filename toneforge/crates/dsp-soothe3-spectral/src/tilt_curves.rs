//! `tilt_curves`: Frequency Tilt Curves around `500 Hz` (low) and `2000 Hz` (high)
//! modulating `Detail`, `Attack`, and `Release` across all `2,049` STFT bins.
//!
//! Mathematical formula (`02_proq4_and_soothe3_dsp.md` Section 2.3.B Step 2):
//! - `S_low(f) = 1.0 / (1.0 + (f / 500.0)^2)`
//! - `S_high(f) = (f / 2000.0)^2 / (1.0 + (f / 2000.0)^2)`
//! - `T_P(f) = (T_{P,low})^{S_low(f)} * (T_{P,high})^{S_high(f)}`

use serde::{Deserialize, Serialize};

pub const TILT_LOW_PIVOT_HZ: f32 = 500.0;
pub const TILT_HIGH_PIVOT_HZ: f32 = 2000.0;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TiltPair {
    pub low_mult: f32,  // 0.1 ..= 10.0 (default 1.0, acts below 500 Hz)
    pub high_mult: f32, // 0.1 ..= 10.0 (default 1.0, acts above 2000 Hz)
}

impl Default for TiltPair {
    fn default() -> Self {
        Self {
            low_mult: 1.0,
            high_mult: 1.0,
        }
    }
}

impl TiltPair {
    pub fn new(low_mult: f32, high_mult: f32) -> Self {
        Self {
            low_mult: low_mult.clamp(0.1, 10.0),
            high_mult: high_mult.clamp(0.1, 10.0),
        }
    }

    #[inline]
    pub fn norm_to_mult(norm: f32) -> f32 {
        // Maps [0.0, 1.0] logarithmically to [0.1, 10.0] with 0.5 -> 1.0
        10.0f32.powf(2.0 * norm.clamp(0.0, 1.0) - 1.0)
    }

    #[inline]
    pub fn mult_to_norm(mult: f32) -> f32 {
        ((mult.clamp(0.1, 10.0).log10() + 1.0) * 0.5).clamp(0.0, 1.0)
    }

    /// Evaluates the smooth frequency tilt multiplier `T_P(f_k)` at `freq_hz`.
    #[inline]
    pub fn evaluate(&self, freq_hz: f32) -> f32 {
        let f = freq_hz.max(0.0);
        let r_low = f / TILT_LOW_PIVOT_HZ;
        let s_low = 1.0 / (1.0 + r_low * r_low);
        let r_high = f / TILT_HIGH_PIVOT_HZ;
        let r_high_sq = r_high * r_high;
        let s_high = r_high_sq / (1.0 + r_high_sq);
        self.low_mult.clamp(0.1, 10.0).powf(s_low) * self.high_mult.clamp(0.1, 10.0).powf(s_high)
    }
}

/// Computes frequency-tilted `(tau_attack_sec, tau_release_sec)` for a bin at `freq_hz` (`update_speed`).
/// Automatically applies a smooth sub-`150 Hz` ballistic floor (`>= 25 ms` attack, `>= 80 ms` release
/// across `50..100 Hz`) to prevent half-cycle sub-bass waveform tearing while preserving fast
/// transient response at `3..10 kHz`.
#[inline]
pub fn compute_bin_ballistic_taus_sec(
    freq_hz: f32,
    attack_eff: f32,
    release_eff: f32,
) -> (f32, f32) {
    let f = freq_hz.max(0.0);
    let a_eff = attack_eff.clamp(0.0, 25.0);
    let r_eff = release_eff.clamp(0.0, 25.0);

    // Base user-controlled time constants (`~1.5..4 ms` fast HF attack, `~15..140 ms` release)
    let tau_att_base = (0.080 / (1.0 + 1.8 * a_eff)).clamp(0.0005, 0.250);
    let tau_rel_base = (0.400 / (1.0 + 1.2 * r_eff)).clamp(0.005, 1.000);

    // Smooth Hermite sub-165 Hz anti-tearing tilt (`0.0` for `f >= 165 Hz`, `>= 0.76` across `50..100 Hz`)
    let u_sub = ((165.0 - f) / 95.0).clamp(0.0, 1.0);
    let w_sub = u_sub * u_sub * (3.0 - 2.0 * u_sub);

    let tau_att_sec = (tau_att_base.max(0.036 * w_sub) + 0.012 * w_sub).clamp(0.0005, 0.350);
    let tau_rel_sec = (tau_rel_base.max(0.115 * w_sub) + 0.030 * w_sub).clamp(0.005, 1.200);
    (tau_att_sec, tau_rel_sec)
}

/// Asymmetric one-pole envelope integrator (`integrate_asym`) in dB (`gr_db <= 0.0`).
#[inline(always)]
pub fn integrate_asym(prev_db: f32, target_db: f32, attack_alpha: f32, release_alpha: f32) -> f32 {
    let alpha = if target_db < prev_db {
        attack_alpha
    } else {
        release_alpha
    };
    alpha * prev_db + (1.0 - alpha) * target_db
}

/// Populates per-bin effective `Detail`, `Attack` alpha, and `Release` alpha arrays
/// for all `num_bins` (`2,049`) bins of a 4096-point STFT at `base_sample_rate`.
#[allow(clippy::too_many_arguments)]
pub fn update_per_bin_tilt_tables<const BINS: usize>(
    fft_size: usize,
    hop_size: usize,
    base_sample_rate: f32,
    detail: f32,
    detail_tilt: &TiltPair,
    attack: f32,
    attack_tilt: &TiltPair,
    release: f32,
    release_tilt: &TiltPair,
    out_detail_per_bin: &mut [f32; BINS],
    out_attack_alpha_per_bin: &mut [f32; BINS],
    out_release_alpha_per_bin: &mut [f32; BINS],
) {
    let fs = base_sample_rate.max(8000.0);
    let dt_hop = (hop_size as f32) / fs;
    let hz_per_bin = fs / (fft_size as f32);

    for k in 0..BINS {
        let fk = (k as f32) * hz_per_bin;
        let d_eff = (detail * detail_tilt.evaluate(fk)).clamp(0.1, 25.0);
        let a_eff = (attack * attack_tilt.evaluate(fk)).clamp(0.0, 25.0);
        let r_eff = (release * release_tilt.evaluate(fk)).clamp(0.0, 25.0);

        out_detail_per_bin[k] = d_eff;

        let (tau_att_sec, tau_rel_sec) = compute_bin_ballistic_taus_sec(fk, a_eff, r_eff);
        out_attack_alpha_per_bin[k] = (-dt_hop / tau_att_sec).exp().clamp(0.0, 0.9985);
        out_release_alpha_per_bin[k] = (-dt_hop / tau_rel_sec).exp().clamp(0.0, 0.9995);
    }
}
