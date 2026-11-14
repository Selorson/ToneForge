//! iZotope Ozone 12 Multi-Band 12AX7 Koren DK-Method Exciter (`DSP::OzoneExciter` `0x1856A9B78`,
//! `DSP::Shared::KState<3, 2, 2, DSP::Shared::KorenFunction>` `0x181DB8BA0`, & `DualTriode` `0x181DF0580`).
//!
//! Features:
//! - Byte-exact binary loaders for **both** single-stage `Triode` (`exciter-triode-<rate:.6>.dat`,
//!   `2,074,532` B, magic `0x8BAD43FC`, schema `v4`) and cascaded two-stage `DualTriode`
//!   (`exciter-dualtriode-<rate:.6>.dat`, `4,149,044` B, magic `0x9DFBCC23`, schema `v4`).
//! - Multi-rate `.dat` auto-discovery & preloading (`1x`, `2x`, `4x` of `host_sr` up to `768,000 Hz`)
//!   during `new` / `prepare` (zero file I/O and zero heap allocation on the RT audio thread).
//! - Exact Holters-Zölzer / Yeh trapezoidal bilinear re-discretization from continuous-time circuit
//!   matrices `(A_c, B_c, C_c, D_x, D_u, D_i)` -> `(M_inv, P, Q, Q_prev, G_d, K_imp, A_d, B_d, C_d)`
//!   for arbitrary sample rates, combined with Norman Koren 12AX7 (`mu=100, kp=600, kvb=300, kg=1060, Ex=1.4`)
//!   2-step analytical Jacobian Newton-Raphson refinement on top of the `360 x 360` 2D LUT initial guess.
//! - True zero-allocation `2x` / `4x` `PolyphaseHalfBandOversampler` stepping the nonlinear tube /
//!   waveshaper stage on every oversampled sub-sample at `effective_sr = host_sr * os_factor`.

use crate::chain_element::{
    DEFAULT_CROSSOVER_FREQS_HZ, LinkwitzRiley4BandCrossover, OZONE_MULTIBAND_COUNT, OzoneDspModule,
    StereoBiquadF64,
};
use crate::oversampler::{OversampleFactor, OversamplerFilterMode, PolyphaseHalfBandOversampler};
use daw_core_graph::{ProcessContext, StereoFrame, UnifiedDspNode, VisualTelemetryFrame};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

pub const IZOTOPE_DK_LUT_MAGIC: u32 = 0x8BAD_43FC;
pub const IZOTOPE_DUALTRIODE_LUT_MAGIC: u32 = 0x9DFB_CC23;
pub const IZOTOPE_DK_LUT_VERSION: u32 = 4;
pub const IZOTOPE_KSTATE_BLOCK_SIZE: usize = 0x1FA740; // 2,074,432 bytes
pub const IZOTOPE_TRIODE_DAT_SIZE: usize = 0x1FA7A4; // 2,074,532 bytes
pub const IZOTOPE_DUALTRIODE_DAT_SIZE: usize = 0x3F4F34; // 4,149,044 bytes

/// Default candidate paths for `exciter-triode-48000.000000.dat` in the ToneForge workspace.
pub const DEFAULT_TRIODE_LUT_PATHS: [&str; 5] = [
    r"f:\FL studio\toneforge\assets\exciter_luts\exciter-triode-48000.000000.dat",
    r"f:\FL studio\toneforge\assets\ozone12\luts\exciter-triode-48000.000000.dat",
    r"assets/exciter_luts/exciter-triode-48000.000000.dat",
    r"assets/ozone12/luts/exciter-triode-48000.000000.dat",
    r"f:\FL studio\extracted_targets\ozone12\exciterCache_component\ExciterCache\Exciter Cache\exciter-triode-48000.000000.dat",
];

/// Standard 19 sample rates cached by Ozone 12 (`8,000 Hz` .. `768,000 Hz`).
pub const STANDARD_EXCITER_LUT_RATES: [u32; 19] = [
    8_000, 11_025, 16_000, 22_050, 32_000, 44_100, 48_000, 64_000, 88_200, 96_000, 128_000,
    176_400, 192_000, 256_000, 352_800, 384_000, 512_000, 705_600, 768_000,
];

/// Norman Koren 12AX7 physical vacuum tube constants (`DSP::Shared::KorenFunction` at `0x181DF4FC0`).
pub const KOREN_12AX7_MU: f64 = 100.0;
pub const KOREN_12AX7_KG: f64 = 1060.0;
pub const KOREN_12AX7_KP: f64 = 600.0;
pub const KOREN_12AX7_KVB: f64 = 300.0;
pub const KOREN_12AX7_EXPONENT: f64 = 1.4;
pub const KOREN_12AX7_RGI_OHMS: f64 = 2000.0;

#[inline]
fn softplus_stable(x: f64) -> (f64, f64) {
    if x > 35.0 {
        let e_neg = (-x).exp();
        (x + e_neg, 1.0 / (1.0 + e_neg))
    } else if x < -35.0 {
        let e_pos = x.exp();
        (e_pos, e_pos)
    } else {
        let e_pos = x.exp();
        ((1.0 + e_pos).ln(), e_pos / (1.0 + e_pos))
    }
}

/// Evaluates the continuous Norman Koren 12AX7 grid and plate currents `[I_g(V_gk), I_p(V_gk, V_pk)]` in Amperes.
#[inline]
pub fn koren_12ax7_currents(v_gk: f64, v_pk: f64) -> [f64; 2] {
    let (currents, _) = koren_12ax7_currents_and_jacobian(v_gk, v_pk);
    currents
}

/// Evaluates Norman Koren 12AX7 `[I_g, I_p]` and its exact `2x2` conductance Jacobian
/// `[[dI_g/dV_gk, dI_g/dV_pk], [dI_p/dV_gk, dI_p/dV_pk]]`.
#[inline]
pub fn koren_12ax7_currents_and_jacobian(v_gk: f64, v_pk: f64) -> ([f64; 2], [[f64; 2]; 2]) {
    let (i_g, dig_dvgk) = if v_gk > 0.0 {
        let e = (-2.0 * v_gk).exp();
        let ig = (v_gk / KOREN_12AX7_RGI_OHMS) * (1.0 - e);
        let dig = (1.0 - e + 2.0 * v_gk * e) / KOREN_12AX7_RGI_OHMS;
        (ig, dig)
    } else {
        (0.0, 0.0)
    };

    if v_pk <= 0.0 {
        return ([i_g, 0.0], [[dig_dvgk, 0.0], [0.0, 0.0]]);
    }

    let d_sq = KOREN_12AX7_KVB + v_pk * v_pk;
    let d = d_sq.sqrt();
    let arg = KOREN_12AX7_KP * (1.0 / KOREN_12AX7_MU + v_gk / d);
    let (sp, sig) = softplus_stable(arg);
    let e1 = (v_pk / KOREN_12AX7_KP) * sp;

    if e1 <= 1.0e-18 {
        return ([i_g, 0.0], [[dig_dvgk, 0.0], [0.0, 0.0]]);
    }

    let e1_pow_m1 = e1.powf(KOREN_12AX7_EXPONENT - 1.0);
    let i_p = (2.0 / KOREN_12AX7_KG) * e1 * e1_pow_m1;
    let dip_de1 = (2.0 * KOREN_12AX7_EXPONENT / KOREN_12AX7_KG) * e1_pow_m1;

    let de1_dvgk = (v_pk * sig) / d;
    let de1_dvpk = (sp / KOREN_12AX7_KP) - (v_pk * v_pk * v_gk * sig) / (d * d_sq);

    (
        [i_g, i_p],
        [
            [dig_dvgk, 0.0],
            [dip_de1 * de1_dvgk, dip_de1 * de1_dvpk],
        ],
    )
}

/// Solves `R(i) = f_Koren(p + K_imp * i) - i = 0` via 2x2 analytical Newton-Raphson.
#[inline]
pub fn koren_12ax7_newton_solve(
    p: [f64; 2],
    k_imp: [[f64; 2]; 2],
    i_init: [f64; 2],
    max_iters: usize,
) -> [f64; 2] {
    let mut cur = i_init;
    for _ in 0..max_iters {
        let v_gk = p[0] + k_imp[0][0] * cur[0] + k_imp[0][1] * cur[1];
        let v_pk = p[1] + k_imp[1][0] * cur[0] + k_imp[1][1] * cur[1];
        let (f_val, jf) = koren_12ax7_currents_and_jacobian(v_gk, v_pk);
        let r0 = f_val[0] - cur[0];
        let r1 = f_val[1] - cur[1];
        if r0.abs() <= 1.0e-15 && r1.abs() <= 1.0e-15 {
            break;
        }
        // J_R = J_f * K_imp - I_2
        let j00 = jf[0][0] * k_imp[0][0] + jf[0][1] * k_imp[1][0] - 1.0;
        let j01 = jf[0][0] * k_imp[0][1] + jf[0][1] * k_imp[1][1];
        let j10 = jf[1][0] * k_imp[0][0] + jf[1][1] * k_imp[1][0];
        let j11 = jf[1][0] * k_imp[0][1] + jf[1][1] * k_imp[1][1] - 1.0;
        let det = j00 * j11 - j01 * j10;
        if det.abs() <= 1.0e-20 {
            break;
        }
        let inv_det = 1.0 / det;
        let di0 = -(j11 * r0 - j01 * r1) * inv_det;
        let di1 = -(-j10 * r0 + j00 * r1) * inv_det;
        cur[0] = (cur[0] + di0.clamp(-0.05, 0.05)).max(0.0);
        cur[1] = (cur[1] + di1.clamp(-0.05, 0.05)).max(0.0);
    }
    cur
}

/// Exact bilinear trapezoidal discretized DK-method matrices (`Mat[6..13]` in `KState<3,2,2>`).
#[derive(Clone, Copy, Debug)]
pub struct DiscretizedDkMatrices {
    pub sample_rate: f64,
    /// `Mat[6]`: `M_inv = (I_3 - (T/2) A_c)^-1` (`[3x3]`).
    pub m_inv: [[f64; 3]; 3],
    /// `Mat[7]`: `P = D_d = D_x * A_d` (`[2x3]`).
    pub d_d: [[f64; 3]; 2],
    /// `Mat[8]`: `Q = E_d = D_x * B_d + E_u` (`[2x2]`).
    pub e_d: [[f64; 2]; 2],
    /// `Mat[9]`: `Q_prev = F_d = D_x * B_d` (`[2x2]`).
    pub f_d: [[f64; 2]; 2],
    /// `Mat[10]`: `G_d = D_x * C_d` (`[2x2]`).
    pub g_d: [[f64; 2]; 2],
    /// Implicit K-matrix `K_imp = D_x * C_d + F_i` (`v_nl = p + K_imp * i_nl`).
    pub k_imp: [[f64; 2]; 2],
    /// `Mat[11]`: `A_d = M_inv * (I_3 + (T/2) A_c)` (`[3x3]`).
    pub a_d: [[f64; 3]; 3],
    /// `Mat[12]`: `B_d = (T/2) * M_inv * B_c` (`[3x2]`).
    pub b_d: [[f64; 2]; 3],
    /// `Mat[13]`: `C_d = (T/2) * M_inv * C_c` (`[3x2]`).
    pub c_d: [[f64; 2]; 3],
}

#[inline]
fn invert_3x3(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    let c01 = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]);
    let c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];

    let c10 = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]);
    let c11 = m[0][0] * m[2][2] - m[0][2] * m[2][0];
    let c12 = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]);

    let c20 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
    let c21 = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]);
    let c22 = m[0][0] * m[1][1] - m[0][1] * m[1][0];

    let det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
    let inv_det = if det.abs() > 1.0e-30 { 1.0 / det } else { 1.0 };

    [
        [c00 * inv_det, c10 * inv_det, c20 * inv_det],
        [c01 * inv_det, c11 * inv_det, c21 * inv_det],
        [c02 * inv_det, c12 * inv_det, c22 * inv_det],
    ]
}

/// Exact Holters-Zölzer / Yeh trapezoidal bilinear discretization (`h = 1 / (2 * target_sr)`).
/// Matches `Mat[6..13]` in `exciter-triode-*.dat` and `exciter-dualtriode-*.dat` to `< 1e-13`.
pub fn rediscretize_dk_matrices(
    a_c: [[f64; 3]; 3],
    b_c: [[f64; 2]; 3],
    c_c: [[f64; 2]; 3],
    d_x: [[f64; 3]; 2],
    e_u: [[f64; 2]; 2],
    f_i: [[f64; 2]; 2],
    target_sr: f64,
) -> DiscretizedDkMatrices {
    let sr = target_sr.max(1000.0);
    let h = 0.5 / sr;

    let mut lhs = [[0.0_f64; 3]; 3];
    let mut rhs = [[0.0_f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            let eye = if r == c { 1.0 } else { 0.0 };
            lhs[r][c] = eye - h * a_c[r][c];
            rhs[r][c] = eye + h * a_c[r][c];
        }
    }

    let inv_lhs = invert_3x3(lhs);
    let mut m_inv = [[0.0_f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            // `Mat[6]` in `KState<3,2,2>` stores `((2 * fs) I_3 - A_c)^-1 = h * (I_3 - h A_c)^-1`
            m_inv[r][c] = h * inv_lhs[r][c];
        }
    }

    let mut a_d = [[0.0_f64; 3]; 3];
    let mut b_d = [[0.0_f64; 2]; 3];
    let mut c_d = [[0.0_f64; 2]; 3];
    for r in 0..3 {
        for c in 0..3 {
            a_d[r][c] =
                inv_lhs[r][0] * rhs[0][c] + inv_lhs[r][1] * rhs[1][c] + inv_lhs[r][2] * rhs[2][c];
        }
        for c in 0..2 {
            b_d[r][c] = m_inv[r][0] * b_c[0][c] + m_inv[r][1] * b_c[1][c] + m_inv[r][2] * b_c[2][c];
            c_d[r][c] = m_inv[r][0] * c_c[0][c] + m_inv[r][1] * c_c[1][c] + m_inv[r][2] * c_c[2][c];
        }
    }

    let mut d_d = [[0.0_f64; 3]; 2];
    let mut f_d = [[0.0_f64; 2]; 2];
    let mut e_d = [[0.0_f64; 2]; 2];
    let mut g_d = [[0.0_f64; 2]; 2];
    let mut k_imp = [[0.0_f64; 2]; 2];

    for r in 0..2 {
        for c in 0..3 {
            d_d[r][c] = d_x[r][0] * a_d[0][c] + d_x[r][1] * a_d[1][c] + d_x[r][2] * a_d[2][c];
        }
        for c in 0..2 {
            f_d[r][c] = d_x[r][0] * b_d[0][c] + d_x[r][1] * b_d[1][c] + d_x[r][2] * b_d[2][c];
            e_d[r][c] = f_d[r][c] + e_u[r][c];
            g_d[r][c] = d_x[r][0] * c_d[0][c] + d_x[r][1] * c_d[1][c] + d_x[r][2] * c_d[2][c];
            k_imp[r][c] = g_d[r][c] + f_i[r][c];
        }
    }

    DiscretizedDkMatrices {
        sample_rate: sr,
        m_inv,
        d_d,
        e_d,
        f_d,
        g_d,
        k_imp,
        a_d,
        b_d,
        c_d,
    }
}

/// Decoded `2,074,432`-byte `KState<3, 2, 2, KorenFunction>` block (`0x181DB8100`).
#[derive(Clone, Debug)]
pub struct DkKStateBlock {
    pub sample_rate: f64,
    pub lut_base_sample_rate: f64,
    pub a_c: [[f64; 3]; 3],
    pub b_c: [[f64; 2]; 3],
    pub c_c: [[f64; 2]; 3],
    pub d_x: [[f64; 3]; 2],
    pub e_u: [[f64; 2]; 2],
    pub f_i: [[f64; 2]; 2],
    pub m_inv: [[f64; 3]; 3],
    pub d_d: [[f64; 3]; 2],
    pub e_d: [[f64; 2]; 2],
    pub f_d: [[f64; 2]; 2],
    pub g_d: [[f64; 2]; 2],
    pub k_imp: [[f64; 2]; 2],
    pub lut_base_k_imp: [[f64; 2]; 2],
    pub a_d: [[f64; 3]; 3],
    pub b_d: [[f64; 2]; 3],
    pub c_d: [[f64; 2]; 3],
    pub bounds: [f64; 4],
    pub grid_dims: [u32; 2],
    pub total_points: u64,
    pub lut: Arc<Vec<[[f64; 2]; 360]>>,
    pub is_synthetic: bool,
}

impl DkKStateBlock {
    /// Parses a `2,074,432`-byte `KState<3,2,2,KorenFunction>` block starting at `base` inside `bytes`.
    pub fn from_slice_at(bytes: &[u8], base: usize) -> Result<Self, &'static str> {
        if bytes.len() < base + IZOTOPE_KSTATE_BLOCK_SIZE {
            return Err("KState<3,2,2> slice shorter than 2,074,432 bytes");
        }
        let read_u32 = |off: usize| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        let read_u64 = |off: usize| u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        let read_f64 = |off: usize| f64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());

        let read_mat_3x3 = |off: usize| -> [[f64; 3]; 3] {
            let mut m = [[0.0; 3]; 3];
            for r in 0..3 {
                for c in 0..3 {
                    m[r][c] = read_f64(off + 8 + (r * 3 + c) * 8);
                }
            }
            m
        };
        let read_mat_3x2 = |off: usize| -> [[f64; 2]; 3] {
            let mut m = [[0.0; 2]; 3];
            for r in 0..3 {
                for c in 0..2 {
                    m[r][c] = read_f64(off + 8 + (r * 2 + c) * 8);
                }
            }
            m
        };
        let read_mat_2x3 = |off: usize| -> [[f64; 3]; 2] {
            let mut m = [[0.0; 3]; 2];
            for r in 0..2 {
                for c in 0..3 {
                    m[r][c] = read_f64(off + 8 + (r * 3 + c) * 8);
                }
            }
            m
        };
        let read_mat_2x2 = |off: usize| -> [[f64; 2]; 2] {
            [
                [read_f64(off + 8), read_f64(off + 16)],
                [read_f64(off + 24), read_f64(off + 32)],
            ]
        };

        let sample_rate = read_f64(base);
        let a_c = read_mat_3x3(base + 0x008);
        let b_c = read_mat_3x2(base + 0x058);
        let c_c = read_mat_3x2(base + 0x090);
        let d_x = read_mat_2x3(base + 0x0C8);
        let e_u = read_mat_2x2(base + 0x100);
        let f_i = read_mat_2x2(base + 0x128);
        let m_inv = read_mat_3x3(base + 0x150);
        let d_d = read_mat_2x3(base + 0x1A0);
        let e_d = read_mat_2x2(base + 0x1D8);
        let f_d = read_mat_2x2(base + 0x200);
        let g_d = read_mat_2x2(base + 0x228);
        let a_d = read_mat_3x3(base + 0x250);
        let b_d = read_mat_3x2(base + 0x2A0);
        let c_d = read_mat_3x2(base + 0x2D8);

        let k_imp = [
            [g_d[0][0] + f_i[0][0], g_d[0][1] + f_i[0][1]],
            [g_d[1][0] + f_i[1][0], g_d[1][1] + f_i[1][1]],
        ];

        let bounds = [
            read_f64(base + 0x310),
            read_f64(base + 0x318),
            read_f64(base + 0x320),
            read_f64(base + 0x328),
        ];
        let grid_dims = [read_u32(base + 0x330), read_u32(base + 0x334)];
        let total_points = read_u64(base + 0x338);
        if grid_dims != [360, 360] {
            return Err("Unexpected KState LUT grid dimensions (expected 360x360)");
        }

        let mut lut_grid = vec![[[0.0f64; 2]; 360]; 360];
        let mut cursor = base + 0x340;
        for row in &mut lut_grid {
            for cell in row.iter_mut() {
                cell[0] = read_f64(cursor);
                cell[1] = read_f64(cursor + 8);
                cursor += 16;
            }
        }

        Ok(Self {
            sample_rate,
            lut_base_sample_rate: sample_rate,
            a_c,
            b_c,
            c_c,
            d_x,
            e_u,
            f_i,
            m_inv,
            d_d,
            e_d,
            f_d,
            g_d,
            k_imp,
            lut_base_k_imp: k_imp,
            a_d,
            b_d,
            c_d,
            bounds,
            grid_dims,
            total_points,
            lut: Arc::new(lut_grid),
            is_synthetic: false,
        })
    }

    /// Returns a clone with exact trapezoidal bilinear re-discretized matrices at `target_sr`,
    /// sharing the `Arc` 2D LUT buffer without copying `2.07 MB`.
    pub fn rediscretized(&self, target_sr: f64) -> Self {
        let disc = rediscretize_dk_matrices(
            self.a_c,
            self.b_c,
            self.c_c,
            self.d_x,
            self.e_u,
            self.f_i,
            target_sr,
        );
        Self {
            sample_rate: disc.sample_rate,
            lut_base_sample_rate: self.lut_base_sample_rate,
            a_c: self.a_c,
            b_c: self.b_c,
            c_c: self.c_c,
            d_x: self.d_x,
            e_u: self.e_u,
            f_i: self.f_i,
            m_inv: disc.m_inv,
            d_d: disc.d_d,
            e_d: disc.e_d,
            f_d: disc.f_d,
            g_d: disc.g_d,
            k_imp: disc.k_imp,
            lut_base_k_imp: self.lut_base_k_imp,
            a_d: disc.a_d,
            b_d: disc.b_d,
            c_d: disc.c_d,
            bounds: self.bounds,
            grid_dims: self.grid_dims,
            total_points: self.total_points,
            lut: Arc::clone(&self.lut),
            is_synthetic: self.is_synthetic,
        }
    }

    #[inline(always)]
    pub fn sample_bilinear(&self, p1: f64, p2: f64) -> [f64; 2] {
        let [p1_min, p1_max, p2_min, p2_max] = self.bounds;
        let u = 359.0 * ((p1 - p1_min) / (p1_max - p1_min)).clamp(0.0, 1.0);
        let v = 359.0 * ((p2 - p2_min) / (p2_max - p2_min)).clamp(0.0, 1.0);

        let i0 = (u.floor() as usize).min(358);
        let j0 = (v.floor() as usize).min(358);
        let fu = u - (i0 as f64);
        let fv = v - (j0 as f64);

        let g00 = self.lut[i0][j0];
        let g10 = self.lut[i0 + 1][j0];
        let g01 = self.lut[i0][j0 + 1];
        let g11 = self.lut[i0 + 1][j0 + 1];

        let mut out = [0.0; 2];
        for c in 0..2 {
            let top = g00[c] * (1.0 - fu) + g10[c] * fu;
            let bot = g01[c] * (1.0 - fu) + g11[c] * fu;
            out[c] = top * (1.0 - fv) + bot * fv;
        }
        out
    }

    /// Evaluates nonlinear tube currents `[I_g, I_p]` at transformed port `p`:
    /// starts from the `360 x 360` 2D bilinear LUT initial guess and, when re-discretized to an
    /// arbitrary rate or running on a synthetic table, applies 2-step analytical Koren 12AX7
    /// Jacobian Newton-Raphson refinement (`0` heap allocations).
    #[inline(always)]
    pub fn solve_currents(&self, p: [f64; 2]) -> [f64; 2] {
        let i_init = self.sample_bilinear(p[0], p[1]);
        if self.is_synthetic {
            return koren_12ax7_newton_solve(p, self.k_imp, i_init, 2);
        }
        if (self.sample_rate - self.lut_base_sample_rate).abs() <= 0.5 {
            return i_init;
        }
        // Arbitrary non-cached sample rate: solve R(i) = LUT_base(p + (K_new - K_base)*i) - i = 0
        // using the analytical Norman Koren 12AX7 Jacobian J_f(v) for quadratic convergence while
        // preserving exact DC equilibrium `< 1e-15` across all sample rates!
        let dk00 = self.k_imp[0][0] - self.lut_base_k_imp[0][0];
        let dk01 = self.k_imp[0][1] - self.lut_base_k_imp[0][1];
        let dk10 = self.k_imp[1][0] - self.lut_base_k_imp[1][0];
        let dk11 = self.k_imp[1][1] - self.lut_base_k_imp[1][1];
        let mut cur = i_init;
        for _ in 0..2 {
            let p_eq0 = p[0] + dk00 * cur[0] + dk01 * cur[1];
            let p_eq1 = p[1] + dk10 * cur[0] + dk11 * cur[1];
            let i_target = self.sample_bilinear(p_eq0, p_eq1);
            let r0 = i_target[0] - cur[0];
            let r1 = i_target[1] - cur[1];
            if r0.abs() <= 1.0e-15 && r1.abs() <= 1.0e-15 {
                break;
            }
            let v_gk = p[0] + self.k_imp[0][0] * cur[0] + self.k_imp[0][1] * cur[1];
            let v_pk = p[1] + self.k_imp[1][0] * cur[0] + self.k_imp[1][1] * cur[1];
            let (_, jf) = koren_12ax7_currents_and_jacobian(v_gk, v_pk);
            let j00 = jf[0][0] * dk00 + jf[0][1] * dk10 - 1.0;
            let j01 = jf[0][0] * dk01 + jf[0][1] * dk11;
            let j10 = jf[1][0] * dk00 + jf[1][1] * dk10;
            let j11 = jf[1][0] * dk01 + jf[1][1] * dk11 - 1.0;
            let det = j00 * j11 - j01 * j10;
            if det.abs() <= 1.0e-20 {
                break;
            }
            let inv_det = 1.0 / det;
            let di0 = -(j11 * r0 - j01 * r1) * inv_det;
            let di1 = -(-j10 * r0 + j00 * r1) * inv_det;
            cur[0] = (cur[0] + di0.clamp(-0.05, 0.05)).max(0.0);
            cur[1] = (cur[1] + di1.clamp(-0.05, 0.05)).max(0.0);
        }
        cur
    }
}

/// Decoded binary representation of `exciter-triode-*.dat` (`KState<3, 2, 2, KorenFunction>`).
#[derive(Clone, Debug)]
pub struct DkTriodeLut {
    pub magic: u32,                     // 0x0000: 0x8BAD43FC
    pub version: u32,                   // 0x0004: 4
    pub dc_offset: f64,                 // 0x0008: -0.04588765735184097 (output normalization scale)
    pub circuit_mode: u32,              // 0x0010: 0 = Triode
    pub x_dc: [f64; 3],                 // 0x001C: [6.68e-14, 196.282586, 1.555761]
    pub u_dc: [f64; 2],                 // 0x003C: [0.0, 300.0]
    pub i_nl_dc: [f64; 2],              // 0x0054: [9.06e-25, 0.001037174]
    pub sample_rate: f64,               // 0x0064: operating sample rate
    pub a_c: [[f64; 3]; 3],             // Mat[0]
    pub b_c: [[f64; 2]; 3],             // Mat[1]
    pub c_c: [[f64; 2]; 3],             // Mat[2]
    pub d_x: [[f64; 3]; 2],             // Mat[3]
    pub e_u: [[f64; 2]; 2],             // Mat[4]
    pub f_i: [[f64; 2]; 2],             // Mat[5]
    pub m_inv: [[f64; 3]; 3],           // Mat[6] (0x01B4)
    pub d_d: [[f64; 3]; 2],             // Mat[7]
    pub e_d: [[f64; 2]; 2],             // Mat[8]
    pub f_d: [[f64; 2]; 2],             // Mat[9]
    pub g_d: [[f64; 2]; 2],             // Mat[10]
    pub a_d: [[f64; 3]; 3],             // Mat[11]
    pub b_d: [[f64; 2]; 3],             // Mat[12]
    pub c_d: [[f64; 2]; 3],             // Mat[13]
    pub bounds: [f64; 4],               // 0x0374: [-600.0, 300.0, 240.0, 303.0]
    pub grid_dims: [u32; 2],            // 0x0394: [360, 360]
    pub lut: Arc<Vec<[[f64; 2]; 360]>>, // 0x03A4..0x1FA7A4
    pub kstate: DkKStateBlock,
}

impl DkTriodeLut {
    /// Decodes `exciter-triode-*.dat` (`2,074,532` bytes = `0x1FA7A4`) byte-for-byte.
    pub fn from_dat_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() < IZOTOPE_TRIODE_DAT_SIZE {
            return Err("File .dat nho hon 2,074,532 bytes (0x1FA7A4)");
        }
        let read_u32 = |off: usize| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        let read_f64 = |off: usize| f64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());

        let magic = read_u32(0x0000);
        let version = read_u32(0x0004);
        if magic != IZOTOPE_DK_LUT_MAGIC || version != IZOTOPE_DK_LUT_VERSION {
            return Err("Sai magic 0x8BAD43FC hoac version != 4");
        }

        let kstate = DkKStateBlock::from_slice_at(bytes, 0x0064)?;
        Ok(Self {
            magic,
            version,
            dc_offset: read_f64(0x0008),
            circuit_mode: read_u32(0x0010),
            x_dc: [read_f64(0x001C), read_f64(0x0024), read_f64(0x002C)],
            u_dc: [read_f64(0x003C), read_f64(0x0044)],
            i_nl_dc: [read_f64(0x0054), read_f64(0x005C)],
            sample_rate: kstate.sample_rate,
            a_c: kstate.a_c,
            b_c: kstate.b_c,
            c_c: kstate.c_c,
            d_x: kstate.d_x,
            e_u: kstate.e_u,
            f_i: kstate.f_i,
            m_inv: kstate.m_inv,
            d_d: kstate.d_d,
            e_d: kstate.e_d,
            f_d: kstate.f_d,
            g_d: kstate.g_d,
            a_d: kstate.a_d,
            b_d: kstate.b_d,
            c_d: kstate.c_d,
            bounds: kstate.bounds,
            grid_dims: kstate.grid_dims,
            lut: Arc::clone(&kstate.lut),
            kstate,
        })
    }

    /// Loads a `.dat` lookup table directly from the given filesystem path.
    pub fn from_dat_file(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
        Self::from_dat_bytes(&bytes).map_err(|e| e.to_string())
    }

    /// Exact trapezoidal bilinear re-discretization to `target_sr` (`< 1e-13` match to `.dat` matrices).
    pub fn rediscretize_for_rate(&self, target_sr: f64) -> Self {
        let kstate = self.kstate.rediscretized(target_sr);
        Self {
            magic: self.magic,
            version: self.version,
            dc_offset: self.dc_offset,
            circuit_mode: self.circuit_mode,
            x_dc: self.x_dc,
            u_dc: self.u_dc,
            i_nl_dc: self.i_nl_dc,
            sample_rate: kstate.sample_rate,
            a_c: kstate.a_c,
            b_c: kstate.b_c,
            c_c: kstate.c_c,
            d_x: kstate.d_x,
            e_u: kstate.e_u,
            f_i: kstate.f_i,
            m_inv: kstate.m_inv,
            d_d: kstate.d_d,
            e_d: kstate.e_d,
            f_d: kstate.f_d,
            g_d: kstate.g_d,
            a_d: kstate.a_d,
            b_d: kstate.b_d,
            c_d: kstate.c_d,
            bounds: kstate.bounds,
            grid_dims: kstate.grid_dims,
            lut: Arc::clone(&kstate.lut),
            kstate,
        }
    }

    /// Loads the real `exciter-triode-48000.000000.dat` from standard ToneForge asset locations,
    /// falling back to the analytical Koren 12AX7 DK-Method table if running without assets on disk.
    pub fn load_default_48k() -> Self {
        load_triode_lut_cached(48_000.0).as_ref().clone()
    }

    /// Generates an analytical 12AX7 DK-Method table at `sample_rate` using exact trapezoidal
    /// discretization and 2x2 analytical Koren Newton-Raphson.
    pub fn synthetic_koren_12ax7_lut(sample_rate: f64) -> Self {
        let sr = sample_rate.max(8_000.0);
        let a_c = [
            [-146.43304130162707, 125.1564455569462, 0.0],
            [2352941.1764705884, -2352941.1764705884, 0.0],
            [0.0, 0.0, -26.666666666666664],
        ];
        let b_c = [
            [146.43304130162707, -125.1564455569462],
            [-2352941.1764705884, 2352941.1764705884],
            [0.0, 0.0],
        ];
        let c_c = [
            [12515644.55569462, 12515644.55569462],
            [164705882352.94122, -235294117647.05887],
            [40000.0, 40000.0],
        ];
        let d_x = [
            [-0.5882352941176471, -0.411764705882353, -1.0],
            [-0.5882352941176471, 0.5882352941176471, -1.0],
        ];
        let e_u = [
            [0.5882352941176471, 0.411764705882353],
            [0.5882352941176471, 0.411764705882353],
        ];
        let f_i = [
            [-41176.470588235294, -41176.470588235294],
            [-41176.470588235294, -41176.470588235294],
        ];

        let disc = rediscretize_dk_matrices(a_c, b_c, c_c, d_x, e_u, f_i, sr);
        let bounds = [-600.0, 300.0, 240.0, 303.0];
        let mut lut_grid = vec![[[0.0f64; 2]; 360]; 360];
        for i in 0..360 {
            let p1 = bounds[0] + (i as f64 / 359.0) * (bounds[1] - bounds[0]);
            let mut prev = [0.0_f64, 0.001_f64];
            for j in 0..360 {
                let p2 = bounds[2] + (j as f64 / 359.0) * (bounds[3] - bounds[2]);
                let solved = koren_12ax7_newton_solve([p1, p2], disc.k_imp, prev, 8);
                lut_grid[i][j] = solved;
                prev = solved;
            }
        }

        // Solve exact DC equilibrium for the analytical Koren 12AX7 model (`u_dc = [0.0, 300.0]`)
        // where R_cathode = 40000 / 26.666666666666664 = 1500 Ohm, R_plate = 100000 Ohm:
        let u_dc = [0.0_f64, 300.0_f64];
        let mut i_p_dc = 0.0007_f64;
        for _ in 0..24 {
            let v_k = 1500.0 * i_p_dc;
            let v_p = 300.0 - 100_000.0 * i_p_dc;
            let [_, ip_new] = koren_12ax7_currents(-v_k, v_p - v_k);
            i_p_dc = 0.5 * i_p_dc + 0.5 * ip_new;
        }
        let x_dc = [0.0, 300.0 - 100_000.0 * i_p_dc, 1500.0 * i_p_dc];
        let i_nl_dc = [0.0, i_p_dc];

        let lut_arc = Arc::new(lut_grid);
        let kstate = DkKStateBlock {
            sample_rate: sr,
            lut_base_sample_rate: sr,
            a_c,
            b_c,
            c_c,
            d_x,
            e_u,
            f_i,
            m_inv: disc.m_inv,
            d_d: disc.d_d,
            e_d: disc.e_d,
            f_d: disc.f_d,
            g_d: disc.g_d,
            k_imp: disc.k_imp,
            lut_base_k_imp: disc.k_imp,
            a_d: disc.a_d,
            b_d: disc.b_d,
            c_d: disc.c_d,
            bounds,
            grid_dims: [360, 360],
            total_points: 129_600,
            lut: Arc::clone(&lut_arc),
            is_synthetic: true,
        };

        Self {
            magic: IZOTOPE_DK_LUT_MAGIC,
            version: IZOTOPE_DK_LUT_VERSION,
            dc_offset: -0.04588765735184097,
            circuit_mode: 0,
            x_dc,
            u_dc,
            i_nl_dc,
            sample_rate: sr,
            a_c,
            b_c,
            c_c,
            d_x,
            e_u,
            f_i,
            m_inv: disc.m_inv,
            d_d: disc.d_d,
            e_d: disc.e_d,
            f_d: disc.f_d,
            g_d: disc.g_d,
            a_d: disc.a_d,
            b_d: disc.b_d,
            c_d: disc.c_d,
            bounds,
            grid_dims: [360, 360],
            lut: lut_arc,
            kstate,
        }
    }

    #[inline(always)]
    pub fn sample_bilinear(&self, p1: f64, p2: f64) -> [f64; 2] {
        self.kstate.sample_bilinear(p1, p2)
    }

    #[inline]
    pub fn compute_quiescent_p(&self) -> [f64; 2] {
        let mut p = [0.0; 2];
        for r in 0..2 {
            p[r] = self.d_d[r][0] * self.x_dc[0]
                + self.d_d[r][1] * self.x_dc[1]
                + self.d_d[r][2] * self.x_dc[2]
                + (self.e_d[r][0] + self.f_d[r][0]) * self.u_dc[0]
                + (self.e_d[r][1] + self.f_d[r][1]) * self.u_dc[1]
                + self.g_d[r][0] * self.i_nl_dc[0]
                + self.g_d[r][1] * self.i_nl_dc[1];
        }
        p
    }
}

/// Decoded binary representation of `exciter-dualtriode-*.dat` (`4,149,044` B, magic `0x9DFBCC23`, version `4`).
/// Contains two cascaded `KState<3, 2, 2, KorenFunction>` 12AX7 stages (`B+ = 286.0 V`).
#[derive(Clone, Debug)]
pub struct DkDualTriodeLut {
    pub magic: u32,
    pub version: u32,
    pub sample_rate: f64,
    pub dc_offset_stage1: f32,
    pub dc_offset_stage2: f32,
    pub x_dc1: [f64; 3],
    pub u_dc1: [f64; 2],
    pub i_nl_dc1: [f64; 2],
    pub x_dc2: [f64; 3],
    pub u_dc2: [f64; 2],
    pub i_nl_dc2: [f64; 2],
    pub stage1: DkKStateBlock,
    pub stage2: DkKStateBlock,
    /// Reciprocal of small-signal 2-stage voltage gain at `1,000 Hz` for unity small-signal calibration.
    pub small_signal_gain_inv: f64,
}

impl DkDualTriodeLut {
    /// Decodes `exciter-dualtriode-*.dat` (`4,149,044` bytes = `0x3F4F34`) byte-for-byte.
    pub fn from_dat_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() < IZOTOPE_DUALTRIODE_DAT_SIZE {
            return Err("File exciter-dualtriode .dat nho hon 4,149,044 bytes (0x3F4F34)");
        }
        let read_u32 = |off: usize| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        let read_f32 = |off: usize| f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        let read_f64 = |off: usize| f64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());

        let magic = read_u32(0x0000);
        let version = read_u32(0x0004);
        if magic != IZOTOPE_DUALTRIODE_LUT_MAGIC || version != IZOTOPE_DK_LUT_VERSION {
            return Err("Sai magic 0x9DFBCC23 hoac version != 4");
        }
        let header_sr = read_f32(0x0008) as f64;
        let dc_offset_stage1 = read_f32(0x000C);
        let dc_offset_stage2 = read_f32(0x0010);

        // Stage 1 DC Operating Point (0x0014..0x0064 = 80 bytes)
        let x_dc1 = [read_f64(0x001C), read_f64(0x0024), read_f64(0x002C)];
        let u_dc1 = [read_f64(0x003C), read_f64(0x0044)];
        let i_nl_dc1 = [read_f64(0x0054), read_f64(0x005C)];

        // Stage 2 DC Operating Point (0x0064..0x00B4 = 80 bytes)
        let x_dc2 = [read_f64(0x006C), read_f64(0x0074), read_f64(0x007C)];
        let u_dc2 = [read_f64(0x008C), read_f64(0x0094)];
        let i_nl_dc2 = [read_f64(0x00A4), read_f64(0x00AC)];

        // Stage 1 KState (0x00B4..0x1FA7F4) & Stage 2 KState (0x1FA7F4..0x3F4F34)
        let stage1 = DkKStateBlock::from_slice_at(bytes, 0x0000_00B4)?;
        let stage2 = DkKStateBlock::from_slice_at(bytes, 0x001F_A7F4)?;
        let sample_rate = if stage1.sample_rate > 0.0 {
            stage1.sample_rate
        } else {
            header_sr
        };

        let mut lut = Self {
            magic,
            version,
            sample_rate,
            dc_offset_stage1,
            dc_offset_stage2,
            x_dc1,
            u_dc1,
            i_nl_dc1,
            x_dc2,
            u_dc2,
            i_nl_dc2,
            stage1,
            stage2,
            small_signal_gain_inv: 1.0 / 3350.0,
        };
        lut.small_signal_gain_inv = lut.measure_small_signal_gain_inv();
        Ok(lut)
    }

    pub fn from_dat_file(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
        Self::from_dat_bytes(&bytes).map_err(|e| e.to_string())
    }

    /// Exact trapezoidal bilinear re-discretization of both cascaded 12AX7 stages to `target_sr`.
    pub fn rediscretize_for_rate(&self, target_sr: f64) -> Self {
        let stage1 = self.stage1.rediscretized(target_sr);
        let stage2 = self.stage2.rediscretized(target_sr);
        let mut out = Self {
            magic: self.magic,
            version: self.version,
            sample_rate: stage1.sample_rate,
            dc_offset_stage1: self.dc_offset_stage1,
            dc_offset_stage2: self.dc_offset_stage2,
            x_dc1: self.x_dc1,
            u_dc1: self.u_dc1,
            i_nl_dc1: self.i_nl_dc1,
            x_dc2: self.x_dc2,
            u_dc2: self.u_dc2,
            i_nl_dc2: self.i_nl_dc2,
            stage1,
            stage2,
            small_signal_gain_inv: self.small_signal_gain_inv,
        };
        out.small_signal_gain_inv = out.measure_small_signal_gain_inv();
        out
    }

    /// Fallback synthetic 2-stage 12AX7 `DualTriode` table when running without `.dat` assets.
    pub fn synthetic_from_triode(triode: &DkTriodeLut) -> Self {
        let mut out = Self {
            magic: IZOTOPE_DUALTRIODE_LUT_MAGIC,
            version: IZOTOPE_DK_LUT_VERSION,
            sample_rate: triode.sample_rate,
            dc_offset_stage1: 0.0,
            dc_offset_stage2: 0.0,
            x_dc1: triode.x_dc,
            u_dc1: triode.u_dc,
            i_nl_dc1: triode.i_nl_dc,
            x_dc2: triode.x_dc,
            u_dc2: triode.u_dc,
            i_nl_dc2: triode.i_nl_dc,
            stage1: triode.kstate.clone(),
            stage2: triode.kstate.clone(),
            small_signal_gain_inv: 1.0,
        };
        out.small_signal_gain_inv = out.measure_small_signal_gain_inv();
        out
    }

    fn measure_small_signal_gain_inv(&self) -> f64 {
        let sr = self.sample_rate.max(8_000.0);
        let mut st = DkDualTriodeChannelState::new_raw(self);
        let amp = 1.0e-5_f64;
        let freq = 1_000.0_f64;
        let n_warm = ((0.025 * sr) as usize).max(256);
        let n_meas = ((0.020 * sr) as usize).max(256);
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        for n in 0..(n_warm + n_meas) {
            let ph = 2.0 * PI * freq * (n as f64) / sr;
            let y = st.tick_unscaled_with(self, amp * ph.sin());
            if n >= n_warm {
                let k = n - n_warm;
                let ph_m = 2.0 * PI * freq * (k as f64) / sr;
                re += y * ph_m.sin();
                im += y * ph_m.cos();
            }
        }
        let peak_out = 2.0 * (re * re + im * im).sqrt() / (n_meas as f64);
        let gain = (peak_out / amp).abs();
        if gain > 1.0e-6 {
            1.0 / gain
        } else {
            1.0 / 3350.0
        }
    }
}

/// Per-channel state for the single-stage 12AX7 DK-Method solver (`0x181DB8BA0`).
#[derive(Clone, Debug)]
pub struct DkTriodeChannelState {
    pub x: [f64; 3],
    pub u_prev: [f64; 2],
    pub i_nl_prev: [f64; 2],
    pub lut: Arc<DkTriodeLut>,
}

impl DkTriodeChannelState {
    pub fn new(lut: Arc<DkTriodeLut>) -> Self {
        Self {
            x: lut.x_dc,
            u_prev: lut.u_dc,
            i_nl_prev: lut.i_nl_dc,
            lut,
        }
    }

    #[inline]
    pub fn set_lut(&mut self, lut: Arc<DkTriodeLut>) {
        self.lut = lut;
        self.reset();
    }

    #[inline]
    pub fn reset(&mut self) {
        self.x = self.lut.x_dc;
        self.u_prev = self.lut.u_dc;
        self.i_nl_prev = self.lut.i_nl_dc;
    }

    /// Steps one sample of the 12AX7 DK-Method state-space equations (`0x181DB8BA0`).
    /// Zero heap allocations.
    #[inline(always)]
    pub fn tick(&mut self, v_in: f64) -> f64 {
        let l = &*self.lut;
        let u = [v_in, l.u_dc[1]];

        let mut p = [0.0f64; 2];
        for r in 0..2 {
            p[r] = l.d_d[r][0] * self.x[0]
                + l.d_d[r][1] * self.x[1]
                + l.d_d[r][2] * self.x[2]
                + l.e_d[r][0] * u[0]
                + l.e_d[r][1] * u[1]
                + l.f_d[r][0] * self.u_prev[0]
                + l.f_d[r][1] * self.u_prev[1]
                + l.g_d[r][0] * self.i_nl_prev[0]
                + l.g_d[r][1] * self.i_nl_prev[1];
        }

        let i_nl = l.kstate.solve_currents(p);

        let u_sum = [u[0] + self.u_prev[0], u[1] + self.u_prev[1]];
        let i_sum = [i_nl[0] + self.i_nl_prev[0], i_nl[1] + self.i_nl_prev[1]];
        let mut x_next = [0.0f64; 3];
        for r in 0..3 {
            let val = l.a_d[r][0] * self.x[0]
                + l.a_d[r][1] * self.x[1]
                + l.a_d[r][2] * self.x[2]
                + l.b_d[r][0] * u_sum[0]
                + l.b_d[r][1] * u_sum[1]
                + l.c_d[r][0] * i_sum[0]
                + l.c_d[r][1] * i_sum[1];
            x_next[r] = if val.abs() < 1e-18 { 0.0 } else { val };
        }

        self.x = x_next;
        self.u_prev = u;
        self.i_nl_prev = i_nl;

        (self.x[1] - l.x_dc[1]) * l.dc_offset
    }
}

/// Per-channel state for the cascaded two-stage 12AX7 `DualTriode` DK-Method solver (`0x181DF0580`).
#[derive(Clone, Debug)]
pub struct DkDualTriodeChannelState {
    pub x1: [f64; 3],
    pub u1_prev: [f64; 2],
    pub i1_prev: [f64; 2],
    pub x2: [f64; 3],
    pub u2_prev: [f64; 2],
    pub i2_prev: [f64; 2],
    pub lut: Option<Arc<DkDualTriodeLut>>,
}

impl DkDualTriodeChannelState {
    pub fn new(lut: Arc<DkDualTriodeLut>) -> Self {
        let mut s = Self::new_raw(&lut);
        s.lut = Some(lut);
        s
    }

    fn new_raw(lut: &DkDualTriodeLut) -> Self {
        Self {
            x1: lut.x_dc1,
            u1_prev: lut.u_dc1,
            i1_prev: lut.i_nl_dc1,
            x2: lut.x_dc2,
            u2_prev: lut.u_dc2,
            i2_prev: lut.i_nl_dc2,
            lut: None,
        }
    }

    #[inline]
    pub fn set_lut(&mut self, lut: Arc<DkDualTriodeLut>) {
        self.x1 = lut.x_dc1;
        self.u1_prev = lut.u_dc1;
        self.i1_prev = lut.i_nl_dc1;
        self.x2 = lut.x_dc2;
        self.u2_prev = lut.u_dc2;
        self.i2_prev = lut.i_nl_dc2;
        self.lut = Some(lut);
    }

    #[inline]
    pub fn reset(&mut self) {
        if let Some(ref lut) = self.lut {
            self.x1 = lut.x_dc1;
            self.u1_prev = lut.u_dc1;
            self.i1_prev = lut.i_nl_dc1;
            self.x2 = lut.x_dc2;
            self.u2_prev = lut.u_dc2;
            self.i2_prev = lut.i_nl_dc2;
        }
    }

    #[inline(always)]
    fn step_kstate(
        ks: &DkKStateBlock,
        x: &mut [f64; 3],
        u_prev: &mut [f64; 2],
        i_prev: &mut [f64; 2],
        u: [f64; 2],
    ) {
        let mut p = [0.0_f64; 2];
        for r in 0..2 {
            p[r] = ks.d_d[r][0] * x[0]
                + ks.d_d[r][1] * x[1]
                + ks.d_d[r][2] * x[2]
                + ks.e_d[r][0] * u[0]
                + ks.e_d[r][1] * u[1]
                + ks.f_d[r][0] * u_prev[0]
                + ks.f_d[r][1] * u_prev[1]
                + ks.g_d[r][0] * i_prev[0]
                + ks.g_d[r][1] * i_prev[1];
        }
        let i_nl = ks.solve_currents(p);
        let u_sum = [u[0] + u_prev[0], u[1] + u_prev[1]];
        let i_sum = [i_nl[0] + i_prev[0], i_nl[1] + i_prev[1]];
        let mut x_next = [0.0_f64; 3];
        for r in 0..3 {
            let val = ks.a_d[r][0] * x[0]
                + ks.a_d[r][1] * x[1]
                + ks.a_d[r][2] * x[2]
                + ks.b_d[r][0] * u_sum[0]
                + ks.b_d[r][1] * u_sum[1]
                + ks.c_d[r][0] * i_sum[0]
                + ks.c_d[r][1] * i_sum[1];
            x_next[r] = if val.abs() < 1e-18 { 0.0 } else { val };
        }
        *x = x_next;
        *u_prev = u;
        *i_prev = i_nl;
    }

    #[inline(always)]
    fn tick_unscaled_with(&mut self, l: &DkDualTriodeLut, v_in: f64) -> f64 {
        let is_real_dualtriode = l.stage1.e_u[0][1].abs() < 1.0e-9;
        if is_real_dualtriode {
            // Stage 1: Miller plate-grid voltage x1[0] = V_p1 - V_g1, coupling cap x1[2] -> V_out1 = x1[0] + u1[0] - x1[2]
            let u1 = [l.u_dc1[0] + v_in, l.u_dc1[1]];
            Self::step_kstate(&l.stage1, &mut self.x1, &mut self.u1_prev, &mut self.i1_prev, u1);
            let v_inter = self.x1[0] + u1[0] - self.x1[2] - (l.dc_offset_stage1 as f64);

            // Stage 2: driven directly by Stage 1's RC-coupled grid node (`u_dc2[0] = 0.05775506 V`)
            let u2 = [v_inter, l.u_dc2[1]];
            Self::step_kstate(&l.stage2, &mut self.x2, &mut self.u2_prev, &mut self.i2_prev, u2);
            let v_out2_dc = l.x_dc2[0] + l.u_dc2[0] - l.x_dc2[2];
            (self.x2[0] + u2[0] - self.x2[2]) - v_out2_dc - (l.dc_offset_stage2 as f64)
        } else {
            // Synthetic fallback from single-triode state topology
            let u1 = [v_in, l.u_dc1[1]];
            Self::step_kstate(&l.stage1, &mut self.x1, &mut self.u1_prev, &mut self.i1_prev, u1);
            let v_mid = (self.x1[1] - l.x_dc1[1]) * (-0.04588765735184097) * 0.85;
            let u2 = [v_mid, l.u_dc2[1]];
            Self::step_kstate(&l.stage2, &mut self.x2, &mut self.u2_prev, &mut self.i2_prev, u2);
            (self.x2[1] - l.x_dc2[1]) * (-0.04588765735184097)
        }
    }

    #[inline(always)]
    pub fn tick(&mut self, v_in: f64) -> f64 {
        self.tick_borrowed(v_in)
    }

    #[inline(always)]
    pub fn tick_borrowed(&mut self, v_in: f64) -> f64 {
        let Some(ref l) = self.lut else {
            return v_in;
        };
        let l_ptr: *const DkDualTriodeLut = Arc::as_ptr(l);
        // SAFETY: `self.lut` remains alive and unmodified for the entire duration of `tick_borrowed`.
        let l_ref = unsafe { &*l_ptr };
        self.tick_unscaled_with(l_ref, v_in) * l_ref.small_signal_gain_inv
    }
}

// ============================================================================
// MULTI-RATE `.dat` DISCOVERY & PROCESS-GLOBAL CACHE (NON-RT ONLY)
// ============================================================================

struct ExciterLutProcessCache {
    triode_by_rate_hz: HashMap<u64, Arc<DkTriodeLut>>,
    dualtriode_by_rate_hz: HashMap<u64, Arc<DkDualTriodeLut>>,
}

fn global_lut_cache() -> &'static Mutex<ExciterLutProcessCache> {
    static CACHE: OnceLock<Mutex<ExciterLutProcessCache>> = OnceLock::new();
    CACHE.get_or_init(|| {
        Mutex::new(ExciterLutProcessCache {
            triode_by_rate_hz: HashMap::new(),
            dualtriode_by_rate_hz: HashMap::new(),
        })
    })
}

fn candidate_lut_directories() -> Vec<PathBuf> {
    let mut dirs = Vec::with_capacity(12);
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    if let Some(ws_root) = manifest.parent().and_then(|p| p.parent()) {
        dirs.push(ws_root.join("assets").join("exciter_luts"));
        dirs.push(ws_root.join("assets").join("ozone12").join("luts"));
    }
    if let Ok(exe) = std::env::current_exe() {
        let mut cur = exe.parent();
        for _ in 0..5 {
            if let Some(d) = cur {
                dirs.push(d.join("assets").join("exciter_luts"));
                dirs.push(d.join("assets").join("ozone12").join("luts"));
                cur = d.parent();
            }
        }
    }
    dirs.push(PathBuf::from(r"f:\FL studio\toneforge\assets\exciter_luts"));
    dirs.push(PathBuf::from(r"f:\FL studio\toneforge\assets\ozone12\luts"));
    dirs.push(PathBuf::from("assets/exciter_luts"));
    dirs.push(PathBuf::from("assets/ozone12/luts"));
    dirs.push(PathBuf::from(
        r"f:\FL studio\extracted_targets\ozone12\exciterCache_component\ExciterCache\Exciter Cache",
    ));
    dirs
}

fn find_dat_file_for_rate(prefix: &str, rate_hz: f64, extra_dir: Option<&Path>) -> Option<PathBuf> {
    let filename = format!("{prefix}{rate_hz:.6}.dat");
    if let Some(d) = extra_dir {
        let p = d.join(&filename);
        if p.exists() {
            return Some(p);
        }
    }
    for dir in candidate_lut_directories() {
        let p = dir.join(&filename);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn nearest_standard_rate(target_sr: f64) -> u32 {
    let mut best = 48_000u32;
    let mut best_diff = f64::MAX;
    for &r in &STANDARD_EXCITER_LUT_RATES {
        let diff = ((r as f64) - target_sr).abs();
        if diff < best_diff {
            best_diff = diff;
            best = r;
        }
    }
    best
}

/// Loads (or retrieves from cache) the single-stage `Triode` LUT at `target_sr` (`Hz`),
/// using the exact `exciter-triode-<rate:.6>.dat` file if available or exact trapezoidal
/// re-discretization from the nearest `.dat` rate.
pub fn load_triode_lut_cached(target_sr: f64) -> Arc<DkTriodeLut> {
    load_triode_lut_with_hint(target_sr, None)
}

fn load_triode_lut_with_hint(target_sr: f64, extra_dir: Option<&Path>) -> Arc<DkTriodeLut> {
    let sr = target_sr.max(8_000.0);
    let key = (sr * 100.0).round() as u64;
    if extra_dir.is_none() {
        if let Ok(guard) = global_lut_cache().lock() {
            if let Some(found) = guard.triode_by_rate_hz.get(&key) {
                return Arc::clone(found);
            }
        }
    }

    let loaded = if let Some(exact_path) = find_dat_file_for_rate("exciter-triode-", sr, extra_dir)
    {
        DkTriodeLut::from_dat_file(&exact_path)
            .ok()
            .map(Arc::new)
    } else {
        let nearest = nearest_standard_rate(sr) as f64;
        find_dat_file_for_rate("exciter-triode-", nearest, extra_dir)
            .or_else(|| find_dat_file_for_rate("exciter-triode-", 48_000.0, extra_dir))
            .and_then(|p| DkTriodeLut::from_dat_file(&p).ok())
            .map(|base| Arc::new(base.rediscretize_for_rate(sr)))
    }
    .unwrap_or_else(|| Arc::new(DkTriodeLut::synthetic_koren_12ax7_lut(sr)));

    if extra_dir.is_none() {
        if let Ok(mut guard) = global_lut_cache().lock() {
            guard.triode_by_rate_hz.insert(key, Arc::clone(&loaded));
        }
    }
    loaded
}

/// Loads (or retrieves from cache) the two-stage `DualTriode` LUT at `target_sr` (`Hz`),
/// using the exact `exciter-dualtriode-<rate:.6>.dat` file if available or exact trapezoidal
/// re-discretization from the nearest `.dat` rate.
pub fn load_dualtriode_lut_cached(target_sr: f64) -> Arc<DkDualTriodeLut> {
    load_dualtriode_lut_with_hint(target_sr, None)
}

fn load_dualtriode_lut_with_hint(
    target_sr: f64,
    extra_dir: Option<&Path>,
) -> Arc<DkDualTriodeLut> {
    let sr = target_sr.max(8_000.0);
    let key = (sr * 100.0).round() as u64;
    if extra_dir.is_none() {
        if let Ok(guard) = global_lut_cache().lock() {
            if let Some(found) = guard.dualtriode_by_rate_hz.get(&key) {
                return Arc::clone(found);
            }
        }
    }

    let loaded =
        if let Some(exact_path) = find_dat_file_for_rate("exciter-dualtriode-", sr, extra_dir) {
            DkDualTriodeLut::from_dat_file(&exact_path)
                .ok()
                .map(Arc::new)
        } else {
            let nearest = nearest_standard_rate(sr) as f64;
            find_dat_file_for_rate("exciter-dualtriode-", nearest, extra_dir)
                .or_else(|| find_dat_file_for_rate("exciter-dualtriode-", 48_000.0, extra_dir))
                .and_then(|p| DkDualTriodeLut::from_dat_file(&p).ok())
                .map(|base| Arc::new(base.rediscretize_for_rate(sr)))
        }
        .unwrap_or_else(|| {
            let tri = load_triode_lut_with_hint(sr, extra_dir);
            Arc::new(DkDualTriodeLut::synthetic_from_triode(&tri))
        });

    if extra_dir.is_none() {
        if let Ok(mut guard) = global_lut_cache().lock() {
            guard.dualtriode_by_rate_hz.insert(key, Arc::clone(&loaded));
        }
    }
    loaded
}

/// Ozone 12 Exciter saturation circuit modes (`AutomatableParamList_Exciter.txt` & `DSP::Shared::Exciter`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExciterMode {
    /// Mode 0: Analog transformer-coupled saturation (balanced odd + subtle even harmonics).
    Analog = 0,
    /// Mode 1: Retro transistor/germanium knee saturation (crisp odd-order harmonics).
    Retro = 1,
    /// Mode 2: Magnetic tape hysteresis saturation (symmetric odd harmonics + HF pre-emphasis).
    Tape = 2,
    /// Mode 3: Warm asymmetric saturation (even-harmonic dominant 2nd harmonic warmth).
    Warm = 3,
    /// Mode 4: Single-stage 12AX7 Vacuum Triode (`DkTriodeChannelState` DK-Method LUT solver).
    Triode = 4,
    /// Mode 5: Cascaded Dual-Stage 12AX7 Vacuum Triode (`DkDualTriodeChannelState` DK-Method solver).
    DualTriode = 5,
    /// Mode 6: Class-A Vacuum Tube saturation (Koren equation asymmetric tube transfer).
    Tube = 6,
}

impl ExciterMode {
    #[inline]
    pub fn from_index(idx: u32) -> Self {
        match idx % 7 {
            0 => Self::Analog,
            1 => Self::Retro,
            2 => Self::Tape,
            3 => Self::Warm,
            4 => Self::Triode,
            5 => Self::DualTriode,
            _ => Self::Tube,
        }
    }
}

/// Per-band configuration for the 4-Band Ozone 12 Exciter.
#[derive(Clone, Copy, Debug)]
pub struct ExciterBandConfig {
    pub enabled: bool,
    pub mode: ExciterMode,
    /// Saturation drive / amount in `[0.0, 100.0]%` (`Band b Amount` in `AutomatableParamList_Exciter.txt`).
    pub amount_pct: f32,
    /// Parallel wet/dry mix in `[0.0, 1.0]` (`Band b Mix` in `AutomatableParamList_Exciter.txt`).
    pub wet_mix: f32,
}

impl Default for ExciterBandConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: ExciterMode::Triode,
            amount_pct: 25.0,
            wet_mix: 1.0,
        }
    }
}

/// Pre-loaded `[1x, 2x, 4x]` rate-matched LUTs for zero-I/O real-time switching.
#[derive(Clone, Debug)]
struct MultiRateExciterLuts {
    triode_1x: Arc<DkTriodeLut>,
    triode_2x: Arc<DkTriodeLut>,
    triode_4x: Arc<DkTriodeLut>,
    dualtriode_1x: Arc<DkDualTriodeLut>,
    dualtriode_2x: Arc<DkDualTriodeLut>,
    dualtriode_4x: Arc<DkDualTriodeLut>,
}

impl MultiRateExciterLuts {
    fn load_for_host_rate(host_sr: f64, extra_dir: Option<&Path>) -> Self {
        let sr = host_sr.max(8_000.0);
        Self {
            triode_1x: load_triode_lut_with_hint(sr, extra_dir),
            triode_2x: load_triode_lut_with_hint(sr * 2.0, extra_dir),
            triode_4x: load_triode_lut_with_hint(sr * 4.0, extra_dir),
            dualtriode_1x: load_dualtriode_lut_with_hint(sr, extra_dir),
            dualtriode_2x: load_dualtriode_lut_with_hint(sr * 2.0, extra_dir),
            dualtriode_4x: load_dualtriode_lut_with_hint(sr * 4.0, extra_dir),
        }
    }

    fn from_explicit_triode(lut: Arc<DkTriodeLut>, host_sr: f64, extra_dir: Option<&Path>) -> Self {
        let sr = host_sr.max(8_000.0);
        let t1 = if (lut.sample_rate - sr).abs() <= 1.0 {
            Arc::clone(&lut)
        } else {
            Arc::new(lut.rediscretize_for_rate(sr))
        };
        let t2 = find_dat_file_for_rate("exciter-triode-", sr * 2.0, extra_dir)
            .and_then(|p| DkTriodeLut::from_dat_file(&p).ok())
            .map(Arc::new)
            .unwrap_or_else(|| Arc::new(lut.rediscretize_for_rate(sr * 2.0)));
        let t4 = find_dat_file_for_rate("exciter-triode-", sr * 4.0, extra_dir)
            .and_then(|p| DkTriodeLut::from_dat_file(&p).ok())
            .map(Arc::new)
            .unwrap_or_else(|| Arc::new(lut.rediscretize_for_rate(sr * 4.0)));
        Self {
            triode_1x: t1,
            triode_2x: t2,
            triode_4x: t4,
            dualtriode_1x: load_dualtriode_lut_with_hint(sr, extra_dir),
            dualtriode_2x: load_dualtriode_lut_with_hint(sr * 2.0, extra_dir),
            dualtriode_4x: load_dualtriode_lut_with_hint(sr * 4.0, extra_dir),
        }
    }

    fn from_explicit_dualtriode(
        lut: Arc<DkDualTriodeLut>,
        host_sr: f64,
        extra_dir: Option<&Path>,
    ) -> Self {
        let sr = host_sr.max(8_000.0);
        let d1 = if (lut.sample_rate - sr).abs() <= 1.0 {
            Arc::clone(&lut)
        } else {
            Arc::new(lut.rediscretize_for_rate(sr))
        };
        let d2 = find_dat_file_for_rate("exciter-dualtriode-", sr * 2.0, extra_dir)
            .and_then(|p| DkDualTriodeLut::from_dat_file(&p).ok())
            .map(Arc::new)
            .unwrap_or_else(|| Arc::new(lut.rediscretize_for_rate(sr * 2.0)));
        let d4 = find_dat_file_for_rate("exciter-dualtriode-", sr * 4.0, extra_dir)
            .and_then(|p| DkDualTriodeLut::from_dat_file(&p).ok())
            .map(Arc::new)
            .unwrap_or_else(|| Arc::new(lut.rediscretize_for_rate(sr * 4.0)));
        Self {
            triode_1x: load_triode_lut_with_hint(sr, extra_dir),
            triode_2x: load_triode_lut_with_hint(sr * 2.0, extra_dir),
            triode_4x: load_triode_lut_with_hint(sr * 4.0, extra_dir),
            dualtriode_1x: d1,
            dualtriode_2x: d2,
            dualtriode_4x: d4,
        }
    }

    #[inline]
    fn triode_for(&self, factor: OversampleFactor) -> &Arc<DkTriodeLut> {
        match factor {
            OversampleFactor::X1 => &self.triode_1x,
            OversampleFactor::X2 => &self.triode_2x,
            OversampleFactor::X4 => &self.triode_4x,
        }
    }

    #[inline]
    fn dualtriode_for(&self, factor: OversampleFactor) -> &Arc<DkDualTriodeLut> {
        match factor {
            OversampleFactor::X1 => &self.dualtriode_1x,
            OversampleFactor::X2 => &self.dualtriode_2x,
            OversampleFactor::X4 => &self.dualtriode_4x,
        }
    }
}

#[derive(Clone, Debug)]
struct ExciterBandRuntimeState {
    oversampler: PolyphaseHalfBandOversampler,
    active_factor: OversampleFactor,
    triode: [DkTriodeChannelState; 2],
    dualtriode: [DkDualTriodeChannelState; 2],
    dc_block_x1: [f64; 2],
    dc_block_y1: [f64; 2],
    tape_lp_z: [f64; 2],
    rms_meter: f32,
}

impl ExciterBandRuntimeState {
    fn new(luts: &MultiRateExciterLuts, factor: OversampleFactor) -> Self {
        let t_lut = Arc::clone(luts.triode_for(factor));
        let d_lut = Arc::clone(luts.dualtriode_for(factor));
        Self {
            oversampler: PolyphaseHalfBandOversampler::new(
                OversamplerFilterMode::EllipticPolyphaseIir,
                factor,
            ),
            active_factor: factor,
            triode: [
                DkTriodeChannelState::new(Arc::clone(&t_lut)),
                DkTriodeChannelState::new(t_lut),
            ],
            dualtriode: [
                DkDualTriodeChannelState::new(Arc::clone(&d_lut)),
                DkDualTriodeChannelState::new(d_lut),
            ],
            dc_block_x1: [0.0; 2],
            dc_block_y1: [0.0; 2],
            tape_lp_z: [0.0; 2],
            rms_meter: 0.0,
        }
    }

    #[inline]
    fn sync_factor(&mut self, luts: &MultiRateExciterLuts, factor: OversampleFactor) {
        if self.active_factor != factor {
            self.active_factor = factor;
            self.oversampler.set_factor(factor);
            let t_lut = luts.triode_for(factor);
            let d_lut = luts.dualtriode_for(factor);
            for ch in 0..2 {
                self.triode[ch].set_lut(Arc::clone(t_lut));
                self.dualtriode[ch].set_lut(Arc::clone(d_lut));
                self.dc_block_x1[ch] = 0.0;
                self.dc_block_y1[ch] = 0.0;
                self.tape_lp_z[ch] = 0.0;
            }
        }
    }

    fn reload_luts(&mut self, luts: &MultiRateExciterLuts, factor: OversampleFactor) {
        self.active_factor = factor;
        self.oversampler.set_factor(factor);
        let t_lut = luts.triode_for(factor);
        let d_lut = luts.dualtriode_for(factor);
        for ch in 0..2 {
            self.triode[ch].set_lut(Arc::clone(t_lut));
            self.dualtriode[ch].set_lut(Arc::clone(d_lut));
        }
        self.reset();
    }

    fn reset(&mut self) {
        self.oversampler.reset();
        for ch in 0..2 {
            self.triode[ch].reset();
            self.dualtriode[ch].reset();
            self.dc_block_x1[ch] = 0.0;
            self.dc_block_y1[ch] = 0.0;
            self.tape_lp_z[ch] = 0.0;
        }
        self.rms_meter = 0.0;
    }

    #[inline(always)]
    fn process_stereo_band(
        &mut self,
        band_in: [f64; 2],
        cfg: ExciterBandConfig,
        tape_alpha: f64,
        dc_g: f64,
        dc_r: f64,
    ) -> [f64; 2] {
        let amt = (cfg.amount_pct as f64 * 0.01).clamp(0.0, 1.0);
        let w = (cfg.wet_mix as f64).clamp(0.0, 1.0);
        let active = cfg.enabled && w > 1.0e-7 && amt > 1.0e-7;

        let triode = &mut self.triode;
        let dualtriode = &mut self.dualtriode;
        let dc_block_x1 = &mut self.dc_block_x1;
        let dc_block_y1 = &mut self.dc_block_y1;
        let tape_lp_z = &mut self.tape_lp_z;

        let out = self.oversampler.process_stereo_frame(band_in, |sub_in| {
            if !active {
                return sub_in;
            }
            let mut sub_out = [0.0_f64; 2];
            for ch in 0..2 {
                let x_in = sub_in[ch];
                let drive = 1.0 + amt * 7.0;
                let inv_drive_norm = 1.0 / (1.0 + amt * 2.5);
                let raw_y = match cfg.mode {
                    ExciterMode::Triode => {
                        let drive_v = 0.35 + amt * 3.15;
                        let y_plate = triode[ch].tick(x_in * drive_v);
                        y_plate / (drive_v * (1.0 + amt * 0.45))
                    }
                    ExciterMode::DualTriode => {
                        let drive_v = 0.006 + amt * 0.068;
                        let y_stage2 = dualtriode[ch].tick_borrowed(x_in * drive_v);
                        y_stage2 / (drive_v * (1.0 + amt * 0.55))
                    }
                    ExciterMode::Tube => {
                        let gx = x_in * drive;
                        let asym = 0.22 * amt;
                        let shaped = (gx + asym * gx * gx).tanh() - (asym * 0.5).tanh();
                        shaped * inv_drive_norm
                    }
                    ExciterMode::Warm => {
                        let gx = x_in * (1.0 + amt * 4.0);
                        let t = gx.tanh();
                        let even = 0.35 * amt * (t * t);
                        (t + even) / (1.0 + amt * 1.4)
                    }
                    ExciterMode::Tape => {
                        tape_lp_z[ch] += tape_alpha * (x_in - tape_lp_z[ch]);
                        let hf_part = x_in - tape_lp_z[ch];
                        let hf_boost = x_in + 0.45 * hf_part;
                        let gx = hf_boost * drive;
                        let odd_sat = (gx / (1.0 + gx * gx).sqrt()) + 0.15 * gx.tanh();
                        odd_sat * 0.88
                    }
                    ExciterMode::Retro => {
                        let gx = x_in * (1.0 + amt * 6.0);
                        let odd_retro = gx.signum() * (1.0 - (-gx.abs()).exp());
                        odd_retro / (1.0 - (-(1.0 + amt * 2.0)).exp()).max(0.5)
                    }
                    ExciterMode::Analog => {
                        let gx = x_in * (1.0 + amt * 4.5);
                        let sat = gx / (1.0 + 0.6 * gx * gx).sqrt() + 0.08 * amt * gx * gx;
                        sat * 0.85
                    }
                };
                let wet = match cfg.mode {
                    ExciterMode::Tape | ExciterMode::Retro => raw_y.clamp(-4.0, 4.0),
                    _ => {
                        let y = dc_g * (raw_y - dc_block_x1[ch]) + dc_r * dc_block_y1[ch];
                        dc_block_x1[ch] = raw_y;
                        dc_block_y1[ch] = if y.abs() < 1e-20 { 0.0 } else { y };
                        y.clamp(-4.0, 4.0)
                    }
                };
                // Blend dry & wet inside the oversampled domain for 100% phase coherence (zero comb filtering!)
                sub_out[ch] = (1.0 - w) * x_in + w * wet;
            }
            sub_out
        });

        let e = 0.5 * (out[0] * out[0] + out[1] * out[1]) as f32;
        self.rms_meter = 0.995 * self.rms_meter + 0.005 * e;
        out
    }
}

/// Complete 4-Band iZotope Ozone 12 Exciter (`DkTriodeLutExciter` / `DkTriodeMultibandExciter` / `DkTriodeExciterModule`).
#[derive(Clone, Debug)]
pub struct DkTriodeLutExciter {
    pub bypass: bool,
    pub multiband_enabled: bool,
    /// Legacy `2x` oversampling toggle (used when `oversample_4x == false`).
    pub oversample_2x: bool,
    /// Production `4x` oversampling toggle (enabled by default matching `DSP::OzoneExciter` `0x181DC88D0`).
    pub oversample_4x: bool,
    pub bands: [ExciterBandConfig; OZONE_MULTIBAND_COUNT],
    pub crossover: LinkwitzRiley4BandCrossover,
    pub post_filter_freq_hz: f64,
    pub post_filter_gain_db: f64,
    pub lut: Arc<DkTriodeLut>,
    pub dualtriode_lut: Arc<DkDualTriodeLut>,
    multi_luts: MultiRateExciterLuts,
    band_states: [ExciterBandRuntimeState; OZONE_MULTIBAND_COUNT],
    post_filter: StereoBiquadF64,
    sample_rate: f64,
    extra_lut_dir: Option<PathBuf>,
}

/// Alias matching `05_master_rust_svelte_blueprint.md` Section 2.
pub type DkTriodeExciterModule = DkTriodeLutExciter;
/// Alias matching Phase 4.2 production API (`DkTriodeMultibandExciter`).
pub type DkTriodeMultibandExciter = DkTriodeLutExciter;

impl Default for DkTriodeLutExciter {
    fn default() -> Self {
        Self::new(48_000.0)
    }
}

impl DkTriodeLutExciter {
    pub fn new(sample_rate: f64) -> Self {
        let sr = sample_rate.max(8_000.0);
        let multi_luts = MultiRateExciterLuts::load_for_host_rate(sr, None);
        Self::from_multi_luts(multi_luts, sr, None)
    }

    /// Loads either a single `Triode` (`0x8BAD43FC`) or `DualTriode` (`0x9DFBCC23`) `.dat` file
    /// and preloads matching multi-rate tables (`1x`, `2x`, `4x`) from its directory.
    pub fn from_dat_file(path: &Path, sample_rate: f64) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
        if bytes.len() < 8 {
            return Err("Exciter .dat file shorter than 8-byte magic header".to_string());
        }
        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let sr = sample_rate.max(8_000.0);
        let extra_dir = path.parent().map(Path::to_path_buf);

        let multi_luts = match magic {
            IZOTOPE_DK_LUT_MAGIC => {
                let lut = Arc::new(DkTriodeLut::from_dat_bytes(&bytes).map_err(|e| e.to_string())?);
                MultiRateExciterLuts::from_explicit_triode(lut, sr, extra_dir.as_deref())
            }
            IZOTOPE_DUALTRIODE_LUT_MAGIC => {
                let lut =
                    Arc::new(DkDualTriodeLut::from_dat_bytes(&bytes).map_err(|e| e.to_string())?);
                MultiRateExciterLuts::from_explicit_dualtriode(lut, sr, extra_dir.as_deref())
            }
            _ => {
                return Err(format!(
                    "Unsupported Exciter .dat magic 0x{magic:08X} in {}",
                    path.display()
                ));
            }
        };
        Ok(Self::from_multi_luts(multi_luts, sr, extra_dir))
    }

    pub fn with_lut(lut: Arc<DkTriodeLut>, sample_rate: f64) -> Self {
        let sr = sample_rate.max(8_000.0);
        let multi_luts = MultiRateExciterLuts::from_explicit_triode(lut, sr, None);
        Self::from_multi_luts(multi_luts, sr, None)
    }

    fn from_multi_luts(
        multi_luts: MultiRateExciterLuts,
        sample_rate: f64,
        extra_lut_dir: Option<PathBuf>,
    ) -> Self {
        let sr = sample_rate.max(8_000.0);
        let default_factor = OversampleFactor::X4;
        let lut = Arc::clone(multi_luts.triode_for(default_factor));
        let dualtriode_lut = Arc::clone(multi_luts.dualtriode_for(default_factor));
        let band_states = [
            ExciterBandRuntimeState::new(&multi_luts, default_factor),
            ExciterBandRuntimeState::new(&multi_luts, default_factor),
            ExciterBandRuntimeState::new(&multi_luts, default_factor),
            ExciterBandRuntimeState::new(&multi_luts, default_factor),
        ];
        let mut post_filter = StereoBiquadF64::new_unity();
        post_filter.set_high_shelf(4500.0, 0.70710678, 0.0, sr);

        Self {
            bypass: false,
            multiband_enabled: true,
            oversample_2x: false,
            oversample_4x: true,
            bands: [ExciterBandConfig::default(); OZONE_MULTIBAND_COUNT],
            crossover: LinkwitzRiley4BandCrossover::new(DEFAULT_CROSSOVER_FREQS_HZ, sr),
            post_filter_freq_hz: 4500.0,
            post_filter_gain_db: 0.0,
            lut,
            dualtriode_lut,
            multi_luts,
            band_states,
            post_filter,
            sample_rate: sr,
            extra_lut_dir,
        }
    }

    /// Sets the oversampling factor (`1`, `2`, or `4`) without allocating.
    #[inline]
    pub fn set_oversampling_factor(&mut self, factor: usize) {
        match factor {
            0 | 1 => {
                self.oversample_4x = false;
                self.oversample_2x = false;
            }
            2 => {
                self.oversample_4x = false;
                self.oversample_2x = true;
            }
            _ => {
                self.oversample_4x = true;
                self.oversample_2x = false;
            }
        }
        let eff = self.effective_oversample_factor();
        self.lut = Arc::clone(self.multi_luts.triode_for(eff));
        self.dualtriode_lut = Arc::clone(self.multi_luts.dualtriode_for(eff));
        for bs in &mut self.band_states {
            bs.sync_factor(&self.multi_luts, eff);
        }
    }

    #[inline]
    pub const fn effective_oversample_factor(&self) -> OversampleFactor {
        if self.oversample_4x {
            OversampleFactor::X4
        } else if self.oversample_2x {
            OversampleFactor::X2
        } else {
            OversampleFactor::X1
        }
    }

    #[inline]
    fn sub_rate_filter_coeffs(effective_sr: f64) -> (f64, f64, f64) {
        // Bilinear 5.0 Hz DC blocker: H(z) = dc_g * (1 - z^-1) / (1 - dc_r * z^-1)
        let k_dc = (PI * 5.0 / effective_sr).tan();
        let dc_g = 1.0 / (1.0 + k_dc);
        let dc_r = (1.0 - k_dc) / (1.0 + k_dc);
        // Sample-rate-invariant 1-pole 3,200 Hz split for Tape pre-emphasis
        let k_tape = (PI * 3_200.0 / effective_sr).tan();
        let tape_alpha = (2.0 * k_tape) / (1.0 + 2.0 * k_tape);
        (tape_alpha, dc_g, dc_r)
    }

    /// Configures all 4 bands to use the same `ExciterMode`, `amount_pct`, and `wet_mix`.
    pub fn set_all_bands_mode(&mut self, mode: ExciterMode, amount_pct: f32, wet_mix: f32) {
        for b in &mut self.bands {
            b.enabled = true;
            b.mode = mode;
            b.amount_pct = amount_pct.clamp(0.0, 100.0);
            b.wet_mix = wet_mix.clamp(0.0, 1.0);
        }
    }

    /// Updates the post-exciter high-shelf tone filter (`AutomatableParamList_Exciter.txt` lines 14-15).
    pub fn set_post_filter(&mut self, freq_hz: f64, gain_db: f64) {
        self.post_filter_freq_hz = freq_hz.clamp(200.0, 20_000.0);
        self.post_filter_gain_db = gain_db.clamp(-12.0, 12.0);
        self.post_filter.set_high_shelf(
            self.post_filter_freq_hz,
            0.70710678,
            self.post_filter_gain_db,
            self.sample_rate,
        );
    }

    /// Processes a single stereo sample pair `(L, R)` (`0` heap allocations).
    #[inline(always)]
    pub fn process_sample_pair(&mut self, left: f32, right: f32) -> StereoFrame {
        self.step_frame([left, right])
    }

    /// Processes a single stereo frame `[L, R]` in-place (Zero Heap Allocation).
    #[inline(always)]
    pub fn step_frame(&mut self, frame: StereoFrame) -> StereoFrame {
        if self.bypass {
            return frame;
        }
        let factor = self.effective_oversample_factor();
        for bs in &mut self.band_states {
            if bs.active_factor != factor {
                bs.sync_factor(&self.multi_luts, factor);
            }
        }

        let effective_sr = self.sample_rate * (factor.as_usize() as f64);
        let (tape_alpha, dc_g, dc_r) = Self::sub_rate_filter_coeffs(effective_sr);
        let in_f64 = [frame[0] as f64, frame[1] as f64];

        let mut sum_out = [0.0f64; 2];
        if self.multiband_enabled {
            let split = self.crossover.split_frame_f64(in_f64);
            for b in 0..OZONE_MULTIBAND_COUNT {
                let band_out = self.band_states[b].process_stereo_band(
                    split[b],
                    self.bands[b],
                    tape_alpha,
                    dc_g,
                    dc_r,
                );
                sum_out[0] += band_out[0];
                sum_out[1] += band_out[1];
            }
        } else {
            sum_out = self.band_states[0].process_stereo_band(
                in_f64,
                self.bands[0],
                tape_alpha,
                dc_g,
                dc_r,
            );
        }

        if self.post_filter_gain_db.abs() > 1e-4 {
            sum_out = self.post_filter.step_stereo(sum_out);
        }

        [sum_out[0] as f32, sum_out[1] as f32]
    }

    /// Exports harmonic drive telemetry into `Ozone12MasterTelemetry.exciter_drive_gr` (`248` floats).
    #[inline]
    pub fn export_telemetry(&self, exciter_drive_gr: &mut [f32; 248]) {
        for (i, slot) in exciter_drive_gr.iter_mut().enumerate() {
            let b = (i * OZONE_MULTIBAND_COUNT) / 248;
            let cfg = self.bands[b];
            let rms_db = 10.0 * self.band_states[b].rms_meter.max(1e-12).log10();
            *slot = if cfg.enabled {
                (cfg.amount_pct * 0.18) + (rms_db + 60.0).max(0.0) * 0.05
            } else {
                0.0
            };
        }
    }
}

impl OzoneDspModule for DkTriodeLutExciter {
    fn name(&self) -> &'static str {
        "iZotope Ozone 12 4-Band 12AX7 DK-Triode Exciter"
    }

    fn reset(&mut self, sample_rate: f64, _max_block_size: usize) {
        let sr = sample_rate.max(8_000.0);
        let rate_changed = (sr - self.sample_rate).abs() > 0.5;
        self.sample_rate = sr;
        let factor = self.effective_oversample_factor();

        if rate_changed {
            self.multi_luts =
                MultiRateExciterLuts::load_for_host_rate(sr, self.extra_lut_dir.as_deref());
            self.lut = Arc::clone(self.multi_luts.triode_for(factor));
            self.dualtriode_lut = Arc::clone(self.multi_luts.dualtriode_for(factor));
            for bs in &mut self.band_states {
                bs.reload_luts(&self.multi_luts, factor);
            }
        } else {
            for bs in &mut self.band_states {
                bs.sync_factor(&self.multi_luts, factor);
                bs.reset();
            }
        }

        self.crossover
            .configure(self.crossover.cutoffs_hz, self.sample_rate);
        self.crossover.reset();
        self.set_post_filter(self.post_filter_freq_hz, self.post_filter_gain_db);
        self.post_filter.reset_state();
    }

    fn process_block(&mut self, left: &mut [f32], right: &mut [f32]) {
        let n = left.len().min(right.len());
        for i in 0..n {
            let [l, r] = self.step_frame([left[i], right[i]]);
            left[i] = l;
            right[i] = r;
        }
    }
}

impl UnifiedDspNode for DkTriodeLutExciter {
    fn name(&self) -> &'static str {
        "iZotope Ozone 12 4-Band 12AX7 DK-Triode Exciter"
    }

    fn latency_samples(&self) -> usize {
        0
    }

    fn prepare(&mut self, sample_rate: f64, max_block_size: usize) {
        self.reset(sample_rate, max_block_size);
    }

    fn reset_state(&mut self) {
        self.crossover.reset();
        for bs in &mut self.band_states {
            bs.reset();
        }
        self.post_filter.reset_state();
    }

    fn set_parameter_normalized(&mut self, param_id: u32, norm_val: f32) {
        let v = norm_val.clamp(0.0, 1.0);
        match param_id {
            0 => self.bypass = v >= 0.5,
            1..=12 => {
                let idx = (param_id - 1) as usize;
                let band_idx = idx / 3;
                let field = idx % 3;
                match field {
                    0 => self.bands[band_idx].amount_pct = v * 100.0,
                    1 => self.bands[band_idx].wet_mix = v,
                    2 => {
                        self.bands[band_idx].mode =
                            ExciterMode::from_index((v * 6.0).round() as u32)
                    }
                    _ => {}
                }
            }
            13 => {
                let freq = 200.0 * (100.0f64).powf(v as f64);
                self.set_post_filter(freq, self.post_filter_gain_db);
            }
            14 => {
                let gain = -12.0 + (v as f64) * 24.0;
                self.set_post_filter(self.post_filter_freq_hz, gain);
            }
            _ => {}
        }
    }

    fn export_visual_telemetry(&self, frame: &mut VisualTelemetryFrame) {
        self.export_telemetry(&mut frame.ozone12_telemetry.exciter_drive_gr);
    }

    fn process_slice(
        &mut self,
        _ctx: &ProcessContext,
        main_io: &mut [StereoFrame],
        _sidechain_in: Option<&[StereoFrame]>,
        _aux_sends_out: &mut [&mut [StereoFrame]],
    ) {
        for frame in main_io.iter_mut() {
            *frame = self.step_frame(*frame);
        }
    }
}
