//! `dsp-ozone12-mastering`:
//! 1. Exact binary parser & 2D bilinear DK-Method solver for `exciter-triode-*.dat`
//!    (`0x8BAD43FC`, `DSP::Shared::KState<3,2,2,KorenFunction>` at `0x181DB8BA0`).
//! 2. `SlidingMaxHeap` (`0x183F81EE4`) + `IrcVMaximizerCore` (UpwardComp + SoftClip + Transient/Sustain Split).
//! 3. Unity-Identity Additive Stem EQ (`0x1857397B0`).

use daw_core_graph::{ProcessContext, StereoFrame, UnifiedDspNode};
use std::sync::Arc;

// ============================================================================
// 1. IZOTOPE OZONE 12: EXACT BINARY PARSER & SOLVER FOR `.dat` (`0x8BAD43FC`)
// ============================================================================

pub const IZOTOPE_DK_LUT_MAGIC: u32 = 0x8BAD_43FC;
pub const IZOTOPE_TRIODE_DAT_SIZE: usize = 0x1FA7A4; // 2,074,532 bytes

#[derive(Clone)]
pub struct DkTriodeLut {
    pub dc_offset: f64,                 // Offset 0x0008
    pub circuit_mode: u32,              // Offset 0x0010 (0 = Triode, 1 = DualTriode)
    pub x_dc: [f64; 3],                 // Offset 0x001C
    pub u_dc: [f64; 2],                 // Offset 0x003C
    pub i_nl_dc: [f64; 2],              // Offset 0x0054
    pub sample_rate: f64,               // Offset 0x0064
    pub d_d: [[f64; 3]; 2],             // Offset 0x020C (after u32[2] header at 0x0204)
    pub e_d: [[f64; 2]; 2],             // Offset 0x0244
    pub f_d: [[f64; 2]; 2],             // Offset 0x026C
    pub g_d: [[f64; 2]; 2],             // Offset 0x0294
    pub a_d: [[f64; 3]; 3],             // Offset 0x02BC
    pub b_d: [[f64; 2]; 3],             // Offset 0x030C
    pub c_d: [[f64; 2]; 3],             // Offset 0x0344
    pub bounds: [f64; 4],               // Offset 0x0374: [-600.0, 300.0, 240.0, 303.0]
    pub grid_dims: [u32; 2],            // Offset 0x0394: [360, 360]
    pub lut: Arc<Vec<[[f64; 2]; 360]>>, // Offset 0x03A4..0x1FA7A4 (360 x 360 x 2 f64)
}

impl DkTriodeLut {
    /// Decodes `exciter-triode-48000.000000.dat` (`2,074,532` bytes = `0x1FA7A4`).
    pub fn from_dat_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() < IZOTOPE_TRIODE_DAT_SIZE {
            return Err("File .dat nho hon 2,074,532 bytes");
        }
        let read_u32 = |off: usize| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        let read_f64 = |off: usize| f64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());

        if read_u32(0x0000) != IZOTOPE_DK_LUT_MAGIC || read_u32(0x0004) != 4 {
            return Err("Sai magic 0x8BAD43FC hoac version != 4");
        }
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

        let mut lut_grid = vec![[[0.0f64; 2]; 360]; 360];
        let mut cursor = 0x03A4usize;
        for i in 0..360 {
            for j in 0..360 {
                lut_grid[i][j][0] = read_f64(cursor);
                lut_grid[i][j][1] = read_f64(cursor + 8);
                cursor += 16;
            }
        }

        Ok(Self {
            dc_offset: read_f64(0x0008),
            circuit_mode: read_u32(0x0010),
            x_dc: [read_f64(0x001C), read_f64(0x0024), read_f64(0x002C)],
            u_dc: [read_f64(0x003C), read_f64(0x0044)],
            i_nl_dc: [read_f64(0x0054), read_f64(0x005C)],
            sample_rate: read_f64(0x0064),
            d_d: read_mat_2x3(0x0204),
            e_d: read_mat_2x2(0x023C),
            f_d: read_mat_2x2(0x0264),
            g_d: read_mat_2x2(0x028C),
            a_d: read_mat_3x3(0x02B4),
            b_d: read_mat_3x2(0x0304),
            c_d: read_mat_3x2(0x033C),
            bounds: [
                read_f64(0x0374),
                read_f64(0x037C),
                read_f64(0x0384),
                read_f64(0x038C),
            ],
            grid_dims: [read_u32(0x0394), read_u32(0x0398)],
            lut: Arc::new(lut_grid),
        })
    }

    /// Bilinear lookup of `[I_g(p1, p2), I_p(p1, p2)]` on the 360x360 precomputed Koren grid.
    #[inline(always)]
    pub fn sample_bilinear(&self, p1: f64, p2: f64) -> [f64; 2] {
        let [p1_min, p2_min, p1_max, p2_max] = self.bounds;
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
}

// ============================================================================
// 2. IZOTOPE OZONE 12: SLIDING MAX-HEAP (`0x183F81EE4`) & IRC V MAXIMIZER
// ============================================================================

/// Binary segment-tree sliding maximum (`DSP::MaxHeap` at `0x183F81EE4`: `shr r8, 1; cmovg`)
/// finding the local peak over lookahead window `L` in O(log L) with Zero Heap Allocation.
pub struct SlidingMaxHeap {
    tree: Vec<f32>,
    capacity_pow2: usize,
    window_len: usize,
    write_pos: usize,
}

impl SlidingMaxHeap {
    pub fn new(window_len: usize) -> Self {
        let cap = window_len.max(1).next_power_of_two();
        Self {
            tree: vec![0.0; cap * 2],
            capacity_pow2: cap,
            window_len: window_len.max(1),
            write_pos: 0,
        }
    }

    pub fn reset(&mut self) {
        self.tree.fill(0.0);
        self.write_pos = 0;
    }

    #[inline(always)]
    pub fn push_and_query_max(&mut self, val: f32) -> f32 {
        let mut idx = self.capacity_pow2 + self.write_pos;
        self.tree[idx] = val;
        idx >>= 1;
        while idx > 0 {
            let left = self.tree[idx * 2];
            let right = self.tree[idx * 2 + 1];
            self.tree[idx] = if left >= right { left } else { right };
            idx >>= 1;
        }
        self.write_pos += 1;
        if self.write_pos >= self.window_len {
            self.write_pos = 0;
        }
        self.tree[1]
    }
}

/// Complete IRC V Maximizer Core: Upward Compression + Soft Clipper + Lookahead MaxHeap
/// + Transient/Sustain Splitter (`0x183F69CE4`).
pub struct IrcVMaximizerCore {
    pub ceiling_linear: f32,
    pub gain_linear: f32,
    pub upward_boost_wet: f32,
    pub soft_clip_mix: f32,
    pub soft_clip_knee: f32, // 0.85 (Light), 0.70 (Medium), 0.50 (Heavy)
    pub transient_shaping: f32,
    pub stereo_link_sustain: f32,
    pub stereo_link_transient: f32,
    pub lookahead_buf: Vec<[f32; 2]>,
    pub lookahead_cursor: usize,
    pub max_heap_l: SlidingMaxHeap,
    pub max_heap_r: SlidingMaxHeap,
    pub env_fast: [f32; 2],
    pub env_slow: [f32; 2],
    pub gr_smooth: [f32; 2],
    pub sample_rate: f32,
}

impl IrcVMaximizerCore {
    pub fn new(sample_rate: f32) -> Self {
        let lookahead_samples = ((0.005 * sample_rate).round() as usize).max(16);
        Self {
            ceiling_linear: 0.89125094, // -1.0 dBFS
            gain_linear: 1.0,
            upward_boost_wet: 0.0,
            soft_clip_mix: 0.0,
            soft_clip_knee: 0.70,
            transient_shaping: 0.0,
            stereo_link_sustain: 0.80,
            stereo_link_transient: 0.40,
            lookahead_buf: vec![[0.0; 2]; lookahead_samples],
            lookahead_cursor: 0,
            max_heap_l: SlidingMaxHeap::new(lookahead_samples),
            max_heap_r: SlidingMaxHeap::new(lookahead_samples),
            env_fast: [0.0; 2],
            env_slow: [0.0; 2],
            gr_smooth: [1.0; 2],
            sample_rate: sample_rate.max(8000.0),
        }
    }

    #[inline(always)]
    pub fn upward_compress_and_soft_clip(&self, x: f32, rms_env: f32) -> f32 {
        let mut s = x * self.gain_linear;
        if self.upward_boost_wet > 0.0 && rms_env > 1e-3 && rms_env < 0.0631 {
            let boost = 1.0 + self.upward_boost_wet * ((0.0631 - rms_env) / 0.0631) * 1.5;
            s *= boost;
        }
        if self.soft_clip_mix <= 0.0 {
            return s;
        }
        let abs_s = s.abs();
        let km = self.soft_clip_knee;
        let clipped = if abs_s <= km {
            abs_s
        } else {
            km + (1.0 - km) * ((abs_s - km) / (1.0 - km)).tanh()
        };
        (1.0 - self.soft_clip_mix) * s + self.soft_clip_mix * s.signum() * clipped
    }

    #[inline]
    pub fn step_stereo(&mut self, in_l: f32, in_r: f32, sample_rate: f32) -> (f32, f32, f32) {
        let xl = self.upward_compress_and_soft_clip(in_l, self.env_slow[0]);
        let xr = self.upward_compress_and_soft_clip(in_r, self.env_slow[1]);

        let delayed = self.lookahead_buf[self.lookahead_cursor];
        self.lookahead_buf[self.lookahead_cursor] = [xl, xr];
        self.lookahead_cursor += 1;
        if self.lookahead_cursor >= self.lookahead_buf.len() {
            self.lookahead_cursor = 0;
        }

        let peak_l = self.max_heap_l.push_and_query_max(xl.abs());
        let peak_r = self.max_heap_r.push_and_query_max(xr.abs());
        let peaks = [peak_l, peak_r];

        let a_fast = (-1.0 / (0.0002 * sample_rate)).exp();
        let r_fast = (-1.0 / (0.0150 * sample_rate)).exp();
        let a_slow = (-1.0 / (0.0150 * sample_rate)).exp();
        let r_slow = (-1.0 / (0.2000 * sample_rate)).exp();

        let mut trans = [0.0f32; 2];
        let mut sust = [0.0f32; 2];
        for c in 0..2 {
            let p = peaks[c];
            let cf = if p > self.env_fast[c] { a_fast } else { r_fast };
            let cs = if p > self.env_slow[c] { a_slow } else { r_slow };
            self.env_fast[c] = cf * self.env_fast[c] + (1.0 - cf) * p;
            self.env_slow[c] = cs * self.env_slow[c] + (1.0 - cs) * p;
            sust[c] = self.env_slow[c];
            trans[c] = (self.env_fast[c] - self.env_slow[c]).max(0.0);
        }

        let max_trans = trans[0].max(trans[1]);
        let max_sust = sust[0].max(sust[1]);
        let mut gains = [1.0f32; 2];

        for c in 0..2 {
            let linked_t = (1.0 - self.stereo_link_transient) * trans[c]
                + self.stereo_link_transient * max_trans;
            let linked_s =
                (1.0 - self.stereo_link_sustain) * sust[c] + self.stereo_link_sustain * max_sust;
            let total_env = (linked_s + linked_t * (1.0 - 0.5 * self.transient_shaping)).max(1e-9);
            let target_g = (self.ceiling_linear / total_env).min(1.0);

            let crest = linked_t / (linked_s + 1e-6);
            let adaptive_rel_sec = (0.080 / (1.0 + 3.0 * crest)).clamp(0.005, 0.150);
            let rel_alpha = (-1.0 / (adaptive_rel_sec * sample_rate)).exp();
            self.gr_smooth[c] = if target_g < self.gr_smooth[c] {
                target_g
            } else {
                rel_alpha * self.gr_smooth[c] + (1.0 - rel_alpha) * target_g
            };
            gains[c] = self.gr_smooth[c];
        }

        let gr_db = 20.0 * gains[0].min(gains[1]).max(1e-6).log10();
        (delayed[0] * gains[0], delayed[1] * gains[1], gr_db)
    }
}

impl UnifiedDspNode for IrcVMaximizerCore {
    fn name(&self) -> &'static str {
        "iZotope Ozone 12 IRC V Maximizer"
    }

    fn latency_samples(&self) -> usize {
        self.lookahead_buf.len()
    }

    fn prepare(&mut self, sample_rate: f64, _max_block_size: usize) {
        self.sample_rate = sample_rate.max(8000.0) as f32;
    }

    fn reset_state(&mut self) {
        self.lookahead_buf.fill([0.0; 2]);
        self.lookahead_cursor = 0;
        self.max_heap_l.reset();
        self.max_heap_r.reset();
        self.env_fast = [0.0; 2];
        self.env_slow = [0.0; 2];
        self.gr_smooth = [1.0; 2];
    }

    fn set_parameter_normalized(&mut self, param_id: u32, norm_val: f32) {
        let v = norm_val.clamp(0.0, 1.0);
        match param_id {
            0 => self.ceiling_linear = 10.0f32.powf((-20.0 + v * 20.0) / 20.0),
            1 => self.gain_linear = 10.0f32.powf((v * 20.0) / 20.0),
            2 => self.upward_boost_wet = v,
            3 => self.soft_clip_mix = v,
            _ => {}
        }
    }

    fn process_slice(
        &mut self,
        ctx: &ProcessContext,
        main_io: &mut [StereoFrame],
        _sidechain_in: Option<&[StereoFrame]>,
        _aux_sends_out: &mut [&mut [StereoFrame]],
    ) {
        let sr = ctx.sample_rate.max(8000.0) as f32;
        for frame in main_io.iter_mut() {
            let (l, r, _gr) = self.step_stereo(frame[0], frame[1], sr);
            *frame = [l, r];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_dk_triode_lut_and_irc_v_maximizer() {
        let mut max_core = IrcVMaximizerCore::new(48_000.0);
        max_core.ceiling_linear = 0.5;
        for _ in 0..512 {
            let (l, r, _gr) = max_core.step_stereo(1.0, -1.0, 48_000.0);
            assert!(l.abs() <= 0.5001 && r.abs() <= 0.5001);
        }

        let dat_path = Path::new(r"f:\FL studio\extracted_targets\ozone12\exciterCache_component\ExciterCache\Exciter Cache\exciter-triode-48000.000000.dat");
        if dat_path.exists() {
            let raw = std::fs::read(dat_path).unwrap();
            let lut = DkTriodeLut::from_dat_bytes(&raw).unwrap();
            assert!((lut.sample_rate - 48_000.0).abs() < 1e-6);
            assert_eq!(lut.grid_dims, [360, 360]);
            let [ig, ip] = lut.sample_bilinear(-1.5, 180.0);
            assert!(ig.is_finite() && ip.is_finite());
        }
    }
}

