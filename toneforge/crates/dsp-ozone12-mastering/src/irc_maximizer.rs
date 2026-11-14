//! iZotope Ozone 12 IRC I–V Multi-Band True-Peak Maximizer (`DSP::OzoneMaximizer` `0x185736808`,
//! `DSP::Shared::IRC::Maximizer` `0x185767808`, `DSP::MaxHeap` `0x183F81EE4`, `SoftClipper` `0x1836839A0`,
//! `IRC::TruePeakDetector` `0x183C2AEE0`, `MBIT+ Dither` `0x183F63430`).
//!
//! Implements:
//! 1. `SlidingMaxHeap`: Binary segment-tree sliding maximum (`shr r8, 1; cmovg`) over the lookahead window
//!    with $O(\log L)$ complexity and Zero Heap Allocation.
//! 2. `4x` Linear-Phase Polyphase FIR Oversampled (`128`-tap Kaiser $\beta = 9.0$) Degree-10 $C^5$
//!    Maximally-Flat Polynomial `SoftClipper` (`Light`: $w=0.19, G=1.19$; `Medium`/`Moderate`:
//!    $w=0.525, G=1.525$; `Heavy`: $w=1.0, G=2.0$) with `5 ms` 1-pole parameter smoothing.
//! 3. 4-Band Phase-Compensated Linkwitz-Riley Crossover (`[150.0, 900.0, 5000.0] Hz` in `IRC V` mode)
//!    with `5.0 ms` fast / `50.0 ms` slow transient-sustain decomposition (`0x18359D2B0`).
//! 4. Dual True-Peak Architecture (`4x` 48-tap `ITU-R BS.1770-4` Table 3 FIR + `9x` 288-tap Kaiser
//!    $\beta = 9.0$ polyphase FIR with analytical 4-point Catmull-Rom cubic extremum solver
//!    $3at^2 + 2bt + c = 0$ and `1.02585` `+0.2217 dB` ISP safety headroom) + stateful continuous
//!    across-block lookahead True-Peak safety verifier and dither headroom clamping.
//! 5. Psychoacoustic `MbitPlusDither` (`8 / 16 / 20 / 24-bit` quantization + noise shaping).

use crate::chain_element::{
    DEFAULT_CROSSOVER_FREQS_HZ, LinkwitzRiley4BandCrossover, OZONE_MULTIBAND_COUNT, OzoneDspModule,
};
use crate::lufs_true_peak_meter::{
    ITU_BS1770_4_POLYPHASE_TAPS, LufsTruePeakMeter, TruePeakDetector4x,
};
use crate::oversampler::{
    KAISER_BETA, OversampleFactor, OversamplerFilterMode, PolyphaseHalfBandOversampler, bessel_i0,
};
use daw_core_graph::{ProcessContext, StereoFrame, UnifiedDspNode, VisualTelemetryFrame};
use std::f64::consts::PI;
use std::sync::OnceLock;

const TP_PREDICTIVE_WINDOW: usize = 24;
const TP_PAST_HISTORY: usize = 11;
const ISP_9X_PHASES: usize = 9;
const ISP_9X_TAPS_PER_PHASE: usize = 32;

/// Exact `IRC V` 4-band crossover frequencies (`0x183583410`: `150 Hz`, `900 Hz`, `5000 Hz`).
pub const IRC5_CROSSOVER_FREQS_HZ: [f64; 3] = [150.0, 900.0, 5000.0];

/// Ozone 12 `IRC V` ISP conservative headroom factor (`1.02585f` = `+0.2217 dB` @ `0x1835A0B10`).
pub const OZONE12_ISP_SAFETY_FACTOR: f32 = 1.025_85;

/// Binary segment-tree sliding maximum (`DSP::MaxHeap` at `0x183F81EE4`: `shr r8, 1; cmovg`)
/// finding the local peak over lookahead window `L` in `O(log L)` with Zero Heap Allocation.
#[derive(Clone, Debug)]
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

    #[inline]
    pub fn clear(&mut self) {
        self.reset();
    }

    #[inline]
    pub fn window_len(&self) -> usize {
        self.window_len
    }

    #[inline(always)]
    pub fn push_and_query_max(&mut self, val: f32) -> f32 {
        let clean_val = if val.is_nan() { 0.0 } else { val };
        let mut idx = self.capacity_pow2 + self.write_pos;
        self.tree[idx] = clean_val;
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

/// Intelligent Release Control algorithm modes (`0x183F69CB2`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IrcMode {
    /// Mode 0 (`0x183F6AF90`): Crest-factor program-dependent adaptive release.
    IrcI = 0,
    /// Mode 1 (`0x183F6B1D0`): Sharp transient-preserving lookahead limiter.
    IrcII = 1,
    /// Mode 2 (`0x185734588`): Psychoacoustic spectral envelope limiter.
    IrcIII = 2,
    /// Mode 3: Multi-band psychoacoustic limiter with character profiles (`Classic`, `Modern`, `Transient`).
    IrcIV = 3,
    /// Mode 4 (`0x183F69CE4`): Ozone 12 4-band (`150/900/5000 Hz`) transient/sustain (`5 ms / 50 ms`) + True-Peak brickwall.
    IrcV = 4,
    /// Mode 5 (`0x183F6AAA0`): Low-latency minimum-phase feedback/short-lookahead limiter.
    IrcLowLatency = 5,
}

impl IrcMode {
    #[inline]
    pub fn from_index(idx: u32) -> Self {
        match idx % 6 {
            0 => Self::IrcI,
            1 => Self::IrcII,
            2 => Self::IrcIII,
            3 => Self::IrcIV,
            4 => Self::IrcV,
            _ => Self::IrcLowLatency,
        }
    }

    #[inline]
    pub fn is_multiband(self) -> bool {
        matches!(self, Self::IrcIII | Self::IrcIV | Self::IrcV)
    }
}

/// IRC Character Sub-Modes (`Max Character` in `AutomatableParamList_Maximizer.txt` & XML presets).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IrcCharacter {
    Pumping = 0,
    Balanced = 1,
    Crisp = 2,
    Classic = 3,
    Modern = 4,
    Transient = 5,
}

impl IrcCharacter {
    #[inline]
    pub fn from_index(idx: u32) -> Self {
        match idx % 6 {
            0 => Self::Pumping,
            1 => Self::Balanced,
            2 => Self::Crisp,
            3 => Self::Classic,
            4 => Self::Modern,
            _ => Self::Transient,
        }
    }

    /// Base release time multiplier and transient preservation factor for this character mode.
    #[inline]
    pub fn ballistics_factors(self) -> (f32, f32) {
        match self {
            Self::Pumping => (0.45, 0.75),
            Self::Balanced => (1.00, 1.00),
            Self::Crisp => (0.70, 1.35),
            Self::Classic => (1.45, 0.85),
            Self::Modern => (0.90, 1.15),
            Self::Transient => (0.60, 1.50),
        }
    }
}

/// Pre-Limiter Soft Clipper Knee Mode (`SoftClipMode` @ `0x18365B020`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoftClipMode {
    /// Light (`w = 0.19`, `G_drive = 1.19`).
    Light = 0,
    /// Moderate / Medium (`w = 0.525`, `G_drive = 1.525`).
    Medium = 1,
    /// Heavy (`w = 1.0`, `G_drive = 2.0`).
    Heavy = 2,
}

impl SoftClipMode {
    /// Alias matching Ozone 12 UI string `"Moderate"` (`enum = 1`).
    #[allow(non_upper_case_globals)]
    pub const Moderate: Self = Self::Medium;

    /// Legacy knee threshold `[0.0, 1.0]` (`1 - w` mapped to legacy range).
    #[inline]
    pub fn knee_threshold(self) -> f32 {
        match self {
            Self::Light => 0.81,
            Self::Medium => 0.475,
            Self::Heavy => 0.0,
        }
    }

    /// Exact Ozone 12 knee half-width `w` (`0x18365B020`: `0.19`, `0.525`, `1.0`).
    #[inline]
    pub const fn knee_half_width(self) -> f64 {
        match self {
            Self::Light => 0.19,
            Self::Medium => 0.525,
            Self::Heavy => 1.0,
        }
    }

    /// Exact Ozone 12 auto-drive `G_drive = 1.0 + w` (`1.19`, `1.525`, `2.0`).
    #[inline]
    pub const fn auto_drive(self) -> f64 {
        1.0 + self.knee_half_width()
    }

    #[inline]
    pub fn from_knee_field(knee: f32) -> Self {
        if knee >= 0.78 {
            Self::Light
        } else if knee <= 0.25 {
            Self::Heavy
        } else {
            Self::Medium
        }
    }
}

/// Evaluates Ozone 12's unique degree-10 $C^5$-continuous maximally-flat polynomial (`order = 4`
/// in `buildMaxFlatPoly` `0x183685D10`) on $t \in [0, 2]$:
///
/// $$P_4(t) = t - \frac{21}{32} t^6 + \frac{15}{16} t^7 - \frac{135}{256} t^8 + \frac{35}{256} t^9 - \frac{7}{512} t^{10}$$
///
/// Satisfies $P_4(0)=0, P_4'(0)=1, P_4^{(2..5)}(0)=0$ and $P_4(2)=1, P_4^{(1..5)}(2)=0$.
#[inline(always)]
pub fn eval_maxflat_c5_poly_p4(t: f64) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }
    if t >= 2.0 {
        return 1.0;
    }
    let t2 = t * t;
    let t6 = t2 * t2 * t2;
    const C6: f64 = -21.0 / 32.0; // -0.65625
    const C7: f64 = 15.0 / 16.0; // +0.9375
    const C8: f64 = -135.0 / 256.0; // -0.52734375
    const C9: f64 = 35.0 / 256.0; // +0.13671875
    const C10: f64 = -7.0 / 512.0; // -0.013671875
    let tail = C6 + t * (C7 + t * (C8 + t * (C9 + t * C10)));
    t + t6 * tail
}

/// Evaluates Ozone 12's rescaled $C^5$ soft-clip curve $S_w(u)$ on knee $[1 - w, 1 + w]$ (`0x183683390` & `0x1836838B0`).
#[inline(always)]
pub fn eval_ozone12_soft_clip_curve(u: f64, w: f64) -> f64 {
    let w_clamped = w.clamp(1.0e-4, 1.0);
    let abs_u = u.abs();
    let knee_lo = 1.0 - w_clamped;
    let knee_hi = 1.0 + w_clamped;
    let mag = if abs_u <= knee_lo {
        abs_u
    } else if abs_u >= knee_hi {
        1.0
    } else {
        let t = (abs_u - knee_lo) / w_clamped;
        knee_lo + w_clamped * eval_maxflat_c5_poly_p4(t)
    };
    u.signum() * mag
}

// ============================================================================
// 9x KAISER POLYPHASE + CATMULL-ROM CUBIC INTER-SAMPLE EXTREMUM DETECTOR
// ============================================================================

/// Solves `p'(t) = 3*a*t^2 + 2*b*t + c = 0` for `t in [0, 1]` on the 4-point Catmull-Rom cubic
/// interpolating `(y_m1, y_0, y_1, y_2)` (`0x183C2AEE0`) and returns `max(|y_0|, |y_1|, |p(t*)|)`.
#[inline(always)]
pub fn catmull_rom_cubic_max_abs(y_m1: f32, y_0: f32, y_1: f32, y_2: f32) -> f32 {
    let p0 = y_0 as f64;
    let p1 = y_1 as f64;
    let pm1 = y_m1 as f64;
    let p2 = y_2 as f64;

    let mut max_abs = p0.abs().max(p1.abs());

    let a = -0.5 * pm1 + 1.5 * p0 - 1.5 * p1 + 0.5 * p2;
    let b = pm1 - 2.5 * p0 + 2.0 * p1 - 0.5 * p2;
    let c = -0.5 * pm1 + 0.5 * p1;
    let d = p0;

    if a.abs() < 1.0e-12 {
        if b.abs() > 1.0e-12 {
            let t = -c / (2.0 * b);
            if t > 0.0 && t < 1.0 {
                let val = ((a * t + b) * t + c) * t + d;
                max_abs = max_abs.max(val.abs());
            }
        }
    } else {
        let disc = b * b - 3.0 * a * c;
        if disc >= 0.0 {
            let root = disc.sqrt();
            let inv_3a = 1.0 / (3.0 * a);
            for t in [(-b - root) * inv_3a, (-b + root) * inv_3a] {
                if t > 0.0 && t < 1.0 {
                    let val = ((a * t + b) * t + c) * t + d;
                    max_abs = max_abs.max(val.abs());
                }
            }
        }
    }
    max_abs as f32
}

fn kaiser_9x_polyphase_table() -> &'static [[f32; ISP_9X_TAPS_PER_PHASE]; ISP_9X_PHASES] {
    static TABLE: OnceLock<[[f32; ISP_9X_TAPS_PER_PHASE]; ISP_9X_PHASES]> = OnceLock::new();
    TABLE.get_or_init(|| {
        const TOTAL: usize = ISP_9X_PHASES * ISP_9X_TAPS_PER_PHASE;
        let center = 0.5 * ((TOTAL - 1) as f64);
        let fc = 0.5 / (ISP_9X_PHASES as f64);
        let i0_beta = bessel_i0(KAISER_BETA);

        let mut proto = [0.0_f64; TOTAL];
        for (m, slot) in proto.iter_mut().enumerate() {
            let x = (m as f64) - center;
            let norm = x / center;
            let win = bessel_i0(KAISER_BETA * (1.0 - norm * norm).max(0.0).sqrt()) / i0_beta;
            let sinc = if x.abs() < 1.0e-12 {
                2.0 * fc
            } else {
                (2.0 * PI * fc * x).sin() / (PI * x)
            };
            *slot = sinc * win;
        }

        let mut phases = [[0.0_f32; ISP_9X_TAPS_PER_PHASE]; ISP_9X_PHASES];
        for p in 0..ISP_9X_PHASES {
            let mut sum = 0.0_f64;
            for k in 0..ISP_9X_TAPS_PER_PHASE {
                sum += proto[k * ISP_9X_PHASES + p];
            }
            let inv = if sum.abs() > 1.0e-15 { 1.0 / sum } else { 1.0 };
            for k in 0..ISP_9X_TAPS_PER_PHASE {
                phases[p][k] = (proto[k * ISP_9X_PHASES + p] * inv) as f32;
            }
        }
        phases
    })
}

/// Ozone 12 `IRC::TruePeakDetector` (`0x183C2AC70` & `0x183C2AEE0`): `9x` Kaiser ($\beta=9.0$)
/// polyphase FIR + analytical Catmull-Rom cubic inter-sample extremum solver (`0` heap allocations).
#[derive(Clone, Copy, Debug)]
struct TruePeakDetector9xCubic {
    hist: [[f32; ISP_9X_TAPS_PER_PHASE]; 2],
    pos: [usize; 2],
    prev_sub: [[f32; 3]; 2],
}

impl TruePeakDetector9xCubic {
    fn new() -> Self {
        let _ = kaiser_9x_polyphase_table();
        Self {
            hist: [[0.0; ISP_9X_TAPS_PER_PHASE]; 2],
            pos: [0; 2],
            prev_sub: [[0.0; 3]; 2],
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.hist = [[0.0; ISP_9X_TAPS_PER_PHASE]; 2];
        self.pos = [0; 2];
        self.prev_sub = [[0.0; 3]; 2];
    }

    #[inline(always)]
    fn step_channel(&mut self, ch: usize, x: f32) -> f32 {
        let table = kaiser_9x_polyphase_table();
        let p_idx = if self.pos[ch] == 0 {
            ISP_9X_TAPS_PER_PHASE - 1
        } else {
            self.pos[ch] - 1
        };
        self.pos[ch] = p_idx;
        self.hist[ch][p_idx] = x;

        let mut subs = [0.0_f32; ISP_9X_PHASES];
        for p in 0..ISP_9X_PHASES {
            let taps = &table[p];
            let mut acc = 0.0_f32;
            let mut idx = p_idx;
            for &tap in taps.iter() {
                acc += tap * self.hist[ch][idx];
                idx += 1;
                if idx == ISP_9X_TAPS_PER_PHASE {
                    idx = 0;
                }
            }
            subs[p] = acc;
        }

        // Form continuous 12-sub-sample sequence [prev_0, prev_1, prev_2, subs_0..subs_8]
        let seq = [
            self.prev_sub[ch][0],
            self.prev_sub[ch][1],
            self.prev_sub[ch][2],
            subs[0],
            subs[1],
            subs[2],
            subs[3],
            subs[4],
            subs[5],
            subs[6],
            subs[7],
            subs[8],
        ];
        self.prev_sub[ch] = [subs[6], subs[7], subs[8]];

        let mut max_tp = x.abs();
        for w in 0..ISP_9X_PHASES {
            let seg_max = catmull_rom_cubic_max_abs(seq[w], seq[w + 1], seq[w + 2], seq[w + 3]);
            if seg_max > max_tp {
                max_tp = seg_max;
            }
        }
        max_tp
    }

    #[inline(always)]
    fn step_stereo(&mut self, xl: f32, xr: f32) -> [f32; 2] {
        [self.step_channel(0, xl), self.step_channel(1, xr)]
    }
}

/// Psychoacoustic MBIT+ Dither Generator (`0x183F63430`).
#[derive(Clone, Copy, Debug)]
pub struct MbitPlusDither {
    pub enabled: bool,
    pub bit_depth: u8, // 8, 16, 20, or 24
    pub dither_amount: f32,
    pub noise_shaping: bool,
    rng_state: u64,
    prev_pdf: [f32; 2],
    err_hist: [[f32; 2]; 2],
}

impl Default for MbitPlusDither {
    fn default() -> Self {
        Self::new()
    }
}

impl MbitPlusDither {
    pub const fn new() -> Self {
        Self {
            enabled: false,
            bit_depth: 24,
            dither_amount: 1.0,
            noise_shaping: true,
            rng_state: 0x853C_49E6_748F_EA9B,
            prev_pdf: [0.0; 2],
            err_hist: [[0.0; 2]; 2],
        }
    }

    pub fn reset(&mut self) {
        self.prev_pdf = [0.0; 2];
        self.err_hist = [[0.0; 2]; 2];
    }

    #[inline(always)]
    fn next_uniform(&mut self) -> f32 {
        let mut x = self.rng_state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng_state = x;
        let u = (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as u32;
        (u as f32) * (1.0 / 16_777_216.0) - 0.5
    }

    #[inline(always)]
    pub fn process_sample(&mut self, ch: usize, x: f32) -> f32 {
        self.process_sample_with_ceiling(ch, x, 1.0)
    }

    /// Quantizes with TPDF + 2nd-order psychoacoustic noise shaping while strictly reserving
    /// `2 LSB` headroom inside `[-ceiling, +ceiling]` so quantization/dither can never push
    /// samples or inter-sample peaks above `ceiling`.
    #[inline(always)]
    pub fn process_sample_with_ceiling(&mut self, ch: usize, x: f32, ceiling: f32) -> f32 {
        if !self.enabled {
            return x;
        }
        let bits = self.bit_depth.clamp(8, 24) as u32;
        let quant_scale = ((1u32 << (bits - 1)) - 1) as f32;
        let inv_quant = 1.0 / quant_scale;
        let max_code = (ceiling * quant_scale - 1.0).floor().max(1.0);
        let safe_in_lim = (max_code - 1.5 * self.dither_amount.clamp(0.0, 2.0)).max(0.0) * inv_quant;
        let x_safe = x.clamp(-safe_in_lim, safe_in_lim);

        let r = self.next_uniform();
        let tpdf = (r - self.prev_pdf[ch]) * self.dither_amount;
        self.prev_pdf[ch] = r;

        let shaped_in = if self.noise_shaping {
            x_safe + (1.55 * self.err_hist[ch][0] - 0.62 * self.err_hist[ch][1]) * inv_quant
        } else {
            x_safe
        };

        let scaled = shaped_in * quant_scale + tpdf;
        let quantized = scaled.round().clamp(-max_code, max_code);
        let err = (shaped_in * quant_scale) - quantized;
        self.err_hist[ch][1] = self.err_hist[ch][0];
        self.err_hist[ch][0] = err.clamp(-2.0, 2.0);
        (quantized * inv_quant).clamp(-ceiling, ceiling)
    }
}

/// Per-band transient/sustain envelope follower (`0x18359D2B0`: `5 ms` fast / `50 ms` slow).
#[derive(Clone, Copy, Debug, Default)]
struct BandEnvelopeLimiter {
    env_fast: [f32; 2],
    env_slow: [f32; 2],
    gr_sust: [f32; 2],
    gr_trans: [f32; 2],
}

impl BandEnvelopeLimiter {
    fn new() -> Self {
        Self {
            env_fast: [0.0; 2],
            env_slow: [0.0; 2],
            gr_sust: [1.0; 2],
            gr_trans: [1.0; 2],
        }
    }

    fn reset(&mut self) {
        self.env_fast = [0.0; 2];
        self.env_slow = [0.0; 2];
        self.gr_sust = [1.0; 2];
        self.gr_trans = [1.0; 2];
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn step_band(
        &mut self,
        band_in: [f32; 2],
        band_ceiling: f32,
        a_fast: f32,
        r_fast: f32,
        a_slow: f32,
        r_slow: f32,
        stereo_link_trans: f32,
        stereo_link_sust: f32,
        transient_shaping: f32,
        rel_scale: f32,
        trans_preserve: f32,
        band_idx: usize,
        sample_rate: f32,
    ) -> [f32; 2] {
        let mut trans_mask = [0.0f32; 2];
        let mut env_f = [0.0f32; 2];
        let mut env_s = [0.0f32; 2];
        for c in 0..2 {
            let p = band_in[c].abs();
            let cf = if p > self.env_fast[c] { a_fast } else { r_fast };
            let cs = if p > self.env_slow[c] { a_slow } else { r_slow };
            self.env_fast[c] = cf * self.env_fast[c] + (1.0 - cf) * p;
            self.env_slow[c] = cs * self.env_slow[c] + (1.0 - cs) * p;
            env_f[c] = self.env_fast[c];
            env_s[c] = self.env_slow[c];
            // Exact IRC V transient ratio mask (0x18359D2B0): clamp(1 - E_slow / (E_fast + eps), 0, 1)
            trans_mask[c] = (1.0 - env_s[c] / (env_f[c] + 1.0e-7)).clamp(0.0, 1.0);
        }

        let max_f = env_f[0].max(env_f[1]);
        let max_s = env_s[0].max(env_s[1]);
        // High bands (900-5000 Hz, 5000+ Hz) recover faster than Sub/Low (< 150 Hz)
        let band_speed_factor = match band_idx {
            0 => 1.35,
            1 => 1.05,
            2 => 0.78,
            _ => 0.62,
        };

        let mut out = [0.0f32; 2];
        for c in 0..2 {
            let x_trans = band_in[c] * trans_mask[c];
            let x_sust = band_in[c] - x_trans;

            let lf = (1.0 - stereo_link_trans) * env_f[c] + stereo_link_trans * max_f;
            let ls = (1.0 - stereo_link_sust) * env_s[c] + stereo_link_sust * max_s;
            let lt = (lf - ls).max(0.0);

            let target_gs = (band_ceiling / ls.max(1.0e-9)).min(1.0);
            let trans_eff =
                (ls + lt * (1.0 - 0.35 * transient_shaping) / trans_preserve).max(1.0e-9);
            let target_gt = (band_ceiling * 1.08 / trans_eff).min(1.0);

            let crest = lt / (ls + 1.0e-6);
            let adaptive_rel_sec =
                ((0.085 * rel_scale * band_speed_factor) / (1.0 + 2.5 * crest)).clamp(0.004, 0.250);
            let rel_alpha = (-1.0 / (adaptive_rel_sec * sample_rate)).exp();

            self.gr_sust[c] = if target_gs < self.gr_sust[c] {
                0.45 * self.gr_sust[c] + 0.55 * target_gs
            } else {
                rel_alpha * self.gr_sust[c] + (1.0 - rel_alpha) * target_gs
            };
            self.gr_trans[c] = if target_gt < self.gr_trans[c] {
                0.35 * self.gr_trans[c] + 0.65 * target_gt
            } else {
                rel_alpha * self.gr_trans[c] + (1.0 - rel_alpha) * target_gt
            };

            out[c] = x_sust * self.gr_sust[c] + x_trans * self.gr_trans[c];
        }
        out
    }
}

/// Core single-band + lookahead Maximizer (`IrcVMaximizerCore`) with `SlidingMaxHeap`,
/// `4x` Linear-Phase Polyphase FIR $C^5$ Polynomial `SoftClipper`, `9x` Catmull-Rom Cubic + `4x`
/// ITU-R BS.1770-4 True-Peak Sidechain, and Stateful Continuous True-Peak Brickwall Finalizer.
#[derive(Clone, Debug)]
pub struct IrcVMaximizerCore {
    pub ceiling_linear: f32,
    pub gain_linear: f32,
    pub upward_boost_wet: f32,
    pub soft_clip_mix: f32,
    pub soft_clip_knee: f32,
    pub soft_clip_mode: SoftClipMode,
    /// Enables `4x` `LinearPhaseFirKaiser128` oversampling in `SoftClipper` (default `true`).
    pub soft_clip_oversample_4x: bool,
    pub transient_shaping: f32,
    pub stereo_link_sustain: f32,
    pub stereo_link_transient: f32,
    pub prevent_intersample_clipping: bool,
    pub lookahead_buf: Vec<[f32; 2]>,
    pub lookahead_cursor: usize,
    pub max_heap_l: SlidingMaxHeap,
    pub max_heap_r: SlidingMaxHeap,
    pub env_fast: [f32; 2],
    pub env_slow: [f32; 2],
    pub gr_smooth: [f32; 2],
    gr_attack_smooth: [f32; 2],
    pub sample_rate: f32,
    soft_clip_oversampler: PolyphaseHalfBandOversampler,
    sc_ceil_smooth: [f64; 2],
    sc_mix_smooth: [f64; 2],
    sc_drive_smooth: [f64; 2],
    sc_smooth_initialized: bool,
    tp_in_detector: TruePeakDetector4x,
    tp_9x_detector: TruePeakDetector9xCubic,
    tp_past_emitted: [[f32; TP_PAST_HISTORY]; 2],
    tp_window: [[f32; TP_PREDICTIVE_WINDOW]; 2],
}

impl IrcVMaximizerCore {
    pub fn new(sample_rate: f32) -> Self {
        let sr = sample_rate.max(8000.0);
        let lookahead_samples = ((0.005 * sr).round() as usize).max(32);
        let default_ceil = 0.891_250_94_f32;
        let default_mode = SoftClipMode::Medium;
        Self {
            ceiling_linear: default_ceil,
            gain_linear: 1.0,
            upward_boost_wet: 0.0,
            soft_clip_mix: 0.0,
            soft_clip_knee: default_mode.knee_threshold(),
            soft_clip_mode: default_mode,
            soft_clip_oversample_4x: true,
            transient_shaping: 0.0,
            stereo_link_sustain: 0.80,
            stereo_link_transient: 0.40,
            prevent_intersample_clipping: true,
            lookahead_buf: vec![[0.0; 2]; lookahead_samples],
            lookahead_cursor: 0,
            max_heap_l: SlidingMaxHeap::new(lookahead_samples),
            max_heap_r: SlidingMaxHeap::new(lookahead_samples),
            env_fast: [0.0; 2],
            env_slow: [0.0; 2],
            gr_smooth: [1.0; 2],
            gr_attack_smooth: [1.0; 2],
            sample_rate: sr,
            soft_clip_oversampler: PolyphaseHalfBandOversampler::new(
                OversamplerFilterMode::LinearPhaseFirKaiser128,
                OversampleFactor::X4,
            ),
            sc_ceil_smooth: [default_ceil as f64; 2],
            sc_mix_smooth: [0.0; 2],
            sc_drive_smooth: [default_mode.auto_drive(); 2],
            sc_smooth_initialized: false,
            tp_in_detector: TruePeakDetector4x::new(),
            tp_9x_detector: TruePeakDetector9xCubic::new(),
            tp_past_emitted: [[0.0; TP_PAST_HISTORY]; 2],
            tp_window: [[0.0; TP_PREDICTIVE_WINDOW]; 2],
        }
    }

    pub fn reset(&mut self) {
        self.lookahead_buf.fill([0.0; 2]);
        self.lookahead_cursor = 0;
        self.max_heap_l.reset();
        self.max_heap_r.reset();
        self.env_fast = [0.0; 2];
        self.env_slow = [0.0; 2];
        self.gr_smooth = [1.0; 2];
        self.gr_attack_smooth = [1.0; 2];
        self.soft_clip_oversampler.reset();
        self.sc_ceil_smooth = [self.ceiling_linear.max(1.0e-4) as f64; 2];
        self.sc_mix_smooth = [self.soft_clip_mix.clamp(0.0, 1.0) as f64; 2];
        self.sc_drive_smooth = [self.soft_clip_mode.auto_drive(); 2];
        self.sc_smooth_initialized = false;
        self.tp_in_detector.reset();
        self.tp_9x_detector.reset();
        self.tp_past_emitted = [[0.0; TP_PAST_HISTORY]; 2];
        self.tp_window = [[0.0; TP_PREDICTIVE_WINDOW]; 2];
    }

    /// Stateless per-sample evaluation of Upward Compression + Ozone 12 Degree-10 $C^5$ Polynomial
    /// Soft-Clipper (`0x183685D10` & `0x1836839A0`). Kept `&self` for API compatibility; stateful
    /// `4x` Linear-Phase Polyphase FIR oversampling + `5 ms` parameter smoothing runs in
    /// [`Self::step_soft_clip_stereo`].
    #[inline(always)]
    pub fn upward_compress_and_soft_clip(&self, x: f32, rms_env: f32) -> f32 {
        let mut s = (x * self.gain_linear) as f64;
        if self.upward_boost_wet > 0.0 && rms_env > 1.0e-3 && rms_env < 0.0631 {
            let boost =
                1.0 + (self.upward_boost_wet as f64) * (((0.0631 - rms_env) / 0.0631) as f64) * 1.5;
            s *= boost;
        }
        let mix = (self.soft_clip_mix as f64).clamp(0.0, 1.0);
        if mix <= 0.0 {
            return s as f32;
        }
        let ceil = (self.ceiling_linear.max(1.0e-4)) as f64;
        let w = self.soft_clip_mode.knee_half_width();
        let drive = 1.0 + w;
        let u = (drive * s) / ceil;
        let clipped = ceil * eval_ozone12_soft_clip_curve(u, w);
        ((1.0 - mix) * s + mix * clipped) as f32
    }

    /// Executes Upward Compression + `4x` Linear-Phase Polyphase FIR (`128`-tap Kaiser $\beta = 9.0$)
    /// Oversampled Degree-10 $C^5$ Polynomial `SoftClipper` (`0x1836839A0`) with `5.0 ms` 1-pole
    /// parameter smoothing and **0 heap allocations**.
    #[inline(always)]
    pub fn step_soft_clip_stereo(&mut self, in_l: f32, in_r: f32, sample_rate: f32) -> [f32; 2] {
        let mut sl = (in_l * self.gain_linear) as f64;
        let mut sr_in = (in_r * self.gain_linear) as f64;

        if self.upward_boost_wet > 0.0 {
            for (s_val, env_val) in [(&mut sl, self.env_slow[0]), (&mut sr_in, self.env_slow[1])] {
                if env_val > 1.0e-3 && env_val < 0.0631 {
                    let boost = 1.0
                        + (self.upward_boost_wet as f64)
                            * (((0.0631 - env_val) / 0.0631) as f64)
                            * 1.5;
                    *s_val *= boost;
                }
            }
        }

        let target_mix = (self.soft_clip_mix as f64).clamp(0.0, 1.0);
        let target_ceil = (self.ceiling_linear.max(1.0e-4)) as f64;
        let target_w = self.soft_clip_mode.knee_half_width();
        let target_drive = 1.0 + target_w;

        if !self.sc_smooth_initialized {
            self.sc_ceil_smooth = [target_ceil; 2];
            self.sc_mix_smooth = [target_mix; 2];
            self.sc_drive_smooth = [target_drive; 2];
            self.sc_smooth_initialized = true;
        }

        if target_mix <= 1.0e-7 && self.sc_mix_smooth[0] <= 1.0e-7 {
            return [sl as f32, sr_in as f32];
        }

        let target_factor = if self.soft_clip_oversample_4x {
            OversampleFactor::X4
        } else {
            OversampleFactor::X1
        };
        if self.soft_clip_oversampler.factor() != target_factor {
            self.soft_clip_oversampler.set_factor(target_factor);
        }

        let os_sr = (sample_rate.max(8000.0) as f64) * (target_factor.as_usize() as f64);
        // 1-pole 5.0 ms smoother at oversampled rate (`0x1836839A0`)
        let alpha = 1.0 - (-1.0 / (0.005 * os_sr)).exp();

        let ceil_sm = &mut self.sc_ceil_smooth;
        let mix_sm = &mut self.sc_mix_smooth;
        let drive_sm = &mut self.sc_drive_smooth;

        let y = self
            .soft_clip_oversampler
            .process_stereo_frame([sl, sr_in], |sub| {
                let mut out = [0.0_f64; 2];
                for ch in 0..2 {
                    ceil_sm[ch] += alpha * (target_ceil - ceil_sm[ch]);
                    mix_sm[ch] += alpha * (target_mix - mix_sm[ch]);
                    drive_sm[ch] += alpha * (target_drive - drive_sm[ch]);

                    let ceil_l = ceil_sm[ch].max(1.0e-5);
                    let mix = mix_sm[ch];
                    let drive = drive_sm[ch];
                    let w = (drive - 1.0).clamp(1.0e-4, 1.0);

                    let x = sub[ch];
                    let u = (drive * x) / ceil_l;
                    let s = eval_ozone12_soft_clip_curve(u, w);
                    out[ch] = mix * (ceil_l * s) + (1.0 - mix) * x;
                }
                out
            });
        [y[0] as f32, y[1] as f32]
    }

    /// Stateful continuous across-block True-Peak lookahead verifier combining:
    /// 1. Exact 12-tap `ITU-R BS.1770-4` Table 3 4-phase convolution over `[past_emitted[11], wbuf[24]]`
    ///    (never skipping indices `0..5`!)
    /// 2. 4-point Catmull-Rom cubic inter-sample extremum check (`3at^2 + 2bt + c = 0`) across the
    ///    reconstructed sub-samples
    /// 3. Exact closed-interval clamp on the emitted sample `wbuf[0]` so `TruePeakDetector4x` is
    ///    mathematically guaranteed `<= ceiling * 0.996` (`<= ceiling_dbtp - 0.035 dB`).
    #[inline(always)]
    fn push_and_prepare_tp_window_channel(
        past: &[f32; TP_PAST_HISTORY],
        wbuf: &mut [f32; TP_PREDICTIVE_WINDOW],
        new_sample: f32,
        ceiling: f32,
    ) -> f32 {
        let soft_limit = ceiling * 0.990;
        let clamped_in = new_sample.clamp(-soft_limit, soft_limit);

        // Shift incoming sample into the end of the 24-sample future lookahead window BEFORE
        // solving across `wbuf[0..24]` (`wbuf[0]` is the candidate to emit on THIS step!).
        // Wait: to keep exact latency == `TP_PREDICTIVE_WINDOW` (`24` samples), extract `cand_0`
        // into `seq[11]`, shift `wbuf` left, place `clamped_in` at `wbuf[23]`, and optimize over
        // all 24 modifiable slots `[cand_0, wbuf[0..23]]`!
        let mut seq = [0.0_f32; TP_PAST_HISTORY + TP_PREDICTIVE_WINDOW];
        for i in 0..TP_PAST_HISTORY {
            // `past[0]` is `y[n-1]`, `past[10]` is `y[n-11]` -> chronological `seq[0..11]`
            seq[i] = past[TP_PAST_HISTORY - 1 - i];
        }
        for i in 0..(TP_PREDICTIVE_WINDOW - 1) {
            seq[TP_PAST_HISTORY + i] = wbuf[i];
        }
        seq[TP_PAST_HISTORY + TP_PREDICTIVE_WINDOW - 1] = clamped_in;

        for _iter in 0..4 {
            let mut worst_ratio = 1.0_f32;
            let mut worst_k = 0usize;

            for k in TP_PAST_HISTORY..(TP_PAST_HISTORY + TP_PREDICTIVE_WINDOW) {
                let mut local_max = seq[k].abs();
                let mut ph_vals = [0.0_f32; 4];
                for (p_idx, phase) in ITU_BS1770_4_POLYPHASE_TAPS.iter().enumerate() {
                    let mut sum = 0.0_f32;
                    for m in 0..12 {
                        sum += phase[m] * seq[k - m];
                    }
                    ph_vals[p_idx] = sum;
                    local_max = local_max.max(sum.abs());
                }
                // Check Catmull-Rom cubic inter-sample extremum across the 4 polyphase sub-samples
                // bracketed by `seq[k - 6]` and `seq[k - 5]`:
                let cubic_pk = catmull_rom_cubic_max_abs(
                    ph_vals[3], ph_vals[2], ph_vals[1], ph_vals[0],
                );
                local_max = local_max.max(cubic_pk);

                if local_max > soft_limit {
                    let ratio = local_max / soft_limit;
                    if ratio > worst_ratio {
                        worst_ratio = ratio;
                        worst_k = k;
                    }
                }
            }

            if worst_ratio <= 1.0 {
                break;
            }

            // Smoothly attenuate all modifiable future samples contributing to `worst_k` (`11..=worst_k`
            // plus a short 4-sample release taper `worst_k+1..=worst_k+4`).
            let g_req = (0.996 / worst_ratio).clamp(0.25, 1.0);
            let start_mod = (worst_k.saturating_sub(11)).max(TP_PAST_HISTORY);
            let end_mod = (worst_k + 4).min(TP_PAST_HISTORY + TP_PREDICTIVE_WINDOW - 1);
            for idx in start_mod..=worst_k {
                seq[idx] *= g_req;
            }
            if end_mod > worst_k {
                let span = (end_mod - worst_k) as f32;
                for idx in (worst_k + 1)..=end_mod {
                    let t = ((idx - worst_k) as f32) / (span + 1.0);
                    let g_taper = g_req + (1.0 - g_req) * t;
                    seq[idx] *= g_taper;
                }
            }
        }

        let cand_emit = seq[TP_PAST_HISTORY];
        for i in 0..(TP_PREDICTIVE_WINDOW - 1) {
            wbuf[i] = seq[TP_PAST_HISTORY + 1 + i];
        }
        wbuf[TP_PREDICTIVE_WINDOW - 1] =
            seq[TP_PAST_HISTORY + TP_PREDICTIVE_WINDOW - 1];

        Self::clamp_emitted_against_past(past, cand_emit, ceiling)
    }

    /// Enforces exact closed-form `ITU-R BS.1770-4` 4-phase linear bounds on `sample_out` given
    /// `past = [y[n-1], ..., y[n-11]]` and records the final emitted sample in `past`.
    #[inline(always)]
    fn finalize_and_record_emitted_channel(
        past: &mut [f32; TP_PAST_HISTORY],
        sample_out: f32,
        ceiling: f32,
    ) -> f32 {
        let safe_out = Self::clamp_emitted_against_past(past, sample_out, ceiling);
        for i in (1..TP_PAST_HISTORY).rev() {
            past[i] = past[i - 1];
        }
        past[0] = safe_out;
        safe_out
    }

    #[inline(always)]
    fn clamp_emitted_against_past(
        past: &[f32; TP_PAST_HISTORY],
        cand: f32,
        ceiling: f32,
    ) -> f32 {
        let hard_tp_limit = ceiling * 0.996;
        let mut lo = -hard_tp_limit;
        let mut hi = hard_tp_limit;

        for phase in &ITU_BS1770_4_POLYPHASE_TAPS {
            let c0 = phase[0];
            let mut rest = 0.0_f32;
            for m in 1..12 {
                rest += phase[m] * past[m - 1];
            }
            if c0.abs() > 1.0e-7 {
                let b1 = (-hard_tp_limit - rest) / c0;
                let b2 = (hard_tp_limit - rest) / c0;
                let p_lo = b1.min(b2);
                let p_hi = b1.max(b2);
                lo = lo.max(p_lo);
                hi = hi.min(p_hi);
            }
        }

        if lo <= hi {
            cand.clamp(lo, hi)
        } else {
            (0.5 * (lo + hi)).clamp(-hard_tp_limit, hard_tp_limit)
        }
    }

    #[inline]
    pub fn step_stereo(&mut self, in_l: f32, in_r: f32, sample_rate: f32) -> (f32, f32, f32) {
        self.soft_clip_mode = SoftClipMode::from_knee_field(self.soft_clip_knee);
        let [xl, xr] = self.step_soft_clip_stereo(in_l, in_r, sample_rate);

        let [tp_in_l, tp_in_r] = if self.prevent_intersample_clipping {
            let tp4 = self.tp_in_detector.step_stereo(xl, xr);
            let tp9 = self.tp_9x_detector.step_stereo(xl, xr);
            [
                tp4[0].max(tp9[0]) * OZONE12_ISP_SAFETY_FACTOR,
                tp4[1].max(tp9[1]) * OZONE12_ISP_SAFETY_FACTOR,
            ]
        } else {
            [xl.abs(), xr.abs()]
        };

        let delayed = self.lookahead_buf[self.lookahead_cursor];
        self.lookahead_buf[self.lookahead_cursor] = [xl, xr];
        self.lookahead_cursor += 1;
        if self.lookahead_cursor >= self.lookahead_buf.len() {
            self.lookahead_cursor = 0;
        }

        let peak_l = self.max_heap_l.push_and_query_max(tp_in_l);
        let peak_r = self.max_heap_r.push_and_query_max(tp_in_r);
        let peaks = [peak_l, peak_r];

        // IRC V dual-timescale envelope followers (`5.0 ms` fast / `50.0 ms` slow @ `0x18359D2B0`)
        let sr = sample_rate.max(8000.0);
        let a_fast = (-1.0 / (0.0005 * sr)).exp();
        let r_fast = (-1.0 / (0.0050 * sr)).exp();
        let a_slow = (-1.0 / (0.0050 * sr)).exp();
        let r_slow = (-1.0 / (0.0500 * sr)).exp();
        let atk_smooth_alpha = (-1.0 / (0.0012 * sr)).exp();

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
        let ceil = self.ceiling_linear.max(1e-5);
        let eff_ceil = if self.prevent_intersample_clipping {
            ceil / OZONE12_ISP_SAFETY_FACTOR
        } else {
            ceil
        };
        let mut gains = [1.0f32; 2];

        for c in 0..2 {
            let linked_t = (1.0 - self.stereo_link_transient) * trans[c]
                + self.stereo_link_transient * max_trans;
            let linked_s =
                (1.0 - self.stereo_link_sustain) * sust[c] + self.stereo_link_sustain * max_sust;
            let total_env = (linked_s + linked_t * (1.0 - 0.5 * self.transient_shaping))
                .max(peaks[c])
                .max(1e-9);
            let target_g = (eff_ceil / total_env).min(1.0);

            let crest = linked_t / (linked_s + 1e-6);
            let adaptive_rel_sec = (0.080 / (1.0 + 3.0 * crest)).clamp(0.005, 0.150);
            let rel_alpha = (-1.0 / (adaptive_rel_sec * sr)).exp();
            self.gr_smooth[c] = if target_g < self.gr_smooth[c] {
                target_g
            } else {
                rel_alpha * self.gr_smooth[c] + (1.0 - rel_alpha) * target_g
            };
            // Smooth C^1 attack transition inside lookahead window to eliminate Gibbs kinks
            self.gr_attack_smooth[c] = if self.gr_smooth[c] < self.gr_attack_smooth[c] {
                atk_smooth_alpha * self.gr_attack_smooth[c]
                    + (1.0 - atk_smooth_alpha) * self.gr_smooth[c]
            } else {
                self.gr_smooth[c]
            };
            gains[c] = self.gr_attack_smooth[c].min(self.gr_smooth[c].max(target_g));
        }

        let cand_l = (delayed[0] * gains[0]).clamp(-ceil, ceil);
        let cand_r = (delayed[1] * gains[1]).clamp(-ceil, ceil);

        let (out_l, out_r) = if self.prevent_intersample_clipping {
            let pre_l = Self::push_and_prepare_tp_window_channel(
                &self.tp_past_emitted[0],
                &mut self.tp_window[0],
                cand_l,
                ceil,
            );
            let pre_r = Self::push_and_prepare_tp_window_channel(
                &self.tp_past_emitted[1],
                &mut self.tp_window[1],
                cand_r,
                ceil,
            );
            let fin_l =
                Self::finalize_and_record_emitted_channel(&mut self.tp_past_emitted[0], pre_l, ceil);
            let fin_r =
                Self::finalize_and_record_emitted_channel(&mut self.tp_past_emitted[1], pre_r, ceil);
            (fin_l, fin_r)
        } else {
            (cand_l, cand_r)
        };

        let gr_db = 20.0 * gains[0].min(gains[1]).max(1e-6).log10();
        (out_l, out_r, gr_db)
    }

    /// Variant of [`Self::step_stereo`] used by [`IrcVMultiBandMaximizer`] so `MbitPlusDither`
    /// runs *before* the final sample is committed to `tp_past_emitted`, guaranteeing dither
    /// noise-shaping can never cause True-Peak overshoot above `ceiling`.
    #[inline]
    fn step_stereo_with_dither(
        &mut self,
        in_l: f32,
        in_r: f32,
        sample_rate: f32,
        dither: &mut MbitPlusDither,
    ) -> (f32, f32, f32) {
        if !dither.enabled {
            return self.step_stereo(in_l, in_r, sample_rate);
        }
        self.soft_clip_mode = SoftClipMode::from_knee_field(self.soft_clip_knee);
        let [xl, xr] = self.step_soft_clip_stereo(in_l, in_r, sample_rate);

        let [tp_in_l, tp_in_r] = if self.prevent_intersample_clipping {
            let tp4 = self.tp_in_detector.step_stereo(xl, xr);
            let tp9 = self.tp_9x_detector.step_stereo(xl, xr);
            [
                tp4[0].max(tp9[0]) * OZONE12_ISP_SAFETY_FACTOR,
                tp4[1].max(tp9[1]) * OZONE12_ISP_SAFETY_FACTOR,
            ]
        } else {
            [xl.abs(), xr.abs()]
        };

        let delayed = self.lookahead_buf[self.lookahead_cursor];
        self.lookahead_buf[self.lookahead_cursor] = [xl, xr];
        self.lookahead_cursor += 1;
        if self.lookahead_cursor >= self.lookahead_buf.len() {
            self.lookahead_cursor = 0;
        }

        let peak_l = self.max_heap_l.push_and_query_max(tp_in_l);
        let peak_r = self.max_heap_r.push_and_query_max(tp_in_r);
        let peaks = [peak_l, peak_r];

        let sr = sample_rate.max(8000.0);
        let a_fast = (-1.0 / (0.0005 * sr)).exp();
        let r_fast = (-1.0 / (0.0050 * sr)).exp();
        let a_slow = (-1.0 / (0.0050 * sr)).exp();
        let r_slow = (-1.0 / (0.0500 * sr)).exp();
        let atk_smooth_alpha = (-1.0 / (0.0012 * sr)).exp();

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
        let ceil = self.ceiling_linear.max(1e-5);
        let eff_ceil = if self.prevent_intersample_clipping {
            ceil / OZONE12_ISP_SAFETY_FACTOR
        } else {
            ceil
        };
        let mut gains = [1.0f32; 2];

        for c in 0..2 {
            let linked_t = (1.0 - self.stereo_link_transient) * trans[c]
                + self.stereo_link_transient * max_trans;
            let linked_s =
                (1.0 - self.stereo_link_sustain) * sust[c] + self.stereo_link_sustain * max_sust;
            let total_env = (linked_s + linked_t * (1.0 - 0.5 * self.transient_shaping))
                .max(peaks[c])
                .max(1e-9);
            let target_g = (eff_ceil / total_env).min(1.0);

            let crest = linked_t / (linked_s + 1e-6);
            let adaptive_rel_sec = (0.080 / (1.0 + 3.0 * crest)).clamp(0.005, 0.150);
            let rel_alpha = (-1.0 / (adaptive_rel_sec * sr)).exp();
            self.gr_smooth[c] = if target_g < self.gr_smooth[c] {
                target_g
            } else {
                rel_alpha * self.gr_smooth[c] + (1.0 - rel_alpha) * target_g
            };
            self.gr_attack_smooth[c] = if self.gr_smooth[c] < self.gr_attack_smooth[c] {
                atk_smooth_alpha * self.gr_attack_smooth[c]
                    + (1.0 - atk_smooth_alpha) * self.gr_smooth[c]
            } else {
                self.gr_smooth[c]
            };
            gains[c] = self.gr_attack_smooth[c].min(self.gr_smooth[c].max(target_g));
        }

        let cand_l = (delayed[0] * gains[0]).clamp(-ceil, ceil);
        let cand_r = (delayed[1] * gains[1]).clamp(-ceil, ceil);

        let (out_l, out_r) = if self.prevent_intersample_clipping {
            let pre_l = Self::push_and_prepare_tp_window_channel(
                &self.tp_past_emitted[0],
                &mut self.tp_window[0],
                cand_l,
                ceil,
            );
            let pre_r = Self::push_and_prepare_tp_window_channel(
                &self.tp_past_emitted[1],
                &mut self.tp_window[1],
                cand_r,
                ceil,
            );
            let dith_l = dither.process_sample_with_ceiling(0, pre_l, ceil * 0.994);
            let dith_r = dither.process_sample_with_ceiling(1, pre_r, ceil * 0.994);
            let fin_l = Self::finalize_and_record_emitted_channel(
                &mut self.tp_past_emitted[0],
                dith_l,
                ceil,
            );
            let fin_r = Self::finalize_and_record_emitted_channel(
                &mut self.tp_past_emitted[1],
                dith_r,
                ceil,
            );
            (fin_l, fin_r)
        } else {
            (
                dither.process_sample_with_ceiling(0, cand_l, ceil),
                dither.process_sample_with_ceiling(1, cand_r, ceil),
            )
        };

        let gr_db = 20.0 * gains[0].min(gains[1]).max(1e-6).log10();
        (out_l, out_r, gr_db)
    }
}

impl UnifiedDspNode for IrcVMaximizerCore {
    fn name(&self) -> &'static str {
        "iZotope Ozone 12 IRC V Maximizer Core"
    }

    fn latency_samples(&self) -> usize {
        if self.prevent_intersample_clipping {
            self.lookahead_buf.len() + TP_PREDICTIVE_WINDOW
        } else {
            self.lookahead_buf.len()
        }
    }

    fn prepare(&mut self, sample_rate: f64, _max_block_size: usize) {
        let sr = sample_rate.max(8000.0) as f32;
        self.sample_rate = sr;
        let lookahead_samples = ((0.005 * sr).round() as usize).max(32);
        if self.lookahead_buf.len() != lookahead_samples {
            self.lookahead_buf = vec![[0.0; 2]; lookahead_samples];
            self.lookahead_cursor = 0;
            self.max_heap_l = SlidingMaxHeap::new(lookahead_samples);
            self.max_heap_r = SlidingMaxHeap::new(lookahead_samples);
        }
    }

    fn reset_state(&mut self) {
        self.reset();
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

/// Complete Multi-Band IRC I–V True-Peak Maximizer (`IrcVMultiBandMaximizer` / `IrcMaximizerModule`).
#[derive(Clone, Debug)]
pub struct IrcVMultiBandMaximizer {
    pub bypass: bool,
    pub irc_mode: IrcMode,
    pub character_mode: IrcCharacter,
    pub character_speed: f32, // 0.0 (Fast) .. 10.0 (Slow)
    pub gain_db: f32,         // [-20.0, +20.0] dB (or Threshold)
    pub ceiling_dbtp: f32,    // [-20.0, 0.0] dBTP (Margin)
    pub link_gain_and_ceiling: bool,
    pub enable_upward_comp: bool,
    pub upward_comp_wet: f32, // [0.0, 1.0]
    pub enable_soft_clip: bool,
    pub soft_clip_mix: f32, // [0.0, 1.0]
    pub soft_clip_mode: SoftClipMode,
    /// Enables `4x` Linear-Phase Polyphase FIR (`128`-tap Kaiser $\beta=9.0$) oversampling for `SoftClipper`.
    pub soft_clip_oversample_4x: bool,
    pub enable_transient_shaping: bool,
    pub transient_shaping: f32, // [-1.0, +1.0]
    pub stereo_link_sustain: f32,
    pub stereo_link_transient: f32,
    pub prevent_intersample_clipping: bool,
    pub dither: MbitPlusDither,
    pub crossover: LinkwitzRiley4BandCrossover,
    band_limiters: [BandEnvelopeLimiter; OZONE_MULTIBAND_COUNT],
    pub final_core: IrcVMaximizerCore,
    pub meter: LufsTruePeakMeter,
    pub last_gr_db: f32,
    sample_rate: f64,
    active_xover_is_irc5: bool,
}

/// Alias matching `05_master_rust_svelte_blueprint.md` Section 2.
pub type IrcMaximizerModule = IrcVMultiBandMaximizer;

impl Default for IrcVMultiBandMaximizer {
    fn default() -> Self {
        Self::new(48_000.0)
    }
}

impl IrcVMultiBandMaximizer {
    pub fn new(sample_rate: f64) -> Self {
        let sr = sample_rate.max(8000.0);
        let mut s = Self {
            bypass: false,
            irc_mode: IrcMode::IrcV,
            character_mode: IrcCharacter::Modern,
            character_speed: 5.0,
            gain_db: 0.0,
            ceiling_dbtp: -1.0,
            link_gain_and_ceiling: false,
            enable_upward_comp: false,
            upward_comp_wet: 0.0,
            enable_soft_clip: false,
            soft_clip_mix: 0.0,
            soft_clip_mode: SoftClipMode::Medium,
            soft_clip_oversample_4x: true,
            enable_transient_shaping: false,
            transient_shaping: 0.0,
            stereo_link_sustain: 0.80,
            stereo_link_transient: 0.40,
            prevent_intersample_clipping: true,
            dither: MbitPlusDither::new(),
            crossover: LinkwitzRiley4BandCrossover::new(IRC5_CROSSOVER_FREQS_HZ, sr),
            band_limiters: [BandEnvelopeLimiter::new(); OZONE_MULTIBAND_COUNT],
            final_core: IrcVMaximizerCore::new(sr as f32),
            meter: LufsTruePeakMeter::new(sr),
            last_gr_db: 0.0,
            sample_rate: sr,
            active_xover_is_irc5: true,
        };
        s.sync_core_parameters();
        s
    }

    /// Sets the input gain (or negative threshold) in dB (`[-20.0, +24.0] dB`).
    pub fn set_gain_db(&mut self, gain_db: f32) {
        self.gain_db = gain_db.clamp(-20.0, 24.0);
        self.sync_core_parameters();
    }

    /// Sets the output True-Peak ceiling (Margin) in dBTP (`[-24.0, 0.0] dBTP`).
    pub fn set_ceiling_dbtp(&mut self, ceiling_dbtp: f32) {
        self.ceiling_dbtp = ceiling_dbtp.clamp(-24.0, 0.0);
        self.sync_core_parameters();
    }

    pub fn sync_core_parameters(&mut self) {
        let want_irc5_xover = self.irc_mode == IrcMode::IrcV;
        if want_irc5_xover != self.active_xover_is_irc5 {
            self.active_xover_is_irc5 = want_irc5_xover;
            let freqs = if want_irc5_xover {
                IRC5_CROSSOVER_FREQS_HZ
            } else {
                DEFAULT_CROSSOVER_FREQS_HZ
            };
            self.crossover.configure(freqs, self.sample_rate);
        }

        let ceiling_lin = 10.0f32.powf(self.ceiling_dbtp / 20.0);
        let gain_lin = 10.0f32.powf(self.gain_db / 20.0);
        self.final_core.ceiling_linear = ceiling_lin;
        self.final_core.gain_linear = if self.irc_mode.is_multiband() {
            1.0
        } else {
            gain_lin
        };
        self.final_core.upward_boost_wet = if self.enable_upward_comp {
            self.upward_comp_wet.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.final_core.soft_clip_mix = if self.enable_soft_clip {
            self.soft_clip_mix.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.final_core.soft_clip_mode = self.soft_clip_mode;
        self.final_core.soft_clip_knee = self.soft_clip_mode.knee_threshold();
        self.final_core.soft_clip_oversample_4x = self.soft_clip_oversample_4x;
        self.final_core.transient_shaping = if self.enable_transient_shaping {
            self.transient_shaping.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        self.final_core.stereo_link_sustain = self.stereo_link_sustain.clamp(0.0, 1.0);
        self.final_core.stereo_link_transient = self.stereo_link_transient.clamp(0.0, 1.0);
        self.final_core.prevent_intersample_clipping = self.prevent_intersample_clipping;
    }

    /// Processes a single stereo frame `[L, R]` through the complete Multi-Band IRC I–V +
    /// `4x` $C^5$ Soft-Clipper + `4x`/`9x` Cubic True-Peak pipeline (`0` heap allocations).
    #[inline]
    pub fn step_frame(&mut self, frame: StereoFrame) -> StereoFrame {
        if self.bypass {
            self.meter.push_frame(frame[0], frame[1], 0.0);
            return frame;
        }
        self.sync_core_parameters();

        let sr = self.sample_rate as f32;
        let ceiling_lin = self.final_core.ceiling_linear;
        let gain_lin = 10.0f32.powf(self.gain_db / 20.0);

        let pre_final = if self.irc_mode.is_multiband() {
            let (rel_char_scale, trans_preserve) = self.character_mode.ballistics_factors();
            let speed_mul = 0.25 + (self.character_speed.clamp(0.0, 10.0) / 10.0) * 1.75;
            let rel_scale = rel_char_scale * speed_mul;

            // Exact IRC V (`0x18359D2B0`): 5.0 ms fast envelope vs 50.0 ms slow sustain envelope
            let a_fast = (-1.0 / (0.0005 * sr)).exp();
            let r_fast = (-1.0 / (0.0050 * sr)).exp();
            let a_slow = (-1.0 / (0.0050 * sr)).exp();
            let r_slow = (-1.0 / (0.0500 * rel_scale * sr)).exp();

            let driven = [(frame[0] * gain_lin) as f64, (frame[1] * gain_lin) as f64];
            let split = self.crossover.split_frame_f64(driven);

            let band_ceilings = [
                ceiling_lin * 0.72,
                ceiling_lin * 0.72,
                ceiling_lin * 0.70,
                ceiling_lin * 0.68,
            ];
            let mut sum_l = 0.0f32;
            let mut sum_r = 0.0f32;
            for b in 0..OZONE_MULTIBAND_COUNT {
                let b_in = [split[b][0] as f32, split[b][1] as f32];
                let b_out = self.band_limiters[b].step_band(
                    b_in,
                    band_ceilings[b],
                    a_fast,
                    r_fast,
                    a_slow,
                    r_slow,
                    self.stereo_link_transient,
                    self.stereo_link_sustain,
                    self.final_core.transient_shaping,
                    rel_scale,
                    trans_preserve,
                    b,
                    sr,
                );
                sum_l += b_out[0];
                sum_r += b_out[1];
            }
            [sum_l, sum_r]
        } else {
            frame
        };

        let (out_l, out_r, gr_db) =
            self.final_core
                .step_stereo_with_dither(pre_final[0], pre_final[1], sr, &mut self.dither);

        self.last_gr_db = gr_db;
        self.meter.push_frame(out_l, out_r, gr_db);
        [out_l, out_r]
    }
}

impl OzoneDspModule for IrcVMultiBandMaximizer {
    fn name(&self) -> &'static str {
        "iZotope Ozone 12 IRC I-V Multi-Band True-Peak Maximizer"
    }

    fn reset(&mut self, sample_rate: f64, _max_block_size: usize) {
        self.sample_rate = sample_rate.max(8000.0);
        let freqs = if self.irc_mode == IrcMode::IrcV {
            self.active_xover_is_irc5 = true;
            IRC5_CROSSOVER_FREQS_HZ
        } else {
            self.active_xover_is_irc5 = false;
            DEFAULT_CROSSOVER_FREQS_HZ
        };
        self.crossover.configure(freqs, self.sample_rate);
        self.crossover.reset();
        for bl in &mut self.band_limiters {
            bl.reset();
        }
        self.final_core.prepare(self.sample_rate, 4096);
        self.final_core.reset();
        self.dither.reset();
        self.meter.set_sample_rate(self.sample_rate);
        self.last_gr_db = 0.0;
    }

    fn process_block(&mut self, left: &mut [f32], right: &mut [f32]) {
        let n = left.len().min(right.len());
        for i in 0..n {
            let [l, r] = self.step_frame([left[i], right[i]]);
            left[i] = l;
            right[i] = r;
        }
    }

    fn latency_samples(&self) -> usize {
        if self.bypass {
            0
        } else {
            self.final_core.latency_samples()
        }
    }
}

impl UnifiedDspNode for IrcVMultiBandMaximizer {
    fn name(&self) -> &'static str {
        "iZotope Ozone 12 IRC I-V Multi-Band True-Peak Maximizer"
    }

    fn latency_samples(&self) -> usize {
        OzoneDspModule::latency_samples(self)
    }

    fn prepare(&mut self, sample_rate: f64, max_block_size: usize) {
        self.reset(sample_rate, max_block_size);
    }

    fn reset_state(&mut self) {
        self.crossover.reset();
        for bl in &mut self.band_limiters {
            bl.reset();
        }
        self.final_core.reset();
        self.dither.reset();
        self.meter.reset();
        self.last_gr_db = 0.0;
    }

    fn set_parameter_normalized(&mut self, param_id: u32, norm_val: f32) {
        let v = norm_val.clamp(0.0, 1.0);
        match param_id {
            0 => self.bypass = v >= 0.5,
            1 => self.set_gain_db(-20.0 + v * 40.0),
            2 => self.set_ceiling_dbtp(-20.0 + v * 20.0),
            3 => self.link_gain_and_ceiling = v >= 0.5,
            4 => self.character_speed = v * 10.0,
            5 => {
                self.enable_upward_comp = v > 0.01;
                self.upward_comp_wet = v;
                self.sync_core_parameters();
            }
            6 => {
                self.enable_soft_clip = v >= 0.5;
                self.sync_core_parameters();
            }
            7 => {
                self.soft_clip_mix = v;
                self.sync_core_parameters();
            }
            8 => {
                self.soft_clip_mode = match (v * 2.0).round() as u32 {
                    0 => SoftClipMode::Light,
                    1 => SoftClipMode::Medium,
                    _ => SoftClipMode::Heavy,
                };
                self.sync_core_parameters();
            }
            9 => {
                self.enable_transient_shaping = (v - 0.5).abs() > 0.01;
                self.transient_shaping = -1.0 + v * 2.0;
                self.sync_core_parameters();
            }
            10 => {
                self.stereo_link_sustain = v;
                self.sync_core_parameters();
            }
            11 => {
                self.stereo_link_transient = v;
                self.sync_core_parameters();
            }
            12 => {
                self.irc_mode = IrcMode::from_index((v * 5.0).round() as u32);
                self.sync_core_parameters();
            }
            13 => {
                self.prevent_intersample_clipping = v >= 0.5;
                self.sync_core_parameters();
            }
            _ => {}
        }
    }

    fn current_gain_reduction_db(&self) -> f32 {
        self.last_gr_db
    }

    fn export_visual_telemetry(&self, frame: &mut VisualTelemetryFrame) {
        self.meter.export_to_visual_telemetry(frame);
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
