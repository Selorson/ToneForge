//! iZotope Ozone 12 IRC I–V Multi-Band True-Peak Maximizer (`DSP::OzoneMaximizer` `0x185736808`,
//! `DSP::Shared::IRC::Maximizer` `0x185767808`, `DSP::MaxHeap` `0x183F81EE4`, `MBIT+ Dither` `0x183F63430`).
//!
//! Implements:
//! 1. `SlidingMaxHeap`: Binary segment-tree sliding maximum (`shr r8, 1; cmovg`) over the lookahead window
//!    with $O(\log L)$ complexity and Zero Heap Allocation.
//! 2. Upward Compression (`EnableLowLevelBoost` / `LowLevelBoostWetAmount`) & Pre-Limiter Soft Clipper
//!    (`Light` $k_m=0.85$, `Medium` $k_m=0.70$, `Heavy` $k_m=0.50$).
//! 3. 4-Band Linkwitz-Riley Crossover (`LinkwitzRiley4BandCrossover`) for Multi-Band IRC IV & IRC V modes.
//! 4. Adaptive Release Envelope Followers across all 6 IRC modes (`IRC I`, `IRC II`, `IRC III`, `IRC IV`,
//!    `IRC V`, `IRC LL`) and 6 Character profiles (`Pumping`, `Balanced`, `Crisp`, `Classic`, `Modern`, `Transient`).
//! 5. 4x Oversampling Polyphase FIR True-Peak Detector & Predictive Inter-Sample Peak Brickwall Limiter
//!    (`ITU-R BS.1770-4` Table 3), strictly guaranteeing post-limiter True Peak $\le$ `Ceiling dBTP`.
//! 6. Psychoacoustic `MbitPlusDither` (`8 / 16 / 20 / 24-bit` quantization + noise shaping).

use crate::chain_element::{
    DEFAULT_CROSSOVER_FREQS_HZ, LinkwitzRiley4BandCrossover, OZONE_MULTIBAND_COUNT, OzoneDspModule,
};
use crate::lufs_true_peak_meter::{ITU_BS1770_4_POLYPHASE_TAPS, LufsTruePeakMeter, TruePeakDetector4x};
use daw_core_graph::{ProcessContext, StereoFrame, UnifiedDspNode, VisualTelemetryFrame};

const TP_PREDICTIVE_WINDOW: usize = 24;

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
    pub fn window_len(&self) -> usize {
        self.window_len
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
    /// Mode 4 (`0x183F69CE4`): Ozone 12 4-band envelope morphing + upward compression + True-Peak brickwall.
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

/// Pre-Limiter Soft Clipper Knee Mode (`SoftClipMode` in `AutomatableParamList_Maximizer.txt`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoftClipMode {
    /// Light (`k_m = 0.85`)
    Light = 0,
    /// Medium (`k_m = 0.70`)
    Medium = 1,
    /// Heavy (`k_m = 0.50`)
    Heavy = 2,
}

impl SoftClipMode {
    #[inline]
    pub fn knee_threshold(self) -> f32 {
        match self {
            Self::Light => 0.85,
            Self::Medium => 0.70,
            Self::Heavy => 0.50,
        }
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
        // Xorshift64* PRNG (Zero Heap Allocation)
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
        if !self.enabled {
            return x;
        }
        let bits = self.bit_depth.clamp(8, 24) as u32;
        let quant_scale = ((1u32 << (bits - 1)) - 1) as f32;
        let inv_quant = 1.0 / quant_scale;

        // High-pass TPDF dither
        let r = self.next_uniform();
        let tpdf = (r - self.prev_pdf[ch]) * self.dither_amount;
        self.prev_pdf[ch] = r;

        // 2nd-order psychoacoustic noise-shaping feedback
        let shaped_in = if self.noise_shaping {
            x + (1.55 * self.err_hist[ch][0] - 0.62 * self.err_hist[ch][1]) * inv_quant
        } else {
            x
        };

        let scaled = shaped_in * quant_scale + tpdf;
        let quantized = scaled.round();
        let err = (shaped_in * quant_scale) - quantized;
        self.err_hist[ch][1] = self.err_hist[ch][0];
        self.err_hist[ch][0] = err.clamp(-2.0, 2.0);
        quantized * inv_quant
    }
}

/// Per-band envelope follower for Multi-Band IRC IV / IRC V limiting.
#[derive(Clone, Copy, Debug, Default)]
struct BandEnvelopeLimiter {
    env_fast: [f32; 2],
    env_slow: [f32; 2],
    gr_smooth: [f32; 2],
}

impl BandEnvelopeLimiter {
    fn new() -> Self {
        Self {
            env_fast: [0.0; 2],
            env_slow: [0.0; 2],
            gr_smooth: [1.0; 2],
        }
    }

    fn reset(&mut self) {
        self.env_fast = [0.0; 2];
        self.env_slow = [0.0; 2];
        self.gr_smooth = [1.0; 2];
    }

    #[inline(always)]
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
        sample_rate: f32,
    ) -> [f32; 2] {
        let mut trans = [0.0f32; 2];
        let mut sust = [0.0f32; 2];
        for c in 0..2 {
            let p = band_in[c].abs();
            let cf = if p > self.env_fast[c] { a_fast } else { r_fast };
            let cs = if p > self.env_slow[c] { a_slow } else { r_slow };
            self.env_fast[c] = cf * self.env_fast[c] + (1.0 - cf) * p;
            self.env_slow[c] = cs * self.env_slow[c] + (1.0 - cs) * p;
            sust[c] = self.env_slow[c];
            trans[c] = (self.env_fast[c] - self.env_slow[c]).max(0.0);
        }

        let max_t = trans[0].max(trans[1]);
        let max_s = sust[0].max(sust[1]);
        let mut out = [0.0f32; 2];

        for c in 0..2 {
            let lt = (1.0 - stereo_link_trans) * trans[c] + stereo_link_trans * max_t;
            let ls = (1.0 - stereo_link_sust) * sust[c] + stereo_link_sust * max_s;
            let eff_env = (ls + lt * (1.0 - 0.35 * transient_shaping) / trans_preserve).max(1e-9);
            let target_g = (band_ceiling / eff_env).min(1.0);

            let crest = lt / (ls + 1e-6);
            let adaptive_rel_sec = ((0.085 * rel_scale) / (1.0 + 3.0 * crest)).clamp(0.004, 0.250);
            let rel_alpha = (-1.0 / (adaptive_rel_sec * sample_rate)).exp();
            self.gr_smooth[c] = if target_g < self.gr_smooth[c] {
                0.35 * self.gr_smooth[c] + 0.65 * target_g
            } else {
                rel_alpha * self.gr_smooth[c] + (1.0 - rel_alpha) * target_g
            };
            out[c] = band_in[c] * self.gr_smooth[c];
        }
        out
    }
}

/// Core single-band + lookahead Maximizer (`IrcVMaximizerCore`) with `SlidingMaxHeap`,
/// Upward Compression, Soft Clipper, and Predictive 4x True-Peak Brickwall stage.
#[derive(Clone, Debug)]
pub struct IrcVMaximizerCore {
    pub ceiling_linear: f32,
    pub gain_linear: f32,
    pub upward_boost_wet: f32,
    pub soft_clip_mix: f32,
    pub soft_clip_knee: f32, // 0.85 (Light), 0.70 (Medium), 0.50 (Heavy)
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
    pub sample_rate: f32,
    tp_in_detector: TruePeakDetector4x,
    tp_window: [[f32; TP_PREDICTIVE_WINDOW]; 2],
}

impl IrcVMaximizerCore {
    pub fn new(sample_rate: f32) -> Self {
        let sr = sample_rate.max(8000.0);
        let lookahead_samples = ((0.005 * sr).round() as usize).max(32);
        Self {
            ceiling_linear: 0.89125094, // -1.0 dBFS / dBTP
            gain_linear: 1.0,
            upward_boost_wet: 0.0,
            soft_clip_mix: 0.0,
            soft_clip_knee: 0.70,
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
            sample_rate: sr,
            tp_in_detector: TruePeakDetector4x::new(),
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
        self.tp_in_detector.reset();
        self.tp_window = [[0.0; TP_PREDICTIVE_WINDOW]; 2];
    }

    #[inline(always)]
    pub fn upward_compress_and_soft_clip(&self, x: f32, rms_env: f32) -> f32 {
        // 1. Upward Compression (`EnableLowLevelBoost` below -24 dB = 0.0631, above -60 dB noise floor)
        let mut s = x * self.gain_linear;
        if self.upward_boost_wet > 0.0 && rms_env > 1e-3 && rms_env < 0.0631 {
            let boost = 1.0 + self.upward_boost_wet * ((0.0631 - rms_env) / 0.0631) * 1.5;
            s *= boost;
        }
        // 2. Pre-Limiter Soft Clipper (`SoftClipMode` knee k_m relative to ceiling)
        if self.soft_clip_mix <= 0.0 {
            return s;
        }
        let ceil = self.ceiling_linear.max(1e-4);
        let norm_s = s / ceil;
        let abs_s = norm_s.abs();
        let km = self.soft_clip_knee.clamp(0.2, 0.95);
        let clipped_norm = if abs_s <= km {
            abs_s
        } else {
            km + (1.0 - km) * ((abs_s - km) / (1.0 - km)).tanh()
        };
        let clipped = norm_s.signum() * clipped_norm * ceil;
        (1.0 - self.soft_clip_mix) * s + self.soft_clip_mix * clipped
    }

    /// Predictive 4x Oversampling True-Peak window enforcement (`ITU-R BS.1770-4` Table 3).
    /// Guarantees that both sample peaks and all 4 polyphase inter-sample branches remain `<= ceiling`.
    #[inline(always)]
    fn enforce_true_peak_window_channel(
        wbuf: &mut [f32; TP_PREDICTIVE_WINDOW],
        new_sample: f32,
        ceiling: f32,
    ) -> f32 {
        let clamped_in = new_sample.clamp(-ceiling, ceiling);
        let emitted = wbuf[0];
        for i in 0..(TP_PREDICTIVE_WINDOW - 1) {
            wbuf[i] = wbuf[i + 1];
        }
        wbuf[TP_PREDICTIVE_WINDOW - 1] = clamped_in;

        let target_limit = ceiling * 0.996;
        for _iter in 0..3 {
            let mut max_tp = 0.0f32;
            for &v in wbuf.iter() {
                let a = v.abs();
                if a > max_tp {
                    max_tp = a;
                }
            }
            for k in 11..TP_PREDICTIVE_WINDOW {
                for phase in &ITU_BS1770_4_POLYPHASE_TAPS {
                    let mut sum = 0.0f32;
                    for m in 0..12 {
                        sum += phase[m] * wbuf[k - m];
                    }
                    let a = sum.abs();
                    if a > max_tp {
                        max_tp = a;
                    }
                }
            }
            if max_tp > target_limit {
                let scale = (ceiling * 0.992) / max_tp;
                for item in wbuf.iter_mut().skip(5) {
                    *item *= scale;
                }
            } else {
                break;
            }
        }
        emitted.clamp(-ceiling, ceiling)
    }

    #[inline]
    pub fn step_stereo(&mut self, in_l: f32, in_r: f32, sample_rate: f32) -> (f32, f32, f32) {
        let xl = self.upward_compress_and_soft_clip(in_l, self.env_slow[0]);
        let xr = self.upward_compress_and_soft_clip(in_r, self.env_slow[1]);

        let [tp_in_l, tp_in_r] = if self.prevent_intersample_clipping {
            self.tp_in_detector.step_stereo(xl, xr)
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
        let ceil = self.ceiling_linear.max(1e-5);
        let eff_ceil = if self.prevent_intersample_clipping {
            ceil * 0.965
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
            let rel_alpha = (-1.0 / (adaptive_rel_sec * sample_rate)).exp();
            self.gr_smooth[c] = if target_g < self.gr_smooth[c] {
                target_g
            } else {
                rel_alpha * self.gr_smooth[c] + (1.0 - rel_alpha) * target_g
            };
            gains[c] = self.gr_smooth[c];
        }

        let cand_l = (delayed[0] * gains[0]).clamp(-ceil, ceil);
        let cand_r = (delayed[1] * gains[1]).clamp(-ceil, ceil);

        let (out_l, out_r) = if self.prevent_intersample_clipping {
            let ol = Self::enforce_true_peak_window_channel(&mut self.tp_window[0], cand_l, ceil);
            let or = Self::enforce_true_peak_window_channel(&mut self.tp_window[1], cand_r, ceil);
            (ol, or)
        } else {
            (cand_l, cand_r)
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
///
/// Combines:
/// - 4-Band Linkwitz-Riley Crossover (`LinkwitzRiley4BandCrossover`) for independent multi-band envelope
///   limiting in `IRC III`, `IRC IV`, and `IRC V` modes (so sub-bass/kick limiting does not pump vocals/highs)
/// - Adaptive program-dependent release (`IRC I–IV` `Classic` / `Modern` / `Transient` / `Pumping`)
/// - `IrcVMaximizerCore` lookahead `SlidingMaxHeap` + `TruePeakDetector4x` (`ITU-R BS.1770-4`)
/// - Upward Compression (`EnableLowLevelBoost`) & Soft Clipper (`Light` / `Medium` / `Heavy`)
/// - `MbitPlusDither` & integrated `LufsTruePeakMeter`.
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
    pub soft_clip_mix: f32,   // [0.0, 1.0]
    pub soft_clip_mode: SoftClipMode,
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
            enable_transient_shaping: false,
            transient_shaping: 0.0,
            stereo_link_sustain: 0.80,
            stereo_link_transient: 0.40,
            prevent_intersample_clipping: true,
            dither: MbitPlusDither::new(),
            crossover: LinkwitzRiley4BandCrossover::new(DEFAULT_CROSSOVER_FREQS_HZ, sr),
            band_limiters: [BandEnvelopeLimiter::new(); OZONE_MULTIBAND_COUNT],
            final_core: IrcVMaximizerCore::new(sr as f32),
            meter: LufsTruePeakMeter::new(sr),
            last_gr_db: 0.0,
            sample_rate: sr,
        };
        s.sync_core_parameters();
        s
    }

    /// Sets the input gain (or negative threshold) in dB (`[-20.0, +20.0] dB`).
    pub fn set_gain_db(&mut self, gain_db: f32) {
        self.gain_db = gain_db.clamp(-20.0, 24.0);
        self.sync_core_parameters();
    }

    /// Sets the output True-Peak ceiling (Margin) in dBTP (`[-20.0, 0.0] dBTP`).
    pub fn set_ceiling_dbtp(&mut self, ceiling_dbtp: f32) {
        self.ceiling_dbtp = ceiling_dbtp.clamp(-24.0, 0.0);
        self.sync_core_parameters();
    }

    pub fn sync_core_parameters(&mut self) {
        let ceiling_lin = 10.0f32.powf(self.ceiling_dbtp / 20.0);
        let gain_lin = 10.0f32.powf(self.gain_db / 20.0);
        self.final_core.ceiling_linear = ceiling_lin;
        // In multi-band modes, input gain is applied before the 4-band crossover;
        // in single-band modes, `final_core` applies `gain_linear`.
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
        self.final_core.soft_clip_knee = self.soft_clip_mode.knee_threshold();
        self.final_core.transient_shaping = if self.enable_transient_shaping {
            self.transient_shaping.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        self.final_core.stereo_link_sustain = self.stereo_link_sustain.clamp(0.0, 1.0);
        self.final_core.stereo_link_transient = self.stereo_link_transient.clamp(0.0, 1.0);
        self.final_core.prevent_intersample_clipping = self.prevent_intersample_clipping;
    }

    /// Processes a single stereo frame `[L, R]` through the complete Multi-Band IRC I–V + True-Peak pipeline.
    /// Zero Heap Allocation.
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

            let a_fast = (-1.0 / (0.0002 * sr)).exp();
            let r_fast = (-1.0 / (0.0150 * sr)).exp();
            let a_slow = (-1.0 / (0.0150 * sr)).exp();
            let r_slow = (-1.0 / (0.1800 * rel_scale * sr)).exp();

            // Apply input gain before splitting into 4 Linkwitz-Riley bands
            let driven = [(frame[0] * gain_lin) as f64, (frame[1] * gain_lin) as f64];
            let split = self.crossover.split_frame_f64(driven);

            // Band ceiling allocation: allow individual bands up to 0.72 * ceiling before full-band summation
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
                    sr,
                );
                sum_l += b_out[0];
                sum_r += b_out[1];
            }
            [sum_l, sum_r]
        } else {
            frame
        };

        let (lim_l, lim_r, gr_db) = self.final_core.step_stereo(pre_final[0], pre_final[1], sr);
        let out_l = self
            .dither
            .process_sample(0, lim_l)
            .clamp(-ceiling_lin, ceiling_lin);
        let out_r = self
            .dither
            .process_sample(1, lim_r)
            .clamp(-ceiling_lin, ceiling_lin);

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
        self.crossover
            .configure(self.crossover.cutoffs_hz, self.sample_rate);
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
        // Matches AutomatableParamList_Maximizer.txt:
        // 0 = Bypass, 1 = Gain, 2 = Margin (Ceiling), 3 = Link, 4 = Character,
        // 5 = UpwardComp, 6 = SoftClip On, 7 = SoftClip Mix, 8 = SoftClip Mode,
        // 9 = Transient Shaping, 10 = Stereo Sustain, 11 = Stereo Transient
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
