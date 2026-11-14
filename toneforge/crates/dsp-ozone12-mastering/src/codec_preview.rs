//! `codec_preview`: Real-Time Psychoacoustic Lossy Codec Audition & Post-Codec True-Peak Monitor
//! for **iZotope Ozone 12** (`CodecPreviewModule`).
//!
//! Architecture:
//! - Exact Princen-Bradley Time-Domain Aliasing Cancellation (TDAC) MDCT/IMDCT with AAC-style
//!   adaptive block switching: `512`-sample LONG frames, LONG_START / LONG_STOP transition windows and
//!   `4 x 128`-sample SHORT blocks chosen one hop ahead by a transient detector (pre-echo control),
//!   exact `768`-sample latency and bit-exact null in `PcmBypass` `Artifact Solo` mode while switching.
//! - Bit reservoir: steady frames bank unused bits, transient frames borrow them.
//! - 24-Band Zwicker Bark psychoacoustic simultaneous masking spreader + codec-specific SMR bitpool
//!   and Joint-Stereo M/S subband quantizer for `MP3 128`, `MP3 320`, `AAC 128`, `AAC 256`, and `Opus 128`.
//! - `Artifact Solo (Delta)` (`codec_wet - delayed_dry`) to audition lossy pre-echo, swirl, and HF loss.
//! - Post-Codec `4x` ITU-R BS.1770-4 True-Peak Detector (`TruePeakDetector4x`) catching lossy
//!   quantization inter-sample overshoots (`dBTP`).
//! - Strict **Zero Heap Allocation** on the real-time audio thread.

use crate::chain_element::OzoneDspModule;
use crate::lufs_true_peak_meter::TruePeakDetector4x;
use daw_core_graph::{ProcessContext, StereoFrame, UnifiedDspNode};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::sync::OnceLock;

pub const CODEC_MDCT_WINDOW_LEN: usize = 512;
pub const CODEC_MDCT_BINS: usize = CODEC_MDCT_WINDOW_LEN / 2; // 256
/// `512`-sample MDCT frame + one `256`-sample look-ahead hop for window-switching decisions.
pub const CODEC_PREVIEW_LATENCY_SAMPLES: usize = CODEC_MDCT_WINDOW_LEN + CODEC_MDCT_BINS; // 768 exact samples
const NUM_BARK_BANDS: usize = 24;

#[repr(u8)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodecTarget {
    PcmBypass = 0,
    Mp3_128 = 1,
    Mp3_320 = 2,
    Aac_128 = 3,
    Aac_256 = 4,
    Opus_128 = 5,
}

impl CodecTarget {
    #[inline]
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::PcmBypass,
            1 => Self::Mp3_128,
            2 => Self::Mp3_320,
            3 => Self::Aac_128,
            4 => Self::Aac_256,
            _ => Self::Opus_128,
        }
    }

    #[inline]
    pub fn label(self) -> &'static str {
        match self {
            Self::PcmBypass => "PCM Lossless Bypass",
            Self::Mp3_128 => "MP3 128 kbps",
            Self::Mp3_320 => "MP3 320 kbps",
            Self::Aac_128 => "AAC 128 kbps",
            Self::Aac_256 => "AAC 256 kbps",
            Self::Opus_128 => "Opus 128 kbps",
        }
    }
}

/// Short-block MDCT geometry for adaptive window switching (`4 x 128`-sample frames, hop `64`).
pub const CODEC_SHORT_WINDOW_LEN: usize = 128;
pub const CODEC_SHORT_BINS: usize = CODEC_SHORT_WINDOW_LEN / 2; // 64
const NUM_SHORT_BLOCKS: usize = CODEC_MDCT_BINS / CODEC_SHORT_BINS; // 4
/// Offset of the first short window inside a `512`-sample frame (AAC geometry: `N/4 - N_s/4`).
const SHORT_BLOCK_OFFSET: usize = CODEC_MDCT_WINDOW_LEN / 4 - CODEC_SHORT_WINDOW_LEN / 4; // 96

/// Window sequence of one `512`-sample frame (AAC `window_sequence`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodecWindowSequence {
    Long,
    LongStart,
    EightShort,
    LongStop,
}

struct MdctStaticTables {
    long_win: [f64; CODEC_MDCT_WINDOW_LEN],
    start_win: [f64; CODEC_MDCT_WINDOW_LEN],
    stop_win: [f64; CODEC_MDCT_WINDOW_LEN],
    short_win: [f64; CODEC_SHORT_WINDOW_LEN],
    dct4_long: Box<[[f64; CODEC_MDCT_BINS]; CODEC_MDCT_BINS]>,
    dct4_short: Box<[[f64; CODEC_SHORT_BINS]; CODEC_SHORT_BINS]>,
    /// Bark spreading weights `[dst][src]` (+25 dB/Bark upward, -10.5 dB/Bark downward).
    spread: [[f64; NUM_BARK_BANDS]; NUM_BARK_BANDS],
}

fn mdct_tables() -> &'static MdctStaticTables {
    static TABLES: OnceLock<MdctStaticTables> = OnceLock::new();
    TABLES.get_or_init(|| {
        let sine = |n: usize, len: usize| (PI * ((n as f64) + 0.5) / (len as f64)).sin();
        let mut long_win = [0.0f64; CODEC_MDCT_WINDOW_LEN];
        for (n, w) in long_win.iter_mut().enumerate() {
            *w = sine(n, CODEC_MDCT_WINDOW_LEN);
        }
        let mut short_win = [0.0f64; CODEC_SHORT_WINDOW_LEN];
        for (n, w) in short_win.iter_mut().enumerate() {
            *w = sine(n, CODEC_SHORT_WINDOW_LEN);
        }
        // LONG_START: long rise | flat | short fall | zeros. LONG_STOP is its time reverse.
        // Both satisfy Princen-Bradley against their long / short neighbours.
        let half = CODEC_MDCT_BINS; // 256
        let flat = (CODEC_MDCT_BINS - CODEC_SHORT_BINS) / 2; // 96
        let mut start_win = [0.0f64; CODEC_MDCT_WINDOW_LEN];
        for n in 0..CODEC_MDCT_WINDOW_LEN {
            start_win[n] = if n < half {
                long_win[n]
            } else if n < half + flat {
                1.0
            } else if n < half + flat + CODEC_SHORT_BINS {
                short_win[CODEC_SHORT_BINS + (n - half - flat)]
            } else {
                0.0
            };
        }
        let mut stop_win = [0.0f64; CODEC_MDCT_WINDOW_LEN];
        for n in 0..CODEC_MDCT_WINDOW_LEN {
            stop_win[n] = start_win[CODEC_MDCT_WINDOW_LEN - 1 - n];
        }
        let mut dct4_long = Box::new([[0.0f64; CODEC_MDCT_BINS]; CODEC_MDCT_BINS]);
        let inv_m = PI / (CODEC_MDCT_BINS as f64);
        for k in 0..CODEC_MDCT_BINS {
            for m in 0..CODEC_MDCT_BINS {
                dct4_long[k][m] = (inv_m * ((m as f64) + 0.5) * ((k as f64) + 0.5)).cos();
            }
        }
        let mut dct4_short = Box::new([[0.0f64; CODEC_SHORT_BINS]; CODEC_SHORT_BINS]);
        let inv_ms = PI / (CODEC_SHORT_BINS as f64);
        for k in 0..CODEC_SHORT_BINS {
            for m in 0..CODEC_SHORT_BINS {
                dct4_short[k][m] = (inv_ms * ((m as f64) + 0.5) * ((k as f64) + 0.5)).cos();
            }
        }
        let mut spread = [[0.0f64; NUM_BARK_BANDS]; NUM_BARK_BANDS];
        for (b_dst, row) in spread.iter_mut().enumerate() {
            for (b_src, w) in row.iter_mut().enumerate() {
                let dz = (b_dst as f64) - (b_src as f64);
                let slope_db = if dz >= 0.0 { -10.5 * dz } else { 25.0 * dz };
                *w = 10.0f64.powf(slope_db / 10.0);
            }
        }
        MdctStaticTables {
            spread,
            long_win,
            start_win,
            stop_win,
            short_win,
            dct4_long,
            dct4_short,
        }
    })
}

/// Windowed TDAC fold + DCT-IV: `frame.len() == 2M`, `win.len() == 2M`, `out.len() == M`.
#[inline]
fn mdct_forward<const M: usize>(
    frame: &[f64],
    win: &[f64],
    dct: &[[f64; M]; M],
    folded: &mut [f64; M],
    out: &mut [f64],
) {
    let half = M / 2;
    for m in 0..half {
        let i1 = half - 1 - m;
        let i2 = half + m;
        folded[m] = -frame[M + i1] * win[M + i1] - frame[M + i2] * win[M + i2];
    }
    for m in half..M {
        let i1 = m - half;
        let i2 = 3 * half - 1 - m;
        folded[m] = frame[i1] * win[i1] - frame[i2] * win[i2];
    }
    for k in 0..M {
        let row = &dct[k];
        let mut acc = 0.0f64;
        for m in 0..M {
            acc += folded[m] * row[m];
        }
        out[k] = acc;
    }
}

/// DCT-IV + TDAC unfold + synthesis window, *accumulated* into `out` (`2M` samples).
#[inline]
fn mdct_inverse_add<const M: usize>(
    spec: &[f64],
    win: &[f64],
    dct: &[[f64; M]; M],
    tmp: &mut [f64; M],
    out: &mut [f64],
) {
    let half = M / 2;
    let norm = 2.0 / (M as f64);
    for m in 0..M {
        let row = &dct[m];
        let mut acc = 0.0f64;
        for k in 0..M {
            acc += spec[k] * row[k];
        }
        tmp[m] = acc * norm;
    }
    for n in 0..half {
        out[n] += win[n] * tmp[n + half];
    }
    for n in half..M {
        out[n] -= win[n] * tmp[3 * half - 1 - n];
    }
    for i in 0..half {
        out[M + i] -= win[M + i] * tmp[half - 1 - i];
    }
    for i in half..M {
        out[M + i] -= win[M + i] * tmp[i - half];
    }
}

/// Real-time zero-allocation Psychoacoustic Codec Preview & Post-Codec True-Peak Monitor with
/// adaptive long/short block switching (pre-echo control) and a bit reservoir.
#[derive(Clone, Debug)]
pub struct CodecPreviewModule {
    pub enabled: bool,
    pub mode: CodecTarget,
    /// Auditions only the codec distortion / quantization delta (`codec_wet - delayed_dry`).
    pub artifact_solo: bool,
    /// Transient-driven LONG -> START -> 4xSHORT -> STOP switching (MPEG/AAC/Opus behaviour).
    pub window_switching: bool,
    sample_rate: f64,
    bin_to_bark: [u8; CODEC_MDCT_BINS],
    short_bin_to_bark: [u8; CODEC_SHORT_BINS],

    /// Three-hop look-ahead pipeline per channel: `[oldest, current, newest]` (`256` samples each).
    hops: Box<[[[f64; CODEC_MDCT_BINS]; 3]; 2]>,
    in_hop: Box<[[f64; CODEC_MDCT_BINS]; 2]>,
    ola_tail: Box<[[f64; CODEC_MDCT_BINS]; 2]>,
    out_fifo: Box<[[f64; CODEC_MDCT_BINS]; 2]>,
    frame_buf: Box<[f64; CODEC_MDCT_WINDOW_LEN]>,
    synth_buf: Box<[[f64; CODEC_MDCT_WINDOW_LEN]; 2]>,
    mdct_spec: Box<[[f64; CODEC_MDCT_BINS]; 2]>,
    short_spec: Box<[[[f64; CODEC_SHORT_BINS]; NUM_SHORT_BLOCKS]; 2]>,
    folded_long: Box<[f64; CODEC_MDCT_BINS]>,
    folded_short: Box<[f64; CODEC_SHORT_BINS]>,
    dry_delay_ring: Box<[[f32; CODEC_PREVIEW_LATENCY_SAMPLES]; 2]>,
    hop_pos: usize,
    dry_cursor: usize,
    hops_seen: usize,

    prev_sequence: CodecWindowSequence,
    pub last_sequence: CodecWindowSequence,
    pub short_frames: u64,
    transient_hp_prev: [f64; 2],
    transient_energy_hist: f64,
    /// Attack flags of the look-ahead pipeline: any attack in the current hop, and attacks in the
    /// second half of the current / oldest hop (see [`Self::detect_attack`]).
    attack_cur: bool,
    late_cur: bool,
    late_prev: bool,
    /// Bit reservoir state (bits) for the bitrate-constrained modes.
    reservoir_bits: f64,

    tp_detector: TruePeakDetector4x,
    artifact_rms_sq: f64,
    pub post_codec_true_peak_dbtp: [f32; 2],
    pub artifact_rms_db: f32,
}

impl Default for CodecPreviewModule {
    fn default() -> Self {
        Self::new(48_000.0)
    }
}

impl CodecPreviewModule {
    /// Zwicker 24 critical band upper edges in Hz.
    const BARK_EDGES_HZ: [f64; NUM_BARK_BANDS] = [
        100.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0, 920.0, 1080.0, 1270.0, 1480.0, 1720.0,
        2000.0, 2320.0, 2700.0, 3150.0, 3700.0, 4400.0, 5300.0, 6400.0, 7700.0, 9500.0, 12000.0,
        15500.0,
    ];

    pub fn new(sample_rate: f64) -> Self {
        let _ = mdct_tables(); // Eagerly initialize static MDCT tables outside RT thread
        let sr = sample_rate.max(8000.0);
        let mut m = Self {
            enabled: false,
            mode: CodecTarget::PcmBypass,
            artifact_solo: false,
            window_switching: true,
            sample_rate: sr,
            bin_to_bark: [0; CODEC_MDCT_BINS],
            short_bin_to_bark: [0; CODEC_SHORT_BINS],
            hops: Box::new([[[0.0; CODEC_MDCT_BINS]; 3]; 2]),
            in_hop: Box::new([[0.0; CODEC_MDCT_BINS]; 2]),
            ola_tail: Box::new([[0.0; CODEC_MDCT_BINS]; 2]),
            out_fifo: Box::new([[0.0; CODEC_MDCT_BINS]; 2]),
            frame_buf: Box::new([0.0; CODEC_MDCT_WINDOW_LEN]),
            synth_buf: Box::new([[0.0; CODEC_MDCT_WINDOW_LEN]; 2]),
            mdct_spec: Box::new([[0.0; CODEC_MDCT_BINS]; 2]),
            short_spec: Box::new([[[0.0; CODEC_SHORT_BINS]; NUM_SHORT_BLOCKS]; 2]),
            folded_long: Box::new([0.0; CODEC_MDCT_BINS]),
            folded_short: Box::new([0.0; CODEC_SHORT_BINS]),
            dry_delay_ring: Box::new([[0.0; CODEC_PREVIEW_LATENCY_SAMPLES]; 2]),
            hop_pos: 0,
            dry_cursor: 0,
            hops_seen: 0,
            prev_sequence: CodecWindowSequence::Long,
            last_sequence: CodecWindowSequence::Long,
            short_frames: 0,
            transient_hp_prev: [0.0; 2],
            transient_energy_hist: 0.0,
            attack_cur: false,
            late_cur: false,
            late_prev: false,
            reservoir_bits: 0.0,
            tp_detector: TruePeakDetector4x::new(),
            artifact_rms_sq: 0.0,
            post_codec_true_peak_dbtp: [-120.0; 2],
            artifact_rms_db: -140.0,
        };
        m.rebuild_bark_map();
        m
    }

    fn bark_index(f_hz: f64) -> u8 {
        for (b, &edge) in Self::BARK_EDGES_HZ.iter().enumerate() {
            if f_hz <= edge {
                return b as u8;
            }
        }
        (NUM_BARK_BANDS - 1) as u8
    }

    fn rebuild_bark_map(&mut self) {
        let sr = self.sample_rate.max(8000.0);
        for k in 0..CODEC_MDCT_BINS {
            let f_hz = ((k as f64) + 0.5) * sr / (CODEC_MDCT_WINDOW_LEN as f64);
            self.bin_to_bark[k] = Self::bark_index(f_hz);
        }
        for k in 0..CODEC_SHORT_BINS {
            let f_hz = ((k as f64) + 0.5) * sr / (CODEC_SHORT_WINDOW_LEN as f64);
            self.short_bin_to_bark[k] = Self::bark_index(f_hz);
        }
    }

    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate.max(8000.0);
        self.rebuild_bark_map();
        self.reset_internal();
    }

    pub fn reset_internal(&mut self) {
        for ch in 0..2 {
            for h in 0..3 {
                self.hops[ch][h].fill(0.0);
            }
            self.in_hop[ch].fill(0.0);
            self.ola_tail[ch].fill(0.0);
            self.out_fifo[ch].fill(0.0);
            self.mdct_spec[ch].fill(0.0);
            self.dry_delay_ring[ch].fill(0.0);
        }
        self.hop_pos = 0;
        self.dry_cursor = 0;
        self.hops_seen = 0;
        self.prev_sequence = CodecWindowSequence::Long;
        self.last_sequence = CodecWindowSequence::Long;
        self.short_frames = 0;
        self.transient_hp_prev = [0.0; 2];
        self.transient_energy_hist = 0.0;
        self.attack_cur = false;
        self.late_cur = false;
        self.late_prev = false;
        self.reservoir_bits = 0.0;
        self.tp_detector.reset();
        self.artifact_rms_sq = 0.0;
        self.post_codec_true_peak_dbtp = [-120.0; 2];
        self.artifact_rms_db = -140.0;
    }

    #[inline]
    pub fn reset_peak_hold(&mut self) {
        self.tp_detector.reset();
        self.post_codec_true_peak_dbtp = [-120.0; 2];
    }

    /// Processes stereo frames in-place with Zero Heap Allocation.
    #[inline]
    pub fn process_frames_in_place(&mut self, io: &mut [StereoFrame]) {
        if !self.enabled {
            return;
        }

        for frame in io.iter_mut() {
            let in_l = frame[0];
            let in_r = frame[1];

            // Exact dry alignment ring (codec latency) for phase-accurate `Artifact Solo` nulling
            let dc = self.dry_cursor;
            let delayed_dry_l = self.dry_delay_ring[0][dc];
            let delayed_dry_r = self.dry_delay_ring[1][dc];
            self.dry_delay_ring[0][dc] = in_l;
            self.dry_delay_ring[1][dc] = in_r;
            self.dry_cursor = (dc + 1) % CODEC_PREVIEW_LATENCY_SAMPLES;

            let hp = self.hop_pos;
            let wet_l = self.out_fifo[0][hp] as f32;
            let wet_r = self.out_fifo[1][hp] as f32;

            self.in_hop[0][hp] = in_l as f64;
            self.in_hop[1][hp] = in_r as f64;
            self.hop_pos = hp + 1;

            if self.hop_pos >= CODEC_MDCT_BINS {
                self.hop_pos = 0;
                self.process_mdct_hop();
            }

            // Track 4x oversampled post-codec True Peak (`ITU-R BS.1770-4`)
            let _ = self.tp_detector.step_channel(0, wet_l);
            let _ = self.tp_detector.step_channel(1, wet_r);

            let delta_l = wet_l - delayed_dry_l;
            let delta_r = wet_r - delayed_dry_r;
            let delta_sq = 0.5 * ((delta_l * delta_l + delta_r * delta_r) as f64);
            self.artifact_rms_sq = 0.9992 * self.artifact_rms_sq + 0.0008 * delta_sq;

            if self.artifact_solo {
                *frame = [delta_l, delta_r];
            } else {
                *frame = [wet_l, wet_r];
            }
        }

        self.post_codec_true_peak_dbtp[0] =
            20.0 * self.tp_detector.max_true_peak_lin[0].max(1e-6).log10();
        self.post_codec_true_peak_dbtp[1] =
            20.0 * self.tp_detector.max_true_peak_lin[1].max(1e-6).log10();
        self.artifact_rms_db = (10.0 * self.artifact_rms_sq.max(1e-16).log10()) as f32;
    }

    /// Attack detector on the newest (look-ahead) hop: high-band energy of any `32`-sample
    /// sub-block exceeding `10x` (`+10 dB`) the recent average. Returns `(attack, late)`, where
    /// `late` marks an attack in the second half of the hop, beyond the short blocks of the frame
    /// that ends with this hop.
    fn detect_attack(&mut self) -> (bool, bool) {
        const SUB: usize = 32;
        let mut attack = false;
        let mut late = false;
        for blk in 0..(CODEC_MDCT_BINS / SUB) {
            let mut e = 0.0f64;
            for i in 0..SUB {
                let idx = blk * SUB + i;
                for ch in 0..2 {
                    let x = self.in_hop[ch][idx];
                    let hp = x - self.transient_hp_prev[ch];
                    self.transient_hp_prev[ch] = x;
                    e += hp * hp;
                }
            }
            e /= (2 * SUB) as f64;
            if e > 10.0 * self.transient_energy_hist && e > 1e-7 {
                attack = true;
                late |= blk * SUB >= CODEC_MDCT_BINS / 2;
            }
            self.transient_energy_hist = 0.8 * self.transient_energy_hist + 0.2 * e;
        }
        (attack, late)
    }

    fn process_mdct_hop(&mut self) {
        // Shift the look-ahead pipeline: [oldest, current, newest] <- [current, newest, in_hop].
        for ch in 0..2 {
            self.hops[ch][0] = self.hops[ch][1];
            self.hops[ch][1] = self.hops[ch][2];
            self.hops[ch][2] = self.in_hop[ch];
        }
        let (attack, late) = self.detect_attack();
        let (attack, late) = (attack && self.window_switching, late && self.window_switching);
        self.hops_seen = self.hops_seen.saturating_add(1);

        // AAC window-sequence state machine for the frame `F = (oldest, current)`; `G = (current,
        // newest)` is the next frame. Short blocks of a frame cover the second half of its first hop
        // and the first half of its second hop, so a frame must be short when its second hop holds an
        // attack or its first hop holds a late (second-half) attack.
        let short_f = self.attack_cur || self.late_prev;
        let short_g = attack || self.late_cur;
        self.late_prev = self.late_cur;
        self.late_cur = late;
        self.attack_cur = attack;
        let seq = match self.prev_sequence {
            CodecWindowSequence::LongStart => CodecWindowSequence::EightShort,
            CodecWindowSequence::EightShort => {
                if short_f || short_g {
                    CodecWindowSequence::EightShort
                } else {
                    CodecWindowSequence::LongStop
                }
            }
            _ => {
                if short_g {
                    CodecWindowSequence::LongStart
                } else {
                    CodecWindowSequence::Long
                }
            }
        };
        self.prev_sequence = seq;
        self.last_sequence = seq;
        if seq == CodecWindowSequence::EightShort {
            self.short_frames += 1;
        }

        let tabs = mdct_tables();
        let win: &[f64] = match seq {
            CodecWindowSequence::Long => &tabs.long_win,
            CodecWindowSequence::LongStart => &tabs.start_win,
            CodecWindowSequence::LongStop => &tabs.stop_win,
            CodecWindowSequence::EightShort => &tabs.long_win,
        };

        // 1. Analysis
        for ch in 0..2 {
            self.frame_buf[..CODEC_MDCT_BINS].copy_from_slice(&self.hops[ch][0]);
            self.frame_buf[CODEC_MDCT_BINS..].copy_from_slice(&self.hops[ch][1]);
            if seq == CodecWindowSequence::EightShort {
                for b in 0..NUM_SHORT_BLOCKS {
                    let off = SHORT_BLOCK_OFFSET + b * CODEC_SHORT_BINS;
                    mdct_forward::<CODEC_SHORT_BINS>(
                        &self.frame_buf[off..off + CODEC_SHORT_WINDOW_LEN],
                        &tabs.short_win,
                        &tabs.dct4_short,
                        &mut self.folded_short,
                        &mut self.short_spec[ch][b],
                    );
                }
            } else {
                mdct_forward::<CODEC_MDCT_BINS>(
                    &self.frame_buf[..],
                    win,
                    &tabs.dct4_long,
                    &mut self.folded_long,
                    &mut self.mdct_spec[ch],
                );
            }
        }

        // 2. Psychoacoustic quantizer + bit reservoir (skipped in `PcmBypass` for bit-clean TDAC)
        if self.mode != CodecTarget::PcmBypass {
            self.quantize_frame(seq);
        }

        // 3. Synthesis + overlap-add
        for ch in 0..2 {
            self.synth_buf[ch].fill(0.0);
            if seq == CodecWindowSequence::EightShort {
                for b in 0..NUM_SHORT_BLOCKS {
                    let off = SHORT_BLOCK_OFFSET + b * CODEC_SHORT_BINS;
                    let spec = self.short_spec[ch][b];
                    mdct_inverse_add::<CODEC_SHORT_BINS>(
                        &spec,
                        &tabs.short_win,
                        &tabs.dct4_short,
                        &mut self.folded_short,
                        &mut self.synth_buf[ch][off..off + CODEC_SHORT_WINDOW_LEN],
                    );
                }
            } else {
                let spec = self.mdct_spec[ch];
                mdct_inverse_add::<CODEC_MDCT_BINS>(
                    &spec,
                    win,
                    &tabs.dct4_long,
                    &mut self.folded_long,
                    &mut self.synth_buf[ch][..],
                );
            }
            for n in 0..CODEC_MDCT_BINS {
                self.out_fifo[ch][n] = self.ola_tail[ch][n] + self.synth_buf[ch][n];
                self.ola_tail[ch][n] = self.synth_buf[ch][CODEC_MDCT_BINS + n];
            }
        }
    }

    /// Codec bitrate (bits/s) driving the bit reservoir.
    fn mode_bitrate(&self) -> f64 {
        match self.mode {
            CodecTarget::PcmBypass => f64::INFINITY,
            CodecTarget::Mp3_128 | CodecTarget::Aac_128 | CodecTarget::Opus_128 => 128_000.0,
            CodecTarget::Mp3_320 => 320_000.0,
            CodecTarget::Aac_256 => 256_000.0,
        }
    }

    fn quantize_frame(&mut self, seq: CodecWindowSequence) {
        // Bit reservoir: each hop earns `bitrate * hop / fs` bits; a frame may borrow up to the
        // reservoir (capped at 4 hops) — transient (short) frames take more, steady frames refill.
        let budget = self.mode_bitrate() * (CODEC_MDCT_BINS as f64) / self.sample_rate;
        let max_reservoir = 4.0 * budget;
        let available = budget + self.reservoir_bits;
        let mut step_scale = 1.0f64;
        let mut used = 0.0f64;
        for _ in 0..6 {
            used = self.quantize_pass(seq, step_scale, true);
            if used <= available {
                break;
            }
            step_scale *= 1.19; // +1.5 dB coarser
        }
        let _ = self.quantize_pass(seq, step_scale, false);
        self.reservoir_bits = (self.reservoir_bits + budget - used).clamp(0.0, max_reservoir);
    }

    /// One quantizer pass over the frame. With `dry_run`, spectra are left untouched and only the
    /// estimated bit cost is returned.
    fn quantize_pass(&mut self, seq: CodecWindowSequence, step_scale: f64, dry_run: bool) -> f64 {
        let mut bits = 0.0f64;
        if seq == CodecWindowSequence::EightShort {
            for b in 0..NUM_SHORT_BLOCKS {
                let mut l = self.short_spec[0][b];
                let mut r = self.short_spec[1][b];
                bits += self.quantize_block(&mut l, &mut r, true, step_scale);
                if !dry_run {
                    self.short_spec[0][b] = l;
                    self.short_spec[1][b] = r;
                }
            }
        } else {
            let mut l = self.mdct_spec[0];
            let mut r = self.mdct_spec[1];
            bits += self.quantize_block(&mut l, &mut r, false, step_scale);
            if !dry_run {
                self.mdct_spec[0] = l;
                self.mdct_spec[1] = r;
            }
        }
        bits
    }

    /// Joint-stereo psychoacoustic quantizer for one MDCT block (`256` or `64` bins).
    /// Returns the estimated bit cost (magnitude entropy + per-band side info).
    fn quantize_block<const M: usize>(
        &self,
        left: &mut [f64; M],
        right: &mut [f64; M],
        short: bool,
        step_scale: f64,
    ) -> f64 {
        let (hf_cutoff_hz, smr_db, side_bit_penalty_db, opus_pvq_norm, mp3_power_law): (
            f64,
            f64,
            f64,
            bool,
            bool,
        ) = match self.mode {
            CodecTarget::PcmBypass => return 0.0,
            CodecTarget::Mp3_128 => (16_000.0, 14.5, 4.5, false, true),
            CodecTarget::Mp3_320 => (20_500.0, 29.0, 1.0, false, true),
            CodecTarget::Aac_128 => (17_000.0, 18.0, 3.0, false, false),
            CodecTarget::Aac_256 => (20_500.0, 27.5, 1.0, false, false),
            CodecTarget::Opus_128 => (20_000.0, 20.5, 2.5, true, false),
        };
        let bark_map: &[u8] = if short {
            &self.short_bin_to_bark
        } else {
            &self.bin_to_bark
        };
        let hz_per_bin = self.sample_rate / ((2 * M) as f64);

        // Joint-Stereo M/S transform (`1 / sqrt(2)` orthonormal)
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        for k in 0..M {
            let l = left[k];
            let r = right[k];
            left[k] = (l + r) * inv_sqrt2; // Mid
            right[k] = (l - r) * inv_sqrt2; // Side
        }

        let mut bits = 0.0f64;
        for ms_ch in 0..2 {
            let spec: &mut [f64; M] = if ms_ch == 0 { left } else { right };
            let mut bark_energy = [1e-16f64; NUM_BARK_BANDS];
            let mut bark_peak = [1e-12f64; NUM_BARK_BANDS];
            let mut bark_count = [0usize; NUM_BARK_BANDS];
            for k in 0..M {
                let b = bark_map[k] as usize;
                let v = spec[k];
                bark_energy[b] += v * v;
                bark_peak[b] = bark_peak[b].max(v.abs());
                bark_count[b] += 1;
            }

            // Simultaneous Bark spreading function (+25 dB/Bark upward, -10 dB/Bark downward)
            let spread_w = &mdct_tables().spread;
            let mut spread_energy = [1e-16f64; NUM_BARK_BANDS];
            for b_dst in 0..NUM_BARK_BANDS {
                let mut acc = 0.0f64;
                for b_src in 0..NUM_BARK_BANDS {
                    acc += bark_energy[b_src] * spread_w[b_dst][b_src];
                }
                spread_energy[b_dst] = acc.max(1e-16);
            }

            let ch_smr_db = if ms_ch == 1 {
                (smr_db - side_bit_penalty_db).max(6.0)
            } else {
                smr_db
            };

            let mut orig_band_e = [0.0f64; NUM_BARK_BANDS];
            let mut quant_band_e = [0.0f64; NUM_BARK_BANDS];
            let mut band_used = [false; NUM_BARK_BANDS];

            for k in 0..M {
                let f_hz = ((k as f64) + 0.5) * hz_per_bin;
                if f_hz > hf_cutoff_hz {
                    spec[k] = 0.0;
                    continue;
                }
                let b = bark_map[k] as usize;
                let x = spec[k];
                orig_band_e[b] += x * x;

                let rms_b = (spread_energy[b] / (bark_count[b].max(1) as f64)).sqrt();
                let ref_amp = 0.65 * bark_peak[b] + 0.35 * rms_b;
                let step = (ref_amp * 10.0f64.powf(-ch_smr_db / 20.0) * step_scale).max(1e-9);

                let (q_val, q_index) = if mp3_power_law {
                    // ISO 11172-3 MP3 non-uniform x^(3/4) dead-zone quantizer
                    let ax = (x.abs() / step).powf(0.75);
                    let qi = (ax - 0.12).max(0.0).round();
                    (x.signum() * qi.powf(4.0 / 3.0) * step, qi)
                } else {
                    // AAC / Opus mid-tread psychoacoustic step quantizer
                    let qi = (x / step).round();
                    (qi * step, qi.abs())
                };
                if q_index > 0.0 {
                    bits += 1.0 + 2.0 * (1.0 + q_index).log2();
                    band_used[b] = true;
                }
                spec[k] = q_val;
                quant_band_e[b] += q_val * q_val;
            }
            bits += 6.0 * band_used.iter().filter(|u| **u).count() as f64;

            // RFC 6716 Opus CELT PVQ energy conservation per Bark band
            if opus_pvq_norm {
                let mut band_gain = [1.0f64; NUM_BARK_BANDS];
                for b in 0..NUM_BARK_BANDS {
                    if quant_band_e[b] > 1e-14 && orig_band_e[b] > 1e-14 {
                        band_gain[b] = (orig_band_e[b] / quant_band_e[b]).sqrt().clamp(0.5, 2.0);
                    }
                }
                for k in 0..M {
                    let b = bark_map[k] as usize;
                    spec[k] *= band_gain[b];
                }
            }
        }

        // Inverse M/S transform back to L/R
        for k in 0..M {
            let m = left[k];
            let s = right[k];
            left[k] = (m + s) * inv_sqrt2;
            right[k] = (m - s) * inv_sqrt2;
        }
        bits
    }
}

impl OzoneDspModule for CodecPreviewModule {
    fn name(&self) -> &'static str {
        "iZotope Ozone 12 Psychoacoustic Codec Preview"
    }

    fn reset(&mut self, sample_rate: f64, _max_block_size: usize) {
        self.set_sample_rate(sample_rate);
    }

    fn process_block(&mut self, left: &mut [f32], right: &mut [f32]) {
        if !self.enabled {
            return;
        }
        let n = left.len().min(right.len());
        let mut frame = [[0.0f32; 2]; 1];
        for i in 0..n {
            frame[0] = [left[i], right[i]];
            self.process_frames_in_place(&mut frame);
            left[i] = frame[0][0];
            right[i] = frame[0][1];
        }
    }

    fn latency_samples(&self) -> usize {
        if self.enabled {
            CODEC_PREVIEW_LATENCY_SAMPLES
        } else {
            0
        }
    }
}

impl UnifiedDspNode for CodecPreviewModule {
    fn name(&self) -> &'static str {
        "iZotope Ozone 12 Psychoacoustic Codec Preview"
    }

    fn latency_samples(&self) -> usize {
        if self.enabled {
            CODEC_PREVIEW_LATENCY_SAMPLES
        } else {
            0
        }
    }

    fn prepare(&mut self, sample_rate: f64, _max_block_size: usize) {
        self.set_sample_rate(sample_rate);
    }

    fn reset_state(&mut self) {
        self.reset_internal();
    }

    fn set_parameter_normalized(&mut self, param_id: u32, norm_val: f32) {
        let v = norm_val.clamp(0.0, 1.0);
        match param_id {
            0 => self.enabled = v >= 0.5,
            1 => self.mode = CodecTarget::from_u8((v * 5.0).round() as u8),
            2 => self.artifact_solo = v >= 0.5,
            _ => {}
        }
    }

    fn process_slice(
        &mut self,
        _ctx: &ProcessContext,
        main_io: &mut [StereoFrame],
        _sidechain_in: Option<&[StereoFrame]>,
        _aux_sends_out: &mut [&mut [StereoFrame]],
    ) {
        self.process_frames_in_place(main_io);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Short frames used for one click at `pos` (input sample index) on a quiet tonal bed.
    fn short_frames_for_click(pos: usize) -> u64 {
        let mut c = CodecPreviewModule::new(48_000.0);
        c.enabled = true;
        c.mode = CodecTarget::Aac_128;
        let mut x: Vec<StereoFrame> = (0..12_288)
            .map(|i| {
                let v = 0.01 * (std::f64::consts::TAU * 220.0 * i as f64 / 48_000.0).sin() as f32;
                [v, v]
            })
            .collect();
        for k in 0..48 {
            let v = 0.8 * (-(k as f32) / 12.0).exp() * if k % 2 == 0 { 1.0 } else { -1.0 };
            x[pos + k] = [v, v];
        }
        for chunk in x.chunks_mut(256) {
            c.process_frames_in_place(chunk);
        }
        c.short_frames
    }

    #[test]
    fn test_late_attack_in_hop_is_covered_by_short_blocks() {
        let hop0 = 20 * CODEC_MDCT_BINS;
        // Early attack: the short blocks of frame (A-1, A) cover it.
        assert_eq!(short_frames_for_click(hop0 + 40), 1);
        // Late attack (second half of hop A): frame (A, A+1) must stay short too, instead of a
        // long STOP window spreading pre-echo ahead of the click.
        assert_eq!(short_frames_for_click(hop0 + 200), 2);
    }
}
