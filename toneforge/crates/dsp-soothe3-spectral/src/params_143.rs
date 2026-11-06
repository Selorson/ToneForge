//! `params_143`: Complete 143-Parameter System of **oeksound Soothe3** (`v1.0.5`)
//! verified against `extracted_targets/soothe3/DigiPageTable.xml` and all 68 `.preset` JSON files.
//!
//! Parameter Breakdown (`143` total):
//! - `26` Global Algorithm & Routing Parameters (`0..26`)
//! - `72` 8-Band Sidechain EQ Parameters (`8 Bands × 9 Params/Band`, `26..98`)
//! - `45` Multichannel / Dolby Atmos Parameters (`16` Channel Enables + `25` 5-Group Params + `4` Group Links, `98..143`)

use crate::tilt_curves::TiltPair;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const SOOTHE3_GLOBAL_PARAMS_COUNT: usize = 26;
pub const SOOTHE3_NUM_SC_BANDS: usize = 8;
pub const SOOTHE3_PARAMS_PER_SC_BAND: usize = 9;
pub const SOOTHE3_SC_BAND_PARAMS_COUNT: usize =
    SOOTHE3_NUM_SC_BANDS * SOOTHE3_PARAMS_PER_SC_BAND; // 72
pub const SOOTHE3_ATMOS_PARAMS_COUNT: usize = 45;
pub const SOOTHE3_TOTAL_PARAMS: usize =
    SOOTHE3_GLOBAL_PARAMS_COUNT + SOOTHE3_SC_BAND_PARAMS_COUNT + SOOTHE3_ATMOS_PARAMS_COUNT; // 143

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SootheProcessMode {
    Soft = 0, // Level-independent relative spectral prominence detector
    Hard = 1, // Level-dependent dynamic threshold suppressor
}

impl SootheProcessMode {
    #[inline]
    pub fn from_norm(norm: f32) -> Self {
        if norm >= 0.5 {
            Self::Hard
        } else {
            Self::Soft
        }
    }

    #[inline]
    pub fn to_norm(self) -> f32 {
        if matches!(self, Self::Hard) {
            1.0
        } else {
            0.0
        }
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SidechainFilterShape {
    LowCut = 0,
    LowShelf = 1,
    Bell = 2,
    Bandpass = 3, // Routed in PARALLEL when enabled
    BandReject = 4,
    Tilt = 5,
    HighShelf = 6,
    HighCut = 7,
}

impl SidechainFilterShape {
    #[inline]
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::LowCut,
            1 => Self::LowShelf,
            2 => Self::Bell,
            3 => Self::Bandpass,
            4 => Self::BandReject,
            5 => Self::Tilt,
            6 => Self::HighShelf,
            _ => Self::HighCut,
        }
    }

    #[inline]
    pub fn from_norm(norm: f32) -> Self {
        Self::from_u8((norm.clamp(0.0, 1.0) * 7.0).round() as u8)
    }

    #[inline]
    pub fn to_norm(self) -> f32 {
        (self as u8 as f32) / 7.0
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeResolutionQuality {
    Normal = 0, // Hop = 512 @ 1x (8x overlap, N_FFT = 4096)
    High = 1,   // Hop = 256 @ 1x (16x overlap, N_FFT = 4096)
    Ultra = 2,  // Hop = 128 @ 1x (32x overlap, N_FFT = 4096)
}

impl TimeResolutionQuality {
    #[inline]
    pub fn from_norm(norm: f32) -> Self {
        match (norm.clamp(0.0, 1.0) * 2.0).round() as u8 {
            0 => Self::Normal,
            1 => Self::High,
            _ => Self::Ultra,
        }
    }

    #[inline]
    pub fn to_norm(self) -> f32 {
        (self as u8 as f32) * 0.5
    }

    #[inline]
    pub const fn hop_size_1x(self) -> usize {
        match self {
            Self::Normal => 512,
            Self::High => 256,
            Self::Ultra => 128,
        }
    }
}

/// 9 Parameters per Sidechain EQ Band (`Band 1..8`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SidechainEqBandParams {
    pub used: bool,                 // +0 (`Band b used`)
    pub enabled: bool,              // +1 (`Band b enabled`)
    pub shape: SidechainFilterShape,// +2 (`Band b shape`: 0..7, 3 = Parallel Bandpass)
    pub freq_hz: f32,               // +3 (`Band b frequency`: 20..20_000 Hz)
    pub depth: f32,                 // +4 (`Band b depth`: -10..+10, corresponding to -24..+24 dB)
    pub q: f32,                     // +5 (`Band b q`: 0.1..16.0, default 0.707)
    pub slope: f32,                 // +6 (`Band b slope`: 0.0..10.0, default 5.0)
    pub focus: f32,                 // +7 (`Band b focus` / balance: 0.0..1.0, default 0.5)
    pub channels: u8,               // +8 (`Band b channels`: 0 = Stereo, 1 = L/Mid, 2 = R/Side)
}

impl Default for SidechainEqBandParams {
    fn default() -> Self {
        Self {
            used: false,
            enabled: false,
            shape: SidechainFilterShape::Bell,
            freq_hz: 1000.0,
            depth: 0.0,
            q: 0.707,
            slope: 5.0,
            focus: 0.5,
            channels: 0,
        }
    }
}

impl SidechainEqBandParams {
    pub const BAND_PARAM_SUFFIXES: [&'static str; SOOTHE3_PARAMS_PER_SC_BAND] = [
        "used",
        "enabled",
        "shape",
        "frequency",
        "depth",
        "q",
        "slope",
        "focus",
        "channels",
    ];

    /// Converts `depth` (`-10.0 .. +10.0`) to gain sensitivity in dB (`-24.0 .. +24.0 dB`).
    #[inline]
    pub fn depth_to_db(&self) -> f32 {
        (self.depth.clamp(-10.0, 10.0)) * 2.4
    }

    pub fn set_offset_normalized(&mut self, offset: usize, norm_val: f32) {
        let v = norm_val.clamp(0.0, 1.0);
        match offset {
            0 => self.used = v >= 0.5,
            1 => self.enabled = v >= 0.5,
            2 => self.shape = SidechainFilterShape::from_norm(v),
            3 => {
                let min_oct = 20.0f32.log2();
                let max_oct = 20_000.0f32.log2();
                self.freq_hz = (min_oct + v * (max_oct - min_oct)).exp2().clamp(20.0, 20_000.0);
            }
            4 => self.depth = 20.0 * v - 10.0,
            5 => self.q = 0.1 * (160.0f32).powf(v),
            6 => self.slope = v * 10.0,
            7 => self.focus = v,
            8 => self.channels = (v * 2.0).round() as u8,
            _ => {}
        }
    }

    pub fn get_offset_normalized(&self, offset: usize) -> f32 {
        match offset {
            0 => if self.used { 1.0 } else { 0.0 },
            1 => if self.enabled { 1.0 } else { 0.0 },
            2 => self.shape.to_norm(),
            3 => {
                let min_oct = 20.0f32.log2();
                let max_oct = 20_000.0f32.log2();
                ((self.freq_hz.clamp(20.0, 20_000.0).log2() - min_oct) / (max_oct - min_oct))
                    .clamp(0.0, 1.0)
            }
            4 => ((self.depth + 10.0) / 20.0).clamp(0.0, 1.0),
            5 => ((self.q.clamp(0.1, 16.0) / 0.1).ln() / 160.0f32.ln()).clamp(0.0, 1.0),
            6 => (self.slope * 0.1).clamp(0.0, 1.0),
            7 => self.focus.clamp(0.0, 1.0),
            8 => (self.channels.min(2) as f32) * 0.5,
            _ => 0.0,
        }
    }
}

/// 5 Parameters per Multichannel Speaker Group (`LR`, `Center`, `LFE`, `Surround`, `Ceiling`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpeakerGroupParams {
    pub enabled: bool,
    pub depth: f32, // 0.0 ..= 2.0 (default 1.0)
    pub delta: bool,
    pub solo: bool,
    pub link_all: bool,
}

impl Default for SpeakerGroupParams {
    fn default() -> Self {
        Self {
            enabled: true,
            depth: 1.0,
            delta: false,
            solo: false,
            link_all: true,
        }
    }
}

/// 26 Global Parameters of Soothe3.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Soothe3GlobalParams {
    pub mode: SootheProcessMode,         // 0 (`Mode`: 0 = soft, 1 = hard)
    pub delta_solo: bool,                // 1 (`Delta`: 0 = suppress, 1 = delta/solo)
    pub depth: f32,                      // 2 (`Depth`: 0.0..10.0, default 2.5)
    pub detail: f32,                     // 3 (`Detail`: 0.0..10.0, default 5.0, merges Sharpness + Selectivity)
    pub detail_tilt: TiltPair,           // 4, 5 (`Detail tilt low`, `Detail tilt high`: 0.1..10.0)
    pub attack: f32,                     // 6 (`Attack`: 0.0..10.0, default 2.0)
    pub attack_tilt: TiltPair,           // 7, 8 (`Attack tilt low`, `Attack tilt high`: 0.1..10.0)
    pub release: f32,                    // 9 (`Release`: 0.0..10.0, default 2.0)
    pub release_tilt: TiltPair,          // 10, 11 (`Release tilt low`, `Release tilt high`: 0.1..10.0)
    pub max_cut_db: f32,                 // 12 (`Max cut`: -36.0..0.0 dB, or +24/36 in preset)
    pub mix_pct: f32,                    // 13 (`Mix`: 0.0..100.0%)
    pub out_trim_db: f32,                // 14 (`Out trim`: -24.0..+24.0 dB)
    pub wet_trim_db: f32,                // 15 (`Wet trim`: -24.0..+24.0 dB)
    pub sidechain: bool,                 // 16 (`Sidechain`)
    pub sidechain_listen: bool,          // 17 (`Sidechain listen`)
    pub linear_phase: bool,              // 18 (`Linear phase`)
    pub low_latency: bool,               // 19 (`Low latency`)
    pub quality: TimeResolutionQuality,  // 20 (`Quality`: 0=normal, 1=high, 2=ultra)
    pub mid_side_mode: bool,             // 21 (`Stereo mode`: 0=L/R, 1=M/S)
    pub stereo_link_pct: f32,            // 22 (`Stereo link`: 0.0..100.0%)
    pub stereo_focus: f32,               // 23 (`Stereo focus`: 0.0..1.0, 0.5 = center)
    pub band_solo: bool,                 // 24 (`Band solo`)
    pub bypass: bool,                    // 25 (`Bypass`)
    pub inverse: bool,                   // Extended: `Inverse` (Spectral Unmasking / Ducking)
    pub auto_gain: bool,                 // Extended: `Auto gain`
}

impl Default for Soothe3GlobalParams {
    fn default() -> Self {
        Self {
            mode: SootheProcessMode::Soft,
            delta_solo: false,
            depth: 2.5,
            detail: 5.0,
            detail_tilt: TiltPair::default(),
            attack: 2.0,
            attack_tilt: TiltPair::default(),
            release: 2.0,
            release_tilt: TiltPair::default(),
            max_cut_db: -24.0,
            mix_pct: 100.0,
            out_trim_db: 0.0,
            wet_trim_db: 0.0,
            sidechain: false,
            sidechain_listen: false,
            linear_phase: false,
            low_latency: false,
            quality: TimeResolutionQuality::Normal,
            mid_side_mode: false,
            stereo_link_pct: 100.0,
            stereo_focus: 0.5,
            band_solo: false,
            bypass: false,
            inverse: false,
            auto_gain: false,
        }
    }
}

/// Complete 143-Parameter State of oeksound Soothe3.
#[derive(Clone, Debug)]
pub struct Soothe3ParameterSet {
    pub global: Soothe3GlobalParams,
    pub sc_bands: [SidechainEqBandParams; SOOTHE3_NUM_SC_BANDS],
    pub channel_enabled: [bool; 16],
    pub group_lr: SpeakerGroupParams,
    pub group_center: SpeakerGroupParams,
    pub group_lfe: SpeakerGroupParams,
    pub group_surround: SpeakerGroupParams,
    pub group_ceiling: SpeakerGroupParams,
    pub link_all: bool,
    pub group_link_lcr: f32,
    pub group_link_surround: f32,
    pub group_link_ceiling: f32,
    pub normalized_mirror: [f32; SOOTHE3_TOTAL_PARAMS],
}

impl Default for Soothe3ParameterSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Serde representation of a `.preset` JSON file in `extracted_targets/soothe3/Presets/`.
#[derive(Deserialize)]
pub struct Soothe3PresetJson {
    pub name: Option<String>,
    pub params: HashMap<String, f32>,
}

impl Soothe3ParameterSet {
    pub const GLOBAL_NAMES: [&'static str; SOOTHE3_GLOBAL_PARAMS_COUNT] = [
        "Mode",
        "Delta",
        "Depth",
        "Detail",
        "Detail tilt low",
        "Detail tilt high",
        "Attack",
        "Attack tilt low",
        "Attack tilt high",
        "Release",
        "Release tilt low",
        "Release tilt high",
        "Max cut",
        "Mix",
        "Out trim",
        "Wet trim",
        "Sidechain",
        "Sidechain listen",
        "Linear phase",
        "Low latency",
        "Quality",
        "Stereo mode",
        "Stereo link",
        "Stereo focus",
        "Band solo",
        "Bypass",
    ];

    pub const ATMOS_NAMES: [&'static str; SOOTHE3_ATMOS_PARAMS_COUNT] = [
        "Channel 1 enabled",
        "Channel 2 enabled",
        "Channel 3 enabled",
        "Channel 4 enabled",
        "Channel 5 enabled",
        "Channel 6 enabled",
        "Channel 7 enabled",
        "Channel 8 enabled",
        "Channel 9 enabled",
        "Channel 10 enabled",
        "Channel 11 enabled",
        "Channel 12 enabled",
        "Channel 13 enabled",
        "Channel 14 enabled",
        "Channel 15 enabled",
        "Channel 16 enabled",
        "LR enabled",
        "LR depth",
        "LR delta",
        "LR solo",
        "LR link all",
        "Center enabled",
        "Center depth",
        "Center delta",
        "Center solo",
        "Center link all",
        "LFE enabled",
        "LFE depth",
        "LFE delta",
        "LFE solo",
        "LFE link all",
        "Surround enabled",
        "Surround depth",
        "Surround delta",
        "Surround solo",
        "Surround link all",
        "Ceiling enabled",
        "Ceiling depth",
        "Ceiling delta",
        "Ceiling solo",
        "Ceiling link all",
        "Link all",
        "Group link LCR",
        "Group link surround",
        "Group link ceiling",
    ];

    pub fn new() -> Self {
        let mut sc_bands = [SidechainEqBandParams::default(); SOOTHE3_NUM_SC_BANDS];
        // Match default band frequencies from `Flat start.preset`
        sc_bands[0].shape = SidechainFilterShape::LowCut;
        sc_bands[0].freq_hz = 100.0;
        sc_bands[1].shape = SidechainFilterShape::Bell;
        sc_bands[1].freq_hz = 4000.0;
        sc_bands[1].depth = 2.5;
        sc_bands[2].shape = SidechainFilterShape::HighCut;
        sc_bands[2].freq_hz = 16000.0;

        let mut s = Self {
            global: Soothe3GlobalParams::default(),
            sc_bands,
            channel_enabled: [true; 16],
            group_lr: SpeakerGroupParams::default(),
            group_center: SpeakerGroupParams {
                link_all: false,
                ..SpeakerGroupParams::default()
            },
            group_lfe: SpeakerGroupParams {
                link_all: false,
                ..SpeakerGroupParams::default()
            },
            group_surround: SpeakerGroupParams::default(),
            group_ceiling: SpeakerGroupParams::default(),
            link_all: false,
            group_link_lcr: 0.0,
            group_link_surround: 0.0,
            group_link_ceiling: 0.0,
            normalized_mirror: [0.0; SOOTHE3_TOTAL_PARAMS],
        };
        s.sync_mirror_from_structured();
        s
    }

    pub fn sync_mirror_from_structured(&mut self) {
        for id in 0..SOOTHE3_TOTAL_PARAMS {
            self.normalized_mirror[id] = self.compute_normalized_for_id(id);
        }
    }

    fn compute_normalized_for_id(&self, idx: usize) -> f32 {
        if idx < SOOTHE3_GLOBAL_PARAMS_COUNT {
            let g = &self.global;
            match idx {
                0 => g.mode.to_norm(),
                1 => if g.delta_solo { 1.0 } else { 0.0 },
                2 => (g.depth * 0.1).clamp(0.0, 1.0),
                3 => (g.detail * 0.1).clamp(0.0, 1.0),
                4 => TiltPair::mult_to_norm(g.detail_tilt.low_mult),
                5 => TiltPair::mult_to_norm(g.detail_tilt.high_mult),
                6 => (g.attack * 0.1).clamp(0.0, 1.0),
                7 => TiltPair::mult_to_norm(g.attack_tilt.low_mult),
                8 => TiltPair::mult_to_norm(g.attack_tilt.high_mult),
                9 => (g.release * 0.1).clamp(0.0, 1.0),
                10 => TiltPair::mult_to_norm(g.release_tilt.low_mult),
                11 => TiltPair::mult_to_norm(g.release_tilt.high_mult),
                12 => (g.max_cut_db.abs() / 36.0).clamp(0.0, 1.0),
                13 => (g.mix_pct * 0.01).clamp(0.0, 1.0),
                14 => ((g.out_trim_db + 24.0) / 48.0).clamp(0.0, 1.0),
                15 => ((g.wet_trim_db + 24.0) / 48.0).clamp(0.0, 1.0),
                16 => if g.sidechain { 1.0 } else { 0.0 },
                17 => if g.sidechain_listen { 1.0 } else { 0.0 },
                18 => if g.linear_phase { 1.0 } else { 0.0 },
                19 => if g.low_latency { 1.0 } else { 0.0 },
                20 => g.quality.to_norm(),
                21 => if g.mid_side_mode { 1.0 } else { 0.0 },
                22 => (g.stereo_link_pct * 0.01).clamp(0.0, 1.0),
                23 => g.stereo_focus.clamp(0.0, 1.0),
                24 => if g.band_solo { 1.0 } else { 0.0 },
                25 => if g.bypass { 1.0 } else { 0.0 },
                _ => 0.0,
            }
        } else if idx < SOOTHE3_GLOBAL_PARAMS_COUNT + SOOTHE3_SC_BAND_PARAMS_COUNT {
            let rel = idx - SOOTHE3_GLOBAL_PARAMS_COUNT;
            let band_idx = rel / SOOTHE3_PARAMS_PER_SC_BAND;
            let off = rel % SOOTHE3_PARAMS_PER_SC_BAND;
            self.sc_bands[band_idx].get_offset_normalized(off)
        } else {
            let atm = idx - (SOOTHE3_GLOBAL_PARAMS_COUNT + SOOTHE3_SC_BAND_PARAMS_COUNT);
            if atm < 16 {
                return if self.channel_enabled[atm] { 1.0 } else { 0.0 };
            }
            let grp_rel = atm - 16;
            if grp_rel < 25 {
                let grp_idx = grp_rel / 5;
                let prop = grp_rel % 5;
                let grp = match grp_idx {
                    0 => &self.group_lr,
                    1 => &self.group_center,
                    2 => &self.group_lfe,
                    3 => &self.group_surround,
                    _ => &self.group_ceiling,
                };
                return match prop {
                    0 => if grp.enabled { 1.0 } else { 0.0 },
                    1 => (grp.depth * 0.5).clamp(0.0, 1.0),
                    2 => if grp.delta { 1.0 } else { 0.0 },
                    3 => if grp.solo { 1.0 } else { 0.0 },
                    _ => if grp.link_all { 1.0 } else { 0.0 },
                };
            }
            match grp_rel - 25 {
                0 => if self.link_all { 1.0 } else { 0.0 },
                1 => self.group_link_lcr.clamp(0.0, 1.0),
                2 => self.group_link_surround.clamp(0.0, 1.0),
                3 => self.group_link_ceiling.clamp(0.0, 1.0),
                _ => 0.0,
            }
        }
    }

    pub fn set_normalized(&mut self, param_id: u32, norm_val: f32) {
        let idx = param_id as usize;
        if idx >= SOOTHE3_TOTAL_PARAMS {
            return;
        }
        let v = norm_val.clamp(0.0, 1.0);
        self.normalized_mirror[idx] = v;

        if idx < SOOTHE3_GLOBAL_PARAMS_COUNT {
            let g = &mut self.global;
            match idx {
                0 => g.mode = SootheProcessMode::from_norm(v),
                1 => g.delta_solo = v >= 0.5,
                2 => g.depth = v * 10.0,
                3 => g.detail = v * 10.0,
                4 => g.detail_tilt.low_mult = TiltPair::norm_to_mult(v),
                5 => g.detail_tilt.high_mult = TiltPair::norm_to_mult(v),
                6 => g.attack = v * 10.0,
                7 => g.attack_tilt.low_mult = TiltPair::norm_to_mult(v),
                8 => g.attack_tilt.high_mult = TiltPair::norm_to_mult(v),
                9 => g.release = v * 10.0,
                10 => g.release_tilt.low_mult = TiltPair::norm_to_mult(v),
                11 => g.release_tilt.high_mult = TiltPair::norm_to_mult(v),
                12 => g.max_cut_db = -(v * 36.0),
                13 => g.mix_pct = v * 100.0,
                14 => g.out_trim_db = 48.0 * v - 24.0,
                15 => g.wet_trim_db = 48.0 * v - 24.0,
                16 => g.sidechain = v >= 0.5,
                17 => g.sidechain_listen = v >= 0.5,
                18 => g.linear_phase = v >= 0.5,
                19 => g.low_latency = v >= 0.5,
                20 => g.quality = TimeResolutionQuality::from_norm(v),
                21 => g.mid_side_mode = v >= 0.5,
                22 => g.stereo_link_pct = v * 100.0,
                23 => g.stereo_focus = v,
                24 => g.band_solo = v >= 0.5,
                25 => g.bypass = v >= 0.5,
                _ => {}
            }
        } else if idx < SOOTHE3_GLOBAL_PARAMS_COUNT + SOOTHE3_SC_BAND_PARAMS_COUNT {
            let rel = idx - SOOTHE3_GLOBAL_PARAMS_COUNT;
            let band_idx = rel / SOOTHE3_PARAMS_PER_SC_BAND;
            let off = rel % SOOTHE3_PARAMS_PER_SC_BAND;
            self.sc_bands[band_idx].set_offset_normalized(off, v);
        } else {
            let atm = idx - (SOOTHE3_GLOBAL_PARAMS_COUNT + SOOTHE3_SC_BAND_PARAMS_COUNT);
            if atm < 16 {
                self.channel_enabled[atm] = v >= 0.5;
                return;
            }
            let grp_rel = atm - 16;
            if grp_rel < 25 {
                let grp_idx = grp_rel / 5;
                let prop = grp_rel % 5;
                let grp = match grp_idx {
                    0 => &mut self.group_lr,
                    1 => &mut self.group_center,
                    2 => &mut self.group_lfe,
                    3 => &mut self.group_surround,
                    _ => &mut self.group_ceiling,
                };
                match prop {
                    0 => grp.enabled = v >= 0.5,
                    1 => grp.depth = v * 2.0,
                    2 => grp.delta = v >= 0.5,
                    3 => grp.solo = v >= 0.5,
                    _ => grp.link_all = v >= 0.5,
                }
                return;
            }
            match grp_rel - 25 {
                0 => self.link_all = v >= 0.5,
                1 => self.group_link_lcr = v,
                2 => self.group_link_surround = v,
                3 => self.group_link_ceiling = v,
                _ => {}
            }
        }
    }

    #[inline]
    pub fn get_normalized(&self, param_id: u32) -> f32 {
        let idx = param_id as usize;
        if idx < SOOTHE3_TOTAL_PARAMS {
            self.normalized_mirror[idx]
        } else {
            0.0
        }
    }

    /// Resolves a Soothe3 parameter name from `DigiPageTable.xml` or `.preset` JSON to `0..143`.
    pub fn find_param_id_by_name(name: &str) -> Option<u32> {
        for (i, &gname) in Self::GLOBAL_NAMES.iter().enumerate() {
            if gname.eq_ignore_ascii_case(name) {
                return Some(i as u32);
            }
        }
        if let Some(rest) = name.strip_prefix("Band ") {
            let (band_str, suffix) = rest.split_once(' ')?;
            let band_1based: usize = band_str.parse().ok()?;
            if !(1..=SOOTHE3_NUM_SC_BANDS).contains(&band_1based) {
                return None;
            }
            let band_idx = band_1based - 1;
            for (off, &suf) in SidechainEqBandParams::BAND_PARAM_SUFFIXES.iter().enumerate() {
                if suf.eq_ignore_ascii_case(suffix) {
                    return Some(
                        (SOOTHE3_GLOBAL_PARAMS_COUNT + band_idx * SOOTHE3_PARAMS_PER_SC_BAND + off)
                            as u32,
                    );
                }
            }
            return None;
        }
        for (a, &aname) in Self::ATMOS_NAMES.iter().enumerate() {
            if aname.eq_ignore_ascii_case(name) {
                return Some(
                    (SOOTHE3_GLOBAL_PARAMS_COUNT + SOOTHE3_SC_BAND_PARAMS_COUNT + a) as u32,
                );
            }
        }
        None
    }

    /// Loads parameter values from a Soothe3 `.preset` JSON string (`143` keys in `params`).
    pub fn load_from_preset_json(&mut self, json_str: &str) -> Result<usize, String> {
        let preset: Soothe3PresetJson =
            serde_json::from_str(json_str).map_err(|e| e.to_string())?;
        let mut loaded = 0usize;
        for (key, &val) in &preset.params {
            if self.set_raw_value_by_name(key, val) {
                loaded += 1;
            }
        }
        self.sync_mirror_from_structured();
        Ok(loaded)
    }

    pub fn set_raw_value_by_name(&mut self, name: &str, raw_val: f32) -> bool {
        let Some(id) = Self::find_param_id_by_name(name) else {
            return false;
        };
        let idx = id as usize;
        if idx < SOOTHE3_GLOBAL_PARAMS_COUNT {
            let g = &mut self.global;
            match idx {
                0 => g.mode = if raw_val >= 0.5 { SootheProcessMode::Hard } else { SootheProcessMode::Soft },
                1 => g.delta_solo = raw_val >= 0.5,
                2 => g.depth = raw_val.clamp(0.0, 10.0),
                3 => g.detail = raw_val.clamp(0.0, 10.0),
                4 => g.detail_tilt.low_mult = raw_val.clamp(0.1, 10.0),
                5 => g.detail_tilt.high_mult = raw_val.clamp(0.1, 10.0),
                6 => g.attack = raw_val.clamp(0.0, 10.0),
                7 => g.attack_tilt.low_mult = raw_val.clamp(0.1, 10.0),
                8 => g.attack_tilt.high_mult = raw_val.clamp(0.1, 10.0),
                9 => g.release = raw_val.clamp(0.0, 10.0),
                10 => g.release_tilt.low_mult = raw_val.clamp(0.1, 10.0),
                11 => g.release_tilt.high_mult = raw_val.clamp(0.1, 10.0),
                12 => g.max_cut_db = -raw_val.abs().clamp(0.0, 36.0),
                13 => g.mix_pct = raw_val.clamp(0.0, 100.0),
                14 => g.out_trim_db = raw_val.clamp(-24.0, 24.0),
                15 => g.wet_trim_db = raw_val.clamp(-24.0, 24.0),
                16 => g.sidechain = raw_val >= 0.5,
                17 => g.sidechain_listen = raw_val >= 0.5,
                18 => g.linear_phase = raw_val >= 0.5,
                19 => g.low_latency = raw_val >= 0.5,
                20 => {
                    g.quality = match raw_val.round() as u8 {
                        0 => TimeResolutionQuality::Normal,
                        1 => TimeResolutionQuality::High,
                        _ => TimeResolutionQuality::Ultra,
                    }
                }
                21 => g.mid_side_mode = raw_val >= 0.5,
                22 => g.stereo_link_pct = if raw_val <= 1.0 { raw_val * 100.0 } else { raw_val.clamp(0.0, 100.0) },
                23 => g.stereo_focus = raw_val.clamp(0.0, 1.0),
                24 => g.band_solo = raw_val >= 0.5,
                25 => g.bypass = raw_val >= 0.5,
                _ => {}
            }
            return true;
        }
        if idx < SOOTHE3_GLOBAL_PARAMS_COUNT + SOOTHE3_SC_BAND_PARAMS_COUNT {
            let rel = idx - SOOTHE3_GLOBAL_PARAMS_COUNT;
            let band_idx = rel / SOOTHE3_PARAMS_PER_SC_BAND;
            let off = rel % SOOTHE3_PARAMS_PER_SC_BAND;
            let b = &mut self.sc_bands[band_idx];
            match off {
                0 => b.used = raw_val >= 0.5,
                1 => b.enabled = raw_val >= 0.5,
                2 => b.shape = SidechainFilterShape::from_u8(raw_val.round() as u8),
                3 => b.freq_hz = raw_val.clamp(20.0, 20_000.0),
                4 => b.depth = raw_val.clamp(-10.0, 10.0),
                5 => b.q = raw_val.clamp(0.1, 16.0),
                6 => b.slope = raw_val.clamp(0.0, 10.0),
                7 => b.focus = raw_val.clamp(0.0, 1.0),
                8 => b.channels = raw_val.round() as u8,
                _ => {}
            }
            return true;
        }
        self.set_normalized(id, raw_val.clamp(0.0, 1.0));
        true
    }
}
