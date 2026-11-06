//! `params_601`: Complete 601-Parameter System (`24 Bands × 24 Params + 25 Global Params`)
//! compatible with `FabFilter Pro-Q 4.clap` (`v4.13`) and `DigiPageTable.xml`.
//!
//! Includes octave-domain frequency mapping (`log2(f_Hz) in [3.321928095, 14.87267488]`),
//! logarithmic `Q` mapping (`Q_UI = 40^(2*x_norm - 1)`), 9 EQ filter shapes (+ AllPass)
//! mapped to 13 internal `AnalogFilterPrototype` indices (`0x10F6` gain-dependent bitmask,
//! `0x1074` Gain-Q interaction bitmask), and `Stereo` / `Left` / `Right` / `Mid` / `Side` routing.

use serde::{Deserialize, Serialize};

pub const PROQ4_NUM_BANDS: usize = 24;
pub const PROQ4_PARAMS_PER_BAND: usize = 24;
pub const PROQ4_BAND_PARAMS_TOTAL: usize = PROQ4_NUM_BANDS * PROQ4_PARAMS_PER_BAND; // 576
pub const PROQ4_GLOBAL_PARAMS: usize = 25;
pub const PROQ4_TOTAL_PARAMS: usize = PROQ4_BAND_PARAMS_TOTAL + PROQ4_GLOBAL_PARAMS; // 601

pub const PROQ4_DISCRETE_PARAM_COUNT: usize = 333;
pub const PROQ4_CONTINUOUS_PARAM_COUNT: usize = 268;

/// Minimum frequency in Hz (`10.0 Hz`) and its base-2 logarithm (`3.321928094887362`).
pub const PROQ4_MIN_FREQ_HZ: f32 = 10.0;
pub const PROQ4_MAX_FREQ_HZ: f32 = 30_000.0;
pub const PROQ4_MIN_FREQ_OCT: f64 = 3.321928094887362; // log2(10.0)
pub const PROQ4_MAX_FREQ_OCT: f64 = 14.872674880270606; // log2(30000.0)
pub const PROQ4_SPAN_FREQ_OCT: f64 = PROQ4_MAX_FREQ_OCT - PROQ4_MIN_FREQ_OCT; // 11.550746785383244

/// Filter prototype bitmask `0x10F6` (`0b0001_0000_1111_0110`) identifying Gain-dependent shapes
/// (`0x18010F7E0`: bits `1, 2, 4, 5, 6, 7, 12`).
pub const PROQ4_GAIN_DEPENDENT_MASK: u32 = 0x10F6;

/// Filter prototype bitmask `0x1074` (`0b0001_0000_0111_0100`) identifying shapes with
/// analog-console Gain-Q interaction (`0x18010FF70`: bits `2, 4, 5, 6, 12`).
pub const PROQ4_GAIN_Q_INTERACTION_MASK: u32 = 0x1074;

/// Analog Gain-Q interaction base constant (`2^(1/8) = 1.0905077326652577` at `0x18010FF70`).
pub const PROQ4_GAIN_Q_INTERACTION_BASE: f64 = 1.0905077326652577;

/// Linear Phase FIR tap table @ 1x sample rate (`0x180212910` / `0x18019EAE0`).
pub const TAP_TABLE_1X: [usize; 5] = [4096, 8192, 16384, 32768, 131072];

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessingMode {
    ZeroLatency = 0,
    NaturalPhase = 1,
    LinearPhase = 2,
}

impl ProcessingMode {
    #[inline]
    pub fn from_norm(norm: f32) -> Self {
        match (norm.clamp(0.0, 1.0) * 2.0).round() as u8 {
            0 => Self::ZeroLatency,
            1 => Self::NaturalPhase,
            _ => Self::LinearPhase,
        }
    }

    #[inline]
    pub fn to_norm(self) -> f32 {
        (self as u8 as f32) * 0.5
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinearPhaseResolution {
    Low = 0,      // 4096 taps @ 1x (2048 samples latency)
    Medium = 1,   // 8192 taps @ 1x (4096 samples latency)
    High = 2,     // 16384 taps @ 1x (8192 samples latency)
    VeryHigh = 3, // 32768 taps @ 1x (16384 samples latency)
    Maximum = 4,  // 131072 taps @ 1x (65536 samples latency)
}

impl LinearPhaseResolution {
    pub const TAP_TABLE_1X: [usize; 5] = TAP_TABLE_1X;

    #[inline]
    pub fn from_norm(norm: f32) -> Self {
        match (norm.clamp(0.0, 1.0) * 4.0).round() as u8 {
            0 => Self::Low,
            1 => Self::Medium,
            2 => Self::High,
            3 => Self::VeryHigh,
            _ => Self::Maximum,
        }
    }

    #[inline]
    pub fn to_norm(self) -> f32 {
        (self as u8 as f32) * 0.25
    }

    #[inline]
    pub fn sample_rate_scale(sample_rate: f32) -> usize {
        ((sample_rate / 48_000.0).round().max(1.0) as usize).next_power_of_two()
    }

    #[inline]
    pub fn fir_length(self, sample_rate: f32) -> usize {
        Self::sample_rate_scale(sample_rate) * Self::TAP_TABLE_1X[self as usize]
    }

    #[inline]
    pub fn latency_samples(self, sample_rate: f32) -> usize {
        self.fir_length(sample_rate) >> 1
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CharacterMode {
    Clean = 0,
    Subtle = 1,
    Warm = 2,
}

impl CharacterMode {
    #[inline]
    pub fn from_norm(norm: f32) -> Self {
        match (norm.clamp(0.0, 1.0) * 2.0).round() as u8 {
            0 => Self::Clean,
            1 => Self::Subtle,
            _ => Self::Warm,
        }
    }

    #[inline]
    pub fn to_norm(self) -> f32 {
        (self as u8 as f32) * 0.5
    }
}

/// UI Filter Shapes (`0..9` in `DigiPageTable.xml`).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterShape {
    Bell = 0,
    LowShelf = 1,
    HighShelf = 2,
    LowCut = 3,
    HighCut = 4,
    Notch = 5,
    BandPass = 6,
    TiltShelf = 7,
    FlatTilt = 8,
    AllPass = 9,
}

/// Internal 13-state `AnalogFilterPrototype` (`[rcx+0x880]` at `0x18010F7E0`),
/// aligned to bitmask `0x10F6` (gain-dependent) and `0x1074` (Gain-Q interaction).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnalogFilterPrototype {
    LowCut = 0,        // Bit 0: non-gain-dependent
    TiltShelf = 1,     // Bit 1: gain-dependent (`0x10F6`)
    Bell = 2,          // Bit 2: gain-dependent (`0x10F6`) & Gain-Q (`0x1074`)
    HighCut = 3,       // Bit 3: non-gain-dependent
    LowShelf = 4,      // Bit 4: gain-dependent (`0x10F6`) & Gain-Q (`0x1074`)
    HighShelf = 5,     // Bit 5: gain-dependent (`0x10F6`) & Gain-Q (`0x1074`)
    BellResonant = 6,  // Bit 6: gain-dependent (`0x10F6`) & Gain-Q (`0x1074`)
    FlatTilt = 7,      // Bit 7: gain-dependent (`0x10F6`)
    Notch = 8,         // Bit 8: non-gain-dependent
    BandPass = 9,      // Bit 9: non-gain-dependent
    AllPass = 10,      // Bit 10: non-gain-dependent
    BrickwallCut = 11, // Bit 11: non-gain-dependent
    ShelfResonant = 12,// Bit 12: gain-dependent (`0x10F6`) & Gain-Q (`0x1074`)
}

impl AnalogFilterPrototype {
    #[inline]
    pub const fn is_gain_dependent(self) -> bool {
        ((PROQ4_GAIN_DEPENDENT_MASK >> (self as u32)) & 1) != 0
    }

    #[inline]
    pub const fn has_gain_q_interaction(self) -> bool {
        ((PROQ4_GAIN_Q_INTERACTION_MASK >> (self as u32)) & 1) != 0
    }
}

impl FilterShape {
    /// All 9 core EQ filter shapes specified in Pro-Q 4.
    pub const ALL_9_SHAPES: [FilterShape; 9] = [
        FilterShape::Bell,
        FilterShape::LowShelf,
        FilterShape::HighShelf,
        FilterShape::LowCut,
        FilterShape::HighCut,
        FilterShape::Notch,
        FilterShape::BandPass,
        FilterShape::TiltShelf,
        FilterShape::FlatTilt,
    ];

    /// All 10 UI filter shapes including `AllPass`.
    pub const ALL_10_SHAPES: [FilterShape; 10] = [
        FilterShape::Bell,
        FilterShape::LowShelf,
        FilterShape::HighShelf,
        FilterShape::LowCut,
        FilterShape::HighCut,
        FilterShape::Notch,
        FilterShape::BandPass,
        FilterShape::TiltShelf,
        FilterShape::FlatTilt,
        FilterShape::AllPass,
    ];

    #[inline]
    pub fn from_norm(norm: f32) -> Self {
        match (norm.clamp(0.0, 1.0) * 9.0).round() as u8 {
            0 => Self::Bell,
            1 => Self::LowShelf,
            2 => Self::HighShelf,
            3 => Self::LowCut,
            4 => Self::HighCut,
            5 => Self::Notch,
            6 => Self::BandPass,
            7 => Self::TiltShelf,
            8 => Self::FlatTilt,
            _ => Self::AllPass,
        }
    }

    #[inline]
    pub fn to_norm(self) -> f32 {
        (self as u8 as f32) / 9.0
    }

    #[inline]
    pub const fn to_prototype(self, is_brickwall: bool, q_ui: f32) -> AnalogFilterPrototype {
        if is_brickwall && matches!(self, Self::LowCut | Self::HighCut) {
            return AnalogFilterPrototype::BrickwallCut;
        }
        match self {
            Self::Bell => {
                if q_ui > 4.0 {
                    AnalogFilterPrototype::BellResonant
                } else {
                    AnalogFilterPrototype::Bell
                }
            }
            Self::LowShelf => {
                if q_ui > 1.5 {
                    AnalogFilterPrototype::ShelfResonant
                } else {
                    AnalogFilterPrototype::LowShelf
                }
            }
            Self::HighShelf => {
                if q_ui > 1.5 {
                    AnalogFilterPrototype::ShelfResonant
                } else {
                    AnalogFilterPrototype::HighShelf
                }
            }
            Self::LowCut => AnalogFilterPrototype::LowCut,
            Self::HighCut => AnalogFilterPrototype::HighCut,
            Self::Notch => AnalogFilterPrototype::Notch,
            Self::BandPass => AnalogFilterPrototype::BandPass,
            Self::TiltShelf => AnalogFilterPrototype::TiltShelf,
            Self::FlatTilt => AnalogFilterPrototype::FlatTilt,
            Self::AllPass => AnalogFilterPrototype::AllPass,
        }
    }

    #[inline]
    pub const fn is_gain_dependent(self) -> bool {
        self.to_prototype(false, 1.0).is_gain_dependent()
    }

    #[inline]
    pub const fn has_gain_q_interaction(self) -> bool {
        self.to_prototype(false, 1.0).has_gain_q_interaction()
    }

    #[inline]
    pub const fn supports_spectral_dynamics(self) -> bool {
        matches!(
            self,
            Self::Bell | Self::LowShelf | Self::HighShelf | Self::TiltShelf
        )
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StereoPlacement {
    Stereo = 0,
    Left = 1,
    Right = 2,
    Mid = 3,
    Side = 4,
}

impl StereoPlacement {
    #[inline]
    pub fn from_norm(norm: f32) -> Self {
        match (norm.clamp(0.0, 1.0) * 4.0).round() as u8 {
            0 => Self::Stereo,
            1 => Self::Left,
            2 => Self::Right,
            3 => Self::Mid,
            _ => Self::Side,
        }
    }

    #[inline]
    pub fn to_norm(self) -> f32 {
        (self as u8 as f32) * 0.25
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChannelMode {
    LeftRight = 0,
    MidSide = 1,
}

impl ChannelMode {
    #[inline]
    pub fn from_norm(norm: f32) -> Self {
        if norm >= 0.5 {
            Self::MidSide
        } else {
            Self::LeftRight
        }
    }

    #[inline]
    pub fn to_norm(self) -> f32 {
        if matches!(self, Self::MidSide) {
            1.0
        } else {
            0.0
        }
    }
}

/// Converts normalized `[0.0, 1.0]` to frequency in Hz (`10.0 ..= 30_000.0`) via octave mapping.
#[inline]
pub fn norm_to_freq_hz(norm: f32) -> f32 {
    let x = (norm as f64).clamp(0.0, 1.0);
    let oct = PROQ4_MIN_FREQ_OCT + PROQ4_SPAN_FREQ_OCT * x;
    oct.exp2().clamp(10.0, 30_000.0) as f32
}

/// Converts frequency in Hz (`10.0 ..= 30_000.0`) to normalized `[0.0, 1.0]` via octave mapping.
#[inline]
pub fn freq_hz_to_norm(freq_hz: f32) -> f32 {
    let f = (freq_hz as f64).clamp(10.0, 30_000.0);
    let oct = f.log2();
    ((oct - PROQ4_MIN_FREQ_OCT) / PROQ4_SPAN_FREQ_OCT).clamp(0.0, 1.0) as f32
}

/// Converts normalized `[0.0, 1.0]` to UI `Q` (`0.025 ..= 40.0`, `0.5 -> 1.0`).
#[inline]
pub fn norm_to_q_ui(norm: f32) -> f32 {
    let x = (norm as f64).clamp(0.0, 1.0);
    40.0f64.powf(2.0 * x - 1.0).clamp(0.025, 40.0) as f32
}

/// Converts UI `Q` (`0.025 ..= 40.0`) to normalized `[0.0, 1.0]`.
#[inline]
pub fn q_ui_to_norm(q_ui: f32) -> f32 {
    let q = (q_ui as f64).clamp(0.025, 40.0);
    ((q.ln() / 40.0f64.ln() + 1.0) * 0.5).clamp(0.0, 1.0) as f32
}

/// Converts normalized `[0.0, 1.0]` to continuous slope in dB/oct (`6.0 ..= 96.0`) and `is_brickwall`.
#[inline]
pub fn norm_to_slope_db_oct(norm: f32) -> (f32, bool) {
    let x = norm.clamp(0.0, 1.0);
    if x >= 0.995 {
        return (96.0, true);
    }
    let db_oct = 6.0 + (x / 0.99) * 90.0;
    (db_oct.clamp(6.0, 96.0), false)
}

/// Converts slope in dB/oct (`6.0 ..= 96.0`) and `is_brickwall` to normalized `[0.0, 1.0]`.
#[inline]
pub fn slope_db_oct_to_norm(slope_db_oct: f32, is_brickwall: bool) -> f32 {
    if is_brickwall {
        return 1.0;
    }
    (((slope_db_oct.clamp(6.0, 96.0) - 6.0) / 90.0) * 0.99).clamp(0.0, 0.99)
}

/// 24 Parameters per Band (`Band 1..24`, offsets `0..24`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EqBandParams {
    pub used: bool,             // +0 (`Band k Used`)
    pub enabled: bool,          // +1 (`Band k Enabled`)
    pub freq_hz: f32,           // +2 (`Band k Frequency`: 10..30_000 Hz)
    pub gain_db: f32,           // +3 (`Band k Gain`: -30..+30 dB)
    pub q_ui: f32,              // +4 (`Band k Q`: 0.025..40.0)
    pub shape: FilterShape,     // +5 (`Band k Shape`: 0..9)
    pub slope_db_oct: f32,      // +6 (`Band k Slope`: 6..96 dB/oct continuous)
    pub is_brickwall: bool,     // +6 (`Slope = 9 Brickwall`)
    pub placement: StereoPlacement, // +7 (`Band k Stereo Placement`: 0..4)
    pub speaker_mask: u64,      // +8 (`Band k Speakers`: 0..64)
    pub dyn_enabled: bool,      // +9 (`Band k Dynamics Enabled`)
    pub dyn_range_db: f32,      // +10 (`Band k Dynamic Range`: -30..+30 dB)
    pub dyn_auto: bool,         // +11 (`Band k Dynamics Auto`)
    pub dyn_threshold_db: f32,  // +12 (`Band k Threshold`: -90..0 dB)
    pub dyn_attack_pct: f32,    // +13 (`Band k Attack`: 0..100%)
    pub dyn_release_pct: f32,   // +14 (`Band k Release`: 0..100%)
    pub ext_sidechain: bool,    // +15 (`Band k External Side Chain`)
    pub sc_filtering: bool,     // +16 (`Band k Side Chain Filtering`)
    pub sc_low_hz: f32,         // +17 (`Band k Side Chain Low Frequency`)
    pub sc_high_hz: f32,        // +18 (`Band k Side Chain High Frequency`)
    pub sc_audition: bool,      // +19 (`Band k Side Chain Audition`)
    pub spectral_enabled: bool, // +20 (`Band k Spectral Enabled`)
    pub spectral_density: f32,  // +21 (`Band k Spectral Density`: 0..100%)
    pub solo: bool,             // +22 (`Band k Solo`)
    pub invert_gain: bool,      // +23 (`Band k Invert Gain` / `Spectral Tilt`)
}

impl Default for EqBandParams {
    fn default() -> Self {
        Self {
            used: false,
            enabled: false,
            freq_hz: 1000.0,
            gain_db: 0.0,
            q_ui: 1.0,
            shape: FilterShape::Bell,
            slope_db_oct: 12.0,
            is_brickwall: false,
            placement: StereoPlacement::Stereo,
            speaker_mask: 0x3, // L + R
            dyn_enabled: false,
            dyn_range_db: 0.0,
            dyn_auto: true,
            dyn_threshold_db: -24.0,
            dyn_attack_pct: 50.0,
            dyn_release_pct: 50.0,
            ext_sidechain: false,
            sc_filtering: false,
            sc_low_hz: 20.0,
            sc_high_hz: 20_000.0,
            sc_audition: false,
            spectral_enabled: false,
            spectral_density: 50.0,
            solo: false,
            invert_gain: false,
        }
    }
}

impl EqBandParams {
    pub const PARAM_NAMES_SUFFIX: [&'static str; PROQ4_PARAMS_PER_BAND] = [
        "Used",
        "Enabled",
        "Frequency",
        "Gain",
        "Q",
        "Shape",
        "Slope",
        "Stereo Placement",
        "Speakers",
        "Dynamics Enabled",
        "Dynamic Range",
        "Dynamics Auto",
        "Threshold",
        "Attack",
        "Release",
        "External Side Chain",
        "Side Chain Filtering",
        "Side Chain Low Frequency",
        "Side Chain High Frequency",
        "Side Chain Audition",
        "Spectral Enabled",
        "Spectral Density",
        "Solo",
        "Invert Gain",
    ];

    #[inline]
    pub fn effective_static_gain_db(&self, global_gain_scale: f32) -> f32 {
        let sign = if self.invert_gain { -1.0 } else { 1.0 };
        self.gain_db * sign * global_gain_scale
    }

    #[inline]
    pub fn effective_dyn_range_db(&self, global_gain_scale: f32) -> f32 {
        let sign = if self.invert_gain { -1.0 } else { 1.0 };
        self.dyn_range_db * sign * global_gain_scale
    }

    pub fn set_offset_normalized(&mut self, offset: usize, norm_val: f32) {
        let v = norm_val.clamp(0.0, 1.0);
        match offset {
            0 => self.used = v >= 0.5,
            1 => self.enabled = v >= 0.5,
            2 => self.freq_hz = norm_to_freq_hz(v),
            3 => self.gain_db = 60.0 * v - 30.0,
            4 => self.q_ui = norm_to_q_ui(v),
            5 => self.shape = FilterShape::from_norm(v),
            6 => {
                let (s, bw) = norm_to_slope_db_oct(v);
                self.slope_db_oct = s;
                self.is_brickwall = bw;
            }
            7 => self.placement = StereoPlacement::from_norm(v),
            8 => self.speaker_mask = ((v * 64.0).round() as u64).max(1),
            9 => self.dyn_enabled = v >= 0.5,
            10 => self.dyn_range_db = 60.0 * v - 30.0,
            11 => self.dyn_auto = v >= 0.5,
            12 => self.dyn_threshold_db = -90.0 + 90.0 * v,
            13 => self.dyn_attack_pct = v * 100.0,
            14 => self.dyn_release_pct = v * 100.0,
            15 => self.ext_sidechain = v >= 0.5,
            16 => self.sc_filtering = v >= 0.5,
            17 => self.sc_low_hz = norm_to_freq_hz(v),
            18 => self.sc_high_hz = norm_to_freq_hz(v),
            19 => self.sc_audition = v >= 0.5,
            20 => self.spectral_enabled = v >= 0.5,
            21 => self.spectral_density = v * 100.0,
            22 => self.solo = v >= 0.5,
            23 => self.invert_gain = v >= 0.5,
            _ => {}
        }
    }

    pub fn get_offset_normalized(&self, offset: usize) -> f32 {
        match offset {
            0 => if self.used { 1.0 } else { 0.0 },
            1 => if self.enabled { 1.0 } else { 0.0 },
            2 => freq_hz_to_norm(self.freq_hz),
            3 => ((self.gain_db + 30.0) / 60.0).clamp(0.0, 1.0),
            4 => q_ui_to_norm(self.q_ui),
            5 => self.shape.to_norm(),
            6 => slope_db_oct_to_norm(self.slope_db_oct, self.is_brickwall),
            7 => self.placement.to_norm(),
            8 => ((self.speaker_mask.min(64) as f32) / 64.0).clamp(0.0, 1.0),
            9 => if self.dyn_enabled { 1.0 } else { 0.0 },
            10 => ((self.dyn_range_db + 30.0) / 60.0).clamp(0.0, 1.0),
            11 => if self.dyn_auto { 1.0 } else { 0.0 },
            12 => ((self.dyn_threshold_db + 90.0) / 90.0).clamp(0.0, 1.0),
            13 => (self.dyn_attack_pct * 0.01).clamp(0.0, 1.0),
            14 => (self.dyn_release_pct * 0.01).clamp(0.0, 1.0),
            15 => if self.ext_sidechain { 1.0 } else { 0.0 },
            16 => if self.sc_filtering { 1.0 } else { 0.0 },
            17 => freq_hz_to_norm(self.sc_low_hz),
            18 => freq_hz_to_norm(self.sc_high_hz),
            19 => if self.sc_audition { 1.0 } else { 0.0 },
            20 => if self.spectral_enabled { 1.0 } else { 0.0 },
            21 => (self.spectral_density * 0.01).clamp(0.0, 1.0),
            22 => if self.solo { 1.0 } else { 0.0 },
            23 => if self.invert_gain { 1.0 } else { 0.0 },
            _ => 0.0,
        }
    }
}

/// 25 Global Parameters (`IDs 577..=601`, 0-based indices `576..601`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProQ4GlobalParams {
    pub processing_mode: ProcessingMode,             // 576 (`Processing Mode`)
    pub processing_resolution: LinearPhaseResolution,// 577 (`Processing Resolution`)
    pub character: CharacterMode,                    // 578 (`Character`)
    pub gain_scale: f32,                             // 579 (`Gain Scale`: 0.0..=2.0, default 1.0)
    pub output_level_db: f32,                        // 580 (`Output Level`: -36..+36 dB)
    pub output_pan: f32,                             // 581 (`Output Pan`: -1.0..+1.0)
    pub output_pan_mode: ChannelMode,                // 582 (`Output Pan Mode`)
    pub bypass: bool,                                // 583 (`Bypass`)
    pub output_invert_phase: bool,                   // 584 (`Output Invert Phase`)
    pub auto_gain: bool,                             // 585 (`Auto Gain`)
    pub analyzer_show_pre: bool,                     // 586 (`Analyzer Show Pre-Processing`)
    pub analyzer_show_post: bool,                    // 587 (`Analyzer Show Post-Processing`)
    pub analyzer_show_sc: bool,                      // 588 (`Analyzer Show Side Chain`)
    pub analyzer_range_step: u8,                     // 589 (`Analyzer Range`: 0=60, 1=90, 2=120 dB)
    pub analyzer_resolution_step: u8,                // 590 (`Analyzer Resolution`: 0..3)
    pub analyzer_speed_step: u8,                     // 591 (`Analyzer Speed`: 0..3)
    pub analyzer_tilt_db_oct: f32,                   // 592 (`Analyzer Tilt`: 0.0, 1.5, 3.0, 4.5, 6.0)
    pub analyzer_freeze: bool,                       // 593 (`Analyzer Freeze`)
    pub spectrum_grab: bool,                         // 594 (`Spectrum Grab`)
    pub display_range_step: u8,                      // 595 (`Display Range`: 0..4)
    pub receive_midi: bool,                          // 596 (`Receive Midi`)
    pub solo_gain_db: f32,                           // 597 (`Solo Gain`: -40..+20 dB)
    pub channel_mode: ChannelMode,                   // 598 (`Channel Mode`: L/R or M/S)
    pub drift_pct: f32,                              // 599 (`Drift`: 0..100%)
    pub show_collisions: bool,                       // 600 (`Show Collisions`)
    pub gain_q_interaction: bool,                    // Analog Gain-Q interaction toggle
}

impl Default for ProQ4GlobalParams {
    fn default() -> Self {
        Self {
            processing_mode: ProcessingMode::ZeroLatency,
            processing_resolution: LinearPhaseResolution::Medium,
            character: CharacterMode::Clean,
            gain_scale: 1.0,
            output_level_db: 0.0,
            output_pan: 0.0,
            output_pan_mode: ChannelMode::LeftRight,
            bypass: false,
            output_invert_phase: false,
            auto_gain: false,
            analyzer_show_pre: true,
            analyzer_show_post: true,
            analyzer_show_sc: false,
            analyzer_range_step: 1,
            analyzer_resolution_step: 1,
            analyzer_speed_step: 2,
            analyzer_tilt_db_oct: 4.5,
            analyzer_freeze: false,
            spectrum_grab: true,
            display_range_step: 2,
            receive_midi: false,
            solo_gain_db: 0.0,
            channel_mode: ChannelMode::LeftRight,
            drift_pct: 0.0,
            show_collisions: false,
            gain_q_interaction: false,
        }
    }
}

impl ProQ4GlobalParams {
    pub const GLOBAL_PARAM_NAMES: [&'static str; PROQ4_GLOBAL_PARAMS] = [
        "Processing Mode",
        "Processing Resolution",
        "Character",
        "Gain Scale",
        "Output Level",
        "Output Pan",
        "Output Pan Mode",
        "Bypass",
        "Output Invert Phase",
        "Auto Gain",
        "Analyzer Show Pre-Processing",
        "Analyzer Show Post-Processing",
        "Analyzer Show Side Chain",
        "Analyzer Range",
        "Analyzer Resolution",
        "Analyzer Speed",
        "Analyzer Tilt",
        "Analyzer Freeze",
        "Spectrum Grab",
        "Display Range",
        "Receive Midi",
        "Solo Gain",
        "Channel Mode",
        "Drift",
        "Show Collisions",
    ];

    pub fn set_global_offset_normalized(&mut self, global_offset: usize, norm_val: f32) {
        let v = norm_val.clamp(0.0, 1.0);
        match global_offset {
            0 => self.processing_mode = ProcessingMode::from_norm(v),
            1 => self.processing_resolution = LinearPhaseResolution::from_norm(v),
            2 => self.character = CharacterMode::from_norm(v),
            3 => self.gain_scale = v * 2.0,
            4 => self.output_level_db = 72.0 * v - 36.0,
            5 => self.output_pan = 2.0 * v - 1.0,
            6 => self.output_pan_mode = ChannelMode::from_norm(v),
            7 => self.bypass = v >= 0.5,
            8 => self.output_invert_phase = v >= 0.5,
            9 => self.auto_gain = v >= 0.5,
            10 => self.analyzer_show_pre = v >= 0.5,
            11 => self.analyzer_show_post = v >= 0.5,
            12 => self.analyzer_show_sc = v >= 0.5,
            13 => self.analyzer_range_step = (v * 2.0).round() as u8,
            14 => self.analyzer_resolution_step = (v * 3.0).round() as u8,
            15 => self.analyzer_speed_step = (v * 3.0).round() as u8,
            16 => self.analyzer_tilt_db_oct = ((v * 4.0).round()) * 1.5,
            17 => self.analyzer_freeze = v >= 0.5,
            18 => self.spectrum_grab = v >= 0.5,
            19 => self.display_range_step = (v * 4.0).round() as u8,
            20 => self.receive_midi = v >= 0.5,
            21 => self.solo_gain_db = -40.0 + 60.0 * v,
            22 => self.channel_mode = ChannelMode::from_norm(v),
            23 => self.drift_pct = v * 100.0,
            24 => self.show_collisions = v >= 0.5,
            _ => {}
        }
    }

    pub fn get_global_offset_normalized(&self, global_offset: usize) -> f32 {
        match global_offset {
            0 => self.processing_mode.to_norm(),
            1 => self.processing_resolution.to_norm(),
            2 => self.character.to_norm(),
            3 => (self.gain_scale * 0.5).clamp(0.0, 1.0),
            4 => ((self.output_level_db + 36.0) / 72.0).clamp(0.0, 1.0),
            5 => ((self.output_pan + 1.0) * 0.5).clamp(0.0, 1.0),
            6 => self.output_pan_mode.to_norm(),
            7 => if self.bypass { 1.0 } else { 0.0 },
            8 => if self.output_invert_phase { 1.0 } else { 0.0 },
            9 => if self.auto_gain { 1.0 } else { 0.0 },
            10 => if self.analyzer_show_pre { 1.0 } else { 0.0 },
            11 => if self.analyzer_show_post { 1.0 } else { 0.0 },
            12 => if self.analyzer_show_sc { 1.0 } else { 0.0 },
            13 => (self.analyzer_range_step as f32 * 0.5).clamp(0.0, 1.0),
            14 => (self.analyzer_resolution_step as f32 / 3.0).clamp(0.0, 1.0),
            15 => (self.analyzer_speed_step as f32 / 3.0).clamp(0.0, 1.0),
            16 => (self.analyzer_tilt_db_oct / 6.0).clamp(0.0, 1.0),
            17 => if self.analyzer_freeze { 1.0 } else { 0.0 },
            18 => if self.spectrum_grab { 1.0 } else { 0.0 },
            19 => (self.display_range_step as f32 * 0.25).clamp(0.0, 1.0),
            20 => if self.receive_midi { 1.0 } else { 0.0 },
            21 => ((self.solo_gain_db + 40.0) / 60.0).clamp(0.0, 1.0),
            22 => self.channel_mode.to_norm(),
            23 => (self.drift_pct * 0.01).clamp(0.0, 1.0),
            24 => if self.show_collisions { 1.0 } else { 0.0 },
            _ => 0.0,
        }
    }
}

/// Complete 601-parameter state container for `ProQ4Processor`.
#[derive(Clone, Debug)]
pub struct ProQ4ParameterSet {
    pub bands: [EqBandParams; PROQ4_NUM_BANDS],
    pub global: ProQ4GlobalParams,
    pub normalized_mirror: [f32; PROQ4_TOTAL_PARAMS],
}

impl Default for ProQ4ParameterSet {
    fn default() -> Self {
        Self::new()
    }
}

impl ProQ4ParameterSet {
    pub fn new() -> Self {
        let bands = [EqBandParams::default(); PROQ4_NUM_BANDS];
        let global = ProQ4GlobalParams::default();
        let mut s = Self {
            bands,
            global,
            normalized_mirror: [0.0; PROQ4_TOTAL_PARAMS],
        };
        s.sync_mirror_from_structured();
        s
    }

    pub fn sync_mirror_from_structured(&mut self) {
        for b in 0..PROQ4_NUM_BANDS {
            let base = b * PROQ4_PARAMS_PER_BAND;
            for off in 0..PROQ4_PARAMS_PER_BAND {
                self.normalized_mirror[base + off] = self.bands[b].get_offset_normalized(off);
            }
        }
        for g in 0..PROQ4_GLOBAL_PARAMS {
            self.normalized_mirror[PROQ4_BAND_PARAMS_TOTAL + g] =
                self.global.get_global_offset_normalized(g);
        }
    }

    #[inline]
    pub fn set_normalized(&mut self, param_id: u32, norm_val: f32) {
        let idx = param_id as usize;
        if idx >= PROQ4_TOTAL_PARAMS {
            return;
        }
        let v = norm_val.clamp(0.0, 1.0);
        self.normalized_mirror[idx] = v;
        if idx < PROQ4_BAND_PARAMS_TOTAL {
            let band_idx = idx / PROQ4_PARAMS_PER_BAND;
            let offset = idx % PROQ4_PARAMS_PER_BAND;
            self.bands[band_idx].set_offset_normalized(offset, v);
        } else {
            let global_offset = idx - PROQ4_BAND_PARAMS_TOTAL;
            self.global.set_global_offset_normalized(global_offset, v);
        }
    }

    #[inline]
    pub fn get_normalized(&self, param_id: u32) -> f32 {
        let idx = param_id as usize;
        if idx < PROQ4_TOTAL_PARAMS {
            self.normalized_mirror[idx]
        } else {
            0.0
        }
    }

    /// Resolves a `DigiPageTable.xml` parameter name (e.g. `"Band 1 Frequency"` or `"Processing Mode"`)
    /// to its 0-based parameter ID (`0..601`).
    pub fn find_param_id_by_name(name: &str) -> Option<u32> {
        if let Some(rest) = name.strip_prefix("Band ") {
            let (band_num_str, suffix) = rest.split_once(' ')?;
            let band_1based: usize = band_num_str.parse().ok()?;
            if !(1..=PROQ4_NUM_BANDS).contains(&band_1based) {
                return None;
            }
            let band_idx = band_1based - 1;
            for (off, &suf) in EqBandParams::PARAM_NAMES_SUFFIX.iter().enumerate() {
                if suf.eq_ignore_ascii_case(suffix)
                    || (off == 23 && suffix.eq_ignore_ascii_case("Spectral Tilt"))
                {
                    return Some((band_idx * PROQ4_PARAMS_PER_BAND + off) as u32);
                }
            }
            return None;
        }
        for (g, &gname) in ProQ4GlobalParams::GLOBAL_PARAM_NAMES.iter().enumerate() {
            if gname.eq_ignore_ascii_case(name)
                || (g == 7 && name.eq_ignore_ascii_case("Master Bypass"))
                || (g == 24 && name.eq_ignore_ascii_case("Analyzer Show Collisions"))
            {
                return Some((PROQ4_BAND_PARAMS_TOTAL + g) as u32);
            }
        }
        None
    }
}
