//! `polyphase_resampler`: Base-Rate Polyphase FIR Resampler (`dsp::Resampler<16, float, 16384>`)
//! normalizing 2x (`88.2 / 96 kHz`) and 4x (`176.4 / 192 kHz`) sample rates to the 1x base rate
//! (`44.1 / 48 kHz`) for the 4096-point STFT core while preserving the ultrasonic band (`> f_base / 2`)
//! on the dry/wet merge branch.
//!
//! Exact Latency Table (`02_proq4_and_soothe3_dsp.md` Section 2.3.A / `soothe3_manual.pdf` p. 100):
//! - Regular (Minimum-Phase): `2,304` @ 1x, `4,704` @ 2x (`2*2304 + 96`), `9,408` @ 4x (`4*2304 + 192`)
//! - Low Latency (Min-Phase): `0` @ 1x, `96` @ 2x, `192` @ 4x
//! - Linear-Phase Mode:       `3,456` @ 1x, `7,008` @ 2x (`2*3456 + 96`), `14,016` @ 4x (`4*3456 + 192`)

use std::f64::consts::PI;

pub const POLYPHASE_TAPS_PER_BRANCH: usize = 16;
pub const MAX_OVERSAMPLE_FACTOR: usize = 4;
pub const MAX_FIR_TAPS: usize = POLYPHASE_TAPS_PER_BRANCH * MAX_OVERSAMPLE_FACTOR; // 64
pub const RESAMPLER_LATENCY_2X: usize = 96;
pub const RESAMPLER_LATENCY_4X: usize = 192;
pub const REGULAR_LATENCY_1X: usize = 2304;
pub const LINEAR_PHASE_LATENCY_1X: usize = 3456;

/// Half-length of the symmetric zero-phase ultrasonic splitter (`2 * 32 + 1 = 65` taps, `< 96` minimum 2x latency).
pub const ULTRASONIC_SPLIT_HALF_TAPS: usize = 32;
pub const ULTRASONIC_SPLIT_TAPS: usize = ULTRASONIC_SPLIT_HALF_TAPS * 2 + 1; // 65

/// Maximum high-rate delay buffer size (supports up to 16,384 samples @ 4x Linear-Phase = 14,016).
pub const HIGH_RATE_DELAY_RING_LEN: usize = 16384;

/// Base-rate normalizer and ultrasonic-preserving polyphase resampler (`dsp::Resampler<16, float, 16384>`).
pub struct BaseRateResampler {
    sample_rate: f32,
    base_sample_rate: f32,
    factor: usize, // 1, 2, or 4
    /// 16-tap polyphase branch coefficients for 2x (32 taps total) and 4x (64 taps total)
    coeffs_2x: [[f32; POLYPHASE_TAPS_PER_BRANCH]; 2],
    coeffs_4x: [[f32; POLYPHASE_TAPS_PER_BRANCH]; 4],
    /// Symmetric 65-tap lowpass kernels for exact zero-phase ultrasonic isolation at 2x and 4x
    ultra_lp_2x: [f32; ULTRASONIC_SPLIT_TAPS],
    ultra_lp_4x: [f32; ULTRASONIC_SPLIT_TAPS],
    /// Downsampler history per channel (`L/R` and `Sidechain L/R`)
    down_hist_main: [[f32; MAX_FIR_TAPS]; 2],
    down_hist_sc: [[f32; MAX_FIR_TAPS]; 2],
    down_phase: usize,
    /// Upsampler history of baseband `wet` and `dry` signals per channel
    up_hist_wet: [[f32; POLYPHASE_TAPS_PER_BRANCH]; 2],
    up_hist_dry: [[f32; POLYPHASE_TAPS_PER_BRANCH]; 2],
    up_phase: usize,
    /// Alignment delay ring (`256` frames) matching polyphase group delay (`32` @ 2x, `64` @ 4x)
    /// to exact resampler latency (`96` @ 2x -> `+64` delay, `192` @ 4x -> `+128` delay)
    baseband_align_wet: [[f32; 2]; 256],
    baseband_align_dry: [[f32; 2]; 256],
    baseband_align_pos: usize,
    /// High-rate full-bandwidth delay ring buffer (`16,384` frames) to preserve ultrasonic band (`> f_base/2`)
    high_rate_delay_ring: Vec<[f32; 2]>,
    high_rate_write_pos: usize,
}

impl BaseRateResampler {
    pub fn new(sample_rate: f32) -> Self {
        let coeffs_2x = Self::design_polyphase_2x();
        let coeffs_4x = Self::design_polyphase_4x();
        let ultra_lp_2x = Self::design_symmetric_lowpass(0.50);
        let ultra_lp_4x = Self::design_symmetric_lowpass(0.25);
        let mut resampler = Self {
            sample_rate: 48000.0,
            base_sample_rate: 48000.0,
            factor: 1,
            coeffs_2x,
            coeffs_4x,
            ultra_lp_2x,
            ultra_lp_4x,
            down_hist_main: [[0.0; MAX_FIR_TAPS]; 2],
            down_hist_sc: [[0.0; MAX_FIR_TAPS]; 2],
            down_phase: 0,
            up_hist_wet: [[0.0; POLYPHASE_TAPS_PER_BRANCH]; 2],
            up_hist_dry: [[0.0; POLYPHASE_TAPS_PER_BRANCH]; 2],
            up_phase: 0,
            baseband_align_wet: [[0.0; 2]; 256],
            baseband_align_dry: [[0.0; 2]; 256],
            baseband_align_pos: 0,
            high_rate_delay_ring: vec![[0.0; 2]; HIGH_RATE_DELAY_RING_LEN],
            high_rate_write_pos: 0,
        };
        resampler.set_sample_rate(sample_rate);
        resampler
    }

    fn design_symmetric_lowpass(cutoff_norm_nyquist: f64) -> [f32; ULTRASONIC_SPLIT_TAPS] {
        let mut kernel = [0.0f32; ULTRASONIC_SPLIT_TAPS];
        let center = ULTRASONIC_SPLIT_HALF_TAPS as f64;
        let mut sum = 0.0f64;
        for i in 0..ULTRASONIC_SPLIT_TAPS {
            let x = (i as f64) - center;
            let sinc = if x.abs() < 1e-9 {
                1.0
            } else {
                (PI * cutoff_norm_nyquist * x).sin() / (PI * cutoff_norm_nyquist * x)
            };
            let w_phase = 2.0 * PI * (i as f64) / ((ULTRASONIC_SPLIT_TAPS - 1) as f64);
            let win = 0.35875 - 0.48829 * w_phase.cos() + 0.14128 * (2.0 * w_phase).cos()
                - 0.01168 * (3.0 * w_phase).cos();
            let v = sinc * win;
            kernel[i] = v as f32;
            sum += v;
        }
        let inv_sum = (1.0 / sum) as f32;
        for k in &mut kernel {
            *k *= inv_sum;
        }
        kernel
    }

    /// Computes Type-I integer-centered Blackman-Harris windowed sinc 16-tap-per-branch polyphase
    /// coefficients for 2x (`32` taps, exact center at index `16`, `tap = 8, branch = 0`).
    fn design_polyphase_2x() -> [[f32; POLYPHASE_TAPS_PER_BRANCH]; 2] {
        let total_taps = POLYPHASE_TAPS_PER_BRANCH * 2;
        let center = (total_taps / 2) as f64; // 16.0 (exact integer group delay = 16 samples)
        let cutoff = 0.455;
        let mut proto = [0.0f64; 32];
        for i in 1..total_taps {
            let x = (i as f64) - center;
            let sinc = if x.abs() < 1e-9 {
                1.0
            } else {
                (PI * cutoff * x).sin() / (PI * cutoff * x)
            };
            let w_phase = 2.0 * PI * ((i - 1) as f64) / ((total_taps - 2) as f64);
            let win = 0.35875 - 0.48829 * w_phase.cos() + 0.14128 * (2.0 * w_phase).cos()
                - 0.01168 * (3.0 * w_phase).cos();
            proto[i] = sinc * win;
        }
        let mut branches = [[0.0f32; POLYPHASE_TAPS_PER_BRANCH]; 2];
        for branch in 0..2 {
            let mut branch_sum = 0.0f64;
            for tap in 0..POLYPHASE_TAPS_PER_BRANCH {
                branch_sum += proto[tap * 2 + branch];
            }
            let norm = 0.5 / branch_sum.max(1e-12);
            for tap in 0..POLYPHASE_TAPS_PER_BRANCH {
                branches[branch][tap] = (proto[tap * 2 + branch] * norm) as f32;
            }
        }
        branches
    }

    /// Computes Type-I integer-centered Blackman-Harris windowed sinc 16-tap-per-branch polyphase
    /// coefficients for 4x (`64` taps, exact center at index `32`, `tap = 8, branch = 0`).
    fn design_polyphase_4x() -> [[f32; POLYPHASE_TAPS_PER_BRANCH]; 4] {
        let total_taps = POLYPHASE_TAPS_PER_BRANCH * 4;
        let center = (total_taps / 2) as f64; // 32.0 (exact integer group delay = 32 samples)
        let cutoff = 0.2275;
        let mut proto = [0.0f64; 64];
        for i in 1..total_taps {
            let x = (i as f64) - center;
            let sinc = if x.abs() < 1e-9 {
                1.0
            } else {
                (PI * cutoff * x).sin() / (PI * cutoff * x)
            };
            let w_phase = 2.0 * PI * ((i - 1) as f64) / ((total_taps - 2) as f64);
            let win = 0.35875 - 0.48829 * w_phase.cos() + 0.14128 * (2.0 * w_phase).cos()
                - 0.01168 * (3.0 * w_phase).cos();
            proto[i] = sinc * win;
        }
        let mut branches = [[0.0f32; POLYPHASE_TAPS_PER_BRANCH]; 4];
        for branch in 0..4 {
            let mut branch_sum = 0.0f64;
            for tap in 0..POLYPHASE_TAPS_PER_BRANCH {
                branch_sum += proto[tap * 4 + branch];
            }
            let norm = 0.25 / branch_sum.max(1e-12);
            for tap in 0..POLYPHASE_TAPS_PER_BRANCH {
                branches[branch][tap] = (proto[tap * 4 + branch] * norm) as f32;
            }
        }
        branches
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        let sr = sample_rate.max(44100.0);
        let factor = if sr >= 150_000.0 {
            4
        } else if sr >= 75_000.0 {
            2
        } else {
            1
        };
        let prev_factor = self.factor;
        self.sample_rate = sr;
        self.factor = factor;
        self.base_sample_rate = sr / (factor as f32);
        if factor != prev_factor {
            self.reset();
        }
    }

    pub fn reset(&mut self) {
        self.down_hist_main = [[0.0; MAX_FIR_TAPS]; 2];
        self.down_hist_sc = [[0.0; MAX_FIR_TAPS]; 2];
        self.down_phase = self.factor.saturating_sub(1);
        self.up_hist_wet = [[0.0; POLYPHASE_TAPS_PER_BRANCH]; 2];
        self.up_hist_dry = [[0.0; POLYPHASE_TAPS_PER_BRANCH]; 2];
        self.up_phase = 0;
        self.baseband_align_wet = [[0.0; 2]; 256];
        self.baseband_align_dry = [[0.0; 2]; 256];
        self.baseband_align_pos = 0;
        for frame in &mut self.high_rate_delay_ring {
            *frame = [0.0, 0.0];
        }
        self.high_rate_write_pos = 0;
    }

    #[inline]
    pub fn oversample_factor(&self) -> usize {
        self.factor
    }

    #[inline]
    pub fn base_sample_rate(&self) -> f32 {
        self.base_sample_rate
    }

    #[inline]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Returns the exact latency in samples at the host sample rate (`soothe3_manual.pdf` p. 100).
    #[inline]
    pub fn total_latency_samples(&self, linear_phase: bool, low_latency: bool) -> u32 {
        let resampler_lat = match self.factor {
            1 => 0,
            2 => RESAMPLER_LATENCY_2X,
            _ => RESAMPLER_LATENCY_4X,
        };
        if low_latency && !linear_phase {
            return resampler_lat as u32;
        }
        let base_lat = if linear_phase {
            LINEAR_PHASE_LATENCY_1X
        } else {
            REGULAR_LATENCY_1X
        };
        (self.factor * base_lat + resampler_lat) as u32
    }

    /// Pushes one high-rate stereo frame into the full-bandwidth delay line and returns both:
    /// 1. `delayed_dry`: the full-bandwidth dry signal delayed by `latency_samples`, and
    /// 2. `ultra_only`: the zero-phase isolated ultrasonic band (`> f_base / 2`) centered at `latency_samples`.
    #[inline]
    pub fn push_and_extract_ultrasonic_at_delay(
        &mut self,
        in_l: f32,
        in_r: f32,
        latency_samples: usize,
    ) -> ([f32; 2], [f32; 2]) {
        let mask = HIGH_RATE_DELAY_RING_LEN - 1;
        let w = self.high_rate_write_pos & mask;
        self.high_rate_delay_ring[w] = [in_l, in_r];
        let center_r = (w + HIGH_RATE_DELAY_RING_LEN - (latency_samples & mask)) & mask;
        self.high_rate_write_pos = (w + 1) & mask;

        let delayed_dry = self.high_rate_delay_ring[center_r];
        if self.factor == 1 || latency_samples < ULTRASONIC_SPLIT_HALF_TAPS {
            return (delayed_dry, [0.0, 0.0]);
        }

        let kernel = if self.factor == 2 {
            &self.ultra_lp_2x
        } else {
            &self.ultra_lp_4x
        };

        let mut lp_l = 0.0f32;
        let mut lp_r = 0.0f32;
        for tap in 0..ULTRASONIC_SPLIT_TAPS {
            let offset = (tap as isize) - (ULTRASONIC_SPLIT_HALF_TAPS as isize);
            let idx = ((center_r as isize) + (HIGH_RATE_DELAY_RING_LEN as isize) + offset) as usize
                & mask;
            let frame = self.high_rate_delay_ring[idx];
            let k = kernel[tap];
            lp_l += frame[0] * k;
            lp_r += frame[1] * k;
        }

        let ultra_only = [delayed_dry[0] - lp_l, delayed_dry[1] - lp_r];
        (delayed_dry, ultra_only)
    }

    /// Feeds one high-rate stereo frame (`main` and `sc`) into the 16-tap polyphase decimator.
    /// Returns `Some(([base_main_l, base_main_r], [base_sc_l, base_sc_r]))` every `factor` samples.
    #[inline]
    pub fn push_high_rate_frame(
        &mut self,
        main_frame: [f32; 2],
        sc_frame: [f32; 2],
    ) -> Option<([f32; 2], [f32; 2])> {
        if self.factor == 1 {
            return Some((main_frame, sc_frame));
        }

        let total_taps = POLYPHASE_TAPS_PER_BRANCH * self.factor;
        for ch in 0..2 {
            self.down_hist_main[ch].copy_within(0..total_taps - 1, 1);
            self.down_hist_main[ch][0] = main_frame[ch];
            self.down_hist_sc[ch].copy_within(0..total_taps - 1, 1);
            self.down_hist_sc[ch][0] = sc_frame[ch];
        }

        self.down_phase += 1;
        if self.down_phase < self.factor {
            return None;
        }
        self.down_phase = 0;

        let mut out_main = [0.0f32; 2];
        let mut out_sc = [0.0f32; 2];

        if self.factor == 2 {
            for ch in 0..2 {
                let mut acc_m = 0.0f32;
                let mut acc_s = 0.0f32;
                for branch in 0..2 {
                    let c = &self.coeffs_2x[branch];
                    for tap in 0..POLYPHASE_TAPS_PER_BRANCH {
                        let idx = tap * 2 + branch;
                        acc_m += self.down_hist_main[ch][idx] * c[tap];
                        acc_s += self.down_hist_sc[ch][idx] * c[tap];
                    }
                }
                out_main[ch] = acc_m;
                out_sc[ch] = acc_s;
            }
        } else {
            for ch in 0..2 {
                let mut acc_m = 0.0f32;
                let mut acc_s = 0.0f32;
                for branch in 0..4 {
                    let c = &self.coeffs_4x[branch];
                    for tap in 0..POLYPHASE_TAPS_PER_BRANCH {
                        let idx = tap * 4 + branch;
                        acc_m += self.down_hist_main[ch][idx] * c[tap];
                        acc_s += self.down_hist_sc[ch][idx] * c[tap];
                    }
                }
                out_main[ch] = acc_m;
                out_sc[ch] = acc_s;
            }
        }

        Some((out_main, out_sc))
    }

    /// Pushes new base-rate `wet_base` and `dry_base` stereo frames into the 16-tap polyphase interpolator.
    #[inline]
    pub fn push_base_rate_wet_and_dry(&mut self, wet_base: [f32; 2], dry_base: [f32; 2]) {
        for ch in 0..2 {
            self.up_hist_wet[ch].copy_within(0..POLYPHASE_TAPS_PER_BRANCH - 1, 1);
            self.up_hist_wet[ch][0] = wet_base[ch];
            self.up_hist_dry[ch].copy_within(0..POLYPHASE_TAPS_PER_BRANCH - 1, 1);
            self.up_hist_dry[ch][0] = dry_base[ch];
        }
        self.up_phase = 0;
    }

    /// Evaluates the next high-rate interpolated baseband `(wet_base_up, dry_base_up)` stereo frames (`< f_base / 2`),
    /// aligned to the exact `96`-sample (2x) / `192`-sample (4x) resampler latency.
    #[inline]
    pub fn next_upsampled_wet_and_dry(&mut self) -> ([f32; 2], [f32; 2]) {
        if self.factor == 1 {
            return (
                [self.up_hist_wet[0][0], self.up_hist_wet[1][0]],
                [self.up_hist_dry[0][0], self.up_hist_dry[1][0]],
            );
        }

        let phase = self.up_phase % self.factor;
        self.up_phase += 1;

        let mut wet_out = [0.0f32; 2];
        let mut dry_out = [0.0f32; 2];
        if self.factor == 2 {
            let c = &self.coeffs_2x[phase];
            for ch in 0..2 {
                let mut sum_w = 0.0f32;
                let mut sum_d = 0.0f32;
                for tap in 0..POLYPHASE_TAPS_PER_BRANCH {
                    sum_w += self.up_hist_wet[ch][tap] * c[tap];
                    sum_d += self.up_hist_dry[ch][tap] * c[tap];
                }
                wet_out[ch] = sum_w * 2.0;
                dry_out[ch] = sum_d * 2.0;
            }
        } else {
            let c = &self.coeffs_4x[phase];
            for ch in 0..2 {
                let mut sum_w = 0.0f32;
                let mut sum_d = 0.0f32;
                for tap in 0..POLYPHASE_TAPS_PER_BRANCH {
                    sum_w += self.up_hist_wet[ch][tap] * c[tap];
                    sum_d += self.up_hist_dry[ch][tap] * c[tap];
                }
                wet_out[ch] = sum_w * 4.0;
                dry_out[ch] = sum_d * 4.0;
            }
        }

        // Exact alignment delay: 2x polyphase filter delay is 32 samples -> +64 = 96 (`RESAMPLER_LATENCY_2X`);
        // 4x polyphase filter delay is 64 samples -> +128 = 192 (`RESAMPLER_LATENCY_4X`).
        let extra_delay = if self.factor == 2 { 64 } else { 128 };
        let w = self.baseband_align_pos & 255;
        self.baseband_align_wet[w] = wet_out;
        self.baseband_align_dry[w] = dry_out;
        let r = (w + 256 - extra_delay) & 255;
        self.baseband_align_pos = (w + 1) & 255;
        (self.baseband_align_wet[r], self.baseband_align_dry[r])
    }
}
