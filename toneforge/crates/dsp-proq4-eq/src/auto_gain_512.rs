//! `auto_gain_512`: 512-Bin 8-Knot Perceptual Loudness Auto-Gain (`0x18019BD20` & `0x18019C1B0`),
//! Cross-Instance Collision Detection (`SharedInstanceCollisionBus` at `0x180230738`), and
//! 120 FPS `ProQ4SpectrumFrame` (`16 KB`) Telemetry Exporter (`AnalyzerProcessor` at `0x1802216B0`).

use crate::collision_bus::ProQ4CollisionBus;
use daw_core_graph::ProQ4SpectrumFrame;
use num_complex::Complex32;
use realfft::{RealFftPlanner, RealToComplex};
use std::f32::consts::TAU;
use std::sync::{Arc, OnceLock};
use wide::f32x4;

pub const AUTO_GAIN_BINS: usize = 512;
pub const SPECTRUM_FRAME_BINS: usize = 1024;
pub const ANALYZER_RING_SIZE: usize = 4096;
pub const ANALYZER_FFT_SIZE: usize = 8192;
pub const ANALYZER_FFT_BINS: usize = (ANALYZER_FFT_SIZE / 2) + 1; // 4097 (5.86 Hz bin spacing @ 48 kHz)

/// 512-Bin Perceptual Loudness Auto-Gain Calculator (`0x18019BD20` & `0x18019C1B0`).
#[derive(Clone, Debug)]
pub struct PerceptualAutoGain {
    pub log_freq_octaves: [f32; AUTO_GAIN_BINS],
    pub freq_hz_table: [f64; AUTO_GAIN_BINS],
    pub normalized_weights: [f32; AUTO_GAIN_BINS],
}

impl Default for PerceptualAutoGain {
    fn default() -> Self {
        Self::new()
    }
}

impl PerceptualAutoGain {
    /// Exact 8-knot table extracted from `.rdata` `0x180212988` & `0x1802129A8`.
    pub const KNOT_FREQS_HZ: [f32; 8] = [10.0, 20.0, 30.0, 60.0, 90.0, 6250.0, 18000.0, 20000.0];
    pub const KNOT_WEIGHTS: [f32; 8] = [0.0, 0.035, 0.20, 0.85, 1.0, 1.0, 0.40, 0.0];

    pub fn new() -> Self {
        let mut log_freq_octaves = [0.0f32; AUTO_GAIN_BINS];
        let mut freq_hz_table = [0.0f64; AUTO_GAIN_BINS];
        let mut weights = [0.0f32; AUTO_GAIN_BINS];
        let min_oct = 20.0f32.log2(); // 4.321928 (`0x180212984`)
        let max_oct = 20000.0f32.log2(); // 14.287712 (`0x1802129C8`)
        let step = (max_oct - min_oct) / ((AUTO_GAIN_BINS - 1) as f32);
        let mut sum_w = 0.0f32;

        for i in 0..AUTO_GAIN_BINS {
            let oct = min_oct + (i as f32) * step;
            let hz = oct.exp2();
            log_freq_octaves[i] = oct;
            freq_hz_table[i] = hz as f64;
            let mut w = 0.0f32;
            for k in 0..7 {
                let f0 = Self::KNOT_FREQS_HZ[k];
                let f1 = Self::KNOT_FREQS_HZ[k + 1];
                if hz >= f0 && hz <= f1 {
                    let t = (hz - f0) / (f1 - f0);
                    w = Self::KNOT_WEIGHTS[k]
                        + t * (Self::KNOT_WEIGHTS[k + 1] - Self::KNOT_WEIGHTS[k]);
                    break;
                }
            }
            weights[i] = w;
            sum_w += w;
        }
        let inv_sum = if sum_w > 0.0 { 1.0 / sum_w } else { 1.0 };
        for w in &mut weights {
            *w *= inv_sum;
        }
        Self {
            log_freq_octaves,
            freq_hz_table,
            normalized_weights: weights,
        }
    }

    /// Evaluates `G_AutoGain = 1.0 / max(1e-9, \sum_{i=0}^{511} sqrt(|H_total(f_i)|^2) * W_norm[i])`
    /// using 4-wide SIMD (`sqrtps`, `mulps`, `addps` at `0x18019BD20`).
    #[inline]
    pub fn compute_compensation_linear(&self, combined_mag_sq_512: &[f32; AUTO_GAIN_BINS]) -> f32 {
        let mut acc_v = f32x4::splat(0.0);
        for i in (0..AUTO_GAIN_BINS).step_by(4) {
            let mag_sq = f32x4::from([
                combined_mag_sq_512[i],
                combined_mag_sq_512[i + 1],
                combined_mag_sq_512[i + 2],
                combined_mag_sq_512[i + 3],
            ]);
            let w = f32x4::from([
                self.normalized_weights[i],
                self.normalized_weights[i + 1],
                self.normalized_weights[i + 2],
                self.normalized_weights[i + 3],
            ]);
            acc_v += mag_sq.sqrt() * w;
        }
        let lanes: [f32; 4] = acc_v.into();
        let weighted_sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
        1.0 / weighted_sum.max(1e-9)
    }

    #[inline]
    pub fn compute_compensation_db(&self, combined_mag_sq_512: &[f32; AUTO_GAIN_BINS]) -> f32 {
        20.0 * self.compute_compensation_linear(combined_mag_sq_512).log10()
    }
}

/// Shared Cross-Instance Spectral Collision Bus (`SharedInstanceState` at `0x180230738`),
/// backed by the 100% lock-free 502-track [`ProQ4CollisionBus`].
pub struct SharedInstanceCollisionBus;

impl SharedInstanceCollisionBus {
    pub fn global() -> &'static Self {
        static BUS: OnceLock<SharedInstanceCollisionBus> = OnceLock::new();
        BUS.get_or_init(|| Self)
    }

    #[inline]
    pub fn publish_and_compute_collisions(
        &self,
        instance_id: u32,
        post_db: &[f32; SPECTRUM_FRAME_BINS],
        out_collision_mask: &mut [f32; SPECTRUM_FRAME_BINS],
    ) {
        ProQ4CollisionBus::global().publish_and_compute_collisions(
            instance_id,
            post_db,
            out_collision_mask,
        );
    }
}

/// Real-Time Zero-Allocation 1024-Bin Spectrum Analyzer (`AnalyzerProcessor` at `0x1802216B0`)
/// populating `ProQ4SpectrumFrame` (`16 KB`) in `VisualTelemetryFrame`.
pub struct ProQ4SpectrumAnalyzer {
    r2c: Arc<dyn RealToComplex<f32>>,
    r2c_scratch: Vec<Complex32>,
    window: [f32; ANALYZER_RING_SIZE],
    pre_ring: Box<[f32; ANALYZER_RING_SIZE]>,
    post_ring: Box<[f32; ANALYZER_RING_SIZE]>,
    sc_ring: Box<[f32; ANALYZER_RING_SIZE]>,
    ring_cursor: usize,
    filled_samples: usize,
    hop_counter: usize,
    fft_time_buf: Box<[f32; ANALYZER_FFT_SIZE]>,
    fft_spec_buf: Box<[Complex32; ANALYZER_FFT_BINS]>,
    log_bin_freq_hz: [f32; SPECTRUM_FRAME_BINS],
    pub latest_frame: ProQ4SpectrumFrame,
}

impl Default for ProQ4SpectrumAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl ProQ4SpectrumAnalyzer {
    pub fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(ANALYZER_FFT_SIZE);
        let r2c_scratch = vec![Complex32::new(0.0, 0.0); r2c.get_scratch_len()];

        let mut window = [0.0f32; ANALYZER_RING_SIZE];
        let inv_n = 1.0 / (ANALYZER_RING_SIZE as f32);
        for (i, w) in window.iter_mut().enumerate() {
            let h = 0.5 * (1.0 - (TAU * (i as f32) * inv_n).cos());
            *w = h * 2.0 * inv_n;
        }

        let min_oct = 20.0f32.log2();
        let max_oct = 20000.0f32.log2();
        let step = (max_oct - min_oct) / ((SPECTRUM_FRAME_BINS - 1) as f32);
        let mut log_bin_freq_hz = [0.0f32; SPECTRUM_FRAME_BINS];
        for (i, f) in log_bin_freq_hz.iter_mut().enumerate() {
            *f = (min_oct + (i as f32) * step).exp2();
        }

        let mut latest_frame = ProQ4SpectrumFrame::default();
        latest_frame.pre_db.fill(-120.0);
        latest_frame.post_db.fill(-120.0);
        latest_frame.sc_db.fill(-120.0);
        latest_frame.collision_mask.fill(0.0);

        Self {
            r2c,
            r2c_scratch,
            window,
            pre_ring: Box::new([0.0; ANALYZER_RING_SIZE]),
            post_ring: Box::new([0.0; ANALYZER_RING_SIZE]),
            sc_ring: Box::new([0.0; ANALYZER_RING_SIZE]),
            ring_cursor: 0,
            filled_samples: 0,
            hop_counter: 0,
            fft_time_buf: Box::new([0.0; ANALYZER_FFT_SIZE]),
            fft_spec_buf: Box::new([Complex32::new(0.0, 0.0); ANALYZER_FFT_BINS]),
            log_bin_freq_hz,
            latest_frame,
        }
    }

    pub fn reset(&mut self) {
        self.pre_ring.fill(0.0);
        self.post_ring.fill(0.0);
        self.sc_ring.fill(0.0);
        self.ring_cursor = 0;
        self.filled_samples = 0;
        self.hop_counter = 0;
        self.latest_frame.pre_db.fill(-120.0);
        self.latest_frame.post_db.fill(-120.0);
        self.latest_frame.sc_db.fill(-120.0);
        self.latest_frame.collision_mask.fill(0.0);
    }

    /// Pushes a stereo-downmixed sample into the analyzer and returns `true` when an FFT hop completed.
    #[inline(always)]
    pub fn push_sample(
        &mut self,
        pre_mono: f32,
        post_mono: f32,
        sc_mono: f32,
        sample_rate: f64,
        tilt_db_oct: f32,
        freeze: bool,
    ) -> bool {
        let c = self.ring_cursor;
        self.pre_ring[c] = pre_mono;
        self.post_ring[c] = post_mono;
        self.sc_ring[c] = sc_mono;
        self.ring_cursor = (c + 1) & (ANALYZER_RING_SIZE - 1);
        if self.filled_samples < ANALYZER_RING_SIZE {
            self.filled_samples += 1;
        }

        self.hop_counter += 1;
        if self.hop_counter >= 512 {
            self.hop_counter = 0;
            if !freeze {
                self.compute_spectra(sample_rate, tilt_db_oct);
                return true;
            }
        }
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_one_ring_db(
        r2c: &Arc<dyn RealToComplex<f32>>,
        r2c_scratch: &mut [Complex32],
        window: &[f32; ANALYZER_RING_SIZE],
        ring: &[f32; ANALYZER_RING_SIZE],
        ring_cursor: usize,
        filled_samples: usize,
        fft_time_buf: &mut [f32; ANALYZER_FFT_SIZE],
        fft_spec_buf: &mut [Complex32; ANALYZER_FFT_BINS],
        log_bin_freq_hz: &[f32; SPECTRUM_FRAME_BINS],
        sample_rate: f32,
        tilt_db_oct: f32,
        out_db: &mut [f32; SPECTRUM_FRAME_BINS],
    ) {
        let n_win = filled_samples.clamp(512, ANALYZER_RING_SIZE);
        fft_time_buf.fill(0.0);
        if n_win == ANALYZER_RING_SIZE {
            for i in 0..ANALYZER_RING_SIZE {
                let idx = (ring_cursor + i) & (ANALYZER_RING_SIZE - 1);
                fft_time_buf[i] = ring[idx] * window[i];
            }
        } else {
            let inv_w = 1.0 / (n_win as f32);
            let start = (ring_cursor + ANALYZER_RING_SIZE - n_win) & (ANALYZER_RING_SIZE - 1);
            for i in 0..n_win {
                let idx = (start + i) & (ANALYZER_RING_SIZE - 1);
                let h = 0.5 * (1.0 - (TAU * (i as f32) * inv_w).cos());
                fft_time_buf[i] = ring[idx] * h * 2.0 * inv_w;
            }
        }
        let _ = r2c.process_with_scratch(fft_time_buf, fft_spec_buf, r2c_scratch);

        let hz_per_bin = sample_rate / (ANALYZER_FFT_SIZE as f32);
        let smooth_alpha = 0.55f32;

        for i in 0..SPECTRUM_FRAME_BINS {
            let f = log_bin_freq_hz[i];
            let bin_f = (f / hz_per_bin).clamp(1.0, (ANALYZER_FFT_BINS - 3) as f32);
            let b_near = bin_f.round() as usize;
            // Catmull-Rom / Quadratic log-magnitude interpolation across 2x zero-padded FFT bins
            let b0 = b_near.clamp(1, ANALYZER_FFT_BINS - 2);
            let m_l = fft_spec_buf[b0 - 1].norm().max(1e-7);
            let m_c = fft_spec_buf[b0].norm().max(1e-7);
            let m_r = fft_spec_buf[b0 + 1].norm().max(1e-7);
            let delta = (bin_f - (b0 as f32)).clamp(-0.5, 0.5);
            let y_l = m_l.ln();
            let y_c = m_c.ln();
            let y_r = m_r.ln();
            let a = 0.5 * (y_l + y_r) - y_c;
            let b = 0.5 * (y_r - y_l);
            let mag = (y_c + delta * (b + delta * a)).exp().max(1e-6);

            // Apply spectral tilt around 1 kHz (`0x180107800`)
            let oct_from_1k = (f / 1000.0).log2();
            let raw_db = (20.0 * mag.log10() + oct_from_1k * tilt_db_oct).clamp(-120.0, 24.0);
            if out_db[i] <= -119.0 {
                out_db[i] = raw_db;
            } else {
                out_db[i] = smooth_alpha * out_db[i] + (1.0 - smooth_alpha) * raw_db;
            }
        }
    }

    pub fn compute_spectra(&mut self, sample_rate: f64, tilt_db_oct: f32) {
        let fs = sample_rate.max(8000.0) as f32;
        Self::compute_one_ring_db(
            &self.r2c,
            &mut self.r2c_scratch,
            &self.window,
            &self.pre_ring,
            self.ring_cursor,
            self.filled_samples,
            &mut self.fft_time_buf,
            &mut self.fft_spec_buf,
            &self.log_bin_freq_hz,
            fs,
            tilt_db_oct,
            &mut self.latest_frame.pre_db,
        );
        Self::compute_one_ring_db(
            &self.r2c,
            &mut self.r2c_scratch,
            &self.window,
            &self.post_ring,
            self.ring_cursor,
            self.filled_samples,
            &mut self.fft_time_buf,
            &mut self.fft_spec_buf,
            &self.log_bin_freq_hz,
            fs,
            tilt_db_oct,
            &mut self.latest_frame.post_db,
        );
        Self::compute_one_ring_db(
            &self.r2c,
            &mut self.r2c_scratch,
            &self.window,
            &self.sc_ring,
            self.ring_cursor,
            self.filled_samples,
            &mut self.fft_time_buf,
            &mut self.fft_spec_buf,
            &self.log_bin_freq_hz,
            fs,
            tilt_db_oct,
            &mut self.latest_frame.sc_db,
        );
        self.latest_frame.collision_mask.fill(0.0);
    }

    #[inline]
    pub fn export_to_frame(&self, dst: &mut ProQ4SpectrumFrame) {
        *dst = self.latest_frame;
    }
}
