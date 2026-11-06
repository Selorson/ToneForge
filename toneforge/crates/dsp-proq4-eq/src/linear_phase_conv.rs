//! `linear_phase_conv`: Linear Phase FIR Synthesis with 60.0 dB Kaiser Window (`bessel_i0` at
//! `0x180102A10`, `0x180232D00`) and Recursive Non-Uniform Partitioned Overlap-Save Convolver
//! (`64 -> 512 -> 4096 -> 32768` at `0x1801A0120`) with Zipper-Free Dual-IR Crossfading (`0x1801A15B0`).
//!
//! Strictly Zero-Allocation in `process_sample`, `process_block`, and `update_impulse_response`.

use num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

/// Partition sizes in the 4-tier non-uniform convolver (`64 -> 512 -> 4096 -> 32768`, $\times 8$ per step).
pub const NON_UNIFORM_PARTITION_SIZES: [usize; 4] = [64, 512, 4096, 32768];

/// Maximum FIR length supported in Real-Time Zero-Allocation mode (`131,072` taps covering
/// `LinearPhaseResolution::Maximum` at 1x/2x/4x sample rates).
pub const DEFAULT_MAX_LINEAR_PHASE_TAPS: usize = 131_072;

/// Kaiser window stopband attenuation (`60.0 dB` at `0x180232D00`: `0x404E000000000000`).
pub const KAISER_ATTENUATION_DB: f64 = 60.0;

/// Kaiser beta parameter for `60.0 dB` stopband attenuation: `0.1102 * (60.0 - 8.7) = 5.65326`.
pub const KAISER_BETA_60DB: f64 = 5.65326;

/// Modified Bessel function of the first kind order 0 (`0x180102A10`) used to generate the
/// 60.0 dB Kaiser window (`0x180232D00`) for Linear Phase FIR filters.
#[inline]
pub fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0f64;
    let mut term = 1.0f64;
    let half_x_sq = 0.25 * x * x;
    for k in 1..=25 {
        term *= half_x_sq / ((k * k) as f64);
        sum += term;
        if term < sum * 1e-15 {
            break;
        }
    }
    sum
}

/// Fills `out_win` with a symmetric 60.0 dB Kaiser window (`0x180232D00`).
pub fn compute_kaiser_60db_window(out_win: &mut [f32]) {
    let n = out_win.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        out_win[0] = 1.0;
        return;
    }
    let inv_i0_beta = 1.0 / bessel_i0(KAISER_BETA_60DB);
    let half = 0.5 * ((n - 1) as f64);
    let mid = n / 2;
    for i in 0..=mid {
        let r = ((i as f64) - half) / half;
        let arg = (1.0 - r * r).max(0.0).sqrt();
        let w = (bessel_i0(KAISER_BETA_60DB * arg) * inv_i0_beta) as f32;
        out_win[i] = w;
        out_win[n - 1 - i] = w;
    }
}

/// Single Partitioned Overlap-Save Tier operating at partition size `P` (FFT size `2P`).
/// Handles `num_partitions` contiguous slices of the impulse response starting at tap `start_tap = P`.
/// Because `start_tap == P`, the 1-block collection delay of size `P` exactly matches `start_tap`,
/// yielding **0 extra buffering latency**!
struct PartitionedTier {
    part_size: usize,
    fft_size: usize,
    num_bins: usize,
    max_partitions: usize,
    active_partitions: [usize; 2],
    r2c: Arc<dyn RealToComplex<f32>>,
    c2r: Arc<dyn ComplexToReal<f32>>,
    /// Dual-IR frequency-domain filter banks: `ir_spectra[bank][part_idx][bin]`
    ir_spectra: [Vec<Vec<Complex32>>; 2],
    /// Input Frequency-Domain Delay Line per channel: `fdl[ch][slot][bin]`
    fdl: [Vec<Vec<Complex32>>; 2],
    fdl_cursor: usize,
    /// Sliding `2P` input history per channel: `[prev_P, curr_P]`
    in_window: [Vec<f32>; 2],
    write_pos: usize,
    /// Pre-computed output blocks `[ch][bank][0..P]` ready for the current `P`-sample period
    out_block: [[Vec<f32>; 2]; 2],
    fft_time_scratch: Vec<f32>,
    fft_accum_spec: Vec<Complex32>,
    r2c_scratch: Vec<Complex32>,
    c2r_scratch: Vec<Complex32>,
}

impl PartitionedTier {
    fn new(planner: &mut RealFftPlanner<f32>, part_size: usize, max_partitions: usize) -> Self {
        let fft_size = part_size * 2;
        let num_bins = part_size + 1;
        let r2c = planner.plan_fft_forward(fft_size);
        let c2r = planner.plan_fft_inverse(fft_size);
        let r2c_scratch_len = r2c.get_scratch_len();
        let c2r_scratch_len = c2r.get_scratch_len();

        let make_ir_bank = || {
            (0..max_partitions)
                .map(|_| vec![Complex32::new(0.0, 0.0); num_bins])
                .collect::<Vec<_>>()
        };
        let make_fdl = || {
            (0..max_partitions.max(1))
                .map(|_| vec![Complex32::new(0.0, 0.0); num_bins])
                .collect::<Vec<_>>()
        };

        Self {
            part_size,
            fft_size,
            num_bins,
            max_partitions,
            active_partitions: [0, 0],
            r2c,
            c2r,
            ir_spectra: [make_ir_bank(), make_ir_bank()],
            fdl: [make_fdl(), make_fdl()],
            fdl_cursor: 0,
            in_window: [vec![0.0; fft_size], vec![0.0; fft_size]],
            write_pos: 0,
            out_block: [
                [vec![0.0; part_size], vec![0.0; part_size]],
                [vec![0.0; part_size], vec![0.0; part_size]],
            ],
            fft_time_scratch: vec![0.0; fft_size],
            fft_accum_spec: vec![Complex32::new(0.0, 0.0); num_bins],
            r2c_scratch: vec![Complex32::new(0.0, 0.0); r2c_scratch_len],
            c2r_scratch: vec![Complex32::new(0.0, 0.0); c2r_scratch_len],
        }
    }

    #[inline(always)]
    fn has_any_active_partitions(&self) -> bool {
        self.active_partitions[0] > 0 || self.active_partitions[1] > 0
    }

    fn reset_state(&mut self) {
        for ch in 0..2 {
            self.in_window[ch].fill(0.0);
            for slot in &mut self.fdl[ch] {
                slot.fill(Complex32::new(0.0, 0.0));
            }
            for bank in 0..2 {
                self.out_block[ch][bank].fill(0.0);
            }
        }
        self.fdl_cursor = 0;
        self.write_pos = 0;
    }

    /// Precomputes the forward FFT of each `P`-sample partition from `ir_slice` (starting at `start_tap = P`)
    /// into `bank` (`0` or `1`) and immediately populates `out_block[ch][bank]` from the latest FDL slot
    /// with Zero Heap Allocation.
    fn load_ir_into_bank(&mut self, bank: usize, ir_slice: &[f32]) {
        let p = self.part_size;
        let inv_fft = 1.0 / (self.fft_size as f32);
        let needed_parts = ir_slice.len().div_ceil(p).min(self.max_partitions);
        self.active_partitions[bank] = needed_parts;

        for m in 0..needed_parts {
            let start = m * p;
            let end = (start + p).min(ir_slice.len());
            let count = end - start;
            self.fft_time_scratch.fill(0.0);
            self.fft_time_scratch[..count].copy_from_slice(&ir_slice[start..end]);
            let _ = self.r2c.process_with_scratch(
                &mut self.fft_time_scratch,
                &mut self.ir_spectra[bank][m],
                &mut self.r2c_scratch,
            );
            for c in &mut self.ir_spectra[bank][m] {
                *c *= inv_fft;
            }
        }
        for m in needed_parts..self.max_partitions {
            self.ir_spectra[bank][m].fill(Complex32::new(0.0, 0.0));
        }

        // Immediately synthesize `out_block[ch][bank]` from the most recently committed FDL slot
        // so that Dual-IR crossfading has valid output samples for the remainder of the current P-block.
        self.recompute_current_out_block_for_bank(bank);
    }

    fn recompute_current_out_block_for_bank(&mut self, bank: usize) {
        let num_parts = self.active_partitions[bank].min(self.max_partitions);
        if num_parts == 0 || self.max_partitions == 0 {
            for ch in 0..2 {
                self.out_block[ch][bank].fill(0.0);
            }
            return;
        }
        let fdl_len = self.fdl[0].len();
        let last_slot = (self.fdl_cursor + fdl_len - 1) % fdl_len;
        let p = self.part_size;

        for ch in 0..2 {
            self.fft_accum_spec.fill(Complex32::new(0.0, 0.0));
            for m in 0..num_parts {
                let slot = (last_slot + fdl_len - m) % fdl_len;
                let x_spec = &self.fdl[ch][slot];
                let h_spec = &self.ir_spectra[bank][m];
                for k in 0..self.num_bins {
                    self.fft_accum_spec[k] += x_spec[k] * h_spec[k];
                }
            }
            self.fft_accum_spec[0].im = 0.0;
            self.fft_accum_spec[self.num_bins - 1].im = 0.0;

            let _ = self.c2r.process_with_scratch(
                &mut self.fft_accum_spec,
                &mut self.fft_time_scratch,
                &mut self.c2r_scratch,
            );
            self.out_block[ch][bank].copy_from_slice(&self.fft_time_scratch[p..2 * p]);
        }
    }

    #[inline(always)]
    fn step_stereo(
        &mut self,
        in_l: f32,
        in_r: f32,
        active_bank: usize,
        crossfading: bool,
    ) -> ([f32; 2], [f32; 2]) {
        let pos = self.write_pos;
        let p = self.part_size;

        // Read current pre-computed output sample for both banks
        let out_a = [self.out_block[0][0][pos], self.out_block[1][0][pos]];
        let out_b = [self.out_block[0][1][pos], self.out_block[1][1][pos]];

        // Record new input sample into the second half `[P .. 2P]` of `in_window`
        self.in_window[0][p + pos] = in_l;
        self.in_window[1][p + pos] = in_r;

        let next_pos = pos + 1;
        if next_pos < p {
            self.write_pos = next_pos;
        } else {
            self.write_pos = 0;
            if self.max_partitions > 0 {
                self.compute_next_block(active_bank, crossfading);
            }
            // Shift `[P .. 2P]` down to `[0 .. P]` for 50% Overlap-Save
            self.in_window[0].copy_within(p..2 * p, 0);
            self.in_window[1].copy_within(p..2 * p, 0);
        }

        (out_a, out_b)
    }

    fn compute_next_block(&mut self, active_bank: usize, crossfading: bool) {
        if self.max_partitions == 0 {
            return;
        }
        let fdl_len = self.fdl[0].len();
        let cur_slot = self.fdl_cursor;
        let p = self.part_size;

        for ch in 0..2 {
            // 1. Always push forward FFT of `[prev_P, curr_P]` into `fdl[ch][cur_slot]`
            // so future IR updates or bank switches have warm FDL history.
            self.fft_time_scratch.copy_from_slice(&self.in_window[ch]);
            let _ = self.r2c.process_with_scratch(
                &mut self.fft_time_scratch,
                &mut self.fdl[ch][cur_slot],
                &mut self.r2c_scratch,
            );

            // 2. Evaluate active_bank (and secondary bank if crossfading)
            for bank in 0..2 {
                if bank != active_bank && !crossfading {
                    continue;
                }
                let num_parts = self.active_partitions[bank].min(self.max_partitions);
                if num_parts == 0 {
                    self.out_block[ch][bank].fill(0.0);
                    continue;
                }
                self.fft_accum_spec.fill(Complex32::new(0.0, 0.0));
                for m in 0..num_parts {
                    let slot = (cur_slot + fdl_len - m) % fdl_len;
                    let x_spec = &self.fdl[ch][slot];
                    let h_spec = &self.ir_spectra[bank][m];
                    for k in 0..self.num_bins {
                        self.fft_accum_spec[k] += x_spec[k] * h_spec[k];
                    }
                }
                // Enforce real DC and Nyquist bins for `realfft` c2r
                self.fft_accum_spec[0].im = 0.0;
                self.fft_accum_spec[self.num_bins - 1].im = 0.0;

                let _ = self.c2r.process_with_scratch(
                    &mut self.fft_accum_spec,
                    &mut self.fft_time_scratch,
                    &mut self.c2r_scratch,
                );
                // Keep second half `[P .. 2P]` (Overlap-Save valid linear convolution output)
                self.out_block[ch][bank].copy_from_slice(&self.fft_time_scratch[p..2 * p]);
            }
        }

        self.fdl_cursor = (cur_slot + 1) % fdl_len;
    }
}

/// Recursive Non-Uniform Partitioned Overlap-Save Convolver (`0x1801A0120`: `64 -> 512 -> 4096 -> 32768`)
/// with Dual-IR Crossfading (`0x1801A15B0`).
pub struct NonUniformPartitionedConvolver {
    /// Direct 64-tap head FIR `[bank][0..64]` for zero additional block latency
    head_taps: [[f32; 64]; 2],
    head_ring: [[f32; 64]; 2],
    head_cursor: usize,
    /// 4 Non-Uniform Partitioned Overlap-Save Tiers (`64`, `512`, `4096`, `32768`)
    tiers: [PartitionedTier; 4],
    active_bank: usize,
    is_primed: bool,
    crossfade_remaining: usize,
    crossfade_total: usize,
    max_fir_len: usize,
    current_fir_len: usize,
    /// Pre-allocated synthesis buffers for zero-allocation Kaiser-windowed IR updates
    synth_c2r_4096: Arc<dyn ComplexToReal<f32>>,
    synth_c2r_8192: Arc<dyn ComplexToReal<f32>>,
    synth_c2r_16384: Arc<dyn ComplexToReal<f32>>,
    synth_c2r_32768: Arc<dyn ComplexToReal<f32>>,
    synth_c2r_65536: Arc<dyn ComplexToReal<f32>>,
    synth_c2r_131072: Arc<dyn ComplexToReal<f32>>,
    synth_spec_buf: Vec<Complex32>,
    synth_scratch: Vec<Complex32>,
    synth_time_buf: Vec<f32>,
    synth_window_buf: Vec<f32>,
}

impl Default for NonUniformPartitionedConvolver {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_LINEAR_PHASE_TAPS)
    }
}

impl NonUniformPartitionedConvolver {
    pub fn new(max_fir_len: usize) -> Self {
        let max_len = max_fir_len.max(4096);
        let mut planner = RealFftPlanner::<f32>::new();

        // Tier 0: P=64, covers taps 64..512 (7 partitions of 64)
        let tier_64 = PartitionedTier::new(&mut planner, 64, 7);
        // Tier 1: P=512, covers taps 512..4096 (7 partitions of 512)
        let tier_512 = PartitionedTier::new(&mut planner, 512, 7);
        // Tier 2: P=4096, covers taps 4096..32768 (7 partitions of 4096)
        let tier_4096 = PartitionedTier::new(&mut planner, 4096, 7);
        // Tier 3: P=32768, covers taps 32768..max_len
        let tier_32768_parts = if max_len > 32768 {
            (max_len - 32768).div_ceil(32768)
        } else {
            1
        };
        let tier_32768 = PartitionedTier::new(&mut planner, 32768, tier_32768_parts);

        let synth_c2r_4096 = planner.plan_fft_inverse(4096);
        let synth_c2r_8192 = planner.plan_fft_inverse(8192);
        let synth_c2r_16384 = planner.plan_fft_inverse(16384);
        let synth_c2r_32768 = planner.plan_fft_inverse(32768);
        let synth_c2r_65536 = planner.plan_fft_inverse(65536);
        let synth_c2r_131072 = planner.plan_fft_inverse(131072);
        let max_scratch = synth_c2r_131072
            .get_scratch_len()
            .max(synth_c2r_65536.get_scratch_len())
            .max(synth_c2r_32768.get_scratch_len())
            .max(synth_c2r_16384.get_scratch_len())
            .max(synth_c2r_8192.get_scratch_len())
            .max(synth_c2r_4096.get_scratch_len());

        let alloc_len = max_len.max(131072);
        let mut conv = Self {
            head_taps: [[0.0; 64]; 2],
            head_ring: [[0.0; 64]; 2],
            head_cursor: 0,
            tiers: [tier_64, tier_512, tier_4096, tier_32768],
            active_bank: 0,
            is_primed: false,
            crossfade_remaining: 0,
            crossfade_total: 512,
            max_fir_len: max_len,
            current_fir_len: 4096,
            synth_c2r_4096,
            synth_c2r_8192,
            synth_c2r_16384,
            synth_c2r_32768,
            synth_c2r_65536,
            synth_c2r_131072,
            synth_spec_buf: vec![Complex32::new(0.0, 0.0); alloc_len / 2 + 1],
            synth_scratch: vec![Complex32::new(0.0, 0.0); max_scratch],
            synth_time_buf: vec![0.0; alloc_len],
            synth_window_buf: vec![0.0; alloc_len],
        };

        // Initialize with a unity delta impulse at the center of a 4096-tap FIR (`tau = 2048`)
        conv.synth_time_buf.fill(0.0);
        conv.synth_time_buf[2048] = 1.0;
        conv.load_ir_into_bank(0, 4096);
        conv.is_primed = true;
        conv
    }

    #[inline]
    pub const fn partition_sizes() -> [usize; 4] {
        NON_UNIFORM_PARTITION_SIZES
    }

    #[inline]
    pub fn max_fir_len(&self) -> usize {
        self.max_fir_len
    }

    #[inline]
    pub fn current_fir_len(&self) -> usize {
        self.current_fir_len
    }

    #[inline]
    pub fn current_latency_samples(&self) -> usize {
        self.current_fir_len / 2
    }

    #[inline]
    pub fn is_crossfading(&self) -> bool {
        self.crossfade_remaining > 0
    }

    pub fn reset_state(&mut self) {
        self.head_ring = [[0.0; 64]; 2];
        self.head_cursor = 0;
        for t in &mut self.tiers {
            t.reset_state();
        }
        self.crossfade_remaining = 0;
    }

    fn load_ir_into_bank(&mut self, bank: usize, fir_len: usize) {
        let len = fir_len.min(self.max_fir_len).max(64);
        self.current_fir_len = len;
        let ir = &self.synth_time_buf[..len];

        // 1. Direct 64-tap head `ir[0..64]`
        self.head_taps[bank].fill(0.0);
        let head_len = len.min(64);
        self.head_taps[bank][..head_len].copy_from_slice(&ir[..head_len]);

        // 2. Tier 0 (`P_0 = 64`): `ir[64 .. 512]`
        if len > 64 {
            let end = len.min(512);
            self.tiers[0].load_ir_into_bank(bank, &ir[64..end]);
        } else {
            self.tiers[0].load_ir_into_bank(bank, &[]);
        }

        // 3. Tier 1 (`P_1 = 512`): `ir[512 .. 4096]`
        if len > 512 {
            let end = len.min(4096);
            self.tiers[1].load_ir_into_bank(bank, &ir[512..end]);
        } else {
            self.tiers[1].load_ir_into_bank(bank, &[]);
        }

        // 4. Tier 2 (`P_2 = 4096`): `ir[4096 .. 32768]`
        if len > 4096 {
            let end = len.min(32768);
            self.tiers[2].load_ir_into_bank(bank, &ir[4096..end]);
        } else {
            self.tiers[2].load_ir_into_bank(bank, &[]);
        }

        // 5. Tier 3 (`P_3 = 32768`): `ir[32768 .. len]`
        if len > 32768 {
            self.tiers[3].load_ir_into_bank(bank, &ir[32768..len]);
        } else {
            self.tiers[3].load_ir_into_bank(bank, &[]);
        }
    }

    /// Updates the impulse response from an arbitrary slice `ir`.
    /// When `smooth_crossfade` is true, loads into the inactive bank and initiates a
    /// sample-accurate Dual-IR Crossfade (`0x1801A15B0`) with Zero Heap Allocation.
    pub fn set_impulse_response(&mut self, ir: &[f32], smooth_crossfade: bool) {
        let len = ir.len().clamp(64, self.max_fir_len);
        self.synth_time_buf[..len].fill(0.0);
        let copy_len = ir.len().min(len);
        self.synth_time_buf[..copy_len].copy_from_slice(&ir[..copy_len]);

        if !self.is_primed || !smooth_crossfade {
            self.load_ir_into_bank(self.active_bank, len);
            self.crossfade_remaining = 0;
            self.is_primed = true;
        } else {
            if self.crossfade_remaining > 0 {
                // Complete current crossfade cleanly before loading the new target IR
                self.active_bank = 1 - self.active_bank;
            }
            let target_bank = 1 - self.active_bank;
            self.load_ir_into_bank(target_bank, len);
            self.crossfade_remaining = self.crossfade_total.max(64);
        }
    }

    /// Synthesizes a symmetric Linear-Phase FIR of length `fir_len` (`4096 .. 131072`)
    /// windowed by a 60.0 dB Kaiser window from a magnitude callback `mag_fn(freq_hz) -> f32`
    /// with Zero Heap Allocation.
    pub fn rebuild_linear_phase_eq_ir<F>(
        &mut self,
        fir_len: usize,
        sample_rate: f64,
        smooth_crossfade: bool,
        mut mag_fn: F,
    ) where
        F: FnMut(f64) -> f32,
    {
        let clamped_req = fir_len.min(self.max_fir_len);
        let n = match clamped_req {
            0..=6144 => 4096,
            6145..=12288 => 8192,
            12289..=24576 => 16384,
            24577..=49152 => 32768,
            49153..=98304 => 65536,
            _ => 131072,
        };
        let half = n / 2;
        let num_bins = half + 1;
        let inv_n = 1.0 / (n as f32);
        let df = sample_rate / (n as f64);

        if n <= 16384 {
            for k in 0..num_bins {
                let freq_hz = (k as f64) * df;
                let mag = mag_fn(freq_hz) * inv_n;
                // Apply linear-phase shift `e^{-j * pi * k}` = `(-1)^k` so impulse center is at `tau = N / 2`
                let sign = if (k & 1) == 0 { 1.0 } else { -1.0 };
                self.synth_spec_buf[k] = Complex32::new(mag * sign, 0.0);
            }
        } else {
            // Exact per-bin evaluation for the lowest 2048 bins (sub-bass / low-mid precision),
            // and piecewise linear interpolation over small stride steps in the upper bins.
            let exact_limit = 2048.min(num_bins - 1);
            for k in 0..=exact_limit {
                let freq_hz = (k as f64) * df;
                let mag = mag_fn(freq_hz) * inv_n;
                let sign = if (k & 1) == 0 { 1.0 } else { -1.0 };
                self.synth_spec_buf[k] = Complex32::new(mag * sign, 0.0);
            }
            let step = (n / 8192).max(2);
            let mut k0 = exact_limit;
            let mut m0 = self.synth_spec_buf[k0].re.abs();
            while k0 < half {
                let k1 = (k0 + step).min(half);
                let m1 = mag_fn((k1 as f64) * df) * inv_n;
                let span = (k1 - k0) as f32;
                for k in (k0 + 1)..=k1 {
                    let t = ((k - k0) as f32) / span;
                    let mag = m0 + t * (m1 - m0);
                    let sign = if (k & 1) == 0 { 1.0 } else { -1.0 };
                    self.synth_spec_buf[k] = Complex32::new(mag * sign, 0.0);
                }
                k0 = k1;
                m0 = m1;
            }
        }

        let spec = &mut self.synth_spec_buf[..num_bins];
        let time_out = &mut self.synth_time_buf[..n];
        match n {
            4096 => {
                let _ = self
                    .synth_c2r_4096
                    .process_with_scratch(spec, time_out, &mut self.synth_scratch);
            }
            8192 => {
                let _ = self
                    .synth_c2r_8192
                    .process_with_scratch(spec, time_out, &mut self.synth_scratch);
            }
            16384 => {
                let _ = self
                    .synth_c2r_16384
                    .process_with_scratch(spec, time_out, &mut self.synth_scratch);
            }
            32768 => {
                let _ = self
                    .synth_c2r_32768
                    .process_with_scratch(spec, time_out, &mut self.synth_scratch);
            }
            65536 => {
                let _ = self
                    .synth_c2r_65536
                    .process_with_scratch(spec, time_out, &mut self.synth_scratch);
            }
            _ => {
                let _ = self
                    .synth_c2r_131072
                    .process_with_scratch(spec, time_out, &mut self.synth_scratch);
            }
        }

        // Apply 60.0 dB Kaiser window (`0x180232D00`)
        compute_kaiser_60db_window(&mut self.synth_window_buf[..n]);
        for i in 0..n {
            self.synth_time_buf[i] *= self.synth_window_buf[i];
        }

        if !self.is_primed || !smooth_crossfade {
            self.load_ir_into_bank(self.active_bank, n);
            self.crossfade_remaining = 0;
            self.is_primed = true;
        } else {
            if self.crossfade_remaining > 0 {
                self.active_bank = 1 - self.active_bank;
            }
            let target_bank = 1 - self.active_bank;
            self.load_ir_into_bank(target_bank, n);
            self.crossfade_remaining = self.crossfade_total.max(64);
        }
    }

    #[inline]
    pub fn process_stereo_sample(&mut self, in_l: f32, in_r: f32) -> [f32; 2] {
        let crossfading = self.crossfade_remaining > 0;
        let act = self.active_bank;

        // 1. Direct 64-tap head
        let c = self.head_cursor;
        self.head_ring[0][c] = in_l;
        self.head_ring[1][c] = in_r;

        let mut y_a = [0.0f32; 2];
        let mut y_b = [0.0f32; 2];

        for k in 0..64 {
            let idx = (c + 64 - k) & 63;
            let xl = self.head_ring[0][idx];
            let xr = self.head_ring[1][idx];
            let ha = self.head_taps[0][k];
            y_a[0] += xl * ha;
            y_a[1] += xr * ha;
            if crossfading || act == 1 {
                let hb = self.head_taps[1][k];
                y_b[0] += xl * hb;
                y_b[1] += xr * hb;
            }
        }
        self.head_cursor = (c + 1) & 63;

        // 2. Accumulate all 4 Non-Uniform Partitioned Overlap-Save Tiers (`64, 512, 4096, 32768`)
        for tier in &mut self.tiers {
            if tier.has_any_active_partitions() {
                let (ta, tb) = tier.step_stereo(in_l, in_r, act, crossfading);
                y_a[0] += ta[0];
                y_a[1] += ta[1];
                y_b[0] += tb[0];
                y_b[1] += tb[1];
            } else {
                // Still advance input history so future IR updates have valid FDL state
                let _ = tier.step_stereo(in_l, in_r, act, false);
            }
        }

        if !crossfading {
            return if act == 0 { y_a } else { y_b };
        }

        // 3. Zipper-free Dual-IR Crossfade (`0x1801A15B0`)
        let total = self.crossfade_total.max(1) as f32;
        let done = (self.crossfade_total - self.crossfade_remaining) as f32;
        let alpha = (done / total).clamp(0.0, 1.0);
        let (old_y, new_y) = if act == 0 { (y_a, y_b) } else { (y_b, y_a) };
        let out = [
            (1.0 - alpha) * old_y[0] + alpha * new_y[0],
            (1.0 - alpha) * old_y[1] + alpha * new_y[1],
        ];

        self.crossfade_remaining -= 1;
        if self.crossfade_remaining == 0 {
            self.active_bank = 1 - self.active_bank;
        }
        out
    }
}
