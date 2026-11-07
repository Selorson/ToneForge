//! Zero-allocation Mixed-Radix & Bluestein Cooley-Tukey FFT engine
//! (`FftPlanner`, `Fft`, `num_complex::Complex32`).
//!
//! Supports:
//! - Exact power-of-two lengths (`2048`, `4096`, ...) via zero-scratch `Radix2Fft`
//! - Exact composite audio STFT lengths (`6144 = 3 * 2048`, `7680 = 15 * 512`, `960`, `480`, ...)
//!   via `MixedRadixPow2Fft` (`O(N log m + r * N)`)
//! - Arbitrary / prime lengths via `BluesteinFft` (`O(M log M)` chirp-z transform).

pub use num_complex;
use num_complex::Complex;
use num_traits::Float;
use std::f64::consts::PI;
use std::sync::Arc;

pub trait FftNum: Float + Send + Sync + 'static {
    fn from_f64(v: f64) -> Self;
}

impl FftNum for f32 {
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}

impl FftNum for f64 {
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FftDirection {
    Forward,
    Inverse,
}

pub trait Fft<T: FftNum>: Send + Sync {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn fft_direction(&self) -> FftDirection;
    fn get_inplace_scratch_len(&self) -> usize;
    fn process(&self, buffer: &mut [Complex<T>]);
    fn process_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]);
}

struct Radix2Fft<T: FftNum> {
    len: usize,
    direction: FftDirection,
    twiddles: Vec<Complex<T>>,
    bit_rev: Vec<usize>,
}

impl<T: FftNum> Radix2Fft<T> {
    fn new(len: usize, direction: FftDirection) -> Self {
        assert!(
            len.is_power_of_two() && len > 0,
            "Radix2Fft requires power-of-two length"
        );
        let bits = len.trailing_zeros();
        let mut bit_rev = vec![0usize; len];
        for (i, slot) in bit_rev.iter_mut().enumerate() {
            *slot = if bits == 0 {
                0
            } else {
                i.reverse_bits() >> (usize::BITS - bits)
            };
        }
        let sign = match direction {
            FftDirection::Forward => -1.0f64,
            FftDirection::Inverse => 1.0f64,
        };
        let mut twiddles = Vec::with_capacity(len);
        let mut half = 1usize;
        while half < len {
            let step = half * 2;
            let angle_step = sign * 2.0 * PI / (step as f64);
            for k in 0..half {
                let theta = angle_step * (k as f64);
                twiddles.push(Complex::new(
                    T::from_f64(theta.cos()),
                    T::from_f64(theta.sin()),
                ));
            }
            half = step;
        }
        Self {
            len,
            direction,
            twiddles,
            bit_rev,
        }
    }

    #[inline]
    fn transform_single_block(&self, block: &mut [Complex<T>]) {
        let n = self.len;
        for i in 0..n {
            let j = self.bit_rev[i];
            if i < j {
                block.swap(i, j);
            }
        }
        let mut half = 1usize;
        let mut tw_offset = 0usize;
        while half < n {
            let step = half * 2;
            let tw_slice = &self.twiddles[tw_offset..tw_offset + half];
            for base in (0..n).step_by(step) {
                for k in 0..half {
                    let u = block[base + k];
                    let v = block[base + k + half] * tw_slice[k];
                    block[base + k] = u + v;
                    block[base + k + half] = u - v;
                }
            }
            tw_offset += half;
            half = step;
        }
    }
}

impl<T: FftNum> Fft<T> for Radix2Fft<T> {
    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn fft_direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn get_inplace_scratch_len(&self) -> usize {
        0
    }

    #[inline]
    fn process(&self, buffer: &mut [Complex<T>]) {
        for chunk in buffer.chunks_exact_mut(self.len) {
            self.transform_single_block(chunk);
        }
    }

    #[inline]
    fn process_with_scratch(&self, buffer: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
        self.process(buffer);
    }
}

/// Fast Mixed-Radix FFT for composite lengths `N = r * m` where `m = 2^k` and `1 < r <= 32`
/// (e.g. `6144 = 3 * 2048`, `7680 = 15 * 512`, `960 = 15 * 64`, `480 = 15 * 32`).
struct MixedRadixPow2Fft<T: FftNum> {
    len: usize,
    r: usize,
    m: usize,
    direction: FftDirection,
    sub_fft: Radix2Fft<T>,
    /// Precomputed `W_N^{p * k}` for `k in 0..len`, `p in 1..r` (`len * (r - 1)` entries).
    twiddles: Vec<Complex<T>>,
}

impl<T: FftNum> MixedRadixPow2Fft<T> {
    fn new(len: usize, r: usize, m: usize, direction: FftDirection) -> Self {
        debug_assert_eq!(len, r * m);
        let sub_fft = Radix2Fft::new(m, direction);
        let sign = match direction {
            FftDirection::Forward => -1.0f64,
            FftDirection::Inverse => 1.0f64,
        };
        let base_angle = sign * 2.0 * PI / (len as f64);
        let mut twiddles = Vec::with_capacity(len * (r - 1));
        for k in 0..len {
            for p in 1..r {
                let theta = base_angle * ((p * k) as f64);
                twiddles.push(Complex::new(
                    T::from_f64(theta.cos()),
                    T::from_f64(theta.sin()),
                ));
            }
        }
        Self {
            len,
            r,
            m,
            direction,
            sub_fft,
            twiddles,
        }
    }

    #[inline]
    fn transform_single_block(&self, block: &mut [Complex<T>], work: &mut [Complex<T>]) {
        let r = self.r;
        let m = self.m;
        // 1. Strided decimation into `r` contiguous power-of-two blocks of size `m`
        for p in 0..r {
            let dst = &mut work[p * m..(p + 1) * m];
            for q in 0..m {
                dst[q] = block[r * q + p];
            }
            self.sub_fft.transform_single_block(dst);
        }

        // 2. Recombine across `p in 0..r`: X[k] = Y_0[k % m] + sum_{p=1..r-1} W_N^{p*k} * Y_p[k % m]
        let r_minus_1 = r - 1;
        let mut q_mod = 0usize;
        for k in 0..self.len {
            let mut acc = work[q_mod];
            let tw_base = k * r_minus_1;
            for p in 1..r {
                acc = acc + work[p * m + q_mod] * self.twiddles[tw_base + (p - 1)];
            }
            block[k] = acc;
            q_mod += 1;
            if q_mod == m {
                q_mod = 0;
            }
        }
    }
}

impl<T: FftNum> Fft<T> for MixedRadixPow2Fft<T> {
    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn fft_direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn get_inplace_scratch_len(&self) -> usize {
        self.len
    }

    fn process(&self, buffer: &mut [Complex<T>]) {
        let mut scratch = vec![Complex::new(T::zero(), T::zero()); self.len];
        self.process_with_scratch(buffer, &mut scratch);
    }

    fn process_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        if scratch.len() >= self.len {
            let work = &mut scratch[..self.len];
            for chunk in buffer.chunks_exact_mut(self.len) {
                self.transform_single_block(chunk, work);
            }
        } else {
            self.process(buffer);
        }
    }
}

/// Exact `O(M log M)` Bluestein Chirp-Z FFT for arbitrary / large-prime lengths `N`.
struct BluesteinFft<T: FftNum> {
    len: usize,
    conv_len: usize,
    direction: FftDirection,
    chirp_conj: Vec<Complex<T>>,
    kernel_fft: Vec<Complex<T>>,
    fwd_conv: Radix2Fft<T>,
    inv_conv: Radix2Fft<T>,
    inv_scale: T,
}

impl<T: FftNum> BluesteinFft<T> {
    fn new(len: usize, direction: FftDirection) -> Self {
        let conv_len = (2 * len - 1).next_power_of_two();
        let fwd_conv = Radix2Fft::new(conv_len, FftDirection::Forward);
        let inv_conv = Radix2Fft::new(conv_len, FftDirection::Inverse);

        let sign = match direction {
            FftDirection::Forward => -1.0f64,
            FftDirection::Inverse => 1.0f64,
        };

        let mut chirp_conj = Vec::with_capacity(len);
        let mut kernel = vec![Complex::new(T::zero(), T::zero()); conv_len];

        for n in 0..len {
            // Use (n^2 mod 2N) so `(n*n)` never loses float precision even for huge N
            let n_sq_mod = ((n as u64) * (n as u64)) % ((2 * len) as u64);
            let angle = sign * PI * (n_sq_mod as f64) / (len as f64);
            let c_conj = Complex::new(T::from_f64(angle.cos()), T::from_f64(angle.sin()));
            let c_pos = Complex::new(T::from_f64(angle.cos()), T::from_f64((-angle).sin()));
            chirp_conj.push(c_conj);
            kernel[n] = c_pos;
            if n > 0 {
                kernel[conv_len - n] = c_pos;
            }
        }

        fwd_conv.transform_single_block(&mut kernel);

        Self {
            len,
            conv_len,
            direction,
            chirp_conj,
            kernel_fft: kernel,
            fwd_conv,
            inv_conv,
            inv_scale: T::from_f64(1.0 / (conv_len as f64)),
        }
    }

    #[inline]
    fn transform_single_block(&self, block: &mut [Complex<T>], work: &mut [Complex<T>]) {
        for slot in work[..self.conv_len].iter_mut() {
            *slot = Complex::new(T::zero(), T::zero());
        }
        for n in 0..self.len {
            work[n] = block[n] * self.chirp_conj[n];
        }
        self.fwd_conv.transform_single_block(&mut work[..self.conv_len]);
        for i in 0..self.conv_len {
            work[i] = work[i] * self.kernel_fft[i];
        }
        self.inv_conv.transform_single_block(&mut work[..self.conv_len]);
        for k in 0..self.len {
            let scaled = Complex::new(work[k].re * self.inv_scale, work[k].im * self.inv_scale);
            block[k] = scaled * self.chirp_conj[k];
        }
    }
}

impl<T: FftNum> Fft<T> for BluesteinFft<T> {
    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn fft_direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn get_inplace_scratch_len(&self) -> usize {
        self.conv_len
    }

    fn process(&self, buffer: &mut [Complex<T>]) {
        let mut scratch = vec![Complex::new(T::zero(), T::zero()); self.conv_len];
        self.process_with_scratch(buffer, &mut scratch);
    }

    fn process_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        if scratch.len() >= self.conv_len {
            let work = &mut scratch[..self.conv_len];
            for chunk in buffer.chunks_exact_mut(self.len) {
                self.transform_single_block(chunk, work);
            }
        } else {
            self.process(buffer);
        }
    }
}

pub struct FftPlanner<T: FftNum> {
    _marker: std::marker::PhantomData<T>,
}

impl<T: FftNum> Default for FftPlanner<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FftNum> FftPlanner<T> {
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }

    pub fn plan_fft(&mut self, len: usize, direction: FftDirection) -> Arc<dyn Fft<T>> {
        assert!(len > 0, "FFT length must be > 0");
        if len.is_power_of_two() {
            return Arc::new(Radix2Fft::new(len, direction));
        }
        let m = 1usize << len.trailing_zeros();
        let r = len / m;
        if m >= 4 && r <= 32 {
            Arc::new(MixedRadixPow2Fft::new(len, r, m, direction))
        } else {
            Arc::new(BluesteinFft::new(len, direction))
        }
    }

    pub fn plan_fft_forward(&mut self, len: usize) -> Arc<dyn Fft<T>> {
        self.plan_fft(len, FftDirection::Forward)
    }

    pub fn plan_fft_inverse(&mut self, len: usize) -> Arc<dyn Fft<T>> {
        self.plan_fft(len, FftDirection::Inverse)
    }
}
