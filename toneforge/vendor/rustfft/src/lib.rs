//! Zero-allocation Radix-2 Cooley-Tukey FFT engine (`FftPlanner`, `Fft`, `num_complex::Complex32`).

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
        assert!(len.is_power_of_two() && len > 0, "Radix2Fft requires power-of-two length");
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
                twiddles.push(Complex::new(T::from_f64(theta.cos()), T::from_f64(theta.sin())));
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
        Arc::new(Radix2Fft::new(len, direction))
    }

    pub fn plan_fft_forward(&mut self, len: usize) -> Arc<dyn Fft<T>> {
        self.plan_fft(len, FftDirection::Forward)
    }

    pub fn plan_fft_inverse(&mut self, len: usize) -> Arc<dyn Fft<T>> {
        self.plan_fft(len, FftDirection::Inverse)
    }
}
