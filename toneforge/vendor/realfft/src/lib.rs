//! Real-to-Complex and Complex-to-Real zero-allocation FFT (`RealFftPlanner`).

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftNum, FftPlanner};
use std::sync::Arc;

pub trait RealToComplex<T: FftNum>: Send + Sync {
    fn len(&self) -> usize;
    fn complex_len(&self) -> usize {
        self.len() / 2 + 1
    }
    fn get_scratch_len(&self) -> usize {
        self.len()
    }
    fn process(&self, input: &mut [T], output: &mut [Complex<T>]) -> Result<(), &'static str>;
    fn process_with_scratch(
        &self,
        input: &mut [T],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), &'static str>;
    fn make_input_vec(&self) -> Vec<T>;
    fn make_output_vec(&self) -> Vec<Complex<T>>;
    fn make_scratch_vec(&self) -> Vec<Complex<T>> {
        vec![Complex::new(T::zero(), T::zero()); self.get_scratch_len()]
    }
}

pub trait ComplexToReal<T: FftNum>: Send + Sync {
    fn len(&self) -> usize;
    fn complex_len(&self) -> usize {
        self.len() / 2 + 1
    }
    fn get_scratch_len(&self) -> usize {
        self.len()
    }
    fn process(&self, input: &mut [Complex<T>], output: &mut [T]) -> Result<(), &'static str>;
    fn process_with_scratch(
        &self,
        input: &mut [Complex<T>],
        output: &mut [T],
        scratch: &mut [Complex<T>],
    ) -> Result<(), &'static str>;
    fn make_input_vec(&self) -> Vec<Complex<T>>;
    fn make_output_vec(&self) -> Vec<T>;
    fn make_scratch_vec(&self) -> Vec<Complex<T>> {
        vec![Complex::new(T::zero(), T::zero()); self.get_scratch_len()]
    }
}

struct R2CPlan<T: FftNum> {
    len: usize,
    fft: Arc<dyn Fft<T>>,
}

impl<T: FftNum> RealToComplex<T> for R2CPlan<T> {
    fn len(&self) -> usize {
        self.len
    }

    fn get_scratch_len(&self) -> usize {
        self.len + self.fft.get_inplace_scratch_len()
    }

    fn process(&self, input: &mut [T], output: &mut [Complex<T>]) -> Result<(), &'static str> {
        let mut work = vec![Complex::new(T::zero(), T::zero()); self.get_scratch_len()];
        self.process_with_scratch(input, output, &mut work)
    }

    #[inline]
    fn process_with_scratch(
        &self,
        input: &mut [T],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), &'static str> {
        if input.len() != self.len || output.len() != self.complex_len() || scratch.len() < self.len
        {
            return Err("invalid buffer length");
        }
        let (work, fft_scratch) = scratch.split_at_mut(self.len);
        for (dst, &src) in work.iter_mut().zip(input.iter()) {
            *dst = Complex::new(src, T::zero());
        }
        self.fft.process_with_scratch(work, fft_scratch);
        output.copy_from_slice(&work[..self.complex_len()]);
        Ok(())
    }

    fn make_input_vec(&self) -> Vec<T> {
        vec![T::zero(); self.len]
    }

    fn make_output_vec(&self) -> Vec<Complex<T>> {
        vec![Complex::new(T::zero(), T::zero()); self.complex_len()]
    }
}

struct C2RPlan<T: FftNum> {
    len: usize,
    ifft: Arc<dyn Fft<T>>,
}

impl<T: FftNum> ComplexToReal<T> for C2RPlan<T> {
    fn len(&self) -> usize {
        self.len
    }

    fn get_scratch_len(&self) -> usize {
        self.len + self.ifft.get_inplace_scratch_len()
    }

    fn process(&self, input: &mut [Complex<T>], output: &mut [T]) -> Result<(), &'static str> {
        let mut work = vec![Complex::new(T::zero(), T::zero()); self.get_scratch_len()];
        self.process_with_scratch(input, output, &mut work)
    }

    #[inline]
    fn process_with_scratch(
        &self,
        input: &mut [Complex<T>],
        output: &mut [T],
        scratch: &mut [Complex<T>],
    ) -> Result<(), &'static str> {
        if input.len() != self.complex_len() || output.len() != self.len || scratch.len() < self.len
        {
            return Err("invalid buffer length");
        }
        let n = self.len;
        let half = n / 2;
        let mirror_limit = (n + 1) / 2;
        let (work, fft_scratch) = scratch.split_at_mut(n);
        for k in 0..=half {
            work[k] = input[k];
            if k > 0 && k < mirror_limit {
                work[n - k] = Complex::new(input[k].re, -input[k].im);
            }
        }
        self.ifft.process_with_scratch(work, fft_scratch);
        for (dst, src) in output.iter_mut().zip(work.iter()) {
            *dst = src.re;
        }
        Ok(())
    }

    fn make_input_vec(&self) -> Vec<Complex<T>> {
        vec![Complex::new(T::zero(), T::zero()); self.complex_len()]
    }

    fn make_output_vec(&self) -> Vec<T> {
        vec![T::zero(); self.len]
    }
}

#[derive(Default)]
pub struct RealFftPlanner<T: FftNum> {
    planner: FftPlanner<T>,
}

impl<T: FftNum> RealFftPlanner<T> {
    pub fn new() -> Self {
        Self {
            planner: FftPlanner::new(),
        }
    }

    pub fn plan_fft_forward(&mut self, len: usize) -> Arc<dyn RealToComplex<T>> {
        Arc::new(R2CPlan {
            len,
            fft: self.planner.plan_fft_forward(len),
        })
    }

    pub fn plan_fft_inverse(&mut self, len: usize) -> Arc<dyn ComplexToReal<T>> {
        Arc::new(C2RPlan {
            len,
            ifft: self.planner.plan_fft_inverse(len),
        })
    }
}
