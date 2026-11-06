//! Portable SIMD vector types (`f32x4`, `f32x8`, `f64x2`, `f64x4`) with aligned memory layout.

use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Sub, SubAssign};

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct f32x4(pub [f32; 4]);

impl f32x4 {
    pub const ZERO: Self = Self([0.0; 4]);
    pub const ONE: Self = Self([1.0; 4]);

    #[inline(always)]
    pub const fn splat(val: f32) -> Self {
        Self([val; 4])
    }

    #[inline(always)]
    pub const fn new(arr: [f32; 4]) -> Self {
        Self(arr)
    }

    #[inline(always)]
    pub const fn to_array(self) -> [f32; 4] {
        self.0
    }

    #[inline(always)]
    pub fn max(self, rhs: Self) -> Self {
        Self([
            self.0[0].max(rhs.0[0]),
            self.0[1].max(rhs.0[1]),
            self.0[2].max(rhs.0[2]),
            self.0[3].max(rhs.0[3]),
        ])
    }

    #[inline(always)]
    pub fn min(self, rhs: Self) -> Self {
        Self([
            self.0[0].min(rhs.0[0]),
            self.0[1].min(rhs.0[1]),
            self.0[2].min(rhs.0[2]),
            self.0[3].min(rhs.0[3]),
        ])
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        Self([
            self.0[0].abs(),
            self.0[1].abs(),
            self.0[2].abs(),
            self.0[3].abs(),
        ])
    }
}

impl Add for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
            self.0[3] + rhs.0[3],
        ])
    }
}

impl AddAssign for f32x4 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
            self.0[3] - rhs.0[3],
        ])
    }
}

impl SubAssign for f32x4 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self([
            self.0[0] * rhs.0[0],
            self.0[1] * rhs.0[1],
            self.0[2] * rhs.0[2],
            self.0[3] * rhs.0[3],
        ])
    }
}

impl MulAssign for f32x4 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self([
            self.0[0] / rhs.0[0],
            self.0[1] / rhs.0[1],
            self.0[2] / rhs.0[2],
            self.0[3] / rhs.0[3],
        ])
    }
}

#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct f32x8(pub [f32; 8]);

impl f32x8 {
    pub const ZERO: Self = Self([0.0; 8]);
    pub const ONE: Self = Self([1.0; 8]);

    #[inline(always)]
    pub const fn splat(val: f32) -> Self {
        Self([val; 8])
    }

    #[inline(always)]
    pub const fn new(arr: [f32; 8]) -> Self {
        Self(arr)
    }

    #[inline(always)]
    pub const fn to_array(self) -> [f32; 8] {
        self.0
    }
}

impl Add for f32x8 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        let mut out = [0.0; 8];
        for i in 0..8 {
            out[i] = self.0[i] + rhs.0[i];
        }
        Self(out)
    }
}

impl Mul for f32x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let mut out = [0.0; 8];
        for i in 0..8 {
            out[i] = self.0[i] * rhs.0[i];
        }
        Self(out)
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct f64x2(pub [f64; 2]);

impl f64x2 {
    #[inline(always)]
    pub const fn splat(val: f64) -> Self {
        Self([val; 2])
    }
}
