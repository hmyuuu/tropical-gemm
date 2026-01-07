use crate::scalar::TropicalScalar;
use crate::traits::{SimdTropical, TropicalSemiring, TropicalWithArgmax};
use std::fmt;
use std::ops::{Add, Mul};

/// TropicalMaxMul semiring: (ℝ⁺, max, ×)
///
/// - Addition (⊕) = max
/// - Multiplication (⊗) = ×
/// - Zero = 0
/// - One = 1
///
/// This is used for:
/// - Probability computations (non-log space)
/// - Fuzzy logic with product t-norm
#[derive(Copy, Clone, PartialEq)]
#[repr(transparent)]
pub struct TropicalMaxMul<T: TropicalScalar>(pub T);

impl<T: TropicalScalar> TropicalMaxMul<T> {
    /// Create a new TropicalMaxMul value.
    #[inline(always)]
    pub fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T: TropicalScalar> TropicalSemiring for TropicalMaxMul<T> {
    type Scalar = T;

    #[inline(always)]
    fn tropical_zero() -> Self {
        Self(T::scalar_zero())
    }

    #[inline(always)]
    fn tropical_one() -> Self {
        Self(T::scalar_one())
    }

    #[inline(always)]
    fn tropical_add(self, rhs: Self) -> Self {
        Self(self.0.scalar_max(rhs.0))
    }

    #[inline(always)]
    fn tropical_mul(self, rhs: Self) -> Self {
        Self(self.0.scalar_mul(rhs.0))
    }

    #[inline(always)]
    fn value(&self) -> T {
        self.0
    }

    #[inline(always)]
    fn from_scalar(s: T) -> Self {
        Self(s)
    }
}

impl<T: TropicalScalar> TropicalWithArgmax for TropicalMaxMul<T> {
    type Index = u32;

    #[inline(always)]
    fn tropical_add_argmax(self, self_idx: u32, rhs: Self, rhs_idx: u32) -> (Self, u32) {
        if self.0 >= rhs.0 {
            (self, self_idx)
        } else {
            (rhs, rhs_idx)
        }
    }
}

impl<T: TropicalScalar> SimdTropical for TropicalMaxMul<T> {
    const SIMD_AVAILABLE: bool = true;
    const SIMD_WIDTH: usize = 8;
}

impl<T: TropicalScalar> Add for TropicalMaxMul<T> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        self.tropical_add(rhs)
    }
}

impl<T: TropicalScalar> Mul for TropicalMaxMul<T> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        self.tropical_mul(rhs)
    }
}

impl<T: TropicalScalar> Default for TropicalMaxMul<T> {
    #[inline(always)]
    fn default() -> Self {
        Self::tropical_zero()
    }
}

impl<T: TropicalScalar> fmt::Debug for TropicalMaxMul<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TropicalMaxMul({})", self.0)
    }
}

impl<T: TropicalScalar> fmt::Display for TropicalMaxMul<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<T: TropicalScalar> From<T> for TropicalMaxMul<T> {
    #[inline(always)]
    fn from(value: T) -> Self {
        Self(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semiring_identity() {
        let a = TropicalMaxMul::new(5.0f64);
        let zero = TropicalMaxMul::tropical_zero();
        let one = TropicalMaxMul::tropical_one();

        // a ⊕ 0 = a
        assert_eq!(a.tropical_add(zero), a);
        // a ⊗ 1 = a
        assert_eq!(a.tropical_mul(one), a);
    }

    #[test]
    fn test_operations() {
        let a = TropicalMaxMul::new(3.0f64);
        let b = TropicalMaxMul::new(5.0f64);

        // max(3, 5) = 5
        assert_eq!(a.tropical_add(b).0, 5.0);
        // 3 * 5 = 15
        assert_eq!(a.tropical_mul(b).0, 15.0);
    }

    #[test]
    fn test_absorbing_zero() {
        let a = TropicalMaxMul::new(5.0f64);
        let zero = TropicalMaxMul::tropical_zero();

        // a ⊗ 0 = 0
        assert_eq!(a.tropical_mul(zero), zero);
    }
}
