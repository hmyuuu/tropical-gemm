use crate::scalar::TropicalScalar;
use crate::traits::{SimdTropical, TropicalSemiring, TropicalWithArgmax};
use std::fmt;
use std::ops::{Add, Mul};

/// TropicalMaxPlus semiring: (ℝ ∪ {-∞}, max, +)
///
/// - Addition (⊕) = max
/// - Multiplication (⊗) = +
/// - Zero = -∞
/// - One = 0
///
/// This is the classic tropical semiring used in:
/// - Viterbi algorithm
/// - Shortest path algorithms (with negated weights)
/// - Log-space probability computations
#[derive(Copy, Clone, PartialEq)]
#[repr(transparent)]
pub struct TropicalMaxPlus<T: TropicalScalar>(pub T);

impl<T: TropicalScalar> TropicalMaxPlus<T> {
    /// Create a new TropicalMaxPlus value.
    #[inline(always)]
    pub fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T: TropicalScalar> TropicalSemiring for TropicalMaxPlus<T> {
    type Scalar = T;

    #[inline(always)]
    fn tropical_zero() -> Self {
        Self(T::neg_infinity())
    }

    #[inline(always)]
    fn tropical_one() -> Self {
        Self(T::scalar_zero())
    }

    #[inline(always)]
    fn tropical_add(self, rhs: Self) -> Self {
        Self(self.0.scalar_max(rhs.0))
    }

    #[inline(always)]
    fn tropical_mul(self, rhs: Self) -> Self {
        Self(self.0.scalar_add(rhs.0))
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

impl<T: TropicalScalar> TropicalWithArgmax for TropicalMaxPlus<T> {
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

impl<T: TropicalScalar> SimdTropical for TropicalMaxPlus<T> {
    const SIMD_AVAILABLE: bool = true;
    const SIMD_WIDTH: usize = 8; // f32x8 for AVX2
}

impl<T: TropicalScalar> Add for TropicalMaxPlus<T> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        self.tropical_add(rhs)
    }
}

impl<T: TropicalScalar> Mul for TropicalMaxPlus<T> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        self.tropical_mul(rhs)
    }
}

impl<T: TropicalScalar> Default for TropicalMaxPlus<T> {
    #[inline(always)]
    fn default() -> Self {
        Self::tropical_zero()
    }
}

impl<T: TropicalScalar> fmt::Debug for TropicalMaxPlus<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TropicalMaxPlus({})", self.0)
    }
}

impl<T: TropicalScalar> fmt::Display for TropicalMaxPlus<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<T: TropicalScalar> From<T> for TropicalMaxPlus<T> {
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
        let a = TropicalMaxPlus::new(5.0f64);
        let zero = TropicalMaxPlus::tropical_zero();
        let one = TropicalMaxPlus::tropical_one();

        // a ⊕ 0 = a
        assert_eq!(a.tropical_add(zero), a);
        // a ⊗ 1 = a
        assert_eq!(a.tropical_mul(one), a);
    }

    #[test]
    fn test_operations() {
        let a = TropicalMaxPlus::new(3.0f64);
        let b = TropicalMaxPlus::new(5.0f64);

        // max(3, 5) = 5
        assert_eq!(a.tropical_add(b).0, 5.0);
        // 3 + 5 = 8
        assert_eq!(a.tropical_mul(b).0, 8.0);
    }

    #[test]
    fn test_argmax() {
        let a = TropicalMaxPlus::new(3.0f64);
        let b = TropicalMaxPlus::new(5.0f64);

        let (result, idx) = a.tropical_add_argmax(0, b, 1);
        assert_eq!(result.0, 5.0);
        assert_eq!(idx, 1);
    }
}
