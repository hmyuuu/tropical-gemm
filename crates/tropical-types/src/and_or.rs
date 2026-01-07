use crate::traits::{SimdTropical, TropicalSemiring, TropicalWithArgmax};
use std::fmt;
use std::ops::{Add, Mul};

/// TropicalAndOr semiring: ({true, false}, OR, AND)
///
/// - Addition (⊕) = OR (logical disjunction)
/// - Multiplication (⊗) = AND (logical conjunction)
/// - Zero = false
/// - One = true
///
/// This is used for:
/// - Boolean matrix multiplication (transitive closure)
/// - Graph reachability
/// - SAT-related computations
#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct TropicalAndOr(pub bool);

impl TropicalAndOr {
    /// Create a new TropicalAndOr value.
    #[inline(always)]
    pub fn new(value: bool) -> Self {
        Self(value)
    }
}

impl TropicalSemiring for TropicalAndOr {
    type Scalar = bool;

    #[inline(always)]
    fn tropical_zero() -> Self {
        Self(false)
    }

    #[inline(always)]
    fn tropical_one() -> Self {
        Self(true)
    }

    #[inline(always)]
    fn tropical_add(self, rhs: Self) -> Self {
        Self(self.0 || rhs.0)
    }

    #[inline(always)]
    fn tropical_mul(self, rhs: Self) -> Self {
        Self(self.0 && rhs.0)
    }

    #[inline(always)]
    fn value(&self) -> bool {
        self.0
    }

    #[inline(always)]
    fn from_scalar(s: bool) -> Self {
        Self(s)
    }
}

impl TropicalWithArgmax for TropicalAndOr {
    type Index = u32;

    #[inline(always)]
    fn tropical_add_argmax(self, self_idx: u32, rhs: Self, rhs_idx: u32) -> (Self, u32) {
        // For OR, return index of first true (or last if both false)
        if self.0 {
            (self, self_idx)
        } else if rhs.0 {
            (rhs, rhs_idx)
        } else {
            (self, self_idx)
        }
    }
}

impl SimdTropical for TropicalAndOr {
    // Bool operations can be SIMD'd via bitmasks
    const SIMD_AVAILABLE: bool = true;
    const SIMD_WIDTH: usize = 256; // 256 bits = 256 bools for AVX2
}

impl Add for TropicalAndOr {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        self.tropical_add(rhs)
    }
}

impl Mul for TropicalAndOr {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        self.tropical_mul(rhs)
    }
}

impl Default for TropicalAndOr {
    #[inline(always)]
    fn default() -> Self {
        Self::tropical_zero()
    }
}

impl fmt::Debug for TropicalAndOr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TropicalAndOr({})", self.0)
    }
}

impl fmt::Display for TropicalAndOr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<bool> for TropicalAndOr {
    #[inline(always)]
    fn from(value: bool) -> Self {
        Self(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semiring_identity() {
        let a = TropicalAndOr::new(true);
        let zero = TropicalAndOr::tropical_zero();
        let one = TropicalAndOr::tropical_one();

        // a ⊕ 0 = a
        assert_eq!(a.tropical_add(zero), a);
        // a ⊗ 1 = a
        assert_eq!(a.tropical_mul(one), a);
    }

    #[test]
    fn test_operations() {
        let t = TropicalAndOr::new(true);
        let f = TropicalAndOr::new(false);

        // OR operations
        assert_eq!(t.tropical_add(f).0, true);
        assert_eq!(f.tropical_add(f).0, false);
        assert_eq!(t.tropical_add(t).0, true);

        // AND operations
        assert_eq!(t.tropical_mul(f).0, false);
        assert_eq!(t.tropical_mul(t).0, true);
        assert_eq!(f.tropical_mul(f).0, false);
    }

    #[test]
    fn test_absorbing_zero() {
        let a = TropicalAndOr::new(true);
        let zero = TropicalAndOr::tropical_zero();

        // a ⊗ 0 = 0
        assert_eq!(a.tropical_mul(zero), zero);
    }

    #[test]
    fn test_reachability_example() {
        // Graph adjacency: can we reach node j from node i?
        // A[0,1] = true (0->1), A[1,2] = true (1->2)
        // (A*A)[0,2] = A[0,0]*A[0,2] OR A[0,1]*A[1,2] = false OR true = true
        let a01 = TropicalAndOr::new(true);
        let a12 = TropicalAndOr::new(true);
        let a00 = TropicalAndOr::new(false);
        let a02 = TropicalAndOr::new(false);

        let result = a00.tropical_mul(a02).tropical_add(a01.tropical_mul(a12));
        assert_eq!(result.0, true);
    }
}
