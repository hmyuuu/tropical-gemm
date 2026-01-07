use std::fmt::{Debug, Display};

/// Trait for scalar types that can be used as underlying values in tropical numbers.
pub trait TropicalScalar:
    Copy + Clone + Send + Sync + Debug + Display + PartialOrd + 'static + Sized
{
    /// The additive identity (standard arithmetic).
    fn scalar_zero() -> Self;

    /// The multiplicative identity (standard arithmetic).
    fn scalar_one() -> Self;

    /// Standard arithmetic addition.
    fn scalar_add(self, rhs: Self) -> Self;

    /// Standard arithmetic multiplication.
    fn scalar_mul(self, rhs: Self) -> Self;

    /// Positive infinity (for MinPlus zero).
    fn pos_infinity() -> Self;

    /// Negative infinity (for MaxPlus zero).
    fn neg_infinity() -> Self;

    /// Maximum of two values.
    fn scalar_max(self, rhs: Self) -> Self;

    /// Minimum of two values.
    fn scalar_min(self, rhs: Self) -> Self;
}

macro_rules! impl_tropical_scalar_float {
    ($($t:ty),*) => {
        $(
            impl TropicalScalar for $t {
                #[inline(always)]
                fn scalar_zero() -> Self {
                    0.0
                }

                #[inline(always)]
                fn scalar_one() -> Self {
                    1.0
                }

                #[inline(always)]
                fn scalar_add(self, rhs: Self) -> Self {
                    self + rhs
                }

                #[inline(always)]
                fn scalar_mul(self, rhs: Self) -> Self {
                    self * rhs
                }

                #[inline(always)]
                fn pos_infinity() -> Self {
                    <$t>::INFINITY
                }

                #[inline(always)]
                fn neg_infinity() -> Self {
                    <$t>::NEG_INFINITY
                }

                #[inline(always)]
                fn scalar_max(self, rhs: Self) -> Self {
                    if self >= rhs { self } else { rhs }
                }

                #[inline(always)]
                fn scalar_min(self, rhs: Self) -> Self {
                    if self <= rhs { self } else { rhs }
                }
            }
        )*
    };
}

macro_rules! impl_tropical_scalar_int {
    ($($t:ty),*) => {
        $(
            impl TropicalScalar for $t {
                #[inline(always)]
                fn scalar_zero() -> Self {
                    0
                }

                #[inline(always)]
                fn scalar_one() -> Self {
                    1
                }

                #[inline(always)]
                fn scalar_add(self, rhs: Self) -> Self {
                    self + rhs
                }

                #[inline(always)]
                fn scalar_mul(self, rhs: Self) -> Self {
                    self * rhs
                }

                #[inline(always)]
                fn pos_infinity() -> Self {
                    <$t>::MAX
                }

                #[inline(always)]
                fn neg_infinity() -> Self {
                    <$t>::MIN
                }

                #[inline(always)]
                fn scalar_max(self, rhs: Self) -> Self {
                    if self >= rhs { self } else { rhs }
                }

                #[inline(always)]
                fn scalar_min(self, rhs: Self) -> Self {
                    if self <= rhs { self } else { rhs }
                }
            }
        )*
    };
}

impl_tropical_scalar_float!(f32, f64);
impl_tropical_scalar_int!(i32, i64, i8, i16, u8, u16, u32, u64);

impl TropicalScalar for bool {
    #[inline(always)]
    fn scalar_zero() -> Self {
        false
    }

    #[inline(always)]
    fn scalar_one() -> Self {
        true
    }

    #[inline(always)]
    fn scalar_add(self, rhs: Self) -> Self {
        self || rhs
    }

    #[inline(always)]
    fn scalar_mul(self, rhs: Self) -> Self {
        self && rhs
    }

    #[inline(always)]
    fn pos_infinity() -> Self {
        true
    }

    #[inline(always)]
    fn neg_infinity() -> Self {
        false
    }

    #[inline(always)]
    fn scalar_max(self, rhs: Self) -> Self {
        self || rhs
    }

    #[inline(always)]
    fn scalar_min(self, rhs: Self) -> Self {
        self && rhs
    }
}
