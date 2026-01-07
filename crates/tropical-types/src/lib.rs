//! Tropical number types and semiring traits.
//!
//! This crate provides the core type system for tropical algebra operations.
//! Tropical semirings replace standard addition with min/max and standard
//! multiplication with addition (or regular multiplication).
//!
//! # Supported Semirings
//!
//! | Type | ⊕ (add) | ⊗ (mul) | Zero | One |
//! |------|---------|---------|------|-----|
//! | [`TropicalMaxPlus<T>`] | max | + | -∞ | 0 |
//! | [`TropicalMinPlus<T>`] | min | + | +∞ | 0 |
//! | [`TropicalMaxMul<T>`] | max | × | 0 | 1 |
//! | [`TropicalAndOr`] | OR | AND | false | true |
//! | [`CountingTropical<T,C>`] | max+count | +,× | (-∞, 0) | (0, 1) |
//!
//! # Example
//!
//! ```
//! use tropical_types::{TropicalMaxPlus, TropicalSemiring};
//!
//! let a = TropicalMaxPlus::new(3.0f64);
//! let b = TropicalMaxPlus::new(5.0f64);
//!
//! // Tropical addition: max(3, 5) = 5
//! assert_eq!(a.tropical_add(b).value(), 5.0);
//!
//! // Tropical multiplication: 3 + 5 = 8
//! assert_eq!(a.tropical_mul(b).value(), 8.0);
//! ```

mod and_or;
mod counting;
mod max_mul;
mod max_plus;
mod min_plus;
mod scalar;
mod traits;

pub use and_or::TropicalAndOr;
pub use counting::CountingTropical;
pub use max_mul::TropicalMaxMul;
pub use max_plus::TropicalMaxPlus;
pub use min_plus::TropicalMinPlus;
pub use scalar::TropicalScalar;
pub use traits::{SimdTropical, TropicalSemiring, TropicalWithArgmax};

/// Prelude module for convenient imports.
pub mod prelude {
    pub use super::{
        CountingTropical, SimdTropical, TropicalAndOr, TropicalMaxMul, TropicalMaxPlus,
        TropicalMinPlus, TropicalScalar, TropicalSemiring, TropicalWithArgmax,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_types_compile() {
        // Ensure all types work with different scalar types
        let _ = TropicalMaxPlus::new(1.0f32);
        let _ = TropicalMaxPlus::new(1.0f64);
        let _ = TropicalMaxPlus::new(1i32);
        let _ = TropicalMaxPlus::new(1i64);

        let _ = TropicalMinPlus::new(1.0f32);
        let _ = TropicalMinPlus::new(1.0f64);

        let _ = TropicalMaxMul::new(1.0f32);
        let _ = TropicalMaxMul::new(1.0f64);

        let _ = TropicalAndOr::new(true);

        let _ = CountingTropical::<f32>::new(1.0, 1.0);
        let _ = CountingTropical::<f64, i64>::new(1.0, 1);
    }
}
