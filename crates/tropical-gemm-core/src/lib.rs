//! Core tropical GEMM algorithms.
//!
//! This crate provides portable (non-SIMD) implementations of tropical
//! matrix multiplication using BLIS-style blocking for cache efficiency.
//!
//! # Features
//!
//! - BLIS-style 5-loop blocking for cache efficiency
//! - Support for transposed inputs
//! - Argmax tracking for backpropagation
//! - Pluggable microkernel architecture
//!
//! # Example
//!
//! ```
//! use tropical_gemm_core::{tropical_gemm_portable, Transpose};
//! use tropical_types::{TropicalMaxPlus, TropicalSemiring};
//!
//! let m = 2;
//! let n = 2;
//! let k = 3;
//!
//! let a: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let b: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let mut c = vec![TropicalMaxPlus::tropical_zero(); m * n];
//!
//! unsafe {
//!     tropical_gemm_portable::<TropicalMaxPlus<f64>>(
//!         m, n, k,
//!         a.as_ptr(), 3, Transpose::NoTrans,
//!         b.as_ptr(), 2, Transpose::NoTrans,
//!         c.as_mut_ptr(), n,
//!     );
//! }
//! ```

mod argmax;
mod gemm;
mod kernel;
mod packing;
mod tiling;

pub use argmax::GemmWithArgmax;
pub use gemm::{
    tropical_gemm_inner, tropical_gemm_portable, tropical_gemm_with_argmax_inner,
    tropical_gemm_with_argmax_portable,
};
pub use kernel::{Microkernel, MicrokernelWithArgmax, PortableMicrokernel};
pub use packing::{pack_a, pack_b, packed_a_size, packed_b_size, Layout, Transpose};
pub use tiling::{BlockIterator, TilingParams};
