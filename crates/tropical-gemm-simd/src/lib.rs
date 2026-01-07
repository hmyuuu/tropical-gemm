//! SIMD-optimized microkernels for tropical GEMM.
//!
//! This crate provides architecture-specific SIMD implementations
//! of the microkernel for tropical matrix multiplication.
//!
//! # Supported Architectures
//!
//! - **x86-64**: AVX2, AVX-512 (where available)
//! - **AArch64**: NEON
//! - **Other**: Portable fallback using `wide` crate
//!
//! # Runtime Dispatch
//!
//! The `dispatch` module provides automatic selection of the best
//! kernel based on CPU features detected at runtime.
//!
//! # Example
//!
//! ```
//! use tropical_gemm_simd::{tropical_gemm_dispatch, simd_level};
//! use tropical_gemm_core::Transpose;
//! use tropical_types::{TropicalMaxPlus, TropicalSemiring};
//!
//! println!("Detected SIMD level: {:?}", simd_level());
//!
//! let m = 64;
//! let n = 64;
//! let k = 64;
//!
//! let a = vec![1.0f32; m * k];
//! let b = vec![1.0f32; k * n];
//! let mut c = vec![TropicalMaxPlus::tropical_zero(); m * n];
//!
//! unsafe {
//!     tropical_gemm_dispatch::<TropicalMaxPlus<f32>>(
//!         m, n, k,
//!         a.as_ptr(), k, Transpose::NoTrans,
//!         b.as_ptr(), n, Transpose::NoTrans,
//!         c.as_mut_ptr(), n,
//!     );
//! }
//! ```

mod detect;
pub mod dispatch;
pub mod kernels;

pub use detect::{simd_level, SimdLevel};
pub use dispatch::{tropical_gemm_dispatch, KernelDispatch};
pub use kernels::*;
