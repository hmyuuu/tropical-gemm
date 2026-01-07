//! High-performance tropical matrix multiplication.
//!
//! This library provides BLAS-level performance for tropical matrix
//! multiplication across multiple semiring types.
//!
//! # GPU Acceleration
//!
//! Enable the `cuda` feature for GPU-accelerated operations:
//!
//! ```toml
//! [dependencies]
//! tropical-gemm = { version = "0.1", features = ["cuda"] }
//! ```
//!
//! Then use the GPU API:
//!
//! ```ignore
//! use tropical_gemm::cuda::{tropical_matmul_gpu, CudaContext};
//! use tropical_gemm::TropicalMaxPlus;
//!
//! let c = tropical_matmul_gpu::<TropicalMaxPlus<f32>>(&a, m, k, &b, n)?;
//! ```
//!
//! # Tropical Semirings
//!
//! Tropical algebra replaces standard arithmetic operations:
//! - Standard addition → tropical addition (typically max or min)
//! - Standard multiplication → tropical multiplication (typically + or ×)
//!
//! | Type | ⊕ (add) | ⊗ (mul) | Zero | One | Use Case |
//! |------|---------|---------|------|-----|----------|
//! | [`TropicalMaxPlus<T>`] | max | + | -∞ | 0 | Viterbi, longest path |
//! | [`TropicalMinPlus<T>`] | min | + | +∞ | 0 | Shortest path |
//! | [`TropicalMaxMul<T>`] | max | × | 0 | 1 | Probability (non-log) |
//! | [`TropicalAndOr`] | OR | AND | false | true | Graph reachability |
//! | [`CountingTropical<T,C>`] | max+count | +,× | (-∞,0) | (0,1) | Path counting |
//!
//! # Quick Start
//!
//! ```
//! use tropical_gemm::{tropical_matmul, TropicalMaxPlus, TropicalSemiring};
//!
//! // Create 2x3 and 3x2 matrices
//! let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
//!
//! // Compute C = A ⊗ B using TropicalMaxPlus semiring
//! let c = tropical_matmul::<TropicalMaxPlus<f32>>(&a, 2, 3, &b, 2);
//!
//! // C[i,j] = max_k(A[i,k] + B[k,j])
//! assert_eq!(c[0].value(), 8.0); // max(1+1, 2+3, 3+5) = 8
//! ```
//!
//! # Argmax Tracking (Backpropagation)
//!
//! For gradient routing in neural networks, you can track which k index
//! produced each optimal value:
//!
//! ```
//! use tropical_gemm::{tropical_matmul_with_argmax, TropicalMaxPlus, TropicalSemiring};
//!
//! let a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let b = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
//!
//! let result = tropical_matmul_with_argmax::<TropicalMaxPlus<f64>>(&a, 2, 3, &b, 2);
//!
//! // Get the optimal value and which k produced it
//! let value = result.get(0, 0).value(); // 8.0
//! let k_idx = result.get_argmax(0, 0);  // 2 (k=2 gave max)
//! ```
//!
//! # Performance
//!
//! The library uses:
//! - BLIS-style cache blocking for memory efficiency
//! - Runtime CPU feature detection for optimal SIMD kernels
//! - AVX2/AVX-512 on x86-64, NEON on ARM
//!
//! ```
//! use tropical_gemm::Backend;
//!
//! println!("Using: {}", Backend::description());
//! ```
//!
//! # BLAS-style API
//!
//! For fine-grained control:
//!
//! ```
//! use tropical_gemm::{TropicalGemm, TropicalMaxPlus, TropicalSemiring};
//!
//! let a = vec![1.0f32; 64 * 64];
//! let b = vec![1.0f32; 64 * 64];
//! let mut c = vec![TropicalMaxPlus::tropical_zero(); 64 * 64];
//!
//! TropicalGemm::<TropicalMaxPlus<f32>>::new(64, 64, 64)
//!     .execute(&a, 64, &b, 64, &mut c, 64);
//! ```

mod api;
mod backend;

pub use api::{tropical_gemm, tropical_matmul, tropical_matmul_with_argmax, TropicalGemm};
pub use backend::{version_info, Backend};

// Re-export types for convenience
pub use tropical_gemm_core::{GemmWithArgmax, Layout, Transpose};
pub use tropical_types::{
    CountingTropical, TropicalAndOr, TropicalMaxMul, TropicalMaxPlus, TropicalMinPlus,
    TropicalScalar, TropicalSemiring, TropicalWithArgmax,
};

/// Prelude module for convenient imports.
pub mod prelude {
    pub use super::{
        tropical_matmul, tropical_matmul_with_argmax, Backend, CountingTropical, GemmWithArgmax,
        Transpose, TropicalAndOr, TropicalGemm, TropicalMaxMul, TropicalMaxPlus, TropicalMinPlus,
        TropicalSemiring, TropicalWithArgmax,
    };
}

/// CUDA backend for GPU-accelerated tropical GEMM.
///
/// This module is only available when the `cuda` feature is enabled.
#[cfg(feature = "cuda")]
pub mod cuda {
    pub use tropical_gemm_cuda::{
        tropical_gemm_gpu, tropical_matmul_gpu, tropical_matmul_gpu_with_ctx, CudaContext,
        CudaError, CudaKernel, GpuMatrix, Result,
    };
}
