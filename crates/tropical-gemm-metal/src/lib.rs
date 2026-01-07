//! Metal backend for tropical matrix multiplication.
//!
//! This crate provides GPU-accelerated tropical GEMM operations using Metal on Apple GPUs.
//!
//! # Quick Start
//!
//! ```ignore
//! use tropical_gemm_metal::{tropical_matmul_gpu, MetalContext};
//! use tropical_types::TropicalMaxPlus;
//!
//! // Simple one-shot API
//! let a = vec![1.0f32; 1024 * 1024];
//! let b = vec![1.0f32; 1024 * 1024];
//! let c = tropical_matmul_gpu::<TropicalMaxPlus<f32>>(&a, 1024, 1024, &b, 1024)?;
//! ```

mod context;
mod error;
mod kernels;
mod memory;

pub use context::MetalContext;
pub use error::{MetalError, Result};
pub use kernels::MetalKernel;
pub use memory::GpuMatrix;

/// One-shot tropical matrix multiplication on GPU.
///
/// This function handles all GPU memory management automatically.
pub fn tropical_matmul_gpu<T>(
    a: &[T::Scalar],
    m: usize,
    k: usize,
    b: &[T::Scalar],
    n: usize,
) -> Result<Vec<T::Scalar>>
where
    T: MetalKernel,
    T::Scalar: Clone + Default + Copy,
{
    if a.len() != m * k {
        return Err(MetalError::DimensionMismatch(format!(
            "A: expected {} elements, got {}",
            m * k,
            a.len()
        )));
    }
    if b.len() != k * n {
        return Err(MetalError::DimensionMismatch(format!(
            "B: expected {} elements, got {}",
            k * n,
            b.len()
        )));
    }

    let ctx = MetalContext::new()?;

    let a_gpu = GpuMatrix::from_host_row_major(&ctx, a, m, k)?;
    let b_gpu = GpuMatrix::from_host_row_major(&ctx, b, k, n)?;
    let mut c_gpu = GpuMatrix::alloc(&ctx, m, n)?;

    T::launch_gemm(&ctx, &a_gpu, &b_gpu, &mut c_gpu)?;

    Ok(c_gpu.to_host_row_major())
}

/// Tropical matrix multiplication with persistent context.
pub fn tropical_gemm_gpu<T>(
    ctx: &MetalContext,
    a: &GpuMatrix<T::Scalar>,
    b: &GpuMatrix<T::Scalar>,
    c: &mut GpuMatrix<T::Scalar>,
) -> Result<()>
where
    T: MetalKernel,
    T::Scalar: Clone + Default + Copy,
{
    if a.cols() != b.rows() {
        return Err(MetalError::DimensionMismatch(format!(
            "A.cols ({}) != B.rows ({})",
            a.cols(),
            b.rows()
        )));
    }
    if c.rows() != a.rows() || c.cols() != b.cols() {
        return Err(MetalError::DimensionMismatch(format!(
            "C dimensions ({}, {}) don't match A*B ({}, {})",
            c.rows(),
            c.cols(),
            a.rows(),
            b.cols()
        )));
    }

    T::launch_gemm(ctx, a, b, c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tropical_types::TropicalMaxPlus;

    #[test]
    fn test_tropical_matmul_gpu_small() {
        // Skip if Metal not available
        let _ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(_) => {
                println!("Metal not available, skipping test");
                return;
            }
        };

        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

        let c = tropical_matmul_gpu::<TropicalMaxPlus<f32>>(
            &a, 2, 3, &b, 2
        ).unwrap();

        // C[0,0] = max(1+1, 2+3, 3+5) = 8
        assert!((c[0] - 8.0).abs() < 1e-5);
        // C[0,1] = max(1+2, 2+4, 3+6) = 9
        assert!((c[1] - 9.0).abs() < 1e-5);
        // C[1,0] = max(4+1, 5+3, 6+5) = 11
        assert!((c[2] - 11.0).abs() < 1e-5);
        // C[1,1] = max(4+2, 5+4, 6+6) = 12
        assert!((c[3] - 12.0).abs() < 1e-5);
    }
}
