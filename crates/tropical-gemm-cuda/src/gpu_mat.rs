//! High-level GPU matrix types with semiring information.
//!
//! This module provides faer-style matrix types for GPU operations:
//! - [`GpuMat<S>`]: GPU matrix with embedded semiring type
//! - [`GpuMatWithArgmax<S>`]: GPU matrix with argmax tracking
//!
//! # Example
//!
//! ```ignore
//! use tropical_gemm::{Mat, MatRef, MaxPlus};
//! use tropical_gemm_cuda::{CudaContext, GpuMat};
//!
//! let ctx = CudaContext::new()?;
//!
//! // Create CPU matrices
//! let a_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let a = MatRef::<MaxPlus<f32>>::from_slice(&a_data, 2, 3);
//!
//! // Upload to GPU
//! let a_gpu = GpuMat::from_matref(&ctx, &a)?;
//!
//! // Compute on GPU
//! let b_gpu = GpuMat::from_matref(&ctx, &b)?;
//! let c_gpu = a_gpu.matmul(&ctx, &b_gpu)?;
//!
//! // Download result
//! let c = c_gpu.to_mat(&ctx)?;
//! ```

use std::marker::PhantomData;

use cudarc::driver::{DeviceRepr, ValidAsZeroBits};
use tropical_gemm::mat::{Mat, MatRef, MatWithArgmax};
use tropical_gemm::TropicalSemiring;

use crate::context::CudaContext;
use crate::error::Result;
use crate::kernels::{CudaKernel, CudaKernelWithArgmax};
use crate::memory::{ArgmaxIndex, GpuMatrix, GpuMatrixWithArgmax};

/// A GPU matrix with embedded semiring type information.
///
/// This provides a higher-level API compared to [`GpuMatrix`], embedding
/// the semiring type so operations know which algebra to use.
pub struct GpuMat<S: TropicalSemiring>
where
    S::Scalar: DeviceRepr,
{
    inner: GpuMatrix<S::Scalar>,
    _phantom: PhantomData<S>,
}

// Basic methods, construction, and conversion
impl<S> GpuMat<S>
where
    S: TropicalSemiring,
    S::Scalar: DeviceRepr + Default + Clone + ValidAsZeroBits,
{
    /// Create a GPU matrix from a CPU MatRef.
    pub fn from_matref(ctx: &CudaContext, mat: &MatRef<S>) -> Result<Self> {
        let inner = GpuMatrix::from_host_row_major(ctx, mat.as_slice(), mat.nrows(), mat.ncols())?;
        Ok(Self {
            inner,
            _phantom: PhantomData,
        })
    }

    /// Create a GPU matrix from raw scalar data.
    pub fn from_slice(
        ctx: &CudaContext,
        data: &[S::Scalar],
        nrows: usize,
        ncols: usize,
    ) -> Result<Self> {
        let inner = GpuMatrix::from_host_row_major(ctx, data, nrows, ncols)?;
        Ok(Self {
            inner,
            _phantom: PhantomData,
        })
    }

    /// Allocate a zeroed GPU matrix.
    pub fn zeros(ctx: &CudaContext, nrows: usize, ncols: usize) -> Result<Self> {
        let inner = GpuMatrix::alloc(ctx, nrows, ncols)?;
        Ok(Self {
            inner,
            _phantom: PhantomData,
        })
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.inner.rows()
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.inner.cols()
    }

    /// Get the underlying GpuMatrix.
    pub fn as_gpu_matrix(&self) -> &GpuMatrix<S::Scalar> {
        &self.inner
    }

    /// Get mutable access to the underlying GpuMatrix.
    pub fn as_gpu_matrix_mut(&mut self) -> &mut GpuMatrix<S::Scalar> {
        &mut self.inner
    }

    /// Convert to a CPU Mat.
    pub fn to_mat(&self, ctx: &CudaContext) -> Result<Mat<S>>
    where
        S::Scalar: Copy,
    {
        let data = self.inner.to_host_row_major(ctx)?;
        Ok(Mat::from_row_major(&data, self.nrows(), self.ncols()))
    }
}

// Kernel operations
impl<S> GpuMat<S>
where
    S: CudaKernel,
    S::Scalar: DeviceRepr + Default + Clone + ValidAsZeroBits,
{
    /// Perform tropical matrix multiplication on GPU.
    ///
    /// Computes C = A ⊗ B where ⊗ is the tropical matmul defined by the semiring S.
    pub fn matmul(&self, ctx: &CudaContext, b: &GpuMat<S>) -> Result<GpuMat<S>> {
        if self.ncols() != b.nrows() {
            return Err(crate::CudaError::DimensionMismatch(format!(
                "A.ncols ({}) != B.nrows ({})",
                self.ncols(),
                b.nrows()
            )));
        }

        let mut c = GpuMat::zeros(ctx, self.nrows(), b.ncols())?;
        S::launch_gemm(ctx, &self.inner, &b.inner, &mut c.inner)?;
        Ok(c)
    }
}

impl<S> GpuMat<S>
where
    S: CudaKernelWithArgmax,
    S::Scalar: DeviceRepr + Default + Clone + ValidAsZeroBits,
{
    /// Perform tropical matrix multiplication with argmax tracking.
    ///
    /// Returns both the result matrix and argmax indices for backward propagation.
    pub fn matmul_argmax(&self, ctx: &CudaContext, b: &GpuMat<S>) -> Result<GpuMatWithArgmax<S>> {
        if self.ncols() != b.nrows() {
            return Err(crate::CudaError::DimensionMismatch(format!(
                "A.ncols ({}) != B.nrows ({})",
                self.ncols(),
                b.nrows()
            )));
        }

        let mut c = GpuMatrixWithArgmax::alloc(ctx, self.nrows(), b.ncols())?;
        S::launch_gemm_with_argmax(ctx, &self.inner, &b.inner, &mut c)?;
        Ok(GpuMatWithArgmax {
            inner: c,
            _phantom: PhantomData,
        })
    }
}

/// A GPU matrix with argmax tracking for backward propagation.
pub struct GpuMatWithArgmax<S: TropicalSemiring>
where
    S::Scalar: DeviceRepr,
{
    inner: GpuMatrixWithArgmax<S::Scalar>,
    _phantom: PhantomData<S>,
}

impl<S> GpuMatWithArgmax<S>
where
    S: TropicalSemiring,
    S::Scalar: DeviceRepr + Default + Clone + ValidAsZeroBits,
{
    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.inner.rows()
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.inner.cols()
    }

    /// Convert to CPU MatWithArgmax.
    pub fn to_mat_with_argmax(&self, ctx: &CudaContext) -> Result<MatWithArgmax<S>>
    where
        S: tropical_gemm::TropicalWithArgmax<Index = u32>,
        S::Scalar: Copy,
    {
        let values_data = self.inner.matrix_to_host_row_major(ctx)?;
        let argmax_data = self.inner.argmax_to_host_row_major(ctx)?;

        let values = Mat::from_row_major(&values_data, self.nrows(), self.ncols());
        let argmax: Vec<u32> = argmax_data.into_iter().map(|x| x as u32).collect();

        Ok(MatWithArgmax { values, argmax })
    }

    /// Get just the result matrix as CPU Mat.
    pub fn to_mat(&self, ctx: &CudaContext) -> Result<Mat<S>>
    where
        S::Scalar: Copy,
    {
        let data = self.inner.matrix_to_host_row_major(ctx)?;
        Ok(Mat::from_row_major(&data, self.nrows(), self.ncols()))
    }

    /// Get just the argmax indices.
    pub fn to_argmax(&self, ctx: &CudaContext) -> Result<Vec<ArgmaxIndex>> {
        self.inner.argmax_to_host_row_major(ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tropical_gemm::{MaxPlus, MinPlus, TropicalSemiring};

    #[test]
    fn test_gpu_mat_basic() {
        let ctx = match CudaContext::new() {
            Ok(c) => c,
            Err(_) => {
                println!("CUDA not available, skipping test");
                return;
            }
        };

        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = MatRef::<MaxPlus<f32>>::from_slice(&data, 2, 3);

        let a_gpu = GpuMat::from_matref(&ctx, &a).unwrap();
        assert_eq!(a_gpu.nrows(), 2);
        assert_eq!(a_gpu.ncols(), 3);

        let a_back = a_gpu.to_mat(&ctx).unwrap();
        assert_eq!(a_back.nrows(), 2);
        assert_eq!(a_back.ncols(), 3);
        assert_eq!(a_back[(0, 0)].value(), 1.0);
        assert_eq!(a_back[(1, 2)].value(), 6.0);
    }

    #[test]
    fn test_gpu_mat_matmul() {
        let ctx = match CudaContext::new() {
            Ok(c) => c,
            Err(_) => {
                println!("CUDA not available, skipping test");
                return;
            }
        };

        let a_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

        let a = MatRef::<MaxPlus<f32>>::from_slice(&a_data, 2, 3);
        let b = MatRef::<MaxPlus<f32>>::from_slice(&b_data, 3, 2);

        let a_gpu = GpuMat::from_matref(&ctx, &a).unwrap();
        let b_gpu = GpuMat::from_matref(&ctx, &b).unwrap();

        let c_gpu = a_gpu.matmul(&ctx, &b_gpu).unwrap();
        let c = c_gpu.to_mat(&ctx).unwrap();

        // C[0,0] = max(1+1, 2+3, 3+5) = 8
        assert!((c[(0, 0)].value() - 8.0).abs() < 1e-5);
        // C[1,1] = max(4+2, 5+4, 6+6) = 12
        assert!((c[(1, 1)].value() - 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_gpu_mat_matmul_argmax() {
        let ctx = match CudaContext::new() {
            Ok(c) => c,
            Err(_) => {
                println!("CUDA not available, skipping test");
                return;
            }
        };

        let a_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

        let a = MatRef::<MaxPlus<f32>>::from_slice(&a_data, 2, 3);
        let b = MatRef::<MaxPlus<f32>>::from_slice(&b_data, 3, 2);

        let a_gpu = GpuMat::from_matref(&ctx, &a).unwrap();
        let b_gpu = GpuMat::from_matref(&ctx, &b).unwrap();

        let result_gpu = a_gpu.matmul_argmax(&ctx, &b_gpu).unwrap();
        let result = result_gpu.to_mat_with_argmax(&ctx).unwrap();

        assert!((result.get(0, 0).value() - 8.0).abs() < 1e-5);
        assert_eq!(result.get_argmax(0, 0), 2); // k=2 gave max
    }

    #[test]
    fn test_gpu_mat_minplus() {
        let ctx = match CudaContext::new() {
            Ok(c) => c,
            Err(_) => {
                println!("CUDA not available, skipping test");
                return;
            }
        };

        let a_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

        let a = MatRef::<MinPlus<f32>>::from_slice(&a_data, 2, 3);
        let b = MatRef::<MinPlus<f32>>::from_slice(&b_data, 3, 2);

        let a_gpu = GpuMat::from_matref(&ctx, &a).unwrap();
        let b_gpu = GpuMat::from_matref(&ctx, &b).unwrap();

        let c_gpu = a_gpu.matmul(&ctx, &b_gpu).unwrap();
        let c = c_gpu.to_mat(&ctx).unwrap();

        // C[0,0] = min(1+1, 2+3, 3+5) = 2
        assert!((c[(0, 0)].value() - 2.0).abs() < 1e-5);
        // C[1,1] = min(4+2, 5+4, 6+6) = 6
        assert!((c[(1, 1)].value() - 6.0).abs() < 1e-5);
    }
}
