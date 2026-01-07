//! CUDA kernel trait and implementations.

use crate::context::CudaContext;
use crate::error::Result;
use crate::memory::GpuMatrix;
use cudarc::driver::{DeviceRepr, LaunchAsync, LaunchConfig, ValidAsZeroBits};
use tropical_types::{TropicalMaxMul, TropicalMaxPlus, TropicalMinPlus, TropicalSemiring};

/// Trait for types that can be computed on GPU.
pub trait CudaKernel: TropicalSemiring
where
    Self::Scalar: DeviceRepr + Default + Clone + ValidAsZeroBits,
{
    /// Kernel function name.
    const KERNEL_NAME: &'static str;

    /// Execute the tropical GEMM kernel.
    ///
    /// Computes C = A ⊗ B where ⊗ is tropical matrix multiplication.
    fn launch_gemm(
        ctx: &CudaContext,
        a: &GpuMatrix<Self::Scalar>,
        b: &GpuMatrix<Self::Scalar>,
        c: &mut GpuMatrix<Self::Scalar>,
    ) -> Result<()>;
}

impl CudaKernel for TropicalMaxPlus<f32> {
    const KERNEL_NAME: &'static str = "tropical_maxplus_f32_nn";

    fn launch_gemm(
        ctx: &CudaContext,
        a: &GpuMatrix<f32>,
        b: &GpuMatrix<f32>,
        c: &mut GpuMatrix<f32>,
    ) -> Result<()> {
        let m = a.rows();
        let k = a.cols();
        let n = b.cols();

        let kernel = ctx.get_kernel(Self::KERNEL_NAME)?;
        let grid = CudaContext::grid_dims_f32(m, n);
        let block = CudaContext::block_dims_f32();

        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: block,
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel.launch(
                cfg,
                (
                    a.as_slice(),
                    b.as_slice(),
                    c.as_slice_mut(),
                    m as i32,
                    n as i32,
                    k as i32,
                ),
            )?;
        }

        ctx.device().synchronize()?;
        Ok(())
    }
}

impl CudaKernel for TropicalMinPlus<f32> {
    const KERNEL_NAME: &'static str = "tropical_minplus_f32_nn";

    fn launch_gemm(
        ctx: &CudaContext,
        a: &GpuMatrix<f32>,
        b: &GpuMatrix<f32>,
        c: &mut GpuMatrix<f32>,
    ) -> Result<()> {
        let m = a.rows();
        let k = a.cols();
        let n = b.cols();

        let kernel = ctx.get_kernel(Self::KERNEL_NAME)?;
        let grid = CudaContext::grid_dims_f32(m, n);
        let block = CudaContext::block_dims_f32();

        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: block,
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel.launch(
                cfg,
                (
                    a.as_slice(),
                    b.as_slice(),
                    c.as_slice_mut(),
                    m as i32,
                    n as i32,
                    k as i32,
                ),
            )?;
        }

        ctx.device().synchronize()?;
        Ok(())
    }
}

impl CudaKernel for TropicalMaxMul<f32> {
    const KERNEL_NAME: &'static str = "tropical_maxmul_f32_nn";

    fn launch_gemm(
        ctx: &CudaContext,
        a: &GpuMatrix<f32>,
        b: &GpuMatrix<f32>,
        c: &mut GpuMatrix<f32>,
    ) -> Result<()> {
        let m = a.rows();
        let k = a.cols();
        let n = b.cols();

        let kernel = ctx.get_kernel(Self::KERNEL_NAME)?;
        let grid = CudaContext::grid_dims_f32(m, n);
        let block = CudaContext::block_dims_f32();

        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: block,
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel.launch(
                cfg,
                (
                    a.as_slice(),
                    b.as_slice(),
                    c.as_slice_mut(),
                    m as i32,
                    n as i32,
                    k as i32,
                ),
            )?;
        }

        ctx.device().synchronize()?;
        Ok(())
    }
}

impl CudaKernel for TropicalMaxPlus<f64> {
    const KERNEL_NAME: &'static str = "tropical_maxplus_f64_nn";

    fn launch_gemm(
        ctx: &CudaContext,
        a: &GpuMatrix<f64>,
        b: &GpuMatrix<f64>,
        c: &mut GpuMatrix<f64>,
    ) -> Result<()> {
        let m = a.rows();
        let k = a.cols();
        let n = b.cols();

        let kernel = ctx.get_kernel(Self::KERNEL_NAME)?;
        let grid = CudaContext::grid_dims_f64(m, n);
        let block = CudaContext::block_dims_f64();

        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: block,
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel.launch(
                cfg,
                (
                    a.as_slice(),
                    b.as_slice(),
                    c.as_slice_mut(),
                    m as i32,
                    n as i32,
                    k as i32,
                ),
            )?;
        }

        ctx.device().synchronize()?;
        Ok(())
    }
}
