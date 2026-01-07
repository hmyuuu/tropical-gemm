use crate::detect::{simd_level, SimdLevel};
use crate::kernels::*;
use tropical_gemm_core::{tropical_gemm_inner, TilingParams, Transpose};
use tropical_types::{TropicalMaxMul, TropicalMaxPlus, TropicalMinPlus, TropicalSemiring};

/// Runtime-dispatched GEMM that selects the best kernel for the current CPU.
///
/// # Safety
/// Same requirements as `tropical_gemm_inner`
pub unsafe fn tropical_gemm_dispatch<T: TropicalSemiring + KernelDispatch>(
    m: usize,
    n: usize,
    k: usize,
    a: *const T::Scalar,
    lda: usize,
    trans_a: Transpose,
    b: *const T::Scalar,
    ldb: usize,
    trans_b: Transpose,
    c: *mut T,
    ldc: usize,
) {
    T::dispatch_gemm(m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc);
}

/// Trait for types that support kernel dispatch.
pub trait KernelDispatch: TropicalSemiring {
    /// Dispatch to the appropriate kernel based on CPU features.
    unsafe fn dispatch_gemm(
        m: usize,
        n: usize,
        k: usize,
        a: *const Self::Scalar,
        lda: usize,
        trans_a: Transpose,
        b: *const Self::Scalar,
        ldb: usize,
        trans_b: Transpose,
        c: *mut Self,
        ldc: usize,
    );
}

impl KernelDispatch for TropicalMaxPlus<f32> {
    unsafe fn dispatch_gemm(
        m: usize,
        n: usize,
        k: usize,
        a: *const f32,
        lda: usize,
        trans_a: Transpose,
        b: *const f32,
        ldb: usize,
        trans_b: Transpose,
        c: *mut Self,
        ldc: usize,
    ) {
        match simd_level() {
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2 | SimdLevel::Avx512 => {
                let kernel = Avx2MaxPlusF32Kernel;
                let params = TilingParams::F32_AVX2;
                tropical_gemm_inner::<Self, _>(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel,
                );
            }
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon => {
                let kernel = NeonMaxPlusF32Kernel;
                let params = TilingParams::new(128, 128, 256, 4, 4);
                tropical_gemm_inner::<Self, _>(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel,
                );
            }
            _ => {
                let kernel = PortableKernel;
                let params = TilingParams::PORTABLE;
                tropical_gemm_inner::<Self, _>(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel,
                );
            }
        }
    }
}

impl KernelDispatch for TropicalMaxPlus<f64> {
    unsafe fn dispatch_gemm(
        m: usize,
        n: usize,
        k: usize,
        a: *const f64,
        lda: usize,
        trans_a: Transpose,
        b: *const f64,
        ldb: usize,
        trans_b: Transpose,
        c: *mut Self,
        ldc: usize,
    ) {
        match simd_level() {
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2 | SimdLevel::Avx512 => {
                let kernel = Avx2MaxPlusF64Kernel;
                let params = TilingParams::F64_AVX2;
                tropical_gemm_inner::<Self, _>(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel,
                );
            }
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon => {
                let kernel = NeonMaxPlusF64Kernel;
                let params = TilingParams::new(64, 64, 128, 2, 2);
                tropical_gemm_inner::<Self, _>(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel,
                );
            }
            _ => {
                let kernel = PortableKernel;
                let params = TilingParams::PORTABLE;
                tropical_gemm_inner::<Self, _>(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel,
                );
            }
        }
    }
}

impl KernelDispatch for TropicalMinPlus<f32> {
    unsafe fn dispatch_gemm(
        m: usize,
        n: usize,
        k: usize,
        a: *const f32,
        lda: usize,
        trans_a: Transpose,
        b: *const f32,
        ldb: usize,
        trans_b: Transpose,
        c: *mut Self,
        ldc: usize,
    ) {
        match simd_level() {
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2 | SimdLevel::Avx512 => {
                let kernel = Avx2MinPlusF32Kernel;
                let params = TilingParams::F32_AVX2;
                tropical_gemm_inner::<Self, _>(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel,
                );
            }
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon => {
                let kernel = NeonMinPlusF32Kernel;
                let params = TilingParams::new(128, 128, 256, 4, 4);
                tropical_gemm_inner::<Self, _>(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel,
                );
            }
            _ => {
                let kernel = PortableKernel;
                let params = TilingParams::PORTABLE;
                tropical_gemm_inner::<Self, _>(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel,
                );
            }
        }
    }
}

impl KernelDispatch for TropicalMaxMul<f32> {
    unsafe fn dispatch_gemm(
        m: usize,
        n: usize,
        k: usize,
        a: *const f32,
        lda: usize,
        trans_a: Transpose,
        b: *const f32,
        ldb: usize,
        trans_b: Transpose,
        c: *mut Self,
        ldc: usize,
    ) {
        match simd_level() {
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2 | SimdLevel::Avx512 => {
                let kernel = Avx2MaxMulF32Kernel;
                let params = TilingParams::F32_AVX2;
                tropical_gemm_inner::<Self, _>(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel,
                );
            }
            _ => {
                let kernel = PortableKernel;
                let params = TilingParams::PORTABLE;
                tropical_gemm_inner::<Self, _>(
                    m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel,
                );
            }
        }
    }
}

// Fallback implementations for other types
macro_rules! impl_kernel_dispatch_portable {
    ($($t:ty),*) => {
        $(
            impl KernelDispatch for $t {
                unsafe fn dispatch_gemm(
                    m: usize,
                    n: usize,
                    k: usize,
                    a: *const Self::Scalar,
                    lda: usize,
                    trans_a: Transpose,
                    b: *const Self::Scalar,
                    ldb: usize,
                    trans_b: Transpose,
                    c: *mut Self,
                    ldc: usize,
                ) {
                    let kernel = PortableKernel;
                    let params = TilingParams::PORTABLE;
                    tropical_gemm_inner::<Self, _>(
                        m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel,
                    );
                }
            }
        )*
    };
}

impl_kernel_dispatch_portable!(
    TropicalMinPlus<f64>,
    TropicalMaxMul<f64>,
    TropicalMaxPlus<i32>,
    TropicalMaxPlus<i64>,
    TropicalMinPlus<i32>,
    TropicalMinPlus<i64>,
    TropicalMaxMul<i32>,
    TropicalMaxMul<i64>
);
