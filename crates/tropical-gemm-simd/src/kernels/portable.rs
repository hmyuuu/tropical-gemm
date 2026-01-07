use tropical_gemm_core::{Microkernel, MicrokernelWithArgmax};
use tropical_types::{TropicalSemiring, TropicalWithArgmax};

/// Portable (non-SIMD) microkernel using the `wide` crate.
///
/// This provides a fallback when no SIMD instructions are available,
/// but uses `wide` types which may still auto-vectorize.
#[derive(Default, Clone, Copy)]
pub struct PortableKernel;

impl<T: TropicalSemiring> Microkernel<T> for PortableKernel {
    const MR: usize = 4;
    const NR: usize = 4;

    unsafe fn execute(
        &self,
        mr: usize,
        nr: usize,
        k: usize,
        a: *const T::Scalar,
        b: *const T::Scalar,
        c: *mut T,
        ldc: usize,
    ) {
        // Delegate to the core portable implementation
        let core_kernel = tropical_gemm_core::PortableMicrokernel;
        core_kernel.execute(mr, nr, k, a, b, c, ldc);
    }
}

impl<T: TropicalWithArgmax<Index = u32>> MicrokernelWithArgmax<T> for PortableKernel {
    unsafe fn execute_with_argmax(
        &self,
        mr: usize,
        nr: usize,
        k: usize,
        k_offset: usize,
        a: *const T::Scalar,
        b: *const T::Scalar,
        c: *mut T,
        argmax: *mut u32,
        ldc: usize,
    ) {
        let core_kernel = tropical_gemm_core::PortableMicrokernel;
        core_kernel.execute_with_argmax(mr, nr, k, k_offset, a, b, c, argmax, ldc);
    }
}
