use tropical_types::{TropicalSemiring, TropicalWithArgmax};

/// Trait for GEMM microkernels.
///
/// A microkernel computes a small block of C += A * B using register blocking.
/// The dimensions mr x nr define the "register tile" that fits in CPU registers.
pub trait Microkernel<T: TropicalSemiring> {
    /// Rows of the microkernel (typically 4-8 for f32).
    const MR: usize;

    /// Columns of the microkernel (typically 4-8 for f32).
    const NR: usize;

    /// Execute the microkernel.
    ///
    /// Computes C[0..mr, 0..nr] = A[0..mr, 0..k] ⊗ B[0..k, 0..nr]
    /// where the result is combined with existing C values using tropical addition.
    ///
    /// # Safety
    /// - `a` must point to at least `mr * k` elements (packed column-major)
    /// - `b` must point to at least `k * nr` elements (packed row-major)
    /// - `c` must point to at least `mr * ldc` elements
    /// - `mr <= Self::MR` and `nr <= Self::NR`
    unsafe fn execute(
        &self,
        mr: usize,
        nr: usize,
        k: usize,
        a: *const T::Scalar,
        b: *const T::Scalar,
        c: *mut T,
        ldc: usize,
    );
}

/// Trait for microkernels that track argmax during computation.
pub trait MicrokernelWithArgmax<T: TropicalWithArgmax<Index = u32>>: Microkernel<T> {
    /// Execute the microkernel with argmax tracking.
    ///
    /// Same as `execute`, but also fills `argmax` with the k-index that
    /// produced each optimal C[i,j] value.
    ///
    /// # Safety
    /// Same requirements as `execute`, plus:
    /// - `argmax` must point to at least `mr * ldc` elements
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
    );
}

/// Portable (non-SIMD) microkernel implementation.
#[derive(Default, Clone, Copy)]
pub struct PortableMicrokernel;

/// Constants for PortableMicrokernel
impl PortableMicrokernel {
    /// Microkernel row dimension.
    pub const MR: usize = 4;
    /// Microkernel column dimension.
    pub const NR: usize = 4;
}

impl<T: TropicalSemiring> Microkernel<T> for PortableMicrokernel {
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
        const MR: usize = 4;
        const NR: usize = 4;

        // Initialize accumulators from C
        let mut acc = [[T::tropical_zero(); NR]; MR];
        for i in 0..mr {
            for j in 0..nr {
                acc[i][j] = *c.add(i * ldc + j);
            }
        }

        // Main loop
        for p in 0..k {
            for i in 0..mr {
                let a_val = T::from_scalar(*a.add(p * MR + i));
                for j in 0..nr {
                    let b_val = T::from_scalar(*b.add(p * NR + j));
                    let product = a_val.tropical_mul(b_val);
                    acc[i][j] = acc[i][j].tropical_add(product);
                }
            }
        }

        // Write back
        for i in 0..mr {
            for j in 0..nr {
                *c.add(i * ldc + j) = acc[i][j];
            }
        }
    }
}

impl<T: TropicalWithArgmax<Index = u32>> MicrokernelWithArgmax<T> for PortableMicrokernel {
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
        const MR: usize = 4;
        const NR: usize = 4;

        // Initialize accumulators from C and existing argmax
        let mut acc = [[T::tropical_zero(); NR]; MR];
        let mut idx = [[0u32; NR]; MR];
        for i in 0..mr {
            for j in 0..nr {
                acc[i][j] = *c.add(i * ldc + j);
                idx[i][j] = *argmax.add(i * ldc + j);
            }
        }

        // Main loop with argmax tracking
        for p in 0..k {
            let current_k = (k_offset + p) as u32;
            for i in 0..mr {
                let a_val = T::from_scalar(*a.add(p * MR + i));
                for j in 0..nr {
                    let b_val = T::from_scalar(*b.add(p * NR + j));
                    let product = a_val.tropical_mul(b_val);
                    let (new_acc, new_idx) =
                        acc[i][j].tropical_add_argmax(idx[i][j], product, current_k);
                    acc[i][j] = new_acc;
                    idx[i][j] = new_idx;
                }
            }
        }

        // Write back
        for i in 0..mr {
            for j in 0..nr {
                *c.add(i * ldc + j) = acc[i][j];
                *argmax.add(i * ldc + j) = idx[i][j];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tropical_types::TropicalMaxPlus;

    #[test]
    fn test_portable_kernel() {
        let kernel = PortableMicrokernel;
        let mr = 2;
        let nr = 2;
        let k = 3;

        // A: 2x3 matrix (packed column-major in MR chunks)
        // A = [[1, 2, 3],
        //      [4, 5, 6]]
        let a: [f64; 12] = [1.0, 4.0, 0.0, 0.0, 2.0, 5.0, 0.0, 0.0, 3.0, 6.0, 0.0, 0.0];

        // B: 3x2 matrix (packed row-major in NR chunks)
        // B = [[1, 2],
        //      [3, 4],
        //      [5, 6]]
        let b: [f64; 12] = [1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0];

        // C: 2x2 output
        let mut c = [TropicalMaxPlus::tropical_zero(); 4];
        let ldc = 2;

        unsafe {
            kernel.execute(mr, nr, k, a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), ldc);
        }

        // C[0,0] = max(A[0,0]+B[0,0], A[0,1]+B[1,0], A[0,2]+B[2,0])
        //        = max(1+1, 2+3, 3+5) = max(2, 5, 8) = 8
        assert_eq!(c[0].0, 8.0);

        // C[0,1] = max(1+2, 2+4, 3+6) = max(3, 6, 9) = 9
        assert_eq!(c[1].0, 9.0);

        // C[1,0] = max(4+1, 5+3, 6+5) = max(5, 8, 11) = 11
        assert_eq!(c[2].0, 11.0);

        // C[1,1] = max(4+2, 5+4, 6+6) = max(6, 9, 12) = 12
        assert_eq!(c[3].0, 12.0);
    }
}
