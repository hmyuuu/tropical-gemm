use crate::argmax::GemmWithArgmax;
use crate::kernel::{Microkernel, MicrokernelWithArgmax, PortableMicrokernel};
use crate::packing::{pack_a, pack_b, packed_a_size, packed_b_size, Layout, Transpose};
use crate::tiling::{BlockIterator, TilingParams};
use tropical_types::{TropicalSemiring, TropicalWithArgmax};

/// Tropical GEMM: C = A ⊗ B
///
/// Computes C[i,j] = ⊕_k (A[i,k] ⊗ B[k,j])
///
/// This is a portable (non-SIMD) implementation using BLIS-style blocking
/// for cache efficiency.
///
/// # Parameters
/// - `m`: Number of rows in A and C
/// - `n`: Number of columns in B and C
/// - `k`: Number of columns in A / rows in B
/// - `a`: Pointer to matrix A data
/// - `lda`: Leading dimension of A
/// - `trans_a`: Whether A is transposed
/// - `b`: Pointer to matrix B data
/// - `ldb`: Leading dimension of B
/// - `trans_b`: Whether B is transposed
/// - `c`: Pointer to matrix C data (output)
/// - `ldc`: Leading dimension of C
///
/// # Safety
/// - All pointers must be valid for the specified dimensions
/// - Memory regions must not overlap inappropriately
pub unsafe fn tropical_gemm_portable<T: TropicalSemiring>(
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
    let params = TilingParams::PORTABLE;
    let kernel = PortableMicrokernel;

    tropical_gemm_inner::<T, PortableMicrokernel>(
        m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc, &params, &kernel,
    );
}

/// Tropical GEMM with custom kernel and tiling parameters.
///
/// # Safety
/// Same requirements as `tropical_gemm_portable`
pub unsafe fn tropical_gemm_inner<T: TropicalSemiring, K: Microkernel<T>>(
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
    params: &TilingParams,
    kernel: &K,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }

    // Allocate packing buffers
    let mut packed_a = vec![T::Scalar::scalar_zero(); packed_a_size(params.mc, params.kc, K::MR)];
    let mut packed_b = vec![T::Scalar::scalar_zero(); packed_b_size(params.kc, params.nc, K::NR)];

    // BLIS-style 5-loop blocking
    // Loop 5: blocks of n
    for (jc, nc) in BlockIterator::new(n, params.nc) {
        // Loop 4: blocks of k
        for (pc, kc) in BlockIterator::new(k, params.kc) {
            // Pack B panel: kc × nc
            pack_b::<T::Scalar>(
                kc,
                nc,
                b_panel_ptr(b, pc, jc, ldb, trans_b),
                ldb,
                Layout::RowMajor,
                trans_b,
                packed_b.as_mut_ptr(),
                K::NR,
            );

            // Loop 3: blocks of m
            for (ic, mc) in BlockIterator::new(m, params.mc) {
                // Pack A panel: mc × kc
                pack_a::<T::Scalar>(
                    mc,
                    kc,
                    a_panel_ptr(a, ic, pc, lda, trans_a),
                    lda,
                    Layout::RowMajor,
                    trans_a,
                    packed_a.as_mut_ptr(),
                    K::MR,
                );

                // Loop 2: micro-blocks of n
                let n_blocks = nc.div_ceil(K::NR);
                for jr in 0..n_blocks {
                    let j_start = jr * K::NR;
                    let nr = (nc - j_start).min(K::NR);

                    // Loop 1: micro-blocks of m
                    let m_blocks = mc.div_ceil(K::MR);
                    for ir in 0..m_blocks {
                        let i_start = ir * K::MR;
                        let mr = (mc - i_start).min(K::MR);

                        // Microkernel
                        let a_ptr = packed_a.as_ptr().add(ir * K::MR * kc);
                        let b_ptr = packed_b.as_ptr().add(jr * K::NR * kc);
                        let c_ptr = c.add((ic + i_start) * ldc + (jc + j_start));

                        kernel.execute(mr, nr, kc, a_ptr, b_ptr, c_ptr, ldc);
                    }
                }
            }
        }
    }
}

/// Tropical GEMM with argmax tracking.
///
/// Same as `tropical_gemm_portable` but also computes argmax indices.
///
/// # Safety
/// Same requirements as `tropical_gemm_portable`
pub unsafe fn tropical_gemm_with_argmax_portable<T: TropicalWithArgmax<Index = u32>>(
    m: usize,
    n: usize,
    k: usize,
    a: *const T::Scalar,
    lda: usize,
    trans_a: Transpose,
    b: *const T::Scalar,
    ldb: usize,
    trans_b: Transpose,
    result: &mut GemmWithArgmax<T>,
) {
    let params = TilingParams::PORTABLE;
    let kernel = PortableMicrokernel;

    tropical_gemm_with_argmax_inner::<T, PortableMicrokernel>(
        m, n, k, a, lda, trans_a, b, ldb, trans_b, result, &params, &kernel,
    );
}

/// Tropical GEMM with argmax tracking and custom kernel.
///
/// # Safety
/// Same requirements as `tropical_gemm_portable`
pub unsafe fn tropical_gemm_with_argmax_inner<
    T: TropicalWithArgmax<Index = u32>,
    K: MicrokernelWithArgmax<T>,
>(
    m: usize,
    n: usize,
    k: usize,
    a: *const T::Scalar,
    lda: usize,
    trans_a: Transpose,
    b: *const T::Scalar,
    ldb: usize,
    trans_b: Transpose,
    result: &mut GemmWithArgmax<T>,
    params: &TilingParams,
    kernel: &K,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }

    let ldc = result.ld;
    let (c, argmax) = result.as_mut_ptrs();

    // Allocate packing buffers
    let mut packed_a = vec![T::Scalar::scalar_zero(); packed_a_size(params.mc, params.kc, K::MR)];
    let mut packed_b = vec![T::Scalar::scalar_zero(); packed_b_size(params.kc, params.nc, K::NR)];

    // BLIS-style 5-loop blocking
    for (jc, nc) in BlockIterator::new(n, params.nc) {
        for (pc, kc) in BlockIterator::new(k, params.kc) {
            pack_b::<T::Scalar>(
                kc,
                nc,
                b_panel_ptr(b, pc, jc, ldb, trans_b),
                ldb,
                Layout::RowMajor,
                trans_b,
                packed_b.as_mut_ptr(),
                K::NR,
            );

            for (ic, mc) in BlockIterator::new(m, params.mc) {
                pack_a::<T::Scalar>(
                    mc,
                    kc,
                    a_panel_ptr(a, ic, pc, lda, trans_a),
                    lda,
                    Layout::RowMajor,
                    trans_a,
                    packed_a.as_mut_ptr(),
                    K::MR,
                );

                let n_blocks = nc.div_ceil(K::NR);
                for jr in 0..n_blocks {
                    let j_start = jr * K::NR;
                    let nr = (nc - j_start).min(K::NR);

                    let m_blocks = mc.div_ceil(K::MR);
                    for ir in 0..m_blocks {
                        let i_start = ir * K::MR;
                        let mr = (mc - i_start).min(K::MR);

                        let a_ptr = packed_a.as_ptr().add(ir * K::MR * kc);
                        let b_ptr = packed_b.as_ptr().add(jr * K::NR * kc);
                        let c_ptr = c.add((ic + i_start) * ldc + (jc + j_start));
                        let argmax_ptr = argmax.add((ic + i_start) * ldc + (jc + j_start));

                        kernel.execute_with_argmax(
                            mr, nr, kc, pc, a_ptr, b_ptr, c_ptr, argmax_ptr, ldc,
                        );
                    }
                }
            }
        }
    }
}

/// Get pointer to A panel considering transpose.
#[inline]
unsafe fn a_panel_ptr<T>(
    a: *const T,
    row: usize,
    col: usize,
    lda: usize,
    trans: Transpose,
) -> *const T {
    match trans {
        Transpose::NoTrans => a.add(row * lda + col),
        Transpose::Trans => a.add(col * lda + row),
    }
}

/// Get pointer to B panel considering transpose.
#[inline]
unsafe fn b_panel_ptr<T>(
    b: *const T,
    row: usize,
    col: usize,
    ldb: usize,
    trans: Transpose,
) -> *const T {
    match trans {
        Transpose::NoTrans => b.add(row * ldb + col),
        Transpose::Trans => b.add(col * ldb + row),
    }
}

use tropical_types::TropicalScalar;

#[cfg(test)]
mod tests {
    use super::*;
    use tropical_types::TropicalMaxPlus;

    #[test]
    fn test_simple_gemm() {
        let m = 2;
        let n = 2;
        let k = 3;

        // A: 2x3 matrix
        let a: [f64; 6] = [
            1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
        ];

        // B: 3x2 matrix
        let b: [f64; 6] = [
            1.0, 2.0, // row 0
            3.0, 4.0, // row 1
            5.0, 6.0, // row 2
        ];

        let mut c = vec![TropicalMaxPlus::tropical_zero(); m * n];

        unsafe {
            tropical_gemm_portable::<TropicalMaxPlus<f64>>(
                m,
                n,
                k,
                a.as_ptr(),
                3,
                Transpose::NoTrans,
                b.as_ptr(),
                2,
                Transpose::NoTrans,
                c.as_mut_ptr(),
                n,
            );
        }

        // C[0,0] = max(1+1, 2+3, 3+5) = max(2, 5, 8) = 8
        assert_eq!(c[0].0, 8.0);
        // C[0,1] = max(1+2, 2+4, 3+6) = max(3, 6, 9) = 9
        assert_eq!(c[1].0, 9.0);
        // C[1,0] = max(4+1, 5+3, 6+5) = max(5, 8, 11) = 11
        assert_eq!(c[2].0, 11.0);
        // C[1,1] = max(4+2, 5+4, 6+6) = max(6, 9, 12) = 12
        assert_eq!(c[3].0, 12.0);
    }

    #[test]
    fn test_gemm_with_argmax() {
        let m = 2;
        let n = 2;
        let k = 3;

        let a: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let mut result: GemmWithArgmax<TropicalMaxPlus<f64>> = GemmWithArgmax::new(m, n);

        unsafe {
            tropical_gemm_with_argmax_portable::<TropicalMaxPlus<f64>>(
                m,
                n,
                k,
                a.as_ptr(),
                3,
                Transpose::NoTrans,
                b.as_ptr(),
                2,
                Transpose::NoTrans,
                &mut result,
            );
        }

        // C[0,0] = max(1+1, 2+3, 3+5) = 8 at k=2
        assert_eq!(result.get(0, 0).0, 8.0);
        assert_eq!(result.get_argmax(0, 0), 2);

        // C[1,1] = max(4+2, 5+4, 6+6) = 12 at k=2
        assert_eq!(result.get(1, 1).0, 12.0);
        assert_eq!(result.get_argmax(1, 1), 2);
    }
}
