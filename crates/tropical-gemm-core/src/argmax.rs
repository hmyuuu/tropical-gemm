use tropical_types::TropicalWithArgmax;

/// Result of GEMM with argmax tracking.
///
/// For each element C[i,j], tracks which k index produced the optimal value:
/// C[i,j] = ⊕_{k} A[i,k] ⊗ B[k,j]
/// argmax[i,j] = argmax_k (A[i,k] ⊗ B[k,j])
#[derive(Debug, Clone)]
pub struct GemmWithArgmax<T: TropicalWithArgmax<Index = u32>> {
    /// The result matrix values.
    pub values: Vec<T>,
    /// The argmax indices for each element.
    pub argmax: Vec<u32>,
    /// Number of rows in the result.
    pub m: usize,
    /// Number of columns in the result.
    pub n: usize,
    /// Leading dimension (stride between rows).
    pub ld: usize,
}

impl<T: TropicalWithArgmax<Index = u32>> GemmWithArgmax<T> {
    /// Create a new result container with tropical zeros.
    pub fn new(m: usize, n: usize) -> Self {
        let size = m * n;
        Self {
            values: vec![T::tropical_zero(); size],
            argmax: vec![0u32; size],
            m,
            n,
            ld: n,
        }
    }

    /// Create a new result container with specified leading dimension.
    pub fn with_ld(m: usize, n: usize, ld: usize) -> Self {
        assert!(ld >= n, "Leading dimension must be >= n");
        let size = m * ld;
        Self {
            values: vec![T::tropical_zero(); size],
            argmax: vec![0u32; size],
            m,
            n,
            ld,
        }
    }

    /// Get value at (i, j).
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> T {
        debug_assert!(i < self.m && j < self.n);
        self.values[i * self.ld + j]
    }

    /// Get argmax at (i, j).
    #[inline]
    pub fn get_argmax(&self, i: usize, j: usize) -> u32 {
        debug_assert!(i < self.m && j < self.n);
        self.argmax[i * self.ld + j]
    }

    /// Get mutable reference to value at (i, j).
    #[inline]
    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut T {
        debug_assert!(i < self.m && j < self.n);
        &mut self.values[i * self.ld + j]
    }

    /// Get mutable reference to argmax at (i, j).
    #[inline]
    pub fn get_argmax_mut(&mut self, i: usize, j: usize) -> &mut u32 {
        debug_assert!(i < self.m && j < self.n);
        &mut self.argmax[i * self.ld + j]
    }

    /// Get raw pointers to the data.
    #[inline]
    pub fn as_mut_ptrs(&mut self) -> (*mut T, *mut u32) {
        (self.values.as_mut_ptr(), self.argmax.as_mut_ptr())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tropical_types::TropicalMaxPlus;

    #[test]
    fn test_gemm_with_argmax_new() {
        let result: GemmWithArgmax<TropicalMaxPlus<f64>> = GemmWithArgmax::new(3, 4);

        assert_eq!(result.m, 3);
        assert_eq!(result.n, 4);
        assert_eq!(result.ld, 4);
        assert_eq!(result.values.len(), 12);
        assert_eq!(result.argmax.len(), 12);

        // All values should be tropical zero (-inf)
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(result.get(i, j).0, f64::NEG_INFINITY);
                assert_eq!(result.get_argmax(i, j), 0);
            }
        }
    }
}
