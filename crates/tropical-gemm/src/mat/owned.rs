//! Owned matrix type.

use std::ops::{Index, IndexMut};

use crate::types::TropicalSemiring;

use super::MatRef;

/// Owned matrix storing semiring values.
///
/// The matrix stores values in row-major order. Use factory methods
/// to create matrices:
///
/// ```
/// use tropical_gemm::{Mat, MaxPlus, TropicalSemiring};
///
/// let zeros = Mat::<MaxPlus<f32>>::zeros(3, 4);
/// let identity = Mat::<MaxPlus<f32>>::identity(3);
/// let custom = Mat::<MaxPlus<f32>>::from_fn(2, 2, |i, j| {
///     MaxPlus::<f32>::from_scalar((i + j) as f32)
/// });
/// ```
#[derive(Debug, Clone)]
pub struct Mat<S: TropicalSemiring> {
    pub(crate) data: Vec<S>,
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
}

impl<S: TropicalSemiring> Mat<S> {
    /// Create a matrix filled with tropical zeros.
    ///
    /// For MaxPlus, this fills with -∞.
    /// For MinPlus, this fills with +∞.
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self {
            data: vec![S::tropical_zero(); nrows * ncols],
            nrows,
            ncols,
        }
    }

    /// Create a tropical identity matrix.
    ///
    /// Diagonal elements are tropical one (0 for MaxPlus/MinPlus).
    /// Off-diagonal elements are tropical zero (-∞ for MaxPlus, +∞ for MinPlus).
    pub fn identity(n: usize) -> Self {
        let mut mat = Self::zeros(n, n);
        for i in 0..n {
            mat.data[i * n + i] = S::tropical_one();
        }
        mat
    }

    /// Create a matrix from a function.
    ///
    /// The function is called with (row, col) indices.
    pub fn from_fn<F>(nrows: usize, ncols: usize, mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> S,
    {
        let data = (0..nrows * ncols)
            .map(|idx| f(idx / ncols, idx % ncols))
            .collect();
        Self { data, nrows, ncols }
    }

    /// Create a matrix from row-major scalar data.
    ///
    /// Each scalar is wrapped in the semiring type.
    pub fn from_row_major(data: &[S::Scalar], nrows: usize, ncols: usize) -> Self
    where
        S::Scalar: Copy,
    {
        assert_eq!(
            data.len(),
            nrows * ncols,
            "data length {} != nrows {} * ncols {}",
            data.len(),
            nrows,
            ncols
        );
        let data = data.iter().map(|&s| S::from_scalar(s)).collect();
        Self { data, nrows, ncols }
    }

    /// Create a matrix from a vector of semiring values.
    pub fn from_vec(data: Vec<S>, nrows: usize, ncols: usize) -> Self {
        assert_eq!(
            data.len(),
            nrows * ncols,
            "data length {} != nrows {} * ncols {}",
            data.len(),
            nrows,
            ncols
        );
        Self { data, nrows, ncols }
    }

    /// Number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Get the underlying data as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[S] {
        &self.data
    }

    /// Get the underlying data as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [S] {
        &mut self.data
    }

    /// Convert to an immutable matrix reference.
    ///
    /// The returned reference views the scalar values.
    pub fn as_ref(&self) -> MatRef<'_, S>
    where
        S::Scalar: Copy,
    {
        // Extract scalars from semiring values
        // This requires that the data is laid out such that we can get scalars
        // For now, we create a view that extracts values on-the-fly
        // This is a limitation - ideally we'd have a separate scalar buffer
        MatRef::from_mat(self)
    }

    /// Get a mutable pointer to the data.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut S {
        self.data.as_mut_ptr()
    }
}

impl<S: TropicalSemiring> Index<(usize, usize)> for Mat<S> {
    type Output = S;

    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &S {
        debug_assert!(
            i < self.nrows,
            "row index {} out of bounds {}",
            i,
            self.nrows
        );
        debug_assert!(
            j < self.ncols,
            "col index {} out of bounds {}",
            j,
            self.ncols
        );
        &self.data[i * self.ncols + j]
    }
}

impl<S: TropicalSemiring> IndexMut<(usize, usize)> for Mat<S> {
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut S {
        debug_assert!(
            i < self.nrows,
            "row index {} out of bounds {}",
            i,
            self.nrows
        );
        debug_assert!(
            j < self.ncols,
            "col index {} out of bounds {}",
            j,
            self.ncols
        );
        &mut self.data[i * self.ncols + j]
    }
}
