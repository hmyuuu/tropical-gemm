//! Mutable matrix reference type.

use crate::types::TropicalSemiring;

/// Mutable view over semiring data.
///
/// Unlike `MatRef`, this holds mutable references to semiring values,
/// not scalars. This is used for in-place operations.
#[derive(Debug)]
pub struct MatMut<'a, S: TropicalSemiring> {
    data: &'a mut [S],
    nrows: usize,
    ncols: usize,
}

impl<'a, S: TropicalSemiring> MatMut<'a, S> {
    /// Create a mutable matrix reference from a slice.
    pub fn from_slice(data: &'a mut [S], nrows: usize, ncols: usize) -> Self {
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

    /// Get the underlying data as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [S] {
        self.data
    }

    /// Get a mutable pointer to the data.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut S {
        self.data.as_mut_ptr()
    }

    /// Get a reference to the value at position (i, j).
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> &S {
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

    /// Get a mutable reference to the value at position (i, j).
    #[inline]
    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut S {
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
