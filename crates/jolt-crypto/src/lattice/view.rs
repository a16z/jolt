//! Matrix views over the public field stream.
//!
//! Every Ajtai matrix is an offset-0 prefix of the stream that
//! [`derive_public_matrix_prefix`](super::derive_public_matrix_prefix)
//! expands, read row-major with each ring entry's `ring_dim` coefficients
//! contiguous. Roles overlap, and a view is a function of its whole shape:
//! changing the width moves every row after the first.

use crate::CryptoError;

/// The shape of a ring matrix: `rows × width` entries of `ring_dim`
/// coefficients each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatrixShape {
    pub rows: usize,
    pub width: usize,
    pub ring_dim: usize,
}

impl MatrixShape {
    /// Number of field elements the view covers, `rows · width · ring_dim`.
    pub fn num_field_elements(&self) -> Result<usize, CryptoError> {
        self.rows
            .checked_mul(self.width)
            .and_then(|entries| entries.checked_mul(self.ring_dim))
            .ok_or_else(|| CryptoError::InvalidSetup("matrix shape overflows usize".into()))
    }
}

/// A borrowed offset-0 view of the public stream.
#[derive(Clone, Copy, Debug)]
pub struct MatrixView<'a, F> {
    shape: MatrixShape,
    data: &'a [F],
}

impl<'a, F> MatrixView<'a, F> {
    /// The view of `shape` over `stream`.
    ///
    /// `check_public_matrix_prefix` accepts a truncated stream, so the
    /// length is checked here, in release builds: a stream shorter than the
    /// shape is an error.
    pub fn new(shape: MatrixShape, stream: &'a [F]) -> Result<Self, CryptoError> {
        let len = shape.num_field_elements()?;
        let data = stream.get(..len).ok_or(CryptoError::InvalidSize {
            expected: len,
            actual: stream.len(),
        })?;
        Ok(Self { shape, data })
    }

    pub fn shape(&self) -> MatrixShape {
        self.shape
    }

    /// The coefficients of entry `(row, col)`.
    pub fn entry(&self, row: usize, col: usize) -> Result<&'a [F], CryptoError> {
        if row >= self.shape.rows || col >= self.shape.width {
            return Err(CryptoError::InvalidInput(format!(
                "matrix entry ({row}, {col}) outside a {}×{} view",
                self.shape.rows, self.shape.width
            )));
        }
        let start = (row * self.shape.width + col) * self.shape.ring_dim;
        self.data
            .get(start..start + self.shape.ring_dim)
            .ok_or_else(|| CryptoError::Internal("matrix entry outside its checked view".into()))
    }
}
