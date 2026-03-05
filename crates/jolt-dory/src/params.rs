//! Instance-local Dory parameters.
//!
//! Unlike the global `DoryGlobals` in jolt-core, `DoryParams` stores all
//! Dory matrix shape parameters on the struct itself, enabling multiple
//! independent Dory instances within a single process.

use serde::{Deserialize, Serialize};

/// Instance-local Dory parameters controlling the commitment matrix shape.
///
/// Dory commits to a multilinear polynomial by interpreting its evaluations
/// as a matrix of shape `max_num_rows x num_columns`. The parameter `t`
/// controls the maximum trace length (cycle count), and together with the
/// matrix dimensions determines how polynomial coefficients map to matrix
/// positions.
///
/// # Matrix shape
///
/// Given a polynomial with `total_vars = log2(K * T)` variables, the matrix
/// is shaped as:
/// - **Even** `total_vars`: square matrix (`num_columns == max_num_rows`).
/// - **Odd** `total_vars`: almost-square (`num_columns == 2 * max_num_rows`).
///
/// The `sigma = log2(num_columns)` column variables and `nu = log2(max_num_rows)`
/// row variables satisfy `sigma + nu = total_vars`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryParams {
    /// Maximum trace length (cycle count). Must be a power of two.
    pub t: usize,
    /// Maximum number of rows in the Dory matrix. Must be a power of two.
    pub max_num_rows: usize,
    /// Number of columns in the Dory matrix. Must be a power of two.
    pub num_columns: usize,
}

impl DoryParams {
    /// Creates new Dory parameters from the given matrix shape.
    ///
    /// All values must be powers of two. No validation is performed here;
    /// use [`DoryParams::from_dimensions`] for automatic calculation.
    pub fn new(t: usize, max_num_rows: usize, num_columns: usize) -> Self {
        Self {
            t,
            max_num_rows,
            num_columns,
        }
    }

    /// Computes optimal matrix dimensions from address space size `k` and trace length `t`.
    ///
    /// Both `k` and `t` must be powers of two. The total size `k * t` is split
    /// into a balanced matrix:
    /// - If `log2(k * t)` is even, produces a square matrix.
    /// - If `log2(k * t)` is odd, produces `num_columns = 2 * max_num_rows`.
    ///
    /// # Panics
    ///
    /// Panics if `k * t` is zero.
    pub fn from_dimensions(k: usize, t: usize) -> Self {
        assert!(k * t > 0, "k * t must be nonzero");
        let total_vars = (k * t).trailing_zeros() as usize;
        let (sigma, nu) = balanced_sigma_nu(total_vars);
        Self {
            t,
            max_num_rows: 1 << nu,
            num_columns: 1 << sigma,
        }
    }

    /// Number of column variables: `sigma = log2(num_columns)`.
    pub fn sigma(&self) -> usize {
        self.num_columns.trailing_zeros() as usize
    }

    /// Number of row variables: `nu = log2(max_num_rows)`.
    pub fn nu(&self) -> usize {
        self.max_num_rows.trailing_zeros() as usize
    }

    /// Total number of polynomial variables supported: `sigma + nu`.
    pub fn total_vars(&self) -> usize {
        self.sigma() + self.nu()
    }
}

/// Splits `total_vars` into a balanced `(sigma, nu)` pair where
/// `sigma = ceil(total_vars / 2)` is the number of column variables
/// and `nu = total_vars - sigma` is the number of row variables.
fn balanced_sigma_nu(total_vars: usize) -> (usize, usize) {
    let sigma = total_vars.div_ceil(2);
    let nu = total_vars - sigma;
    (sigma, nu)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_dimensions_square() {
        let params = DoryParams::from_dimensions(4, 4);
        assert_eq!(params.num_columns, 4);
        assert_eq!(params.max_num_rows, 4);
        assert_eq!(params.t, 4);
        assert_eq!(params.sigma(), 2);
        assert_eq!(params.nu(), 2);
        assert_eq!(params.total_vars(), 4);
    }

    #[test]
    fn from_dimensions_almost_square() {
        let params = DoryParams::from_dimensions(4, 8);
        assert_eq!(params.num_columns, 8);
        assert_eq!(params.max_num_rows, 4);
        assert_eq!(params.t, 8);
        assert_eq!(params.sigma(), 3);
        assert_eq!(params.nu(), 2);
        assert_eq!(params.total_vars(), 5);
    }

    #[test]
    fn serialization_round_trip() {
        let params = DoryParams::new(16, 8, 8);
        let serialized = serde_json::to_string(&params).expect("serialize");
        let deserialized: DoryParams = serde_json::from_str(&serialized).expect("deserialize");
        assert_eq!(params, deserialized);
    }

    #[test]
    fn new_stores_values() {
        let params = DoryParams::new(32, 16, 16);
        assert_eq!(params.t, 32);
        assert_eq!(params.max_num_rows, 16);
        assert_eq!(params.num_columns, 16);
    }
}
