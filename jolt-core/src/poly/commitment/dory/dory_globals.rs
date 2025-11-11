//! Global state management for Dory parameters

use crate::utils::math::Math;
use dory::backends::arkworks::{init_cache, is_cached, ArkG1, ArkG2};
use once_cell::sync::OnceCell;

static mut GLOBAL_T: OnceCell<usize> = OnceCell::new();
static mut MAX_NUM_ROWS: OnceCell<usize> = OnceCell::new();
static mut NUM_COLUMNS: OnceCell<usize> = OnceCell::new();

/// Global state management for Dory parameters
pub struct DoryGlobals;

impl DoryGlobals {
    pub fn set_max_num_rows(max_num_rows: usize) {
        #[allow(static_mut_refs)]
        unsafe {
            let _ = MAX_NUM_ROWS.set(max_num_rows);
        }
    }

    pub fn get_max_num_rows() -> usize {
        #[allow(static_mut_refs)]
        unsafe {
            *MAX_NUM_ROWS.get().expect("max_num_rows not initialized")
        }
    }

    pub fn set_num_columns(num_columns: usize) {
        #[allow(static_mut_refs)]
        unsafe {
            let _ = NUM_COLUMNS.set(num_columns);
        }
    }

    pub fn get_num_columns() -> usize {
        #[allow(static_mut_refs)]
        unsafe {
            *NUM_COLUMNS.get().expect("num_columns not initialized")
        }
    }

    pub fn set_T(t: usize) {
        #[allow(static_mut_refs)]
        unsafe {
            let _ = GLOBAL_T.set(t);
        }
    }

    pub fn get_T() -> usize {
        #[allow(static_mut_refs)]
        unsafe {
            *GLOBAL_T.get().expect("t not initialized")
        }
    }

    /// Initialize the globals for the dory matrix
    ///
    /// # Arguments
    /// * `K` - Maximum address space size (K in OneHot polynomials)
    /// * `T` - Maximum trace length (cycle count)
    ///
    /// The matrix dimensions are calculated to minimize padding:
    /// - If log2(K*T) is even: creates a square matrix
    /// - If log2(K*T) is odd: creates an almost-square matrix (columns = 2*rows)
    pub fn initialize(K: usize, T: usize) -> Option<()> {
        let total_size = K * T;
        let total_vars = total_size.log_2();

        // Calculate optimal matrix dimensions
        let (num_columns, num_rows): (usize, usize) = if total_vars % 2 == 0 {
            // Even total vars: square matrix
            let side = 1 << (total_vars / 2);
            (side, side)
        } else {
            // Odd total vars: almost square (columns = 2*rows)
            let sigma = (total_vars + 1) / 2;
            let nu = total_vars - sigma;
            (1 << sigma, 1 << nu)
        };

        Self::set_num_columns(num_columns);
        Self::set_T(T);
        Self::set_max_num_rows(num_rows);
        Some(())
    }

    /// Reset global state (for testing only)
    #[cfg(test)]
    pub fn reset() {
        #[allow(static_mut_refs)]
        unsafe {
            let _ = GLOBAL_T.take();
            let _ = MAX_NUM_ROWS.take();
            let _ = NUM_COLUMNS.take();
        }
    }

    /// Initialize the prepared point cache for faster pairing operations
    ///
    /// This should be called once after creating the prover setup to cache
    /// prepared versions of the G1 and G2 generators for ~20-30% speedup
    /// in repeated pairing operations.
    ///
    /// If the cache is already initialized, this function returns early without
    /// doing anything.
    ///
    /// # Arguments
    /// * `g1_vec` - Vector of G1 generators from the prover setup
    /// * `g2_vec` - Vector of G2 generators from the prover setup
    pub fn init_prepared_cache(g1_vec: &[ArkG1], g2_vec: &[ArkG2]) {
        if !is_cached() {
            init_cache(g1_vec, g2_vec);
        }
    }
}
