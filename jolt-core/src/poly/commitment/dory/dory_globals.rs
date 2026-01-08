//! Global state management for Dory parameters

use crate::utils::math::Math;
use dory::backends::arkworks::{init_cache, is_cached, ArkG1, ArkG2};
use std::sync::{
    atomic::{AtomicU8, Ordering},
    OnceLock,
};

/// Dory matrix layout - determines how polynomial coefficients map to matrix positions.
///
/// In Dory, polynomial coefficients are arranged in a 2D matrix for commitment.
/// This enum controls the mapping from linear coefficient index to (row, col) position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DoryLayout {
    /// Row-major layout: coeff[i] -> row = i / num_cols, col = i % num_cols
    ///
    /// Coefficients in the same row are contiguous in memory.
    /// This is the default layout used by the current Rust implementation.
    #[default]
    RowMajor,

    /// Column-major layout (k-chunk major): coeff[i] -> row = i % num_rows, col = i / num_rows
    ///
    /// Coefficients in the same column are contiguous in memory.
    /// This layout was used in earlier versions and is compatible with certain GPU implementations.
    ColumnMajor,
}

impl DoryLayout {
    /// Convert a linear coefficient index to a (row, col) matrix position.
    #[inline]
    pub fn index_to_position(&self, coeff_idx: usize, num_rows: usize, num_cols: usize) -> (usize, usize) {
        match self {
            DoryLayout::RowMajor => {
                let row = coeff_idx / num_cols;
                let col = coeff_idx % num_cols;
                (row, col)
            }
            DoryLayout::ColumnMajor => {
                let row = coeff_idx % num_rows;
                let col = coeff_idx / num_rows;
                (row, col)
            }
        }
    }

    /// Convert a (row, col) matrix position to a linear coefficient index.
    #[inline]
    pub fn position_to_index(&self, row: usize, col: usize, num_rows: usize, num_cols: usize) -> usize {
        match self {
            DoryLayout::RowMajor => row * num_cols + col,
            DoryLayout::ColumnMajor => col * num_rows + row,
        }
    }
}

impl From<u8> for DoryLayout {
    fn from(value: u8) -> Self {
        match value {
            0 => DoryLayout::RowMajor,
            1 => DoryLayout::ColumnMajor,
            _ => panic!("Invalid DoryLayout value: {value}"),
        }
    }
}

// Main polynomial globals
static mut GLOBAL_T: OnceLock<usize> = OnceLock::new();
static mut MAX_NUM_ROWS: OnceLock<usize> = OnceLock::new();
static mut NUM_COLUMNS: OnceLock<usize> = OnceLock::new();

// Trusted advice globals
static mut TRUSTED_ADVICE_T: OnceLock<usize> = OnceLock::new();
static mut TRUSTED_ADVICE_MAX_NUM_ROWS: OnceLock<usize> = OnceLock::new();
static mut TRUSTED_ADVICE_NUM_COLUMNS: OnceLock<usize> = OnceLock::new();

// Untrusted advice globals
static mut UNTRUSTED_ADVICE_T: OnceLock<usize> = OnceLock::new();
static mut UNTRUSTED_ADVICE_MAX_NUM_ROWS: OnceLock<usize> = OnceLock::new();
static mut UNTRUSTED_ADVICE_NUM_COLUMNS: OnceLock<usize> = OnceLock::new();

// Context tracking: 0=Main, 1=TrustedAdvice, 2=UntrustedAdvice
static CURRENT_CONTEXT: AtomicU8 = AtomicU8::new(0);

// Layout tracking: 0=RowMajor, 1=ColumnMajor
static CURRENT_LAYOUT: AtomicU8 = AtomicU8::new(0);

/// Dory commitment context - determines which set of global parameters to use
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DoryContext {
    Main = 0,
    TrustedAdvice = 1,
    UntrustedAdvice = 2,
}

impl From<u8> for DoryContext {
    fn from(value: u8) -> Self {
        match value {
            0 => DoryContext::Main,
            1 => DoryContext::TrustedAdvice,
            2 => DoryContext::UntrustedAdvice,
            _ => panic!("Invalid DoryContext value: {value}"),
        }
    }
}

pub struct DoryContextGuard {
    previous_context: DoryContext,
}

impl Drop for DoryContextGuard {
    fn drop(&mut self) {
        CURRENT_CONTEXT.store(self.previous_context as u8, Ordering::SeqCst);
    }
}

/// Global state management for Dory parameters
pub struct DoryGlobals;

impl DoryGlobals {
    /// Get the current Dory context
    pub fn current_context() -> DoryContext {
        CURRENT_CONTEXT.load(Ordering::SeqCst).into()
    }

    /// Set the Dory context and return a guard that restores the previous context on drop
    pub fn with_context(context: DoryContext) -> DoryContextGuard {
        let previous = Self::current_context();
        CURRENT_CONTEXT.store(context as u8, Ordering::SeqCst);
        DoryContextGuard {
            previous_context: previous,
        }
    }

    /// Get the current Dory matrix layout
    pub fn get_layout() -> DoryLayout {
        CURRENT_LAYOUT.load(Ordering::SeqCst).into()
    }

    /// Set the Dory matrix layout
    ///
    /// This should be called once at initialization time, before any Dory operations.
    /// The layout determines how polynomial coefficients are mapped to matrix positions:
    /// - `RowMajor`: coeff[i] -> row = i / num_cols, col = i % num_cols (default)
    /// - `ColumnMajor`: coeff[i] -> row = i % num_rows, col = i / num_rows
    pub fn set_layout(layout: DoryLayout) {
        CURRENT_LAYOUT.store(layout as u8, Ordering::SeqCst);
    }

    /// Compute coefficient index from row and column using the current layout
    #[inline]
    pub fn coeff_index(row: usize, col: usize) -> usize {
        let num_rows = Self::get_max_num_rows();
        let num_cols = Self::get_num_columns();
        Self::get_layout().position_to_index(row, col, num_rows, num_cols)
    }

    /// Compute row and column from coefficient index using the current layout
    #[inline]
    pub fn row_col_from_index(coeff_idx: usize) -> (usize, usize) {
        let num_rows = Self::get_max_num_rows();
        let num_cols = Self::get_num_columns();
        Self::get_layout().index_to_position(coeff_idx, num_rows, num_cols)
    }

    fn set_max_num_rows_for_context(max_num_rows: usize, context: DoryContext) {
        #[allow(static_mut_refs)]
        unsafe {
            match context {
                DoryContext::Main => {
                    let _ = MAX_NUM_ROWS.set(max_num_rows);
                }
                DoryContext::TrustedAdvice => {
                    let _ = TRUSTED_ADVICE_MAX_NUM_ROWS.set(max_num_rows);
                }
                DoryContext::UntrustedAdvice => {
                    let _ = UNTRUSTED_ADVICE_MAX_NUM_ROWS.set(max_num_rows);
                }
            }
        }
    }

    pub fn get_max_num_rows() -> usize {
        let context = Self::current_context();
        #[allow(static_mut_refs)]
        unsafe {
            match context {
                DoryContext::Main => *MAX_NUM_ROWS.get().expect("max_num_rows not initialized"),
                DoryContext::TrustedAdvice => *TRUSTED_ADVICE_MAX_NUM_ROWS
                    .get()
                    .expect("trusted_advice max_num_rows not initialized"),
                DoryContext::UntrustedAdvice => *UNTRUSTED_ADVICE_MAX_NUM_ROWS
                    .get()
                    .expect("untrusted_advice max_num_rows not initialized"),
            }
        }
    }

    fn set_num_columns_for_context(num_columns: usize, context: DoryContext) {
        #[allow(static_mut_refs)]
        unsafe {
            match context {
                DoryContext::Main => {
                    let _ = NUM_COLUMNS.set(num_columns);
                }
                DoryContext::TrustedAdvice => {
                    let _ = TRUSTED_ADVICE_NUM_COLUMNS.set(num_columns);
                }
                DoryContext::UntrustedAdvice => {
                    let _ = UNTRUSTED_ADVICE_NUM_COLUMNS.set(num_columns);
                }
            }
        }
    }

    pub fn get_num_columns() -> usize {
        let context = Self::current_context();
        #[allow(static_mut_refs)]
        unsafe {
            match context {
                DoryContext::Main => *NUM_COLUMNS.get().expect("num_columns not initialized"),
                DoryContext::TrustedAdvice => *TRUSTED_ADVICE_NUM_COLUMNS
                    .get()
                    .expect("trusted_advice num_columns not initialized"),
                DoryContext::UntrustedAdvice => *UNTRUSTED_ADVICE_NUM_COLUMNS
                    .get()
                    .expect("untrusted_advice num_columns not initialized"),
            }
        }
    }

    fn set_T_for_context(t: usize, context: DoryContext) {
        #[allow(static_mut_refs)]
        unsafe {
            match context {
                DoryContext::Main => {
                    let _ = GLOBAL_T.set(t);
                }
                DoryContext::TrustedAdvice => {
                    let _ = TRUSTED_ADVICE_T.set(t);
                }
                DoryContext::UntrustedAdvice => {
                    let _ = UNTRUSTED_ADVICE_T.set(t);
                }
            }
        }
    }

    pub fn get_T() -> usize {
        let context = Self::current_context();
        #[allow(static_mut_refs)]
        unsafe {
            match context {
                DoryContext::Main => *GLOBAL_T.get().expect("t not initialized"),
                DoryContext::TrustedAdvice => *TRUSTED_ADVICE_T
                    .get()
                    .expect("trusted_advice t not initialized"),
                DoryContext::UntrustedAdvice => *UNTRUSTED_ADVICE_T
                    .get()
                    .expect("untrusted_advice t not initialized"),
            }
        }
    }

    /// Calculate optimal matrix dimensions for given K and T
    fn calculate_dimensions(K: usize, T: usize) -> (usize, usize, usize) {
        let total_size = K * T;
        let total_vars = total_size.log_2();

        let (num_columns, num_rows) = if total_vars % 2 == 0 {
            // Even total vars: square matrix
            let side = 1 << (total_vars / 2);
            (side, side)
        } else {
            // Odd total vars: almost square (columns = 2*rows)
            let sigma = total_vars.div_ceil(2);
            let nu = total_vars - sigma;
            (1 << sigma, 1 << nu)
        };

        (num_columns, num_rows, T)
    }

    /// Initialize the globals for the main Dory matrix
    ///
    /// # Arguments
    /// * `K` - Maximum address space size (K in OneHot polynomials)
    /// * `T` - Maximum trace length (cycle count)
    ///
    /// The matrix dimensions are calculated to minimize padding:
    /// - If log2(K*T) is even: creates a square matrix
    /// - If log2(K*T) is odd: creates an almost-square matrix (columns = 2*rows)
    pub fn initialize(K: usize, T: usize) -> Option<()> {
        let (num_columns, num_rows, t) = Self::calculate_dimensions(K, T);
        Self::set_num_columns_for_context(num_columns, DoryContext::Main);
        Self::set_T_for_context(t, DoryContext::Main);
        Self::set_max_num_rows_for_context(num_rows, DoryContext::Main);
        Some(())
    }

    /// Initialize the globals for trusted advice commitments
    pub fn initialize_trusted_advice(K: usize, T: usize) -> Option<()> {
        let (num_columns, num_rows, t) = Self::calculate_dimensions(K, T);
        Self::set_num_columns_for_context(num_columns, DoryContext::TrustedAdvice);
        Self::set_T_for_context(t, DoryContext::TrustedAdvice);
        Self::set_max_num_rows_for_context(num_rows, DoryContext::TrustedAdvice);
        Some(())
    }

    /// Initialize the globals for untrusted advice commitments
    pub fn initialize_untrusted_advice(K: usize, T: usize) -> Option<()> {
        let (num_columns, num_rows, t) = Self::calculate_dimensions(K, T);
        Self::set_num_columns_for_context(num_columns, DoryContext::UntrustedAdvice);
        Self::set_T_for_context(t, DoryContext::UntrustedAdvice);
        Self::set_max_num_rows_for_context(num_rows, DoryContext::UntrustedAdvice);
        Some(())
    }

    /// Reset global state
    #[cfg(test)]
    pub fn reset() {
        #[allow(static_mut_refs)]
        unsafe {
            // Reset main globals
            let _ = GLOBAL_T.take();
            let _ = MAX_NUM_ROWS.take();
            let _ = NUM_COLUMNS.take();

            // Reset layout to default (RowMajor)
            CURRENT_LAYOUT.store(0, Ordering::SeqCst);

            // Reset trusted advice globals
            let _ = TRUSTED_ADVICE_T.take();
            let _ = TRUSTED_ADVICE_MAX_NUM_ROWS.take();
            let _ = TRUSTED_ADVICE_NUM_COLUMNS.take();

            // Reset untrusted advice globals
            let _ = UNTRUSTED_ADVICE_T.take();
            let _ = UNTRUSTED_ADVICE_MAX_NUM_ROWS.take();
            let _ = UNTRUSTED_ADVICE_NUM_COLUMNS.take();
        }

        // Reset context to Main
        CURRENT_CONTEXT.store(0, Ordering::SeqCst);
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
