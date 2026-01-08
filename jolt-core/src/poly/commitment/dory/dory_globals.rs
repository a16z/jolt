//! Global state management for Dory parameters

use crate::utils::math::Math;
use dory::backends::arkworks::{init_cache, is_cached, ArkG1, ArkG2};
use std::sync::{
    atomic::{AtomicU8, Ordering},
    OnceLock,
};

/// Dory matrix layout for OneHot polynomials.
///
/// This enum controls how polynomial coefficients (indexed by address k and cycle t)
/// are mapped to matrix positions for Dory commitment.
///
/// For a OneHot polynomial with K addresses and T cycles:
/// - Total coefficients = K * T
/// - Matrix dimensions = sqrt(K * T) x sqrt(K * T) (approximately square)
///
/// The layout determines the mapping from (address, cycle) to matrix (row, col).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DoryLayout {
    /// Cycle-major layout (current main branch behavior).
    ///
    /// Coefficients are ordered by address first, then by cycle within each address:
    /// ```text
    /// Memory: [a0_t0, a0_t1, ..., a0_tT-1, a1_t0, a1_t1, ..., a1_tT-1, ...]
    ///          └──── address 0 cycles ────┘ └──── address 1 cycles ────┘
    ///
    /// global_index = address * T + cycle
    /// ```
    ///
    /// Matrix layout (K=4 addresses, T=4 cycles):
    /// ```text
    ///            col0    col1    col2    col3
    ///      ┌────────┬────────┬────────┬────────┐
    /// row0 │ a0,t0  │ a0,t1  │ a0,t2  │ a0,t3  │  ← All of address 0
    ///      ├────────┼────────┼────────┼────────┤
    /// row1 │ a1,t0  │ a1,t1  │ a1,t2  │ a1,t3  │  ← All of address 1
    ///      ├────────┼────────┼────────┼────────┤
    /// row2 │ a2,t0  │ a2,t1  │ a2,t2  │ a2,t3  │  ← All of address 2
    ///      ├────────┼────────┼────────┼────────┤
    /// row3 │ a3,t0  │ a3,t1  │ a3,t2  │ a3,t3  │  ← All of address 3
    ///      └────────┴────────┴────────┴────────┘
    /// ```
    #[default]
    CycleMajor,

    /// Address-major layout (from PR #1038, GPU-compatible).
    ///
    /// Coefficients are ordered by cycle first, then by address within each cycle:
    /// ```text
    /// Memory: [t0_a0, t0_a1, ..., t0_aK-1, t1_a0, t1_a1, ..., t1_aK-1, ...]
    ///          └──── cycle 0 addresses ───┘ └──── cycle 1 addresses ───┘
    ///
    /// global_index = cycle * K + address
    /// ```
    ///
    /// Matrix layout (K=4 addresses, T=4 cycles):
    /// ```text
    ///            col0    col1    col2    col3
    ///      ┌────────┬────────┬────────┬────────┐
    /// row0 │ a0,t0  │ a1,t0  │ a2,t0  │ a3,t0  │  ← All of cycle 0
    ///      ├────────┼────────┼────────┼────────┤
    /// row1 │ a0,t1  │ a1,t1  │ a2,t1  │ a3,t1  │  ← All of cycle 1
    ///      ├────────┼────────┼────────┼────────┤
    /// row2 │ a0,t2  │ a1,t2  │ a2,t2  │ a3,t2  │  ← All of cycle 2
    ///      ├────────┼────────┼────────┼────────┤
    /// row3 │ a0,t3  │ a1,t3  │ a2,t3  │ a3,t3  │  ← All of cycle 3
    ///      └────────┴────────┴────────┴────────┘
    /// ```
    ///
    /// This layout is optimized for OneHot polynomials where each cycle has exactly
    /// one nonzero entry (at the accessed address). Each row commitment becomes a
    /// single basis element lookup rather than an MSM.
    AddressMajor,
}

impl DoryLayout {
    /// Convert a (address, cycle) pair to a linear coefficient index.
    ///
    /// # Arguments
    /// * `address` - The address index (0 to K-1)
    /// * `cycle` - The cycle index (0 to T-1)
    /// * `K` - Total number of addresses
    /// * `T` - Total number of cycles
    #[inline]
    pub fn address_cycle_to_index(&self, address: usize, cycle: usize, K: usize, T: usize) -> usize {
        match self {
            DoryLayout::CycleMajor => address * T + cycle,
            DoryLayout::AddressMajor => cycle * K + address,
        }
    }

    /// Convert a linear coefficient index to a (address, cycle) pair.
    ///
    /// # Arguments
    /// * `index` - The linear coefficient index
    /// * `K` - Total number of addresses
    /// * `T` - Total number of cycles
    #[inline]
    pub fn index_to_address_cycle(&self, index: usize, K: usize, T: usize) -> (usize, usize) {
        match self {
            DoryLayout::CycleMajor => {
                let address = index / T;
                let cycle = index % T;
                (address, cycle)
            }
            DoryLayout::AddressMajor => {
                let cycle = index / K;
                let address = index % K;
                (address, cycle)
            }
        }
    }

    /// Convert a linear coefficient index to a (row, col) matrix position.
    ///
    /// For dense polynomials, this is standard row-major indexing regardless of layout.
    /// The layout distinction primarily affects OneHot polynomial interpretation.
    #[inline]
    pub fn index_to_position(&self, coeff_idx: usize, _num_rows: usize, num_cols: usize) -> (usize, usize) {
        // For dense polynomials, both layouts use row-major matrix storage
        let row = coeff_idx / num_cols;
        let col = coeff_idx % num_cols;
        (row, col)
    }

    /// Convert a (row, col) matrix position to a linear coefficient index.
    ///
    /// For dense polynomials, this is standard row-major indexing regardless of layout.
    #[inline]
    pub fn position_to_index(&self, row: usize, col: usize, _num_rows: usize, num_cols: usize) -> usize {
        // For dense polynomials, both layouts use row-major matrix storage
        row * num_cols + col
    }
}

impl From<u8> for DoryLayout {
    fn from(value: u8) -> Self {
        match value {
            0 => DoryLayout::CycleMajor,
            1 => DoryLayout::AddressMajor,
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

// Layout tracking: 0=CycleMajor, 1=AddressMajor
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
    /// The layout determines how OneHot polynomial coefficients are organized:
    /// - `CycleMajor`: Coefficients ordered by address, then cycle (current main branch)
    /// - `AddressMajor`: Coefficients ordered by cycle, then address (GPU-compatible, PR #1038)
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

            // Reset layout to default (CycleMajor)
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
