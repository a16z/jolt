//! Global state management for Dory parameters

use crate::utils::math::Math;
use allocative::Allocative;
use dory::backends::arkworks::{init_cache, ArkG1, ArkG2};
use std::sync::{
    atomic::{AtomicU8, AtomicUsize, Ordering},
    RwLock,
};

/// Dory matrix layout for OneHot polynomials.
///
/// This enum controls how polynomial coefficients (indexed by address k and cycle t)
/// are mapped to matrix positions for Dory commitment.
///
/// For a OneHot polynomial with K addresses and T cycles:
/// - Total coefficients = K * T
/// - The Dory matrix shape is chosen by [`DoryGlobals::calculate_dimensions`] as either:
///   - square: `num_rows == num_cols` when `log2(K*T)` is even, or
///   - almost-square: `num_cols == 2*num_rows` when `log2(K*T)` is odd.
///
/// The layout determines the mapping from (address, cycle) to matrix (row, col).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Allocative)]
pub enum DoryLayout {
    /// Cycle-major layout
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

    /// Address-major layout
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
    AddressMajor,
}

impl DoryLayout {
    /// Convert a (address, cycle) pair to a coefficient index.
    ///
    /// # Arguments
    /// * `address` - The address index (0 to K-1)
    /// * `cycle` - The cycle index (0 to T-1)
    /// * `K` - Total number of addresses
    /// * `T` - Total number of cycles
    pub fn address_cycle_to_index(
        &self,
        address: usize,
        cycle: usize,
        K: usize,
        T: usize,
    ) -> usize {
        match self {
            DoryLayout::CycleMajor => address * T + cycle,
            DoryLayout::AddressMajor => cycle * K + address,
        }
    }

    /// Convert a coefficient index to a (address, cycle) pair.
    ///
    /// # Arguments
    /// * `index` - The linear coefficient index
    /// * `K` - Total number of addresses
    /// * `T` - Total number of cycles
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

impl From<DoryLayout> for u8 {
    fn from(layout: DoryLayout) -> Self {
        match layout {
            DoryLayout::CycleMajor => 0,
            DoryLayout::AddressMajor => 1,
        }
    }
}

// Main polynomial globals
static GLOBAL_T: RwLock<Option<usize>> = RwLock::new(None);
static MAIN_K_CHUNK: RwLock<Option<usize>> = RwLock::new(None);
static MAX_NUM_ROWS: RwLock<Option<usize>> = RwLock::new(None);
static NUM_COLUMNS: RwLock<Option<usize>> = RwLock::new(None);

// Trusted advice globals
static TRUSTED_ADVICE_T: RwLock<Option<usize>> = RwLock::new(None);
static TRUSTED_ADVICE_MAX_NUM_ROWS: RwLock<Option<usize>> = RwLock::new(None);
static TRUSTED_ADVICE_NUM_COLUMNS: RwLock<Option<usize>> = RwLock::new(None);

// Untrusted advice globals
static UNTRUSTED_ADVICE_T: RwLock<Option<usize>> = RwLock::new(None);
static UNTRUSTED_ADVICE_MAX_NUM_ROWS: RwLock<Option<usize>> = RwLock::new(None);
static UNTRUSTED_ADVICE_NUM_COLUMNS: RwLock<Option<usize>> = RwLock::new(None);

// Context tracking: 0=Main, 1=TrustedAdvice, 2=UntrustedAdvice
static CURRENT_CONTEXT: AtomicU8 = AtomicU8::new(0);

// Layout tracking: 0=CycleMajor, 1=AddressMajor
static CURRENT_LAYOUT: AtomicU8 = AtomicU8::new(0);
// Largest Main log-embedding needed for precommitted/embed calculations.
static MAIN_LOG_EMBEDDING: AtomicUsize = AtomicUsize::new(0);

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
    /// Split `total_vars` into a *balanced* pair `(sigma, nu)` where:
    /// - **sigma** is the number of **column** variables
    /// - **nu** is the number of **row** variables
    ///
    /// Dory matrices are conceptually shaped as `2^nu` rows × `2^sigma` columns (row-major).
    /// We use the balanced policy `sigma = ceil(total_vars / 2)` and `nu = total_vars - sigma`.
    #[inline]
    pub fn balanced_sigma_nu(total_vars: usize) -> (usize, usize) {
        let sigma = total_vars.div_ceil(2);
        let nu = total_vars - sigma;
        (sigma, nu)
    }

    /// Convenience helper for the main Dory matrix where `total_vars = log_k_chunk + log_t`.
    #[inline]
    pub fn main_sigma_nu(log_k_chunk: usize, log_t: usize) -> (usize, usize) {
        Self::balanced_sigma_nu(log_k_chunk + log_t)
    }

    /// Computes balanced `(sigma, nu)` dimensions directly from a max advice byte budget.
    ///
    /// - `max_advice_size_bytes` is interpreted as bytes of 64-bit words.
    /// - Rounds word count up to the next power of two (minimum 1) and computes log2 as `advice_vars`.
    /// - Returns `(sigma, nu)` where `sigma = ⌈advice_vars/2⌉` and `nu = advice_vars - sigma`.
    #[inline]
    pub fn advice_sigma_nu_from_max_bytes(max_advice_size_bytes: usize) -> (usize, usize) {
        let words = max_advice_size_bytes / 8;
        let len = words.next_power_of_two().max(1);
        let advice_vars = len.log_2();
        Self::balanced_sigma_nu(advice_vars)
    }

    /// How many row variables of the *cycle* segment exist in the unified point:
    /// `row_cycle_len = max(0, log_t - sigma_main)`.
    #[inline]
    pub fn cycle_row_len(log_t: usize, sigma_main: usize) -> usize {
        log_t.saturating_sub(sigma_main)
    }

    #[inline]
    pub fn get_main_log_embedding() -> usize {
        let stored = MAIN_LOG_EMBEDDING.load(Ordering::SeqCst);
        if stored > 0 {
            stored
        } else {
            let main_cols = Self::configured_main_num_columns();
            let main_rows = *MAX_NUM_ROWS
                .read()
                .unwrap()
                .as_ref()
                .expect("main max_num_rows not initialized");
            main_cols.log_2() + main_rows.log_2()
        }
    }

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

    /// Set the Dory matrix layout directly (test-only).
    ///
    /// In production code, prefer passing the layout to `initialize_context` instead.
    #[cfg(test)]
    pub fn set_layout(layout: DoryLayout) {
        CURRENT_LAYOUT.store(layout as u8, Ordering::SeqCst);
    }

    /// Returns the configured Dory matrix shape `(num_rows, num_cols)` for the current context.
    pub fn matrix_shape() -> (usize, usize) {
        (Self::get_max_num_rows(), Self::get_num_columns())
    }

    #[inline]
    pub(crate) fn main_k() -> usize {
        *MAIN_K_CHUNK
            .read()
            .unwrap()
            .as_ref()
            .expect("main k not initialized")
    }

    #[inline]
    pub(crate) fn main_t() -> usize {
        *GLOBAL_T
            .read()
            .unwrap()
            .as_ref()
            .expect("main t not initialized")
    }

    #[inline]
    pub(crate) fn configured_main_num_columns() -> usize {
        *NUM_COLUMNS
            .read()
            .unwrap()
            .as_ref()
            .expect("main num_columns not initialized")
    }

    #[inline]
    fn main_embedding_extra_vars() -> usize {
        let main_total_vars = Self::main_k().log_2() + Self::get_T().log_2();
        Self::get_main_log_embedding().saturating_sub(main_total_vars)
    }

    /// Column stride for one-hot embeddings in the current layout/context.
    pub fn one_hot_stride() -> usize {
        if Self::current_context() != DoryContext::Main
            || Self::get_layout() != DoryLayout::AddressMajor
        {
            return 1;
        }
        1usize << Self::main_embedding_extra_vars()
    }

    /// Column stride for dense trace-domain embeddings in the current layout/context.
    pub fn dense_stride() -> usize {
        if Self::current_context() != DoryContext::Main
            || Self::get_layout() != DoryLayout::AddressMajor
        {
            return 1;
        }
        let dense_stride_log = Self::main_embedding_extra_vars() + Self::main_k().log_2();
        1usize << dense_stride_log
    }

    /// Returns the embedded cycle-domain size for the current Dory matrix.
    pub fn get_embedded_t() -> usize {
        let context = Self::current_context();
        if context != DoryContext::Main {
            return Self::get_T();
        }

        let k = Self::main_k();
        let num_rows = Self::get_max_num_rows();
        let num_cols = Self::get_num_columns();
        let total = num_rows * num_cols;
        debug_assert_eq!(
            total % k,
            0,
            "Invalid Main DoryGlobals: num_rows*num_cols must be divisible by K"
        );
        total / k
    }

    /// Returns the "K" used to initialize the *main* Dory matrix for OneHot polynomials.
    pub fn k_from_matrix_shape() -> usize {
        if Self::current_context() == DoryContext::Main {
            return Self::main_k();
        }
        let (num_rows, num_cols) = Self::matrix_shape();
        let t = Self::get_T();
        debug_assert_eq!(
            (num_rows * num_cols) % t,
            0,
            "Invalid DoryGlobals: num_rows*num_cols must be divisible by T"
        );
        (num_rows * num_cols) / t
    }

    fn set_max_num_rows_for_context(max_num_rows: usize, context: DoryContext) {
        match context {
            DoryContext::Main => {
                *MAX_NUM_ROWS.write().unwrap() = Some(max_num_rows);
            }
            DoryContext::TrustedAdvice => {
                *TRUSTED_ADVICE_MAX_NUM_ROWS.write().unwrap() = Some(max_num_rows);
            }
            DoryContext::UntrustedAdvice => {
                *UNTRUSTED_ADVICE_MAX_NUM_ROWS.write().unwrap() = Some(max_num_rows);
            }
        }
    }

    pub fn get_max_num_rows() -> usize {
        let context = Self::current_context();
        match context {
            DoryContext::Main => MAX_NUM_ROWS
                .read()
                .unwrap()
                .expect("max_num_rows not initialized"),
            DoryContext::TrustedAdvice => TRUSTED_ADVICE_MAX_NUM_ROWS
                .read()
                .unwrap()
                .expect("trusted_advice max_num_rows not initialized"),
            DoryContext::UntrustedAdvice => UNTRUSTED_ADVICE_MAX_NUM_ROWS
                .read()
                .unwrap()
                .expect("untrusted_advice max_num_rows not initialized"),
        }
    }

    fn set_num_columns_for_context(num_columns: usize, context: DoryContext) {
        match context {
            DoryContext::Main => {
                *NUM_COLUMNS.write().unwrap() = Some(num_columns);
            }
            DoryContext::TrustedAdvice => {
                *TRUSTED_ADVICE_NUM_COLUMNS.write().unwrap() = Some(num_columns);
            }
            DoryContext::UntrustedAdvice => {
                *UNTRUSTED_ADVICE_NUM_COLUMNS.write().unwrap() = Some(num_columns);
            }
        }
    }

    fn set_main_k(k: usize) {
        *MAIN_K_CHUNK.write().unwrap() = Some(k);
    }

    pub fn get_num_columns() -> usize {
        let context = Self::current_context();
        match context {
            DoryContext::Main => NUM_COLUMNS
                .read()
                .unwrap()
                .expect("num_columns not initialized"),
            DoryContext::TrustedAdvice => TRUSTED_ADVICE_NUM_COLUMNS
                .read()
                .unwrap()
                .expect("trusted_advice num_columns not initialized"),
            DoryContext::UntrustedAdvice => UNTRUSTED_ADVICE_NUM_COLUMNS
                .read()
                .unwrap()
                .expect("untrusted_advice num_columns not initialized"),
        }
    }

    fn set_T_for_context(t: usize, context: DoryContext) {
        match context {
            DoryContext::Main => {
                *GLOBAL_T.write().unwrap() = Some(t);
            }
            DoryContext::TrustedAdvice => {
                *TRUSTED_ADVICE_T.write().unwrap() = Some(t);
            }
            DoryContext::UntrustedAdvice => {
                *UNTRUSTED_ADVICE_T.write().unwrap() = Some(t);
            }
        }
    }

    pub fn get_T() -> usize {
        let context = Self::current_context();
        match context {
            DoryContext::Main => GLOBAL_T.read().unwrap().expect("t not initialized"),
            DoryContext::TrustedAdvice => TRUSTED_ADVICE_T
                .read()
                .unwrap()
                .expect("trusted_advice t not initialized"),
            DoryContext::UntrustedAdvice => UNTRUSTED_ADVICE_T
                .read()
                .unwrap()
                .expect("untrusted_advice t not initialized"),
        }
    }

    /// Calculate optimal matrix dimensions for given K and T
    fn calculate_dimensions(K: usize, T: usize) -> (usize, usize, usize) {
        let total_size = K * T;
        let total_vars = total_size.log_2();

        let (num_columns, num_rows) = if total_vars.is_multiple_of(2) {
            // Even total vars: square matrix
            let side = 1 << (total_vars / 2);
            (side, side)
        } else {
            // Odd total vars: almost square (columns = 2*rows)
            let (sigma, nu) = Self::balanced_sigma_nu(total_vars);
            (1 << sigma, 1 << nu)
        };

        (num_columns, num_rows, T)
    }

    fn initialize_context_common(
        K: usize,
        embedded_t: usize,
        stored_t: usize,
        context: DoryContext,
    ) -> Option<()> {
        let (num_columns, num_rows, _) = Self::calculate_dimensions(K, embedded_t);
        Self::set_num_columns_for_context(num_columns, context);
        Self::set_T_for_context(stored_t, context);
        Self::set_max_num_rows_for_context(num_rows, context);

        Some(())
    }

    /// Initialize the globals for a specific Dory context
    ///
    /// # Arguments
    /// * `K` - Maximum address space size (K in OneHot polynomials)
    /// * `T` - Maximum trace length (cycle count)
    /// * `context` - The Dory context to initialize (Main, TrustedAdvice, or UntrustedAdvice)
    /// * `layout` - Optional layout for the Dory matrix. Only applies to Main context.
    ///   If `Some(layout)`, sets the layout. If `None`, leaves the existing layout
    ///   unchanged (defaults to `CycleMajor` after `reset()`). Ignored for advice contexts.
    ///
    /// The matrix dimensions are calculated to minimize padding:
    /// - If log2(K*T) is even: creates a square matrix
    /// - If log2(K*T) is odd: creates an almost-square matrix (columns = 2*rows)
    pub fn initialize_context(
        K: usize,
        T: usize,
        context: DoryContext,
        layout: Option<DoryLayout>,
    ) -> Option<()> {
        if context == DoryContext::Main {
            return Self::initialize_main_with_log_embedding(K, T, K.log_2() + T.log_2(), layout);
        }
        Self::initialize_context_common(K, T, T, context)?;
        Some(())
    }

    /// Initialize Main context with execution `T` and explicit `main_log_embedding` for
    /// global precommitted geometry.
    pub fn initialize_main_with_log_embedding(
        K: usize,
        T: usize,
        matrix_total_vars: usize,
        layout: Option<DoryLayout>,
    ) -> Option<()> {
        let log_k = K.log_2();
        let embedded_t = 1usize << matrix_total_vars.saturating_sub(log_k);
        Self::initialize_context_common(K, embedded_t, T, DoryContext::Main)?;
        Self::set_main_k(K);
        if let Some(l) = layout {
            CURRENT_LAYOUT.store(l as u8, Ordering::SeqCst);
        }
        CURRENT_CONTEXT.store(DoryContext::Main as u8, Ordering::SeqCst);
        MAIN_LOG_EMBEDDING.store(matrix_total_vars, Ordering::SeqCst);
        Some(())
    }

    /// Reset global state
    #[cfg(test)]
    pub fn reset() {
        // Reset main globals
        *GLOBAL_T.write().unwrap() = None;
        *MAIN_K_CHUNK.write().unwrap() = None;
        *MAX_NUM_ROWS.write().unwrap() = None;
        *NUM_COLUMNS.write().unwrap() = None;

        // Reset layout to default (CycleMajor)
        CURRENT_LAYOUT.store(0, Ordering::SeqCst);

        // Reset trusted advice globals
        *TRUSTED_ADVICE_T.write().unwrap() = None;
        *TRUSTED_ADVICE_MAX_NUM_ROWS.write().unwrap() = None;
        *TRUSTED_ADVICE_NUM_COLUMNS.write().unwrap() = None;

        // Reset untrusted advice globals
        *UNTRUSTED_ADVICE_T.write().unwrap() = None;
        *UNTRUSTED_ADVICE_MAX_NUM_ROWS.write().unwrap() = None;
        *UNTRUSTED_ADVICE_NUM_COLUMNS.write().unwrap() = None;

        CURRENT_CONTEXT.store(0, Ordering::SeqCst);
        MAIN_LOG_EMBEDDING.store(0, Ordering::SeqCst);
    }

    /// Initialize the prepared point cache for faster pairing operations
    ///
    /// This should be called once after creating the prover setup to cache
    /// prepared versions of the G1 and G2 generators for ~20-30% speedup
    /// in repeated pairing operations.
    ///
    /// init_cache handles smart re-initialization internally:
    /// - If cache doesn't exist, creates it
    /// - If cache is too small, replaces with larger one
    /// - If cache is large enough, no-op (reuses existing)
    ///
    /// # Arguments
    /// * `g1_vec` - Vector of G1 generators from the prover setup
    /// * `g2_vec` - Vector of G2 generators from the prover setup
    pub fn init_prepared_cache(g1_vec: &[ArkG1], g2_vec: &[ArkG2]) {
        init_cache(g1_vec, g2_vec);
    }
}
