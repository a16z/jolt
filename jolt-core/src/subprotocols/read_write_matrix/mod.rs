//! Sparse matrix representations for read/write checking sumchecks.
//!
//! This module provides efficient data structures for representing the `ra(k, j)` and
//! `Val(k, j)` polynomials used in memory read/write checking sumchecks. These polynomials
//! are conceptually K×T matrices (addresses × cycles) but are far too large to store
//! explicitly. Instead, we use sparse representations that store only non-zero entries.
//!
//! # Three-Phase Sumcheck Structure
//!
//! Both RAM and Registers use a flexible 3-phase sumcheck:
//!
//! - **Phase 1 (Cycle Binding)**: Bind cycle variables using cycle-major sparse representation.
//! - **Phase 2 (Address Binding)**: Bind address variables using address-major sparse representation.
//! - **Phase 3 (Materialized)**: Standard sumcheck on dense polynomials.
//!
//! The phase boundaries are configurable via `phase1_num_rounds` and `phase2_num_rounds`.
//!
//! # Two Representations
//!
//! - **Cycle-Major**: Entries sorted by `(row, col)` i.e. cycle-major order.
//!   Uses Array-of-Structs layout. Used when binding *cycle variables* first.
//!
//! - **Address-Major**: Entries sorted by `(col, row)` i.e. address-major order.
//!   Uses Struct-of-Arrays layout with dense `val_init`/`val_final` arrays.
//!   Used when binding *address variables* first.
//!
//! The choice of representation affects cache performance during sumcheck operations.
//!
//! # Organization
//!
//! - [`ram`]: RAM-specific implementations with a single `ra` coefficient per entry.
//! - [`registers`]: Register-specific implementations with multiple access types
//!   (`rs1_ra`, `rs2_ra`, `rd_wa`) per entry.
//!
//! # Generic Column Index
//!
//! Both representations are generic over the column index type `I: ColIndex`:
//! - `usize` for RAM (large address space, up to ~2^20 addresses)
//! - `u8` for registers (128 registers)

use crate::field::JoltField;
use crate::poly::opening_proof::{OpeningPoint, BIG_ENDIAN};
use num::Integer;
use std::fmt::Debug;

pub mod ram;
pub mod registers;

// Re-export the main types for convenience
pub use ram::{ReadWriteEntry, ReadWriteMatrixAddressMajor, ReadWriteMatrixCycleMajor};
pub use registers::{
    RegisterEntry, RegisterMatrixAddressMajor, RegisterMatrixAddressMajorOptimized,
    RegisterMatrixCycleMajor,
};

/// Trait for column index types used in sparse read-write matrices.
///
/// This allows the same matrix implementations to be used for both:
/// - RAM (column = memory address, using `usize`)
/// - Registers (column = register index, using `u8`)
///
/// The trait requires numeric operations needed for column-pair merging during
/// address variable binding.
pub trait ColIndex:
    Copy + Clone + Ord + Eq + Debug + Integer + Send + Sync + Default + 'static
{
    /// Convert to `usize` for array indexing.
    fn to_usize(self) -> usize;

    /// Create from `usize`. May truncate if the value doesn't fit.
    fn from_usize(val: usize) -> Self;
}

impl ColIndex for usize {
    #[inline(always)]
    fn to_usize(self) -> usize {
        self
    }

    #[inline(always)]
    fn from_usize(val: usize) -> Self {
        val
    }
}

impl ColIndex for u8 {
    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }

    #[inline(always)]
    fn from_usize(val: usize) -> Self {
        val as u8
    }
}

// ============================================================================
// Shared Utilities for 3-Phase Read-Write Checking Sumchecks
// ============================================================================

/// Phase of the 3-phase read-write checking sumcheck.
///
/// Both RAM and Registers use this same phase structure, though the number
/// of rounds in each phase is configurable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadWritePhase {
    /// Phase 1: Binding cycle variables using sparse cycle-major representation.
    CycleBinding,
    /// Phase 2: Binding address variables using sparse address-major representation.
    AddressBinding,
    /// Phase 3: Standard sumcheck on materialized dense polynomials.
    Materialized,
}

impl ReadWritePhase {
    /// Determine the phase for a given sumcheck round.
    ///
    /// # Arguments
    /// - `round`: The current round index (0-based).
    /// - `phase1_rounds`: Number of rounds in Phase 1 (cycle binding).
    /// - `phase2_rounds`: Number of rounds in Phase 2 (address binding).
    #[inline]
    pub fn for_round(round: usize, phase1_rounds: usize, phase2_rounds: usize) -> Self {
        if round < phase1_rounds {
            Self::CycleBinding
        } else if round < phase1_rounds + phase2_rounds {
            Self::AddressBinding
        } else {
            Self::Materialized
        }
    }
}

/// Compute the final opening point from sumcheck challenges.
///
/// This function implements the shared logic for both RAM and Registers:
/// - Phase 1 challenges are bound low-to-high for cycle variables
/// - Phase 2 challenges are bound low-to-high for address variables
/// - Phase 3 binds remaining cycle variables, then remaining address variables
///
/// The output format is `[r_address..., r_cycle...]` in big-endian order,
/// where address variables are "higher" than cycle variables.
///
/// # Arguments
/// - `sumcheck_challenges`: All challenges from the sumcheck, in order.
/// - `log_k`: Number of address variables (log₂(K)).
/// - `log_t`: Number of cycle variables (log₂(T)).
/// - `phase1_rounds`: Number of cycle variables bound in Phase 1.
/// - `phase2_rounds`: Number of address variables bound in Phase 2.
pub fn compute_rw_opening_point<F: JoltField>(
    sumcheck_challenges: &[F::Challenge],
    log_k: usize,
    log_t: usize,
    phase1_rounds: usize,
    phase2_rounds: usize,
) -> OpeningPoint<BIG_ENDIAN, F> {
    debug_assert_eq!(
        sumcheck_challenges.len(),
        log_k + log_t,
        "Expected {} challenges, got {}",
        log_k + log_t,
        sumcheck_challenges.len()
    );

    // Split challenges by phase
    let (phase1_challenges, rest) = sumcheck_challenges.split_at(phase1_rounds);
    let (phase2_challenges, phase3_challenges) = rest.split_at(phase2_rounds);

    // Phase 3 binds remaining cycle vars first, then remaining address vars
    let remaining_cycle_vars = log_t - phase1_rounds;
    let (phase3_cycle, phase3_addr) = phase3_challenges.split_at(remaining_cycle_vars);

    // Reconstruct full r_cycle and r_address in big-endian order.
    // LowToHigh binding means challenges are in reverse order relative to big-endian.
    let r_cycle: Vec<_> = phase3_cycle
        .iter()
        .rev()
        .chain(phase1_challenges.iter().rev())
        .cloned()
        .collect();

    let r_address: Vec<_> = phase3_addr
        .iter()
        .rev()
        .chain(phase2_challenges.iter().rev())
        .cloned()
        .collect();

    // Final point: [r_address..., r_cycle...] (address bits are "higher")
    [r_address, r_cycle].concat().into()
}
