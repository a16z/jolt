//! Sparse matrix representations for read/write checking sumchecks.
//!
//! This module provides efficient data structures for representing the `ra(k, j)` and
//! `Val(k, j)` polynomials used in memory read/write checking sumchecks. These polynomials
//! are conceptually K×T matrices (addresses × cycles) but are far too large to store
//! explicitly. Instead, we use sparse representations that store only non-zero entries.
//!
//! # Two Representations
//!
//! - [`ReadWriteMatrixCycleMajor`]: Entries sorted by `(row, col)` i.e. cycle-major order.
//!   Uses Array-of-Structs layout. Used when binding *cycle variables* first.
//!
//! - [`ReadWriteMatrixAddressMajor`]: Entries sorted by `(col, row)` i.e. address-major order.
//!   Uses Struct-of-Arrays layout with dense `val_init`/`val_final` arrays.
//!   Used when binding *address variables* first.
//!
//! The choice of representation affects cache performance during sumcheck operations.
//!
//! # Generic Column Index
//!
//! Both representations are generic over the column index type `I: ColIndex`:
//! - `usize` for RAM (large address space, up to ~2^20 addresses)
//! - `u8` for registers (128 registers)

use num::Integer;
use std::fmt::Debug;

mod address_major;
mod cycle_major;

pub use address_major::ReadWriteMatrixAddressMajor;
pub use cycle_major::{ReadWriteEntry, ReadWriteMatrixCycleMajor};

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
