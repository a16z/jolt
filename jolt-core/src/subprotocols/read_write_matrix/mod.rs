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
//!   Used when binding *cycle variables* first.
//!
//! - [`ReadWriteMatrixAddressMajor`]: Entries sorted by `(col, row)` i.e. address-major order.
//!   Used when binding *address variables* first.
//!
//! The choice of representation affects cache performance during sumcheck operations.

mod address_major;
mod cycle_major;

pub use address_major::ReadWriteMatrixAddressMajor;
pub use cycle_major::{ReadWriteEntry, ReadWriteMatrixCycleMajor};
