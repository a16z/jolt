//! RAM read-write checking sparse matrix representations.
//!
//! This module contains the cycle-major and address-major representations
//! for RAM memory operations.

mod address_major;
mod cycle_major;

pub use address_major::ReadWriteMatrixAddressMajor;
pub use cycle_major::{ReadWriteEntry, ReadWriteMatrixCycleMajor};

