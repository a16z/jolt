//! Register read-write checking sparse matrix representations.
//!
//! This module contains the cycle-major and address-major representations
//! for register operations.

mod address_major;
mod address_major_optimized;
mod cycle_major;

pub use address_major::RegisterMatrixAddressMajor;
pub use address_major_optimized::RegisterMatrixAddressMajorOptimized;
pub use cycle_major::{RegisterEntry, RegisterMatrixCycleMajor};

