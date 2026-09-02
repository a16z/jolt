//! Streaming kernels for Jolt's prefix-packed trace one-hot polynomial.
//!
//! A trace row gives the selected one-hot row for each column. Byte zero normally
//! denotes no stored coefficient; a per-row column mask distinguishes selected row
//! zero for families such as RAM. Jolt's kernels consume that row-major source
//! directly and lay the columns out as consecutive `K * T` segments inside one
//! physical polynomial. Padding selector slots are zero.

#![expect(
    clippy::indexing_slicing,
    reason = "hot kernels index geometry validated by TracePackedOneHot and their plans"
)]

const NO_SELECTED_ROW: u8 = 0;
const MAX_WIDE_ACCUMULATIONS: usize = 1 << 15;
const TASKS_PER_RAYON_WORKER: usize = 4;
const ROTATED_CHALLENGE_TABLE_BUDGET: usize = 1 << 28;
const DECOMPOSE_POSITION_WORKING_SET_TARGET: usize = 1 << 21;
const SHARED_SHIFT_MIN_COLUMNS: u8 = 3;
const K256_ROW_BATCH: usize = 1 << 13;
const _: () = assert!(K256_ROW_BATCH <= i16::MAX as usize);

mod commit;
mod decomposition;
mod grouped;
mod kernels;
mod opening;
mod source;
mod traversal;

#[cfg(test)]
mod tests;

pub(crate) use grouped::GroupedRootSource;
pub use source::{no_selected_row, TraceOneHotRows, TracePackedOneHot};

#[cfg(test)]
use decomposition::{
    decompose_fold_packed_with_mode, prepare_rotations, DecomposeRotationMode, PreparedRotations,
};
#[cfg(test)]
use traversal::{
    coefficient_packing_partials_packed, visit_segment_ring_range, AkitaWideRing,
    DeferredFp128Ring, K16FourRowShiftGroups,
};
