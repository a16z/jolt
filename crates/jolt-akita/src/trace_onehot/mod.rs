//! Streaming kernels over the native trace commitment batch.

#![expect(
    clippy::indexing_slicing,
    reason = "hot kernels index geometry validated by TraceOneHotColumn and their plans"
)]

use crate::AkitaField;
use jolt_field::WithCommitAccumulator;

const NO_SELECTED_ROW: u8 = 0;
const MAX_WIDE_ACCUMULATIONS: usize = AkitaField::MAX_COMMIT_ACCUMULATIONS;
const TASKS_PER_RAYON_WORKER: usize = 4;
const ROTATED_CHALLENGE_TABLE_BUDGET: usize = 1 << 28;
const DECOMPOSE_POSITION_WORKING_SET_TARGET: usize = 1 << 21;
const K256_ROW_BATCH: usize = 1 << 13;
const _: () = assert!(K256_ROW_BATCH <= i16::MAX as usize);

mod commit;
mod decomposition;
mod digit_windows;
mod kernels;
mod source;
mod traversal;

#[cfg(test)]
mod tests;

pub use source::{no_selected_row, TraceOneHotColumn, TraceOneHotRows};

#[cfg(test)]
use decomposition::{decompose_fold_columns_with_mode, DecomposeRotationMode};
#[cfg(test)]
use traversal::{
    coefficient_packing_partials_columns, visit_segment_ring_range, AkitaWideRing,
    DeferredFp128Ring,
};
