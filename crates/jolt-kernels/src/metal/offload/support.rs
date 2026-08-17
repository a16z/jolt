//! Device-handoff accessors on the shared optimized support types.

use jolt_field::Field;

use crate::optimized::support::SplitLt;

impl<F: Field> SplitLt<F> {
    pub(crate) fn split_lo(&self) -> Option<&[F]> {
        match self {
            Self::Split { lt_lo, .. } => Some(lt_lo),
            Self::Dense(_) => None,
        }
    }

    pub(crate) fn current_len(&self) -> usize {
        match self {
            Self::Split { lt_lo, lt_hi, .. } => lt_lo.len() * lt_hi.len(),
            Self::Dense(table) => table.len(),
        }
    }
}
