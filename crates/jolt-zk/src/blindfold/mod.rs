//! The curve back end: BlindFold over committed pads.
//!
//! The residual relation is lowered to relaxed R1CS whose witness rows begin
//! with the pad rows, committed before the first challenge. The prover folds
//! the committed instance with a random satisfying instance (NovaBlindFold,
//! HyperNova §7) and proves the folded instance with Spartan's outer and inner
//! sumchecks; the folded witness is uniform, so the tail reveals nothing.

#[cfg(feature = "blindfold")]
mod backend;
mod error;
#[cfg(feature = "blindfold")]
mod proof;
#[cfg(feature = "blindfold")]
mod protocol;
#[cfg(all(feature = "prover", feature = "blindfold"))]
mod prove;
#[cfg(feature = "blindfold")]
mod relaxed;
#[cfg(feature = "blindfold")]
mod verify;

#[cfg(feature = "blindfold")]
pub use backend::{BlindFold, BlindFoldPadOpening, BlindFoldResidualWitness};
pub use error::RelaxedError;
#[cfg(feature = "blindfold")]
pub use error::{ProverError, VerificationError};
#[cfg(feature = "blindfold")]
pub use proof::{BlindFoldProof, FoldedEvaluation, FoldingCommitments, OuterClaims};
#[cfg(feature = "blindfold")]
pub use protocol::{
    pad_row_count, BlindFoldDimensions, BlindFoldProtocol, FinalOpeningWitnessCoordinates,
    RowDimensions, WitnessCoordinate, WitnessRowLayout,
};
#[cfg(all(feature = "prover", feature = "blindfold"))]
pub use prove::{
    prove, prove_with_row_committer, BlindFoldRowCommitter, BlindFoldWitness,
    DirectBlindFoldRowCommitter,
};
#[cfg(feature = "blindfold")]
pub use relaxed::{RelaxedInstance, RelaxedWitness};
