//! The curve back end: BlindFold over committed pads.
//!
//! The residual relation is lowered to relaxed R1CS whose witness rows begin
//! with the pad rows, committed before the first challenge. The prover folds
//! the committed instance with a random satisfying instance (NovaBlindFold,
//! HyperNova §7) and proves the folded instance with Spartan's outer and inner
//! sumchecks; the folded witness is uniform, so the tail reveals nothing.

mod backend;
mod error;
mod protocol;
mod prove;
mod relaxed;
mod verify;
mod wire;

pub use backend::{BlindFold, BlindFoldPadOpening, BlindFoldResidualWitness};
pub use error::{ProverError, RelaxedError, VerificationError};
pub use protocol::{
    pad_row_count, BlindFoldDimensions, BlindFoldProtocol, FinalOpeningWitnessCoordinates,
    RowDimensions, WitnessCoordinate, WitnessRowLayout,
};
pub use prove::{
    prove, prove_with_row_committer, BlindFoldRowCommitter, BlindFoldWitness,
    DirectBlindFoldRowCommitter,
};
pub use relaxed::{RelaxedInstance, RelaxedWitness};
