//! Generic BlindFold claim, protocol, layout, and verifier-equation types.

mod builder;
mod error;
pub mod protocol;
mod prove;
pub mod r1cs;
mod relaxed;
mod statements;
mod verify;

pub use builder::{BlindFoldProtocolBuilder, BlindFoldStageBuilder};
pub use error::{Error, LayoutError, ProverError, RelaxedError, VerificationError};
pub use protocol::{
    BlindFoldDimensions, BlindFoldProtocol, FinalOpeningWitnessCoordinates, RowDimensions,
    WitnessCoordinate, WitnessRowLayout,
};
pub use prove::{
    prove, prove_with_row_committer, BlindFoldRowCommitter, BlindFoldWitness,
    DirectBlindFoldRowCommitter,
};
pub use relaxed::{RelaxedInstance, RelaxedWitness};
pub use statements::{
    BlindFoldStage, BlindFoldStatement, CommittedClaimRows, FinalOpeningBinding, OpeningAlias,
};
