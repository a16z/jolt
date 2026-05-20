//! Generic BlindFold claim, protocol, layout, and verifier-equation types.

mod builder;
mod error;
mod proof;
pub mod protocol;
pub mod r1cs;
mod relaxed;
mod statements;
mod verify;

pub use builder::{BlindFoldProtocolBuilder, BlindFoldStageBuilder};
pub use error::{Error, LayoutError, RelaxedError, VerificationError};
pub use proof::BlindFoldProof;
pub use protocol::{
    BlindFoldDimensions, BlindFoldProtocol, FinalOpeningWitnessCoordinates, RowDimensions,
    WitnessCoordinate, WitnessRowLayout,
};
pub use relaxed::{RelaxedInstance, RelaxedWitness};
pub use statements::{
    BlindFoldStage, BlindFoldStatement, CommittedClaimRows, FinalOpeningBinding, OpeningAlias,
};
pub use verify::verify;
