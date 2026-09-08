//! Generic BlindFold claim, protocol, layout, and verifier-equation types.

// In the jolt-verifier runtime closure: stricter panic and unsafe discipline
// than the workspace lints (specs/verifier-closure-lints.md).
#![forbid(unsafe_code)]
#![deny(
    clippy::indexing_slicing,
    clippy::get_unwrap,
    clippy::string_slice,
    clippy::fallible_impl_from,
    clippy::mem_forget,
    clippy::exit,
    clippy::panic_in_result_fn,
    clippy::let_underscore_must_use,
    clippy::host_endian_bytes,
    clippy::wildcard_enum_match_arm
)]

mod assignment;
mod builder;
mod error;
mod proof;
pub mod protocol;
mod prove;
pub mod r1cs;
mod relaxed;
mod statements;
mod verify;

pub use assignment::AssignedBlindFoldWitness;
pub use builder::{BlindFoldProtocolBuilder, BlindFoldStageBuilder};
pub use error::{Error, LayoutError, ProverError, RelaxedError, VerificationError};
pub use proof::BlindFoldProof;
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
