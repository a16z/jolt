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

#[cfg(feature = "blindfold")]
mod assignment;
#[cfg(feature = "blindfold")]
mod builder;
#[cfg(feature = "blindfold")]
mod error;
#[cfg(feature = "blindfold")]
mod proof;
#[cfg(feature = "blindfold")]
pub mod protocol;
#[cfg(feature = "prover")]
mod prove;
#[cfg(feature = "blindfold")]
pub mod r1cs;
#[cfg(feature = "blindfold")]
mod relaxed;
#[cfg(feature = "blindfold")]
mod statements;
#[cfg(feature = "blindfold")]
mod verify;

#[cfg(feature = "blindfold")]
pub use assignment::AssignedBlindFoldWitness;
#[cfg(feature = "blindfold")]
pub use builder::{BlindFoldProtocolBuilder, BlindFoldStageBuilder};
#[cfg(feature = "blindfold")]
pub use error::{Error, LayoutError, ProverError, RelaxedError, VerificationError};
#[cfg(feature = "blindfold")]
pub use proof::BlindFoldProof;
#[cfg(feature = "blindfold")]
pub use protocol::{
    BlindFoldDimensions, BlindFoldProtocol, FinalOpeningWitnessCoordinates, RowDimensions,
    WitnessCoordinate, WitnessRowLayout,
};
#[cfg(feature = "prover")]
pub use prove::{
    prove, prove_with_row_committer, BlindFoldRowCommitter, BlindFoldWitness,
    DirectBlindFoldRowCommitter,
};
#[cfg(feature = "blindfold")]
pub use relaxed::{RelaxedInstance, RelaxedWitness};
#[cfg(feature = "blindfold")]
pub use statements::{
    BlindFoldStage, BlindFoldStatement, CommittedClaimRows, FinalOpeningBinding, OpeningAlias,
};
