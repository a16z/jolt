//! Zero knowledge for Jolt: generic BlindFold claim, protocol, layout and
//! verifier-equation types, and the masking front end ([`masking`]).

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
pub mod masking;
mod proof;
pub mod protocol;
#[cfg(feature = "prover")]
mod prove;
pub mod r1cs;
mod relaxed;
mod secret;
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
#[cfg(feature = "prover")]
pub use prove::{
    prove, prove_with_row_committer, BlindFoldRowCommitter, BlindFoldWitness,
    DirectBlindFoldRowCommitter,
};
pub use relaxed::{RelaxedInstance, RelaxedWitness};
pub use statements::{
    BlindFoldStage, BlindFoldStatement, CommittedClaimRows, FinalOpeningBinding, OpeningAlias,
};
