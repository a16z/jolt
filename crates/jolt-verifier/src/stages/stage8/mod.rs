//! Stage 8: the final PCS opening. [`verify`] is the per-build entry point;
//! the feature-specific statement assembly lives beside it.

pub mod outputs;
/// Packed-build statement assembly: per-object packings, leaf-claim
/// resolution, and the joint opening call.
#[cfg(feature = "akita")]
mod packed;
#[cfg(not(feature = "akita"))]
pub mod precommitted;
/// The reconstruction phase that opens the stage-8 region on the packed path:
/// settles every virtualized word/chunk claim against its committed one-hot
/// decomposition, producing the packed leaf claims the opening consumes.
/// Public because its output-claims aggregate is part of the proof's clear
/// claims.
#[cfg(feature = "akita")]
pub mod reconstruction;
mod verify;

#[cfg(not(feature = "akita"))]
pub use outputs::Stage8ClearOutput;
pub use outputs::{Stage8Output, Stage8ZkOutput};
#[cfg(not(feature = "akita"))]
pub use precommitted::precommitted_final_openings;
pub use verify::verify;
#[cfg(not(feature = "akita"))]
pub use verify::{batch_entries, Stage8BatchEntry};

/// The commitment/setup metadata Stage 8 enforces before dispatching a
/// native OneHotTrace opening — the generic [`jolt_openings`] traits, applied here
/// to the OneHotTrace group (impls live beside the concrete PCS types).
#[cfg(feature = "akita")]
pub use jolt_openings::{
    GroupCommitmentMetadata as OneHotTraceCommitmentMetadata,
    GroupSetupMetadata as OneHotTraceSetupMetadata,
};
