//! Stage 8: the final PCS opening. [`verify`] is the per-build entry point;
//! the feature-specific statement assembly lives beside it.

/// Homomorphic-build statement assembly: batch entries with unified-point
/// embedding scales.
#[cfg(not(feature = "akita"))]
mod homomorphic;
pub mod outputs;
/// Packed-build statement assembly: per-object packings, leaf-claim
/// resolution, and the joint opening call.
#[cfg(feature = "akita")]
mod packed;
#[cfg(not(feature = "akita"))]
mod precommitted;
/// The reconstruction phase that opens the stage-8 region on the packed path:
/// settles every virtualized word/chunk claim against its committed one-hot
/// decomposition, producing the packed leaf claims the opening consumes.
/// Public because its output-claims aggregate is part of the proof's clear
/// claims.
#[cfg(feature = "akita")]
pub mod reconstruction;
mod verify;

pub use outputs::{Stage8Output, Stage8ZkOutput};
pub use verify::verify;
