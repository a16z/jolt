//! Backend compute implementations for modular Jolt proving.
//!
//! Backends own compute traits and request/result types. Protocol crates decide
//! what work to schedule.

#[cfg(feature = "zk")]
mod blindfold;
mod commitments;
mod error;
mod ids;
mod openings;
mod sumcheck;
mod traits;

#[cfg(feature = "cpu")]
pub mod cpu;

#[cfg(feature = "zk")]
pub use blindfold::{
    BlindFoldPrivateOpening, BlindFoldRequest, BlindFoldResult, BlindFoldRoundRequest,
    BlindFoldSlot,
};
pub use commitments::{
    CommitmentRequest, CommitmentRequestItem, CommitmentResult, CommitmentSlot,
    CommittedPolynomialOutput, ResolvedWitnessRequirement, StreamedWitnessChunk,
    StreamedWitnessOutput,
};
pub use error::BackendError;
pub use ids::{BackendRelationId, BackendValueSlot};
pub use openings::{
    OpeningEvaluationOutput, OpeningProofOutput, OpeningQueryRequest, OpeningRequest,
    OpeningResult, OpeningSlot,
};
pub use sumcheck::{
    SumcheckEvaluationOutput, SumcheckInstanceRequest, SumcheckProofOutput, SumcheckRequest,
    SumcheckResult, SumcheckSlot,
};
#[cfg(feature = "zk")]
pub use traits::BlindFoldBackend;
pub use traits::{Backend, CommitmentBackend, OpeningBackend, SumcheckBackend};
