mod request;
mod result;

pub use request::{
    CommitmentMode, CommitmentRequest, CommitmentRequestItem, CommitmentSlot,
    TracePolynomialEmbedding,
};
pub use result::{
    CommitmentResult, CommittedPolynomialOutput, ResolvedWitnessRequirement, StreamedWitnessChunk,
    StreamedWitnessOutput,
};
