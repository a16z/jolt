mod input;
mod output;
mod prove;
mod request;
#[cfg(test)]
mod tests;

#[cfg(feature = "field-inline")]
pub use input::FieldInlineCommitmentWitness;
pub use input::{CommitmentStageConfig, CommitmentStageInput};
pub use output::{CommitmentStageOutput, CommitmentStageProverState};
pub use prove::{prove, CommitmentStageBackend};
pub use request::build_commitment_request;
