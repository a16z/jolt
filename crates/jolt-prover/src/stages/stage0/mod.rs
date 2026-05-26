mod input;
mod output;
mod prove;
mod request;
#[cfg(test)]
mod tests;

pub use input::CommitmentStageConfig;
pub use output::CommitmentStageOutput;
#[cfg(feature = "field-inline")]
pub use output::FieldInlineCommittedPolynomialOutput;
pub use prove::{commit_witness, prove_jolt_vm_commitments};
pub use request::build_commitment_request;
