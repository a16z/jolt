pub(crate) mod prepare;
mod prove;
#[cfg(test)]
mod tests;

pub use prove::{prove, CommitmentComponent, CommitmentStageBackend, CommitmentStageInput};
