//! Public prover API entry points and setup artifacts.

pub mod commitment;
mod scheme;
pub mod setup;
pub mod setup_prefix;

pub use commitment::{
    batched_commit, batched_commit_with_params, commit, commit_with_params,
    prepare_batched_commit_inputs, prepare_commit_inputs,
};
pub use scheme::CommitmentProver;
pub use setup::AkitaProverSetup;
pub use setup_prefix::commit_setup_prefix;
