mod batch;
mod io;
pub(crate) mod prepare;
pub mod prove;
mod relation_state;
mod verifier_output;

#[cfg(feature = "zk")]
pub use io::Stage6CommittedProofComponent;
pub use io::{Stage6ProofComponent, Stage6ProverConfig, Stage6ProverInput};
