//! Wrapper SNARK infrastructure for configured Jolt verifier R1CS.

mod builder;
mod error;
mod proof;
mod protocol;
pub mod r1cs;
pub mod snark_backends;
mod statements;
mod verify;
mod witness;

pub use builder::WrapperProtocolBuilder;
pub use error::Error;
pub use proof::WrapperProof;
pub use protocol::{WrapperLayout, WrapperProtocol};
pub use statements::WrapperStatement;
pub use verify::verify_r1cs_witness;
pub use witness::WrapperWitness;
