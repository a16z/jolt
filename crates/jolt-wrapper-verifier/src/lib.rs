//! Verifier-side wrapper protocol for configured Jolt verifier R1CS.

mod config;
mod error;
mod proof;
mod r1cs_builder;
pub mod stages;
mod verifier;

pub use config::{validate_proof_config, WrapperVerifierConfig, WrapperVerifierKey};
#[cfg(feature = "zk")]
pub use config::{validate_zk_proof_config, WrapperZkVerifierConfig};
pub use error::{Error, WrapperError};
pub use proof::{HyperKzgProof, R1csRelationStatement, SpartanProof, WrapperProof};
#[cfg(feature = "zk")]
pub use proof::{SpartanZkProof, WrapperZkProof};
pub use r1cs_builder::{
    verify_r1cs_witness, WrapperR1csBuilder, WrapperR1csLayout, WrapperR1csProtocol,
};
#[cfg(feature = "zk")]
pub use verifier::verify_zk;
pub use verifier::{verify, CheckedInputs, WrapperVerifierInputs};
