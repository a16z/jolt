//! Proof and key types re-exported from [`jolt_verifier`].

use jolt_field::Field;
use jolt_openings::CommitmentScheme;

pub use jolt_verifier::proof::{JoltProof, StageProof};

pub struct JoltProvingKey<F: Field, PCS: CommitmentScheme<Field = F>> {
    pub pcs_prover_setup: PCS::ProverSetup,
    pub pcs_verifier_setup: PCS::VerifierSetup,
}
pub use jolt_verifier::{
    verify_openings, JoltError, JoltVerifyingKey, OneHotConfig, OneHotParams, ProverConfig,
    ReadWriteConfig,
};
