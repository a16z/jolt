//! Proof and key types re-exported from [`jolt_verifier`].

use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_spartan::UniformSpartanKey;

pub use jolt_verifier::proof::{JoltProof, StageProof};
pub use jolt_verifier::{verify_openings, verify_spartan};
pub use jolt_verifier::{JoltError, JoltVerifyingKey, OneHotConfig, OneHotParams, ProverConfig, ReadWriteConfig};

pub struct JoltProvingKey<F: Field, PCS: CommitmentScheme<Field = F>> {
    pub spartan_key: UniformSpartanKey<F>,
    pub pcs_prover_setup: PCS::ProverSetup,
    pub pcs_verifier_setup: PCS::VerifierSetup,
}
