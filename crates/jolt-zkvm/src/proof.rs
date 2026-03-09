//! Proof and key types for the complete Jolt pipeline.
//!
//! Proof types are defined in [`jolt_verifier`] and re-exported here.
//! [`JoltProvingKey`] is prover-specific and lives in this crate.

use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_spartan::UniformSpartanKey;

pub use jolt_verifier::proof::{BatchOpeningProofs, JoltProof, SumcheckStageProof};
pub use jolt_verifier::{JoltError, JoltVerifyingKey, VerifierStage};
pub use jolt_verifier::{verify, verify_openings, verify_spartan};

/// Proving key containing preprocessed circuit data and PCS parameters.
///
/// Constructed once during preprocessing; reused across multiple prove() calls
/// for the same circuit. Uses the uniform Spartan key which stores per-cycle
/// sparse constraints, achieving O(K) key size regardless of cycle count.
pub struct JoltProvingKey<F: Field, PCS: CommitmentScheme<Field = F>> {
    /// Uniform Spartan key with per-cycle sparse constraint matrices.
    pub spartan_key: UniformSpartanKey<F>,
    /// PCS prover-side structured reference string.
    pub pcs_prover_setup: PCS::ProverSetup,
    /// PCS verifier-side structured reference string.
    pub pcs_verifier_setup: PCS::VerifierSetup,
}
