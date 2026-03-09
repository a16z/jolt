//! Proof and key types for the complete Jolt pipeline.
//!
//! [`JoltProof`] bundles the Spartan proof, per-stage sumcheck proofs, and
//! batch opening proofs into a single serializable artifact. [`JoltProvingKey`]
//! holds the preprocessed circuit data and PCS parameters needed by the prover.

use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_spartan::{SpartanKey, SpartanProof};
use jolt_sumcheck::SumcheckProof;
use serde::{Deserialize, Serialize};

use crate::stages::s8_opening::OpeningProofs;

/// Complete Jolt proof for one program execution.
///
/// Produced by running stages S1 (Spartan) through S8 (batch openings).
/// The verifier replays the Fiat-Shamir transcript and checks each component
/// in sequence.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct JoltProof<F: Field, PCS: CommitmentScheme<Field = F>> {
    /// Spartan R1CS proof (outer + inner sumcheck + witness opening).
    pub spartan_proof: SpartanProof<F, PCS>,
    /// Per-stage sumcheck proofs from stages S2–S7.
    pub sumcheck_proofs: Vec<SumcheckProof<F>>,
    /// Batch PCS opening proofs for all polynomial claims.
    pub opening_proofs: OpeningProofs<PCS>,
}

/// Proving key containing preprocessed circuit data and PCS parameters.
///
/// Constructed once during preprocessing; reused across multiple prove() calls
/// for the same circuit.
pub struct JoltProvingKey<F: Field, PCS: CommitmentScheme<Field = F>> {
    /// Spartan key derived from the R1CS constraint system.
    pub spartan_key: SpartanKey<F>,
    /// PCS prover-side structured reference string.
    pub pcs_prover_setup: PCS::ProverSetup,
    /// PCS verifier-side structured reference string.
    pub pcs_verifier_setup: PCS::VerifierSetup,
}
