//! Proving key type.

use jolt_field::Field;
use jolt_openings::CommitmentScheme;

/// PCS setup material for the prover.
pub struct JoltProvingKey<F: Field, PCS: CommitmentScheme<Field = F>> {
    pub pcs_prover_setup: PCS::ProverSetup,
    pub pcs_verifier_setup: PCS::VerifierSetup,
}
