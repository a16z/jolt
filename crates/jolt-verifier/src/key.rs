use jolt_compiler::module::Module;
use jolt_compiler::{PolynomialSpec, VerifierSchedule};
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_r1cs::R1csKey;
use serde::{Deserialize, Serialize};

/// Contains the PCS verifier setup, compiled verifier schedule, and R1CS key.
///
/// Constructed once during preprocessing and reused across multiple proof
/// verifications for the same circuit. The schedule is the compiler's output
/// describing the verifier's computation — the verifier is a generic
/// interpreter over it.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct JoltVerifyingKey<P: PolynomialSpec, F: Field, PCS: CommitmentScheme<Field = F>> {
    /// PCS verifier-side structured reference string.
    pub pcs_setup: PCS::VerifierSetup,
    /// Compiled verifier schedule: stages, claim formulas, sumcheck params.
    pub schedule: VerifierSchedule<P>,
    /// Preprocessed R1CS key for evaluating matrix-vector products.
    pub r1cs_key: R1csKey<F>,
}

impl<P: PolynomialSpec, F: Field, PCS: CommitmentScheme<Field = F>> JoltVerifyingKey<P, F, PCS> {
    /// Construct from a compiled module, PCS verifier setup, and R1CS key.
    pub fn new(module: &Module<P>, pcs_setup: PCS::VerifierSetup, r1cs_key: R1csKey<F>) -> Self {
        Self {
            pcs_setup,
            schedule: module.verifier.clone(),
            r1cs_key,
        }
    }
}
