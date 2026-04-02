use jolt_compiler::module::Module;
use jolt_compiler::{PolyId, VerifierSchedule};
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use serde::{Deserialize, Serialize};

/// Contains the PCS verifier setup and the compiled verifier schedule.
///
/// Constructed once during preprocessing and reused across multiple proof
/// verifications for the same circuit. The schedule is the compiler's output
/// describing the verifier's computation — the verifier is a generic
/// interpreter over it.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct JoltVerifyingKey<P: PolyId, F: Field, PCS: CommitmentScheme<Field = F>> {
    /// PCS verifier-side structured reference string.
    pub pcs_setup: PCS::VerifierSetup,
    /// Compiled verifier schedule: stages, claim formulas, sumcheck params.
    pub schedule: VerifierSchedule<P>,
}

impl<P: PolyId, F: Field, PCS: CommitmentScheme<Field = F>> JoltVerifyingKey<P, F, PCS> {
    /// Construct from a compiled module and PCS verifier setup.
    ///
    /// Call once at preprocessing time; reuse across verifications.
    pub fn from_module(module: &Module<P>, pcs_setup: PCS::VerifierSetup) -> Self {
        Self {
            pcs_setup,
            schedule: module.verifier.clone(),
        }
    }
}
