use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use serde::{Deserialize, Serialize};

/// Contains the PCS verifier setup. Constructed once during preprocessing
/// and reused across multiple proof verifications for the same circuit.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct JoltVerifyingKey<F: Field, PCS: CommitmentScheme<Field = F>> {
    /// PCS verifier-side structured reference string.
    pub pcs_setup: PCS::VerifierSetup,
}
