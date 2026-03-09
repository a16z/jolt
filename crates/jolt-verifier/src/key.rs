//! Verification key for Jolt proofs.

use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_spartan::SpartanKey;
use serde::{Deserialize, Serialize};

/// Verification key for a Jolt proof.
///
/// Contains the Spartan key (precomputed matrix MLEs) and the PCS verifier
/// setup. Constructed once during preprocessing and reused across multiple
/// proof verifications for the same circuit.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct JoltVerifyingKey<F: Field, PCS: CommitmentScheme<Field = F>> {
    /// Spartan key derived from the R1CS constraint system.
    pub spartan_key: SpartanKey<F>,
    /// PCS verifier-side structured reference string.
    pub pcs_setup: PCS::VerifierSetup,
}
