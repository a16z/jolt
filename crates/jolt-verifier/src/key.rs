//! Verification key for Jolt proofs.

use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_spartan::UniformSpartanKey;
use serde::{Deserialize, Serialize};

/// Verification key for a Jolt proof.
///
/// Contains the uniform Spartan key (per-cycle sparse constraints) and the
/// PCS verifier setup. Constructed once during preprocessing and reused
/// across multiple proof verifications for the same circuit.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct JoltVerifyingKey<F: Field, PCS: CommitmentScheme<Field = F>> {
    /// Uniform Spartan key with per-cycle sparse constraint matrices.
    pub spartan_key: UniformSpartanKey<F>,
    /// PCS verifier-side structured reference string.
    pub pcs_setup: PCS::VerifierSetup,
}
