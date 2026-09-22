use jolt_crypto::{Bn254G1, Bn254G2, JoltGroup};
use jolt_field::Fr;
use jolt_transcript::{Transcript, U64Word};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Public powers from an authenticated BN254 KZG ceremony.
///
/// `g1_powers[j] = beta^j G1`, `beta_g2 = beta G2`. The importer must
/// authenticate the ceremony and the consistency of these powers. The scheme
/// checks dimensions and nonidentity setup elements, not ceremony provenance.
/// Do not pass powers generated with a known or retained `beta`.
pub struct HyperKZGSetupParams {
    /// Capacity for honest polynomial tables; not an adversarial degree bound.
    pub g1_powers: Vec<Bn254G1>,
    /// Authenticated application/ceremony policy identifier.
    pub setup_id: [u8; 32],
    /// Largest degree supported by all public powers under this trapdoor,
    /// including powers not imported here. This is a trusted policy assertion.
    pub max_public_degree: u64,
    pub g2: Bn254G2,
    pub beta_g2: Bn254G2,
}

/// Imported immutable prover powers. Only the checked import constructs this.
#[derive(Clone, Debug)]
pub struct HyperKZGProverSetup {
    pub(crate) g1_powers: Vec<Bn254G1>,
    pub(crate) verifier: HyperKZGVerifierSetup,
}

/// Authenticated verification key; proof verification rechecks shape invariants.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HyperKZGVerifierSetup {
    pub(crate) num_powers: u64,
    pub(crate) setup_id: [u8; 32],
    pub(crate) max_public_degree: u64,
    pub(crate) g1: Bn254G1,
    pub(crate) g2: Bn254G2,
    pub(crate) beta_g2: Bn254G2,
}

/// Validated public setup material used by the preprocessed Spartan key.
/// Application authentication remains the caller's responsibility.
#[derive(Clone, Debug)]
pub struct HyperKZGSetupBinding {
    pub num_powers: u64,
    pub max_public_degree: u64,
    pub setup_id: [u8; 32],
    pub canonical_bytes: Vec<u8>,
}

impl HyperKZGVerifierSetup {
    /// Validates this imported setup and returns the v2 setup-digest preimage.
    pub fn binding(&self) -> Result<HyperKZGSetupBinding, HyperKZGError> {
        let _ = self.validate()?;
        let mut bytes = b"JOLT-HKZG-SETUP\0".to_vec();
        bytes.extend(self.setup_id);
        bytes.extend(self.num_powers.to_le_bytes());
        bytes.extend(self.max_public_degree.to_le_bytes());
        bytes.extend(self.g1.compressed_bytes());
        bytes.extend(self.g2.compressed_bytes());
        bytes.extend(self.beta_g2.compressed_bytes());
        Ok(HyperKZGSetupBinding {
            num_powers: self.num_powers,
            max_public_degree: self.max_public_degree,
            setup_id: self.setup_id,
            canonical_bytes: bytes,
        })
    }

    pub(crate) fn validate(&self) -> Result<usize, HyperKZGError> {
        let capacity = usize::try_from(self.num_powers).map_err(|_| HyperKZGError::InvalidSetup)?;
        if capacity < 2
            || self.max_public_degree < self.num_powers.saturating_sub(1)
            || self.g1.is_identity()
            || self.g2.is_identity()
            || self.beta_g2.is_identity()
        {
            return Err(HyperKZGError::InvalidSetup);
        }
        Ok(capacity)
    }

    pub(crate) fn check_arity(&self, num_vars: usize) -> Result<usize, HyperKZGError> {
        let capacity = self.validate()?;
        let shift = u32::try_from(num_vars).map_err(|_| HyperKZGError::InvalidArity)?;
        let len = 1usize
            .checked_shl(shift)
            .ok_or(HyperKZGError::InvalidArity)?;
        if num_vars == 0 || len > capacity {
            return Err(HyperKZGError::InvalidArity);
        }
        Ok(len)
    }

    pub(crate) fn append_statement(
        &self,
        commitment: &Bn254G1,
        point: &[Fr],
        evaluation: Fr,
        transcript: &mut impl Transcript<Challenge = Fr>,
    ) {
        transcript.append_labeled(b"hyperkzg-binary-bn254-v1", &U64Word(self.num_powers));
        transcript.append_bytes(&self.setup_id);
        transcript.append(&U64Word(self.max_public_degree));
        transcript.append(&self.g1);
        transcript.append(&self.g2);
        transcript.append(&self.beta_g2);
        transcript.append(commitment);
        transcript.append_values(b"opening-point", point);
        transcript.append(&evaluation);
    }
}

/// Binary fold commitments, evaluations at `[r, -r, r^2]`, and KZG witnesses.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HyperKZGProof {
    pub com: Vec<Bn254G1>,
    pub v: [Vec<Fr>; 3],
    pub w: [Bn254G1; 3],
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum HyperKZGError {
    #[error("invalid imported HyperKZG setup")]
    InvalidSetup,
    #[error("opening arity must be nonzero and fit the setup and address space")]
    InvalidArity,
    #[error("polynomial table length disagrees with its arity")]
    PolynomialShape,
    #[error("polynomial evaluation disagrees with the opening claim")]
    WrongEvaluation,
    #[error("opening proof has the wrong number of fold commitments or evaluations")]
    ProofShape,
    #[error("HyperKZG received the degenerate zero challenge")]
    DegenerateChallenge,
    #[error("HyperKZG fold equation failed")]
    Folding,
    #[error("KZG pairing equation failed")]
    Pairing,
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test fixture serialization fails loudly"
)]
mod tests {
    use jolt_crypto::Bn254;
    use jolt_field::{One, Ring, Zero};
    use jolt_transcript::Blake2bTranscript;

    use super::*;
    use crate::HyperKZGScheme;

    #[test]
    fn decoded_verifier_metadata_is_rechecked() {
        let valid = HyperKZGVerifierSetup {
            num_powers: 4,
            max_public_degree: 3,
            setup_id: [1; 32],
            g1: Bn254::g1_generator(),
            g2: Bn254::g2_generator(),
            beta_g2: Bn254::g2_generator().scalar_mul(&Fr::from_u64(7)),
        };
        let proof = HyperKZGProof {
            com: vec![],
            v: std::array::from_fn(|_| vec![Fr::zero()]),
            w: [Bn254G1::identity(); 3],
        };
        let verify = |key: &HyperKZGVerifierSetup| {
            HyperKZGScheme::verify_opening(
                &Bn254G1::identity(),
                &[Fr::one()],
                Fr::zero(),
                &proof,
                key,
                &mut Blake2bTranscript::new(b"decoded-key"),
            )
        };
        verify(&valid).unwrap();
        let mut invalid = std::array::from_fn::<_, 5, _>(|_| valid.clone());
        let [capacity, degree, g1, g2, beta_g2] = &mut invalid;
        capacity.num_powers = 0;
        degree.max_public_degree = 2;
        g1.g1 = Bn254G1::identity();
        g2.g2 = Bn254G2::identity();
        beta_g2.beta_g2 = Bn254G2::identity();
        for key in invalid {
            let bytes = bincode::serde::encode_to_vec(&key, bincode::config::standard()).unwrap();
            let (decoded, _): (HyperKZGVerifierSetup, _) =
                bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
            assert_eq!(verify(&decoded), Err(HyperKZGError::InvalidSetup));
        }
    }
}
