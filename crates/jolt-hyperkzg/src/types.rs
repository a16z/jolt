use jolt_crypto::{Bn254G1, Bn254G2, JoltGroup};
use jolt_field::Fr;
use jolt_transcript::{Channel, TranscriptError};
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
    pub(crate) g1: Bn254G1,
    pub(crate) g2: Bn254G2,
    pub(crate) beta_g2: Bn254G2,
}

impl HyperKZGVerifierSetup {
    pub(crate) fn validate(&self) -> Result<usize, HyperKZGError> {
        let capacity = usize::try_from(self.num_powers).map_err(|_| HyperKZGError::InvalidSetup)?;
        if capacity < 2
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

    /// Binds the setup and the whole opening statement before any opening
    /// message, on either side.
    pub(crate) fn bind_statement<C: Channel>(
        &self,
        commitment: &Bn254G1,
        point: &[Fr],
        evaluation: Fr,
        channel: &mut C,
    ) {
        channel.public(&self.num_powers.to_le_bytes());
        channel.public(&self.setup_id);
        channel.public(&self.g1);
        channel.public(&self.g2);
        channel.public(&self.beta_g2);
        channel.public(commitment);
        channel.public_all(point);
        channel.public(&evaluation);
    }
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
    #[error(transparent)]
    Transcript(#[from] TranscriptError),
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test fixture serialization fails loudly"
)]
mod tests {
    use jolt_crypto::Bn254;
    use jolt_field::{One, Ring, Zero};
    use jolt_transcript::{Blake2b512, ProtocolId, VerifierTranscript};

    use super::*;
    use crate::HyperKZGScheme;

    const PROTOCOL: ProtocolId = ProtocolId::new::<Blake2b512>("jolt-hyperkzg/test");

    #[test]
    fn decoded_verifier_metadata_is_rechecked() {
        let valid = HyperKZGVerifierSetup {
            num_powers: 4,
            setup_id: [1; 32],
            g1: Bn254::g1_generator(),
            g2: Bn254::g2_generator(),
            beta_g2: Bn254::g2_generator().scalar_mul(&Fr::from_u64(7)),
        };
        let verify = |key: &HyperKZGVerifierSetup| {
            let mut transcript =
                VerifierTranscript::<Blake2b512>::new(&PROTOCOL, b"decoded-key", &[]);
            HyperKZGScheme::verify_opening(
                &Bn254G1::identity(),
                &[Fr::one()],
                Fr::zero(),
                key,
                &mut transcript,
            )
        };
        // A valid key reaches the (here empty) opening messages.
        assert_eq!(
            verify(&valid),
            Err(HyperKZGError::Transcript(TranscriptError::Truncated))
        );
        let mut invalid = std::array::from_fn::<_, 4, _>(|_| valid.clone());
        let [capacity, g1, g2, beta_g2] = &mut invalid;
        capacity.num_powers = 0;
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
