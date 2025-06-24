//! General Inner Product Arguments
//!
//! This is the building block of other Product Arguments like Tipa
use std::marker::PhantomData;

use super::Error;
use crate::msm::Icicle;
use crate::{
    field::JoltField,
    utils::transcript::Transcript,
};
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

pub type CommitmentSteps<P> = Vec<(
    (PairingOutput<P>, <P as Pairing>::G1),
    (PairingOutput<P>, <P as Pairing>::G1),
)>;

/// Proof for General Inner Product Argument
///
/// This is basically how bullet-proofs are built
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone)]
pub struct GipaProof<P: Pairing, ProofTranscript> {
    pub(crate) commitment_steps: CommitmentSteps<P>,
    pub(crate) final_message: (P::G1, P::ScalarField),
    // auxiliary info
    pub(crate) scalar_transcript: Vec<P::ScalarField>,
    pub(crate) final_commitment_param: P::G2,
    // we use fn because we need it to be Send without specifying bounds
    pub(crate) _transcript: PhantomData<fn() -> ProofTranscript>,
}

impl<P, ProofTranscript> GipaProof<P, ProofTranscript>
where
    P: Pairing,
    P::G1: Icicle,
    P::ScalarField: JoltField,
    ProofTranscript: Transcript,
{
    // Helper function used to calculate recursive challenges from proof execution (transcript in reverse)
    #[allow(clippy::type_complexity)]
    pub fn verify(
        (mut com_a, mut com_t): (PairingOutput<P>, P::G1),
        commitment_steps: &CommitmentSteps<P>,
        transcript: &mut ProofTranscript,
    ) -> Result<((PairingOutput<P>, P::G1), Vec<P::ScalarField>), Error> {
        let mut r_transcript: Vec<P::ScalarField> = Vec::new();
        for (com_l, com_r) in commitment_steps.iter().rev() {
            // Fiat-Shamir challenge

            transcript.append_serializable(&com_l.0);
            transcript.append_point(&com_l.1);
            transcript.append_serializable(&com_r.0);
            transcript.append_point(&com_r.1);

            let c: P::ScalarField = transcript.challenge_scalar();
            let c_inv = JoltField::inverse(&c).unwrap();

            com_a = com_l.0 * c + com_a + com_r.0 * c_inv;
            com_t = com_l.1 * c + com_t + com_r.1 * c_inv;

            r_transcript.push(c);
        }
        r_transcript.reverse();
        Ok(((com_a, com_t), r_transcript))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        super::afgho::{random_generators, AfghoCommitment},
        super::inner_products::MultiexponentiationInnerProduct,
        *,
    };
    use crate::msm::{use_icicle, Icicle, VariableBaseMSM};
    use crate::utils::transcript::KeccakTranscript;
    use ark_bn254::Bn254;
    use ark_ec::pairing::Pairing;
    use ark_ec::CurveGroup;
    use ark_ff::UniformRand;
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use ark_std::One;

    /// Inner pairing product commitment in G1
    type AfghoBn245 = AfghoCommitment<Bn254>;
    const TEST_SIZE: usize = 8;

    // used only for tests
    impl<P, ProofTranscript> GipaProof<P, ProofTranscript>
    where
        P: Pairing,
        P::G1: Icicle,
        P::G2: Icicle,
        P::ScalarField: JoltField,
        ProofTranscript: Transcript,
    {
        fn _compute_final_commitment_keys(
            l_params: &[P::G2],
            transcript: &[P::ScalarField],
        ) -> Result<P::G2, Error> {
            // Calculate base commitment keys
            assert!(l_params.len().is_power_of_two());

            let mut ck_a_agg_challenge_exponents = vec![P::ScalarField::one()];
            let mut ck_b_agg_challenge_exponents = vec![P::ScalarField::one()];
            for (i, c) in transcript.iter().enumerate() {
                let c_inv = JoltField::inverse(c).unwrap();
                for j in 0..2_usize.pow(i as u32) {
                    ck_a_agg_challenge_exponents.push(ck_a_agg_challenge_exponents[j] * c_inv);
                    ck_b_agg_challenge_exponents.push(ck_b_agg_challenge_exponents[j] * c);
                }
            }
            assert_eq!(ck_a_agg_challenge_exponents.len(), l_params.len());
            let ck_a_base = <P::G2 as VariableBaseMSM>::msm_field_elements(
                &P::G2::normalize_batch(l_params),
                None,
                &ck_a_agg_challenge_exponents,
                None,
                use_icicle(),
            )?;
            Ok(ck_a_base)
        }

        fn _verify_base_commitment(
            base_ck: &P::G2,
            base_com: (PairingOutput<P>, P::G1),
            proof: &GipaProof<P, ProofTranscript>,
        ) -> Result<bool, Error> {
            let (com_a, com_t) = base_com;
            let ck_a_base = base_ck;
            let a_base = vec![proof.final_message.0];
            let b_base = vec![proof.final_message.1];
            let t_base = MultiexponentiationInnerProduct::inner_product(&a_base, &b_base)?;

            let same_ip_commit = t_base == com_t;

            Ok(AfghoCommitment::verify(&[*ck_a_base], &a_base, &com_a)? && same_ip_commit)
        }
    }

    #[test]
    fn multiexponentiation_inner_product_test() {
        type MultiExpGIPA = GipaProof<Bn254, KeccakTranscript>;

        let mut rng = StdRng::seed_from_u64(0u64);

        let params = AfghoBn245::setup(&mut rng, TEST_SIZE);
        let m_a = random_generators(&mut rng, TEST_SIZE);
        let mut m_b = Vec::new();
        for _ in 0..TEST_SIZE {
            m_b.push(<Bn254 as Pairing>::ScalarField::rand(&mut rng));
        }
        let l_commit = AfghoBn245::commit(&params, &m_a).unwrap();
        let ip_commit = MultiexponentiationInnerProduct::inner_product(&m_a, &m_b).unwrap();

        let mut transcript = KeccakTranscript::new(b"test");

        let proof = MultiExpGIPA::prove(m_a, params.clone(), m_b, &mut transcript).unwrap();

        let mut transcript = KeccakTranscript::new(b"test");

        // Calculate base commitment and transcript
        let (base_com, transcript) = MultiExpGIPA::verify(
            (l_commit, ip_commit),
            &proof.commitment_steps,
            &mut transcript,
        )
        .unwrap();
        // Calculate base commitment keys
        let ck_a_base = MultiExpGIPA::_compute_final_commitment_keys(&params, &transcript).unwrap();
        // Verify base commitment
        assert!(
            MultiExpGIPA::_verify_base_commitment(&ck_a_base, (base_com.0, base_com.1), &proof)
                .unwrap()
        )
    }
}
