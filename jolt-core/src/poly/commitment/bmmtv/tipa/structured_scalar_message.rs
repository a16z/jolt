use std::marker::PhantomData;

use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ff::One;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{cfg_iter, end_timer, start_timer};

use super::super::{
    gipa::Gipa,
    inner_products::InnerProduct,
    tipa::{prove_commitment_key_kzg_opening, verify_commitment_key_g2_kzg_opening},
    Error,
};
use crate::{
    field::JoltField,
    msm::Icicle,
    poly::commitment::{
        bmmtv::{
            commitments::{
                afgho16::AfghoCommitment,
                identity::{IdentityCommitment, IdentityOutput},
            },
            gipa::CommitmentSteps,
            inner_products::MultiexponentiationInnerProduct,
        },
        kzg::KZGVerifierKey,
    },
    utils::transcript::Transcript,
};

/// Pairing-based instantiation of GIPA with an updatable
/// (trusted) structured reference string (SRS) to achieve
/// logarithmic-time verification
pub struct TipaWithSsm<P, Transcript> {
    _pair: PhantomData<P>,
    _transcript: PhantomData<Transcript>,
}

/// Proof of [`TipaWithSsm`]
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone)]
pub struct TipaWithSsmProof<P>
where
    P: Pairing,
{
    commitment_steps: CommitmentSteps<P>,
    final_message: (P::G1, P::ScalarField),
    final_ck: P::G2,
    final_ck_proof: P::G2,
}

impl<P, ProofTranscript> TipaWithSsm<P, ProofTranscript>
where
    P: Pairing,
    P::G1: Icicle,
    P::ScalarField: JoltField,
    ProofTranscript: Transcript,
{
    pub fn prove_with_structured_scalar_message(
        h_beta_powers: &[P::G2],
        values: (&[P::G1], &[P::ScalarField]),
        ck: &[P::G2],
        transcript: &mut ProofTranscript,
    ) -> Result<TipaWithSsmProof<P>, Error> {
        // Run GIPA
        let gipa = start_timer!(|| "GIPA");
        let proof = Gipa::<P, ProofTranscript>::prove_with_aux(values, ck, transcript)?;
        end_timer!(gipa);

        // Prove final commitment key is wellformed
        let ck_kzg = start_timer!(|| "Prove commitment key");
        let ck_a_final = proof.final_commitment_param;
        let transcript_inverse = cfg_iter!(proof.scalar_transcript)
            .map(|x| JoltField::inverse(x).unwrap())
            .collect::<Vec<_>>();

        // KZG challenge point
        transcript.append_serializable(&ck_a_final);
        let c = transcript.challenge_scalar();

        // Complete KZG proof
        let ck_a_kzg_opening = prove_commitment_key_kzg_opening(
            h_beta_powers,
            &transcript_inverse,
            &<P::ScalarField>::one(), // r_shift = one, why?
            &c,
        )?;
        end_timer!(ck_kzg);

        Ok(TipaWithSsmProof {
            final_message: proof.final_message,
            commitment_steps: proof.commitment_steps,
            final_ck: ck_a_final,
            final_ck_proof: ck_a_kzg_opening,
        })
    }

    pub fn verify_with_structured_scalar_message(
        v_srs: &KZGVerifierKey<P>,
        com: (&PairingOutput<P>, &IdentityOutput<P::G1>),
        scalar_b: &P::ScalarField,
        proof: &TipaWithSsmProof<P>,
        transcript: &mut ProofTranscript,
    ) -> Result<bool, Error> {
        let (base_com, gipa_transcript) =
            Gipa::<P, ProofTranscript>::verify_recursive_challenge_transcript(
                (com.0, scalar_b, com.1),
                &proof.commitment_steps,
                transcript,
            )?;
        let transcript_inverse = cfg_iter!(gipa_transcript)
            .map(|x| JoltField::inverse(x).unwrap())
            .collect::<Vec<_>>();

        let ck_a_final = &proof.final_ck;
        let ck_a_proof = &proof.final_ck_proof;

        // KZG challenge point
        transcript.append_serializable(ck_a_final);
        let c = transcript.challenge_scalar();

        // Check commitment key
        let ck_a_valid = verify_commitment_key_g2_kzg_opening(
            v_srs,
            ck_a_final,
            ck_a_proof,
            &transcript_inverse,
            &P::ScalarField::one(),
            &c,
        )?;

        // Compute final scalar
        let mut power_2_b = *scalar_b;
        let mut product_form = Vec::new();
        for x in gipa_transcript.iter() {
            product_form
                .push(<P::ScalarField>::one() + (JoltField::inverse(x).unwrap() * power_2_b));
            power_2_b *= power_2_b;
        }
        let b_base = cfg_iter!(product_form).product::<P::ScalarField>();

        // Verify base inner product commitment
        let (com_a, _, com_t) = base_com;
        let a_base = vec![proof.final_message.0];
        let t_base = vec![MultiexponentiationInnerProduct::<P::G1>::inner_product(
            &a_base,
            &[b_base],
        )?];
        let base_valid = AfghoCommitment::verify(&[*ck_a_final], &a_base, &com_a)?
            && IdentityCommitment::verify(&t_base, &com_t);

        Ok(ck_a_valid && base_valid)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        super::super::{
            commitments::{afgho16::AfghoCommitment, random_generators},
            inner_products::{InnerProduct, MultiexponentiationInnerProduct},
        },
        *,
    };
    use crate::poly::commitment::bmmtv::tipa::Field;
    use crate::poly::commitment::kzg::SRS;
    use crate::utils::transcript::KeccakTranscript;
    use ark_bn254::Bn254;
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use ark_std::UniformRand;
    use std::sync::Arc;

    type BnAfghoG1 = AfghoCommitment<Bn254>;
    type BnScalarField = <Bn254 as Pairing>::ScalarField;
    type BnG1 = <Bn254 as Pairing>::G1;

    const TEST_SIZE: usize = 8;

    fn structured_scalar_power<F: Field>(num: usize, s: &F) -> Vec<F> {
        let mut powers = vec![F::one()];
        for i in 1..num {
            powers.push(powers[i - 1] * s);
        }
        powers
    }

    #[test]
    fn tipa_ssm_multiexponentiation_inner_product_test() {
        type IP = MultiexponentiationInnerProduct<BnG1>;
        type MultiExpTipa = TipaWithSsm<Bn254, KeccakTranscript>;

        let mut rng = StdRng::seed_from_u64(0u64);
        let srs = SRS::setup(&mut rng, 2 * (TEST_SIZE - 1), 2 * (TEST_SIZE - 1));

        let ck_a = srs.get_commitment_keys();
        let powers_len = srs.g1_powers.len();
        let (p_srs, v_srs) = SRS::trim(Arc::new(srs), powers_len - 1);

        let m_a = random_generators(&mut rng, TEST_SIZE);
        let b = BnScalarField::rand(&mut rng);
        let m_b = structured_scalar_power(TEST_SIZE, &b);
        let com_a = BnAfghoG1::commit(&ck_a, &m_a).unwrap();
        let t = vec![IP::inner_product(&m_a, &m_b).unwrap()];
        let com_t = IdentityOutput(t);

        let mut transcript = KeccakTranscript::new(b"TipaTest");

        let proof = MultiExpTipa::prove_with_structured_scalar_message(
            &p_srs.h_beta_powers(),
            (&m_a, &m_b),
            &ck_a,
            &mut transcript,
        )
        .unwrap();

        let mut transcript = KeccakTranscript::new(b"TipaTest");

        assert!(MultiExpTipa::verify_with_structured_scalar_message(
            &v_srs,
            (&com_a, &com_t),
            &b,
            &proof,
            &mut transcript,
        )
        .unwrap());
    }
}
