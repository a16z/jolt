use super::super::{
    commitments::Dhc,
    gipa::{Gipa, GipaProof},
    inner_products::InnerProduct,
    tipa::{
        prove_commitment_key_kzg_opening, structured_generators_scalar_power,
        verify_commitment_key_g2_kzg_opening, Srs, VerifierSrs,
    },
    Error,
};
use crate::field::JoltField;
use crate::poly::commitment::bmmtv::commitments::afgho16::AfghoCommitment;
use crate::poly::commitment::bmmtv::commitments::identity::{IdentityCommitment, IdentityOutput};
use crate::poly::commitment::bmmtv::inner_products::MultiexponentiationInnerProduct;
use crate::utils::transcript::Transcript;
use ark_ec::pairing::PairingOutput;
use ark_ec::{pairing::Pairing, Group};
use ark_ff::{One, PrimeField, UniformRand};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{cfg_iter, rand::Rng};
use ark_std::{end_timer, start_timer};
use std::marker::PhantomData;
//TODO: Properly generalize the non-committed message approach of SIPP and MIPP to GIPA
//TODO: Structured message is a special case of the non-committed message and does not rely on TIPA
//TODO: Can support structured group element messages as well as structured scalar messages

/// Use placeholder commitment to commit to vector in clear during GIPA execution
#[derive(Clone)]
pub struct SsmDummyCommitment<F> {
    _field: PhantomData<F>,
}

impl<F: PrimeField> Dhc for SsmDummyCommitment<F> {
    type Scalar = F;
    type Message = F;
    type Param = ();
    type Output = F;

    fn setup<R: Rng>(_rng: &mut R, _size: usize) -> Result<Vec<Self::Param>, Error> {
        Ok(vec![])
    }

    //TODO: Doesn't include message which means scalar b not included in generating challenges
    fn commit(_k: &[Self::Param], _m: &[Self::Message]) -> Result<Self::Output, Error> {
        Ok(F::zero())
    }
}

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
    gipa_proof: GipaProof<P>,
    final_ck: P::G2,
    final_ck_proof: P::G2,
}

impl<P, ProofTranscript> TipaWithSsm<P, ProofTranscript>
where
    P: Pairing,
    P::ScalarField: JoltField,
    ProofTranscript: Transcript,
{
    //TODO: Don't need full TIPA SRS since only using one set of powers
    pub fn setup<R: Rng>(rng: &mut R, size: usize) -> Result<Srs<P>, Error> {
        let alpha = <P::ScalarField>::rand(rng);
        let beta = <P::ScalarField>::rand(rng);
        let g = P::G1::generator();
        let h = P::G2::generator();
        Ok(Srs {
            g_alpha_powers: structured_generators_scalar_power(2 * size - 1, &g, &alpha),
            h_beta_powers: structured_generators_scalar_power(2 * size - 1, &h, &beta),
            g_beta: g * beta,
            h_alpha: h * alpha,
        })
    }

    pub fn prove_with_structured_scalar_message(
        h_beta_powers: &[P::G2],
        values: (&[P::G1], &[P::ScalarField]),
        ck: &[P::G2],
        transcript: &mut ProofTranscript,
    ) -> Result<TipaWithSsmProof<P>, Error> {
        // Run GIPA
        let gipa = start_timer!(|| "GIPA");
        let (proof, aux) = Gipa::<P, ProofTranscript>::prove_with_aux(values, ck, transcript)?;
        end_timer!(gipa);

        // Prove final commitment key is wellformed
        let ck_kzg = start_timer!(|| "Prove commitment key");
        let ck_a_final = aux.final_commitment_param;
        let transcript_inverse = cfg_iter!(aux.scalar_transcript)
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
            gipa_proof: proof,
            final_ck: ck_a_final,
            final_ck_proof: ck_a_kzg_opening,
        })
    }

    pub fn verify_with_structured_scalar_message(
        v_srs: &VerifierSrs<P>,
        com: (&PairingOutput<P>, &IdentityOutput<P::G1>),
        scalar_b: &P::ScalarField,
        proof: &TipaWithSsmProof<P>,
        transcript: &mut ProofTranscript,
    ) -> Result<bool, Error> {
        let (base_com, gipa_transcript) =
            Gipa::<P, ProofTranscript>::verify_recursive_challenge_transcript(
                (com.0, scalar_b, com.1),
                &proof.gipa_proof,
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
        let a_base = vec![proof.gipa_proof.final_message.0];
        let t_base = vec![MultiexponentiationInnerProduct::<P::G1>::inner_product(
            &a_base,
            &[b_base],
        )?];
        let base_valid = AfghoCommitment::verify(&[*ck_a_final], &a_base, &com_a)?
            && IdentityCommitment::<P::G1, P::ScalarField>::verify(&[], &t_base, &com_t)?;

        Ok(ck_a_valid && base_valid)
    }
}

// #[cfg(test)]
// mod tests {
//     use super::{
//         super::super::{
//             commitments::{
//                 afgho16::AfghoCommitment, identity::IdentityCommitment, random_generators,
//             },
//             inner_products::{InnerProduct, MultiexponentiationInnerProduct},
//         },
//         *,
//     };
//     use crate::poly::commitment::bmmtv::tipa::Field;
//     use crate::utils::transcript::KeccakTranscript;
//     use ark_bn254::Bn254;
//     use ark_std::rand::{rngs::StdRng, SeedableRng};
//
//     type BlsAfghoG1 = AfghoCommitment<Bn254>;
//     type BlsScalarField = <Bn254 as Pairing>::ScalarField;
//     type BlsG1 = <Bn254 as Pairing>::G1;
//
//     const TEST_SIZE: usize = 8;
//
//     fn structured_scalar_power<F: Field>(num: usize, s: &F) -> Vec<F> {
//         let mut powers = vec![F::one()];
//         for i in 1..num {
//             powers.push(powers[i - 1] * s);
//         }
//         powers
//     }
//
//     #[test]
//     fn tipa_ssm_multiexponentiation_inner_product_test() {
//         type IP = MultiexponentiationInnerProduct<BlsG1>;
//         type Ipc = IdentityCommitment<BlsG1, BlsScalarField>;
//         type MultiExpTipa = TipaWithSsm<IP, BlsAfghoG1, Ipc, Bn254, KeccakTranscript>;
//
//         let mut rng = StdRng::seed_from_u64(0u64);
//         let (srs, ck_t) = MultiExpTipa::setup(&mut rng, TEST_SIZE).unwrap();
//         let ck_a = srs.get_commitment_keys();
//         let v_srs = srs.get_verifier_key();
//         let m_a = random_generators(&mut rng, TEST_SIZE);
//         let b = BlsScalarField::rand(&mut rng);
//         let m_b = structured_scalar_power(TEST_SIZE, &b);
//         let com_a = BlsAfghoG1::commit(&ck_a, &m_a).unwrap();
//         let t = vec![IP::inner_product(&m_a, &m_b).unwrap()];
//         let com_t = Ipc::commit(&[ck_t.clone()], &t).unwrap();
//
//         let mut transcript = KeccakTranscript::new(b"TipaTest");
//
//         let proof = MultiExpTipa::prove_with_structured_scalar_message(
//             &srs.h_beta_powers,
//             (&m_a, &m_b),
//             (&ck_a, &ck_t),
//             &mut transcript,
//         )
//         .unwrap();
//
//         let mut transcript = KeccakTranscript::new(b"TipaTest");
//
//         assert!(MultiExpTipa::verify_with_structured_scalar_message(
//             &v_srs,
//             &ck_t,
//             (&com_a, &com_t),
//             &b,
//             &proof,
//             &mut transcript,
//         )
//         .unwrap());
//     }
// }
