use crate::poly::commitment::bmmtv::gipa::GipaProof;
use ark_ec::AffineRepr;
use std::marker::PhantomData;
use tracing::Level;

use ark_ec::{
    pairing::{Pairing, PairingOutput},
    CurveGroup,
};
use ark_ff::{Field, One};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::Itertools;
use rayon::prelude::*;

use super::{
    afgho::AfghoCommitment, gipa::CommitmentSteps, inner_products::MultiexponentiationInnerProduct,
    Error,
};
use crate::{
    field::JoltField,
    msm::Icicle,
    poly::{
        commitment::kzg::{KZGProverKey, KZGVerifierKey, UnivariateKZG, G2},
        unipoly::UniPoly,
    },
    utils::transcript::Transcript,
};

/// Calculates KZG opening in G2
pub fn prove_commitment_key_kzg_opening<P: Pairing>(
    srs_powers: &[P::G2],
    transcript: &[P::ScalarField],
    r_shift: P::ScalarField,
    point: P::ScalarField,
) -> Result<P::G2, Error>
where
    P::ScalarField: JoltField,
    P::G2: Icicle,
{
    let ck_polynomial =
        UniPoly::from_coeff(polynomial_coefficients_from_transcript(transcript, r_shift));
    assert_eq!(srs_powers.len(), ck_polynomial.coeffs.len());

    let ck_polynomial_c_eval =
        polynomial_evaluation_product_form_from_transcript(transcript, point, r_shift);

    let poly = ck_polynomial - UniPoly::from_coeff(vec![ck_polynomial_c_eval]);
    let powers = P::G2::normalize_batch(srs_powers);

    Ok(UnivariateKZG::<P, G2>::generic_open(
        &powers, None, &poly, point,
    )?)
}

pub fn verify_kzg_g2<P: Pairing>(
    v_srs: &KZGVerifierKey<P>,
    commitment: P::G2,
    proof: P::G2,
    transcript: &[P::ScalarField],
    r_shift: P::ScalarField,
    point: P::ScalarField,
) -> bool {
    let evaluation = polynomial_evaluation_product_form_from_transcript(transcript, point, r_shift);
    UnivariateKZG::<P, G2>::verify_g2(v_srs, commitment, point, proof, evaluation)
}

#[tracing::instrument(name = "polynomial eval", skip_all)]
fn polynomial_evaluation_product_form_from_transcript<F: Field>(
    transcript: &[F],
    z: F,
    r_shift: F,
) -> F {
    let mut power_2_zr = (z * z) * r_shift;
    let mut product_form = Vec::new();
    for x in transcript.iter().cloned() {
        product_form.push(F::one() + (x * power_2_zr));
        power_2_zr *= power_2_zr;
    }
    product_form.iter().product()
}

/// We create a polynomial using the transcript
/// This is why we need 2x srs for g2 we interleave it with zeroes
fn polynomial_coefficients_from_transcript<F: Field>(transcript: &[F], r_shift: F) -> Vec<F> {
    let mut coefficients = vec![F::one()];
    let mut power_2_r = r_shift;
    for (i, x) in transcript.iter().enumerate() {
        for j in 0..2_usize.pow(i as u32) {
            coefficients.push(coefficients[j] * (*x * power_2_r));
        }
        power_2_r *= power_2_r;
    }
    // Interleave with 0 coefficients
    coefficients
        .iter()
        .interleave([F::zero()].iter().cycle().take(coefficients.len() - 1))
        .cloned()
        .collect()
}

/// Multiexponentiation with known field vector
///
/// In the MIPPk protocol a prover demonstrates knowledge of [A] ∈ [G1] such
/// that A commits to pairing commitment T under *v* and U = A^b for a public vector [b] ∈ [F].
pub struct MippK<P, Transcript> {
    _pair: PhantomData<P>,
    _transcript: PhantomData<Transcript>,
}

/// Proof of [`MippK`]
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct MippKProof<P>
where
    P: Pairing,
{
    commitment_steps: CommitmentSteps<P>,
    final_message: (P::G1, P::ScalarField),
    final_ck: P::G2,
    final_ck_proof: P::G2,
}

impl<P, ProofTranscript> MippK<P, ProofTranscript>
where
    P: Pairing,
    P::G1: Icicle,
    P::G2: Icicle,
    P::ScalarField: JoltField,
    ProofTranscript: Transcript,
{
    #[tracing::instrument(name = "MippK::prove", skip_all)]
    pub fn prove(
        p_srs: &KZGProverKey<P>,
        values: (Vec<P::G1>, Vec<P::ScalarField>),
        transcript: &mut ProofTranscript,
    ) -> Result<MippKProof<P>, Error> {
        let h_beta_powers = p_srs
            .g2_powers()
            .iter()
            .map(|aff| aff.into_group())
            .collect::<Vec<_>>();
        let commitment_key = p_srs
            .g2_powers()
            .iter()
            .map(|aff| aff.into_group())
            .step_by(2)
            .collect();
        // Run GIPA
        let proof =
            GipaProof::<P, ProofTranscript>::prove(values.0, commitment_key, values.1, transcript)?;

        // Prove final commitment key is wellformed
        let (ck_a_final, ck_a_kzg_opening) = {
            let ck_kzg = tracing::span!(Level::TRACE, "Prove commitment key");
            let _guard = ck_kzg.enter();
            let ck_a_final = proof.final_commitment_param;
            let transcript_inverse = proof
                .scalar_transcript
                .par_iter()
                .map(|x| JoltField::inverse(x).unwrap())
                .collect::<Vec<_>>();

            // KZG challenge point
            transcript.append_point(&ck_a_final);
            let c: P::ScalarField = transcript.challenge_scalar();

            // Complete KZG proof
            let ck_a_kzg_opening = prove_commitment_key_kzg_opening::<P>(
                &h_beta_powers,
                &transcript_inverse,
                P::ScalarField::one(), // r_shift = one, why?
                c,
            )?;
            (ck_a_final, ck_a_kzg_opening)
        };

        Ok(MippKProof {
            final_message: proof.final_message,
            commitment_steps: proof.commitment_steps,
            final_ck: ck_a_final,
            final_ck_proof: ck_a_kzg_opening,
        })
    }

    pub fn verify(
        v_srs: &KZGVerifierKey<P>,
        com: (PairingOutput<P>, P::G1),
        scalar_b: P::ScalarField,
        proof: &MippKProof<P>,
        transcript: &mut ProofTranscript,
    ) -> Result<bool, Error> {
        let (base_com, gipa_transcript) =
            GipaProof::<P, ProofTranscript>::verify(com, &proof.commitment_steps, transcript)?;
        let transcript_inverse = gipa_transcript
            .par_iter()
            .map(|x| JoltField::inverse(x).unwrap())
            .collect::<Vec<_>>();

        // KZG challenge point
        transcript.append_point(&proof.final_ck);
        let c = transcript.challenge_scalar();

        // Check commitment key
        let ck_a_valid = verify_kzg_g2(
            v_srs,
            proof.final_ck,
            proof.final_ck_proof,
            &transcript_inverse,
            P::ScalarField::one(),
            c,
        );

        // Compute final scalar
        let mut power_2_b = scalar_b;
        let mut product_form = Vec::new();
        for x in transcript_inverse.iter() {
            product_form.push(<P::ScalarField>::one() + (*x * power_2_b));
            power_2_b *= power_2_b;
        }
        let b_base = product_form.par_iter().product::<P::ScalarField>();

        // Verify base inner product commitment
        let (com_a, com_t) = base_com;
        let a_base = vec![proof.final_message.0];
        let t_base = MultiexponentiationInnerProduct::inner_product(&a_base, &[b_base])?;

        let same_ip_commit = t_base == com_t;
        let base_valid =
            AfghoCommitment::verify(&[proof.final_ck], &a_base, &com_a)? && same_ip_commit;

        Ok(ck_a_valid && base_valid)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ark_bn254::Bn254;
    use ark_std::{
        rand::{rngs::StdRng, SeedableRng},
        UniformRand,
    };

    use super::{
        super::afgho::{random_generators, AfghoCommitment},
        *,
    };
    use crate::{
        poly::commitment::{bmmtv::mipp_k::Field, kzg::SRS},
        utils::transcript::KeccakTranscript,
    };

    type BnAfghoG1 = AfghoCommitment<Bn254>;
    type BnScalarField = <Bn254 as Pairing>::ScalarField;

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
        type MultiExpTipa = MippK<Bn254, KeccakTranscript>;

        let mut rng = StdRng::seed_from_u64(0u64);
        let srs = SRS::setup(&mut rng, 2 * (TEST_SIZE - 1), 2 * (TEST_SIZE - 1));

        let powers_len = srs.g1_powers.len();
        let (p_srs, v_srs) = SRS::trim(Arc::new(srs), powers_len - 1);

        let ck_a = p_srs
            .g2_powers()
            .iter()
            .map(|aff: &<Bn254 as Pairing>::G2Affine| aff.into_group())
            .step_by(2)
            .collect::<Vec<_>>();

        let m_a = random_generators(&mut rng, TEST_SIZE);
        let b = BnScalarField::rand(&mut rng);
        let m_b = structured_scalar_power(TEST_SIZE, &b);
        let com_a = BnAfghoG1::commit(&ck_a, &m_a).unwrap();
        let com_t = MultiexponentiationInnerProduct::inner_product(&m_a, &m_b).unwrap();

        let mut transcript = KeccakTranscript::new(b"TipaTest");

        let proof = MultiExpTipa::prove(&p_srs, (m_a, m_b), &mut transcript).unwrap();

        let mut transcript = KeccakTranscript::new(b"TipaTest");

        assert!(MultiExpTipa::verify(&v_srs, (com_a, com_t), b, &proof, &mut transcript,).unwrap());
    }
}
