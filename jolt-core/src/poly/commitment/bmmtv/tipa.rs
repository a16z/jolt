use ark_ec::scalar_mul::fixed_base::FixedBase;
use ark_ec::Group;
use ark_ec::{pairing::Pairing, CurveGroup};
use ark_ff::{Field, One, PrimeField, UniformRand, Zero};
use ark_poly::polynomial::{univariate::DensePolynomial, DenseUVPolynomial};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::Rng;
use ark_std::{end_timer, start_timer};
use digest::Digest;
use itertools::Itertools;
use std::{marker::PhantomData, ops::MulAssign};

use super::{
    commitments::Dhc,
    gipa::GipaParams,
    gipa::{Gipa, GipaProof},
    inner_products::{InnerProduct, MultiexponentiationInnerProduct},
    Error,
};

pub mod structured_scalar_message;

//TODO: Could generalize: Don't need TIPA over G1 and G2, would work with G1 and G1 or over different pairing engines

//TODO: May need to add "reverse" MultiexponentiationInnerProduct to allow for MIP with G2 messages (because TIP hard-coded G1 left and G2 right)
pub struct Tipa<IP, LMC, RMC, IPC, P, D> {
    _inner_product: PhantomData<IP>,
    _left_commitment: PhantomData<LMC>,
    _right_commitment: PhantomData<RMC>,
    _inner_product_commitment: PhantomData<IPC>,
    _pair: PhantomData<P>,
    _digest: PhantomData<D>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone)]
pub struct TipaProof<LCom, RCom, IpCom, P>
where
    P: Pairing,
    LCom: Dhc,
    RCom: Dhc,
    IpCom: Dhc,
{
    gipa_proof: GipaProof<LCom, RCom, IpCom>,
    final_ck: (LCom::Param, RCom::Param),
    final_ck_proof: (P::G2, P::G1),
}

/// Structured Reference String
///
/// This is also known as the trusted setup
#[derive(Clone)]
pub struct Srs<P: Pairing> {
    pub g_alpha_powers: Vec<P::G1>,
    pub h_beta_powers: Vec<P::G2>,
    pub g_beta: P::G1,
    pub h_alpha: P::G2,
}

#[derive(Clone)]
pub struct VerifierSrs<P: Pairing> {
    pub g: P::G1,
    pub h: P::G2,
    pub g_beta: P::G1,
    pub h_alpha: P::G2,
}

//TODO: Change SRS to return reference iterator - requires changes to TIPA and GIPA signatures
impl<P: Pairing> Srs<P> {
    pub fn get_commitment_keys(&self) -> (Vec<P::G2>, Vec<P::G1>) {
        let ck_1 = self.h_beta_powers.iter().step_by(2).cloned().collect();
        let ck_2 = self.g_alpha_powers.iter().step_by(2).cloned().collect();
        (ck_1, ck_2)
    }

    pub fn get_verifier_key(&self) -> VerifierSrs<P> {
        VerifierSrs {
            g: self.g_alpha_powers[0],
            h: self.h_beta_powers[0],
            g_beta: self.g_beta,
            h_alpha: self.h_alpha,
        }
    }
}

impl<Ip, LCom, RCom, IpCom, P, D> Tipa<Ip, LCom, RCom, IpCom, P, D>
where
    D: Digest,
    P: Pairing,
    Ip: InnerProduct<
        LeftMessage = LCom::Message,
        RightMessage = RCom::Message,
        Output = IpCom::Message,
    >,
    LCom: Dhc<Scalar = P::ScalarField, Param = P::G2>,
    RCom: Dhc<Scalar = LCom::Scalar, Param = P::G1>,
    IpCom: Dhc<Scalar = LCom::Scalar>,
    LCom::Message: MulAssign<P::ScalarField>,
    RCom::Message: MulAssign<P::ScalarField>,
    IpCom::Message: MulAssign<P::ScalarField>,
    IpCom::Param: MulAssign<P::ScalarField>,
    LCom::Output: MulAssign<P::ScalarField>,
    RCom::Output: MulAssign<P::ScalarField>,
    IpCom::Output: MulAssign<P::ScalarField>,
{
    pub fn setup<R: Rng>(rng: &mut R, size: usize) -> Result<(Srs<P>, IpCom::Param), Error> {
        let alpha = <P::ScalarField>::rand(rng);
        let beta = <P::ScalarField>::rand(rng);
        let g = P::G1::generator();
        let h = P::G2::generator();
        Ok((
            Srs {
                g_alpha_powers: structured_generators_scalar_power(2 * size - 1, &g, &alpha),
                h_beta_powers: structured_generators_scalar_power(2 * size - 1, &h, &beta),
                g_beta: g * beta,
                h_alpha: h * alpha,
            },
            IpCom::setup(rng, 1)?.pop().unwrap(),
        ))
    }

    pub fn prove(
        srs: &Srs<P>,
        values: (&[Ip::LeftMessage], &[Ip::RightMessage]),
        ck: (&[LCom::Param], &[RCom::Param], &IpCom::Param),
    ) -> Result<TipaProof<LCom, RCom, IpCom, P>, Error> {
        Self::prove_with_srs_shift(srs, values, ck, &<P::ScalarField>::one())
    }

    // Shifts KZG proof for left message by scalar r (used for efficient composition with aggregation protocols)
    // LMC commitment key should already be shifted before being passed as input
    pub fn prove_with_srs_shift(
        srs: &Srs<P>,
        values: (&[Ip::LeftMessage], &[Ip::RightMessage]),
        ck: (&[LCom::Param], &[RCom::Param], &IpCom::Param),
        r_shift: &P::ScalarField,
    ) -> Result<TipaProof<LCom, RCom, IpCom, P>, Error> {
        // Run GIPA
        let (proof, aux) = <Gipa<Ip, LCom, RCom, IpCom, D>>::prove_with_aux(
            values,
            &GipaParams::new_aux(ck.0, ck.1, &[ck.2.clone()]),
        )?;

        // Prove final commitment keys are wellformed
        let (ck_a_final, ck_b_final) = aux.final_commitment_param;
        let transcript = aux.scalar_transcript;
        let transcript_inverse = transcript
            .iter()
            .map(|x| x.inverse().unwrap())
            .collect::<Vec<_>>();
        let r_inverse = r_shift.inverse().unwrap();

        // KZG challenge point
        let mut counter_nonce: usize = 0;
        let c = loop {
            let mut hash_input = Vec::new();
            hash_input.extend_from_slice(&counter_nonce.to_be_bytes()[..]);
            transcript
                .first()
                .unwrap()
                .serialize_uncompressed(&mut hash_input)?;
            ck_a_final.serialize_uncompressed(&mut hash_input)?;
            ck_b_final.serialize_uncompressed(&mut hash_input)?;
            if let Some(c) = LCom::Scalar::from_random_bytes(&D::digest(&hash_input)) {
                break c;
            };
            counter_nonce += 1;
        };

        // Complete KZG proofs
        let ck_a_kzg_opening = prove_commitment_key_kzg_opening(
            &srs.h_beta_powers,
            &transcript_inverse,
            &r_inverse,
            &c,
        )?;
        let ck_b_kzg_opening = prove_commitment_key_kzg_opening(
            &srs.g_alpha_powers,
            &transcript,
            &<P::ScalarField>::one(),
            &c,
        )?;

        Ok(TipaProof {
            gipa_proof: proof,
            final_ck: (ck_a_final, ck_b_final),
            final_ck_proof: (ck_a_kzg_opening, ck_b_kzg_opening),
        })
    }

    pub fn verify(
        v_srs: &VerifierSrs<P>,
        ck_t: &IpCom::Param,
        com: (&LCom::Output, &RCom::Output, &IpCom::Output),
        proof: &TipaProof<LCom, RCom, IpCom, P>,
    ) -> Result<bool, Error> {
        Self::verify_with_srs_shift(v_srs, ck_t, com, proof, &<P::ScalarField>::one())
    }

    pub fn verify_with_srs_shift(
        v_srs: &VerifierSrs<P>,
        ck_t: &IpCom::Param,
        com: (&LCom::Output, &RCom::Output, &IpCom::Output),
        proof: &TipaProof<LCom, RCom, IpCom, P>,
        r_shift: &P::ScalarField,
    ) -> Result<bool, Error> {
        let (base_com, transcript) =
            Gipa::<Ip, LCom, RCom, IpCom, D>::verify_recursive_challenge_transcript(
                com,
                &proof.gipa_proof,
            )?;
        let transcript_inverse = transcript
            .iter()
            .map(|x| x.inverse().unwrap())
            .collect::<Vec<_>>();

        // Verify commitment keys wellformed
        let (ck_a_final, ck_b_final) = &proof.final_ck;
        let (ck_a_proof, ck_b_proof) = &proof.final_ck_proof;

        // KZG challenge point
        let mut counter_nonce: usize = 0;
        let c = loop {
            let mut hash_input = Vec::new();
            hash_input.extend_from_slice(&counter_nonce.to_be_bytes()[..]);
            transcript
                .first()
                .unwrap()
                .serialize_uncompressed(&mut hash_input)?;
            ck_a_final.serialize_uncompressed(&mut hash_input)?;
            ck_b_final.serialize_uncompressed(&mut hash_input)?;
            if let Some(c) = LCom::Scalar::from_random_bytes(&D::digest(&hash_input)) {
                break c;
            };
            counter_nonce += 1;
        };

        let ck_a_valid = verify_commitment_key_g2_kzg_opening(
            v_srs,
            ck_a_final,
            ck_a_proof,
            &transcript_inverse,
            &r_shift.inverse().unwrap(),
            &c,
        )?;
        let ck_b_valid = verify_commitment_key_g1_kzg_opening(
            v_srs,
            ck_b_final,
            ck_b_proof,
            &transcript,
            &<P::ScalarField>::one(),
            &c,
        )?;

        // Verify base inner product commitment
        let (com_a, com_b, com_t) = base_com;
        let a_base = vec![proof.gipa_proof.final_message.0.clone()];
        let b_base = vec![proof.gipa_proof.final_message.1.clone()];
        let t_base = vec![Ip::inner_product(&a_base, &b_base)?];
        let base_valid = LCom::verify(&[*ck_a_final], &a_base, &com_a)?
            && RCom::verify(&[*ck_b_final], &b_base, &com_b)?
            && IpCom::verify(&[ck_t.clone()], &t_base, &com_t)?;

        Ok(ck_a_valid && ck_b_valid && base_valid)
    }
}

/// Returns the proof that the polynomial
///
/// Calculate commitment of quotient polynomial such that w(X).(X-z) = P(X) - v
///
/// `kzg_challenge`: X
/// `srs_powers`:
pub fn prove_commitment_key_kzg_opening<G: CurveGroup>(
    srs_powers: &[G],
    transcript: &[G::ScalarField],
    r_shift: &G::ScalarField,
    kzg_challenge: &G::ScalarField,
) -> Result<G, Error> {
    let ck_polynomial = DensePolynomial::from_coefficients_slice(
        &polynomial_coefficients_from_transcript(transcript, r_shift),
    );
    assert_eq!(srs_powers.len(), ck_polynomial.coeffs.len());

    let eval = start_timer!(|| "polynomial eval");
    let ck_polynomial_c_eval =
        polynomial_evaluation_product_form_from_transcript(transcript, kzg_challenge, r_shift);
    end_timer!(eval);

    let quotient = start_timer!(|| "polynomial quotient");
    let quotient_polynomial = &(&ck_polynomial
        - &DensePolynomial::from_coefficients_vec(vec![ck_polynomial_c_eval]))
        / &(DensePolynomial::from_coefficients_vec(vec![-*kzg_challenge, <G::ScalarField>::one()]));
    end_timer!(quotient);

    let mut quotient_polynomial_coeffs = quotient_polynomial.coeffs;
    quotient_polynomial_coeffs.resize(srs_powers.len(), <G::ScalarField>::zero());

    let multiexp = start_timer!(|| "opening multiexp");
    let opening =
        MultiexponentiationInnerProduct::inner_product(srs_powers, &quotient_polynomial_coeffs)?;
    end_timer!(multiexp);
    Ok(opening)
}

//TODO: Figure out how to avoid needing two separate methods for verification of opposite groups
pub fn verify_commitment_key_g2_kzg_opening<P: Pairing>(
    v_srs: &VerifierSrs<P>,
    ck_final: &P::G2,
    ck_opening: &P::G2,
    transcript: &[P::ScalarField],
    r_shift: &P::ScalarField,
    kzg_challenge: &P::ScalarField,
) -> Result<bool, Error> {
    let ck_polynomial_c_eval =
        polynomial_evaluation_product_form_from_transcript(transcript, kzg_challenge, r_shift);
    Ok(
        P::pairing(v_srs.g, *ck_final - v_srs.h * ck_polynomial_c_eval)
            == P::pairing(v_srs.g_beta - v_srs.g * kzg_challenge, *ck_opening),
    )
}

pub fn verify_commitment_key_g1_kzg_opening<P: Pairing>(
    v_srs: &VerifierSrs<P>,
    ck_final: &P::G1,
    ck_opening: &P::G1,
    transcript: &[P::ScalarField],
    r_shift: &P::ScalarField,
    kzg_challenge: &P::ScalarField,
) -> Result<bool, Error> {
    let ck_polynomial_c_eval =
        polynomial_evaluation_product_form_from_transcript(transcript, kzg_challenge, r_shift);
    Ok(
        P::pairing(*ck_final - v_srs.g * ck_polynomial_c_eval, v_srs.h)
            == P::pairing(*ck_opening, v_srs.h_alpha - v_srs.h * kzg_challenge),
    )
}

pub fn structured_generators_scalar_power<G: CurveGroup>(
    num: usize,
    g: &G,
    s: &G::ScalarField,
) -> Vec<G> {
    assert!(num > 0);
    let mut powers_of_scalar = vec![];
    let mut pow_s = G::ScalarField::one();
    for _ in 0..num {
        powers_of_scalar.push(pow_s);
        pow_s *= s;
    }

    let window_size = FixedBase::get_mul_window_size(num);

    let scalar_bits = G::ScalarField::MODULUS_BIT_SIZE as usize;
    let g_table = FixedBase::get_window_table(scalar_bits, window_size, g.clone());
    let powers_of_g = FixedBase::msm::<G>(scalar_bits, window_size, &g_table, &powers_of_scalar);
    powers_of_g
}

fn polynomial_evaluation_product_form_from_transcript<F: Field>(
    transcript: &[F],
    z: &F,
    r_shift: &F,
) -> F {
    let mut power_2_zr = (*z * z) * r_shift;
    let mut product_form = Vec::new();
    for x in transcript.iter() {
        product_form.push(F::one() + (*x * power_2_zr));
        power_2_zr *= power_2_zr;
    }
    product_form.iter().product()
}

fn polynomial_coefficients_from_transcript<F: Field>(transcript: &[F], r_shift: &F) -> Vec<F> {
    let mut coefficients = vec![F::one()];
    let mut power_2_r = *r_shift;
    for (i, x) in transcript.iter().enumerate() {
        for j in 0..(2_usize).pow(i as u32) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::commitment::bmmtv::{
        commitments::{
            afgho16::{AfghoCommitmentG1, AfghoCommitmentG2},
            identity::IdentityCommitment,
            pedersen::PedersenCommitment,
            random_generators,
        },
        inner_products::{
            InnerProduct, MultiexponentiationInnerProduct, PairingInnerProduct, ScalarInnerProduct,
        },
        tipa::structured_scalar_message::structured_scalar_power,
    };
    use ark_bn254::Bn254;
    use ark_ec::pairing::PairingOutput;
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use sha3::Sha3_256;

    type BlsScalarField = <Bn254 as Pairing>::ScalarField;
    type BnG1 = <Bn254 as Pairing>::G1;
    type BnG2 = <Bn254 as Pairing>::G2;

    type BlsAfghoG1 = AfghoCommitmentG1<Bn254>;
    type BlsAfghoG2 = AfghoCommitmentG2<Bn254>;
    type BlsPedersenG1 = PedersenCommitment<BnG1>;
    type BlsPedersenG2 = PedersenCommitment<BnG2>;

    const TEST_SIZE: usize = 8;

    #[test]
    fn pairing_inner_product_test() {
        type PairingInnerProd = PairingInnerProduct<Bn254>;
        type Identity = IdentityCommitment<PairingOutput<Bn254>, BlsScalarField>;
        type PairingTipa =
            Tipa<PairingInnerProd, BlsAfghoG1, BlsAfghoG2, Identity, Bn254, Sha3_256>;

        let mut rng = StdRng::seed_from_u64(0u64);
        let (srs, ck_t) = PairingTipa::setup(&mut rng, TEST_SIZE).unwrap();
        let (ck_a, ck_b) = srs.get_commitment_keys();
        let v_srs = srs.get_verifier_key();
        let m_a = random_generators(&mut rng, TEST_SIZE);
        let m_b = random_generators(&mut rng, TEST_SIZE);
        let com_a = BlsAfghoG1::commit(&ck_a, &m_a).unwrap();
        let com_b = BlsAfghoG2::commit(&ck_b, &m_b).unwrap();
        let t = vec![PairingInnerProd::inner_product(&m_a, &m_b).unwrap()];
        let com_t = Identity::commit(&[ck_t.clone()], &t).unwrap();

        let proof = PairingTipa::prove(&srs, (&m_a, &m_b), (&ck_a, &ck_b, &ck_t)).unwrap();

        assert!(PairingTipa::verify(&v_srs, &ck_t, (&com_a, &com_b, &com_t), &proof).unwrap());
    }

    #[test]
    fn multiexponentiation_inner_product_test() {
        type MultiExpInnerProd = MultiexponentiationInnerProduct<BnG1>;
        type Identity = IdentityCommitment<BnG1, BlsScalarField>;
        type MultiExpTipa =
            Tipa<MultiExpInnerProd, BlsAfghoG1, BlsPedersenG1, Identity, Bn254, Sha3_256>;

        let mut rng = StdRng::seed_from_u64(0u64);
        let (srs, ck_t) = MultiExpTipa::setup(&mut rng, TEST_SIZE).unwrap();
        let (ck_a, ck_b) = srs.get_commitment_keys();
        let v_srs = srs.get_verifier_key();
        let m_a = random_generators(&mut rng, TEST_SIZE);
        let mut m_b = Vec::new();
        for _ in 0..TEST_SIZE {
            m_b.push(BlsScalarField::rand(&mut rng));
        }
        let com_a = BlsAfghoG1::commit(&ck_a, &m_a).unwrap();
        let com_b = BlsPedersenG1::commit(&ck_b, &m_b).unwrap();
        let t = vec![MultiExpInnerProd::inner_product(&m_a, &m_b).unwrap()];
        let com_t = Identity::commit(&[ck_t.clone()], &t).unwrap();

        let proof = MultiExpTipa::prove(&srs, (&m_a, &m_b), (&ck_a, &ck_b, &ck_t)).unwrap();

        assert!(MultiExpTipa::verify(&v_srs, &ck_t, (&com_a, &com_b, &com_t), &proof).unwrap());
    }

    #[test]
    fn scalar_inner_product_test() {
        type ScalarInnerProd = ScalarInnerProduct<BlsScalarField>;
        type Identity = IdentityCommitment<BlsScalarField, BlsScalarField>;
        type ScalarTipa =
            Tipa<ScalarInnerProd, BlsPedersenG2, BlsPedersenG1, Identity, Bn254, Sha3_256>;

        let mut rng = StdRng::seed_from_u64(0u64);
        let (srs, ck_t) = ScalarTipa::setup(&mut rng, TEST_SIZE).unwrap();
        let (ck_a, ck_b) = srs.get_commitment_keys();
        let v_srs = srs.get_verifier_key();
        let mut m_a = Vec::new();
        let mut m_b = Vec::new();
        for _ in 0..TEST_SIZE {
            m_a.push(BlsScalarField::rand(&mut rng));
            m_b.push(BlsScalarField::rand(&mut rng));
        }
        let com_a = BlsPedersenG2::commit(&ck_a, &m_a).unwrap();
        let com_b = BlsPedersenG1::commit(&ck_b, &m_b).unwrap();
        let t = vec![ScalarInnerProd::inner_product(&m_a, &m_b).unwrap()];
        let com_t = Identity::commit(&[ck_t.clone()], &t).unwrap();

        let proof = ScalarTipa::prove(&srs, (&m_a, &m_b), (&ck_a, &ck_b, &ck_t)).unwrap();

        assert!(ScalarTipa::verify(&v_srs, &ck_t, (&com_a, &com_b, &com_t), &proof).unwrap());
    }

    #[test]
    fn pairing_inner_product_with_srs_shift_test() {
        type PairingInnerProd = PairingInnerProduct<Bn254>;
        type Identity = IdentityCommitment<PairingOutput<Bn254>, BlsScalarField>;
        type PairingTipa =
            Tipa<PairingInnerProd, BlsAfghoG1, BlsAfghoG2, Identity, Bn254, Sha3_256>;

        let mut rng = StdRng::seed_from_u64(0u64);
        let (srs, ck_t) = PairingTipa::setup(&mut rng, TEST_SIZE).unwrap();
        let (ck_a, ck_b) = srs.get_commitment_keys();
        let v_srs = srs.get_verifier_key();

        let m_a = random_generators(&mut rng, TEST_SIZE);
        let m_b = random_generators(&mut rng, TEST_SIZE);
        let com_a = BlsAfghoG1::commit(&ck_a, &m_a).unwrap();
        let com_b = BlsAfghoG2::commit(&ck_b, &m_b).unwrap();

        let r_scalar = BlsScalarField::rand(&mut rng);
        let r_vec = structured_scalar_power(TEST_SIZE, &r_scalar);
        let m_a_r = m_a
            .iter()
            .zip(&r_vec)
            .map(|(&a, r)| a * r)
            .collect::<Vec<BnG1>>();
        let ck_a_r = ck_a
            .iter()
            .zip(&r_vec)
            .map(|(&ck, r)| ck * r.inverse().unwrap())
            .collect::<Vec<BnG2>>();

        let t = vec![PairingInnerProd::inner_product(&m_a_r, &m_b).unwrap()];
        let com_t = Identity::commit(&[ck_t.clone()], &t).unwrap();

        assert_eq!(
            com_a,
            PairingInnerProd::inner_product(&m_a_r, &ck_a_r).unwrap()
        );

        let proof = PairingTipa::prove_with_srs_shift(
            &srs,
            (&m_a_r, &m_b),
            (&ck_a_r, &ck_b, &ck_t),
            &r_scalar,
        )
        .unwrap();

        assert!(PairingTipa::verify_with_srs_shift(
            &v_srs,
            &ck_t,
            (&com_a, &com_b, &com_t),
            &proof,
            &r_scalar
        )
        .unwrap());
    }
}
