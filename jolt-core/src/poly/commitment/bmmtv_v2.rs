use std::{marker::PhantomData, sync::Arc};

use afgho::AFGHOCommitmentG1;
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::Field;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{cfg_iter, end_timer, start_timer, One, Zero};
use eyre::Error;
use rand_core::{CryptoRng, RngCore};

use crate::{field::JoltField, msm::Icicle, poly::unipoly::UniPoly};

use super::kzg::{KZGProverKey, KZGVerifierKey, UnivariateKZG, SRS};

mod afgho;

pub struct BivariatePolynomial<F> {
    y_polynomials: Vec<UniPoly<F>>,
}

impl<F: JoltField> BivariatePolynomial<F> {
    pub fn evaluate(&self, (x, y): &(F, F)) -> F {
        let mut point_x_powers = vec![];
        let mut cur = F::one();
        for _ in 0..(self.y_polynomials.len()) {
            point_x_powers.push(cur);
            cur *= *x;
        }
        point_x_powers
            .iter()
            .zip(&self.y_polynomials)
            .map(|(x_power, y_polynomial)| {
                let var_name = y;
                *x_power * y_polynomial.evaluate(var_name)
            })
            .sum()
    }
}

pub struct OpeningProof<P: Pairing> {
    ip_proof: TIPAWithSSMProof<P>,
    y_eval_comm: P::G1Affine,
    kzg_proof: P::G1Affine,
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct TIPAWithSSMProof<P>
where
    P: Pairing,
{
    gipa_proof: GIPAProof<P>,
    final_ck: P::G2,
    final_ck_proof: P::G2,
}

/// Why is this needed?
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Default, Eq, PartialEq)]
pub struct HomomorphicPlaceholderValue;

// pub fn prove_with_aux<P: Pairing>(
//     values: (&[P::G1], &[P::ScalarField]),
//     ck: (
//         &[P::G2],
//         &[HomomorphicPlaceholderValue],
//         &[HomomorphicPlaceholderValue],
//     ),
// ) -> Result<(GIPAProof<P>, GIPAAux<P>), Error> {
//     let (m_a, m_b) = values;
//     let (ck_a, ck_b, ck_t) = ck;
//     let (mut m_a, mut m_b) = values;
//     let (mut ck_a, mut ck_b, ck_t) = ck;
//     let mut r_commitment_steps = Vec::new();
//     let mut r_transcript: Vec<P::ScalarField> = Vec::new();
//     assert!(m_a.len().is_power_of_two());
//     let (m_base, ck_base) = 'recurse: loop {
//         let recurse = start_timer!(|| format!("Recurse round size {}", m_a.len()));
//         if m_a.len() == 1 {
//             // base case
//             break 'recurse (
//                 (m_a[0].clone(), m_b[0].clone()),
//                 (ck_a[0].clone(), ck_b[0].clone()),
//             );
//         } else {
//             // recursive step
//             // Recurse with problem of half size
//             let split = m_a.len() / 2;
//
//             let m_a_1 = &m_a[split..];
//             let m_a_2 = &m_a[..split];
//             let ck_a_1 = &ck_a[..split];
//             let ck_a_2 = &ck_a[split..];
//
//             let m_b_1 = &m_b[..split];
//             let m_b_2 = &m_b[split..];
//             let ck_b_1 = &ck_b[split..];
//             let ck_b_2 = &ck_b[..split];
//
//             let cl = start_timer!(|| "Commit L");
//             let com_1 = (
//                 LMC::commit(ck_a_1, m_a_1)?,
//                 RMC::commit(ck_b_1, m_b_1)?,
//                 IPC::commit(&ck_t, &[IP::inner_product(m_a_1, m_b_1)?])?,
//             );
//             end_timer!(cl);
//             let cr = start_timer!(|| "Commit R");
//             let com_2 = (
//                 LMC::commit(ck_a_2, m_a_2)?,
//                 RMC::commit(ck_b_2, m_b_2)?,
//                 IPC::commit(&ck_t, &[IP::inner_product(m_a_2, m_b_2)?])?,
//             );
//             end_timer!(cr);
//
//             // Fiat-Shamir challenge
//             let mut counter_nonce: usize = 0;
//             let default_transcript = Default::default();
//             let transcript = r_transcript.last().unwrap_or(&default_transcript);
//             let (c, c_inv) = 'challenge: loop {
//                 let mut hash_input = Vec::new();
//                 hash_input.extend_from_slice(&counter_nonce.to_be_bytes()[..]);
//                 transcript.serialize_uncompressed(&mut hash_input)?;
//                 com_1.0.serialize_uncompressed(&mut hash_input)?;
//                 com_1.1.serialize_uncompressed(&mut hash_input)?;
//                 com_1.2.serialize_uncompressed(&mut hash_input)?;
//                 com_2.0.serialize_uncompressed(&mut hash_input)?;
//                 com_2.1.serialize_uncompressed(&mut hash_input)?;
//                 com_2.2.serialize_uncompressed(&mut hash_input)?;
//                 let c: LMC::Scalar = u128::from_be_bytes(
//                     D::digest(&hash_input).as_slice()[0..16].try_into().unwrap(),
//                 )
//                 .into();
//                 if let Some(c_inv) = c.inverse() {
//                     // Optimization for multiexponentiation to rescale G2 elements with 128-bit challenge
//                     // Swap 'c' and 'c_inv' since can't control bit size of c_inv
//                     break 'challenge (c_inv, c);
//                 }
//                 counter_nonce += 1;
//             };
//
//             // Set up values for next step of recursion
//             let rescale_m1 = start_timer!(|| "Rescale M1");
//             m_a = cfg_iter!(m_a_1)
//                 .map(|a| mul_helper(a, &c))
//                 .zip(m_a_2)
//                 .map(|(a_1, a_2)| a_1 + a_2.clone())
//                 .collect::<Vec<LMC::Message>>();
//             end_timer!(rescale_m1);
//
//             let rescale_m2 = start_timer!(|| "Rescale M2");
//             m_b = cfg_iter!(m_b_2)
//                 .map(|b| mul_helper(b, &c_inv))
//                 .zip(m_b_1)
//                 .map(|(b_1, b_2)| b_1 + b_2.clone())
//                 .collect::<Vec<RMC::Message>>();
//             end_timer!(rescale_m2);
//
//             let rescale_ck1 = start_timer!(|| "Rescale CK1");
//             ck_a = cfg_iter!(ck_a_2)
//                 .map(|a| mul_helper(a, &c_inv))
//                 .zip(ck_a_1)
//                 .map(|(a_1, a_2)| a_1 + a_2.clone())
//                 .collect::<Vec<LMC::Key>>();
//             end_timer!(rescale_ck1);
//
//             let rescale_ck2 = start_timer!(|| "Rescale CK2");
//             ck_b = cfg_iter!(ck_b_1)
//                 .map(|b| mul_helper(b, &c))
//                 .zip(ck_b_2)
//                 .map(|(b_1, b_2)| b_1 + b_2.clone())
//                 .collect::<Vec<RMC::Key>>();
//             end_timer!(rescale_ck2);
//
//             r_commitment_steps.push((com_1, com_2));
//             r_transcript.push(c);
//             end_timer!(recurse);
//         }
//     };
//     r_transcript.reverse();
//     r_commitment_steps.reverse();
//     Ok((
//         GIPAProof {
//             r_commitment_steps,
//             r_base: m_base,
//         },
//         GIPAAux {
//             r_transcript,
//             ck_base,
//         },
//     ))
// }

// pub fn prove_commitment_key_kzg_opening<G: AffineRepr>(
//     srs_powers: &[G],
//     transcript: &[G::ScalarField],
//     r_shift: &G::ScalarField,
//     kzg_challenge: &G::ScalarField,
// ) -> Result<G, Error> {
//     let ck_polynomial = UniPoly::from_coeff(&polynomial_coefficients_from_transcript(
//         transcript, r_shift,
//     ));
//     assert_eq!(srs_powers.len(), ck_polynomial.coeffs.len());
//
//     let eval = start_timer!(|| "polynomial eval");
//     let ck_polynomial_c_eval =
//         polynomial_evaluation_product_form_from_transcript(transcript, kzg_challenge, r_shift);
//     end_timer!(eval);
//
//     let quotient = start_timer!(|| "polynomial quotient");
//     let quotient_polynomial = &(&ck_polynomial
//         - &DensePolynomial::from_coefficients_vec(vec![ck_polynomial_c_eval]))
//         / &(DensePolynomial::from_coefficients_vec(vec![-*kzg_challenge, <G::ScalarField>::one()]));
//     end_timer!(quotient);
//
//     let mut quotient_polynomial_coeffs = quotient_polynomial.coeffs;
//     quotient_polynomial_coeffs.resize(srs_powers.len(), <G::ScalarField>::zero());
//
//     let multiexp = start_timer!(|| "opening multiexp");
//     let opening =
//         MultiexponentiationInnerProduct::inner_product(srs_powers, &quotient_polynomial_coeffs);
//     end_timer!(multiexp);
//     opening
// }

// impl<P: Pairing> TIPAWithSSMProof<P>
// where
//     P::G1: Icicle,
// {
//     pub fn prove_with_structured_scalar_message(
//         srs: &SRS<P>,
//         values: (&[P::G1], &[P::ScalarField]),
//         ck: (&[P::G2], &HomomorphicPlaceholderValue),
//     ) -> Result<Self, Error> {
//         // Run GIPA
//         let gipa = start_timer!(|| "GIPA");
//         let (proof, aux) = prove_with_aux(
//             values,
//             (
//                 ck.0,
//                 &vec![HomomorphicPlaceholderValue {}; values.1.len()],
//                 &[ck.1.clone()],
//             ),
//         )?;
//         end_timer!(gipa);
//
//         // Prove final commitment key is wellformed
//         let ck_kzg = start_timer!(|| "Prove commitment key");
//         let (ck_a_final, _) = aux.ck_base;
//         let transcript = aux.r_transcript;
//         let transcript_inverse: Vec<_> = cfg_iter!(transcript)
//             .map(|x: &P::ScalarField| x.inverse().unwrap())
//             .collect();
//
//         // KZG challenge point
//         let mut counter_nonce: usize = 0;
//         // Take from Transcript!
//         let c = todo!();
//         // let c = loop {
//         //     let mut hash_input = Vec::new();
//         //     hash_input.extend_from_slice(&counter_nonce.to_be_bytes()[..]);
//         //     transcript
//         //         .first()
//         //         .unwrap()
//         //         .serialize_uncompressed(&mut hash_input)?;
//         //     ck_a_final.serialize_uncompressed(&mut hash_input)?;
//         //     if let Some(c) = LMC::Scalar::from_random_bytes(&D::digest(&hash_input)) {
//         //         break c;
//         //     };
//         //     counter_nonce += 1;
//         // };
//
//         // Complete KZG proof
//         let ck_a_kzg_opening = prove_commitment_key_kzg_opening(
//             &srs.g2_powers,
//             &transcript_inverse,
//             &<P::ScalarField>::one(),
//             &c,
//         )?;
//         end_timer!(ck_kzg);
//
//         Ok(TIPAWithSSMProof {
//             gipa_proof: proof,
//             final_ck: ck_a_final,
//             final_ck_proof: ck_a_kzg_opening,
//         })
//     }
// }

type Step<P> = (
    // LMC::Output
    PairingOutput<P>,
    // RMC::Output
    <P as Pairing>::ScalarField,
    // IPC::Output
    IdentityOutput<<P as Pairing>::G1>,
);

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct GIPAProof<P>
where
    P: Pairing,
{
    pub(crate) r_commitment_steps: Vec<(Step<P>, Step<P>)>,
    pub(crate) r_base: (P::G1, P::ScalarField),
}

#[derive(Clone)]
pub struct GIPAAux<P: Pairing> {
    pub(crate) r_transcript: Vec<P::ScalarField>,
    pub(crate) ck_base: (P::G2, P::ScalarField),
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Default, Eq, PartialEq)]
pub struct IdentityOutput<T>(pub Vec<T>)
where
    T: CanonicalSerialize + CanonicalDeserialize + Clone + Default + Eq;

pub struct BivariatePolynomialCommitment<P: Pairing> {
    _pairing: PhantomData<P>,
}

pub fn get_commitment_keys<P>(srs: &SRS<P>) -> (Vec<P::G2Affine>, Vec<P::G1Affine>)
where
    P: Pairing,
    P::G1: Icicle,
{
    let ck_1 = srs.g2_powers.iter().step_by(2).cloned().collect();
    let ck_2 = srs.g1_powers.iter().step_by(2).cloned().collect();
    (ck_1, ck_2)
}

impl<P: Pairing> BivariatePolynomialCommitment<P>
where
    P::ScalarField: JoltField + Field,
    P::G1: Icicle,
{
    pub fn setup<R: RngCore + CryptoRng>(rng: &mut R, x_degree: usize, y_degree: usize) -> SRS<P> {
        SRS::setup(rng, y_degree + 1, 2 * x_degree + 1)
    }

    pub fn commit(
        srs: &SRS<P>,
        bivariate_polynomial: &BivariatePolynomial<P::ScalarField>,
    ) -> Result<(PairingOutput<P>, Vec<P::G1Affine>), Error> {
        let (ck, _) = get_commitment_keys(srs);
        let (kzg_pk, _) = SRS::trim(Arc::new(srs.clone()), ck.len());
        // Create KZG commitments to Y polynomials
        let y_polynomial_coms = bivariate_polynomial
            .y_polynomials
            .iter()
            .chain([UniPoly::zero()].iter().cycle())
            .take(ck.len())
            .map(|y_polynomial| UnivariateKZG::<P>::commit(&kzg_pk, y_polynomial))
            .collect::<Result<Vec<P::G1Affine>, _>>()?;

        // Create AFGHO commitment to Y polynomial commitments
        Ok((
            AFGHOCommitmentG1::<P>::commit(&ck, &y_polynomial_coms)?,
            y_polynomial_coms,
        ))
    }

    pub fn open(
        srs: &SRS<P>,
        bivariate_polynomial: &BivariatePolynomial<P::ScalarField>,
        y_polynomial_comms: &[P::G1],
        point: &(P::ScalarField, P::ScalarField),
    ) -> Result<OpeningProof<P>, Error> {
        let (x, y) = point;

        let (ck_1, _) = get_commitment_keys(srs);
        let (kzg_pk, _) = SRS::trim(
            Arc::new(srs.clone()),
            bivariate_polynomial.y_polynomials.len(),
        );

        // let (ip_srs, kzg_srs) = srs;
        // let (ck_1, _) = ip_srs.get_commitment_keys();
        // assert!(ck_1.len() >= bivariate_polynomial.y_polynomials.len());

        let precomp_time = start_timer!(|| "Computing coefficients and KZG commitment");
        let mut powers_of_x = vec![];
        let mut cur = P::ScalarField::one();
        for _ in 0..(ck_1.len()) {
            powers_of_x.push(cur);
            cur *= x;
        }

        let coeffs = bivariate_polynomial
            .y_polynomials
            .iter()
            .chain([UniPoly::<P::ScalarField>::zero()].iter().cycle())
            .take(ck_1.len())
            .map(|y_polynomial: &UniPoly<_>| {
                let mut c = y_polynomial.coeffs.to_vec();
                c.resize(kzg_pk.g1_powers().len(), <P::ScalarField>::zero());
                c
            })
            .collect::<Vec<Vec<P::ScalarField>>>();
        let y_eval_coeffs = (0..kzg_pk.g1_powers().len())
            .map(|j| (0..ck_1.len()).map(|i| powers_of_x[i] * coeffs[i][j]).sum())
            .collect::<Vec<P::ScalarField>>();
        // Can unwrap because y_eval_coeffs.len() is guarnateed to be equal to kzg_srs.len()
        let y_eval_comm = todo!();
        // P::G1::msm(kzg_srs, &y_eval_coeffs).unwrap();
        end_timer!(precomp_time);

        let ipa_time = start_timer!(|| "Computing IPA proof");
        let ip_proof = todo!();
        // PolynomialEvaluationSecondTierIPA::<P>::prove_with_structured_scalar_message(
        //     ip_srs,
        //     (y_polynomial_comms, &powers_of_x),
        //     (&ck_1, &HomomorphicPlaceholderValue),
        // )?;
        end_timer!(ipa_time);
        let kzg_time = start_timer!(|| "Computing KZG opening proof");
        let (kzg_proof, _eval) =
            UnivariateKZG::<P>::open(&kzg_pk, &UniPoly::from_coeff(y_eval_coeffs), y)?;
        end_timer!(kzg_time);

        Ok(OpeningProof {
            ip_proof,
            y_eval_comm,
            kzg_proof,
        })
    }

    pub fn verify(
        v_srs: &KZGVerifierKey<P>,
        com: &PairingOutput<P>,
        point: &(P::ScalarField, P::ScalarField),
        eval: &P::ScalarField,
        proof: &OpeningProof<P>,
    ) -> Result<bool, Error> {
        todo!()
    }
}

pub struct UnivariatePolynomialCommitment<P: Pairing> {
    _pairing: PhantomData<P>,
}

impl<P: Pairing> UnivariatePolynomialCommitment<P>
where
    P::ScalarField: JoltField + Field,
    P::G1: Icicle,
{
    fn bivariate_degrees(univariate_degree: usize) -> (usize, usize) {
        //(((univariate_degree + 1) as f64).sqrt().ceil() as usize).next_power_of_two() - 1;
        let sqrt = (((univariate_degree + 1) as f64).sqrt().ceil() as usize).next_power_of_two();
        // Skew split between bivariate degrees to account for KZG being less expensive than MIPP
        let skew_factor = if sqrt >= 32 { 16_usize } else { sqrt / 2 };
        (sqrt / skew_factor - 1, sqrt * skew_factor - 1)
    }

    fn parse_bivariate_degrees_from_srs(srs: &SRS<P>) -> (usize, usize) {
        let x_degree = (srs.g2_powers.len() - 1) / 2;
        let y_degree = srs.g1_powers.len() - 1;
        (x_degree, y_degree)
    }

    fn bivariate_form(
        bivariate_degrees: (usize, usize),
        polynomial: &UniPoly<P::ScalarField>,
    ) -> BivariatePolynomial<P::ScalarField> {
        let (x_degree, y_degree) = bivariate_degrees;
        let default_zero = [P::ScalarField::zero()];
        let mut coeff_iter = polynomial
            .coeffs
            .iter()
            .chain(default_zero.iter().cycle())
            .take((x_degree + 1) * (y_degree + 1));

        let mut y_polynomials = Vec::new();
        for _ in 0..x_degree + 1 {
            let mut y_polynomial_coeffs = vec![];
            for _ in 0..y_degree + 1 {
                y_polynomial_coeffs.push(Clone::clone(coeff_iter.next().unwrap()))
            }
            y_polynomials.push(UniPoly::from_coeff(y_polynomial_coeffs));
        }
        BivariatePolynomial { y_polynomials }
    }

    pub fn setup<R: RngCore + CryptoRng>(rng: &mut R, degree: usize) -> SRS<P> {
        let (x_degree, y_degree) = Self::bivariate_degrees(degree);
        BivariatePolynomialCommitment::setup(rng, x_degree, y_degree)
    }

    pub fn commit(
        srs: &SRS<P>,
        polynomial: &UniPoly<P::ScalarField>,
    ) -> Result<(PairingOutput<P>, Vec<P::G1Affine>), Error> {
        let bivariate_degrees = Self::parse_bivariate_degrees_from_srs(srs);
        BivariatePolynomialCommitment::commit(
            srs,
            &Self::bivariate_form(bivariate_degrees, polynomial),
        )
    }

    pub fn open(
        srs: &SRS<P>,
        polynomial: &UniPoly<P::ScalarField>,
        y_polynomial_comms: &[P::G1],
        point: &P::ScalarField,
    ) -> Result<OpeningProof<P>, Error> {
        let (x_degree, y_degree) = Self::parse_bivariate_degrees_from_srs(srs);
        let y = *point;
        let x = point.pow([(y_degree + 1) as u64]);
        BivariatePolynomialCommitment::<P>::open(
            srs,
            &Self::bivariate_form((x_degree, y_degree), polynomial),
            y_polynomial_comms,
            &(x, y),
        )
    }

    pub fn verify(
        v_srs: &KZGVerifierKey<P>,
        max_degree: usize,
        com: &PairingOutput<P>,
        point: &P::ScalarField,
        eval: &P::ScalarField,
        proof: &OpeningProof<P>,
    ) -> Result<bool, Error> {
        let (_, y_degree) = Self::bivariate_degrees(max_degree);
        let y = *point;
        let x = y.pow([(y_degree + 1) as u64]);
        BivariatePolynomialCommitment::<P>::verify(v_srs, com, &(x, y), eval, proof)
    }
}
