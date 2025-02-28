use std::{marker::PhantomData, sync::Arc};

use afgho::inner_product;
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ff::Field;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{One, Zero};
use eyre::Error;
use rand_core::{CryptoRng, RngCore};

use crate::{field::JoltField, msm::Icicle, poly::unipoly::UniPoly};

use super::kzg::{KZGVerifierKey, UnivariateKZG, SRS};

mod afgho;

struct ProverKeySRS<P: Pairing> {
    g_powers: Vec<P::G1>,
    h_powers: Vec<P::G2>,
}

struct VerifierKeySRS<P: Pairing> {
    g_betha: P::G1,
    h_alpha: P::G2,
}

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
    /// SRS currently uses only betha and pows to both g1 and g2, should we use alpha and betha or
    /// does it work the same?
    pub fn setup<R: RngCore + CryptoRng>(rng: &mut R, x_degree: usize, y_degree: usize) -> SRS<P> {
        SRS::setup(rng, y_degree + 1, 2 * x_degree + 1)
    }

    pub fn commit(
        srs: &SRS<P>,
        bivariate_polynomial: &BivariatePolynomial<P::ScalarField>,
    ) -> Result<(PairingOutput<P>, Vec<P::G1Affine>), Error> {
        let (ck, _) = get_commitment_keys(srs);
        let (kzg_pk, _) = SRS::trim(Arc::new(srs.clone()), ck.len());
        // We create a KZG commitment for each y_polynomial
        let y_polynomial_coms = UnivariateKZG::commit_variable_batch_univariate(
            &kzg_pk,
            bivariate_polynomial.y_polynomials.as_slice(),
        )?;

        // Create AFGHO commitment to Y polynomial commitments
        Ok((inner_product(&y_polynomial_coms, &ck)?, y_polynomial_coms))
    }

    pub fn open(
        srs: &SRS<P>,
        bivariate_polynomial: &BivariatePolynomial<P::ScalarField>,
        y_polynomial_comms: &[P::G1Affine],
        point: &(P::ScalarField, P::ScalarField),
    ) -> Result<OpeningProof<P>, Error> {
        let (x, y) = point;

        let (ck_1, kzg_srs) = get_commitment_keys(srs);
        let (kzg_pk, _) = SRS::trim(
            Arc::new(srs.clone()),
            bivariate_polynomial.y_polynomials.len(),
        );

        // let (ip_srs, kzg_srs) = srs;
        // let (ck_1, _) = ip_srs.get_commitment_keys();
        assert!(ck_1.len() >= bivariate_polynomial.y_polynomials.len());

        let powers_of_x = {
            let mut powers_of_x = vec![];
            let mut cur = P::ScalarField::one();
            for _ in 0..(ck_1.len()) {
                powers_of_x.push(cur);
                cur *= x;
            }
            powers_of_x
        };

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
        let y_eval_coeff_poly = UniPoly::from_coeff(y_eval_coeffs);
        // Can unwrap because y_eval_coeffs.len() is guarnateed to be equal to kzg_srs.len()
        let y_eval_comm = UnivariateKZG::commit(&kzg_pk, &y_eval_coeff_poly).unwrap();

        let ip_proof = todo!();
        // PolynomialEvaluationSecondTierIPA::<P>::prove_with_structured_scalar_message(
        //     ip_srs,
        //     (y_polynomial_comms, &powers_of_x),
        //     (&ck_1),
        // )?;
        let (kzg_proof, _eval) = UnivariateKZG::<P>::open(&kzg_pk, &y_eval_coeff_poly, y)?;

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
        let (x, y) = point;
        let ip_proof_valid = todo!();
        // PolynomialEvaluationSecondTierIPA::verify_with_structured_scalar_message(
        //     v_srs,
        //     (com, &IdentityOutput(vec![proof.y_eval_comm])),
        //     x,
        //     &proof.ip_proof,
        // )?;
        let kzg_proof_valid =
            UnivariateKZG::verify(v_srs, &proof.y_eval_comm, y, &proof.kzg_proof, eval)?;
        Ok(ip_proof_valid && kzg_proof_valid)
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
        y_polynomial_comms: &[P::G1Affine],
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
