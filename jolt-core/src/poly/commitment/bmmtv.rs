use std::marker::PhantomData;

use afgho::AFGHOCommitmentG1;
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ec::scalar_mul::fixed_base::FixedBase;
use ark_ec::{AffineRepr, CurveGroup, Group};
use ark_ff::{Field, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{end_timer, start_timer};
use ark_std::{rand::Rng, One, UniformRand, Zero};
use eyre::Error;
use identity::{IdentityCommitment, IdentityOutput};

use crate::msm::Icicle;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::{field::JoltField, poly::unipoly::UniPoly as UnivariatePolynomial};

use super::kzg::{KZGProverKey, KZGVerifierKey as VerifierSRS, UnivariateKZG as KZG};

mod afgho;
mod identity;
mod inner_product;

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct GIPAProof<IP, LMC, RMC, IPC, D>
where
    IP: InnerProduct<
        LeftMessage = LMC::Message,
        RightMessage = RMC::Message,
        Output = IPC::Message,
    >,
    LMC: DoublyHomomorphicCommitment,
    RMC: DoublyHomomorphicCommitment<Scalar = LMC::Scalar>,
    IPC: DoublyHomomorphicCommitment<Scalar = LMC::Scalar>,
{
    pub(crate) r_commitment_steps: Vec<(
        (LMC::Output, RMC::Output, IPC::Output),
        (LMC::Output, RMC::Output, IPC::Output),
    )>,
    pub(crate) r_base: (LMC::Message, RMC::Message),
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct TIPAWithSSMProof<IP, LMC, IPC, P, D>
where
    P: Pairing,
    IP: InnerProduct<LeftMessage = LMC::Message, RightMessage = LMC::Scalar, Output = IPC::Message>,
    LMC: DoublyHomomorphicCommitment<Scalar = P::ScalarField, Key = P::G2> + TIPACompatibleSetup,
    IPC: DoublyHomomorphicCommitment<Scalar = LMC::Scalar>,
{
    gipa_proof: GIPAProof<IP, LMC, SSMPlaceholderCommitment<LMC::Scalar>, IPC, D>,
    final_ck: P::G2,
    final_ck_proof: P::G2,
}

type PolynomialEvaluationSecondTierIPA<P> = TIPAWithSSM<
    MultiexponentiationInnerProduct<<P as Pairing>::G1>,
    AFGHOCommitmentG1<P>,
    IdentityCommitment<<P as Pairing>::G1, <P as Pairing>::ScalarField>,
    P,
>;

type PolynomialEvaluationSecondTierIPAProof<P> = TIPAWithSSMProof<
    MultiexponentiationInnerProduct<<P as Pairing>::G1>,
    AFGHOCommitmentG1<P>,
    IdentityCommitment<<P as Pairing>::G1, <P as Pairing>::ScalarField>,
    P,
>;

pub struct BivariatePolynomial<F> {
    y_polynomials: Vec<UnivariatePolynomial<F>>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Default, Eq, PartialEq)]
pub struct HomomorphicPlaceholderValue;

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
    ip_proof: PolynomialEvaluationSecondTierIPAProof<P>,
    y_eval_comm: P::G1Affine,
    kzg_proof: P::G1Affine,
}

pub struct BivariatePolynomialCommitment<P: Pairing> {
    _pairing: PhantomData<P>,
}

#[derive(Clone)]
pub struct SRS<P: Pairing> {
    pub g_alpha_powers: Vec<P::G1>,
    pub h_beta_powers: Vec<P::G2>,
    pub g_beta: P::G1,
    pub h_alpha: P::G2,
}

// #[derive(Clone)]
// pub struct VerifierSRS<P: Pairing> {
//     pub g: P::G1,
//     pub h: P::G2,
//     pub g_beta: P::G1,
//     pub h_alpha: P::G2,
// }

impl<P: Pairing> SRS<P> {
    pub fn get_commitment_keys(&self) -> (Vec<P::G2>, Vec<P::G1>) {
        let ck_1 = self.h_beta_powers.iter().step_by(2).cloned().collect();
        let ck_2 = self.g_alpha_powers.iter().step_by(2).cloned().collect();
        (ck_1, ck_2)
    }

    pub fn get_verifier_key(&self) -> VerifierSRS<P> {
        VerifierSRS {
            g1: self.g_alpha_powers[0].into_affine(),
            // g: self.g_alpha_powers[0],
            g2: self.h_beta_powers[0].into_affine(),
            // h: self.h_beta_powers[0],
            // g_beta: self.g_beta,
            beta_g2: self.h_alpha.into_affine(),
            // h_alpha: self.h_alpha,
        }
    }
}

pub fn structured_generators_scalar_power<G: CurveGroup>(
    num: usize,
    g: &G,
    s: &G::ScalarField,
) -> Vec<G> {
    assert!(num > 0);
    let mut powers_of_scalar = vec![];
    let mut pow_s = <G as Group>::ScalarField::one();
    for _ in 0..num {
        powers_of_scalar.push(pow_s);
        pow_s *= s;
    }

    let window_size = FixedBase::get_mul_window_size(num);

    let scalar_bits = <<G as Group>::ScalarField as PrimeField>::MODULUS_BIT_SIZE as usize;
    let g_table = FixedBase::get_window_table(scalar_bits, window_size, *g);
    FixedBase::msm::<G>(scalar_bits, window_size, &g_table, &powers_of_scalar)
}

impl<P: Pairing> BivariatePolynomialCommitment<P>
where
    P::ScalarField: JoltField + Field,
    P::G1: Icicle,
{
    pub fn setup<R: Rng>(
        rng: &mut R,
        x_degree: usize,
        y_degree: usize,
    ) -> Result<(SRS<P>, Vec<P::G1Affine>), Error> {
        let alpha = <P::ScalarField>::rand(rng);
        let beta = <P::ScalarField>::rand(rng);
        let g = P::G1::generator();
        let h = P::G2::generator();
        let kzg_srs = <P as Pairing>::G1::normalize_batch(&structured_generators_scalar_power(
            y_degree + 1,
            &g,
            &alpha,
        ));
        let srs = SRS {
            g_alpha_powers: vec![g],
            h_beta_powers: structured_generators_scalar_power(2 * x_degree + 1, &h, &beta),
            g_beta: g * beta,
            h_alpha: h * alpha,
        };
        Ok((srs, kzg_srs))
    }

    pub fn commit(
        srs: &(SRS<P>, KZGProverKey<P>),
        bivariate_polynomial: &BivariatePolynomial<P::ScalarField>,
    ) -> Result<(PairingOutput<P>, Vec<P::G1Affine>), Error> {
        let (ip_srs, kzg_srs) = srs;
        let (ck, _) = ip_srs.get_commitment_keys();
        assert!(ck.len() >= bivariate_polynomial.y_polynomials.len());

        // Create KZG commitments to Y polynomials
        let y_polynomial_coms = bivariate_polynomial
            .y_polynomials
            .iter()
            .chain([UnivariatePolynomial::zero()].iter().cycle())
            .take(ck.len())
            .map(|y_polynomial| KZG::commit(kzg_srs, y_polynomial))
            .collect::<Result<Vec<P::G1Affine>, _>>()?;

        // Create AFGHO commitment to Y polynomial commitments
        Ok((
            AFGHOCommitmentG1::commit(
                &ck,
                &y_polynomial_coms
                    .iter()
                    .map(|g| g.into_group())
                    .collect::<Vec<_>>(),
            )?,
            y_polynomial_coms,
        ))
    }

    pub fn open(
        srs: &(SRS<P>, KZGProverKey<P>),
        bivariate_polynomial: &BivariatePolynomial<P::ScalarField>,
        y_polynomial_comms: &[P::G1],
        point: &(P::ScalarField, P::ScalarField),
    ) -> Result<OpeningProof<P>, Error> {
        let (x, y) = point;
        let (ip_srs, kzg_srs) = srs;
        let (ck_1, _) = ip_srs.get_commitment_keys();
        assert!(ck_1.len() >= bivariate_polynomial.y_polynomials.len());

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
            .chain(
                [UnivariatePolynomial::<P::ScalarField>::zero()]
                    .iter()
                    .cycle(),
            )
            .take(ck_1.len())
            .map(|y_polynomial| {
                let mut c = y_polynomial.coeffs.to_vec();
                c.resize(kzg_srs.len(), <P::ScalarField>::zero());
                c
            })
            .collect::<Vec<Vec<P::ScalarField>>>();
        let y_eval_coeffs = (0..kzg_srs.len())
            .map(|j| (0..ck_1.len()).map(|i| powers_of_x[i] * coeffs[i][j]).sum())
            .collect::<Vec<P::ScalarField>>();
        // Can unwrap because y_eval_coeffs.len() is guarnateed to be equal to kzg_srs.len()
        let polynomial =
            MultilinearPolynomial::LargeScalars(DensePolynomial::new(y_eval_coeffs.clone()).into());
        let y_eval_comm = KZG::commit_as_univariate(kzg_srs, &polynomial)?;
        end_timer!(precomp_time);

        let ipa_time = start_timer!(|| "Computing IPA proof");
        let ip_proof = PolynomialEvaluationSecondTierIPA::prove_with_structured_scalar_message(
            ip_srs,
            (y_polynomial_comms, &powers_of_x),
            (&ck_1, &HomomorphicPlaceholderValue),
        )?;
        end_timer!(ipa_time);
        let kzg_time = start_timer!(|| "Computing KZG opening proof");
        let (kzg_proof, _evaluation) =
            KZG::open(kzg_srs, &UnivariatePolynomial::from_coeff(y_eval_coeffs), y)?;
        end_timer!(kzg_time);

        Ok(OpeningProof {
            ip_proof,
            y_eval_comm,
            kzg_proof,
        })
    }

    pub fn verify(
        v_srs: &VerifierSRS<P>,
        com: &PairingOutput<P>,
        point: &(P::ScalarField, P::ScalarField),
        eval: &P::ScalarField,
        proof: &OpeningProof<P>,
    ) -> Result<bool, Error> {
        let (x, y) = point;
        let ip_proof_valid =
            PolynomialEvaluationSecondTierIPA::verify_with_structured_scalar_message(
                v_srs,
                &HomomorphicPlaceholderValue,
                (com, &IdentityOutput(vec![proof.y_eval_comm])),
                x,
                &proof.ip_proof,
            )?;
        let kzg_proof_valid = KZG::verify(v_srs, &proof.y_eval_comm, y, &proof.kzg_proof, eval)?;
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

    fn parse_bivariate_degrees_from_srs(srs: &(SRS<P>, KZGProverKey<P>)) -> (usize, usize) {
        let x_degree = (srs.0.h_beta_powers.len() - 1) / 2;
        let y_degree = srs.1.len() - 1;
        (x_degree, y_degree)
    }

    fn bivariate_form(
        bivariate_degrees: (usize, usize),
        polynomial: &UnivariatePolynomial<P::ScalarField>,
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
            y_polynomials.push(UnivariatePolynomial::from_coeff(y_polynomial_coeffs));
        }
        BivariatePolynomial { y_polynomials }
    }

    pub fn setup<R: Rng>(rng: &mut R, degree: usize) -> Result<(SRS<P>, Vec<P::G1Affine>), Error> {
        let (x_degree, y_degree) = Self::bivariate_degrees(degree);
        BivariatePolynomialCommitment::setup(rng, x_degree, y_degree)
    }

    pub fn commit(
        srs: &(SRS<P>, KZGProverKey<P>),
        polynomial: &UnivariatePolynomial<P::ScalarField>,
    ) -> Result<(PairingOutput<P>, Vec<P::G1Affine>), Error> {
        let bivariate_degrees = Self::parse_bivariate_degrees_from_srs(srs);
        BivariatePolynomialCommitment::commit(
            srs,
            &Self::bivariate_form(bivariate_degrees, polynomial),
        )
    }

    pub fn open(
        srs: &(SRS<P>, KZGProverKey<P>),
        polynomial: &UnivariatePolynomial<P::ScalarField>,
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
        v_srs: &VerifierSRS<P>,
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
