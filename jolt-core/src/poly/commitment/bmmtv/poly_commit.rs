use ark_ec::Group;
use ark_ec::{
    pairing::{Pairing, PairingOutput},
    scalar_mul::variable_base::VariableBaseMSM,
    CurveGroup,
};
use ark_ff::{Field, One, UniformRand, Zero};
use ark_poly::polynomial::{
    univariate::DensePolynomial as UnivariatePolynomial, DenseUVPolynomial, Polynomial,
};

use ark_std::{end_timer, start_timer};
use std::marker::PhantomData;

use ark_std::rand::Rng;
use digest::Digest;

use super::{
    commitments::{
        afgho16::AfghoCommitment,
        identity::{DummyParam, IdentityCommitment, IdentityOutput},
        Dhc,
    },
    inner_products::MultiexponentiationInnerProduct,
    tipa::{
        structured_generators_scalar_power,
        structured_scalar_message::{TipaWithSsm, TipaWithSsmProof},
        Srs, VerifierSrs,
    },
    Error,
};

type G1<P> = <P as Pairing>::G1;
type ScalarField<P> = <P as Pairing>::ScalarField;

type PolynomialEvaluationSecondTierIpa<P, D> = TipaWithSsm<
    MultiexponentiationInnerProduct<G1<P>>,
    AfghoCommitment<P>,
    IdentityCommitment<G1<P>, ScalarField<P>>,
    P,
    D,
>;

type PolynomialEvaluationSecondTierIpaProof<P> =
    TipaWithSsmProof<P, AfghoCommitment<P>, IdentityCommitment<G1<P>, ScalarField<P>>>;

pub struct Kzg<P: Pairing> {
    _pairing: PhantomData<P>,
}

// Simple implementation of KZG polynomial commitment scheme
impl<P: Pairing> Kzg<P> {
    pub fn setup<R: Rng>(
        rng: &mut R,
        degree: usize,
    ) -> Result<(Vec<P::G1Affine>, VerifierSrs<P>), Error> {
        let alpha = P::ScalarField::rand(rng);
        let beta = P::ScalarField::rand(rng);
        let g = P::G1::generator();
        let h = P::G2::generator();
        let g_alpha_powers = structured_generators_scalar_power(degree + 1, &g, &alpha);
        Ok((
            P::G1::normalize_batch(&g_alpha_powers),
            VerifierSrs {
                g,
                h,
                g_beta: g * beta,
                h_alpha: h * alpha,
            },
        ))
    }

    pub fn commit(
        powers: &[P::G1Affine],
        polynomial: &UnivariatePolynomial<P::ScalarField>,
    ) -> Result<P::G1, Error> {
        assert!(powers.len() > polynomial.degree());
        let mut coeffs = polynomial.coeffs.to_vec();
        coeffs.resize(powers.len(), P::ScalarField::zero());

        // Can unwrap because coeffs.len() is guaranteed to be equal to powers.len()
        Ok(P::G1::msm(powers, &coeffs).unwrap())
    }

    pub fn open(
        powers: &[P::G1Affine],
        polynomial: &UnivariatePolynomial<P::ScalarField>,
        point: &P::ScalarField,
    ) -> Result<P::G1, Error> {
        assert!(powers.len() > polynomial.degree());

        // Trick to calculate (p(x) - p(z)) / (x - z) as p(x) / (x - z) ignoring remainder p(z)
        let quotient_polynomial = polynomial
            / &UnivariatePolynomial::from_coefficients_vec(vec![-*point, P::ScalarField::one()]);
        let mut quotient_coeffs = quotient_polynomial.coeffs.to_vec();
        quotient_coeffs.resize(powers.len(), P::ScalarField::zero());

        // Can unwrap because quotient_coeffs.len() is guaranteed to be equal to powers.len()
        Ok(P::G1::msm(powers, &quotient_coeffs).unwrap())
    }

    pub fn verify(
        v_srs: &VerifierSrs<P>,
        com: &P::G1,
        point: &P::ScalarField,
        eval: &P::ScalarField,
        proof: &P::G1,
    ) -> Result<bool, Error> {
        Ok(P::pairing(*com - v_srs.g * eval, v_srs.h)
            == P::pairing(*proof, v_srs.h_alpha - v_srs.h * point))
    }
}

pub struct BivariatePolynomial<F: Field> {
    y_polynomials: Vec<UnivariatePolynomial<F>>,
}

impl<F: Field> BivariatePolynomial<F> {
    pub fn evaluate(&self, point: &(F, F)) -> F {
        let (x, y) = point;
        let mut point_x_powers = vec![];
        let mut cur = F::one();
        for _ in 0..(self.y_polynomials.len()) {
            point_x_powers.push(cur);
            cur *= x;
        }
        point_x_powers
            .iter()
            .zip(&self.y_polynomials)
            .map(|(x_power, y_polynomial)| *x_power * y_polynomial.evaluate(y))
            .sum()
    }
}

pub struct OpeningProof<P: Pairing> {
    ip_proof: PolynomialEvaluationSecondTierIpaProof<P>,
    y_eval_comm: P::G1,
    kzg_proof: P::G1,
}

pub struct BivariatePolynomialCommitment<P: Pairing, D: Digest>(
    PolynomialEvaluationSecondTierIpa<P, D>,
);

impl<P: Pairing, D: Digest> BivariatePolynomialCommitment<P, D> {
    pub fn setup<R: Rng>(
        rng: &mut R,
        x_degree: usize,
        y_degree: usize,
    ) -> Result<(Srs<P>, Vec<P::G1Affine>), Error> {
        let alpha = P::ScalarField::rand(rng);
        let beta = P::ScalarField::rand(rng);
        let g = P::G1::generator();
        let h = P::G2::generator();
        let kzg_srs = P::G1::normalize_batch(&structured_generators_scalar_power(
            y_degree + 1,
            &g,
            &alpha,
        ));
        let srs = Srs {
            g_alpha_powers: vec![g],
            // why 2x
            // MAYBE artifact of afgho or because of ceiling
            h_beta_powers: structured_generators_scalar_power(2 * x_degree + 1, &h, &beta),
            g_beta: g * beta,
            h_alpha: h * alpha,
        };
        Ok((srs, kzg_srs))
    }

    pub fn commit(
        srs: &(Srs<P>, Vec<P::G1Affine>),
        bivariate_polynomial: &BivariatePolynomial<P::ScalarField>,
    ) -> Result<(PairingOutput<P>, Vec<P::G1>), Error> {
        let (ip_srs, kzg_srs) = srs;
        let (ck, _) = ip_srs.get_commitment_keys();
        assert!(ck.len() >= bivariate_polynomial.y_polynomials.len());

        // Create KZG commitments to Y polynomials
        let y_polynomial_coms = bivariate_polynomial
            .y_polynomials
            .iter()
            .chain([UnivariatePolynomial::zero()].iter().cycle())
            .take(ck.len())
            .map(|y_polynomial| Kzg::<P>::commit(kzg_srs, y_polynomial))
            .collect::<Result<Vec<P::G1>, Error>>()?;

        // Create AFGHO commitment to Y polynomial commitments
        Ok((
            AfghoCommitment::<P>::commit(&ck, &y_polynomial_coms)?,
            y_polynomial_coms,
        ))
    }

    pub fn open(
        srs: &(Srs<P>, Vec<P::G1Affine>),
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
            .chain([UnivariatePolynomial::zero()].iter().cycle())
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
        let y_eval_comm = P::G1::msm(kzg_srs, &y_eval_coeffs).unwrap();
        end_timer!(precomp_time);

        let ipa_time = start_timer!(|| "Computing IPA proof");
        let ip_proof =
            PolynomialEvaluationSecondTierIpa::<P, D>::prove_with_structured_scalar_message(
                ip_srs,
                (y_polynomial_comms, &powers_of_x),
                (&ck_1, &DummyParam),
            )?;
        end_timer!(ipa_time);
        let kzg_time = start_timer!(|| "Computing KZG opening proof");
        let kzg_proof = Kzg::<P>::open(
            kzg_srs,
            &UnivariatePolynomial::from_coefficients_slice(&y_eval_coeffs),
            y,
        )?;
        end_timer!(kzg_time);

        Ok(OpeningProof {
            ip_proof,
            y_eval_comm,
            kzg_proof,
        })
    }

    pub fn verify(
        v_srs: &VerifierSrs<P>,
        com: &PairingOutput<P>,
        point: &(P::ScalarField, P::ScalarField),
        eval: &P::ScalarField,
        proof: &OpeningProof<P>,
    ) -> Result<bool, Error> {
        let (x, y) = point;
        let ip_proof_valid =
            PolynomialEvaluationSecondTierIpa::<P, D>::verify_with_structured_scalar_message(
                v_srs,
                &DummyParam,
                (com, &IdentityOutput(vec![proof.y_eval_comm])),
                x,
                &proof.ip_proof,
            )?;
        let kzg_proof_valid =
            Kzg::<P>::verify(v_srs, &proof.y_eval_comm, y, eval, &proof.kzg_proof)?;
        Ok(ip_proof_valid && kzg_proof_valid)
    }
}

pub struct UnivariatePolynomialCommitment<P: Pairing, D: Digest> {
    _pairing: PhantomData<P>,
    _digest: PhantomData<D>,
}

impl<P: Pairing, D: Digest> UnivariatePolynomialCommitment<P, D> {
    fn bivariate_degrees(univariate_degree: usize) -> (usize, usize) {
        //(((univariate_degree + 1) as f64).sqrt().ceil() as usize).next_power_of_two() - 1;
        let sqrt = (((univariate_degree + 1) as f64).sqrt().ceil() as usize).next_power_of_two();
        // Skew split between bivariate degrees to account for KZG being less expensive than MIPP
        let skew_factor = if sqrt >= 32 { 16_usize } else { sqrt / 2 };
        (sqrt / skew_factor - 1, sqrt * skew_factor - 1)
    }

    fn parse_bivariate_degrees_from_srs(srs: &(Srs<P>, Vec<P::G1Affine>)) -> (usize, usize) {
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
            y_polynomials.push(UnivariatePolynomial::from_coefficients_slice(
                &y_polynomial_coeffs,
            ));
        }
        BivariatePolynomial { y_polynomials }
    }

    pub fn setup<R: Rng>(rng: &mut R, degree: usize) -> Result<(Srs<P>, Vec<P::G1Affine>), Error> {
        let (x_degree, y_degree) = Self::bivariate_degrees(degree);
        BivariatePolynomialCommitment::<P, D>::setup(rng, x_degree, y_degree)
    }

    pub fn commit(
        srs: &(Srs<P>, Vec<P::G1Affine>),
        polynomial: &UnivariatePolynomial<P::ScalarField>,
    ) -> Result<(PairingOutput<P>, Vec<P::G1>), Error> {
        let bivariate_degrees = Self::parse_bivariate_degrees_from_srs(srs);
        BivariatePolynomialCommitment::<P, D>::commit(
            srs,
            &Self::bivariate_form(bivariate_degrees, polynomial),
        )
    }

    pub fn open(
        srs: &(Srs<P>, Vec<P::G1Affine>),
        polynomial: &UnivariatePolynomial<P::ScalarField>,
        y_polynomial_comms: &[P::G1],
        point: &P::ScalarField,
    ) -> Result<OpeningProof<P>, Error> {
        let (x_degree, y_degree) = Self::parse_bivariate_degrees_from_srs(srs);
        let y = *point;
        let x = point.pow(vec![(y_degree + 1) as u64]);
        BivariatePolynomialCommitment::<P, D>::open(
            srs,
            &Self::bivariate_form((x_degree, y_degree), polynomial),
            y_polynomial_comms,
            &(x, y),
        )
    }

    pub fn verify(
        v_srs: &VerifierSrs<P>,
        max_degree: usize,
        com: &PairingOutput<P>,
        point: &P::ScalarField,
        eval: &P::ScalarField,
        proof: &OpeningProof<P>,
    ) -> Result<bool, Error> {
        let (_, y_degree) = Self::bivariate_degrees(max_degree);
        let y = *point;
        let x = y.pow(vec![(y_degree + 1) as u64]);
        BivariatePolynomialCommitment::<P, D>::verify(v_srs, com, &(x, y), eval, proof)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Bn254;
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use sha3::Sha3_256;

    const BIVARIATE_X_DEGREE: usize = 7;
    const BIVARIATE_Y_DEGREE: usize = 7;
    //const UNIVARIATE_DEGREE: usize = 56;
    const UNIVARIATE_DEGREE: usize = 65535;
    //const UNIVARIATE_DEGREE: usize = 1048575;

    type TestBivariatePolyCommitment = BivariatePolynomialCommitment<Bn254, Sha3_256>;
    type TestUnivariatePolyCommitment = UnivariatePolynomialCommitment<Bn254, Sha3_256>;

    #[test]
    fn bivariate_poly_commit_test() {
        let mut rng = StdRng::seed_from_u64(0u64);
        let srs =
            TestBivariatePolyCommitment::setup(&mut rng, BIVARIATE_X_DEGREE, BIVARIATE_Y_DEGREE)
                .unwrap();
        let v_srs = srs.0.get_verifier_key();

        let mut y_polynomials = Vec::new();
        for _ in 0..BIVARIATE_X_DEGREE + 1 {
            let mut y_polynomial_coeffs = vec![];
            for _ in 0..BIVARIATE_Y_DEGREE + 1 {
                y_polynomial_coeffs.push(<Bn254 as Pairing>::ScalarField::rand(&mut rng));
            }
            y_polynomials.push(UnivariatePolynomial::from_coefficients_slice(
                &y_polynomial_coeffs,
            ));
        }
        let bivariate_polynomial = BivariatePolynomial { y_polynomials };

        // Commit to polynomial
        let (com, y_polynomial_comms) =
            TestBivariatePolyCommitment::commit(&srs, &bivariate_polynomial).unwrap();

        // Evaluate at challenge point
        let point = (UniformRand::rand(&mut rng), UniformRand::rand(&mut rng));
        let eval_proof = TestBivariatePolyCommitment::open(
            &srs,
            &bivariate_polynomial,
            &y_polynomial_comms,
            &point,
        )
        .unwrap();
        let eval = bivariate_polynomial.evaluate(&point);

        // Verify proof
        assert!(
            TestBivariatePolyCommitment::verify(&v_srs, &com, &point, &eval, &eval_proof).unwrap()
        );
    }

    // `cargo test univariate_poly_commit_test --release --features print-trace -- --ignored --nocapture`
    // #[ignore]
    #[test]
    fn univariate_poly_commit_test() {
        let mut rng = StdRng::seed_from_u64(0u64);
        let srs = TestUnivariatePolyCommitment::setup(&mut rng, UNIVARIATE_DEGREE).unwrap();
        let v_srs = srs.0.get_verifier_key();

        let mut polynomial_coeffs = vec![];
        for _ in 0..UNIVARIATE_DEGREE + 1 {
            polynomial_coeffs.push(<Bn254 as Pairing>::ScalarField::rand(&mut rng));
        }
        let polynomial = UnivariatePolynomial::from_coefficients_slice(&polynomial_coeffs);

        // Commit to polynomial
        let (com, y_polynomial_comms) =
            TestUnivariatePolyCommitment::commit(&srs, &polynomial).unwrap();

        // Evaluate at challenge point
        let point = UniformRand::rand(&mut rng);
        let eval_proof =
            TestUnivariatePolyCommitment::open(&srs, &polynomial, &y_polynomial_comms, &point)
                .unwrap();
        let eval = polynomial.evaluate(&point);

        // Verify proof
        assert!(TestUnivariatePolyCommitment::verify(
            &v_srs,
            UNIVARIATE_DEGREE,
            &com,
            &point,
            &eval,
            &eval_proof
        )
        .unwrap());
    }
}
