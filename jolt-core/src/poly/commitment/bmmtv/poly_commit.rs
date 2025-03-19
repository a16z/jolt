use crate::msm::Icicle;
use ark_ec::AffineRepr;
use ark_ec::{
    pairing::{Pairing, PairingOutput},
    scalar_mul::variable_base::VariableBaseMSM,
    CurveGroup,
};
use ark_ff::{Field, One, Zero};

use ark_std::{end_timer, start_timer};
use std::marker::PhantomData;

use super::{
    commitments::afgho16::AfghoCommitment,
    tipa::structured_scalar_message::{TipaWithSsm, TipaWithSsmProof},
    Error,
};
use crate::field::JoltField;
use crate::poly::commitment::kzg::{KZGProverKey, KZGVerifierKey, UnivariateKZG, SRS};
use crate::poly::unipoly::UniPoly as UnivariatePolynomial;
use crate::utils::transcript::Transcript;
use ark_std::rand::Rng;
use rand_core::CryptoRng;

pub struct BivariatePolynomial<F: Field> {
    y_polynomials: Vec<UnivariatePolynomial<F>>,
}

impl<F: Field> BivariatePolynomial<F>
where
    F: JoltField,
{
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
    ip_proof: TipaWithSsmProof<P>,
    y_eval_comm: P::G1,
    kzg_proof: P::G1,
}

pub struct BivariatePolynomialCommitment<P: Pairing, D>(TipaWithSsm<P, D>);

impl<P: Pairing, ProofTranscript: Transcript> BivariatePolynomialCommitment<P, ProofTranscript>
where
    P::ScalarField: JoltField,
    P::G1: Icicle,
{
    pub fn setup<R: Rng + CryptoRng>(
        rng: &mut R,
        x_degree: usize,
        y_degree: usize,
    ) -> Result<SRS<P>, Error> {
        Ok(SRS::setup(rng, y_degree, 2 * x_degree))
    }

    pub fn commit(
        srs: &(Vec<P::G2>, KZGProverKey<P>),
        bivariate_polynomial: &BivariatePolynomial<P::ScalarField>,
    ) -> Result<(PairingOutput<P>, Vec<P::G1>), Error> {
        let (ck, kzg_srs) = srs;
        assert!(ck.len() >= bivariate_polynomial.y_polynomials.len());

        // Create KZG commitments to Y polynomials
        let y_polynomial_coms = bivariate_polynomial
            .y_polynomials
            .iter()
            .chain([UnivariatePolynomial::zero()].iter().cycle())
            .take(ck.len())
            .map(|y_polynomial| UnivariateKZG::<P>::commit(kzg_srs, y_polynomial))
            .collect::<Result<Vec<P::G1Affine>, _>>()?;
        // TODO update
        let y_polynomial_coms = y_polynomial_coms
            .into_iter()
            .map(|affine| affine.into_group())
            .collect::<Vec<_>>();

        // Create AFGHO commitment to Y polynomial commitments
        Ok((
            AfghoCommitment::<P>::commit(ck, &y_polynomial_coms)?,
            y_polynomial_coms,
        ))
    }

    pub fn open(
        srs: &(Vec<P::G2>, KZGProverKey<P>),
        bivariate_polynomial: &BivariatePolynomial<P::ScalarField>,
        y_polynomial_comms: &[P::G1],
        point: &(P::ScalarField, P::ScalarField),
        transcript: &mut ProofTranscript,
    ) -> Result<OpeningProof<P>, Error> {
        let (x, y) = point;
        let (ck_1, kzg_srs) = srs;
        assert!(ck_1.len() >= bivariate_polynomial.y_polynomials.len());

        let precomp_time = start_timer!(|| "Computing coefficients and KZG commitment");
        let mut powers_of_x = vec![];
        let mut cur = P::ScalarField::one();
        for _ in 0..(ck_1.len()) {
            powers_of_x.push(cur);
            cur *= *x;
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
        let y_eval_comm = P::G1::msm(kzg_srs.g1_powers(), &y_eval_coeffs).unwrap();
        end_timer!(precomp_time);

        let ipa_time = start_timer!(|| "Computing IPA proof");
        let ip_proof = TipaWithSsm::<P, ProofTranscript>::prove_with_structured_scalar_message(
            &kzg_srs.h_beta_powers(),
            (y_polynomial_comms, &powers_of_x),
            ck_1,
            transcript,
        )?;
        end_timer!(ipa_time);
        let kzg_time = start_timer!(|| "Computing KZG opening proof");
        let (kzg_proof, _eval) =
            UnivariateKZG::<P>::open(kzg_srs, &UnivariatePolynomial::from_coeff(y_eval_coeffs), y)?;
        end_timer!(kzg_time);

        Ok(OpeningProof {
            ip_proof,
            y_eval_comm,
            kzg_proof: kzg_proof.into_group(),
        })
    }

    pub fn verify(
        v_srs: &KZGVerifierKey<P>,
        com: &PairingOutput<P>,
        point: &(P::ScalarField, P::ScalarField),
        eval: &P::ScalarField,
        proof: &OpeningProof<P>,
        transcript: &mut ProofTranscript,
    ) -> Result<bool, Error> {
        let (x, y) = point;
        let ip_proof_valid =
            TipaWithSsm::<P, ProofTranscript>::verify_with_structured_scalar_message(
                v_srs,
                (com, &proof.y_eval_comm),
                x,
                &proof.ip_proof,
                transcript,
            )?;
        let kzg_proof_valid = UnivariateKZG::<P>::verify(
            v_srs,
            &proof.y_eval_comm.into_affine(),
            y,
            &proof.kzg_proof.into_affine(),
            eval,
        )?;
        println!(
            "ip_proof_valid: {}, kzg_proof_valid: {}",
            ip_proof_valid, kzg_proof_valid
        );
        Ok(ip_proof_valid && kzg_proof_valid)
    }
}

pub struct UnivariatePolynomialCommitment<P, D> {
    _pairing: PhantomData<P>,
    _digest: PhantomData<D>,
}

impl<P: Pairing, ProofTranscript: Transcript> UnivariatePolynomialCommitment<P, ProofTranscript>
where
    P::ScalarField: JoltField,
    P::G1: Icicle,
{
    fn bivariate_degrees(univariate_degree: usize) -> (usize, usize) {
        //(((univariate_degree + 1) as f64).sqrt().ceil() as usize).next_power_of_two() - 1;
        let sqrt = (((univariate_degree + 1) as f64).sqrt().ceil() as usize).next_power_of_two();
        // Skew split between bivariate degrees to account for KZG being less expensive than MIPP
        let skew_factor = if sqrt >= 32 { 16_usize } else { sqrt / 2 };
        (sqrt / skew_factor - 1, sqrt * skew_factor - 1)
    }

    fn parse_bivariate_degrees_from_srs(srs: &KZGProverKey<P>) -> (usize, usize) {
        let x_degree = (srs.h_beta_powers().len() - 1) / 2;
        let y_degree = srs.len() - 1;
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

    pub fn setup<R: Rng + CryptoRng>(rng: &mut R, degree: usize) -> Result<SRS<P>, Error> {
        let (x_degree, y_degree) = Self::bivariate_degrees(degree);
        BivariatePolynomialCommitment::<P, ProofTranscript>::setup(rng, x_degree, y_degree)
    }

    pub fn commit(
        srs: &(Vec<P::G2>, KZGProverKey<P>),
        polynomial: &UnivariatePolynomial<P::ScalarField>,
    ) -> Result<(PairingOutput<P>, Vec<P::G1>), Error> {
        let bivariate_degrees = Self::parse_bivariate_degrees_from_srs(&srs.1);
        BivariatePolynomialCommitment::<P, ProofTranscript>::commit(
            srs,
            &Self::bivariate_form(bivariate_degrees, polynomial),
        )
    }

    pub fn open(
        srs: &(Vec<P::G2>, KZGProverKey<P>),
        polynomial: &UnivariatePolynomial<P::ScalarField>,
        y_polynomial_comms: &[P::G1],
        point: &P::ScalarField,
        transcript: &mut ProofTranscript,
    ) -> Result<OpeningProof<P>, Error> {
        let (x_degree, y_degree) = Self::parse_bivariate_degrees_from_srs(&srs.1);
        let y = *point;
        let x = point.pow(vec![(y_degree + 1) as u64]);
        BivariatePolynomialCommitment::<P, ProofTranscript>::open(
            srs,
            &Self::bivariate_form((x_degree, y_degree), polynomial),
            y_polynomial_comms,
            &(x, y),
            transcript,
        )
    }

    pub fn verify(
        v_srs: &KZGVerifierKey<P>,
        max_degree: usize,
        com: &PairingOutput<P>,
        point: &P::ScalarField,
        eval: &P::ScalarField,
        proof: &OpeningProof<P>,
        transcript: &mut ProofTranscript,
    ) -> Result<bool, Error> {
        let (_, y_degree) = Self::bivariate_degrees(max_degree);
        let y = *point;
        let x = y.pow(vec![(y_degree + 1) as u64]);
        BivariatePolynomialCommitment::<P, ProofTranscript>::verify(
            v_srs,
            com,
            &(x, y),
            eval,
            proof,
            transcript,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::transcript::KeccakTranscript;
    use ark_bn254::Bn254;
    use ark_ff::UniformRand;
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use std::sync::Arc;

    const BIVARIATE_X_DEGREE: usize = 7;
    const BIVARIATE_Y_DEGREE: usize = 7;
    //const UNIVARIATE_DEGREE: usize = 56;
    const UNIVARIATE_DEGREE: usize = 65535;
    //const UNIVARIATE_DEGREE: usize = 1048575;

    type TestBivariatePolyCommitment = BivariatePolynomialCommitment<Bn254, KeccakTranscript>;
    type TestUnivariatePolyCommitment = UnivariatePolynomialCommitment<Bn254, KeccakTranscript>;

    #[test]
    fn bivariate_poly_commit_test() {
        let mut rng = StdRng::seed_from_u64(0u64);
        let srs =
            TestBivariatePolyCommitment::setup(&mut rng, BIVARIATE_X_DEGREE, BIVARIATE_Y_DEGREE)
                .unwrap();
        let ck = srs.get_commitment_keys();
        let (p_srs, v_srs) = SRS::trim(Arc::new(srs), BIVARIATE_X_DEGREE);

        let mut y_polynomials = Vec::new();
        for _ in 0..BIVARIATE_X_DEGREE + 1 {
            let mut y_polynomial_coeffs = vec![];
            for _ in 0..BIVARIATE_Y_DEGREE + 1 {
                y_polynomial_coeffs.push(<Bn254 as Pairing>::ScalarField::rand(&mut rng));
            }
            y_polynomials.push(UnivariatePolynomial::from_coeff(y_polynomial_coeffs));
        }
        let bivariate_polynomial = BivariatePolynomial { y_polynomials };
        let srs = (ck, p_srs);

        // Commit to polynomial
        let (com, y_polynomial_comms) =
            TestBivariatePolyCommitment::commit(&srs, &bivariate_polynomial).unwrap();

        let mut transcript = KeccakTranscript::new(b"test");

        // Evaluate at challenge point
        let point = (UniformRand::rand(&mut rng), UniformRand::rand(&mut rng));
        let eval_proof = TestBivariatePolyCommitment::open(
            &srs,
            &bivariate_polynomial,
            &y_polynomial_comms,
            &point,
            &mut transcript,
        )
        .unwrap();
        let eval = bivariate_polynomial.evaluate(&point);

        let mut transcript = KeccakTranscript::new(b"test");

        // Verify proof
        assert!(TestBivariatePolyCommitment::verify(
            &v_srs,
            &com,
            &point,
            &eval,
            &eval_proof,
            &mut transcript
        )
        .unwrap());
    }

    // `cargo test univariate_poly_commit_test --release --features print-trace -- --ignored --nocapture`
    // #[ignore]
    #[test]
    fn univariate_poly_commit_test() {
        let mut rng = StdRng::seed_from_u64(0u64);
        let srs = TestUnivariatePolyCommitment::setup(&mut rng, UNIVARIATE_DEGREE).unwrap();
        let ck = srs.get_commitment_keys();
        let powers_len = srs.g1_powers.len();
        let (p_srs, v_srs) = SRS::trim(Arc::new(srs), powers_len - 1);
        // let v_srs = srs.0.get_verifier_key();

        let mut polynomial_coeffs = vec![];
        for _ in 0..UNIVARIATE_DEGREE + 1 {
            polynomial_coeffs.push(<Bn254 as Pairing>::ScalarField::rand(&mut rng));
        }
        let polynomial = UnivariatePolynomial::from_coeff(polynomial_coeffs);

        let srs = (ck, p_srs);

        // Commit to polynomial
        let (com, y_polynomial_comms) =
            TestUnivariatePolyCommitment::commit(&srs, &polynomial).unwrap();

        let mut transcript = KeccakTranscript::new(b"test");
        // Evaluate at challenge point
        let point = UniformRand::rand(&mut rng);
        let eval_proof = TestUnivariatePolyCommitment::open(
            &srs,
            &polynomial,
            &y_polynomial_comms,
            &point,
            &mut transcript,
        )
        .unwrap();
        let eval = polynomial.evaluate(&point);
        let mut transcript = KeccakTranscript::new(b"test");

        // Verify proof
        assert!(TestUnivariatePolyCommitment::verify(
            &v_srs,
            UNIVARIATE_DEGREE,
            &com,
            &point,
            &eval,
            &eval_proof,
            &mut transcript,
        )
        .unwrap());
    }
}
