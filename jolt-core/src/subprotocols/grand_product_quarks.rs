use super::sumcheck::SumcheckInstanceProof;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::field::JoltField;
use crate::utils::math::Math;
use crate::utils::transcript::{AppendToTranscript, ProofTranscript};
use ark_serialize::*;
use itertools::Itertools;
use thiserror::Error;

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct QuarkGrandProductProof<C: CommitmentScheme> {
    sumcheck_proof: SumcheckInstanceProof<C::Field>,
    v_commitment: C::Commitment,
    f_commitment: C::Commitment,
    claimed_eval_f_0_r: (C::Field, C::Proof),
    claimed_eval_f_1_r: (C::Field, C::Proof),
    claimed_eval_f_r_0: (C::Field, C::Proof),
    claimed_eval_f_r_1: (C::Field, C::Proof),
    sum_opening: C::Proof,
    v_opening_proof: C::Proof,
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum QuarkError {
    /// returned if the sumcheck fails
    #[error("InvalidSumcheck")]
    InvalidQuarkSumcheck,
    /// Returned if a quark opening proof fails
    #[error("IvalidOpeningProof")]
    InvalidOpeningProof,
    /// Returned if eq(tau, r)*(f(1, r) - f(r, 0)*f(r,1)) does not match the result from sumcheck
    #[error("IvalidOpeningProof")]
    InvalidBinding,
}

pub trait QuarkGrandProduct<C: CommitmentScheme>: Sized {
    /// Computes a grand product proof using the Section 5 technique from Quarks Paper
    /// First - Extends the evals of v to create an f poly, then commits to it and evals
    /// Then - Constructs a g poly and preforms sumcheck proof that sum == 0
    /// Finally - computes opening proofs for a random sampled during sumcheck proof and returns
    fn prove(
        v: &DensePolynomial<C::Field>,
        transcript: &mut ProofTranscript,
        setup: &C::Setup,
    ) -> (Self, C::Field);

    /// Verifies the given grand product proof.
    fn verify(
        self,
        claim: &C::Field,
        transcript: &mut ProofTranscript,
        n_rounds: usize,
        setup: &C::Setup,
    ) -> Result<(), QuarkError>;
}

impl<C: CommitmentScheme> QuarkGrandProduct<C> for QuarkGrandProductProof<C> {
    /// Computes a grand product proof using the Section 5 technique from Quarks Paper
    /// First - Extends the evals of v to create an f poly, then commits to it and evals
    /// Then - Constructs a g poly and preforms sumcheck proof that sum == 0
    /// Finally - computes opening proofs for a random sampled during sumcheck proof and returns
    fn prove(
        v: &DensePolynomial<C::Field>,
        transcript: &mut ProofTranscript,
        setup: &C::Setup,
    ) -> (Self, C::Field) {
        let v_length = v.len();
        let v_variables = v_length.log_2();

        assert_eq!(
            v_length,
            v_variables.pow2(),
            "Only grand products on length power of two are currently supported"
        );

        let mut f_evals = vec![C::Field::zero(); 2 * v_length];
        let (evals, _) = v.split_evals(v.len());
        f_evals[..v_length].clone_from_slice(evals);

        // Todo (aleph_v) - problems when f length is equal to the usize
        for i in v_length..2 * v_length {
            let i_shift_mod = (i << 1) % (2 * v_length);
            // The transform follows the logic of the paper and to accumulate
            // the partial sums into the correct indices.
            f_evals[i] = f_evals[i_shift_mod] * f_evals[i_shift_mod + 1]
        }

        // We pull out the co-efficient which instantiate the lower d polys for the sumcheck
        let f_1_x = f_evals[v_length..].to_vec();

        let mut f_x_0 = Vec::new();
        let mut f_x_1 = Vec::new();
        for (i, x) in f_evals.iter().enumerate() {
            if i % 2 == 0 {
                f_x_0.push(*x);
            } else {
                f_x_1.push(*x);
            }
        }

        // f(1, ..., 1, 0) = P
        let product = f_evals[2 * v_length - 2];
        let f = DensePolynomial::new(f_evals);

        // We bind to these polynomials
        transcript.append_scalar(b"grand product claim", &product);
        let v_commitment = C::commit(v, setup);
        let f_commitment = C::commit(&f, setup);
        v_commitment.append_to_transcript(b"v commitment", transcript);
        f_commitment.append_to_transcript(b"f commitment", transcript);

        // Now we do the sumcheck using the prove arbitrary

        // First instantiate our polynomials
        let tau = transcript.challenge_vector(b"element for eval poly", v_variables);
        let evals = DensePolynomial::new(EqPolynomial::evals(&tau));
        let mut sumcheck_polys = vec![
            evals,
            DensePolynomial::new(f_1_x),
            DensePolynomial::new(f_x_0),
            DensePolynomial::new(f_x_1),
        ];

        // We define a closure using vals[0] = eq(tau, x), vals[1] = f(1, x), vals[1] = f(x, 0), vals[2] = f(x, 1)
        let output_check_fn =
            |vals: &[C::Field]| -> C::Field { vals[0] * (vals[1] - vals[2] * vals[3]) };

        // Now run the sumcheck in arbitrary mode
        // TODO (aleph_v): Use a trait implementation as is done for batched cubic
        // Note - We use the final randomness from binding all variables (x) as the source random for the openings so the verifier can
        //        check that the base layer is the same as is committed too.
        let (sumcheck_proof, x, _) = SumcheckInstanceProof::<C::Field>::prove_arbitrary::<_>(
            &C::Field::zero(),
            v_variables,
            &mut sumcheck_polys,
            output_check_fn,
            3,
            transcript,
        );

        // TODO (aleph_v) - Batch opens and a line reduction to make this 3 openings
        let mut challenge_0_x = vec![C::Field::zero()];
        challenge_0_x.append(&mut x.clone());
        let point_0_x = f.evaluate(&challenge_0_x);
        let proof_0_x = C::prove(setup, &f, &challenge_0_x, transcript);
        let claimed_eval_f_0_r = (point_0_x, proof_0_x);

        let mut challenge_1_x = vec![C::Field::one()];
        challenge_1_x.append(&mut x.clone());
        let point_1_x = f.evaluate(&challenge_1_x);
        let proof_1_x = C::prove(setup, &f, &challenge_1_x, transcript);
        let claimed_eval_f_1_r = (point_1_x, proof_1_x);

        let mut challenge_x_0 = x.clone();
        challenge_x_0.push(C::Field::zero());
        let point_x_0 = f.evaluate(&challenge_x_0);
        let proof_x_0 = C::prove(setup, &f, &challenge_x_0, transcript);
        let claimed_eval_f_r_0 = (point_x_0, proof_x_0);

        let mut challenge_x_1 = x.clone();
        challenge_x_1.push(C::Field::one());
        let point_x_1 = f.evaluate(&challenge_x_1);
        let proof_x_1 = C::prove(setup, &f, &challenge_x_1, transcript);
        let claimed_eval_f_r_1 = (point_x_1, proof_x_1);

        let mut challenge_sum = vec![C::Field::one(); x.len()];
        challenge_sum.push(C::Field::zero());
        // Here we don't calculate an eval because we should know it from the product recorded above
        let sum_opening = C::prove(setup, &f, &challenge_sum, transcript);

        // Here we don't calculate an eval because it should be equal to f(0, x) which is the first point we open
        let v_opening_proof = C::prove(setup, v, &x, transcript);

        (
            Self {
                sumcheck_proof,
                v_commitment,
                f_commitment,
                claimed_eval_f_0_r,
                claimed_eval_f_1_r,
                claimed_eval_f_r_0,
                claimed_eval_f_r_1,
                sum_opening,
                v_opening_proof,
            },
            product,
        )
    }

    /// Verifies the given grand product proof.
    fn verify(
        self,
        claim: &C::Field,
        transcript: &mut ProofTranscript,
        n_rounds: usize,
        setup: &C::Setup,
    ) -> Result<(), QuarkError> {
        // First we append the claimed values for the commitment and the product
        transcript.append_scalar(b"grand product claim", claim);
        self.v_commitment
            .append_to_transcript(b"v commitment", transcript);
        self.f_commitment
            .append_to_transcript(b"f commitment", transcript);

        //Next sample the tau and construct the evals poly
        let tau: Vec<C::Field> = transcript.challenge_vector(b"element for eval poly", n_rounds);

        // To complete the sumcheck proof we have to validate that our polynomial openings match and are right.
        let (expected, r) = self
            .sumcheck_proof
            .verify(C::Field::zero(), n_rounds, 3, transcript)
            .map_err(|_| QuarkError::InvalidQuarkSumcheck)?;

        // First we will confirm that the various opening of f are correct at the points defined by r
        let mut challenge_0_r = vec![C::Field::zero()];
        challenge_0_r.append(&mut r.clone());
        C::verify(
            &self.claimed_eval_f_0_r.1,
            setup,
            transcript,
            &challenge_0_r,
            &self.claimed_eval_f_0_r.0,
            &self.f_commitment,
        )
        .map_err(|_| QuarkError::InvalidOpeningProof)?;

        let mut challenge_1_r = vec![C::Field::one()];
        challenge_1_r.append(&mut r.clone());
        let point_f_1_r = &self.claimed_eval_f_1_r.0;
        C::verify(
            &self.claimed_eval_f_1_r.1,
            setup,
            transcript,
            &challenge_1_r,
            point_f_1_r,
            &self.f_commitment,
        )
        .map_err(|_| QuarkError::InvalidOpeningProof)?;

        let mut challenge_r_0 = r.clone();
        challenge_r_0.push(C::Field::zero());
        let point_f_r_0 = &self.claimed_eval_f_r_0.0;
        C::verify(
            &self.claimed_eval_f_r_0.1,
            setup,
            transcript,
            &challenge_r_0,
            point_f_r_0,
            &self.f_commitment,
        )
        .map_err(|_| QuarkError::InvalidOpeningProof)?;

        let mut challenge_r_1 = r.clone();
        challenge_r_1.push(C::Field::one());
        let point_f_r_1 = &self.claimed_eval_f_r_1.0;
        C::verify(
            &self.claimed_eval_f_r_1.1,
            setup,
            transcript,
            &challenge_r_1,
            point_f_r_1,
            &self.f_commitment,
        )
        .map_err(|_| QuarkError::InvalidOpeningProof)?;

        // We enforce that f opened at (1,1,...,1, 0) is in fact the product
        let mut challenge_sum = vec![C::Field::one(); r.len()];
        challenge_sum.push(C::Field::zero());
        C::verify(
            &self.sum_opening,
            setup,
            transcript,
            &challenge_sum,
            claim,
            &self.f_commitment,
        )
        .map_err(|_| QuarkError::InvalidOpeningProof)?;

        // Now we enforce that f(0, r) = v(r) via an opening proof on v
        C::verify(
            &self.v_opening_proof,
            setup,
            transcript,
            &r,
            &self.claimed_eval_f_0_r.0,
            &self.v_commitment,
        )
        .map_err(|_| QuarkError::InvalidOpeningProof)?;

        // Use the log(n) form to calculate eq(tau, r)
        let eq_eval: C::Field = r
            .iter()
            .zip_eq(tau.iter())
            .map(|(&r_gp, &r_sc)| r_gp * r_sc + (C::Field::one() - r_gp) * (C::Field::one() - r_sc))
            .product();

        // Finally we check that in fact the polynomial bound by the sumcheck is equal to eq(tau, r)*(f(1, r) - f(r, 0)*f(r,1))
        let result_from_openings = eq_eval * (*point_f_1_r - *point_f_r_0 * point_f_r_1);
        if result_from_openings != expected {
            return Err(QuarkError::InvalidBinding);
        }

        Ok(())
    }
}

#[cfg(test)]
mod quark_grand_product_tests {
    use super::*;
    use crate::poly::commitment::zeromorph::*;
    use ark_bn254::{Bn254, Fr};
    use rand_core::SeedableRng;

    #[test]
    fn quark_e2e() {
        const LAYER_SIZE: usize = 1 << 8;

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(9 as u64);

        let leaves: Vec<Fr> = std::iter::repeat_with(|| Fr::random(&mut rng))
            .take(LAYER_SIZE)
            .collect();
        let known_product: Fr = leaves.iter().product();
        let v = DensePolynomial::new(leaves);
        let mut transcript: ProofTranscript = ProofTranscript::new(b"test_transcript");

        let srs = ZeromorphSRS::<Bn254>::setup(&mut rng, 1 << 9);
        let setup = srs.trim(1 << 9);

        let (proof, product) =
            QuarkGrandProductProof::<Zeromorph<Bn254>>::prove(&v, &mut transcript, &setup);

        assert_eq!(&product, &known_product, "Not the product of v");

        // Note resetting the transcript is important
        transcript = ProofTranscript::new(b"test_transcript");
        let result = proof.verify(&product, &mut transcript, 8, &setup);

        assert_eq!(result, Ok(()), "Proof doesn't verify");
    }
}
