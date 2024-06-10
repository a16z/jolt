use super::sumcheck::SumcheckInstanceProof;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::{BatchType, CommitmentScheme};
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::utils::math::Math;
use crate::utils::transcript::{AppendToTranscript, ProofTranscript};
use ark_serialize::*;
use itertools::Itertools;
use thiserror::Error;

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct QuarkGrandProductProof<C: CommitmentScheme> {
    sumcheck_proof: SumcheckInstanceProof<C::Field>,
    v_commitment: Vec<C::Commitment>,
    f_commitment: Vec<C::Commitment>,
    claimed_eval_f_0_r: (Vec<C::Field>, C::BatchedProof),
    claimed_eval_f_1_r: (Vec<C::Field>, C::BatchedProof),
    claimed_eval_f_r_0: (Vec<C::Field>, C::BatchedProof),
    claimed_eval_f_r_1: (Vec<C::Field>, C::BatchedProof),
    sum_openings: C::BatchedProof,
    v_opening_proof: C::BatchedProof,
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

pub trait QuarkBatchedGrandProduct<C: CommitmentScheme>: Sized {
    type Leaves = Vec<C::Field>;
    /// Computes a grand product proof using the Section 5 technique from Quarks Paper
    /// First - Extends the evals of v to create an f poly, then commits to it and evals
    /// Then - Constructs a g poly and preforms sumcheck proof that sum == 0
    /// Finally - computes opening proofs for a random sampled during sumcheck proof and returns
    fn prove(
        v: Vec<DensePolynomial<C::Field>>,
        transcript: &mut ProofTranscript,
        setup: &C::Setup,
    ) -> (Self, Vec<C::Field>);

    /// Verifies the given grand product proof.
    fn verify(
        self,
        claim: &[C::Field],
        transcript: &mut ProofTranscript,
        n_rounds: usize,
        setup: &C::Setup,
    ) -> Result<(), QuarkError>;
}

impl<C: CommitmentScheme> QuarkBatchedGrandProduct<C> for QuarkGrandProductProof<C> {
    /// Computes a grand product proof using the Section 5 technique from Quarks Paper
    /// First - Extends the evals of v to create an f poly, then commits to it and evals
    /// Then - Constructs a g poly and preforms sumcheck proof that sum == 0
    /// Finally - computes opening proofs for a random sampled during sumcheck proof and returns
    fn prove(
        v_polys: Vec<DensePolynomial<C::Field>>,
        transcript: &mut ProofTranscript,
        setup: &C::Setup,
    ) -> (Self, Vec<C::Field>) {
        // TODO (alpeh_v): all v the same size?
        let v_length = v_polys[0].len();
        let v_variables = v_length.log_2();

        let mut f_polys = Vec::<DensePolynomial<C::Field>>::new();
        let mut sumcheck_polys = Vec::<DensePolynomial<C::Field>>::new();
        let mut products = Vec::<C::Field>::new();

        for v in v_polys.iter() {
            let (f, f_1_r, f_r_0, f_r_1, p) = v_into_f::<C>(v);
            f_polys.push(f);
            sumcheck_polys.push(f_1_r);
            sumcheck_polys.push(f_r_0);
            sumcheck_polys.push(f_r_1);
            products.push(p);
        }

        // We bind to these polynomials
        transcript.append_scalars(b"grand product claim", &products);
        let v_commitment = C::batch_commit_polys(&v_polys, setup, BatchType::Big);
        let f_commitment = C::batch_commit_polys(&f_polys, setup, BatchType::Big);
        for v in v_commitment.iter() {
            v.append_to_transcript(b"v commitment", transcript);
        }
        for f in f_commitment.iter() {
            f.append_to_transcript(b"f commitment", transcript);
        }

        // Now we do the sumcheck using the prove arbitrary
        // First instantiate our polynomials
        let tau: Vec<C::Field> = transcript.challenge_vector(b"element for eval poly", v_variables);
        let evals = DensePolynomial::new(EqPolynomial::evals(&tau));
        //We add evals as the last polynomial in the sumcheck
        sumcheck_polys.push(evals);

        // Sample a constant to do a random linear combination to combine the sumchecks
        let r_combination: Vec<C::Field> =
            transcript.challenge_vector(b"for the random linear comb", v_polys.len());

        // We define a closure using vals[0] = eq(tau, x), vals[1] = f(1, x), vals[1] = f(x, 0), vals[2] = f(x, 1)
        let output_check_fn = |vals: &[C::Field]| -> C::Field {
            let eval = vals[vals.len() - 1];
            let mut sum = C::Field::zero();

            for i in 0..(vals.len() / 3) {
                sum += r_combination[i] * (vals[i * 3] - vals[3 * i + 1] * vals[3 * i + 2]);
            }
            sum * eval
        };

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

        // Interface for the batch proof is &[&Poly] but we have &[Poly]
        let borrowed_f: Vec<&DensePolynomial<C::Field>> = f_polys.iter().collect();
        let borrowed_v: Vec<&DensePolynomial<C::Field>> = v_polys.iter().collect();

        // TODO (aleph_v) - Batch opens and a line reduction to make this 3 openings
        let mut challenge_0_x = vec![C::Field::zero()];
        challenge_0_x.append(&mut x.clone());
        let claimed_eval_f_0_r = open_and_prove::<C>(&challenge_0_x, &f_polys, setup, transcript);

        let mut challenge_1_x = vec![C::Field::one()];
        challenge_1_x.append(&mut x.clone());
        let claimed_eval_f_1_r = open_and_prove::<C>(&challenge_1_x, &f_polys, setup, transcript);

        let mut challenge_x_0 = x.clone();
        challenge_x_0.push(C::Field::zero());
        let claimed_eval_f_r_0 = open_and_prove::<C>(&challenge_x_0, &f_polys, setup, transcript);

        let mut challenge_x_1 = x.clone();
        challenge_x_1.push(C::Field::one());
        let claimed_eval_f_r_1 = open_and_prove::<C>(&challenge_x_1, &f_polys, setup, transcript);

        let mut challenge_sum = vec![C::Field::one(); x.len()];
        challenge_sum.push(C::Field::zero());
        // Here we don't calculate an eval because we should know it from the product recorded above
        let sum_openings = C::batch_prove(
            setup,
            &borrowed_f,
            &challenge_sum,
            &products,
            BatchType::Big,
            transcript,
        );

        // Here we don't calculate an eval because it should be equal to f(0, x) which is the first point we open
        let v_opening_proof = C::batch_prove(
            setup,
            &borrowed_v,
            &x,
            &claimed_eval_f_0_r.0,
            BatchType::Big,
            transcript,
        );

        (
            Self {
                sumcheck_proof,
                v_commitment,
                f_commitment,
                claimed_eval_f_0_r,
                claimed_eval_f_1_r,
                claimed_eval_f_r_0,
                claimed_eval_f_r_1,
                sum_openings,
                v_opening_proof,
            },
            products,
        )
    }

    /// Verifies the given grand product proof.
    fn verify(
        self,
        claims: &[C::Field],
        transcript: &mut ProofTranscript,
        n_rounds: usize,
        setup: &C::Setup,
    ) -> Result<(), QuarkError> {
        // First we append the claimed values for the commitment and the product
        transcript.append_scalars(b"grand product claim", claims);
        for v in self.v_commitment.iter() {
            v.append_to_transcript(b"v commitment", transcript);
        }
        for f in self.f_commitment.iter() {
            f.append_to_transcript(b"f commitment", transcript);
        }

        //Next sample the tau and construct the evals poly
        let tau: Vec<C::Field> = transcript.challenge_vector(b"element for eval poly", n_rounds);
        let r_combination: Vec<C::Field> =
            transcript.challenge_vector(b"for the random linear comb", self.v_commitment.len());

        // To complete the sumcheck proof we have to validate that our polynomial openings match and are right.
        let (expected, r) = self
            .sumcheck_proof
            .verify(C::Field::zero(), n_rounds, 3, transcript)
            .map_err(|_| QuarkError::InvalidQuarkSumcheck)?;

        // Again the batch verify expects we have a slice of borrows but we have a slice of Commitments
        let borrowed_f: Vec<&C::Commitment> = self.f_commitment.iter().collect();
        let borrowed_v: Vec<&C::Commitment> = self.v_commitment.iter().collect();

        // First we will confirm that the various opening of f are correct at the points defined by r
        let mut challenge_0_r = vec![C::Field::zero()];
        challenge_0_r.append(&mut r.clone());
        C::batch_verify(
            &self.claimed_eval_f_0_r.1,
            setup,
            &challenge_0_r,
            &self.claimed_eval_f_0_r.0,
            &borrowed_f,
            transcript,
        )
        .map_err(|_| QuarkError::InvalidOpeningProof)?;

        let mut challenge_1_r = vec![C::Field::one()];
        challenge_1_r.append(&mut r.clone());
        let one_r = self.claimed_eval_f_1_r.0;
        C::batch_verify(
            &self.claimed_eval_f_1_r.1,
            setup,
            &challenge_1_r,
            &one_r,
            &borrowed_f,
            transcript,
        )
        .map_err(|_| QuarkError::InvalidOpeningProof)?;

        let mut challenge_r_0 = r.clone();
        challenge_r_0.push(C::Field::zero());
        let r_0 = self.claimed_eval_f_r_0.0;
        C::batch_verify(
            &self.claimed_eval_f_r_0.1,
            setup,
            &challenge_r_0,
            &r_0,
            &borrowed_f,
            transcript,
        )
        .map_err(|_| QuarkError::InvalidOpeningProof)?;

        let mut challenge_r_1 = r.clone();
        challenge_r_1.push(C::Field::one());
        let r_1 = self.claimed_eval_f_r_1.0;
        C::batch_verify(
            &self.claimed_eval_f_r_1.1,
            setup,
            &challenge_r_1,
            &r_1,
            &borrowed_f,
            transcript,
        )
        .map_err(|_| QuarkError::InvalidOpeningProof)?;

        // We enforce that f opened at (1,1,...,1, 0) is in fact the product
        let mut challenge_sum = vec![C::Field::one(); r.len()];
        challenge_sum.push(C::Field::zero());
        C::batch_verify(
            &self.sum_openings,
            setup,
            &challenge_sum,
            claims,
            &borrowed_f,
            transcript,
        )
        .map_err(|_| QuarkError::InvalidOpeningProof)?;

        // Now we enforce that f(0, r) = v(r) via an opening proof on v
        C::batch_verify(
            &self.v_opening_proof,
            setup,
            &r,
            &self.claimed_eval_f_0_r.0,
            &borrowed_v,
            transcript,
        )
        .map_err(|_| QuarkError::InvalidOpeningProof)?;

        // Use the log(n) form to calculate eq(tau, r)
        let eq_eval: C::Field = r
            .iter()
            .zip_eq(tau.iter())
            .map(|(&r_gp, &r_sc)| r_gp * r_sc + (C::Field::one() - r_gp) * (C::Field::one() - r_sc))
            .product();

        // Finally we check that in fact the polynomial bound by the sumcheck is equal to eq(tau, r)*(f(1, r) - f(r, 0)*f(r,1))
        let mut result_from_openings = C::Field::zero();
        for i in 0..r_0.len() {
            result_from_openings += r_combination[i] * eq_eval * (one_r[i] - r_0[i] * r_1[i]);
        }

        if result_from_openings != expected {
            return Err(QuarkError::InvalidBinding);
        }

        Ok(())
    }
}

// Computes f and slices of f for the sumcheck
#[allow(clippy::type_complexity)]
fn v_into_f<C: CommitmentScheme>(
    v: &DensePolynomial<C::Field>,
) -> (
    DensePolynomial<C::Field>,
    DensePolynomial<C::Field>,
    DensePolynomial<C::Field>,
    DensePolynomial<C::Field>,
    C::Field,
) {
    let v_length = v.len();
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

    let f_r_0 = DensePolynomial::new(f_x_0);
    let f_r_1 = DensePolynomial::new(f_x_1);
    let f_1_r = DensePolynomial::new(f_1_x);

    // f(1, ..., 1, 0) = P
    let product = f_evals[2 * v_length - 2];
    let f = DensePolynomial::new(f_evals);

    (f, f_1_r, f_r_0, f_r_1, product)
}

// Open a set of polynomials at a point and return the openings and proof
fn open_and_prove<C: CommitmentScheme>(
    x: &[C::Field],
    f_polys: &[DensePolynomial<C::Field>],
    setup: &C::Setup,
    transcript: &mut ProofTranscript,
) -> (Vec<C::Field>, C::BatchedProof) {
    let chis = EqPolynomial::evals(x);
    let openings: Vec<C::Field> = f_polys
        .iter()
        .map(|f| f.evaluate_at_chi_low_optimized(&chis))
        .collect();
    // Batch proof requires  &[&]
    let borrowed: Vec<&DensePolynomial<C::Field>> = f_polys.iter().collect();

    let proof = C::batch_prove(setup, &borrowed, x, &openings, BatchType::Big, transcript);

    (openings, proof)
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

        let leaves_1: Vec<Fr> = std::iter::repeat_with(|| Fr::random(&mut rng))
            .take(LAYER_SIZE)
            .collect();
        let leaves_2: Vec<Fr> = std::iter::repeat_with(|| Fr::random(&mut rng))
            .take(LAYER_SIZE)
            .collect();
        let known_products = vec![leaves_1.iter().product(), leaves_2.iter().product()];
        let v_1 = DensePolynomial::new(leaves_1);
        let v_2 = DensePolynomial::new(leaves_2);
        let v = vec![v_1, v_2];
        let mut transcript: ProofTranscript = ProofTranscript::new(b"test_transcript");

        let srs = ZeromorphSRS::<Bn254>::setup(&mut rng, 1 << 9);
        let setup = srs.trim(1 << 9);

        let (proof, product) =
            QuarkGrandProductProof::<Zeromorph<Bn254>>::prove(v, &mut transcript, &setup);

        for (x, y) in product.iter().zip(known_products.iter()) {
            assert_eq!(x, y, "Proof doesn't produce correct products");
        }

        // Note resetting the transcript is important
        transcript = ProofTranscript::new(b"test_transcript");
        let result = proof.verify(&product, &mut transcript, 8, &setup);

        assert_eq!(result, Ok(()), "Proof doesn't verify");
    }
}
