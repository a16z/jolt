use super::sumcheck::{BatchedCubicSumcheck, SumcheckInstanceProof};
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::field::JoltField;
use crate::poly::{dense_mlpoly::DensePolynomial, unipoly::UniPoly};
// todo specify 
use crate::poly::commitment::commitment_scheme::{CommitmentScheme};
use crate::utils::math::Math;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::{AppendToTranscript, ProofTranscript};
use ark_ff::Zero;
use ark_serialize::*;
use itertools::Itertools;
use rayon::prelude::*;

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
    v_opening_proof: C::Proof
}

pub trait QuarkGrandProduct<C: CommitmentScheme>: Sized {

    /// Computes a grand product proof using the Section 5 technique from Quarks Paper
    /// First - Extends the evals of v to create an f poly, then commits to it and evals
    /// Then - Constructs a g poly and preforms sumcheck proof that sum == 0
    /// Finally - computes opening proofs for a random sampled during sumcheck proof and returns
    fn prove_grand_product(
        &mut self,
        v: &DensePolynomial<C::Field>,
        transcript: &mut ProofTranscript,
        setup: &C::Setup
    ) -> (QuarkGrandProductProof<C>, C::Field) {
        let v_length = v.len();
        let v_variables = v_length.log_2();

        assert_eq!(v_length, v_variables.pow2(), "Only grand products on length power of two are currently supported");
        let mut f_evals = vec![C::Field::zero(); 2*v_length];
        let (evals, _) = v.split_evals(v.len());
        f_evals[..v_length].clone_from_slice(evals);

        // Todo - problems when f length is equal to the usize
        for i in v_length..2*v_length {
            let i_shift_mod = (i << 1) % 2*v_length;
            // this just works tm
            f_evals[i] = f_evals[i_shift_mod]*f_evals[i_shift_mod + 1]
        }

        // We pull out the co-efficient which instantiate the lower d polys for the sumcheck
        let mut f_1_x = Vec::new();
        f_1_x.clone_from_slice(&f_evals[v_length..]);

        let mut f_x_0 = Vec::new();
        let mut f_x_1 = Vec::new();
        for (i, x) in (&f_evals).into_iter().enumerate() {
            if i % 2 == 0 {
                f_x_0.push(x.clone());
            } else {
                f_x_1.push(x.clone());
            }
        }

        let product = f_evals[2*v_length - 2];
        let f = DensePolynomial::new(f_evals);

        // We bind to these polynomials
        transcript.append_scalar(b"grand product claim",&product);
        let v_commitment = C::commit(v, setup);
        let f_commitment = C::commit(&f, setup);
        v_commitment.append_to_transcript(b"v commitment", transcript);
        f_commitment.append_to_transcript(b"f commitment", transcript);

        // Now we do the sumcheck using the prove arbitrary

        // First insatiate our polynomials
        let tau = transcript.challenge_vector(b"element for eval poly", v_variables);
        let evals = DensePolynomial::new(EqPolynomial::evals(&tau));
        let mut sumcheck_polys = vec![evals, DensePolynomial::new(f_1_x), DensePolynomial::new(f_x_0), DensePolynomial::new(f_x_1)];

        // We define a closure using vals[0] = eq(tau, x), vals[1] = f(1, x), vals[1] = f(x, 0), vals[2] = f(x, 0)
        let output_check_fn = |vals: &[C::Field]| -> C::Field { vals[0]*(vals[1] - vals[2]*vals[3]) };

        // Now run the sumcheck in arbitrary mode
        // TODO (aleph_v): Use a trait implementation as is done for batched cubic
        // Note - We use the final randomness from binding all variables (x) as the source random for the openings so the verifier can
        //        check that the base layer is the same as is committed too.
        // TODO - Do we need final_evals in struct
        let (sumcheck_proof, x, final_evals) = SumcheckInstanceProof::<C::Field>::prove_arbitrary::<_>(
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
        let proof_0_x = C::prove(&f, &challenge_0_x, transcript);
        let claimed_eval_f_0_r = (point_0_x, proof_0_x);

        let mut challenge_1_x = vec![C::Field::one()];
        challenge_1_x.append(&mut x.clone());
        let point_1_x = f.evaluate(&challenge_1_x);
        let proof_1_x = C::prove(&f, &challenge_1_x, transcript);
        let claimed_eval_f_1_r = (point_1_x, proof_1_x);

        let mut challenge_x_0 = x.clone();
        challenge_x_0.push(C::Field::zero());
        let point_x_0 = f.evaluate(&challenge_x_0);
        let proof_x_0 = C::prove(&f, &challenge_x_0, transcript);
        let claimed_eval_f_r_0 = (point_x_0, proof_x_0);

        let mut challenge_x_1 = x.clone();
        challenge_x_1.push(C::Field::one());
        let point_x_1 = f.evaluate(&challenge_x_1);
        let proof_x_1 = C::prove(&f, &challenge_x_1, transcript);
        let claimed_eval_f_r_1 = (point_x_1, proof_x_1);

        let mut challenge_sum = vec![C::Field::one()];
        challenge_sum.push(C::Field::zero());
        // Here we don't calculate an eval because we should know it from the product recorded above
        let sum_opening = C::prove(&f, &challenge_sum, transcript);

        // Here we don't calculate an eval because it should be equal to f(0, x) which is the first point we open
        let v_opening_proof = C::prove(&v, &x, transcript);


        (QuarkGrandProductProof::<C>{
            sumcheck_proof,
            v_commitment,
            f_commitment,
            claimed_eval_f_0_r,
            claimed_eval_f_1_r,
            claimed_eval_f_r_0,
            claimed_eval_f_r_1,
            sum_opening,
            v_opening_proof
        }, product)
    }


    // Verifies the given grand product proof.
    // fn verify_grand_product(
    //     proof: &QuarkGrandProductProof<F>,
    //     transcript: &mut ProofTranscript,
    // ) -> (Vec<F>, Vec<F>) {
    //     let mut r_grand_product: Vec<F> = Vec::new();
    //     let mut claims_to_verify = claims.to_owned();

    //     for (layer_index, layer_proof) in proof.layers.iter().enumerate() {
    //         // produce a fresh set of coeffs
    //         let coeffs: Vec<F> =
    //             transcript.challenge_vector(b"rand_coeffs_next_layer", claims_to_verify.len());
    //         // produce a joint claim
    //         let claim = claims_to_verify
    //             .iter()
    //             .zip(coeffs.iter())
    //             .map(|(&claim, &coeff)| claim * coeff)
    //             .sum();

    //         let (sumcheck_claim, r_sumcheck) =
    //             layer_proof.verify(claim, layer_index, 3, transcript);
    //         assert_eq!(claims.len(), layer_proof.left_claims.len());
    //         assert_eq!(claims.len(), layer_proof.right_claims.len());

    //         for (left, right) in layer_proof
    //             .left_claims
    //             .iter()
    //             .zip(layer_proof.right_claims.iter())
    //         {
    //             transcript.append_scalar(b"sumcheck left claim", left);
    //             transcript.append_scalar(b"sumcheck right claim", right);
    //         }

    //         assert_eq!(r_grand_product.len(), r_sumcheck.len());

    //         let eq_eval: F = r_grand_product
    //             .iter()
    //             .zip_eq(r_sumcheck.iter().rev())
    //             .map(|(&r_gp, &r_sc)| r_gp * r_sc + (F::one() - r_gp) * (F::one() - r_sc))
    //             .product();

    //         r_grand_product = r_sumcheck.into_iter().rev().collect();

    //         Self::verify_sumcheck_claim(
    //             &proof.layers,
    //             layer_index,
    //             &coeffs,
    //             sumcheck_claim,
    //             eq_eval,
    //             &mut claims_to_verify,
    //             &mut r_grand_product,
    //             transcript,
    //         );
    //     }

    //     (claims_to_verify, r_grand_product)
    // }
}



#[cfg(test)]
mod grand_product_tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_core::RngCore;


}
