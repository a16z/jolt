use std::marker::PhantomData;
use tracing::{span, Level};

use crate::field::JoltField;
use crate::jolt::vm::JoltCommitments;
use crate::jolt::vm::JoltPolynomials;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::multilinear_polynomial::PolynomialEvaluation;
use crate::poly::opening_proof::ProverOpeningAccumulator;
use crate::poly::opening_proof::VerifierOpeningAccumulator;
use crate::poly::split_eq_poly::SplitEqPolynomial;
use crate::r1cs::key::UniformSpartanKey;
use crate::utils::math::Math;
use crate::utils::thread::drop_in_background_thread;

use crate::utils::transcript::Transcript;
use ark_serialize::CanonicalDeserialize;
use ark_serialize::CanonicalSerialize;

use thiserror::Error;

use crate::{
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::{EqPlusOnePolynomial, EqPolynomial},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
};

use super::builder::CombinedUniformBuilder;
use super::inputs::ConstraintInput;

use rayon::prelude::*;

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum SpartanError {
    /// returned if the supplied row or col in (row,col,val) tuple is out of range
    #[error("InvalidIndex")]
    InvalidIndex,

    /// returned when an invalid sum-check proof is provided
    #[error("InvalidSumcheckProof")]
    InvalidSumcheckProof,

    /// returned when the recursive sumcheck proof fails
    #[error("InvalidOuterSumcheckProof")]
    InvalidOuterSumcheckProof,

    /// returned when the final sumcheck opening proof fails
    #[error("InvalidOuterSumcheckClaim")]
    InvalidOuterSumcheckClaim,

    /// returned when the recursive sumcheck proof fails
    #[error("InvalidInnerSumcheckProof")]
    InvalidInnerSumcheckProof,

    /// returned when the final sumcheck opening proof fails
    #[error("InvalidInnerSumcheckClaim")]
    InvalidInnerSumcheckClaim,

    /// returned if the supplied witness is not of the right length
    #[error("InvalidWitnessLength")]
    InvalidWitnessLength,

    /// returned when an invalid PCS proof is provided
    #[error("InvalidPCSProof")]
    InvalidPCSProof,
}

/// A succinct proof of knowledge of a witness to a relaxed R1CS instance
/// The proof is produced using Spartan's combination of the sum-check and
/// the commitment to a vector viewed as a polynomial commitment
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct UniformSpartanProof<
    const C: usize,
    I: ConstraintInput,
    F: JoltField,
    ProofTranscript: Transcript,
> {
    _inputs: PhantomData<I>,
    pub(crate) outer_sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub(crate) outer_sumcheck_claims: (F, F, F),
    pub(crate) inner_sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub(crate) shift_sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub(crate) shift_sumcheck_claim: F,
    pub(crate) claimed_witness_evals: Vec<F>,
    pub(crate) shift_sumcheck_witness_evals: Vec<F>,
    _marker: PhantomData<ProofTranscript>,
}

impl<const C: usize, I, F, ProofTranscript> UniformSpartanProof<C, I, F, ProofTranscript>
where
    I: ConstraintInput,
    F: JoltField,
    ProofTranscript: Transcript,
{
    #[tracing::instrument(skip_all, name = "Spartan::setup")]
    pub fn setup(
        constraint_builder: &CombinedUniformBuilder<C, F, I>,
        padded_num_steps: usize,
    ) -> UniformSpartanKey<C, I, F> {
        assert_eq!(
            padded_num_steps,
            constraint_builder.uniform_repeat().next_power_of_two()
        );
        UniformSpartanKey::from_builder(constraint_builder)
    }

    #[tracing::instrument(skip_all, name = "Spartan::prove")]
    pub fn prove<PCS>(
        constraint_builder: &CombinedUniformBuilder<C, F, I>,
        key: &UniformSpartanKey<C, I, F>,
        polynomials: &JoltPolynomials<F>,
        opening_accumulator: &mut ProverOpeningAccumulator<F, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<Self, SpartanError>
    where
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
    {
        let flattened_polys: Vec<&MultilinearPolynomial<F>> = I::flatten::<C>()
            .iter()
            .map(|var| var.get_ref(polynomials))
            .collect();

        let num_rounds_x = key.num_rows_bits();

        /* Sumcheck 1: Outer sumcheck */

        let tau = (0..num_rounds_x)
            .map(|_i| transcript.challenge_scalar())
            .collect::<Vec<F>>();
        let mut eq_tau = SplitEqPolynomial::new(&tau);

        let mut az_bz_cz_poly = constraint_builder.compute_spartan_Az_Bz_Cz(&flattened_polys);
        let (outer_sumcheck_proof, outer_sumcheck_r, outer_sumcheck_claims) =
            SumcheckInstanceProof::prove_spartan_cubic(
                num_rounds_x,
                &mut eq_tau,
                &mut az_bz_cz_poly,
                transcript,
            );
        let outer_sumcheck_r: Vec<F> = outer_sumcheck_r.into_iter().rev().collect();
        drop_in_background_thread((az_bz_cz_poly, eq_tau));

        ProofTranscript::append_scalars(transcript, &outer_sumcheck_claims);
        // claims from the end of sum-check
        // claim_Az is the (scalar) value v_A = \sum_y A(r_x, y) * z(r_x) where r_x is the sumcheck randomness
        let (claim_Az, claim_Bz, claim_Cz): (F, F, F) = (
            outer_sumcheck_claims[0],
            outer_sumcheck_claims[1],
            outer_sumcheck_claims[2],
        );

        /* Sumcheck 2: Inner sumcheck
            RLC of claims Az, Bz, Cz
            where claim_Az = \sum_{y_var} A(rx, y_var || rx_step) * z(y_var || rx_step)
                                + A_shift(..) * z_shift(..)
            and shift denotes the values at the next time step "rx_step+1" for cross-step constraints
            - A_shift(rx, y_var || rx_step) = \sum_t A(rx, y_var || t) * eq_plus_one(rx_step, t)
            - z_shift(y_var || rx_step) = \sum z(y_var || rx_step) * eq_plus_one(rx_step, t)
        */

        let num_steps = key.num_steps;
        let num_steps_bits = num_steps.ilog2() as usize;
        let num_vars_uniform = key.num_vars_uniform_padded().next_power_of_two();

        let inner_sumcheck_RLC: F = transcript.challenge_scalar();
        let claim_inner_joint = claim_Az
            + inner_sumcheck_RLC * claim_Bz
            + inner_sumcheck_RLC * inner_sumcheck_RLC * claim_Cz;

        let (rx_step, rx_constr) = outer_sumcheck_r.split_at(num_steps_bits);

        let (eq_rx_step, eq_plus_one_rx_step) = EqPlusOnePolynomial::evals(rx_step, None);
        let (eq_rx_step_r2, eq_plus_one_rx_step_r2) =
            EqPlusOnePolynomial::evals(rx_step, F::montgomery_r2());

        /* Compute the two polynomials provided as input to the second sumcheck:
           - poly_ABC: A(r_x, y_var || rx_step), A_shift(..) at all variables y_var
           - poly_z: z(y_var || rx_step), z_shift(..)
        */

        let poly_ABC = DensePolynomial::new(key.evaluate_matrix_mle_partial(
            rx_constr,
            rx_step,
            inner_sumcheck_RLC,
        ));

        // Binding z and z_shift polynomials at point rx_step
        let span = span!(Level::INFO, "binding_z_and_shift_z");
        let _guard = span.enter();

        let mut bind_z = vec![F::zero(); num_vars_uniform * 2];
        let mut bind_shift_z = vec![F::zero(); num_vars_uniform * 2];

        flattened_polys
            .par_iter()
            .zip(bind_z.par_iter_mut().zip(bind_shift_z.par_iter_mut()))
            .for_each(|(poly, (eval, eval_shifted))| {
                *eval = poly.dot_product(Some(&eq_rx_step), Some(&eq_rx_step_r2));
                *eval_shifted =
                    poly.dot_product(Some(&eq_plus_one_rx_step), Some(&eq_plus_one_rx_step_r2));
            });

        bind_z[num_vars_uniform] = F::one();

        drop(_guard);
        drop(span);

        let poly_z =
            DensePolynomial::new(bind_z.into_iter().chain(bind_shift_z.into_iter()).collect());
        assert_eq!(poly_z.len(), poly_ABC.len());

        let num_rounds_inner_sumcheck = poly_ABC.len().log_2();

        let mut polys = vec![
            MultilinearPolynomial::LargeScalars(poly_ABC),
            MultilinearPolynomial::LargeScalars(poly_z),
        ];

        let comb_func = |poly_evals: &[F]| -> F {
            assert_eq!(poly_evals.len(), 2);
            poly_evals[0] * poly_evals[1]
        };

        let (inner_sumcheck_proof, inner_sumcheck_r, _claims_inner) =
            SumcheckInstanceProof::prove_arbitrary(
                &claim_inner_joint,
                num_rounds_inner_sumcheck,
                &mut polys,
                comb_func,
                2,
                transcript,
            );

        drop_in_background_thread(polys);

        /*  Sumcheck 3: Shift sumcheck
            sumcheck claim is = z_shift(ry_var || rx_step) = \sum_t z(ry_var || t) * eq_plus_one(rx_step, t)
        */

        let ry_var = inner_sumcheck_r[1..].to_vec();
        let eq_ry_var = EqPolynomial::evals(&ry_var);
        let eq_ry_var_r2 = EqPolynomial::evals_with_r2(&ry_var);

        let mut bind_z_ry_var: Vec<F> = Vec::with_capacity(num_steps);

        let span = span!(Level::INFO, "bind_z_ry_var");
        let _guard = span.enter();
        let num_steps_unpadded = constraint_builder.uniform_repeat();
        (0..num_steps_unpadded) // unpadded number of steps is sufficient
            .into_par_iter()
            .map(|t| {
                flattened_polys
                    .iter()
                    .enumerate()
                    .map(|(i, poly)| poly.scale_coeff(t, eq_ry_var[i], eq_ry_var_r2[i]))
                    .sum()
            })
            .collect_into_vec(&mut bind_z_ry_var);
        drop(_guard);
        drop(span);

        let num_rounds_shift_sumcheck = num_steps_bits;
        assert_eq!(bind_z_ry_var.len(), eq_plus_one_rx_step.len());

        let mut shift_sumcheck_polys = vec![
            MultilinearPolynomial::from(bind_z_ry_var),
            MultilinearPolynomial::from(eq_plus_one_rx_step),
        ];

        let shift_sumcheck_claim = (0..1 << num_rounds_shift_sumcheck)
            .into_par_iter()
            .map(|i| {
                let params: Vec<F> = shift_sumcheck_polys
                    .iter()
                    .map(|poly| poly.get_coeff(i))
                    .collect();
                comb_func(&params)
            })
            .reduce(|| F::zero(), |acc, x| acc + x);

        let (shift_sumcheck_proof, shift_sumcheck_r, _shift_sumcheck_claims) =
            SumcheckInstanceProof::prove_arbitrary(
                &shift_sumcheck_claim,
                num_rounds_shift_sumcheck,
                &mut shift_sumcheck_polys,
                comb_func,
                2,
                transcript,
            );

        drop_in_background_thread(shift_sumcheck_polys);

        // Inner sumcheck evaluations: evaluate z on rx_step
        let (claimed_witness_evals, chis) =
            MultilinearPolynomial::batch_evaluate(&flattened_polys, rx_step);

        opening_accumulator.append(
            &flattened_polys,
            DensePolynomial::new(chis),
            rx_step.to_vec(),
            &claimed_witness_evals,
            transcript,
        );

        // Shift sumcheck evaluations: evaluate z on ry_var
        let (shift_sumcheck_witness_evals, chis2) =
            MultilinearPolynomial::batch_evaluate(&flattened_polys, &shift_sumcheck_r);

        opening_accumulator.append(
            &flattened_polys,
            DensePolynomial::new(chis2),
            shift_sumcheck_r.to_vec(),
            &shift_sumcheck_witness_evals,
            transcript,
        );

        // Outer sumcheck claims: [A(r_x), B(r_x), C(r_x)]
        let outer_sumcheck_claims = (
            outer_sumcheck_claims[0],
            outer_sumcheck_claims[1],
            outer_sumcheck_claims[2],
        );
        Ok(UniformSpartanProof {
            _inputs: PhantomData,
            outer_sumcheck_proof,
            outer_sumcheck_claims,
            inner_sumcheck_proof,
            shift_sumcheck_proof,
            shift_sumcheck_claim,
            claimed_witness_evals,
            shift_sumcheck_witness_evals,
            _marker: PhantomData,
        })
    }

    #[tracing::instrument(skip_all, name = "Spartan::verify")]
    pub fn verify<PCS>(
        &self,
        key: &UniformSpartanKey<C, I, F>,
        commitments: &JoltCommitments<PCS, ProofTranscript>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), SpartanError>
    where
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
        ProofTranscript: Transcript,
    {
        let num_rounds_x = key.num_rows_total().log_2();

        /* Sumcheck 1: Outer sumcheck
         */
        let tau = (0..num_rounds_x)
            .map(|_i| transcript.challenge_scalar())
            .collect::<Vec<F>>();

        let (claim_outer_final, outer_sumcheck_r) = self
            .outer_sumcheck_proof
            .verify(F::zero(), num_rounds_x, 3, transcript)
            .map_err(|_| SpartanError::InvalidOuterSumcheckProof)?;

        // Outer sumcheck is bound from the top, reverse the fiat shamir randomness
        let outer_sumcheck_r: Vec<F> = outer_sumcheck_r.into_iter().rev().collect();

        let (claim_Az, claim_Bz, claim_Cz) = self.outer_sumcheck_claims;
        let taus_bound_rx = EqPolynomial::new(tau).evaluate(&outer_sumcheck_r);
        let claim_outer_final_expected = taus_bound_rx * (claim_Az * claim_Bz - claim_Cz);
        if claim_outer_final != claim_outer_final_expected {
            return Err(SpartanError::InvalidOuterSumcheckClaim);
        }

        transcript.append_scalars(
            [
                self.outer_sumcheck_claims.0,
                self.outer_sumcheck_claims.1,
                self.outer_sumcheck_claims.2,
            ]
            .as_slice(),
        );

        /* Sumcheck 2: Inner sumcheck
           - claim is an RLC of claims_Az, Bz, Cz
           where claim_Az = \sum_{y_var} A(rx, y_var || rx_step) * z(y_var || rx_step)
                               + A_shift(..) * z_shift(..)
           - verifying it involves computing each term with randomness ry_var
        */
        let inner_sumcheck_RLC: F = transcript.challenge_scalar();
        let claim_inner_joint = self.outer_sumcheck_claims.0
            + inner_sumcheck_RLC * self.outer_sumcheck_claims.1
            + inner_sumcheck_RLC * inner_sumcheck_RLC * self.outer_sumcheck_claims.2;

        let num_rounds_inner_sumcheck = (2 * key.num_vars_uniform_padded()).log_2() + 1; // +1 for shift evals
        let (claim_inner_final, inner_sumcheck_r) = self
            .inner_sumcheck_proof
            .verify(claim_inner_joint, num_rounds_inner_sumcheck, 2, transcript)
            .map_err(|_| SpartanError::InvalidInnerSumcheckProof)?;

        let num_steps_bits = key.num_steps.log_2();

        let (rx_step, rx_constr) = outer_sumcheck_r.split_at(num_steps_bits);

        let r_non_uni = inner_sumcheck_r[0];
        let ry_var = inner_sumcheck_r[1..].to_vec();
        let eval_z =
            key.evaluate_z_mle_with_segment_evals(&self.claimed_witness_evals, &ry_var, true);

        let (eval_a, eval_b, eval_c) = key.evaluate_matrix_mle_full(rx_constr, &ry_var, &r_non_uni);

        let left_expected =
            eval_a + inner_sumcheck_RLC * eval_b + inner_sumcheck_RLC * inner_sumcheck_RLC * eval_c;
        let right_expected =
            (F::one() - r_non_uni) * eval_z + r_non_uni * self.shift_sumcheck_claim;

        let claim_inner_final_expected = left_expected * right_expected;
        if claim_inner_final != claim_inner_final_expected {
            return Err(SpartanError::InvalidInnerSumcheckClaim);
        }

        /* Sumcheck 3: Shift sumcheck
            - claim = \sum_t z(ry_var || t) * eq_plus_one(rx_step, t)
            - verifying it involves checking that claim = z(ry_var || r_t) * eq_plus_one(rx_step, r_t)
            where r_t = shift_sumcheck_r
        */

        let num_rounds_shift_sumcheck = num_steps_bits;
        let (claim_shift_sumcheck, shift_sumcheck_r) = self
            .shift_sumcheck_proof
            .verify(
                self.shift_sumcheck_claim,
                num_rounds_shift_sumcheck,
                2,
                transcript,
            )
            .map_err(|_| SpartanError::InvalidInnerSumcheckProof)?;

        let eval_z_shift_sumcheck = key.evaluate_z_mle_with_segment_evals(
            &self.shift_sumcheck_witness_evals,
            &ry_var,
            false,
        );
        let eq_plus_one_shift_sumcheck =
            EqPlusOnePolynomial::new(rx_step.to_vec()).evaluate(&shift_sumcheck_r);
        let claim_shift_sumcheck_expected = eval_z_shift_sumcheck * eq_plus_one_shift_sumcheck;

        if claim_shift_sumcheck != claim_shift_sumcheck_expected {
            return Err(SpartanError::InvalidInnerSumcheckClaim);
        }

        let flattened_commitments: Vec<_> = I::flatten::<C>()
            .iter()
            .map(|var| var.get_ref(commitments))
            .collect();

        opening_accumulator.append(
            &flattened_commitments,
            rx_step.to_vec(),
            &self.claimed_witness_evals.iter().collect::<Vec<_>>(),
            transcript,
        );

        opening_accumulator.append(
            &flattened_commitments,
            shift_sumcheck_r.to_vec(),
            &self.shift_sumcheck_witness_evals.iter().collect::<Vec<_>>(),
            transcript,
        );

        Ok(())
    }
}

// #[cfg(test)]
// mod test {
//     use ark_bn254::Fr;
//     use ark_std::One;

//     use crate::poly::commitment::{commitment_scheme::CommitShape, hyrax::HyraxScheme};

//     use super::*;

//     #[test]
//     fn integration() {
//         let (builder, key) = simp_test_builder_key();
//         let witness_segments: Vec<Vec<Fr>> = vec![
//             vec![Fr::one(), Fr::from(5), Fr::from(9), Fr::from(13)], /* Q */
//             vec![Fr::one(), Fr::from(5), Fr::from(9), Fr::from(13)], /* R */
//             vec![Fr::one(), Fr::from(5), Fr::from(9), Fr::from(13)], /* S */
//         ];

//         // Create a witness and commit
//         let witness_segments_ref: Vec<&[Fr]> = witness_segments
//             .iter()
//             .map(|segment| segment.as_slice())
//             .collect();
//         let gens = HyraxScheme::setup(&[CommitShape::new(16, BatchType::Small)]);
//         let witness_commitment =
//             HyraxScheme::batch_commit(&witness_segments_ref, &gens, BatchType::Small);

//         // Prove spartan!
//         let mut prover_transcript = ProofTranscript::new(b"stuff");
//         let proof =
//             UniformSpartanProof::<Fr, HyraxScheme<ark_bn254::G1Projective>>::prove_precommitted::<
//                 SimpTestIn,
//             >(
//                 &gens,
//                 builder,
//                 &key,
//                 witness_segments,
//                 todo!("opening accumulator"),
//                 &mut prover_transcript,
//             )
//             .unwrap();

//         let mut verifier_transcript = ProofTranscript::new(b"stuff");
//         let witness_commitment_ref: Vec<&_> = witness_commitment.iter().collect();
//         proof
//             .verify_precommitted(
//                 &key,
//                 witness_commitment_ref,
//                 &gens,
//                 &mut verifier_transcript,
//             )
//             .expect("Spartan verifier failed");
//     }
// }
