#![allow(clippy::len_without_is_empty)]

use std::marker::PhantomData;
use tracing::{span, Level}; 
use rayon::prelude::*;

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
    poly::{dense_mlpoly::DensePolynomial, eq_poly::{EqPolynomial, EqPlusOnePolynomial}},
    subprotocols::sumcheck::SumcheckInstanceProof,
};

use super::builder::CombinedUniformBuilder;
use super::inputs::ConstraintInput;

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum SpartanError {
    /// returned if the supplied row or col in (row,col,val) tuple is out of range
    #[error("InvalidIndex")]
    InvalidIndex,

    /// returned when an invalid sum-check proof is provided
    #[error("InvalidSumcheckProof")]
    InvalidSumcheckProof,

    /// returned when the recusive sumcheck proof fails
    #[error("InvalidOuterSumcheckProof")]
    InvalidOuterSumcheckProof,

    /// returned when the final sumcheck opening proof fails
    #[error("InvalidOuterSumcheckClaim")]
    InvalidOuterSumcheckClaim,

    /// returned when the recusive sumcheck proof fails
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
    pub(crate) claimed_witness_evals_shift_sumcheck: Vec<F>,
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
        let num_rounds_y = key.num_cols_total().log_2();

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
        
        /* Sumcheck 2: Inner sumcheck */
        /*
            sumcheck claim:  
         */

        let num_steps_padded = constraint_builder.uniform_repeat().next_power_of_two();
        let num_steps_bits = num_steps_padded.ilog2() as usize;

        let inner_sumcheck_RLC: F = transcript.challenge_scalar();
        let claim_inner_joint = claim_Az
            + inner_sumcheck_RLC * claim_Bz
            + inner_sumcheck_RLC * inner_sumcheck_RLC * claim_Cz;

        // this is the polynomial extended from the vector r_A * A(r_x, y) + r_B * B(r_x, y) + r_C * C(r_x, y) for all y
        let num_vars_padded = key.num_vars_uniform_padded();
        let num_constr_bits = constraint_builder.padded_rows_per_step().ilog2() as usize;
        let num_step_bits = outer_sumcheck_r.len() - num_constr_bits;

        let (rx_step, rx_con) = outer_sumcheck_r.split_at(num_step_bits);

        let eq_rx_step = EqPolynomial::evals(rx_step);

        let mut eq_plus_one_rx_step  = vec![];
        let span = span!(Level::INFO, "evals_eq_plus_one");
        {
            eq_plus_one_rx_step = EqPlusOnePolynomial::evals(rx_step);
        }

        // A(rx_con || rx_step, y_var || rx_step) evaluated at each y_var
        let poly_ABC = DensePolynomial::new(
            key.evaluate_r1cs_mle_rlc(rx_con, rx_step, inner_sumcheck_RLC)
        );

        let mut evals: Vec<F> = Vec::with_capacity(num_vars_padded * 2);
        let mut evals_shifted: Vec<F> = Vec::with_capacity(num_vars_padded * 2);

        let span = span!(Level::INFO, "evals_z");
        {
        evals = flattened_polys
            .iter()
            .map(|poly| {
                (0..poly.original_len())
                    .into_iter()
                    .map(|t| {
                        poly.get_coeff(t) *  eq_rx_step[t]
                    })
                    .sum()
            })
            .collect();
        evals.resize(evals.len().next_power_of_two(), F::zero());
        evals.push(F::one()); // Constant
        evals.resize(evals.len().next_power_of_two(), F::zero());
        }


        let span = span!(Level::INFO, "evals_z");
        {
        evals_shifted = flattened_polys
            .iter()
            .map(|poly| {
                (0..poly.original_len())
                .into_iter()
                .map(|t| {
                        poly.get_coeff(t) * eq_plus_one_rx_step[t] 
                    })
                    .sum()
            })
            .collect();
        evals_shifted.resize(evals.len(), F::zero());
        }

        let poly_z = DensePolynomial::new(evals.into_iter().chain(evals_shifted.into_iter()).collect());
        assert_eq!(poly_z.len(), poly_ABC.len());

        let num_rounds = poly_ABC.len().log_2();
        
        let mut polys = vec![
            MultilinearPolynomial::LargeScalars(poly_ABC), 
            MultilinearPolynomial::LargeScalars(poly_z)
        ]; 

        let comb_func = |poly_evals: &[F]| -> F {
            assert_eq!(poly_evals.len(), 2);
            poly_evals[0] * poly_evals[1]
        };

        let (inner_sumcheck_proof, inner_sumcheck_r, _claims_inner) =
            SumcheckInstanceProof::prove_arbitrary(
                &claim_inner_joint, // r_A * v_A + r_B * v_B + r_C * v_C
                num_rounds,
                &mut polys, // r_A * A(r_x, y) + r_B * B(r_x, y) + r_C * C(r_x, y) for all y
                comb_func,
                2,
                transcript,
            );
        drop_in_background_thread(polys);

        let r_y_var = inner_sumcheck_r[1..].to_vec();
        assert_eq!(r_y_var.len(), key.num_vars_uniform_padded().log_2() + 1);

        let eq_ry_var = EqPolynomial::evals(&r_y_var);

        /*  Sumcheck 3: the shift sumcheck */

        /*  Binding 2: evaluating z on r_y_var
            TODO(arasuarun): this might lead to inefficient memory paging 
            as we access each poly in flattened_poly num_steps_padded-many times.
        */ 
        let mut evals_z_r_y_var: Vec<F> = Vec::with_capacity(num_steps_padded);

        let span = span!(Level::INFO, "evals_z_r_y_var");
        {
        let _enter = span.enter();
        evals_z_r_y_var = (0..constraint_builder.uniform_repeat())
            .into_par_iter()
            .map(|t| {
                flattened_polys
                    .par_iter()
                    .enumerate()
                    .map(|(i, poly)| {
                        poly.get_coeff(t) * eq_ry_var[i]
                    })
                    .sum()
            })
            .collect();
        }

        let num_rounds_shift_sumcheck = num_steps_bits; 
        assert_eq!(evals_z_r_y_var.len(), 1 << num_rounds_shift_sumcheck); 
        let mut shift_sumcheck_polys = vec![
            MultilinearPolynomial::LargeScalars(DensePolynomial::new(evals_z_r_y_var)),
            MultilinearPolynomial::LargeScalars(DensePolynomial::new(eq_plus_one_rx_step)),
        ];

        let shift_sumcheck_claim = (0..1 << num_rounds_shift_sumcheck) 
            .into_par_iter()
            .map(|i| {
                let params: Vec<F> = shift_sumcheck_polys.iter().map(|poly| poly.get_coeff(i)).collect();
                comb_func(&params)
            })
            .reduce(|| F::zero(), |acc, x| acc + x);

        let (shift_sumcheck_proof, shift_sumcheck_r, shift_sumcheck_claims) =
            SumcheckInstanceProof::prove_arbitrary(
                &shift_sumcheck_claim, 
                num_rounds_shift_sumcheck, 
                &mut shift_sumcheck_polys, 
                comb_func, 
                2, 
                transcript);
        drop_in_background_thread(shift_sumcheck_polys);

        let (claimed_witness_evals, chis) =
            MultilinearPolynomial::batch_evaluate(&flattened_polys, rx_step);

        opening_accumulator.append(
            &flattened_polys,
            DensePolynomial::new(chis),
            rx_step.to_vec(),
            &claimed_witness_evals,
            transcript,
        );

        let (claimed_witness_evals_shift_sumcheck, chis2) =
            MultilinearPolynomial::batch_evaluate(&flattened_polys, &shift_sumcheck_r);

        opening_accumulator.append(
            &flattened_polys,
            DensePolynomial::new(chis2),
            shift_sumcheck_r.to_vec(),
            &claimed_witness_evals_shift_sumcheck,
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
            claimed_witness_evals_shift_sumcheck,
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
        let num_rounds_y = key.num_cols_total().log_2();

        // outer sum-check
        let tau = (0..num_rounds_x)
            .map(|_i| transcript.challenge_scalar())
            .collect::<Vec<F>>();

        let (claim_outer_final, outer_sumcheck_r) = self
            .outer_sumcheck_proof
            .verify(F::zero(), num_rounds_x, 3, transcript)
            .map_err(|_| SpartanError::InvalidOuterSumcheckProof)?;

        // Outer sumcheck is bound from the top, reverse the fiat shamir randomness
        let outer_sumcheck_r: Vec<F> = outer_sumcheck_r.into_iter().rev().collect();

        // verify claim_outer_final
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

        // inner sum-check
        let inner_sumcheck_RLC: F = transcript.challenge_scalar();
        let claim_inner_joint = self.outer_sumcheck_claims.0
            + inner_sumcheck_RLC * self.outer_sumcheck_claims.1
            + inner_sumcheck_RLC * inner_sumcheck_RLC * self.outer_sumcheck_claims.2;

        let num_rounds = (2 * key.num_vars_uniform_padded()).log_2() + 1; // +1 for cross-step
        let (claim_inner_final, inner_sumcheck_r) = self
            .inner_sumcheck_proof
            .verify(claim_inner_joint, num_rounds, 2, transcript) 
            .map_err(|_| SpartanError::InvalidInnerSumcheckProof)?;

        // let inner_sumcheck_r: Vec<F> = inner_sumcheck_r.into_iter().rev().collect();

        let n_constraint_bits_uniform = key.uniform_r1cs.num_rows.next_power_of_two().log_2();
        let num_step_bits = key.num_steps.log_2();
        let outer_sumcheck_r_step = &outer_sumcheck_r[..num_step_bits];
        
        let r_choice = inner_sumcheck_r[0]; 
        let r_y_var = inner_sumcheck_r[1..].to_vec();
        let y_prime = [r_y_var.clone(), outer_sumcheck_r_step.to_owned()].concat();
        let eval_z = key.evaluate_z_mle(&self.claimed_witness_evals, &y_prime, true);

        let r = [outer_sumcheck_r.clone(), y_prime].concat();
        let (eval_a, eval_b, eval_c) = key.evaluate_r1cs_matrix_mles(&r, &r_choice);

        let left_expected = eval_a
            + inner_sumcheck_RLC * eval_b
            + inner_sumcheck_RLC * inner_sumcheck_RLC * eval_c;
        let right_expected = 
            (F::one() - r_choice) * eval_z + 
            r_choice * self.shift_sumcheck_claim; 

        let claim_inner_final_expected = left_expected * right_expected;

        assert_eq!(claim_inner_final, claim_inner_final_expected);
        if claim_inner_final != claim_inner_final_expected {
            return Err(SpartanError::InvalidInnerSumcheckClaim);
        }

        let num_steps_bits = outer_sumcheck_r_step.len();
        let num_rounds_shift_sumcheck = num_steps_bits;
        let (claim_shift_final, shift_sumcheck_r) = self
            .shift_sumcheck_proof
            .verify(self.shift_sumcheck_claim, num_rounds_shift_sumcheck, 2, transcript) 
            .map_err(|_| SpartanError::InvalidInnerSumcheckProof)?;

        let y_prime_shift_sumcheck = [r_y_var, shift_sumcheck_r.to_owned()].concat(); 
        let eval_z_shift_sumcheck = key.evaluate_z_mle(&self.claimed_witness_evals_shift_sumcheck, &y_prime_shift_sumcheck, false);
        let eq_plus_one_shift_sumcheck = EqPlusOnePolynomial::new(outer_sumcheck_r_step.to_vec()).evaluate(&shift_sumcheck_r);
        let claim_shift_sumcheck_expected = eval_z_shift_sumcheck * eq_plus_one_shift_sumcheck; 
        assert_eq!(claim_shift_final, claim_shift_sumcheck_expected);
        if claim_shift_final != claim_shift_sumcheck_expected {
            return Err(SpartanError::InvalidInnerSumcheckClaim);
        }

        let flattened_commitments: Vec<_> = I::flatten::<C>()
            .iter()
            .map(|var| var.get_ref(commitments))
            .collect();

        let r_y_point = outer_sumcheck_r_step; 
        opening_accumulator.append(
            &flattened_commitments,
            r_y_point.to_vec(),
            &self.claimed_witness_evals.iter().collect::<Vec<_>>(),
            transcript,
        );

        opening_accumulator.append(
            &flattened_commitments,
            shift_sumcheck_r.to_vec(),
            &self.claimed_witness_evals_shift_sumcheck.iter().collect::<Vec<_>>(),
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
