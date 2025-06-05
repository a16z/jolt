use std::marker::PhantomData;
use tracer::instruction::RV32IMCycle;
use tracing::{span, Level};

use crate::field::JoltField;
use crate::jolt::vm::JoltProverPreprocessing;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::multilinear_polynomial::PolynomialEvaluation;
use crate::poly::opening_proof::ProverOpeningAccumulator;
use crate::poly::opening_proof::VerifierOpeningAccumulator;
use crate::r1cs::inputs::ALL_R1CS_INPUTS;
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
    utils::small_value::NUM_SVO_ROUNDS,
};

use super::builder::CombinedUniformBuilder;

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
pub struct UniformSpartanProof<F: JoltField, ProofTranscript: Transcript> {
    pub(crate) outer_sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub(crate) outer_sumcheck_claims: (F, F, F),
    pub(crate) inner_sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub(crate) shift_sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub(crate) shift_sumcheck_claim: F,
    pub(crate) claimed_witness_evals: Vec<F>,
    pub(crate) shift_sumcheck_witness_evals: Vec<F>,
    _marker: PhantomData<ProofTranscript>,
}

impl<F, ProofTranscript> UniformSpartanProof<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    #[tracing::instrument(skip_all, name = "Spartan::setup")]
    pub fn setup(
        constraint_builder: &CombinedUniformBuilder<F>,
        padded_num_steps: usize,
    ) -> UniformSpartanKey<F> {
        assert_eq!(
            padded_num_steps,
            constraint_builder.uniform_repeat().next_power_of_two()
        );
        UniformSpartanKey::from_builder(constraint_builder)
    }

    #[tracing::instrument(skip_all, name = "Spartan::prove")]
    pub fn prove<PCS>(
        preprocessing: &JoltProverPreprocessing<F, PCS, ProofTranscript>,
        constraint_builder: &CombinedUniformBuilder<F>,
        key: &UniformSpartanKey<F>,
        trace: &[RV32IMCycle],
        _opening_accumulator: &mut ProverOpeningAccumulator<F, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<Self, SpartanError>
    where
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
    {
        let input_polys: Vec<MultilinearPolynomial<F>> = ALL_R1CS_INPUTS
            .par_iter()
            .map(|var| var.generate_witness(trace, preprocessing))
            .collect();

        let num_rounds_x = key.num_rows_bits();

        /* Sumcheck 1: Outer sumcheck */

        let tau: Vec<F> = transcript.challenge_vector(num_rounds_x);
        // For the first sumcheck, we only use uniform constraints (no cross-step constraints)
        // This makes A block-diagonal with blocks being A_small
        let uniform_constraints_only_padded = constraint_builder.uniform_builder.constraints.len().next_power_of_two();
        let (outer_sumcheck_proof, outer_sumcheck_r, outer_sumcheck_claims) =
            SumcheckInstanceProof::prove_spartan_small_value::<NUM_SVO_ROUNDS>(
                num_rounds_x,
                uniform_constraints_only_padded,
                &constraint_builder.uniform_builder.constraints,
                &input_polys,
                &tau,
                transcript,
            );
        let outer_sumcheck_r: Vec<F> = outer_sumcheck_r.into_iter().rev().collect();

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
            - z_shift(y_var || rx_step) = \sum_t z(y_var || t) * eq_plus_one(rx_step, t)
        */

        let num_cycles = key.num_steps;
        let num_cycles_bits = num_cycles.ilog2() as usize;
        let num_vars_uniform = key.num_vars_uniform_padded().next_power_of_two();

        let inner_sumcheck_RLC: F = transcript.challenge_scalar();
        let claim_inner_joint =
            claim_Az + inner_sumcheck_RLC * claim_Bz + inner_sumcheck_RLC.square() * claim_Cz;

        let (r_cycle, r_var) = outer_sumcheck_r.split_at(num_cycles_bits);

        let (eq_r_cycle, eq_plus_one_r_cycle) = EqPlusOnePolynomial::evals(r_cycle, None);

        /* Compute the two polynomials provided as input to the second sumcheck:
           - poly_ABC: A(r_x, y_var || rx_step), A_shift(..) at all variables y_var
           - poly_z: z(y_var || rx_step), z_shift(..)
        */

        let poly_abc_small =
            DensePolynomial::new(key.evaluate_small_matrix_rlc(r_var, inner_sumcheck_RLC));

        let span = span!(Level::INFO, "binding_z_second_sumcheck");
        let _guard = span.enter();

        let mut bind_z = vec![F::zero(); num_vars_uniform];

        input_polys
            .par_iter()
            .take(num_vars_uniform)
            .zip(bind_z.par_iter_mut())
            .for_each(|(poly, eval)| {
                *eval = poly.dot_product(&eq_r_cycle);
            });

        // Set the constant value at the appropriate position
        if key.uniform_r1cs.num_vars < num_vars_uniform {
            bind_z[key.uniform_r1cs.num_vars] = F::one();
        }

        drop(_guard);
        drop(span);

        let poly_z = DensePolynomial::new(bind_z);
        assert_eq!(poly_z.len(), poly_abc_small.len());

        let num_rounds_inner_sumcheck = poly_abc_small.len().log_2();

        let mut polys = vec![
            MultilinearPolynomial::LargeScalars(poly_abc_small),
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

        /*  Sumcheck 3: Shift sumcheck with RLC
            sumcheck claim is: VirtualInstructionAddress(r_cycle) + r * PCNEXT(r_cycle) = 
                              \sum_t (VirtualInstructionAddress(t) + 1 + r * PC(t)) * eq_plus_one(r_cycle, t)
        */
        let span = span!(Level::INFO, "shift_sumcheck_rlc");
        let _guard = span.enter();
        
        // Get RLC coefficient from transcript
        let shift_rlc_coeff: F = transcript.challenge_scalar();
        
        // Extract the polynomials
        let virtual_seq_poly = &input_polys[0]; // VirtualInstructionAddress is at index 0
        let pc_poly = &input_polys[1];          // RealInstructionAddress is at index 1
        
        // Create polynomial for the sumcheck: (VirtualInstructionAddress(t) + 1 + r * PC(t))
        // We need to add 1 to each element of virtual_seq
        let virtual_seq_plus_one: Vec<F> = (0..virtual_seq_poly.len())
            .map(|i| virtual_seq_poly.get_coeff(i) + F::one())
            .collect();
        let virtual_seq_plus_one_poly = MultilinearPolynomial::from(virtual_seq_plus_one);
        
        // Create RLC polynomial: (virtual_seq + 1) + shift_rlc_coeff * pc
        let rlc_poly = MultilinearPolynomial::linear_combination(
            &[&virtual_seq_plus_one_poly, pc_poly],
            &[F::one(), shift_rlc_coeff],
        );
        
        let num_rounds_shift_sumcheck = num_cycles_bits;
        
        // For the shift sumcheck, we use the RLC polynomial and eq_plus_one
        let mut shift_sumcheck_polys = vec![
            rlc_poly,
            MultilinearPolynomial::from(eq_plus_one_r_cycle),
        ];
        
        drop(_guard);
        drop(span);

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

        let flattened_polys_ref: Vec<_> = input_polys.iter().collect();
        // Inner sumcheck evaluations: evaluate z on rx_step
        let (claimed_witness_evals, _chis) =
            MultilinearPolynomial::batch_evaluate(&flattened_polys_ref, r_cycle);

        // opening_accumulator.append(
        //     &flattened_polys_ref,
        //     DensePolynomial::new(chis),
        //     rx_step.to_vec(),
        //     &claimed_witness_evals,
        //     transcript,
        // );

        // Shift sumcheck evaluations: evaluate z on ry_var
        let (shift_sumcheck_witness_evals, _chis2) =
            MultilinearPolynomial::batch_evaluate(&flattened_polys_ref, &shift_sumcheck_r);

        // opening_accumulator.append(
        //     &flattened_polys_ref,
        //     DensePolynomial::new(chis2),
        //     shift_sumcheck_r.to_vec(),
        //     &shift_sumcheck_witness_evals,
        //     transcript,
        // );

        // Outer sumcheck claims: [A(r_x), B(r_x), C(r_x)]
        let outer_sumcheck_claims = (
            outer_sumcheck_claims[0],
            outer_sumcheck_claims[1],
            outer_sumcheck_claims[2],
        );
        Ok(UniformSpartanProof {
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
        key: &UniformSpartanKey<F>,
        // commitments: &JoltCommitments<PCS, ProofTranscript>,
        _opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), SpartanError>
    where
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
        ProofTranscript: Transcript,
    {
        let num_rounds_x = key.num_rows_total().log_2();

        /* Sumcheck 1: Outer sumcheck
         */
        let tau: Vec<F> = transcript.challenge_vector(num_rounds_x);

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
            + inner_sumcheck_RLC.square() * self.outer_sumcheck_claims.2;

        let num_rounds_inner_sumcheck = key.num_vars_uniform_padded().log_2();
        let (claim_inner_final, inner_sumcheck_r) = self
            .inner_sumcheck_proof
            .verify(claim_inner_joint, num_rounds_inner_sumcheck, 2, transcript)
            .map_err(|_| SpartanError::InvalidInnerSumcheckProof)?;

        let num_steps_bits = key.num_steps.log_2();

        let (rx_step, rx_constr) = outer_sumcheck_r.split_at(num_steps_bits);

        let ry_var = inner_sumcheck_r.to_vec();
        let eval_z =
            key.evaluate_z_mle_with_segment_evals(&self.claimed_witness_evals, &ry_var, true);

        // For the verifier, we need to evaluate A(rx_constr, ry_var), B(rx_constr, ry_var), C(rx_constr, ry_var)
        // Since we removed cross-step constraints, we only evaluate the uniform part
        let eval_a = key.evaluate_uniform_a_at_point(rx_constr, &ry_var);
        let eval_b = key.evaluate_uniform_b_at_point(rx_constr, &ry_var);
        let eval_c = key.evaluate_uniform_c_at_point(rx_constr, &ry_var);

        let left_expected =
            eval_a + inner_sumcheck_RLC * eval_b + inner_sumcheck_RLC * inner_sumcheck_RLC * eval_c;

        let claim_inner_final_expected = left_expected * eval_z;
        if claim_inner_final != claim_inner_final_expected {
            return Err(SpartanError::InvalidInnerSumcheckClaim);
        }

        /* Sumcheck 3: Shift sumcheck with RLC
            - claim = VirtualInstructionAddress(r_cycle) + r * PCNEXT(r_cycle) = 
                     \sum_t (VirtualInstructionAddress(t) + 1 + r * PC(t)) * eq_plus_one(r_cycle, t)
            - verifying it involves checking that claim = (virtual_seq(r_t) + 1 + r * pc(r_t)) * eq_plus_one(r_cycle, r_t)
            where r_t = shift_sumcheck_r
        */

        // Get the same RLC coefficient from transcript
        let shift_rlc_coeff: F = transcript.challenge_scalar();

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

        // Extract evaluations from shift_sumcheck_witness_evals
        let eval_virtual_seq_at_shift_r = self.shift_sumcheck_witness_evals[0]; // VirtualInstructionAddress
        let eval_pc_at_shift_r = self.shift_sumcheck_witness_evals[1];         // RealInstructionAddress
        
        // Compute RLC evaluation: (virtual_seq(r_t) + 1) + r * pc(r_t)
        let eval_rlc_at_shift_r = (eval_virtual_seq_at_shift_r + F::one()) + shift_rlc_coeff * eval_pc_at_shift_r;
        
        let eq_plus_one_shift_sumcheck =
            EqPlusOnePolynomial::new(rx_step.to_vec()).evaluate(&shift_sumcheck_r);
        let claim_shift_sumcheck_expected = eval_rlc_at_shift_r * eq_plus_one_shift_sumcheck;

        if claim_shift_sumcheck != claim_shift_sumcheck_expected {
            return Err(SpartanError::InvalidInnerSumcheckClaim);
        }

        // TODO(moodlezoup): Openings

        // let flattened_commitments: Vec<_> = I::flatten()
        //     .iter()
        //     .map(|var| var.get_ref(commitments))
        //     .collect();

        // opening_accumulator.append(
        //     &flattened_commitments,
        //     rx_step.to_vec(),
        //     &self.claimed_witness_evals.iter().collect::<Vec<_>>(),
        //     transcript,
        // );

        // opening_accumulator.append(
        //     &flattened_commitments,
        //     shift_sumcheck_r.to_vec(),
        //     &self.shift_sumcheck_witness_evals.iter().collect::<Vec<_>>(),
        //     transcript,
        // );

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
