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

        /* Sumcheck 1: Outer sumcheck
           Proves: \sum_x eq(tau, x) * (Az(x) * Bz(x) - Cz(x)) = 0
           Only uses uniform constraints, making A, B, C block-diagonal with blocks A_small, B_small, C_small
        */

        let tau: Vec<F> = transcript.challenge_vector(num_rounds_x);
        let uniform_constraints_only_padded = constraint_builder
            .uniform_builder
            .constraints
            .len()
            .next_power_of_two();
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
        let (claim_Az, claim_Bz, claim_Cz): (F, F, F) = (
            outer_sumcheck_claims[0],
            outer_sumcheck_claims[1],
            outer_sumcheck_claims[2],
        );

        /* Sumcheck 2: Inner sumcheck
           Proves: claim_Az + r * claim_Bz + r^2 * claim_Cz =
                   \sum_y (A_small(rx, y) + r * B_small(rx, y) + r^2 * C_small(rx, y)) * z(y)
           Uses only uniform constraints (A_small, B_small, C_small)
        */

        let num_cycles = key.num_steps;
        let num_cycles_bits = num_cycles.ilog2() as usize;
        let num_vars_uniform = key.num_vars_uniform_padded().next_power_of_two();

        let inner_sumcheck_RLC: F = transcript.challenge_scalar();
        let claim_inner_joint =
            claim_Az + inner_sumcheck_RLC * claim_Bz + inner_sumcheck_RLC.square() * claim_Cz;

        let (r_cycle, r_var) = outer_sumcheck_r.split_at(num_cycles_bits);

        let (eq_r_cycle, eq_plus_one_r_cycle) = EqPlusOnePolynomial::evals(r_cycle, None);

        // Evaluate A_small, B_small, C_small combined with RLC at point r_var
        let poly_abc_small =
            DensePolynomial::new(key.evaluate_small_matrix_rlc(r_var, inner_sumcheck_RLC));

        let span = span!(Level::INFO, "binding_z_second_sumcheck");
        let _guard = span.enter();

        // Bind witness polynomials z at point r_cycle
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

        let (inner_sumcheck_proof, inner_sumcheck_r, claims_inner) =
            SumcheckInstanceProof::prove_arbitrary(
                &claim_inner_joint,
                num_rounds_inner_sumcheck,
                &mut polys,
                comb_func,
                2,
                transcript,
            );

        drop_in_background_thread(polys);

        // Evaluate all witness polynomials P_i at r_cycle for the verifier
        // The verifier will compute z(r_inner, r_cycle) = Σ_i eq(r_inner, i) * P_i(r_cycle)
        let flattened_polys_ref: Vec<_> = input_polys.iter().collect();
        let (claimed_witness_evals, chis) =
            MultilinearPolynomial::batch_evaluate(&flattened_polys_ref, r_cycle);

        /*  Sumcheck 3: Shift sumcheck for NextPC verification
            Proves: NextPC(r_cycle) = \sum_t PC(t) * eq_plus_one(r_cycle, t)
            This allows the verifier to verify claimed_witness_evals[18] (NextPC(r_cycle))
            using only PC(shift_r) without needing a separate opening proof for NextPC
        */
        let span = span!(Level::INFO, "shift_sumcheck_pc");
        let _guard = span.enter();

        // Extract the polynomials
        let pc_poly = &input_polys[1]; // RealInstructionAddress/PC is at index 1
        let next_pc_poly = &input_polys[18]; // NextPC is at index 18

        let num_rounds_shift_sumcheck = num_cycles_bits;

        // The claim is the sum itself (not NextPC(r_cycle))
        // This is because we're proving the sumcheck relation directly

        // For the shift sumcheck, we use PC polynomial and eq_plus_one
        let eq_plus_one_len = eq_plus_one_r_cycle.len(); // Save length before move
        let mut shift_sumcheck_polys = vec![
            pc_poly.clone(),
            MultilinearPolynomial::from(eq_plus_one_r_cycle),
        ];

        drop(_guard);
        drop(span);

        // Debug: Let's trace through the sum computation
        println!("DEBUG: Computing shift sumcheck claim...");
        println!("DEBUG: r_cycle = {:?}", r_cycle);
        println!("DEBUG: eq_plus_one_r_cycle len = {}", eq_plus_one_len);
        
        let shift_sumcheck_claim = (0..1 << num_rounds_shift_sumcheck)
            .into_par_iter()
            .map(|i| {
                let params: Vec<F> = shift_sumcheck_polys
                    .iter()
                    .map(|poly| poly.get_coeff(i))
                    .collect();
                let result = comb_func(&params);
                // Debug first few non-zero contributions
                if !result.is_zero() && i < 10 {
                    println!("DEBUG: i={}, PC[{}]={:?}, eq_plus_one={:?}, product={:?}", 
                             i, i, params[0], params[1], result);
                }
                result
            })
            .reduce(|| F::zero(), |acc, x| acc + x);
        
        println!("DEBUG: shift_sumcheck_claim = {:?}", shift_sumcheck_claim);
        println!("DEBUG: claimed_witness_evals[18] (NextPC at r_cycle) = {:?}", claimed_witness_evals[18]);

        // Debug: Let's verify the constraint holds for concrete values
        let num_cycles = 1 << num_cycles_bits;
        println!("DEBUG: Total cycles = {}, PC poly len = {}, NextPC poly len = {}", 
                 num_cycles, pc_poly.len(), input_polys[18].len());
        
        // PC[0] is 
        println!("DEBUG: PC[0] = {:?}", pc_poly.get_coeff(0));
        // Check first few cycles
        for i in 0..num_cycles.min(10) {
            let next_pc_i = input_polys[18].get_coeff(i);
            let pc_i_plus_1 = if i + 1 < pc_poly.len() {
                pc_poly.get_coeff(i + 1)
            } else {
                F::zero()
            };
            println!("DEBUG: Cycle {}: NextPC = {:?}, PC[{}] = {:?}", 
                     i, next_pc_i, i+1, pc_i_plus_1);
            if next_pc_i != pc_i_plus_1 {
                println!("  MISMATCH!");
            }
        }

        
        // Check last cycle
        let last = num_cycles - 1;
        println!("DEBUG: Last cycle {}: NextPC = {:?}, no PC[{}]", 
                 last, input_polys[18].get_coeff(last), last+1);
        println!("DEBUG: Pre last cycle PC: {}", pc_poly.get_coeff(last));
        
        // Check last 10 cycles
        for i in last-10..last {
            let next_pc_i = input_polys[18].get_coeff(i);
            let pc_i_plus_1 = if i + 1 < pc_poly.len() {
                pc_poly.get_coeff(i + 1)
            } else {
                F::zero()
            };
            println!("DEBUG: Cycle {}: NextPC = {:?}, PC[{}] = {:?}", 
                     i, next_pc_i, i+1, pc_i_plus_1);
            if next_pc_i != pc_i_plus_1 {
                println!("  MISMATCH!");
            }
        }
        
        // The third sumcheck proves that NextPC at r_cycle can be computed from PC values
        // We verify that our computed sum equals NextPC(r_cycle) from claimed_witness_evals
        assert_eq!(
            shift_sumcheck_claim, claimed_witness_evals[18],
            "Shift sumcheck claim (sum of PC(t) * eq_plus_one(r_cycle, t)) should equal NextPC(r_cycle)"
        );

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

        // opening_accumulator.append(
        //     &flattened_polys_ref,
        //     DensePolynomial::new(chis),
        //     rx_step.to_vec(),
        //     &claimed_witness_evals,
        //     transcript,
        // );

        // For shift sumcheck, we need PC evaluation at shift_r
        let shift_polys_to_evaluate = vec![&input_polys[1]]; // PC only
        let (shift_sumcheck_witness_evals_partial, chis2) =
            MultilinearPolynomial::batch_evaluate(&shift_polys_to_evaluate, &shift_sumcheck_r);

        // Pad with zeros for compatibility with existing proof structure
        let mut shift_sumcheck_witness_evals = vec![F::zero(); input_polys.len()];
        shift_sumcheck_witness_evals[1] = shift_sumcheck_witness_evals_partial[0]; // PC(shift_r)

        // opening_accumulator.append(
        //     &flattened_polys_ref,
        //     DensePolynomial::new(chis2),
        //     shift_sumcheck_r.to_vec(),
        //     &shift_sumcheck_witness_evals,
        //     transcript,
        // );

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
          Verifies: \sum_x eq(tau, x) * (Az(x) * Bz(x) - Cz(x)) = 0
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
           Verifies: claim_Az + r * claim_Bz + r^2 * claim_Cz =
                    (A_small(rx, ry) + r * B_small(rx, ry) + r^2 * C_small(rx, ry)) * z(ry)
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

        // Evaluate uniform matrices A_small, B_small, C_small at point (rx_constr, ry_var)
        let eval_a = key.evaluate_uniform_a_at_point(rx_constr, &ry_var);
        let eval_b = key.evaluate_uniform_b_at_point(rx_constr, &ry_var);
        let eval_c = key.evaluate_uniform_c_at_point(rx_constr, &ry_var);

        let left_expected =
            eval_a + inner_sumcheck_RLC * eval_b + inner_sumcheck_RLC * inner_sumcheck_RLC * eval_c;

        let claim_inner_final_expected = left_expected * eval_z;
        if claim_inner_final != claim_inner_final_expected {
            return Err(SpartanError::InvalidInnerSumcheckClaim);
        }

        /* Sumcheck 3: Shift sumcheck for PC
            - claim = \sum_t PC(t) * eq_plus_one(r_cycle, t)
            - verifying it involves checking that claim = pc(r_t) * eq_plus_one(r_cycle, r_t)
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

        // Extract PC evaluation from shift_sumcheck_witness_evals
        let eval_pc_at_shift_r = self.shift_sumcheck_witness_evals[1]; // PC

        let eq_plus_one_shift_sumcheck =
            EqPlusOnePolynomial::new(rx_step.to_vec()).evaluate(&shift_sumcheck_r);
        let claim_shift_sumcheck_expected = eval_pc_at_shift_r * eq_plus_one_shift_sumcheck;

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
