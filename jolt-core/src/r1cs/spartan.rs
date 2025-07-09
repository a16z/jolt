use crate::field::JoltField;
use crate::jolt::vm::JoltCommitments;
use crate::jolt::vm::JoltProverPreprocessing;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{
    process_large_scalar_polys, process_small_scalar_polys, PolynomialEvaluation,
};
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::poly::opening_proof::ProverOpeningAccumulator;
use crate::poly::opening_proof::VerifierOpeningAccumulator;
use crate::r1cs::inputs::COMMITTED_R1CS_INPUTS;
use crate::r1cs::inputs::{JoltR1CSInputs, R1CSInputsOracle, ALL_R1CS_INPUTS};
use crate::r1cs::key::UniformSpartanKey;
use crate::utils::math::Math;
use std::marker::PhantomData;
use tracer::instruction::RV32IMCycle;
use tracing::{span, Level};

use crate::utils::transcript::Transcript;
use ark_serialize::CanonicalDeserialize;
use ark_serialize::CanonicalSerialize;

use thiserror::Error;

use crate::{
    poly::{dense_mlpoly::DensePolynomial, eq_poly::EqPlusOnePolynomial},
    subprotocols::sumcheck::{BatchableSumcheckInstance, SumcheckInstanceProof},
    utils::small_value::NUM_SVO_ROUNDS,
};

use super::builder::CombinedUniformBuilder;

use crate::poly::split_eq_poly::SplitEqPolynomial;
use crate::subprotocols::sumcheck::eq_plus_one_shards;
use crate::utils::thread::drop_in_background_thread;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct OuterSumcheckClaims<F: JoltField> {
    az: F,
    bz: F,
    cz: F,
}

#[derive(Clone, Debug)]
pub struct InnerSumcheckParams<F: JoltField> {
    r_cycle: Vec<F>,
    rx_var: Vec<F>,
}

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

    /// returned when the recursive sumcheck proof fails
    #[error("InvalidShiftSumcheckProof")]
    InvalidShiftSumcheckProof,

    /// returned when the final sumcheck opening proof fails
    #[error("InvalidShiftSumcheckClaim")]
    InvalidShiftSumcheckClaim,

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
#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct UniformSpartanProof<F: JoltField, ProofTranscript: Transcript> {
    pub(crate) outer_sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub(crate) outer_sumcheck_claims: (F, F, F),
    pub(crate) inner_sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub(crate) shift_sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub(crate) claimed_witness_evals: Vec<F>,
    pub(crate) shift_sumcheck_witness_eval: Vec<F>,
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
        opening_accumulator: &mut ProverOpeningAccumulator<F, PCS, ProofTranscript>,
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

           The matrices A, B, C have a block-diagonal structure with repeated blocks
           A_small, B_small, C_small corresponding to the uniform constraints.
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

        // claims from the end of sum-check
        // claim_Az is the (scalar) value v_A = \sum_y A(r_x, y) * z(r_x) where r_x is the sumcheck randomness
        let (claim_Az, claim_Bz, claim_Cz): (F, F, F) = (
            outer_sumcheck_claims[0],
            outer_sumcheck_claims[1],
            outer_sumcheck_claims[2],
        );

        /* Sumcheck 2: Inner sumcheck
           Proves: claim_Az + r * claim_Bz + r^2 * claim_Cz =
                   \sum_y (A_small(rx, y) + r * B_small(rx, y) + r^2 * C_small(rx, y)) * z(y)

           Evaluates the uniform constraint matrices A_small, B_small, C_small at the point
           determined by the outer sumcheck.
        */

        let num_cycles = key.num_steps;
        let num_cycles_bits = num_cycles.ilog2() as usize;

        let inner_sumcheck_RLC: F = transcript.challenge_scalar();

        let (r_cycle, rx_var) = outer_sumcheck_r.split_at(num_cycles_bits);

        let claims = OuterSumcheckClaims {
            az: claim_Az,
            bz: claim_Bz,
            cz: claim_Cz,
        };

        let params = InnerSumcheckParams {
            r_cycle: r_cycle.to_vec(),
            rx_var: rx_var.to_vec(),
        };
        let (inner_sumcheck_proof, _inner_sumcheck_r) = Self::prove_inner_sumcheck(
            key,
            &input_polys,
            &claims,
            &params,
            inner_sumcheck_RLC,
            transcript,
        );

        // Evaluate all witness polynomials P_i at r_cycle for the verifier
        // Verifier computes: z(r_inner, r_cycle) = Σ_i eq(r_inner, i) * P_i(r_cycle)
        let flattened_polys_ref: Vec<_> = input_polys.iter().collect();
        let (claimed_witness_evals, chis) =
            MultilinearPolynomial::batch_evaluate(&flattened_polys_ref, r_cycle);

        let (_, eq_plus_one_r_cycle) = EqPlusOnePolynomial::evals(r_cycle, None);

        /*  Sumcheck 3: Batched sumcheck for NextUnexpandedPC and NextPC verification
            Proves: NextUnexpandedPC(r_cycle) + r * NextPC(r_cycle) =
                    \sum_t (UnexpandedPC(t) + r * PC(t)) * eq_plus_one(r_cycle, t)

            This batched sumcheck simultaneously proves:
            1. NextUnexpandedPC(r_cycle) = \sum_t UnexpandedPC(t) * eq_plus_one(r_cycle, t)
            2. NextPC(r_cycle) = \sum_t PC(t) * eq_plus_one(r_cycle, t)
        */

        let (shift_sumcheck_proof, shift_sumcheck_witness_eval) = Self::prove_pc_sumcheck(
            &input_polys,
            &claimed_witness_evals,
            eq_plus_one_r_cycle,
            transcript,
        );

        // Only non-virtual (i.e. committed) polynomials' openings are
        // proven using the PCS opening proof. Virtual polynomial openings
        // are proven in some subsequent sumcheck.
        let committed_polys: Vec<_> = COMMITTED_R1CS_INPUTS
            .iter()
            .map(|input| &input_polys[input.to_index()])
            .collect();
        let committed_poly_claims: Vec<_> = COMMITTED_R1CS_INPUTS
            .iter()
            .map(|input| claimed_witness_evals[input.to_index()])
            .collect();

        opening_accumulator.append_dense(
            &committed_polys,
            chis,
            r_cycle.to_vec(),
            &committed_poly_claims,
            transcript,
        );

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
            claimed_witness_evals,
            shift_sumcheck_witness_eval,
            _marker: PhantomData,
        })
    }

    #[tracing::instrument(skip_all, name = "Spartan::prove_streaming")]
    pub fn prove_streaming<PCS>(
        preprocessing: &JoltProverPreprocessing<F, PCS, ProofTranscript>,
        constraint_builder: &CombinedUniformBuilder<F>,
        key: &UniformSpartanKey<F>,
        trace: &[RV32IMCycle],
        shard_length: usize,
        opening_accumulator: &mut ProverOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<Self, SpartanError>
    where
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
    {
        // We require that shard length be at least 2^{ceil{trace length / 2}}.
        assert!(shard_length.is_power_of_two());

        let num_rounds_x = key.num_rows_bits();

        /* Sumcheck 1: Outer sumcheck
           Proves: \sum_x eq(tau, x) * (Az(x) * Bz(x) - Cz(x)) = 0

           The matrices A, B, C have a block-diagonal structure with repeated blocks
           A_small, B_small, C_small corresponding to the uniform constraints.
        */

        let tau: Vec<F> = transcript.challenge_vector(num_rounds_x);

        let (outer_sumcheck_proof, outer_sumcheck_r, outer_sumcheck_claims) =
            SumcheckInstanceProof::prove_spartan_small_value_streaming::<NUM_SVO_ROUNDS, PCS>(
                num_rounds_x,
                constraint_builder.padded_rows_per_step(),
                &constraint_builder.uniform_builder.constraints,
                trace,
                preprocessing,
                shard_length,
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
           Proves: claim_Az + r * claim_Bz + r^2 * claim_Cz =
                   \sum_y (A_small(rx, y) + r * B_small(rx, y) + r^2 * C_small(rx, y)) * z(y)

           Evaluates the uniform constraint matrices A_small, B_small, C_small at the point
           determined by the outer sumcheck.
        */

        let num_cycles = key.num_steps;
        let num_cycles_bits = num_cycles.ilog2() as usize;
        let num_vars_uniform = key.num_vars_uniform_padded();

        let inner_sumcheck_RLC: F = transcript.challenge_scalar();
        let claim_inner_joint =
            claim_Az + inner_sumcheck_RLC * claim_Bz + inner_sumcheck_RLC.square() * claim_Cz;

        let (r_cycle, rx_var) = outer_sumcheck_r.split_at(num_cycles_bits);

        // Evaluate A_small, B_small, C_small combined with RLC at point rx_var
        let poly_abc_small =
            DensePolynomial::new(key.evaluate_small_matrix_rlc(rx_var, inner_sumcheck_RLC));

        let span = span!(Level::INFO, "binding_z_and_shift_z");
        let _guard = span.enter();

        let mut input_polys_oracle = R1CSInputsOracle::new(shard_length, trace, preprocessing);
        let num_shards = trace.len() / shard_length;

        let mut bind_z_stream = vec![F::zero(); num_vars_uniform];

        let eq_rx_step = SplitEqPolynomial::new(r_cycle);
        let num_x1_bits = eq_rx_step.E1_len.log_2();
        let x1_bitmask = (1 << (num_x1_bits)) - 1;

        let mut shards = input_polys_oracle.next().unwrap();
        let num_polys = shards.len();

        for shard_idx in 0..num_shards {
            let base_poly_idx = shard_idx * shard_length;
            (_, shards) = rayon::join(
                || {
                    shards
                        .par_iter()
                        .zip(bind_z_stream.par_iter_mut().take(num_polys))
                        .for_each(|(poly, bind_z_eval)| {
                            match poly {
                                MultilinearPolynomial::LargeScalars(poly) => {
                                    process_large_scalar_polys(
                                        &poly.Z,
                                        bind_z_eval,
                                        &eq_rx_step,
                                        base_poly_idx,
                                        x1_bitmask,
                                        num_x1_bits,
                                        shard_length,
                                    );
                                }

                                MultilinearPolynomial::U8Scalars(poly) => {
                                    process_small_scalar_polys(
                                        &poly.coeffs,
                                        bind_z_eval,
                                        &eq_rx_step,
                                        base_poly_idx,
                                        x1_bitmask,
                                        num_x1_bits,
                                        shard_length,
                                    );
                                }

                                MultilinearPolynomial::U16Scalars(poly) => {
                                    process_small_scalar_polys(
                                        &poly.coeffs,
                                        bind_z_eval,
                                        &eq_rx_step,
                                        base_poly_idx,
                                        x1_bitmask,
                                        num_x1_bits,
                                        shard_length,
                                    );
                                }

                                MultilinearPolynomial::U32Scalars(poly) => {
                                    process_small_scalar_polys(
                                        &poly.coeffs,
                                        bind_z_eval,
                                        &eq_rx_step,
                                        base_poly_idx,
                                        x1_bitmask,
                                        num_x1_bits,
                                        shard_length,
                                    );
                                }

                                MultilinearPolynomial::U64Scalars(poly) => {
                                    process_small_scalar_polys(
                                        &poly.coeffs,
                                        bind_z_eval,
                                        &eq_rx_step,
                                        base_poly_idx,
                                        x1_bitmask,
                                        num_x1_bits,
                                        shard_length,
                                    );
                                }

                                MultilinearPolynomial::I64Scalars(poly) => {
                                    process_small_scalar_polys(
                                        &poly.coeffs,
                                        bind_z_eval,
                                        &eq_rx_step,
                                        base_poly_idx,
                                        x1_bitmask,
                                        num_x1_bits,
                                        shard_length,
                                    );
                                }
                                _ => unimplemented!("Unexpected MultilinearPolynomial variant"),
                            };
                        });
                },
                || {
                    let mut shard = Vec::with_capacity(num_polys);
                    if shard_idx != num_shards - 1 {
                        shard = input_polys_oracle.next().unwrap();
                    };
                    shard
                },
            );
        }

        // Set the constant value at the appropriate position
        if key.uniform_r1cs.num_vars < num_vars_uniform {
            bind_z_stream[key.uniform_r1cs.num_vars] = F::one();
        }

        drop(_guard);
        drop(span);

        let poly_z = DensePolynomial::new(bind_z_stream);

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
        let (inner_sumcheck_proof, _inner_sumcheck_r, _claims_inner) =
            SumcheckInstanceProof::prove_arbitrary(
                &claim_inner_joint,
                num_rounds_inner_sumcheck,
                &mut polys,
                comb_func,
                2,
                BindingOrder::HighToLow,
                transcript,
            );

        drop_in_background_thread(polys);

        // Evaluate all witness polynomials P_i at r_cycle for the verifier
        // Verifier computes: z(r_inner, r_cycle) = Σ_i eq(r_inner, i) * P_i(r_cycle)
        let claimed_witness_evals = MultilinearPolynomial::stream_batch_evaluate(
            &mut input_polys_oracle,
            r_cycle,
            num_shards,
            shard_length,
        );

        /*  Sumcheck 3: Batched sumcheck for NextUnexpandedPC and NextPC verification
            Proves: NextUnexpandedPC(r_cycle) + r * NextPC(r_cycle) =
                    \sum_t (UnexpandedPC(t) + r * PC(t)) * eq_plus_one(r_cycle, t)

            This batched sumcheck simultaneously proves:
            1. NextUnexpandedPC(r_cycle) = \sum_t UnexpandedPC(t) * eq_plus_one(r_cycle, t)
            2. NextPC(r_cycle) = \sum_t PC(t) * eq_plus_one(r_cycle, t)
        */

        // Get random challenge r for batching
        let r: F = transcript.challenge_scalar();

        let num_rounds_shift_sumcheck = num_cycles_bits;
        let mut shift_sumcheck_polys_oracle =
            ShiftSumCheckOracle::new(trace, shard_length, preprocessing, r_cycle);

        // Define the batched combining function:
        // (unexpanded_pc(t) + r * pc(t)) * eq_plus_one(r_cycle, t)
        let batched_comb_func = move |poly_evals: &[F]| -> F {
            assert_eq!(poly_evals.len(), 3);
            let unexpanded_pc_eval = poly_evals[0];
            let pc_eval = poly_evals[1];
            let eq_eval = poly_evals[2];

            let batched_eval = unexpanded_pc_eval + r * pc_eval;
            batched_eval * eq_eval
        };

        // The batched claim equals NextUnexpandedPC(r_cycle) + r * NextPC(r_cycle)
        let next_unexpanded_pc_index = JoltR1CSInputs::NextUnexpandedPC.to_index();
        let next_pc_index = JoltR1CSInputs::NextPC.to_index();
        let shift_sumcheck_claim = claimed_witness_evals[next_unexpanded_pc_index]
            + r * claimed_witness_evals[next_pc_index];
        let (shift_sumcheck_proof, _shift_sumcheck_r, shift_sumcheck_claims) =
            SumcheckInstanceProof::shift_sumcheck(
                num_rounds_shift_sumcheck,
                shift_sumcheck_claim,
                &mut shift_sumcheck_polys_oracle,
                batched_comb_func,
                shard_length,
                transcript,
                2,
            );

        // Only non-virtual (i.e. committed) polynomials' openings are
        // proven using the PCS opening proof. Virtual polynomial openings
        // are proven in some subsequent sumcheck.
        //TODO:- We need to stream input polys here. Can be done while integrating Streamed version of PCS.
        let input_polys: Vec<MultilinearPolynomial<F>> = ALL_R1CS_INPUTS
            .par_iter()
            .map(|var| var.generate_witness(trace, preprocessing))
            .collect();

        let committed_polys: Vec<_> = COMMITTED_R1CS_INPUTS
            .iter()
            .map(|input| &input_polys[input.to_index()])
            .collect();
        let committed_poly_claims: Vec<_> = COMMITTED_R1CS_INPUTS
            .iter()
            .map(|input| claimed_witness_evals[input.to_index()])
            .collect();

        //TODO:- Use SplitEQ for chis. Can be done while integrating Streamed version of PCS.
        let chis = EqPolynomial::evals(r_cycle);
        opening_accumulator.append_dense(
            &committed_polys,
            chis,
            r_cycle.to_vec(),
            &committed_poly_claims,
            transcript,
        );

        let unexpanded_pc_eval_at_shift_r = shift_sumcheck_claims[0];
        let pc_eval_at_shift_r = shift_sumcheck_claims[1];
        let shift_sumcheck_witness_eval = vec![unexpanded_pc_eval_at_shift_r, pc_eval_at_shift_r];

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
            claimed_witness_evals,
            shift_sumcheck_witness_eval,
            _marker: PhantomData,
        })
    }
    #[tracing::instrument(skip_all)]
    fn prove_inner_sumcheck(
        key: &UniformSpartanKey<F>,
        input_polys: &[MultilinearPolynomial<F>],
        claims: &OuterSumcheckClaims<F>,
        params: &InnerSumcheckParams<F>,
        inner_sumcheck_RLC: F,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>) {
        let mut inner_sumcheck =
            InnerSumcheck::new_prover(key, input_polys, claims, params, inner_sumcheck_RLC);

        let (inner_sumcheck_proof, r) = inner_sumcheck.prove_single(transcript);

        (inner_sumcheck_proof, r)
    }
    fn prove_pc_sumcheck(
        input_polys: &[MultilinearPolynomial<F>],
        claimed_witness_evals: &[F],
        eq_plus_one_r_cycle: Vec<F>,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>) {
        let span = span!(Level::INFO, "shift_sumcheck_pc");
        let _guard = span.enter();

        let r: F = transcript.challenge_scalar();

        let mut pc_sumcheck =
            PCSumcheck::new_prover(input_polys, claimed_witness_evals, eq_plus_one_r_cycle, r);

        let (shift_sumcheck_proof, _r) = pc_sumcheck.prove_single(transcript);

        let cached_claims = pc_sumcheck.cached_claims.expect("Claims not cached");
        let unexpanded_pc_eval_at_shift_r = cached_claims.0;
        let pc_eval_at_shift_r = cached_claims.1;
        let shift_sumcheck_witness_eval = vec![unexpanded_pc_eval_at_shift_r, pc_eval_at_shift_r];

        drop(_guard);
        drop(span);

        (shift_sumcheck_proof, shift_sumcheck_witness_eval)
    }
    #[tracing::instrument(skip_all, name = "Spartan::verify")]
    pub fn verify<PCS>(
        &self,
        key: &UniformSpartanKey<F>,
        commitments: &JoltCommitments<F, PCS, ProofTranscript>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
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

        // Outer sumcheck is bound from the top, reverse the challenge
        let outer_sumcheck_r: Vec<F> = outer_sumcheck_r.into_iter().rev().collect();

        let (claim_Az, claim_Bz, claim_Cz) = self.outer_sumcheck_claims;
        let taus_bound_rx = EqPolynomial::mle(&tau, &outer_sumcheck_r);
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

        let num_cycles_bits = key.num_steps.log_2();
        let (r_cycle, rx_var) = outer_sumcheck_r.split_at(num_cycles_bits);

        #[cfg(not(feature = "streaming"))]
        {
            let inner_sumcheck = InnerSumcheck::<F>::new_verifier(
                claim_inner_joint,
                key,
                rx_var.to_vec(),
                self.claimed_witness_evals.clone(),
                inner_sumcheck_RLC,
            );

            // Verify the inner sumcheck
            let _inner_sumcheck_r = inner_sumcheck
                .verify_single(&self.inner_sumcheck_proof, transcript)
                .map_err(|_| SpartanError::InvalidInnerSumcheckProof)?;
        }

        #[cfg(feature = "streaming")]
        {
            let num_rounds_inner_sumcheck = key.num_vars_uniform_padded().log_2();
            let (claim_inner_final, inner_sumcheck_r) = self
                .inner_sumcheck_proof
                .verify(claim_inner_joint, num_rounds_inner_sumcheck, 2, transcript)
                .map_err(|_| SpartanError::InvalidInnerSumcheckProof)?;

            let num_cycles_bits = key.num_steps.log_2();

            let ry_var = inner_sumcheck_r.to_vec();
            let eval_z =
                key.evaluate_z_mle_with_segment_evals(&self.claimed_witness_evals, &ry_var, true);

            // Evaluate uniform matrices A_small, B_small, C_small at point (rx_constr, ry_var)
            let eval_a = key.evaluate_uniform_a_at_point(rx_var, &ry_var);
            let eval_b = key.evaluate_uniform_b_at_point(rx_var, &ry_var);
            let eval_c = key.evaluate_uniform_c_at_point(rx_var, &ry_var);

            let left_expected = eval_a
                + inner_sumcheck_RLC * eval_b
                + inner_sumcheck_RLC * inner_sumcheck_RLC * eval_c;

            let claim_inner_final_expected = left_expected * eval_z;
            if claim_inner_final != claim_inner_final_expected {
                return Err(SpartanError::InvalidInnerSumcheckClaim);
            }
        }
        /* Sumcheck 3: Batched sumcheck for NextUnexpandedPC and NextPC verification
           Verifies the batched constraint for both NextUnexpandedPC and NextPC
        */
        let r: F = transcript.challenge_scalar();

        let next_unexpanded_pc_index = JoltR1CSInputs::NextUnexpandedPC.to_index();
        let next_pc_index = JoltR1CSInputs::NextPC.to_index();

        // The batched claim equals NextUnexpandedPC(r_cycle) + r * NextPC(r_cycle)
        let shift_sumcheck_claim = self.claimed_witness_evals[next_unexpanded_pc_index]
            + r * self.claimed_witness_evals[next_pc_index];

        let unexpanded_pc_eval_at_shift_r = self.shift_sumcheck_witness_eval[0];
        let pc_eval_at_shift_r = self.shift_sumcheck_witness_eval[1];

        #[cfg(not(feature = "streaming"))]
        {
            let pc_sumcheck = PCSumcheck::<F>::new_verifier(
                shift_sumcheck_claim,
                r_cycle.to_vec(),
                r,
                unexpanded_pc_eval_at_shift_r,
                pc_eval_at_shift_r,
            );
            let _shift_sumcheck_r = pc_sumcheck
                .verify_single(&self.shift_sumcheck_proof, transcript)
                .map_err(|_| SpartanError::InvalidShiftSumcheckProof)?;
        }

        #[cfg(feature = "streaming")]
        {
            let (claim_shift_sumcheck, shift_sumcheck_r) = self
                .shift_sumcheck_proof
                .verify(shift_sumcheck_claim, num_cycles_bits, 2, transcript)
                .map_err(|_| SpartanError::InvalidShiftSumcheckProof)?;
            let shift_sumcheck_r: Vec<F> = shift_sumcheck_r.iter().rev().copied().collect();

            let batched_eval_at_shift_r = unexpanded_pc_eval_at_shift_r + r * pc_eval_at_shift_r;
            let eq_plus_one_shift_sumcheck =
                EqPlusOnePolynomial::new(r_cycle.to_vec()).evaluate(&shift_sumcheck_r);

            let claim_shift_sumcheck_expected =
                batched_eval_at_shift_r * eq_plus_one_shift_sumcheck;

            if claim_shift_sumcheck != claim_shift_sumcheck_expected {
                return Err(SpartanError::InvalidShiftSumcheckClaim);
            }
        }

        // TODO(moodlezoup): Relies on ordering of commitments
        let r1cs_input_commitments = &commitments
            .commitments
            .iter()
            .take(COMMITTED_R1CS_INPUTS.len())
            .collect::<Vec<_>>();

        let claims: Vec<_> = COMMITTED_R1CS_INPUTS
            .iter()
            .map(|input| self.claimed_witness_evals[input.to_index()])
            .collect();
        opening_accumulator.append(
            r1cs_input_commitments,
            r_cycle.to_vec(),
            &claims,
            transcript,
        );

        Ok(())
    }
}

pub struct ShiftSumCheckOracle<
    'a,
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
> {
    pub step: usize,
    pub shard_length: usize,
    pub trace: &'a [RV32IMCycle],
    pub preprocessing: &'a JoltProverPreprocessing<F, PCS, ProofTranscript>,
    eq_rx_step: SplitEqPolynomial<F>,
}

impl<
        'a,
        F: JoltField,
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
        ProofTranscript: Transcript,
    > ShiftSumCheckOracle<'a, F, PCS, ProofTranscript>
{
    pub fn new(
        trace: &'a [RV32IMCycle],
        shard_length: usize,
        preprocessing: &'a JoltProverPreprocessing<F, PCS, ProofTranscript>,
        r_cycle: &[F],
    ) -> Self {
        let eq_rx_step = SplitEqPolynomial::new(r_cycle);

        Self {
            step: 0,
            shard_length,
            trace,
            preprocessing,
            eq_rx_step,
        }
    }

    fn compute_evals(&self, trace_shard: &[RV32IMCycle]) -> Vec<MultilinearPolynomial<F>> {
        let is_last_shard = self.step == self.trace.len() - self.shard_length;
        let pc_coeff: Vec<u64> = if !is_last_shard {
            self.preprocessing
                .shared
                .bytecode
                .map_trace_to_pc_streaming(trace_shard)
                .collect()
        } else {
            self.preprocessing
                .shared
                .bytecode
                .map_trace_to_pc(trace_shard)
                .collect()
        };

        let unexpanded_pc_coeff: Vec<u64> = trace_shard
            .par_iter()
            .map(|cycle| cycle.instruction().normalize().address as u64)
            .collect();

        vec![unexpanded_pc_coeff.into(), pc_coeff.into()]
    }
}

impl<'a, F: JoltField, PCS, ProofTranscript> Iterator
    for ShiftSumCheckOracle<'a, F, PCS, ProofTranscript>
where
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    type Item = Vec<MultilinearPolynomial<F>>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut shard = self.compute_evals(&self.trace[self.step..self.step + self.shard_length]);
        let num_x1_bits = self.eq_rx_step.E1_len.log_2();
        let x1_bitmask = (1 << num_x1_bits) - 1;
        shard.push(eq_plus_one_shards(
            self.step % self.trace.len(),
            self.shard_length,
            &self.eq_rx_step,
            num_x1_bits,
            x1_bitmask,
        ));

        self.step += self.shard_length;
        self.step %= self.trace.len();
        assert_eq!(self.shard_length, shard[0].len(), "Incorrect shard length");

        //Make sure that the shard length is more than equal to the square root of trace length.
        let log2_trace_len = self.trace.len().log_2();
        let shard_length = 1 << (log2_trace_len - (log2_trace_len / 2));
        assert!(
            self.shard_length >= shard_length,
            "expected shard length {} is less than square root of trace length {}",
            self.shard_length,
            shard_length
        );
        Some(shard)
    }
}

struct InnerSumcheckProverState<F: JoltField> {
    poly_abc_small: MultilinearPolynomial<F>,
    poly_z: MultilinearPolynomial<F>,
}

struct InnerSumcheckVerifierState<'a, F: JoltField> {
    key: &'a UniformSpartanKey<F>,
    rx_var: Vec<F>,
    claimed_witness_evals: Vec<F>,
    inner_sumcheck_RLC: F,
}

pub struct InnerSumcheck<'a, F: JoltField> {
    input_claim: F,
    prover_state: Option<InnerSumcheckProverState<F>>,
    verifier_state: Option<InnerSumcheckVerifierState<'a, F>>,
    cached_claims: Option<(F, F)>, // (final_poly_abc_eval, final_poly_z_eval)
}

impl<'a, F: JoltField> InnerSumcheck<'a, F> {
    pub fn new_prover(
        key: &UniformSpartanKey<F>,
        input_polys: &[MultilinearPolynomial<F>],
        claims: &OuterSumcheckClaims<F>,
        params: &InnerSumcheckParams<F>,
        inner_sumcheck_RLC: F,
    ) -> Self {
        let num_vars_uniform = key.num_vars_uniform_padded();
        let claim_inner_joint =
            claims.az + inner_sumcheck_RLC * claims.bz + inner_sumcheck_RLC.square() * claims.cz;

        let (eq_r_cycle, _) = EqPlusOnePolynomial::evals(&params.r_cycle, None);

        // Evaluate A_small, B_small, C_small combined with RLC at point rx_var
        let poly_abc_small =
            DensePolynomial::new(key.evaluate_small_matrix_rlc(&params.rx_var, inner_sumcheck_RLC));

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

        Self {
            input_claim: claim_inner_joint,
            prover_state: Some(InnerSumcheckProverState {
                poly_abc_small: MultilinearPolynomial::LargeScalars(poly_abc_small),
                poly_z: MultilinearPolynomial::LargeScalars(poly_z),
            }),
            verifier_state: None,
            cached_claims: None,
        }
    }
    pub fn new_prover_streaming<PCS, ProofTranscript>(
        key: &UniformSpartanKey<F>,
        claims: &OuterSumcheckClaims<F>,
        params: &InnerSumcheckParams<F>,
        inner_sumcheck_RLC: F,
        trace: &'a [RV32IMCycle],
        preprocessing: &'a JoltProverPreprocessing<F, PCS, ProofTranscript>,
        shard_length: usize,
    ) -> Self
    where
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
        ProofTranscript: Transcript,
    {
        let num_vars_uniform = key.num_vars_uniform_padded();
        let claim_inner_joint =
            claims.az + inner_sumcheck_RLC * claims.bz + inner_sumcheck_RLC.square() * claims.cz;

        // Evaluate A_small, B_small, C_small combined with RLC at point rx_var
        let poly_abc_small =
            DensePolynomial::new(key.evaluate_small_matrix_rlc(&params.rx_var, inner_sumcheck_RLC));

        let span = span!(Level::INFO, "binding_z_second_sumcheck");
        let _guard = span.enter();

        let mut input_polys_oracle = R1CSInputsOracle::new(shard_length, trace, preprocessing);
        let num_shards = trace.len() / shard_length;

        let mut bind_z_stream = vec![F::zero(); num_vars_uniform];

        let eq_r_cycle = SplitEqPolynomial::new(&params.r_cycle);
        let num_x1_bits = eq_r_cycle.E1_len.log_2();
        let x1_bitmask = (1 << (num_x1_bits)) - 1;

        let mut shards = input_polys_oracle.next().unwrap();
        let num_polys = shards.len();

        for shard_idx in 0..num_shards {
            let base_poly_idx = shard_idx * shard_length;
            (_, shards) = rayon::join(
                || {
                    shards
                        .par_iter()
                        .zip(bind_z_stream.par_iter_mut().take(num_polys))
                        .for_each(|(poly, bind_z_eval)| {
                            match poly {
                                MultilinearPolynomial::LargeScalars(poly) => {
                                    process_large_scalar_polys(
                                        &poly.Z,
                                        bind_z_eval,
                                        &eq_r_cycle,
                                        base_poly_idx,
                                        x1_bitmask,
                                        num_x1_bits,
                                        shard_length,
                                    );
                                }

                                MultilinearPolynomial::U8Scalars(poly) => {
                                    process_small_scalar_polys(
                                        &poly.coeffs,
                                        bind_z_eval,
                                        &eq_r_cycle,
                                        base_poly_idx,
                                        x1_bitmask,
                                        num_x1_bits,
                                        shard_length,
                                    );
                                }

                                MultilinearPolynomial::U16Scalars(poly) => {
                                    process_small_scalar_polys(
                                        &poly.coeffs,
                                        bind_z_eval,
                                        &eq_r_cycle,
                                        base_poly_idx,
                                        x1_bitmask,
                                        num_x1_bits,
                                        shard_length,
                                    );
                                }

                                MultilinearPolynomial::U32Scalars(poly) => {
                                    process_small_scalar_polys(
                                        &poly.coeffs,
                                        bind_z_eval,
                                        &eq_r_cycle,
                                        base_poly_idx,
                                        x1_bitmask,
                                        num_x1_bits,
                                        shard_length,
                                    );
                                }

                                MultilinearPolynomial::U64Scalars(poly) => {
                                    process_small_scalar_polys(
                                        &poly.coeffs,
                                        bind_z_eval,
                                        &eq_r_cycle,
                                        base_poly_idx,
                                        x1_bitmask,
                                        num_x1_bits,
                                        shard_length,
                                    );
                                }

                                MultilinearPolynomial::I64Scalars(poly) => {
                                    process_small_scalar_polys(
                                        &poly.coeffs,
                                        bind_z_eval,
                                        &eq_r_cycle,
                                        base_poly_idx,
                                        x1_bitmask,
                                        num_x1_bits,
                                        shard_length,
                                    );
                                }
                                _ => unimplemented!("Unexpected MultilinearPolynomial variant"),
                            };
                        });
                },
                || {
                    let mut shard = Vec::with_capacity(num_polys);
                    if shard_idx != num_shards - 1 {
                        shard = input_polys_oracle.next().unwrap();
                    };
                    shard
                },
            );
        }

        // Set the constant value at the appropriate position
        if key.uniform_r1cs.num_vars < num_vars_uniform {
            bind_z_stream[key.uniform_r1cs.num_vars] = F::one();
        }

        drop(_guard);
        drop(span);

        let poly_z = DensePolynomial::new(bind_z_stream);
        assert_eq!(poly_z.len(), poly_abc_small.len());

        Self {
            input_claim: claim_inner_joint,
            prover_state: Some(InnerSumcheckProverState {
                poly_abc_small: MultilinearPolynomial::LargeScalars(poly_abc_small),
                poly_z: MultilinearPolynomial::LargeScalars(poly_z),
            }),
            verifier_state: None,
            cached_claims: None,
        }
    }
    pub fn new_verifier(
        input_claim: F,
        key: &'a UniformSpartanKey<F>,
        rx_var: Vec<F>,
        claimed_witness_evals: Vec<F>,
        inner_sumcheck_RLC: F,
    ) -> Self {
        Self {
            input_claim,
            prover_state: None,
            verifier_state: Some(InnerSumcheckVerifierState {
                key,
                rx_var,
                claimed_witness_evals,
                inner_sumcheck_RLC,
            }),
            cached_claims: None,
        }
    }
}

impl<F: JoltField, ProofTranscript: Transcript> BatchableSumcheckInstance<F, ProofTranscript>
    for InnerSumcheck<'_, F>
{
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        if let Some(prover_state) = &self.prover_state {
            prover_state.poly_abc_small.original_len().log_2()
        } else if let Some(verifier_state) = &self.verifier_state {
            verifier_state.key.num_vars_uniform_padded().log_2()
        } else {
            panic!("Neither prover state nor verifier state is initialized");
        }
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    fn compute_prover_message(&self, _round: usize) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        let degree = <Self as BatchableSumcheckInstance<F, ProofTranscript>>::degree(self);

        let univariate_poly_evals: Vec<F> = (0..prover_state.poly_abc_small.len() / 2)
            .into_par_iter()
            .map(|i| {
                let abc_evals =
                    prover_state
                        .poly_abc_small
                        .sumcheck_evals(i, degree, BindingOrder::HighToLow);
                let z_evals =
                    prover_state
                        .poly_z
                        .sumcheck_evals(i, degree, BindingOrder::HighToLow);

                vec![
                    abc_evals[0] * z_evals[0], // eval at 0
                    abc_evals[1] * z_evals[1], // eval at 2
                ]
            })
            .reduce(
                || vec![F::zero(); degree],
                |mut running, new| {
                    for i in 0..degree {
                        running[i] += new[i];
                    }
                    running
                },
            );

        univariate_poly_evals
    }

    fn bind(&mut self, r_j: F, _round: usize) {
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        // Bind both polynomials in parallel
        rayon::join(
            || {
                prover_state
                    .poly_abc_small
                    .bind_parallel(r_j, BindingOrder::HighToLow)
            },
            || {
                prover_state
                    .poly_z
                    .bind_parallel(r_j, BindingOrder::HighToLow)
            },
        );
    }

    fn cache_openings(&mut self) {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let final_poly_abc = prover_state.poly_abc_small.final_sumcheck_claim();
        let final_poly_z = prover_state.poly_z.final_sumcheck_claim();

        self.cached_claims = Some((final_poly_abc, final_poly_z));
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        let verifier_state = self
            .verifier_state
            .as_ref()
            .expect("Verifier state not initialized");

        // The verifier needs to compute:
        // (A_small(rx, ry) + r * B_small(rx, ry) + r^2 * C_small(rx, ry)) * z(ry)

        // Evaluate uniform matrices A_small, B_small, C_small at point (rx_var, ry_var)
        let eval_a = verifier_state
            .key
            .evaluate_uniform_a_at_point(&verifier_state.rx_var, r);
        let eval_b = verifier_state
            .key
            .evaluate_uniform_b_at_point(&verifier_state.rx_var, r);
        let eval_c = verifier_state
            .key
            .evaluate_uniform_c_at_point(&verifier_state.rx_var, r);

        let left_expected = eval_a
            + verifier_state.inner_sumcheck_RLC * eval_b
            + verifier_state.inner_sumcheck_RLC * verifier_state.inner_sumcheck_RLC * eval_c;

        // Evaluate z(ry)
        let eval_z = verifier_state.key.evaluate_z_mle_with_segment_evals(
            &verifier_state.claimed_witness_evals,
            r,
            true,
        );

        left_expected * eval_z
    }
}

struct PCSumcheckProverState<F: JoltField> {
    unexpanded_pc_poly: MultilinearPolynomial<F>,
    pc_poly: MultilinearPolynomial<F>,
    eq_plus_one_poly: MultilinearPolynomial<F>,
    r: F,
}

struct PCSumcheckVerifierState<F: JoltField> {
    r_cycle: Vec<F>,
    r: F,
    unexpanded_pc_eval_at_shift_r: F,
    pc_eval_at_shift_r: F,
}

pub struct PCSumcheck<F: JoltField> {
    input_claim: F,
    prover_state: Option<PCSumcheckProverState<F>>,
    verifier_state: Option<PCSumcheckVerifierState<F>>,
    cached_claims: Option<(F, F)>, // (unexpanded_pc_eval, pc_eval)
}

impl<F: JoltField> PCSumcheck<F> {
    pub fn new_prover(
        input_polys: &[MultilinearPolynomial<F>],
        claimed_witness_evals: &[F],
        eq_plus_one_r_cycle: Vec<F>,
        r: F,
    ) -> Self {
        let unexpanded_pc_index = JoltR1CSInputs::UnexpandedPC.to_index();
        let pc_index = JoltR1CSInputs::PC.to_index();
        let next_unexpanded_pc_index = JoltR1CSInputs::NextUnexpandedPC.to_index();
        let next_pc_index = JoltR1CSInputs::NextPC.to_index();

        // The batched claim equals NextUnexpandedPC(r_cycle) + r * NextPC(r_cycle)
        let input_claim = claimed_witness_evals[next_unexpanded_pc_index]
            + r * claimed_witness_evals[next_pc_index];

        Self {
            input_claim,
            prover_state: Some(PCSumcheckProverState {
                unexpanded_pc_poly: input_polys[unexpanded_pc_index].clone(),
                pc_poly: input_polys[pc_index].clone(),
                eq_plus_one_poly: MultilinearPolynomial::from(eq_plus_one_r_cycle),
                r,
            }),
            verifier_state: None,
            cached_claims: None,
        }
    }

    pub fn new_verifier(
        input_claim: F,
        r_cycle: Vec<F>,
        r: F,
        unexpanded_pc_eval_at_shift_r: F,
        pc_eval_at_shift_r: F,
    ) -> Self {
        Self {
            input_claim,
            prover_state: None,
            verifier_state: Some(PCSumcheckVerifierState {
                r_cycle,
                r,
                unexpanded_pc_eval_at_shift_r,
                pc_eval_at_shift_r,
            }),
            cached_claims: None,
        }
    }
}

impl<F: JoltField, ProofTranscript: Transcript> BatchableSumcheckInstance<F, ProofTranscript>
    for PCSumcheck<F>
{
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        if let Some(prover_state) = &self.prover_state {
            prover_state.unexpanded_pc_poly.get_num_vars()
        } else if let Some(verifier_state) = &self.verifier_state {
            verifier_state.r_cycle.len()
        } else {
            panic!("Neither prover state nor verifier state is initialized");
        }
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    fn compute_prover_message(&self, _round: usize) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        let degree = <Self as BatchableSumcheckInstance<F, ProofTranscript>>::degree(self);

        let univariate_poly_evals: Vec<F> = (0..prover_state.unexpanded_pc_poly.len() / 2)
            .into_par_iter()
            .map(|i| {
                let unexpanded_pc_evals = prover_state.unexpanded_pc_poly.sumcheck_evals(
                    i,
                    degree,
                    BindingOrder::HighToLow,
                );
                let pc_evals =
                    prover_state
                        .pc_poly
                        .sumcheck_evals(i, degree, BindingOrder::HighToLow);
                let eq_evals = prover_state.eq_plus_one_poly.sumcheck_evals(
                    i,
                    degree,
                    BindingOrder::HighToLow,
                );

                vec![
                    (unexpanded_pc_evals[0] + prover_state.r * pc_evals[0]) * eq_evals[0], // eval at 0
                    (unexpanded_pc_evals[1] + prover_state.r * pc_evals[1]) * eq_evals[1], // eval at 2
                ]
            })
            .reduce(
                || vec![F::zero(); degree],
                |mut running, new| {
                    for i in 0..degree {
                        running[i] += new[i];
                    }
                    running
                },
            );

        univariate_poly_evals
    }

    fn bind(&mut self, r_j: F, _round: usize) {
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        rayon::join(
            || {
                rayon::join(
                    || {
                        prover_state
                            .unexpanded_pc_poly
                            .bind_parallel(r_j, BindingOrder::HighToLow)
                    },
                    || {
                        prover_state
                            .pc_poly
                            .bind_parallel(r_j, BindingOrder::HighToLow)
                    },
                )
            },
            || {
                prover_state
                    .eq_plus_one_poly
                    .bind_parallel(r_j, BindingOrder::HighToLow)
            },
        );
    }

    fn cache_openings(&mut self) {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let unexpanded_pc_eval = prover_state.unexpanded_pc_poly.final_sumcheck_claim();
        let pc_eval = prover_state.pc_poly.final_sumcheck_claim();

        self.cached_claims = Some((unexpanded_pc_eval, pc_eval));
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        let verifier_state = self
            .verifier_state
            .as_ref()
            .expect("Verifier state not initialized");

        // Compute the expected claim:
        // batched_eval_at_shift_r * eq_plus_one_shift_sumcheck
        let batched_eval_at_shift_r = verifier_state.unexpanded_pc_eval_at_shift_r
            + verifier_state.r * verifier_state.pc_eval_at_shift_r;

        let eq_plus_one_shift_sumcheck =
            EqPlusOnePolynomial::new(verifier_state.r_cycle.clone()).evaluate(r);

        batched_eval_at_shift_r * eq_plus_one_shift_sumcheck
    }
}
