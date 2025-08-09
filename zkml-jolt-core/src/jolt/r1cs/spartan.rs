use jolt_core::field::JoltField;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::multilinear_polynomial::PolynomialEvaluation;
use jolt_core::poly::multilinear_polynomial::{
    BindingOrder, MultilinearPolynomial, PolynomialBinding,
};
use jolt_core::subprotocols::sumcheck::SumcheckInstanceProof;
use jolt_core::utils::math::Math;
use jolt_core::utils::small_value::NUM_SVO_ROUNDS;
use jolt_core::utils::transcript::Transcript;
use jolt_core::{
    poly::{dense_mlpoly::DensePolynomial, eq_poly::EqPlusOnePolynomial},
    subprotocols::sumcheck::BatchableSumcheckInstance,
};

use rayon::prelude::*;
use std::marker::PhantomData;
use thiserror::Error;
use tracing::{Level, span};

use crate::jolt::JoltProverPreprocessing;
use crate::jolt::execution_trace::{
    ALL_R1CS_INPUTS, JoltONNXCycle, JoltONNXR1CSInputs, WitnessGenerator,
};
use crate::jolt::r1cs::builder::CombinedUniformBuilder;
use crate::jolt::r1cs::key::UniformSpartanKey;
#[cfg(test)]
use crate::jolt::r1cs::spartan_interleaved_poly::SpartanInterleavedPolynomial;

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
#[derive(Debug, Clone)]
pub struct UniformSpartanProof<F: JoltField, ProofTranscript: Transcript> {
    pub(crate) outer_sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub(crate) outer_sumcheck_claims: (F, F, F),
    pub(crate) inner_sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub(crate) claimed_witness_evals: Vec<F>,
    pub(crate) shift_sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
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
        trace: &[JoltONNXCycle],
        // opening_accumulator: &mut ProverOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<Self, SpartanError>
    where
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
    {
        let input_polys: Vec<MultilinearPolynomial<F>> = ALL_R1CS_INPUTS
            .par_iter()
            .map(|var| var.generate_witness::<F, PCS, ProofTranscript>(trace, preprocessing))
            .collect();

        #[cfg(test)]
        {
            SpartanInterleavedPolynomial::new(
                &constraint_builder.uniform_builder.constraints,
                &input_polys,
                constraint_builder.padded_rows_per_step(),
            );
        }

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
        let (claimed_witness_evals, _chis) =
            MultilinearPolynomial::batch_evaluate(&flattened_polys_ref, r_cycle);

        let (_, eq_plus_one_r_cycle) = EqPlusOnePolynomial::evals(r_cycle, None);

        /*  Sumcheck 3: sumcheck for NextPC verification
            Proves: NextPC(r_cycle) =
                    \sum_t  PC(t)) * eq_plus_one(r_cycle, t)

            1. NextPC(r_cycle) = \sum_t PC(t) * eq_plus_one(r_cycle, t)
        */
        let (shift_sumcheck_proof, shift_sumcheck_witness_eval) = Self::prove_pc_sumcheck(
            &input_polys,
            &claimed_witness_evals,
            eq_plus_one_r_cycle,
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
            claimed_witness_evals,
            shift_sumcheck_proof,
            shift_sumcheck_witness_eval,
            _marker: PhantomData,
        })
    }

    fn prove_pc_sumcheck(
        input_polys: &[MultilinearPolynomial<F>],
        claimed_witness_evals: &[F],
        eq_plus_one_r_cycle: Vec<F>,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>) {
        let span = span!(Level::INFO, "shift_sumcheck_pc");
        let _guard = span.enter();

        let mut pc_sumcheck =
            PCSumcheck::new_prover(input_polys, claimed_witness_evals, eq_plus_one_r_cycle);

        let (shift_sumcheck_proof, _r) = pc_sumcheck.prove_single(transcript);

        let cached_claims = pc_sumcheck.cached_claims.expect("Claims not cached");
        let pc_eval_at_shift_r = cached_claims;
        let shift_sumcheck_witness_eval = vec![pc_eval_at_shift_r];

        drop(_guard);
        drop(span);

        (shift_sumcheck_proof, shift_sumcheck_witness_eval)
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

    #[tracing::instrument(skip_all, name = "Spartan::verify")]
    pub fn verify<PCS>(
        &self,
        key: &UniformSpartanKey<F>,
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

        /* Sumcheck 3: Batched sumcheck NextPC verification
           Verifies the batched constraint for NextPC
        */

        let next_pc_index = JoltONNXR1CSInputs::NextPC.to_index();

        // The batched claim equals NextPC(r_cycle)
        let shift_sumcheck_claim = self.claimed_witness_evals[next_pc_index];

        let pc_eval_at_shift_r = self.shift_sumcheck_witness_eval[0];

        let pc_sumcheck = PCSumcheck::<F>::new_verifier(
            shift_sumcheck_claim,
            r_cycle.to_vec(),
            pc_eval_at_shift_r,
        );

        let _shift_sumcheck_r = pc_sumcheck
            .verify_single(&self.shift_sumcheck_proof, transcript)
            .map_err(|_| SpartanError::InvalidShiftSumcheckProof)?;
        Ok(())
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

    fn compute_prover_message(&mut self, _round: usize) -> Vec<F> {
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
    pc_poly: MultilinearPolynomial<F>,
    eq_plus_one_poly: MultilinearPolynomial<F>,
}

struct PCSumcheckVerifierState<F: JoltField> {
    r_cycle: Vec<F>,
    pc_eval_at_shift_r: F,
}

pub struct PCSumcheck<F: JoltField> {
    input_claim: F,
    prover_state: Option<PCSumcheckProverState<F>>,
    verifier_state: Option<PCSumcheckVerifierState<F>>,
    cached_claims: Option<F>, // (pc_eval)
}

impl<F: JoltField> PCSumcheck<F> {
    pub fn new_prover(
        input_polys: &[MultilinearPolynomial<F>],
        claimed_witness_evals: &[F],
        eq_plus_one_r_cycle: Vec<F>,
    ) -> Self {
        let pc_index = JoltONNXR1CSInputs::PC.to_index();
        let next_pc_index = JoltONNXR1CSInputs::NextPC.to_index();

        // The claim equals NextPC(r_cycle)
        let input_claim = claimed_witness_evals[next_pc_index];

        Self {
            input_claim,
            prover_state: Some(PCSumcheckProverState {
                pc_poly: input_polys[pc_index].clone(),
                eq_plus_one_poly: MultilinearPolynomial::from(eq_plus_one_r_cycle),
            }),
            verifier_state: None,
            cached_claims: None,
        }
    }

    pub fn new_verifier(input_claim: F, r_cycle: Vec<F>, pc_eval_at_shift_r: F) -> Self {
        Self {
            input_claim,
            prover_state: None,
            verifier_state: Some(PCSumcheckVerifierState {
                r_cycle,
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
            prover_state.pc_poly.get_num_vars()
        } else if let Some(verifier_state) = &self.verifier_state {
            verifier_state.r_cycle.len()
        } else {
            panic!("Neither prover state nor verifier state is initialized");
        }
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    fn compute_prover_message(&mut self, _round: usize) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        let degree = <Self as BatchableSumcheckInstance<F, ProofTranscript>>::degree(self);

        let univariate_poly_evals: Vec<F> = (0..prover_state.pc_poly.len() / 2)
            .into_par_iter()
            .map(|i| {
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
                    (pc_evals[0]) * eq_evals[0], // eval at 0
                    (pc_evals[1]) * eq_evals[1], // eval at 2
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
                prover_state
                    .pc_poly
                    .bind_parallel(r_j, BindingOrder::HighToLow)
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

        let pc_eval = prover_state.pc_poly.final_sumcheck_claim();

        self.cached_claims = Some(pc_eval);
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        let verifier_state = self
            .verifier_state
            .as_ref()
            .expect("Verifier state not initialized");

        // Compute the expected claim:
        // batched_eval_at_shift_r * eq_plus_one_shift_sumcheck
        let batched_eval_at_shift_r = verifier_state.pc_eval_at_shift_r;

        let eq_plus_one_shift_sumcheck =
            EqPlusOnePolynomial::new(verifier_state.r_cycle.clone()).evaluate(r);

        batched_eval_at_shift_r * eq_plus_one_shift_sumcheck
    }
}
