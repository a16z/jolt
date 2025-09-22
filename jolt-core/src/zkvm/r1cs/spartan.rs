use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;
use tracing::{span, Level};

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::poly::opening_proof::{
    OpeningId, OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
    BIG_ENDIAN,
};
use crate::utils::math::Math;
#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
use crate::zkvm::dag::stage::SumcheckStages;
use crate::zkvm::dag::state_manager::{ProofData, ProofKeys, StateManager};
use crate::zkvm::instruction::CircuitFlags;
use crate::zkvm::r1cs::inputs::{
    compute_claimed_witness_evals, generate_pc_noop_witnesses, JoltR1CSInputs,
    TraceWitnessAccessor, WitnessRowAccessor, ALL_R1CS_INPUTS, COMMITTED_R1CS_INPUTS,
};
use crate::zkvm::r1cs::key::UniformSpartanKey;
use crate::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};

use crate::transcripts::Transcript;
use crate::utils::small_value::NUM_SVO_ROUNDS;
use ark_serialize::CanonicalDeserialize;
use ark_serialize::CanonicalSerialize;

use thiserror::Error;

use crate::{
    poly::{dense_mlpoly::DensePolynomial, eq_poly::EqPlusOnePolynomial},
    subprotocols::sumcheck::{SumcheckInstance, SumcheckInstanceProof},
};
use rayon::prelude::*;

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum SpartanError {
    /// returned when the outer sumcheck proof fails
    #[error("InvalidOuterSumcheckProof")]
    InvalidOuterSumcheckProof,

    /// returned when the final sumcheck opening proof fails
    #[error("InvalidOuterSumcheckClaim")]
    InvalidOuterSumcheckClaim,

    /// returned when the recursive sumcheck proof fails
    #[error("InvalidInnerSumcheckProof")]
    InvalidInnerSumcheckProof,

    /// returned when the recursive sumcheck proof fails
    #[error("InvalidShiftSumcheckProof")]
    InvalidShiftSumcheckProof,
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

impl<F: JoltField, ProofTranscript> UniformSpartanProof<F, ProofTranscript>
where
    ProofTranscript: Transcript,
{
    #[tracing::instrument(skip_all, name = "Spartan::setup")]
    #[inline]
    pub fn setup(padded_num_steps: usize) -> UniformSpartanKey<F> {
        UniformSpartanKey::new(padded_num_steps)
    }

    #[tracing::instrument(skip_all)]
    fn prove_outer_sumcheck(
        num_rounds_x: usize,
        accessor: &dyn WitnessRowAccessor<F>,
        tau: &[F],
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, [F; 3]) {
        SumcheckInstanceProof::prove_spartan_small_value::<NUM_SVO_ROUNDS>(
            num_rounds_x,
            accessor,
            tau,
            transcript,
        )
    }
}

#[derive(Allocative)]
struct InnerSumcheckProverState<F: JoltField> {
    poly_abc_small: MultilinearPolynomial<F>,
    poly_z: MultilinearPolynomial<F>,
}

#[derive(Allocative)]
pub struct InnerSumcheck<F: JoltField> {
    input_claim: F,
    prover_state: Option<InnerSumcheckProverState<F>>,
    #[allocative(skip)]
    key: Option<Arc<UniformSpartanKey<F>>>,
    gamma: Option<F>,
}

impl<F: JoltField> InnerSumcheck<F> {
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        key: Arc<UniformSpartanKey<F>>,
    ) -> Self {
        let num_vars_uniform = key.num_vars_uniform_padded();
        let num_cycles = key.num_steps;
        let num_cycles_bits = num_cycles.ilog2() as usize;

        // Get gamma challenge for batching
        let gamma: F = state_manager.transcript.borrow_mut().challenge_scalar();

        // Get opening_point and claims from accumulator (Az, Bz, Cz all have the same point)
        let (outer_sumcheck_r, claim_Az) = state_manager
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanAz, SumcheckId::SpartanOuter);
        let (_, claim_Bz) = state_manager
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanBz, SumcheckId::SpartanOuter);
        let (_, claim_Cz) = state_manager
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanCz, SumcheckId::SpartanOuter);

        let (_r_cycle, rx_var) = outer_sumcheck_r.r.split_at(num_cycles_bits);

        let claim_inner_joint = claim_Az + gamma * claim_Bz + gamma.square() * claim_Cz;

        // Evaluate A_small, B_small, C_small combined with RLC at point rx_var
        let poly_abc_small = DensePolynomial::new(key.evaluate_small_matrix_rlc(rx_var, gamma));

        let span = span!(Level::INFO, "binding_z_second_sumcheck");
        let _guard = span.enter();

        // Bind witness polynomials z at point r_cycle
        let mut bind_z = vec![F::zero(); num_vars_uniform];

        ALL_R1CS_INPUTS
            .into_iter()
            .zip(bind_z.iter_mut())
            .for_each(|(r1cs_input, dest)| {
                let accumulator = state_manager.get_prover_accumulator();
                let accumulator = accumulator.borrow();
                let key = OpeningId::try_from(&r1cs_input).expect(
                    "Failed to map R1CS input to OpeningId (neither virtual nor committed)",
                );
                let (_, claim) = accumulator
                    .openings
                    .get(&key)
                    .expect("Missing opening claim for expected OpeningId in bind_z");
                *dest = *claim;
            });

        // Set the constant value at the appropriate position
        let const_col = JoltR1CSInputs::num_inputs();
        if const_col < num_vars_uniform {
            bind_z[const_col] = F::one();
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
            key: None,
            gamma: None,
        }
    }

    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        key: Arc<UniformSpartanKey<F>>,
    ) -> Self {
        // Pull the outer sumcheck claims from the accumulator
        let accumulator = state_manager.get_verifier_accumulator();
        let accumulator_ref = accumulator.borrow();
        let (_, claim_Az) = accumulator_ref
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanAz, SumcheckId::SpartanOuter);
        let (_, claim_Bz) = accumulator_ref
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanBz, SumcheckId::SpartanOuter);
        let (_, claim_Cz) = accumulator_ref
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanCz, SumcheckId::SpartanOuter);
        drop(accumulator_ref);

        // Get gamma challenge for batching
        let gamma: F = state_manager.transcript.borrow_mut().challenge_scalar();

        // Compute joint claim
        let input_claim = claim_Az + gamma * claim_Bz + gamma.square() * claim_Cz;

        Self {
            input_claim,
            prover_state: None,
            key: Some(key),
            gamma: Some(gamma),
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for InnerSumcheck<F> {
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        if let Some(prover_state) = &self.prover_state {
            prover_state.poly_abc_small.original_len().log_2()
        } else if let Some(key) = &self.key {
            key.num_vars_uniform_padded().log_2()
        } else {
            panic!("Neither prover state nor key is initialized");
        }
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    #[tracing::instrument(skip_all, name = "InnerSumcheck::compute_prover_message")]
    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        const DEGREE: usize = 2;

        let univariate_poly_evals: [F; DEGREE] = (0..prover_state.poly_abc_small.len() / 2)
            .into_par_iter()
            .map(|i| {
                let abc_evals = prover_state
                    .poly_abc_small
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let z_evals = prover_state
                    .poly_z
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);

                [
                    abc_evals[0] * z_evals[0], // eval at 0
                    abc_evals[1] * z_evals[1], // eval at 2
                ]
            })
            .reduce(
                || [F::zero(); DEGREE],
                |mut running, new| {
                    for i in 0..DEGREE {
                        running[i] += new[i];
                    }
                    running
                },
            );

        univariate_poly_evals.into()
    }

    #[tracing::instrument(skip_all, name = "InnerSumcheck::bind")]
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

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F],
    ) -> F {
        let key = self.key.as_ref().expect("Key not initialized");
        let gamma = self.gamma.expect("gamma not initialized");

        let accumulator = accumulator.as_ref().unwrap().borrow();

        // Get rx_var from the outer sumcheck opening point in accumulator
        let (outer_sumcheck_opening, _) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanAz, SumcheckId::SpartanOuter);
        let num_cycles_bits = key.num_steps.log_2();
        let (_r_cycle, rx_var) = outer_sumcheck_opening.r.split_at(num_cycles_bits);

        // Pull claimed witness evaluations from the accumulator
        let claimed_witness_evals: Vec<F> = ALL_R1CS_INPUTS
            .into_iter()
            .map(|r1cs_input| {
                let (_, claim) = accumulator
                    .openings
                    .get(&OpeningId::try_from(&r1cs_input).ok().unwrap())
                    .unwrap();
                *claim
            })
            .collect();

        // The verifier needs to compute:
        // (A_small(rx, ry) + r * B_small(rx, ry) + r^2 * C_small(rx, ry)) * z(ry)

        // Evaluate uniform matrices A_small, B_small, C_small at point (rx_var, ry_var)
        let eval_a = key.evaluate_uniform_a_at_point(rx_var, r);
        let eval_b = key.evaluate_uniform_b_at_point(rx_var, r);
        let eval_c = key.evaluate_uniform_c_at_point(rx_var, r);

        let left_expected = eval_a + gamma * eval_b + gamma.square() * eval_c;

        // Evaluate z(ry)
        let eval_z = key.evaluate_z_mle_with_segment_evals(&claimed_witness_evals, r, true);

        left_expected * eval_z
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings_prover(
        &self,
        _accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        _opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        // Nothing to cache
    }

    fn cache_openings_verifier(
        &self,
        _accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        _opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        // Nothing to cache
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

#[derive(Allocative)]
struct PCSumcheckProverState<F: JoltField> {
    unexpanded_pc_poly: MultilinearPolynomial<F>,
    pc_poly: MultilinearPolynomial<F>,
    is_noop_poly: MultilinearPolynomial<F>,
    eq_plus_one_poly: MultilinearPolynomial<F>,
}

#[derive(Allocative)]
pub struct PCSumcheck<F: JoltField> {
    input_claim: F,
    gamma: F,
    gamma_squared: F,
    log_T: usize,
    prover_state: Option<PCSumcheckProverState<F>>,
}

impl<F: JoltField> PCSumcheck<F> {
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        key: Arc<UniformSpartanKey<F>>,
    ) -> Self {
        let (preprocessing, trace, _program_io, _final_memory_state) =
            state_manager.get_prover_data();

        // Stream once to generate PC, UnexpandedPC and IsNoop witnesses
        let (unexpanded_pc_poly, pc_poly, is_noop_poly) =
            generate_pc_noop_witnesses(preprocessing, trace);

        let num_cycles = key.num_steps;
        let num_cycles_bits = num_cycles.ilog2() as usize;

        // Get opening_point and claims from accumulator
        let accumulator = state_manager.get_prover_accumulator();
        let (outer_sumcheck_r, next_pc_eval) = accumulator
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::NextPC, SumcheckId::SpartanOuter);
        let (_, next_unexpanded_pc_eval) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::NextUnexpandedPC,
            SumcheckId::SpartanOuter,
        );
        let (_, next_is_noop_eval) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::NextIsNoop,
            SumcheckId::SpartanOuter,
        );

        let (r_cycle, _rx_var) = outer_sumcheck_r.split_at(num_cycles_bits);

        let (_, eq_plus_one_r_cycle) = EqPlusOnePolynomial::evals(&r_cycle.r, None);

        let gamma: F = state_manager.transcript.borrow_mut().challenge_scalar();
        let gamma_squared = gamma.square();

        let input_claim =
            next_unexpanded_pc_eval + gamma * next_pc_eval + gamma_squared * next_is_noop_eval;

        Self {
            input_claim,
            log_T: r_cycle.len(),
            prover_state: Some(PCSumcheckProverState {
                unexpanded_pc_poly,
                pc_poly,
                is_noop_poly,
                eq_plus_one_poly: MultilinearPolynomial::from(eq_plus_one_r_cycle),
            }),
            gamma,
            gamma_squared,
        }
    }

    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        key: Arc<UniformSpartanKey<F>>,
    ) -> Self {
        // Get batching challenge for combining NextUnexpandedPC and NextPC
        let gamma: F = state_manager.transcript.borrow_mut().challenge_scalar();
        let gamma_squared = gamma.square();

        // Get the Next* evaluations from the accumulator
        let accumulator = state_manager.get_verifier_accumulator();
        let (_, next_pc_eval) = accumulator
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::NextPC, SumcheckId::SpartanOuter);
        let (_, next_unexpanded_pc_eval) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::NextUnexpandedPC,
            SumcheckId::SpartanOuter,
        );
        let (_, next_is_noop_eval) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::NextIsNoop,
            SumcheckId::SpartanOuter,
        );

        let input_claim =
            next_unexpanded_pc_eval + gamma * next_pc_eval + gamma_squared * next_is_noop_eval;
        let log_T = key.num_steps.log_2();

        Self {
            input_claim,
            prover_state: None,
            log_T,
            gamma,
            gamma_squared,
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for PCSumcheck<F> {
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.log_T
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    #[tracing::instrument(skip_all, name = "PCSumcheck::compute_prover_message")]
    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        const DEGREE: usize = 2;

        let univariate_poly_evals: [F; DEGREE] = (0..prover_state.unexpanded_pc_poly.len() / 2)
            .into_par_iter()
            .map(|i| {
                let unexpanded_pc_evals = prover_state
                    .unexpanded_pc_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let pc_evals = prover_state
                    .pc_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let eq_evals = prover_state
                    .eq_plus_one_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let is_noop_evals = prover_state
                    .is_noop_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);

                [
                    (unexpanded_pc_evals[0]
                        + self.gamma * pc_evals[0]
                        + self.gamma_squared * is_noop_evals[0])
                        * eq_evals[0], // eval at 0
                    (unexpanded_pc_evals[1]
                        + self.gamma * pc_evals[1]
                        + self.gamma_squared * is_noop_evals[1])
                        * eq_evals[1], // eval at 2
                ]
            })
            .reduce(
                || [F::zero(); DEGREE],
                |mut running, new| {
                    for i in 0..DEGREE {
                        running[i] += new[i];
                    }
                    running
                },
            );

        univariate_poly_evals.into()
    }

    #[tracing::instrument(skip_all, name = "PCSumcheck::bind")]
    fn bind(&mut self, r_j: F, _round: usize) {
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        rayon::scope(|s| {
            s.spawn(|_| {
                prover_state
                    .unexpanded_pc_poly
                    .bind_parallel(r_j, BindingOrder::HighToLow)
            });
            s.spawn(|_| {
                prover_state
                    .pc_poly
                    .bind_parallel(r_j, BindingOrder::HighToLow)
            });
            s.spawn(|_| {
                prover_state
                    .is_noop_poly
                    .bind_parallel(r_j, BindingOrder::HighToLow)
            });
            s.spawn(|_| {
                prover_state
                    .eq_plus_one_poly
                    .bind_parallel(r_j, BindingOrder::HighToLow)
            });
        });
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F],
    ) -> F {
        let accumulator = accumulator.as_ref().unwrap().borrow();

        // Get r_cycle from the SpartanOuter sumcheck opening point
        let (outer_sumcheck_opening, _) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::NextPC, SumcheckId::SpartanOuter);
        let outer_sumcheck_r = &outer_sumcheck_opening.r;
        let num_cycles_bits = self.log_T;
        let (r_cycle, _) = outer_sumcheck_r.split_at(num_cycles_bits);

        // Get the shift evaluations from the accumulator
        let (_, unexpanded_pc_eval_at_shift_r) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
        );
        let (_, pc_eval_at_shift_r) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanShift);
        let (_, is_noop_eval_at_shift_r) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::IsNoop),
            SumcheckId::SpartanShift,
        );

        let batched_eval_at_shift_r = unexpanded_pc_eval_at_shift_r
            + self.gamma * pc_eval_at_shift_r
            + self.gamma_squared * is_noop_eval_at_shift_r;

        let eq_plus_one_shift_sumcheck = EqPlusOnePolynomial::new(r_cycle.to_vec()).evaluate(r);

        batched_eval_at_shift_r * eq_plus_one_shift_sumcheck
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let unexpanded_pc_eval = prover_state.unexpanded_pc_poly.final_sumcheck_claim();
        let pc_eval = prover_state.pc_poly.final_sumcheck_claim();
        let is_noop_eval = prover_state.is_noop_poly.final_sumcheck_claim();

        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
            opening_point.clone(),
            unexpanded_pc_eval,
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::PC,
            SumcheckId::SpartanShift,
            opening_point.clone(),
            pc_eval,
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::OpFlags(CircuitFlags::IsNoop),
            SumcheckId::SpartanShift,
            opening_point,
            is_noop_eval,
        );
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
            opening_point.clone(),
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::PC,
            SumcheckId::SpartanShift,
            opening_point.clone(),
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::OpFlags(CircuitFlags::IsNoop),
            SumcheckId::SpartanShift,
            opening_point,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct SpartanDag<F: JoltField> {
    /// Cached key to avoid recomputation across stages
    key: Arc<UniformSpartanKey<F>>,
}

impl<F: JoltField> SpartanDag<F> {
    pub fn new<ProofTranscript: Transcript>(padded_trace_length: usize) -> Self {
        let key = Arc::new(UniformSpartanProof::<F, ProofTranscript>::setup(
            padded_trace_length,
        ));
        Self { key }
    }
}

impl<F, ProofTranscript, PCS> SumcheckStages<F, ProofTranscript, PCS> for SpartanDag<F>
where
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    fn stage1_prove(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Result<(), anyhow::Error> {
        /* Sumcheck 1: Outer sumcheck
           Proves: \sum_x eq(tau, x) * (Az(x) * Bz(x) - Cz(x)) = 0

           The matrices A, B, C have a block-diagonal structure with repeated blocks
           A_small, B_small, C_small corresponding to the uniform constraints.
        */
        let (preprocessing, trace, _program_io, _final_memory_state) =
            state_manager.get_prover_data();

        let key = self.key.clone();

        // Streaming accessor (no materialization of all input_polys)
        let accessor = TraceWitnessAccessor::<F, PCS>::new(preprocessing, trace);

        let num_rounds_x = key.num_rows_bits();

        let tau: Vec<F> = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector(num_rounds_x);

        let (outer_sumcheck_proof, outer_sumcheck_r, outer_sumcheck_claims) = {
            let mut transcript = state_manager.transcript.borrow_mut();
            UniformSpartanProof::<F, ProofTranscript>::prove_outer_sumcheck(
                num_rounds_x,
                &accessor,
                &tau,
                &mut transcript,
            )
        };

        let outer_sumcheck_r: Vec<F> = outer_sumcheck_r.into_iter().rev().collect();

        ProofTranscript::append_scalars(
            &mut *state_manager.transcript.borrow_mut(),
            &outer_sumcheck_claims,
        );

        // Store Az, Bz, Cz claims with the outer sumcheck point
        let accumulator = state_manager.get_prover_accumulator();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::SpartanAz,
            SumcheckId::SpartanOuter,
            OpeningPoint::new(outer_sumcheck_r.clone()),
            outer_sumcheck_claims[0],
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::SpartanBz,
            SumcheckId::SpartanOuter,
            OpeningPoint::new(outer_sumcheck_r.clone()),
            outer_sumcheck_claims[1],
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::SpartanCz,
            SumcheckId::SpartanOuter,
            OpeningPoint::new(outer_sumcheck_r.clone()),
            outer_sumcheck_claims[2],
        );

        // Append the outer sumcheck proof to the state manager
        state_manager.proofs.borrow_mut().insert(
            ProofKeys::Stage1Sumcheck,
            ProofData::SumcheckProof(outer_sumcheck_proof),
        );

        let num_cycles = key.num_steps;
        let num_cycles_bits = num_cycles.ilog2() as usize;

        let (r_cycle, _rx_var) = outer_sumcheck_r.split_at(num_cycles_bits);

        // Compute claimed witness evals at r_cycle via streaming
        let claimed_witness_evals = compute_claimed_witness_evals::<F>(r_cycle, &accessor);

        // Only non-virtual (i.e. committed) polynomials' openings are
        // proven using the PCS opening proof, which we add for future opening proof here
        let committed_polys: Vec<_> = COMMITTED_R1CS_INPUTS
            .iter()
            .map(|input| CommittedPolynomial::try_from(input).ok().unwrap())
            .collect();
        let committed_poly_claims: Vec<_> = COMMITTED_R1CS_INPUTS
            .iter()
            .map(|input| claimed_witness_evals[input.to_index()])
            .collect();

        let accumulator = state_manager.get_prover_accumulator();
        accumulator.borrow_mut().append_dense(
            committed_polys,
            SumcheckId::SpartanOuter,
            r_cycle.to_vec(),
            &committed_poly_claims,
        );

        // Add virtual polynomial evaluations to the accumulator
        // These are needed by the verifier for future sumchecks and are not part of the PCS opening proof
        for (input, eval) in ALL_R1CS_INPUTS.iter().zip(claimed_witness_evals.iter()) {
            // Skip if it's a committed input (already added above)
            if !COMMITTED_R1CS_INPUTS.contains(input) {
                accumulator.borrow_mut().append_virtual(
                    VirtualPolynomial::try_from(input).ok().unwrap(),
                    SumcheckId::SpartanOuter,
                    OpeningPoint::new(r_cycle.to_vec()),
                    *eval,
                );
            }
        }

        Ok(())
    }

    fn stage1_verify(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Result<(), anyhow::Error> {
        let key = self.key.clone();

        let num_rounds_x = key.num_rows_bits();

        let tau: Vec<F> = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector(num_rounds_x);

        // Get the outer sumcheck proof
        let proofs = state_manager.proofs.borrow();
        let proof_data = {
            proofs
                .get(&ProofKeys::Stage1Sumcheck)
                .expect("Outer sumcheck proof not found")
        };

        let outer_sumcheck_proof = match proof_data {
            ProofData::SumcheckProof(proof) => proof,
            _ => panic!("Invalid proof data type"),
        };

        // Get the claims:
        let accumulator = state_manager.get_verifier_accumulator();
        let accumulator_ref = accumulator.borrow();
        let (_, claim_Az) = accumulator_ref
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanAz, SumcheckId::SpartanOuter);
        let (_, claim_Bz) = accumulator_ref
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanBz, SumcheckId::SpartanOuter);
        let (_, claim_Cz) = accumulator_ref
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanCz, SumcheckId::SpartanOuter);
        drop(accumulator_ref);
        let outer_sumcheck_claims = [claim_Az, claim_Bz, claim_Cz];

        // Run the main sumcheck verifier:
        let (claim_outer_final, outer_sumcheck_r_original) = {
            let transcript = &mut state_manager.transcript.borrow_mut();
            match outer_sumcheck_proof.verify(F::zero(), num_rounds_x, 3, transcript) {
                Ok(result) => result,
                Err(_) => return Err(anyhow::anyhow!("Outer sumcheck verification failed")),
            }
        };

        // Outer sumcheck is bound from the top, reverse the challenge
        // TODO(markosg04): Make use of Endianness here?
        let outer_sumcheck_r_reversed: Vec<F> =
            outer_sumcheck_r_original.iter().rev().cloned().collect();
        let opening_point = OpeningPoint::new(outer_sumcheck_r_reversed.clone());

        // Populate the opening points for Az, Bz, Cz claims now that we have outer_sumcheck_r
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::SpartanAz,
            SumcheckId::SpartanOuter,
            opening_point.clone(),
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::SpartanBz,
            SumcheckId::SpartanOuter,
            opening_point.clone(),
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::SpartanCz,
            SumcheckId::SpartanOuter,
            opening_point.clone(),
        );

        let tau_bound_rx = EqPolynomial::mle(&tau, &outer_sumcheck_r_reversed);
        let claim_outer_final_expected = tau_bound_rx * (claim_Az * claim_Bz - claim_Cz);
        if claim_outer_final != claim_outer_final_expected {
            return Err(anyhow::anyhow!("Invalid outer sumcheck claim"));
        }

        ProofTranscript::append_scalars(
            &mut state_manager.transcript.borrow_mut(),
            &outer_sumcheck_claims[..],
        );

        // Add the commitments to verifier accumulator
        let num_cycles = key.num_steps;
        let num_cycles_bits = num_cycles.ilog2() as usize;

        let (r_cycle, _rx_var) = outer_sumcheck_r_reversed.split_at(num_cycles_bits);

        let accumulator = state_manager.get_verifier_accumulator();

        // Only non-virtual (i.e. committed) polynomials' openings are
        // proven using the PCS opening proof, which we add for future opening proof here
        let committed_polys: Vec<_> = COMMITTED_R1CS_INPUTS
            .iter()
            .map(|input| CommittedPolynomial::try_from(input).ok().unwrap())
            .collect();
        accumulator.borrow_mut().append_dense(
            committed_polys,
            SumcheckId::SpartanOuter,
            r_cycle.to_vec(),
        );

        ALL_R1CS_INPUTS.iter().for_each(|input| {
            // Skip if it's a committed input (already added above)
            if !COMMITTED_R1CS_INPUTS.contains(input) {
                accumulator.borrow_mut().append_virtual(
                    VirtualPolynomial::try_from(input).ok().unwrap(),
                    SumcheckId::SpartanOuter,
                    OpeningPoint::new(r_cycle.to_vec()),
                );
            }
        });

        Ok(())
    }

    fn stage2_prover_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        /* Sumcheck 2: Inner sumcheck
            Proves: claim_Az + r * claim_Bz + r^2 * claim_Cz =
                    \sum_y (A_small(rx, y) + r * B_small(rx, y) + r^2 * C_small(rx, y)) * z(y)

            Evaluates the uniform constraint matrices A_small, B_small, C_small at the point
            determined by the outer sumcheck.
        */
        let key = self.key.clone();
        let inner_sumcheck = InnerSumcheck::new_prover(state_manager, key);

        #[cfg(feature = "allocative")]
        print_data_structure_heap_usage("Spartan InnerSumcheck", &inner_sumcheck);

        vec![Box::new(inner_sumcheck)]
    }

    fn stage2_verifier_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        /* Sumcheck 2: Inner sumcheck
           Verifies: claim_Az + r * claim_Bz + r^2 * claim_Cz =
                    (A_small(rx, ry) + r * B_small(rx, ry) + r^2 * C_small(rx, ry)) * z(ry)
        */
        let key = self.key.clone();
        let inner_sumcheck = InnerSumcheck::<F>::new_verifier(state_manager, key);
        vec![Box::new(inner_sumcheck)]
    }

    fn stage3_prover_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        /*  Sumcheck 3: Batched sumcheck for NextUnexpandedPC and NextPC verification
            Proves: NextUnexpandedPC(r_cycle) + r * NextPC(r_cycle) =
                    \sum_t (UnexpandedPC(t) + r * PC(t)) * eq_plus_one(r_cycle, t)

            This batched sumcheck simultaneously proves:
            1. NextUnexpandedPC(r_cycle) = \sum_t UnexpandedPC(t) * eq_plus_one(r_cycle, t)
            2. NextPC(r_cycle) = \sum_t PC(t) * eq_plus_one(r_cycle, t)
        */
        let key = self.key.clone();
        let pc_sumcheck = PCSumcheck::<F>::new_prover(state_manager, key);

        #[cfg(feature = "allocative")]
        print_data_structure_heap_usage("Spartan PCSumcheck", &pc_sumcheck);

        vec![Box::new(pc_sumcheck)]
    }

    fn stage3_verifier_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        /* Sumcheck 3: Batched sumcheck for NextUnexpandedPC and NextPC verification
           Verifies the batched constraint for both NextUnexpandedPC and NextPC
        */
        let key = self.key.clone();
        let pc_sumcheck = PCSumcheck::<F>::new_verifier(state_manager, key);
        vec![Box::new(pc_sumcheck)]
    }
}
