use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use allocative::Allocative;
use tracer::instruction::Cycle;

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::eq_poly::EqPlusOnePolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
    VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::subprotocols::sumcheck::SumcheckInstance;
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::dag::state_manager::StateManager;
use crate::zkvm::instruction::{CircuitFlags, InstructionFlags};
use crate::zkvm::r1cs::inputs::{
    evaluate_shift_sumcheck_witnesses, generate_shift_sumcheck_witnesses,
};
use crate::zkvm::r1cs::key::UniformSpartanKey;
use crate::zkvm::witness::VirtualPolynomial;
use rayon::prelude::*;

// Spartan PC sumcheck
//
// Proves the batched identity over cycles j:
//   Σ_j EqPlusOne(r_cycle, j) ⋅ (UnexpandedPC_shift(j) + γ·PC_shift(j) + γ²·IsNoop_shift(j))
//   = NextUnexpandedPC(r_cycle) + γ·NextPC(r_cycle) + γ²·NextIsNoop(r_cycle),
//
// where:
// - EqPlusOne(r_cycle, j): MLE of the function that,
//     on (i,j) returns 1 iff i = j + 1; no wrap-around at j = 2^{log T} − 1
// - UnexpandedPC_shift(j), PC_shift(j), IsNoop_shift(j):
//     SpartanShift MLEs encoding f(j+1) aligned at cycle j
// - NextUnexpandedPC(r_cycle), NextPC(r_cycle), NextIsNoop(r_cycle)
//     are claims from Spartan outer sumcheck
// - γ: batching scalar drawn from the transcript

#[derive(Allocative)]
struct ShiftSumcheckProverState<F: JoltField> {
    combined_witness_poly: MultilinearPolynomial<F>,
    is_noop_poly: MultilinearPolynomial<F>,
    eq_plus_one_r_cycle: MultilinearPolynomial<F>,
    eq_plus_one_r_product: MultilinearPolynomial<F>,
    #[allocative(skip)]
    trace: Arc<Vec<Cycle>>,
    #[allocative(skip)]
    bytecode_preprocessing: BytecodePreprocessing,
}

#[derive(Allocative)]
pub struct ShiftSumcheck<F: JoltField> {
    gamma_powers: [F; 5],
    log_T: usize,
    prover_state: Option<ShiftSumcheckProverState<F>>,
}

impl<F: JoltField> ShiftSumcheck<F> {
    #[tracing::instrument(skip_all, name = "ShiftSumcheck::new_prover")]
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        key: Arc<UniformSpartanKey<F>>,
    ) -> Self {
        let (preprocessing, _, _program_io, _final_memory_state) = state_manager.get_prover_data();
        let trace = state_manager.get_trace_arc();

        let num_cycles = key.num_steps;
        let num_cycles_bits = num_cycles.ilog2() as usize;

        // Get opening_point from accumulator
        let (outer_sumcheck_r, _) = state_manager
            .get_virtual_polynomial_opening(VirtualPolynomial::NextPC, SumcheckId::SpartanOuter);

        let (product_sumcheck_r, _) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::NextIsNoop,
            SumcheckId::ProductVirtualization,
        );

        let (r_cycle, _rx_var) = outer_sumcheck_r.split_at(num_cycles_bits);
        let (r_product, _) = product_sumcheck_r.split_at(num_cycles_bits);

        let (_, eq_plus_one_r_cycle) = EqPlusOnePolynomial::<F>::evals(&r_cycle.r, None);
        let (_, eq_plus_one_r_product) = EqPlusOnePolynomial::<F>::evals(&r_product.r, None);

        let gamma_powers: Vec<F> = state_manager
            .transcript
            .borrow_mut()
            .challenge_scalar_powers(5);

        // Stream once to generate PC, UnexpandedPC and IsNoop witnesses
        let (combined_witness_poly, is_noop_poly) =
            generate_shift_sumcheck_witnesses(&preprocessing.shared, &trace, &gamma_powers);

        Self {
            log_T: r_cycle.len(),
            prover_state: Some(ShiftSumcheckProverState {
                combined_witness_poly,
                is_noop_poly,
                eq_plus_one_r_cycle: MultilinearPolynomial::from(eq_plus_one_r_cycle),
                eq_plus_one_r_product: MultilinearPolynomial::from(eq_plus_one_r_product),
                bytecode_preprocessing: preprocessing.shared.bytecode.clone(), // HACK
                trace,
            }),
            gamma_powers: gamma_powers.try_into().unwrap(),
        }
    }

    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        key: Arc<UniformSpartanKey<F>>,
    ) -> Self {
        // Get batching challenge for combining claims
        let gamma_powers: Vec<F> = state_manager
            .transcript
            .borrow_mut()
            .challenge_scalar_powers(5);
        let log_T = key.num_steps.log_2();

        Self {
            prover_state: None,
            log_T,
            gamma_powers: gamma_powers.try_into().unwrap(),
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for ShiftSumcheck<F> {
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.log_T
    }

    fn input_claim(&self, acc: Option<&RefCell<dyn OpeningAccumulator<F>>>) -> F {
        let acc = acc.unwrap().borrow();
        let (_, next_pc_eval) =
            acc.get_virtual_polynomial_opening(VirtualPolynomial::NextPC, SumcheckId::SpartanOuter);
        let (_, next_unexpanded_pc_eval) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::NextUnexpandedPC,
            SumcheckId::SpartanOuter,
        );
        let (_, next_is_virtual_eval) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::NextIsVirtual,
            SumcheckId::SpartanOuter,
        );
        let (_, next_is_first_in_sequence_eval) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::NextIsFirstInSequence,
            SumcheckId::SpartanOuter,
        );
        let (_, next_is_noop_eval) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::NextIsNoop,
            SumcheckId::ProductVirtualization,
        );
        [
            next_unexpanded_pc_eval,
            next_pc_eval,
            next_is_virtual_eval,
            next_is_first_in_sequence_eval,
            F::one() - next_is_noop_eval,
        ]
        .iter()
        .zip(self.gamma_powers.iter())
        .map(|(eval, gamma)| *gamma * eval)
        .sum()
    }

    #[tracing::instrument(skip_all, name = "ShiftSumcheck::compute_prover_message")]
    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        const DEGREE: usize = 2;

        let univariate_poly_evals: [F; DEGREE] = (0..prover_state.combined_witness_poly.len() / 2)
            .into_par_iter()
            .map(|i| {
                let combined_witness_evals = prover_state
                    .combined_witness_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let eq_r_cycle_evals = prover_state
                    .eq_plus_one_r_cycle
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let eq_r_product_evals = prover_state
                    .eq_plus_one_r_product
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let is_noop_evals = prover_state
                    .is_noop_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                std::array::from_fn(|i| {
                    combined_witness_evals[i] * eq_r_cycle_evals[i]
                        + self.gamma_powers[4]
                            * (F::one() - is_noop_evals[i])
                            * eq_r_product_evals[i]
                })
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

    #[tracing::instrument(skip_all, name = "ShiftSumcheck::bind")]
    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        rayon::scope(|s| {
            s.spawn(|_| {
                prover_state
                    .combined_witness_poly
                    .bind_parallel(r_j, BindingOrder::LowToHigh)
            });
            s.spawn(|_| {
                prover_state
                    .is_noop_poly
                    .bind_parallel(r_j, BindingOrder::LowToHigh)
            });
            s.spawn(|_| {
                prover_state
                    .eq_plus_one_r_cycle
                    .bind_parallel(r_j, BindingOrder::LowToHigh)
            });
            s.spawn(|_| {
                prover_state
                    .eq_plus_one_r_product
                    .bind_parallel(r_j, BindingOrder::LowToHigh)
            });
        });
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F::Challenge],
    ) -> F {
        let accumulator = accumulator.as_ref().unwrap().borrow();

        // Get r_cycle from the SpartanOuter sumcheck opening point
        let (outer_sumcheck_opening, _) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::NextPC, SumcheckId::SpartanOuter);
        let outer_sumcheck_r = &outer_sumcheck_opening.r;
        let num_cycles_bits = self.log_T;
        let (r_cycle, _) = outer_sumcheck_r.split_at(num_cycles_bits);

        let (product_sumcheck_opening, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NextIsNoop,
            SumcheckId::ProductVirtualization,
        );
        let product_sumcheck_r = &product_sumcheck_opening.r;
        let (r_product, _) = product_sumcheck_r.split_at(num_cycles_bits);

        // Get the shift evaluations from the accumulator
        let (_, unexpanded_pc_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
        );
        let (_, pc_claim) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanShift);
        let (_, is_virtual_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::VirtualInstruction),
            SumcheckId::SpartanShift,
        );
        let (_, is_first_in_sequence_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::IsFirstInSequence),
            SumcheckId::SpartanShift,
        );
        let (_, is_noop_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionFlags(InstructionFlags::IsNoop),
            SumcheckId::SpartanShift,
        );

        let r = <Self as SumcheckInstance<F, T>>::normalize_opening_point(self, r);
        let eq_plus_one_r_cycle_at_shift =
            EqPlusOnePolynomial::<F>::new(r_cycle.to_vec()).evaluate(&r.r);
        let eq_plus_one_r_product_at_shift =
            EqPlusOnePolynomial::<F>::new(r_product.to_vec()).evaluate(&r.r);

        [
            unexpanded_pc_claim,
            pc_claim,
            is_virtual_claim,
            is_first_in_sequence_claim,
        ]
        .iter()
        .zip(self.gamma_powers.iter())
        .map(|(eval, gamma)| *gamma * eval)
        .sum::<F>()
            * eq_plus_one_r_cycle_at_shift
            + self.gamma_powers[4] * (F::one() - is_noop_claim) * eq_plus_one_r_product_at_shift
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        let is_noop_eval = prover_state.is_noop_poly.final_sumcheck_claim();

        let [unexpanded_pc_eval, pc_eval, is_virtual_eval, is_first_in_sequence_eval] =
            evaluate_shift_sumcheck_witnesses(
                &prover_state.bytecode_preprocessing,
                &prover_state.trace,
                &opening_point,
            );

        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
            opening_point.clone(),
            unexpanded_pc_eval,
        );
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::PC,
            SumcheckId::SpartanShift,
            opening_point.clone(),
            pc_eval,
        );
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::VirtualInstruction),
            SumcheckId::SpartanShift,
            opening_point.clone(),
            is_virtual_eval,
        );
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::IsFirstInSequence),
            SumcheckId::SpartanShift,
            opening_point.clone(),
            is_first_in_sequence_eval,
        );
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::InstructionFlags(InstructionFlags::IsNoop),
            SumcheckId::SpartanShift,
            opening_point,
            is_noop_eval,
        );
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(opening_point.to_vec()).match_endianness()
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
            opening_point.clone(),
        );
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::PC,
            SumcheckId::SpartanShift,
            opening_point.clone(),
        );
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::VirtualInstruction),
            SumcheckId::SpartanShift,
            opening_point.clone(),
        );
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::IsFirstInSequence),
            SumcheckId::SpartanShift,
            opening_point.clone(),
        );
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::InstructionFlags(InstructionFlags::IsNoop),
            SumcheckId::SpartanShift,
            opening_point,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}
