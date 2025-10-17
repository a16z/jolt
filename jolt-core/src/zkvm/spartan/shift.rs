use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use allocative::Allocative;

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::eq_poly::EqPlusOnePolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::poly::opening_proof::{
    OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator, BIG_ENDIAN,
};
use crate::subprotocols::sumcheck::SumcheckInstance;
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use crate::zkvm::dag::state_manager::StateManager;
use crate::zkvm::instruction::InstructionFlags;
use crate::zkvm::r1cs::inputs::generate_pc_noop_witnesses;
use crate::zkvm::r1cs::key::UniformSpartanKey;
use crate::zkvm::witness::VirtualPolynomial;
use rayon::prelude::*;

#[derive(Allocative)]
struct ShiftSumcheckProverState<F: JoltField> {
    unexpanded_pc_poly: MultilinearPolynomial<F>,
    pc_poly: MultilinearPolynomial<F>,
    is_noop_poly: MultilinearPolynomial<F>,
    eq_plus_one_r_cycle: MultilinearPolynomial<F>,
    eq_plus_one_r_product: MultilinearPolynomial<F>,
}

#[derive(Allocative)]
pub struct ShiftSumcheck<F: JoltField> {
    input_claim: F,
    gamma: F,
    gamma_squared: F,
    log_T: usize,
    prover_state: Option<ShiftSumcheckProverState<F>>,
}

impl<F: JoltField> ShiftSumcheck<F> {
    #[tracing::instrument(skip_all, name = "ShiftSumcheck::new_prover")]
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        key: Arc<UniformSpartanKey<F>>,
    ) -> Self {
        let (preprocessing, trace, _program_io, _final_memory_state) =
            state_manager.get_prover_data();

        // Stream once to generate PC, UnexpandedPC and IsNoop witnesses
        let (unexpanded_pc_poly, pc_poly, is_noop_poly) =
            generate_pc_noop_witnesses(&preprocessing.shared, trace);

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

        let (product_sumcheck_r, next_is_noop_eval) =
            accumulator.borrow().get_virtual_polynomial_opening(
                VirtualPolynomial::NextIsNoop,
                SumcheckId::ShouldJumpVirtualization,
            );

        let (r_cycle, _rx_var) = outer_sumcheck_r.split_at(num_cycles_bits);
        let (r_product, _) = product_sumcheck_r.split_at(num_cycles_bits);

        let (_, eq_plus_one_r_cycle) = EqPlusOnePolynomial::<F>::evals(&r_cycle.r, None);
        let (_, eq_plus_one_r_product) = EqPlusOnePolynomial::<F>::evals(&r_product.r, None);

        let gamma: F = state_manager.transcript.borrow_mut().challenge_scalar();
        let gamma_squared = gamma.square();

        let input_claim = next_unexpanded_pc_eval
            + gamma * next_pc_eval
            + gamma_squared * (F::one() - next_is_noop_eval);

        Self {
            input_claim,
            log_T: r_cycle.len(),
            prover_state: Some(ShiftSumcheckProverState {
                unexpanded_pc_poly,
                pc_poly,
                is_noop_poly,
                eq_plus_one_r_cycle: MultilinearPolynomial::from(eq_plus_one_r_cycle),
                eq_plus_one_r_product: MultilinearPolynomial::from(eq_plus_one_r_product),
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
            SumcheckId::ShouldJumpVirtualization,
        );

        let input_claim = next_unexpanded_pc_eval
            + gamma * next_pc_eval
            + gamma_squared * (F::one() - next_is_noop_eval);
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

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for ShiftSumcheck<F> {
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.log_T
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    #[tracing::instrument(skip_all, name = "ShiftSumcheck::compute_prover_message")]
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
                let eq_r_cycle_evals = prover_state
                    .eq_plus_one_r_cycle
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let eq_r_product_evals = prover_state
                    .eq_plus_one_r_product
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let is_noop_evals = prover_state
                    .is_noop_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);

                [
                    (unexpanded_pc_evals[0] + self.gamma * pc_evals[0]) * eq_r_cycle_evals[0]
                        + self.gamma_squared
                            * (F::one() - is_noop_evals[0])
                            * eq_r_product_evals[0],
                    (unexpanded_pc_evals[1] + self.gamma * pc_evals[1]) * eq_r_cycle_evals[1]
                        + self.gamma_squared
                            * (F::one() - is_noop_evals[1])
                            * eq_r_product_evals[1],
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

    #[tracing::instrument(skip_all, name = "ShiftSumcheck::bind")]
    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
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
                    .eq_plus_one_r_cycle
                    .bind_parallel(r_j, BindingOrder::HighToLow)
            });
            s.spawn(|_| {
                prover_state
                    .eq_plus_one_r_product
                    .bind_parallel(r_j, BindingOrder::HighToLow)
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
            SumcheckId::ShouldJumpVirtualization,
        );
        let product_sumcheck_r = &product_sumcheck_opening.r;
        let (r_product, _) = product_sumcheck_r.split_at(num_cycles_bits);

        // Get the shift evaluations from the accumulator
        let (_, unexpanded_pc_eval_at_shift_r) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
        );
        let (_, pc_eval_at_shift_r) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanShift);
        let (_, is_noop_eval_at_shift_r) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionFlags(InstructionFlags::IsNoop),
            SumcheckId::SpartanShift,
        );

        let eq_plus_one_r_cycle_at_shift =
            EqPlusOnePolynomial::<F>::new(r_cycle.to_vec()).evaluate(r);
        let eq_plus_one_r_product_at_shift =
            EqPlusOnePolynomial::<F>::new(r_product.to_vec()).evaluate(r);

        (unexpanded_pc_eval_at_shift_r + self.gamma * pc_eval_at_shift_r)
            * eq_plus_one_r_cycle_at_shift
            + self.gamma_squared
                * (F::one() - is_noop_eval_at_shift_r)
                * eq_plus_one_r_product_at_shift
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

        let unexpanded_pc_eval = prover_state.unexpanded_pc_poly.final_sumcheck_claim();
        let pc_eval = prover_state.pc_poly.final_sumcheck_claim();
        let is_noop_eval = prover_state.is_noop_poly.final_sumcheck_claim();

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
        OpeningPoint::new(opening_point.to_vec())
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
