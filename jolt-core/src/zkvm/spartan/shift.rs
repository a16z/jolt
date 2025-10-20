use std::array;
use std::cell::RefCell;
use std::iter::zip;
use std::rc::Rc;
use std::sync::Arc;

use allocative::Allocative;

use crate::field::{ChallengeFieldOps, JoltField};
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::poly::opening_proof::{
    OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator, BIG_ENDIAN,
};
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck::SumcheckInstance;
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use crate::zkvm::dag::state_manager::StateManager;
use crate::zkvm::instruction::CircuitFlags;
use crate::zkvm::r1cs::inputs::generate_shift_sumcheck_witnesses;
use crate::zkvm::r1cs::key::UniformSpartanKey;
use crate::zkvm::witness::VirtualPolynomial;
use rayon::prelude::*;

#[derive(Allocative)]
struct ShiftSumcheckProverState<F: JoltField> {
    unexpanded_pc_poly: MultilinearPolynomial<F>,
    pc_poly: MultilinearPolynomial<F>,
    is_virtual_poly: MultilinearPolynomial<F>,
    is_first_in_sequence_poly: MultilinearPolynomial<F>,
    is_noop_poly: MultilinearPolynomial<F>,
    gruen_eq_r_cycle_plus_one: GruenSplitEqPolynomial<F>,
    gruen_eq_r_product_plus_one: GruenSplitEqPolynomial<F>,
    prev_claim_r_cycle: F,
    prev_claim_r_product: F,
    prev_round_poly_r_cycle: Option<UniPoly<F>>,
    prev_round_poly_r_product: Option<UniPoly<F>>,
}

#[derive(Allocative)]
pub struct ShiftSumcheck<F: JoltField> {
    input_claim: F,
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
        let (preprocessing, trace, _program_io, _final_memory_state) =
            state_manager.get_prover_data();

        // Stream once to generate PC, UnexpandedPC and IsNoop witnesses
        let (unexpanded_pc_poly, pc_poly, is_noop_poly, is_virtual_poly, is_first_in_sequence_poly) =
            generate_shift_sumcheck_witnesses(&preprocessing.shared, trace);

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
        let (_, next_is_virtual_eval) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::NextIsVirtual,
            SumcheckId::SpartanOuter,
        );
        let (_, next_is_first_in_sequence_eval) =
            accumulator.borrow().get_virtual_polynomial_opening(
                VirtualPolynomial::NextIsFirstInSequence,
                SumcheckId::SpartanOuter,
            );

        let (product_sumcheck_r, next_is_noop_eval) =
            accumulator.borrow().get_virtual_polynomial_opening(
                VirtualPolynomial::NextIsNoop,
                SumcheckId::ShouldJumpVirtualization,
            );

        let (r_cycle, _rx_var) = outer_sumcheck_r.split_at(num_cycles_bits);
        let (r_product, _) = product_sumcheck_r.split_at(num_cycles_bits);

        let gruen_eq_r_cycle_plus_one =
            GruenSplitEqPolynomial::new(&r_plus_1(&r_cycle.r), BindingOrder::HighToLow);
        let gruen_eq_r_product_plus_one =
            GruenSplitEqPolynomial::new(&r_plus_1(&r_product.r), BindingOrder::HighToLow);

        let gamma_powers: Vec<F> = state_manager
            .transcript
            .borrow_mut()
            .challenge_scalar_powers(5);

        let prev_claim_r_cycle = next_unexpanded_pc_eval
            + gamma_powers[1] * next_pc_eval
            + gamma_powers[2] * next_is_virtual_eval
            + gamma_powers[3] * next_is_first_in_sequence_eval;

        let prev_claim_r_product = F::one() - next_is_noop_eval;

        let input_claim = prev_claim_r_cycle + gamma_powers[4] * prev_claim_r_product;

        Self {
            input_claim,
            log_T: r_cycle.len(),
            prover_state: Some(ShiftSumcheckProverState {
                unexpanded_pc_poly,
                pc_poly,
                is_virtual_poly,
                is_first_in_sequence_poly,
                is_noop_poly,
                gruen_eq_r_cycle_plus_one,
                gruen_eq_r_product_plus_one,
                prev_claim_r_cycle,
                prev_claim_r_product,
                prev_round_poly_r_cycle: None,
                prev_round_poly_r_product: None,
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

        // Get the Next* evaluations from the accumulator
        let accumulator = state_manager.get_verifier_accumulator();
        let (_, next_pc_eval) = accumulator
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::NextPC, SumcheckId::SpartanOuter);
        let (_, next_unexpanded_pc_eval) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::NextUnexpandedPC,
            SumcheckId::SpartanOuter,
        );
        let (_, next_is_virtual_eval) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::NextIsVirtual,
            SumcheckId::SpartanOuter,
        );
        let (_, next_is_first_in_sequence_eval) =
            accumulator.borrow().get_virtual_polynomial_opening(
                VirtualPolynomial::NextIsFirstInSequence,
                SumcheckId::SpartanOuter,
            );
        let (_, next_is_noop_eval) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::NextIsNoop,
            SumcheckId::ShouldJumpVirtualization,
        );

        let input_claim = [
            next_unexpanded_pc_eval,
            next_pc_eval,
            next_is_virtual_eval,
            next_is_first_in_sequence_eval,
            F::one() - next_is_noop_eval,
        ]
        .iter()
        .zip(gamma_powers.iter())
        .map(|(eval, gamma)| *gamma * eval)
        .sum();
        let log_T = key.num_steps.log_2();

        Self {
            input_claim,
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

    fn input_claim(&self) -> F {
        self.input_claim
    }

    #[tracing::instrument(skip_all, name = "ShiftSumcheck::compute_prover_message")]
    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let state = self.prover_state.as_mut().unwrap();
        let in_evals_r_cycle_plus_1 = state.gruen_eq_r_cycle_plus_one.E_in_current();
        let in_evals_r_product_plus_1 = state.gruen_eq_r_product_plus_one.E_in_current();
        let out_evals_r_cycle_plus_1 = state.gruen_eq_r_cycle_plus_one.E_out_current();
        let out_evals_r_product_plus_1 = state.gruen_eq_r_product_plus_one.E_out_current();

        let out_len = out_evals_r_cycle_plus_1.len();
        let in_len = in_evals_r_cycle_plus_1.len();
        let out_n_vars = out_len.ilog2();

        let [eval_at_0_for_r_cycle, eval_at_0_for_r_product]: [F; 2] = (0..in_len)
            .into_par_iter()
            .map(|j_hi| {
                let mut eval_at_0_for_r_cycle = F::zero();
                let mut eval_at_0_for_r_product = F::zero();

                for j_lo in 0..out_len {
                    let j = j_lo + (j_hi << out_n_vars);

                    // Eval UnexpandedPc(x) at (r', 0, j).
                    let unexpanded_pc_at_0_j = state.unexpanded_pc_poly.get_bound_coeff(j);

                    // Eval Pc(x) at (r', 0, j).
                    let pc_at_0_j = state.pc_poly.get_bound_coeff(j);

                    // Eval IsVirtual(x) at (r', 0, j).
                    let is_virtual_at_0_j = state.is_virtual_poly.get_bound_coeff(j);

                    // Eval IsFirstInSequence(x) at (r', 0, j).
                    let is_first_in_sequence_at_0_j =
                        state.is_first_in_sequence_poly.get_bound_coeff(j);

                    // Eval IsNoOp(x) at (r', 0, j).
                    let is_noop_at_0_j = state.is_noop_poly.get_bound_coeff(j);

                    let [_, g1, g2, g3, _] = self.gamma_powers;
                    eval_at_0_for_r_cycle += out_evals_r_cycle_plus_1[j_lo]
                        * (unexpanded_pc_at_0_j
                            + g1 * pc_at_0_j
                            + g2 * is_virtual_at_0_j
                            + g3 * is_first_in_sequence_at_0_j);
                    eval_at_0_for_r_product +=
                        out_evals_r_product_plus_1[j_lo] * (F::one() - is_noop_at_0_j);
                }

                [
                    in_evals_r_cycle_plus_1[j_hi] * eval_at_0_for_r_cycle,
                    in_evals_r_product_plus_1[j_hi] * eval_at_0_for_r_product,
                ]
            })
            .reduce(|| [F::zero(); 2], |a, b| array::from_fn(|i| a[i] + b[i]));

        let univariate_evals_r_cycle = state
            .gruen_eq_r_cycle_plus_one
            .gruen_evals_deg_2(eval_at_0_for_r_cycle, state.prev_claim_r_cycle);
        let univariate_evals_r_product = state
            .gruen_eq_r_product_plus_one
            .gruen_evals_deg_2(eval_at_0_for_r_product, state.prev_claim_r_product);
        state.prev_round_poly_r_cycle = Some(UniPoly::from_evals_and_hint(
            state.prev_claim_r_cycle,
            &univariate_evals_r_cycle,
        ));
        state.prev_round_poly_r_product = Some(UniPoly::from_evals_and_hint(
            state.prev_claim_r_product,
            &univariate_evals_r_product,
        ));
        zip(univariate_evals_r_cycle, univariate_evals_r_product)
            .map(|(eval_r_cycle, eval_r_product)| {
                eval_r_cycle + self.gamma_powers[4] * eval_r_product
            })
            .collect()
    }

    #[tracing::instrument(skip_all, name = "ShiftSumcheck::bind")]
    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        let ShiftSumcheckProverState {
            unexpanded_pc_poly,
            pc_poly,
            is_virtual_poly,
            is_first_in_sequence_poly,
            is_noop_poly,
            gruen_eq_r_cycle_plus_one,
            gruen_eq_r_product_plus_one,
            prev_claim_r_cycle,
            prev_claim_r_product,
            prev_round_poly_r_cycle,
            prev_round_poly_r_product,
        } = self.prover_state.as_mut().unwrap();
        unexpanded_pc_poly.bind_parallel(r_j, BindingOrder::HighToLow);
        pc_poly.bind_parallel(r_j, BindingOrder::HighToLow);
        is_virtual_poly.bind_parallel(r_j, BindingOrder::HighToLow);
        is_first_in_sequence_poly.bind_parallel(r_j, BindingOrder::HighToLow);
        is_noop_poly.bind_parallel(r_j, BindingOrder::HighToLow);
        gruen_eq_r_cycle_plus_one.bind(r_j);
        gruen_eq_r_product_plus_one.bind(r_j);
        *prev_claim_r_cycle = prev_round_poly_r_cycle.take().unwrap().evaluate(&r_j);
        *prev_claim_r_product = prev_round_poly_r_product.take().unwrap().evaluate(&r_j);
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
            VirtualPolynomial::OpFlags(CircuitFlags::IsNoop),
            SumcheckId::SpartanShift,
        );

        let r_cycle_prime = r_plus_1(&r_cycle);
        let r_product_prime = r_plus_1(&r_product);
        let eq_r_cycle_prime_eval = EqPolynomial::mle(&r_cycle_prime, &r);
        let eq_r_prouct_prime_eval = EqPolynomial::mle(&r_product_prime, &r);

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
            * eq_r_cycle_prime_eval
            + self.gamma_powers[4] * (F::one() - is_noop_claim) * eq_r_prouct_prime_eval
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
        let is_virtual_eval = prover_state.is_virtual_poly.final_sumcheck_claim();
        let is_first_in_sequence_eval = prover_state
            .is_first_in_sequence_poly
            .final_sumcheck_claim();
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
            VirtualPolynomial::OpFlags(CircuitFlags::IsNoop),
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
            VirtualPolynomial::OpFlags(CircuitFlags::IsNoop),
            SumcheckId::SpartanShift,
            opening_point,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

/// Computes r + 1.
///
/// `r` should be in big-endian.
fn r_plus_1<F: JoltField, C: ChallengeFieldOps<F>>(r: &[C]) -> Vec<F> {
    let mut r_prime = Vec::new();
    let mut carry = F::one();

    for &r_i in r.iter().rev() {
        let carry_xor_r_i = r_i + carry - r_i * carry * F::from_u8(2);
        let carry_and_r_i = r_i * carry;
        r_prime.push(carry_xor_r_i);
        carry = carry_and_r_i;
    }

    // Put in big-endian.
    r_prime.reverse();

    r_prime
}
