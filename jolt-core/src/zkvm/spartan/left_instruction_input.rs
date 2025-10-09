use std::{array, cell::RefCell, iter::zip, rc::Rc};

use allocative::Allocative;
use rayon::prelude::*;
use tracer::instruction::Cycle;

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::sumcheck::SumcheckInstance,
    transcripts::Transcript,
    zkvm::{
        dag::state_manager::StateManager,
        instruction::{CircuitFlags, InstructionFlags},
        witness::VirtualPolynomial,
    },
};

/// A sumcheck instance for `LeftInstructionInput(r_cycle) + gamma * LeftInstructionInput(r_product)`.
///
/// Where
///
/// ```text
/// LeftInstructionInput(r) = sum_j eq(r, j) * (
///     LeftInstructionInputIsRs1(j) * Rs1Value(j) +
///     LeftInstructionInputIsPc(j) * UnexpandedPc(j))
/// ```
///
/// Note:
/// - `r_cycle` is the randomness from Spartan outer sumcheck.
/// - `r_product` is the randomness from instruction product sumcheck.
#[derive(Allocative)]
pub struct LeftInstructionInputSumcheck<F: JoltField> {
    /// Randomness from the spartan outer sumcheck.
    r_cycle: OpeningPoint<BIG_ENDIAN, F>,
    input_claim_for_r_cycle: F,
    /// Randomness from instruction product sumcheck.
    r_product: OpeningPoint<BIG_ENDIAN, F>,
    input_claim_for_r_product: F,
    /// Randomness use to combine sumchecks.
    gamma: F,
    prover_state: Option<ProverState<F>>,
}

impl<F: JoltField> LeftInstructionInputSumcheck<F> {
    #[tracing::instrument(skip_all, name = "LeftInstructionInputSumcheck::new_prover")]
    pub fn new_prover(
        state_manager: &mut StateManager<'_, F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let accumulator = state_manager.get_prover_accumulator();
        let (r_cycle, input_claim_for_r_cycle) =
            accumulator.borrow().get_virtual_polynomial_opening(
                VirtualPolynomial::LeftInstructionInput,
                SumcheckId::SpartanOuter,
            );
        let (r_product, input_claim_for_r_product) =
            accumulator.borrow().get_virtual_polynomial_opening(
                VirtualPolynomial::LeftInstructionInput,
                SumcheckId::ProductVirtualization,
            );

        let (_, trace, _, _) = state_manager.get_prover_data();
        let prover_state = ProverState::gen(
            trace,
            &r_cycle,
            input_claim_for_r_cycle,
            &r_product,
            input_claim_for_r_product,
        );

        Self {
            r_cycle,
            input_claim_for_r_cycle,
            r_product,
            input_claim_for_r_product,
            gamma: state_manager.transcript.borrow_mut().challenge_scalar(),
            prover_state: Some(prover_state),
        }
    }

    pub fn new_verifier<T: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, T, PCS>,
    ) -> Self {
        let accumulator = state_manager.get_verifier_accumulator();
        let (r_cycle, input_claim_for_r_cycle) =
            accumulator.borrow().get_virtual_polynomial_opening(
                VirtualPolynomial::LeftInstructionInput,
                SumcheckId::SpartanOuter,
            );
        let (r_product, input_claim_for_r_product) =
            accumulator.borrow().get_virtual_polynomial_opening(
                VirtualPolynomial::LeftInstructionInput,
                SumcheckId::ProductVirtualization,
            );
        Self {
            r_cycle: r_cycle.match_endianness(),
            input_claim_for_r_cycle,
            r_product: r_product.match_endianness(),
            input_claim_for_r_product,
            gamma: state_manager.transcript.borrow_mut().challenge_scalar(),
            prover_state: None,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for LeftInstructionInputSumcheck<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len()
    }

    fn input_claim(&self) -> F {
        self.input_claim_for_r_cycle + self.gamma * self.input_claim_for_r_product
    }

    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let state = self.prover_state.as_mut().unwrap();

        let r_cycle_out_evals = state.eq_r_cycle.E_out_current();
        let r_cycle_in_evals = state.eq_r_cycle.E_in_current();
        let r_product_out_evals = state.eq_r_product.E_out_current();
        let r_product_in_evals = state.eq_r_product.E_in_current();

        let out_len = r_product_out_evals.len();
        let out_n_vars = out_len.ilog2();
        let in_len = r_product_in_evals.len();
        let half_n = state.rs1_value_mle.len() / 2;

        let [eval_at_0_for_r_cycle, eval_at_inf_for_r_cycle, eval_at_0_for_r_product, eval_at_inf_for_r_product] =
            (0..out_len)
                .into_par_iter()
                .map(|j_hi| {
                    let mut eval_at_0_for_r_cycle = F::zero();
                    let mut eval_at_inf_for_r_cycle = F::zero();
                    let mut eval_at_0_for_r_product = F::zero();
                    let mut eval_at_inf_for_r_product = F::zero();

                    for j_lo in 0..in_len {
                        let j = j_lo + (j_hi << out_n_vars);

                        // Evaluations of Rs1Value(x) at (r', {0, 1, inf}, j).
                        let rs1_value_at_0_j = state.rs1_value_mle.get_bound_coeff(j);
                        let rs1_value_at_1_j = state.rs1_value_mle.get_bound_coeff(j + half_n);
                        let rs1_value_at_inf_j = rs1_value_at_1_j - rs1_value_at_0_j;

                        // Evaluations of LeftInstructionInputIsRs1(x) at (r', {0, 1, inf}, j).
                        let is_rs1_value_at_0_j = state.is_rs1_value_mle.get_bound_coeff(j);
                        let is_rs1_value_at_1_j =
                            state.is_rs1_value_mle.get_bound_coeff(j + half_n);
                        let is_rs1_value_at_inf_j = is_rs1_value_at_1_j - is_rs1_value_at_0_j;

                        // Evaluations of UnexpandedPc(x) at (r', {0, 1, inf}, j).
                        let unexpanded_pc_at_0_j = state.unexpanded_pc_mle.get_bound_coeff(j);
                        let unexpanded_pc_at_1_j =
                            state.unexpanded_pc_mle.get_bound_coeff(j + half_n);
                        let unexpanded_pc_at_inf_j = unexpanded_pc_at_1_j - unexpanded_pc_at_0_j;

                        // Evaluations of LeftInstructionInputIsPc(x) at (r', {0, 1, inf}, j).
                        let is_pc_at_0_j = state.is_pc_mle.get_bound_coeff(j);
                        let is_pc_at_1_j = state.is_pc_mle.get_bound_coeff(j + half_n);
                        let is_pc_at_inf_j = is_pc_at_1_j - is_pc_at_0_j;

                        // Eval left(x) = LeftInstructionInputIsRs1(x) * Rs1Value(x) + LeftInstructionInputIsPc(x) * UnexpandedPc(x)
                        // at (r', {0, inf}, j).
                        let left_at_0_j = is_rs1_value_at_0_j * rs1_value_at_0_j
                            + is_pc_at_0_j * unexpanded_pc_at_0_j;
                        let left_at_inf_j = is_rs1_value_at_inf_j * rs1_value_at_inf_j
                            + is_pc_at_inf_j * unexpanded_pc_at_inf_j;

                        eval_at_0_for_r_cycle += r_cycle_in_evals[j_lo] * left_at_0_j;
                        eval_at_inf_for_r_cycle += r_cycle_in_evals[j_lo] * left_at_inf_j;
                        eval_at_0_for_r_product += r_product_in_evals[j_lo] * left_at_0_j;
                        eval_at_inf_for_r_product += r_product_in_evals[j_lo] * left_at_inf_j;
                    }

                    [
                        r_cycle_out_evals[j_hi] * eval_at_0_for_r_cycle,
                        r_cycle_out_evals[j_hi] * eval_at_inf_for_r_cycle,
                        r_product_out_evals[j_hi] * eval_at_0_for_r_product,
                        r_product_out_evals[j_hi] * eval_at_inf_for_r_product,
                    ]
                })
                .reduce(|| [F::zero(); 4], |a, b| array::from_fn(|i| a[i] + b[i]));

        let evals_for_r_cycle = state.eq_r_cycle.gruen_evals_deg_3(
            eval_at_0_for_r_cycle,
            eval_at_inf_for_r_cycle,
            state.prev_claim_for_r_cycle,
        );

        let evals_for_r_product = state.eq_r_product.gruen_evals_deg_3(
            eval_at_0_for_r_product,
            eval_at_inf_for_r_product,
            state.prev_claim_for_r_product,
        );

        state.prev_round_poly_for_r_cycle = Some(poly_from_evals_and_hint(
            state.prev_claim_for_r_cycle,
            &evals_for_r_cycle,
        ));

        state.prev_round_poly_for_r_product = Some(poly_from_evals_and_hint(
            state.prev_claim_for_r_product,
            &evals_for_r_product,
        ));

        zip(evals_for_r_cycle, evals_for_r_product)
            .map(|(eval_for_r_cycle, eval_for_r_product)| {
                eval_for_r_cycle + self.gamma * eval_for_r_product
            })
            .collect()
    }

    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        let prover_state = self.prover_state.as_mut().unwrap();
        prover_state
            .is_rs1_value_mle
            .bind_parallel(r_j, BindingOrder::HighToLow);
        prover_state
            .rs1_value_mle
            .bind(r_j, BindingOrder::HighToLow);
        prover_state
            .is_pc_mle
            .bind_parallel(r_j, BindingOrder::HighToLow);
        prover_state
            .unexpanded_pc_mle
            .bind(r_j, BindingOrder::HighToLow);
        prover_state.eq_r_cycle.bind(r_j);
        prover_state.eq_r_product.bind(r_j);
        let round_poly_for_r_cycle = prover_state.prev_round_poly_for_r_cycle.take().unwrap();
        prover_state.prev_claim_for_r_cycle = round_poly_for_r_cycle.evaluate(&r_j);
        let round_poly_for_r_product = prover_state.prev_round_poly_for_r_product.take().unwrap();
        prover_state.prev_claim_for_r_product = round_poly_for_r_product.evaluate(&r_j);
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F::Challenge],
    ) -> F {
        let r = OpeningPoint::<BIG_ENDIAN, F>::new(r.to_vec());
        let eq_eval_at_r_cycle = EqPolynomial::mle_endian(&r, &self.r_cycle);
        let eq_eval_at_r_product = EqPolynomial::mle_endian(&r, &self.r_product);

        let accumulator = accumulator.as_ref().unwrap().borrow();
        let (_, rs1_value_eval) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Value,
            SumcheckId::LeftInstructionInputVirtualization,
        );
        let (_, is_rs1_value_eval) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::LeftOperandIsRs1Value),
            SumcheckId::LeftInstructionInputVirtualization,
        );
        let (_, unexpanded_pc_eval) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::LeftInstructionInputVirtualization,
        );
        let (_, is_pc_eval) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::LeftOperandIsPC),
            SumcheckId::LeftInstructionInputVirtualization,
        );

        (eq_eval_at_r_cycle + self.gamma * eq_eval_at_r_product)
            * (is_rs1_value_eval * rs1_value_eval + is_pc_eval * unexpanded_pc_eval)
    }

    fn normalize_opening_point(
        &self,
        sumcheck_challenges: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<BIG_ENDIAN, F>::new(sumcheck_challenges.to_vec())
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let state = self.prover_state.as_ref().unwrap();
        let mut accumulator = accumulator.borrow_mut();
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::LeftOperandIsRs1Value),
            SumcheckId::LeftInstructionInputVirtualization,
            opening_point.clone(),
            state.is_rs1_value_mle.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::LeftOperandIsPC),
            SumcheckId::LeftInstructionInputVirtualization,
            opening_point.clone(),
            state.is_pc_mle.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Rs1Value,
            SumcheckId::LeftInstructionInputVirtualization,
            opening_point.clone(),
            state.rs1_value_mle.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::LeftInstructionInputVirtualization,
            opening_point,
            state.unexpanded_pc_mle.final_sumcheck_claim(),
        );
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let mut accumulator = accumulator.borrow_mut();
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::LeftOperandIsRs1Value),
            SumcheckId::LeftInstructionInputVirtualization,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::LeftOperandIsPC),
            SumcheckId::LeftInstructionInputVirtualization,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Rs1Value,
            SumcheckId::LeftInstructionInputVirtualization,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::LeftInstructionInputVirtualization,
            opening_point,
        );
    }
}

#[derive(Allocative)]
struct ProverState<F: JoltField> {
    rs1_value_mle: MultilinearPolynomial<F>,
    is_rs1_value_mle: MultilinearPolynomial<F>,
    unexpanded_pc_mle: MultilinearPolynomial<F>,
    is_pc_mle: MultilinearPolynomial<F>,
    eq_r_cycle: GruenSplitEqPolynomial<F>,
    eq_r_product: GruenSplitEqPolynomial<F>,
    prev_claim_for_r_cycle: F,
    prev_claim_for_r_product: F,
    prev_round_poly_for_r_cycle: Option<UniPoly<F>>,
    prev_round_poly_for_r_product: Option<UniPoly<F>>,
}

impl<F: JoltField> ProverState<F> {
    fn gen(
        trace: &[Cycle],
        r_cycle: &OpeningPoint<BIG_ENDIAN, F>,
        input_claim_for_r_cycle: F,
        r_product: &OpeningPoint<BIG_ENDIAN, F>,
        input_claim_for_r_product: F,
    ) -> Self {
        // Generate MLEs.
        let mut rs1_value_mle: Vec<u64> = vec![0; trace.len()];
        let mut is_rs1_value_mle: Vec<u8> = vec![0; trace.len()];
        let mut unexpanded_pc_mle: Vec<u64> = vec![0; trace.len()];
        let mut is_pc_mle: Vec<u8> = vec![0; trace.len()];
        (
            &mut rs1_value_mle,
            &mut is_rs1_value_mle,
            &mut unexpanded_pc_mle,
            &mut is_pc_mle,
            trace,
        )
            .into_par_iter()
            .for_each(|(rs1_value, is_rs1_value, unexpanded_pc, is_pc, cycle)| {
                let instruction = cycle.instruction();
                let flags = instruction.circuit_flags();
                *is_rs1_value = flags[CircuitFlags::LeftOperandIsRs1Value].into();
                *is_pc = flags[CircuitFlags::LeftOperandIsPC].into();
                *rs1_value = cycle.rs1_read().1;
                *unexpanded_pc = instruction.normalize().address as u64;
            });

        let eq_r_cycle = GruenSplitEqPolynomial::new(&r_cycle.r, BindingOrder::HighToLow);
        let eq_r_product = GruenSplitEqPolynomial::new(&r_product.r, BindingOrder::HighToLow);

        Self {
            rs1_value_mle: rs1_value_mle.into(),
            is_rs1_value_mle: is_rs1_value_mle.into(),
            unexpanded_pc_mle: unexpanded_pc_mle.into(),
            is_pc_mle: is_pc_mle.into(),
            eq_r_cycle,
            eq_r_product,
            prev_claim_for_r_cycle: input_claim_for_r_cycle,
            prev_claim_for_r_product: input_claim_for_r_product,
            prev_round_poly_for_r_cycle: None,
            prev_round_poly_for_r_product: None,
        }
    }
}

/// Interpolate a polynomial `p(x)` from its evaluations at the points `0, 2, ..., degree-1` and a `hint = p(0) + p(1)`.
pub fn poly_from_evals_and_hint<F: JoltField>(hint: F, evals: &[F]) -> UniPoly<F> {
    let mut evals = evals.to_vec();
    let eval_at_1 = hint - evals[0];
    evals.insert(1, eval_at_1);
    UniPoly::from_evals(&evals)
}
