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

use super::left_instruction_input::poly_from_evals_and_hint;

/// A sumcheck instance for `RightInstructionInput(r_cycle) + gamma * RightInstructionInput(r_product)`.
///
/// Where
///
/// ```text
/// RightInstructionInput(r) = sum_j eq(r, j) * (
///     RightInstructionInputIsRs2(j) * Rs2Value(j) +
///     RightInstructionInputIsImm(j) * Imm(j))
/// ```
///
/// Note:
/// - `r_cycle` is the randomness from Spartan outer sumcheck.
/// - `r_product` is the randomness from instruction product sumcheck.
#[derive(Allocative)]
pub struct RightInstructionInputSumcheck<F: JoltField> {
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

impl<F: JoltField> RightInstructionInputSumcheck<F> {
    #[tracing::instrument(skip_all, name = "RightInstructionInputSumcheck::new_prover")]
    pub fn new_prover(
        state_manager: &mut StateManager<'_, F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let accumulator = state_manager.get_prover_accumulator();
        let (r_cycle, input_claim_for_r_cycle) =
            accumulator.borrow().get_virtual_polynomial_opening(
                VirtualPolynomial::RightInstructionInput,
                SumcheckId::SpartanOuter,
            );
        let (r_product, input_claim_for_r_product) =
            accumulator.borrow().get_virtual_polynomial_opening(
                VirtualPolynomial::RightInstructionInput,
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
                VirtualPolynomial::RightInstructionInput,
                SumcheckId::SpartanOuter,
            );
        let (r_product, input_claim_for_r_product) =
            accumulator.borrow().get_virtual_polynomial_opening(
                VirtualPolynomial::RightInstructionInput,
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

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for RightInstructionInputSumcheck<F> {
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
        let half_n = state.imm_mle.len() / 2;

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

                        // Evaluations of Rs2Value(x) at (r', {0, 1, inf}, j).
                        let rs2_value_at_0_j = state.rs2_value_mle.get_bound_coeff(j);
                        let rs2_value_at_1_j = state.rs2_value_mle.get_bound_coeff(j + half_n);
                        let rs2_value_at_inf_j = rs2_value_at_1_j - rs2_value_at_0_j;

                        // Evaluations of RightInstructionInputIsRs2(x) at (r', {0, 1, inf}, j).
                        let is_rs2_value_at_0_j = state.is_rs2_value_mle.get_bound_coeff(j);
                        let is_rs2_value_at_1_j =
                            state.is_rs2_value_mle.get_bound_coeff(j + half_n);
                        let is_rs2_value_at_inf_j = is_rs2_value_at_1_j - is_rs2_value_at_0_j;

                        // Evaluations of Imm(x) at (r', {0, 1, inf}, j).
                        let imm_at_0_j = state.imm_mle.get_bound_coeff(j);
                        let imm_at_1_j = state.imm_mle.get_bound_coeff(j + half_n);
                        let imm_at_inf_j = imm_at_1_j - imm_at_0_j;

                        // Evaluations of RightInstructionInputIsImm(x) at (r', {0, 1, inf}, j).
                        let is_imm_at_0_j = state.is_imm_mle.get_bound_coeff(j);
                        let is_imm_at_1_j = state.is_imm_mle.get_bound_coeff(j + half_n);
                        let is_imm_at_inf_j = is_imm_at_1_j - is_imm_at_0_j;

                        // Eval right(x) = RightInstructionInputIsRs2(x) * Rs2Value(x) + RightInstructionInputIsImm(x) * Imm(x)
                        // at (r', {0, inf}, j).
                        let right_at_0_j =
                            is_rs2_value_at_0_j * rs2_value_at_0_j + is_imm_at_0_j * imm_at_0_j;
                        let right_at_inf_j = is_rs2_value_at_inf_j * rs2_value_at_inf_j
                            + is_imm_at_inf_j * imm_at_inf_j;

                        eval_at_0_for_r_cycle += r_cycle_in_evals[j_lo] * right_at_0_j;
                        eval_at_inf_for_r_cycle += r_cycle_in_evals[j_lo] * right_at_inf_j;
                        eval_at_0_for_r_product += r_product_in_evals[j_lo] * right_at_0_j;
                        eval_at_inf_for_r_product += r_product_in_evals[j_lo] * right_at_inf_j;
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
            .is_rs2_value_mle
            .bind_parallel(r_j, BindingOrder::HighToLow);
        prover_state
            .rs2_value_mle
            .bind(r_j, BindingOrder::HighToLow);
        prover_state
            .is_imm_mle
            .bind_parallel(r_j, BindingOrder::HighToLow);
        prover_state.imm_mle.bind(r_j, BindingOrder::HighToLow);
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
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r = OpeningPoint::<BIG_ENDIAN, F>::new(sumcheck_challenges.to_vec());
        let eq_eval_at_r_cycle = EqPolynomial::mle_endian(&r, &self.r_cycle);
        let eq_eval_at_r_product = EqPolynomial::mle_endian(&r, &self.r_product);

        let accumulator = accumulator.as_ref().unwrap().borrow();
        let (_, rs2_value_eval) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs2Value,
            SumcheckId::RightInstructionInputVirtualization,
        );
        let (_, is_rs2_value_eval) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::RightOperandIsRs2Value),
            SumcheckId::RightInstructionInputVirtualization,
        );
        let (_, imm_eval) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Imm,
            SumcheckId::RightInstructionInputVirtualization,
        );
        let (_, is_imm_eval) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::RightOperandIsImm),
            SumcheckId::RightInstructionInputVirtualization,
        );

        (eq_eval_at_r_cycle + self.gamma * eq_eval_at_r_product)
            * (is_rs2_value_eval * rs2_value_eval + is_imm_eval * imm_eval)
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
            VirtualPolynomial::OpFlags(CircuitFlags::RightOperandIsRs2Value),
            SumcheckId::RightInstructionInputVirtualization,
            opening_point.clone(),
            state.is_rs2_value_mle.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::RightOperandIsImm),
            SumcheckId::RightInstructionInputVirtualization,
            opening_point.clone(),
            state.is_imm_mle.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Rs2Value,
            SumcheckId::RightInstructionInputVirtualization,
            opening_point.clone(),
            state.rs2_value_mle.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Imm,
            SumcheckId::RightInstructionInputVirtualization,
            opening_point,
            state.imm_mle.final_sumcheck_claim(),
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
            VirtualPolynomial::OpFlags(CircuitFlags::RightOperandIsRs2Value),
            SumcheckId::RightInstructionInputVirtualization,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::RightOperandIsImm),
            SumcheckId::RightInstructionInputVirtualization,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Rs2Value,
            SumcheckId::RightInstructionInputVirtualization,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Imm,
            SumcheckId::RightInstructionInputVirtualization,
            opening_point,
        );
    }
}

#[derive(Allocative)]
struct ProverState<F: JoltField> {
    rs2_value_mle: MultilinearPolynomial<F>,
    is_rs2_value_mle: MultilinearPolynomial<F>,
    imm_mle: MultilinearPolynomial<F>,
    is_imm_mle: MultilinearPolynomial<F>,
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
        let mut rs2_value_mle: Vec<u64> = vec![0; trace.len()];
        let mut is_rs2_value_mle: Vec<u8> = vec![0; trace.len()];
        let mut imm_mle: Vec<i128> = vec![0; trace.len()];
        let mut is_imm_mle: Vec<u8> = vec![0; trace.len()];
        (
            &mut rs2_value_mle,
            &mut is_rs2_value_mle,
            &mut imm_mle,
            &mut is_imm_mle,
            trace,
        )
            .into_par_iter()
            .for_each(|(rs2_value, is_rs2_value, imm, is_imm, cycle)| {
                let instruction = cycle.instruction();
                let flags = instruction.circuit_flags();
                *is_rs2_value = flags[CircuitFlags::RightOperandIsRs2Value].into();
                *is_imm = flags[CircuitFlags::RightOperandIsImm].into();
                *rs2_value = cycle.rs2_read().1;
                *imm = instruction.normalize().operands.imm;
            });

        let eq_r_cycle = GruenSplitEqPolynomial::new(&r_cycle.r, BindingOrder::HighToLow);
        let eq_r_product = GruenSplitEqPolynomial::new(&r_product.r, BindingOrder::HighToLow);

        Self {
            rs2_value_mle: rs2_value_mle.into(),
            is_rs2_value_mle: is_rs2_value_mle.into(),
            imm_mle: imm_mle.into(),
            is_imm_mle: is_imm_mle.into(),
            eq_r_cycle,
            eq_r_product,
            prev_claim_for_r_cycle: input_claim_for_r_cycle,
            prev_claim_for_r_product: input_claim_for_r_product,
            prev_round_poly_for_r_cycle: None,
            prev_round_poly_for_r_product: None,
        }
    }
}
