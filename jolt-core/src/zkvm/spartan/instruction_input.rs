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
    utils::math::Math,
    zkvm::{
        dag::state_manager::StateManager,
        instruction::{CircuitFlags, InstructionFlags},
        witness::VirtualPolynomial,
    },
};

/// A sumcheck instance for:
///
/// ```text
/// sum_j (eq(r_cycle_stage_1, j) + gamma^2 * eq(r_cycle_stage_2, j)) * (RightInstructionInput(j) + gamma * LeftInstructionInput(j))
/// ```
///
/// Where
///
/// ```text
/// LeftInstructionInput(x) = LeftInstructionInputIsRs1(x) * Rs1Value(x) + LeftInstructionInputIsPc(x) * UnexpandedPc(x)
/// RightInstructionInput(x) = RightInstructionInputIsRs2(x) * Rs2Value(x) + RightInstructionInputIsImm(x) * Imm(x)
/// ```
///
/// Note:
/// - `r_cycle_stage_1` is the randomness from the log(T) rounds of Spartan outer sumcheck (stage 1).
/// - `r_cycle_stage_2` is the randomness from instruction product sumcheck (stage 2).
#[derive(Allocative)]
pub struct InstructionInputSumcheck<F: JoltField> {
    input_sample_stage_1: (OpeningPoint<BIG_ENDIAN, F>, F),
    input_sample_stage_2: (OpeningPoint<BIG_ENDIAN, F>, F),
    prover_state: Option<ProverState<F>>,
    gamma: F,
}

impl<F: JoltField> InstructionInputSumcheck<F> {
    #[tracing::instrument(skip_all, name = "InstructionInputSumcheck::new_prover")]
    pub fn new_prover(
        state_manager: &mut StateManager<'_, F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        // Get claimed samples.
        let accumulator = state_manager.get_prover_accumulator();
        let (r_cycle_stage_1, left_claim_stage_1) =
            accumulator.borrow().get_virtual_polynomial_opening(
                VirtualPolynomial::LeftInstructionInput,
                SumcheckId::SpartanOuter,
            );
        let (_, right_claim_stage_1) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RightInstructionInput,
            SumcheckId::SpartanOuter,
        );
        let (r_cycle_stage_2, left_claim_stage_2) =
            accumulator.borrow().get_virtual_polynomial_opening(
                VirtualPolynomial::LeftInstructionInput,
                SumcheckId::ProductVirtualization,
            );
        let (_, right_claim_stage_2) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RightInstructionInput,
            SumcheckId::ProductVirtualization,
        );

        let gamma = state_manager.transcript.borrow_mut().challenge_scalar();
        let claim_stage_1 = right_claim_stage_1 + gamma * left_claim_stage_1;
        let claim_stage_2 = right_claim_stage_2 + gamma * left_claim_stage_2;

        let input_sample_stage_1 = (r_cycle_stage_1, claim_stage_1);
        let input_sample_stage_2 = (r_cycle_stage_2, claim_stage_2);

        let (_, trace, _, _) = state_manager.get_prover_data();
        let prover_state = ProverState::gen(trace, &input_sample_stage_1, &input_sample_stage_2);

        Self {
            input_sample_stage_1,
            input_sample_stage_2,
            prover_state: Some(prover_state),
            gamma,
        }
    }

    pub fn new_verifier<T: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, T, PCS>,
    ) -> Self {
        let accumulator = state_manager.get_verifier_accumulator();
        let (r_outer, left_claim_for_stage_1) =
            accumulator.borrow().get_virtual_polynomial_opening(
                VirtualPolynomial::LeftInstructionInput,
                SumcheckId::SpartanOuter,
            );
        let (_, _, T) = state_manager.get_verifier_data();
        let (r_cycle_stage_1, _) = r_outer.split_at(T.log_2());
        let (_, right_claim_for_stage_1) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RightInstructionInput,
            SumcheckId::SpartanOuter,
        );
        let (r_cycle_stage_2, left_claim_for_stage_2) =
            accumulator.borrow().get_virtual_polynomial_opening(
                VirtualPolynomial::LeftInstructionInput,
                SumcheckId::ProductVirtualization,
            );
        let (_, right_claim_for_stage_2) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RightInstructionInput,
            SumcheckId::ProductVirtualization,
        );

        let gamma = state_manager.transcript.borrow_mut().challenge_scalar();
        let claim_for_stage_1 = right_claim_for_stage_1 + gamma * left_claim_for_stage_1;
        let claim_for_stage_2 = right_claim_for_stage_2 + gamma * left_claim_for_stage_2;

        let input_sample_stage_1 = (r_cycle_stage_1, claim_for_stage_1);
        let input_sample_stage_2 = (r_cycle_stage_2, claim_for_stage_2);

        Self {
            input_sample_stage_1,
            input_sample_stage_2,
            prover_state: None,
            gamma,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for InstructionInputSumcheck<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.input_sample_stage_1.0.len() // = log(T)
    }

    fn input_claim(&self) -> F {
        self.input_sample_stage_1.1 + self.gamma.square() * self.input_sample_stage_2.1
    }

    #[tracing::instrument(skip_all, name = "InstructionInputSumcheck::compute_prover_message")]
    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let state = self.prover_state.as_mut().unwrap();

        let out_evals_r_cycle_stage_1 = state.eq_r_cycle_stage_1.E_out_current();
        let in_evals_r_cycle_stage_1 = state.eq_r_cycle_stage_1.E_in_current();
        let out_evals_r_cycle_stage_2 = state.eq_r_cycle_stage_2.E_out_current();
        let in_evals_r_cycle_stage_2 = state.eq_r_cycle_stage_2.E_in_current();

        let out_len = out_evals_r_cycle_stage_1.len();
        let in_len = in_evals_r_cycle_stage_1.len();
        let out_n_vars = out_len.ilog2();
        let half_n = state.rs1_value_poly.len() / 2;

        let [eval_at_0_for_stage_1, eval_at_inf_for_stage_1, eval_at_0_for_stage_2, eval_at_inf_for_stage_2] =
            (0..in_len)
                .into_par_iter()
                .map(|j_hi| {
                    let mut eval_at_0_for_stage_1 = F::zero();
                    let mut eval_at_inf_for_stage_1 = F::zero();
                    let mut eval_at_0_for_stage_2 = F::zero();
                    let mut eval_at_inf_for_stage_2 = F::zero();

                    for j_lo in 0..out_len {
                        let j = j_lo + (j_hi << out_n_vars);

                        // Eval RightInstructionInputIsRs2(x) at (r', {0, 1, inf}, j).
                        let right_is_rs2_at_0_j = state.right_is_rs2_poly.get_bound_coeff(j);
                        let right_is_rs2_at_1_j =
                            state.right_is_rs2_poly.get_bound_coeff(j + half_n);
                        let right_is_rs2_at_inf_j = right_is_rs2_at_1_j - right_is_rs2_at_0_j;

                        // Eval Rs2Value(x) at (r', {0, 1, inf}, j).
                        let rs2_value_at_0_j = state.rs2_value_poly.get_bound_coeff(j);
                        let rs2_value_at_1_j = state.rs2_value_poly.get_bound_coeff(j + half_n);
                        let rs2_value_at_inf_j = rs2_value_at_1_j - rs2_value_at_0_j;

                        // Eval RightInstructionInputIsImm(x) at (r', {0, 1, inf}, j).
                        let right_is_imm_at_0_j = state.right_is_imm_poly.get_bound_coeff(j);
                        let right_is_imm_at_1_j =
                            state.right_is_imm_poly.get_bound_coeff(j + half_n);
                        let right_is_imm_at_inf_j = right_is_imm_at_1_j - right_is_imm_at_0_j;

                        // Eval Imm(x) at (r', {0, 1, inf}, j).
                        let imm_at_0_j = state.imm_poly.get_bound_coeff(j);
                        let imm_at_1_j = state.imm_poly.get_bound_coeff(j + half_n);
                        let imm_at_inf_j = imm_at_1_j - imm_at_0_j;

                        // Eval RightInstructionInput(x) at (r', {0, inf}, j).
                        let right_at_0_j = right_is_rs2_at_0_j * rs2_value_at_0_j
                            + right_is_imm_at_0_j * imm_at_0_j;
                        let right_at_inf_j = right_is_rs2_at_inf_j * rs2_value_at_inf_j
                            + right_is_imm_at_inf_j * imm_at_inf_j;

                        // Eval LeftInstructionInputIsRs1(x) at (r', {0, 1, inf}, j).
                        let left_is_rs1_at_0_j = state.left_is_rs1_poly.get_bound_coeff(j);
                        let left_is_rs1_at_1_j = state.left_is_rs1_poly.get_bound_coeff(j + half_n);
                        let left_is_rs1_at_inf_j = left_is_rs1_at_1_j - left_is_rs1_at_0_j;

                        // Eval Rs1Value(x) at (r', {0, 1, inf}, j).
                        let rs1_value_at_0_j = state.rs1_value_poly.get_bound_coeff(j);
                        let rs1_value_at_1_j = state.rs1_value_poly.get_bound_coeff(j + half_n);
                        let rs1_value_at_inf_j = rs1_value_at_1_j - rs1_value_at_0_j;

                        // Eval LeftInstructionInputIsPc(x) at (r', {0, 1, inf}, j).
                        let left_is_pc_at_0_j = state.left_is_pc_poly.get_bound_coeff(j);
                        let left_is_pc_at_1_j = state.left_is_pc_poly.get_bound_coeff(j + half_n);
                        let left_is_pc_at_inf_j = left_is_pc_at_1_j - left_is_pc_at_0_j;

                        // Eval UnexpandedPc(x) at (r', {0, 1, inf}, j).
                        let unexpanded_pc_at_0_j = state.unexpanded_pc_poly.get_bound_coeff(j);
                        let unexpanded_pc_at_1_j =
                            state.unexpanded_pc_poly.get_bound_coeff(j + half_n);
                        let unexpanded_pc_at_inf_j = unexpanded_pc_at_1_j - unexpanded_pc_at_0_j;

                        // Eval LeftInstructionInput(x) at (r', {0, inf}, j).
                        let left_at_0_j = left_is_rs1_at_0_j * rs1_value_at_0_j
                            + left_is_pc_at_0_j * unexpanded_pc_at_0_j;
                        let left_at_inf_j = left_is_rs1_at_inf_j * rs1_value_at_inf_j
                            + left_is_pc_at_inf_j * unexpanded_pc_at_inf_j;

                        // Eval Input(x) = RightInstructionInput(x) + gamma * LeftInstructionInput(x) at (r', {0, inf}, j).
                        let input_at_0_j = right_at_0_j + self.gamma * left_at_0_j;
                        let input_at_inf_j = right_at_inf_j + self.gamma * left_at_inf_j;

                        eval_at_0_for_stage_1 += out_evals_r_cycle_stage_1[j_lo] * input_at_0_j;
                        eval_at_inf_for_stage_1 += out_evals_r_cycle_stage_1[j_lo] * input_at_inf_j;
                        eval_at_0_for_stage_2 += out_evals_r_cycle_stage_2[j_lo] * input_at_0_j;
                        eval_at_inf_for_stage_2 += out_evals_r_cycle_stage_2[j_lo] * input_at_inf_j;
                    }

                    [
                        in_evals_r_cycle_stage_1[j_hi] * eval_at_0_for_stage_1,
                        in_evals_r_cycle_stage_1[j_hi] * eval_at_inf_for_stage_1,
                        in_evals_r_cycle_stage_2[j_hi] * eval_at_0_for_stage_2,
                        in_evals_r_cycle_stage_2[j_hi] * eval_at_inf_for_stage_2,
                    ]
                })
                .reduce(|| [F::zero(); 4], |a, b| array::from_fn(|i| a[i] + b[i]));

        let univariate_evals_stage_1 = state.eq_r_cycle_stage_1.gruen_evals_deg_3(
            eval_at_0_for_stage_1,
            eval_at_inf_for_stage_1,
            state.prev_claim_stage_1,
        );
        let univariate_evals_stage_2 = state.eq_r_cycle_stage_2.gruen_evals_deg_3(
            eval_at_0_for_stage_2,
            eval_at_inf_for_stage_2,
            state.prev_claim_stage_2,
        );
        state.prev_round_poly_stage_1 = Some(poly_from_evals_and_hint(
            state.prev_claim_stage_1,
            &univariate_evals_stage_1,
        ));
        state.prev_round_poly_stage_2 = Some(poly_from_evals_and_hint(
            state.prev_claim_stage_2,
            &univariate_evals_stage_2,
        ));
        zip(univariate_evals_stage_1, univariate_evals_stage_2)
            .map(|(eval_stage_1, eval_stage_2)| eval_stage_1 + self.gamma.square() * eval_stage_2)
            .collect()
    }

    #[tracing::instrument(skip_all, name = "InstructionInputSumcheck::bind")]
    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        let ProverState {
            left_is_rs1_poly,
            left_is_pc_poly,
            right_is_rs2_poly,
            right_is_imm_poly,
            rs1_value_poly,
            rs2_value_poly,
            imm_poly,
            unexpanded_pc_poly,
            eq_r_cycle_stage_1,
            eq_r_cycle_stage_2,
            prev_claim_stage_1,
            prev_claim_stage_2,
            prev_round_poly_stage_1,
            prev_round_poly_stage_2,
        } = self.prover_state.as_mut().unwrap();
        left_is_rs1_poly.bind_parallel(r_j, BindingOrder::HighToLow);
        left_is_pc_poly.bind_parallel(r_j, BindingOrder::HighToLow);
        right_is_rs2_poly.bind_parallel(r_j, BindingOrder::HighToLow);
        right_is_imm_poly.bind_parallel(r_j, BindingOrder::HighToLow);
        rs1_value_poly.bind_parallel(r_j, BindingOrder::HighToLow);
        rs2_value_poly.bind_parallel(r_j, BindingOrder::HighToLow);
        imm_poly.bind_parallel(r_j, BindingOrder::HighToLow);
        unexpanded_pc_poly.bind_parallel(r_j, BindingOrder::HighToLow);
        eq_r_cycle_stage_1.bind(r_j);
        eq_r_cycle_stage_2.bind(r_j);
        *prev_claim_stage_1 = prev_round_poly_stage_1.take().unwrap().evaluate(&r_j);
        *prev_claim_stage_2 = prev_round_poly_stage_2.take().unwrap().evaluate(&r_j);
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F::Challenge],
    ) -> F {
        let r = OpeningPoint::<BIG_ENDIAN, F>::new(r.to_vec());
        let eq_eval_at_r_cycle_stage_1 = EqPolynomial::mle_endian(&r, &self.input_sample_stage_1.0);
        let eq_eval_at_r_cycle_stage_2 = EqPolynomial::mle_endian(&r, &self.input_sample_stage_2.0);

        let accumulator = accumulator.as_ref().unwrap().borrow();
        let (_, rs1_value_eval) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Value,
            SumcheckId::InstructionInputVirtualization,
        );
        let (_, left_is_rs1_eval) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::LeftOperandIsRs1Value),
            SumcheckId::InstructionInputVirtualization,
        );
        let (_, unexpanded_pc_eval) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::InstructionInputVirtualization,
        );
        let (_, left_is_pc_eval) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::LeftOperandIsPC),
            SumcheckId::InstructionInputVirtualization,
        );
        let (_, rs2_value_eval) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs2Value,
            SumcheckId::InstructionInputVirtualization,
        );
        let (_, right_is_rs2_eval) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::RightOperandIsRs2Value),
            SumcheckId::InstructionInputVirtualization,
        );
        let (_, imm_eval) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Imm,
            SumcheckId::InstructionInputVirtualization,
        );
        let (_, right_is_imm_eval) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::RightOperandIsImm),
            SumcheckId::InstructionInputVirtualization,
        );

        let left_instruction_input =
            left_is_rs1_eval * rs1_value_eval + left_is_pc_eval * unexpanded_pc_eval;
        let right_instruction_input =
            right_is_rs2_eval * rs2_value_eval + right_is_imm_eval * imm_eval;

        (eq_eval_at_r_cycle_stage_1 + self.gamma.square() * eq_eval_at_r_cycle_stage_2)
            * (right_instruction_input + self.gamma * left_instruction_input)
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
        r: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let state = self.prover_state.as_ref().unwrap();
        let mut accumulator = accumulator.borrow_mut();
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::LeftOperandIsRs1Value),
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
            state.left_is_rs1_poly.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Rs1Value,
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
            state.rs1_value_poly.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::LeftOperandIsPC),
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
            state.left_is_pc_poly.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
            state.unexpanded_pc_poly.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::RightOperandIsRs2Value),
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
            state.right_is_rs2_poly.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Rs2Value,
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
            state.rs2_value_poly.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::RightOperandIsImm),
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
            state.right_is_imm_poly.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Imm,
            SumcheckId::InstructionInputVirtualization,
            r,
            state.imm_poly.final_sumcheck_claim(),
        );
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        transcript: &mut T,
        r: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let mut accumulator = accumulator.borrow_mut();
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::LeftOperandIsRs1Value),
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Rs1Value,
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::LeftOperandIsPC),
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::RightOperandIsRs2Value),
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Rs2Value,
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::RightOperandIsImm),
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Imm,
            SumcheckId::InstructionInputVirtualization,
            r,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

#[derive(Allocative)]
struct ProverState<F: JoltField> {
    left_is_rs1_poly: MultilinearPolynomial<F>,
    left_is_pc_poly: MultilinearPolynomial<F>,
    right_is_rs2_poly: MultilinearPolynomial<F>,
    right_is_imm_poly: MultilinearPolynomial<F>,
    rs1_value_poly: MultilinearPolynomial<F>,
    rs2_value_poly: MultilinearPolynomial<F>,
    imm_poly: MultilinearPolynomial<F>,
    unexpanded_pc_poly: MultilinearPolynomial<F>,
    eq_r_cycle_stage_1: GruenSplitEqPolynomial<F>,
    eq_r_cycle_stage_2: GruenSplitEqPolynomial<F>,
    prev_claim_stage_1: F,
    prev_claim_stage_2: F,
    prev_round_poly_stage_1: Option<UniPoly<F>>,
    prev_round_poly_stage_2: Option<UniPoly<F>>,
}

impl<F: JoltField> ProverState<F> {
    fn gen(
        trace: &[Cycle],
        sample_stage_1: &(OpeningPoint<BIG_ENDIAN, F>, F),
        sample_stage_2: &(OpeningPoint<BIG_ENDIAN, F>, F),
    ) -> Self {
        // Compute MLEs.
        let mut left_is_rs1_poly = vec![0u8; trace.len()];
        let mut left_is_pc_poly = vec![0u8; trace.len()];
        let mut right_is_rs2_poly = vec![0u8; trace.len()];
        let mut right_is_imm_poly = vec![0u8; trace.len()];
        let mut rs1_value_poly = vec![0; trace.len()];
        let mut rs2_value_poly = vec![0; trace.len()];
        let mut imm_poly = vec![0; trace.len()];
        let mut unexpanded_pc_poly = vec![0; trace.len()];
        (
            &mut left_is_rs1_poly,
            &mut left_is_pc_poly,
            &mut right_is_rs2_poly,
            &mut right_is_imm_poly,
            &mut rs1_value_poly,
            &mut rs2_value_poly,
            &mut imm_poly,
            &mut unexpanded_pc_poly,
            trace,
        )
            .into_par_iter()
            .for_each(
                |(
                    left_is_rs1_eval,
                    left_is_pc_eval,
                    right_is_rs2_eval,
                    right_is_imm_eval,
                    rs1_value_eval,
                    rs2_value_eval,
                    imm_eval,
                    unexpanded_pc_eval,
                    cycle,
                )| {
                    let instruction = cycle.instruction();
                    let instruction_norm = instruction.normalize();
                    let flags = instruction.circuit_flags();
                    *left_is_rs1_eval = flags[CircuitFlags::LeftOperandIsRs1Value].into();
                    *left_is_pc_eval = flags[CircuitFlags::LeftOperandIsPC].into();
                    *right_is_rs2_eval = flags[CircuitFlags::RightOperandIsRs2Value].into();
                    *right_is_imm_eval = flags[CircuitFlags::RightOperandIsImm].into();
                    *rs1_value_eval = cycle.rs1_read().1;
                    *rs2_value_eval = cycle.rs2_read().1;
                    *imm_eval = instruction_norm.operands.imm;
                    *unexpanded_pc_eval = instruction_norm.address as u64;
                },
            );

        let eq_r_cycle_stage_1 =
            GruenSplitEqPolynomial::new(&sample_stage_1.0.r, BindingOrder::HighToLow);
        let eq_r_cycle_stage_2 =
            GruenSplitEqPolynomial::new(&sample_stage_2.0.r, BindingOrder::HighToLow);

        Self {
            left_is_rs1_poly: left_is_rs1_poly.into(),
            left_is_pc_poly: left_is_pc_poly.into(),
            right_is_rs2_poly: right_is_rs2_poly.into(),
            right_is_imm_poly: right_is_imm_poly.into(),
            rs1_value_poly: rs1_value_poly.into(),
            rs2_value_poly: rs2_value_poly.into(),
            imm_poly: imm_poly.into(),
            unexpanded_pc_poly: unexpanded_pc_poly.into(),
            eq_r_cycle_stage_1,
            eq_r_cycle_stage_2,
            prev_claim_stage_1: sample_stage_1.1,
            prev_claim_stage_2: sample_stage_2.1,
            prev_round_poly_stage_1: None,
            prev_round_poly_stage_2: None,
        }
    }
}

/// Interpolate a polynomial `p(x)` from its evaluations at the points `0, 2, ..., degree-1` and a `hint = p(0) + p(1)`.
fn poly_from_evals_and_hint<F: JoltField>(hint: F, evals: &[F]) -> UniPoly<F> {
    let mut evals = evals.to_vec();
    let eval_at_1 = hint - evals[0];
    evals.insert(1, eval_at_1);
    UniPoly::from_evals(&evals)
}
