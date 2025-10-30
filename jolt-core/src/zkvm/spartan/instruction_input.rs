use std::{array, iter::zip};

use allocative::Allocative;
use rayon::prelude::*;

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    zkvm::{
        dag::state_manager::StateManager,
        instruction::{Flags, InstructionFlags},
        witness::VirtualPolynomial,
    },
};

/// Degree bound of the sumcheck round polynomials.
const DEGREE_BOUND: usize = 3;

/// Sumcheck prover for [`InstructionInputSumcheckVerifier`].
// TODO: do 3 round compression SVO on each of the 8 multilinears, then bind directly
#[derive(Allocative)]
pub struct InstructionInputSumcheckProver<F: JoltField> {
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
    #[allocative(skip)]
    params: InstructionInputParams<F>,
}

impl<F: JoltField> InstructionInputSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "InstructionInputSumcheckProver::gen")]
    pub fn gen(
        state_manager: &mut StateManager<'_, F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let params = InstructionInputParams::new(state_manager);

        let (_, _, trace, _, _) = state_manager.get_prover_data();

        // Compute MLEs.
        let mut left_is_rs1_poly = vec![false; trace.len()];
        let mut left_is_pc_poly = vec![false; trace.len()];
        let mut right_is_rs2_poly = vec![false; trace.len()];
        let mut right_is_imm_poly = vec![false; trace.len()];
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
                    let flags = instruction.instruction_flags();
                    *left_is_rs1_eval = flags[InstructionFlags::LeftOperandIsRs1Value];
                    *left_is_pc_eval = flags[InstructionFlags::LeftOperandIsPC];
                    *right_is_rs2_eval = flags[InstructionFlags::RightOperandIsRs2Value];
                    *right_is_imm_eval = flags[InstructionFlags::RightOperandIsImm];
                    *rs1_value_eval = cycle.rs1_read().1;
                    *rs2_value_eval = cycle.rs2_read().1;
                    *imm_eval = instruction_norm.operands.imm;
                    *unexpanded_pc_eval = instruction_norm.address as u64;
                },
            );

        let eq_r_cycle_stage_1 =
            GruenSplitEqPolynomial::new(&params.r_cycle_stage_1.r, BindingOrder::LowToHigh);
        let eq_r_cycle_stage_2 =
            GruenSplitEqPolynomial::new(&params.r_cycle_stage_2.r, BindingOrder::LowToHigh);

        let (_, left_claim_stage_1) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::LeftInstructionInput,
            SumcheckId::SpartanOuter,
        );
        let (_, right_claim_stage_1) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::RightInstructionInput,
            SumcheckId::SpartanOuter,
        );
        let (_, left_claim_stage_2) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::LeftInstructionInput,
            SumcheckId::ProductVirtualization,
        );
        let (_, right_claim_stage_2) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::RightInstructionInput,
            SumcheckId::ProductVirtualization,
        );
        let claim_stage_1 = right_claim_stage_1 + params.gamma * left_claim_stage_1;
        let claim_stage_2 = right_claim_stage_2 + params.gamma * left_claim_stage_2;

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
            prev_claim_stage_1: claim_stage_1,
            prev_claim_stage_2: claim_stage_2,
            prev_round_poly_stage_1: None,
            prev_round_poly_stage_2: None,
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for InstructionInputSumcheckProver<F>
{
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.params.input_claim(accumulator)
    }

    #[tracing::instrument(
        skip_all,
        name = "InstructionInputSumcheckProver::compute_prover_message"
    )]
    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let out_evals_r_cycle_stage_1 = self.eq_r_cycle_stage_1.E_out_current();
        let in_evals_r_cycle_stage_1 = self.eq_r_cycle_stage_1.E_in_current();
        let out_evals_r_cycle_stage_2 = self.eq_r_cycle_stage_2.E_out_current();
        let in_evals_r_cycle_stage_2 = self.eq_r_cycle_stage_2.E_in_current();

        let out_len = out_evals_r_cycle_stage_1.len();
        let in_len = in_evals_r_cycle_stage_1.len();
        let in_n_vars = in_len.ilog2();

        let [eval_at_0_for_stage_1, eval_at_inf_for_stage_1, eval_at_0_for_stage_2, eval_at_inf_for_stage_2] =
            (0..out_len)
                .into_par_iter()
                .map(|j_hi| {
                    let mut eval_at_0_for_stage_1 = F::zero();
                    let mut eval_at_inf_for_stage_1 = F::zero();
                    let mut eval_at_0_for_stage_2 = F::zero();
                    let mut eval_at_inf_for_stage_2 = F::zero();

                    for j_lo in 0..in_len {
                        let j = j_lo + (j_hi << in_n_vars);

                        // Eval RightInstructionInputIsRs2(x) at (r', j, {0, inf}).
                        let right_is_rs2_at_j_0 = self.right_is_rs2_poly.get_bound_coeff(j * 2);
                        let right_is_rs2_at_j_inf =
                            self.right_is_rs2_poly.get_bound_coeff(j * 2 + 1) - right_is_rs2_at_j_0;

                        // Eval Rs2Value(x) at (r', j, {0, inf}).
                        let rs2_value_at_j_0 = self.rs2_value_poly.get_bound_coeff(j * 2);
                        let rs2_value_at_j_inf =
                            self.rs2_value_poly.get_bound_coeff(j * 2 + 1) - rs2_value_at_j_0;

                        // Eval RightInstructionInputIsImm(x) at (r', j, {0, inf}).
                        let right_is_imm_at_j_0 = self.right_is_imm_poly.get_bound_coeff(j * 2);
                        let right_is_imm_at_j_inf =
                            self.right_is_imm_poly.get_bound_coeff(j * 2 + 1) - right_is_imm_at_j_0;

                        // Eval Imm(x) at (r', j, {0, inf}).
                        let imm_at_j_0 = self.imm_poly.get_bound_coeff(j * 2);
                        let imm_at_j_inf = self.imm_poly.get_bound_coeff(j * 2 + 1) - imm_at_j_0;

                        // Eval RightInstructionInput(x) at (r', j, {0, inf}).
                        let right_at_j_0 = right_is_rs2_at_j_0 * rs2_value_at_j_0
                            + right_is_imm_at_j_0 * imm_at_j_0;
                        let right_at_j_inf = right_is_rs2_at_j_inf * rs2_value_at_j_inf
                            + right_is_imm_at_j_inf * imm_at_j_inf;

                        // Eval LeftInstructionInputIsRs1(x) at (r', j, {0, inf}).
                        let left_is_rs1_at_j_0 = self.left_is_rs1_poly.get_bound_coeff(j * 2);
                        let left_is_rs1_at_j_inf =
                            self.left_is_rs1_poly.get_bound_coeff(j * 2 + 1) - left_is_rs1_at_j_0;

                        // Eval Rs1Value(x) at (r', j, {0, inf}).
                        let rs1_value_at_j_0 = self.rs1_value_poly.get_bound_coeff(j * 2);
                        let rs1_value_at_j_inf =
                            self.rs1_value_poly.get_bound_coeff(j * 2 + 1) - rs1_value_at_j_0;

                        // Eval LeftInstructionInputIsPc(x) at (r', j, {0, inf}).
                        let left_is_pc_at_j_0 = self.left_is_pc_poly.get_bound_coeff(j * 2);
                        let left_is_pc_at_j_inf =
                            self.left_is_pc_poly.get_bound_coeff(j * 2 + 1) - left_is_pc_at_j_0;

                        // Eval UnexpandedPc(x) at (r', j, {0, inf}).
                        let unexpanded_pc_at_j_0 = self.unexpanded_pc_poly.get_bound_coeff(j * 2);
                        let unexpanded_pc_at_j_inf =
                            self.unexpanded_pc_poly.get_bound_coeff(j * 2 + 1)
                                - unexpanded_pc_at_j_0;

                        // Eval LeftInstructionInput(x) at (r', {0, inf}, j).
                        let left_at_j_0 = left_is_rs1_at_j_0 * rs1_value_at_j_0
                            + left_is_pc_at_j_0 * unexpanded_pc_at_j_0;
                        let left_at_j_inf = left_is_rs1_at_j_inf * rs1_value_at_j_inf
                            + left_is_pc_at_j_inf * unexpanded_pc_at_j_inf;

                        // Eval Input(x) = RightInstructionInput(x) + gamma * LeftInstructionInput(x) at (r', {0, inf}, j).
                        let input_at_j_0 = right_at_j_0 + self.params.gamma * left_at_j_0;
                        let input_at_j_inf = right_at_j_inf + self.params.gamma * left_at_j_inf;

                        eval_at_0_for_stage_1 += in_evals_r_cycle_stage_1[j_lo] * input_at_j_0;
                        eval_at_inf_for_stage_1 += in_evals_r_cycle_stage_1[j_lo] * input_at_j_inf;
                        eval_at_0_for_stage_2 += in_evals_r_cycle_stage_2[j_lo] * input_at_j_0;
                        eval_at_inf_for_stage_2 += in_evals_r_cycle_stage_2[j_lo] * input_at_j_inf;
                    }

                    [
                        out_evals_r_cycle_stage_1[j_hi] * eval_at_0_for_stage_1,
                        out_evals_r_cycle_stage_1[j_hi] * eval_at_inf_for_stage_1,
                        out_evals_r_cycle_stage_2[j_hi] * eval_at_0_for_stage_2,
                        out_evals_r_cycle_stage_2[j_hi] * eval_at_inf_for_stage_2,
                    ]
                })
                .reduce(|| [F::zero(); 4], |a, b| array::from_fn(|i| a[i] + b[i]));

        let univariate_evals_stage_1 = self.eq_r_cycle_stage_1.gruen_evals_deg_3(
            eval_at_0_for_stage_1,
            eval_at_inf_for_stage_1,
            self.prev_claim_stage_1,
        );
        let univariate_evals_stage_2 = self.eq_r_cycle_stage_2.gruen_evals_deg_3(
            eval_at_0_for_stage_2,
            eval_at_inf_for_stage_2,
            self.prev_claim_stage_2,
        );
        self.prev_round_poly_stage_1 = Some(UniPoly::from_evals_and_hint(
            self.prev_claim_stage_1,
            &univariate_evals_stage_1,
        ));
        self.prev_round_poly_stage_2 = Some(UniPoly::from_evals_and_hint(
            self.prev_claim_stage_2,
            &univariate_evals_stage_2,
        ));
        zip(univariate_evals_stage_1, univariate_evals_stage_2)
            .map(|(eval_stage_1, eval_stage_2)| {
                eval_stage_1 + self.params.gamma.square() * eval_stage_2
            })
            .collect()
    }

    #[tracing::instrument(skip_all, name = "InstructionInputSumcheckProver::bind")]
    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        let Self {
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
            params: _,
        } = self;
        left_is_rs1_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        left_is_pc_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        right_is_rs2_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        right_is_imm_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        rs1_value_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        rs2_value_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        imm_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        unexpanded_pc_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        eq_r_cycle_stage_1.bind(r_j);
        eq_r_cycle_stage_2.bind(r_j);
        *prev_claim_stage_1 = prev_round_poly_stage_1.take().unwrap().evaluate(&r_j);
        *prev_claim_stage_2 = prev_round_poly_stage_2.take().unwrap().evaluate(&r_j);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r = get_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::InstructionFlags(InstructionFlags::LeftOperandIsRs1Value),
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
            self.left_is_rs1_poly.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Rs1Value,
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
            self.rs1_value_poly.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::InstructionFlags(InstructionFlags::LeftOperandIsPC),
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
            self.left_is_pc_poly.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
            self.unexpanded_pc_poly.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::InstructionFlags(InstructionFlags::RightOperandIsRs2Value),
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
            self.right_is_rs2_poly.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Rs2Value,
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
            self.rs2_value_poly.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::InstructionFlags(InstructionFlags::RightOperandIsImm),
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
            self.right_is_imm_poly.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Imm,
            SumcheckId::InstructionInputVirtualization,
            r,
            self.imm_poly.final_sumcheck_claim(),
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

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
pub struct InstructionInputSumcheckVerifier<F: JoltField> {
    params: InstructionInputParams<F>,
}

impl<F: JoltField> InstructionInputSumcheckVerifier<F> {
    pub fn new(
        state_manager: &mut StateManager<'_, F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let params = InstructionInputParams::new(state_manager);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for InstructionInputSumcheckVerifier<F>
{
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, accumulator: &VerifierOpeningAccumulator<F>) -> F {
        self.params.input_claim(accumulator)
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r = get_opening_point(sumcheck_challenges);
        let eq_eval_at_r_cycle_stage_1 = EqPolynomial::mle_endian(&r, &self.params.r_cycle_stage_1);
        let eq_eval_at_r_cycle_stage_2 = EqPolynomial::mle_endian(&r, &self.params.r_cycle_stage_2);

        let (_, rs1_value_eval) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Value,
            SumcheckId::InstructionInputVirtualization,
        );
        let (_, left_is_rs1_eval) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionFlags(InstructionFlags::LeftOperandIsRs1Value),
            SumcheckId::InstructionInputVirtualization,
        );
        let (_, unexpanded_pc_eval) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::InstructionInputVirtualization,
        );
        let (_, left_is_pc_eval) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionFlags(InstructionFlags::LeftOperandIsPC),
            SumcheckId::InstructionInputVirtualization,
        );
        let (_, rs2_value_eval) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs2Value,
            SumcheckId::InstructionInputVirtualization,
        );
        let (_, right_is_rs2_eval) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionFlags(InstructionFlags::RightOperandIsRs2Value),
            SumcheckId::InstructionInputVirtualization,
        );
        let (_, imm_eval) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Imm,
            SumcheckId::InstructionInputVirtualization,
        );
        let (_, right_is_imm_eval) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionFlags(InstructionFlags::RightOperandIsImm),
            SumcheckId::InstructionInputVirtualization,
        );

        let left_instruction_input =
            left_is_rs1_eval * rs1_value_eval + left_is_pc_eval * unexpanded_pc_eval;
        let right_instruction_input =
            right_is_rs2_eval * rs2_value_eval + right_is_imm_eval * imm_eval;

        (eq_eval_at_r_cycle_stage_1 + self.params.gamma.square() * eq_eval_at_r_cycle_stage_2)
            * (right_instruction_input + self.params.gamma * left_instruction_input)
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r = get_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::InstructionFlags(InstructionFlags::LeftOperandIsRs1Value),
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
            VirtualPolynomial::InstructionFlags(InstructionFlags::LeftOperandIsPC),
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
            VirtualPolynomial::InstructionFlags(InstructionFlags::RightOperandIsRs2Value),
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
            VirtualPolynomial::InstructionFlags(InstructionFlags::RightOperandIsImm),
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
}

struct InstructionInputParams<F: JoltField> {
    r_cycle_stage_1: OpeningPoint<BIG_ENDIAN, F>,
    r_cycle_stage_2: OpeningPoint<BIG_ENDIAN, F>,
    gamma: F,
}

impl<F: JoltField> InstructionInputParams<F> {
    fn new(
        state_manager: &mut StateManager<'_, F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let (r_cycle_stage_1, _) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::LeftInstructionInput,
            SumcheckId::SpartanOuter,
        );
        let (r_cycle_stage_2, _) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::LeftInstructionInput,
            SumcheckId::ProductVirtualization,
        );
        let gamma = state_manager.transcript.borrow_mut().challenge_scalar();
        Self {
            r_cycle_stage_1,
            r_cycle_stage_2,
            gamma,
        }
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle_stage_1.len()
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, left_claim_stage_1) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LeftInstructionInput,
            SumcheckId::SpartanOuter,
        );
        let (_, right_claim_stage_1) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RightInstructionInput,
            SumcheckId::SpartanOuter,
        );
        let (_, left_claim_stage_2) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LeftInstructionInput,
            SumcheckId::ProductVirtualization,
        );
        let (_, right_claim_stage_2) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RightInstructionInput,
            SumcheckId::ProductVirtualization,
        );

        let claim_stage_1 = right_claim_stage_1 + self.gamma * left_claim_stage_1;
        let claim_stage_2 = right_claim_stage_2 + self.gamma * left_claim_stage_2;

        claim_stage_1 + self.gamma.square() * claim_stage_2
    }
}

fn get_opening_point<F: JoltField>(
    sumcheck_challenges: &[F::Challenge],
) -> OpeningPoint<BIG_ENDIAN, F> {
    OpeningPoint::<LITTLE_ENDIAN, F>::new(sumcheck_challenges.to_vec()).match_endianness()
}
