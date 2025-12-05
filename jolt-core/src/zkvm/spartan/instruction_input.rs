use ark_ff::Zero;

use allocative::Allocative;
use rayon::prelude::*;
use tracer::instruction::Cycle;

use crate::{
    field::JoltField,
    poly::{
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
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    zkvm::{
        instruction::{Flags, InstructionFlags},
        witness::VirtualPolynomial,
    },
};

/// Degree bound of the sumcheck round polynomials.
const DEGREE_BOUND: usize = 3;

pub struct InstructionInputParams<F: JoltField> {
    r_cycle_stage_1: OpeningPoint<BIG_ENDIAN, F>,
    r_cycle_stage_2: OpeningPoint<BIG_ENDIAN, F>,
    gamma: F,
}

impl<F: JoltField> InstructionInputParams<F> {
    pub fn new(
        opening_accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let (r_cycle_stage_1, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LeftInstructionInput,
            SumcheckId::SpartanOuter,
        );
        let (r_cycle_stage_2, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LeftInstructionInput,
            SumcheckId::ProductVirtualization,
        );
        let gamma = transcript.challenge_scalar();
        Self {
            r_cycle_stage_1,
            r_cycle_stage_2,
            gamma,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for InstructionInputParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
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

    fn normalize_opening_point(
        &self,
        sumcheck_challenges: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(sumcheck_challenges.to_vec()).match_endianness()
    }
}

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
    #[tracing::instrument(skip_all, name = "InstructionInputSumcheckProver::initialize")]
    pub fn initialize(
        params: InstructionInputParams<F>,
        trace: &[Cycle],
        opening_accumulator: &ProverOpeningAccumulator<F>,
    ) -> Self {
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

        let (_, left_claim_stage_1) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LeftInstructionInput,
            SumcheckId::SpartanOuter,
        );
        let (_, right_claim_stage_1) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RightInstructionInput,
            SumcheckId::SpartanOuter,
        );
        let (_, left_claim_stage_2) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LeftInstructionInput,
            SumcheckId::ProductVirtualization,
        );
        let (_, right_claim_stage_2) = opening_accumulator.get_virtual_polynomial_opening(
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
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "InstructionInputSumcheckProver::compute_message")]
    fn compute_message(&mut self, _round: usize, _previous_claim: F) -> UniPoly<F> {
        // Lockstep requirement: the two split-eq polynomials must have identical split sizes
        debug_assert_eq!(
            self.eq_r_cycle_stage_1.E_out_current_len(),
            self.eq_r_cycle_stage_2.E_out_current_len(),
            "eq_r_cycle_stage_1 and eq_r_cycle_stage_2 must have same E_out length"
        );
        debug_assert_eq!(
            self.eq_r_cycle_stage_1.E_in_current_len(),
            self.eq_r_cycle_stage_2.E_in_current_len(),
            "eq_r_cycle_stage_1 and eq_r_cycle_stage_2 must have same E_in length"
        );

        let e_out_stage_2 = self.eq_r_cycle_stage_2.E_out_current();
        let e_in_stage_2 = self.eq_r_cycle_stage_2.E_in_current();

        // Fold over stage 1's split-eq; use indices to access stage 2's corresponding weights
        let [eval_at_0_for_stage_1, eval_at_inf_for_stage_1, eval_at_0_for_stage_2, eval_at_inf_for_stage_2] =
            self.eq_r_cycle_stage_1
                .par_fold_out_in(
                    || [F::Unreduced::<9>::zero(); 4],
                    |inner, j, x_in, e_in1| {
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

                        // Stage 2 e_in mirrors stage 1's x_in indexing; when fully bound, treat as 1
                        let e_in2 = if e_in_stage_2.len() <= 1 {
                            F::one()
                        } else {
                            e_in_stage_2[x_in]
                        };

                        // Accumulate in Montgomery-unreduced form to minimize reductions
                        inner[0] += e_in1.mul_unreduced::<9>(input_at_j_0);
                        inner[1] += e_in1.mul_unreduced::<9>(input_at_j_inf);
                        inner[2] += e_in2.mul_unreduced::<9>(input_at_j_0);
                        inner[3] += e_in2.mul_unreduced::<9>(input_at_j_inf);
                    },
                    |x_out, e_out1, inner| {
                        let mut out = [F::Unreduced::<9>::zero(); 4];
                        let reduced0 = F::from_montgomery_reduce::<9>(inner[0]);
                        let reduced1 = F::from_montgomery_reduce::<9>(inner[1]);
                        let reduced2 = F::from_montgomery_reduce::<9>(inner[2]);
                        let reduced3 = F::from_montgomery_reduce::<9>(inner[3]);
                        let e_out2 = if e_out_stage_2.len() <= 1 {
                            F::one()
                        } else {
                            e_out_stage_2[x_out]
                        };
                        out[0] = e_out1.mul_unreduced::<9>(reduced0);
                        out[1] = e_out1.mul_unreduced::<9>(reduced1);
                        out[2] = e_out2.mul_unreduced::<9>(reduced2);
                        out[3] = e_out2.mul_unreduced::<9>(reduced3);
                        out
                    },
                    |mut a, b| {
                        for i in 0..4 {
                            a[i] += b[i];
                        }
                        a
                    },
                )
                .map(|x| F::from_montgomery_reduce::<9>(x));

        let round_poly_stage_1 = self.eq_r_cycle_stage_1.gruen_poly_deg_3(
            eval_at_0_for_stage_1,
            eval_at_inf_for_stage_1,
            self.prev_claim_stage_1,
        );
        let round_poly_stage_2 = self.eq_r_cycle_stage_2.gruen_poly_deg_3(
            eval_at_0_for_stage_2,
            eval_at_inf_for_stage_2,
            self.prev_claim_stage_2,
        );
        let gamma_squared = self.params.gamma.square();
        let res = &round_poly_stage_1 + &(&round_poly_stage_2 * gamma_squared);
        self.prev_round_poly_stage_1 = Some(round_poly_stage_1);
        self.prev_round_poly_stage_2 = Some(round_poly_stage_2);
        res
    }

    #[tracing::instrument(skip_all, name = "InstructionInputSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
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
        let r = self.params.normalize_opening_point(sumcheck_challenges);
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
        opening_accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params = InstructionInputParams::new(opening_accumulator, transcript);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for InstructionInputSumcheckVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r = self.params.normalize_opening_point(sumcheck_challenges);
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
        let r = self.params.normalize_opening_point(sumcheck_challenges);
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
