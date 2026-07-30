use allocative::Allocative;
use ark_ff::Zero;
use jolt_riscv::JoltTraceRow;
use rayon::prelude::*;
use std::cmp::Ordering;

#[cfg(feature = "zk")]
use crate::poly::opening_proof::OpeningId;
#[cfg(feature = "zk")]
use crate::subprotocols::blindfold::{
    InputClaimConstraint, OutputClaimConstraint, ProductTerm, ValueSource,
};
use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::BindingOrder,
        opening_proof::{
            AbstractVerifierOpeningAccumulator, OpeningAccumulator, OpeningPoint, PolynomialId,
            ProverOpeningAccumulator, SumcheckId, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_claim::{
            CachedPointRef, ChallengePart, Claim, ClaimExpr, InputOutputClaims, SumcheckFrontend,
            VerifierEvaluablePolynomial,
        },
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::small_scalar::SmallScalar,
    zkvm::{
        instruction::{Flags, InstructionFlags},
        witness::VirtualPolynomial,
    },
};

/// Degree bound of the sumcheck round polynomials.
const DEGREE_BOUND: usize = 3;

#[derive(Allocative, Clone)]
pub struct InstructionInputParams<F: JoltField> {
    pub r_cycle_stage_2: OpeningPoint<BIG_ENDIAN, F>,
    pub gamma: F,
}

impl<F: JoltField> InstructionInputParams<F> {
    pub fn new(
        opening_accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let (r_cycle_stage_2, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LeftInstructionInput,
            SumcheckId::SpartanProductVirtualization,
        );
        let gamma = transcript.challenge_scalar();
        Self {
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
        self.r_cycle_stage_2.len()
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (r_left_claim_instruction, left_claim_instruction) = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LeftInstructionInput,
                SumcheckId::InstructionClaimReduction,
            );
        let (r_right_claim_instruction, right_claim_instruction) = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RightInstructionInput,
                SumcheckId::InstructionClaimReduction,
            );

        let (r_left_claim_stage_2, left_claim_stage_2) = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LeftInstructionInput,
                SumcheckId::SpartanProductVirtualization,
            );
        let (r_right_claim_stage_2, right_claim_stage_2) = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RightInstructionInput,
                SumcheckId::SpartanProductVirtualization,
            );

        // Soundness: InstructionClaimReduction and SpartanProductVirtualization must produce
        // the same claims at the same opening points.
        assert_eq!(r_left_claim_instruction, r_left_claim_stage_2);
        assert_eq!(left_claim_instruction, left_claim_stage_2);
        assert_eq!(r_right_claim_instruction, r_right_claim_stage_2);
        assert_eq!(right_claim_instruction, right_claim_stage_2);

        right_claim_stage_2 + self.gamma * left_claim_stage_2
    }

    fn normalize_opening_point(
        &self,
        sumcheck_challenges: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(sumcheck_challenges.to_vec()).match_endianness()
    }

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        InputClaimConstraint::weighted_openings(&[
            OpeningId::virt(
                VirtualPolynomial::RightInstructionInput,
                SumcheckId::SpartanProductVirtualization,
            ),
            OpeningId::virt(
                VirtualPolynomial::LeftInstructionInput,
                SumcheckId::SpartanProductVirtualization,
            ),
        ])
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(&self, _: &dyn OpeningAccumulator<F>) -> Vec<F> {
        vec![self.gamma]
    }

    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        // expected_output_claim = E2 * (right_input + γ * left_input)
        // where:
        //   left_input = left_is_rs1 * rs1_value + left_is_pc * unexpanded_pc
        //   right_input = right_is_rs2 * rs2_value + right_is_imm * imm
        //
        // Challenges:
        // - Challenge(0) = E2
        // - Challenge(1) = γ * E2

        let left_is_rs1 = OpeningId::virt(
            VirtualPolynomial::InstructionFlags(InstructionFlags::LeftOperandIsRs1Value),
            SumcheckId::InstructionInputVirtualization,
        );
        let rs1_value = OpeningId::virt(
            VirtualPolynomial::Rs1Value,
            SumcheckId::InstructionInputVirtualization,
        );
        let left_is_pc = OpeningId::virt(
            VirtualPolynomial::InstructionFlags(InstructionFlags::LeftOperandIsPC),
            SumcheckId::InstructionInputVirtualization,
        );
        let unexpanded_pc = OpeningId::virt(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::InstructionInputVirtualization,
        );
        let right_is_rs2 = OpeningId::virt(
            VirtualPolynomial::InstructionFlags(InstructionFlags::RightOperandIsRs2Value),
            SumcheckId::InstructionInputVirtualization,
        );
        let rs2_value = OpeningId::virt(
            VirtualPolynomial::Rs2Value,
            SumcheckId::InstructionInputVirtualization,
        );
        let right_is_imm = OpeningId::virt(
            VirtualPolynomial::InstructionFlags(InstructionFlags::RightOperandIsImm),
            SumcheckId::InstructionInputVirtualization,
        );
        let imm = OpeningId::virt(
            VirtualPolynomial::Imm,
            SumcheckId::InstructionInputVirtualization,
        );

        let e2 = ValueSource::Challenge(0);
        let gamma_e2 = ValueSource::Challenge(1);

        let terms = vec![
            // E2 * right_is_rs2 * rs2_value
            ProductTerm::product(vec![
                e2.clone(),
                ValueSource::Opening(right_is_rs2),
                ValueSource::Opening(rs2_value),
            ]),
            // E2 * right_is_imm * imm
            ProductTerm::product(vec![
                e2,
                ValueSource::Opening(right_is_imm),
                ValueSource::Opening(imm),
            ]),
            // γ*E2 * left_is_rs1 * rs1_value
            ProductTerm::product(vec![
                gamma_e2.clone(),
                ValueSource::Opening(left_is_rs1),
                ValueSource::Opening(rs1_value),
            ]),
            // γ*E2 * left_is_pc * unexpanded_pc
            ProductTerm::product(vec![
                gamma_e2,
                ValueSource::Opening(left_is_pc),
                ValueSource::Opening(unexpanded_pc),
            ]),
        ];

        Some(OutputClaimConstraint::sum_of_products(terms))
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        let r = self.normalize_opening_point(sumcheck_challenges);
        let e2 = EqPolynomial::mle_endian(&r, &self.r_cycle_stage_2);
        vec![e2, self.gamma * e2]
    }
}

const MAX_SVO_ROUNDS: usize = 3;

/// For Fp128, retains small coefficients through the first three low-to-high
/// binds and materializes the field representation at one eighth the length.
#[derive(Allocative)]
struct SvoPolynomial<T: SmallScalar, F: JoltField> {
    coeffs: Vec<T>,
    bound_coeffs: Vec<F>,
    delayed_challenges: Vec<F::Challenge>,
    len: usize,
}

impl<T: SmallScalar, F: JoltField> SvoPolynomial<T, F> {
    const MATERIALIZATION_ROUND: usize = if F::NUM_BYTES == 16 {
        MAX_SVO_ROUNDS
    } else {
        1
    };

    fn new(coeffs: Vec<T>) -> Self {
        assert!(coeffs.len().is_power_of_two());
        let len = coeffs.len();
        Self {
            coeffs,
            bound_coeffs: Vec::new(),
            delayed_challenges: Vec::with_capacity(Self::MATERIALIZATION_ROUND),
            len,
        }
    }

    #[cfg(all(test, feature = "akita"))]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn interpolate_small(a: T, b: T, r: F::Challenge) -> F {
        match a.cmp(&b) {
            Ordering::Equal => a.to_field(),
            Ordering::Less => a.to_field::<F>() + b.diff_mul_field::<F>(a, r.into()),
            Ordering::Greater => a.to_field::<F>() - a.diff_mul_field::<F>(b, r.into()),
        }
    }

    #[inline]
    fn evaluate_delayed(coeffs: &[T], challenges: &[F::Challenge], index: usize) -> F {
        if challenges.is_empty() {
            return coeffs[index].to_field();
        }

        let block_len = 1 << challenges.len();
        let block = &coeffs[index * block_len..(index + 1) * block_len];
        let mut values = [F::zero(); 1 << MAX_SVO_ROUNDS];
        let mut width = block_len / 2;
        for i in 0..width {
            values[i] = Self::interpolate_small(block[2 * i], block[2 * i + 1], challenges[0]);
        }
        for &challenge in &challenges[1..] {
            for i in 0..width / 2 {
                values[i] = values[2 * i] + (values[2 * i + 1] - values[2 * i]) * challenge;
            }
            width /= 2;
        }
        values[0]
    }

    #[inline]
    fn get_bound_coeff(&self, index: usize) -> F {
        if self.bound_coeffs.is_empty() {
            Self::evaluate_delayed(&self.coeffs, &self.delayed_challenges, index)
        } else {
            self.bound_coeffs[index]
        }
    }

    fn bind_parallel(&mut self, r: F::Challenge) {
        let n = self.len / 2;
        if self.bound_coeffs.is_empty() {
            self.delayed_challenges.push(r);
            self.len = n;
            if self.delayed_challenges.len() == Self::MATERIALIZATION_ROUND || n == 1 {
                self.bound_coeffs = (0..n)
                    .into_par_iter()
                    .map(|index| {
                        Self::evaluate_delayed(&self.coeffs, &self.delayed_challenges, index)
                    })
                    .collect();
                self.coeffs = Vec::new();
                self.delayed_challenges = Vec::new();
            }
        } else {
            let mut bound_coeffs = Vec::with_capacity(n);
            (
                bound_coeffs.spare_capacity_mut(),
                self.bound_coeffs.par_chunks_exact(2),
            )
                .into_par_iter()
                .with_min_len(512 * 32 / F::NUM_BYTES)
                .for_each(|(bound_coeff, coeffs)| {
                    bound_coeff.write(coeffs[0] + (coeffs[1] - coeffs[0]) * r);
                });
            // SAFETY: every spare-capacity element was initialized above.
            unsafe { bound_coeffs.set_len(n) };
            self.bound_coeffs = bound_coeffs;
            self.len = n;
        }
    }

    fn final_sumcheck_claim(&self) -> F {
        assert_eq!(self.len, 1);
        self.get_bound_coeff(0)
    }
}

#[derive(Allocative)]
pub struct InstructionInputSumcheckProver<F: JoltField> {
    left_is_rs1_poly: SvoPolynomial<bool, F>,
    left_is_pc_poly: SvoPolynomial<bool, F>,
    right_is_rs2_poly: SvoPolynomial<bool, F>,
    right_is_imm_poly: SvoPolynomial<bool, F>,
    rs1_value_poly: SvoPolynomial<u64, F>,
    rs2_value_poly: SvoPolynomial<u64, F>,
    imm_poly: SvoPolynomial<i128, F>,
    unexpanded_pc_poly: SvoPolynomial<u64, F>,
    eq_r_cycle_stage_2: GruenSplitEqPolynomial<F>,
    pub params: InstructionInputParams<F>,
}

impl<F: JoltField> InstructionInputSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "InstructionInputSumcheckProver::initialize")]
    pub fn initialize(
        params: InstructionInputParams<F>,
        trace: &[JoltTraceRow],
        _opening_accumulator: &ProverOpeningAccumulator<F>,
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
                    trace_row,
                )| {
                    let flags = Flags::instruction_flags(trace_row);
                    *left_is_rs1_eval = flags[InstructionFlags::LeftOperandIsRs1Value];
                    *left_is_pc_eval = flags[InstructionFlags::LeftOperandIsPC];
                    *right_is_rs2_eval = flags[InstructionFlags::RightOperandIsRs2Value];
                    *right_is_imm_eval = flags[InstructionFlags::RightOperandIsImm];
                    *rs1_value_eval = trace_row.rs1_value();
                    *rs2_value_eval = trace_row.rs2_value();
                    *imm_eval = trace_row.imm();
                    *unexpanded_pc_eval = trace_row.unexpanded_pc();
                },
            );

        let eq_r_cycle_stage_2 =
            GruenSplitEqPolynomial::new(&params.r_cycle_stage_2.r, BindingOrder::LowToHigh);

        Self {
            left_is_rs1_poly: SvoPolynomial::new(left_is_rs1_poly),
            left_is_pc_poly: SvoPolynomial::new(left_is_pc_poly),
            right_is_rs2_poly: SvoPolynomial::new(right_is_rs2_poly),
            right_is_imm_poly: SvoPolynomial::new(right_is_imm_poly),
            rs1_value_poly: SvoPolynomial::new(rs1_value_poly),
            rs2_value_poly: SvoPolynomial::new(rs2_value_poly),
            imm_poly: SvoPolynomial::new(imm_poly),
            unexpanded_pc_poly: SvoPolynomial::new(unexpanded_pc_poly),
            eq_r_cycle_stage_2,
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
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let [eval_at_0, eval_at_inf] = self
            .eq_r_cycle_stage_2
            .par_fold_out_in(
                || [F::UnreducedProductAccum::zero(); 2],
                |inner, j, _x_in, e_in| {
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
                    let right_at_j_0 =
                        right_is_rs2_at_j_0 * rs2_value_at_j_0 + right_is_imm_at_j_0 * imm_at_j_0;
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
                        self.unexpanded_pc_poly.get_bound_coeff(j * 2 + 1) - unexpanded_pc_at_j_0;
                    // Eval LeftInstructionInput(x) at (r', {0, inf}, j).
                    let left_at_j_0 = left_is_rs1_at_j_0 * rs1_value_at_j_0
                        + left_is_pc_at_j_0 * unexpanded_pc_at_j_0;
                    let left_at_j_inf = left_is_rs1_at_j_inf * rs1_value_at_j_inf
                        + left_is_pc_at_j_inf * unexpanded_pc_at_j_inf;

                    // Eval Input(x) = RightInstructionInput(x) + gamma * LeftInstructionInput(x) at (r', {0, inf}, j).
                    let input_at_j_0 = right_at_j_0 + self.params.gamma * left_at_j_0;
                    let input_at_j_inf = right_at_j_inf + self.params.gamma * left_at_j_inf;

                    // Accumulate in Montgomery-unreduced form to minimize reductions
                    inner[0] += e_in.mul_to_product_accum(input_at_j_0);
                    inner[1] += e_in.mul_to_product_accum(input_at_j_inf);
                },
                |_x_out, e_out, inner| {
                    let mut out = [F::UnreducedProductAccum::zero(); 2];
                    let reduced0 = F::reduce_product_accum(inner[0]);
                    let reduced1 = F::reduce_product_accum(inner[1]);
                    out[0] = e_out.mul_to_product_accum(reduced0);
                    out[1] = e_out.mul_to_product_accum(reduced1);
                    out
                },
                |mut a, b| {
                    for i in 0..2 {
                        a[i] += b[i];
                    }
                    a
                },
            )
            .map(|x| F::reduce_product_accum(x));

        self.eq_r_cycle_stage_2
            .gruen_poly_deg_3(eval_at_0, eval_at_inf, previous_claim)
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
            eq_r_cycle_stage_2,
            params: _,
        } = self;
        left_is_rs1_poly.bind_parallel(r_j);
        left_is_pc_poly.bind_parallel(r_j);
        right_is_rs2_poly.bind_parallel(r_j);
        right_is_imm_poly.bind_parallel(r_j);
        rs1_value_poly.bind_parallel(r_j);
        rs2_value_poly.bind_parallel(r_j);
        imm_poly.bind_parallel(r_j);
        unexpanded_pc_poly.bind_parallel(r_j);
        eq_r_cycle_stage_2.bind(r_j);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            VirtualPolynomial::InstructionFlags(InstructionFlags::LeftOperandIsRs1Value),
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
            self.left_is_rs1_poly.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            VirtualPolynomial::Rs1Value,
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
            self.rs1_value_poly.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            VirtualPolynomial::InstructionFlags(InstructionFlags::LeftOperandIsPC),
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
            self.left_is_pc_poly.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
            self.unexpanded_pc_poly.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            VirtualPolynomial::InstructionFlags(InstructionFlags::RightOperandIsRs2Value),
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
            self.right_is_rs2_poly.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            VirtualPolynomial::Rs2Value,
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
            self.rs2_value_poly.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            VirtualPolynomial::InstructionFlags(InstructionFlags::RightOperandIsImm),
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
            self.right_is_imm_poly.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
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
/// sum_j eq(r_cycle_stage_2, j) * (RightInstructionInput(j) + gamma * LeftInstructionInput(j))
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
/// - `r_cycle_stage_2` is the randomness from instruction product sumcheck (stage 2).
pub struct InstructionInputSumcheckVerifier<F: JoltField> {
    params: InstructionInputParams<F>,
}

impl<F: JoltField> InstructionInputSumcheckVerifier<F> {
    pub fn new<A: AbstractVerifierOpeningAccumulator<F>>(
        opening_accumulator: &A,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params = InstructionInputParams::new(opening_accumulator, transcript);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript, A: AbstractVerifierOpeningAccumulator<F>>
    SumcheckInstanceVerifier<F, T, A> for InstructionInputSumcheckVerifier<F>
{
    fn input_claim(&self, accumulator: &A) -> F {
        let result = self.params.input_claim(accumulator);

        #[cfg(test)]
        {
            let claims = Self::input_output_claims();
            let gamma_pows: Vec<F> =
                std::iter::successors(Some(F::one()), |prev| Some(*prev * self.params.gamma))
                    .take(claims.claims.len())
                    .collect();
            let reference_result = claims.input_claim(&gamma_pows, accumulator);
            assert_eq!(result, reference_result);
        }

        result
    }

    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(&self, accumulator: &A, sumcheck_challenges: &[F::Challenge]) -> F {
        let r = self.params.normalize_opening_point(sumcheck_challenges);

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

        let result = eq_eval_at_r_cycle_stage_2
            * (right_instruction_input + self.params.gamma * left_instruction_input);

        #[cfg(test)]
        {
            let claims = Self::input_output_claims();
            let gamma_pows: Vec<F> =
                std::iter::successors(Some(F::one()), |prev| Some(*prev * self.params.gamma))
                    .take(claims.claims.len())
                    .collect();
            let reference_result = claims.expected_output_claim(&r, &gamma_pows, accumulator);

            assert_eq!(result, reference_result);
        }

        result
    }

    fn cache_openings(&self, accumulator: &mut A, sumcheck_challenges: &[F::Challenge]) {
        let r = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            VirtualPolynomial::InstructionFlags(InstructionFlags::LeftOperandIsRs1Value),
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
        );
        accumulator.append_virtual(
            VirtualPolynomial::Rs1Value,
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
        );
        accumulator.append_virtual(
            VirtualPolynomial::InstructionFlags(InstructionFlags::LeftOperandIsPC),
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
        );
        accumulator.append_virtual(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
        );
        accumulator.append_virtual(
            VirtualPolynomial::InstructionFlags(InstructionFlags::RightOperandIsRs2Value),
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
        );
        accumulator.append_virtual(
            VirtualPolynomial::Rs2Value,
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
        );
        accumulator.append_virtual(
            VirtualPolynomial::InstructionFlags(InstructionFlags::RightOperandIsImm),
            SumcheckId::InstructionInputVirtualization,
            r.clone(),
        );
        accumulator.append_virtual(
            VirtualPolynomial::Imm,
            SumcheckId::InstructionInputVirtualization,
            r,
        );
    }
}

impl<F: JoltField> SumcheckFrontend<F> for InstructionInputSumcheckVerifier<F> {
    fn input_output_claims() -> InputOutputClaims<F> {
        let right_instruction_input: ClaimExpr<F> = VirtualPolynomial::RightInstructionInput.into();
        let left_instruction_input: ClaimExpr<F> = VirtualPolynomial::LeftInstructionInput.into();

        let rs1_value: ClaimExpr<F> = VirtualPolynomial::Rs1Value.into();
        let left_is_rs1: ClaimExpr<F> =
            VirtualPolynomial::InstructionFlags(InstructionFlags::LeftOperandIsRs1Value).into();
        let unexpanded_pc: ClaimExpr<F> = VirtualPolynomial::UnexpandedPC.into();
        let left_is_pc: ClaimExpr<F> =
            VirtualPolynomial::InstructionFlags(InstructionFlags::LeftOperandIsPC).into();
        let rs2_value: ClaimExpr<F> = VirtualPolynomial::Rs2Value.into();
        let right_is_rs2: ClaimExpr<F> =
            VirtualPolynomial::InstructionFlags(InstructionFlags::RightOperandIsRs2Value).into();
        let imm: ClaimExpr<F> = VirtualPolynomial::Imm.into();
        let right_is_imm: ClaimExpr<F> =
            VirtualPolynomial::InstructionFlags(InstructionFlags::RightOperandIsImm).into();

        let left_instruction_input_eval = left_is_rs1 * rs1_value + left_is_pc * unexpanded_pc;
        let right_instruction_input_eval = right_is_rs2 * rs2_value + right_is_imm * imm;

        let eq_r_stage2 = VerifierEvaluablePolynomial::Eq(CachedPointRef {
            opening: PolynomialId::Virtual(VirtualPolynomial::LeftInstructionInput),
            sumcheck: SumcheckId::SpartanProductVirtualization,
            part: ChallengePart::Cycle,
        });

        InputOutputClaims {
            claims: vec![
                Claim {
                    input_sumcheck_id: SumcheckId::SpartanProductVirtualization,
                    input_claim_expr: right_instruction_input,
                    batching_poly: eq_r_stage2,
                    expected_output_claim_expr: right_instruction_input_eval,
                },
                Claim {
                    input_sumcheck_id: SumcheckId::SpartanProductVirtualization,
                    input_claim_expr: left_instruction_input,
                    batching_poly: eq_r_stage2,
                    expected_output_claim_expr: left_instruction_input_eval,
                },
            ],
            output_sumcheck_id: SumcheckId::InstructionInputVirtualization,
        }
    }
}

#[cfg(all(test, feature = "akita"))]
mod tests {
    use super::*;
    use crate::field::akita::AkitaFp128;
    use crate::poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialBinding};

    fn svo_matches_compact<T>(coeffs: Vec<T>)
    where
        T: crate::utils::small_scalar::SmallScalar,
        MultilinearPolynomial<AkitaFp128>: From<Vec<T>>,
    {
        let num_vars = coeffs.len().ilog2() as usize;
        let mut expected = MultilinearPolynomial::<AkitaFp128>::from(coeffs.clone());
        let mut actual = SvoPolynomial::<T, AkitaFp128>::new(coeffs);
        let challenges = [3, 5, 11, 19, 27, 41]
            .map(<AkitaFp128 as JoltField>::from_u64)
            .into_iter()
            .take(num_vars);

        for challenge in challenges {
            assert_eq!(actual.len(), expected.len());
            for index in 0..actual.len() {
                assert_eq!(
                    actual.get_bound_coeff(index),
                    expected.get_bound_coeff(index)
                );
            }
            actual.bind_parallel(challenge);
            expected.bind_parallel(challenge, BindingOrder::LowToHigh);
        }

        assert_eq!(
            actual.final_sumcheck_claim(),
            expected.final_sumcheck_claim()
        );
    }

    #[test]
    fn svo_polynomial_matches_compact_bindings() {
        svo_matches_compact(vec![false, true]);
        svo_matches_compact(vec![2u64, 7, 1, 8]);
        svo_matches_compact(vec![3i128, -5, 8, 13, -21, 34, 55, -89]);
        svo_matches_compact((0..64).map(|i| i % 7 == 0).collect::<Vec<_>>());
        svo_matches_compact(
            (0..64)
                .map(|i| ((i * 0x9e37) ^ (i >> 2)) as u64)
                .collect::<Vec<_>>(),
        );
        svo_matches_compact(
            (0..64)
                .map(|i| (i as i128 - 17) * (i as i128 + 3))
                .collect::<Vec<_>>(),
        );
    }
}
