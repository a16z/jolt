use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

use super::super::{
    DoryAssistCopyConstraint, DoryAssistOpeningId, DoryAssistPublicId, DoryAssistRelationClaims,
    DoryAssistRelationId, DoryAssistValueRef, DoryAssistValueType, DoryAssistVirtualPolynomial,
    G1Polynomial, G2Polynomial, GtPolynomial, MillerLoopPolynomial,
};
use super::dimensions::{
    DoryAssistDimensions, DoryAssistFormulaDimensionsError, PrefixPackingDimensions,
};
use super::{artifacts, dory_reduce, g1, g2, gt, miller_loop, packing, wiring};

const LOCAL_ROW: usize = 0;
const NEXT_ROW: usize = 1;
pub const PAIR_PRODUCT_GT_MUL_ROW: usize = 0;
pub const ACCUMULATOR_MUL_GT_ROW: usize = 1;
pub const ACCUMULATOR_SQUARE_GT_ROW: usize = 2;
pub const GT_MULTIPLICATION_ROWS: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistPackingEntry {
    pub opening: DoryAssistOpeningId,
    pub native_vars: usize,
}

impl DoryAssistPackingEntry {
    pub const fn new(opening: DoryAssistOpeningId, native_vars: usize) -> Self {
        Self {
            opening,
            native_vars,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistPackingCatalog {
    entries: Vec<DoryAssistPackingEntry>,
}

impl DoryAssistPackingCatalog {
    pub fn entries(&self) -> &[DoryAssistPackingEntry] {
        &self.entries
    }

    pub fn openings(&self) -> Vec<DoryAssistOpeningId> {
        self.entries.iter().map(|entry| entry.opening).collect()
    }

    pub fn num_claims(&self) -> usize {
        self.entries.len()
    }

    pub fn max_poly_vars(&self) -> usize {
        self.entries
            .iter()
            .map(|entry| entry.native_vars)
            .max()
            .unwrap_or(0)
    }

    pub fn minimal_dimensions(
        &self,
    ) -> Result<PrefixPackingDimensions, DoryAssistFormulaDimensionsError> {
        let max_poly_vars = self.max_poly_vars();
        let prefix_vars = ceil_log2_usize(self.num_claims());
        PrefixPackingDimensions::new(
            max_poly_vars + prefix_vars,
            max_poly_vars,
            self.num_claims(),
        )
    }

    fn extend_openings<I>(&mut self, native_vars: usize, openings: I)
    where
        I: IntoIterator<Item = DoryAssistOpeningId>,
    {
        for opening in openings {
            self.push(DoryAssistPackingEntry::new(opening, native_vars));
        }
    }

    fn push(&mut self, new_entry: DoryAssistPackingEntry) {
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|entry| entry.opening == new_entry.opening)
        {
            entry.native_vars = entry.native_vars.max(new_entry.native_vars);
        } else {
            self.entries.push(new_entry);
        }
    }
}

pub fn prefix_packing_catalog(dimensions: DoryAssistDimensions) -> DoryAssistPackingCatalog {
    let mut catalog = DoryAssistPackingCatalog::default();

    catalog.extend_openings(
        dimensions.gt.exp_batched_rounds(),
        gt::exponentiation_input_openings(),
    );
    catalog.extend_openings(
        dimensions.gt.exp_batched_rounds(),
        gt::exponentiation_output_openings(),
    );
    catalog.extend_openings(
        dimensions.gt.exp_batched_rounds(),
        gt::exponentiation_digit_selector_input_openings(),
    );
    catalog.extend_openings(
        dimensions.gt.exp_batched_rounds(),
        gt::exponentiation_digit_selector_output_openings(),
    );
    catalog.extend_openings(
        dimensions.gt.exp_base_power_rounds(),
        gt::exponentiation_base_power_output_openings(),
    );
    catalog.extend_openings(
        dimensions.gt.exp_batched_rounds(),
        gt::exponentiation_digit_bitness_output_openings(),
    );
    catalog.extend_openings(
        dimensions.gt.exp_batched_rounds(),
        gt::exponentiation_shift_input_openings(),
    );
    catalog.extend_openings(
        dimensions.gt.exp_batched_rounds(),
        gt::exponentiation_shift_output_openings(),
    );
    catalog.extend_openings(
        dimensions.gt.exp_batched_rounds(),
        gt::exponentiation_boundary_output_openings(),
    );
    catalog.extend_openings(
        dimensions.gt.mul_rounds(),
        gt::multiplication_input_openings(),
    );
    catalog.extend_openings(
        dimensions.gt.mul_rounds(),
        gt::multiplication_output_openings(),
    );

    catalog.extend_openings(
        dimensions.g1.scalar_mul_rounds(),
        g1::scalar_multiplication_input_openings(),
    );
    catalog.extend_openings(
        dimensions.g1.scalar_mul_shift_sumcheck().rounds,
        g1::scalar_multiplication_shift_input_openings(),
    );
    catalog.extend_openings(
        dimensions.g1.scalar_mul_shift_sumcheck().rounds,
        g1::scalar_multiplication_shift_output_openings(),
    );
    catalog.extend_openings(
        dimensions.g1.scalar_mul_rounds(),
        g1::scalar_multiplication_boundary_output_openings(),
    );
    catalog.extend_openings(dimensions.g1.add_rounds(), g1::addition_input_openings());

    catalog.extend_openings(
        dimensions.g2.scalar_mul_rounds(),
        g2::scalar_multiplication_input_openings(),
    );
    catalog.extend_openings(
        dimensions.g2.scalar_mul_shift_sumcheck().rounds,
        g2::scalar_multiplication_shift_input_openings(),
    );
    catalog.extend_openings(
        dimensions.g2.scalar_mul_shift_sumcheck().rounds,
        g2::scalar_multiplication_shift_output_openings(),
    );
    catalog.extend_openings(
        dimensions.g2.scalar_mul_rounds(),
        g2::scalar_multiplication_boundary_output_openings(),
    );
    catalog.extend_openings(dimensions.g2.add_rounds(), g2::addition_input_openings());

    catalog.extend_openings(
        dimensions.miller_loop.line_step_rounds(),
        miller_loop::line_step_output_openings(),
    );
    catalog.extend_openings(
        dimensions.miller_loop.line_evaluation_rounds(),
        miller_loop::line_evaluation_output_openings(),
    );
    catalog.extend_openings(
        dimensions.miller_loop.pair_product_rounds(),
        miller_loop::pair_product_output_openings(),
    );
    catalog.extend_openings(
        dimensions.miller_loop.accumulator_rounds(),
        miller_loop::accumulator_input_openings(),
    );
    catalog.extend_openings(
        dimensions.miller_loop.accumulator_rounds(),
        miller_loop::accumulator_output_openings(),
    );
    catalog.extend_openings(
        dimensions.miller_loop.boundary_rounds(),
        miller_loop::boundary_output_openings(),
    );

    catalog.extend_openings(
        dimensions.dory_reduce.reduce_round_vars(),
        dory_reduce::gt_transition_openings(),
    );
    catalog.extend_openings(
        dimensions.dory_reduce.reduce_round_vars(),
        dory_reduce::g1_transition_openings(),
    );
    catalog.extend_openings(
        dimensions.dory_reduce.reduce_round_vars(),
        dory_reduce::g2_transition_openings(),
    );
    catalog.extend_openings(
        dimensions.dory_reduce.reduce_round_vars(),
        dory_reduce::scalar_fold_input_openings(),
    );
    catalog.extend_openings(
        dimensions.dory_reduce.reduce_round_vars(),
        dory_reduce::scalar_fold_output_openings(),
    );
    if dimensions.dory_reduce.reduce_rounds() > 1 {
        catalog.extend_openings(
            dimensions.dory_reduce.reduce_round_vars(),
            dory_reduce::state_chain_input_openings(),
        );
        catalog.extend_openings(
            dimensions.dory_reduce.reduce_round_vars(),
            dory_reduce::state_chain_output_openings(),
        );
        catalog.extend_openings(
            dimensions.dory_reduce.reduce_round_vars(),
            dory_reduce::boundary_output_openings(),
        );
    }

    catalog.extend_openings(
        dimensions.wiring.log_edges(),
        wiring::copy_zero_check_output_openings(DoryAssistRelationId::WiringGt),
    );
    catalog.extend_openings(
        dimensions.wiring.log_edges(),
        wiring::copy_zero_check_output_openings(DoryAssistRelationId::WiringG1),
    );
    catalog.extend_openings(
        dimensions.wiring.log_edges(),
        wiring::copy_zero_check_output_openings(DoryAssistRelationId::WiringG2),
    );

    for edge in public_input_copy_constraints()
        .into_iter()
        .chain(gt_copy_constraints())
        .chain(g1_copy_constraints())
        .chain(g2_copy_constraints())
        .chain(miller_loop_copy_constraints())
    {
        for endpoint in [edge.source, edge.target] {
            if let Some(opening) = endpoint.witness_opening() {
                catalog.push(DoryAssistPackingEntry::new(
                    opening,
                    native_vars_for_value_ref(dimensions, endpoint),
                ));
            }
        }
    }

    catalog
}

pub fn prefix_packing_claims<F>(
    dimensions: PrefixPackingDimensions,
    catalog: &DoryAssistPackingCatalog,
) -> DoryAssistRelationClaims<F>
where
    F: RingCore,
{
    packing::prefix_packing(dimensions, catalog.openings())
}

pub fn gt_copy_constraints() -> Vec<DoryAssistCopyConstraint> {
    let mut constraints = Vec::new();

    constraints.push(copy_constraint(
        DoryAssistValueType::Scalar,
        gt_relation_ref(
            DoryAssistRelationId::GtExponentiation,
            GtPolynomial::ExpDigitSelector,
        ),
        gt_relation_ref(
            DoryAssistRelationId::GtExponentiationDigitSelector,
            GtPolynomial::ExpDigitSelector,
        ),
    ));

    for bit_index in 0..2 {
        constraints.push(copy_constraint(
            DoryAssistValueType::Scalar,
            gt_relation_ref(
                DoryAssistRelationId::GtExponentiationDigitSelector,
                GtPolynomial::ExpDigitBit(bit_index),
            ),
            gt_relation_ref(
                DoryAssistRelationId::GtExponentiationDigitBitness,
                GtPolynomial::ExpDigitBit(bit_index),
            ),
        ));
    }

    for power in 1..=3 {
        constraints.push(copy_constraint(
            DoryAssistValueType::Scalar,
            gt_relation_ref(
                DoryAssistRelationId::GtExponentiationDigitSelector,
                GtPolynomial::ExpBasePower(power),
            ),
            gt_relation_ref(
                DoryAssistRelationId::GtExponentiationBasePower,
                GtPolynomial::ExpBasePower(power),
            ),
        ));
    }

    constraints.push(copy_constraint(
        DoryAssistValueType::Scalar,
        gt_relation_ref(
            DoryAssistRelationId::GtExponentiation,
            GtPolynomial::Modulus,
        ),
        gt_relation_ref(
            DoryAssistRelationId::GtExponentiationBasePower,
            GtPolynomial::Modulus,
        ),
    ));
    constraints.push(copy_constraint(
        DoryAssistValueType::Scalar,
        gt_relation_ref(
            DoryAssistRelationId::GtExponentiation,
            GtPolynomial::ExpAccumulator,
        ),
        gt_relation_ref(
            DoryAssistRelationId::GtExponentiationShift,
            GtPolynomial::ExpAccumulator,
        ),
    ));
    constraints.push(copy_constraint(
        DoryAssistValueType::Scalar,
        gt_relation_ref(
            DoryAssistRelationId::GtExponentiation,
            GtPolynomial::ExpAccumulator,
        ),
        gt_relation_ref(
            DoryAssistRelationId::GtExponentiationBoundary,
            GtPolynomial::ExpAccumulator,
        ),
    ));
    constraints.push(copy_constraint(
        DoryAssistValueType::Scalar,
        gt_relation_ref(
            DoryAssistRelationId::GtExponentiation,
            GtPolynomial::ExpShiftedAccumulator,
        ),
        gt_relation_ref(
            DoryAssistRelationId::GtExponentiationBoundary,
            GtPolynomial::ExpShiftedAccumulator,
        ),
    ));

    constraints
}

pub fn public_input_copy_constraints() -> Vec<DoryAssistCopyConstraint> {
    let mut constraints = vec![copy_constraint(
        DoryAssistValueType::Scalar,
        DoryAssistValueRef::public(
            DoryAssistPublicId::DoryProofArtifact(artifacts::DORY_VMV_C_START),
            0,
        ),
        gt_relation_ref(
            DoryAssistRelationId::GtExponentiation,
            GtPolynomial::ExpAccumulator,
        ),
    )];

    for (component, polynomial) in [
        (0, MillerLoopPolynomial::G1PointX),
        (1, MillerLoopPolynomial::G1PointY),
    ] {
        constraints.push(copy_constraint(
            DoryAssistValueType::G1,
            DoryAssistValueRef::public(
                DoryAssistPublicId::DoryProofArtifact(artifacts::DORY_VMV_E1_START + component),
                component,
            ),
            miller_ref(
                DoryAssistRelationId::MillerLoopLineEvaluation,
                polynomial,
                LOCAL_ROW,
                component,
            ),
        ));
    }

    constraints
}

pub fn g1_copy_constraints() -> Vec<DoryAssistCopyConstraint> {
    let mut constraints = Vec::new();

    for (polynomial, component) in [
        (G1Polynomial::ScalarMulAccumulatorX, 0),
        (G1Polynomial::ScalarMulAccumulatorY, 1),
        (G1Polynomial::ScalarMulShiftedAccumulatorX, 0),
        (G1Polynomial::ScalarMulShiftedAccumulatorY, 1),
    ] {
        constraints.push(copy_constraint(
            DoryAssistValueType::G1,
            g1_ref(
                DoryAssistRelationId::G1ScalarMultiplication,
                polynomial,
                component,
            ),
            g1_ref(
                DoryAssistRelationId::G1ScalarMultiplicationShift,
                polynomial,
                component,
            ),
        ));
    }

    for (polynomial, component) in [
        (G1Polynomial::ScalarMulAccumulatorX, 0),
        (G1Polynomial::ScalarMulAccumulatorY, 1),
        (G1Polynomial::ScalarMulAccumulatorInfinity, 2),
        (G1Polynomial::ScalarMulShiftedAccumulatorX, 0),
        (G1Polynomial::ScalarMulShiftedAccumulatorY, 1),
    ] {
        constraints.push(copy_constraint(
            DoryAssistValueType::G1,
            g1_ref(
                DoryAssistRelationId::G1ScalarMultiplication,
                polynomial,
                component,
            ),
            g1_ref(
                DoryAssistRelationId::G1ScalarMultiplicationBoundary,
                polynomial,
                component,
            ),
        ));
    }

    constraints
}

pub fn g2_copy_constraints() -> Vec<DoryAssistCopyConstraint> {
    let mut constraints = Vec::new();

    for (polynomial, component) in [
        (G2Polynomial::ScalarMulAccumulatorX0, 0),
        (G2Polynomial::ScalarMulAccumulatorX1, 1),
        (G2Polynomial::ScalarMulAccumulatorY0, 2),
        (G2Polynomial::ScalarMulAccumulatorY1, 3),
        (G2Polynomial::ScalarMulShiftedAccumulatorX0, 0),
        (G2Polynomial::ScalarMulShiftedAccumulatorX1, 1),
        (G2Polynomial::ScalarMulShiftedAccumulatorY0, 2),
        (G2Polynomial::ScalarMulShiftedAccumulatorY1, 3),
    ] {
        constraints.push(copy_constraint(
            DoryAssistValueType::G2,
            g2_ref(
                DoryAssistRelationId::G2ScalarMultiplication,
                polynomial,
                component,
            ),
            g2_ref(
                DoryAssistRelationId::G2ScalarMultiplicationShift,
                polynomial,
                component,
            ),
        ));
    }

    for (polynomial, component) in [
        (G2Polynomial::ScalarMulAccumulatorX0, 0),
        (G2Polynomial::ScalarMulAccumulatorX1, 1),
        (G2Polynomial::ScalarMulAccumulatorY0, 2),
        (G2Polynomial::ScalarMulAccumulatorY1, 3),
        (G2Polynomial::ScalarMulAccumulatorInfinity, 4),
        (G2Polynomial::ScalarMulShiftedAccumulatorX0, 0),
        (G2Polynomial::ScalarMulShiftedAccumulatorX1, 1),
        (G2Polynomial::ScalarMulShiftedAccumulatorY0, 2),
        (G2Polynomial::ScalarMulShiftedAccumulatorY1, 3),
    ] {
        constraints.push(copy_constraint(
            DoryAssistValueType::G2,
            g2_ref(
                DoryAssistRelationId::G2ScalarMultiplication,
                polynomial,
                component,
            ),
            g2_ref(
                DoryAssistRelationId::G2ScalarMultiplicationBoundary,
                polynomial,
                component,
            ),
        ));
    }

    constraints
}

pub fn miller_loop_line_copy_constraints() -> Vec<DoryAssistCopyConstraint> {
    let mut constraints = Vec::new();
    extend_line_step_state_shift_constraints(&mut constraints);
    extend_line_step_to_line_evaluation_constraints(&mut constraints);
    constraints
}

pub fn miller_loop_active_copy_constraints() -> Vec<DoryAssistCopyConstraint> {
    let mut constraints = miller_loop_line_copy_constraints();
    extend_pair_product_active_constraints(&mut constraints);
    extend_accumulator_active_constraints(&mut constraints);
    extend_boundary_output_constraints(&mut constraints);
    constraints
}

pub fn native_vars_for_relation(
    dimensions: DoryAssistDimensions,
    relation: DoryAssistRelationId,
) -> usize {
    match relation {
        DoryAssistRelationId::GtExponentiation
        | DoryAssistRelationId::GtExponentiationDigitSelector
        | DoryAssistRelationId::GtExponentiationDigitBitness
        | DoryAssistRelationId::GtExponentiationShift
        | DoryAssistRelationId::GtExponentiationBoundary => dimensions.gt.exp_batched_rounds(),
        DoryAssistRelationId::GtExponentiationBasePower => dimensions.gt.exp_base_power_rounds(),
        DoryAssistRelationId::GtMultiplication => dimensions.gt.mul_rounds(),
        DoryAssistRelationId::G1ScalarMultiplication
        | DoryAssistRelationId::G1ScalarMultiplicationBoundary => dimensions.g1.scalar_mul_rounds(),
        DoryAssistRelationId::G1ScalarMultiplicationShift => {
            dimensions.g1.scalar_mul_shift_sumcheck().rounds
        }
        DoryAssistRelationId::G1Addition => dimensions.g1.add_rounds(),
        DoryAssistRelationId::G2ScalarMultiplication
        | DoryAssistRelationId::G2ScalarMultiplicationBoundary => dimensions.g2.scalar_mul_rounds(),
        DoryAssistRelationId::G2ScalarMultiplicationShift => {
            dimensions.g2.scalar_mul_shift_sumcheck().rounds
        }
        DoryAssistRelationId::G2Addition => dimensions.g2.add_rounds(),
        DoryAssistRelationId::MillerLoopLineStep => dimensions.miller_loop.line_step_rounds(),
        DoryAssistRelationId::MillerLoopLineEvaluation => {
            dimensions.miller_loop.line_evaluation_rounds()
        }
        DoryAssistRelationId::MillerLoopPairProduct => dimensions.miller_loop.pair_product_rounds(),
        DoryAssistRelationId::MillerLoopAccumulator => dimensions.miller_loop.accumulator_rounds(),
        DoryAssistRelationId::MillerLoopBoundary => dimensions.miller_loop.boundary_rounds(),
        DoryAssistRelationId::WiringGt
        | DoryAssistRelationId::WiringG1
        | DoryAssistRelationId::WiringG2 => dimensions.wiring.log_edges(),
        DoryAssistRelationId::DoryReduceGtTransition
        | DoryAssistRelationId::DoryReduceG1Transition
        | DoryAssistRelationId::DoryReduceG2Transition
        | DoryAssistRelationId::DoryReduceStateChain
        | DoryAssistRelationId::DoryReduceBoundary => dimensions.dory_reduce.reduce_round_vars(),
        DoryAssistRelationId::DoryReduceScalarFold => dimensions.dory_reduce.reduce_round_vars(),
        DoryAssistRelationId::PrefixPacking => dimensions.packing.packed_vars(),
    }
}

pub fn miller_loop_copy_constraints() -> Vec<DoryAssistCopyConstraint> {
    let mut constraints = Vec::new();
    extend_line_step_state_shift_constraints(&mut constraints);
    extend_line_step_to_line_evaluation_constraints(&mut constraints);
    extend_pair_product_gt_multiplication_constraints(&mut constraints);
    extend_accumulator_gt_multiplication_constraints(&mut constraints);
    extend_boundary_output_constraints(&mut constraints);
    constraints
}

fn extend_line_step_state_shift_constraints(constraints: &mut Vec<DoryAssistCopyConstraint>) {
    for (source, target, component) in [
        (
            MillerLoopPolynomial::G2LineShiftedStateX0,
            MillerLoopPolynomial::G2LineStateX0,
            0,
        ),
        (
            MillerLoopPolynomial::G2LineShiftedStateX1,
            MillerLoopPolynomial::G2LineStateX1,
            1,
        ),
        (
            MillerLoopPolynomial::G2LineShiftedStateY0,
            MillerLoopPolynomial::G2LineStateY0,
            0,
        ),
        (
            MillerLoopPolynomial::G2LineShiftedStateY1,
            MillerLoopPolynomial::G2LineStateY1,
            1,
        ),
        (
            MillerLoopPolynomial::G2LineShiftedStateZ0,
            MillerLoopPolynomial::G2LineStateZ0,
            0,
        ),
        (
            MillerLoopPolynomial::G2LineShiftedStateZ1,
            MillerLoopPolynomial::G2LineStateZ1,
            1,
        ),
    ] {
        constraints.push(copy_constraint(
            DoryAssistValueType::Fq2,
            miller_ref(
                DoryAssistRelationId::MillerLoopLineStep,
                source,
                LOCAL_ROW,
                component,
            ),
            miller_ref(
                DoryAssistRelationId::MillerLoopLineStep,
                target,
                NEXT_ROW,
                component,
            ),
        ));
    }
}

fn extend_line_step_to_line_evaluation_constraints(
    constraints: &mut Vec<DoryAssistCopyConstraint>,
) {
    for coefficient in 0..miller_loop::MILLER_LOOP_LINE_COEFFICIENTS {
        for component in 0..2 {
            constraints.push(copy_constraint(
                DoryAssistValueType::Fq2,
                miller_ref(
                    DoryAssistRelationId::MillerLoopLineStep,
                    MillerLoopPolynomial::LineCoefficient {
                        coefficient,
                        component,
                    },
                    LOCAL_ROW,
                    component,
                ),
                miller_ref(
                    DoryAssistRelationId::MillerLoopLineEvaluation,
                    MillerLoopPolynomial::LineCoefficient {
                        coefficient,
                        component,
                    },
                    LOCAL_ROW,
                    component,
                ),
            ));
        }
    }
}

fn extend_pair_product_gt_multiplication_constraints(
    constraints: &mut Vec<DoryAssistCopyConstraint>,
) {
    for component in 0..miller_loop::MILLER_LOOP_GT_COEFFS {
        constraints.push(copy_constraint(
            DoryAssistValueType::Gt,
            miller_ref(
                DoryAssistRelationId::MillerLoopPairProduct,
                MillerLoopPolynomial::PairProductAccumulatorCoeff(component),
                LOCAL_ROW,
                component,
            ),
            gt_ref(GtPolynomial::MulLeft, PAIR_PRODUCT_GT_MUL_ROW, component),
        ));
        constraints.push(copy_constraint(
            DoryAssistValueType::Gt,
            miller_ref(
                DoryAssistRelationId::MillerLoopLineEvaluation,
                MillerLoopPolynomial::LineEvaluationCoeff(component),
                LOCAL_ROW,
                component,
            ),
            gt_ref(GtPolynomial::MulRight, PAIR_PRODUCT_GT_MUL_ROW, component),
        ));
        constraints.push(copy_constraint(
            DoryAssistValueType::Gt,
            gt_ref(GtPolynomial::MulOutput, PAIR_PRODUCT_GT_MUL_ROW, component),
            miller_ref(
                DoryAssistRelationId::MillerLoopPairProduct,
                MillerLoopPolynomial::PairProductShiftedAccumulatorCoeff(component),
                LOCAL_ROW,
                component,
            ),
        ));
        constraints.push(copy_constraint(
            DoryAssistValueType::Gt,
            miller_ref(
                DoryAssistRelationId::MillerLoopPairProduct,
                MillerLoopPolynomial::PairProductQuotientCoeff(component),
                LOCAL_ROW,
                component,
            ),
            gt_ref(
                GtPolynomial::MulQuotient,
                PAIR_PRODUCT_GT_MUL_ROW,
                component,
            ),
        ));
    }
}

fn extend_accumulator_gt_multiplication_constraints(
    constraints: &mut Vec<DoryAssistCopyConstraint>,
) {
    for component in 0..miller_loop::MILLER_LOOP_GT_COEFFS {
        constraints.push(copy_constraint(
            DoryAssistValueType::Gt,
            miller_ref(
                DoryAssistRelationId::MillerLoopAccumulator,
                MillerLoopPolynomial::AccumulatorCoeff(component),
                LOCAL_ROW,
                component,
            ),
            gt_ref(GtPolynomial::MulLeft, ACCUMULATOR_SQUARE_GT_ROW, component),
        ));
        constraints.push(copy_constraint(
            DoryAssistValueType::Gt,
            miller_ref(
                DoryAssistRelationId::MillerLoopAccumulator,
                MillerLoopPolynomial::AccumulatorCoeff(component),
                LOCAL_ROW,
                component,
            ),
            gt_ref(GtPolynomial::MulRight, ACCUMULATOR_SQUARE_GT_ROW, component),
        ));
        constraints.push(copy_constraint(
            DoryAssistValueType::Gt,
            gt_ref(
                GtPolynomial::MulOutput,
                ACCUMULATOR_SQUARE_GT_ROW,
                component,
            ),
            miller_ref(
                DoryAssistRelationId::MillerLoopAccumulator,
                MillerLoopPolynomial::ShiftedAccumulatorCoeff(component),
                LOCAL_ROW,
                component,
            ),
        ));
        constraints.push(copy_constraint(
            DoryAssistValueType::Gt,
            miller_ref(
                DoryAssistRelationId::MillerLoopAccumulator,
                MillerLoopPolynomial::AccumulatorQuotientCoeff(component),
                LOCAL_ROW,
                component,
            ),
            gt_ref(
                GtPolynomial::MulQuotient,
                ACCUMULATOR_SQUARE_GT_ROW,
                component,
            ),
        ));

        constraints.push(copy_constraint(
            DoryAssistValueType::Gt,
            miller_ref(
                DoryAssistRelationId::MillerLoopAccumulator,
                MillerLoopPolynomial::AccumulatorCoeff(component),
                LOCAL_ROW,
                component,
            ),
            gt_ref(GtPolynomial::MulLeft, ACCUMULATOR_MUL_GT_ROW, component),
        ));
        constraints.push(copy_constraint(
            DoryAssistValueType::Gt,
            miller_ref(
                DoryAssistRelationId::MillerLoopPairProduct,
                MillerLoopPolynomial::PairLineProductCoeff(component),
                LOCAL_ROW,
                component,
            ),
            gt_ref(GtPolynomial::MulRight, ACCUMULATOR_MUL_GT_ROW, component),
        ));
        constraints.push(copy_constraint(
            DoryAssistValueType::Gt,
            gt_ref(GtPolynomial::MulOutput, ACCUMULATOR_MUL_GT_ROW, component),
            miller_ref(
                DoryAssistRelationId::MillerLoopAccumulator,
                MillerLoopPolynomial::ShiftedAccumulatorCoeff(component),
                LOCAL_ROW,
                component,
            ),
        ));
        constraints.push(copy_constraint(
            DoryAssistValueType::Gt,
            miller_ref(
                DoryAssistRelationId::MillerLoopAccumulator,
                MillerLoopPolynomial::AccumulatorQuotientCoeff(component),
                LOCAL_ROW,
                component,
            ),
            gt_ref(GtPolynomial::MulQuotient, ACCUMULATOR_MUL_GT_ROW, component),
        ));
    }
}

fn extend_pair_product_active_constraints(constraints: &mut Vec<DoryAssistCopyConstraint>) {
    for component in 0..miller_loop::MILLER_LOOP_GT_COEFFS {
        constraints.push(copy_constraint(
            DoryAssistValueType::Gt,
            miller_ref(
                DoryAssistRelationId::MillerLoopPairProduct,
                MillerLoopPolynomial::PairProductAccumulatorCoeff(component),
                LOCAL_ROW,
                component,
            ),
            gt_ref(GtPolynomial::MulLeft, PAIR_PRODUCT_GT_MUL_ROW, component),
        ));
        constraints.push(copy_constraint(
            DoryAssistValueType::Gt,
            miller_ref(
                DoryAssistRelationId::MillerLoopLineEvaluation,
                MillerLoopPolynomial::LineEvaluationCoeff(component),
                LOCAL_ROW,
                component,
            ),
            gt_ref(GtPolynomial::MulRight, PAIR_PRODUCT_GT_MUL_ROW, component),
        ));
        constraints.push(copy_constraint(
            DoryAssistValueType::Gt,
            gt_ref(GtPolynomial::MulOutput, PAIR_PRODUCT_GT_MUL_ROW, component),
            miller_ref(
                DoryAssistRelationId::MillerLoopPairProduct,
                MillerLoopPolynomial::PairProductShiftedAccumulatorCoeff(component),
                LOCAL_ROW,
                component,
            ),
        ));
    }
}

fn extend_accumulator_active_constraints(constraints: &mut Vec<DoryAssistCopyConstraint>) {
    for component in 0..miller_loop::MILLER_LOOP_GT_COEFFS {
        constraints.push(copy_constraint(
            DoryAssistValueType::Gt,
            miller_ref(
                DoryAssistRelationId::MillerLoopAccumulator,
                MillerLoopPolynomial::AccumulatorCoeff(component),
                LOCAL_ROW,
                component,
            ),
            gt_ref(GtPolynomial::MulLeft, ACCUMULATOR_SQUARE_GT_ROW, component),
        ));
        constraints.push(copy_constraint(
            DoryAssistValueType::Gt,
            miller_ref(
                DoryAssistRelationId::MillerLoopAccumulator,
                MillerLoopPolynomial::AccumulatorCoeff(component),
                LOCAL_ROW,
                component,
            ),
            gt_ref(GtPolynomial::MulRight, ACCUMULATOR_SQUARE_GT_ROW, component),
        ));
        constraints.push(copy_constraint(
            DoryAssistValueType::Gt,
            gt_ref(
                GtPolynomial::MulOutput,
                ACCUMULATOR_SQUARE_GT_ROW,
                component,
            ),
            miller_ref(
                DoryAssistRelationId::MillerLoopAccumulator,
                MillerLoopPolynomial::ShiftedAccumulatorCoeff(component),
                LOCAL_ROW,
                component,
            ),
        ));
        constraints.push(copy_constraint(
            DoryAssistValueType::Gt,
            miller_ref(
                DoryAssistRelationId::MillerLoopAccumulator,
                MillerLoopPolynomial::AccumulatorCoeff(component),
                LOCAL_ROW,
                component,
            ),
            gt_ref(GtPolynomial::MulLeft, ACCUMULATOR_MUL_GT_ROW, component),
        ));
        constraints.push(copy_constraint(
            DoryAssistValueType::Gt,
            miller_ref(
                DoryAssistRelationId::MillerLoopPairProduct,
                MillerLoopPolynomial::PairLineProductCoeff(component),
                LOCAL_ROW,
                component,
            ),
            gt_ref(GtPolynomial::MulRight, ACCUMULATOR_MUL_GT_ROW, component),
        ));
        constraints.push(copy_constraint(
            DoryAssistValueType::Gt,
            gt_ref(GtPolynomial::MulOutput, ACCUMULATOR_MUL_GT_ROW, component),
            miller_ref(
                DoryAssistRelationId::MillerLoopAccumulator,
                MillerLoopPolynomial::ShiftedAccumulatorCoeff(component),
                LOCAL_ROW,
                component,
            ),
        ));
    }
}

fn extend_boundary_output_constraints(constraints: &mut Vec<DoryAssistCopyConstraint>) {
    for component in 0..miller_loop::MILLER_LOOP_GT_COEFFS {
        constraints.push(copy_constraint(
            DoryAssistValueType::Gt,
            miller_ref(
                DoryAssistRelationId::MillerLoopAccumulator,
                MillerLoopPolynomial::AccumulatorCoeff(component),
                LOCAL_ROW,
                component,
            ),
            miller_ref(
                DoryAssistRelationId::MillerLoopBoundary,
                MillerLoopPolynomial::AccumulatorCoeff(component),
                LOCAL_ROW,
                component,
            ),
        ));
        constraints.push(copy_constraint(
            DoryAssistValueType::Gt,
            miller_ref(
                DoryAssistRelationId::MillerLoopAccumulator,
                MillerLoopPolynomial::ShiftedAccumulatorCoeff(component),
                LOCAL_ROW,
                component,
            ),
            miller_ref(
                DoryAssistRelationId::MillerLoopBoundary,
                MillerLoopPolynomial::ShiftedAccumulatorCoeff(component),
                LOCAL_ROW,
                component,
            ),
        ));
        constraints.push(copy_constraint(
            DoryAssistValueType::Gt,
            miller_ref(
                DoryAssistRelationId::MillerLoopBoundary,
                MillerLoopPolynomial::ShiftedAccumulatorCoeff(component),
                LOCAL_ROW,
                component,
            ),
            DoryAssistValueRef::public(
                DoryAssistPublicId::MillerLoopOutputGt(component),
                component,
            ),
        ));
    }
}

fn native_vars_for_value_ref(
    dimensions: DoryAssistDimensions,
    value_ref: DoryAssistValueRef,
) -> usize {
    match value_ref {
        DoryAssistValueRef::Witness { relation, .. } => {
            native_vars_for_relation(dimensions, relation)
        }
        DoryAssistValueRef::Public { .. }
        | DoryAssistValueRef::Challenge(_)
        | DoryAssistValueRef::Constant(_) => 0,
    }
}

fn copy_constraint(
    value_type: DoryAssistValueType,
    source: DoryAssistValueRef,
    target: DoryAssistValueRef,
) -> DoryAssistCopyConstraint {
    DoryAssistCopyConstraint::new(value_type, source, target)
}

fn miller_ref(
    relation: DoryAssistRelationId,
    polynomial: MillerLoopPolynomial,
    row: usize,
    component: usize,
) -> DoryAssistValueRef {
    DoryAssistValueRef::witness(
        relation,
        DoryAssistVirtualPolynomial::MillerLoop(polynomial),
        row,
        component,
    )
}

fn gt_ref(polynomial: GtPolynomial, row: usize, component: usize) -> DoryAssistValueRef {
    DoryAssistValueRef::witness(
        DoryAssistRelationId::GtMultiplication,
        DoryAssistVirtualPolynomial::Gt(polynomial),
        row,
        component,
    )
}

fn gt_relation_ref(relation: DoryAssistRelationId, polynomial: GtPolynomial) -> DoryAssistValueRef {
    DoryAssistValueRef::witness(
        relation,
        DoryAssistVirtualPolynomial::Gt(polynomial),
        LOCAL_ROW,
        0,
    )
}

fn g1_ref(
    relation: DoryAssistRelationId,
    polynomial: G1Polynomial,
    component: usize,
) -> DoryAssistValueRef {
    DoryAssistValueRef::witness(
        relation,
        DoryAssistVirtualPolynomial::G1(polynomial),
        LOCAL_ROW,
        component,
    )
}

fn g2_ref(
    relation: DoryAssistRelationId,
    polynomial: G2Polynomial,
    component: usize,
) -> DoryAssistValueRef {
    DoryAssistValueRef::witness(
        relation,
        DoryAssistVirtualPolynomial::G2(polynomial),
        LOCAL_ROW,
        component,
    )
}

fn ceil_log2_usize(value: usize) -> usize {
    if value <= 1 {
        0
    } else {
        usize::BITS as usize - (value - 1).leading_zeros() as usize
    }
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::unwrap_used,
        reason = "tests fail loudly on invalid fixture dimensions"
    )]

    use std::collections::BTreeSet;

    use jolt_field::Fr;

    use super::super::super::DoryAssistPolynomialId;
    use super::super::dimensions::{
        DoryReduceDimensions, G1Dimensions, G2Dimensions, GtDimensions, MillerLoopDimensions,
        WiringDimensions,
    };
    use super::*;

    fn dimensions() -> DoryAssistDimensions {
        DoryAssistDimensions::new(
            GtDimensions::new(7, 2, 3),
            G1Dimensions::new(8, 2, 3),
            G2Dimensions::new(8, 2, 3),
            MillerLoopDimensions::new(7, 2, 8),
            DoryReduceDimensions::new(2, 1),
            WiringDimensions::new(6),
            PrefixPackingDimensions::new(0, 0, 0).unwrap(),
        )
    }

    fn multi_round_dory_reduce_dimensions() -> DoryAssistDimensions {
        DoryAssistDimensions::new(
            GtDimensions::new(7, 2, 3),
            G1Dimensions::new(8, 2, 3),
            G2Dimensions::new(8, 2, 3),
            MillerLoopDimensions::new(7, 2, 8),
            DoryReduceDimensions::new(4, 2),
            WiringDimensions::new(6),
            PrefixPackingDimensions::new(0, 0, 0).unwrap(),
        )
    }

    #[test]
    fn prefix_packing_catalog_spans_all_component_families_without_duplicates() {
        let catalog = prefix_packing_catalog(dimensions());
        let openings = catalog.openings();
        let unique_openings = openings.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(openings.len(), unique_openings.len());
        for expected_relation in [
            DoryAssistRelationId::GtExponentiation,
            DoryAssistRelationId::GtMultiplication,
            DoryAssistRelationId::G1ScalarMultiplication,
            DoryAssistRelationId::G1Addition,
            DoryAssistRelationId::G2ScalarMultiplication,
            DoryAssistRelationId::G2Addition,
            DoryAssistRelationId::MillerLoopLineStep,
            DoryAssistRelationId::MillerLoopLineEvaluation,
            DoryAssistRelationId::MillerLoopPairProduct,
            DoryAssistRelationId::MillerLoopAccumulator,
            DoryAssistRelationId::MillerLoopBoundary,
            DoryAssistRelationId::DoryReduceScalarFold,
            DoryAssistRelationId::WiringGt,
            DoryAssistRelationId::WiringG1,
            DoryAssistRelationId::WiringG2,
        ] {
            assert!(
                openings.iter().any(|opening| matches!(
                    opening,
                    DoryAssistOpeningId::Polynomial { relation, .. }
                        if *relation == expected_relation
                )),
                "missing relation {expected_relation:?}"
            );
        }
    }

    #[test]
    fn minimal_prefix_packing_dimensions_fit_the_catalog() {
        let catalog = prefix_packing_catalog(dimensions());
        let packing_dimensions = catalog.minimal_dimensions().unwrap();

        assert_eq!(packing_dimensions.max_poly_vars(), catalog.max_poly_vars());
        assert_eq!(packing_dimensions.num_claims(), catalog.num_claims());
        assert!((1usize << packing_dimensions.prefix_vars()) >= catalog.num_claims());
        if packing_dimensions.prefix_vars() > 0 {
            assert!((1usize << (packing_dimensions.prefix_vars() - 1)) < catalog.num_claims());
        }
    }

    #[test]
    fn prefix_packing_catalog_includes_dory_reduce_state_chain_and_boundary_when_multiround() {
        let catalog = prefix_packing_catalog(multi_round_dory_reduce_dimensions());
        let openings = catalog.openings();

        assert!(openings.iter().any(|opening| matches!(
            opening,
            DoryAssistOpeningId::Polynomial {
                relation: DoryAssistRelationId::DoryReduceStateChain,
                ..
            }
        )));
        assert!(openings.iter().any(|opening| matches!(
            opening,
            DoryAssistOpeningId::Polynomial {
                relation: DoryAssistRelationId::DoryReduceBoundary,
                ..
            }
        )));
    }

    #[test]
    fn prefix_packing_claims_use_the_catalog_opening_order() {
        let catalog = prefix_packing_catalog(dimensions());
        let packing_dimensions = catalog.minimal_dimensions().unwrap();
        let claims = prefix_packing_claims::<Fr>(packing_dimensions, &catalog);

        assert_eq!(claims.id, DoryAssistRelationId::PrefixPacking);
        assert_eq!(
            claims.input.required_openings,
            vec![packing::dense_witness_opening()]
        );
        assert_eq!(claims.output.required_openings, catalog.openings());
        assert_eq!(
            claims.output.required_publics,
            (0..catalog.num_claims())
                .map(DoryAssistPublicId::PrefixPackingWeight)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn public_input_copy_constraints_bind_dory_artifacts_to_trace_endpoints() {
        let constraints = public_input_copy_constraints();

        assert_eq!(
            constraints,
            vec![
                copy_constraint(
                    DoryAssistValueType::Scalar,
                    DoryAssistValueRef::public(
                        DoryAssistPublicId::DoryProofArtifact(artifacts::DORY_VMV_C_START),
                        0,
                    ),
                    gt_relation_ref(
                        DoryAssistRelationId::GtExponentiation,
                        GtPolynomial::ExpAccumulator,
                    ),
                ),
                copy_constraint(
                    DoryAssistValueType::G1,
                    DoryAssistValueRef::public(
                        DoryAssistPublicId::DoryProofArtifact(artifacts::DORY_VMV_E1_START),
                        0,
                    ),
                    miller_ref(
                        DoryAssistRelationId::MillerLoopLineEvaluation,
                        MillerLoopPolynomial::G1PointX,
                        LOCAL_ROW,
                        0,
                    ),
                ),
                copy_constraint(
                    DoryAssistValueType::G1,
                    DoryAssistValueRef::public(
                        DoryAssistPublicId::DoryProofArtifact(artifacts::DORY_VMV_E1_START + 1),
                        1,
                    ),
                    miller_ref(
                        DoryAssistRelationId::MillerLoopLineEvaluation,
                        MillerLoopPolynomial::G1PointY,
                        LOCAL_ROW,
                        1,
                    ),
                ),
            ]
        );
    }

    #[test]
    fn gt_copy_constraints_cover_stage1_operation_wiring() {
        let constraints = gt_copy_constraints();

        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::Scalar,
            gt_relation_ref(
                DoryAssistRelationId::GtExponentiation,
                GtPolynomial::ExpDigitSelector,
            ),
            gt_relation_ref(
                DoryAssistRelationId::GtExponentiationDigitSelector,
                GtPolynomial::ExpDigitSelector,
            ),
        )));
        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::Scalar,
            gt_relation_ref(
                DoryAssistRelationId::GtExponentiationDigitSelector,
                GtPolynomial::ExpDigitBit(0),
            ),
            gt_relation_ref(
                DoryAssistRelationId::GtExponentiationDigitBitness,
                GtPolynomial::ExpDigitBit(0),
            ),
        )));
        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::Scalar,
            gt_relation_ref(
                DoryAssistRelationId::GtExponentiationDigitSelector,
                GtPolynomial::ExpBasePower(3),
            ),
            gt_relation_ref(
                DoryAssistRelationId::GtExponentiationBasePower,
                GtPolynomial::ExpBasePower(3),
            ),
        )));
        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::Scalar,
            gt_relation_ref(
                DoryAssistRelationId::GtExponentiation,
                GtPolynomial::ExpAccumulator,
            ),
            gt_relation_ref(
                DoryAssistRelationId::GtExponentiationShift,
                GtPolynomial::ExpAccumulator,
            ),
        )));
        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::Scalar,
            gt_relation_ref(
                DoryAssistRelationId::GtExponentiation,
                GtPolynomial::ExpShiftedAccumulator,
            ),
            gt_relation_ref(
                DoryAssistRelationId::GtExponentiationBoundary,
                GtPolynomial::ExpShiftedAccumulator,
            ),
        )));
    }

    #[test]
    fn g1_copy_constraints_cover_scalar_mul_reductions() {
        let constraints = g1_copy_constraints();

        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::G1,
            g1_ref(
                DoryAssistRelationId::G1ScalarMultiplication,
                G1Polynomial::ScalarMulAccumulatorX,
                0,
            ),
            g1_ref(
                DoryAssistRelationId::G1ScalarMultiplicationShift,
                G1Polynomial::ScalarMulAccumulatorX,
                0,
            ),
        )));
        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::G1,
            g1_ref(
                DoryAssistRelationId::G1ScalarMultiplication,
                G1Polynomial::ScalarMulShiftedAccumulatorY,
                1,
            ),
            g1_ref(
                DoryAssistRelationId::G1ScalarMultiplicationShift,
                G1Polynomial::ScalarMulShiftedAccumulatorY,
                1,
            ),
        )));
        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::G1,
            g1_ref(
                DoryAssistRelationId::G1ScalarMultiplication,
                G1Polynomial::ScalarMulAccumulatorInfinity,
                2,
            ),
            g1_ref(
                DoryAssistRelationId::G1ScalarMultiplicationBoundary,
                G1Polynomial::ScalarMulAccumulatorInfinity,
                2,
            ),
        )));
        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::G1,
            g1_ref(
                DoryAssistRelationId::G1ScalarMultiplication,
                G1Polynomial::ScalarMulShiftedAccumulatorX,
                0,
            ),
            g1_ref(
                DoryAssistRelationId::G1ScalarMultiplicationBoundary,
                G1Polynomial::ScalarMulShiftedAccumulatorX,
                0,
            ),
        )));
    }

    #[test]
    fn g2_copy_constraints_cover_scalar_mul_reductions() {
        let constraints = g2_copy_constraints();

        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::G2,
            g2_ref(
                DoryAssistRelationId::G2ScalarMultiplication,
                G2Polynomial::ScalarMulAccumulatorX0,
                0,
            ),
            g2_ref(
                DoryAssistRelationId::G2ScalarMultiplicationShift,
                G2Polynomial::ScalarMulAccumulatorX0,
                0,
            ),
        )));
        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::G2,
            g2_ref(
                DoryAssistRelationId::G2ScalarMultiplication,
                G2Polynomial::ScalarMulShiftedAccumulatorY1,
                3,
            ),
            g2_ref(
                DoryAssistRelationId::G2ScalarMultiplicationShift,
                G2Polynomial::ScalarMulShiftedAccumulatorY1,
                3,
            ),
        )));
        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::G2,
            g2_ref(
                DoryAssistRelationId::G2ScalarMultiplication,
                G2Polynomial::ScalarMulAccumulatorInfinity,
                4,
            ),
            g2_ref(
                DoryAssistRelationId::G2ScalarMultiplicationBoundary,
                G2Polynomial::ScalarMulAccumulatorInfinity,
                4,
            ),
        )));
        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::G2,
            g2_ref(
                DoryAssistRelationId::G2ScalarMultiplication,
                G2Polynomial::ScalarMulShiftedAccumulatorX1,
                1,
            ),
            g2_ref(
                DoryAssistRelationId::G2ScalarMultiplicationBoundary,
                G2Polynomial::ScalarMulShiftedAccumulatorX1,
                1,
            ),
        )));
    }

    #[test]
    fn miller_loop_copy_constraints_cover_core_composition_edges() {
        let constraints = miller_loop_copy_constraints();

        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::Fq2,
            miller_ref(
                DoryAssistRelationId::MillerLoopLineStep,
                MillerLoopPolynomial::G2LineShiftedStateX0,
                LOCAL_ROW,
                0,
            ),
            miller_ref(
                DoryAssistRelationId::MillerLoopLineStep,
                MillerLoopPolynomial::G2LineStateX0,
                NEXT_ROW,
                0,
            ),
        )));
        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::Fq2,
            miller_ref(
                DoryAssistRelationId::MillerLoopLineStep,
                MillerLoopPolynomial::LineCoefficient {
                    coefficient: 0,
                    component: 0,
                },
                LOCAL_ROW,
                0,
            ),
            miller_ref(
                DoryAssistRelationId::MillerLoopLineEvaluation,
                MillerLoopPolynomial::LineCoefficient {
                    coefficient: 0,
                    component: 0,
                },
                LOCAL_ROW,
                0,
            ),
        )));
        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::Gt,
            miller_ref(
                DoryAssistRelationId::MillerLoopLineEvaluation,
                MillerLoopPolynomial::LineEvaluationCoeff(0),
                LOCAL_ROW,
                0,
            ),
            gt_ref(GtPolynomial::MulRight, PAIR_PRODUCT_GT_MUL_ROW, 0),
        )));
        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::Gt,
            miller_ref(
                DoryAssistRelationId::MillerLoopPairProduct,
                MillerLoopPolynomial::PairProductAccumulatorCoeff(0),
                LOCAL_ROW,
                0,
            ),
            gt_ref(GtPolynomial::MulLeft, PAIR_PRODUCT_GT_MUL_ROW, 0),
        )));
        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::Gt,
            miller_ref(
                DoryAssistRelationId::MillerLoopPairProduct,
                MillerLoopPolynomial::PairLineProductCoeff(0),
                LOCAL_ROW,
                0,
            ),
            gt_ref(GtPolynomial::MulRight, ACCUMULATOR_MUL_GT_ROW, 0),
        )));
        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::Gt,
            miller_ref(
                DoryAssistRelationId::MillerLoopAccumulator,
                MillerLoopPolynomial::AccumulatorCoeff(0),
                LOCAL_ROW,
                0,
            ),
            gt_ref(GtPolynomial::MulRight, ACCUMULATOR_SQUARE_GT_ROW, 0),
        )));
        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::Gt,
            miller_ref(
                DoryAssistRelationId::MillerLoopBoundary,
                MillerLoopPolynomial::ShiftedAccumulatorCoeff(0),
                LOCAL_ROW,
                0,
            ),
            DoryAssistValueRef::public(DoryAssistPublicId::MillerLoopOutputGt(0), 0),
        )));
    }

    #[test]
    fn miller_loop_line_copy_constraints_are_the_enabled_line_subset() {
        let constraints = miller_loop_line_copy_constraints();

        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::Fq2,
            miller_ref(
                DoryAssistRelationId::MillerLoopLineStep,
                MillerLoopPolynomial::G2LineShiftedStateX0,
                LOCAL_ROW,
                0,
            ),
            miller_ref(
                DoryAssistRelationId::MillerLoopLineStep,
                MillerLoopPolynomial::G2LineStateX0,
                NEXT_ROW,
                0,
            ),
        )));
        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::Fq2,
            miller_ref(
                DoryAssistRelationId::MillerLoopLineStep,
                MillerLoopPolynomial::LineCoefficient {
                    coefficient: 0,
                    component: 0,
                },
                LOCAL_ROW,
                0,
            ),
            miller_ref(
                DoryAssistRelationId::MillerLoopLineEvaluation,
                MillerLoopPolynomial::LineCoefficient {
                    coefficient: 0,
                    component: 0,
                },
                LOCAL_ROW,
                0,
            ),
        )));
        assert!(constraints
            .iter()
            .all(|constraint| constraint.source.witness_opening().is_some()
                && constraint.target.witness_opening().is_some()));
    }

    #[test]
    fn miller_loop_active_copy_constraints_cover_enabled_non_quotient_edges() {
        let constraints = miller_loop_active_copy_constraints();

        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::Gt,
            miller_ref(
                DoryAssistRelationId::MillerLoopPairProduct,
                MillerLoopPolynomial::PairProductAccumulatorCoeff(0),
                LOCAL_ROW,
                0,
            ),
            gt_ref(GtPolynomial::MulLeft, PAIR_PRODUCT_GT_MUL_ROW, 0),
        )));
        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::Gt,
            miller_ref(
                DoryAssistRelationId::MillerLoopAccumulator,
                MillerLoopPolynomial::AccumulatorCoeff(0),
                LOCAL_ROW,
                0,
            ),
            gt_ref(GtPolynomial::MulLeft, ACCUMULATOR_SQUARE_GT_ROW, 0),
        )));
        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::Gt,
            miller_ref(
                DoryAssistRelationId::MillerLoopBoundary,
                MillerLoopPolynomial::ShiftedAccumulatorCoeff(0),
                LOCAL_ROW,
                0,
            ),
            DoryAssistValueRef::public(DoryAssistPublicId::MillerLoopOutputGt(0), 0),
        )));
        assert!(!constraints.iter().any(|constraint| {
            [constraint.source, constraint.target]
                .iter()
                .any(|endpoint| {
                    matches!(
                        endpoint.witness_opening(),
                        Some(DoryAssistOpeningId::Polynomial {
                            polynomial: DoryAssistPolynomialId::Virtual(
                                DoryAssistVirtualPolynomial::MillerLoop(
                                    MillerLoopPolynomial::PairProductQuotientCoeff(_)
                                        | MillerLoopPolynomial::AccumulatorQuotientCoeff(_),
                                ),
                            ),
                            ..
                        })
                    )
                })
        }));
    }

    #[test]
    fn copy_constraint_witness_endpoints_are_in_the_packing_catalog() {
        let catalog_openings = prefix_packing_catalog(dimensions())
            .openings()
            .into_iter()
            .collect::<BTreeSet<_>>();

        for constraint in public_input_copy_constraints()
            .into_iter()
            .chain(gt_copy_constraints())
            .chain(g1_copy_constraints())
            .chain(g2_copy_constraints())
            .chain(miller_loop_copy_constraints())
        {
            for endpoint in [constraint.source, constraint.target] {
                if let Some(opening) = endpoint.witness_opening() {
                    assert!(
                        catalog_openings.contains(&opening),
                        "missing copy endpoint opening {opening:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn copy_constraints_are_directed_equality_edges() {
        let constraints = public_input_copy_constraints()
            .into_iter()
            .chain(gt_copy_constraints())
            .chain(g1_copy_constraints())
            .chain(g2_copy_constraints())
            .chain(miller_loop_copy_constraints())
            .collect::<Vec<_>>();
        let unique_constraints = constraints.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(constraints.len(), unique_constraints.len());
        assert!(constraints
            .iter()
            .all(|constraint| constraint.source != constraint.target));
    }
}
