use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

use super::super::{
    DoryAssistCopyConstraint, DoryAssistOpeningId, DoryAssistPublicId, DoryAssistRelationClaims,
    DoryAssistRelationId, DoryAssistValueRef, DoryAssistValueType, DoryAssistVirtualPolynomial,
    GtPolynomial, MillerLoopPolynomial,
};
use super::dimensions::{
    DoryAssistDimensions, DoryAssistFormulaDimensionsError, PrefixPackingDimensions,
};
use super::{g1, g2, gt, miller_loop, packing, wiring};

const LOCAL_ROW: usize = 0;
const NEXT_ROW: usize = 1;
const PAIR_PRODUCT_GT_MUL_ROW: usize = 0;
const ACCUMULATOR_MUL_GT_ROW: usize = 1;
const ACCUMULATOR_SQUARE_GT_ROW: usize = 2;

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

    for edge in miller_loop_copy_constraints() {
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

    use super::super::dimensions::{
        G1Dimensions, G2Dimensions, GtDimensions, MillerLoopDimensions, WiringDimensions,
    };
    use super::*;

    fn dimensions() -> DoryAssistDimensions {
        DoryAssistDimensions::new(
            GtDimensions::new(7, 2, 3),
            G1Dimensions::new(8, 2, 3),
            G2Dimensions::new(8, 2, 3),
            MillerLoopDimensions::new(7, 2, 8),
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
    fn copy_constraint_witness_endpoints_are_in_the_packing_catalog() {
        let catalog_openings = prefix_packing_catalog(dimensions())
            .openings()
            .into_iter()
            .collect::<BTreeSet<_>>();

        for constraint in miller_loop_copy_constraints() {
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
        let constraints = miller_loop_copy_constraints();
        let unique_constraints = constraints.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(constraints.len(), unique_constraints.len());
        assert!(constraints
            .iter()
            .all(|constraint| constraint.source != constraint.target));
    }
}
