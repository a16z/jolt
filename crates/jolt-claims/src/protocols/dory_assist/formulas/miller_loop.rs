use jolt_field::{FromPrimitiveInt, RingCore};

use crate::{challenge, constant, opening, public};

use super::super::{
    DoryAssistBoundaryEndpoint, DoryAssistChallengeId, DoryAssistExpr, DoryAssistOpeningId,
    DoryAssistPublicId, DoryAssistRelationClaims, DoryAssistRelationId,
    DoryAssistVirtualPolynomial, MillerLoopChallenge, MillerLoopConstant, MillerLoopPolynomial,
    MillerLoopSelector,
};
use super::dimensions::{DoryAssistSumcheckSpec, MillerLoopDimensions};

pub const MILLER_LOOP_GT_COEFFS: usize = 16;
pub const MILLER_LOOP_LINE_COEFFICIENTS: usize = 3;
pub const MILLER_LOOP_LINE_EVALUATION_NONZERO_COEFFS: usize = 6;

pub const fn line_step_sumcheck(dimensions: MillerLoopDimensions) -> DoryAssistSumcheckSpec {
    dimensions.line_step_sumcheck(4)
}

pub const fn line_evaluation_sumcheck(dimensions: MillerLoopDimensions) -> DoryAssistSumcheckSpec {
    dimensions.line_evaluation_sumcheck(2)
}

pub const fn pair_product_sumcheck(dimensions: MillerLoopDimensions) -> DoryAssistSumcheckSpec {
    dimensions.pair_product_sumcheck(2)
}

pub const fn accumulator_sumcheck(dimensions: MillerLoopDimensions) -> DoryAssistSumcheckSpec {
    dimensions.accumulator_sumcheck(2)
}

pub const fn boundary_sumcheck(dimensions: MillerLoopDimensions) -> DoryAssistSumcheckSpec {
    dimensions.boundary_sumcheck()
}

pub fn line_step<F>(dimensions: MillerLoopDimensions) -> DoryAssistRelationClaims<F>
where
    F: RingCore + FromPrimitiveInt,
{
    DoryAssistRelationClaims::new(
        DoryAssistRelationId::MillerLoopLineStep,
        line_step_sumcheck(dimensions),
        constant(F::zero()),
        line_step_constraint_expression(),
    )
}

pub fn line_evaluation<F>(dimensions: MillerLoopDimensions) -> DoryAssistRelationClaims<F>
where
    F: RingCore + FromPrimitiveInt,
{
    DoryAssistRelationClaims::new(
        DoryAssistRelationId::MillerLoopLineEvaluation,
        line_evaluation_sumcheck(dimensions),
        constant(F::zero()),
        line_evaluation_constraint_expression(),
    )
}

pub fn pair_product<F>(dimensions: MillerLoopDimensions) -> DoryAssistRelationClaims<F>
where
    F: RingCore + FromPrimitiveInt,
{
    DoryAssistRelationClaims::new(
        DoryAssistRelationId::MillerLoopPairProduct,
        pair_product_sumcheck(dimensions),
        constant(F::zero()),
        pair_product_constraint_expression(),
    )
    .with_input_challenges([DoryAssistChallengeId::from(
        MillerLoopChallenge::PairProductBatch,
    )])
    .with_auxiliary_openings(pair_product_quotient_openings())
}

pub fn accumulator<F>(dimensions: MillerLoopDimensions) -> DoryAssistRelationClaims<F>
where
    F: RingCore,
{
    let relation = DoryAssistRelationId::MillerLoopAccumulator;
    let gamma = miller_loop_challenge(MillerLoopChallenge::AccumulatorBatch);
    let input = combine_coefficients(
        gamma.clone(),
        (0..MILLER_LOOP_GT_COEFFS).map(accumulator_shifted_opening),
    );
    let output = miller_loop_shift_eq_kernel(relation)
        * combine_coefficients(gamma, (0..MILLER_LOOP_GT_COEFFS).map(accumulator_opening));

    DoryAssistRelationClaims::new(relation, accumulator_sumcheck(dimensions), input, output)
        .with_auxiliary_openings(accumulator_quotient_openings())
}

pub fn boundary<F>(dimensions: MillerLoopDimensions) -> DoryAssistRelationClaims<F>
where
    F: RingCore + FromPrimitiveInt,
{
    let relation = DoryAssistRelationId::MillerLoopBoundary;
    let gamma = miller_loop_challenge(MillerLoopChallenge::BoundaryPoint);
    let initial = boundary_selector(relation, DoryAssistBoundaryEndpoint::Initial)
        * combine_boundary_differences(
            gamma.clone(),
            (0..MILLER_LOOP_GT_COEFFS).map(|index| {
                (
                    boundary_accumulator_opening(index),
                    boundary_initial_value(index),
                )
            }),
        );
    let final_value = boundary_selector(relation, DoryAssistBoundaryEndpoint::Final)
        * combine_boundary_differences(
            gamma.clone(),
            (0..MILLER_LOOP_GT_COEFFS).map(|index| {
                (
                    boundary_shifted_accumulator_opening(index),
                    output_gt(index),
                )
            }),
        );

    DoryAssistRelationClaims::new(
        relation,
        boundary_sumcheck(dimensions),
        constant(F::zero()),
        initial + gamma.pow(MILLER_LOOP_GT_COEFFS) * final_value,
    )
}

pub fn line_step_output_openings() -> Vec<DoryAssistOpeningId> {
    let relation = DoryAssistRelationId::MillerLoopLineStep;
    let mut openings = vec![
        line_state_x0_opening(relation),
        line_state_x1_opening(relation),
        line_state_y0_opening(relation),
        line_state_y1_opening(relation),
        line_state_z0_opening(relation),
        line_state_z1_opening(relation),
        line_addend_x0_opening(relation),
        line_addend_x1_opening(relation),
        line_addend_y0_opening(relation),
        line_addend_y1_opening(relation),
        line_shifted_state_x0_opening(relation),
        line_shifted_state_x1_opening(relation),
        line_shifted_state_y0_opening(relation),
        line_shifted_state_y1_opening(relation),
        line_shifted_state_z0_opening(relation),
        line_shifted_state_z1_opening(relation),
    ];
    openings.extend(line_coefficient_openings(relation));
    openings
}

pub fn line_evaluation_output_openings() -> Vec<DoryAssistOpeningId> {
    let relation = DoryAssistRelationId::MillerLoopLineEvaluation;
    let mut openings = vec![g1_point_x_opening(relation), g1_point_y_opening(relation)];
    openings.extend(line_coefficient_openings(relation));
    openings.extend((0..MILLER_LOOP_GT_COEFFS).map(line_evaluation_coeff_opening));
    openings
}

pub fn pair_product_output_openings() -> Vec<DoryAssistOpeningId> {
    let mut openings = Vec::with_capacity(3 * MILLER_LOOP_GT_COEFFS);
    openings.extend((0..MILLER_LOOP_GT_COEFFS).map(pair_product_accumulator_opening));
    openings.extend((0..MILLER_LOOP_GT_COEFFS).map(pair_product_shifted_accumulator_opening));
    openings.extend((0..MILLER_LOOP_GT_COEFFS).map(pair_line_product_opening));
    openings
}

pub fn pair_product_quotient_openings() -> Vec<DoryAssistOpeningId> {
    (0..MILLER_LOOP_GT_COEFFS)
        .map(pair_product_quotient_opening)
        .collect()
}

pub fn accumulator_input_openings() -> Vec<DoryAssistOpeningId> {
    (0..MILLER_LOOP_GT_COEFFS)
        .map(accumulator_shifted_opening)
        .collect()
}

pub fn accumulator_output_openings() -> Vec<DoryAssistOpeningId> {
    (0..MILLER_LOOP_GT_COEFFS)
        .map(accumulator_opening)
        .collect()
}

pub fn accumulator_quotient_openings() -> Vec<DoryAssistOpeningId> {
    (0..MILLER_LOOP_GT_COEFFS)
        .map(accumulator_quotient_opening)
        .collect()
}

pub fn boundary_output_openings() -> Vec<DoryAssistOpeningId> {
    let mut openings = Vec::with_capacity(2 * MILLER_LOOP_GT_COEFFS);
    openings.extend((0..MILLER_LOOP_GT_COEFFS).map(boundary_accumulator_opening));
    openings.extend((0..MILLER_LOOP_GT_COEFFS).map(boundary_shifted_accumulator_opening));
    openings
}

pub fn line_step_constraint_expression<F>() -> DoryAssistExpr<F>
where
    F: RingCore + FromPrimitiveInt,
{
    let relation = DoryAssistRelationId::MillerLoopLineStep;
    let state = line_state(relation);
    let addend = line_addend(relation);
    let shifted = line_shifted_state(relation);
    let line_coefficients = line_coefficients(relation);
    let double_selector = selector(MillerLoopSelector::LineDouble);
    let add_selector = selector(MillerLoopSelector::LineAdd);

    let (double_state, double_coefficients) = double_step(state.clone());
    let (add_state, add_coefficients) = add_step(state, addend);

    let mut constraints = Vec::new();
    constraints.extend(selected_fq2_constraints(
        shifted.x,
        double_state.x,
        add_state.x,
        double_selector.clone(),
        add_selector.clone(),
    ));
    constraints.extend(selected_fq2_constraints(
        shifted.y,
        double_state.y,
        add_state.y,
        double_selector.clone(),
        add_selector.clone(),
    ));
    constraints.extend(selected_fq2_constraints(
        shifted.z,
        double_state.z,
        add_state.z,
        double_selector.clone(),
        add_selector.clone(),
    ));
    for (actual, (double_expected, add_expected)) in line_coefficients
        .into_iter()
        .zip(double_coefficients.into_iter().zip(add_coefficients))
    {
        constraints.extend(selected_fq2_constraints(
            actual,
            double_expected,
            add_expected,
            double_selector.clone(),
            add_selector.clone(),
        ));
    }

    batch_constraints(
        miller_loop_challenge(MillerLoopChallenge::LineStepBatch),
        constraints,
    )
}

pub fn line_evaluation_constraint_expression<F>() -> DoryAssistExpr<F>
where
    F: RingCore + FromPrimitiveInt,
{
    let relation = DoryAssistRelationId::MillerLoopLineEvaluation;
    let g1_x = miller_loop_polynomial(MillerLoopPolynomial::G1PointX, relation);
    let g1_y = miller_loop_polynomial(MillerLoopPolynomial::G1PointY, relation);
    let line_coefficients = line_coefficients(relation);

    let mut constraints = vec![
        miller_loop_polynomial(MillerLoopPolynomial::LineEvaluationCoeff(0), relation)
            - line_coefficients[0].c0.clone() * g1_y.clone(),
        miller_loop_polynomial(MillerLoopPolynomial::LineEvaluationCoeff(1), relation)
            - line_coefficients[0].c1.clone() * g1_y,
        miller_loop_polynomial(MillerLoopPolynomial::LineEvaluationCoeff(2), relation)
            - line_coefficients[1].c0.clone() * g1_x.clone(),
        miller_loop_polynomial(MillerLoopPolynomial::LineEvaluationCoeff(3), relation)
            - line_coefficients[1].c1.clone() * g1_x,
        miller_loop_polynomial(MillerLoopPolynomial::LineEvaluationCoeff(4), relation)
            - line_coefficients[2].c0.clone(),
        miller_loop_polynomial(MillerLoopPolynomial::LineEvaluationCoeff(5), relation)
            - line_coefficients[2].c1.clone(),
    ];
    constraints.extend(
        (MILLER_LOOP_LINE_EVALUATION_NONZERO_COEFFS..MILLER_LOOP_GT_COEFFS).map(|index| {
            miller_loop_polynomial(MillerLoopPolynomial::LineEvaluationCoeff(index), relation)
        }),
    );

    batch_constraints(
        miller_loop_challenge(MillerLoopChallenge::LineEvaluationBatch),
        constraints,
    )
}

pub fn pair_product_constraint_expression<F>() -> DoryAssistExpr<F>
where
    F: RingCore + FromPrimitiveInt,
{
    let relation = DoryAssistRelationId::MillerLoopPairProduct;
    let gamma = miller_loop_challenge(MillerLoopChallenge::PairProductBatch);
    let shifted = combine_coefficients(
        gamma.clone(),
        (0..MILLER_LOOP_GT_COEFFS).map(pair_product_shifted_accumulator_opening),
    );
    let current = combine_coefficients(
        gamma.clone(),
        (0..MILLER_LOOP_GT_COEFFS).map(pair_product_accumulator_opening),
    );
    let line_product = combine_coefficients(
        gamma.clone(),
        (0..MILLER_LOOP_GT_COEFFS).map(pair_line_product_opening),
    );
    let initial = boundary_selector(relation, DoryAssistBoundaryEndpoint::Initial)
        * combine_boundary_differences(
            gamma.clone(),
            (0..MILLER_LOOP_GT_COEFFS).map(|index| {
                (
                    pair_product_accumulator_opening(index),
                    boundary_value(relation, DoryAssistBoundaryEndpoint::Initial, index),
                )
            }),
        );
    let final_value = boundary_selector(relation, DoryAssistBoundaryEndpoint::Final)
        * (shifted.clone() - line_product);

    shifted - miller_loop_shift_eq_kernel(relation) * current
        + gamma.clone().pow(MILLER_LOOP_GT_COEFFS) * initial
        + gamma.pow(2 * MILLER_LOOP_GT_COEFFS) * final_value
}

fn double_step<F>(
    state: G2HomState<F>,
) -> (G2HomState<F>, [Fq2Expr<F>; MILLER_LOOP_LINE_COEFFICIENTS])
where
    F: RingCore + FromPrimitiveInt,
{
    let two_inv = public(DoryAssistPublicId::MillerLoopConstant(
        MillerLoopConstant::TwoInverse,
    ));
    let a = fq2_scalar_mul(fq2_mul(state.x.clone(), state.y.clone()), two_inv.clone());
    let b = fq2_square(state.y.clone());
    let c = fq2_square(state.z.clone());
    let e = fq2_mul(
        twist_b(),
        fq2_scalar_mul(c.clone(), constant(F::from_u64(3))),
    );
    let f = fq2_scalar_mul(e.clone(), constant(F::from_u64(3)));
    let g = fq2_scalar_mul(fq2_add(b.clone(), f.clone()), two_inv);
    let h = fq2_sub(
        fq2_square(fq2_add(state.y.clone(), state.z)),
        fq2_add(b.clone(), c),
    );
    let i = fq2_sub(e.clone(), b.clone());
    let j = fq2_square(state.x);
    let e_square = fq2_square(e);

    let next = G2HomState {
        x: fq2_mul(a, fq2_sub(b.clone(), f)),
        y: fq2_sub(
            fq2_square(g),
            fq2_scalar_mul(e_square, constant(F::from_u64(3))),
        ),
        z: fq2_mul(b, h.clone()),
    };
    let coefficients = [fq2_neg(h), fq2_scalar_mul(j, constant(F::from_u64(3))), i];

    (next, coefficients)
}

fn add_step<F>(
    state: G2HomState<F>,
    addend: G2AffinePoint<F>,
) -> (G2HomState<F>, [Fq2Expr<F>; MILLER_LOOP_LINE_COEFFICIENTS])
where
    F: RingCore + FromPrimitiveInt,
{
    let theta = fq2_sub(state.y.clone(), fq2_mul(addend.y.clone(), state.z.clone()));
    let lambda = fq2_sub(state.x.clone(), fq2_mul(addend.x.clone(), state.z.clone()));
    let c = fq2_square(theta.clone());
    let d = fq2_square(lambda.clone());
    let e = fq2_mul(lambda.clone(), d.clone());
    let f = fq2_mul(state.z.clone(), c);
    let g = fq2_mul(state.x.clone(), d);
    let h = fq2_sub(
        fq2_add(e.clone(), f),
        fq2_scalar_mul(g.clone(), constant(F::from_u64(2))),
    );
    let next = G2HomState {
        x: fq2_mul(lambda.clone(), h.clone()),
        y: fq2_sub(
            fq2_mul(theta.clone(), fq2_sub(g.clone(), h)),
            fq2_mul(e.clone(), state.y),
        ),
        z: fq2_mul(state.z, e),
    };
    let j = fq2_sub(
        fq2_mul(theta.clone(), addend.x),
        fq2_mul(lambda.clone(), addend.y),
    );
    let coefficients = [lambda, fq2_neg(theta), j];

    (next, coefficients)
}

fn selected_fq2_constraints<F>(
    actual: Fq2Expr<F>,
    double_expected: Fq2Expr<F>,
    add_expected: Fq2Expr<F>,
    double_selector: DoryAssistExpr<F>,
    add_selector: DoryAssistExpr<F>,
) -> [DoryAssistExpr<F>; 2]
where
    F: RingCore,
{
    [
        double_selector.clone() * (actual.c0.clone() - double_expected.c0)
            + add_selector.clone() * (actual.c0 - add_expected.c0),
        double_selector * (actual.c1.clone() - double_expected.c1)
            + add_selector * (actual.c1 - add_expected.c1),
    ]
}

fn line_state<F>(relation: DoryAssistRelationId) -> G2HomState<F>
where
    F: RingCore,
{
    G2HomState {
        x: fq2_polynomial(
            MillerLoopPolynomial::G2LineStateX0,
            MillerLoopPolynomial::G2LineStateX1,
            relation,
        ),
        y: fq2_polynomial(
            MillerLoopPolynomial::G2LineStateY0,
            MillerLoopPolynomial::G2LineStateY1,
            relation,
        ),
        z: fq2_polynomial(
            MillerLoopPolynomial::G2LineStateZ0,
            MillerLoopPolynomial::G2LineStateZ1,
            relation,
        ),
    }
}

fn line_shifted_state<F>(relation: DoryAssistRelationId) -> G2HomState<F>
where
    F: RingCore,
{
    G2HomState {
        x: fq2_polynomial(
            MillerLoopPolynomial::G2LineShiftedStateX0,
            MillerLoopPolynomial::G2LineShiftedStateX1,
            relation,
        ),
        y: fq2_polynomial(
            MillerLoopPolynomial::G2LineShiftedStateY0,
            MillerLoopPolynomial::G2LineShiftedStateY1,
            relation,
        ),
        z: fq2_polynomial(
            MillerLoopPolynomial::G2LineShiftedStateZ0,
            MillerLoopPolynomial::G2LineShiftedStateZ1,
            relation,
        ),
    }
}

fn line_addend<F>(relation: DoryAssistRelationId) -> G2AffinePoint<F>
where
    F: RingCore,
{
    G2AffinePoint {
        x: fq2_polynomial(
            MillerLoopPolynomial::G2LineAddendX0,
            MillerLoopPolynomial::G2LineAddendX1,
            relation,
        ),
        y: fq2_polynomial(
            MillerLoopPolynomial::G2LineAddendY0,
            MillerLoopPolynomial::G2LineAddendY1,
            relation,
        ),
    }
}

fn line_coefficients<F>(
    relation: DoryAssistRelationId,
) -> [Fq2Expr<F>; MILLER_LOOP_LINE_COEFFICIENTS]
where
    F: RingCore,
{
    [
        line_coefficient(0, relation),
        line_coefficient(1, relation),
        line_coefficient(2, relation),
    ]
}

fn line_coefficient<F>(coefficient: usize, relation: DoryAssistRelationId) -> Fq2Expr<F>
where
    F: RingCore,
{
    fq2_polynomial(
        MillerLoopPolynomial::LineCoefficient {
            coefficient,
            component: 0,
        },
        MillerLoopPolynomial::LineCoefficient {
            coefficient,
            component: 1,
        },
        relation,
    )
}

fn line_coefficient_openings(relation: DoryAssistRelationId) -> Vec<DoryAssistOpeningId> {
    (0..MILLER_LOOP_LINE_COEFFICIENTS)
        .flat_map(|coefficient| {
            [
                miller_loop_opening(
                    MillerLoopPolynomial::LineCoefficient {
                        coefficient,
                        component: 0,
                    },
                    relation,
                ),
                miller_loop_opening(
                    MillerLoopPolynomial::LineCoefficient {
                        coefficient,
                        component: 1,
                    },
                    relation,
                ),
            ]
        })
        .collect()
}

fn combine_coefficients<F, I>(gamma: DoryAssistExpr<F>, openings: I) -> DoryAssistExpr<F>
where
    F: RingCore,
    I: IntoIterator<Item = DoryAssistOpeningId>,
{
    openings
        .into_iter()
        .enumerate()
        .fold(constant(F::zero()), |acc, (index, opening_id)| {
            acc + gamma.clone().pow(index) * opening(opening_id)
        })
}

fn combine_boundary_differences<F, I>(gamma: DoryAssistExpr<F>, terms: I) -> DoryAssistExpr<F>
where
    F: RingCore,
    I: IntoIterator<Item = (DoryAssistOpeningId, DoryAssistExpr<F>)>,
{
    terms.into_iter().enumerate().fold(
        constant(F::zero()),
        |acc, (index, (opening_id, expected))| {
            acc + gamma.clone().pow(index) * (opening(opening_id) - expected)
        },
    )
}

fn batch_constraints<F, I>(challenge: DoryAssistExpr<F>, constraints: I) -> DoryAssistExpr<F>
where
    F: RingCore,
    I: IntoIterator<Item = DoryAssistExpr<F>>,
{
    constraints
        .into_iter()
        .enumerate()
        .fold(constant(F::zero()), |acc, (index, constraint)| {
            acc + challenge.clone().pow(index) * constraint
        })
}

fn twist_b<F>() -> Fq2Expr<F>
where
    F: RingCore,
{
    Fq2Expr {
        c0: public(DoryAssistPublicId::MillerLoopConstant(
            MillerLoopConstant::TwistB0,
        )),
        c1: public(DoryAssistPublicId::MillerLoopConstant(
            MillerLoopConstant::TwistB1,
        )),
    }
}

fn fq2_polynomial<F>(
    c0: MillerLoopPolynomial,
    c1: MillerLoopPolynomial,
    relation: DoryAssistRelationId,
) -> Fq2Expr<F>
where
    F: RingCore,
{
    Fq2Expr {
        c0: miller_loop_polynomial(c0, relation),
        c1: miller_loop_polynomial(c1, relation),
    }
}

fn fq2_add<F>(left: Fq2Expr<F>, right: Fq2Expr<F>) -> Fq2Expr<F>
where
    F: RingCore,
{
    Fq2Expr {
        c0: left.c0 + right.c0,
        c1: left.c1 + right.c1,
    }
}

fn fq2_sub<F>(left: Fq2Expr<F>, right: Fq2Expr<F>) -> Fq2Expr<F>
where
    F: RingCore,
{
    Fq2Expr {
        c0: left.c0 - right.c0,
        c1: left.c1 - right.c1,
    }
}

fn fq2_neg<F>(value: Fq2Expr<F>) -> Fq2Expr<F>
where
    F: RingCore,
{
    Fq2Expr {
        c0: constant(F::zero()) - value.c0,
        c1: constant(F::zero()) - value.c1,
    }
}

fn fq2_mul<F>(left: Fq2Expr<F>, right: Fq2Expr<F>) -> Fq2Expr<F>
where
    F: RingCore,
{
    Fq2Expr {
        c0: left.c0.clone() * right.c0.clone() - left.c1.clone() * right.c1.clone(),
        c1: left.c0 * right.c1 + left.c1 * right.c0,
    }
}

fn fq2_square<F>(value: Fq2Expr<F>) -> Fq2Expr<F>
where
    F: RingCore,
{
    fq2_mul(value.clone(), value)
}

fn fq2_scalar_mul<F>(value: Fq2Expr<F>, scalar: DoryAssistExpr<F>) -> Fq2Expr<F>
where
    F: RingCore,
{
    Fq2Expr {
        c0: value.c0 * scalar.clone(),
        c1: value.c1 * scalar,
    }
}

fn selector<F>(selector: MillerLoopSelector) -> DoryAssistExpr<F>
where
    F: RingCore,
{
    public(DoryAssistPublicId::MillerLoopSelector {
        relation: DoryAssistRelationId::MillerLoopLineStep,
        selector,
    })
}

fn miller_loop_shift_eq_kernel<F>(relation: DoryAssistRelationId) -> DoryAssistExpr<F>
where
    F: RingCore,
{
    public(DoryAssistPublicId::MillerLoopShiftEqKernel(relation))
}

fn boundary_selector<F>(
    relation: DoryAssistRelationId,
    endpoint: DoryAssistBoundaryEndpoint,
) -> DoryAssistExpr<F>
where
    F: RingCore,
{
    public(DoryAssistPublicId::BoundarySelector { relation, endpoint })
}

fn boundary_value<F>(
    relation: DoryAssistRelationId,
    endpoint: DoryAssistBoundaryEndpoint,
    component: usize,
) -> DoryAssistExpr<F>
where
    F: RingCore,
{
    public(DoryAssistPublicId::BoundaryValue {
        relation,
        endpoint,
        component,
    })
}

fn boundary_initial_value<F>(component: usize) -> DoryAssistExpr<F>
where
    F: RingCore,
{
    boundary_value(
        DoryAssistRelationId::MillerLoopBoundary,
        DoryAssistBoundaryEndpoint::Initial,
        component,
    )
}

fn output_gt<F>(component: usize) -> DoryAssistExpr<F>
where
    F: RingCore,
{
    public(DoryAssistPublicId::MillerLoopOutputGt(component))
}

fn miller_loop_challenge<F>(id: MillerLoopChallenge) -> DoryAssistExpr<F>
where
    F: RingCore,
{
    challenge(DoryAssistChallengeId::from(id))
}

fn miller_loop_polynomial<F>(
    polynomial: MillerLoopPolynomial,
    relation: DoryAssistRelationId,
) -> DoryAssistExpr<F>
where
    F: RingCore,
{
    opening(miller_loop_opening(polynomial, relation))
}

fn miller_loop_opening(
    polynomial: MillerLoopPolynomial,
    relation: DoryAssistRelationId,
) -> DoryAssistOpeningId {
    DoryAssistOpeningId::virtual_polynomial(
        DoryAssistVirtualPolynomial::MillerLoop(polynomial),
        relation,
    )
}

fn g1_point_x_opening(relation: DoryAssistRelationId) -> DoryAssistOpeningId {
    miller_loop_opening(MillerLoopPolynomial::G1PointX, relation)
}

fn g1_point_y_opening(relation: DoryAssistRelationId) -> DoryAssistOpeningId {
    miller_loop_opening(MillerLoopPolynomial::G1PointY, relation)
}

fn line_state_x0_opening(relation: DoryAssistRelationId) -> DoryAssistOpeningId {
    miller_loop_opening(MillerLoopPolynomial::G2LineStateX0, relation)
}

fn line_state_x1_opening(relation: DoryAssistRelationId) -> DoryAssistOpeningId {
    miller_loop_opening(MillerLoopPolynomial::G2LineStateX1, relation)
}

fn line_state_y0_opening(relation: DoryAssistRelationId) -> DoryAssistOpeningId {
    miller_loop_opening(MillerLoopPolynomial::G2LineStateY0, relation)
}

fn line_state_y1_opening(relation: DoryAssistRelationId) -> DoryAssistOpeningId {
    miller_loop_opening(MillerLoopPolynomial::G2LineStateY1, relation)
}

fn line_state_z0_opening(relation: DoryAssistRelationId) -> DoryAssistOpeningId {
    miller_loop_opening(MillerLoopPolynomial::G2LineStateZ0, relation)
}

fn line_state_z1_opening(relation: DoryAssistRelationId) -> DoryAssistOpeningId {
    miller_loop_opening(MillerLoopPolynomial::G2LineStateZ1, relation)
}

fn line_addend_x0_opening(relation: DoryAssistRelationId) -> DoryAssistOpeningId {
    miller_loop_opening(MillerLoopPolynomial::G2LineAddendX0, relation)
}

fn line_addend_x1_opening(relation: DoryAssistRelationId) -> DoryAssistOpeningId {
    miller_loop_opening(MillerLoopPolynomial::G2LineAddendX1, relation)
}

fn line_addend_y0_opening(relation: DoryAssistRelationId) -> DoryAssistOpeningId {
    miller_loop_opening(MillerLoopPolynomial::G2LineAddendY0, relation)
}

fn line_addend_y1_opening(relation: DoryAssistRelationId) -> DoryAssistOpeningId {
    miller_loop_opening(MillerLoopPolynomial::G2LineAddendY1, relation)
}

fn line_shifted_state_x0_opening(relation: DoryAssistRelationId) -> DoryAssistOpeningId {
    miller_loop_opening(MillerLoopPolynomial::G2LineShiftedStateX0, relation)
}

fn line_shifted_state_x1_opening(relation: DoryAssistRelationId) -> DoryAssistOpeningId {
    miller_loop_opening(MillerLoopPolynomial::G2LineShiftedStateX1, relation)
}

fn line_shifted_state_y0_opening(relation: DoryAssistRelationId) -> DoryAssistOpeningId {
    miller_loop_opening(MillerLoopPolynomial::G2LineShiftedStateY0, relation)
}

fn line_shifted_state_y1_opening(relation: DoryAssistRelationId) -> DoryAssistOpeningId {
    miller_loop_opening(MillerLoopPolynomial::G2LineShiftedStateY1, relation)
}

fn line_shifted_state_z0_opening(relation: DoryAssistRelationId) -> DoryAssistOpeningId {
    miller_loop_opening(MillerLoopPolynomial::G2LineShiftedStateZ0, relation)
}

fn line_shifted_state_z1_opening(relation: DoryAssistRelationId) -> DoryAssistOpeningId {
    miller_loop_opening(MillerLoopPolynomial::G2LineShiftedStateZ1, relation)
}

fn line_evaluation_coeff_opening(component: usize) -> DoryAssistOpeningId {
    miller_loop_opening(
        MillerLoopPolynomial::LineEvaluationCoeff(component),
        DoryAssistRelationId::MillerLoopLineEvaluation,
    )
}

fn pair_product_accumulator_opening(component: usize) -> DoryAssistOpeningId {
    miller_loop_opening(
        MillerLoopPolynomial::PairProductAccumulatorCoeff(component),
        DoryAssistRelationId::MillerLoopPairProduct,
    )
}

fn pair_product_shifted_accumulator_opening(component: usize) -> DoryAssistOpeningId {
    miller_loop_opening(
        MillerLoopPolynomial::PairProductShiftedAccumulatorCoeff(component),
        DoryAssistRelationId::MillerLoopPairProduct,
    )
}

fn pair_line_product_opening(component: usize) -> DoryAssistOpeningId {
    miller_loop_opening(
        MillerLoopPolynomial::PairLineProductCoeff(component),
        DoryAssistRelationId::MillerLoopPairProduct,
    )
}

fn pair_product_quotient_opening(component: usize) -> DoryAssistOpeningId {
    miller_loop_opening(
        MillerLoopPolynomial::PairProductQuotientCoeff(component),
        DoryAssistRelationId::MillerLoopPairProduct,
    )
}

fn accumulator_opening(component: usize) -> DoryAssistOpeningId {
    miller_loop_opening(
        MillerLoopPolynomial::AccumulatorCoeff(component),
        DoryAssistRelationId::MillerLoopAccumulator,
    )
}

fn accumulator_shifted_opening(component: usize) -> DoryAssistOpeningId {
    miller_loop_opening(
        MillerLoopPolynomial::ShiftedAccumulatorCoeff(component),
        DoryAssistRelationId::MillerLoopAccumulator,
    )
}

fn accumulator_quotient_opening(component: usize) -> DoryAssistOpeningId {
    miller_loop_opening(
        MillerLoopPolynomial::AccumulatorQuotientCoeff(component),
        DoryAssistRelationId::MillerLoopAccumulator,
    )
}

fn boundary_accumulator_opening(component: usize) -> DoryAssistOpeningId {
    miller_loop_opening(
        MillerLoopPolynomial::AccumulatorCoeff(component),
        DoryAssistRelationId::MillerLoopBoundary,
    )
}

fn boundary_shifted_accumulator_opening(component: usize) -> DoryAssistOpeningId {
    miller_loop_opening(
        MillerLoopPolynomial::ShiftedAccumulatorCoeff(component),
        DoryAssistRelationId::MillerLoopBoundary,
    )
}

#[derive(Clone)]
struct G2HomState<F> {
    x: Fq2Expr<F>,
    y: Fq2Expr<F>,
    z: Fq2Expr<F>,
}

struct G2AffinePoint<F> {
    x: Fq2Expr<F>,
    y: Fq2Expr<F>,
}

#[derive(Clone)]
struct Fq2Expr<F> {
    c0: DoryAssistExpr<F>,
    c1: DoryAssistExpr<F>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn dimensions() -> MillerLoopDimensions {
        MillerLoopDimensions::new(7, 2, 8)
    }

    fn assert_openings_include(actual: &[DoryAssistOpeningId], expected: &[DoryAssistOpeningId]) {
        for opening in expected {
            assert!(
                actual.contains(opening),
                "missing expected opening {opening:?}"
            );
        }
    }

    #[test]
    fn line_step_claims_expose_schedule_and_curve_constants() {
        let claims = line_step::<Fr>(dimensions());

        assert_eq!(claims.id, DoryAssistRelationId::MillerLoopLineStep);
        assert_eq!(claims.sumcheck, line_step_sumcheck(dimensions()));
        assert!(claims.input.required_openings.is_empty());
        assert_openings_include(
            &claims.output.required_openings,
            &line_step_output_openings(),
        );
        assert_eq!(
            claims.required_challenges(),
            vec![DoryAssistChallengeId::from(
                MillerLoopChallenge::LineStepBatch
            )]
        );
        assert!(claims
            .required_publics()
            .contains(&DoryAssistPublicId::MillerLoopSelector {
                relation: DoryAssistRelationId::MillerLoopLineStep,
                selector: MillerLoopSelector::LineDouble,
            }));
        assert!(claims
            .required_publics()
            .contains(&DoryAssistPublicId::MillerLoopConstant(
                MillerLoopConstant::TwoInverse
            )));
    }

    #[test]
    fn line_step_accepts_zero_double_row_and_rejects_bad_shift() {
        let expr = line_step_constraint_expression::<Fr>();
        let zero = Fr::from_u64(0);

        let good = expr.evaluate(
            |_| zero,
            |_| Fr::from_u64(5),
            |public| match *public {
                DoryAssistPublicId::MillerLoopSelector {
                    selector: MillerLoopSelector::LineDouble,
                    ..
                } => Fr::from_u64(1),
                DoryAssistPublicId::MillerLoopSelector {
                    selector: MillerLoopSelector::LineAdd,
                    ..
                } => zero,
                DoryAssistPublicId::MillerLoopConstant(MillerLoopConstant::TwoInverse) => {
                    Fr::from_u64(9)
                }
                _ => zero,
            },
        );
        let bad = expr.evaluate(
            |opening| match *opening {
                id if id
                    == line_shifted_state_x0_opening(DoryAssistRelationId::MillerLoopLineStep) =>
                {
                    Fr::from_u64(1)
                }
                _ => zero,
            },
            |_| Fr::from_u64(5),
            |public| match *public {
                DoryAssistPublicId::MillerLoopSelector {
                    selector: MillerLoopSelector::LineDouble,
                    ..
                } => Fr::from_u64(1),
                DoryAssistPublicId::MillerLoopSelector {
                    selector: MillerLoopSelector::LineAdd,
                    ..
                } => zero,
                DoryAssistPublicId::MillerLoopConstant(MillerLoopConstant::TwoInverse) => {
                    Fr::from_u64(9)
                }
                _ => zero,
            },
        );

        assert_eq!(good, zero);
        assert_ne!(bad, zero);
    }

    #[test]
    fn line_evaluation_claims_expose_sparse_mul_by_034_shape() {
        let claims = line_evaluation::<Fr>(dimensions());

        assert_eq!(claims.id, DoryAssistRelationId::MillerLoopLineEvaluation);
        assert_eq!(claims.sumcheck, line_evaluation_sumcheck(dimensions()));
        assert!(claims.input.required_openings.is_empty());
        assert_openings_include(
            &claims.output.required_openings,
            &line_evaluation_output_openings(),
        );
        assert_eq!(
            claims.required_challenges(),
            vec![DoryAssistChallengeId::from(
                MillerLoopChallenge::LineEvaluationBatch
            )]
        );
    }

    #[test]
    fn line_evaluation_accepts_sparse_line_embedding() {
        let expr = line_evaluation_constraint_expression::<Fr>();
        let relation = DoryAssistRelationId::MillerLoopLineEvaluation;
        let zero = Fr::from_u64(0);

        let value = expr.evaluate(
            |opening| match *opening {
                id if id == g1_point_x_opening(relation) => Fr::from_u64(3),
                id if id == g1_point_y_opening(relation) => Fr::from_u64(5),
                id if id
                    == miller_loop_opening(
                        MillerLoopPolynomial::LineCoefficient {
                            coefficient: 0,
                            component: 0,
                        },
                        relation,
                    ) =>
                {
                    Fr::from_u64(2)
                }
                id if id
                    == miller_loop_opening(
                        MillerLoopPolynomial::LineCoefficient {
                            coefficient: 0,
                            component: 1,
                        },
                        relation,
                    ) =>
                {
                    Fr::from_u64(7)
                }
                id if id
                    == miller_loop_opening(
                        MillerLoopPolynomial::LineCoefficient {
                            coefficient: 1,
                            component: 0,
                        },
                        relation,
                    ) =>
                {
                    Fr::from_u64(11)
                }
                id if id
                    == miller_loop_opening(
                        MillerLoopPolynomial::LineCoefficient {
                            coefficient: 1,
                            component: 1,
                        },
                        relation,
                    ) =>
                {
                    Fr::from_u64(13)
                }
                id if id
                    == miller_loop_opening(
                        MillerLoopPolynomial::LineCoefficient {
                            coefficient: 2,
                            component: 0,
                        },
                        relation,
                    ) =>
                {
                    Fr::from_u64(17)
                }
                id if id
                    == miller_loop_opening(
                        MillerLoopPolynomial::LineCoefficient {
                            coefficient: 2,
                            component: 1,
                        },
                        relation,
                    ) =>
                {
                    Fr::from_u64(19)
                }
                id if id == line_evaluation_coeff_opening(0) => Fr::from_u64(10),
                id if id == line_evaluation_coeff_opening(1) => Fr::from_u64(35),
                id if id == line_evaluation_coeff_opening(2) => Fr::from_u64(33),
                id if id == line_evaluation_coeff_opening(3) => Fr::from_u64(39),
                id if id == line_evaluation_coeff_opening(4) => Fr::from_u64(17),
                id if id == line_evaluation_coeff_opening(5) => Fr::from_u64(19),
                _ => zero,
            },
            |_| Fr::from_u64(23),
            |_| zero,
        );

        assert_eq!(value, zero);
    }

    #[test]
    fn line_evaluation_rejects_nonzero_padding_slot() {
        let expr = line_evaluation_constraint_expression::<Fr>();
        let zero = Fr::from_u64(0);

        let value = expr.evaluate(
            |opening| match *opening {
                id if id == line_evaluation_coeff_opening(6) => Fr::from_u64(1),
                _ => zero,
            },
            |_| Fr::from_u64(23),
            |_| zero,
        );

        assert_ne!(value, zero);
    }

    #[test]
    fn pair_product_claims_expose_shift_and_boundary_dependencies() {
        let claims = pair_product::<Fr>(dimensions());

        assert_eq!(claims.id, DoryAssistRelationId::MillerLoopPairProduct);
        assert_eq!(claims.sumcheck, pair_product_sumcheck(dimensions()));
        assert!(claims.input.required_openings.is_empty());
        assert_openings_include(
            &claims.output.required_openings,
            &pair_product_output_openings(),
        );
        assert_openings_include(
            &claims.output.required_openings,
            &pair_product_quotient_openings(),
        );
        assert!(claims
            .required_publics()
            .contains(&DoryAssistPublicId::MillerLoopShiftEqKernel(
                DoryAssistRelationId::MillerLoopPairProduct
            )));
    }

    #[test]
    fn pair_product_checks_shift_and_pair_boundaries() {
        let expr = pair_product_constraint_expression::<Fr>();
        let relation = DoryAssistRelationId::MillerLoopPairProduct;
        let zero = Fr::from_u64(0);

        let value = expr.evaluate(
            |opening| match *opening {
                id if id == pair_product_accumulator_opening(0) => Fr::from_u64(2),
                id if id == pair_product_shifted_accumulator_opening(0) => Fr::from_u64(6),
                id if id == pair_line_product_opening(0) => Fr::from_u64(6),
                _ => zero,
            },
            |_| Fr::from_u64(7),
            |public| match *public {
                DoryAssistPublicId::MillerLoopShiftEqKernel(id) if id == relation => {
                    Fr::from_u64(3)
                }
                DoryAssistPublicId::BoundarySelector {
                    relation: id,
                    endpoint: DoryAssistBoundaryEndpoint::Initial,
                } if id == relation => Fr::from_u64(1),
                DoryAssistPublicId::BoundarySelector {
                    relation: id,
                    endpoint: DoryAssistBoundaryEndpoint::Final,
                } if id == relation => Fr::from_u64(1),
                DoryAssistPublicId::BoundaryValue {
                    relation: id,
                    endpoint: DoryAssistBoundaryEndpoint::Initial,
                    component: 0,
                } if id == relation => Fr::from_u64(2),
                _ => zero,
            },
        );

        assert_eq!(value, zero);
    }

    #[test]
    fn accumulator_shift_reduces_shifted_accumulator_to_current_accumulator() {
        let claims = accumulator::<Fr>(dimensions());
        let zero = Fr::from_u64(0);

        assert_eq!(claims.id, DoryAssistRelationId::MillerLoopAccumulator);
        assert_eq!(claims.sumcheck, accumulator_sumcheck(dimensions()));
        assert_eq!(claims.input.required_openings, accumulator_input_openings());
        assert_openings_include(
            &claims.output.required_openings,
            &accumulator_output_openings(),
        );
        assert_openings_include(
            &claims.output.required_openings,
            &accumulator_quotient_openings(),
        );
        assert_eq!(
            claims.required_challenges(),
            vec![DoryAssistChallengeId::from(
                MillerLoopChallenge::AccumulatorBatch
            )]
        );

        let input = claims.input.expression().evaluate(
            |opening| match *opening {
                id if id == accumulator_shifted_opening(0) => Fr::from_u64(12),
                _ => zero,
            },
            |_| Fr::from_u64(5),
            |_| zero,
        );
        let output = claims.output.expression().evaluate(
            |opening| match *opening {
                id if id == accumulator_opening(0) => Fr::from_u64(4),
                _ => zero,
            },
            |_| Fr::from_u64(5),
            |public| match *public {
                DoryAssistPublicId::MillerLoopShiftEqKernel(
                    DoryAssistRelationId::MillerLoopAccumulator,
                ) => Fr::from_u64(3),
                _ => zero,
            },
        );

        assert_eq!(input, output);
    }

    #[test]
    fn boundary_checks_initial_accumulator_and_public_output() {
        let claims = boundary::<Fr>(dimensions());
        let relation = DoryAssistRelationId::MillerLoopBoundary;
        let zero = Fr::from_u64(0);

        assert_eq!(claims.id, relation);
        assert_eq!(claims.sumcheck, boundary_sumcheck(dimensions()));
        assert_eq!(claims.output.required_openings, boundary_output_openings());
        assert_eq!(
            claims.required_challenges(),
            vec![DoryAssistChallengeId::from(
                MillerLoopChallenge::BoundaryPoint
            )]
        );
        assert!(claims
            .required_publics()
            .contains(&DoryAssistPublicId::MillerLoopOutputGt(0)));

        let output = claims.output.expression().evaluate(
            |opening| match *opening {
                id if id == boundary_accumulator_opening(0) => Fr::from_u64(1),
                id if id == boundary_shifted_accumulator_opening(0) => Fr::from_u64(9),
                _ => zero,
            },
            |_| Fr::from_u64(7),
            |public| match *public {
                DoryAssistPublicId::BoundarySelector {
                    relation: id,
                    endpoint: DoryAssistBoundaryEndpoint::Initial,
                } if id == relation => Fr::from_u64(1),
                DoryAssistPublicId::BoundarySelector {
                    relation: id,
                    endpoint: DoryAssistBoundaryEndpoint::Final,
                } if id == relation => Fr::from_u64(1),
                DoryAssistPublicId::BoundaryValue {
                    relation: id,
                    endpoint: DoryAssistBoundaryEndpoint::Initial,
                    component: 0,
                } if id == relation => Fr::from_u64(1),
                DoryAssistPublicId::MillerLoopOutputGt(0) => Fr::from_u64(9),
                _ => zero,
            },
        );

        assert_eq!(output, zero);
    }
}
