use jolt_field::{FromPrimitiveInt, RingCore};

use crate::{challenge, constant, opening, public};

use super::super::{
    DoryAssistBoundaryEndpoint, DoryAssistChallengeId, DoryAssistExpr, DoryAssistOpeningId,
    DoryAssistPublicId, DoryAssistRelationClaims, DoryAssistRelationId,
    DoryAssistVirtualPolynomial, G2Challenge, G2Polynomial,
};
use super::dimensions::{DoryAssistSumcheckSpec, G2Dimensions};

#[derive(Clone)]
struct Fq2Expr<F> {
    c0: DoryAssistExpr<F>,
    c1: DoryAssistExpr<F>,
}

pub const fn scalar_multiplication_sumcheck(dimensions: G2Dimensions) -> DoryAssistSumcheckSpec {
    dimensions.scalar_mul_sumcheck()
}

pub const fn scalar_multiplication_shift_sumcheck(
    dimensions: G2Dimensions,
) -> DoryAssistSumcheckSpec {
    dimensions.scalar_mul_shift_sumcheck()
}

pub const fn scalar_multiplication_boundary_sumcheck(
    dimensions: G2Dimensions,
) -> DoryAssistSumcheckSpec {
    dimensions.scalar_mul_boundary_sumcheck()
}

pub const fn addition_sumcheck(dimensions: G2Dimensions) -> DoryAssistSumcheckSpec {
    dimensions.add_sumcheck()
}

pub fn scalar_multiplication<F>(dimensions: G2Dimensions) -> DoryAssistRelationClaims<F>
where
    F: RingCore + FromPrimitiveInt,
{
    DoryAssistRelationClaims::new(
        DoryAssistRelationId::G2ScalarMultiplication,
        scalar_multiplication_sumcheck(dimensions),
        constant(F::zero()),
        scalar_multiplication_constraint_expression(),
    )
}

pub fn scalar_multiplication_shift<F>(dimensions: G2Dimensions) -> DoryAssistRelationClaims<F>
where
    F: RingCore,
{
    let gamma = g2_challenge(G2Challenge::ShiftGamma);
    let input = opening(scalar_mul_shifted_accumulator_x0_opening())
        + gamma.clone() * opening(scalar_mul_shifted_accumulator_x1_opening())
        + gamma.clone().pow(2) * opening(scalar_mul_shifted_accumulator_y0_opening())
        + gamma.clone().pow(3) * opening(scalar_mul_shifted_accumulator_y1_opening());
    let output = opening(scalar_mul_accumulator_x0_opening())
        + gamma.clone() * opening(scalar_mul_accumulator_x1_opening())
        + gamma.clone().pow(2) * opening(scalar_mul_accumulator_y0_opening())
        + gamma.pow(3) * opening(scalar_mul_accumulator_y1_opening());

    DoryAssistRelationClaims::new(
        DoryAssistRelationId::G2ScalarMultiplicationShift,
        scalar_multiplication_shift_sumcheck(dimensions),
        input,
        output,
    )
}

pub fn scalar_multiplication_boundary<F>(dimensions: G2Dimensions) -> DoryAssistRelationClaims<F>
where
    F: RingCore,
{
    let relation = DoryAssistRelationId::G2ScalarMultiplicationBoundary;
    let gamma = g2_challenge(G2Challenge::BoundaryPoint);
    let initial = boundary_selector(relation, DoryAssistBoundaryEndpoint::Initial)
        * (opening(scalar_mul_boundary_accumulator_x0_opening())
            - boundary_value(relation, DoryAssistBoundaryEndpoint::Initial, 0)
            + gamma.clone()
                * (opening(scalar_mul_boundary_accumulator_x1_opening())
                    - boundary_value(relation, DoryAssistBoundaryEndpoint::Initial, 1))
            + gamma.clone().pow(2)
                * (opening(scalar_mul_boundary_accumulator_y0_opening())
                    - boundary_value(relation, DoryAssistBoundaryEndpoint::Initial, 2))
            + gamma.clone().pow(3)
                * (opening(scalar_mul_boundary_accumulator_y1_opening())
                    - boundary_value(relation, DoryAssistBoundaryEndpoint::Initial, 3))
            + gamma.clone().pow(4)
                * (opening(scalar_mul_boundary_accumulator_infinity_opening())
                    - boundary_value(relation, DoryAssistBoundaryEndpoint::Initial, 4)));
    let final_value = boundary_selector(relation, DoryAssistBoundaryEndpoint::Final)
        * (opening(scalar_mul_boundary_shifted_accumulator_x0_opening())
            - boundary_value(relation, DoryAssistBoundaryEndpoint::Final, 0)
            + gamma.clone()
                * (opening(scalar_mul_boundary_shifted_accumulator_x1_opening())
                    - boundary_value(relation, DoryAssistBoundaryEndpoint::Final, 1))
            + gamma.clone().pow(2)
                * (opening(scalar_mul_boundary_shifted_accumulator_y0_opening())
                    - boundary_value(relation, DoryAssistBoundaryEndpoint::Final, 2))
            + gamma.clone().pow(3)
                * (opening(scalar_mul_boundary_shifted_accumulator_y1_opening())
                    - boundary_value(relation, DoryAssistBoundaryEndpoint::Final, 3)));

    DoryAssistRelationClaims::new(
        relation,
        scalar_multiplication_boundary_sumcheck(dimensions),
        constant(F::zero()),
        initial + gamma.pow(5) * final_value,
    )
}

pub fn addition<F>(dimensions: G2Dimensions) -> DoryAssistRelationClaims<F>
where
    F: RingCore + FromPrimitiveInt,
{
    DoryAssistRelationClaims::new(
        DoryAssistRelationId::G2Addition,
        addition_sumcheck(dimensions),
        constant(F::zero()),
        addition_constraint_expression(),
    )
}

pub fn scalar_multiplication_input_openings() -> Vec<DoryAssistOpeningId> {
    let relation = DoryAssistRelationId::G2ScalarMultiplication;
    vec![
        g2_opening(G2Polynomial::ScalarMulAccumulatorX0, relation),
        g2_opening(G2Polynomial::ScalarMulAccumulatorX1, relation),
        g2_opening(G2Polynomial::ScalarMulAccumulatorY0, relation),
        g2_opening(G2Polynomial::ScalarMulAccumulatorY1, relation),
        g2_opening(G2Polynomial::ScalarMulAccumulatorInfinity, relation),
        g2_opening(G2Polynomial::ScalarMulDoubledX0, relation),
        g2_opening(G2Polynomial::ScalarMulDoubledX1, relation),
        g2_opening(G2Polynomial::ScalarMulDoubledY0, relation),
        g2_opening(G2Polynomial::ScalarMulDoubledY1, relation),
        g2_opening(G2Polynomial::ScalarMulDoubledInfinity, relation),
        g2_opening(G2Polynomial::ScalarMulShiftedAccumulatorX0, relation),
        g2_opening(G2Polynomial::ScalarMulShiftedAccumulatorX1, relation),
        g2_opening(G2Polynomial::ScalarMulShiftedAccumulatorY0, relation),
        g2_opening(G2Polynomial::ScalarMulShiftedAccumulatorY1, relation),
        g2_opening(G2Polynomial::ScalarMulBit, relation),
        g2_opening(G2Polynomial::ScalarMulBaseX0, relation),
        g2_opening(G2Polynomial::ScalarMulBaseX1, relation),
        g2_opening(G2Polynomial::ScalarMulBaseY0, relation),
        g2_opening(G2Polynomial::ScalarMulBaseY1, relation),
    ]
}

pub fn scalar_multiplication_shift_input_openings() -> [DoryAssistOpeningId; 4] {
    [
        scalar_mul_shifted_accumulator_x0_opening(),
        scalar_mul_shifted_accumulator_x1_opening(),
        scalar_mul_shifted_accumulator_y0_opening(),
        scalar_mul_shifted_accumulator_y1_opening(),
    ]
}

pub fn scalar_multiplication_shift_output_openings() -> [DoryAssistOpeningId; 4] {
    [
        scalar_mul_accumulator_x0_opening(),
        scalar_mul_accumulator_x1_opening(),
        scalar_mul_accumulator_y0_opening(),
        scalar_mul_accumulator_y1_opening(),
    ]
}

pub fn scalar_multiplication_boundary_output_openings() -> [DoryAssistOpeningId; 9] {
    [
        scalar_mul_boundary_accumulator_x0_opening(),
        scalar_mul_boundary_accumulator_x1_opening(),
        scalar_mul_boundary_accumulator_y0_opening(),
        scalar_mul_boundary_accumulator_y1_opening(),
        scalar_mul_boundary_accumulator_infinity_opening(),
        scalar_mul_boundary_shifted_accumulator_x0_opening(),
        scalar_mul_boundary_shifted_accumulator_x1_opening(),
        scalar_mul_boundary_shifted_accumulator_y0_opening(),
        scalar_mul_boundary_shifted_accumulator_y1_opening(),
    ]
}

pub fn addition_input_openings() -> Vec<DoryAssistOpeningId> {
    let relation = DoryAssistRelationId::G2Addition;
    vec![
        g2_opening(G2Polynomial::AddInputLeftX0, relation),
        g2_opening(G2Polynomial::AddInputLeftX1, relation),
        g2_opening(G2Polynomial::AddInputLeftY0, relation),
        g2_opening(G2Polynomial::AddInputLeftY1, relation),
        g2_opening(G2Polynomial::AddInputLeftInfinity, relation),
        g2_opening(G2Polynomial::AddInputRightX0, relation),
        g2_opening(G2Polynomial::AddInputRightX1, relation),
        g2_opening(G2Polynomial::AddInputRightY0, relation),
        g2_opening(G2Polynomial::AddInputRightY1, relation),
        g2_opening(G2Polynomial::AddInputRightInfinity, relation),
        g2_opening(G2Polynomial::AddOutputX0, relation),
        g2_opening(G2Polynomial::AddOutputX1, relation),
        g2_opening(G2Polynomial::AddOutputY0, relation),
        g2_opening(G2Polynomial::AddOutputY1, relation),
        g2_opening(G2Polynomial::AddOutputInfinity, relation),
        g2_opening(G2Polynomial::AddSlope0, relation),
        g2_opening(G2Polynomial::AddSlope1, relation),
        g2_opening(G2Polynomial::AddInverse0, relation),
        g2_opening(G2Polynomial::AddInverse1, relation),
        g2_opening(G2Polynomial::AddBranchSelector(0), relation),
        g2_opening(G2Polynomial::AddBranchSelector(1), relation),
    ]
}

pub fn scalar_mul_accumulator_x0_opening() -> DoryAssistOpeningId {
    g2_opening(
        G2Polynomial::ScalarMulAccumulatorX0,
        DoryAssistRelationId::G2ScalarMultiplicationShift,
    )
}

pub fn scalar_mul_accumulator_x1_opening() -> DoryAssistOpeningId {
    g2_opening(
        G2Polynomial::ScalarMulAccumulatorX1,
        DoryAssistRelationId::G2ScalarMultiplicationShift,
    )
}

pub fn scalar_mul_accumulator_y0_opening() -> DoryAssistOpeningId {
    g2_opening(
        G2Polynomial::ScalarMulAccumulatorY0,
        DoryAssistRelationId::G2ScalarMultiplicationShift,
    )
}

pub fn scalar_mul_accumulator_y1_opening() -> DoryAssistOpeningId {
    g2_opening(
        G2Polynomial::ScalarMulAccumulatorY1,
        DoryAssistRelationId::G2ScalarMultiplicationShift,
    )
}

pub fn scalar_mul_shifted_accumulator_x0_opening() -> DoryAssistOpeningId {
    g2_opening(
        G2Polynomial::ScalarMulShiftedAccumulatorX0,
        DoryAssistRelationId::G2ScalarMultiplicationShift,
    )
}

pub fn scalar_mul_shifted_accumulator_x1_opening() -> DoryAssistOpeningId {
    g2_opening(
        G2Polynomial::ScalarMulShiftedAccumulatorX1,
        DoryAssistRelationId::G2ScalarMultiplicationShift,
    )
}

pub fn scalar_mul_shifted_accumulator_y0_opening() -> DoryAssistOpeningId {
    g2_opening(
        G2Polynomial::ScalarMulShiftedAccumulatorY0,
        DoryAssistRelationId::G2ScalarMultiplicationShift,
    )
}

pub fn scalar_mul_shifted_accumulator_y1_opening() -> DoryAssistOpeningId {
    g2_opening(
        G2Polynomial::ScalarMulShiftedAccumulatorY1,
        DoryAssistRelationId::G2ScalarMultiplicationShift,
    )
}

pub fn scalar_mul_boundary_accumulator_x0_opening() -> DoryAssistOpeningId {
    g2_opening(
        G2Polynomial::ScalarMulAccumulatorX0,
        DoryAssistRelationId::G2ScalarMultiplicationBoundary,
    )
}

pub fn scalar_mul_boundary_accumulator_x1_opening() -> DoryAssistOpeningId {
    g2_opening(
        G2Polynomial::ScalarMulAccumulatorX1,
        DoryAssistRelationId::G2ScalarMultiplicationBoundary,
    )
}

pub fn scalar_mul_boundary_accumulator_y0_opening() -> DoryAssistOpeningId {
    g2_opening(
        G2Polynomial::ScalarMulAccumulatorY0,
        DoryAssistRelationId::G2ScalarMultiplicationBoundary,
    )
}

pub fn scalar_mul_boundary_accumulator_y1_opening() -> DoryAssistOpeningId {
    g2_opening(
        G2Polynomial::ScalarMulAccumulatorY1,
        DoryAssistRelationId::G2ScalarMultiplicationBoundary,
    )
}

pub fn scalar_mul_boundary_accumulator_infinity_opening() -> DoryAssistOpeningId {
    g2_opening(
        G2Polynomial::ScalarMulAccumulatorInfinity,
        DoryAssistRelationId::G2ScalarMultiplicationBoundary,
    )
}

pub fn scalar_mul_boundary_shifted_accumulator_x0_opening() -> DoryAssistOpeningId {
    g2_opening(
        G2Polynomial::ScalarMulShiftedAccumulatorX0,
        DoryAssistRelationId::G2ScalarMultiplicationBoundary,
    )
}

pub fn scalar_mul_boundary_shifted_accumulator_x1_opening() -> DoryAssistOpeningId {
    g2_opening(
        G2Polynomial::ScalarMulShiftedAccumulatorX1,
        DoryAssistRelationId::G2ScalarMultiplicationBoundary,
    )
}

pub fn scalar_mul_boundary_shifted_accumulator_y0_opening() -> DoryAssistOpeningId {
    g2_opening(
        G2Polynomial::ScalarMulShiftedAccumulatorY0,
        DoryAssistRelationId::G2ScalarMultiplicationBoundary,
    )
}

pub fn scalar_mul_boundary_shifted_accumulator_y1_opening() -> DoryAssistOpeningId {
    g2_opening(
        G2Polynomial::ScalarMulShiftedAccumulatorY1,
        DoryAssistRelationId::G2ScalarMultiplicationBoundary,
    )
}

pub fn scalar_multiplication_constraint_expression<F>() -> DoryAssistExpr<F>
where
    F: RingCore + FromPrimitiveInt,
{
    let relation = DoryAssistRelationId::G2ScalarMultiplication;
    let x_a = scalar_mul_accumulator_x(relation);
    let y_a = scalar_mul_accumulator_y(relation);
    let iota_a = g2_polynomial(G2Polynomial::ScalarMulAccumulatorInfinity, relation);
    let x_t = scalar_mul_doubled_x(relation);
    let y_t = scalar_mul_doubled_y(relation);
    let iota_t = g2_polynomial(G2Polynomial::ScalarMulDoubledInfinity, relation);
    let x_a_next = scalar_mul_shifted_accumulator_x(relation);
    let y_a_next = scalar_mul_shifted_accumulator_y(relation);
    let bit = g2_polynomial(G2Polynomial::ScalarMulBit, relation);
    let x_p = scalar_mul_base_x(relation);
    let y_p = scalar_mul_base_y(relation);

    let dx = fq2_sub(&x_p, &x_t);
    let dy = fq2_sub(&y_p, &y_t);
    let bit_boolean = bit.clone() * (one::<F>() - bit.clone());
    let x_a_square = fq2_square(&x_a);
    let y_a_square = fq2_square(&y_a);

    let double_x = fq2_sub(
        &fq2_scale_i128(
            &fq2_mul(&y_a_square, &fq2_add(&x_t, &fq2_scale_i128(&x_a, 2))),
            4,
        ),
        &fq2_scale_i128(&fq2_square(&x_a_square), 9),
    );
    let double_y = fq2_add(
        &fq2_scale_i128(&fq2_mul(&x_a_square, &fq2_sub(&x_t, &x_a)), 3),
        &fq2_scale_i128(&fq2_mul(&y_a, &fq2_add(&y_t, &y_a)), 2),
    );
    let conditional_x = fq2_add(
        &fq2_add(
            &fq2_scale_expr(&fq2_sub(&x_a_next, &x_t), one::<F>() - bit.clone()),
            &fq2_scale_expr(&fq2_sub(&x_a_next, &x_p), bit.clone() * iota_t.clone()),
        ),
        &fq2_scale_expr(
            &fq2_sub(
                &fq2_mul(&fq2_add(&fq2_add(&x_a_next, &x_t), &x_p), &fq2_square(&dx)),
                &fq2_square(&dy),
            ),
            bit.clone() * (one::<F>() - iota_t.clone()),
        ),
    );
    let conditional_y = fq2_add(
        &fq2_add(
            &fq2_scale_expr(&fq2_sub(&y_a_next, &y_t), one::<F>() - bit.clone()),
            &fq2_scale_expr(&fq2_sub(&y_a_next, &y_p), bit.clone() * iota_t.clone()),
        ),
        &fq2_scale_expr(
            &fq2_sub(
                &fq2_mul(&fq2_add(&y_a_next, &y_t), &dx),
                &fq2_mul(&dy, &fq2_sub(&x_t, &x_a_next)),
            ),
            bit * (one::<F>() - iota_t.clone()),
        ),
    );

    batch_constraints(
        g2_challenge(G2Challenge::ConstraintBatch),
        [
            bit_boolean,
            double_x.c0,
            double_x.c1,
            double_y.c0,
            double_y.c1,
            conditional_x.c0,
            conditional_x.c1,
            conditional_y.c0,
            conditional_y.c1,
            iota_a * (one::<F>() - iota_t.clone()),
            iota_t.clone() * x_t.c0,
            iota_t.clone() * x_t.c1,
            iota_t.clone() * y_t.c0,
            iota_t * y_t.c1,
        ],
    )
}

pub fn addition_constraint_expression<F>() -> DoryAssistExpr<F>
where
    F: RingCore + FromPrimitiveInt,
{
    let relation = DoryAssistRelationId::G2Addition;
    let x_p = add_input_left_x(relation);
    let y_p = add_input_left_y(relation);
    let iota_p = g2_polynomial(G2Polynomial::AddInputLeftInfinity, relation);
    let x_q = add_input_right_x(relation);
    let y_q = add_input_right_y(relation);
    let iota_q = g2_polynomial(G2Polynomial::AddInputRightInfinity, relation);
    let x_r = add_output_x(relation);
    let y_r = add_output_y(relation);
    let iota_r = g2_polynomial(G2Polynomial::AddOutputInfinity, relation);
    let lambda = Fq2Expr {
        c0: g2_polynomial(G2Polynomial::AddSlope0, relation),
        c1: g2_polynomial(G2Polynomial::AddSlope1, relation),
    };
    let mu = Fq2Expr {
        c0: g2_polynomial(G2Polynomial::AddInverse0, relation),
        c1: g2_polynomial(G2Polynomial::AddInverse1, relation),
    };
    let sigma_1 = g2_polynomial(G2Polynomial::AddBranchSelector(0), relation);
    let sigma_2 = g2_polynomial(G2Polynomial::AddBranchSelector(1), relation);

    let dx = fq2_sub(&x_q, &x_p);
    let dy = fq2_sub(&y_q, &y_p);
    let phi = (one::<F>() - iota_p.clone()) * (one::<F>() - iota_q.clone());
    let add_branch = one::<F>() - sigma_1.clone() - sigma_2.clone();
    let not_inverse = one::<F>() - sigma_2.clone();
    let mu_dx = fq2_mul(&mu, &dx);
    let lambda_dx = fq2_mul(&lambda, &dx);
    let y_p_lambda = fq2_mul(&y_p, &lambda);
    let x_p_square = fq2_square(&x_p);
    let lambda_square = fq2_square(&lambda);
    let lambda_x_p_minus_r = fq2_mul(&lambda, &fq2_sub(&x_p, &x_r));

    let constraints = [
        iota_p.clone() * (one::<F>() - iota_p.clone()),
        iota_q.clone() * (one::<F>() - iota_q.clone()),
        iota_r.clone() * (one::<F>() - iota_r.clone()),
        iota_p.clone() * x_p.c0.clone(),
        iota_p.clone() * x_p.c1.clone(),
        iota_p.clone() * y_p.c0.clone(),
        iota_p.clone() * y_p.c1.clone(),
        iota_q.clone() * x_q.c0.clone(),
        iota_q.clone() * x_q.c1.clone(),
        iota_q.clone() * y_q.c0.clone(),
        iota_q.clone() * y_q.c1.clone(),
        iota_r.clone() * x_r.c0.clone(),
        iota_r.clone() * x_r.c1.clone(),
        iota_r.clone() * y_r.c0.clone(),
        iota_r.clone() * y_r.c1.clone(),
        iota_p.clone() * (x_r.c0.clone() - x_q.c0.clone()),
        iota_p.clone() * (x_r.c1.clone() - x_q.c1.clone()),
        iota_p.clone() * (y_r.c0.clone() - y_q.c0.clone()),
        iota_p.clone() * (y_r.c1.clone() - y_q.c1.clone()),
        iota_p.clone() * (iota_r.clone() - iota_q.clone()),
        iota_q.clone() * (one::<F>() - iota_p.clone()) * (x_r.c0.clone() - x_p.c0.clone()),
        iota_q.clone() * (one::<F>() - iota_p.clone()) * (x_r.c1.clone() - x_p.c1.clone()),
        iota_q.clone() * (one::<F>() - iota_p.clone()) * (y_r.c0.clone() - y_p.c0.clone()),
        iota_q.clone() * (one::<F>() - iota_p.clone()) * (y_r.c1.clone() - y_p.c1.clone()),
        iota_q.clone() * (one::<F>() - iota_p.clone()) * (iota_r.clone() - iota_p),
        phi.clone() * sigma_1.clone() * (one::<F>() - sigma_1.clone()),
        phi.clone() * sigma_2.clone() * (one::<F>() - sigma_2.clone()),
        phi.clone() * sigma_1.clone() * sigma_2.clone(),
        phi.clone() * add_branch.clone() * (one::<F>() - mu_dx.c0),
        phi.clone() * add_branch.clone() * mu_dx.c1,
        phi.clone() * sigma_1.clone() * dx.c0.clone(),
        phi.clone() * sigma_1.clone() * dx.c1.clone(),
        phi.clone() * sigma_1.clone() * dy.c0.clone(),
        phi.clone() * sigma_1.clone() * dy.c1.clone(),
        phi.clone() * sigma_2.clone() * dx.c0.clone(),
        phi.clone() * sigma_2.clone() * dx.c1.clone(),
        phi.clone() * sigma_2.clone() * (y_q.c0.clone() + y_p.c0.clone()),
        phi.clone() * sigma_2.clone() * (y_q.c1.clone() + y_p.c1.clone()),
        phi.clone() * add_branch.clone() * (lambda_dx.c0 - dy.c0),
        phi.clone() * add_branch.clone() * (lambda_dx.c1 - dy.c1),
        phi.clone() * sigma_1.clone() * (2 * y_p_lambda.c0 - 3 * x_p_square.c0),
        phi.clone() * sigma_1.clone() * (2 * y_p_lambda.c1 - 3 * x_p_square.c1),
        phi.clone() * sigma_2.clone() * (one::<F>() - iota_r.clone()),
        phi.clone() * not_inverse.clone() * iota_r,
        phi.clone()
            * not_inverse.clone()
            * (x_r.c0.clone() - lambda_square.c0 + x_p.c0.clone() + x_q.c0),
        phi.clone()
            * not_inverse.clone()
            * (x_r.c1.clone() - lambda_square.c1 + x_p.c1.clone() + x_q.c1),
        phi.clone() * not_inverse.clone() * (y_r.c0 - lambda_x_p_minus_r.c0 + y_p.c0),
        phi * not_inverse * (y_r.c1 - lambda_x_p_minus_r.c1 + y_p.c1),
    ];

    batch_constraints(g2_challenge(G2Challenge::ConstraintBatch), constraints)
}

fn batch_constraints<F, const N: usize>(
    delta: DoryAssistExpr<F>,
    constraints: [DoryAssistExpr<F>; N],
) -> DoryAssistExpr<F>
where
    F: RingCore + FromPrimitiveInt,
{
    constraints
        .into_iter()
        .enumerate()
        .fold(constant(F::zero()), |acc, (index, constraint)| {
            acc + delta.clone().pow(index) * constraint
        })
}

fn fq2_add<F>(lhs: &Fq2Expr<F>, rhs: &Fq2Expr<F>) -> Fq2Expr<F>
where
    F: RingCore,
{
    Fq2Expr {
        c0: lhs.c0.clone() + rhs.c0.clone(),
        c1: lhs.c1.clone() + rhs.c1.clone(),
    }
}

fn fq2_sub<F>(lhs: &Fq2Expr<F>, rhs: &Fq2Expr<F>) -> Fq2Expr<F>
where
    F: RingCore,
{
    Fq2Expr {
        c0: lhs.c0.clone() - rhs.c0.clone(),
        c1: lhs.c1.clone() - rhs.c1.clone(),
    }
}

fn fq2_mul<F>(lhs: &Fq2Expr<F>, rhs: &Fq2Expr<F>) -> Fq2Expr<F>
where
    F: RingCore,
{
    Fq2Expr {
        c0: lhs.c0.clone() * rhs.c0.clone() - lhs.c1.clone() * rhs.c1.clone(),
        c1: lhs.c0.clone() * rhs.c1.clone() + lhs.c1.clone() * rhs.c0.clone(),
    }
}

fn fq2_square<F>(value: &Fq2Expr<F>) -> Fq2Expr<F>
where
    F: RingCore + FromPrimitiveInt,
{
    Fq2Expr {
        c0: square(value.c0.clone()) - square(value.c1.clone()),
        c1: 2 * value.c0.clone() * value.c1.clone(),
    }
}

fn fq2_scale_expr<F>(value: &Fq2Expr<F>, scalar: DoryAssistExpr<F>) -> Fq2Expr<F>
where
    F: RingCore,
{
    Fq2Expr {
        c0: scalar.clone() * value.c0.clone(),
        c1: scalar * value.c1.clone(),
    }
}

fn fq2_scale_i128<F>(value: &Fq2Expr<F>, scalar: i128) -> Fq2Expr<F>
where
    F: RingCore + FromPrimitiveInt,
{
    Fq2Expr {
        c0: scalar * value.c0.clone(),
        c1: scalar * value.c1.clone(),
    }
}

fn square<F>(value: DoryAssistExpr<F>) -> DoryAssistExpr<F>
where
    F: RingCore,
{
    value.clone() * value
}

fn one<F>() -> DoryAssistExpr<F>
where
    F: RingCore,
{
    constant(F::one())
}

fn scalar_mul_accumulator_x<F>(relation: DoryAssistRelationId) -> Fq2Expr<F>
where
    F: RingCore,
{
    Fq2Expr {
        c0: g2_polynomial(G2Polynomial::ScalarMulAccumulatorX0, relation),
        c1: g2_polynomial(G2Polynomial::ScalarMulAccumulatorX1, relation),
    }
}

fn scalar_mul_accumulator_y<F>(relation: DoryAssistRelationId) -> Fq2Expr<F>
where
    F: RingCore,
{
    Fq2Expr {
        c0: g2_polynomial(G2Polynomial::ScalarMulAccumulatorY0, relation),
        c1: g2_polynomial(G2Polynomial::ScalarMulAccumulatorY1, relation),
    }
}

fn scalar_mul_doubled_x<F>(relation: DoryAssistRelationId) -> Fq2Expr<F>
where
    F: RingCore,
{
    Fq2Expr {
        c0: g2_polynomial(G2Polynomial::ScalarMulDoubledX0, relation),
        c1: g2_polynomial(G2Polynomial::ScalarMulDoubledX1, relation),
    }
}

fn scalar_mul_doubled_y<F>(relation: DoryAssistRelationId) -> Fq2Expr<F>
where
    F: RingCore,
{
    Fq2Expr {
        c0: g2_polynomial(G2Polynomial::ScalarMulDoubledY0, relation),
        c1: g2_polynomial(G2Polynomial::ScalarMulDoubledY1, relation),
    }
}

fn scalar_mul_shifted_accumulator_x<F>(relation: DoryAssistRelationId) -> Fq2Expr<F>
where
    F: RingCore,
{
    Fq2Expr {
        c0: g2_polynomial(G2Polynomial::ScalarMulShiftedAccumulatorX0, relation),
        c1: g2_polynomial(G2Polynomial::ScalarMulShiftedAccumulatorX1, relation),
    }
}

fn scalar_mul_shifted_accumulator_y<F>(relation: DoryAssistRelationId) -> Fq2Expr<F>
where
    F: RingCore,
{
    Fq2Expr {
        c0: g2_polynomial(G2Polynomial::ScalarMulShiftedAccumulatorY0, relation),
        c1: g2_polynomial(G2Polynomial::ScalarMulShiftedAccumulatorY1, relation),
    }
}

fn scalar_mul_base_x<F>(relation: DoryAssistRelationId) -> Fq2Expr<F>
where
    F: RingCore,
{
    Fq2Expr {
        c0: g2_polynomial(G2Polynomial::ScalarMulBaseX0, relation),
        c1: g2_polynomial(G2Polynomial::ScalarMulBaseX1, relation),
    }
}

fn scalar_mul_base_y<F>(relation: DoryAssistRelationId) -> Fq2Expr<F>
where
    F: RingCore,
{
    Fq2Expr {
        c0: g2_polynomial(G2Polynomial::ScalarMulBaseY0, relation),
        c1: g2_polynomial(G2Polynomial::ScalarMulBaseY1, relation),
    }
}

fn add_input_left_x<F>(relation: DoryAssistRelationId) -> Fq2Expr<F>
where
    F: RingCore,
{
    Fq2Expr {
        c0: g2_polynomial(G2Polynomial::AddInputLeftX0, relation),
        c1: g2_polynomial(G2Polynomial::AddInputLeftX1, relation),
    }
}

fn add_input_left_y<F>(relation: DoryAssistRelationId) -> Fq2Expr<F>
where
    F: RingCore,
{
    Fq2Expr {
        c0: g2_polynomial(G2Polynomial::AddInputLeftY0, relation),
        c1: g2_polynomial(G2Polynomial::AddInputLeftY1, relation),
    }
}

fn add_input_right_x<F>(relation: DoryAssistRelationId) -> Fq2Expr<F>
where
    F: RingCore,
{
    Fq2Expr {
        c0: g2_polynomial(G2Polynomial::AddInputRightX0, relation),
        c1: g2_polynomial(G2Polynomial::AddInputRightX1, relation),
    }
}

fn add_input_right_y<F>(relation: DoryAssistRelationId) -> Fq2Expr<F>
where
    F: RingCore,
{
    Fq2Expr {
        c0: g2_polynomial(G2Polynomial::AddInputRightY0, relation),
        c1: g2_polynomial(G2Polynomial::AddInputRightY1, relation),
    }
}

fn add_output_x<F>(relation: DoryAssistRelationId) -> Fq2Expr<F>
where
    F: RingCore,
{
    Fq2Expr {
        c0: g2_polynomial(G2Polynomial::AddOutputX0, relation),
        c1: g2_polynomial(G2Polynomial::AddOutputX1, relation),
    }
}

fn add_output_y<F>(relation: DoryAssistRelationId) -> Fq2Expr<F>
where
    F: RingCore,
{
    Fq2Expr {
        c0: g2_polynomial(G2Polynomial::AddOutputY0, relation),
        c1: g2_polynomial(G2Polynomial::AddOutputY1, relation),
    }
}

fn g2_polynomial<F>(polynomial: G2Polynomial, relation: DoryAssistRelationId) -> DoryAssistExpr<F>
where
    F: RingCore,
{
    opening(g2_opening(polynomial, relation))
}

fn g2_opening(polynomial: G2Polynomial, relation: DoryAssistRelationId) -> DoryAssistOpeningId {
    DoryAssistOpeningId::virtual_polynomial(DoryAssistVirtualPolynomial::G2(polynomial), relation)
}

pub fn g2_challenge<F>(id: G2Challenge) -> DoryAssistExpr<F>
where
    F: RingCore,
{
    challenge(DoryAssistChallengeId::from(id))
}

pub fn boundary_selector<F>(
    relation: DoryAssistRelationId,
    endpoint: DoryAssistBoundaryEndpoint,
) -> DoryAssistExpr<F>
where
    F: RingCore,
{
    public(DoryAssistPublicId::BoundarySelector { relation, endpoint })
}

pub fn boundary_value<F>(
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

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    #[test]
    fn scalar_mul_shift_batches_fq2_coordinates_with_gamma() {
        let claims = scalar_multiplication_shift::<Fr>(G2Dimensions::new(8, 0, 0));
        let zero = Fr::from_u64(0);

        assert_eq!(
            claims.input.required_openings,
            scalar_multiplication_shift_input_openings().to_vec()
        );
        assert_eq!(
            claims.output.required_openings,
            scalar_multiplication_shift_output_openings().to_vec()
        );
        assert_eq!(
            claims.required_challenges(),
            vec![DoryAssistChallengeId::from(G2Challenge::ShiftGamma)]
        );

        let input = claims.input.expression().evaluate(
            |opening| match *opening {
                id if id == scalar_mul_shifted_accumulator_x0_opening() => Fr::from_u64(2),
                id if id == scalar_mul_shifted_accumulator_x1_opening() => Fr::from_u64(3),
                id if id == scalar_mul_shifted_accumulator_y0_opening() => Fr::from_u64(5),
                id if id == scalar_mul_shifted_accumulator_y1_opening() => Fr::from_u64(7),
                _ => zero,
            },
            |_| Fr::from_u64(11),
            |_| zero,
        );

        assert_eq!(input, Fr::from_u64(9957));
    }

    #[test]
    fn local_constraint_relations_are_batched_zero_checks() {
        let dimensions = G2Dimensions::new(8, 2, 3);
        let scalar_mul = scalar_multiplication::<Fr>(dimensions);
        let addition = addition::<Fr>(dimensions);

        assert_eq!(
            scalar_mul.sumcheck,
            scalar_multiplication_sumcheck(dimensions)
        );
        let mut actual_scalar_mul_openings = scalar_mul.output.required_openings.clone();
        let mut expected_scalar_mul_openings = scalar_multiplication_input_openings();
        actual_scalar_mul_openings.sort();
        expected_scalar_mul_openings.sort();
        assert_eq!(actual_scalar_mul_openings, expected_scalar_mul_openings);
        assert_eq!(
            scalar_mul.required_challenges(),
            vec![DoryAssistChallengeId::from(G2Challenge::ConstraintBatch)]
        );
        assert_eq!(addition.sumcheck, addition_sumcheck(dimensions));
        let mut actual_addition_openings = addition.output.required_openings.clone();
        let mut expected_addition_openings = addition_input_openings();
        actual_addition_openings.sort();
        expected_addition_openings.sort();
        assert_eq!(actual_addition_openings, expected_addition_openings);
        assert_eq!(
            addition.required_challenges(),
            vec![DoryAssistChallengeId::from(G2Challenge::ConstraintBatch)]
        );
    }

    #[test]
    fn scalar_mul_constraints_accept_all_infinity_zero_row() {
        let claims = scalar_multiplication::<Fr>(G2Dimensions::new(8, 0, 0));
        let output = claims.output.expression().evaluate(
            |opening| match *opening {
                DoryAssistOpeningId::Polynomial {
                    polynomial:
                        super::super::super::DoryAssistPolynomialId::Virtual(
                            DoryAssistVirtualPolynomial::G2(
                                G2Polynomial::ScalarMulAccumulatorInfinity
                                | G2Polynomial::ScalarMulDoubledInfinity,
                            ),
                        ),
                    ..
                } => Fr::from_u64(1),
                DoryAssistOpeningId::Polynomial { .. } => Fr::from_u64(0),
            },
            |_| Fr::from_u64(2),
            |_| Fr::from_u64(0),
        );

        assert_eq!(output, Fr::from_u64(0));
    }

    #[test]
    fn scalar_mul_constraints_detect_invalid_infinity_transition() {
        let claims = scalar_multiplication::<Fr>(G2Dimensions::new(8, 0, 0));
        let output = claims.output.expression().evaluate(
            |opening| match *opening {
                DoryAssistOpeningId::Polynomial {
                    polynomial:
                        super::super::super::DoryAssistPolynomialId::Virtual(
                            DoryAssistVirtualPolynomial::G2(
                                G2Polynomial::ScalarMulAccumulatorInfinity,
                            ),
                        ),
                    ..
                } => Fr::from_u64(1),
                DoryAssistOpeningId::Polynomial { .. } => Fr::from_u64(0),
            },
            |_| Fr::from_u64(2),
            |_| Fr::from_u64(0),
        );

        assert_ne!(output, Fr::from_u64(0));
    }

    #[test]
    fn scalar_mul_constraints_reject_non_boolean_bit() {
        let claims = scalar_multiplication::<Fr>(G2Dimensions::new(8, 0, 0));
        let output = claims.output.expression().evaluate(
            |opening| match *opening {
                DoryAssistOpeningId::Polynomial {
                    polynomial:
                        super::super::super::DoryAssistPolynomialId::Virtual(
                            DoryAssistVirtualPolynomial::G2(G2Polynomial::ScalarMulBit),
                        ),
                    ..
                } => Fr::from_u64(2),
                DoryAssistOpeningId::Polynomial { .. } => Fr::from_u64(0),
            },
            |_| Fr::from_u64(2),
            |_| Fr::from_u64(0),
        );

        assert_ne!(output, Fr::from_u64(0));
    }

    #[test]
    fn scalar_mul_boundary_checks_initial_and_final_points() {
        let dimensions = G2Dimensions::new(8, 2, 0);
        let relation = DoryAssistRelationId::G2ScalarMultiplicationBoundary;
        let claims = scalar_multiplication_boundary::<Fr>(dimensions);
        let zero = Fr::from_u64(0);

        assert_eq!(claims.id, relation);
        assert_eq!(
            claims.sumcheck,
            scalar_multiplication_boundary_sumcheck(dimensions)
        );
        assert_eq!(
            claims.output.required_openings,
            scalar_multiplication_boundary_output_openings().to_vec()
        );
        assert_eq!(
            claims.required_challenges(),
            vec![DoryAssistChallengeId::from(G2Challenge::BoundaryPoint)]
        );

        let output = claims.output.expression().evaluate(
            |opening| match *opening {
                id if id == scalar_mul_boundary_accumulator_infinity_opening() => Fr::from_u64(1),
                id if id == scalar_mul_boundary_shifted_accumulator_x0_opening() => {
                    Fr::from_u64(13)
                }
                id if id == scalar_mul_boundary_shifted_accumulator_x1_opening() => {
                    Fr::from_u64(17)
                }
                id if id == scalar_mul_boundary_shifted_accumulator_y0_opening() => {
                    Fr::from_u64(19)
                }
                id if id == scalar_mul_boundary_shifted_accumulator_y1_opening() => {
                    Fr::from_u64(23)
                }
                _ => zero,
            },
            |_| Fr::from_u64(5),
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
                    component: 4,
                } if id == relation => Fr::from_u64(1),
                DoryAssistPublicId::BoundaryValue {
                    relation: id,
                    endpoint: DoryAssistBoundaryEndpoint::Final,
                    component: 0,
                } if id == relation => Fr::from_u64(13),
                DoryAssistPublicId::BoundaryValue {
                    relation: id,
                    endpoint: DoryAssistBoundaryEndpoint::Final,
                    component: 1,
                } if id == relation => Fr::from_u64(17),
                DoryAssistPublicId::BoundaryValue {
                    relation: id,
                    endpoint: DoryAssistBoundaryEndpoint::Final,
                    component: 2,
                } if id == relation => Fr::from_u64(19),
                DoryAssistPublicId::BoundaryValue {
                    relation: id,
                    endpoint: DoryAssistBoundaryEndpoint::Final,
                    component: 3,
                } if id == relation => Fr::from_u64(23),
                _ => zero,
            },
        );

        assert_eq!(output, zero);
    }

    #[test]
    fn addition_constraints_accept_all_infinity_row() {
        let claims = addition::<Fr>(G2Dimensions::new(8, 0, 0));
        let output = claims.output.expression().evaluate(
            |opening| match *opening {
                DoryAssistOpeningId::Polynomial {
                    polynomial:
                        super::super::super::DoryAssistPolynomialId::Virtual(
                            DoryAssistVirtualPolynomial::G2(
                                G2Polynomial::AddInputLeftInfinity
                                | G2Polynomial::AddInputRightInfinity
                                | G2Polynomial::AddOutputInfinity,
                            ),
                        ),
                    ..
                } => Fr::from_u64(1),
                DoryAssistOpeningId::Polynomial { .. } => Fr::from_u64(0),
            },
            |_| Fr::from_u64(2),
            |_| Fr::from_u64(0),
        );

        assert_eq!(output, Fr::from_u64(0));
    }
}
