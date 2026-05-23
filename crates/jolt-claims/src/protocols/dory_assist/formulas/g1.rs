use jolt_field::{FromPrimitiveInt, RingCore};

use crate::{challenge, constant, opening};

use super::super::{
    DoryAssistChallengeId, DoryAssistExpr, DoryAssistOpeningId, DoryAssistRelationClaims,
    DoryAssistRelationId, DoryAssistVirtualPolynomial, G1Challenge, G1Polynomial,
};
use super::dimensions::{DoryAssistSumcheckSpec, G1Dimensions};

pub const fn scalar_multiplication_sumcheck(dimensions: G1Dimensions) -> DoryAssistSumcheckSpec {
    dimensions.scalar_mul_sumcheck()
}

pub const fn scalar_multiplication_shift_sumcheck(
    dimensions: G1Dimensions,
) -> DoryAssistSumcheckSpec {
    dimensions.scalar_mul_shift_sumcheck()
}

pub const fn addition_sumcheck(dimensions: G1Dimensions) -> DoryAssistSumcheckSpec {
    dimensions.add_sumcheck()
}

pub fn scalar_multiplication<F>(dimensions: G1Dimensions) -> DoryAssistRelationClaims<F>
where
    F: RingCore + FromPrimitiveInt,
{
    DoryAssistRelationClaims::new(
        DoryAssistRelationId::G1ScalarMultiplication,
        scalar_multiplication_sumcheck(dimensions),
        constant(F::zero()),
        scalar_multiplication_constraint_expression(),
    )
}

pub fn scalar_multiplication_shift<F>(dimensions: G1Dimensions) -> DoryAssistRelationClaims<F>
where
    F: RingCore,
{
    let gamma = g1_challenge(G1Challenge::ShiftGamma);
    let input = opening(scalar_mul_shifted_accumulator_x_opening())
        + gamma.clone() * opening(scalar_mul_shifted_accumulator_y_opening());
    let output = opening(scalar_mul_accumulator_x_opening())
        + gamma * opening(scalar_mul_accumulator_y_opening());

    DoryAssistRelationClaims::new(
        DoryAssistRelationId::G1ScalarMultiplicationShift,
        scalar_multiplication_shift_sumcheck(dimensions),
        input,
        output,
    )
}

pub fn addition<F>(dimensions: G1Dimensions) -> DoryAssistRelationClaims<F>
where
    F: RingCore + FromPrimitiveInt,
{
    DoryAssistRelationClaims::new(
        DoryAssistRelationId::G1Addition,
        addition_sumcheck(dimensions),
        constant(F::zero()),
        addition_constraint_expression(),
    )
}

pub fn scalar_multiplication_input_openings() -> Vec<DoryAssistOpeningId> {
    let relation = DoryAssistRelationId::G1ScalarMultiplication;
    vec![
        g1_opening(G1Polynomial::ScalarMulAccumulatorX, relation),
        g1_opening(G1Polynomial::ScalarMulAccumulatorY, relation),
        g1_opening(G1Polynomial::ScalarMulAccumulatorInfinity, relation),
        g1_opening(G1Polynomial::ScalarMulDoubledX, relation),
        g1_opening(G1Polynomial::ScalarMulDoubledY, relation),
        g1_opening(G1Polynomial::ScalarMulDoubledInfinity, relation),
        g1_opening(G1Polynomial::ScalarMulShiftedAccumulatorX, relation),
        g1_opening(G1Polynomial::ScalarMulShiftedAccumulatorY, relation),
        g1_opening(G1Polynomial::ScalarMulBit, relation),
        g1_opening(G1Polynomial::ScalarMulBaseX, relation),
        g1_opening(G1Polynomial::ScalarMulBaseY, relation),
    ]
}

pub fn scalar_multiplication_shift_input_openings() -> [DoryAssistOpeningId; 2] {
    [
        scalar_mul_shifted_accumulator_x_opening(),
        scalar_mul_shifted_accumulator_y_opening(),
    ]
}

pub fn scalar_multiplication_shift_output_openings() -> [DoryAssistOpeningId; 2] {
    [
        scalar_mul_accumulator_x_opening(),
        scalar_mul_accumulator_y_opening(),
    ]
}

pub fn addition_input_openings() -> Vec<DoryAssistOpeningId> {
    let relation = DoryAssistRelationId::G1Addition;
    vec![
        g1_opening(G1Polynomial::AddInputLeftX, relation),
        g1_opening(G1Polynomial::AddInputLeftY, relation),
        g1_opening(G1Polynomial::AddInputLeftInfinity, relation),
        g1_opening(G1Polynomial::AddInputRightX, relation),
        g1_opening(G1Polynomial::AddInputRightY, relation),
        g1_opening(G1Polynomial::AddInputRightInfinity, relation),
        g1_opening(G1Polynomial::AddOutputX, relation),
        g1_opening(G1Polynomial::AddOutputY, relation),
        g1_opening(G1Polynomial::AddOutputInfinity, relation),
        g1_opening(G1Polynomial::AddSlope, relation),
        g1_opening(G1Polynomial::AddInverse, relation),
        g1_opening(G1Polynomial::AddBranchSelector(0), relation),
        g1_opening(G1Polynomial::AddBranchSelector(1), relation),
    ]
}

pub fn scalar_mul_accumulator_x_opening() -> DoryAssistOpeningId {
    g1_opening(
        G1Polynomial::ScalarMulAccumulatorX,
        DoryAssistRelationId::G1ScalarMultiplicationShift,
    )
}

pub fn scalar_mul_accumulator_y_opening() -> DoryAssistOpeningId {
    g1_opening(
        G1Polynomial::ScalarMulAccumulatorY,
        DoryAssistRelationId::G1ScalarMultiplicationShift,
    )
}

pub fn scalar_mul_shifted_accumulator_x_opening() -> DoryAssistOpeningId {
    g1_opening(
        G1Polynomial::ScalarMulShiftedAccumulatorX,
        DoryAssistRelationId::G1ScalarMultiplicationShift,
    )
}

pub fn scalar_mul_shifted_accumulator_y_opening() -> DoryAssistOpeningId {
    g1_opening(
        G1Polynomial::ScalarMulShiftedAccumulatorY,
        DoryAssistRelationId::G1ScalarMultiplicationShift,
    )
}

pub fn scalar_multiplication_constraint_expression<F>() -> DoryAssistExpr<F>
where
    F: RingCore + FromPrimitiveInt,
{
    let relation = DoryAssistRelationId::G1ScalarMultiplication;
    let x_a = g1_polynomial(G1Polynomial::ScalarMulAccumulatorX, relation);
    let y_a = g1_polynomial(G1Polynomial::ScalarMulAccumulatorY, relation);
    let iota_a = g1_polynomial(G1Polynomial::ScalarMulAccumulatorInfinity, relation);
    let x_t = g1_polynomial(G1Polynomial::ScalarMulDoubledX, relation);
    let y_t = g1_polynomial(G1Polynomial::ScalarMulDoubledY, relation);
    let iota_t = g1_polynomial(G1Polynomial::ScalarMulDoubledInfinity, relation);
    let x_a_next = g1_polynomial(G1Polynomial::ScalarMulShiftedAccumulatorX, relation);
    let y_a_next = g1_polynomial(G1Polynomial::ScalarMulShiftedAccumulatorY, relation);
    let bit = g1_polynomial(G1Polynomial::ScalarMulBit, relation);
    let x_p = g1_polynomial(G1Polynomial::ScalarMulBaseX, relation);
    let y_p = g1_polynomial(G1Polynomial::ScalarMulBaseY, relation);

    let dx = x_p.clone() - x_t.clone();
    let dy = y_p.clone() - y_t.clone();

    let c1 =
        4 * square(y_a.clone()) * (x_t.clone() + 2 * x_a.clone()) - 9 * square(square(x_a.clone()));
    let c2 = 3 * square(x_a.clone()) * (x_t.clone() - x_a.clone())
        + 2 * y_a.clone() * (y_t.clone() + y_a);
    let c3 = (one::<F>() - bit.clone()) * (x_a_next.clone() - x_t.clone())
        + bit.clone() * iota_t.clone() * (x_a_next.clone() - x_p.clone())
        + bit.clone()
            * (one::<F>() - iota_t.clone())
            * ((x_a_next.clone() + x_t.clone() + x_p) * square(dx.clone()) - square(dy.clone()));
    let c4 = (one::<F>() - bit.clone()) * (y_a_next.clone() - y_t.clone())
        + bit.clone() * iota_t.clone() * (y_a_next.clone() - y_p)
        + bit
            * (one::<F>() - iota_t.clone())
            * ((y_a_next + y_t.clone()) * dx - dy * (x_t.clone() - x_a_next));
    let c5 = iota_a * (one::<F>() - iota_t.clone());
    let c6 = iota_t.clone() * x_t;
    let c7 = iota_t * y_t;

    batch_constraints(
        g1_challenge(G1Challenge::ConstraintBatch),
        [c1, c2, c3, c4, c5, c6, c7],
    )
}

pub fn addition_constraint_expression<F>() -> DoryAssistExpr<F>
where
    F: RingCore + FromPrimitiveInt,
{
    let relation = DoryAssistRelationId::G1Addition;
    let x_p = g1_polynomial(G1Polynomial::AddInputLeftX, relation);
    let y_p = g1_polynomial(G1Polynomial::AddInputLeftY, relation);
    let iota_p = g1_polynomial(G1Polynomial::AddInputLeftInfinity, relation);
    let x_q = g1_polynomial(G1Polynomial::AddInputRightX, relation);
    let y_q = g1_polynomial(G1Polynomial::AddInputRightY, relation);
    let iota_q = g1_polynomial(G1Polynomial::AddInputRightInfinity, relation);
    let x_r = g1_polynomial(G1Polynomial::AddOutputX, relation);
    let y_r = g1_polynomial(G1Polynomial::AddOutputY, relation);
    let iota_r = g1_polynomial(G1Polynomial::AddOutputInfinity, relation);
    let lambda = g1_polynomial(G1Polynomial::AddSlope, relation);
    let mu = g1_polynomial(G1Polynomial::AddInverse, relation);
    let sigma_1 = g1_polynomial(G1Polynomial::AddBranchSelector(0), relation);
    let sigma_2 = g1_polynomial(G1Polynomial::AddBranchSelector(1), relation);

    let dx = x_q.clone() - x_p.clone();
    let dy = y_q.clone() - y_p.clone();
    let phi = (one::<F>() - iota_p.clone()) * (one::<F>() - iota_q.clone());
    let add_branch = one::<F>() - sigma_1.clone() - sigma_2.clone();
    let not_inverse = one::<F>() - sigma_2.clone();

    let constraints = [
        iota_p.clone() * (one::<F>() - iota_p.clone()),
        iota_q.clone() * (one::<F>() - iota_q.clone()),
        iota_r.clone() * (one::<F>() - iota_r.clone()),
        iota_p.clone() * x_p.clone(),
        iota_p.clone() * y_p.clone(),
        iota_q.clone() * x_q.clone(),
        iota_q.clone() * y_q.clone(),
        iota_r.clone() * x_r.clone(),
        iota_r.clone() * y_r.clone(),
        iota_p.clone() * (x_r.clone() - x_q.clone()),
        iota_p.clone() * (y_r.clone() - y_q.clone()),
        iota_p.clone() * (iota_r.clone() - iota_q.clone()),
        iota_q.clone() * (one::<F>() - iota_p.clone()) * (x_r.clone() - x_p.clone()),
        iota_q.clone() * (one::<F>() - iota_p.clone()) * (y_r.clone() - y_p.clone()),
        iota_q.clone() * (one::<F>() - iota_p.clone()) * (iota_r.clone() - iota_p),
        phi.clone() * sigma_1.clone() * (one::<F>() - sigma_1.clone()),
        phi.clone() * sigma_2.clone() * (one::<F>() - sigma_2.clone()),
        phi.clone() * add_branch.clone() * (one::<F>() - mu * dx.clone()),
        phi.clone() * sigma_1.clone() * dx.clone(),
        phi.clone() * sigma_1.clone() * dy.clone(),
        phi.clone() * sigma_2.clone() * dx.clone(),
        phi.clone() * sigma_2.clone() * (y_q.clone() + y_p.clone()),
        phi.clone()
            * (add_branch.clone() * (lambda.clone() * dx - dy)
                + sigma_1 * (2 * y_p.clone() * lambda.clone() - 3 * square(x_p.clone()))),
        phi.clone() * sigma_2.clone() * (one::<F>() - iota_r.clone()),
        phi.clone() * not_inverse.clone() * iota_r,
        phi.clone()
            * not_inverse.clone()
            * (x_r.clone() - square(lambda.clone()) + x_p.clone() + x_q),
        phi * not_inverse * (y_r - lambda * (x_p - x_r) + y_p),
    ];

    batch_constraints(g1_challenge(G1Challenge::ConstraintBatch), constraints)
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

fn g1_polynomial<F>(polynomial: G1Polynomial, relation: DoryAssistRelationId) -> DoryAssistExpr<F>
where
    F: RingCore,
{
    opening(g1_opening(polynomial, relation))
}

fn g1_opening(polynomial: G1Polynomial, relation: DoryAssistRelationId) -> DoryAssistOpeningId {
    DoryAssistOpeningId::virtual_polynomial(DoryAssistVirtualPolynomial::G1(polynomial), relation)
}

pub fn g1_challenge<F>(id: G1Challenge) -> DoryAssistExpr<F>
where
    F: RingCore,
{
    challenge(DoryAssistChallengeId::from(id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    #[test]
    fn scalar_mul_shift_batches_coordinates_with_gamma() {
        let claims = scalar_multiplication_shift::<Fr>(G1Dimensions::new(8, 0, 0));
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
            vec![DoryAssistChallengeId::from(G1Challenge::ShiftGamma)]
        );

        let input = claims.input.expression().evaluate(
            |opening| match *opening {
                id if id == scalar_mul_shifted_accumulator_x_opening() => Fr::from_u64(3),
                id if id == scalar_mul_shifted_accumulator_y_opening() => Fr::from_u64(5),
                _ => zero,
            },
            |_| Fr::from_u64(7),
            |_| zero,
        );

        assert_eq!(input, Fr::from_u64(38));
    }

    #[test]
    fn local_constraint_relations_are_batched_zero_checks() {
        let dimensions = G1Dimensions::new(8, 2, 3);
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
            vec![DoryAssistChallengeId::from(G1Challenge::ConstraintBatch)]
        );
        assert_eq!(addition.sumcheck, addition_sumcheck(dimensions));
        let mut actual_addition_openings = addition.output.required_openings.clone();
        let mut expected_addition_openings = addition_input_openings();
        actual_addition_openings.sort();
        expected_addition_openings.sort();
        assert_eq!(actual_addition_openings, expected_addition_openings);
        assert_eq!(
            addition.required_challenges(),
            vec![DoryAssistChallengeId::from(G1Challenge::ConstraintBatch)]
        );
    }

    #[test]
    fn scalar_mul_constraints_accept_all_infinity_zero_row() {
        let claims = scalar_multiplication::<Fr>(G1Dimensions::new(8, 0, 0));
        let output = claims.output.expression().evaluate(
            |opening| match *opening {
                DoryAssistOpeningId::Polynomial {
                    polynomial:
                        super::super::super::DoryAssistPolynomialId::Virtual(
                            DoryAssistVirtualPolynomial::G1(
                                G1Polynomial::ScalarMulAccumulatorInfinity
                                | G1Polynomial::ScalarMulDoubledInfinity,
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
        let claims = scalar_multiplication::<Fr>(G1Dimensions::new(8, 0, 0));
        let output = claims.output.expression().evaluate(
            |opening| match *opening {
                DoryAssistOpeningId::Polynomial {
                    polynomial:
                        super::super::super::DoryAssistPolynomialId::Virtual(
                            DoryAssistVirtualPolynomial::G1(
                                G1Polynomial::ScalarMulAccumulatorInfinity,
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
    fn addition_constraints_accept_all_infinity_row() {
        let claims = addition::<Fr>(G1Dimensions::new(8, 0, 0));
        let output = claims.output.expression().evaluate(
            |opening| match *opening {
                DoryAssistOpeningId::Polynomial {
                    polynomial:
                        super::super::super::DoryAssistPolynomialId::Virtual(
                            DoryAssistVirtualPolynomial::G1(
                                G1Polynomial::AddInputLeftInfinity
                                | G1Polynomial::AddInputRightInfinity
                                | G1Polynomial::AddOutputInfinity,
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
