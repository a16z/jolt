use jolt_field::RingCore;

use crate::{challenge, opening, public};

use super::super::{
    DoryAssistChallengeId, DoryAssistOpeningId, DoryAssistPublicId, DoryAssistRelationClaims,
    DoryAssistRelationId, DoryAssistVirtualPolynomial, GtChallenge, GtPolynomial,
};
use super::dimensions::{DoryAssistSumcheckSpec, GtDimensions, GT_EXP_BASE};

pub const fn exponentiation_sumcheck(dimensions: GtDimensions) -> DoryAssistSumcheckSpec {
    dimensions.exp_sumcheck(GT_EXP_BASE + 4)
}

pub const fn exponentiation_shift_sumcheck(dimensions: GtDimensions) -> DoryAssistSumcheckSpec {
    dimensions.exp_shift_sumcheck()
}

pub const fn multiplication_sumcheck(dimensions: GtDimensions) -> DoryAssistSumcheckSpec {
    dimensions.mul_sumcheck()
}

pub fn exponentiation<F>(dimensions: GtDimensions) -> DoryAssistRelationClaims<F>
where
    F: RingCore,
{
    let output = opening(exp_accumulator_opening()).pow(GT_EXP_BASE)
        * opening(exp_digit_selector_opening())
        + opening(exp_quotient_opening()) * opening(exp_modulus_opening());

    DoryAssistRelationClaims::new(
        DoryAssistRelationId::GtExponentiation,
        exponentiation_sumcheck(dimensions),
        opening(exp_shifted_accumulator_opening()),
        output,
    )
    .with_input_challenges([DoryAssistChallengeId::from(GtChallenge::InstanceBatch)])
}

pub fn exponentiation_shift<F>(dimensions: GtDimensions) -> DoryAssistRelationClaims<F>
where
    F: RingCore,
{
    let output = gt_shift_eq_kernel() * opening(exp_accumulator_shift_opening());

    DoryAssistRelationClaims::new(
        DoryAssistRelationId::GtExponentiationShift,
        exponentiation_shift_sumcheck(dimensions),
        opening(exp_shifted_accumulator_opening()),
        output,
    )
}

pub fn multiplication<F>(dimensions: GtDimensions) -> DoryAssistRelationClaims<F>
where
    F: RingCore,
{
    let output = opening(mul_output_opening())
        + opening(mul_quotient_opening()) * opening(mul_modulus_opening());

    DoryAssistRelationClaims::new(
        DoryAssistRelationId::GtMultiplication,
        multiplication_sumcheck(dimensions),
        opening(mul_left_opening()) * opening(mul_right_opening()),
        output,
    )
}

pub fn exponentiation_input_openings() -> [DoryAssistOpeningId; 1] {
    [exp_shifted_accumulator_opening()]
}

pub fn exponentiation_output_openings() -> [DoryAssistOpeningId; 4] {
    [
        exp_accumulator_opening(),
        exp_digit_selector_opening(),
        exp_quotient_opening(),
        exp_modulus_opening(),
    ]
}

pub fn exponentiation_shift_input_openings() -> [DoryAssistOpeningId; 1] {
    [exp_shifted_accumulator_opening()]
}

pub fn exponentiation_shift_output_openings() -> [DoryAssistOpeningId; 1] {
    [exp_accumulator_shift_opening()]
}

pub fn multiplication_input_openings() -> [DoryAssistOpeningId; 2] {
    [mul_left_opening(), mul_right_opening()]
}

pub fn multiplication_output_openings() -> [DoryAssistOpeningId; 3] {
    [
        mul_output_opening(),
        mul_quotient_opening(),
        mul_modulus_opening(),
    ]
}

pub fn exp_accumulator_opening() -> DoryAssistOpeningId {
    gt_opening(
        GtPolynomial::ExpAccumulator,
        DoryAssistRelationId::GtExponentiation,
    )
}

pub fn exp_shifted_accumulator_opening() -> DoryAssistOpeningId {
    gt_opening(
        GtPolynomial::ExpShiftedAccumulator,
        DoryAssistRelationId::GtExponentiation,
    )
}

pub fn exp_accumulator_shift_opening() -> DoryAssistOpeningId {
    gt_opening(
        GtPolynomial::ExpAccumulator,
        DoryAssistRelationId::GtExponentiationShift,
    )
}

pub fn exp_quotient_opening() -> DoryAssistOpeningId {
    gt_opening(
        GtPolynomial::ExpQuotient,
        DoryAssistRelationId::GtExponentiation,
    )
}

pub fn exp_digit_selector_opening() -> DoryAssistOpeningId {
    gt_opening(
        GtPolynomial::ExpDigitSelector,
        DoryAssistRelationId::GtExponentiation,
    )
}

pub fn exp_modulus_opening() -> DoryAssistOpeningId {
    gt_opening(
        GtPolynomial::Modulus,
        DoryAssistRelationId::GtExponentiation,
    )
}

pub fn mul_modulus_opening() -> DoryAssistOpeningId {
    gt_opening(
        GtPolynomial::Modulus,
        DoryAssistRelationId::GtMultiplication,
    )
}

pub fn mul_left_opening() -> DoryAssistOpeningId {
    gt_opening(
        GtPolynomial::MulLeft,
        DoryAssistRelationId::GtMultiplication,
    )
}

pub fn mul_right_opening() -> DoryAssistOpeningId {
    gt_opening(
        GtPolynomial::MulRight,
        DoryAssistRelationId::GtMultiplication,
    )
}

pub fn mul_output_opening() -> DoryAssistOpeningId {
    gt_opening(
        GtPolynomial::MulOutput,
        DoryAssistRelationId::GtMultiplication,
    )
}

pub fn mul_quotient_opening() -> DoryAssistOpeningId {
    gt_opening(
        GtPolynomial::MulQuotient,
        DoryAssistRelationId::GtMultiplication,
    )
}

fn gt_opening(polynomial: GtPolynomial, relation: DoryAssistRelationId) -> DoryAssistOpeningId {
    DoryAssistOpeningId::virtual_polynomial(DoryAssistVirtualPolynomial::Gt(polynomial), relation)
}

pub fn gt_challenge<F>(id: GtChallenge) -> super::super::DoryAssistExpr<F>
where
    F: RingCore,
{
    challenge(DoryAssistChallengeId::from(id))
}

fn gt_shift_eq_kernel<F>() -> super::super::DoryAssistExpr<F>
where
    F: RingCore,
{
    public(DoryAssistPublicId::GtShiftEqKernel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    #[test]
    fn multiplication_claims_expose_polynomial_division_identity() {
        let dimensions = GtDimensions::new(7, 2, 3);
        let claims = multiplication::<Fr>(dimensions);

        assert_eq!(claims.id, DoryAssistRelationId::GtMultiplication);
        assert_eq!(claims.sumcheck, multiplication_sumcheck(dimensions));
        assert_eq!(
            claims.input.required_openings,
            multiplication_input_openings().to_vec()
        );
        assert_eq!(
            claims.output.required_openings,
            multiplication_output_openings().to_vec()
        );
    }

    #[test]
    fn multiplication_evaluates_quotient_identity_shape() {
        let claims = multiplication::<Fr>(GtDimensions::new(7, 0, 0));
        let zero = Fr::from_u64(0);

        let input = claims.input.expression().evaluate(
            |opening| match *opening {
                id if id == mul_left_opening() => Fr::from_u64(6),
                id if id == mul_right_opening() => Fr::from_u64(7),
                _ => zero,
            },
            |_| zero,
            |_| zero,
        );
        let output = claims.output.expression().evaluate(
            |opening| match *opening {
                id if id == mul_output_opening() => Fr::from_u64(10),
                id if id == mul_quotient_opening() => Fr::from_u64(4),
                id if id == mul_modulus_opening() => Fr::from_u64(8),
                _ => zero,
            },
            |_| zero,
            |_| zero,
        );

        assert_eq!(input, Fr::from_u64(42));
        assert_eq!(output, Fr::from_u64(42));
    }

    #[test]
    fn exponentiation_shift_connects_rho_next_to_shifted_rho_kernel() {
        let claims = exponentiation_shift::<Fr>(GtDimensions::new(7, 2, 0));

        assert_eq!(claims.id, DoryAssistRelationId::GtExponentiationShift);
        assert_eq!(
            claims.sumcheck,
            exponentiation_shift_sumcheck(GtDimensions::new(7, 2, 0))
        );
        assert_eq!(
            claims.input.required_openings,
            exponentiation_shift_input_openings().to_vec()
        );
        assert_eq!(
            claims.output.required_openings,
            exponentiation_shift_output_openings().to_vec()
        );
        assert_eq!(
            claims.output.required_publics,
            vec![DoryAssistPublicId::GtShiftEqKernel]
        );
        assert!(claims.required_challenges().is_empty());

        let output = claims.output.expression().evaluate(
            |opening| match *opening {
                id if id == exp_accumulator_shift_opening() => Fr::from_u64(11),
                _ => Fr::from_u64(0),
            },
            |_| Fr::from_u64(0),
            |public| match *public {
                DoryAssistPublicId::GtShiftEqKernel => Fr::from_u64(7),
                _ => Fr::from_u64(0),
            },
        );

        assert_eq!(output, Fr::from_u64(77));
    }
}
