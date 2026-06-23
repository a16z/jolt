use jolt_field::RingCore;

use crate::{challenge, opening, public};

use super::super::{
    FieldInlineChallengeId, FieldInlineCommittedPolynomial, FieldInlineExpr, FieldInlineOpeningId,
    FieldInlinePublicId, FieldInlineRelationClaims, FieldInlineRelationId,
    FieldInlineVirtualPolynomial, FieldRegistersReadWriteChallenge, FieldRegistersReadWritePublic,
    FieldRegistersValEvaluationPublic,
};
use super::dimensions::{
    FieldInlineSumcheckSpec, FieldRegistersReadWriteDimensions, FieldRegistersTraceDimensions,
};

pub const fn read_write_checking_sumcheck(
    dimensions: FieldRegistersReadWriteDimensions,
) -> FieldInlineSumcheckSpec {
    dimensions.read_write_sumcheck()
}

pub fn read_write_checking<F>(
    dimensions: FieldRegistersReadWriteDimensions,
) -> FieldInlineRelationClaims<F>
where
    F: RingCore,
{
    let gamma = read_write_challenge(FieldRegistersReadWriteChallenge::Gamma);
    let eq_cycle = read_write_public(FieldRegistersReadWritePublic::EqCycle);

    let input = opening(field_rd_value_claim())
        + gamma.clone() * opening(field_rs1_value_claim())
        + gamma.clone().pow(2) * opening(field_rs2_value_claim());

    let output =
        eq_cycle.clone() * opening(field_rd_wa_read_write()) * opening(field_rd_inc_read_write())
            + eq_cycle.clone()
                * opening(field_rd_wa_read_write())
                * opening(field_registers_val_read_write())
            + eq_cycle.clone()
                * gamma.clone()
                * opening(field_rs1_ra_read_write())
                * opening(field_registers_val_read_write())
            + eq_cycle
                * gamma.pow(2)
                * opening(field_rs2_ra_read_write())
                * opening(field_registers_val_read_write());

    FieldInlineRelationClaims::new(
        FieldInlineRelationId::FieldRegistersReadWriteChecking,
        read_write_checking_sumcheck(dimensions),
        input,
        output,
    )
}

pub fn val_evaluation<F>(dimensions: FieldRegistersTraceDimensions) -> FieldInlineRelationClaims<F>
where
    F: RingCore,
{
    let input = opening(field_registers_val_read_write());
    let output = val_evaluation_public(FieldRegistersValEvaluationPublic::LtCycle)
        * opening(field_rd_inc_val_evaluation())
        * opening(field_rd_wa_val_evaluation());

    FieldInlineRelationClaims::new(
        FieldInlineRelationId::FieldRegistersValEvaluation,
        dimensions.sumcheck(3),
        input,
        output,
    )
}

pub fn read_write_checking_input_openings() -> [FieldInlineOpeningId; 3] {
    [
        field_rd_value_claim(),
        field_rs1_value_claim(),
        field_rs2_value_claim(),
    ]
}

pub fn read_write_checking_output_openings() -> [FieldInlineOpeningId; 5] {
    [
        field_registers_val_read_write(),
        field_rs1_ra_read_write(),
        field_rs2_ra_read_write(),
        field_rd_wa_read_write(),
        field_rd_inc_read_write(),
    ]
}

pub fn val_evaluation_input_openings() -> [FieldInlineOpeningId; 1] {
    [field_registers_val_read_write()]
}

pub fn val_evaluation_output_openings() -> [FieldInlineOpeningId; 2] {
    [field_rd_inc_val_evaluation(), field_rd_wa_val_evaluation()]
}

fn read_write_challenge<F>(id: FieldRegistersReadWriteChallenge) -> FieldInlineExpr<F>
where
    F: RingCore,
{
    challenge(FieldInlineChallengeId::from(id))
}

fn read_write_public<F>(id: FieldRegistersReadWritePublic) -> FieldInlineExpr<F>
where
    F: RingCore,
{
    public(FieldInlinePublicId::from(id))
}

fn val_evaluation_public<F>(id: FieldRegistersValEvaluationPublic) -> FieldInlineExpr<F>
where
    F: RingCore,
{
    public(FieldInlinePublicId::from(id))
}

fn field_rd_value_claim() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRdValue,
        FieldInlineRelationId::FieldRegistersClaimReduction,
    )
}

fn field_rs1_value_claim() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRs1Value,
        FieldInlineRelationId::FieldRegistersClaimReduction,
    )
}

fn field_rs2_value_claim() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRs2Value,
        FieldInlineRelationId::FieldRegistersClaimReduction,
    )
}

fn field_registers_val_read_write() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRegistersVal,
        FieldInlineRelationId::FieldRegistersReadWriteChecking,
    )
}

fn field_rs1_ra_read_write() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRs1Ra,
        FieldInlineRelationId::FieldRegistersReadWriteChecking,
    )
}

fn field_rs2_ra_read_write() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRs2Ra,
        FieldInlineRelationId::FieldRegistersReadWriteChecking,
    )
}

fn field_rd_wa_read_write() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRdWa,
        FieldInlineRelationId::FieldRegistersReadWriteChecking,
    )
}

fn field_rd_inc_read_write() -> FieldInlineOpeningId {
    FieldInlineOpeningId::committed(
        FieldInlineCommittedPolynomial::FieldRdInc,
        FieldInlineRelationId::FieldRegistersReadWriteChecking,
    )
}

fn field_rd_inc_val_evaluation() -> FieldInlineOpeningId {
    FieldInlineOpeningId::committed(
        FieldInlineCommittedPolynomial::FieldRdInc,
        FieldInlineRelationId::FieldRegistersValEvaluation,
    )
}

fn field_rd_wa_val_evaluation() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRdWa,
        FieldInlineRelationId::FieldRegistersValEvaluation,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn trace_dimensions() -> FieldRegistersTraceDimensions {
        FieldRegistersTraceDimensions::new(5)
    }

    fn read_write_dimensions() -> FieldRegistersReadWriteDimensions {
        FieldRegistersReadWriteDimensions::new(5, 4, 2, 1)
    }

    #[test]
    fn read_write_claims_expose_expected_dependencies() {
        let claims = read_write_checking::<Fr>(read_write_dimensions());

        assert_eq!(
            claims.id,
            FieldInlineRelationId::FieldRegistersReadWriteChecking
        );
        assert_eq!(
            claims.sumcheck,
            read_write_checking_sumcheck(read_write_dimensions())
        );
        assert_eq!(
            claims.input.required_openings,
            read_write_checking_input_openings().to_vec()
        );
        assert_eq!(
            claims.output.required_openings,
            vec![
                field_rd_wa_read_write(),
                field_rd_inc_read_write(),
                field_registers_val_read_write(),
                field_rs1_ra_read_write(),
                field_rs2_ra_read_write(),
            ]
        );
        assert_eq!(
            read_write_checking_output_openings(),
            [
                field_registers_val_read_write(),
                field_rs1_ra_read_write(),
                field_rs2_ra_read_write(),
                field_rd_wa_read_write(),
                field_rd_inc_read_write(),
            ]
        );
        assert_eq!(
            claims.input.required_challenges,
            vec![FieldInlineChallengeId::from(
                FieldRegistersReadWriteChallenge::Gamma
            )]
        );
        assert_eq!(
            claims.output.required_challenges,
            vec![FieldInlineChallengeId::from(
                FieldRegistersReadWriteChallenge::Gamma
            )]
        );
        assert_eq!(
            claims.required_challenges(),
            vec![FieldInlineChallengeId::from(
                FieldRegistersReadWriteChallenge::Gamma
            )]
        );
        assert_eq!(
            claims.output.required_publics,
            vec![FieldInlinePublicId::from(
                FieldRegistersReadWritePublic::EqCycle
            )]
        );
        assert_eq!(
            claims.required_publics(),
            vec![FieldInlinePublicId::from(
                FieldRegistersReadWritePublic::EqCycle
            )]
        );
        assert_eq!(claims.num_challenges(), 1);
    }

    #[test]
    fn read_write_claims_evaluate_like_field_register_twist_formula() {
        let claims = read_write_checking::<Fr>(read_write_dimensions());

        let rd_value = Fr::from_u64(3);
        let rs1_value = Fr::from_u64(5);
        let rs2_value = Fr::from_u64(7);
        let val = Fr::from_u64(11);
        let rs1_ra = Fr::from_u64(13);
        let rs2_ra = Fr::from_u64(17);
        let rd_wa = Fr::from_u64(19);
        let inc = Fr::from_u64(23);
        let gamma = Fr::from_u64(29);
        let eq_cycle = Fr::from_u64(31);
        let zero = Fr::from_u64(0);

        let input = claims.input.expression().evaluate(
            |id| match *id {
                id if id == field_rd_value_claim() => rd_value,
                id if id == field_rs1_value_claim() => rs1_value,
                id if id == field_rs2_value_claim() => rs2_value,
                _ => zero,
            },
            |id| match *id {
                FieldInlineChallengeId::FieldRegistersReadWrite(
                    FieldRegistersReadWriteChallenge::Gamma,
                ) => gamma,
                _ => zero,
            },
            |_| zero,
        );

        let output = claims.output.expression().evaluate(
            |id| match *id {
                id if id == field_registers_val_read_write() => val,
                id if id == field_rs1_ra_read_write() => rs1_ra,
                id if id == field_rs2_ra_read_write() => rs2_ra,
                id if id == field_rd_wa_read_write() => rd_wa,
                id if id == field_rd_inc_read_write() => inc,
                _ => zero,
            },
            |id| match *id {
                FieldInlineChallengeId::FieldRegistersReadWrite(
                    FieldRegistersReadWriteChallenge::Gamma,
                ) => gamma,
                _ => zero,
            },
            |id| match *id {
                FieldInlinePublicId::FieldRegistersReadWrite(
                    FieldRegistersReadWritePublic::EqCycle,
                ) => eq_cycle,
                _ => zero,
            },
        );

        assert_eq!(
            input,
            rd_value + gamma * rs1_value + gamma * gamma * rs2_value
        );
        assert_eq!(
            output,
            eq_cycle * (rd_wa * (inc + val) + gamma * rs1_ra * val + gamma * gamma * rs2_ra * val)
        );
    }

    #[test]
    fn val_evaluation_claims_expose_expected_dependencies() {
        let claims = val_evaluation::<Fr>(trace_dimensions());

        assert_eq!(
            claims.id,
            FieldInlineRelationId::FieldRegistersValEvaluation
        );
        assert_eq!(claims.sumcheck, trace_dimensions().sumcheck(3));
        assert_eq!(
            claims.input.required_openings,
            val_evaluation_input_openings().to_vec()
        );
        assert_eq!(
            claims.output.required_openings,
            val_evaluation_output_openings().to_vec()
        );
        assert!(claims.output.required_challenges.is_empty());
        assert!(claims.required_challenges().is_empty());
        assert_eq!(
            claims.output.required_publics,
            vec![FieldInlinePublicId::from(
                FieldRegistersValEvaluationPublic::LtCycle
            )]
        );
        assert_eq!(
            claims.required_publics(),
            vec![FieldInlinePublicId::from(
                FieldRegistersValEvaluationPublic::LtCycle
            )]
        );
        assert_eq!(claims.num_challenges(), 0);
    }

    #[test]
    fn val_evaluation_claims_evaluate_like_field_register_twist_formula() {
        let claims = val_evaluation::<Fr>(trace_dimensions());

        let val = Fr::from_u64(3);
        let inc = Fr::from_u64(5);
        let wa = Fr::from_u64(7);
        let lt_cycle = Fr::from_u64(11);
        let zero = Fr::from_u64(0);

        let input = claims.input.expression().evaluate(
            |id| match *id {
                id if id == field_registers_val_read_write() => val,
                _ => zero,
            },
            |_| zero,
            |_| zero,
        );

        let output = claims.output.expression().evaluate(
            |id| match *id {
                id if id == field_rd_inc_val_evaluation() => inc,
                id if id == field_rd_wa_val_evaluation() => wa,
                _ => zero,
            },
            |_| zero,
            |id| match *id {
                FieldInlinePublicId::FieldRegistersValEvaluation(
                    FieldRegistersValEvaluationPublic::LtCycle,
                ) => lt_cycle,
                _ => zero,
            },
        );

        assert_eq!(input, val);
        assert_eq!(output, lt_cycle * inc * wa);
    }
}
