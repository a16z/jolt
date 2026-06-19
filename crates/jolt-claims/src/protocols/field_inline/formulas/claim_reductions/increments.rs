use jolt_field::RingCore;

use crate::{challenge, opening, public};

use super::super::super::{
    FieldInlineChallengeId, FieldInlineCommittedPolynomial, FieldInlineExpr, FieldInlineOpeningId,
    FieldInlinePublicId, FieldInlineRelationClaims, FieldInlineRelationId,
    FieldRegistersIncClaimReductionChallenge, FieldRegistersIncClaimReductionPublic,
};
use super::super::dimensions::FieldRegistersTraceDimensions;

pub fn claim_reduction<F>(dimensions: FieldRegistersTraceDimensions) -> FieldInlineRelationClaims<F>
where
    F: RingCore,
{
    let eta = inc_challenge(FieldRegistersIncClaimReductionChallenge::Gamma);

    let input =
        opening(field_rd_inc_read_write()) + eta.clone() * opening(field_rd_inc_val_evaluation());

    let output_coeff = inc_public(FieldRegistersIncClaimReductionPublic::EqReadWrite)
        + eta * inc_public(FieldRegistersIncClaimReductionPublic::EqValEvaluation);
    let output = output_coeff * opening(field_rd_inc_reduced());

    FieldInlineRelationClaims::new(
        FieldInlineRelationId::FieldRegistersIncClaimReduction,
        dimensions.sumcheck(2),
        input,
        output,
    )
}

pub fn claim_reduction_input_openings() -> [FieldInlineOpeningId; 2] {
    [field_rd_inc_read_write(), field_rd_inc_val_evaluation()]
}

pub fn claim_reduction_output_openings() -> [FieldInlineOpeningId; 1] {
    [field_rd_inc_reduced()]
}

pub fn field_rd_inc_read_write_opening() -> FieldInlineOpeningId {
    field_rd_inc_read_write()
}

pub fn field_rd_inc_val_evaluation_opening() -> FieldInlineOpeningId {
    field_rd_inc_val_evaluation()
}

pub fn field_rd_inc_reduced_opening() -> FieldInlineOpeningId {
    field_rd_inc_reduced()
}

fn inc_challenge<F>(id: FieldRegistersIncClaimReductionChallenge) -> FieldInlineExpr<F>
where
    F: RingCore,
{
    challenge(FieldInlineChallengeId::from(id))
}

fn inc_public<F>(id: FieldRegistersIncClaimReductionPublic) -> FieldInlineExpr<F>
where
    F: RingCore,
{
    public(FieldInlinePublicId::from(id))
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

fn field_rd_inc_reduced() -> FieldInlineOpeningId {
    FieldInlineOpeningId::committed(
        FieldInlineCommittedPolynomial::FieldRdInc,
        FieldInlineRelationId::FieldRegistersIncClaimReduction,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn dimensions() -> FieldRegistersTraceDimensions {
        FieldRegistersTraceDimensions::new(5)
    }

    #[test]
    fn claim_reduction_exposes_expected_dependencies() {
        let claims = claim_reduction::<Fr>(dimensions());

        assert_eq!(
            claims.id,
            FieldInlineRelationId::FieldRegistersIncClaimReduction
        );
        assert_eq!(claims.sumcheck, dimensions().sumcheck(2));
        assert_eq!(
            claims.input.required_openings,
            claim_reduction_input_openings().to_vec()
        );
        assert_eq!(
            claims.output.required_openings,
            claim_reduction_output_openings().to_vec()
        );
        assert_eq!(
            claims.required_challenges(),
            vec![FieldInlineChallengeId::from(
                FieldRegistersIncClaimReductionChallenge::Gamma
            )]
        );
        assert_eq!(
            claims.required_publics(),
            vec![
                FieldInlinePublicId::from(FieldRegistersIncClaimReductionPublic::EqReadWrite),
                FieldInlinePublicId::from(FieldRegistersIncClaimReductionPublic::EqValEvaluation),
            ]
        );
        assert_eq!(claims.num_challenges(), 1);
    }

    #[test]
    fn claim_reduction_evaluates_like_field_rd_inc_reduction_formula() {
        let claims = claim_reduction::<Fr>(dimensions());

        let read_write_inc = Fr::from_u64(3);
        let val_evaluation_inc = Fr::from_u64(5);
        let reduced_inc = Fr::from_u64(7);
        let eta = Fr::from_u64(11);
        let eq_read_write = Fr::from_u64(13);
        let eq_val_evaluation = Fr::from_u64(17);
        let zero = Fr::from_u64(0);

        let input = claims.input.expression().evaluate(
            |id| match *id {
                id if id == field_rd_inc_read_write() => read_write_inc,
                id if id == field_rd_inc_val_evaluation() => val_evaluation_inc,
                _ => zero,
            },
            |id| match *id {
                FieldInlineChallengeId::FieldRegistersIncClaimReduction(
                    FieldRegistersIncClaimReductionChallenge::Gamma,
                ) => eta,
                _ => zero,
            },
            |_| zero,
        );

        let output = claims.output.expression().evaluate(
            |id| match *id {
                id if id == field_rd_inc_reduced() => reduced_inc,
                _ => zero,
            },
            |id| match *id {
                FieldInlineChallengeId::FieldRegistersIncClaimReduction(
                    FieldRegistersIncClaimReductionChallenge::Gamma,
                ) => eta,
                _ => zero,
            },
            |id| match *id {
                FieldInlinePublicId::FieldRegistersIncClaimReduction(
                    FieldRegistersIncClaimReductionPublic::EqReadWrite,
                ) => eq_read_write,
                FieldInlinePublicId::FieldRegistersIncClaimReduction(
                    FieldRegistersIncClaimReductionPublic::EqValEvaluation,
                ) => eq_val_evaluation,
                _ => zero,
            },
        );

        assert_eq!(input, read_write_inc + eta * val_evaluation_inc);
        assert_eq!(
            output,
            (eq_read_write + eta * eq_val_evaluation) * reduced_inc
        );
    }
}
