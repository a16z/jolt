//! field_inline rd-inc claim-reduction symbolic sumcheck relation.

use jolt_field::RingCore;

use crate::protocols::field_inline::formulas::claim_reductions::increments::{
    field_rd_inc_read_write, field_rd_inc_reduced, field_rd_inc_val_evaluation, inc_challenge,
    inc_public,
};
use crate::protocols::field_inline::{
    FieldInlineChallengeId, FieldInlineExpr, FieldInlineOpeningId, FieldInlinePublicId,
    FieldInlineRelationId, FieldInlineSumcheckSpec, FieldRegistersIncClaimReductionChallenge,
    FieldRegistersIncClaimReductionPublic, FieldRegistersTraceDimensions,
};
use crate::{opening, SymbolicSumcheck};

/// Reduces the two `FieldRdInc` openings (read/write and val-evaluation) to a
/// single reduced `FieldRdInc` opening, folding by `eta` and weighting by the
/// `EqReadWrite`/`EqValEvaluation` publics.
pub struct ClaimReduction {
    shape: FieldRegistersTraceDimensions,
}

impl SymbolicSumcheck for ClaimReduction {
    type RelationId = FieldInlineRelationId;
    type OpeningId = FieldInlineOpeningId;
    type PublicId = FieldInlinePublicId;
    type ChallengeId = FieldInlineChallengeId;
    type Shape = FieldRegistersTraceDimensions;

    fn new(shape: FieldRegistersTraceDimensions) -> Self {
        Self { shape }
    }

    fn id() -> FieldInlineRelationId {
        FieldInlineRelationId::FieldRegistersIncClaimReduction
    }

    fn spec(&self) -> FieldInlineSumcheckSpec {
        self.shape.sumcheck(2)
    }

    fn input_expression<F: RingCore>(&self) -> FieldInlineExpr<F> {
        let eta = inc_challenge(FieldRegistersIncClaimReductionChallenge::Gamma);

        opening(field_rd_inc_read_write()) + eta * opening(field_rd_inc_val_evaluation())
    }

    fn output_expression<F: RingCore>(&self) -> FieldInlineExpr<F> {
        let eta = inc_challenge(FieldRegistersIncClaimReductionChallenge::Gamma);

        let output_coeff = inc_public(FieldRegistersIncClaimReductionPublic::EqReadWrite)
            + eta * inc_public(FieldRegistersIncClaimReductionPublic::EqValEvaluation);
        output_coeff * opening(field_rd_inc_reduced())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::field_inline::formulas::claim_reductions::increments::{
        claim_reduction_input_openings, claim_reduction_output_openings,
    };
    use jolt_field::{Fr, FromPrimitiveInt};

    fn dimensions() -> FieldRegistersTraceDimensions {
        FieldRegistersTraceDimensions::new(5)
    }

    #[test]
    fn claim_reduction_exposes_expected_dependencies() {
        let relation = ClaimReduction::new(dimensions());

        assert_eq!(
            ClaimReduction::id(),
            FieldInlineRelationId::FieldRegistersIncClaimReduction
        );
        assert_eq!(relation.spec(), dimensions().sumcheck(2));
        assert_eq!(
            relation.input_expression::<Fr>().required_openings(),
            claim_reduction_input_openings().to_vec()
        );
        assert_eq!(
            relation.output_expression::<Fr>().required_openings(),
            claim_reduction_output_openings().to_vec()
        );
        assert_eq!(
            relation.required_challenges::<Fr>(),
            vec![FieldInlineChallengeId::from(
                FieldRegistersIncClaimReductionChallenge::Gamma
            )]
        );
        assert_eq!(
            relation.required_publics::<Fr>(),
            vec![
                FieldInlinePublicId::from(FieldRegistersIncClaimReductionPublic::EqReadWrite),
                FieldInlinePublicId::from(FieldRegistersIncClaimReductionPublic::EqValEvaluation),
            ]
        );
    }

    #[test]
    fn claim_reduction_evaluates_like_field_rd_inc_reduction_formula() {
        let relation = ClaimReduction::new(dimensions());

        let read_write_inc = Fr::from_u64(3);
        let val_evaluation_inc = Fr::from_u64(5);
        let reduced_inc = Fr::from_u64(7);
        let eta = Fr::from_u64(11);
        let eq_read_write = Fr::from_u64(13);
        let eq_val_evaluation = Fr::from_u64(17);
        let zero = Fr::from_u64(0);

        let input = relation.input_expression::<Fr>().evaluate(
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

        let output = relation.output_expression::<Fr>().evaluate(
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
