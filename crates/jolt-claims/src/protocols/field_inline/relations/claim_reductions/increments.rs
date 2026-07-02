//! field_inline rd-inc claim-reduction symbolic sumcheck relation.

use jolt_field::RingCore;

use crate::protocols::field_inline::geometry::claim_reductions::increments::{
    field_rd_inc_read_write, field_rd_inc_reduced, field_rd_inc_val_evaluation,
};
use crate::protocols::field_inline::{
    FieldInlineChallengeId, FieldInlineDerivedId, FieldInlineExpr, FieldInlineOpeningId,
    FieldInlineRelationId, FieldRegistersIncClaimReductionChallenge,
    FieldRegistersIncClaimReductionPublic, FieldRegistersTraceDimensions,
};
use crate::{challenge, derived, opening, SymbolicSumcheck};

/// Reduces the two `FieldRdInc` openings (read/write and val-evaluation) to a
/// single reduced `FieldRdInc` opening, folding by `eta` and weighting by the
/// `EqReadWrite`/`EqValEvaluation` publics.
pub struct ClaimReduction {
    shape: FieldRegistersTraceDimensions,
}

impl SymbolicSumcheck for ClaimReduction {
    type RelationId = FieldInlineRelationId;
    type OpeningId = FieldInlineOpeningId;
    type DerivedId = FieldInlineDerivedId;
    type ChallengeId = FieldInlineChallengeId;
    type Shape = FieldRegistersTraceDimensions;
    type Challenges<F> = crate::NoChallenges<F>;
    type Inputs<C> = crate::NoInputs<C>;
    type Outputs<C> = crate::NoOutputs<C>;

    fn new(shape: FieldRegistersTraceDimensions) -> Self {
        Self { shape }
    }

    fn id() -> FieldInlineRelationId {
        FieldInlineRelationId::FieldRegistersIncClaimReduction
    }

    fn rounds(&self) -> usize {
        self.shape.log_t()
    }

    fn degree(&self) -> usize {
        2
    }

    fn input_expression<F: RingCore>(&self) -> FieldInlineExpr<F> {
        let eta = challenge(FieldRegistersIncClaimReductionChallenge::Gamma);

        opening(field_rd_inc_read_write()) + eta * opening(field_rd_inc_val_evaluation())
    }

    fn output_expression<F: RingCore>(&self) -> FieldInlineExpr<F> {
        let eta = challenge(FieldRegistersIncClaimReductionChallenge::Gamma);

        let output_coeff = derived(FieldRegistersIncClaimReductionPublic::EqReadWrite)
            + eta * derived(FieldRegistersIncClaimReductionPublic::EqValEvaluation);
        output_coeff * opening(field_rd_inc_reduced())
    }
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
        let relation = ClaimReduction::new(dimensions());

        assert_eq!(
            ClaimReduction::id(),
            FieldInlineRelationId::FieldRegistersIncClaimReduction
        );
        assert_eq!(relation.rounds(), dimensions().log_t());
        assert_eq!(relation.degree(), 2);
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
                FieldInlineDerivedId::FieldRegistersIncClaimReduction(
                    FieldRegistersIncClaimReductionPublic::EqReadWrite,
                ) => eq_read_write,
                FieldInlineDerivedId::FieldRegistersIncClaimReduction(
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
