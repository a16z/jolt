//! field_inline registers claim-reduction symbolic sumcheck relation.

use jolt_field::RingCore;

use crate::protocols::field_inline::geometry::claim_reductions::registers::{
    field_rd_value_reduced, field_rd_value_spartan, field_rs1_value_reduced,
    field_rs1_value_spartan, field_rs2_value_reduced, field_rs2_value_spartan,
};
use crate::protocols::field_inline::{
    FieldInlineChallengeId, FieldInlineDerivedId, FieldInlineExpr, FieldInlineOpeningId,
    FieldInlineRelationId, FieldInlineSumcheckSpec, FieldRegistersClaimReductionChallenge,
    FieldRegistersClaimReductionPublic, FieldRegistersTraceDimensions,
};
use crate::{challenge, derived, opening, SymbolicSumcheck};

/// Batches the native field-register Spartan-outer openings (`FieldRdValue`,
/// `FieldRs1Value`, `FieldRs2Value`) by `gamma` and reduces them to the
/// registers-claim-reduction openings weighted by the `EqSpartan` public.
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

    fn new(shape: FieldRegistersTraceDimensions) -> Self {
        Self { shape }
    }

    fn id() -> FieldInlineRelationId {
        FieldInlineRelationId::FieldRegistersClaimReduction
    }

    fn spec(&self) -> FieldInlineSumcheckSpec {
        self.shape.sumcheck(2)
    }

    fn input_expression<F: RingCore>(&self) -> FieldInlineExpr<F> {
        let gamma = challenge(FieldRegistersClaimReductionChallenge::Gamma);

        opening(field_rd_value_spartan())
            + gamma.clone() * opening(field_rs1_value_spartan())
            + gamma.clone().pow(2) * opening(field_rs2_value_spartan())
    }

    fn output_expression<F: RingCore>(&self) -> FieldInlineExpr<F> {
        let gamma = challenge(FieldRegistersClaimReductionChallenge::Gamma);
        let eq_spartan = derived(FieldRegistersClaimReductionPublic::EqSpartan);

        eq_spartan.clone() * opening(field_rd_value_reduced())
            + eq_spartan.clone() * gamma.clone() * opening(field_rs1_value_reduced())
            + eq_spartan * gamma.pow(2) * opening(field_rs2_value_reduced())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::field_inline::geometry::claim_reductions::registers::{
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
            FieldInlineRelationId::FieldRegistersClaimReduction
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
                FieldRegistersClaimReductionChallenge::Gamma
            )]
        );
        assert_eq!(
            relation.required_deriveds::<Fr>(),
            vec![FieldInlineDerivedId::from(
                FieldRegistersClaimReductionPublic::EqSpartan
            )]
        );
    }

    #[test]
    fn claim_reduction_evaluates_like_field_register_twist_formula() {
        let relation = ClaimReduction::new(dimensions());

        let rd_spartan = Fr::from_u64(3);
        let rs1_spartan = Fr::from_u64(5);
        let rs2_spartan = Fr::from_u64(7);
        let rd_reduced = Fr::from_u64(11);
        let rs1_reduced = Fr::from_u64(13);
        let rs2_reduced = Fr::from_u64(17);
        let gamma = Fr::from_u64(19);
        let eq_spartan = Fr::from_u64(23);
        let zero = Fr::from_u64(0);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == field_rd_value_spartan() => rd_spartan,
                id if id == field_rs1_value_spartan() => rs1_spartan,
                id if id == field_rs2_value_spartan() => rs2_spartan,
                _ => zero,
            },
            |id| match *id {
                FieldInlineChallengeId::FieldRegistersClaimReduction(
                    FieldRegistersClaimReductionChallenge::Gamma,
                ) => gamma,
                _ => zero,
            },
            |_| zero,
        );

        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == field_rd_value_reduced() => rd_reduced,
                id if id == field_rs1_value_reduced() => rs1_reduced,
                id if id == field_rs2_value_reduced() => rs2_reduced,
                _ => zero,
            },
            |id| match *id {
                FieldInlineChallengeId::FieldRegistersClaimReduction(
                    FieldRegistersClaimReductionChallenge::Gamma,
                ) => gamma,
                _ => zero,
            },
            |id| match *id {
                FieldInlineDerivedId::FieldRegistersClaimReduction(
                    FieldRegistersClaimReductionPublic::EqSpartan,
                ) => eq_spartan,
                _ => zero,
            },
        );

        assert_eq!(
            input,
            rd_spartan + gamma * rs1_spartan + gamma * gamma * rs2_spartan
        );
        assert_eq!(
            output,
            eq_spartan * (rd_reduced + gamma * rs1_reduced + gamma * gamma * rs2_reduced)
        );
    }
}
