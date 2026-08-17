//! field_inline rd-inc claim-reduction symbolic sumcheck relation.

use jolt_field::RingCore;

use crate::protocols::field_inline::geometry::claim_reductions::increments::{
    field_rd_inc_read_write, field_rd_inc_reduced, field_rd_inc_val_evaluation,
};
use serde::{Deserialize, Serialize};

use crate::protocols::field_inline::{
    FieldInlineChallengeId, FieldInlineDerivedId, FieldInlineExpr, FieldInlineOpeningId,
    FieldInlineRelationId, FieldRegistersIncClaimReductionChallenge,
    FieldRegistersIncClaimReductionPublic, FieldRegistersTraceDimensions,
};
use crate::{
    challenge, derived, opening, InputClaims, OutputClaims, SumcheckChallenges, SymbolicSumcheck,
};

/// The single reduced `FieldRdInc` opening handed to the final opening planner.
/// Mirrors `geometry::claim_reductions::increments::claim_reduction_output_openings()`.
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[protocol(field_inline)]
#[relation(FieldRegistersIncClaimReduction)]
pub struct FieldRegistersIncClaimReductionOutputClaims<C> {
    #[opening(committed = FieldRdInc)]
    pub rd_inc: C,
}

/// The two semantic `FieldRdInc` openings consumed by the reduction, wired from
/// the stage-4 FR read/write checking and the stage-5 FR val evaluation.
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
#[protocol(field_inline)]
pub struct FieldRegistersIncClaimReductionInputClaims<C> {
    #[opening(committed = FieldRdInc, from = FieldRegistersReadWriteChecking)]
    pub rd_inc_read_write: C,
    #[opening(committed = FieldRdInc, from = FieldRegistersValEvaluation)]
    pub rd_inc_val_evaluation: C,
}

/// Fiat-Shamir challenge drawn by the FR increment claim-reduction sumcheck
/// (the challenge the protocol spec names `eta`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, SumcheckChallenges)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[protocol(field_inline)]
pub struct FieldRegistersIncClaimReductionChallenges<F> {
    #[challenge(FieldRegistersIncClaimReductionChallenge::Gamma)]
    pub gamma: F,
}

/// Reduces the two `FieldRdInc` openings (read/write and val-evaluation) to a
/// single reduced `FieldRdInc` opening, folding by the drawn challenge and
/// weighting by the `EqReadWrite`/`EqValEvaluation` publics.
#[derive(Clone)]
pub struct ClaimReduction {
    shape: FieldRegistersTraceDimensions,
}

impl SymbolicSumcheck for ClaimReduction {
    type RelationId = FieldInlineRelationId;
    type OpeningId = FieldInlineOpeningId;
    type DerivedId = FieldInlineDerivedId;
    type ChallengeId = FieldInlineChallengeId;
    type Shape = FieldRegistersTraceDimensions;
    type Challenges<F> = FieldRegistersIncClaimReductionChallenges<F>;
    type Inputs<C> = FieldRegistersIncClaimReductionInputClaims<C>;
    type Outputs<C> = FieldRegistersIncClaimReductionOutputClaims<C>;

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
        let gamma = challenge(FieldRegistersIncClaimReductionChallenge::Gamma);

        opening(field_rd_inc_read_write()) + gamma * opening(field_rd_inc_val_evaluation())
    }

    fn output_expression<F: RingCore>(&self) -> FieldInlineExpr<F> {
        let gamma = challenge(FieldRegistersIncClaimReductionChallenge::Gamma);

        let output_coeff = derived(FieldRegistersIncClaimReductionPublic::EqReadWrite)
            + gamma * derived(FieldRegistersIncClaimReductionPublic::EqValEvaluation);
        output_coeff * opening(field_rd_inc_reduced())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::protocols::field_inline::geometry::claim_reductions::increments::{
        claim_reduction_input_openings, claim_reduction_output_openings,
    };
    use jolt_field::{Fr, FromPrimitiveInt};

    fn dimensions() -> FieldRegistersTraceDimensions {
        FieldRegistersTraceDimensions::new(5)
    }

    #[test]
    fn claim_struct_field_order_matches_geometry_opening_order() {
        let value = Fr::from_u64(1);

        let outputs = FieldRegistersIncClaimReductionOutputClaims::<Fr> { rd_inc: value };
        assert_eq!(outputs.canonical_order(), claim_reduction_output_openings());

        let inputs = FieldRegistersIncClaimReductionInputClaims::<Fr> {
            rd_inc_read_write: value,
            rd_inc_val_evaluation: value,
        };
        assert_eq!(inputs.canonical_order(), claim_reduction_input_openings());
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
