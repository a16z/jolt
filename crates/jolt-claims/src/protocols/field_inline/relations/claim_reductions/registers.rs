//! field_inline registers claim-reduction symbolic sumcheck relation.

use crate::protocols::field_inline::geometry::claim_reductions::registers::{
    field_rd_value_reduced, field_rd_value_spartan, field_rs1_value_reduced,
    field_rs1_value_spartan, field_rs2_value_reduced, field_rs2_value_spartan,
};
use serde::{Deserialize, Serialize};

use crate::protocols::field_inline::{
    FieldInlineChallengeId, FieldInlineDerivedId, FieldInlineOpeningId, FieldInlineRelationId,
    FieldRegistersClaimReductionChallenge, FieldRegistersClaimReductionPublic,
    FieldRegistersTraceDimensions,
};
use crate::twist::claim_reductions as twist;
use crate::{InputClaims, OutputClaims, SumcheckChallenges};

/// Produced field-register claim-reduction openings (`FieldRdValue`,
/// `FieldRs1Value`, `FieldRs2Value` reduced to the Spartan point), all sharing
/// the single reduction opening point. Field declaration order is the canonical
/// Fiat-Shamir order and mirrors
/// `geometry::claim_reductions::registers::claim_reduction_output_openings()`.
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[protocol(field_inline)]
#[relation(FieldRegistersClaimReduction)]
pub struct FieldRegistersClaimReductionOutputClaims<C> {
    #[opening(FieldRdValue)]
    pub rd_value: C,
    #[opening(FieldRs1Value)]
    pub rs1_value: C,
    #[opening(FieldRs2Value)]
    pub rs2_value: C,
}

/// Consumed field-register openings reduced by this sumcheck, wired from the
/// field-inline extension of stage 1's outer sumcheck. Generic over the cell.
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
#[protocol(field_inline)]
pub struct FieldRegistersClaimReductionInputClaims<C> {
    #[opening(FieldRdValue, from = FieldRegistersSpartanOuter)]
    pub rd_value: C,
    #[opening(FieldRs1Value, from = FieldRegistersSpartanOuter)]
    pub rs1_value: C,
    #[opening(FieldRs2Value, from = FieldRegistersSpartanOuter)]
    pub rs2_value: C,
}

/// Fiat-Shamir challenge drawn by the FR claim-reduction sumcheck.
#[derive(Clone, Copy, Debug, PartialEq, Eq, SumcheckChallenges)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[protocol(field_inline)]
pub struct FieldRegistersClaimReductionChallenges<F> {
    #[challenge(FieldRegistersClaimReductionChallenge::Gamma)]
    pub gamma: F,
}

/// Batches the native field-register Spartan-outer openings (`FieldRdValue`,
/// `FieldRs1Value`, `FieldRs2Value`) by `gamma` and reduces them to the
/// registers-claim-reduction openings weighted by the `EqSpartan` public.
#[derive(Clone)]
pub struct ClaimReduction {
    shape: FieldRegistersTraceDimensions,
}

twist::instantiate_value_reduction! {
    relation = ClaimReduction,
    id = FieldInlineRelationId::FieldRegistersClaimReduction,
    ids = (FieldInlineRelationId, FieldInlineOpeningId, FieldInlineDerivedId, FieldInlineChallengeId),
    dimensions = FieldRegistersTraceDimensions,
    challenges = FieldRegistersClaimReductionChallenges,
    inputs = FieldRegistersClaimReductionInputClaims,
    outputs = FieldRegistersClaimReductionOutputClaims,
    consumed = [
        field_rd_value_spartan(),
        field_rs1_value_spartan(),
        field_rs2_value_spartan(),
    ],
    reduced = [
        field_rd_value_reduced(),
        field_rs1_value_reduced(),
        field_rs2_value_reduced(),
    ],
    gamma = FieldRegistersClaimReductionChallenge::Gamma,
    eq_spartan = FieldRegistersClaimReductionPublic::EqSpartan,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SymbolicSumcheck;

    use crate::protocols::field_inline::geometry::claim_reductions::registers::{
        claim_reduction_input_openings, claim_reduction_output_openings,
    };
    use jolt_field::{Fr, Ring};

    fn dimensions() -> FieldRegistersTraceDimensions {
        FieldRegistersTraceDimensions::new(5)
    }

    #[test]
    fn claim_struct_field_order_matches_geometry_opening_order() {
        let value = Fr::from_u64(1);

        let outputs = FieldRegistersClaimReductionOutputClaims::<Fr> {
            rd_value: value,
            rs1_value: value,
            rs2_value: value,
        };
        assert_eq!(outputs.canonical_order(), claim_reduction_output_openings());

        let inputs = FieldRegistersClaimReductionInputClaims::<Fr> {
            rd_value: value,
            rs1_value: value,
            rs2_value: value,
        };
        assert_eq!(inputs.canonical_order(), claim_reduction_input_openings());
    }

    #[test]
    fn claim_reduction_exposes_expected_dependencies() {
        let relation = ClaimReduction::new(dimensions());

        assert_eq!(
            ClaimReduction::id(),
            FieldInlineRelationId::FieldRegistersClaimReduction
        );
        assert_eq!(relation.rounds(), dimensions().log_t());
        assert_eq!(relation.degree(), 2);
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
