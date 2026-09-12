//! field_inline registers symbolic sumcheck relations.

use serde::{Deserialize, Serialize};

use crate::protocols::field_inline::geometry::registers::{
    field_rd_inc_read_write, field_rd_inc_val_evaluation, field_rd_value_claim,
    field_rd_wa_read_write, field_rd_wa_val_evaluation, field_registers_val_read_write,
    field_rs1_ra_read_write, field_rs1_value_claim, field_rs2_ra_read_write, field_rs2_value_claim,
};
use crate::protocols::field_inline::{
    FieldInlineChallengeId, FieldInlineDerivedId, FieldInlineOpeningId, FieldInlineRelationId,
    FieldRegistersReadWriteChallenge, FieldRegistersReadWriteDimensions,
    FieldRegistersReadWritePublic, FieldRegistersTraceDimensions,
    FieldRegistersValEvaluationPublic,
};
use crate::twist::memory_checking as twist;
use crate::{InputClaims, NoChallenges, OutputClaims, SumcheckChallenges};

/// Produced field-register read-write openings, all sharing the single FR
/// read-write opening point. Generic over the opening cell (`F` for the
/// serialized wire value, `Vec<F>` for the derived opening point). Field
/// declaration order is the canonical Fiat-Shamir order and mirrors
/// `geometry::registers::read_write_checking_output_openings()`.
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[protocol(field_inline)]
#[relation(FieldRegistersReadWriteChecking)]
pub struct FieldRegistersReadWriteOutputClaims<C> {
    #[opening(FieldRegistersVal)]
    pub registers_val: C,
    #[opening(FieldRs1Ra)]
    pub rs1_ra: C,
    #[opening(FieldRs2Ra)]
    pub rs2_ra: C,
    #[opening(FieldRdWa)]
    pub rd_wa: C,
    #[opening(committed = FieldRdInc)]
    pub rd_inc: C,
}

/// Consumed field-register openings reduced by the FR read-write checking
/// sumcheck, wired from the upstream field-register claim reduction (stage 2).
/// Generic over the cell.
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
#[protocol(field_inline)]
pub struct FieldRegistersReadWriteInputClaims<C> {
    #[opening(FieldRdValue, from = FieldRegistersClaimReduction)]
    pub rd_value: C,
    #[opening(FieldRs1Value, from = FieldRegistersClaimReduction)]
    pub rs1_value: C,
    #[opening(FieldRs2Value, from = FieldRegistersClaimReduction)]
    pub rs2_value: C,
}

/// Fiat-Shamir challenge drawn by the FR read/write-checking sumcheck.
#[derive(Clone, Copy, Debug, PartialEq, Eq, SumcheckChallenges)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[protocol(field_inline)]
pub struct FieldRegistersReadWriteChallenges<F> {
    #[challenge(FieldRegistersReadWriteChallenge::Gamma)]
    pub gamma: F,
}

/// The native field-register read/write checking sumcheck: relates the read-value
/// claims (`FieldRdValue`, `FieldRs1Value`, `FieldRs2Value`) folded by `gamma` to
/// the register `val`/`ra`/`inc` openings weighted by the `EqCycle` public.
#[derive(Clone)]
pub struct ReadWriteChecking {
    shape: FieldRegistersReadWriteDimensions,
}

twist::instantiate_read_write_checking! {
    relation = ReadWriteChecking,
    id = FieldInlineRelationId::FieldRegistersReadWriteChecking,
    ids = (FieldInlineRelationId, FieldInlineOpeningId, FieldInlineDerivedId, FieldInlineChallengeId),
    dimensions = FieldRegistersReadWriteDimensions,
    challenges = FieldRegistersReadWriteChallenges,
    inputs = FieldRegistersReadWriteInputClaims,
    outputs = FieldRegistersReadWriteOutputClaims,
    rd_value = field_rd_value_claim(),
    rs1_value = field_rs1_value_claim(),
    rs2_value = field_rs2_value_claim(),
    registers_val = field_registers_val_read_write(),
    rs1_ra = field_rs1_ra_read_write(),
    rs2_ra = field_rs2_ra_read_write(),
    rd_wa = field_rd_wa_read_write(),
    rd_inc = field_rd_inc_read_write(),
    gamma = FieldRegistersReadWriteChallenge::Gamma,
    eq_cycle = FieldRegistersReadWritePublic::EqCycle,
}

/// Produced field-register val-evaluation openings. Field declaration order is
/// the canonical Fiat-Shamir order and mirrors
/// `geometry::registers::val_evaluation_output_openings()`.
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[protocol(field_inline)]
#[relation(FieldRegistersValEvaluation)]
pub struct FieldRegistersValEvaluationOutputClaims<C> {
    #[opening(committed = FieldRdInc)]
    pub rd_inc: C,
    #[opening(FieldRdWa)]
    pub rd_wa: C,
}

/// Consumed field-register value-evaluation opening, wired from the upstream FR
/// read-write checking.
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
#[protocol(field_inline)]
pub struct FieldRegistersValEvaluationInputClaims<C> {
    #[opening(FieldRegistersVal, from = FieldRegistersReadWriteChecking)]
    pub registers_val: C,
}

/// The native field-register val-evaluation sumcheck: relates the register `val`
/// opening to `rd_inc * rd_wa` weighted by the `LtCycle` public.
#[derive(Clone)]
pub struct ValEvaluation {
    shape: FieldRegistersTraceDimensions,
}

twist::instantiate_val_evaluation! {
    relation = ValEvaluation,
    id = FieldInlineRelationId::FieldRegistersValEvaluation,
    ids = (FieldInlineRelationId, FieldInlineOpeningId, FieldInlineDerivedId, FieldInlineChallengeId),
    dimensions = FieldRegistersTraceDimensions,
    challenges = NoChallenges,
    inputs = FieldRegistersValEvaluationInputClaims,
    outputs = FieldRegistersValEvaluationOutputClaims,
    registers_val = field_registers_val_read_write(),
    rd_inc = field_rd_inc_val_evaluation(),
    rd_wa = field_rd_wa_val_evaluation(),
    lt_cycle = FieldRegistersValEvaluationPublic::LtCycle,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SymbolicSumcheck;

    use crate::protocols::field_inline::geometry::registers::{
        read_write_checking_input_openings, read_write_checking_output_openings,
        val_evaluation_input_openings, val_evaluation_output_openings,
    };
    use jolt_field::{Fr, Ring};

    fn trace_dimensions() -> FieldRegistersTraceDimensions {
        FieldRegistersTraceDimensions::new(5)
    }

    #[test]
    fn claim_struct_field_order_matches_geometry_opening_order() {
        let value = Fr::from_u64(1);

        let outputs = FieldRegistersReadWriteOutputClaims::<Fr> {
            registers_val: value,
            rs1_ra: value,
            rs2_ra: value,
            rd_wa: value,
            rd_inc: value,
        };
        assert_eq!(
            outputs.canonical_order(),
            read_write_checking_output_openings()
        );

        let inputs = FieldRegistersReadWriteInputClaims::<Fr> {
            rd_value: value,
            rs1_value: value,
            rs2_value: value,
        };
        assert_eq!(
            inputs.canonical_order(),
            read_write_checking_input_openings()
        );

        let outputs = FieldRegistersValEvaluationOutputClaims::<Fr> {
            rd_inc: value,
            rd_wa: value,
        };
        assert_eq!(outputs.canonical_order(), val_evaluation_output_openings());

        let inputs = FieldRegistersValEvaluationInputClaims::<Fr> {
            registers_val: value,
        };
        assert_eq!(inputs.canonical_order(), val_evaluation_input_openings());
    }

    #[test]
    fn read_write_challenges_resolve_by_field_inline_id() {
        let gamma = Fr::from_u64(29);
        let challenges = FieldRegistersReadWriteChallenges { gamma };
        assert_eq!(
            challenges.resolve_challenge(&FieldInlineChallengeId::FieldRegistersReadWrite(
                FieldRegistersReadWriteChallenge::Gamma,
            )),
            Some(gamma)
        );
    }

    fn read_write_dimensions() -> FieldRegistersReadWriteDimensions {
        FieldRegistersReadWriteDimensions::new(5, 4, 2, 1)
    }

    #[test]
    fn read_write_claims_expose_expected_dependencies() {
        let relation = ReadWriteChecking::new(read_write_dimensions());

        assert_eq!(
            ReadWriteChecking::id(),
            FieldInlineRelationId::FieldRegistersReadWriteChecking
        );
        assert_eq!(
            relation.rounds(),
            read_write_dimensions().read_write_rounds()
        );
        assert_eq!(relation.degree(), 3);
    }

    #[test]
    fn read_write_claims_evaluate_like_field_register_twist_formula() {
        let relation = ReadWriteChecking::new(read_write_dimensions());

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

        let input = relation.input_expression::<Fr>().evaluate(
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

        let output = relation.output_expression::<Fr>().evaluate(
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
                FieldInlineDerivedId::FieldRegistersReadWrite(
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
        let relation = ValEvaluation::new(trace_dimensions());

        assert_eq!(
            ValEvaluation::id(),
            FieldInlineRelationId::FieldRegistersValEvaluation
        );
        assert_eq!(relation.rounds(), trace_dimensions().log_t());
        assert_eq!(relation.degree(), 3);
    }

    #[test]
    fn val_evaluation_claims_evaluate_like_field_register_twist_formula() {
        let relation = ValEvaluation::new(trace_dimensions());

        let val = Fr::from_u64(3);
        let inc = Fr::from_u64(5);
        let wa = Fr::from_u64(7);
        let lt_cycle = Fr::from_u64(11);
        let zero = Fr::from_u64(0);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == field_registers_val_read_write() => val,
                _ => zero,
            },
            |_| zero,
            |_| zero,
        );

        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == field_rd_inc_val_evaluation() => inc,
                id if id == field_rd_wa_val_evaluation() => wa,
                _ => zero,
            },
            |_| zero,
            |id| match *id {
                FieldInlineDerivedId::FieldRegistersValEvaluation(
                    FieldRegistersValEvaluationPublic::LtCycle,
                ) => lt_cycle,
                _ => zero,
            },
        );

        assert_eq!(input, val);
        assert_eq!(output, lt_cycle * inc * wa);
    }
}
