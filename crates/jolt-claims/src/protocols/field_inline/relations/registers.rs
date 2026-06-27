//! field_inline registers symbolic sumcheck relations.

use jolt_field::RingCore;

use crate::protocols::field_inline::geometry::registers::{
    field_rd_inc_read_write, field_rd_inc_val_evaluation, field_rd_value_claim,
    field_rd_wa_read_write, field_rd_wa_val_evaluation, field_registers_val_read_write,
    field_rs1_ra_read_write, field_rs1_value_claim, field_rs2_ra_read_write, field_rs2_value_claim,
};
use crate::protocols::field_inline::{
    FieldInlineChallengeId, FieldInlineDerivedId, FieldInlineExpr, FieldInlineOpeningId,
    FieldInlineRelationId, FieldInlineSumcheckSpec, FieldRegistersReadWriteChallenge,
    FieldRegistersReadWriteDimensions, FieldRegistersReadWritePublic,
    FieldRegistersTraceDimensions, FieldRegistersValEvaluationPublic,
};
use crate::SymbolicSumcheck;
use crate::{challenge, derived, opening};

/// The native field-register read/write checking sumcheck: relates the read-value
/// claims (`FieldRdValue`, `FieldRs1Value`, `FieldRs2Value`) folded by `gamma` to
/// the register `val`/`ra`/`inc` openings weighted by the `EqCycle` public.
pub struct ReadWriteChecking {
    shape: FieldRegistersReadWriteDimensions,
}

impl SymbolicSumcheck for ReadWriteChecking {
    type RelationId = FieldInlineRelationId;
    type OpeningId = FieldInlineOpeningId;
    type DerivedId = FieldInlineDerivedId;
    type ChallengeId = FieldInlineChallengeId;
    type Shape = FieldRegistersReadWriteDimensions;
    type Challenges<F> = crate::NoChallenges<F>;
    type Inputs<C> = crate::NoInputs<C>;
    type Outputs<C> = crate::NoOutputs<C>;

    fn new(shape: FieldRegistersReadWriteDimensions) -> Self {
        Self { shape }
    }

    fn id() -> FieldInlineRelationId {
        FieldInlineRelationId::FieldRegistersReadWriteChecking
    }

    fn spec(&self) -> FieldInlineSumcheckSpec {
        self.shape.read_write_sumcheck()
    }

    fn input_expression<F: RingCore>(&self) -> FieldInlineExpr<F> {
        let gamma = challenge(FieldRegistersReadWriteChallenge::Gamma);
        opening(field_rd_value_claim())
            + gamma.clone() * opening(field_rs1_value_claim())
            + gamma.clone().pow(2) * opening(field_rs2_value_claim())
    }

    fn output_expression<F: RingCore>(&self) -> FieldInlineExpr<F> {
        let gamma = challenge(FieldRegistersReadWriteChallenge::Gamma);
        let eq_cycle = derived(FieldRegistersReadWritePublic::EqCycle);
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
                * opening(field_registers_val_read_write())
    }
}

/// The native field-register val-evaluation sumcheck: relates the register `val`
/// opening to `rd_inc * rd_wa` weighted by the `LtCycle` public.
pub struct ValEvaluation {
    shape: FieldRegistersTraceDimensions,
}

impl SymbolicSumcheck for ValEvaluation {
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
        FieldInlineRelationId::FieldRegistersValEvaluation
    }

    fn spec(&self) -> FieldInlineSumcheckSpec {
        self.shape.sumcheck(3)
    }

    fn input_expression<F: RingCore>(&self) -> FieldInlineExpr<F> {
        opening(field_registers_val_read_write())
    }

    fn output_expression<F: RingCore>(&self) -> FieldInlineExpr<F> {
        derived(FieldRegistersValEvaluationPublic::LtCycle)
            * opening(field_rd_inc_val_evaluation())
            * opening(field_rd_wa_val_evaluation())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::field_inline::geometry::registers::{
        read_write_checking_input_openings, val_evaluation_input_openings,
        val_evaluation_output_openings,
    };
    use jolt_field::{Fr, FromPrimitiveInt};

    fn trace_dimensions() -> FieldRegistersTraceDimensions {
        FieldRegistersTraceDimensions::new(5)
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
            relation.spec(),
            read_write_dimensions().read_write_sumcheck()
        );
        assert_eq!(
            relation.input_expression::<Fr>().required_openings(),
            read_write_checking_input_openings().to_vec()
        );
        assert_eq!(
            relation.output_expression::<Fr>().required_openings(),
            vec![
                field_rd_wa_read_write(),
                field_rd_inc_read_write(),
                field_registers_val_read_write(),
                field_rs1_ra_read_write(),
                field_rs2_ra_read_write(),
            ]
        );
        assert_eq!(
            relation.required_challenges::<Fr>(),
            vec![FieldInlineChallengeId::from(
                FieldRegistersReadWriteChallenge::Gamma
            )]
        );
        assert_eq!(relation.required_challenges::<Fr>().len(), 1);
        assert_eq!(
            relation.required_deriveds::<Fr>(),
            vec![FieldInlineDerivedId::from(
                FieldRegistersReadWritePublic::EqCycle
            )]
        );
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
        assert_eq!(relation.spec(), trace_dimensions().sumcheck(3));
        assert_eq!(
            relation.input_expression::<Fr>().required_openings(),
            val_evaluation_input_openings().to_vec()
        );
        assert_eq!(
            relation.output_expression::<Fr>().required_openings(),
            val_evaluation_output_openings().to_vec()
        );
        assert!(relation.required_challenges::<Fr>().is_empty());
        assert_eq!(
            relation.required_deriveds::<Fr>(),
            vec![FieldInlineDerivedId::from(
                FieldRegistersValEvaluationPublic::LtCycle
            )]
        );
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
