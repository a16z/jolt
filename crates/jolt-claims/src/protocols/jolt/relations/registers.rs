//! registers symbolic sumcheck relations.

use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::registers::{
    rd_inc_read_write, rd_inc_val_evaluation, rd_wa_read_write, rd_wa_val_evaluation,
    rd_write_value_claim, registers_val_read_write, rs1_ra_read_write, rs1_value_claim,
    rs2_ra_read_write, rs2_value_claim,
};
use crate::protocols::jolt::{
    JoltExpr, JoltRelationId, JoltSumcheckSpec, ReadWriteDimensions, RegistersReadWriteChallenge,
    RegistersReadWritePublic, RegistersValEvaluationPublic, TraceDimensions,
};
use crate::SymbolicSumcheck;
use crate::{challenge, opening, derived, InputClaims, OutputClaims};

/// Produced register read-write openings, all sharing the single read-write
/// opening point. Generic over the cell (`F` on the wire, `Vec<F>` for ZK points,
/// `OpeningClaim<F>` on the clear path).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(RegistersReadWriteChecking)]
pub struct RegistersReadWriteOutputClaims<C> {
    #[opening(RegistersVal)]
    pub registers_val: C,
    #[opening(Rs1Ra)]
    pub rs1_ra: C,
    #[opening(Rs2Ra)]
    pub rs2_ra: C,
    #[opening(RdWa)]
    pub rd_wa: C,
    #[opening(committed = RdInc)]
    pub rd_inc: C,
}

/// Consumed register openings reduced by the read-write checking sumcheck, wired
/// from the upstream registers claim-reduction relation (stage 3). Generic over
/// the cell.
#[derive(Clone, Debug, InputClaims)]
pub struct RegistersReadWriteInputClaims<C> {
    #[opening(RdWriteValue, from = RegistersClaimReduction)]
    pub rd_write_value: C,
    #[opening(Rs1Value, from = RegistersClaimReduction)]
    pub rs1_value: C,
    #[opening(Rs2Value, from = RegistersClaimReduction)]
    pub rs2_value: C,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(RegistersValEvaluation)]
pub struct RegistersValEvaluationOutputClaims<C> {
    #[opening(committed = RdInc)]
    pub rd_inc: C,
    #[opening(RdWa)]
    pub rd_wa: C,
}

/// Consumed register value-evaluation opening, wired from the upstream register
/// read-write checking.
#[derive(Clone, Debug, InputClaims)]
pub struct RegistersValEvaluationInputClaims<C> {
    #[opening(RegistersVal, from = RegistersReadWriteChecking)]
    pub registers_val: C,
}

/// The registers read/write checking sumcheck: relates the read-value claims
/// (`RdWriteValue`, `Rs1Value`, `Rs2Value`) folded by `gamma` to the register
/// `val`/`ra`/`inc` openings weighted by the `EqCycle` public.
pub struct ReadWriteChecking {
    shape: ReadWriteDimensions,
}

impl SymbolicSumcheck for ReadWriteChecking {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = ReadWriteDimensions;

    fn new(shape: ReadWriteDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::RegistersReadWriteChecking
    }

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.read_write_sumcheck()
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(RegistersReadWriteChallenge::Gamma);
        opening(rd_write_value_claim())
            + gamma.clone() * opening(rs1_value_claim())
            + gamma.clone().pow(2) * opening(rs2_value_claim())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(RegistersReadWriteChallenge::Gamma);
        let eq_cycle = derived(RegistersReadWritePublic::EqCycle);
        eq_cycle.clone() * opening(rd_wa_read_write()) * opening(rd_inc_read_write())
            + eq_cycle.clone() * opening(rd_wa_read_write()) * opening(registers_val_read_write())
            + eq_cycle.clone()
                * gamma.clone()
                * opening(rs1_ra_read_write())
                * opening(registers_val_read_write())
            + eq_cycle
                * gamma.pow(2)
                * opening(rs2_ra_read_write())
                * opening(registers_val_read_write())
    }
}

/// The registers val-evaluation sumcheck: relates the register `val` opening to
/// `rd_inc * rd_wa` weighted by the `LtCycle` public.
pub struct ValEvaluation {
    shape: TraceDimensions,
}

impl SymbolicSumcheck for ValEvaluation {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = TraceDimensions;

    fn new(shape: TraceDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::RegistersValEvaluation
    }

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.sumcheck(3)
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(registers_val_read_write())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        derived(RegistersValEvaluationPublic::LtCycle)
            * opening(rd_inc_val_evaluation())
            * opening(rd_wa_val_evaluation())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::{JoltChallengeId, JoltDerivedId};
    use jolt_field::{Fr, FromPrimitiveInt};

    fn read_write_dimensions() -> ReadWriteDimensions {
        ReadWriteDimensions::new(5, 7, 2, 1)
    }

    fn trace_dimensions() -> TraceDimensions {
        TraceDimensions::new(5)
    }

    #[test]
    fn read_write_claims_evaluate_like_core_formula() {
        let relation = ReadWriteChecking::new(read_write_dimensions());

        let rd_write_value = Fr::from_u64(3);
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
                id if id == rd_write_value_claim() => rd_write_value,
                id if id == rs1_value_claim() => rs1_value,
                id if id == rs2_value_claim() => rs2_value,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::RegistersReadWrite(RegistersReadWriteChallenge::Gamma) => gamma,
                JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaClaimReduction(_)
                | JoltChallengeId::RegistersClaimReduction(_)
                | JoltChallengeId::InstructionClaimReduction(_)
                | JoltChallengeId::InstructionInput(_)
                | JoltChallengeId::InstructionReadRaf(_)
                | JoltChallengeId::InstructionRaVirtualization(_)
                | JoltChallengeId::Booleanity(_)
                | JoltChallengeId::IncClaimReduction(_)
                | JoltChallengeId::HammingWeightClaimReduction(_)
                | JoltChallengeId::BytecodeReadRaf(_)
                | JoltChallengeId::BytecodeClaimReduction(_)
                | JoltChallengeId::SpartanShift(_) => zero,
            },
            |_| zero,
        );

        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == registers_val_read_write() => val,
                id if id == rs1_ra_read_write() => rs1_ra,
                id if id == rs2_ra_read_write() => rs2_ra,
                id if id == rd_wa_read_write() => rd_wa,
                id if id == rd_inc_read_write() => inc,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::RegistersReadWrite(RegistersReadWriteChallenge::Gamma) => gamma,
                JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaClaimReduction(_)
                | JoltChallengeId::RegistersClaimReduction(_)
                | JoltChallengeId::InstructionClaimReduction(_)
                | JoltChallengeId::InstructionInput(_)
                | JoltChallengeId::InstructionReadRaf(_)
                | JoltChallengeId::InstructionRaVirtualization(_)
                | JoltChallengeId::Booleanity(_)
                | JoltChallengeId::IncClaimReduction(_)
                | JoltChallengeId::HammingWeightClaimReduction(_)
                | JoltChallengeId::BytecodeReadRaf(_)
                | JoltChallengeId::BytecodeClaimReduction(_)
                | JoltChallengeId::SpartanShift(_) => zero,
            },
            |id| match *id {
                JoltDerivedId::RegistersReadWrite(RegistersReadWritePublic::EqCycle) => eq_cycle,
                _ => zero,
            },
        );

        assert_eq!(
            input,
            rd_write_value + gamma * rs1_value + gamma * gamma * rs2_value
        );
        assert_eq!(
            output,
            eq_cycle * (rd_wa * (inc + val) + gamma * rs1_ra * val + gamma * gamma * rs2_ra * val)
        );
    }

    #[test]
    fn val_evaluation_claims_evaluate_like_core_formula() {
        let relation = ValEvaluation::new(trace_dimensions());

        let val = Fr::from_u64(3);
        let inc = Fr::from_u64(5);
        let wa = Fr::from_u64(7);
        let lt_cycle = Fr::from_u64(11);
        let zero = Fr::from_u64(0);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == registers_val_read_write() => val,
                _ => zero,
            },
            |_| zero,
            |_| zero,
        );

        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == rd_inc_val_evaluation() => inc,
                id if id == rd_wa_val_evaluation() => wa,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaClaimReduction(_)
                | JoltChallengeId::RegistersReadWrite(_)
                | JoltChallengeId::RegistersClaimReduction(_)
                | JoltChallengeId::InstructionClaimReduction(_)
                | JoltChallengeId::InstructionInput(_)
                | JoltChallengeId::InstructionReadRaf(_)
                | JoltChallengeId::InstructionRaVirtualization(_)
                | JoltChallengeId::Booleanity(_)
                | JoltChallengeId::IncClaimReduction(_)
                | JoltChallengeId::HammingWeightClaimReduction(_)
                | JoltChallengeId::BytecodeReadRaf(_)
                | JoltChallengeId::BytecodeClaimReduction(_)
                | JoltChallengeId::SpartanShift(_) => zero,
            },
            |id| match *id {
                JoltDerivedId::RegistersValEvaluation(RegistersValEvaluationPublic::LtCycle) => {
                    lt_cycle
                }
                _ => zero,
            },
        );

        assert_eq!(input, val);
        assert_eq!(output, lt_cycle * inc * wa);
    }

    #[test]
    fn read_write_checking_symbolic_matches_dependencies() {
        let relation = ReadWriteChecking::new(read_write_dimensions());
        assert_eq!(
            ReadWriteChecking::id(),
            JoltRelationId::RegistersReadWriteChecking
        );
        assert_eq!(
            relation.spec(),
            read_write_dimensions().read_write_sumcheck()
        );
        assert_eq!(
            relation.required_openings::<Fr>(),
            vec![
                rd_write_value_claim(),
                rs1_value_claim(),
                rs2_value_claim(),
                rd_wa_read_write(),
                rd_inc_read_write(),
                registers_val_read_write(),
                rs1_ra_read_write(),
                rs2_ra_read_write(),
            ]
        );
        assert_eq!(
            relation.required_challenges::<Fr>(),
            vec![JoltChallengeId::from(RegistersReadWriteChallenge::Gamma)]
        );
        assert_eq!(
            relation.required_deriveds::<Fr>(),
            vec![JoltDerivedId::from(RegistersReadWritePublic::EqCycle)]
        );
    }

    #[test]
    fn val_evaluation_symbolic_matches_dependencies() {
        let relation = ValEvaluation::new(trace_dimensions());
        assert_eq!(ValEvaluation::id(), JoltRelationId::RegistersValEvaluation);
        assert_eq!(relation.spec(), trace_dimensions().sumcheck(3));
        assert_eq!(
            relation.required_openings::<Fr>(),
            vec![
                registers_val_read_write(),
                rd_inc_val_evaluation(),
                rd_wa_val_evaluation(),
            ]
        );
        assert!(relation.required_challenges::<Fr>().is_empty());
        assert_eq!(
            relation.required_deriveds::<Fr>(),
            vec![JoltDerivedId::from(RegistersValEvaluationPublic::LtCycle)]
        );
    }
}
