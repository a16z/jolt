//! registers val-evaluation symbolic sumcheck relation.

use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::registers::{
    rd_inc_val_evaluation, rd_wa_val_evaluation, registers_val_read_write,
};
use crate::protocols::jolt::{
    JoltExpr, JoltRelationId, RegistersValEvaluationPublic, TraceDimensions,
};
use crate::SymbolicSumcheck;
use crate::{derived, opening, InputClaims, OutputClaims};

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
#[derive(Clone, Debug, PartialEq, Eq, InputClaims)]
pub struct RegistersValEvaluationInputClaims<C> {
    #[opening(RegistersVal, from = RegistersReadWriteChecking)]
    pub registers_val: C,
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
    type Challenges<F> = crate::NoChallenges<F>;
    type Inputs<C> = RegistersValEvaluationInputClaims<C>;
    type Outputs<C> = RegistersValEvaluationOutputClaims<C>;

    fn new(shape: TraceDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::RegistersValEvaluation
    }

    fn rounds(&self) -> usize {
        self.shape.log_t()
    }

    fn degree(&self) -> usize {
        3
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

    fn trace_dimensions() -> TraceDimensions {
        TraceDimensions::new(5)
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
    fn val_evaluation_symbolic_matches_dependencies() {
        let relation = ValEvaluation::new(trace_dimensions());
        assert_eq!(ValEvaluation::id(), JoltRelationId::RegistersValEvaluation);
        assert_eq!(relation.rounds(), trace_dimensions().log_t());
        assert_eq!(relation.degree(), 3);
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
