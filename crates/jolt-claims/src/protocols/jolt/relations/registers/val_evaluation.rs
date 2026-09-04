//! registers val-evaluation symbolic sumcheck relation.

use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::registers::{
    rd_inc_val_evaluation, rd_wa_val_evaluation, registers_val_read_write,
};
use crate::protocols::jolt::{
    JoltChallengeId, JoltDerivedId, JoltOpeningId, JoltRelationId, RegistersValEvaluationPublic,
    TraceDimensions,
};
use crate::twist::memory_checking as twist;
use crate::{InputClaims, NoChallenges, OutputClaims};

#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
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
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct RegistersValEvaluationInputClaims<C> {
    #[opening(RegistersVal, from = RegistersReadWriteChecking)]
    pub registers_val: C,
}

/// The registers val-evaluation sumcheck: relates the register `val` opening to
/// `rd_inc * rd_wa` weighted by the `LtCycle` public.
#[derive(Clone)]
pub struct ValEvaluation {
    shape: TraceDimensions,
}

twist::instantiate_val_evaluation! {
    relation = ValEvaluation,
    id = JoltRelationId::RegistersValEvaluation,
    ids = (JoltRelationId, JoltOpeningId, JoltDerivedId, JoltChallengeId),
    dimensions = TraceDimensions,
    challenges = NoChallenges,
    inputs = RegistersValEvaluationInputClaims,
    outputs = RegistersValEvaluationOutputClaims,
    registers_val = registers_val_read_write(),
    rd_inc = rd_inc_val_evaluation(),
    rd_wa = rd_wa_val_evaluation(),
    lt_cycle = RegistersValEvaluationPublic::LtCycle,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::JoltDerivedId;
    use crate::SymbolicSumcheck;
    use jolt_field::{Fr, Ring};

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
            |_| zero,
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
    }
}
