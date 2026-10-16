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
