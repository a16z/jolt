use jolt_field::RingCore;

use crate::{challenge, public};

use super::super::super::{
    IncClaimReductionChallenge, IncClaimReductionPublic, JoltChallengeId, JoltCommittedPolynomial,
    JoltExpr, JoltOpeningId, JoltPublicId, JoltRelationId,
};

pub(crate) fn inc_challenge<F>(id: IncClaimReductionChallenge) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

pub(crate) fn inc_public<F>(id: IncClaimReductionPublic) -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(id))
}

pub fn claim_reduction_input_openings() -> [JoltOpeningId; 4] {
    [
        ram_inc_read_write(),
        ram_inc_val_check(),
        rd_inc_read_write(),
        rd_inc_val_evaluation(),
    ]
}

pub fn claim_reduction_output_openings() -> [JoltOpeningId; 2] {
    [ram_inc_reduced(), rd_inc_reduced()]
}

pub(crate) fn ram_inc_read_write() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RamInc,
        JoltRelationId::RamReadWriteChecking,
    )
}

pub(crate) fn ram_inc_val_check() -> JoltOpeningId {
    JoltOpeningId::committed(JoltCommittedPolynomial::RamInc, JoltRelationId::RamValCheck)
}

pub(crate) fn rd_inc_read_write() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RdInc,
        JoltRelationId::RegistersReadWriteChecking,
    )
}

pub(crate) fn rd_inc_val_evaluation() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RdInc,
        JoltRelationId::RegistersValEvaluation,
    )
}

pub(crate) fn ram_inc_reduced() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RamInc,
        JoltRelationId::IncClaimReduction,
    )
}

pub(crate) fn rd_inc_reduced() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RdInc,
        JoltRelationId::IncClaimReduction,
    )
}
