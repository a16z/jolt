use super::super::super::{JoltCommittedPolynomial, JoltOpeningId, JoltRelationId};

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

pub fn ram_inc_reduced() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RamInc,
        JoltRelationId::IncClaimReduction,
    )
}

pub fn rd_inc_reduced() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RdInc,
        JoltRelationId::IncClaimReduction,
    )
}
