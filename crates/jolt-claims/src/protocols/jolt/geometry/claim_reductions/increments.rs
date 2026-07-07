use jolt_field::RingCore;

use super::super::super::{JoltCommittedPolynomial, JoltExpr, JoltOpeningId, JoltRelationId};
use crate::opening;

/// The γ-batched consumption of the four inc consumer claims. Shared by the
/// base `IncClaimReduction` and the lattice `IncVirtualization` relations,
/// which must consume exactly the same set with the same γ order.
pub(crate) fn inc_consumers_input<F>(gamma: JoltExpr<F>) -> JoltExpr<F>
where
    F: RingCore,
{
    opening(ram_inc_read_write())
        + gamma.clone() * opening(ram_inc_val_check())
        + gamma.clone().pow(2) * opening(rd_inc_read_write())
        + gamma.pow(3) * opening(rd_inc_val_evaluation())
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
