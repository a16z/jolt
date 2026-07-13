use jolt_field::RingCore;

use super::super::super::{JoltCommittedPolynomial, JoltExpr, JoltOpeningId, JoltRelationId};
use super::super::ram::{ram_inc, ram_inc_val_check};
use super::super::registers::{rd_inc_read_write, rd_inc_val_evaluation};
use crate::opening;

/// The γ-batched consumption of the four inc consumer claims. Shared by the
/// base `IncClaimReduction` and the lattice `IncVirtualization` relations,
/// which must consume exactly the same set with the same γ order.
pub(crate) fn inc_consumers_input<F>(gamma: JoltExpr<F>) -> JoltExpr<F>
where
    F: RingCore,
{
    opening(ram_inc())
        + gamma.clone() * opening(ram_inc_val_check())
        + gamma.clone().pow(2) * opening(rd_inc_read_write())
        + gamma.pow(3) * opening(rd_inc_val_evaluation())
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
