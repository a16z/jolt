use jolt_field::RingCore;

use crate::{challenge, public};

use super::super::{
    JoltChallengeId, JoltCommittedPolynomial, JoltExpr, JoltOpeningId, JoltPublicId,
    JoltRelationId, JoltVirtualPolynomial, RegistersReadWriteChallenge, RegistersReadWritePublic,
    RegistersValEvaluationPublic,
};
use super::dimensions::{JoltSumcheckSpec, ReadWriteDimensions};

pub const fn read_write_checking_sumcheck(dimensions: ReadWriteDimensions) -> JoltSumcheckSpec {
    dimensions.read_write_sumcheck()
}

pub fn read_write_checking_input_openings() -> [JoltOpeningId; 3] {
    [rd_write_value_claim(), rs1_value_claim(), rs2_value_claim()]
}

pub fn read_write_checking_output_openings() -> [JoltOpeningId; 5] {
    [
        registers_val_read_write(),
        rs1_ra_read_write(),
        rs2_ra_read_write(),
        rd_wa_read_write(),
        rd_inc_read_write(),
    ]
}

pub fn val_evaluation_input_openings() -> [JoltOpeningId; 1] {
    [registers_val_read_write()]
}

pub fn val_evaluation_output_openings() -> [JoltOpeningId; 2] {
    [rd_inc_val_evaluation(), rd_wa_val_evaluation()]
}

pub(crate) fn read_write_challenge<F>(id: RegistersReadWriteChallenge) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

pub(crate) fn read_write_public<F>(id: RegistersReadWritePublic) -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(id))
}

pub(crate) fn val_evaluation_public<F>(id: RegistersValEvaluationPublic) -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(id))
}

pub(crate) fn rd_write_value_claim() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RdWriteValue,
        JoltRelationId::RegistersClaimReduction,
    )
}

pub(crate) fn rs1_value_claim() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Rs1Value,
        JoltRelationId::RegistersClaimReduction,
    )
}

pub(crate) fn rs2_value_claim() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Rs2Value,
        JoltRelationId::RegistersClaimReduction,
    )
}

pub(crate) fn registers_val_read_write() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RegistersVal,
        JoltRelationId::RegistersReadWriteChecking,
    )
}

pub(crate) fn rs1_ra_read_write() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Rs1Ra,
        JoltRelationId::RegistersReadWriteChecking,
    )
}

pub(crate) fn rs2_ra_read_write() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Rs2Ra,
        JoltRelationId::RegistersReadWriteChecking,
    )
}

pub(crate) fn rd_wa_read_write() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RdWa,
        JoltRelationId::RegistersReadWriteChecking,
    )
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

pub(crate) fn rd_wa_val_evaluation() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RdWa,
        JoltRelationId::RegistersValEvaluation,
    )
}
