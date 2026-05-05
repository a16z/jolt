use super::super::result_count::LoweredResultCount;
use super::notation::{
    SUMCHECK_BATCH_ATTRS, SUMCHECK_CLAIM_ATTRS, SUMCHECK_DRIVER_ATTRS, SUMCHECK_KERNEL_CLAIM_ATTRS,
    SUMCHECK_KERNEL_DRIVER_ATTRS,
};

#[derive(Clone, Copy)]
pub(super) struct SumcheckProofShape {
    pub(super) operand_start: usize,
    pub(super) attrs: &'static [&'static str],
    pub(super) result_count: LoweredResultCount,
}

pub(super) const SUMCHECK_CLAIM_SHAPE: SumcheckProofShape = SumcheckProofShape {
    operand_start: 0,
    attrs: SUMCHECK_CLAIM_ATTRS,
    result_count: LoweredResultCount::One,
};

pub(super) const SUMCHECK_DRIVER_SHAPE: SumcheckProofShape = SumcheckProofShape {
    operand_start: 0,
    attrs: SUMCHECK_DRIVER_ATTRS,
    result_count: LoweredResultCount::Four,
};

pub(super) const SUMCHECK_KERNEL_CLAIM_SHAPE: SumcheckProofShape = SumcheckProofShape {
    operand_start: 0,
    attrs: SUMCHECK_KERNEL_CLAIM_ATTRS,
    result_count: LoweredResultCount::One,
};

pub(super) const SUMCHECK_KERNEL_DRIVER_SHAPE: SumcheckProofShape = SumcheckProofShape {
    operand_start: 0,
    attrs: SUMCHECK_KERNEL_DRIVER_ATTRS,
    result_count: LoweredResultCount::Four,
};

pub(super) fn sumcheck_batch_shape(operand_start: usize) -> SumcheckProofShape {
    SumcheckProofShape {
        operand_start,
        attrs: SUMCHECK_BATCH_ATTRS,
        result_count: LoweredResultCount::One,
    }
}
