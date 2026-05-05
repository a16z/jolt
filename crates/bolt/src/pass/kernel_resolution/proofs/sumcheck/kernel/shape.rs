use crate::pass::support::{
    LoweredResultCount, COMPUTE_SUMCHECK_CLAIM_RESULT_TYPES, COMPUTE_SUMCHECK_DRIVER_RESULT_TYPES,
    SUMCHECK_KERNEL_CLAIM_SOURCE_ATTRS, SUMCHECK_KERNEL_DRIVER_SOURCE_ATTRS,
};

#[derive(Clone, Copy)]
pub(super) struct KernelSumcheckShape {
    pub(super) target_op: &'static str,
    pub(super) source_attrs: &'static [&'static str],
    pub(super) result_types: &'static [&'static str],
    pub(super) result_count: LoweredResultCount,
}

pub(super) const KERNEL_CLAIM_SHAPE: KernelSumcheckShape = KernelSumcheckShape {
    target_op: "compute.sumcheck_kernel_claim",
    source_attrs: SUMCHECK_KERNEL_CLAIM_SOURCE_ATTRS,
    result_types: COMPUTE_SUMCHECK_CLAIM_RESULT_TYPES,
    result_count: LoweredResultCount::One,
};

pub(super) const KERNEL_DRIVER_SHAPE: KernelSumcheckShape = KernelSumcheckShape {
    target_op: "compute.sumcheck_kernel_driver",
    source_attrs: SUMCHECK_KERNEL_DRIVER_SOURCE_ATTRS,
    result_types: COMPUTE_SUMCHECK_DRIVER_RESULT_TYPES,
    result_count: LoweredResultCount::Four,
};
