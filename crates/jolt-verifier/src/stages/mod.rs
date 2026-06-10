//! Typed verifier stage entry points.

use common::jolt_device::MemoryLayout;
use jolt_claims::protocols::jolt::{
    formulas::claim_reductions::advice, AdviceClaimReductionLayout, PrecommittedClaimReduction,
    TracePolynomialOrder,
};

pub mod stage1;
pub mod stage2;
pub mod stage3;
pub mod stage4;
pub mod stage5;
pub mod stage6;
pub mod stage7;
pub mod stage8;
pub(crate) mod zk;

pub(crate) struct AdviceLayouts {
    pub trusted: Option<AdviceClaimReductionLayout>,
    pub untrusted: Option<AdviceClaimReductionLayout>,
}

/// Builds the per-kind advice claim-reduction layouts over the shared
/// precommitted scheduling reference. The reference spans all present advice
/// polynomials, so both layouts must be derived together.
pub(crate) fn advice_layouts(
    trace_order: TracePolynomialOrder,
    log_t: usize,
    log_k_chunk: usize,
    memory_layout: &MemoryLayout,
    trusted_advice_present: bool,
    untrusted_advice_present: bool,
) -> AdviceLayouts {
    let trusted_max_bytes =
        trusted_advice_present.then_some(memory_layout.max_trusted_advice_size as usize);
    let untrusted_max_bytes =
        untrusted_advice_present.then_some(memory_layout.max_untrusted_advice_size as usize);
    let candidates = advice::precommitted_candidates(trusted_max_bytes, untrusted_max_bytes);
    let scheduling_reference = PrecommittedClaimReduction::scheduling_reference(
        log_t + log_k_chunk,
        &candidates,
        log_k_chunk,
    );
    AdviceLayouts {
        trusted: trusted_max_bytes.map(|max_bytes| {
            AdviceClaimReductionLayout::balanced(
                trace_order,
                log_t,
                scheduling_reference,
                max_bytes,
            )
        }),
        untrusted: untrusted_max_bytes.map(|max_bytes| {
            AdviceClaimReductionLayout::balanced(
                trace_order,
                log_t,
                scheduling_reference,
                max_bytes,
            )
        }),
    }
}
