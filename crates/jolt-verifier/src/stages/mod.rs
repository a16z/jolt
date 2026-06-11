//! Typed verifier stage entry points.

use common::jolt_device::MemoryLayout;
use jolt_claims::protocols::jolt::{
    formulas::claim_reductions::advice, AdviceClaimReductionLayout, JoltAdviceKind,
    PrecommittedClaimReduction, TracePolynomialOrder,
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

/// Per-polynomial claim-reduction layouts over the shared precommitted
/// scheduling reference, derived once during input validation.
///
/// The reference spans all present precommitted polynomials, so the layouts
/// must be built together; stages read them from `CheckedInputs` instead of
/// re-deriving the schedule.
#[derive(Clone, Debug, PartialEq)]
pub struct PrecommittedSchedule {
    pub trusted_advice: Option<AdviceClaimReductionLayout>,
    pub untrusted_advice: Option<AdviceClaimReductionLayout>,
}

impl PrecommittedSchedule {
    pub fn new(
        trace_order: TracePolynomialOrder,
        log_t: usize,
        log_k_chunk: usize,
        memory_layout: &MemoryLayout,
        trusted_advice_present: bool,
        untrusted_advice_present: bool,
    ) -> Self {
        let trusted_max_bytes =
            trusted_advice_present.then_some(memory_layout.max_trusted_advice_size as usize);
        let untrusted_max_bytes =
            untrusted_advice_present.then_some(memory_layout.max_untrusted_advice_size as usize);
        let candidates = advice::candidate_total_vars(trusted_max_bytes, untrusted_max_bytes);
        let scheduling_reference = PrecommittedClaimReduction::scheduling_reference(
            log_t + log_k_chunk,
            &candidates,
            log_k_chunk,
        );
        let layout = |max_bytes: Option<usize>| {
            max_bytes.map(|max_bytes| {
                AdviceClaimReductionLayout::balanced(
                    trace_order,
                    log_t,
                    scheduling_reference,
                    max_bytes,
                )
            })
        };
        Self {
            trusted_advice: layout(trusted_max_bytes),
            untrusted_advice: layout(untrusted_max_bytes),
        }
    }

    pub fn advice(&self, kind: JoltAdviceKind) -> Option<&AdviceClaimReductionLayout> {
        match kind {
            JoltAdviceKind::Trusted => self.trusted_advice.as_ref(),
            JoltAdviceKind::Untrusted => self.untrusted_advice.as_ref(),
        }
    }
}
