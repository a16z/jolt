#![expect(
    dead_code,
    reason = "the resident producer owner is wired in a later integration slice"
)]

use super::ScatterDispatchPlan;

/// Integration boundary for a real resident-buffer owner.
///
/// An implementation must bind `required_buffers` in their declared Metal slot
/// order and own the exact layout frozen in the plan. It must validate buffer
/// lengths, non-aliasing, device identity, source generation, pipeline thread
/// limits, and the device's maximum buffer length. Before encoding it clears
/// the status word; after completion it rejects every nonzero status. A
/// [`ScatterDispatchPlan`] alone proves none of those runtime properties. The
/// cycle lookup and claim planes remain alive for cycle-round consumers after
/// scatter completion.
pub(super) trait ResidentProducerBufferOwner {
    type Error;
    type BoundScatter<'a>
    where
        Self: 'a;

    fn bind_checked<'a>(
        &'a self,
        plan: &ScatterDispatchPlan,
    ) -> std::result::Result<Self::BoundScatter<'a>, Self::Error>;
}
