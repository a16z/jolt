use metal::Buffer;

use super::{PlaneRole, ScatterDispatchPlan};

pub(super) struct BoundScatter<'a> {
    buffers: [&'a Buffer; 9],
}

impl<'a> BoundScatter<'a> {
    pub(super) const fn new(buffers: [&'a Buffer; 9]) -> Self {
        Self { buffers }
    }

    pub(super) fn buffer(&self, role: PlaneRole) -> &'a Buffer {
        self.buffers[role.metal_buffer_slot()]
    }
}

pub(super) trait ResidentProducerBufferOwner {
    type Error;

    fn bind_checked<'a>(
        &'a self,
        plan: &ScatterDispatchPlan,
    ) -> std::result::Result<BoundScatter<'a>, Self::Error>;
}
