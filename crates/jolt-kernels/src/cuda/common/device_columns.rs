#[cfg(test)]
use std::sync::Arc;

#[cfg(test)]
use cudarc::driver::CudaSlice;

pub(crate) fn witness_identity<T: ?Sized>(witness: &T) -> usize {
    std::ptr::from_ref(witness).cast::<()>() as usize
}

#[cfg(test)]
pub(crate) struct DeviceTraceColumns {
    pub(crate) lookup: Arc<CudaSlice<u64>>,
    pub(crate) pc: Arc<CudaSlice<u32>>,
    pub(crate) ram: Arc<CudaSlice<u32>>,
}
