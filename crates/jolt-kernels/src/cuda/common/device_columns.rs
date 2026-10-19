use std::sync::Arc;

use cudarc::driver::CudaSlice;

pub(crate) struct DeviceTraceColumns {
    pub(crate) lookup: Arc<CudaSlice<u64>>,
    pub(crate) pc: Arc<CudaSlice<u32>>,
    pub(crate) ram: Arc<CudaSlice<u32>>,
}
