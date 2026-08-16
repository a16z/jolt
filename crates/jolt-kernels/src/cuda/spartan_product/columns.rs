use cudarc::driver::CudaSlice;

use super::witness::{self, Packed, NARROW, WIDE};
use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::error::CudaError;

pub struct DeviceProductColumns {
    narrow: CudaSlice<u64>,
    wide: CudaSlice<u64>,
    flags: CudaSlice<u32>,
    layout: CudaSlice<u32>,
    cycles: usize,
}

impl DeviceProductColumns {
    pub fn new(context: &CudaKernelContext, packed: &Packed) -> Result<Self, CudaError> {
        let cycles = packed.flags.len();
        if cycles == 0
            || packed.narrow.len() != cycles * NARROW
            || packed.wide.len() != cycles * WIDE * 2
        {
            return Err(CudaError::InvariantViolation {
                reason: "the packed product columns must hold one entry per cycle per column",
            });
        }
        Ok(Self {
            narrow: context.upload_u64_slice(&packed.narrow)?,
            wide: context.upload_u64_slice(&packed.wide)?,
            flags: context.upload_u32_slice(&packed.flags)?,
            layout: context.upload_u32_slice(&witness::CLAIM_LAYOUT)?,
            cycles,
        })
    }

    pub const fn cycles(&self) -> usize {
        self.cycles
    }

    pub const fn narrow(&self) -> &CudaSlice<u64> {
        &self.narrow
    }

    pub const fn wide(&self) -> &CudaSlice<u64> {
        &self.wide
    }

    pub const fn flags(&self) -> &CudaSlice<u32> {
        &self.flags
    }

    pub const fn layout(&self) -> &CudaSlice<u32> {
        &self.layout
    }
}
