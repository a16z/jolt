//! Bounded, reusable host storage for asynchronous trace uploads.
//!
//! CUDA events on the pinned columns protect them until their last copy ends.
//! Acquiring mutable slices waits for those events before packing new rows.
//! Cacheable pinned memory supports the row encoder's scattered writes and
//! the instruction-support check that reads the packed kind words.

use cudarc::driver::{CudaContext, DeviceRepr, DriverError, PinnedHostSlice, ValidAsZeroBits};
use jolt_riscv::JoltTraceRow;
use std::sync::Arc;

use super::packed::{fill_rows, EXTRA_WORDS};

pub(super) const STAGING_CYCLES: usize = 1 << 18;
// Smaller windows do not amortize allocating and pinning both buffers.
pub(super) const MIN_PINNED_CYCLES: usize = 4 * STAGING_CYCLES;

pub(super) struct PinnedTrace {
    pub(super) is_noop: PinnedHostSlice<u8>,
    pub(super) address: PinnedHostSlice<u64>,
    pub(super) virtual_sequence: PinnedHostSlice<u32>,
    pub(super) ram_address: PinnedHostSlice<u64>,
    pub(super) extras: PinnedHostSlice<u64>,
}

fn zeroed_pinned<T: DeviceRepr + ValidAsZeroBits>(
    context: &Arc<CudaContext>,
    len: usize,
) -> Result<PinnedHostSlice<T>, DriverError> {
    // SAFETY: initialize every element before exposing a slice or scheduling
    // a copy. Zero is valid for T and this allocation has no in-flight readers.
    unsafe {
        let mut buffer = context.alloc_pinned_with_flags::<T>(len, 0)?;
        buffer.as_mut_ptr()?.write_bytes(0, len);
        Ok(buffer)
    }
}

impl PinnedTrace {
    pub(super) fn new(context: &Arc<CudaContext>) -> Result<Self, DriverError> {
        Ok(Self {
            is_noop: zeroed_pinned(context, STAGING_CYCLES)?,
            address: zeroed_pinned(context, STAGING_CYCLES)?,
            virtual_sequence: zeroed_pinned(context, STAGING_CYCLES)?,
            ram_address: zeroed_pinned(context, STAGING_CYCLES)?,
            extras: zeroed_pinned(context, STAGING_CYCLES * EXTRA_WORDS)?,
        })
    }

    pub(super) fn pack(&mut self, rows: &[JoltTraceRow]) -> Result<(), DriverError> {
        let len = rows.len();
        fill_rows(
            rows,
            0,
            &mut self.is_noop.as_mut_slice()?[..len],
            &mut self.address.as_mut_slice()?[..len],
            &mut self.virtual_sequence.as_mut_slice()?[..len],
            &mut self.ram_address.as_mut_slice()?[..len],
            &mut self.extras.as_mut_slice()?[..len * EXTRA_WORDS],
        );
        Ok(())
    }
}
