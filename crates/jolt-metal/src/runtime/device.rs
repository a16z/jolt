use crate::error::{CapacityLimit, MetalError};
use crate::runtime::sys;

/// Device limits read once when the [`Device`] is created.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceLimits {
    /// Highest supported Apple GPU family (7 = M1 and later).
    pub apple_family: u32,
    /// `MTLDevice.maxBufferLength`: the largest single buffer, in bytes.
    pub max_buffer_length: usize,
    /// `MTLDevice.recommendedMaxWorkingSetSize`: bytes the device can use
    /// without degrading performance. Allocations beyond it are refused.
    pub recommended_max_working_set_size: u64,
    /// `MTLDevice.maxThreadsPerThreadgroup.width`.
    pub max_threads_per_threadgroup: usize,
}

/// What the platform backend reads from a newly opened device.
pub(crate) struct DeviceInfo {
    pub(crate) name: String,
    pub(crate) registry_id: u64,
    pub(crate) limits: DeviceLimits,
}

/// A Metal GPU and its command queue.
///
/// Requires Apple GPU family 7 or later. Buffers and pipelines remember the
/// device that created them and are rejected by any other device.
pub struct Device {
    pub(crate) sys: sys::Device,
    name: String,
    pub(crate) registry_id: u64,
    limits: DeviceLimits,
}

impl Device {
    /// Opens the system's default Metal device.
    ///
    /// Returns [`MetalError::Unavailable`] on non-macOS targets, when no
    /// device exists, or when the device is below Apple GPU family 7.
    pub fn system_default() -> Result<Self, MetalError> {
        let (sys, info) = sys::Device::system_default()?;
        Ok(Self {
            sys,
            name: info.name,
            registry_id: info.registry_id,
            limits: info.limits,
        })
    }

    /// The device's marketing name, e.g. `Apple M4 Max`.
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn limits(&self) -> DeviceLimits {
        self.limits
    }

    /// Refuses an allocation of `bytes` that exceeds a device limit.
    pub(crate) fn admit_allocation(&self, bytes: usize) -> Result<(), MetalError> {
        let exceeded = |limit, requested: usize, available| MetalError::CapacityExceeded {
            limit,
            requested: requested as u128,
            available,
        };
        if bytes > self.limits.max_buffer_length {
            return Err(exceeded(
                CapacityLimit::MaxBufferLength,
                bytes,
                self.limits.max_buffer_length as u128,
            ));
        }
        let budget = self.limits.recommended_max_working_set_size as u128;
        let allocated = self.sys.allocated_bytes()? as u128;
        if allocated.saturating_add(bytes as u128) > budget {
            return Err(exceeded(
                CapacityLimit::WorkingSet,
                bytes,
                budget.saturating_sub(allocated),
            ));
        }
        Ok(())
    }
}
