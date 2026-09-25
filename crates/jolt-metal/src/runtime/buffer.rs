use std::marker::PhantomData;
use std::mem::size_of;

use bytemuck::checked::{self, CheckedBitPattern};
use bytemuck::{NoUninit, Zeroable};

use crate::error::{CapacityLimit, MetalError};
use crate::runtime::device::Device;
use crate::runtime::sys::RawBuffer;

/// A shared-storage Metal buffer holding `len` values of `T`.
///
/// # Host and GPU access
///
/// A [`Batch`](crate::runtime::Batch) borrows the buffers it binds for its
/// whole lifetime, and the GPU runs the batch only inside
/// [`Batch::commit_and_wait`](crate::runtime::Batch::commit_and_wait), which
/// blocks until the GPU finishes. Host views need `&mut self`, so the borrow
/// checker rules out a host view while the GPU may touch the buffer.
pub struct DeviceBuffer<T> {
    pub(crate) sys: RawBuffer,
    len: usize,
    byte_len: usize,
    pub(crate) device_id: u64,
    _element: PhantomData<T>,
}

impl<T> DeviceBuffer<T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn byte_len(device: &Device, len: usize) -> Result<usize, MetalError> {
        let byte_len = len
            .checked_mul(size_of::<T>())
            .ok_or(MetalError::CapacityExceeded {
                limit: CapacityLimit::AddressSpace,
                requested: (len as u128).saturating_mul(size_of::<T>() as u128),
                available: usize::MAX as u128,
            })?;
        device.admit_allocation(byte_len)?;
        Ok(byte_len)
    }

    fn from_sys(device: &Device, sys: RawBuffer, len: usize, byte_len: usize) -> Self {
        Self {
            sys,
            len,
            byte_len,
            device_id: device.registry_id,
            _element: PhantomData,
        }
    }
}

impl<T: NoUninit> DeviceBuffer<T> {
    /// Copies `data` into a new device buffer.
    pub fn from_slice(device: &Device, data: &[T]) -> Result<Self, MetalError> {
        let byte_len = Self::byte_len(device, data.len())?;
        let sys = device
            .sys
            .new_buffer_with_bytes(bytemuck::cast_slice(data))?;
        Ok(Self::from_sys(device, sys, data.len(), byte_len))
    }

    /// A new device buffer of `len` all-zero values.
    pub fn zeroed(device: &Device, len: usize) -> Result<Self, MetalError>
    where
        T: Zeroable,
    {
        let byte_len = Self::byte_len(device, len)?;
        let sys = device.sys.new_zeroed_buffer(byte_len)?;
        Ok(Self::from_sys(device, sys, len, byte_len))
    }
}

impl<T: CheckedBitPattern> DeviceBuffer<T> {
    /// Views the buffer's contents after checking that every element is a
    /// valid `T`. A kernel that wrote an invalid value (for a field element,
    /// a non-canonical one) yields [`MetalError::InvalidReadback`].
    pub fn read(&mut self) -> Result<&[T], MetalError> {
        let byte_len = self.byte_len;
        let bytes = self
            .sys
            .host_bytes(byte_len)
            .ok_or_else(|| MetalError::InvalidReadback {
                reason: format!("allocation is smaller than {byte_len} bytes"),
            })?;
        checked::try_cast_slice(bytes).map_err(|error| {
            let element = bytes
                .chunks_exact(size_of::<T>().max(1))
                .position(|chunk| checked::try_from_bytes::<T>(chunk).is_err());
            MetalError::InvalidReadback {
                reason: match element {
                    Some(index) => format!("element {index} is not a valid value ({error})"),
                    None => error.to_string(),
                },
            }
        })
    }
}
