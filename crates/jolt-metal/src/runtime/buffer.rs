use std::marker::PhantomData;
use std::mem::{size_of, MaybeUninit};
use std::sync::atomic::{AtomicBool, Ordering};

use bytemuck::checked::{self, CheckedBitPattern};
use bytemuck::{NoUninit, Zeroable};

use crate::error::{CapacityLimit, MetalError};
use crate::runtime::device::Device;
use crate::runtime::sys::RawBuffer;

#[derive(Default)]
pub(crate) struct BufferAccess {
    completion_pending: AtomicBool,
}

impl BufferAccess {
    pub(crate) fn require_available(&self, operation: &'static str) -> Result<(), MetalError> {
        if self.completion_pending.load(Ordering::Acquire) {
            return Err(MetalError::BufferUnavailable { operation });
        }
        Ok(())
    }

    pub(crate) fn begin_submission(&self) -> Result<(), MetalError> {
        self.completion_pending
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| ())
            .map_err(|_| MetalError::BufferUnavailable {
                operation: "GPU dispatch",
            })
    }

    pub(crate) fn confirm_completion(&self) {
        self.completion_pending.store(false, Ordering::Release);
    }
}

/// A shared-storage Metal buffer holding `len` values of `T`.
/// Zero-sized element types are rejected at construction.
///
/// # Host and GPU access
///
/// A [`Batch`](crate::runtime::Batch) borrows the buffers it binds for its
/// whole lifetime, and the GPU runs the batch only inside
/// [`Batch::commit_and_wait`](crate::runtime::Batch::commit_and_wait), which
/// blocks until the GPU finishes. Host views need `&mut self`, so the borrow
/// checker rules out a host view while the call is active. If the runtime
/// cannot confirm completion, the buffer remains unavailable to host reads
/// and later dispatches.
/// Ownership may move between threads when `T: Send`; host views still require
/// an exclusive borrow and confirmed GPU completion.
pub struct DeviceBuffer<T> {
    pub(crate) sys: RawBuffer,
    pub(crate) access: BufferAccess,
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
        if size_of::<T>() == 0 {
            return Err(MetalError::ZeroSizedElement);
        }
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
            access: BufferAccess::default(),
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

impl<T> DeviceBuffer<MaybeUninit<T>> {
    /// Allocates storage without initializing its contents. It can be filled
    /// through an exclusive host slice or bound to a GPU writer. Call
    /// [`Self::assume_init`] only after every logical element has been written.
    /// Uninitialized storage has no safe readback method:
    ///
    /// ```compile_fail
    /// use std::mem::MaybeUninit;
    /// use jolt_metal::runtime::DeviceBuffer;
    /// fn read_uninitialized(buffer: &mut DeviceBuffer<MaybeUninit<u64>>) {
    ///     let _ = buffer.read();
    /// }
    /// ```
    pub fn uninit(device: &Device, len: usize) -> Result<Self, MetalError> {
        let byte_len = Self::byte_len(device, len)?;
        let sys = device.sys.new_uninit_buffer(byte_len)?;
        Ok(Self::from_sys(device, sys, len, byte_len))
    }

    /// Exclusive storage for host initialization. It cannot overlap an encoded
    /// batch borrow, and uncertain GPU completion prevents access.
    pub fn write_uninit(&mut self) -> Result<&mut [MaybeUninit<T>], MetalError> {
        self.access.require_available("host initialization")?;
        self.sys
            .host_uninit(self.len)
            .ok_or_else(|| MetalError::InvalidReadback {
                reason: "initialization storage has incompatible size or alignment".into(),
            })
    }

    /// Marks a completely initialized buffer ready for ordinary use.
    ///
    /// # Safety
    /// All bytes of every logical element, including padding, must be
    /// initialized, and each element must be a valid `T`. A GPU initializer
    /// must have successfully completed and write every element on all paths.
    pub unsafe fn assume_init(self) -> Result<DeviceBuffer<T>, MetalError> {
        self.access.require_available("initialization completion")?;
        Ok(DeviceBuffer {
            sys: self.sys,
            access: self.access,
            len: self.len,
            byte_len: self.byte_len,
            device_id: self.device_id,
            _element: PhantomData,
        })
    }
}

impl<T: CheckedBitPattern + NoUninit> DeviceBuffer<T> {
    /// Views the buffer's contents after checking that every element is a
    /// valid `T`. A kernel that wrote an invalid value (for a field element,
    /// a non-canonical one) yields [`MetalError::InvalidReadback`].
    pub fn read(&mut self) -> Result<&[T], MetalError> {
        self.access.require_available("host read")?;
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
