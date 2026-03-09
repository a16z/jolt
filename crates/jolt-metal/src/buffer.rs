use std::ffi::c_void;
use std::marker::PhantomData;

use jolt_compute::Scalar;
use metal::{Device, MTLResourceOptions};

/// Typed wrapper around a Metal buffer (`MTLBuffer`).
///
/// Stores the element count and phantom type so that the `ComputeBackend`
/// trait can provide a typed API. On Apple Silicon unified memory, the
/// buffer's `contents()` pointer is directly accessible from CPU without
/// an explicit copy.
pub struct MetalBuffer<T: Scalar> {
    pub(crate) raw: metal::Buffer,
    pub(crate) len: usize,
    pub(crate) _marker: PhantomData<T>,
}

/// SAFETY: Metal buffers are refcounted and safe to share across threads.
/// The device accesses them only during command buffer execution, which is
/// synchronized via command buffer completion handlers.
unsafe impl<T: Scalar> Send for MetalBuffer<T> {}
/// SAFETY: See above — Metal buffers are immutable after upload and
/// refcounted by the Metal runtime.
unsafe impl<T: Scalar> Sync for MetalBuffer<T> {}

impl<T: Scalar> MetalBuffer<T> {
    /// Number of `T` elements in this buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer has zero elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw Metal buffer handle.
    pub fn raw(&self) -> &metal::Buffer {
        &self.raw
    }

    /// Byte length of the buffer contents.
    pub fn byte_len(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// Upload host data into a shared-memory Metal buffer.
    ///
    /// On Apple Silicon, `StorageModeShared` means CPU and Metal access the
    /// same physical memory — no DMA copy occurs after the initial memcpy.
    pub(crate) fn from_data(device: &Device, data: &[T]) -> Self {
        let byte_len = std::mem::size_of_val(data) as u64;
        let raw = if byte_len == 0 {
            // Metal doesn't allow zero-length buffers on all implementations.
            device.new_buffer(4, MTLResourceOptions::StorageModeShared)
        } else {
            device.new_buffer_with_data(
                data.as_ptr().cast::<c_void>(),
                byte_len,
                MTLResourceOptions::StorageModeShared,
            )
        };
        Self {
            raw,
            len: data.len(),
            _marker: PhantomData,
        }
    }

    /// Allocate a zeroed Metal buffer with room for `len` elements.
    pub(crate) fn zeroed(device: &Device, len: usize) -> Self {
        let byte_len = len * std::mem::size_of::<T>();
        // Guarantee at least 4 bytes so Metal always returns a valid buffer.
        let raw = device.new_buffer(
            byte_len.max(4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Self {
            raw,
            len,
            _marker: PhantomData,
        }
    }

    /// Read buffer contents back to host memory.
    ///
    /// Caller must ensure all Metal commands writing to this buffer have
    /// completed (via `wait_until_completed`) before calling.
    pub(crate) fn to_vec(&self) -> Vec<T> {
        if self.len == 0 {
            return Vec::new();
        }
        // SAFETY: StorageModeShared guarantees CPU-coherent access after
        // command buffer completion. The caller upholds this invariant.
        unsafe {
            let ptr = self.raw.contents().cast::<T>();
            std::slice::from_raw_parts(ptr, self.len).to_vec()
        }
    }
}
