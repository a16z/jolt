use std::cell::Cell;
use std::marker::PhantomData;
use std::mem::size_of;
use std::sync::Arc;

use bytemuck::NoUninit;

use crate::error::MetalError;
use crate::runtime::buffer::{BufferAccess, DeviceBuffer};
use crate::runtime::device::Device;
use crate::runtime::library::Pipeline;
use crate::runtime::sys::{RawBatchOutcome, RawBuffer, RawCommandBatch};

/// Metal's limit for inline argument data (`setBytes`).
const MAX_INLINE_BYTES: usize = 4096;

/// One kernel argument. Bindings are positional: the `i`-th binding of a
/// dispatch goes to `[[buffer(i)]]`.
#[derive(Clone, Copy)]
pub struct Binding<'a> {
    pub(crate) kind: BindingKind<'a>,
}

#[derive(Clone, Copy)]
pub(crate) enum BindingKind<'a> {
    Buffer {
        buffer: &'a RawBuffer,
        access: &'a BufferAccess,
        element_size: usize,
        device_id: u64,
    },
    Bytes(&'a [u8]),
}

impl<'a> Binding<'a> {
    /// Binds a whole buffer to a `device T*` argument.
    pub fn buffer<T>(buffer: &'a DeviceBuffer<T>) -> Self {
        Self {
            kind: BindingKind::Buffer {
                buffer: &buffer.sys,
                access: &buffer.access,
                element_size: size_of::<T>(),
                device_id: buffer.device_id,
            },
        }
    }

    /// Binds a copy of `value` to a `constant T&` argument.
    pub fn value<T: NoUninit>(value: &'a T) -> Self {
        Self {
            kind: BindingKind::Bytes(bytemuck::bytes_of(value)),
        }
    }
}

/// A one-dimensional dispatch shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    threads: usize,
    threads_per_threadgroup: usize,
}

impl Grid {
    /// `threads` threads in threadgroups of `threads_per_threadgroup`; the
    /// last threadgroup may be partial.
    pub fn linear(threads: usize, threads_per_threadgroup: usize) -> Self {
        Self {
            threads,
            threads_per_threadgroup,
        }
    }
}

/// A sequence of dispatches submitted as one command buffer.
///
/// Dispatches run in order, and each sees the writes of the ones before it.
/// Nothing runs until [`Batch::commit_and_wait`]. Dropping a batch discards
/// it without running anything.
pub struct Batch<'a> {
    sys: RawCommandBatch,
    device_id: u64,
    /// Pipeline of every encoded dispatch, for failure attribution.
    dispatched: Vec<Arc<str>>,
    bound_buffers: Vec<&'a BufferAccess>,
    /// A backend encoding error can leave a partial dispatch in the command
    /// buffer. Such a batch may only be dropped, never submitted.
    encoding_failed: bool,
    /// Invariant in `'a`: every bound buffer stays borrowed until the batch
    /// is committed or dropped.
    _bindings: PhantomData<Cell<&'a ()>>,
}

impl<'a> Batch<'a> {
    pub fn new(device: &Device) -> Result<Self, MetalError> {
        Ok(Self {
            sys: device.sys.command_batch()?,
            device_id: device.registry_id,
            dispatched: Vec::new(),
            bound_buffers: Vec::new(),
            encoding_failed: false,
            _bindings: PhantomData,
        })
    }

    /// Encodes `pipeline` over `grid` with `bindings`.
    ///
    /// The bindings are checked against the kernel's reflected signature
    /// before anything is encoded: one binding per buffer argument, buffer
    /// element size equal to the argument's pointee size, and inline values
    /// of exactly the argument's size. A dispatch rejected by those checks
    /// leaves the batch unchanged. A grid of zero threads encodes nothing.
    ///
    /// # Safety
    ///
    /// For every dispatched thread, the kernel must confine device-memory
    /// accesses to the allocations supplied in `bindings` and keep all device
    /// and threadgroup accesses within their respective allocations. Every
    /// bound buffer must be large enough for the kernel's complete access
    /// pattern, including access derived from inline values. Aliasing between
    /// bindings, concurrent accesses between threads, and synchronization
    /// within the kernel must obey the Metal Shading Language memory model.
    /// These requirements apply to reads as well as writes; pipeline
    /// reflection validates argument types, but cannot validate access bounds
    /// or the shader's memory behavior. Validation failures leave the batch
    /// unchanged. A backend error during encoding makes the batch unusable.
    pub unsafe fn dispatch_unchecked(
        &mut self,
        pipeline: &Pipeline,
        bindings: &[Binding<'a>],
        grid: Grid,
    ) -> Result<(), MetalError> {
        let invalid = |reason: String| MetalError::InvalidDispatch {
            pipeline: pipeline.name().to_owned(),
            reason,
        };
        if self.encoding_failed {
            return Err(invalid(
                "the batch is unusable after an earlier encoding failure".to_owned(),
            ));
        }
        if pipeline.device_id != self.device_id {
            return Err(invalid("pipeline belongs to a different device".to_owned()));
        }
        let slots = &pipeline.info.slots;
        if bindings.len() != slots.len() {
            return Err(invalid(format!(
                "kernel takes {} buffer arguments, got {} bindings",
                slots.len(),
                bindings.len()
            )));
        }
        for (slot, binding) in slots.iter().zip(bindings) {
            let size = match binding.kind {
                BindingKind::Buffer {
                    access,
                    element_size,
                    device_id,
                    ..
                } => {
                    access.require_available("GPU dispatch")?;
                    if device_id != self.device_id {
                        return Err(invalid(format!(
                            "buffer for `{}` belongs to a different device",
                            slot.name
                        )));
                    }
                    element_size
                }
                BindingKind::Bytes(bytes) => {
                    if bytes.len() > MAX_INLINE_BYTES {
                        return Err(invalid(format!(
                            "inline value for `{}` is {} bytes; the limit is {MAX_INLINE_BYTES}",
                            slot.name,
                            bytes.len()
                        )));
                    }
                    bytes.len()
                }
            };
            if size != slot.data_size {
                return Err(invalid(format!(
                    "`{}` (buffer {}) expects {}-byte data; the binding has {size}-byte elements",
                    slot.name, slot.index, slot.data_size
                )));
            }
        }
        if u32::try_from(grid.threads).is_err() {
            return Err(invalid(format!(
                "{} threads overflows the 32-bit thread position",
                grid.threads
            )));
        }
        let max_group = pipeline.max_total_threads_per_threadgroup();
        if grid.threads_per_threadgroup == 0 || grid.threads_per_threadgroup > max_group {
            return Err(invalid(format!(
                "threadgroup size {} is outside 1..={max_group}",
                grid.threads_per_threadgroup
            )));
        }
        if grid.threads == 0 {
            return Ok(());
        }
        self.encoding_failed = true;
        self.sys.dispatch(
            &pipeline.sys,
            bindings,
            grid.threads,
            grid.threads_per_threadgroup,
        )?;
        self.encoding_failed = false;
        for binding in bindings {
            if let BindingKind::Buffer { access, .. } = binding.kind {
                if !self
                    .bound_buffers
                    .iter()
                    .any(|bound| std::ptr::eq(*bound, access))
                {
                    self.bound_buffers.push(access);
                }
            }
        }
        self.dispatched.push(Arc::clone(&pipeline.name));
        Ok(())
    }

    /// Runs the batch and blocks until the GPU finishes it.
    ///
    /// On a GPU failure the error names every pipeline in the batch; split
    /// the batch to attribute a fault to a single dispatch.
    pub fn commit_and_wait(self) -> Result<(), MetalError> {
        if self.encoding_failed {
            return Err(MetalError::InvalidBatch {
                reason: "a dispatch failed while it was being encoded",
            });
        }
        let Self {
            sys,
            dispatched,
            bound_buffers,
            ..
        } = self;
        let submission = Submission::begin(&bound_buffers)?;
        let result = match sys.commit_and_wait() {
            RawBatchOutcome::CompletionConfirmed(result) => {
                submission.confirm_completion();
                result
            }
            RawBatchOutcome::CompletionUncertain(error) => Err(error),
        };
        result.map_err(|mut error| {
            if let MetalError::CommandBuffer { pipelines, .. } = &mut error {
                for name in &dispatched {
                    if !pipelines.iter().any(|seen| **seen == **name) {
                        pipelines.push(name.to_string());
                    }
                }
            }
            error
        })
    }
}

/// Bound buffers remain unavailable if this guard is dropped without an
/// explicit completion confirmation, including during unwinding.
struct Submission<'a> {
    buffers: &'a [&'a BufferAccess],
}

impl<'a> Submission<'a> {
    fn begin(buffers: &'a [&'a BufferAccess]) -> Result<Self, MetalError> {
        let mut acquired = Vec::with_capacity(buffers.len());
        for buffer in buffers {
            if let Err(error) = buffer.begin_submission() {
                for acquired_buffer in acquired {
                    BufferAccess::confirm_completion(acquired_buffer);
                }
                return Err(error);
            }
            acquired.push(*buffer);
        }
        Ok(Self { buffers })
    }

    fn confirm_completion(self) {
        for buffer in self.buffers {
            buffer.confirm_completion();
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests may panic on assertion failures")]
mod tests {
    use super::*;

    #[test]
    fn uncertain_submission_keeps_buffers_unavailable() {
        let access = BufferAccess::default();
        let buffers = [&access];
        {
            // Simulate successful submission followed by a failure or unwind
            // before completion can be confirmed.
            let _submission = Submission::begin(&buffers).unwrap();
        }

        assert!(matches!(
            access.require_available("host read"),
            Err(MetalError::BufferUnavailable {
                operation: "host read"
            })
        ));
        assert!(matches!(
            access.require_available("GPU dispatch"),
            Err(MetalError::BufferUnavailable {
                operation: "GPU dispatch"
            })
        ));
        assert!(Submission::begin(&buffers).is_err());
    }

    #[test]
    fn confirmed_completion_releases_buffers() {
        let access = BufferAccess::default();
        let buffers = [&access];
        Submission::begin(&buffers).unwrap().confirm_completion();

        access.require_available("host read").unwrap();
        access.require_available("GPU dispatch").unwrap();
    }

    #[cfg(target_os = "macos")]
    mod gpu {
        use super::*;

        #[test]
        fn uncertain_submission_rejects_device_buffer_read() {
            let device = Device::system_default().unwrap();
            let mut buffer = DeviceBuffer::<u32>::zeroed(&device, 1).unwrap();
            {
                let buffers = [&buffer.access];
                let _submission = Submission::begin(&buffers).unwrap();
            }

            assert!(matches!(
                buffer.read(),
                Err(MetalError::BufferUnavailable {
                    operation: "host read"
                })
            ));
        }
    }

    #[test]
    fn rejected_submission_does_not_clear_an_existing_poison() {
        let ready = BufferAccess::default();
        let poisoned = BufferAccess::default();
        {
            let poisoned_buffers = [&poisoned];
            let _submission = Submission::begin(&poisoned_buffers).unwrap();
        }

        let buffers = [&ready, &poisoned];
        assert!(Submission::begin(&buffers).is_err());

        ready.require_available("host read").unwrap();
        assert!(poisoned.require_available("host read").is_err());
    }
}
