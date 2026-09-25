use std::cell::Cell;
use std::marker::PhantomData;
use std::mem::size_of;
use std::sync::Arc;

use bytemuck::NoUninit;

use crate::error::MetalError;
use crate::runtime::buffer::DeviceBuffer;
use crate::runtime::device::Device;
use crate::runtime::library::Pipeline;
use crate::runtime::sys;

/// Metal's limit for inline argument data (`setBytes`).
const MAX_INLINE_BYTES: usize = 4096;

/// One kernel argument. Bindings are positional: the `i`-th binding of a
/// dispatch goes to `[[buffer(i)]]`.
pub struct Binding<'a> {
    pub(crate) kind: BindingKind<'a>,
}

#[derive(Clone, Copy)]
pub(crate) enum BindingKind<'a> {
    Buffer {
        buffer: &'a sys::Buffer,
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
    sys: sys::CommandBatch,
    device_id: u64,
    /// Pipeline of every encoded dispatch, for failure attribution.
    dispatched: Vec<Arc<str>>,
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
            _bindings: PhantomData,
        })
    }

    /// Encodes `pipeline` over `grid` with `bindings`.
    ///
    /// The bindings are checked against the kernel's reflected signature
    /// before anything is encoded: one binding per buffer argument, buffer
    /// element size equal to the argument's pointee size, and inline values
    /// of exactly the argument's size. A rejected dispatch leaves the batch
    /// unchanged. A grid of zero threads encodes nothing.
    pub fn dispatch(
        &mut self,
        pipeline: &Pipeline,
        bindings: &[Binding<'a>],
        grid: Grid,
    ) -> Result<(), MetalError> {
        let invalid = |reason: String| MetalError::InvalidDispatch {
            pipeline: pipeline.name().to_owned(),
            reason,
        };
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
                    element_size,
                    device_id,
                    ..
                } => {
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
        self.sys.dispatch(
            &pipeline.sys,
            bindings,
            grid.threads,
            grid.threads_per_threadgroup,
        )?;
        self.dispatched.push(Arc::clone(&pipeline.name));
        Ok(())
    }

    /// Runs the batch and blocks until the GPU finishes it.
    ///
    /// On a GPU failure the error names every pipeline in the batch; split
    /// the batch to attribute a fault to a single dispatch.
    pub fn commit_and_wait(self) -> Result<(), MetalError> {
        let Self {
            sys, dispatched, ..
        } = self;
        sys.commit_and_wait().map_err(|mut error| {
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
