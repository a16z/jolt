//! The only module that talks to Metal. Every Objective-C message send runs
//! inside [`objc`], which drains an autorelease pool and turns a raised
//! Objective-C exception into [`MetalError::ObjcException`] instead of letting
//! it unwind into Rust and abort the process.
//!
//! Callers validate arguments before calling in here; this layer only checks
//! what Metal itself reports (nil objects, `NSError`s, command-buffer status).

use std::panic::AssertUnwindSafe;
use std::ptr::NonNull;
use std::slice;
use std::time::Duration;

use objc2::exception;
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::{NSObjectProtocol, ProtocolObject};
use objc2::ProtocolType;
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLBinding, MTLBindingType, MTLBuffer, MTLBufferBinding, MTLCommandBuffer,
    MTLCommandBufferErrorDomain, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLDispatchType, MTLGPUFamily, MTLLanguageVersion,
    MTLLibrary, MTLPipelineOption, MTLResourceOptions, MTLSize,
};

use crate::error::{CommandBufferError, MetalError};
use crate::runtime::batch::{Binding, BindingKind};
use crate::runtime::device::{DeviceInfo, DeviceLimits};
use crate::runtime::library::{ArgumentSlot, PipelineInfo};

// `MTLCreateSystemDefaultDevice` returns nil unless CoreGraphics is linked
// (see the `objc2-metal` crate docs). Headless command-line processes do not
// otherwise link it.
#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {}

/// The Apple GPU families probed, lowest first. Family 7 (M1) is the floor.
const APPLE_FAMILIES: [(MTLGPUFamily, u32); 4] = [
    (MTLGPUFamily::Apple7, 7),
    (MTLGPUFamily::Apple8, 8),
    (MTLGPUFamily::Apple9, 9),
    (MTLGPUFamily::Apple10, 10),
];

/// Metal rejects zero-length buffers; empty `DeviceBuffer`s get this many bytes.
const MIN_ALLOCATION: usize = 16;

/// Runs `f` inside an autorelease pool and catches any Objective-C exception.
///
/// After an exception the state of the Metal objects involved is unspecified,
/// which is why the result is a [`Fault`](crate::ErrorClass::Fault).
fn objc<R>(operation: &'static str, f: impl FnOnce() -> R) -> Result<R, MetalError> {
    autoreleasepool(|_| exception::catch(AssertUnwindSafe(f))).map_err(|raised| {
        MetalError::ObjcException {
            operation,
            description: raised.map_or_else(|| "nil exception".to_owned(), |e| format!("{e:?}")),
        }
    })
}

fn describe(error: &NSError) -> String {
    error.localizedDescription().to_string()
}

pub(crate) struct RawDevice {
    raw: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

impl RawDevice {
    pub(crate) fn system_default() -> Result<(Self, DeviceInfo), MetalError> {
        objc("device discovery", || {
            let Some(raw) = MTLCreateSystemDefaultDevice() else {
                return Err(MetalError::Unavailable {
                    reason: "MTLCreateSystemDefaultDevice returned no device".to_owned(),
                });
            };
            let apple_family = APPLE_FAMILIES
                .iter()
                .filter(|(family, _)| raw.supportsFamily(*family))
                .map(|(_, number)| *number)
                .max()
                .ok_or_else(|| MetalError::Unavailable {
                    reason: format!("`{}` is below Apple GPU family 7", raw.name()),
                })?;
            let queue = raw.newCommandQueue().ok_or(MetalError::Unavailable {
                reason: "the device returned no command queue".to_owned(),
            })?;
            let info = DeviceInfo {
                name: raw.name().to_string(),
                registry_id: raw.registryID(),
                limits: DeviceLimits {
                    apple_family,
                    max_buffer_length: raw.maxBufferLength(),
                    recommended_max_working_set_size: raw.recommendedMaxWorkingSetSize(),
                    max_threads_per_threadgroup: raw.maxThreadsPerThreadgroup().width,
                },
            };
            Ok((Self { raw, queue }, info))
        })?
    }

    pub(crate) fn allocated_bytes(&self) -> Result<usize, MetalError> {
        objc("currentAllocatedSize", || self.raw.currentAllocatedSize())
    }

    pub(crate) fn new_zeroed_buffer(&self, bytes: usize) -> Result<RawBuffer, MetalError> {
        let length = bytes.max(MIN_ALLOCATION);
        let raw = objc("newBufferWithLength", || {
            self.raw
                .newBufferWithLength_options(length, buffer_options())
        })?
        .ok_or(MetalError::AllocationFailed { bytes })?;
        let buffer = RawBuffer::new(raw)?;
        // SAFETY: `contents` points to `allocated` writable bytes of a
        // shared-storage buffer that no command buffer references yet.
        unsafe { buffer.contents.write_bytes(0, buffer.allocated) };
        Ok(buffer)
    }

    pub(crate) fn new_buffer_with_bytes(&self, data: &[u8]) -> Result<RawBuffer, MetalError> {
        if data.is_empty() {
            return self.new_zeroed_buffer(0);
        }
        let raw = objc("newBufferWithBytes", || {
            // SAFETY: `data` is a live, non-empty slice of `data.len()`
            // initialized bytes; Metal copies them before returning.
            unsafe {
                self.raw.newBufferWithBytes_length_options(
                    NonNull::from(data).cast(),
                    data.len(),
                    buffer_options(),
                )
            }
        })?
        .ok_or(MetalError::AllocationFailed { bytes: data.len() })?;
        RawBuffer::new(raw)
    }

    pub(crate) fn compile(&self, source: &str) -> Result<RawLibrary, MetalError> {
        objc("newLibraryWithSource", || {
            let options = MTLCompileOptions::new();
            options.setLanguageVersion(MTLLanguageVersion::Version3_0);
            // `mathMode` needs macOS 15; the platform floor is macOS 13.
            #[expect(deprecated, reason = "mathMode is unavailable below macOS 15")]
            options.setFastMathEnabled(false);
            self.raw
                .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
                .map(RawLibrary)
                .map_err(|error| MetalError::ShaderCompile {
                    log: describe(&error),
                })
        })?
    }

    pub(crate) fn command_batch(&self) -> Result<RawCommandBatch, MetalError> {
        objc("command batch creation", || {
            let buffer = self.queue.commandBuffer().ok_or(MetalError::NilObject {
                object: "command buffer",
            })?;
            let encoder = buffer
                .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
                .ok_or(MetalError::NilObject {
                    object: "compute command encoder",
                })?;
            Ok(RawCommandBatch {
                buffer,
                encoder,
                encoding: true,
            })
        })?
    }
}

fn buffer_options() -> MTLResourceOptions {
    MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeTracked
}

pub(crate) struct RawLibrary(Retained<ProtocolObject<dyn MTLLibrary>>);

impl RawLibrary {
    pub(crate) fn pipeline(
        &self,
        device: &RawDevice,
        kernel: &str,
    ) -> Result<(RawPipeline, PipelineInfo), MetalError> {
        let failed = |reason: String| MetalError::Pipeline {
            kernel: kernel.to_owned(),
            reason,
        };
        objc("pipeline creation", || {
            let function = self
                .0
                .newFunctionWithName(&NSString::from_str(kernel))
                .ok_or_else(|| failed("no such function in the library".to_owned()))?;
            let mut reflection = None;
            // SAFETY: creating a pipeline does not run `function`; the
            // reflection out-pointer is a live `Option` owned by this frame.
            let raw = unsafe {
                device
                    .raw
                    .newComputePipelineStateWithFunction_options_reflection_error(
                        &function,
                        MTLPipelineOption::BindingInfo | MTLPipelineOption::BufferTypeInfo,
                        Some(&mut reflection),
                    )
            }
            .map_err(|error| failed(describe(&error)))?;
            let reflection =
                reflection.ok_or_else(|| failed("Metal returned no reflection".to_owned()))?;
            let buffer_binding = <dyn MTLBufferBinding>::protocol()
                .ok_or_else(|| failed("MTLBufferBinding protocol is not registered".to_owned()))?;
            let mut slots = Vec::new();
            for binding in reflection.bindings() {
                if !binding.isArgument() {
                    continue;
                }
                if binding.r#type() != MTLBindingType::Buffer
                    || !binding.conformsToProtocol(buffer_binding)
                {
                    return Err(failed(format!(
                        "argument `{}` is not a buffer; the runtime binds buffers only",
                        binding.name()
                    )));
                }
                // SAFETY: the binding conforms to `MTLBufferBinding` (checked
                // above), so viewing it through that protocol is valid.
                let binding: Retained<ProtocolObject<dyn MTLBufferBinding>> =
                    unsafe { Retained::cast_unchecked(binding) };
                slots.push(ArgumentSlot {
                    name: binding.name().to_string(),
                    index: binding.index(),
                    data_size: binding.bufferDataSize(),
                });
            }
            let info = PipelineInfo {
                max_total_threads_per_threadgroup: raw.maxTotalThreadsPerThreadgroup(),
                thread_execution_width: raw.threadExecutionWidth(),
                slots,
            };
            Ok((RawPipeline(raw), info))
        })?
    }
}

pub(crate) struct RawPipeline(Retained<ProtocolObject<dyn MTLComputePipelineState>>);

pub(crate) struct RawBuffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Cached `contents()`; stable for the buffer's lifetime (shared storage).
    contents: NonNull<u8>,
    /// Allocation size in bytes (at least [`MIN_ALLOCATION`]).
    allocated: usize,
}

impl RawBuffer {
    fn new(raw: Retained<ProtocolObject<dyn MTLBuffer>>) -> Result<Self, MetalError> {
        let (contents, allocated) = objc("buffer contents", || (raw.contents(), raw.length()))?;
        Ok(Self {
            raw,
            contents: contents.cast(),
            allocated,
        })
    }

    /// The first `len` bytes of the buffer, for the host.
    ///
    /// `&mut self` is the proof that no GPU work touches the buffer: a
    /// [`Batch`](crate::runtime::Batch) holds `&Buffer` from the dispatch
    /// that binds it until it is committed (and the GPU is done) or dropped
    /// (and nothing ran). `None` if `len` exceeds the allocation.
    pub(crate) fn host_bytes(&mut self, len: usize) -> Option<&[u8]> {
        if len > self.allocated {
            return None;
        }
        // SAFETY: `contents` points to `allocated >= len` bytes initialized
        // at creation (zeroed or copied) and since written only by completed
        // GPU work; no GPU access is in flight, per the borrow argument
        // above; the slice borrows `self`, so the buffer outlives it.
        Some(unsafe { slice::from_raw_parts(self.contents.as_ptr(), len) })
    }
}

pub(crate) struct RawCommandBatch {
    buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    /// Whether `endEncoding` is still owed. Metal raises if an encoder is
    /// released while encoding, so [`Drop`] ends it for uncommitted batches.
    encoding: bool,
}

impl RawCommandBatch {
    /// Encodes one 1-D dispatch. The caller has validated every binding
    /// against the pipeline's reflection and the grid against its limits.
    pub(crate) fn dispatch(
        &mut self,
        pipeline: &RawPipeline,
        bindings: &[Binding<'_>],
        threads: usize,
        threads_per_threadgroup: usize,
    ) -> Result<(), MetalError> {
        objc("dispatch encoding", || {
            self.encoder.setComputePipelineState(&pipeline.0);
            for (index, binding) in bindings.iter().enumerate() {
                match binding.kind {
                    // SAFETY: the buffer outlives the batch (borrowed for its
                    // lifetime) and offset 0 is in bounds of every buffer.
                    BindingKind::Buffer { buffer, .. } => unsafe {
                        self.encoder
                            .setBuffer_offset_atIndex(Some(&buffer.raw), 0, index);
                    },
                    // SAFETY: `bytes` is a live slice of `bytes.len()`
                    // initialized bytes (at most 4 KiB, checked by the
                    // caller); Metal copies them during this call.
                    BindingKind::Bytes(bytes) => unsafe {
                        self.encoder.setBytes_length_atIndex(
                            NonNull::from(bytes).cast(),
                            bytes.len(),
                            index,
                        );
                    },
                }
            }
            self.encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: threads,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threads_per_threadgroup,
                    height: 1,
                    depth: 1,
                },
            );
        })
    }

    /// Submits the batch and blocks until the GPU finishes it.
    ///
    /// WARNING: if an exception escapes `commit` or `waitUntilCompleted`, the
    /// GPU may still be running the batch when this returns. The command
    /// buffer retains its buffers, so memory stays valid, but their contents
    /// are unspecified; the resulting `Fault` tells the consumer to discard
    /// the device and everything allocated on it.
    pub(crate) fn commit_and_wait(mut self) -> Result<Duration, MetalError> {
        self.encoding = false;
        objc("command buffer submission", || {
            self.encoder.endEncoding();
            self.buffer.commit();
            self.buffer.waitUntilCompleted();
            let status = self.buffer.status();
            if status == MTLCommandBufferStatus::Completed {
                return Ok(gpu_time(
                    self.buffer.GPUStartTime(),
                    self.buffer.GPUEndTime(),
                ));
            }
            let (code, description) = match self.buffer.error() {
                Some(error) => {
                    // SAFETY: reading an immutable framework-provided
                    // `NSString` constant.
                    let domain = unsafe { MTLCommandBufferErrorDomain };
                    let code = if *error.domain() == *domain {
                        CommandBufferError::from_code(error.code())
                    } else {
                        CommandBufferError::Unknown(error.code())
                    };
                    (code, format!("{}: {}", error.domain(), describe(&error)))
                }
                None => (
                    CommandBufferError::Unreported,
                    format!("command buffer ended with status {}", status.0),
                ),
            };
            Err(MetalError::CommandBuffer {
                code,
                description,
                pipelines: Vec::new(),
            })
        })?
    }
}

/// The GPU execution time between two host-clock timestamps in seconds.
///
/// A completed command buffer has both timestamps and `end >= start`; a
/// value outside that contract reads as zero rather than failing a batch
/// whose results are valid.
fn gpu_time(start: f64, end: f64) -> Duration {
    Duration::try_from_secs_f64(end - start).unwrap_or(Duration::ZERO)
}

impl Drop for RawCommandBatch {
    fn drop(&mut self) {
        if self.encoding {
            // Nothing to report from a destructor: a failure here means the
            // batch was never submitted, which is what dropping it asks for.
            let _ = objc("endEncoding on drop", || self.encoder.endEncoding());
        }
    }
}
