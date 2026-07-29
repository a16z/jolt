//! Process-global Metal context: device, command queue, and the compiled
//! compute pipelines.
//!
//! Shader sources are embedded `.metal` files (`include_str!`) concatenated
//! behind the generated constants preamble and compiled ONCE per process
//! with `newLibraryWithSource` (~30 ms on an M4 — no offline `metallib`, no
//! Xcode dependency). Every pipeline is built eagerly at context creation,
//! so [`JoltBackend::metal`](crate::JoltBackend::metal) fails closed: a
//! missing device, a shader that no longer compiles, or a pipeline whose
//! register pressure undercuts the fixed threadgroup width all surface at
//! construction, never mid-proof.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::OnceLock;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLSize,
};

use super::buffers::DeviceBuffer;
use super::error::MetalError;
use super::field;

/// Fixed compute threadgroup width for every kernel; the shader-side
/// `JK_TG_SIZE` is generated from this constant, so the two cannot drift.
pub const THREADGROUP_SIZE: usize = 256;

/// Capacity of `jk_fr_bind_eval`'s eval-point set (`JK_MAX_EVAL_POINTS`
/// shader-side). The actual point count is a runtime parameter.
pub const MAX_EVAL_POINTS: usize = 8;

/// The compute kernels compiled into the global library. `name()` must match
/// the `[[kernel]]` function names in `shaders/kernels.metal`; prewarm
/// catches a mismatch at construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelId {
    Noop,
    FrMul,
    FrAdd,
    FrSub,
    FrPow2k,
    FrBind,
    FrBindEval,
}

impl KernelId {
    pub const ALL: [Self; 7] = [
        Self::Noop,
        Self::FrMul,
        Self::FrAdd,
        Self::FrSub,
        Self::FrPow2k,
        Self::FrBind,
        Self::FrBindEval,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Noop => "jk_noop",
            Self::FrMul => "jk_fr_mul",
            Self::FrAdd => "jk_fr_add",
            Self::FrSub => "jk_fr_sub",
            Self::FrPow2k => "jk_fr_pow2k",
            Self::FrBind => "jk_fr_bind",
            Self::FrBindEval => "jk_fr_bind_eval",
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

/// Device + queue + prewarmed pipelines. One per process, behind
/// [`MetalContext::global`]; all members are documented thread-safe by Metal
/// (command buffers and encoders, which are not, live in [`ComputePass`] and
/// never cross threads).
pub struct MetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// Indexed by [`KernelId::index`] (construction order = `KernelId::ALL`).
    pipelines: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
}

static CONTEXT: OnceLock<Result<MetalContext, MetalError>> = OnceLock::new();

impl MetalContext {
    /// The process-global context, initialized (device lookup, shader
    /// compilation, pipeline prewarm) on first call.
    pub fn global() -> Result<&'static Self, MetalError> {
        CONTEXT
            .get_or_init(Self::new)
            .as_ref()
            .map_err(Clone::clone)
    }

    fn new() -> Result<Self, MetalError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::NoDevice)?;
        let source = format!(
            "{}\n{}\n{}",
            field::constants_preamble(),
            include_str!("shaders/fr.metal"),
            include_str!("shaders/kernels.metal"),
        );
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(&source), None)
            .map_err(|e| MetalError::Compile(e.localizedDescription().to_string()))?;

        let mut pipelines = Vec::with_capacity(KernelId::ALL.len());
        for kernel in KernelId::ALL {
            let function = library
                .newFunctionWithName(&NSString::from_str(kernel.name()))
                .ok_or(MetalError::MissingFunction(kernel.name()))?;
            let pipeline = device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| MetalError::Pipeline {
                    name: kernel.name(),
                    reason: e.localizedDescription().to_string(),
                })?;
            let max = pipeline.maxTotalThreadsPerThreadgroup();
            if max < THREADGROUP_SIZE {
                return Err(MetalError::ThreadgroupTooSmall {
                    name: kernel.name(),
                    max,
                    need: THREADGROUP_SIZE,
                });
            }
            pipelines.push(pipeline);
        }

        let queue = device.newCommandQueue().ok_or(MetalError::NoCommandQueue)?;
        Ok(Self {
            device,
            queue,
            pipelines,
        })
    }

    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    pub(super) fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// Open a command buffer + compute encoder for one or more dispatches.
    /// `'b` is the dispatched buffers' minimum lifetime: every buffer handed
    /// to [`ComputePass::dispatch`] must outlive the pass, so no-copy
    /// wrappers cannot lose their backing memory while the GPU may touch it.
    pub fn begin_pass<'b>(&self) -> Result<ComputePass<'_, 'b>, MetalError> {
        let cb = self
            .queue
            .commandBuffer()
            .ok_or(MetalError::NoCommandBuffer)?;
        let encoder = cb
            .computeCommandEncoder()
            .ok_or(MetalError::NoCommandBuffer)?;
        Ok(ComputePass {
            ctx: self,
            cb,
            encoder,
            _buffers: PhantomData,
        })
    }

    /// One synchronous dispatch: encode, commit, wait.
    pub fn run_once(
        &self,
        kernel: KernelId,
        params: &[u32],
        buffers: &[&DeviceBuffer<'_>],
        threads: usize,
    ) -> Result<(), MetalError> {
        let mut pass = self.begin_pass()?;
        pass.dispatch(kernel, params, buffers, threads);
        pass.run()
    }
}

/// A single command buffer with an open compute encoder. Dispatches are
/// encoded eagerly; [`run`](Self::run) commits and blocks until completion.
/// Dropping without `run` abandons the (uncommitted) work.
pub struct ComputePass<'c, 'b> {
    ctx: &'c MetalContext,
    cb: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    _buffers: PhantomData<&'b ()>,
}

impl<'b> ComputePass<'_, 'b> {
    /// Encode one dispatch of `threads` logical threads (rounded up to full
    /// threadgroups; kernels bounds-check against their params).
    ///
    /// Binding convention: `buffers[i]` at `[[buffer(i)]]`, and `params` (a
    /// flat all-`u32` parameter struct, possibly empty) via `setBytes` at
    /// index `buffers.len()`.
    pub fn dispatch(
        &mut self,
        kernel: KernelId,
        params: &[u32],
        buffers: &[&DeviceBuffer<'b>],
        threads: usize,
    ) {
        self.encoder
            .setComputePipelineState(&self.ctx.pipelines[kernel.index()]);
        for (index, buffer) in buffers.iter().enumerate() {
            // SAFETY: the buffer is a live MTLBuffer; index/offset are in
            // range for the pipeline's argument table.
            unsafe {
                self.encoder
                    .setBuffer_offset_atIndex(Some(buffer.raw()), 0, index);
            }
        }
        if !params.is_empty() {
            let bytes: NonNull<c_void> = NonNull::from(&params[0]).cast();
            // SAFETY: `bytes` points at `params.len() * 4` readable bytes;
            // Metal copies them into the command stream during this call.
            unsafe {
                self.encoder
                    .setBytes_length_atIndex(bytes, size_of_val(params), buffers.len());
            }
        }
        let groups = MTLSize {
            width: threads.div_ceil(THREADGROUP_SIZE).max(1),
            height: 1,
            depth: 1,
        };
        let per_group = MTLSize {
            width: THREADGROUP_SIZE,
            height: 1,
            depth: 1,
        };
        self.encoder
            .dispatchThreadgroups_threadsPerThreadgroup(groups, per_group);
    }

    /// Commit and block until the GPU finishes; surfaces device-side errors.
    pub fn run(self) -> Result<(), MetalError> {
        self.encoder.endEncoding();
        self.cb.commit();
        self.cb.waitUntilCompleted();
        if self.cb.status() != MTLCommandBufferStatus::Completed {
            let reason = self.cb.error().map_or_else(
                || format!("status {:?}", self.cb.status()),
                |e| e.localizedDescription().to_string(),
            );
            return Err(MetalError::Execution(reason));
        }
        Ok(())
    }
}
