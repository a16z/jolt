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
use std::time::Instant;

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

/// Per-dispatch selector capacity of `jk_opening_fold_onehot`
/// (`JK_OPENING_MAX_SEL` shader-side): each selector costs one full-width
/// `Fr` accumulator of thread registers, so wider one-hot families split
/// into several dispatches over the same column buffer.
pub const OPENING_MAX_SEL: usize = 8;

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
    FrBind4,
    FrBindEval,
    IncRound,
    TablePairsRound,
    HammingRound,
    G1SegSum,
    G1CombineRows,
    G1ScalarMulAdd,
    G1ProjectiveMulAdd,
    DoryMsmHist,
    DoryMsmOffsets,
    DoryMsmScatter,
    G1DoryMsmOwner,
    G1DoryMsmWindowFold,
    G2ScalarMulAdd,
    G2ProjectiveMulAdd,
    G2DoryMsmOwner,
    G2DoryMsmWindowFold,
    G2FixedBaseMul,
    OpeningFoldDense,
    OpeningFoldOneHot,
    IrrPhaseScan,
    IrrSuffixScan,
    IrrReduce,
    IrrCycleInit,
    IrrCycleRound,
    SuffixProbe,
    RaMaterialize,
    BoolLazyRound,
    BoolDenseRound,
    BoolAdoptRound,
    RavLazyRound,
    RavAdoptRound,
    RavDenseRound,
    BytecodeInit,
    BytecodeLazyRound,
    BytecodeAdopt,
    BytecodeDenseRound,
    BytecodeOffsetProbe,
    InstrInputQ0,
    InstrInputBindNative,
    InstrInputRound,
    OuterT1,
    OuterAzbz,
    OuterRound,
    OuterClaims,
    ProductT1,
    ProductLr,
    IcrInit,
    IcrRound,
    RamRwMessage,
    RamRwBind,
    RegRwMessageIdx,
    RegRwMessageF,
    RegRwBindIdx,
    RegRwBindIdxToF,
    RegRwBindF,
    Fq6Mul,
    Fq6Sqr,
    Fq12Mul,
    Fq12Sqr,
    Fq12Mul034,
    MillerTable,
    MillerFly,
    MillerFlyIndexed,
}

impl KernelId {
    pub const ALL: [Self; 71] = [
        Self::Noop,
        Self::FrMul,
        Self::FrAdd,
        Self::FrSub,
        Self::FrPow2k,
        Self::FrBind,
        Self::FrBind4,
        Self::FrBindEval,
        Self::IncRound,
        Self::TablePairsRound,
        Self::HammingRound,
        Self::G1SegSum,
        Self::G1CombineRows,
        Self::G1ScalarMulAdd,
        Self::G1ProjectiveMulAdd,
        Self::DoryMsmHist,
        Self::DoryMsmOffsets,
        Self::DoryMsmScatter,
        Self::G1DoryMsmOwner,
        Self::G1DoryMsmWindowFold,
        Self::G2ScalarMulAdd,
        Self::G2ProjectiveMulAdd,
        Self::G2DoryMsmOwner,
        Self::G2DoryMsmWindowFold,
        Self::G2FixedBaseMul,
        Self::OpeningFoldDense,
        Self::OpeningFoldOneHot,
        Self::IrrPhaseScan,
        Self::IrrSuffixScan,
        Self::IrrReduce,
        Self::IrrCycleInit,
        Self::IrrCycleRound,
        Self::SuffixProbe,
        Self::RaMaterialize,
        Self::BoolLazyRound,
        Self::BoolDenseRound,
        Self::BoolAdoptRound,
        Self::RavLazyRound,
        Self::RavAdoptRound,
        Self::RavDenseRound,
        Self::BytecodeInit,
        Self::BytecodeLazyRound,
        Self::BytecodeAdopt,
        Self::BytecodeDenseRound,
        Self::BytecodeOffsetProbe,
        Self::InstrInputQ0,
        Self::InstrInputBindNative,
        Self::InstrInputRound,
        Self::OuterT1,
        Self::OuterAzbz,
        Self::OuterRound,
        Self::OuterClaims,
        Self::ProductT1,
        Self::ProductLr,
        Self::IcrInit,
        Self::IcrRound,
        Self::RamRwMessage,
        Self::RamRwBind,
        Self::RegRwMessageIdx,
        Self::RegRwMessageF,
        Self::RegRwBindIdx,
        Self::RegRwBindIdxToF,
        Self::RegRwBindF,
        Self::Fq6Mul,
        Self::Fq6Sqr,
        Self::Fq12Mul,
        Self::Fq12Sqr,
        Self::Fq12Mul034,
        Self::MillerTable,
        Self::MillerFly,
        Self::MillerFlyIndexed,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Noop => "jk_noop",
            Self::FrMul => "jk_fr_mul",
            Self::FrAdd => "jk_fr_add",
            Self::FrSub => "jk_fr_sub",
            Self::FrPow2k => "jk_fr_pow2k",
            Self::FrBind => "jk_fr_bind",
            Self::FrBind4 => "jk_fr_bind4",
            Self::FrBindEval => "jk_fr_bind_eval",
            Self::IncRound => "jk_inc_round",
            Self::TablePairsRound => "jk_table_pairs_round",
            Self::HammingRound => "jk_hamming_round",
            Self::G1SegSum => "jk_g1_seg_sum",
            Self::G1CombineRows => "jk_g1_combine_rows",
            Self::G1ScalarMulAdd => "jk_g1_scalar_mul_add",
            Self::G1ProjectiveMulAdd => "jk_g1_projective_mul_add",
            Self::DoryMsmHist => "jk_dory_msm_hist",
            Self::DoryMsmOffsets => "jk_dory_msm_offsets",
            Self::DoryMsmScatter => "jk_dory_msm_scatter",
            Self::G1DoryMsmOwner => "jk_g1_dory_msm_owner",
            Self::G1DoryMsmWindowFold => "jk_g1_dory_msm_window_fold",
            Self::G2ScalarMulAdd => "jk_g2_scalar_mul_add",
            Self::G2ProjectiveMulAdd => "jk_g2_projective_mul_add",
            Self::G2DoryMsmOwner => "jk_g2_dory_msm_owner",
            Self::G2DoryMsmWindowFold => "jk_g2_dory_msm_window_fold",
            Self::G2FixedBaseMul => "jk_g2_fixed_base_mul",
            Self::OpeningFoldDense => "jk_opening_fold_dense",
            Self::OpeningFoldOneHot => "jk_opening_fold_onehot",
            Self::IrrPhaseScan => "jk_irr_phase_scan",
            Self::IrrSuffixScan => "jk_irr_suffix_scan",
            Self::IrrReduce => "jk_irr_reduce",
            Self::IrrCycleInit => "jk_irr_cycle_init",
            Self::IrrCycleRound => "jk_irr_cycle_round",
            Self::SuffixProbe => "jk_suffix_probe",
            Self::RaMaterialize => "jk_ra_materialize",
            Self::BoolLazyRound => "jk_bool_lazy_round",
            Self::BoolDenseRound => "jk_bool_dense_round",
            Self::BoolAdoptRound => "jk_bool_adopt_round",
            Self::RavLazyRound => "jk_rav_lazy_round",
            Self::RavAdoptRound => "jk_rav_adopt_round",
            Self::RavDenseRound => "jk_rav_dense_round",
            Self::BytecodeInit => "jk_bytecode_init",
            Self::BytecodeLazyRound => "jk_bytecode_lazy_round",
            Self::BytecodeAdopt => "jk_bytecode_adopt",
            Self::BytecodeDenseRound => "jk_bytecode_dense_round",
            Self::BytecodeOffsetProbe => "jk_bytecode_offset_probe",
            Self::InstrInputQ0 => "jk_instr_input_q0",
            Self::InstrInputBindNative => "jk_instr_input_bind_native",
            Self::InstrInputRound => "jk_instr_input_round",
            Self::OuterT1 => "jk_outer_t1",
            Self::OuterAzbz => "jk_outer_azbz",
            Self::OuterRound => "jk_outer_round",
            Self::OuterClaims => "jk_outer_claims",
            Self::ProductT1 => "jk_product_t1",
            Self::ProductLr => "jk_product_lr",
            Self::IcrInit => "jk_icr_init",
            Self::IcrRound => "jk_icr_round",
            Self::RamRwMessage => "jk_ram_rw_message",
            Self::RamRwBind => "jk_ram_rw_bind",
            Self::RegRwMessageIdx => "jk_reg_rw_message_idx",
            Self::RegRwMessageF => "jk_reg_rw_message_f",
            Self::RegRwBindIdx => "jk_reg_rw_bind_idx",
            Self::RegRwBindIdxToF => "jk_reg_rw_bind_idx_to_f",
            Self::RegRwBindF => "jk_reg_rw_bind_f",
            Self::Fq6Mul => "jk_fq6_mul",
            Self::Fq6Sqr => "jk_fq6_sqr",
            Self::Fq12Mul => "jk_fq12_mul",
            Self::Fq12Sqr => "jk_fq12_sqr",
            Self::Fq12Mul034 => "jk_fq12_mul034",
            Self::MillerTable => "jk_miller_table",
            Self::MillerFly => "jk_miller_fly",
            Self::MillerFlyIndexed => "jk_miller_fly_indexed",
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

/// `JOLT_METAL_CB_TRACE=1` prints one stderr line per committed command
/// buffer: commit time (relative to the first traced CB), device execution
/// window from `GPUStartTime`/`GPUEndTime`, CPU blocked-wait time, and the
/// dispatch mix. The dispatch-batching audit tool; free when unset.
fn cb_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("JOLT_METAL_CB_TRACE").is_some_and(|v| v != "0"))
}

fn trace_epoch() -> Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

/// Dispatch mix of one traced command buffer, in encode order.
struct CbTrace {
    /// (kernel, logical threads) per dispatch.
    dispatches: Vec<(KernelId, usize)>,
}

impl CbTrace {
    /// Run-length-collapsed dispatch summary: `FrBind×3(2048)` = three
    /// consecutive FrBind dispatches, 2048 threads max among them.
    fn summary(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        let mut runs = self.dispatches.iter().peekable();
        while let Some(&(kernel, threads)) = runs.next() {
            let mut count = 1usize;
            let mut max_threads = threads;
            while let Some(&&(next, t)) = runs.peek() {
                if next != kernel {
                    break;
                }
                count += 1;
                max_threads = max_threads.max(t);
                let _ = runs.next();
            }
            if !out.is_empty() {
                out.push(' ');
            }
            let _ = write!(out, "{:?}×{count}({max_threads})", kernel);
        }
        out
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
            "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
            field::constants_preamble(),
            super::miller::pairing_preamble(),
            include_str!("shaders/fr.metal"),
            include_str!("shaders/kernels.metal"),
            include_str!("shaders/g1.metal"),
            include_str!("shaders/g2.metal"),
            include_str!("shaders/fq12.metal"),
            include_str!("shaders/instruction.metal"),
            include_str!("shaders/ra_lazy.metal"),
            include_str!("shaders/bytecode_read_raf.metal"),
            include_str!("shaders/spartan.metal"),
            include_str!("shaders/ram_read_write.metal"),
            include_str!("shaders/registers_read_write.metal"),
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

    /// Compiler-derived pipeline limits `(max_total_threads_per_threadgroup,
    /// thread_execution_width)` — the public occupancy proxy: the compiler
    /// lowers the first below the device's 1024 cap as per-thread register
    /// footprint grows, so it doubles as an indirect register-pressure
    /// reading (Metal exposes no direct register count).
    pub fn pipeline_stats(&self, kernel: KernelId) -> (usize, usize) {
        let pipeline = &self.pipelines[kernel.index()];
        (
            pipeline.maxTotalThreadsPerThreadgroup(),
            pipeline.threadExecutionWidth(),
        )
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
            trace: cb_trace_enabled().then(|| CbTrace {
                dispatches: Vec::new(),
            }),
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
    trace: Option<CbTrace>,
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
        self.dispatch_width(kernel, params, buffers, threads, THREADGROUP_SIZE);
    }

    /// Encode one dispatch with a kernel-specific threadgroup width.
    pub fn dispatch_width(
        &mut self,
        kernel: KernelId,
        params: &[u32],
        buffers: &[&DeviceBuffer<'b>],
        threads: usize,
        width: usize,
    ) {
        assert!(width.is_power_of_two());
        assert!(width <= self.ctx.pipelines[kernel.index()].maxTotalThreadsPerThreadgroup());
        if let Some(trace) = &mut self.trace {
            trace.dispatches.push((kernel, threads));
        }
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
            width: threads.div_ceil(width).max(1),
            height: 1,
            depth: 1,
        };
        let per_group = MTLSize {
            width,
            height: 1,
            depth: 1,
        };
        self.encoder
            .dispatchThreadgroups_threadsPerThreadgroup(groups, per_group);
    }

    /// Commit and block until the GPU finishes; surfaces device-side errors.
    pub fn run(self) -> Result<(), MetalError> {
        self.commit().wait()
    }

    /// Commit without blocking: the GPU starts executing while the caller
    /// keeps the CPU busy; [`PendingPass::wait`] collects completion. The
    /// pending pass extends the dispatched buffers' borrows, so backing
    /// memory stays alive until the wait.
    pub fn commit(self) -> PendingPass<'b> {
        self.encoder.endEncoding();
        let committed_at = self.trace.as_ref().map(|_| {
            let epoch = trace_epoch();
            Instant::now().duration_since(epoch)
        });
        self.cb.commit();
        PendingPass {
            cb: self.cb,
            trace: self.trace.zip(committed_at),
            _buffers: PhantomData,
        }
    }
}

/// A committed, in-flight command buffer. Dropping without
/// [`wait`](Self::wait) does not cancel the GPU work — callers must wait
/// before reading results.
pub struct PendingPass<'b> {
    cb: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    trace: Option<(CbTrace, std::time::Duration)>,
    _buffers: PhantomData<&'b ()>,
}

impl PendingPass<'_> {
    /// Block until the GPU finishes; surfaces device-side errors.
    pub fn wait(self) -> Result<(), MetalError> {
        wait_completed(&self.cb, self.trace.as_ref())
    }

    /// Erase the dispatched buffers' borrows, so the pass can stay in
    /// flight across two `&mut self` calls on the struct that owns the
    /// buffers (the two-phase round contract).
    ///
    /// # Safety
    ///
    /// The caller must uphold what the erased `'b` borrows proved: every
    /// dispatched buffer's backing memory stays alive — and is neither read
    /// (kernel-written buffers) nor written (any buffer) by the host —
    /// until [`DetachedPass::wait`] returns or the detached pass drops.
    pub unsafe fn detach(self) -> DetachedPass {
        DetachedPass {
            cb: self.cb,
            trace: self.trace,
            waited: false,
        }
    }
}

/// Block until `cb` finishes; surfaces device-side errors and emits the
/// env-gated `[jk-cb]` audit line. `blocked_us` measures from this call, so
/// a command buffer that already finished (overlapped execution) reports a
/// near-zero blocked time against its full `gpu_us` window.
#[expect(clippy::print_stderr, reason = "env-gated audit trace")]
fn wait_completed(
    cb: &ProtocolObject<dyn MTLCommandBuffer>,
    trace: Option<&(CbTrace, std::time::Duration)>,
) -> Result<(), MetalError> {
    let wait_start = trace.map(|_| Instant::now());
    cb.waitUntilCompleted();
    if cb.status() != MTLCommandBufferStatus::Completed {
        let reason = cb.error().map_or_else(
            || format!("status {:?}", cb.status()),
            |e| e.localizedDescription().to_string(),
        );
        return Err(MetalError::Execution(reason));
    }
    if let Some((trace, committed_at)) = trace {
        // GPUStartTime/GPUEndTime are device timestamps in seconds on a
        // shared mach timebase; their difference is the CB's execution
        // window (includes any queue wait ahead of it).
        let gpu_us = (cb.GPUEndTime() - cb.GPUStartTime()) * 1e6;
        let blocked_us = wait_start.map_or(0.0, |w| w.elapsed().as_secs_f64() * 1e6);
        eprintln!(
            "[jk-cb] commit=+{:.6}s gpu_us={gpu_us:.0} blocked_us={blocked_us:.0} disp={} {}",
            committed_at.as_secs_f64(),
            trace.dispatches.len(),
            trace.summary(),
        );
    }
    Ok(())
}

/// A committed command buffer released from [`PendingPass`]'s borrow
/// tracking (see [`PendingPass::detach`]). Dropping without
/// [`wait`](Self::wait) blocks until the GPU finishes: with the borrows
/// erased, nothing else proves the dispatched buffers' backing memory
/// outlives the in-flight work, so the drop path must not let it free under
/// a running command buffer.
pub struct DetachedPass {
    cb: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    trace: Option<(CbTrace, std::time::Duration)>,
    waited: bool,
}

impl DetachedPass {
    /// Block until the GPU finishes; surfaces device-side errors.
    pub fn wait(mut self) -> Result<(), MetalError> {
        self.waited = true;
        wait_completed(&self.cb, self.trace.take().as_ref())
    }
}

impl Drop for DetachedPass {
    fn drop(&mut self) {
        if !self.waited {
            self.cb.waitUntilCompleted();
        }
    }
}
