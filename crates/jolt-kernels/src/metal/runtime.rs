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
#[cfg(feature = "bench-utils")]
use std::time::Duration;
use std::time::Instant;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBarrierScope, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineDescriptor, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLPipelineOption, MTLSize,
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
    IncPrepare,
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
    G2FixedBaseTable,
    OpeningFoldDense,
    OpeningFoldOneHot,
    IrrPhaseScan,
    IrrSuffixScan,
    IrrEqOuter,
    IrrReduce,
    IrrCycleInit,
    IrrCycleInitFused,
    IrrCycleRound,
    SuffixProbe,
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
    OuterT1Lazy,
    OuterAzbzLazy,
    OuterRound,
    OuterClaimsLazy,
    ProductT1,
    ProductLr,
    IcrInit,
    IcrRound,
    RamRwMessage,
    RamRwBind,
    RegRwBuild,
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
    MillerFlyLines,
    MillerFlyFold,
    RegistersValRound,
}

impl KernelId {
    pub const ALL: [Self; 76] = [
        Self::Noop,
        Self::FrMul,
        Self::FrAdd,
        Self::FrSub,
        Self::FrPow2k,
        Self::FrBind,
        Self::FrBind4,
        Self::FrBindEval,
        Self::IncPrepare,
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
        Self::G2FixedBaseTable,
        Self::OpeningFoldDense,
        Self::OpeningFoldOneHot,
        Self::IrrPhaseScan,
        Self::IrrSuffixScan,
        Self::IrrEqOuter,
        Self::IrrReduce,
        Self::IrrCycleInit,
        Self::IrrCycleInitFused,
        Self::IrrCycleRound,
        Self::SuffixProbe,
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
        Self::OuterT1Lazy,
        Self::OuterAzbzLazy,
        Self::OuterRound,
        Self::OuterClaimsLazy,
        Self::ProductT1,
        Self::ProductLr,
        Self::IcrInit,
        Self::IcrRound,
        Self::RamRwMessage,
        Self::RamRwBind,
        Self::RegRwBuild,
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
        Self::MillerFlyLines,
        Self::MillerFlyFold,
        Self::RegistersValRound,
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
            Self::IncPrepare => "jk_inc_prepare",
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
            Self::G2FixedBaseTable => "jk_g2_fixed_base_table",
            Self::OpeningFoldDense => "jk_opening_fold_dense",
            Self::OpeningFoldOneHot => "jk_opening_fold_onehot",
            Self::IrrPhaseScan => "jk_irr_phase_scan",
            Self::IrrSuffixScan => "jk_irr_suffix_scan",
            Self::IrrEqOuter => "jk_irr_eq_outer",
            Self::IrrReduce => "jk_irr_reduce",
            Self::IrrCycleInit => "jk_irr_cycle_init",
            Self::IrrCycleInitFused => "jk_irr_cycle_init_fused",
            Self::IrrCycleRound => "jk_irr_cycle_round",
            Self::SuffixProbe => "jk_suffix_probe",
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
            Self::OuterT1Lazy => "jk_outer_t1_lazy",
            Self::OuterAzbzLazy => "jk_outer_azbz_lazy",
            Self::OuterRound => "jk_outer_round",
            Self::OuterClaimsLazy => "jk_outer_claims_lazy",
            Self::ProductT1 => "jk_product_t1",
            Self::ProductLr => "jk_product_lr",
            Self::IcrInit => "jk_icr_init",
            Self::IcrRound => "jk_icr_round",
            Self::RamRwMessage => "jk_ram_rw_message",
            Self::RamRwBind => "jk_ram_rw_bind",
            Self::RegRwBuild => "jk_reg_rw_build",
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
            Self::MillerFlyLines => "jk_miller_fly_lines",
            Self::MillerFlyFold => "jk_miller_fly_fold",
            Self::RegistersValRound => "jk_registers_val_round",
        }
    }

    const fn index(self) -> usize {
        self as usize
    }

    /// The pairing/tower family: every kernel whose per-thread live set
    /// (an Fq12 accumulator and up) is past the register budget the AGX
    /// compiler will hold at the device's 1024-thread pipeline cap.
    const fn is_pairing_family(self) -> bool {
        matches!(
            self,
            Self::Fq6Mul
                | Self::Fq6Sqr
                | Self::Fq12Mul
                | Self::Fq12Sqr
                | Self::Fq12Mul034
                | Self::MillerTable
                | Self::MillerFly
                | Self::MillerFlyLines
                | Self::MillerFlyFold
        )
    }

    /// Compile-time `maxTotalThreadsPerThreadgroup` cap for this kernel's
    /// pipeline, or `None` for the device default (1024). Declaring a
    /// lower cap on the pipeline DESCRIPTOR is the one public lever that
    /// invites the AGX compiler to trade occupancy for per-thread
    /// registers instead of spilling; dispatch width adapts via
    /// [`ComputePass::dispatch`].
    ///
    /// **Default: cap 64 for the stage-8 fly kernels ONLY; everything
    /// else uncapped.** The trade is real but regime- and context-bound
    /// (W4-fly, W4 bundle):
    ///
    /// - Fly at 8192 pairs (µs/pair): caps 1024/256 codegen-inert at
    ///   3.47, 128 → 3.08, 64 → 3.05 (plateau), 32 → 3.07. Hook walls:
    ///   2^13 −8.6%, 2^17 −2.8% — registers buy back serial ladder
    ///   latency; at saturation full occupancy already hides the spill.
    /// - `jk_miller_fly` runs solo-dominant in stage-8 reduce rounds →
    ///   capped (with `MillerFlyLines`/`Fold`, same lane).
    ///
    /// `JOLT_METAL_PAIRING_TG_CAP` overrides the WHOLE pairing family for
    /// experiments (`0` = uncapped everywhere), read once at context
    /// build.
    fn thread_cap(self) -> Option<usize> {
        const FLY_TG_CAP: usize = 64;
        #[derive(Clone, Copy)]
        enum CapOverride {
            /// Env unset: per-kernel defaults below.
            Defaults,
            /// `0`: uncapped everywhere (the W3-baseline ablation arm).
            Uncapped,
            /// `N`: the whole pairing family capped at N.
            Family(usize),
        }
        if !self.is_pairing_family() {
            return None;
        }
        static OVERRIDE: OnceLock<CapOverride> = OnceLock::new();
        let env_cap = *OVERRIDE.get_or_init(|| {
            match std::env::var("JOLT_METAL_PAIRING_TG_CAP")
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
            {
                None => CapOverride::Defaults,
                Some(0) => CapOverride::Uncapped,
                Some(cap) => CapOverride::Family(cap.next_power_of_two().clamp(32, 1024)),
            }
        });
        const TABLE_TG_CAP: usize = 32;
        match env_cap {
            CapOverride::Family(cap) => Some(cap),
            CapOverride::Uncapped => None,
            CapOverride::Defaults
                if matches!(
                    self,
                    Self::MillerFly | Self::MillerFlyLines | Self::MillerFlyFold
                ) =>
            {
                Some(FLY_TG_CAP)
            }
            // W5-st8: the isolated −24% at cap 32 (W4-fly handoff) holds on
            // the table's own dispatch shapes, and the W4 §5 in-pipeline
            // family-cap-32 commit probe measured wall parity (1.207 s both
            // arms) — cap 32 is co-run-safe where cap 64 inverted. The
            // kernel runs only on the tier-2 fallback path (fly-commit gate
            // declined, mid-size traces).
            CapOverride::Defaults if matches!(self, Self::MillerTable) => Some(TABLE_TG_CAP),
            CapOverride::Defaults => None,
        }
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
    /// Default dispatch width per kernel: [`THREADGROUP_SIZE`], shrunk to
    /// the pipeline's declared maximum for register-capped pipelines
    /// (cached so the encode path stays free of per-dispatch objc calls).
    dispatch_widths: Vec<usize>,
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

    fn library_source() -> String {
        format!(
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
        )
    }

    fn new() -> Result<Self, MetalError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::NoDevice)?;
        let source = Self::library_source();
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(&source), None)
            .map_err(|e| MetalError::Compile(e.localizedDescription().to_string()))?;

        let mut pipelines = Vec::with_capacity(KernelId::ALL.len());
        let mut dispatch_widths = Vec::with_capacity(KernelId::ALL.len());
        for kernel in KernelId::ALL {
            let function = library
                .newFunctionWithName(&NSString::from_str(kernel.name()))
                .ok_or(MetalError::MissingFunction(kernel.name()))?;
            let pipeline = if let Some(cap) = kernel.thread_cap() {
                let descriptor = MTLComputePipelineDescriptor::new();
                descriptor.setComputeFunction(Some(&function));
                descriptor.setMaxTotalThreadsPerThreadgroup(cap);
                device
                    .newComputePipelineStateWithDescriptor_options_reflection_error(
                        &descriptor,
                        MTLPipelineOption::None,
                        None,
                    )
                    .map_err(|e| MetalError::Pipeline {
                        name: kernel.name(),
                        reason: e.localizedDescription().to_string(),
                    })?
            } else {
                device
                    .newComputePipelineStateWithFunction_error(&function)
                    .map_err(|e| MetalError::Pipeline {
                        name: kernel.name(),
                        reason: e.localizedDescription().to_string(),
                    })?
            };
            let max = pipeline.maxTotalThreadsPerThreadgroup();
            // Capped pipelines dispatch at their (smaller) declared width,
            // so they only need a full simdgroup; everything else keeps the
            // fixed-width fail-closed check.
            let need = if kernel.thread_cap().is_some() {
                32
            } else {
                THREADGROUP_SIZE
            };
            if max < need {
                return Err(MetalError::ThreadgroupTooSmall {
                    name: kernel.name(),
                    max,
                    need,
                });
            }
            dispatch_widths.push(THREADGROUP_SIZE.min(max));
            pipelines.push(pipeline);
        }

        let queue = device.newCommandQueue().ok_or(MetalError::NoCommandQueue)?;
        Ok(Self {
            device,
            queue,
            pipelines,
            dispatch_widths,
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

    /// Compile a bench-only kernel variant: the full production library
    /// source plus `extra_source` appended, so variants can reuse every
    /// shader-side helper (`fq_mul`, `g1_xyzz_madd`, …). Attribution rig
    /// for kernel experiments — never a production dispatch path.
    #[cfg(feature = "bench-utils")]
    pub fn compile_variant(
        &self,
        extra_source: &str,
        entry: &str,
    ) -> Result<VariantPipeline, MetalError> {
        let source = format!("{}\n{extra_source}", Self::library_source());
        let library = self
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(&source), None)
            .map_err(|e| MetalError::Compile(e.localizedDescription().to_string()))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(entry))
            .ok_or(MetalError::Compile(format!(
                "missing variant entry {entry}"
            )))?;
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| MetalError::Pipeline {
                name: "variant",
                reason: e.localizedDescription().to_string(),
            })?;
        Ok(VariantPipeline { pipeline })
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

    /// As [`begin_pass`](Self::begin_pass), on a fresh private queue:
    /// Metal schedules a queue's command buffers in order, so a pass whose
    /// SCHEDULING is expensive (wiring large fresh no-copy buffers) would
    /// stall every later pass on the shared queue. A side queue wires
    /// concurrently instead. The command buffer retains its queue, so the
    /// throwaway queue outlives the pass.
    pub fn begin_pass_side<'b>(&self) -> Result<ComputePass<'_, 'b>, MetalError> {
        let queue = self
            .device
            .newCommandQueue()
            .ok_or(MetalError::NoCommandQueue)?;
        let cb = queue.commandBuffer().ok_or(MetalError::NoCommandBuffer)?;
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

/// Bench-only ad-hoc pipeline from [`MetalContext::compile_variant`].
#[cfg(feature = "bench-utils")]
pub struct VariantPipeline {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

#[cfg(feature = "bench-utils")]
impl VariantPipeline {
    /// `(maxTotalThreadsPerThreadgroup, threadExecutionWidth)` — the same
    /// register-pressure proxy as [`MetalContext::pipeline_stats`].
    pub fn stats(&self) -> (usize, usize) {
        (
            self.pipeline.maxTotalThreadsPerThreadgroup(),
            self.pipeline.threadExecutionWidth(),
        )
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
        // Capped pipelines (register/occupancy trade) shrink the width to
        // their declared maximum; everyone else keeps the fixed width.
        let width = self.ctx.dispatch_widths[kernel.index()];
        self.dispatch_width(kernel, params, buffers, threads, width);
    }

    /// Encode one dispatch of a bench-only variant pipeline.
    #[cfg(feature = "bench-utils")]
    pub fn dispatch_variant(
        &mut self,
        variant: &VariantPipeline,
        params: &[u32],
        buffers: &[&DeviceBuffer<'b>],
        threads: usize,
        width: usize,
    ) {
        assert!(width.is_power_of_two());
        assert!(width <= variant.pipeline.maxTotalThreadsPerThreadgroup());
        self.encoder.setComputePipelineState(&variant.pipeline);
        for (index, buffer) in buffers.iter().enumerate() {
            // SAFETY: live MTLBuffer; index/offset in range for the table.
            unsafe {
                self.encoder
                    .setBuffer_offset_atIndex(Some(buffer.raw()), 0, index);
            }
        }
        if !params.is_empty() {
            let bytes: NonNull<c_void> = NonNull::from(&params[0]).cast();
            // SAFETY: `bytes` spans `params.len() * 4` readable bytes; Metal
            // copies them into the command stream during this call.
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
        #[cfg(any(test, feature = "bench-utils"))]
        super::testing::note_device_dispatch();
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

    /// Make buffer writes visible before a dependent dispatch in this pass.
    pub fn buffer_barrier(&mut self) {
        self.encoder
            .memoryBarrierWithScope(MTLBarrierScope::Buffers);
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

    /// Wait and return the command buffer's device execution window.
    #[cfg(feature = "bench-utils")]
    pub fn wait_timed(self) -> Result<Duration, MetalError> {
        wait_completed(&self.cb, self.trace.as_ref())?;
        Ok(Duration::from_secs_f64(
            self.cb.GPUEndTime() - self.cb.GPUStartTime(),
        ))
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
            "[jk-cb] commit=+{:.6}s gstart={:.6} gpu_us={gpu_us:.0} blocked_us={blocked_us:.0} disp={} {}",
            committed_at.as_secs_f64(),
            cb.GPUStartTime(),
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
