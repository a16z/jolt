//! Link-time kernel compilation and prover executable.
//!
//! [`link`] compiles a target-agnostic [`Module`] into an [`Executable`] by
//! lowering each [`KernelDef`]'s [`KernelSpec`] to a backend-specific compiled
//! kernel. The resulting `Executable` can be executed by the prover runtime
//! without further compilation.

use jolt_compiler::module::{Module, Op};
use jolt_field::Field;

use crate::ComputeBackend;

/// Dual-path validation mode for the backend's [`fuse_ops`](ComputeBackend::fuse_ops)
/// rewrite pass.
///
/// Toggled via the `JOLT_FUSE_DEBUG=1` environment variable. When on, the
/// linker keeps the pre-fusion op stream as `Executable::shadow_ops` and the
/// runtime asserts per-`BatchRoundEvaluate` equivalence between the fused
/// path and a shadow replay of the pre-fusion per-instance reduces. Off in
/// release (no shadow stream, zero overhead).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FuseDebugMode {
    /// Release path: runtime executes the fused stream only.
    #[default]
    Off,
    /// Debug path: linker keeps the pre-fusion stream; runtime shadow-replays
    /// per-instance reduces at each `BatchRoundEvaluate` and asserts eval
    /// equality across every active instance.
    On,
}

impl FuseDebugMode {
    /// Read the mode from the `JOLT_FUSE_DEBUG` environment variable.
    ///
    /// `JOLT_FUSE_DEBUG=1` → [`Self::On`]. Unset or any other value → [`Self::Off`].
    pub fn from_env() -> Self {
        match std::env::var("JOLT_FUSE_DEBUG").as_deref() {
            Ok("1") => Self::On,
            _ => Self::Off,
        }
    }

    #[inline]
    pub fn is_on(self) -> bool {
        matches!(self, Self::On)
    }
}

/// Switch between the legacy reduce ops and the unified `Op::Reduce` path.
///
/// Toggled via the `JOLT_UNIFIED_REDUCE` environment variable. Phase C bridge:
/// - [`Self::Off`] (default): runtime executes the legacy reduce ops
///   (`SumcheckRound`, `InstanceReduce`, `InstanceSegmentedReduce`,
///   `BatchRoundEvaluate`); `Op::Reduce` is a no-op.
/// - [`Self::On`]: runtime executes `Op::Reduce` only; legacy reduce-side work
///   is skipped (`SumcheckRound` still performs its bind, but not the reduce).
/// - [`Self::Shadow`]: like `On`, but additionally runs
///   [`per_instance_reference_reduce`](crate::per_instance_reference_reduce)
///   in lockstep and asserts byte-identical output before writing back —
///   catches state-wiring regressions in Phase C and fused-kernel divergence
///   in Phase E.
///
/// Phase D deletes this toggle entirely along with the legacy ops.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReduceDebugMode {
    /// Legacy path only (Phase B behavior).
    #[default]
    Off,
    /// Unified `Op::Reduce` path only.
    On,
    /// Unified path with reference-impl shadow assertions.
    Shadow,
}

impl ReduceDebugMode {
    /// Read the mode from the `JOLT_UNIFIED_REDUCE` environment variable.
    ///
    /// `=1` → [`Self::On`], `=shadow` → [`Self::Shadow`], unset/other →
    /// [`Self::Off`].
    pub fn from_env() -> Self {
        match std::env::var("JOLT_UNIFIED_REDUCE").as_deref() {
            Ok("1") => Self::On,
            Ok("shadow") => Self::Shadow,
            _ => Self::Off,
        }
    }

    /// Runtime should execute `Op::Reduce` rather than the legacy ops.
    #[inline]
    pub fn unified_active(self) -> bool {
        matches!(self, Self::On | Self::Shadow)
    }

    /// Runtime should also run the reference shadow for cross-check.
    #[inline]
    pub fn shadow_active(self) -> bool {
        matches!(self, Self::Shadow)
    }
}

/// A linked schedule ready for execution on backend `B`.
///
/// Produced by [`link`]. Contains the compiled module, flat op sequence,
/// and backend-compiled kernels. The prover runtime walks `ops` sequentially,
/// dispatching compute operations to `B` via the compiled kernels.
pub struct Executable<B: ComputeBackend, F: Field> {
    /// The compiled module metadata (poly decls, challenge decls, verifier schedule).
    pub module: Module,
    /// Full op sequence executed by the runtime. If the backend's
    /// [`fuse_ops`](ComputeBackend::fuse_ops) returned `Some`, this is the
    /// fused stream; otherwise it is the compiler's original stream.
    pub ops: Vec<Op>,
    /// Pre-fusion op stream, populated only when [`FuseDebugMode::On`] AND
    /// the backend returned a non-identity fuse rewrite. Used by the runtime
    /// to shadow-replay per-instance reduces and assert equivalence against
    /// each fused `BatchRoundEvaluate`.
    pub shadow_ops: Option<Vec<Op>>,
    /// Unified-reduce dispatch mode, resolved from `JOLT_UNIFIED_REDUCE` at
    /// link time. Runtime reads this to decide whether legacy reduce ops
    /// (`InstanceReduce`, `InstanceSegmentedReduce`, `BatchRoundEvaluate`,
    /// reduce-side of `SumcheckRound`) fire or are skipped in favor of the
    /// unified `Op::Reduce` specs.
    pub reduce_mode: ReduceDebugMode,
    /// Backend-compiled kernels, indexed by compute ops (`SumcheckRound`, `AbsorbRoundPoly`).
    pub kernels: Vec<B::CompiledKernel<F>>,
}

impl<B: ComputeBackend, F: Field> Executable<B, F> {
    /// Returns `true` iff the linker installed a shadow stream for
    /// dual-path validation.
    #[inline]
    pub fn has_fuse_debug(&self) -> bool {
        self.shadow_ops.is_some()
    }
}

/// Compile a [`Module`] into an [`Executable`] by lowering kernel specs
/// to backend-specific compiled kernels.
///
/// Each [`KernelDef`]'s [`KernelSpec`] is compiled via
/// [`ComputeBackend::compile`], which bakes in the formula, iteration
/// pattern, evaluation grid, and binding order. Challenge values are
/// resolved at execution time from the Fiat-Shamir transcript.
///
/// The backend's [`fuse_ops`](ComputeBackend::fuse_ops) is invoked once on
/// the compiler's op stream. If it returns `Some`, the rewritten stream is
/// stored on the executable; in [`FuseDebugMode::On`], the pre-fusion stream
/// is also kept as a shadow for dual-path validation.
pub fn link<B: ComputeBackend, F: Field>(module: Module, backend: &B) -> Executable<B, F> {
    let kernels: Vec<B::CompiledKernel<F>> = module
        .prover
        .kernels
        .iter()
        .map(|def| backend.compile(&def.spec))
        .collect();
    let raw_ops = module.prover.ops.clone();

    let (ops, shadow_ops) = match backend.fuse_ops(&raw_ops) {
        Some(fused) => match FuseDebugMode::from_env() {
            FuseDebugMode::On => (fused, Some(raw_ops)),
            FuseDebugMode::Off => (fused, None),
        },
        None => (raw_ops, None),
    };

    Executable {
        module,
        ops,
        shadow_ops,
        reduce_mode: ReduceDebugMode::from_env(),
        kernels,
    }
}
