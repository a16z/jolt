//! Link-time kernel compilation and prover executable.
//!
//! [`link`] compiles a target-agnostic [`Module`] into an [`Executable`] by
//! lowering each [`KernelDef`] to a backend-specific compiled kernel.
//! The resulting `Executable` can be executed by the prover runtime (in
//! `jolt-zkvm`) without further compilation.

use jolt_compiler::module::{Module, Op};
use jolt_field::Field;

use crate::ComputeBackend;

/// A linked schedule ready for execution on backend `B`.
///
/// Produced by [`link`]. Contains the compiled module, flat op sequence,
/// and backend-compiled kernels. The prover runtime walks `ops` sequentially,
/// dispatching compute operations to `B` via the compiled kernels.
///
/// Kernels capture formula *structure* only — challenge values are passed
/// at dispatch time during execution.
pub struct Executable<B: ComputeBackend, F: Field> {
    /// The compiled module metadata (poly decls, challenge decls, verifier schedule).
    pub module: Module,
    /// Flat op sequence from the prover schedule.
    pub ops: Vec<Op>,
    /// Backend-compiled kernels, indexed by `Op::SumcheckRound { kernel, .. }`.
    pub kernels: Vec<B::CompiledKernel<F>>,
}

/// Compile a [`Module`] into an [`Executable`] by lowering kernel definitions
/// to backend-specific compiled kernels.
///
/// Each [`KernelDef`] in the module's prover schedule is compiled via
/// [`ComputeBackend::compile_kernel`], which captures the formula shape
/// without baking challenge values. Challenges are resolved at execution
/// time from the Fiat-Shamir transcript.
pub fn link<B: ComputeBackend, F: Field>(module: Module, backend: &B) -> Executable<B, F> {
    let kernels: Vec<B::CompiledKernel<F>> = module
        .prover
        .kernels
        .iter()
        .map(|def| backend.compile_kernel(&def.formula))
        .collect();
    let ops = module.prover.ops.clone();
    Executable {
        module,
        ops,
        kernels,
    }
}

impl<B: ComputeBackend, F: Field> Executable<B, F> {
    #[inline]
    pub fn kernel(&self, idx: usize) -> &B::CompiledKernel<F> {
        &self.kernels[idx]
    }
}
