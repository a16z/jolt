//! Link-time kernel compilation and prover executable.
//!
//! [`link`] compiles a target-agnostic [`Module`] into an [`Executable`] by
//! lowering each [`KernelDef`]'s [`KernelSpec`] to a backend-specific compiled
//! kernel. The resulting `Executable` can be executed by the prover runtime
//! without further compilation.

use jolt_compiler::module::{Module, Op};
use jolt_compiler::PolynomialSpec;
use jolt_field::Field;

use crate::ComputeBackend;

/// A linked schedule ready for execution on backend `B`.
///
/// Produced by [`link`]. Contains the compiled module, flat op sequence,
/// and backend-compiled kernels. The prover runtime walks `ops` sequentially,
/// dispatching compute operations to `B` via the compiled kernels.
pub struct Executable<P: PolynomialSpec, B: ComputeBackend, F: Field> {
    /// The compiled module metadata (poly decls, challenge decls, verifier schedule).
    pub module: Module<P>,
    /// Full op sequence: compute, PCS, and orchestration ops.
    pub ops: Vec<Op<P>>,
    /// Backend-compiled kernels, indexed by compute ops (`SumcheckRound`, `AbsorbRoundPoly`).
    pub kernels: Vec<B::CompiledKernel<F>>,
}

/// Compile a [`Module`] into an [`Executable`] by lowering kernel specs
/// to backend-specific compiled kernels.
///
/// Each [`KernelDef`]'s [`KernelSpec`] is compiled via
/// [`ComputeBackend::compile`], which bakes in the formula, iteration
/// pattern, evaluation grid, and binding order. Challenge values are
/// resolved at execution time from the Fiat-Shamir transcript.
pub fn link<P: PolynomialSpec, B: ComputeBackend, F: Field>(
    module: Module<P>,
    backend: &B,
) -> Executable<P, B, F> {
    let kernels: Vec<B::CompiledKernel<F>> = module
        .prover
        .kernels
        .iter()
        .map(|def| backend.compile(&def.spec))
        .collect();
    let ops = module.prover.ops.clone();
    Executable {
        module,
        ops,
        kernels,
    }
}

impl<P: PolynomialSpec, B: ComputeBackend, F: Field> Executable<P, B, F> {
    #[inline]
    pub fn kernel(&self, idx: usize) -> &B::CompiledKernel<F> {
        &self.kernels[idx]
    }
}
