//! Compiler passes for the SNARK protocol IR.
//!
//! - **validate**: Reject malformed protocols
//! - **analyze**: Compute derived properties (topo order, depth, degree)
//! - **stage**: Group vertices into batched sumcheck stages
//! - **emit**: Produce prover schedule + verifier schedule

pub(crate) mod analyze;
pub mod cost;
mod emit;
pub(crate) mod stage;
mod validate;

pub use analyze::IRInfo;
pub use cost::{CompileParams, Cost, Objective, SolverConfig};
pub use validate::Diagnostic;

use crate::ir::Protocol;
use crate::module::Module;
use crate::polynomial_id::PolynomialId;

/// Validate an L0 protocol and compute derived properties.
pub fn analyze(protocol: &Protocol) -> Result<IRInfo, Vec<Diagnostic>> {
    let diagnostics = validate::validate(protocol);
    if !diagnostics.is_empty() {
        return Err(diagnostics);
    }
    Ok(analyze::compute(protocol))
}

/// Compile a protocol into a prover schedule and verifier script.
///
/// `poly_map` maps protocol polynomial indices (from `Protocol::poly()`)
/// to concrete [`PolynomialId`] values. Length must equal
/// `protocol.polynomials.len()`.
pub fn compile(
    protocol: &Protocol,
    params: &CompileParams,
    config: &SolverConfig,
    poly_map: &[PolynomialId],
) -> Result<Module, CompileError> {
    assert_eq!(
        poly_map.len(),
        protocol.polynomials.len(),
        "poly_map length ({}) must match protocol polynomial count ({})",
        poly_map.len(),
        protocol.polynomials.len(),
    );
    let diagnostics = validate::validate(protocol);
    if !diagnostics.is_empty() {
        return Err(CompileError::Validation(diagnostics));
    }
    let info = analyze::compute(protocol);
    let staging = stage::stage(protocol, &info, params, config)?;
    let module = emit::emit(&staging, params, poly_map);

    // Post-emit primitive-form rail. See `crates/jolt-compiler/OPS.md` and
    // `crates/jolt-bench/opt/05-streamlining.md` §O3. Today `is_primitive()`
    // returns `true` for every variant, so this is a no-op. Phases O4 and O5
    // flip variants to `false` as they lower them, at which point this
    // assertion catches any emission site that still uses the old form.
    debug_assert!(
        module
            .prover
            .ops
            .iter()
            .all(crate::module::Op::is_primitive),
        "compiler emitted non-primitive op: {:?}",
        module.prover.ops.iter().find(|op| !op.is_primitive())
    );

    Ok(module)
}

/// Errors from the compilation pipeline.
#[derive(Clone, Debug)]
pub enum CompileError {
    /// Validation failures.
    Validation(Vec<Diagnostic>),
    /// No feasible staging under the given constraints.
    Infeasible { cost: Cost, violated: Vec<String> },
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Validation(diags) => {
                write!(f, "validation errors:")?;
                for d in diags {
                    write!(f, "\n  {d}")?;
                }
                Ok(())
            }
            Self::Infeasible { violated, .. } => {
                write!(f, "infeasible:")?;
                for v in violated {
                    write!(f, "\n  {v}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for CompileError {}
