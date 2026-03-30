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

/// Validate an L0 protocol and compute derived properties.
pub fn analyze(protocol: &Protocol) -> Result<IRInfo, Vec<Diagnostic>> {
    let diagnostics = validate::validate(protocol);
    if !diagnostics.is_empty() {
        return Err(diagnostics);
    }
    Ok(analyze::compute(protocol))
}

/// Compile a protocol into a prover schedule and verifier script.
pub fn compile(
    protocol: &Protocol,
    params: &CompileParams,
    config: &SolverConfig,
) -> Result<Module, CompileError> {
    let diagnostics = validate::validate(protocol);
    if !diagnostics.is_empty() {
        return Err(CompileError::Validation(diagnostics));
    }
    let info = analyze::compute(protocol);
    let staging = stage::stage(protocol, &info, params, config)?;
    Ok(emit::emit(&staging, params))
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
