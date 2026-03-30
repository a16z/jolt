pub mod compiler;
pub mod dot;
pub mod formula;
pub mod ir;
pub mod output;

pub use compiler::{
    analyze, compile, CompileError, CompileParams, Cost, Diagnostic, IRInfo, Objective,
    SolverConfig,
};
pub use formula::{BindingOrder, CompositionFormula, Factor, ProductTerm};
pub use ir::expr::{Challenge, Expr, Poly, Term};
pub use ir::{Claim, ClaimId, PolyDef, PolyKind, Protocol, PublicPoly, Vertex};
pub use output::{
    ChallengeSource, ChallengeSpec, ClaimFormula, CompilerOutput, EqMode, EvalSpec, KernelSpec,
    PolySpec, ProverSchedule, ProverStep, VerifierScript, VerifierStage,
};
