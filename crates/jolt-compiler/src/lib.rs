pub mod compiler;
pub mod dot;
pub mod formula;
pub mod ir;
pub mod module;

pub use compiler::{
    analyze, compile, CompileError, CompileParams, Cost, Diagnostic, IRInfo, Objective,
    SolverConfig,
};
pub use formula::{BindingOrder, Factor, Formula, ProductTerm};
pub use ir::expr::{Challenge, Expr, Poly, Term};
pub use ir::{Claim, ClaimId, PolyDef, PolyKind, Protocol, PublicPoly, Vertex};
pub use module::{
    ChallengeDecl, ChallengeSource, ClaimFormula, Evaluation, InputBinding, KernelDef, Module, Op,
    PolyDecl, Schedule, VerifierSchedule, VerifierStage,
};
