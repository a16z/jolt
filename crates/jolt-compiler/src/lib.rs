pub mod compiler;
pub mod dot;
pub mod formula;
pub mod ir;
pub mod kernel_spec;
pub mod module;
pub mod params;
pub mod polynomial_id;

pub use compiler::{
    analyze, compile, CompileError, CompileParams, Cost, Diagnostic, IRInfo, Objective,
    SolverConfig,
};
pub use formula::{BindingOrder, Factor, Formula, ProductTerm};
pub use ir::expr::{Challenge, Expr, Poly, Term};
pub use ir::{Claim, ClaimId, Density, PolyDef, PolyKind, Protocol, PublicPoly, Vertex};
pub use kernel_spec::{Iteration, KernelSpec};
pub use module::{
    ChallengeDecl, ChallengeSource, ClaimFactor, ClaimFormula, ClaimTerm, Evaluation, InputBinding,
    KernelDef, Module, Op, PointNormalization, PolyDecl, PolyId, R1CSMatrix, Schedule,
    SumcheckInstance, TranscriptTag, UniskipVerify, VerifierOp, VerifierSchedule,
    VerifierStageIndex,
};
pub use params::ModuleParams;
pub use polynomial_id::PolynomialId;
