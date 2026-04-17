pub mod builder;
pub mod checkpoint_lowering;
pub mod compiler;
pub mod descriptor;
pub mod dot;
pub mod formula;
pub mod ir;
pub mod kernel_spec;
pub mod module;
pub mod params;
pub mod polynomial_id;
pub mod prefix_mle_lowering;

pub use compiler::{
    analyze, compile, CompileError, CompileParams, Cost, Diagnostic, IRInfo, Objective,
    SolverConfig,
};
pub use descriptor::{PolySource, PolynomialDescriptor, R1csColumn, StorageHint, WitnessSlot};
pub use formula::{BindingOrder, Factor, Formula, ProductTerm};
pub use ir::expr::{Challenge, Expr, Poly, Term};
pub use ir::{Claim, ClaimId, Density, PolyDef, PolyKind, Protocol, PublicPoly, Vertex};
pub use kernel_spec::{GruenHint, Iteration, KernelSpec, LinComboQ};
pub use module::{
    BatchIdx, ChallengeDecl, ChallengeIdx, ChallengeSource, ClaimFactor, ClaimFormula, ClaimTerm,
    CombineEntry, Comparison, DomainSeparator, EvalMode, Evaluation, InputBinding, InstanceConfig,
    InstanceIdx, IntBitOp, KernelDef, Module, Op, PointNormalization, PolyDecl, PrefixMleFormula,
    PrefixMleRule, R1CSMatrix, RemainingTest, RoundPolyEncoding, Schedule, SumcheckInstance,
    UniskipVerify, VerifierOp, VerifierSchedule, VerifierStageIndex,
};
pub use params::ModuleParams;
pub use polynomial_id::PolynomialId;
