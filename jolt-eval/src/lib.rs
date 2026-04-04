#![allow(non_snake_case)]

// Allow `jolt_eval::` paths in macro-generated code within this crate.
extern crate self as jolt_eval;

// Force the linker to keep inline instruction registrations from these
// crates. Without this, inventory::submit! symbols get dead-stripped
// and the tracer panics with "No inline registered for opcode=...".
extern crate jolt_inlines_secp256k1;
extern crate jolt_inlines_sha2;

pub mod agent;
pub mod guests;
pub mod invariant;
pub mod objective;

pub use guests::{GuestConfig, GuestProgram, JoltDevice, ProofVerifyError};
pub use invariant::{
    CheckError, Invariant, InvariantTargets, InvariantViolation, JoltInvariants, SynthesisTarget,
};
pub use objective::objective_fn::ObjectiveFunction;
pub use objective::{
    MeasurementError, Objective, OptimizationObjective, PerformanceObjective,
    StaticAnalysisObjective,
};

// Re-exports used by the #[invariant] proc macro generated code.
pub use arbitrary;
pub use rand;
