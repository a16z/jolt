#![allow(non_snake_case)]

// Allow `jolt_eval::` paths in macro-generated code within this crate.
extern crate self as jolt_eval;

pub mod agent;
pub mod guests;
pub mod invariant;
pub mod objective;

pub use guests::{JoltDevice, ProofVerifyError, TestCase};
pub use invariant::{
    CheckError, Invariant, InvariantTargets, InvariantViolation, JoltInvariants, SynthesisTarget,
};
pub use objective::{AbstractObjective, Direction, MeasurementError, Objective};

// Re-exports used by the #[invariant] proc macro generated code.
pub use arbitrary;
pub use rand;
