#![allow(non_snake_case)]

// Allow `jolt_eval::` paths in macro-generated code within this crate.
extern crate self as jolt_eval;

pub mod agent;
pub mod guests;
pub mod invariant;
pub mod objective;

pub use guests::{
    deserialize_proof, serialize_proof, CommitmentScheme, GuestProgram, JoltDevice, Proof,
    ProofVerifyError, ProverPreprocessing, Serializable, SharedSetup, TestCase,
    VerifierPreprocessing, F, FS, PCS,
};
pub use invariant::{
    CheckError, Invariant, InvariantTargets, InvariantViolation, JoltInvariants, SynthesisTarget,
};
pub use objective::{AbstractObjective, Direction, MeasurementError, Objective};

// Re-exports used by the #[invariant] proc macro generated code.
pub use arbitrary;
pub use rand;

/// Run all provided invariants, returning results keyed by name.
pub fn check_all_invariants(
    invariants: &[JoltInvariants],
    num_random: usize,
) -> std::collections::HashMap<String, Vec<Result<(), InvariantViolation>>> {
    invariants
        .iter()
        .map(|inv| {
            let name = inv.name().to_string();
            let results = inv.run_checks(num_random);
            (name, results)
        })
        .collect()
}

/// Measure all provided objectives, returning results keyed by name.
pub fn measure_all_objectives(
    objectives: &[Objective],
) -> std::collections::HashMap<String, Result<f64, MeasurementError>> {
    objectives
        .iter()
        .map(|obj| {
            let name = obj.name().to_string();
            let result = obj.collect_measurement();
            (name, result)
        })
        .collect()
}
