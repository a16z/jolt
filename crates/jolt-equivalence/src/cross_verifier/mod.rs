//! Cross-verifier soundness suite (spec
//! `crates/jolt-equivalence/docs/verifier_parity_and_soundness.md`).
//!
//! Reusable framework that runs jolt-core's verifier (`V_core`) and the
//! modular verifier (`V_mod`) on the same proof to detect rejection
//! divergence. The suite quantifies the gap between the two verifiers
//! via the `KnownGapRegistry` and shrinks the registry as the
//! verifier-parity work lands stage by stage.
//!
//! Module layout:
//! - [`categories`] — `Constraint`, `RejectionCategory`, `TamperKind`
//! - [`tamper`] — tamper definitions and `apply_tamper`
//! - [`conversion`] — modular ↔ core proof conversion
//! - [`fixture`] — cached honest fixture (~5–10s amortized)
//! - [`runner`] — dual-verifier runner + outcome reporting
//! - [`registry`] — `KnownGap` list

pub mod categories;
pub mod conversion;
pub mod fixture;
pub mod registry;
pub mod runner;
pub mod tamper;

pub use categories::{Constraint, RejectionCategory, TamperKind};
pub use conversion::{modular_to_core, CoreScaffold};
pub use fixture::{fixture, HonestFixture};
pub use registry::{is_registered, registered_count, KnownGap, KNOWN_GAPS};
pub use runner::{assert_consistent, run_both_verifiers, run_tampered, DualVerifyResult};
pub use tamper::{
    apply_tamper, ConfigField, ConstraintTemplate, DegreeKind, ExpectedResult, IoField,
    TamperLocation, TamperMutation, TamperOutcome, TamperPoint, TAMPER_COVERAGE,
};
