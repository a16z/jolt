//! Dual-verifier runner.
//!
//! Given an honest modular proof + a tamper, applies the tamper, runs
//! both verifiers (V_mod and V_core via `modular_to_core` conversion),
//! and reports the agreement / disagreement (S5'). Spec §3.8.

use jolt_dory::DoryScheme;
use jolt_field::Fr as NewFr;
use jolt_verifier::JoltProof as ModularJoltProof;

use super::categories::RejectionCategory;
use super::conversion::modular_to_core;
use super::fixture::{verify_with_core, HonestFixture};
use super::registry::is_registered;
use super::tamper::{apply_tamper, ExpectedResult, TamperOutcome, TamperPoint};

/// Result of running both verifiers on the same (potentially tampered)
/// proof. `core_category` and `modular_category` are populated only on
/// rejection.
#[derive(Debug)]
pub struct DualVerifyResult {
    pub core: Result<(), String>,
    pub modular: Result<(), String>,
    pub core_category: Option<RejectionCategory>,
    pub modular_category: Option<RejectionCategory>,
}

impl DualVerifyResult {
    pub fn both_accept(&self) -> bool {
        self.core.is_ok() && self.modular.is_ok()
    }
    pub fn both_reject(&self) -> bool {
        self.core.is_err() && self.modular.is_err()
    }
    pub fn core_rejects_modular_accepts(&self) -> bool {
        self.core.is_err() && self.modular.is_ok()
    }
    pub fn core_accepts_modular_rejects(&self) -> bool {
        self.core.is_ok() && self.modular.is_err()
    }

    pub fn outcome_label(&self) -> &'static str {
        match (self.core.is_ok(), self.modular.is_ok()) {
            (true, true) => "BothAccept",
            (false, false) => "BothReject",
            (false, true) => "CoreRejectsModularAccepts",
            (true, false) => "CoreAcceptsModularRejects",
        }
    }
}

/// Run V_core and V_mod on a proof produced by tampering an honest
/// modular proof. Returns the dual-verifier outcome.
///
/// The conversion uses the fixture's `core_scaffold` to fill structural
/// fields the modular proof doesn't carry; the scaffold's
/// opening_claims come from a HONEST core proof, so any modular eval
/// tamper that survives conversion will only be visible to V_core
/// through downstream sumcheck/PCS checks (the suite intentionally
/// keeps the scaffold honest to isolate the verifier-side gap).
pub fn run_both_verifiers(
    modular_proof: ModularJoltProof<NewFr, DoryScheme>,
    fixture: &HonestFixture,
) -> DualVerifyResult {
    // Modular verify (V_mod).
    let modular = jolt_verifier::verify(
        fixture.modular_verifying_key,
        &modular_proof,
        &fixture.io_hash,
    )
    .map_err(|e| format!("modular: {e}"));

    // Convert to core proof and verify (V_core).
    let core_proof = modular_to_core(&modular_proof, &fixture.core_scaffold);
    let core = verify_with_core(core_proof, fixture);

    let modular_category = modular
        .as_ref()
        .err()
        .map(|e| categorize_modular_error(e.as_str()));
    let core_category = core
        .as_ref()
        .err()
        .map(|e| categorize_core_error(e.as_str()));

    DualVerifyResult {
        core,
        modular,
        core_category,
        modular_category,
    }
}

/// Apply a tamper to a fresh clone of the honest fixture proof, run
/// both verifiers, and return the outcome plus whether the tamper
/// could be applied at all.
pub fn run_tampered(fixture: &HonestFixture, tamper: &TamperPoint) -> Option<DualVerifyResult> {
    let mut proof = fixture.modular_proof.clone();
    match apply_tamper(&mut proof, tamper) {
        TamperOutcome::Applied => Some(run_both_verifiers(proof, fixture)),
        TamperOutcome::Vacuous(_) => None,
    }
}

/// Coarse-grained classification of a modular verifier error string.
/// Best-effort — exact string matching against `JoltError` variants.
fn categorize_modular_error(err: &str) -> RejectionCategory {
    let s = err;
    if s.contains("io_hash") || s.contains("IoHash") {
        RejectionCategory::PreambleMismatch
    } else if s.contains("sumcheck error") {
        // Could be SumcheckRoundPolySum or SumcheckFinalEval; without
        // a structured discriminator we report SumcheckRoundPolySum
        // (the most common cause of T1/T8 detection).
        RejectionCategory::SumcheckRoundPolySum
    } else if s.contains("opening verification") {
        RejectionCategory::OpeningProofInvalid
    } else if s.contains("evaluation check") || s.contains("EvaluationMismatch") {
        RejectionCategory::OpeningClaimEvalMismatch
    } else if s.contains("invalid proof structure") || s.contains("InvalidProof") {
        RejectionCategory::StructuralInvalid
    } else {
        RejectionCategory::Other
    }
}

fn categorize_core_error(err: &str) -> RejectionCategory {
    let s = err;
    if s.contains("preamble") {
        RejectionCategory::PreambleMismatch
    } else if s.contains("stage1")
        || s.contains("stage2")
        || s.contains("stage3")
        || s.contains("stage4")
        || s.contains("stage5")
        || s.contains("stage6")
        || s.contains("stage7")
    {
        RejectionCategory::SumcheckRoundPolySum
    } else if s.contains("stage8") {
        RejectionCategory::OpeningProofInvalid
    } else {
        RejectionCategory::Other
    }
}

/// Assert the dual-verify outcome matches the tamper's expectation,
/// modulo the `KnownGapRegistry` (S5').
///
/// Returns `Ok(())` if the outcome matches; `Err` with a diagnostic
/// otherwise. Hard-fail callers (KGC, T1, T8) propagate the error.
pub fn assert_consistent(tamper: &TamperPoint, result: &DualVerifyResult) -> Result<(), String> {
    let actual = match (result.core.is_ok(), result.modular.is_ok()) {
        (true, true) => ExpectedResult::BothAccept,
        (false, false) => ExpectedResult::BothReject,
        (false, true) => ExpectedResult::CoreRejectsModularAccepts,
        (true, false) => ExpectedResult::CoreAcceptsModularRejects,
    };

    if actual == tamper.expected {
        return Ok(());
    }

    // Allow registered gaps: if the tamper is registered as a known gap,
    // the outcome must be `CoreRejectsModularAccepts`.
    if is_registered(tamper.stage, tamper.kind)
        && actual == ExpectedResult::CoreRejectsModularAccepts
    {
        return Ok(());
    }

    Err(format!(
        "tamper {label} (stage={stage}, kind={kind:?}) expected {expected:?} but got \
         {actual:?}\n  core: {core:?}\n  modular: {modular:?}",
        label = tamper.label,
        stage = tamper.stage,
        kind = tamper.kind,
        expected = tamper.expected,
        actual = actual,
        core = result.core,
        modular = result.modular,
    ))
}
