#[cfg(not(feature = "akita"))]
use std::path::Path;

use crate::support::tamper_manifest::{
    all_targets, clear_claim_leaf_paths, manifest_paths, proof_field_paths,
    target_names_are_unique, verifier_owned_targets_without_active_coverage, TamperCoverage,
};

#[test]
fn tamper_manifest_target_names_are_unique() {
    assert!(target_names_are_unique());
}

/// Closes the Active ⇒ test direction: `assert_verifier_fixture_tamper_rejects`
/// proves an exercised target is Active, but nothing else stops a target from
/// being flipped Active (inflating the enforced tamper ratio) without a test.
/// Every tamper test names its target by string literal, so requiring each
/// Active name to appear in the tampering test sources makes the ratio honest;
/// deleting a test or its target reference fails here.
///
/// Standard-mode only: the akita suite tampers its targets structurally
/// (destructured field mutation in tampering/akita.rs) without naming them,
/// so the name-literal convention this audit enforces does not apply to the
/// akita-gated target arrays.
#[cfg(not(feature = "akita"))]
#[test]
#[expect(
    clippy::expect_used,
    reason = "manifest audits should fail loudly if test sources are unreadable"
)]
fn active_tamper_targets_are_referenced_by_tampering_tests() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/soundness/tampering");
    let mut sources = String::new();
    for entry in std::fs::read_dir(&dir).expect("tampering test directory is readable") {
        let path = entry.expect("tampering test directory entry").path();
        if path.extension().is_some_and(|ext| ext == "rs")
            && path.file_name().is_some_and(|name| name != "manifest.rs")
        {
            sources.push_str(&std::fs::read_to_string(&path).expect("tampering test source"));
        }
    }

    let unreferenced = all_targets()
        .into_iter()
        .filter(|target| target.coverage == TamperCoverage::Active)
        .filter(|target| !sources.contains(&format!("\"{}\"", target.name)))
        .map(|target| target.name)
        .collect::<Vec<_>>();

    assert!(
        unreferenced.is_empty(),
        "Active tamper targets with no referencing tamper test: {unreferenced:?}"
    );
}

#[test]
fn tamper_manifest_covers_clear_claim_fields() {
    let manifest_paths = manifest_paths();
    let missing = clear_claim_leaf_paths()
        .into_iter()
        .filter(|path| !manifest_paths.contains(path.as_str()))
        .collect::<Vec<_>>();

    assert!(
        missing.is_empty(),
        "clear claim fields missing from tamper manifest: {missing:?}"
    );
}

#[test]
fn tamper_manifest_covers_top_level_proof_fields() {
    let manifest_paths = manifest_paths();
    let missing = proof_field_paths()
        .iter()
        .copied()
        .filter(|path| !manifest_paths.contains(path))
        .collect::<Vec<_>>();

    assert!(
        missing.is_empty(),
        "top-level proof fields missing from tamper manifest: {missing:?}"
    );
}

#[test]
fn verifier_owned_inactive_tamper_targets_are_documented() {
    let undocumented = verifier_owned_targets_without_active_coverage()
        .into_iter()
        .filter(|target| target.reason.is_empty())
        .collect::<Vec<_>>();

    assert!(
        undocumented.is_empty(),
        "verifier-owned tamper targets without active coverage need a reason: {undocumented:?}"
    );
}

#[test]
fn deferred_tamper_targets_are_documented() {
    let undocumented = all_targets()
        .into_iter()
        .filter(|target| target.coverage != TamperCoverage::Active)
        .filter(|target| target.reason.is_empty())
        .collect::<Vec<_>>();

    assert!(
        undocumented.is_empty(),
        "deferred or ignored tamper targets need a reason: {undocumented:?}"
    );
}
