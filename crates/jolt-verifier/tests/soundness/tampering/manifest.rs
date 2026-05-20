use crate::support::tamper_manifest::{
    all_targets, checked_now_without_active_coverage, clear_claim_leaf_paths, manifest_paths,
    proof_field_paths, target_names_are_unique, TamperCoverage,
};

#[test]
fn tamper_manifest_target_names_are_unique() {
    assert!(target_names_are_unique());
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
fn inactive_unlocked_tamper_targets_are_documented() {
    let undocumented = checked_now_without_active_coverage()
        .into_iter()
        .filter(|target| target.reason.is_empty())
        .collect::<Vec<_>>();

    assert!(
        undocumented.is_empty(),
        "unlocked tamper targets without active coverage need a reason: {undocumented:?}"
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
