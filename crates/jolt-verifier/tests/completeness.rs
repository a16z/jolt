#![expect(
    dead_code,
    reason = "Verifier prover-fixture harness registers fixture metadata used by feature-gated cases."
)]

#[path = "completeness/mod.rs"]
mod completeness;
mod support;

#[test]
fn completeness_case_registry_is_wired() {
    completeness::assert_registry_is_wired();
}
