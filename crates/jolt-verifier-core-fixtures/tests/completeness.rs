#![expect(
    dead_code,
    reason = "Verifier compatibility harness registers fixture metadata used by feature-gated cases."
)]

#[path = "../../jolt-verifier/tests/completeness/mod.rs"]
mod completeness;
mod support;

#[test]
fn completeness_case_registry_is_wired() {
    completeness::assert_registry_is_wired();
}
