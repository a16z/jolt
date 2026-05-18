#![expect(
    dead_code,
    reason = "Incremental verifier harness skeleton registers future stage-gated cases."
)]

#[path = "completeness/mod.rs"]
mod completeness;
mod support;

#[test]
fn completeness_case_registry_is_wired() {
    completeness::assert_registry_is_wired();
}
