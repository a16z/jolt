#![expect(
    dead_code,
    reason = "Completeness oracle scaffolding shares registry metadata with focused test modules."
)]

#[path = "completeness/mod.rs"]
mod completeness;
mod support;

#[test]
fn completeness_case_registry_is_wired() {
    completeness::assert_registry_is_wired();
}
