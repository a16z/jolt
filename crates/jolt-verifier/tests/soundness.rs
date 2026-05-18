#![expect(
    dead_code,
    reason = "Incremental verifier harness skeleton registers future stage-gated cases."
)]

#[path = "soundness/mod.rs"]
mod soundness;
mod support;

#[test]
fn soundness_case_registry_is_wired() {
    soundness::assert_registry_is_wired();
}
