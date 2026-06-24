#![expect(
    dead_code,
    reason = "Verifier fixture harness registers fixture metadata used by feature-gated cases."
)]

#[path = "soundness/mod.rs"]
mod soundness;
mod support;

#[test]
fn soundness_case_registry_is_wired() {
    soundness::assert_registry_is_wired();
}
