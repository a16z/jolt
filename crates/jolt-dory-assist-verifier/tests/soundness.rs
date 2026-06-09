#![expect(
    dead_code,
    reason = "Soundness oracle scaffolding shares registry metadata with focused tamper modules."
)]

#[path = "soundness/mod.rs"]
mod soundness;
mod support;

#[test]
fn soundness_case_registry_is_wired() {
    soundness::assert_registry_is_wired();
}
