#![expect(
    dead_code,
    reason = "Verifier statistical-independence harness shares fixture metadata with feature-gated cases."
)]

#[path = "../../jolt-verifier/tests/statistical_independence/mod.rs"]
mod statistical_independence;
mod support;
