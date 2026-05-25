#![expect(
    dead_code,
    reason = "Verifier statistical-independence harness shares fixture metadata with feature-gated cases."
)]

#[path = "statistical_independence/mod.rs"]
mod statistical_independence;
mod support;
