//! jolt-verifier's fixture-driven suites, linked as one binary so the prover
//! that generates their fixtures is compiled and linked once rather than per
//! suite. `fs_attacks` and `fs_obligations` stay separate targets: the
//! Fiat-Shamir workflow builds each alone with `--test`.

#![expect(
    dead_code,
    reason = "the shared support module is only partially used per feature configuration"
)]

mod completeness;
mod generate_fuzz_fixture;
mod soundness;
mod statistical_independence;
mod support;
