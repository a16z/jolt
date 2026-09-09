pub mod cognitive;
pub mod halstead_bugs;
pub mod lloc;

pub(crate) const PROOF_SYSTEM_CRATE_DIRS: &[&str] = &[
    "crates/jolt-akita",
    "crates/jolt-blindfold",
    "crates/jolt-claims",
    "crates/jolt-crypto",
    "crates/jolt-dory",
    "crates/jolt-field",
    "crates/jolt-host",
    "crates/jolt-kernels",
    "crates/jolt-lookup-tables",
    "crates/jolt-openings",
    "crates/jolt-poly",
    "crates/jolt-program",
    "crates/jolt-prover",
    "crates/jolt-r1cs",
    "crates/jolt-riscv",
    "crates/jolt-sumcheck",
    "crates/jolt-transcript",
    "crates/jolt-verifier",
    "crates/jolt-witness",
];
