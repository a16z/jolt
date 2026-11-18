//! jolt-prover's integration tests, linked as one binary so the prover and
//! its dependencies are compiled and linked once rather than per file.
//!
//! A suite is a module declared here: a file in this directory that no `mod`
//! line names is not compiled, and any other `tests/*.rs` file is built as a
//! separate binary that compiles the prover again.

#[cfg(feature = "prover-fixtures")]
mod support;

mod akita_e2e;
mod akita_field_inline_e2e;
mod backend_contract;
mod e2e_matrix;
mod engine_twins;
mod field_inline_e2e;
mod field_inline_limbs;
mod misaligned_memory;
mod profiling_smoke;
mod zk_e2e;
