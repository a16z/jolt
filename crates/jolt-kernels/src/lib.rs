//! Coarse CPU kernels used by generated Bolt/Jolt Rust.
//!
//! This crate is intentionally above the primitive protocol crates and below
//! generated code. It owns the temporary coarse CPU ABI while the compiler
//! grows finer compute lowerings.

pub mod stage1;
