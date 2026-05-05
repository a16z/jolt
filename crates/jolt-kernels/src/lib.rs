//! Coarse CPU kernels used by generated Bolt/Jolt Rust.
//!
//! This crate is intentionally above the primitive protocol crates and below
//! generated code. It owns the temporary coarse CPU ABI while the compiler
//! grows finer compute lowerings.

mod dense;
mod split_eq;

pub mod stage1;
pub mod stage2;
pub mod stage3;
pub mod stage4;
pub mod stage5;
pub mod stage6;
pub mod stage7;
pub mod trace;
