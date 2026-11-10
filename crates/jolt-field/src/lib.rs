//! `jolt-field` is the former name of [`jolt_algebra`]. This crate re-exports
//! it unchanged and forwards its features, so dependents that name
//! `jolt-field`, including Akita through its Git pin, keep building until they
//! switch to `jolt-algebra`.
#![forbid(unsafe_code)]

pub use jolt_algebra::*;
