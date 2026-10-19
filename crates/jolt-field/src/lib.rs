//! `jolt-field` is the former name of [`jolt_algebra`]. This crate re-exports
//! it unchanged, with every feature forwarded, so dependents (including the
//! pinned Akita, which names `jolt-field` by Git source) keep compiling until
//! they switch to `jolt-algebra`.
#![forbid(unsafe_code)]

pub use jolt_algebra::*;
