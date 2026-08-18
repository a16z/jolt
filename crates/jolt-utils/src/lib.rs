//! Dependency-light utilities shared across the Jolt workspace.
//!
//! This crate sits at the very bottom of the workspace dependency graph: it
//! must not depend on any other jolt crate, so every crate — jolt-field
//! included — can use it without cycles. Anything that needs a `Field` bound
//! belongs elsewhere.
//!
//! - [`alloc`]: `unsafe_allocate_zero_vec` — zero-init allocation via `alloc_zeroed`
//! - [`math`]: the `Math` trait (`pow2`, `log_2`) and power-of-two log helpers
//! - [`thread`] (feature `parallel`): `drop_in_background_thread`, plus the
//!   deterministic-error index-parallel collection primitives
//!   ([`FirstErrorLatch`], [`par_collect_windows`])

pub mod alloc;
pub mod math;
#[cfg(feature = "parallel")]
pub mod thread;

pub use alloc::unsafe_allocate_zero_vec;
pub use math::{checked_log2_power_of_two, log2_power_of_two, Math};
#[cfg(feature = "parallel")]
pub use thread::{drop_in_background_thread, par_collect_windows, FirstErrorLatch};
