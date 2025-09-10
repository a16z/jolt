//! Common utilities and constants for Jolt inline implementations
#![cfg_attr(not(feature = "save_trace"), no_std)]

pub mod constants;
#[cfg(feature = "save_trace")]
pub mod trace_writer;
