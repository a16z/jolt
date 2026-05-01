#![cfg_attr(not(feature = "host"), no_std)]

pub mod multiplication;
pub use multiplication::*;

#[cfg(feature = "host")]
jolt_inlines_sdk::register_inlines! {
    trace_file: "bigint_mul256_trace.joltinline",
    ops: [multiplication::sequence_builder::BigintMul256],
}
