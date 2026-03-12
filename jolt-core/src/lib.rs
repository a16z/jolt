#![allow(
    clippy::assertions_on_result_states,
    clippy::from_over_into,
    clippy::len_without_is_empty,
    clippy::needless_range_loop,
    clippy::new_without_default,
    clippy::too_long_first_doc_paragraph,
    long_running_const_eval,
    non_snake_case,
    type_alias_bounds
)]
#[cfg(feature = "host")]
pub mod host;

pub mod curve;
pub mod field;
pub mod guest;
pub mod msm;
pub mod poly;
pub mod subprotocols;
pub mod transcripts;
pub mod utils;
pub mod zkvm;
pub use ark_bn254;

// Re-export AdviceTape type for use in generated code
pub use tracer::emulator::cpu::AdviceTape;

/// Register all inline implementations linked into the binary.
/// Discovers plugins via `inventory` — external inline crates are included automatically.
/// Idempotent — safe to call multiple times.
#[cfg(feature = "host")]
pub fn register_all_inlines() {
    jolt_inlines_sdk::host::register_all_inlines();
}
