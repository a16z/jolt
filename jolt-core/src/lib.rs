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

/// Register all inline implementations compiled in via the `host` feature.
/// Idempotent — safe to call multiple times (uses `Once` internally).
#[cfg(feature = "host")]
pub fn register_all_inlines() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        jolt_inlines_sha2::init_inlines().expect("Failed to register sha2 inlines");
        jolt_inlines_keccak256::init_inlines().expect("Failed to register keccak256 inlines");
    });
}
