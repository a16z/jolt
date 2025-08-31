//! BigInt multiplication inline implementation module

#![cfg_attr(not(feature = "host"), no_std)]

pub mod multiplication;
pub use multiplication::*;

#[cfg(feature = "host")]
use jolt_inlines_common::constants;
#[cfg(feature = "host")]
use tracer::register_inline;

// Initialize and register inlines
#[cfg(feature = "host")]
pub fn init_inlines() -> Result<(), String> {
    // Register 256-bit Int multiplication
    register_inline(
        constants::INLINE_OPCODE,
        constants::bigint::mul256::FUNCT3,
        constants::bigint::mul256::FUNCT7,
        constants::bigint::mul256::NAME,
        std::boxed::Box::new(exec::bigint_mul_exec),
        std::boxed::Box::new(trace_generator::bigint_mul_sequence_builder),
    )?;
    Ok(())
}

// Automatic registration when the library is loaded
#[cfg(all(feature = "host", not(target_arch = "wasm32")))]
#[ctor::ctor]
fn auto_register() {
    if let Err(e) = init_inlines() {
        eprintln!("Failed to register BIGINT256_MUL inlines: {e}");
    }
}
