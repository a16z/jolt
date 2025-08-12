//! BigInt multiplication inline implementation module

#![cfg_attr(not(feature = "host"), no_std)]

pub mod sdk;
pub use sdk::*;

// Only include inline registration when compiling for host
#[cfg(feature = "host")]
pub mod exec;
#[cfg(feature = "host")]
pub mod trace_generator;

#[cfg(feature = "host")]
use tracer::register_inline;

// Initialize and register inlines
#[cfg(feature = "host")]
pub fn init_inlines() -> Result<(), String> {
    // Register 256-bit Int multiplication with funct3=0x00 and funct7=0x01
    register_inline(
        0x0B,
        0x00,
        0x01,
        "BIGINT256_MUL",
        std::boxed::Box::new(exec::bigint_mul_exec),
        std::boxed::Box::new(trace_generator::bigint_mul_sequence_builder),
    )?;
    Ok(())
}

// Automatic registration when the library is loaded
#[cfg(feature = "host")]
#[ctor::ctor]
fn auto_register() {
    if let Err(e) = init_inlines() {
        eprintln!("Failed to register BIGINT256_MUL inlines: {e}");
    }
}
