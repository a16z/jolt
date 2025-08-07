//! SHA256 inline implementation module

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
    // Register SHA256 with funct3=0x00 and funct7=0x00 (matching the SDK's assembly instruction)
    register_inline(
        0x0B,
        0x00,
        0x00,
        "SHA256_INLINE",
        std::boxed::Box::new(exec::sha2_exec),
        std::boxed::Box::new(trace_generator::sha2_virtual_sequence_builder),
    )?;

    // Register SHA256 with funct3=0x01 and funct7=0x00 (matching the SDK's assembly instruction)
    register_inline(
        0x0B,
        0x01,
        0x00,
        "SHA256_INIT_INLINE",
        std::boxed::Box::new(exec::sha2_init_exec),
        std::boxed::Box::new(trace_generator::sha2_init_virtual_sequence_builder),
    )?;
    Ok(())
}

// Automatic registration when the library is loaded
#[cfg(feature = "host")]
#[ctor::ctor]
fn auto_register() {
    if let Err(e) = init_inlines() {
        eprintln!("Failed to register SHA256 inlines: {}", e);
    }
}
