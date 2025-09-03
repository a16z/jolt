//! BigInt multiplication inline implementation module

#![cfg_attr(not(feature = "host"), no_std)]

pub mod multiplication;
pub use multiplication::*;

#[cfg(feature = "host")]
use jolt_inlines_common::constants;
#[cfg(feature = "host")]
use tracer::register_inline;

#[cfg(feature = "host")]
use jolt_inlines_common::trace_writer::{write_inline_trace, InlineDescriptor, SequenceInputs};
#[cfg(feature = "host")]
use tracer::emulator::cpu::Xlen;

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

#[cfg(feature = "host")]
pub fn store_inlines() -> Result<(), String> {
    let inline_info = InlineDescriptor::new(
        constants::bigint::mul256::NAME.to_string(),
        constants::INLINE_OPCODE,
        constants::bigint::mul256::FUNCT3,
        constants::bigint::mul256::FUNCT7,
    );
    let sequence_inputs = SequenceInputs::default();
    let instructions = trace_generator::bigint_mul_sequence_builder(
        sequence_inputs.address,
        sequence_inputs.is_compressed,
        Xlen::Bit64,
        sequence_inputs.rs1,
        sequence_inputs.rs2,
        sequence_inputs.rd,
    );
    write_inline_trace(
        "bigint_mul256_trace.joltinline",
        &inline_info,
        &sequence_inputs,
        &instructions,
        false,
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

    #[cfg(feature = "host")]
    if let Err(e) = store_inlines() {
        eprintln!("Failed to store BIGINT256_MUL inline traces: {e}");
    }
}
