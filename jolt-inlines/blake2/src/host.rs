//! Host-side implementation and registration.
pub use crate::trace_generator;
use jolt_inlines_common::constants;
use jolt_inlines_common::trace_writer::{write_inline_trace, InlineDescriptor, SequenceInputs};
use tracer::emulator::cpu::Xlen;
use tracer::register_inline;

pub fn init_inlines() -> Result<(), String> {
    register_inline(
        constants::INLINE_OPCODE,
        constants::blake2::FUNCT3,
        constants::blake2::FUNCT7,
        constants::blake2::NAME,
        std::boxed::Box::new(trace_generator::blake2b_inline_sequence_builder),
    )?;

    Ok(())
}

pub fn store_inlines() -> Result<(), String> {
    let inline_info = InlineDescriptor::new(
        constants::blake2::NAME.to_string(),
        constants::INLINE_OPCODE,
        constants::blake2::FUNCT3,
        constants::blake2::FUNCT7,
    );
    let sequence_inputs = SequenceInputs::default();
    let instructions = trace_generator::blake2b_inline_sequence_builder(
        sequence_inputs.address,
        sequence_inputs.is_compressed,
        Xlen::Bit64,
        sequence_inputs.rs1,
        sequence_inputs.rs2,
        sequence_inputs.rs3,
    );
    write_inline_trace(
        "blake2_trace.joltinline",
        &inline_info,
        &sequence_inputs,
        &instructions,
        false,
    )
    .map_err(|e| e.to_string())?;

    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
#[ctor::ctor]
fn auto_register() {
    if let Err(e) = init_inlines() {
        eprintln!("Failed to register BLAKE2 inlines: {e}");
    }

    if let Err(e) = store_inlines() {
        eprintln!("Failed to store BLAKE2 inline traces: {e}");
    }
}
