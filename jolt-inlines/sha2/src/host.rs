//! Host-side implementation and registration.
pub use crate::sequence_builder;

use jolt_inlines_common::constants;
use tracer::register_inline;

use jolt_inlines_common::trace_writer::{
    write_inline_trace, AppendMode, InlineDescriptor, SequenceInputs,
};

pub fn init_inlines() -> Result<(), String> {
    register_inline(
        constants::INLINE_OPCODE,
        constants::sha256::default::FUNCT3,
        constants::sha256::default::FUNCT7,
        constants::sha256::default::NAME,
        std::boxed::Box::new(sequence_builder::sha2_inline_sequence_builder),
    )?;

    register_inline(
        constants::INLINE_OPCODE,
        constants::sha256::init::FUNCT3,
        constants::sha256::init::FUNCT7,
        constants::sha256::init::NAME,
        std::boxed::Box::new(sequence_builder::sha2_init_inline_sequence_builder),
    )?;

    Ok(())
}

pub fn store_inlines() -> Result<(), String> {
    // Store SHA256 default inline trace
    let inline_info = InlineDescriptor::new(
        constants::sha256::default::NAME.to_string(),
        constants::INLINE_OPCODE,
        constants::sha256::default::FUNCT3,
        constants::sha256::default::FUNCT7,
    );
    let sequence_inputs = SequenceInputs::default();
    let instructions = sequence_builder::sha2_inline_sequence_builder(
        sequence_inputs.address,
        sequence_inputs.is_compressed,
        sequence_inputs.xlen,
        sequence_inputs.rs1,
        sequence_inputs.rs2,
        sequence_inputs.rs3,
    );
    write_inline_trace(
        "sha256_trace.joltinline",
        &inline_info,
        &sequence_inputs,
        &instructions,
        AppendMode::Overwrite,
    )
    .map_err(|e| e.to_string())?;

    // Store SHA256 init inline trace (append to the same file)
    let inline_info = InlineDescriptor::new(
        constants::sha256::init::NAME.to_string(),
        constants::INLINE_OPCODE,
        constants::sha256::init::FUNCT3,
        constants::sha256::init::FUNCT7,
    );
    let sequence_inputs = SequenceInputs::default();
    let instructions = sequence_builder::sha2_init_inline_sequence_builder(
        sequence_inputs.address,
        sequence_inputs.is_compressed,
        sequence_inputs.xlen,
        sequence_inputs.rs1,
        sequence_inputs.rs2,
        sequence_inputs.rs3,
    );
    write_inline_trace(
        "sha256_trace.joltinline",
        &inline_info,
        &sequence_inputs,
        &instructions,
        AppendMode::Append,
    )
    .map_err(|e| e.to_string())?;

    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
#[ctor::ctor]
fn auto_register() {
    if let Err(e) = init_inlines() {
        tracing::error!("Failed to register SHA256 inlines: {e}");
    }

    if let Err(e) = store_inlines() {
        eprintln!("Failed to store SHA256 inline traces: {e}");
    }
}
