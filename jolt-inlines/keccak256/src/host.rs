//! Host-side implementation and registration.
pub use crate::exec;
pub use crate::trace_generator;

use jolt_inlines_common::constants;
use tracer::register_inline;

use jolt_inlines_common::trace_writer::{write_inline_trace, InlineDescriptor, SequenceInputs};

pub fn init_inlines() -> Result<(), String> {
    register_inline(
        constants::INLINE_OPCODE,
        constants::keccak256::FUNCT3,
        constants::keccak256::FUNCT7,
        constants::keccak256::NAME,
        std::boxed::Box::new(exec::keccak256_exec),
        std::boxed::Box::new(trace_generator::keccak256_inline_sequence_builder),
    )?;

    Ok(())
}

pub fn store_inlines() -> Result<(), String> {
    let inline_info = InlineDescriptor::new(
        constants::keccak256::NAME.to_string(),
        constants::INLINE_OPCODE,
        constants::keccak256::FUNCT3,
        constants::keccak256::FUNCT7,
    );
    let sequence_inputs = SequenceInputs::default();
    let instructions = trace_generator::keccak256_inline_sequence_builder(
        sequence_inputs.address,
        sequence_inputs.is_compressed,
        sequence_inputs.xlen,
        sequence_inputs.rs1,
        sequence_inputs.rs2,
        sequence_inputs.rs3,
    );
    write_inline_trace(
        "keccak256_trace.joltinline",
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
        tracing::error!("Failed to register Keccak256 inlines: {e}");
    }

    if let Err(e) = store_inlines() {
        eprintln!("Failed to store Keccak256 inline traces: {e}");
    }

    if let Err(e) = store_inlines() {
        eprintln!("Failed to store Keccak256 inline traces: {e}");
    }
}
