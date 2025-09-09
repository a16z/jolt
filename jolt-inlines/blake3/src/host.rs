//! Host-side implementation and registration.
pub use crate::sequence_builder;
use crate::{BLAKE3_FUNCT3, BLAKE3_FUNCT7, BLAKE3_NAME, INLINE_OPCODE};
use tracer::emulator::cpu::Xlen;
use tracer::register_inline;
use tracer::utils::inline_sequence_writer::{
    write_inline_trace, AppendMode, InlineDescriptor, SequenceInputs,
};

pub fn init_inlines() -> Result<(), String> {
    register_inline(
        INLINE_OPCODE,
        BLAKE3_FUNCT3,
        BLAKE3_FUNCT7,
        BLAKE3_NAME,
        std::boxed::Box::new(sequence_builder::blake3_inline_sequence_builder),
    )?;

    Ok(())
}

pub fn store_inlines() -> Result<(), String> {
    let inline_info = InlineDescriptor::new(
        BLAKE3_NAME.to_string(),
        INLINE_OPCODE,
        BLAKE3_FUNCT3,
        BLAKE3_FUNCT7,
    );
    let sequence_inputs = SequenceInputs::default();
    let instructions = sequence_builder::blake3_inline_sequence_builder(
        sequence_inputs.address,
        sequence_inputs.is_compressed,
        Xlen::Bit64,
        sequence_inputs.rs1,
        sequence_inputs.rs2,
        sequence_inputs.rs3,
    );
    write_inline_trace(
        "blake3_trace.joltinline",
        &inline_info,
        &sequence_inputs,
        &instructions,
        AppendMode::Overwrite,
    )
    .map_err(|e| e.to_string())?;

    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
#[ctor::ctor]
fn auto_register() {
    if let Err(e) = init_inlines() {
        eprintln!("Failed to register BLAKE3 inlines: {e}");
    }

    if let Err(e) = store_inlines() {
        eprintln!("Failed to store BLAKE3 inline traces: {e}");
    }
}
