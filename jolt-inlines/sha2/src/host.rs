//! Host-side implementation and registration.
pub use crate::exec;
pub use crate::trace_generator;

#[cfg(feature = "host")]
use jolt_inlines_common::constants;
#[cfg(feature = "host")]
use tracer::register_inline;

#[cfg(feature = "save_trace")]
use jolt_inlines_common::trace_writer::{write_inline_trace, InlineDescriptor, SequenceInputs};
#[cfg(feature = "save_trace")]
use tracer::emulator::cpu::Xlen;

#[cfg(feature = "host")]
pub fn init_inlines() -> Result<(), String> {
    register_inline(
        constants::INLINE_OPCODE,
        constants::sha256::default::FUNCT3,
        constants::sha256::default::FUNCT7,
        constants::sha256::default::NAME,
        std::boxed::Box::new(exec::sha2_exec),
        std::boxed::Box::new(trace_generator::sha2_inline_sequence_builder),
    )?;

    register_inline(
        constants::INLINE_OPCODE,
        constants::sha256::init::FUNCT3,
        constants::sha256::init::FUNCT7,
        constants::sha256::init::NAME,
        std::boxed::Box::new(exec::sha2_init_exec),
        std::boxed::Box::new(trace_generator::sha2_init_inline_sequence_builder),
    )?;

    Ok(())
}

#[cfg(feature = "save_trace")]
pub fn store_inlines() -> Result<(), String> {
    // Store SHA256 default inline trace
    let inline_info = InlineDescriptor::new(
        constants::sha256::default::NAME.to_string(),
        constants::INLINE_OPCODE,
        constants::sha256::default::FUNCT3,
        constants::sha256::default::FUNCT7,
    );
    let sequence_inputs = SequenceInputs::default();
    let instructions = trace_generator::sha2_inline_sequence_builder(
        sequence_inputs.address,
        sequence_inputs.is_compressed,
        Xlen::Bit64,
        sequence_inputs.rs1,
        sequence_inputs.rs2,
        sequence_inputs.rd,
    );
    write_inline_trace(
        "sha256_trace.txt",
        &inline_info,
        &sequence_inputs,
        &instructions,
        false, // Don't append for the first one
    )?;

    // Store SHA256 init inline trace (append to the same file)
    let inline_info = InlineDescriptor::new(
        constants::sha256::init::NAME.to_string(),
        constants::INLINE_OPCODE,
        constants::sha256::init::FUNCT3,
        constants::sha256::init::FUNCT7,
    );
    let sequence_inputs = SequenceInputs::default();
    let instructions = trace_generator::sha2_init_inline_sequence_builder(
        sequence_inputs.address,
        sequence_inputs.is_compressed,
        Xlen::Bit64,
        sequence_inputs.rs1,
        sequence_inputs.rs2,
        sequence_inputs.rd,
    );
    write_inline_trace(
        "sha256_trace.txt",
        &inline_info,
        &sequence_inputs,
        &instructions,
        true, // Append to the existing file
    )?;

    Ok(())
}

#[cfg(all(feature = "host", not(target_arch = "wasm32")))]
#[ctor::ctor]
fn auto_register() {
    if let Err(e) = init_inlines() {
        eprintln!("Failed to register SHA256 inlines: {e}");
    }

    #[cfg(feature = "save_trace")]
    if let Err(e) = store_inlines() {
        eprintln!("Failed to store SHA256 inline traces: {e}");
    }
}
