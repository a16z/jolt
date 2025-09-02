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
        constants::blake2::FUNCT3,
        constants::blake2::FUNCT7,
        constants::blake2::NAME,
        std::boxed::Box::new(exec::blake2b_exec),
        std::boxed::Box::new(trace_generator::blake2b_inline_sequence_builder),
    )?;

    Ok(())
}

#[cfg(feature = "save_trace")]
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
        "blake2_trace.txt",
        &inline_info,
        &sequence_inputs,
        &instructions,
        false,
    )?;

    Ok(())
}

#[cfg(all(feature = "host", not(target_arch = "wasm32")))]
#[ctor::ctor]
fn auto_register() {
    if let Err(e) = init_inlines() {
        eprintln!("Failed to register Blake2 inlines: {e}");
    }

    #[cfg(feature = "save_trace")]
    if let Err(e) = store_inlines() {
        eprintln!("Failed to store Blake2 inline traces: {e}");
    }
}