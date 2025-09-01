//! Host-side implementation and registration.
pub use crate::exec;
pub use crate::trace_generator;

#[cfg(feature = "host")]
use jolt_inlines_common::constants;
#[cfg(feature = "host")]
use tracer::register_inline;

#[cfg(feature = "save_trace")]
use jolt_inlines_common::trace_writer::{InlineDescriptor, SequenceInputs, write_inline_trace};
#[cfg(feature = "save_trace")]
use tracer::emulator::cpu::Xlen;

#[cfg(feature = "host")]
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

#[cfg(feature = "save_trace")]
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
        Xlen::Bit64,
        sequence_inputs.rs1,
        sequence_inputs.rs2,
        sequence_inputs.rd,
    );
    write_inline_trace(
        "keccak256_trace.txt",
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
        eprintln!("Failed to register Keccak256 inlines: {e}");
    }
    
    #[cfg(feature = "save_trace")]
    if let Err(e) = store_inlines() {
        eprintln!("Failed to store Keccak256 inline traces: {e}");
    }
}
