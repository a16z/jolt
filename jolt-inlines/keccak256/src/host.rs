//! Host-side implementation and registration.
pub use crate::exec;
pub use crate::trace_generator;

#[cfg(feature = "host")]
use jolt_inlines_common::constants;
#[cfg(feature = "host")]
use tracer::register_inline;

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

#[cfg(all(feature = "host", not(target_arch = "wasm32")))]
#[ctor::ctor]
fn auto_register() {
    if let Err(e) = init_inlines() {
        eprintln!("Failed to register Keccak256 inlines: {e}");
    }
}
