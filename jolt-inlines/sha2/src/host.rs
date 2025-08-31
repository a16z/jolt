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

#[cfg(all(feature = "host", not(target_arch = "wasm32")))]
#[ctor::ctor]
fn auto_register() {
    if let Err(e) = init_inlines() {
        eprintln!("Failed to register SHA256 inlines: {e}");
    }
}
