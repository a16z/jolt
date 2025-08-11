//! Host-side implementation and registration.
pub use crate::exec;
pub use crate::trace_generator;

use tracer::register_inline;

pub fn init_inlines() -> Result<(), String> {
    register_inline(
        0x0B,
        0x00,
        0x01,
        "KECCAK256_INLINE",
        std::boxed::Box::new(exec::keccak256_exec),
        std::boxed::Box::new(trace_generator::keccak256_virtual_sequence_builder),
    )?;
    Ok(())
}

#[ctor::ctor]
fn auto_register() {
    if let Err(e) = init_inlines() {
        eprintln!("Failed to register Keccak256 inlines: {e}");
    }
}
