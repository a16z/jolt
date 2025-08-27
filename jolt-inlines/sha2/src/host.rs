//! Host-side implementation and registration.
pub use crate::exec;
pub use crate::trace_generator;

use tracer::register_inline;

pub fn init_inlines() -> Result<(), String> {
    register_inline(
        0x0B,
        0x00,
        0x00,
        "SHA256_INLINE",
        std::boxed::Box::new(exec::sha2_exec),
        std::boxed::Box::new(trace_generator::sha2_inline_sequence_builder),
    )?;
    register_inline(
        0x0B,
        0x01,
        0x00,
        "SHA256_INIT_INLINE",
        std::boxed::Box::new(exec::sha2_init_exec),
        std::boxed::Box::new(trace_generator::sha2_init_inline_sequence_builder),
    )?;
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
#[ctor::ctor]
fn auto_register() {
    if let Err(e) = init_inlines() {
        eprintln!("Failed to register SHA256 inlines: {e}");
    }
}
