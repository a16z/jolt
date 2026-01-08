//! Host-side implementation and registration.
pub use crate::sequence_builder;
use crate::{BLAKE2_FUNCT3, BLAKE2_FUNCT7, BLAKE2_NAME, INLINE_OPCODE};
use tracer::register_inline;
use tracer::utils::inline_sequence_writer::{
    write_inline_trace, AppendMode, InlineDescriptor, SequenceInputs,
};

pub fn init_inlines() -> Result<(), String> {
    register_inline(
        INLINE_OPCODE,
        BLAKE2_FUNCT3,
        BLAKE2_FUNCT7,
        BLAKE2_NAME,
        std::boxed::Box::new(sequence_builder::blake2b_inline_sequence_builder),
        None,
    )?;

    Ok(())
}

pub fn store_inlines() -> Result<(), String> {
    let inline_info = InlineDescriptor::new(
        BLAKE2_NAME.to_string(),
        INLINE_OPCODE,
        BLAKE2_FUNCT3,
        BLAKE2_FUNCT7,
    );
    let inputs = SequenceInputs::default();
    let instructions =
        sequence_builder::blake2b_inline_sequence_builder((&inputs).into(), (&inputs).into());
    write_inline_trace(
        "blake2_trace.joltinline",
        &inline_info,
        &inputs,
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
        tracing::error!("Failed to register BLAKE2 inlines: {e}");
    }

    if std::env::var("STORE_INLINE").unwrap_or_default() == "true" {
        if let Err(e) = store_inlines() {
            tracing::error!("Failed to store BLAKE2 inline traces: {e}");
        }
    }
}
