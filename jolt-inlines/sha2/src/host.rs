//! Host-side implementation and registration.
pub use crate::sequence_builder;

use crate::{
    INLINE_OPCODE, SHA256_FUNCT3, SHA256_FUNCT7, SHA256_INIT_FUNCT3, SHA256_INIT_FUNCT7,
    SHA256_INIT_NAME, SHA256_NAME,
};
use tracer::register_inline;

use tracer::utils::inline_sequence_writer::{
    write_inline_trace, AppendMode, InlineDescriptor, SequenceInputs,
};

pub fn init_inlines() -> Result<(), String> {
    register_inline(
        INLINE_OPCODE,
        SHA256_FUNCT3,
        SHA256_FUNCT7,
        SHA256_NAME,
        std::boxed::Box::new(sequence_builder::sha2_inline_sequence_builder),
        None,
    )?;

    register_inline(
        INLINE_OPCODE,
        SHA256_INIT_FUNCT3,
        SHA256_INIT_FUNCT7,
        SHA256_INIT_NAME,
        std::boxed::Box::new(sequence_builder::sha2_init_inline_sequence_builder),
        None,
    )?;

    Ok(())
}

pub fn store_inlines() -> Result<(), String> {
    // Store SHA256 default inline trace
    let inline_info = InlineDescriptor::new(
        SHA256_NAME.to_string(),
        INLINE_OPCODE,
        SHA256_FUNCT3,
        SHA256_FUNCT7,
    );
    let inputs = SequenceInputs::default();
    let instructions =
        sequence_builder::sha2_inline_sequence_builder((&inputs).into(), (&inputs).into());
    write_inline_trace(
        "sha256_trace.joltinline",
        &inline_info,
        &inputs,
        &instructions,
        AppendMode::Overwrite,
    )
    .map_err(|e| e.to_string())?;

    // Store SHA256 init inline trace (append to the same file)
    let inline_info = InlineDescriptor::new(
        SHA256_INIT_NAME.to_string(),
        INLINE_OPCODE,
        SHA256_INIT_FUNCT3,
        SHA256_INIT_FUNCT7,
    );
    let inputs = SequenceInputs::default();
    let instructions =
        sequence_builder::sha2_init_inline_sequence_builder((&inputs).into(), (&inputs).into());
    write_inline_trace(
        "sha256_trace.joltinline",
        &inline_info,
        &inputs,
        &instructions,
        AppendMode::Append,
    )
    .map_err(|e| e.to_string())?;

    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
#[ctor::ctor]
fn auto_register() {
    if let Err(e) = init_inlines() {
        tracing::error!("Failed to register SHA256 inlines: {e}");
    }

    if std::env::var("STORE_INLINE").unwrap_or_default() == "true" {
        if let Err(e) = store_inlines() {
            eprintln!("Failed to store SHA256 inline traces: {e}");
        }
    }
}
