//! Host-side implementation and registration.
pub use crate::sequence_builder;
use crate::{
    BLAKE3_FUNCT3, BLAKE3_FUNCT7, BLAKE3_KEYED64_FUNCT3, BLAKE3_KEYED64_NAME, BLAKE3_NAME,
    INLINE_OPCODE,
};
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
        None,
    )?;

    register_inline(
        INLINE_OPCODE,
        BLAKE3_KEYED64_FUNCT3,
        BLAKE3_FUNCT7,
        BLAKE3_KEYED64_NAME,
        std::boxed::Box::new(sequence_builder::blake3_keyed64_inline_sequence_builder),
        None,
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
    let inputs = SequenceInputs::default();
    let instructions =
        sequence_builder::blake3_inline_sequence_builder((&inputs).into(), (&inputs).into());
    write_inline_trace(
        "blake3_trace.joltinline",
        &inline_info,
        &inputs,
        &instructions,
        AppendMode::Overwrite,
    )
    .map_err(|e| e.to_string())?;

    let inline_info = InlineDescriptor::new(
        BLAKE3_KEYED64_NAME.to_string(),
        INLINE_OPCODE,
        BLAKE3_KEYED64_FUNCT3,
        BLAKE3_FUNCT7,
    );
    let inputs = SequenceInputs::default();
    let instructions = sequence_builder::blake3_keyed64_inline_sequence_builder(
        (&inputs).into(),
        (&inputs).into(),
    );
    write_inline_trace(
        "blake3_trace.joltinline",
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
        eprintln!("Failed to register BLAKE3 inlines: {e}");
    }

    if std::env::var("STORE_INLINE").unwrap_or_default() == "true" {
        if let Err(e) = store_inlines() {
            eprintln!("Failed to store BLAKE3 inline traces: {e}");
        }
    }
}
