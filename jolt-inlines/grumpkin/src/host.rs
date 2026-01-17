//! Host-side implementation and registration.
pub use crate::sequence_builder;

use crate::{
    GRUMPKIN_DIVQ_ADV_FUNCT3, GRUMPKIN_DIVQ_ADV_NAME, GRUMPKIN_DIVR_ADV_FUNCT3,
    GRUMPKIN_DIVR_ADV_NAME, GRUMPKIN_FUNCT7, GRUMPKIN_GLVR_ADV_FUNCT3, GRUMPKIN_GLVR_ADV_NAME,
    INLINE_OPCODE,
};
use tracer::register_inline;

use tracer::utils::inline_sequence_writer::{
    write_inline_trace, AppendMode, InlineDescriptor, SequenceInputs,
};

pub fn init_inlines() -> Result<(), String> {
    register_inline(
        INLINE_OPCODE,
        GRUMPKIN_DIVQ_ADV_FUNCT3,
        GRUMPKIN_FUNCT7,
        GRUMPKIN_DIVQ_ADV_NAME,
        std::boxed::Box::new(sequence_builder::grumpkin_divq_adv_sequence_builder),
        Some(std::boxed::Box::new(
            sequence_builder::grumpkin_divq_adv_advice,
        )),
    )?;
    register_inline(
        INLINE_OPCODE,
        GRUMPKIN_DIVR_ADV_FUNCT3,
        GRUMPKIN_FUNCT7,
        GRUMPKIN_DIVR_ADV_NAME,
        std::boxed::Box::new(sequence_builder::grumpkin_divr_adv_sequence_builder),
        Some(std::boxed::Box::new(
            sequence_builder::grumpkin_divr_adv_advice,
        )),
    )?;
    register_inline(
        INLINE_OPCODE,
        GRUMPKIN_GLVR_ADV_FUNCT3,
        GRUMPKIN_FUNCT7,
        GRUMPKIN_GLVR_ADV_NAME,
        std::boxed::Box::new(sequence_builder::grumpkin_glvr_adv_sequence_builder),
        Some(std::boxed::Box::new(
            sequence_builder::grumpkin_glvr_adv_advice,
        )),
    )?;
    Ok(())
}

pub fn store_inlines() -> Result<(), String> {
    // Store grumpkin divq inline trace
    let inline_info = InlineDescriptor::new(
        GRUMPKIN_DIVQ_ADV_NAME.to_string(),
        INLINE_OPCODE,
        GRUMPKIN_DIVQ_ADV_FUNCT3,
        GRUMPKIN_FUNCT7,
    );
    let inputs = SequenceInputs::default();
    let instructions =
        sequence_builder::grumpkin_divq_adv_sequence_builder((&inputs).into(), (&inputs).into());
    write_inline_trace(
        "grumpkin_trace.joltinline",
        &inline_info,
        &inputs,
        &instructions,
        AppendMode::Overwrite,
    )
    .map_err(|e| e.to_string())?;
    // Append grumpkin divr inline trace
    let inline_info = InlineDescriptor::new(
        GRUMPKIN_DIVR_ADV_NAME.to_string(),
        INLINE_OPCODE,
        GRUMPKIN_DIVR_ADV_FUNCT3,
        GRUMPKIN_FUNCT7,
    );
    let inputs = SequenceInputs::default();
    let instructions =
        sequence_builder::grumpkin_divr_adv_sequence_builder((&inputs).into(), (&inputs).into());
    write_inline_trace(
        "grumpkin_trace.joltinline",
        &inline_info,
        &inputs,
        &instructions,
        AppendMode::Append,
    )
    .map_err(|e| e.to_string())?;
    // Append grumpkin glvr inline trace
    let inline_info = InlineDescriptor::new(
        GRUMPKIN_GLVR_ADV_NAME.to_string(),
        INLINE_OPCODE,
        GRUMPKIN_GLVR_ADV_FUNCT3,
        GRUMPKIN_FUNCT7,
    );
    let inputs = SequenceInputs::default();
    let instructions =
        sequence_builder::grumpkin_glvr_adv_sequence_builder((&inputs).into(), (&inputs).into());
    write_inline_trace(
        "grumpkin_trace.joltinline",
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
        tracing::error!("Failed to register grumpkin inlines: {e}");
    }

    if std::env::var("STORE_INLINE").unwrap_or_default() == "true" {
        if let Err(e) = store_inlines() {
            tracing::error!("Failed to store grumpkin inline traces: {e}");
        }
    }
}
