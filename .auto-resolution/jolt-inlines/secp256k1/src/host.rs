//! Host-side implementation and registration.
pub use crate::sequence_builder;

use crate::{
    INLINE_OPCODE, SECP256K1_DIVQ_ADV_FUNCT3, SECP256K1_DIVQ_ADV_NAME, SECP256K1_DIVR_ADV_FUNCT3,
    SECP256K1_DIVR_ADV_NAME, SECP256K1_FUNCT7, SECP256K1_GLVR_ADV_FUNCT3, SECP256K1_GLVR_ADV_NAME,
};
use tracer::register_inline;

use tracer::utils::inline_sequence_writer::{
    write_inline_trace, AppendMode, InlineDescriptor, SequenceInputs,
};

pub fn init_inlines() -> Result<(), String> {
    register_inline(
        INLINE_OPCODE,
        SECP256K1_DIVQ_ADV_FUNCT3,
        SECP256K1_FUNCT7,
        SECP256K1_DIVQ_ADV_NAME,
        std::boxed::Box::new(sequence_builder::secp256k1_divq_adv_sequence_builder),
        Some(std::boxed::Box::new(
            sequence_builder::secp256k1_divq_adv_advice,
        )),
    )?;
    register_inline(
        INLINE_OPCODE,
        SECP256K1_DIVR_ADV_FUNCT3,
        SECP256K1_FUNCT7,
        SECP256K1_DIVR_ADV_NAME,
        std::boxed::Box::new(sequence_builder::secp256k1_divr_adv_sequence_builder),
        Some(std::boxed::Box::new(
            sequence_builder::secp256k1_divr_adv_advice,
        )),
    )?;
    register_inline(
        INLINE_OPCODE,
        SECP256K1_GLVR_ADV_FUNCT3,
        SECP256K1_FUNCT7,
        SECP256K1_GLVR_ADV_NAME,
        std::boxed::Box::new(sequence_builder::secp256k1_glvr_adv_sequence_builder),
        Some(std::boxed::Box::new(
            sequence_builder::secp256k1_glvr_adv_advice,
        )),
    )?;
    Ok(())
}

pub fn store_inlines() -> Result<(), String> {
    // Store secp256k1 divq inline trace
    let inline_info = InlineDescriptor::new(
        SECP256K1_DIVQ_ADV_NAME.to_string(),
        INLINE_OPCODE,
        SECP256K1_DIVQ_ADV_FUNCT3,
        SECP256K1_FUNCT7,
    );
    let inputs = SequenceInputs::default();
    let instructions =
        sequence_builder::secp256k1_divq_adv_sequence_builder((&inputs).into(), (&inputs).into());
    write_inline_trace(
        "secp256k1_trace.joltinline",
        &inline_info,
        &inputs,
        &instructions,
        AppendMode::Overwrite,
    )
    .map_err(|e| e.to_string())?;
    // Append secp256k1 divr inline trace
    let inline_info = InlineDescriptor::new(
        SECP256K1_DIVR_ADV_NAME.to_string(),
        INLINE_OPCODE,
        SECP256K1_DIVR_ADV_FUNCT3,
        SECP256K1_FUNCT7,
    );
    let inputs = SequenceInputs::default();
    let instructions =
        sequence_builder::secp256k1_divr_adv_sequence_builder((&inputs).into(), (&inputs).into());
    write_inline_trace(
        "secp256k1_trace.joltinline",
        &inline_info,
        &inputs,
        &instructions,
        AppendMode::Append,
    )
    .map_err(|e| e.to_string())?;
    // Append secp256k1 glvr inline trace
    let inline_info = InlineDescriptor::new(
        SECP256K1_GLVR_ADV_NAME.to_string(),
        INLINE_OPCODE,
        SECP256K1_GLVR_ADV_FUNCT3,
        SECP256K1_FUNCT7,
    );
    let inputs = SequenceInputs::default();
    let instructions =
        sequence_builder::secp256k1_glvr_adv_sequence_builder((&inputs).into(), (&inputs).into());
    write_inline_trace(
        "secp256k1_trace.joltinline",
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
        tracing::error!("Failed to register secp256k1 inlines: {e}");
    }

    if std::env::var("STORE_INLINE").unwrap_or_default() == "true" {
        if let Err(e) = store_inlines() {
            tracing::error!("Failed to store secp256k1 inline traces: {e}");
        }
    }
}
