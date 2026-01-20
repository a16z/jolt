//! BigInt multiplication inline implementation module

#![cfg_attr(not(feature = "host"), no_std)]

pub mod multiplication;
pub use multiplication::*;

#[cfg(feature = "host")]
use tracer::register_inline;

#[cfg(feature = "host")]
use tracer::utils::inline_sequence_writer::{write_inline_trace, InlineDescriptor, SequenceInputs};

// Initialize and register inlines
#[cfg(feature = "host")]
pub fn init_inlines() -> Result<(), String> {
    // Register 256-bit Int multiplication
    register_inline(
        multiplication::INLINE_OPCODE,
        multiplication::BIGINT256_MUL_FUNCT3,
        multiplication::BIGINT256_MUL_FUNCT7,
        multiplication::BIGINT256_MUL_NAME,
        std::boxed::Box::new(sequence_builder::bigint_mul_sequence_builder),
        None,
    )?;

    Ok(())
}

#[cfg(feature = "host")]
pub fn store_inlines() -> Result<(), String> {
    use tracer::utils::inline_sequence_writer::AppendMode;

    let inline_info = InlineDescriptor::new(
        multiplication::BIGINT256_MUL_NAME.to_string(),
        multiplication::INLINE_OPCODE,
        multiplication::BIGINT256_MUL_FUNCT3,
        multiplication::BIGINT256_MUL_FUNCT7,
    );
    let inputs = SequenceInputs::default();
    let instructions =
        sequence_builder::bigint_mul_sequence_builder((&inputs).into(), (&inputs).into());
    write_inline_trace(
        "bigint_mul256_trace.joltinline",
        &inline_info,
        &inputs,
        &instructions,
        AppendMode::Overwrite,
    )
    .map_err(|e| e.to_string())?;

    Ok(())
}

// Automatic registration when the library is loaded
#[cfg(all(feature = "host", not(target_arch = "wasm32")))]
#[ctor::ctor]
fn auto_register() {
    if let Err(e) = init_inlines() {
        tracing::error!("Failed to register BIGINT256_MUL inlines: {e}");
    }

    if std::env::var("STORE_INLINE").unwrap_or_default() == "true" {
        if let Err(e) = store_inlines() {
            tracing::error!("Failed to store BIGINT256_MUL inline traces: {e}");
        }
    }
}
