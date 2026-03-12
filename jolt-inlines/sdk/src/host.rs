use std::collections::VecDeque;

use tracer::utils::inline_sequence_writer::{write_inline_trace, InlineDescriptor, SequenceInputs};

pub use tracer::emulator::cpu::{Cpu, Xlen};
pub use tracer::instruction;
pub use tracer::instruction::format::format_inline::FormatInline;
pub use tracer::instruction::inline::InlineRegistration;
pub use tracer::instruction::Instruction;
pub use tracer::utils::inline_helpers::{InstrAssembler, Value};
pub use tracer::utils::inline_sequence_writer::AppendMode;
pub use tracer::utils::virtual_registers::VirtualRegisterGuard;

/// Trait for declaring an inline operation's metadata and sequence builder.
///
/// Implement this for each sub-inline (e.g. `Sha256Compression`, `Secp256k1MulQ`),
/// then pass the types to [`register_inlines!`] to generate registration boilerplate.
pub trait InlineOp: Send + Sync {
    const OPCODE: u32;
    const FUNCT3: u32;
    const FUNCT7: u32;
    const NAME: &'static str;

    fn build_sequence(asm: InstrAssembler, operands: FormatInline) -> Vec<Instruction>;

    fn build_advice(
        _asm: InstrAssembler,
        _operands: FormatInline,
        _cpu: &mut Cpu,
    ) -> Option<VecDeque<u64>> {
        None
    }
}

/// Write the default inline trace for a single `InlineOp` to `file` with the given `mode`.
pub fn store_trace<T: InlineOp>(file: &str, mode: AppendMode) -> Result<(), String> {
    let inline_info = InlineDescriptor::new(T::NAME.to_string(), T::OPCODE, T::FUNCT3, T::FUNCT7);
    let inputs = SequenceInputs::default();
    let instructions = T::build_sequence((&inputs).into(), (&inputs).into());
    write_inline_trace(file, &inline_info, &inputs, &instructions, mode).map_err(|e| e.to_string())
}

/// Generate `store_inlines()` and submit `InlineRegistration` entries to `inventory`.
///
/// Each `InlineOp` type gets an `inventory::submit!` that registers it at link time.
/// The tracer's INLINE instruction discovers these registrations automatically.
///
/// ```ignore
/// register_inlines! {
///     trace_file: "sha256_trace.joltinline",
///     ops: [Sha256Compression, Sha256CompressionInitial],
/// }
/// ```
#[macro_export]
macro_rules! register_inlines {
    (
        trace_file: $trace_file:expr,
        ops: [$first:ty $(, $rest:ty)*$(,)?]$(,)?
    ) => {
        pub fn store_inlines() -> Result<(), String> {
            $crate::host::store_trace::<$first>(
                $trace_file,
                $crate::host::AppendMode::Overwrite,
            )?;
            $($crate::host::store_trace::<$rest>(
                $trace_file,
                $crate::host::AppendMode::Append,
            )?;)*
            Ok(())
        }

        $crate::__submit_inline_op!($first);
        $($crate::__submit_inline_op!($rest);)*
    };
}

/// Helper macro to submit a single `InlineOp` to inventory.
#[macro_export]
macro_rules! __submit_inline_op {
    ($op:ty) => {
        inventory::submit! {
            $crate::host::InlineRegistration {
                opcode: <$op as $crate::host::InlineOp>::OPCODE,
                funct3: <$op as $crate::host::InlineOp>::FUNCT3,
                funct7: <$op as $crate::host::InlineOp>::FUNCT7,
                name: <$op as $crate::host::InlineOp>::NAME,
                build_sequence: <$op as $crate::host::InlineOp>::build_sequence,
                build_advice: <$op as $crate::host::InlineOp>::build_advice,
            }
        }
    };
}
