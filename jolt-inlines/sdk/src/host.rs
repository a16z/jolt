use std::collections::VecDeque;

use tracer::{
    emulator::cpu::Cpu,
    instruction::{format::format_inline::FormatInline, Instruction},
    register_inline,
    utils::{
        inline_helpers::InstrAssembler,
        inline_sequence_writer::{write_inline_trace, InlineDescriptor, SequenceInputs},
    },
};

pub use tracer::utils::inline_sequence_writer::AppendMode;

/// Trait for declaring an inline operation's metadata and sequence builder.
///
/// Implement this for each sub-inline (e.g. `Sha256Compression`, `Secp256k1MulQ`),
/// then pass the types to [`register_inlines!`] to generate registration boilerplate.
pub trait InlineOp: Send + Sync {
    const OPCODE: u32;
    const FUNCT3: u32;
    const FUNCT7: u32;
    const NAME: &'static str;
    const HAS_ADVICE: bool = false;

    fn build_sequence(asm: InstrAssembler, operands: FormatInline) -> Vec<Instruction>;

    fn build_advice(
        _asm: InstrAssembler,
        _operands: FormatInline,
        _cpu: &mut Cpu,
    ) -> Option<VecDeque<u64>> {
        None
    }
}

type BoxedAdviceFn =
    Box<dyn Fn(InstrAssembler, FormatInline, &mut Cpu) -> VecDeque<u64> + Send + Sync>;

/// Register a single `InlineOp` type with the tracer's global inline registry.
pub fn register<T: InlineOp + 'static>() -> Result<(), String> {
    let advice_fn: Option<BoxedAdviceFn> = if T::HAS_ADVICE {
        Some(Box::new(|asm, operands, cpu| {
            T::build_advice(asm, operands, cpu)
                .expect("HAS_ADVICE=true but build_advice returned None")
        }))
    } else {
        None
    };

    register_inline(
        T::OPCODE,
        T::FUNCT3,
        T::FUNCT7,
        T::NAME,
        Box::new(T::build_sequence),
        advice_fn,
    )
}

/// Write the default inline trace for a single `InlineOp` to `file` with the given `mode`.
pub fn store_trace<T: InlineOp>(file: &str, mode: AppendMode) -> Result<(), String> {
    let inline_info = InlineDescriptor::new(T::NAME.to_string(), T::OPCODE, T::FUNCT3, T::FUNCT7);
    let inputs = SequenceInputs::default();
    let instructions = T::build_sequence((&inputs).into(), (&inputs).into());
    write_inline_trace(file, &inline_info, &inputs, &instructions, mode).map_err(|e| e.to_string())
}

/// Plugin registration entry submitted by each inline crate via [`register_inlines!`].
/// Collected at link time by `inventory` — no manual bookkeeping needed.
pub struct InlineRegistration {
    pub init: fn() -> Result<(), String>,
}

inventory::collect!(InlineRegistration);

/// Register all inline implementations that are linked into the binary.
/// Idempotent — safe to call multiple times.
pub fn register_all_inlines() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        for entry in inventory::iter::<InlineRegistration> {
            (entry.init)().expect("Failed to register inlines");
        }
    });
}

/// Generate `init_inlines()` and `store_inlines()`, and register the crate's inlines
/// with the global plugin registry via `inventory`.
///
/// Any binary that links this crate will automatically discover its inlines
/// when [`register_all_inlines()`] is called.
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
        pub fn init_inlines() -> Result<(), String> {
            $crate::host::register::<$first>()?;
            $($crate::host::register::<$rest>()?;)*
            Ok(())
        }

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

        inventory::submit! {
            $crate::host::InlineRegistration {
                init: init_inlines,
            }
        }
    };
}
