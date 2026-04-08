use crate::flags::Flags;

/// A Jolt instruction modelled as a pure function from two 64-bit operands to a 64-bit result.
///
/// Implementations must be stateless and deterministic. The [`execute`](Instruction::execute)
/// method provides ground-truth computation for arithmetic and lookup-backed instructions.
/// Runtime-managed opcodes such as advice and host I/O remain in the registry for opcode and
/// flag lookup, but their `execute` implementation is only a placeholder and is not a full
/// emulator for tracer state.
///
/// Every instruction also implements [`Flags`] to declare its static R1CS
/// and witness-generation flag configuration. Dynamic flags (virtual sequence
/// state, compression, rd!=0) are applied by the runtime based on trace context.
pub trait Instruction: Flags + Send + Sync + 'static {
    /// Human-readable mnemonic (e.g., `"ADD"`, `"SRL"`).
    fn name(&self) -> &'static str;

    /// Execute the instruction on two 64-bit operands, returning a 64-bit result.
    ///
    /// For RV64I/M instructions this uses wrapping arithmetic matching the RISC-V
    /// specification. For W-suffix instructions, the result is sign-extended
    /// from 32 bits to 64 bits unless the instruction explicitly models a zero-extending
    /// virtual operation such as `VIRTUAL_ROTRIW`.
    fn execute(&self, x: u64, y: u64) -> u64;
}
