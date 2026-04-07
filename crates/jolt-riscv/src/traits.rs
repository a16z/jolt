use crate::flags::Flags;
use crate::LookupTableKind;

/// A RISC-V instruction: a pure function from two 64-bit operands to a 64-bit result.
///
/// Implementations must be stateless and deterministic. The [`execute`](Instruction::execute)
/// method provides ground-truth computation using native Rust arithmetic.
/// [`lookup_table`](Instruction::lookup_table) declares which lookup table this
/// instruction decomposes into for the proving system.
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
    /// from 32 bits to 64 bits.
    fn execute(&self, x: u64, y: u64) -> u64;

    /// The lookup table this instruction decomposes into, if any.
    ///
    /// Returns `None` for instructions that don't use lookup tables (loads, stores,
    /// system instructions). The prover uses this to route instruction evaluations
    /// to the correct table during the instruction sumcheck.
    fn lookup_table(&self) -> Option<LookupTableKind>;
}
