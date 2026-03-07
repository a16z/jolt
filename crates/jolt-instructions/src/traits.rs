//! Core traits for Jolt instruction definitions and lookup table decompositions.
//!
//! The [`Instruction`] trait defines the interface for all RISC-V instructions
//! in the Jolt zkVM. Each instruction declares its static flags, lookup table
//! association, and execution semantics.
//!
//! The [`LookupTable`] trait defines small-domain functions used by the prover,
//! including multilinear extension evaluation for the sumcheck protocol.

use jolt_field::Field;
use std::fmt::Debug;

use crate::flags::Flags;
use crate::tables::LookupTableKind;

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
    /// Unique opcode identifying this instruction within the [`JoltInstructionSet`](crate::JoltInstructionSet).
    fn opcode(&self) -> u32;

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

/// A lookup table mapping interleaved operand indices to scalar values.
///
/// Tables are materialized once during preprocessing and their multilinear
/// extensions are evaluated during the sumcheck protocol. The index space is
/// `0..2^(2*XLEN)` where `XLEN` is the word size (8 for tests, 64 for production).
///
/// The `XLEN` const generic determines the word size. Challenge point `r` passed
/// to [`evaluate_mle`](LookupTable::evaluate_mle) has length `2 * XLEN`.
pub trait LookupTable<const XLEN: usize>: Clone + Debug + Send + Sync {
    /// Compute the raw table value at the given interleaved index.
    ///
    /// For tables with two operands, `index` contains interleaved bits of `(x, y)`.
    /// Use [`uninterleave_bits`](crate::uninterleave_bits) to recover the operands.
    fn materialize_entry(&self, index: u128) -> u64;

    /// Evaluate the multilinear extension of this table at challenge point `r`.
    ///
    /// `r` has length `2 * XLEN`. For interleaved-operand tables, even indices
    /// correspond to the first operand and odd indices to the second.
    ///
    /// This is the critical method for the sumcheck verifier: it checks that
    /// the prover's claimed table evaluation matches the table's MLE.
    fn evaluate_mle<F: Field>(&self, r: &[F]) -> F;

    /// Materialize the entire table as a dense vector (test-only, XLEN=8).
    #[cfg(test)]
    fn materialize(&self) -> Vec<u64> {
        (0..1u128 << (2 * XLEN))
            .map(|i| self.materialize_entry(i))
            .collect()
    }
}
