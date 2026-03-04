//! Core traits for Jolt instruction definitions and lookup table decompositions.
//!
//! The [`Instruction`] trait defines the interface for all RISC-V instructions
//! in the Jolt zkVM. Each instruction can be executed on concrete operands and
//! decomposed into lookup table queries for the proving system.
//!
//! The [`LookupTable`] trait defines small-domain functions used by the prover
//! to evaluate instruction decompositions as multilinear extensions.

use jolt_field::Field;
use serde::{Deserialize, Serialize};

/// Identifies a lookup table by a unique 16-bit index.
///
/// Table IDs are assigned sequentially during instruction set registration
/// and remain stable for the lifetime of a proving session.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct TableId(pub u16);

/// A single lookup table query produced by instruction decomposition.
///
/// During proving, each instruction execution is decomposed into a sequence
/// of these queries. The prover evaluates each query against the corresponding
/// [`LookupTable`] to produce field elements for the sumcheck protocol.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LookupQuery {
    /// Which table to query.
    pub table: TableId,
    /// The input value to the table (typically a small chunk of the operand).
    pub input: u64,
}

/// A RISC-V instruction: a pure function from two 64-bit operands to a 64-bit result.
///
/// Implementations must be stateless and deterministic. The [`execute`](Instruction::execute)
/// method provides the ground-truth computation using native Rust arithmetic,
/// while [`lookups`](Instruction::lookups) decomposes the same computation into
/// small lookup table queries suitable for the Jolt proving system.
pub trait Instruction: Send + Sync + 'static {
    /// Unique opcode identifying this instruction within the [`JoltInstructionSet`](crate::JoltInstructionSet).
    fn opcode(&self) -> u32;

    /// Human-readable mnemonic (e.g., `"ADD"`, `"SRL"`).
    fn name(&self) -> &'static str;

    /// Execute the instruction on two 64-bit operands, returning a 64-bit result.
    ///
    /// For RV64I/M instructions this uses wrapping arithmetic matching the RISC-V
    /// specification. For RV64 W-suffix instructions, the result is sign-extended
    /// from 32 bits to 64 bits.
    fn execute(&self, x: u64, y: u64) -> u64;

    /// Decompose the instruction execution into lookup table queries.
    ///
    /// Returns an empty vector for instructions whose lookup decomposition
    /// has not yet been implemented. Full decomposition will be provided
    /// when the prover pipeline is integrated.
    fn lookups(&self, x: u64, y: u64) -> Vec<LookupQuery>;
}

/// A lookup table mapping a small input domain to field elements.
///
/// Tables are materialized once during preprocessing and queried many times
/// during the sumcheck-based proving protocol. The [`size`](LookupTable::size)
/// determines the input domain `0..size`.
pub trait LookupTable<F: Field>: Send + Sync {
    /// Unique identifier for this table.
    fn id(&self) -> TableId;

    /// Human-readable name for debugging and profiling.
    fn name(&self) -> &'static str;

    /// Number of entries (input domain is `0..size`).
    fn size(&self) -> usize;

    /// Evaluate the table at a single input.
    fn evaluate(&self, input: u64) -> F;

    /// Materialize the entire table as a dense vector of field elements.
    ///
    /// The default implementation evaluates each input sequentially.
    /// Implementations may override this for better performance.
    fn materialize(&self) -> Vec<F> {
        (0..self.size() as u64).map(|i| self.evaluate(i)).collect()
    }
}
