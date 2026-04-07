use jolt_field::Field;
use std::fmt::Debug;

use crate::challenge_ops::{ChallengeOps, FieldOps};

/// A lookup table mapping interleaved operand indices to scalar values.
///
/// Tables are materialized once during preprocessing and their multilinear
/// extensions are evaluated during the sumcheck protocol. The index space is
/// `0..2^(2*XLEN)` where `XLEN` is the word size (8 for tests, 64 for production).
///
/// The `XLEN` const generic determines the word size. Challenge point `r` passed
/// to [`evaluate_mle`](LookupTable::evaluate_mle) has length `2 * XLEN`.
///
/// The `evaluate_mle` method is generic over a challenge type `C` to support
/// smaller-than-field-element challenge values (e.g., 128-bit challenges with
/// a 254-bit field), which is a critical performance optimization for the
/// sumcheck prover.
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
    /// `C` is the challenge type (may be smaller than `F` for performance).
    /// When `C = F`, this degenerates to standard field evaluation.
    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>;

    /// Materialize the entire table as a dense vector (test-only, XLEN=8).
    #[cfg(test)]
    fn materialize(&self) -> Vec<u64> {
        (0..1u128 << (2 * XLEN))
            .map(|i| self.materialize_entry(i))
            .collect()
    }
}
