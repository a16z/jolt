use jolt_field::JoltField;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

/// Bound-bit accumulator for [`AlignAddrTable`](crate::tables::align_addr):
/// `Σ 2^k·b_k` over the bound index bits `k ∈ [3, XLEN)`. Bits at or above
/// `XLEN` (the carry) contribute nothing; bits 2..0 are cleared.
pub enum AlignAddrPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for AlignAddrPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let j_start = 2 * XLEN - suffix_len - b.len();
        // Ignore chunks entirely above the low XLEN bits (the carry range).
        // Zeroing is only sound for whole chunks: a chunk straddling index
        // bit XLEN would have its below-XLEN contribution dropped, so phase
        // boundaries must never fall inside the carry/value split.
        debug_assert!(j_start >= XLEN || suffix_len >= XLEN);
        if j_start < XLEN {
            return F::zero();
        }
        checkpoints[Prefixes::AlignAddr] + F::from_u128((u128::from(b) << suffix_len) & !7)
    }
}
