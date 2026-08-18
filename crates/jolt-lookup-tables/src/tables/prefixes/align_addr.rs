use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

/// Bound-bit accumulator for [`AlignAddrTable`](crate::tables::align_addr):
/// `Σ 2^k·b_k` over the bound index bits `k ∈ [3, XLEN)`. Bits at or above
/// `XLEN` (the carry) contribute nothing; bits 2..0 are cleared.
pub enum AlignAddrPrefix {}

impl<F: Field> SparseDensePrefix<F> for AlignAddrPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let j_start = 2 * XLEN - suffix_len - b.len();
        // Ignore chunks entirely above the low XLEN bits (the carry range).
        if j_start < XLEN {
            return F::zero();
        }
        checkpoints[Prefixes::AlignAddr] + F::from_u128((u128::from(b) << suffix_len) & !7)
    }
}
