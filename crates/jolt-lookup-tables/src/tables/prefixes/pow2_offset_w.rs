use jolt_field::Field;

use crate::lookup_bits::LookupBits;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

/// `2^(32·ea_2)` where `ea_2` is bit 2 of the (non-interleaved) lookup index:
/// the word-offset half of a doubleword effective address.
pub enum Pow2OffsetWPrefix {}

impl<F: Field> SparseDensePrefix<F> for Pow2OffsetWPrefix {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        // Bit 2 of the raw index stays in the suffix until the final phase;
        // this pairing with the suffix's `b.len() < 3` guard assumes phase
        // boundaries never fall inside the low three index bits.
        debug_assert!(suffix_len == 0 || suffix_len >= 3);
        if suffix_len != 0 {
            return F::one();
        }
        let bits: u128 = b.into();
        let bit2 = ((bits >> 2) & 1) as u32;
        checkpoints[Prefixes::Pow2OffsetW] * F::from_u64(1u64 << (32 * bit2))
    }
}
