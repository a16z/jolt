use jolt_field::JoltField;

use crate::lookup_bits::LookupBits;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

/// `2^(32·ea_2)` where `ea_2` is bit 2 of the (non-interleaved) lookup index:
/// the word-offset half of a doubleword effective address.
pub enum Pow2OffsetWPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for Pow2OffsetWPrefix {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        // Phase-boundary-agnostic split of the index around bit 2: the suffix
        // owns bits [0, suffix_len), this phase's bits `b` own
        // [suffix_len, suffix_len + b.len()), and everything above is bound
        // into the checkpoint.
        if suffix_len >= 3 {
            // Bit 2 is in the suffix; `Pow2OffsetWSuffix` supplies the factor.
            return F::one();
        }
        let checkpoint = checkpoints[Prefixes::Pow2OffsetW];
        if suffix_len + b.len() > 2 {
            // Bit 2 is among this phase's bits.
            let bits: u128 = b.into();
            let bit2 = ((bits >> (2 - suffix_len)) & 1) as u32;
            checkpoint * F::from_u64(1u64 << (32 * bit2))
        } else {
            // Bit 2 was bound in an earlier phase; its factor lives in the
            // checkpoint.
            checkpoint
        }
    }
}
