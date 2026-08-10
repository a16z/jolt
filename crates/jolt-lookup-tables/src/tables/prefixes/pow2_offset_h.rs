use jolt_field::Field;

use crate::lookup_bits::LookupBits;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

/// `2^(8·(ea mod 8 & !1))` where `ea` is the (non-interleaved) lookup index:
/// the halfword-offset scale of a doubleword effective address (bit 0
/// ignored).
pub enum Pow2OffsetHPrefix {}

impl<F: Field> SparseDensePrefix<F> for Pow2OffsetHPrefix {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        if suffix_len != 0 {
            return F::one();
        }
        let bits: u128 = b.into();
        let offset = (bits & 6) as u32;
        checkpoints[Prefixes::Pow2OffsetH] * F::from_u64(1u64 << (8 * offset))
    }
}
