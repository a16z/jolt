use jolt_field::Field;

use crate::lookup_bits::LookupBits;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

/// `2^(8·(ea mod 8))` where `ea` is the (non-interleaved) lookup index: the
/// byte-offset scale of a doubleword effective address.
pub enum Pow2OffsetBPrefix {}

impl<F: Field> SparseDensePrefix<F> for Pow2OffsetBPrefix {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        // The low 3 index bits stay in the suffix until the final phase.
        if suffix_len != 0 {
            return F::one();
        }
        let bits: u128 = b.into();
        let offset = (bits & 7) as u32;
        checkpoints[Prefixes::Pow2OffsetB] * F::from_u64(1u64 << (8 * offset))
    }
}
