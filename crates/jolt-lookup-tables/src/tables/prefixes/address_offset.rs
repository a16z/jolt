use jolt_field::JoltField;

use crate::lookup_bits::LookupBits;
use crate::tables::address_offset::AddressOffsetTable;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

/// Bound-bit accumulator for
/// [`AddressOffsetTable`](crate::tables::address_offset::AddressOffsetTable):
/// `Σ 2^k·b_k` over the offset bits `k < 3` the prefix owns, restricted to the
/// width's offset mask. The suffix owns index bits `[0, suffix_len)`, this
/// phase's bits `b` the next `b.len()`, and offset bits bound in earlier
/// phases sit in the checkpoint.
pub enum AddressOffsetPrefix<const WIDTH: usize> {}

impl<const WIDTH: usize> AddressOffsetPrefix<WIDTH> {
    pub(crate) const VARIANT: Prefixes = match WIDTH {
        1 => Prefixes::AddressOffsetB,
        2 => Prefixes::AddressOffsetH,
        4 => Prefixes::AddressOffsetW,
        _ => panic!("unsupported WIDTH"),
    };
}

impl<const WIDTH: usize, F: JoltField> SparseDensePrefix<F> for AddressOffsetPrefix<WIDTH> {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        if suffix_len >= 3 {
            // Every offset bit is still in the suffix.
            return F::zero();
        }
        let bits = (u128::from(b) << suffix_len) & AddressOffsetTable::<XLEN, WIDTH>::OFFSET_MASK;
        checkpoints[Self::VARIANT] + F::from_u128(bits)
    }
}
