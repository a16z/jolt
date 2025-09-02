use crate::zkvm::instruction_lookups::read_raf_checking::current_suffix_len;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};
use crate::field::MontU128;
use super::{PrefixCheckpoint, SparseDensePrefix};

pub enum LsbPrefix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize, F: JoltField> SparseDensePrefix<F> for LsbPrefix<WORD_SIZE> {
    fn prefix_mle(_: &[PrefixCheckpoint<F>], _: Option<MontU128>, c: u32, b: LookupBits, j: usize) -> F {
        if j == 2 * WORD_SIZE - 1 {
            // in the log(K)th round, `c` corresponds to the LSB
            debug_assert_eq!(b.len(), 0);
            F::from_u32(c)
        } else if current_suffix_len(2 * WORD_SIZE, j) == 0 {
            // in the (log(K)-1)th round, the LSB of `b` is the LSB
            F::from_u32(u32::from(b) & 1)
        } else {
            F::one()
        }
    }

    fn update_prefix_checkpoint(
        _: &[PrefixCheckpoint<F>],
        _: MontU128,
        r_y: MontU128,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        if j == 2 * WORD_SIZE - 1 {
            Some(F::from_u128_mont(r_y)).into()
        } else {
            Some(F::one()).into()
        }
    }
}
