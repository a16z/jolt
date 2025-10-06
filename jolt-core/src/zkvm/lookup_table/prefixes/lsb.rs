use crate::utils::lookup_bits::LookupBits;
use crate::zkvm::instruction_lookups::read_raf_checking::current_suffix_len;
use jolt_field::JoltField;

use super::{PrefixCheckpoint, SparseDensePrefix};

pub enum LsbPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for LsbPrefix<XLEN> {
    fn prefix_mle(_: &[PrefixCheckpoint<F>], _: Option<F>, c: u32, b: LookupBits, j: usize) -> F {
        if j == 2 * XLEN - 1 {
            // in the log(K)th round, `c` corresponds to the LSB
            debug_assert_eq!(b.len(), 0);
            F::from_u32(c)
        } else if current_suffix_len(j) == 0 {
            // in the (log(K)-1)th round, the LSB of `b` is the LSB
            F::from_u32(u32::from(b) & 1)
        } else {
            F::one()
        }
    }

    fn update_prefix_checkpoint(
        _: &[PrefixCheckpoint<F>],
        _: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        if j == 2 * XLEN - 1 {
            Some(r_y).into()
        } else {
            Some(F::one()).into()
        }
    }
}
