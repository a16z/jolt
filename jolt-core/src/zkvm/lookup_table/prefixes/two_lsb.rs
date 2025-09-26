use crate::zkvm::instruction_lookups::read_raf_checking::current_suffix_len;
use crate::zkvm::lookup_table::prefixes::Prefixes;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

use super::{PrefixCheckpoint, SparseDensePrefix};

pub enum TwoLsbPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for TwoLsbPrefix<XLEN> {
    fn prefix_mle(
        _: &[PrefixCheckpoint<F>],
        r_x: Option<F::Challenge>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F {
        if j == 2 * XLEN - 1 {
            // in the log(K)th round, `c` corresponds to bit 0
            // and `r_x` corresponds to bit 1
            debug_assert_eq!(b.len(), 0);
            (F::one() - F::from_u32(c)) * (F::one() - r_x.unwrap())
        } else if j == 2 * XLEN - 2 {
            // in the (log(K)-1)th round, `c` corresponds to bit 1
            debug_assert_eq!(b.len(), 1);
            let bit0 = u32::from(b) & 1;
            let bit1 = c;
            (F::one() - F::from_u32(bit0)) * (F::one() - F::from_u32(bit1))
        } else if current_suffix_len(j) == 0 {
            // in the (log(K)-2)th round, the two LSBs of `b` are the two LSBs
            match u32::from(b) & 0b11 {
                0b00 => F::one(),
                _ => F::zero(),
            }
        } else {
            F::one()
        }
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F::Challenge,
        r_y: F::Challenge,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        if j == 2 * XLEN - 1 {
            Some((F::one() - r_x) * (F::one() - r_y)).into()
        } else {
            checkpoints[Prefixes::TwoLsb].into()
        }
    }
    fn update_prefix_checkpoint_field(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        if j == 2 * XLEN - 1 {
            Some((F::one() - r_x) * (F::one() - r_y)).into()
        } else {
            checkpoints[Prefixes::TwoLsb].into()
        }
    }

    fn prefix_mle_field(
        _checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F {
        if j == 2 * XLEN - 1 {
            // in the log(K)th round, `c` corresponds to bit 0
            // and `r_x` corresponds to bit 1
            debug_assert_eq!(b.len(), 0);
            (F::one() - F::from_u32(c)) * (F::one() - r_x.unwrap())
        } else if j == 2 * XLEN - 2 {
            // in the (log(K)-1)th round, `c` corresponds to bit 1
            debug_assert_eq!(b.len(), 1);
            let bit0 = u32::from(b) & 1;
            let bit1 = c;
            (F::one() - F::from_u32(bit0)) * (F::one() - F::from_u32(bit1))
        } else if current_suffix_len(j) == 0 {
            // in the (log(K)-2)th round, the two LSBs of `b` are the two LSBs
            match u32::from(b) & 0b11 {
                0b00 => F::one(),
                _ => F::zero(),
            }
        } else {
            F::one()
        }
    }
}
