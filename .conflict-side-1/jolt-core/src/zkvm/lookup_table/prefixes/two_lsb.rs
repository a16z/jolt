use crate::field::{ChallengeFieldOps, FieldChallengeOps};
use crate::zkvm::instruction_lookups::LOG_K;
use crate::zkvm::lookup_table::prefixes::Prefixes;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

use super::{PrefixCheckpoint, SparseDensePrefix};

pub enum TwoLsbPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for TwoLsbPrefix<XLEN> {
    fn prefix_mle<C>(
        _: &[PrefixCheckpoint<F>],
        r_x: Option<C>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        let suffix_len = LOG_K - j - b.len() - 1;
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
        } else if suffix_len == 0 {
            // in the (log(K)-2)th round, the two LSBs of `b` are the two LSBs
            match u32::from(b) & 0b11 {
                0b00 => F::one(),
                _ => F::zero(),
            }
        } else {
            F::one()
        }
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: C,
        r_y: C,
        j: usize,
        _suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        if j == 2 * XLEN - 1 {
            Some((F::one() - r_x) * (F::one() - r_y)).into()
        } else {
            checkpoints[Prefixes::TwoLsb].into()
        }
    }
}
