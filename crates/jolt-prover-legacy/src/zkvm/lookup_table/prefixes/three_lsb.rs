use crate::field::{ChallengeFieldOps, FieldChallengeOps};
use crate::zkvm::instruction_lookups::LOG_K;
use crate::zkvm::lookup_table::prefixes::Prefixes;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

use super::{PrefixCheckpoint, SparseDensePrefix};

pub enum ThreeLsbPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for ThreeLsbPrefix<XLEN> {
    fn prefix_mle<C>(
        checkpoints: &[PrefixCheckpoint<F>],
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
            debug_assert_eq!(b.len(), 0);
            checkpoints[Prefixes::ThreeLsb].unwrap()
                * (F::one() - F::from_u32(c))
                * (F::one() - r_x.unwrap())
        } else if j == 2 * XLEN - 2 {
            debug_assert_eq!(b.len(), 1);
            let bit0 = u32::from(b) & 1;
            checkpoints[Prefixes::ThreeLsb].unwrap()
                * (F::one() - F::from_u32(bit0))
                * (F::one() - F::from_u32(c))
        } else if j == 2 * XLEN - 3 {
            if suffix_len == 2 {
                debug_assert_eq!(b.len(), 0);
                F::one() - F::from_u32(c)
            } else {
                debug_assert_eq!(suffix_len, 0);
                debug_assert_eq!(b.len(), 2);
                if u32::from(b).trailing_zeros() >= 2 {
                    F::one() - F::from_u32(c)
                } else {
                    F::zero()
                }
            }
        } else if suffix_len < 3 {
            let prefix_alignment_bits = 3 - suffix_len;
            if u32::from(b).trailing_zeros() >= prefix_alignment_bits as u32 {
                F::one()
            } else {
                F::zero()
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
        if j == 2 * XLEN - 3 {
            Some(F::one() - r_y).into()
        } else if j == 2 * XLEN - 1 {
            Some(checkpoints[Prefixes::ThreeLsb].unwrap() * (F::one() - r_x) * (F::one() - r_y))
                .into()
        } else {
            checkpoints[Prefixes::ThreeLsb].into()
        }
    }
}
