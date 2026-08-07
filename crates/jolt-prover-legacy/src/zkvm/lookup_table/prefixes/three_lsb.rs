use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};
use crate::utils::lookup_bits::LookupBits;
use crate::zkvm::instruction_lookups::LOG_K;

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

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
        if suffix_len != 0 {
            return F::zero();
        }
        if b.len() >= 3 {
            return F::from_u64((b & 7) as u64);
        }

        let mut result = checkpoints[Prefixes::ThreeLsb].unwrap_or(F::zero());
        result += F::from_u64((b & 7) as u64);
        result += F::from_u64(1 << b.len()) * F::from_u32(c);
        if let Some(r_x) = r_x.filter(|_| b.len() + 1 < 3) {
            result += F::from_u64(1 << (b.len() + 1)) * r_x;
        }
        result
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
            Some(F::from_u64(4) * r_y).into()
        } else if j == 2 * XLEN - 1 {
            let result =
                checkpoints[Prefixes::ThreeLsb].unwrap_or(F::zero()) + F::from_u64(2) * r_x + r_y;
            Some(result).into()
        } else {
            None.into()
        }
    }
}
