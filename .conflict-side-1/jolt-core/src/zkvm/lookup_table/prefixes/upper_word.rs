use crate::field::{ChallengeFieldOps, FieldChallengeOps};
use crate::zkvm::instruction_lookups::LOG_K;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

#[derive(Default)]
pub struct UpperWordPrefix<const XLEN: usize>;

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for UpperWordPrefix<XLEN> {
    fn prefix_mle<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<C>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        let suffix_len = LOG_K - j - b.len() - 1;
        let mut result = checkpoints[Prefixes::UpperWord].unwrap_or(F::zero());
        // Ignore low-order variables
        if j >= XLEN {
            return result;
        }

        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            let x_shift = XLEN - j;
            let y_shift = XLEN - j - 1;
            result += F::from_u64(1 << x_shift) * r_x;
            result += F::from_u64(1 << y_shift) * y;
        } else {
            let x = F::from_u8(c as u8);
            let y_msb = b.pop_msb();
            let x_shift = XLEN - j - 1;
            let y_shift = XLEN - j - 2;
            result += F::from_u64(1 << x_shift) * x;
            result += F::from_u64(1 << y_shift) * F::from_u8(y_msb);
        }

        // Add in bits of `b` that fall in upper word
        if suffix_len > XLEN {
            result += F::from_u64(u64::from(b) << (suffix_len - XLEN));
        } else {
            let (b_high, _) = b.split(XLEN - suffix_len);
            result += F::from_u64(u64::from(b_high));
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
        if j >= XLEN {
            return checkpoints[Prefixes::UpperWord].into();
        }
        let x_shift = XLEN - j;
        let y_shift = XLEN - j - 1;
        let updated = checkpoints[Prefixes::UpperWord].unwrap_or(F::zero())
            + F::from_u64(1 << x_shift) * r_x
            + F::from_u64(1 << y_shift) * r_y;
        Some(updated).into()
    }
}
