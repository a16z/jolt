use crate::field::{ChallengeFieldOps, FieldChallengeOps};
use crate::zkvm::instruction_lookups::LOG_K;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum LowerHalfWordPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for LowerHalfWordPrefix<XLEN> {
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
        let half_word_size = XLEN / 2;
        // Ignore high-order variables (those above the half-word boundary)
        if j < XLEN + half_word_size {
            return F::zero();
        }
        let mut result = checkpoints[Prefixes::LowerHalfWord].unwrap_or(F::zero());

        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            let x_shift = 2 * XLEN - j;
            let y_shift = 2 * XLEN - j - 1;
            result += F::from_u128(1u128 << x_shift) * r_x;
            result += F::from_u128(1u128 << y_shift) * y;
        } else {
            let x = F::from_u8(c as u8);
            let y_msb = b.pop_msb();
            let x_shift = 2 * XLEN - j - 1;
            let y_shift = 2 * XLEN - j - 2;
            result += F::from_u128(1 << x_shift) * x;
            result += F::from_u128(1 << y_shift) * F::from_u8(y_msb);
        }

        // Add in low-order bits from `b`
        result += F::from_u128(u128::from(b) << suffix_len);

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
        let half_word_size = XLEN / 2;
        if j < XLEN + half_word_size {
            return None.into();
        }
        let x_shift = 2 * XLEN - j;
        let y_shift = 2 * XLEN - j - 1;
        let mut updated = checkpoints[Prefixes::LowerHalfWord].unwrap_or(F::zero());
        updated += F::from_u128(1 << x_shift) * r_x;
        updated += F::from_u128(1 << y_shift) * r_y;
        Some(updated).into()
    }
}
