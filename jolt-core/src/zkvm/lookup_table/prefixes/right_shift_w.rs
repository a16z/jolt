use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

/// Right-shifts the left operand according to the bitmask given by
/// the right operand, processing the second half of bits (j > XLEN).
pub enum RightShiftWPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for RightShiftWPrefix<XLEN> {
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
        // Only process when j >= XLEN
        if j < XLEN {
            return F::zero();
        }

        let mut result = checkpoints[Prefixes::RightShiftW].unwrap_or(F::zero());
        if let Some(r_x) = r_x {
            result *= F::from_u32(1 + c);
            result += r_x * F::from_u32(c);
        } else {
            let y_msb = b.pop_msb();
            result *= F::from_u8(1 + y_msb);
            result += F::from_u8(c as u8 * y_msb);
        }
        let (x, y) = b.uninterleave();
        result *= F::from_u32(1 << y.leading_ones());
        result += F::from_u32(u32::from(x) >> y.trailing_zeros());

        result
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: C,
        r_y: C,
        j: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        if j >= XLEN {
            let mut updated = checkpoints[Prefixes::RightShiftW].unwrap_or(F::zero());
            updated *= F::one() + r_y;
            updated += r_x * r_y;
            Some(updated).into()
        } else {
            Some(F::zero()).into()
        }
    }
}
