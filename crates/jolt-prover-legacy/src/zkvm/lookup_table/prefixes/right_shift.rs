use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

/// Right-shifts the left operand according to the bitmask given by
/// the right operand.
/// e.g. if the right operand is 0b11100000
/// then this suffix would shift the left operand by 5.
pub enum RightShiftPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for RightShiftPrefix {
    fn prefix_mle<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<C>,
        c: u32,
        mut b: LookupBits,
        _: usize,
    ) -> F
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        let mut result = checkpoints[Prefixes::RightShift].unwrap_or(F::zero());
        if let Some(r_x) = r_x {
            result *= F::from_u32(1 + c);
            result += r_x * F::from_u32(c);
        } else {
            let y_msb = b.pop_msb();
            result *= F::from_u8(1 + y_msb);
            result += F::from_u8(c as u8 * y_msb);
        }
        // Faithful form of the per-round recurrence at binary points
        //   result = result·(1+y_i) + x_i·y_i,
        // i.e. result·2^popcount(y) + pext(x, y). Valid for any mask shape;
        // for contiguous masks 1..10..0 it agrees with the previous
        // leading-ones/trailing-zeros shortcut.
        let (x, y) = b.uninterleave();
        let (x_val, y_val) = (u64::from(x), u64::from(y));
        let mut pext = 0u64;
        for i in (0..y.len()).rev() {
            if (y_val >> i) & 1 == 1 {
                pext = (pext << 1) | ((x_val >> i) & 1);
            }
        }
        result *= F::from_u64(1 << y_val.count_ones());
        result += F::from_u64(pext);

        result
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: C,
        r_y: C,
        _: usize,
        _suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        let mut updated = checkpoints[Prefixes::RightShift].unwrap_or(F::zero());
        updated *= F::one() + r_y;
        updated += r_x * r_y;
        Some(updated).into()
    }
}
