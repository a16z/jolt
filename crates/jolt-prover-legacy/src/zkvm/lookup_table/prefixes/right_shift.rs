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
        let (mut x, mut y) = b.uninterleave();
        let mut suffix = 0;
        let mut selected = 0;
        while x.len() != 0 {
            let x_i = x.pop_msb();
            let y_i = y.pop_msb();
            suffix *= 1 + u32::from(y_i);
            suffix += u32::from(x_i * y_i);
            selected += u32::from(y_i);
        }
        result *= F::from_u32(1 << selected);
        result += F::from_u32(suffix);

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
