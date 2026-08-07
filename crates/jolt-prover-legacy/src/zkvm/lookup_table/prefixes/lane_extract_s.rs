use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};
use crate::utils::lookup_bits::LookupBits;

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum LaneSignTPrefix<const XLEN: usize> {}
pub enum LaneSignSPrefix {}
pub enum LastMaskBitPrefix {}

fn apply_remaining_pairs<F: JoltField>(
    b: LookupBits,
    mut previous_mask: F,
    mut apply: impl FnMut(F, u64, u64, F) -> F,
    mut result: F,
) -> F {
    let (mut x, mut y) = b.uninterleave();
    while x.len() != 0 {
        let x_i = u64::from(x.pop_msb());
        let y_i = u64::from(y.pop_msb());
        result = apply(result, x_i, y_i, previous_mask);
        previous_mask = F::from_u64(y_i);
    }
    result
}

fn current_pair<F, C>(r_x: Option<C>, c: u32, b: &mut LookupBits) -> (F, F)
where
    C: ChallengeFieldOps<F>,
    F: JoltField + FieldChallengeOps<C>,
{
    if let Some(r_x) = r_x {
        (r_x.into(), F::from_u32(c))
    } else {
        (F::from_u32(c), F::from_u8(b.pop_msb()))
    }
}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for LaneSignTPrefix<XLEN> {
    fn prefix_mle<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<C>,
        c: u32,
        mut b: LookupBits,
        _j: usize,
    ) -> F
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        let mut result = checkpoints[Prefixes::LaneSignT].unwrap_or(F::zero());
        let mut previous_mask = checkpoints[Prefixes::LastMaskBit].unwrap_or(F::zero());
        let (x_i, y_i) = current_pair(r_x, c, &mut b);
        result += F::from_u128(1u128 << XLEN) * x_i * y_i * (F::one() - previous_mask);
        previous_mask = y_i;
        apply_remaining_pairs(
            b,
            previous_mask,
            |value, x, y, previous| {
                value + F::from_u128(1u128 << XLEN) * F::from_u64(x * y) * (F::one() - previous)
            },
            result,
        )
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: C,
        r_y: C,
        _j: usize,
        _suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        let result = checkpoints[Prefixes::LaneSignT].unwrap_or(F::zero())
            + F::from_u128(1u128 << XLEN)
                * r_x
                * r_y
                * (F::one() - checkpoints[Prefixes::LastMaskBit].unwrap_or(F::zero()));
        Some(result).into()
    }
}

impl<F: JoltField> SparseDensePrefix<F> for LaneSignSPrefix {
    fn prefix_mle<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<C>,
        c: u32,
        mut b: LookupBits,
        _j: usize,
    ) -> F
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        let mut result = checkpoints[Prefixes::LaneSignS].unwrap_or(F::zero());
        let mut previous_mask = checkpoints[Prefixes::LastMaskBit].unwrap_or(F::zero());
        let (x_i, y_i) = current_pair(r_x, c, &mut b);
        result =
            result * (F::one() + y_i) - F::from_u64(2) * x_i * y_i * (F::one() - previous_mask);
        previous_mask = y_i;
        apply_remaining_pairs(
            b,
            previous_mask,
            |value, x, y, previous| {
                value * F::from_u64(1 + y) - F::from_u64(2 * x * y) * (F::one() - previous)
            },
            result,
        )
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: C,
        r_y: C,
        _j: usize,
        _suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        let result = checkpoints[Prefixes::LaneSignS].unwrap_or(F::zero()) * (F::one() + r_y)
            - F::from_u64(2)
                * r_x
                * r_y
                * (F::one() - checkpoints[Prefixes::LastMaskBit].unwrap_or(F::zero()));
        Some(result).into()
    }
}

impl<F: JoltField> SparseDensePrefix<F> for LastMaskBitPrefix {
    fn prefix_mle<C>(
        _checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<C>,
        c: u32,
        mut b: LookupBits,
        _j: usize,
    ) -> F
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        let (_, mut y_i) = current_pair(r_x, c, &mut b);
        let (_, mut y) = b.uninterleave();
        while y.len() != 0 {
            y_i = F::from_u8(y.pop_msb());
        }
        y_i
    }

    fn update_prefix_checkpoint<C>(
        _checkpoints: &[PrefixCheckpoint<F>],
        _r_x: C,
        r_y: C,
        _j: usize,
        _suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        Some(r_y.into()).into()
    }
}
