use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum ChangeDivisorWPrefix<const XLEN: usize> {}

/// Calculates the prefix for the change_divisor_w instruction
/// Equivalently, it's a (2 - 2^XLEN) * eq(x, 100...000) * eq(y, 111...111)
/// where x and y are the lower word parts of operands
impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for ChangeDivisorWPrefix<XLEN> {
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
        if j < XLEN {
            return F::zero();
        }

        let mut result = if j == XLEN || j == XLEN + 1 {
            F::from_u64(2) - F::from_u128(1u128 << XLEN)
        } else {
            checkpoints[Prefixes::ChangeDivisorW].unwrap()
        };

        if j == XLEN {
            let x_msb = b.pop_msb() as u32;
            if x_msb == 0 {
                return F::zero();
            }
            let (x, y) = b.uninterleave();
            if u64::from(x) != 0 || u64::from(y) != (1u64 << y.len()) - 1 {
                return F::zero();
            }
            result = result.mul_u64(c as u64);
        } else if let Some(r_x) = r_x {
            if j > XLEN {
                let (x, y) = b.uninterleave();
                if u64::from(x) != 0 || u64::from(y) != (1u64 << y.len()) - 1 || c == 0 {
                    return F::zero();
                }

                if j == XLEN + 1 {
                    result *= (r_x) * F::from_u64(c as u64);
                } else {
                    result *= (F::one() - r_x) * F::from_u64(c as u64);
                }
            }
        } else if j > XLEN {
            let (x, y) = b.uninterleave();
            if b.len() > 0 && u64::from(x) != 0 || u64::from(y) != (1u64 << y.len()) - 1 {
                return F::zero();
            }
            result *= F::one() - F::from_u64(c as u64);
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
        if j < XLEN {
            return Some(F::zero()).into();
        }

        let updated = if j == XLEN + 1 {
            (F::from_u64(2) - F::from_u128(1u128 << XLEN)) * r_x * r_y
        } else {
            checkpoints[Prefixes::ChangeDivisorW].unwrap() * ((F::one() - r_x) * r_y)
        };
        Some(updated).into()
    }
}
