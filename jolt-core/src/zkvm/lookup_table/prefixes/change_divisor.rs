use crate::{field::JoltField, utils::lookup_bits::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum ChangeDivisorPrefix<const XLEN: usize> {}

/// Calculates the prefix for the change_divisor instruction
/// Equivalently, it's a (2 - 2^XLEN) * eq(x, 100...000) * eq(y, 111...111)
impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for ChangeDivisorPrefix<XLEN> {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F::Challenge>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        let mut result = checkpoints[Prefixes::ChangeDivisor]
            .unwrap_or(F::from_u64(2) - F::from_u128(1u128 << XLEN));
        if j == 0 {
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
            let (x, y) = b.uninterleave();
            if u64::from(x) != 0 || u64::from(y) != (1u64 << y.len()) - 1 || c == 0 {
                return F::zero();
            }
            result *= (F::one() - r_x) * F::from_u64(c as u64);
        } else {
            let (x, y) = b.uninterleave();
            if b.len() > 0 && u64::from(x) != 0 || u64::from(y) != (1u64 << y.len()) - 1 {
                return F::zero();
            }
            result *= F::one() - F::from_u64(c as u64);
        }
        result
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F::Challenge,
        r_y: F::Challenge,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        let updated = checkpoints[Prefixes::ChangeDivisor]
            .unwrap_or(F::from_u64(2) - F::from_u128(1u128 << XLEN))
            * if j == 1 {
                r_x * r_y
            } else {
                (F::one() - r_x) * r_y
            };
        Some(updated).into()
    }

    fn prefix_mle_field(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        let mut result = checkpoints[Prefixes::ChangeDivisor]
            .unwrap_or(F::from_u64(2) - F::from_u128(1u128 << XLEN));
        if j == 0 {
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
            let (x, y) = b.uninterleave();
            if u64::from(x) != 0 || u64::from(y) != (1u64 << y.len()) - 1 || c == 0 {
                return F::zero();
            }
            result *= (F::one() - r_x) * F::from_u64(c as u64);
        } else {
            let (x, y) = b.uninterleave();
            if b.len() > 0 && u64::from(x) != 0 || u64::from(y) != (1u64 << y.len()) - 1 {
                return F::zero();
            }
            result *= F::one() - F::from_u64(c as u64);
        }
        result
    }

    fn update_prefix_checkpoint_field(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        let updated = checkpoints[Prefixes::ChangeDivisor]
            .unwrap_or(F::from_u64(2) - F::from_u128(1u128 << XLEN))
            * if j == 1 {
                r_x * r_y
            } else {
                (F::one() - r_x) * r_y
            };
        Some(updated).into()
    }
}
