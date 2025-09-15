use crate::{field::JoltField, utils::lookup_bits::LookupBits};
use crate::field::MontU128;
use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum ChangeDivisorWPrefix<const XLEN: usize> {}

/// Calculates the prefix for the change_divisor_w instruction
/// Equivalently, it's a (2 - 2^XLEN) * eq(x, 100...000) * eq(y, 111...111)
/// where x and y are the lower word parts of operands
impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for ChangeDivisorWPrefix<XLEN> {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<MontU128>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        let mut result = checkpoints[Prefixes::ChangeDivisorW]
            .unwrap_or(F::from_u64(2) - F::from_u128(1u128 << XLEN));
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
                result *= (F::one() - F::from_u128_mont(r_x)) * F::from_u64(c as u64);
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

    fn prefix_mle_field(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        let mut result = checkpoints[Prefixes::ChangeDivisorW]
            .unwrap_or(F::from_u64(2) - F::from_u128(1u128 << XLEN));
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
                result *= (F::one() - r_x) * F::from_u64(c as u64);
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

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: MontU128,
        r_y: MontU128,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        let updated = checkpoints[Prefixes::ChangeDivisorW]
            .unwrap_or(F::from_u64(2) - F::from_u128(1u128 << XLEN))
            * if j == XLEN + 1 {
                F::from_u128_mont(r_x).mul_u128_mont_form(r_y)
            } else if j > XLEN + 1 {
                (F::one() - F::from_u128_mont(r_x)).mul_u128_mont_form(r_y)
            } else {
                F::one()
            };
        Some(updated).into()
    }

    fn update_prefix_checkpoint_field(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        let updated = checkpoints[Prefixes::ChangeDivisorW]
            .unwrap_or(F::from_u64(2) - F::from_u128(1u128 << XLEN))
            * if j == XLEN + 1 {
            r_x*r_y
        } else if j > XLEN + 1 {
            (F::one() - r_x)*r_y
        } else {
            F::one()
        };
        Some(updated).into()
    }
}
