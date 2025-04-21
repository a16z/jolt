use crate::{field::JoltField, subprotocols::sparse_dense_shout::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum NegativeDivisorGreaterThanRemainderPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for NegativeDivisorGreaterThanRemainderPrefix {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        if j == 0 {
            let divisor_sign = F::from_u8(b.pop_msb());
            let (remainder, divisor) = b.uninterleave();
            if u64::from(remainder) <= u64::from(divisor) {
                return F::zero();
            } else {
                // `c` is the sign "bit" of the remainder.
                // This prefix handles the case where both remainder and
                // divisor are negative, i.e. their sign bits are one.
                return F::from_u32(c) * divisor_sign;
            }
        }
        if j == 1 {
            let (remainder, divisor) = b.uninterleave();
            if u64::from(remainder) <= u64::from(divisor) {
                return F::zero();
            } else {
                // `r_x` is the sign "bit" of the remainder.
                // `c` is the sign "bit" of the divisor.
                // This prefix handles the case where both remainder and
                // divisor are negative, i.e. their sign bits are one.
                return r_x.unwrap() * F::from_u32(c);
            }
        }

        let mut gt = checkpoints[Prefixes::NegativeDivisorGreaterThanRemainder].unwrap();
        let mut eq = checkpoints[Prefixes::NegativeDivisorEqualsRemainder].unwrap();

        // For j=2 and j=3, the two checkpoints are the same (they both store isNegative(divisor))
        // so to avoid double-counting we multiply `gt` by x * (1 - y) instead of adding
        // eq * x * (1 - y) as we do in subsequent rounds.
        if j == 2 {
            let c = F::from_u32(c);
            let y_msb = F::from_u8(b.pop_msb());
            let (x, y) = b.uninterleave();
            gt *= c * (F::one() - y_msb);
            if u64::from(x) > u64::from(y) {
                eq *= c * y_msb + (F::one() - c) * (F::one() - y_msb);
                gt += eq;
            }
            return gt;
        }
        if j == 3 {
            let r_x = r_x.unwrap();
            let c = F::from_u32(c);
            let (x, y) = b.uninterleave();
            gt *= r_x * (F::one() - c);
            if u64::from(x) > u64::from(y) {
                eq *= r_x * c + (F::one() - r_x) * (F::one() - c);
                gt += eq;
            }
            return gt;
        }

        if let Some(r_x) = r_x {
            let c = F::from_u32(c);
            gt += eq * r_x * (F::one() - c);
            let (x, y) = b.uninterleave();
            if u64::from(x) > u64::from(y) {
                eq *= r_x * c + (F::one() - r_x) * (F::one() - c);
                gt += eq;
            }
        } else {
            let c = F::from_u32(c);
            let y_msb = F::from_u8(b.pop_msb());
            gt += eq * c * (F::one() - y_msb);
            let (x, y) = b.uninterleave();
            if u64::from(x) > u64::from(y) {
                eq *= c * y_msb + (F::one() - c) * (F::one() - y_msb);
                gt += eq;
            }
        }

        gt
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        if j == 1 {
            // `r_x` is the sign bit of the remainder
            // `r_y` is the sign bit of the divisor
            // This prefix handles the case where both remainder and
            // divisor are negative.
            return Some(r_x * r_y).into();
        }

        let gt_checkpoint = checkpoints[Prefixes::NegativeDivisorGreaterThanRemainder].unwrap();
        let eq_checkpoint = checkpoints[Prefixes::NegativeDivisorEqualsRemainder].unwrap();

        if j == 3 {
            return Some(gt_checkpoint * r_x * (F::one() - r_y)).into();
        }

        let gt_updated = gt_checkpoint + eq_checkpoint * r_x * (F::one() - r_y);
        Some(gt_updated).into()
    }
}
