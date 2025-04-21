use crate::{field::JoltField, subprotocols::sparse_dense_shout::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum PositiveRemainderLessThanDivisorPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for PositiveRemainderLessThanDivisorPrefix {
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
            if u64::from(remainder) >= u64::from(divisor) {
                return F::zero();
            } else {
                // `c` is the sign "bit" of the remainder.
                // This prefix handles the case where both remainder and divisor
                // are positive, i.e. their sign bits are zero.
                return (F::one() - F::from_u32(c)) * (F::one() - divisor_sign);
            }
        }
        if j == 1 {
            let (remainder, divisor) = b.uninterleave();
            if u64::from(remainder) >= u64::from(divisor) {
                return F::zero();
            } else {
                // `r_x` is the sign "bit" of the remainder.
                // `c` is the sign "bit" of the divisor.
                // This prefix handles the case where both remainder and divisor
                // are positive, i.e. their sign bits are zero.
                return (F::one() - r_x.unwrap()) * (F::one() - F::from_u32(c));
            }
        }

        let mut lt = checkpoints[Prefixes::PositiveRemainderLessThanDivisor].unwrap();
        let mut eq = checkpoints[Prefixes::PositiveRemainderEqualsDivisor].unwrap();

        // For j=2 and j=3, the two checkpoints are the same (they both store isNegative(divisor))
        // so to avoid double-counting we multiply `lt` by (1 - x) * y instead of adding
        // eq * (1 - x) * y as we do in subsequent rounds.
        if j == 2 {
            let c = F::from_u32(c);
            let y_msb = F::from_u8(b.pop_msb());
            let (x, y) = b.uninterleave();
            lt *= (F::one() - c) * y_msb;
            if u64::from(x) < u64::from(y) {
                eq *= c * y_msb + (F::one() - c) * (F::one() - y_msb);
                lt += eq;
            }
            return lt;
        }
        if j == 3 {
            let r_x = r_x.unwrap();
            let c = F::from_u32(c);
            let (x, y) = b.uninterleave();
            lt *= (F::one() - r_x) * c;
            if u64::from(x) < u64::from(y) {
                eq *= r_x * c + (F::one() - r_x) * (F::one() - c);
                lt += eq;
            }
            return lt;
        }

        if let Some(r_x) = r_x {
            let c = F::from_u32(c);
            lt += eq * (F::one() - r_x) * c;
            let (x, y) = b.uninterleave();
            if u64::from(x) < u64::from(y) {
                eq *= r_x * c + (F::one() - r_x) * (F::one() - c);
                lt += eq;
            }
        } else {
            let c = F::from_u32(c);
            let y_msb = F::from_u8(b.pop_msb());
            lt += eq * (F::one() - c) * y_msb;
            let (x, y) = b.uninterleave();
            if u64::from(x) < u64::from(y) {
                eq *= c * y_msb + (F::one() - c) * (F::one() - y_msb);
                lt += eq;
            }
        }

        lt
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
            // This prefix handles the case where both remainder and divisor
            // are positive, i.e. their sign bits are zero.
            return Some((F::one() - r_x) * (F::one() - r_y)).into();
        }

        let lt_checkpoint = checkpoints[Prefixes::PositiveRemainderLessThanDivisor].unwrap();
        let eq_checkpoint = checkpoints[Prefixes::PositiveRemainderEqualsDivisor].unwrap();

        if j == 3 {
            return Some(lt_checkpoint * (F::one() - r_x) * r_y).into();
        }

        let lt_updated = lt_checkpoint + eq_checkpoint * (F::one() - r_x) * r_y;
        Some(lt_updated).into()
    }
}
