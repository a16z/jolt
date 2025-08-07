use crate::{field::JoltField, utils::lookup_bits::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum ChangeDivisorWPrefix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize, F: JoltField> SparseDensePrefix<F>
    for ChangeDivisorWPrefix<WORD_SIZE>
{
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        let mut result = checkpoints[Prefixes::ChangeDivisorW]
            .unwrap_or(F::from_u64(2) - F::from_u128(1u128 << WORD_SIZE));
        if j == WORD_SIZE {
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
            if j > WORD_SIZE {
                let (x, y) = b.uninterleave();
                if u64::from(x) != 0 || u64::from(y) != (1u64 << y.len()) - 1 || c == 0 {
                    return F::zero();
                }
                result *= (F::one() - r_x) * F::from_u64(c as u64);
            }
        } else if j > WORD_SIZE {
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
        r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        let updated = checkpoints[Prefixes::ChangeDivisorW]
            .unwrap_or(F::from_u64(2) - F::from_u128(1u128 << WORD_SIZE))
            * if j == WORD_SIZE + 1 {
                r_x * r_y
            } else if j > WORD_SIZE + 1 {
                (F::one() - r_x) * r_y
            } else {
                F::one()
            };
        Some(updated).into()
    }
}
