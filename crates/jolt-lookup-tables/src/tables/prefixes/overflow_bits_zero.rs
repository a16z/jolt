use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixCheckpoint, PrefixEval, Prefixes, SparseDensePrefix, LOG_K};

pub enum OverflowBitsZeroPrefix {}

impl<F: Field> SparseDensePrefix<F> for OverflowBitsZeroPrefix {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let j_start = 2 * XLEN - suffix_len - b.len();
        if j_start >= 128 - XLEN {
            return checkpoints[Prefixes::OverflowBitsZero];
        }

        // Overflow region = interleaved positions 0..XLEN.
        // Phase bits in overflow = top portion of `b`.
        let overflow_bits = if suffix_len >= XLEN {
            u128::from(b)
        } else {
            u128::from(b) >> (XLEN - suffix_len)
        };

        if overflow_bits != 0 {
            F::zero()
        } else {
            checkpoints[Prefixes::OverflowBitsZero]
        }
    }

    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        let _ = (checkpoints, r_x, c, b, j);
        let suffix_len = LOG_K - j - b.len() - 1;
        if j >= 128 - XLEN {
            return checkpoints[Prefixes::OverflowBitsZero].unwrap_or(F::one());
        }

        let mut result = checkpoints[Prefixes::OverflowBitsZero].unwrap_or(F::one());

        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            result *= (F::one() - r_x) * (F::one() - y);
        } else {
            let x = F::from_u32(c);
            let y = F::from_u8(b.pop_msb());
            result *= (F::one() - x) * (F::one() - y);
        }

        let rest = u128::from(b);
        let temp = F::from_u64((((rest << suffix_len) >> XLEN) == 0) as u64);
        result *= temp;

        result
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
        suffix_len: usize,
    ) -> PrefixCheckpoint<F> {
        let _ = (checkpoints, r_x, r_y, j, suffix_len);
        if j >= 128 - XLEN {
            return checkpoints[Prefixes::OverflowBitsZero].into();
        }
        let updated = checkpoints[Prefixes::OverflowBitsZero].unwrap_or(F::one())
            * (F::one() - r_x)
            * (F::one() - r_y);

        Some(updated).into()
    }
}
