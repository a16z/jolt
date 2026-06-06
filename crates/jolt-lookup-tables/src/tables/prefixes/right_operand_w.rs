use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixCheckpoint, PrefixEval, Prefixes, SparseDensePrefix, LOG_K};

pub enum RightOperandWPrefix {}

impl<F: Field> SparseDensePrefix<F> for RightOperandWPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let mut result = checkpoints[Prefixes::RightOperandW];
        if suffix_len < XLEN {
            let (_, y) = b.uninterleave();
            result += F::from_u128(u128::from(y) << (suffix_len / 2));
        }
        result
    }

    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F {
        let _ = (checkpoints, r_x, c, b, j);
        let suffix_len = LOG_K - j - b.len() - 1;
        let mut result = checkpoints[Prefixes::RightOperandW].unwrap_or(F::zero());

        if j % 2 == 1 && j > XLEN {
            // c is of the right operand
            let shift = XLEN - 1 - j / 2;
            result += F::from_u128((c as u128) << shift);
        }

        if suffix_len < XLEN {
            let (_x, y) = b.uninterleave();
            result += F::from_u128(u128::from(y) << (suffix_len / 2));
        }

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
        if j > XLEN {
            let shift = XLEN - 1 - j / 2;
            let updated = checkpoints[Prefixes::RightOperandW].unwrap_or(F::zero())
                + (F::from_u64(1 << shift) * r_y);
            Some(updated).into()
        } else {
            checkpoints[Prefixes::RightOperandW].into()
        }
    }
}
