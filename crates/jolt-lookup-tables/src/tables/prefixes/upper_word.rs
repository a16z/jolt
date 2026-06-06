use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixCheckpoint, PrefixEval, Prefixes, SparseDensePrefix, LOG_K};

pub enum UpperWordPrefix {}

impl<F: Field> SparseDensePrefix<F> for UpperWordPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let j_start = 2 * XLEN - suffix_len - b.len();
        if j_start >= XLEN {
            return checkpoints[Prefixes::UpperWord];
        }

        let mut result = checkpoints[Prefixes::UpperWord];
        if suffix_len > XLEN {
            result += F::from_u64(u64::from(b) << (suffix_len - XLEN));
        } else {
            let (b_high, _) = b.split(XLEN - suffix_len);
            result += F::from_u64(u64::from(b_high));
        }
        result
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
        let mut result = checkpoints[Prefixes::UpperWord].unwrap_or(F::zero());
        // Ignore low-order variables
        if j >= XLEN {
            return result;
        }

        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            let x_shift = XLEN - j;
            let y_shift = XLEN - j - 1;
            result += F::from_u64(1 << x_shift) * r_x;
            result += F::from_u64(1 << y_shift) * y;
        } else {
            let x = F::from_u8(c as u8);
            let y_msb = b.pop_msb();
            let x_shift = XLEN - j - 1;
            let y_shift = XLEN - j - 2;
            result += F::from_u64(1 << x_shift) * x;
            result += F::from_u64(1 << y_shift) * F::from_u8(y_msb);
        }

        // Add in bits of `b` that fall in upper word
        if suffix_len > XLEN {
            result += F::from_u64(u64::from(b) << (suffix_len - XLEN));
        } else {
            let (b_high, _) = b.split(XLEN - suffix_len);
            result += F::from_u64(u64::from(b_high));
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
        if j >= XLEN {
            return checkpoints[Prefixes::UpperWord].into();
        }
        let x_shift = XLEN - j;
        let y_shift = XLEN - j - 1;
        let updated = checkpoints[Prefixes::UpperWord].unwrap_or(F::zero())
            + F::from_u64(1 << x_shift) * r_x
            + F::from_u64(1 << y_shift) * r_y;
        Some(updated).into()
    }
}
