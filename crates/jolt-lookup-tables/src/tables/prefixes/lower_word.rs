use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixCheckpoint, PrefixEval, Prefixes, SparseDensePrefix, LOG_K};

pub enum LowerWordPrefix {}

impl<F: Field> SparseDensePrefix<F> for LowerWordPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let j_start = 2 * XLEN - suffix_len - b.len();
        if j_start < XLEN {
            return F::zero();
        }
        checkpoints[Prefixes::LowerWord] + F::from_u128(u128::from(b) << suffix_len)
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
        // Ignore high-order variables
        if j < XLEN {
            return F::zero();
        }
        let mut result = checkpoints[Prefixes::LowerWord].unwrap_or(F::zero());

        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            let x_shift = 2 * XLEN - j;
            let y_shift = 2 * XLEN - j - 1;
            result += F::from_u128(1u128 << x_shift) * r_x;
            result += F::from_u128(1u128 << y_shift) * y;
        } else {
            let x = F::from_u8(c as u8);
            let y_msb = b.pop_msb();
            let x_shift = 2 * XLEN - j - 1;
            let y_shift = 2 * XLEN - j - 2;
            result += F::from_u128(1 << x_shift) * x;
            result += F::from_u128(1 << y_shift) * F::from_u8(y_msb);
        }

        // Add in low-order bits from `b`
        result += F::from_u128(u128::from(b) << suffix_len);

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
        if j < XLEN {
            return None.into();
        }
        let x_shift = 2 * XLEN - j;
        let y_shift = 2 * XLEN - j - 1;
        let mut updated = checkpoints[Prefixes::LowerWord].unwrap_or(F::zero());
        updated += F::from_u128(1 << x_shift) * r_x;
        updated += F::from_u128(1 << y_shift) * r_y;
        Some(updated).into()
    }
}
