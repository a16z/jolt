use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixCheckpoint, PrefixEval, Prefixes, SparseDensePrefix};

pub enum RightShiftWPrefix {}

impl<F: Field> SparseDensePrefix<F> for RightShiftWPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let j_start = 2 * XLEN - suffix_len - b.len();
        if j_start < XLEN {
            return F::zero();
        }

        let (x, y) = b.uninterleave();
        let (x_val, y_val) = (u64::from(x), u64::from(y));
        let n = y.len();

        let mut result = checkpoints[Prefixes::RightShiftW];
        for i in 0..n {
            let x_i = (x_val >> (n - 1 - i)) & 1;
            let y_i = (y_val >> (n - 1 - i)) & 1;
            if y_i == 1 {
                result = result + result + F::from_u64(x_i);
            }
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
        // Only process when j >= XLEN
        if j < XLEN {
            return F::zero();
        }

        let mut result = checkpoints[Prefixes::RightShiftW].unwrap_or(F::zero());
        if let Some(r_x) = r_x {
            result *= F::from_u32(1 + c);
            result += r_x * F::from_u32(c);
        } else {
            let y_msb = b.pop_msb();
            result *= F::from_u8(1 + y_msb);
            result += F::from_u8(c as u8 * y_msb);
        }
        let (x, y) = b.uninterleave();
        result *= F::from_u32(1 << y.leading_ones());
        result += F::from_u32(u32::from(x) >> y.trailing_zeros());

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
            let mut updated = checkpoints[Prefixes::RightShiftW].unwrap_or(F::zero());
            updated *= F::one() + r_y;
            updated += r_x * r_y;
            Some(updated).into()
        } else {
            Some(F::zero()).into()
        }
    }
}
