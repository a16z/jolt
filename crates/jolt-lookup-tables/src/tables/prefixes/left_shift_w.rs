use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixCheckpoint, PrefixEval, Prefixes, SparseDensePrefix};

pub enum LeftShiftWPrefix {}

impl<F: Field> SparseDensePrefix<F> for LeftShiftWPrefix {
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

        let mut result = checkpoints[Prefixes::LeftShiftW];
        let mut prod = checkpoints[Prefixes::LeftShiftWHelper];
        let bit_index_base = XLEN - 1 - j_start / 2;
        for i in 0..n {
            let x_i = (x_val >> (n - 1 - i)) & 1;
            let y_i = (y_val >> (n - 1 - i)) & 1;
            if y_i == 0 && x_i == 1 {
                result += prod * F::from_u64(1u64.wrapping_shl((bit_index_base - i) as u32));
            }
            if y_i == 1 {
                prod = prod + prod;
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

        let mut result = checkpoints[Prefixes::LeftShiftW].unwrap_or(F::zero());
        let mut prod_one_plus_y = checkpoints[Prefixes::LeftShiftWHelper].unwrap_or(F::one());

        // Calculate shift for the second half: when j >= XLEN, we're processing
        // bits from XLEN/2-1 down to 0
        let bit_index = XLEN - 1 - j / 2;

        if let Some(r_x) = r_x {
            result += r_x
                * (F::one() - F::from_u8(c as u8))
                * prod_one_plus_y
                * F::from_u64(1u64.wrapping_shl(bit_index as u32));
            prod_one_plus_y *= F::from_u8(1 + c as u8);
        } else {
            let y_msb = b.pop_msb();
            result += F::from_u8(c as u8 * (1 - y_msb))
                * prod_one_plus_y
                * F::from_u64(1u64.wrapping_shl(bit_index as u32));
            prod_one_plus_y *= F::from_u8(1 + y_msb);
        }

        let (x, y) = b.uninterleave();
        let (x, y_u) = (u64::from(x), u64::from(y));
        let x = x & !y_u;
        let shift = (y.leading_ones() as usize + bit_index - y.len()) as u32;
        result += F::from_u64(x.unbounded_shl(shift)) * prod_one_plus_y;

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
            let mut updated = checkpoints[Prefixes::LeftShiftW].unwrap_or(F::zero());
            let prod_one_plus_y = checkpoints[Prefixes::LeftShiftWHelper].unwrap_or(F::one());
            let bit_index = XLEN - 1 - j / 2;
            updated += r_x
                * (F::one() - r_y)
                * prod_one_plus_y
                * F::from_u64(1u64.wrapping_shl(bit_index as u32));
            Some(updated).into()
        } else {
            Some(F::zero()).into()
        }
    }
}
