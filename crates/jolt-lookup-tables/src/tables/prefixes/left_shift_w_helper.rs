use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixCheckpoint, PrefixEval, Prefixes, SparseDensePrefix};

pub enum LeftShiftWHelperPrefix {}

impl<F: Field> SparseDensePrefix<F> for LeftShiftWHelperPrefix {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let j_start = 2 * XLEN - suffix_len - b.len();
        if j_start < XLEN {
            return F::one();
        }

        let (_x, y) = b.uninterleave();
        checkpoints[Prefixes::LeftShiftWHelper] * F::from_u64(1u64 << u64::from(y).count_ones())
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
            return F::one();
        }

        let mut result = checkpoints[Prefixes::LeftShiftWHelper].unwrap_or(F::one());

        if r_x.is_some() {
            result *= F::from_u32(1 + c);
        } else {
            let y_msb = b.pop_msb();
            result *= F::from_u8(1 + y_msb);
        }

        let (_, y) = b.uninterleave();
        result *= F::from_u32(1 << y.leading_ones());

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
            let mut updated = checkpoints[Prefixes::LeftShiftWHelper].unwrap_or(F::one());
            updated *= F::one() + r_y;
            Some(updated).into()
        } else {
            Some(F::one()).into()
        }
    }
}
