use jolt_field::Field;

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::lookup_bits::LookupBits;

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum LeftShiftWPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: Field> SparseDensePrefix<F> for LeftShiftWPrefix<XLEN> {
    fn prefix_mle<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<C>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F
    where
        C: ChallengeOps<F>,
        F: FieldOps<C>,
    {
        if j < XLEN {
            return F::zero();
        }

        let mut result = checkpoints[Prefixes::LeftShiftW].unwrap_or(F::zero());
        let mut prod_one_plus_y = checkpoints[Prefixes::LeftShiftWHelper].unwrap_or(F::one());

        // When j >= XLEN, we're processing bits from XLEN/2-1 down to 0
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

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: C,
        r_y: C,
        j: usize,
        _suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeOps<F>,
        F: FieldOps<C>,
    {
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
