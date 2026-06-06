use jolt_field::Field;

use crate::lookup_bits::LookupBits;

use super::{PrefixCheckpoint, PrefixEval, Prefixes, SparseDensePrefix};

pub enum RightOperandIsZeroPrefix {}

impl<F: Field> SparseDensePrefix<F> for RightOperandIsZeroPrefix {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, _suffix_len: usize) -> F {
        let (_, y) = b.uninterleave();
        if u64::from(y) != 0 {
            F::zero()
        } else {
            checkpoints[Prefixes::RightOperandIsZero]
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
        let (_, y) = b.uninterleave();
        // Short-circuit if low-order bits of `y` are not 0s
        if u64::from(y) != 0 {
            return F::zero();
        }

        let mut result = checkpoints[Prefixes::RightOperandIsZero].unwrap_or(F::one());

        if r_x.is_some() {
            let y = F::from_u8(c as u8);
            result *= F::one() - y;
        } else {
            let y = F::from_u8(b.pop_msb());
            result *= F::one() - y;
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
        // checkpoint *= (1 - r_y)
        let updated =
            checkpoints[Prefixes::RightOperandIsZero].unwrap_or(F::one()) * (F::one() - r_y);
        Some(updated).into()
    }
}
