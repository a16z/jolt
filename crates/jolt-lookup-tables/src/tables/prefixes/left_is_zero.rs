use jolt_field::Field;

use crate::lookup_bits::LookupBits;

use super::{PrefixCheckpoint, PrefixEval, Prefixes, SparseDensePrefix};

pub enum LeftOperandIsZeroPrefix {}

impl<F: Field> SparseDensePrefix<F> for LeftOperandIsZeroPrefix {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, _suffix_len: usize) -> F {
        let (x, _) = b.uninterleave();
        if u64::from(x) != 0 {
            F::zero()
        } else {
            checkpoints[Prefixes::LeftOperandIsZero]
        }
    }

    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F {
        let _ = (checkpoints, r_x, c, b, j);
        let (x, _) = b.uninterleave();
        // Short-circuit if low-order bits of `x` are not 0s
        if u64::from(x) != 0 {
            return F::zero();
        }

        let mut result = checkpoints[Prefixes::LeftOperandIsZero].unwrap_or(F::one());

        if let Some(r_x) = r_x {
            result *= F::one() - r_x;
        } else {
            let x = F::from_u8(c as u8);
            result *= F::one() - x;
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
        // checkpoint *= (1 - r_x)
        let updated =
            checkpoints[Prefixes::LeftOperandIsZero].unwrap_or(F::one()) * (F::one() - r_x);
        Some(updated).into()
    }
}
