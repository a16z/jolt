use crate::{field::JoltField, utils::lookup_bits::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum RightOperandIsZeroPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for RightOperandIsZeroPrefix {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F::Challenge>,
        c: u32,
        mut b: LookupBits,
        _: usize,
    ) -> F {
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
        _: F::Challenge,
        r_y: F::Challenge,
        _: usize,
    ) -> PrefixCheckpoint<F> {
        // checkpoint *= (1 - r_y)
        let updated =
            checkpoints[Prefixes::RightOperandIsZero].unwrap_or(F::one()) * (F::one() - r_y);
        Some(updated).into()
    }
    fn update_prefix_checkpoint_field(
        checkpoints: &[PrefixCheckpoint<F>],
        _r_x: F,
        r_y: F,
        _j: usize,
    ) -> PrefixCheckpoint<F> {
        // checkpoint *= (1 - r_y)
        let updated =
            checkpoints[Prefixes::RightOperandIsZero].unwrap_or(F::one()) * (F::one() - r_y);
        Some(updated).into()
    }

    fn prefix_mle_field(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        _: usize,
    ) -> F {
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
}
