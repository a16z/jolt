use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};
use crate::field::MontU128;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

/// Right-shifts the left operand according to the bitmask given by
/// the right operand.
/// e.g. if the right operand is 0b11100000
/// then this suffix would shift the left operand by 5.
pub enum RightShiftPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for RightShiftPrefix {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<MontU128>,
        c: u32,
        mut b: LookupBits,
        _: usize,
    ) -> F {
        let mut result = checkpoints[Prefixes::RightShift].unwrap_or(F::zero());
        if let Some(r_x) = r_x {
            result *= F::from_u32(1 + c);
            result += F::from_u32(c).mul_u128_mont_form(r_x);
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

    fn prefix_mle_field(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        _: usize,
    ) -> F {
        let mut result = checkpoints[Prefixes::RightShift].unwrap_or(F::zero());
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
        r_x: MontU128,
        r_y: MontU128,
        _: usize,
    ) -> PrefixCheckpoint<F> {
        let mut updated = checkpoints[Prefixes::RightShift].unwrap_or(F::zero());
        updated *= F::one() + F::from_u128_mont(r_y);
        updated += F::from_u128_mont(r_x) * F::from_u128_mont(r_y);
        Some(updated).into()
    }

    fn update_prefix_checkpoint_field(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        _: usize,
    ) -> PrefixCheckpoint<F> {
        let mut updated = checkpoints[Prefixes::RightShift].unwrap_or(F::zero());
        updated *= F::one() + r_y;
        updated += r_x * r_y;
        Some(updated).into()
    }
}
