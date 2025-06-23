use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};
use crate::{field::JoltField, subprotocols::sparse_dense_shout::LookupBits};

/// Left-shifts the left operand according to the bitmask given by
/// the right operand.
/// e.g. if the right operand is 0b11111000
/// then this prefix would shift the left operand by 5 to the left.
pub enum LeftShiftPrefix<const WORD_SIZE: usize> {}

impl<F: JoltField, const WORD_SIZE: usize> SparseDensePrefix<F> for LeftShiftPrefix<WORD_SIZE> {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        let mut result = checkpoints[Prefixes::LeftShift].unwrap_or(F::zero());
        let mut prod_one_plus_y = checkpoints[Prefixes::LeftShiftHelper].unwrap_or(F::one());

        if let Some(r_x) = r_x {
            result += r_x
                * (F::one() - F::from_u8(c as u8))
                * prod_one_plus_y
                * F::from_u32(1 << (WORD_SIZE - 1 - j / 2));
            prod_one_plus_y *= F::from_u8(1 + c as u8);
        } else {
            let y_msb = b.pop_msb();
            result += F::from_u8(c as u8 * (1 - y_msb))
                * prod_one_plus_y
                * F::from_u64(1 << (WORD_SIZE - 1 - j / 2));
            prod_one_plus_y *= F::from_u8(1 + y_msb);
        }

        let (x, y) = b.uninterleave();
        let (x, y_u) = (u32::from(x), u32::from(y));
        let x = x & !y_u;
        let shift = (y.leading_ones() as usize + WORD_SIZE - 1 - j / 2 - y.len()) as u32;
        result += F::from_u32(x.unbounded_shl(shift)) * prod_one_plus_y;

        result
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        let mut updated = checkpoints[Prefixes::LeftShift].unwrap_or(F::zero());
        let prod_one_plus_y = checkpoints[Prefixes::LeftShiftHelper].unwrap_or(F::one());
        updated +=
            r_x * (F::one() - r_y) * prod_one_plus_y * F::from_u64(1 << (WORD_SIZE - 1 - j / 2));
        Some(updated).into()
    }
}
