use crate::zkvm::instruction_lookups::read_raf_checking::current_suffix_len;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum RightOperandWPrefix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize, F: JoltField> SparseDensePrefix<F> for RightOperandWPrefix<WORD_SIZE> {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        _r_x: Option<F>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F {
        let mut result = checkpoints[Prefixes::RightOperandW].unwrap_or(F::zero());

        if j % 2 == 1 && j > WORD_SIZE {
            // c is of the right operand
            let shift = WORD_SIZE - 1 - j / 2;
            result += F::from_u128((c as u128) << shift);
        }

        let suffix_len = current_suffix_len(j);
        if suffix_len < WORD_SIZE {
            let (_x, y) = b.uninterleave();
            result += F::from_u128(u128::from(y) << (suffix_len / 2));
        }

        result
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        _r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        if j > WORD_SIZE {
            let shift = WORD_SIZE - 1 - j / 2;
            let updated = checkpoints[Prefixes::RightOperandW].unwrap_or(F::zero())
                + (F::from_u64(1 << shift) * r_y);
            Some(updated).into()
        } else {
            checkpoints[Prefixes::RightOperandW].into()
        }
    }
}
