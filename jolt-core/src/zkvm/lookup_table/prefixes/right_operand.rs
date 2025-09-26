use crate::zkvm::instruction_lookups::read_raf_checking::current_suffix_len;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum RightOperandPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for RightOperandPrefix<XLEN> {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        _r_x: Option<F::Challenge>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F {
        let mut result = checkpoints[Prefixes::RightOperand].unwrap_or(F::zero());

        if j % 2 == 1 {
            // c is of the right operand
            let shift = XLEN - 1 - j / 2;
            result += F::from_u128((c as u128) << shift);
        }

        let (_x, y) = b.uninterleave();
        let suffix_len = current_suffix_len(j);
        result += F::from_u128(u128::from(y) << (suffix_len / 2));

        result
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        _r_x: F::Challenge,
        r_y: F::Challenge,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        let shift = XLEN - 1 - j / 2;
        let updated = checkpoints[Prefixes::RightOperand].unwrap_or(F::zero())
            + (F::from_u64(1 << shift) * r_y);
        Some(updated).into()
    }
    fn update_prefix_checkpoint_field(
        checkpoints: &[PrefixCheckpoint<F>],
        _r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        let shift = XLEN - 1 - j / 2;
        let updated = checkpoints[Prefixes::RightOperand].unwrap_or(F::zero())
            + (F::from_u64(1 << shift) * r_y);
        Some(updated).into()
    }

    fn prefix_mle_field(
        checkpoints: &[PrefixCheckpoint<F>],
        _r_x: Option<F>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F {
        let mut result = checkpoints[Prefixes::RightOperand].unwrap_or(F::zero());

        if j % 2 == 1 {
            // c is of the right operand
            let shift = XLEN - 1 - j / 2;
            result += F::from_u128((c as u128) << shift);
        }

        let (_x, y) = b.uninterleave();
        let suffix_len = current_suffix_len(j);
        result += F::from_u128(u128::from(y) << (suffix_len / 2));

        result
    }
}
