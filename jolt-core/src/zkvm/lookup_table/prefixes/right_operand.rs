use crate::zkvm::instruction_lookups::read_raf_checking::current_suffix_len;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum RightOperandPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for RightOperandPrefix<XLEN> {
    fn prefix_mle(
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
            if shift >= 64 {
                // For shifts >= 64, we can't use from_u64 directly
                // This shouldn't happen for XLEN <= 64
                panic!("Shift {} too large for XLEN {}", shift, XLEN);
            } else {
                result += F::from_u64((c as u64) << shift);
            }
        }

        let (_x, y) = b.uninterleave();
        let suffix_len = current_suffix_len(2 * XLEN, j);
        let y_shift = suffix_len / 2;
        if y_shift >= 64 {
            panic!("Y shift {} too large", y_shift);
        } else {
            result += F::from_u64(u64::from(y) << y_shift);
        }

        result
    }

    fn update_prefix_checkpoint(
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
}
