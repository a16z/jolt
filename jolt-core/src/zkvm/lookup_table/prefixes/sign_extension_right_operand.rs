use crate::zkvm::instruction_lookups::read_raf_checking::current_suffix_len;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum SignExtensionRightOperandPrefix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize, F: JoltField> SparseDensePrefix<F>
    for SignExtensionRightOperandPrefix<WORD_SIZE>
{
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        _r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        let suffix_len = current_suffix_len(j);

        // If suffix handles sign extension, return 1
        if suffix_len >= WORD_SIZE {
            return F::one();
        }

        if j == WORD_SIZE {
            // Sign bit is msb of b
            let sign_bit = b.pop_msb();
            F::from_u128((1u128 << WORD_SIZE) - (1u128 << (WORD_SIZE / 2))).mul_u64(sign_bit as u64)
        } else if j == WORD_SIZE + 1 {
            // Sign bit is in c
            F::from_u128((1u128 << WORD_SIZE) - (1u128 << (WORD_SIZE / 2))).mul_u64(c as u64)
        } else if j >= WORD_SIZE + 2 {
            // Sign bit has been processed, use checkpoint
            checkpoints[Prefixes::SignExtensionRightOperand].unwrap_or(F::zero())
        } else {
            unreachable!("This case should never happen if our prefixes start at half_word_size");
        }
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        _r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        if j == WORD_SIZE + 1 {
            // Sign bit is in r_y
            let value = F::from_u128((1u128 << WORD_SIZE) - (1u128 << (WORD_SIZE / 2))) * r_y;
            Some(value).into()
        } else {
            checkpoints[Prefixes::SignExtensionRightOperand].into()
        }
    }
}
