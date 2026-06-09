use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixCheckpoint, PrefixEval, Prefixes, SparseDensePrefix, LOG_K};

pub enum SignExtensionRightOperandPrefix {}

impl<F: Field> SparseDensePrefix<F> for SignExtensionRightOperandPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        if suffix_len >= XLEN {
            return F::one();
        }

        let j_start = 2 * XLEN - suffix_len - b.len();
        if j_start >= XLEN + 2 {
            return checkpoints[Prefixes::SignExtensionRightOperand];
        }

        // Sign bit is the y-bit at position XLEN (the first y-bit in the lower half).
        // Extract it from b: it's the MSB of the y portion after uninterleaving
        // the bits that include position XLEN and XLEN+1.
        let (_, y) = b.uninterleave();
        let sign_bit = u64::from(y) >> (y.len() - 1);
        F::from_u128((1u128 << XLEN) - (1u128 << (XLEN / 2))).mul_u64(sign_bit)
    }

    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        let _ = (checkpoints, r_x, c, b, j);
        let suffix_len = LOG_K - j - b.len() - 1;

        // If suffix handles sign extension, return 1
        if suffix_len >= XLEN {
            return F::one();
        }

        if j == XLEN {
            // Sign bit is msb of b
            let sign_bit = b.pop_msb();
            F::from_u128((1u128 << XLEN) - (1u128 << (XLEN / 2))).mul_u64(sign_bit as u64)
        } else if j == XLEN + 1 {
            // Sign bit is in c
            F::from_u128((1u128 << XLEN) - (1u128 << (XLEN / 2))).mul_u64(c as u64)
        } else if j >= XLEN + 2 {
            // Sign bit has been processed, use checkpoint
            checkpoints[Prefixes::SignExtensionRightOperand].unwrap_or(F::zero())
        } else {
            unreachable!("This case should never happen if our prefixes start at half_word_size");
        }
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
        suffix_len: usize,
    ) -> PrefixCheckpoint<F> {
        let _ = (checkpoints, r_x, r_y, j, suffix_len);
        if j == XLEN + 1 {
            // Sign bit is in r_y
            let value = F::from_u128((1u128 << XLEN) - (1u128 << (XLEN / 2))) * r_y;
            Some(value).into()
        } else {
            checkpoints[Prefixes::SignExtensionRightOperand].into()
        }
    }
}
