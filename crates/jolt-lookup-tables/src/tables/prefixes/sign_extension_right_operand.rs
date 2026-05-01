use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

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
}
