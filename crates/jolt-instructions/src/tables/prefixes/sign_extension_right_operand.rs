use jolt_field::Field;

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::lookup_bits::LookupBits;

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum SignExtensionRightOperandPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: Field> SparseDensePrefix<F> for SignExtensionRightOperandPrefix<XLEN> {
    fn prefix_mle<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        _r_x: Option<C>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F
    where
        C: ChallengeOps<F>,
        F: FieldOps<C>,
    {
        let suffix_len = 2 * XLEN - j - b.len() - 1;

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

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        _r_x: C,
        r_y: C,
        j: usize,
        _suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeOps<F>,
        F: FieldOps<C>,
    {
        if j == XLEN + 1 {
            // Sign bit is in r_y
            let value = F::from_u128((1u128 << XLEN) - (1u128 << (XLEN / 2))) * r_y;
            Some(value).into()
        } else {
            checkpoints[Prefixes::SignExtensionRightOperand].into()
        }
    }
}
