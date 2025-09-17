use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

/// Prefix for detecting when all overflow bits and sign bit are one.
pub enum SignedOverflowBitsOnePrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for SignedOverflowBitsOnePrefix<XLEN> {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        if j >= XLEN + 2 {
            return checkpoints[Prefixes::SignedOverflowBitsOne].unwrap_or(F::one());
        }

        let mut result = checkpoints[Prefixes::SignedOverflowBitsOne].unwrap_or(F::one());

        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            if j == XLEN + 1 {
                // Special case: Only the sign bit (r_x) contributes
                // The y variable is from a lower-order bit that doesn't affect overflow
                result *= r_x;
            } else {
                result *= r_x * y;
            }
        } else {
            let x = F::from_u32(c);
            let y = F::from_u8(b.pop_msb());
            if j == XLEN {
                // Special case: x is the sign bit, y is from lower bits
                // Only the sign bit (x) contributes to overflow checking
                result *= x;
            } else {
                result *= x * y;
            }
        }

        if j < XLEN {
            while b.len() > 0 {
                result *= F::from_u8(b.pop_msb());
            }
        }
        result
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        let updated;
        if j < XLEN {
            updated = checkpoints[Prefixes::SignedOverflowBitsOne].unwrap_or(F::one()) * r_x * r_y;
        } else if j == XLEN + 1 {
            // Special case: Only the sign bit (r_x) is part of the overflow region we're checking.
            // The lower bit (r_y) doesn't affect signed overflow detection.
            updated = checkpoints[Prefixes::SignedOverflowBitsOne].unwrap_or(F::one()) * r_x;
        } else {
            return checkpoints[Prefixes::SignedOverflowBitsOne].into();
        }

        Some(updated).into()
    }
}
