use crate::{
    field::JoltField, utils::lookup_bits::LookupBits,
    zkvm::instruction_lookups::read_raf_checking::current_suffix_len,
};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

/// Prefix that returns 1 if all upper bits (everything except the lower XLEN bits) are one
pub enum OverflowBitsOnePrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for OverflowBitsOnePrefix<XLEN> {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        // We process bits from MSB to LSB
        // If j > 128-XLEN, we've already checked all upper bits (those beyond XLEN)
        // and now we're in the lower XLEN bits region, so just return the checkpoint
        if j > 128 - XLEN {
            return checkpoints[Prefixes::OverflowBitsOne].unwrap_or(F::one());
        }

        // We're checking if all upper bits are one
        // This computes the product a_i for each upper bit
        
        let mut result = checkpoints[Prefixes::OverflowBitsOne].unwrap_or(F::one());

        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            // Check if bit is one: r_x * y
            result *= r_x * y;
        } else {
            let x = F::from_u32(c);
            let y = F::from_u8(b.pop_msb());
            // Check if both bits are one: x * y
            result *= x * y;
        }

        let rest = u128::from(b);
        let suffix_len = current_suffix_len(j);
        // Check if remaining upper bits are all ones
        let upper_mask = ((1u128 << suffix_len) - 1) << XLEN;
        let upper_bits = rest << suffix_len;
        let temp = F::from_u64(((upper_bits & upper_mask) == upper_mask) as u64);
        result *= temp;

        result
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        // If j > 128-XLEN, we're in the lower XLEN bits region
        // Just return the existing checkpoint unchanged
        if j > 128 - XLEN {
            return checkpoints[Prefixes::OverflowBitsOne].into();
        }

        // We're in the upper bits region
        // Both r_x and r_y should be 1 (checking if all upper bits are one)
        let updated = checkpoints[Prefixes::OverflowBitsOne].unwrap_or(F::one())
            * r_x * r_y;

        Some(updated).into()
    }
}
