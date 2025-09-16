use crate::{
    field::JoltField, utils::lookup_bits::LookupBits,
    zkvm::instruction_lookups::read_raf_checking::current_suffix_len,
};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

/// Prefix that returns 1 if all upper bits (everything except the lower XLEN bits) are zero
pub enum UpperWordIsZeroPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for UpperWordIsZeroPrefix<XLEN> {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        // We process bits from MSB to LSB
        // If j >= 128-XLEN, we've already checked all upper bits (those beyond XLEN)
        // and now we're in the lower XLEN bits region, so just return the checkpoint
        if j >= 128 - XLEN {
            return checkpoints[Prefixes::UpperWordIsZero].unwrap_or(F::one());
        }

        let mut result = checkpoints[Prefixes::UpperWordIsZero].unwrap_or(F::one());

        // Check that the current bit(s) are zero
        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            // We have r_x, so we're checking one bit
            result *= (F::one() - r_x) * (F::one() - y);
        } else {
            // We're checking two bits: c and the MSB of b
            let x = F::from_u32(c);
            let y = F::from_u8(b.pop_msb());
            // Both x and y must be zero: (1-x)(1-y)
            result *= (F::one() - x) * (F::one() - y);
        }

        let rest = u128::from(b);
        let temp = F::from_u64((((rest << current_suffix_len(j)) >> XLEN) == 0) as u64);
        result *= temp;

        result
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        // If j >= XLEN, we're in the lower XLEN bits region
        // Just return the existing checkpoint unchanged
        if j >= 128 - XLEN {
            return checkpoints[Prefixes::UpperWordIsZero].into();
        }

        // We're still in the upper bits region (those that should be zero)
        // Both r_x and r_y must be zero: (1-r_x)(1-r_y)
        let updated = checkpoints[Prefixes::UpperWordIsZero].unwrap_or(F::one())
            * (F::one() - r_x)
            * (F::one() - r_y);

        Some(updated).into()
    }
}
