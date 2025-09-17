use crate::{
    field::JoltField, utils::lookup_bits::LookupBits,
    zkvm::instruction_lookups::read_raf_checking::current_suffix_len,
};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

/// Returns 1 if the upper 128-XLEN bits are all zero (no overflow), 0 otherwise.
pub enum OverflowBitsZeroPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for OverflowBitsZeroPrefix<XLEN> {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        if j >= 128 - XLEN {
            return checkpoints[Prefixes::OverflowBitsZero].unwrap_or(F::one());
        }

        let mut result = checkpoints[Prefixes::OverflowBitsZero].unwrap_or(F::one());

        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            result *= (F::one() - r_x) * (F::one() - y);
        } else {
            let x = F::from_u32(c);
            let y = F::from_u8(b.pop_msb());
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
        if j >= 128 - XLEN {
            return checkpoints[Prefixes::OverflowBitsZero].into();
        }
        let updated = checkpoints[Prefixes::OverflowBitsZero].unwrap_or(F::one())
            * (F::one() - r_x)
            * (F::one() - r_y);

        Some(updated).into()
    }
}
