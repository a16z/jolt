use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

/// Computes 2^(y.leading_ones()) for j >= XLEN
pub enum LeftShiftWHelperPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for LeftShiftWHelperPrefix<XLEN> {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F::Challenge>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        // Only process when j >= XLEN
        if j < XLEN {
            return F::one();
        }

        let mut result = checkpoints[Prefixes::LeftShiftWHelper].unwrap_or(F::one());

        if r_x.is_some() {
            result *= F::from_u32(1 + c);
        } else {
            let y_msb = b.pop_msb();
            result *= F::from_u8(1 + y_msb);
        }

        let (_, y) = b.uninterleave();
        result *= F::from_u32(1 << y.leading_ones());

        result
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        _r_x: F::Challenge,
        r_y: F::Challenge,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        if j >= XLEN {
            let mut updated = checkpoints[Prefixes::LeftShiftWHelper].unwrap_or(F::one());
            updated *= F::one() + r_y;
            Some(updated).into()
        } else {
            Some(F::one()).into()
        }
    }

    fn update_prefix_checkpoint_field(
        checkpoints: &[PrefixCheckpoint<F>],
        _r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        if j >= XLEN {
            let mut updated = checkpoints[Prefixes::LeftShiftWHelper].unwrap_or(F::one());
            updated *= F::one() + r_y;
            Some(updated).into()
        } else {
            Some(F::one()).into()
        }
    }
    fn prefix_mle_field(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        // Only process when j >= XLEN
        if j < XLEN {
            return F::one();
        }

        let mut result = checkpoints[Prefixes::LeftShiftWHelper].unwrap_or(F::one());

        if r_x.is_some() {
            result *= F::from_u32(1 + c);
        } else {
            let y_msb = b.pop_msb();
            result *= F::from_u8(1 + y_msb);
        }

        let (_, y) = b.uninterleave();
        result *= F::from_u32(1 << y.leading_ones());

        result
    }
}
