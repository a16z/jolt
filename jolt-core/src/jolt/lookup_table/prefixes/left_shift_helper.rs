use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};
use crate::{field::JoltField, subprotocols::sparse_dense_shout::LookupBits};

/// Computes 2^(y.leading_ones())
pub enum LeftShiftHelperPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for LeftShiftHelperPrefix {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        _: usize,
    ) -> F {
        let mut result = checkpoints[Prefixes::LeftShiftHelper].unwrap_or(F::one());

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
        _r_x: F,
        r_y: F,
        _: usize,
    ) -> PrefixCheckpoint<F> {
        let mut updated = checkpoints[Prefixes::LeftShiftHelper].unwrap_or(F::one());
        updated *= F::one() + r_y;
        Some(updated).into()
    }
}
