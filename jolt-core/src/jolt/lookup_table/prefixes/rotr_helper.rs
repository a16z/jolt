use crate::{field::JoltField, subprotocols::sparse_dense_shout::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum RotrHelperPrefix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize, F: JoltField> SparseDensePrefix<F> for RotrHelperPrefix<WORD_SIZE> {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        b: LookupBits,
        _j: usize,
    ) -> F {
        let mut result = checkpoints[Prefixes::RotrHelper].unwrap_or(F::one());

        let y = if r_x.is_some() {
            if c != 0 {
                let y = F::from_u8(c as u8);
                result *= (F::one() + y) * F::from_u8(2).inverse().unwrap();
            }
            b.uninterleave().1
        } else {
            b.uninterleave().1
        };

        if y.len() != y.leading_ones() as usize {
            result *= F::from_u64(1 << (y.len() - y.leading_ones() as usize))
                .inverse()
                .unwrap();
        }

        result
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        _r_x: F,
        r_y: F,
        _j: usize,
    ) -> PrefixCheckpoint<F> {
        let mut result = checkpoints[Prefixes::RotrHelper].unwrap_or(F::one());
        result *= (F::one() + r_y) * F::from_u8(2).inverse().unwrap();
        Some(result).into()
    }
}
