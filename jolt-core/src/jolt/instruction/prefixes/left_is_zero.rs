use crate::{field::JoltField, subprotocols::sparse_dense_shout::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum LeftOperandIsZeroPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for LeftOperandIsZeroPrefix {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        b: LookupBits,
        _: usize,
    ) -> F {
        let (x, _) = b.uninterleave();
        // Short-circuit if low-order bits of `x` are not 0s
        if u64::from(x) != 0 {
            return F::zero();
        }

        let mut result = checkpoints[Prefixes::LeftOperandIsZero].unwrap_or(F::one());

        if let Some(r_x) = r_x {
            result *= F::one() - r_x;
        } else {
            let x = F::from_u8(c as u8);
            result *= F::one() - x;
        }
        result
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        _: F,
        _: usize,
    ) -> PrefixCheckpoint<F> {
        // checkpoint *= (1 - r_x)
        let updated =
            checkpoints[Prefixes::LeftOperandIsZero].unwrap_or(F::one()) * (F::one() - r_x);
        Some(updated).into()
    }
}
