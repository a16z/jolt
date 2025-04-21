use crate::{field::JoltField, subprotocols::sparse_dense_shout::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum EqPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for EqPrefix {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        _: usize,
    ) -> F {
        let mut result = checkpoints[Prefixes::Eq].unwrap_or(F::one());

        // EQ high-order variables of x and y
        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            result *= r_x * y + (F::one() - r_x) * (F::one() - y);
        } else {
            let x = F::from_u8(c as u8);
            let y_msb = F::from_u8(b.pop_msb());
            result *= x * y_msb + (F::one() - x) * (F::one() - y_msb);
        }
        // EQ remaining x and y bits
        let (x, y) = b.uninterleave();
        if x != y {
            return F::zero();
        }
        result
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        _: usize,
    ) -> PrefixCheckpoint<F> {
        // checkpoint *= r_x * r_y + (1 - r_x) * (1 - r_y)
        let updated = checkpoints[Prefixes::Eq].unwrap_or(F::one())
            * (r_x * r_y + (F::one() - r_x) * (F::one() - r_y));
        Some(updated).into()
    }
}
