use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

pub enum RightMsbPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for RightMsbPrefix {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        _: Option<F::Challenge>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        if j == 0 {
            let y_msb = b.pop_msb();
            F::from_u8(y_msb)
        } else if j == 1 {
            F::from_u32(c)
        } else {
            checkpoints[Prefixes::RightOperandMsb].unwrap()
        }
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        _: F::Challenge,
        r_y: F::Challenge,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        if j == 1 {
            Some(r_y.into()).into()
        } else {
            checkpoints[Prefixes::RightOperandMsb].into()
        }
    }
    fn update_prefix_checkpoint_field(
        checkpoints: &[PrefixCheckpoint<F>],
        _r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        if j == 1 {
            Some(r_y).into()
        } else {
            checkpoints[Prefixes::RightOperandMsb].into()
        }
    }

    fn prefix_mle_field(
        checkpoints: &[PrefixCheckpoint<F>],
        _: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        if j == 0 {
            let y_msb = b.pop_msb();
            F::from_u8(y_msb)
        } else if j == 1 {
            F::from_u32(c)
        } else {
            checkpoints[Prefixes::RightOperandMsb].unwrap()
        }
    }
}
