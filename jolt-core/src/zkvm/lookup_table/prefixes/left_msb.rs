use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};
use crate::field::IntoField;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

pub enum LeftMsbPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for LeftMsbPrefix {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F::Challenge>,
        c: u32,
        _: LookupBits,
        j: usize,
    ) -> F {
        if j == 0 {
            F::from_u32(c)
        } else if j == 1 {
            r_x.unwrap().into_F()
        } else {
            checkpoints[Prefixes::LeftOperandMsb].unwrap()
        }
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F::Challenge,
        _: F::Challenge,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        if j == 1 {
            Some(r_x.into_F()).into()
        } else {
            checkpoints[Prefixes::LeftOperandMsb].into()
        }
    }

    fn update_prefix_checkpoint_field(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        _r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        if j == 1 {
            Some(r_x).into()
        } else {
            checkpoints[Prefixes::LeftOperandMsb].into()
        }
    }

    fn prefix_mle_field(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        _b: LookupBits,
        j: usize,
    ) -> F {
        if j == 0 {
            F::from_u32(c)
        } else if j == 1 {
            r_x.unwrap()
        } else {
            checkpoints[Prefixes::LeftOperandMsb].unwrap()
        }
    }
}
