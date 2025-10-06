use crate::utils::lookup_bits::LookupBits;
use jolt_field::JoltField;

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum LeftMsbPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for LeftMsbPrefix {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        _: LookupBits,
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

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        _: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        if j == 1 {
            Some(r_x).into()
        } else {
            checkpoints[Prefixes::LeftOperandMsb].into()
        }
    }
}
