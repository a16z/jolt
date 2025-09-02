use crate::{field::JoltField, utils::lookup_bits::LookupBits};
use crate::field::MontU128;
use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum LeftMsbPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for LeftMsbPrefix {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<MontU128>,
        c: u32,
        _: LookupBits,
        j: usize,
    ) -> F {
        if j == 0 {
            F::from_u32(c)
        } else if j == 1 {
            F::from_u128_mont(r_x.unwrap())
        } else {
            checkpoints[Prefixes::LeftOperandMsb].unwrap()
        }
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: MontU128,
        _: MontU128,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        if j == 1 {
            Some(F::from_u128_mont(r_x)).into()
        } else {
            checkpoints[Prefixes::LeftOperandMsb].into()
        }
    }
}
