use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};
use crate::field::MontU128;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

pub enum RightMsbPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for RightMsbPrefix {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        _: Option<MontU128>,
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

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        _: MontU128,
        r_y: MontU128,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        if j == 1 {
            Some(F::from_u128_mont(r_y)).into()
        } else {
            checkpoints[Prefixes::RightOperandMsb].into()
        }
    }
}
