use crate::utils::lookup_bits::LookupBits;
use jolt_field::JoltField;

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum LessThanPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for LessThanPrefix {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        _: usize,
    ) -> F {
        let mut lt = checkpoints[Prefixes::LessThan].unwrap_or(F::zero());
        let mut eq = checkpoints[Prefixes::Eq].unwrap_or(F::one());

        if let Some(r_x) = r_x {
            let c = F::from_u32(c);
            lt += eq * (F::one() - r_x) * c;
            let (x, y) = b.uninterleave();
            if u64::from(x) < u64::from(y) {
                eq *= r_x * c + (F::one() - r_x) * (F::one() - c);
                lt += eq;
            }
        } else {
            let c = F::from_u32(c);
            let y_msb = F::from_u8(b.pop_msb());
            lt += eq * (F::one() - c) * y_msb;
            let (x, y) = b.uninterleave();
            if u64::from(x) < u64::from(y) {
                eq *= c * y_msb + (F::one() - c) * (F::one() - y_msb);
                lt += eq;
            }
        }

        lt
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        _: usize,
    ) -> PrefixCheckpoint<F> {
        let lt_checkpoint = checkpoints[Prefixes::LessThan].unwrap_or(F::zero());
        let eq_checkpoint = checkpoints[Prefixes::Eq].unwrap_or(F::one());
        let lt_updated = lt_checkpoint + eq_checkpoint * (F::one() - r_x) * r_y;
        Some(lt_updated).into()
    }
}
