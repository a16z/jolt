use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};
use crate::field::MontU128;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

pub enum LessThanPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for LessThanPrefix {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<MontU128>,
        c: u32,
        mut b: LookupBits,
        _: usize,
    ) -> F {
        let mut lt = checkpoints[Prefixes::LessThan].unwrap_or(F::zero());
        let mut eq = checkpoints[Prefixes::Eq].unwrap_or(F::one());

        if let Some(r_x) = r_x {
            let c = F::from_u32(c);
            lt += eq * (F::one() - F::from_u128_mont(r_x)) * c;
            let (x, y) = b.uninterleave();
            if u64::from(x) < u64::from(y) {
                eq *= c.mul_u128_mont_form(r_x)
                    + (F::one() - F::from_u128_mont(r_x)) * (F::one() - c);
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

    fn prefix_mle_field(
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
        r_x: MontU128,
        r_y: MontU128,
        _: usize,
    ) -> PrefixCheckpoint<F> {
        let lt_checkpoint = checkpoints[Prefixes::LessThan].unwrap_or(F::zero());
        let eq_checkpoint = checkpoints[Prefixes::Eq].unwrap_or(F::one());
        let lt_updated = lt_checkpoint
            + eq_checkpoint * (F::one() - F::from_u128_mont(r_x)).mul_u128_mont_form(r_y);
        Some(lt_updated).into()
    }

    fn update_prefix_checkpoint_field(
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
