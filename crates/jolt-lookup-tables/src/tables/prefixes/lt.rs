use jolt_field::Field;

use crate::lookup_bits::LookupBits;

use super::{PrefixCheckpoint, PrefixEval, Prefixes, SparseDensePrefix};

pub enum LessThanPrefix {}

impl<F: Field> SparseDensePrefix<F> for LessThanPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, _suffix_len: usize) -> F {
        let (x, y) = b.uninterleave();
        if u64::from(x) < u64::from(y) {
            checkpoints[Prefixes::LessThan] + checkpoints[Prefixes::Eq]
        } else {
            checkpoints[Prefixes::LessThan]
        }
    }

    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        let _ = (checkpoints, r_x, c, b, j);
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
        j: usize,
        suffix_len: usize,
    ) -> PrefixCheckpoint<F> {
        let _ = (checkpoints, r_x, r_y, j, suffix_len);
        let lt_checkpoint = checkpoints[Prefixes::LessThan].unwrap_or(F::zero());
        let eq_checkpoint = checkpoints[Prefixes::Eq].unwrap_or(F::one());
        let lt_updated = lt_checkpoint + eq_checkpoint * (F::one() - r_x) * r_y;
        Some(lt_updated).into()
    }
}
