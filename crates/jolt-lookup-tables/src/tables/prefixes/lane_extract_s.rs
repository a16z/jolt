use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum LaneSignTPrefix {}
pub enum LaneSignSPrefix {}
pub enum LastMaskBitPrefix {}

fn pairs(b: LookupBits) -> impl Iterator<Item = (u64, u64)> {
    let (x, y) = b.uninterleave();
    let (x, y) = (u64::from(x), u64::from(y));
    let len = b.len() / 2;
    (0..len).map(move |i| {
        let shift = len - 1 - i;
        ((x >> shift) & 1, (y >> shift) & 1)
    })
}

impl<F: Field> SparseDensePrefix<F> for LaneSignTPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, _suffix_len: usize) -> F {
        let mut result = checkpoints[Prefixes::LaneSignT];
        let mut previous_mask = checkpoints[Prefixes::LastMaskBit];
        for (x_i, y_i) in pairs(b) {
            let top = F::from_u64(x_i * y_i) * (F::one() - previous_mask);
            result += F::from_u128(1u128 << XLEN) * top;
            previous_mask = F::from_u64(y_i);
        }
        result
    }
}

impl<F: Field> SparseDensePrefix<F> for LaneSignSPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, _suffix_len: usize) -> F {
        let mut result = checkpoints[Prefixes::LaneSignS];
        let mut previous_mask = checkpoints[Prefixes::LastMaskBit];
        for (x_i, y_i) in pairs(b) {
            let top = F::from_u64(x_i * y_i) * (F::one() - previous_mask);
            result = result * F::from_u64(1 + y_i) - F::from_u64(2) * top;
            previous_mask = F::from_u64(y_i);
        }
        result
    }
}

impl<F: Field> SparseDensePrefix<F> for LastMaskBitPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, _suffix_len: usize) -> F {
        pairs(b)
            .last()
            .map_or(checkpoints[Prefixes::LastMaskBit], |(_, y)| F::from_u64(y))
    }
}
