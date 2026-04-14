use jolt_field::Field;

use crate::lookup_bits::LookupBits;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum DivByZeroPrefix {}

impl<F: Field> SparseDensePrefix<F> for DivByZeroPrefix {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, _suffix_len: usize) -> F {
        let (divisor, quotient) = b.uninterleave();
        if u64::from(divisor) != 0 || u64::from(quotient) != (1 << quotient.len()) - 1 {
            F::zero()
        } else {
            checkpoints[Prefixes::DivByZero]
        }
    }
}
