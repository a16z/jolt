use ark_ff::PrimeField;
use ark_std::log2;

use crate::{jolt::subtable::LassoSubtable, utils::split_bits};

pub struct LessThanSubtable;

impl<F> LassoSubtable<F> for LessThanSubtable
where
    F: PrimeField,
{
    fn materialize(&self, M: usize) -> Vec<F> {
        let mut entries = Vec::with_capacity(M);
        let bits_per_operand = (log2(M) / 2) as usize;
        for idx in 0..M {
            let (x, y) = split_bits(idx, bits_per_operand);
            let val = if x < y { F::one() } else { F::zero() };
            entries.push(val)
        }
        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        todo!()
    }
}
