use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::{LookupMaterializer, MaterializerBackend, U128Materializer};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct AndTable<const XLEN: usize>;

impl<const XLEN: usize> LookupTable for AndTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        self.materialize(&mut U128Materializer::<XLEN>::new(index))
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: crate::LookupEval + FieldOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let mut result = F::zero();
        for i in 0..XLEN {
            let x_i = r[2 * i];
            let y_i = r[2 * i + 1];
            result += F::from_u64(1u64 << (XLEN - 1 - i)) * x_i * y_i;
        }
        result
    }
}

impl<const XLEN: usize> LookupMaterializer<XLEN> for AndTable<XLEN> {
    fn materialize<B: MaterializerBackend>(&self, backend: &mut B) -> B::Output {
        let bits: [B::Bit; XLEN] = std::array::from_fn(|i| {
            let left = backend.input_bit(2 * i);
            let right = backend.input_bit(2 * i + 1);
            backend.and(left, right)
        });
        backend.bits_be(bits)
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for AndTable<XLEN> {
    fn prefixes(&self) -> &'static [Prefixes] {
        &[Prefixes::And]
    }

    fn suffixes(&self) -> &'static [Suffixes] {
        &[Suffixes::One, Suffixes::And]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let [one, and] = suffixes.try_into().unwrap();
        prefixes[Prefixes::And] * one + and
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::test_utils::{mle_full_hypercube_test, mle_random_test, prefix_suffix_test};
    use crate::{uninterleave_bits, XLEN};
    use jolt_field::Fr;

    #[test]
    fn mle_random() {
        mle_random_test::<XLEN, Fr, AndTable<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        mle_full_hypercube_test::<8, Fr, AndTable<8>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, AndTable<XLEN>>();
    }

    #[test]
    fn shared_materializer_matches_bitwise_and() {
        const TEST_XLEN: usize = 8;
        for index in 0..(1u128 << (2 * TEST_XLEN)) {
            let (left, right) = uninterleave_bits(index);
            assert_eq!(AndTable::<TEST_XLEN>.materialize_entry(index), left & right);
        }
    }
}
