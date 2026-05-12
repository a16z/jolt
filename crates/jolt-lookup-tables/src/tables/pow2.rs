use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;

/// Computes `2^(index % XLEN)`.
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Pow2Table<const XLEN: usize>;

impl<const XLEN: usize> LookupTable for Pow2Table<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        1 << (index % XLEN as u128) as u64
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let log_xlen = XLEN.trailing_zeros() as usize;
        let mut result = F::one();
        for i in 0..log_xlen {
            result *= F::one() + (F::from_u64((1 << (1 << i)) - 1)) * r[r.len() - i - 1];
        }
        result
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for Pow2Table<XLEN> {
    fn suffixes(&self) -> &'static [Suffixes] {
        &[Suffixes::Pow2]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let [pow2] = suffixes.try_into().unwrap();
        prefixes[Prefixes::Pow2] * pow2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::test_utils::{mle_full_hypercube_test, mle_random_test, prefix_suffix_test};
    use crate::XLEN;
    use jolt_field::Fr;

    #[test]
    fn mle_full_hypercube() {
        mle_full_hypercube_test::<8, Fr, Pow2Table<8>>();
    }

    #[test]
    fn mle_random() {
        mle_random_test::<XLEN, Fr, Pow2Table<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, Pow2Table<XLEN>>();
    }
}
