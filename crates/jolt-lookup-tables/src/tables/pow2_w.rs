use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Pow2WTable<const XLEN: usize>;

impl<const XLEN: usize> LookupTable for Pow2WTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        1 << (index % (XLEN / 2) as u128) as u64
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let log_half = (XLEN / 2).trailing_zeros() as usize;
        let mut result = F::one();
        for i in 0..log_half {
            result *= F::one() + (F::from_u64((1 << (1 << i)) - 1)) * r[r.len() - i - 1];
        }
        result
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for Pow2WTable<XLEN> {
    fn suffixes(&self) -> &'static [Suffixes] {
        &[Suffixes::Pow2W]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [pow2w] = suffixes.try_into().unwrap();
        prefixes[Prefixes::Pow2W] * pow2w
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
        mle_full_hypercube_test::<8, Fr, Pow2WTable<8>>();
    }

    #[test]
    fn mle_random() {
        mle_random_test::<XLEN, Fr, Pow2WTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, Pow2WTable<XLEN>>();
    }
}
