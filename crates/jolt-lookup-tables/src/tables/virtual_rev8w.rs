use std::array;
use std::iter;

use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;

#[inline]
pub(crate) fn rev8w(v: u64) -> u64 {
    let lo = (v as u32).swap_bytes();
    let hi = ((v >> 32) as u32).swap_bytes();
    lo as u64 + ((hi as u64) << 32)
}

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct VirtualRev8WTable;

impl LookupTable for VirtualRev8WTable {
    fn materialize_entry(&self, index: u128) -> u64 {
        rev8w(index as u64)
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        let mut bits = r.iter().rev();
        let mut bytes = iter::from_fn(|| {
            let bit_chunk = (&mut bits).take(8).enumerate();
            Some(
                bit_chunk
                    .map(|(i, b)| Into::<F>::into(*b).mul_u64(1 << i))
                    .sum::<F>(),
            )
        });

        // SAFETY: `bytes` is an infinite iterator via `iter::from_fn` returning `Some`
        #[expect(clippy::unwrap_used)]
        let [a, b, c, d, e, f, g, h] = array::from_fn(|_| bytes.next().unwrap());
        [d, c, b, a, h, g, f, e]
            .iter()
            .enumerate()
            .map(|(i, b)| b.mul_u64(1 << (i * 8)))
            .sum()
    }
}

impl PrefixSuffixDecomposition for VirtualRev8WTable {
    fn suffixes(&self) -> &'static [Suffixes] {
        &[Suffixes::One, Suffixes::Rev8W]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let [one, rev8w] = suffixes.try_into().unwrap();
        prefixes[Prefixes::Rev8W] * one + rev8w
    }

    // Rev8WPrefix returns zero when suffix >= 64 bits, and Rev8WSuffix only
    // handles the lower 32 bits. The decomposition is therefore only valid for
    // lookup indices whose value fits in 32 bits (upper word = 0).
    #[cfg(test)]
    fn random_lookup_index(rng: &mut rand::rngs::StdRng) -> u128 {
        rand::RngCore::next_u32(rng) as u128
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::test_utils::{mle_random_test, prefix_suffix_test};
    use jolt_field::Fr;

    #[test]
    fn mle_random() {
        mle_random_test::<Fr, VirtualRev8WTable>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<Fr, VirtualRev8WTable>();
    }
}
