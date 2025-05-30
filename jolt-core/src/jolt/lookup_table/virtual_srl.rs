use serde::{Deserialize, Serialize};

use super::prefixes::PrefixEval;
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::JoltField;
use crate::jolt::lookup_table::prefixes::Prefixes;
use crate::subprotocols::sparse_dense_shout::LookupBits;
use crate::utils::uninterleave_bits;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct VirtualSRLTable<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> JoltLookupTable for VirtualSRLTable<WORD_SIZE> {
    fn materialize_entry(&self, index: u64) -> u64 {
        let (x, y) = uninterleave_bits(index);
        let mut x = LookupBits::new(x as u64, WORD_SIZE);
        let mut y = LookupBits::new(y as u64, WORD_SIZE);

        let mut entry = 0;
        for _ in 0..WORD_SIZE {
            let x_i = x.pop_msb();
            let y_i = y.pop_msb();
            entry *= 1 + y_i as u64;
            entry += (x_i * y_i) as u64;
        }
        entry
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * WORD_SIZE);
        let mut result = F::zero();
        for i in 0..WORD_SIZE {
            let x_i = r[2 * i];
            let y_i = r[2 * i + 1];
            result *= F::one() + y_i;
            result += x_i * y_i;
        }
        result
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE> for VirtualSRLTable<WORD_SIZE> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::RightShift, Suffixes::RightShiftHelper]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [right_shift, right_shift_helper] = suffixes.try_into().unwrap();
        prefixes[Prefixes::RightShift] * right_shift_helper + right_shift
    }

    #[cfg(test)]
    fn random_lookup_index(rng: &mut rand::rngs::StdRng) -> u64 {
        super::test::gen_bitmask_lookup_index(rng)
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::VirtualSRLTable;
    use crate::jolt::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, VirtualSRLTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, VirtualSRLTable<32>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<Fr, VirtualSRLTable<32>>();
    }
}
