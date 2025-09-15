use serde::{Deserialize, Serialize};

use super::{
    prefixes::{PrefixEval, Prefixes},
    suffixes::{SuffixEval, Suffixes},
    JoltLookupTable, PrefixSuffixDecomposition,
};
use crate::field::MontU128;
use crate::{field::JoltField, utils::uninterleave_bits};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct EqualTable<const XLEN: usize>;

impl<const WORD_SIZE: usize> JoltLookupTable for EqualTable<WORD_SIZE> {
    fn evaluate_mle<F: JoltField>(&self, r: &[MontU128]) -> F {
        debug_assert!(r.len().is_multiple_of(2));

        let x = r.iter().step_by(2);
        let y = r.iter().skip(1).step_by(2);
        x.zip(y)
            .map(|(x_i, y_i)| {
                F::from_u128_mont(*x_i) * F::from_u128_mont(*y_i)
                    + (F::one() - F::from_u128_mont(*x_i)) * (F::one() - F::from_u128_mont(*y_i))
            })
            .product()
    }

    fn evaluate_mle_field<F: JoltField>(&self, r: &[F]) -> F {
        debug_assert!(r.len().is_multiple_of(2));

        let x = r.iter().step_by(2);
        let y = r.iter().skip(1).step_by(2);
        x.zip(y)
            .map(|(x_i, y_i)| *x_i * y_i + (F::one() - x_i) * (F::one() - y_i))
            .product()
    }

    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, y) = uninterleave_bits(index);
        (x == y).into()
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for EqualTable<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::Eq]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [eq] = suffixes.try_into().unwrap();
        prefixes[Prefixes::Eq] * eq
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test,
        lookup_table_mle_random_test,
        prefix_suffix_test,
    };
    use common::constants::XLEN;

    use super::EqualTable;

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, EqualTable<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, EqualTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, EqualTable<XLEN>>();
    }
}
