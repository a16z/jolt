use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::MontU128;
use crate::{field::JoltField, utils::uninterleave_bits};
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct SignedLessThanTable<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for SignedLessThanTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, y) = uninterleave_bits(index);
        match XLEN {
            #[cfg(test)]
            8 => ((x as i8) < y as i8).into(),
            32 => ((x as i32) < y as i32).into(),
            64 => ((x as i64) < y as i64).into(),
            _ => panic!("{XLEN}-bit word size is unsupported"),
        }
    }

    fn evaluate_mle_field<F: JoltField>(&self, r: &[F]) -> F {
        let x_sign = r[0];
        let y_sign = r[1];

        let mut lt = F::zero();
        let mut eq = F::one();
        for i in 0..XLEN {
            let x_i = r[2 * i];
            let y_i = r[2 * i + 1];
            lt += (F::one() - x_i) * y_i * eq;
            eq *= x_i * y_i + (F::one() - x_i) * (F::one() - y_i);
        }

        x_sign - y_sign + lt
    }
    fn evaluate_mle<F: JoltField>(&self, r: &[MontU128]) -> F {
        let x_sign = r[0];
        let y_sign = r[1];

        let mut lt = F::zero();
        let mut eq = F::one();
        for i in 0..XLEN {
            let x_i = r[2 * i];
            let x_i_f = F::from_u128_mont(x_i);
            let y_i = r[2 * i + 1];
            lt += (F::one() - x_i_f).mul_u128_mont_form(y_i) * eq;
            eq *= x_i_f.mul_u128_mont_form(y_i) + (F::one() - x_i_f) * (F::one() - F::from_u128_mont(y_i));
        }

        F::from_u128_mont(x_sign) - F::from_u128_mont(y_sign) + lt
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for SignedLessThanTable<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::LessThan]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, less_than] = suffixes.try_into().unwrap();
        prefixes[Prefixes::LeftOperandMsb] * one - prefixes[Prefixes::RightOperandMsb] * one
            + prefixes[Prefixes::LessThan] * one
            + prefixes[Prefixes::Eq] * less_than
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

    use super::SignedLessThanTable;

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, SignedLessThanTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, SignedLessThanTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, SignedLessThanTable<XLEN>>();
    }
}
