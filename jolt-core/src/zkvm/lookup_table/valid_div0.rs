use serde::{Deserialize, Serialize};

use super::{
    prefixes::{PrefixEval, Prefixes},
    suffixes::{SuffixEval, Suffixes},
    JoltLookupTable,
    PrefixSuffixDecomposition,
};
use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::uninterleave_bits,
};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
/// (divisor, quotient)
pub struct ValidDiv0Table<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for ValidDiv0Table<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (divisor, quotient) = uninterleave_bits(index);
        if divisor == 0 {
            match XLEN {
                8 => (quotient == u8::MAX as u64).into(),
                32 => (quotient == u32::MAX as u64).into(),
                _ => panic!("{XLEN}-bit word size is unsupported"),
            }
        } else {
            1
        }
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        let mut divisor_is_zero = F::one();
        let mut is_valid_div_by_zero = F::one();

        for i in 0..XLEN {
            let x_i = r[2 * i];
            let y_i = r[2 * i + 1];
            divisor_is_zero *= F::one() - x_i;
            is_valid_div_by_zero *= (F::one() - x_i) * y_i;
        }

        F::one() - divisor_is_zero + is_valid_div_by_zero
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for ValidDiv0Table<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![
            Suffixes::One,
            Suffixes::LeftOperandIsZero,
            Suffixes::DivByZero,
        ]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, left_operand_is_zero, div_by_zero] = suffixes.try_into().unwrap();
        // If the divisor is *not* zero, both:
        // `prefixes[Prefixes::LeftOperandIsZero] * left_operand_is_zero` and
        // `prefixes[Prefixes::DivByZero] * div_by_zero`
        // will be zero.
        //
        // If the divisor *is* zero, returns 1 (on the Boolean hypercube)
        // iff the quotient is valid (i.e. 2^XLEN - 1).
        one - prefixes[Prefixes::LeftOperandIsZero] * left_operand_is_zero
            + prefixes[Prefixes::DivByZero] * div_by_zero
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use common::constants::XLEN;

    use super::ValidDiv0Table;
    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test,
        lookup_table_mle_random_test,
        prefix_suffix_test,
    };

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, ValidDiv0Table<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, ValidDiv0Table<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, ValidDiv0Table<XLEN>>();
    }
}
