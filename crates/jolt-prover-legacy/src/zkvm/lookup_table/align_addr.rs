use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};

/// Aligned containing-doubleword address: `(index mod 2^XLEN) & !7`.
///
/// The (non-interleaved) index is the unwrapped effective-address sum
/// `rs1 + imm`; truncating mod `2^XLEN` drops the carry bit (matching the
/// wrapped ADDI it fuses) and clearing bits 0-2 aligns down to the containing
/// doubleword.
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct AlignAddrTable<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for AlignAddrTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        ((index & (1u128 << XLEN).wrapping_sub(1)) as u64) & !7
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let mut result = F::zero();
        // Skip bits 2..0 (the three lowest positions).
        for i in 0..XLEN - 3 {
            let shift = XLEN - 1 - i;
            result += F::from_u128(1u128 << shift) * r[XLEN + i];
        }
        result
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for AlignAddrTable<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::AlignAddr]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, align_addr] = suffixes.try_into().unwrap();
        prefixes[Prefixes::AlignAddr] * one + align_addr
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::AlignAddrTable;
    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use common::constants::XLEN;

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, AlignAddrTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, AlignAddrTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, AlignAddrTable<XLEN>>();
    }
}
