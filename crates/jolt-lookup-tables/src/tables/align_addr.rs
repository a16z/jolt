use jolt_field::JoltField;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;

/// Aligned containing-doubleword address: `(index mod 2^XLEN) & !7`.
///
/// The (non-interleaved) index is the unwrapped effective-address sum
/// `rs1 + imm`; truncating mod `2^XLEN` drops the carry bit (matching the
/// wrapped ADDI it fuses) and clearing bits 0-2 aligns down to the containing
/// doubleword.
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct AlignAddrTable<const XLEN: usize>;

impl<const XLEN: usize> LookupTable for AlignAddrTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        ((index & (1u128 << XLEN).wrapping_sub(1)) as u64) & !7
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: JoltField + FieldOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let mut result = F::zero();
        // Skip bits 2..0 (the three lowest positions).
        for i in 0..XLEN - 3 {
            let shift = XLEN - 1 - i;
            let b_i: F = r[XLEN + i].into();
            result += F::from_u128(1u128 << shift) * b_i;
        }
        result
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for AlignAddrTable<XLEN> {
    fn prefixes(&self) -> &'static [Prefixes] {
        &[Prefixes::AlignAddr]
    }

    fn suffixes(&self) -> &'static [Suffixes] {
        &[Suffixes::One, Suffixes::AlignAddr]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, align_addr] = suffixes.try_into().unwrap();
        prefixes[Prefixes::AlignAddr] * one + align_addr
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::test_utils::{mle_full_hypercube_test, mle_random_test, prefix_suffix_test};
    use crate::XLEN;
    use jolt_field::Fr;

    #[test]
    fn mle_random() {
        mle_random_test::<XLEN, Fr, AlignAddrTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, AlignAddrTable<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        mle_full_hypercube_test::<8, Fr, AlignAddrTable<8>>();
    }
}
