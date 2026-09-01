use serde::{Deserialize, Serialize};

use super::prefixes::PrefixEval;
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};
use crate::utils::lookup_bits::LookupBits;
use crate::utils::uninterleave_bits;
use crate::zkvm::lookup_table::prefixes::Prefixes;

/// Parallel bit extract: packs `x`'s bits at `y`'s set positions (MSB-first),
/// zero-extended. The unsigned sibling of
/// [`PextSignedTable`](super::pext_signed::PextSignedTable): with `y` a
/// contiguous window mask, this is the zero-extended lane of `x` at the
/// mask's byte offset. Total on the full index domain.
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct PextTable<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for PextTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, y) = uninterleave_bits(index);
        let x = LookupBits::new(x as u128, XLEN);
        let y = LookupBits::new(y as u128, XLEN);
        let (x_val, y_val) = (u64::from(x), u64::from(y));

        crate::zkvm::lookup_table::suffixes::pext::pext(x_val, y_val)
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let mut pext = F::zero();
        for i in 0..XLEN {
            let x_i = r[2 * i];
            let y_i = r[2 * i + 1];
            let xy: F = x_i * y_i;
            pext = pext * (F::one() + y_i) + xy;
        }
        pext
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for PextTable<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::Pext, Suffixes::PextHelper]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [pext, pext_helper] = suffixes.try_into().unwrap();
        prefixes[Prefixes::RightShift] * pext_helper + pext
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::PextTable;
    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use common::constants::XLEN;

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, PextTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, PextTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, PextTable<XLEN>>();
    }
}
