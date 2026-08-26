use jolt_field::JoltField;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::lookup_bits::LookupBits;
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::uninterleave_bits;

/// Parallel bit extract: packs `x`'s bits at `y`'s set positions (MSB-first),
/// zero-extended. The unsigned sibling of
/// [`PextSignedTable`](super::pext_signed::PextSignedTable): with `y` a
/// contiguous window mask, this is the zero-extended lane of `x` at the
/// mask's byte offset. Total on the full index domain.
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct PextTable<const XLEN: usize>;

impl<const XLEN: usize> LookupTable for PextTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, y) = uninterleave_bits(index);
        let x = LookupBits::new(x as u128, XLEN);
        let y = LookupBits::new(y as u128, XLEN);
        let (x_val, y_val) = (u64::from(x), u64::from(y));

        crate::tables::suffixes::pext(x_val, y_val)
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: JoltField + FieldOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let mut pext = F::zero();
        for i in 0..XLEN {
            let x_i: F = r[2 * i].into();
            let y_i: F = r[2 * i + 1].into();
            pext = pext * (F::one() + y_i) + x_i * y_i;
        }
        pext
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for PextTable<XLEN> {
    fn prefixes(&self) -> &'static [Prefixes] {
        &[Prefixes::RightShift]
    }

    fn suffixes(&self) -> &'static [Suffixes] {
        &[Suffixes::Pext, Suffixes::PextHelper]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [pext, pext_helper] = suffixes.try_into().unwrap();
        prefixes[Prefixes::RightShift] * pext_helper + pext
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
        mle_random_test::<XLEN, Fr, PextTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, PextTable<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        mle_full_hypercube_test::<8, Fr, PextTable<8>>();
    }
}
