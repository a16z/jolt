use jolt_field::JoltField;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::lookup_bits::LookupBits;
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::uninterleave_bits;

/// Sign-extending parallel bit extract: packs `x`'s bits at `y`'s set
/// positions (MSB-first), then sign-extends by the extracted window's top
/// bit: `pext(x, y) + σ·(2^XLEN − 2^popcount(y))` where `σ` is `x`'s bit at
/// `y`'s most significant set bit.
///
/// With `y` a contiguous window mask of `8w` ones at byte offset `o` (as
/// produced by the window-mask tables), this is the sign-extended `w`-byte
/// lane of `x` at offset `o` — the fused extract for sub-word loads. The
/// definition is total: the identity between `materialize_entry`,
/// `evaluate_mle`, and the prefix-suffix decomposition holds on the full
/// index domain, for any mask shape.
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct PextSignedTable<const XLEN: usize>;

/// The table's semantics on already-masked `XLEN`-bit operands:
/// `pext(x, y) + σ·(2^XLEN − 2^popcount(y))` where `σ` is the window sign.
/// Shared by `materialize_entry` and the instruction's `to_lookup_output`.
pub(crate) fn pext_signed<const XLEN: usize>(x: u64, y: u64) -> u64 {
    let pc = y.count_ones();
    if pc == 0 {
        return 0;
    }
    let pext = crate::tables::suffixes::pext(x, y);
    let sign = crate::tables::suffixes::window_sign_bit(x, y);
    // pext < 2^pc, so the sum never overflows XLEN bits.
    let ext = if sign == 1 {
        ((1u128 << XLEN) - (1u128 << pc)) as u64
    } else {
        0
    };
    pext + ext
}

impl<const XLEN: usize> LookupTable for PextSignedTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, y) = uninterleave_bits(index);
        let x = LookupBits::new(x as u128, XLEN);
        let y = LookupBits::new(y as u128, XLEN);
        pext_signed::<XLEN>(u64::from(x), u64::from(y))
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: JoltField + FieldOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        // pext:   result·(1+y_i) + x_i·y_i
        // σ:      Σ_i x_i·y_i·Π_{j<i}(1−y_j)
        // σ2pc:   multilinear extension of σ·2^popcount(y):
        //         Σ_i x_i·y_i·2·Π_{j<i}(1−y_j)·Π_{k>i}(1+y_k)
        let mut pext = F::zero();
        let mut sigma = F::zero();
        let mut sig2pc = F::zero();
        let mut none = F::one();
        for i in 0..XLEN {
            let x_i: F = r[2 * i].into();
            let y_i: F = r[2 * i + 1].into();
            let xy = x_i * y_i;
            pext = pext * (F::one() + y_i) + xy;
            sig2pc = sig2pc * (F::one() + y_i) + none * (xy + xy);
            sigma += none * xy;
            none *= F::one() - y_i;
        }
        pext + sigma * F::from_u128(1u128 << XLEN) - sig2pc
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for PextSignedTable<XLEN> {
    fn prefixes(&self) -> &'static [Prefixes] {
        &[
            Prefixes::RightShift,
            Prefixes::RightOperandIsZero,
            Prefixes::WindowSign,
            Prefixes::WindowSignPow2,
        ]
    }

    fn suffixes(&self) -> &'static [Suffixes] {
        &[
            Suffixes::One,
            Suffixes::Pext,
            Suffixes::PextHelper,
            Suffixes::WindowSign,
            Suffixes::WindowSignPow2,
        ]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, pext, pext_helper, window_sign, window_sign_pow2] = suffixes.try_into().unwrap();
        let pow_xlen = F::from_u128(1u128 << XLEN);
        // pext part: prefix accumulator scaled past the suffix + suffix pext
        prefixes[Prefixes::RightShift] * pext_helper + pext
            // + 2^XLEN·σ, split as σ_prefix + none_prefix·σ_suffix
            + pow_xlen * (prefixes[Prefixes::WindowSign] * one)
            + pow_xlen * (prefixes[Prefixes::RightOperandIsZero] * window_sign)
            // − σ·2^popcount(y), same split with the popcount scale factored in
            - prefixes[Prefixes::WindowSignPow2] * pext_helper
            - prefixes[Prefixes::RightOperandIsZero] * window_sign_pow2
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
        mle_random_test::<XLEN, Fr, PextSignedTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, PextSignedTable<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        mle_full_hypercube_test::<8, Fr, PextSignedTable<8>>();
    }
}
