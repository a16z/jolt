use crate::field::{ChallengeFieldOps, FieldChallengeOps};
use crate::zkvm::instruction_lookups::LOG_K;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

/// Bound-bit accumulator for the aligned-address table: `Σ 2^k·b_k` over the
/// bound index bits `k ∈ [3, XLEN)`. Bits at or above `XLEN` (the carry)
/// contribute nothing; bits 2..0 are cleared.
pub enum AlignAddrPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for AlignAddrPrefix<XLEN> {
    fn prefix_mle<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<C>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        let suffix_len = LOG_K - j - b.len() - 1;
        // Ignore high-order variables (the carry range at or above bit XLEN).
        if j < XLEN {
            return F::zero();
        }
        let mut result = checkpoints[Prefixes::AlignAddr].unwrap_or(F::zero());

        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            let x_shift = 2 * XLEN - j;
            let y_shift = 2 * XLEN - j - 1;
            if x_shift >= 3 {
                result += F::from_u128(1u128 << x_shift) * r_x;
            }
            if y_shift >= 3 {
                result += F::from_u128(1u128 << y_shift) * y;
            }
        } else {
            let x = F::from_u8(c as u8);
            let y_msb = b.pop_msb();
            let x_shift = 2 * XLEN - j - 1;
            let y_shift = 2 * XLEN - j - 2;
            if x_shift >= 3 {
                result += F::from_u128(1 << x_shift) * x;
            }
            if y_shift >= 3 {
                result += F::from_u128((1 << y_shift) * u128::from(y_msb));
            }
        }

        // Add in low-order bits from `b`, clearing bits 2..0.
        result += F::from_u128((u128::from(b) << suffix_len) & !7);

        result
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: C,
        r_y: C,
        j: usize,
        _suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        if j < XLEN {
            return None.into();
        }
        let x_shift = 2 * XLEN - j;
        let y_shift = 2 * XLEN - j - 1;
        let mut updated = checkpoints[Prefixes::AlignAddr].unwrap_or(F::zero());
        if x_shift >= 3 {
            updated += F::from_u128(1 << x_shift) * r_x;
        }
        if y_shift >= 3 {
            updated += F::from_u128(1 << y_shift) * r_y;
        }
        Some(updated).into()
    }
}
