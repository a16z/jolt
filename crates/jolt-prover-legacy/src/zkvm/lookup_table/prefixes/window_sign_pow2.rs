use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
    zkvm::lookup_table::suffixes::window_sign::window_sign_bit,
};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

/// Multilinear extension of `σ·2^popcount(y)` where σ is the window sign
/// (see [`super::window_sign::WindowSignPrefix`]):
/// `Σ_i x_i·y_i·2·Π_{j<i}(1−y_j)·Π_{k>i}(1+y_k)` (indices MSB-first).
/// At binary points this equals `σ·2^popcount(y)`.
pub enum WindowSignPow2Prefix {}

impl<F: JoltField> SparseDensePrefix<F> for WindowSignPow2Prefix {
    fn prefix_mle<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<C>,
        c: u32,
        mut b: LookupBits,
        _: usize,
    ) -> F
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        let mut result = checkpoints[Prefixes::WindowSignPow2].unwrap_or(F::zero());
        // Π(1−y) over the bound pairs, maintained by `RightOperandIsZeroPrefix`
        let mut none = checkpoints[Prefixes::RightOperandIsZero].unwrap_or(F::one());

        // Per-pair recurrence: result = result·(1+y) + none·2·x·y
        if let Some(r_x) = r_x {
            // (r_x, c) is the current (x, y) pair
            let y = F::from_u32(c);
            result = result * (F::one() + y) + none * (r_x * (y + y));
            none *= F::one() - y;
        } else {
            let y_msb = b.pop_msb();
            result *= F::from_u8(1 + y_msb);
            result += none * F::from_u32(2 * c * y_msb as u32);
            none *= F::from_u8(1 - y_msb);
        }

        let (x, y) = b.uninterleave();
        let (x_val, y_val) = (u64::from(x), u64::from(y));
        let sigma_b = window_sign_bit(x_val, y_val);
        (result + none * F::from_u64(sigma_b)) * F::from_u128(1u128 << y_val.count_ones())
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: C,
        r_y: C,
        _: usize,
        _suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        // checkpoint = checkpoint·(1+r_y) + Π(1−y)·2·r_x·r_y
        let r_xy = r_x * r_y;
        let updated = checkpoints[Prefixes::WindowSignPow2].unwrap_or(F::zero()) * (F::one() + r_y)
            + checkpoints[Prefixes::RightOperandIsZero].unwrap_or(F::one()) * (r_xy + r_xy);
        Some(updated).into()
    }
}
