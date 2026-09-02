use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
    zkvm::lookup_table::suffixes::window_sign::window_sign_bit,
};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

/// Sign bit of a mask window: the left operand's bit at the right operand's
/// most significant set bit. Accumulated as
/// `σ = Σ_i x_i·y_i·Π_{j<i}(1−y_j)` (indices MSB-first).
pub enum WindowSignPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for WindowSignPrefix {
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
        let mut result = checkpoints[Prefixes::WindowSign].unwrap_or(F::zero());
        // Π(1−y) over the bound pairs, maintained by `RightOperandIsZeroPrefix`
        let mut none = checkpoints[Prefixes::RightOperandIsZero].unwrap_or(F::one());

        if let Some(r_x) = r_x {
            // (r_x, c) is the current (x, y) pair
            result += none * (r_x * F::from_u32(c));
            none *= F::one() - F::from_u32(c);
        } else {
            let y_msb = b.pop_msb();
            result += none * F::from_u32(c * y_msb as u32);
            none *= F::from_u8(1 - y_msb);
        }

        let (x, y) = b.uninterleave();
        let sigma_b = window_sign_bit(u64::from(x), u64::from(y));
        result + none * F::from_u64(sigma_b)
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
        // checkpoint += Π(1−y)·r_x·r_y
        let updated = checkpoints[Prefixes::WindowSign].unwrap_or(F::zero())
            + checkpoints[Prefixes::RightOperandIsZero].unwrap_or(F::one()) * (r_x * r_y);
        Some(updated).into()
    }
}
