use jolt_field::JoltField;

use crate::lookup_bits::LookupBits;
use crate::tables::suffixes::window_sign_bit;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

/// Sign bit of a mask window: the left operand's bit at the right operand's
/// most significant set bit. Accumulated as
/// `σ = Σ_i x_i·y_i·Π_{j<i}(1−y_j)` (indices MSB-first).
pub enum WindowSignPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for WindowSignPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, _suffix_len: usize) -> F {
        let (x, y) = b.uninterleave();
        let sigma_b = window_sign_bit(u64::from(x), u64::from(y));
        checkpoints[Prefixes::WindowSign]
            + checkpoints[Prefixes::RightOperandIsZero] * F::from_u64(sigma_b)
    }
}
