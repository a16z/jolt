use jolt_field::Field;

use crate::lookup_bits::LookupBits;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

/// Sign bit of a mask window: the left operand's bit at the right operand's
/// most significant set bit. Accumulated as
/// `σ = Σ_i x_i·y_i·Π_{j<i}(1−y_j)` (indices MSB-first).
pub enum WindowSignPrefix {}

impl<F: Field> SparseDensePrefix<F> for WindowSignPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, _suffix_len: usize) -> F {
        let (x, y) = b.uninterleave();
        let (x_val, y_val) = (u64::from(x), u64::from(y));
        // σ(b): x's bit at y's most significant set bit, 0 if y == 0
        let sigma_b = if y_val == 0 {
            0
        } else {
            let i = y_val.ilog2();
            (x_val >> i) & 1
        };
        checkpoints[Prefixes::WindowSign]
            + checkpoints[Prefixes::RightOperandIsZero] * F::from_u64(sigma_b)
    }
}
