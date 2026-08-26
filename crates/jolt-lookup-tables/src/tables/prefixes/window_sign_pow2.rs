use jolt_field::JoltField;

use crate::lookup_bits::LookupBits;
use crate::tables::suffixes::window_sign_bit;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

/// Multilinear extension of `σ·2^popcount(y)` where σ is the window sign
/// (see [`super::window_sign::WindowSignPrefix`]):
/// `Σ_i x_i·y_i·2·Π_{j<i}(1−y_j)·Π_{k>i}(1+y_k)` (indices MSB-first).
/// At binary points this equals `σ·2^popcount(y)`.
pub enum WindowSignPow2Prefix {}

impl<F: JoltField> SparseDensePrefix<F> for WindowSignPow2Prefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, _suffix_len: usize) -> F {
        let (x, y) = b.uninterleave();
        let (x_val, y_val) = (u64::from(x), u64::from(y));
        let sigma_b = window_sign_bit(x_val, y_val);
        let pow2_pc = F::from_u128(1u128 << y_val.count_ones());
        (checkpoints[Prefixes::WindowSignPow2]
            + checkpoints[Prefixes::RightOperandIsZero] * F::from_u64(sigma_b))
            * pow2_pc
    }
}
