use jolt_field::JoltField;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum SignExtensionPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for SignExtensionPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let j_start = 2 * XLEN - suffix_len - b.len();

        let (_x, y) = b.uninterleave();
        let y_val = u64::from(y);
        let y_len = y.len();

        if j_start == 0 {
            let (x, _) = b.uninterleave();
            let x_val = u64::from(x);
            let sign_bit = (x_val >> (x.len() - 1)) & 1;
            if sign_bit == 0 {
                return F::zero();
            }

            let mut sum = 0u64;
            for i in 1..y_len {
                let y_bit = (y_val >> (y_len - 1 - i)) & 1;
                if y_bit == 0 {
                    sum += 1u64 << i;
                }
            }
            return F::from_u64(sum);
        }

        let sign_bit = checkpoints[Prefixes::LeftOperandMsb];
        let base_index = j_start / 2;
        let mut new_sum = F::zero();
        for i in 0..y_len {
            let y_bit = (y_val >> (y_len - 1 - i)) & 1;
            if y_bit == 0 {
                new_sum += F::from_u64(1u64 << (base_index + i));
            }
        }

        checkpoints[Prefixes::SignExtension] + sign_bit * new_sum
    }
}
