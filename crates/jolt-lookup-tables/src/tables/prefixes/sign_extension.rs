use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum SignExtensionPrefix {}

impl<F: Field> SparseDensePrefix<F> for SignExtensionPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let j_start = 2 * XLEN - suffix_len - b.len();

        // sign_extension = sign_bit * sum(2^i for each y_i == 0, i >= 1)
        //
        // The sign bit is x_0 (MSB of x). For the first phase (j_start == 0),
        // we extract x_0 from b and sum over y_1..y_{n-1} (skipping y_0).
        // For subsequent phases, checkpoint already contains the accumulated
        // sign_bit * sum, and we add sign_bit * contributions from this phase's y bits.

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

            // y_0 is paired with x_0 (the sign bit). Sign extension starts
            // at bit position 1, processing y_1 onwards. Skip y_0 (index 0).
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
