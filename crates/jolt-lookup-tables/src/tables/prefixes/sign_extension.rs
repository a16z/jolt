use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixCheckpoint, PrefixEval, Prefixes, SparseDensePrefix};

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

    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        let _ = (checkpoints, r_x, c, b, j);
        if j == 0 {
            let sign_bit = F::from_u8(c as u8);
            if sign_bit.is_zero() {
                return F::zero();
            }
            let _ = b.pop_msb();
            let (_, mut y) = b.uninterleave();
            let mut result = F::zero();
            let mut index = 1;
            for _ in 0..y.len() {
                let y_i = y.pop_msb() as u64;
                result += F::from_u64((1 - y_i) << index);
                index += 1;
            }
            return result * sign_bit;
        }
        if j == 1 {
            let sign_bit = r_x.unwrap();
            let (_, mut y) = b.uninterleave();
            let mut result = F::zero();
            let mut index = 1;
            for _ in 0..y.len() {
                let y_i = y.pop_msb() as u64;
                result += F::from_u64((1 - y_i) << index);
                index += 1;
            }
            return result * sign_bit;
        }

        let sign_bit = checkpoints[Prefixes::LeftOperandMsb].unwrap();
        let mut result = checkpoints[Prefixes::SignExtension].unwrap_or(F::zero());

        if r_x.is_some() {
            result += F::from_u64(1 << (j / 2)) * (F::one() - F::from_u32(c));
        } else {
            let y_msb = b.pop_msb();
            if y_msb == 0 {
                result += F::from_u64(1 << (j / 2));
            }
        }
        let (_, mut y) = b.uninterleave();
        let mut index = j / 2;
        for _ in 0..y.len() {
            index += 1;
            if y.pop_msb() == 0 {
                result += F::from_u64(1 << index);
            }
        }

        result *= sign_bit;
        result
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
        suffix_len: usize,
    ) -> PrefixCheckpoint<F> {
        let _ = (checkpoints, r_x, r_y, j, suffix_len);
        if j == 1 {
            return None.into();
        }
        let mut updated = checkpoints[Prefixes::SignExtension].unwrap_or(F::zero());
        updated += F::from_u64(1 << (j / 2)) * (F::one() - r_y);
        if j == 2 * XLEN - 1 {
            updated *= checkpoints[Prefixes::LeftOperandMsb].unwrap();
        }
        Some(updated).into()
    }
}
