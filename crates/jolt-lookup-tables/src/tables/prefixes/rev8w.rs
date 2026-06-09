use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::tables::virtual_rev8w::rev8w;
use crate::XLEN;

use super::{PrefixCheckpoint, PrefixEval, Prefixes, SparseDensePrefix, LOG_K};

pub enum Rev8WPrefix {}

impl<F: Field> SparseDensePrefix<F> for Rev8WPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        if suffix_len >= 64 {
            return F::zero();
        }

        // Each bit at position `i` (from MSB=0) in the interleaved stream maps to
        // bit position `rev8w(1 << i)` in the output. At binary points, both x and y
        // bits land at concrete positions. The interleaved b bits represent positions
        // starting at `suffix_len` upward.
        let b_contribution = rev8w(u64::from(b) << suffix_len);
        checkpoints[Prefixes::Rev8W] + F::from_u64(b_contribution)
    }

    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F {
        let _ = (checkpoints, r_x, c, b, j);
        let suffix_len = LOG_K - j - b.len() - 1;
        // The prefix-suffix MLE is only defined on the 64 LSBs.
        let suffix_n_bits = suffix_len;
        if suffix_n_bits >= 64 {
            return F::zero();
        }

        let mut eval = checkpoints[Prefixes::Rev8W].unwrap_or(F::zero());

        // Add `c` contribution.
        let c_bit_index = suffix_n_bits + b.len();
        if c_bit_index < 64 {
            let shift = rev8w(1 << c_bit_index).trailing_zeros();
            eval += F::from_u128((c as u128) << shift);
        }

        // Add `r_x` contribution.
        let r_x_bit_index = c_bit_index + 1;
        if r_x_bit_index < 64 {
            if let Some(r_x) = r_x {
                let rev_pow2 = rev8w(1 << r_x_bit_index);
                eval += r_x.mul_u64(rev_pow2);
            }
        }

        // Add `b` contribution.
        let b_contribution = rev8w(u64::from(b) << suffix_n_bits);
        eval += F::from_u64(b_contribution);

        eval
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
        suffix_len: usize,
    ) -> PrefixCheckpoint<F> {
        let _ = (checkpoints, r_x, r_y, j, suffix_len);
        let mut res = checkpoints[Prefixes::Rev8W].unwrap_or(F::zero());

        let r_y_bit_index = 2 * XLEN - 1 - j;
        if r_y_bit_index < 64 {
            let rev_pow2 = rev8w(1 << r_y_bit_index);
            res += r_y.mul_u64(rev_pow2);
        }

        let r_x_bit_index = r_y_bit_index + 1;
        if r_x_bit_index < 64 {
            let rev_pow2 = rev8w(1 << r_x_bit_index);
            res += r_x.mul_u64(rev_pow2);
        }

        Some(res).into()
    }
}
