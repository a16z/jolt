use common::constants::XLEN;
use tracer::instruction::virtual_rev8w::rev8w;

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};
use crate::field::IntoField;
use crate::{
    field::JoltField, utils::lookup_bits::LookupBits,
    zkvm::instruction_lookups::read_raf_checking::current_suffix_len,
};

pub enum Rev8WPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for Rev8WPrefix {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F::Challenge>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F {
        // The prefix-suffix MLE is only defined on the 64 LSBs.
        let suffix_n_bits = current_suffix_len(j);
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
                eval += r_x.into_F().mul_u64(rev_pow2);
            }
        }

        // Add `b` contribution.
        let b_contribution = rev8w(u64::from(b) << suffix_n_bits);
        eval += F::from_u64(b_contribution);

        eval
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F::Challenge,
        r_y: F::Challenge,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        let mut res = checkpoints[Prefixes::Rev8W].unwrap_or(F::zero());

        let r_y_bit_index = 2 * XLEN - 1 - j;
        if r_y_bit_index < 64 {
            let rev_pow2 = rev8w(1 << r_y_bit_index);
            res += r_y.into_F().mul_u64(rev_pow2);
        }

        let r_x_bit_index = r_y_bit_index + 1;
        if r_x_bit_index < 64 {
            let rev_pow2 = rev8w(1 << r_x_bit_index);
            res += r_x.into_F().mul_u64(rev_pow2);
        }

        Some(res).into()
    }
    fn update_prefix_checkpoint_field(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        todo!()
    }

    fn prefix_mle_field(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F {
        todo!()
    }
}
