use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};
use crate::field::MontU128;
use crate::zkvm::instruction_lookups::read_raf_checking::current_suffix_len;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

pub enum SignExtensionUpperHalfPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for SignExtensionUpperHalfPrefix<XLEN> {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<MontU128>,
        c: u32,
        _b: LookupBits,
        j: usize,
    ) -> F {
        let half_word_size = XLEN / 2;
        let suffix_len = current_suffix_len(j);

        // If suffix handles sign extension, return 1
        if suffix_len >= half_word_size {
            return F::one();
        }

        if j == XLEN + half_word_size {
            // Sign bit is in c
            F::from_u128(((1u128 << (half_word_size)) - 1) << (half_word_size)).mul_u64(c as u64)
        } else if j == XLEN + half_word_size + 1 {
            // Sign bit is in r_x
            F::from_u128(((1u128 << (half_word_size)) - 1) << (half_word_size))
                .mul_u128_mont_form(r_x.unwrap())
        } else if j > XLEN + half_word_size + 1 {
            // Sign bit has been processed, use checkpoint
            checkpoints[Prefixes::SignExtensionUpperHalf].unwrap_or(F::zero())
        } else {
            unreachable!("This case should never happen if our prefixes start at half_word_size");
        }
    }
    fn prefix_mle_field(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        _b: LookupBits,
        j: usize,
    ) -> F {
        let half_word_size = XLEN / 2;
        let suffix_len = current_suffix_len(j);

        // If suffix handles sign extension, return 1
        if suffix_len >= half_word_size {
            return F::one();
        }

        if j == XLEN + half_word_size {
            // Sign bit is in c
            F::from_u128(((1u128 << (half_word_size)) - 1) << (half_word_size)).mul_u64(c as u64)
        } else if j == XLEN + half_word_size + 1 {
            // Sign bit is in r_x
            F::from_u128(((1u128 << (half_word_size)) - 1) << (half_word_size)) * r_x.unwrap()
        } else if j > XLEN + half_word_size + 1 {
            // Sign bit has been processed, use checkpoint
            checkpoints[Prefixes::SignExtensionUpperHalf].unwrap_or(F::zero())
        } else {
            unreachable!("This case should never happen if our prefixes start at half_word_size");
        }
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: MontU128,
        _r_y: MontU128,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        let half_word_size = XLEN / 2;

        if j == XLEN + half_word_size + 1 {
            // Sign bit is in r_x
            let value = F::from_u128(((1u128 << (half_word_size)) - 1) << (half_word_size))
                .mul_u128_mont_form(r_x);
            Some(value).into()
        } else {
            checkpoints[Prefixes::SignExtensionUpperHalf].into()
        }
    }

    fn update_prefix_checkpoint_field(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        _r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        let half_word_size = XLEN / 2;

        if j == XLEN + half_word_size + 1 {
            // Sign bit is in r_x
            let value = F::from_u128(((1u128 << (half_word_size)) - 1) << (half_word_size)) * r_x;
            Some(value).into()
        } else {
            checkpoints[Prefixes::SignExtensionUpperHalf].into()
        }
    }
}
