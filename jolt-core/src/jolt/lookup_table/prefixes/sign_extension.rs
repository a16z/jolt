use crate::{field::JoltField, utils::lookup_bits::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum SignExtensionPrefix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize, F: JoltField> SparseDensePrefix<F> for SignExtensionPrefix<WORD_SIZE> {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
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
        _: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        if j == 1 {
            return None.into();
        }
        let mut updated = checkpoints[Prefixes::SignExtension].unwrap_or(F::zero());
        updated += F::from_u64(1 << (j / 2)) * (F::one() - r_y);
        if j == 2 * WORD_SIZE - 1 {
            updated *= checkpoints[Prefixes::LeftOperandMsb].unwrap();
        }
        Some(updated).into()
    }
}
