use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum SignExtensionPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for SignExtensionPrefix<XLEN> {
    fn prefix_mle<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<C>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
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

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        _: C,
        r_y: C,
        j: usize,
        _suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
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
