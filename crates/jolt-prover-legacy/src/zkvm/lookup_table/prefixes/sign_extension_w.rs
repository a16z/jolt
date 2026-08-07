use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum SignExtensionWPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for SignExtensionWPrefix<XLEN> {
    fn prefix_mle<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<C>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        if j < XLEN {
            return F::zero();
        }

        let sign_bit = if j == XLEN {
            F::from_u32(c)
        } else if j == XLEN + 1 {
            r_x.unwrap().into()
        } else {
            checkpoints[Prefixes::WordMsb].unwrap()
        };
        let mut result = if j <= XLEN + 1 {
            sign_bit * F::from_u128((1u128 << XLEN) - (1u128 << (XLEN / 2)))
        } else {
            checkpoints[Prefixes::SignExtensionW].unwrap()
        };

        if !j.is_multiple_of(2) {
            let position = j / 2 - XLEN / 2;
            if position != 0 {
                result += sign_bit * F::from_u128(1u128 << position) * (F::one() - F::from_u32(c));
            }
        }

        let raw = u128::from(b);
        for offset in 0..b.len() {
            let round = j + 1 + offset;
            if round.is_multiple_of(2) {
                continue;
            }
            let pair = round / 2;
            if pair < XLEN / 2 {
                continue;
            }
            let position = pair - XLEN / 2;
            if position == 0 {
                continue;
            }
            let y_bit = (raw >> (b.len() - 1 - offset)) & 1;
            result += sign_bit * F::from_u128((1 - y_bit) << position);
        }

        result
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: C,
        r_y: C,
        j: usize,
        _: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        if j < XLEN + 1 {
            return None.into();
        }
        if j == XLEN + 1 {
            let base = F::from_u128((1u128 << XLEN) - (1u128 << (XLEN / 2)));
            return Some(base * r_x).into();
        }

        let sign_bit = checkpoints[Prefixes::WordMsb].unwrap();
        let mut result = checkpoints[Prefixes::SignExtensionW].unwrap();
        let position = j / 2 - XLEN / 2;
        if position != 0 {
            result += sign_bit * F::from_u128(1u128 << position) * (F::one() - r_y);
        }
        Some(result).into()
    }
}
