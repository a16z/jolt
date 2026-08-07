use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};
use crate::utils::lookup_bits::LookupBits;
use crate::zkvm::instruction_lookups::LOG_K;
use crate::zkvm::lookup_table::lane_mask::lane_value;

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum LaneMaskPrefix<const XLEN: usize, const WIDTH_BYTES: usize> {}

fn factor<const WIDTH_BYTES: usize>(bit: usize) -> u64 {
    if bit >= 3 {
        return 1;
    }
    let first_bit = match WIDTH_BYTES {
        0 | 1 => 0,
        2 => 1,
        4 => 2,
        _ => unreachable!("unsupported lane width"),
    };
    if bit < first_bit {
        1
    } else {
        1u64 << (8 * (1 << bit))
    }
}

fn prefix_kind<const WIDTH_BYTES: usize>() -> Prefixes {
    match WIDTH_BYTES {
        0 => Prefixes::Pow2Lane,
        1 => Prefixes::LaneMaskB,
        2 => Prefixes::LaneMaskH,
        4 => Prefixes::LaneMaskW,
        _ => unreachable!("unsupported lane width"),
    }
}

fn base<const WIDTH_BYTES: usize>() -> u64 {
    match WIDTH_BYTES {
        0 => 1,
        1 => 0xff,
        2 => 0xffff,
        4 => 0xffff_ffff,
        _ => unreachable!("unsupported lane width"),
    }
}

impl<const XLEN: usize, const WIDTH_BYTES: usize, F: JoltField> SparseDensePrefix<F>
    for LaneMaskPrefix<XLEN, WIDTH_BYTES>
{
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
        let suffix_len = LOG_K - j - b.len() - 1;
        if suffix_len != 0 {
            return F::one();
        }
        if b.len() >= 3 {
            return F::from_u64(lane_value::<WIDTH_BYTES>((b & 7) as u64));
        }

        let mut result = F::from_u64(base::<WIDTH_BYTES>());
        for bit in 0..b.len() {
            let value = ((b & 7) as u64 >> bit) & 1;
            result *= F::from_u64(1 + (factor::<WIDTH_BYTES>(bit) - 1) * value);
        }
        result *= F::one() + F::from_u64(factor::<WIDTH_BYTES>(b.len()) - 1) * F::from_u32(c);
        if let Some(r_x) = r_x {
            result *= F::one() + F::from_u64(factor::<WIDTH_BYTES>(b.len() + 1) - 1) * r_x;
        }
        result *= checkpoints[prefix_kind::<WIDTH_BYTES>()].unwrap_or(F::one());
        result
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: C,
        r_y: C,
        j: usize,
        suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        if suffix_len != 0 {
            return Some(F::one()).into();
        }
        if j == 2 * XLEN - 3 {
            let value = F::one() + F::from_u64(factor::<WIDTH_BYTES>(2) - 1) * r_y;
            return Some(value).into();
        }
        if j == 2 * XLEN - 1 {
            let mut value = checkpoints[prefix_kind::<WIDTH_BYTES>()].unwrap_or(F::one());
            value *= F::one() + F::from_u64(factor::<WIDTH_BYTES>(1) - 1) * r_x;
            value *= F::one() + F::from_u64(factor::<WIDTH_BYTES>(0) - 1) * r_y;
            return Some(value).into();
        }
        Some(F::one()).into()
    }
}
