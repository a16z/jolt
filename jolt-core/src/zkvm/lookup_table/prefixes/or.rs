use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};
use crate::field::MontU128;
use crate::zkvm::instruction_lookups::read_raf_checking::current_suffix_len;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum OrPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for OrPrefix<XLEN> {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        let mut result = checkpoints[Prefixes::Or].unwrap_or(F::zero());

        // OR high-order variables of x and y
        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            let shift = XLEN - 1 - j / 2;
            result += F::from_u64(1 << shift) * (r_x + y - (r_x * y));
        } else {
            let y_msb = b.pop_msb() as u32;
            let shift = XLEN - 1 - j / 2;
            result += F::from_u32(c + y_msb - c * y_msb) * F::from_u64(1 << shift);
        }
        // OR remaining x and y bits
        let (x, y) = b.uninterleave();
        let suffix_len = current_suffix_len(j);
        result += F::from_u64((u64::from(x) | u64::from(y)) << (suffix_len / 2));

        result
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        let shift = XLEN - 1 - j / 2;
        // checkpoint += 2^shift * (r_x + r_y - r_x * r_y)
        let updated = checkpoints[Prefixes::Or].unwrap_or(F::zero())
            + F::from_u64(1 << shift) * (r_x + r_y - r_x * r_y);
        Some(updated).into()
    }
}
