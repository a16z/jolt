use crate::zkvm::instruction_lookups::read_raf_checking::current_suffix_len;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum XorPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for XorPrefix<XLEN> {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F::Challenge>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        let mut result = checkpoints[Prefixes::Xor].unwrap_or(F::zero());

        // XOR high-order variables of x and y
        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            let shift = XLEN - 1 - j / 2;
            result += F::from_u64(1 << shift) * ((F::one() - r_x) * y + r_x * (F::one() - y));
        } else {
            let x = F::from_u32(c);
            let y_msb = F::from_u8(b.pop_msb());
            let shift = XLEN - 1 - j / 2;
            result += F::from_u64(1 << shift) * ((F::one() - x) * y_msb + x * (F::one() - y_msb));
        }
        // XOR remaining x and y bits
        let (x, y) = b.uninterleave();
        let suffix_len = current_suffix_len(j);
        result += F::from_u64((u64::from(x) ^ u64::from(y)) << (suffix_len / 2));

        result
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F::Challenge,
        r_y: F::Challenge,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        let shift = XLEN - 1 - j / 2;
        // checkpoint += 2^shift * ((1 - r_x) * r_y + r_x * (1 - r_y))
        let updated = checkpoints[Prefixes::Xor].unwrap_or(F::zero())
            + F::from_u64(1 << shift) * ((F::one() - r_x) * r_y + r_x * (F::one() - r_y));
        Some(updated).into()
    }
}
