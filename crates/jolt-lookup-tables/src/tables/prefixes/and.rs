use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixCheckpoint, PrefixEval, Prefixes, SparseDensePrefix, LOG_K};

pub enum AndPrefix {}

impl<F: Field> SparseDensePrefix<F> for AndPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let (x, y) = b.uninterleave();
        checkpoints[Prefixes::And] + F::from_u64((u64::from(x) & u64::from(y)) << (suffix_len / 2))
    }

    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        let _ = (checkpoints, r_x, c, b, j);
        let suffix_len = LOG_K - j - b.len() - 1;
        let mut result = checkpoints[Prefixes::And].unwrap_or(F::zero());

        // AND high-order variables of x and y
        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            let shift = XLEN - 1 - j / 2;
            result += F::from_u64(1 << shift) * r_x * y;
        } else {
            let y_msb = b.pop_msb() as u32;
            let shift = XLEN - 1 - j / 2;
            result += F::from_u32(c * y_msb) * F::from_u64(1 << shift);
        }
        // AND remaining x and y bits
        let (x, y) = b.uninterleave();
        result += F::from_u64((u64::from(x) & u64::from(y)) << (suffix_len / 2));

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
        let shift = XLEN - 1 - j / 2;
        // checkpoint += 2^shift * r_x * r_y
        let updated =
            checkpoints[Prefixes::And].unwrap_or(F::zero()) + F::from_u64(1 << shift) * r_x * r_y;
        Some(updated).into()
    }
}
