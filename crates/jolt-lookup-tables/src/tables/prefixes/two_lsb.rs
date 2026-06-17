use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixCheckpoint, PrefixEval, Prefixes, SparseDensePrefix, LOG_K};

pub enum TwoLsbPrefix {}

impl<F: Field> SparseDensePrefix<F> for TwoLsbPrefix {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        if suffix_len == 0 {
            if u32::from(b).trailing_zeros() >= 2 {
                F::one()
            } else {
                F::zero()
            }
        } else {
            checkpoints[Prefixes::TwoLsb]
        }
    }

    #[expect(clippy::unwrap_used)]
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F {
        let _ = (checkpoints, r_x, c, b, j);
        let suffix_len = LOG_K - j - b.len() - 1;
        if j == 2 * XLEN - 1 {
            // in the log(K)th round, `c` corresponds to bit 0
            // and `r_x` corresponds to bit 1
            debug_assert_eq!(b.len(), 0);
            (F::one() - F::from_u32(c)) * (F::one() - r_x.unwrap())
        } else if j == 2 * XLEN - 2 {
            // in the (log(K)-1)th round, `c` corresponds to bit 1
            debug_assert_eq!(b.len(), 1);
            let bit0 = u32::from(b) & 1;
            let bit1 = c;
            (F::one() - F::from_u32(bit0)) * (F::one() - F::from_u32(bit1))
        } else if suffix_len == 0 {
            // in the (log(K)-2)th round, the two LSBs of `b` are the two LSBs
            match u32::from(b) & 0b11 {
                0b00 => F::one(),
                _ => F::zero(),
            }
        } else {
            F::one()
        }
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
        suffix_len: usize,
    ) -> PrefixCheckpoint<F> {
        let _ = (checkpoints, r_x, r_y, j, suffix_len);
        if j == 2 * XLEN - 1 {
            Some((F::one() - r_x) * (F::one() - r_y)).into()
        } else {
            checkpoints[Prefixes::TwoLsb].into()
        }
    }
}
