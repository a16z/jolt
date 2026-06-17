use jolt_field::Field;

use crate::lookup_bits::LookupBits;

use super::{PrefixCheckpoint, PrefixEval, Prefixes, SparseDensePrefix};

pub enum LeftShiftHelperPrefix {}

impl<F: Field> SparseDensePrefix<F> for LeftShiftHelperPrefix {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, _suffix_len: usize) -> F {
        // Tracks product of (1 + y_i) across rounds.
        // At binary points, (1 + y_i) = 1 when y_i=0, 2 when y_i=1.
        // So product = 2^(popcount(y_phase_bits)).
        // But the leading run of y=1 bits is the only part that matters for the
        // prefix_mle structure: checkpoint * prod(1+y_i) for phase bits, which at
        // binary points = checkpoint * 2^(leading_ones of y) for the multiplicative chain,
        // then remaining y=0 bits contribute factor 1.
        // Actually the full product is needed: each y_i independently multiplies.
        // At binary points: prod over all y_i in phase of (1+y_i) = 2^(count of y_i=1).
        let (_x, y) = b.uninterleave();
        checkpoints[Prefixes::LeftShiftHelper] * F::from_u64(1u64 << u64::from(y).count_ones())
    }

    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        let _ = (checkpoints, r_x, c, b, j);
        let mut result = checkpoints[Prefixes::LeftShiftHelper].unwrap_or(F::one());

        if r_x.is_some() {
            result *= F::from_u32(1 + c);
        } else {
            let y_msb = b.pop_msb();
            result *= F::from_u8(1 + y_msb);
        }

        let (_, y) = b.uninterleave();
        result *= F::from_u32(1 << y.leading_ones());

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
        let mut updated = checkpoints[Prefixes::LeftShiftHelper].unwrap_or(F::one());
        updated *= F::one() + r_y;
        Some(updated).into()
    }
}
