use jolt_field::JoltField;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

struct PrefixState<F> {
    acc: F,
    straddle: F,
    wrap: F,
}

fn evaluate_state<F: JoltField>(
    checkpoints: &[PrefixEval<F>],
    b: LookupBits,
    suffix_len: usize,
) -> PrefixState<F> {
    let suffix_pairs = suffix_len / 2;
    let phase_pairs = b.len() / 2;
    let (x, y) = b.uninterleave();
    let x = u64::from(x);
    let y = u64::from(y);
    let first_phase = suffix_pairs + phase_pairs == XLEN;

    let mut acc = if first_phase {
        let y_top = (y >> (phase_pairs - 1)) & 1;
        F::from_u64(y_top)
    } else {
        checkpoints[Prefixes::XorRotL1Acc]
            + checkpoints[Prefixes::XorRotL1Straddle] * F::from_u64((y >> (phase_pairs - 1)) & 1)
    };

    for local_bit in 1..phase_pairs {
        let output_bit = suffix_pairs + local_bit;
        let x_bit = (x >> local_bit) & 1;
        let y_bit = (y >> (local_bit - 1)) & 1;
        acc += F::from_u64((x_bit ^ y_bit) << output_bit);
    }

    let x_bottom = x & 1;
    if suffix_pairs == 0 {
        acc += if first_phase {
            let y_top = (y >> (phase_pairs - 1)) & 1;
            F::from_u64(x_bottom ^ y_top)
        } else {
            checkpoints[Prefixes::XorRotL1Wrap] * F::from_u64(x_bottom)
        };
    } else {
        acc += F::from_u64(x_bottom << suffix_pairs);
    }

    let straddle = if suffix_pairs == 0 {
        F::zero()
    } else {
        F::from_u64(1 << suffix_pairs) * (F::one() - F::from_u64(2 * x_bottom))
    };
    let wrap = if first_phase {
        let y_top = (y >> (phase_pairs - 1)) & 1;
        F::one() - F::from_u64(2 * y_top)
    } else {
        checkpoints[Prefixes::XorRotL1Wrap]
    };

    PrefixState {
        acc,
        straddle,
        wrap,
    }
}

macro_rules! impl_prefix {
    ($name:ident, $field:ident) => {
        pub enum $name {}

        impl<F: JoltField> SparseDensePrefix<F> for $name {
            fn default_checkpoint() -> F {
                F::zero()
            }

            fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
                evaluate_state(checkpoints, b, suffix_len).$field
            }
        }
    };
}

impl_prefix!(XorRotL1AccPrefix, acc);
impl_prefix!(XorRotL1StraddlePrefix, straddle);
impl_prefix!(XorRotL1WrapPrefix, wrap);
