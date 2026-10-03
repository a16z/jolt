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
    let y_top = (y >> (phase_pairs - 1)) & 1;
    let x_bottom = x & 1;

    // Output bit 0 is `x_0 ^ y_{XLEN-1}`. The first phase seeds `acc` with
    // `y_{XLEN-1}` and carries `wrap = 1 - 2*y_{XLEN-1}`, so `acc += wrap * x_0`
    // completes the XOR in whichever phase binds `x_0`.
    let (mut acc, wrap) = if first_phase {
        (F::from_u64(y_top), F::one() - F::from_u64(2 * y_top))
    } else {
        (
            checkpoints[Prefixes::XorRotL1Acc]
                + checkpoints[Prefixes::XorRotL1Straddle] * F::from_u64(y_top),
            checkpoints[Prefixes::XorRotL1Wrap],
        )
    };

    for local_bit in 1..phase_pairs {
        let output_bit = suffix_pairs + local_bit;
        let x_bit = (x >> local_bit) & 1;
        let y_bit = (y >> (local_bit - 1)) & 1;
        acc += F::from_u64((x_bit ^ y_bit) << output_bit);
    }

    let straddle = if suffix_pairs == 0 {
        acc += wrap * F::from_u64(x_bottom);
        F::zero()
    } else {
        acc += F::from_u64(x_bottom << suffix_pairs);
        F::from_u64(1 << suffix_pairs) * (F::one() - F::from_u64(2 * x_bottom))
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

#[cfg(test)]
mod tests {
    use jolt_field::{Fr, Ring};
    use rand::rngs::StdRng;
    use rand::{RngCore, SeedableRng};

    use super::*;
    use crate::tables::prefixes::ALL_PREFIXES;
    use crate::tables::virtual_xor_rotl1::VirtualXORROTL1Table;
    use crate::traits::LookupTable;

    /// One phase spanning all `2 * XLEN` bits: no suffix, no checkpoints, so
    /// `acc` alone must equal the table entry. The phased materialization
    /// tests never reach this geometry.
    #[test]
    fn single_phase_acc_is_table_entry() {
        let checkpoints: Vec<PrefixEval<Fr>> = ALL_PREFIXES
            .iter()
            .map(|prefix| prefix.default_checkpoint())
            .collect();
        let mut rng = StdRng::seed_from_u64(0x0e7);
        for _ in 0..64 {
            let index = (u128::from(rng.next_u64()) << 64) | u128::from(rng.next_u64());
            let acc =
                XorRotL1AccPrefix::evaluate(&checkpoints, LookupBits::new(index, 2 * XLEN), 0);
            let expected = VirtualXORROTL1Table::<XLEN>.materialize_entry(index);
            assert_eq!(acc, Fr::from_u64(expected), "index {index:#034x}");
        }
    }
}
