use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
    zkvm::instruction_lookups::LOG_K,
};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

#[derive(Clone, Copy)]
struct PrefixState<F> {
    acc: F,
    straddle: F,
    wrap: F,
}

fn bind_pair<F: JoltField>(state: &mut PrefixState<F>, x: F, y: F, bit: usize, first_pair: bool) {
    if first_pair {
        state.acc = y + F::from_u64(1 << bit) * x;
        state.straddle = F::from_u64(1 << bit) * (F::one() - x - x);
        state.wrap = F::one() - y - y;
    } else {
        state.acc += state.straddle * y;
        if bit == 0 {
            state.acc += state.wrap * x;
            state.straddle = F::zero();
        } else {
            state.acc += F::from_u64(1 << bit) * x;
            state.straddle = F::from_u64(1 << bit) * (F::one() - x - x);
        }
    }
}

fn evaluate_state<F, C>(
    checkpoints: &[PrefixCheckpoint<F>],
    r_x: Option<C>,
    c: u32,
    mut b: LookupBits,
    j: usize,
) -> PrefixState<F>
where
    C: ChallengeFieldOps<F>,
    F: JoltField + FieldChallengeOps<C>,
{
    let suffix_len = LOG_K - j - b.len() - 1;
    let suffix_pairs = suffix_len / 2;
    let current_bit = 63 - j / 2;
    let first_pair = current_bit == 63;
    let mut state = if first_pair {
        PrefixState {
            acc: F::zero(),
            straddle: F::zero(),
            wrap: F::zero(),
        }
    } else {
        PrefixState {
            acc: checkpoints[Prefixes::XorRotL1Acc].unwrap(),
            straddle: checkpoints[Prefixes::XorRotL1Straddle].unwrap(),
            wrap: checkpoints[Prefixes::XorRotL1Wrap].unwrap(),
        }
    };

    if let Some(r_x) = r_x {
        bind_pair(
            &mut state,
            r_x.into(),
            F::from_u32(c),
            current_bit,
            first_pair,
        );
    } else {
        let y = b.pop_msb();
        bind_pair(
            &mut state,
            F::from_u32(c),
            F::from_u8(y),
            current_bit,
            first_pair,
        );
    }

    let (x, y) = b.uninterleave();
    let x = u64::from(x);
    let y = u64::from(y);
    for bit in (suffix_pairs..current_bit).rev() {
        let local_bit = bit - suffix_pairs;
        bind_pair(
            &mut state,
            F::from_u64((x >> local_bit) & 1),
            F::from_u64((y >> local_bit) & 1),
            bit,
            false,
        );
    }

    state
}

fn update_state<F, C>(
    checkpoints: &[PrefixCheckpoint<F>],
    r_x: C,
    r_y: C,
    j: usize,
) -> PrefixState<F>
where
    C: ChallengeFieldOps<F>,
    F: JoltField + FieldChallengeOps<C>,
{
    let bit = 63 - j / 2;
    let first_pair = bit == 63;
    let mut state = if first_pair {
        PrefixState {
            acc: F::zero(),
            straddle: F::zero(),
            wrap: F::zero(),
        }
    } else {
        PrefixState {
            acc: checkpoints[Prefixes::XorRotL1Acc].unwrap(),
            straddle: checkpoints[Prefixes::XorRotL1Straddle].unwrap(),
            wrap: checkpoints[Prefixes::XorRotL1Wrap].unwrap(),
        }
    };
    bind_pair(&mut state, r_x.into(), r_y.into(), bit, first_pair);
    state
}

macro_rules! impl_prefix {
    ($name:ident, $field:ident) => {
        pub enum $name {}

        impl<F: JoltField> SparseDensePrefix<F> for $name {
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
                evaluate_state(checkpoints, r_x, c, b, j).$field
            }

            fn update_prefix_checkpoint<C>(
                checkpoints: &[PrefixCheckpoint<F>],
                r_x: C,
                r_y: C,
                j: usize,
                _suffix_len: usize,
            ) -> PrefixCheckpoint<F>
            where
                C: ChallengeFieldOps<F>,
                F: FieldChallengeOps<C>,
            {
                Some(update_state(checkpoints, r_x, r_y, j).$field).into()
            }
        }
    };
}

impl_prefix!(XorRotL1AccPrefix, acc);
impl_prefix!(XorRotL1StraddlePrefix, straddle);
impl_prefix!(XorRotL1WrapPrefix, wrap);
