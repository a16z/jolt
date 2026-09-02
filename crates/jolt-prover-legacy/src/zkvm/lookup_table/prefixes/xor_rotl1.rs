use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
    zkvm::instruction_lookups::LOG_K,
};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

/// Running state of `out[p] = x[p] ^ y[p - 1 mod 64]` bound MSB-first.
///
/// `acc` holds the completed output bits; `straddle` is the coefficient
/// `2^bit * (1 - 2 x_bit)` awaiting the next lower `y`; `wrap` is
/// `1 - 2 y_63`, awaiting `x_0`.
#[derive(Clone, Copy)]
struct PrefixState<F> {
    acc: F,
    straddle: F,
    wrap: F,
}

impl<F: JoltField> PrefixState<F> {
    /// Restores the state before pair `bit` and binds `(x, y)` at that pair.
    ///
    /// Checkpoints are `None` only before the first pair (`bit == 63`).
    /// Seeding `straddle` with one makes `straddle * y` supply the wrap seed
    /// `y_63` (output bit 0 is `x_0 ^ y_63`, and `xor(x, y) = y + (1 - 2y) x`).
    fn bound(checkpoints: &[PrefixCheckpoint<F>], x: F, y: F, bit: usize) -> Self {
        let mut state = Self {
            acc: checkpoints[Prefixes::XorRotL1Acc].unwrap_or(F::zero()),
            straddle: checkpoints[Prefixes::XorRotL1Straddle].unwrap_or(F::one()),
            wrap: checkpoints[Prefixes::XorRotL1Wrap].unwrap_or(F::zero()),
        };
        if bit == 63 {
            state.wrap = F::one() - y - y;
        }
        state.acc += state.straddle * y;
        if bit == 0 {
            state.acc += state.wrap * x;
            state.straddle = F::zero();
        } else {
            let weight = F::from_u64(1 << bit);
            state.acc += weight * x;
            state.straddle = weight * (F::one() - x - x);
        }
        state
    }

    /// Binds the Boolean pairs below the bound pair `current_bit`, down to
    /// (and excluding) the suffix. All bits are 0/1, so the pairs fold into
    /// one integer and the field coefficients reduce to conditional adds.
    fn bind_tail(&mut self, x: u64, y: u64, current_bit: usize, suffix_pairs: usize) {
        let tail_pairs = current_bit - suffix_pairs;
        if tail_pairs == 0 {
            return;
        }
        if (y >> (tail_pairs - 1)) & 1 == 1 {
            self.acc += self.straddle;
        }
        // Local pair `i >= 1` is `x_i ^ y_{i-1}` at output bit `suffix_pairs + i`.
        let mask = (1u64 << tail_pairs) - 1;
        let mut tail = ((x ^ (y << 1)) & mask & !1) << suffix_pairs;
        let x_bottom = x & 1;
        if suffix_pairs == 0 {
            if x_bottom == 1 {
                self.acc += self.wrap;
            }
            self.straddle = F::zero();
        } else {
            tail |= x_bottom << suffix_pairs;
            let weight = F::from_u64(1 << suffix_pairs);
            self.straddle = if x_bottom == 0 { weight } else { -weight };
        }
        self.acc += F::from_u64(tail);
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
    let (x, y) = match r_x {
        Some(r_x) => (r_x.into(), F::from_u32(c)),
        None => (F::from_u32(c), F::from_u8(b.pop_msb())),
    };
    let mut state = PrefixState::bound(checkpoints, x, y, current_bit);

    let (x, y) = b.uninterleave();
    state.bind_tail(u64::from(x), u64::from(y), current_bit, suffix_pairs);
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
                Some(PrefixState::bound(checkpoints, r_x.into(), r_y.into(), 63 - j / 2).$field)
                    .into()
            }
        }
    };
}

impl_prefix!(XorRotL1AccPrefix, acc);
impl_prefix!(XorRotL1StraddlePrefix, straddle);
impl_prefix!(XorRotL1WrapPrefix, wrap);
