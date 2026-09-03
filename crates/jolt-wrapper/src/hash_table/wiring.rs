//! Structural wiring: which committed word every wired column of a row copies,
//! as a function of the row's position inside its 128-row compression cell.
//!
//! Row `b · 128 + p` is position `p` of compression `b`: `p < 112` half-G steps
//! (`(round · 8 + g) · 2 + half`), `112..120` chaining rows, `120..122`
//! challenge rows (two output words each), `122..128` zero. Blake3 reads fixed
//! state lanes with a fixed message permutation, so every wired slot at
//! position `p` reads one committed word group at a fixed row offset with
//! fixed bit weights — [`source`] is that table, a constant of Blake3 and the
//! layout the verifier holds as code. Compression-dependent inputs (block
//! length, flags, constant message half-words) are public columns committed
//! once in the verifier key ([`VkColumns`]); the first compression's chaining
//! value and the public preamble tail are public inputs ([`PublicInputs`]).
//!
//! The copy constraints are proven by [`super::wiring_prover::WiringProver`],
//! a degree-3 zero-check whose verifier side is [`WiringStatement`]:
//! `Σ_row eq(τ, row) · Σ_s γ_s (w_s(row) − source_s(row)) = 0`, each shifted
//! source re-indexed so the final claim is `Σ_κ K_κ(τ, r) · V_κ(r)` with kernel
//! weights `K_κ(τ, r) = eq(τ_hi, r_hi) · Σ_{p} eqτ[p] · eqr[p − δ]` (previous-cell
//! reads use `eq+1` on the 11 cell bits) — O(positions × slots) verifier work.

use jolt_field::{Fr, One, Ring, Zero};
use jolt_poly::{EqPlusOnePolynomial, EqPolynomial};

use super::blake3::{last_writer, schedule, G_INDICES, HALF_STEPS, IV, ROTATIONS, ROUNDS};
use super::layout::{
    ColumnEvals, WiredWord, WordColumn, MESSAGE, WIRED_BITS, WIRED_WORDS, WORD_BITS,
};

pub const LOG_CELL: usize = 7;
pub const CELL_ROWS: usize = 1 << LOG_CELL;
pub const STEP_ROWS: usize = HALF_STEPS;
pub const CHAINING_POS: usize = STEP_ROWS;
pub const CHALLENGE_POS: usize = CHAINING_POS + 8;
pub const CHALLENGE_ROWS: usize = 2;
/// Round-0 positions a field-element wire can start at: byte 0 of its word
/// (positions 0, 8) or two bytes in (positions 5, 13: items absorbed before
/// the first squeeze sit 22 bytes into the segment).
pub const WIRE_START_POSITIONS: [usize; 4] = [0, 8, 5, 13];
pub const SHIFTED_WIRE_START_POSITIONS: [usize; 2] = [5, 13];
/// Batching coefficients of the wiring zero-check: one per wired slot (64
/// bits, then the words), then the low / high half-word pins.
pub const WIRING_TERMS: usize = WIRED_BITS + WIRED_WORDS + 2;

/// A wired 32-bit word: the two XOR operands are committed as bits, the rest
/// as words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WordSlot {
    Din,
    Bin,
    Word(WiredWord),
}

impl WordSlot {
    pub const COUNT: usize = 2 + WIRED_WORDS;

    pub fn all() -> impl Iterator<Item = Self> {
        [Self::Din, Self::Bin]
            .into_iter()
            .chain(WiredWord::ALL.into_iter().map(Self::Word))
    }

    /// Index of the slot's first batching coefficient.
    pub fn gamma_index(self) -> usize {
        match self {
            Self::Din => 0,
            Self::Bin => WORD_BITS,
            Self::Word(word) => WIRED_BITS + word.index(),
        }
    }
}

/// Fixed bit weights applied to a source word.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Weights {
    /// `rotate_right(rot)`: bit `k` of the value is source bit `(k + rot) % 32`.
    Rot(u8),
    /// Byte swap (big-endian reading of a little-endian word).
    Bswap,
    /// The low `n` bits.
    Mask(u8),
    /// Byte swap of the low half-word (`bytes[0] · 2^8 + bytes[1]`).
    BswapLo16,
}

impl Weights {
    pub fn apply(self, word: u32) -> u32 {
        match self {
            Self::Rot(rot) => word.rotate_right(u32::from(rot)),
            Self::Bswap => word.swap_bytes(),
            Self::Mask(n) => word & ((1u32 << n) - 1),
            Self::BswapLo16 => (word & 0xffff).swap_bytes() >> 16,
        }
    }

    /// Coefficient of source bit `j` in the value: `value = Σ_j coef_j · bit_j`.
    pub fn coefficient(self, j: usize) -> Option<usize> {
        match self {
            Self::Rot(rot) => Some((j + WORD_BITS - usize::from(rot)) % WORD_BITS),
            Self::Bswap => Some(8 * (3 - j / 8) + j % 8),
            Self::Mask(n) => (j < usize::from(n)).then_some(j),
            Self::BswapLo16 => (j < 16).then(|| 8 * (1 - j / 8) + j % 8),
        }
    }
}

/// Public columns committed once in the verifier key (u16 / bit typed): the
/// half-word pins of `m` — 1 where the half-word is a protocol constant, zero
/// padding, or the next compression's block length / flags (positions 122,
/// 123), and the pinned value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VkColumn {
    LoIsConst,
    LoConst,
    HiIsConst,
    HiConst,
}

impl VkColumn {
    pub const ALL: [Self; 4] = [
        Self::LoIsConst,
        Self::LoConst,
        Self::HiIsConst,
        Self::HiConst,
    ];

    pub fn is_bit(self) -> bool {
        matches!(self, Self::LoIsConst | Self::HiIsConst)
    }
}

/// The verifier-key columns over the padded row domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VkColumns {
    pub lo_is_const: Vec<u8>,
    pub lo_const: Vec<u16>,
    pub hi_is_const: Vec<u8>,
    pub hi_const: Vec<u16>,
}

impl VkColumns {
    pub fn zero(rows: usize) -> Self {
        Self {
            lo_is_const: vec![0; rows],
            lo_const: vec![0; rows],
            hi_is_const: vec![0; rows],
            hi_const: vec![0; rows],
        }
    }

    pub fn value(&self, column: VkColumn, row: usize) -> u32 {
        match column {
            VkColumn::LoIsConst => u32::from(self.lo_is_const[row]),
            VkColumn::LoConst => u32::from(self.lo_const[row]),
            VkColumn::HiIsConst => u32::from(self.hi_is_const[row]),
            VkColumn::HiConst => u32::from(self.hi_const[row]),
        }
    }
}

/// Public inputs of the table: the chaining value, block length and flags
/// the first compression starts from (what its predecessor's cell would hold
/// at positions 112..120, 122, 123) and the preamble bytes sharing its first
/// block (an even number, pinned as half-words of the block's message words).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicInputs {
    pub state_in: [u32; 8],
    pub block_len: u32,
    pub flags: u32,
    pub tail: Vec<u8>,
}

/// Positions of a cell holding the next compression's block length / flags in
/// `m` (pinned by the verifier key).
pub const NEXT_LEN_POS: usize = CHALLENGE_POS + CHALLENGE_ROWS;
pub const NEXT_FLAGS_POS: usize = NEXT_LEN_POS + 1;

impl PublicInputs {
    /// The word a `Previous` read of `position` yields for the first cell.
    pub fn previous_word(&self, position: usize) -> u32 {
        match position {
            CHAINING_POS..CHALLENGE_POS => self.state_in[position - CHAINING_POS],
            NEXT_LEN_POS => self.block_len,
            NEXT_FLAGS_POS => self.flags,
            _ => 0,
        }
    }

    /// `(word, high half, value)` of every pinned half-word of the first block.
    pub fn tail_halves(&self) -> impl Iterator<Item = (usize, bool, u16)> + '_ {
        self.tail
            .chunks_exact(2)
            .enumerate()
            .map(|(j, pair)| (j / 2, j % 2 == 1, u16::from_le_bytes([pair[0], pair[1]])))
    }
}

/// Where a wired slot of a position reads from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Zero,
    /// A protocol constant (IV word).
    Const(u32),
    /// A committed word group of the same cell, `delta` rows earlier
    /// (negative: later).
    Cell {
        group: WordColumn,
        weights: Weights,
        delta: i8,
    },
    /// A committed word group at `position` of the previous cell; the first
    /// cell reads `PublicInputs::previous_word(position)`.
    Previous {
        group: WordColumn,
        weights: Weights,
        position: u8,
    },
    /// A committed word group at `position` of the next cell (a shifted wire
    /// straddling two blocks); the last cell reads zero.
    Next {
        group: WordColumn,
        weights: Weights,
        position: u8,
    },
}

/// The message word `steps` rows after a wire start at `p`, in this cell or
/// the next.
fn wire_word(p: usize, steps: usize, weights: Weights) -> Source {
    let target = p + steps;
    if target < 16 {
        Source::Cell {
            group: WordColumn::Message,
            weights,
            delta: -(steps as i8),
        }
    } else {
        Source::Next {
            group: WordColumn::Message,
            weights,
            position: (target - 16) as u8,
        }
    }
}

/// The word group and second-half rotation holding state index `index` after
/// a half-G step: `a` / `c` are add outputs, `b` / `d` rotated XOR outputs.
fn lane(index: usize) -> (WordColumn, u8) {
    let (rot_d, rot_b) = ROTATIONS[1];
    match index / 4 {
        0 => (WordColumn::AOut, 0),
        1 => (WordColumn::BXor, rot_b as u8),
        2 => (WordColumn::COut, 0),
        _ => (WordColumn::DXor, rot_d as u8),
    }
}

/// Position of the second half of step `g` of `round`.
fn second_half(round: usize, g: usize) -> usize {
    round * 16 + 2 * g + 1
}

/// The state word `index` as position `p` sees it, written at `writer`.
fn state(p: usize, writer: usize, index: usize) -> Source {
    let (group, rot) = lane(index);
    Source::Cell {
        group,
        weights: Weights::Rot(rot),
        delta: (p as i16 - writer as i16) as i8,
    }
}

/// State index `index` after the seven rounds (its last writer is a diagonal
/// step of round 6).
fn final_state(p: usize, index: usize) -> Source {
    state(p, second_half(ROUNDS - 1, last_writer(index)), index)
}

fn previous(group: WordColumn, position: usize) -> Source {
    Source::Previous {
        group,
        weights: Weights::Rot(0),
        position: position as u8,
    }
}

/// The wiring table: what slot `slot` of position `p` copies.
pub fn source(p: usize, slot: WordSlot) -> Source {
    if p < STEP_ROWS {
        let (round, rest) = (p / 16, p % 16);
        let (g, half) = (rest / 2, rest % 2);
        let [a, b, c, d] = G_INDICES[g];
        let (rot_d, _) = ROTATIONS[half];
        return match slot {
            WordSlot::Word(WiredWord::RotD) => Source::Cell {
                group: WordColumn::DXor,
                weights: Weights::Rot(rot_d as u8),
                delta: 0,
            },
            WordSlot::Word(WiredWord::MIn) => Source::Cell {
                group: WordColumn::Message,
                weights: Weights::Rot(0),
                delta: (p - schedule(round, rest)) as i8,
            },
            WordSlot::Word(WiredWord::FrNext(i)) if WIRE_START_POSITIONS.contains(&p) => {
                wire_word(p, usize::from(i), Weights::Bswap)
            }
            WordSlot::Word(WiredWord::FrTail) if SHIFTED_WIRE_START_POSITIONS.contains(&p) => {
                wire_word(p, 8, Weights::BswapLo16)
            }
            WordSlot::Word(_)
                if !matches!(slot, WordSlot::Word(WiredWord::AIn | WiredWord::CIn)) =>
            {
                Source::Zero
            }
            _ => {
                let index = match slot {
                    WordSlot::Din => d,
                    WordSlot::Bin => b,
                    WordSlot::Word(WiredWord::AIn) => a,
                    WordSlot::Word(_) => c,
                };
                if half == 1 {
                    // The first half of the same step wrote every lane.
                    let (first_rot_d, first_rot_b) = ROTATIONS[0];
                    let (group, rot) = match slot {
                        WordSlot::Din => (WordColumn::DXor, first_rot_d as u8),
                        WordSlot::Bin => (WordColumn::BXor, first_rot_b as u8),
                        WordSlot::Word(WiredWord::AIn) => (WordColumn::AOut, 0),
                        WordSlot::Word(_) => (WordColumn::COut, 0),
                    };
                    Source::Cell {
                        group,
                        weights: Weights::Rot(rot),
                        delta: 1,
                    }
                } else if g >= 4 {
                    state(p, second_half(round, index % 4), index)
                } else if round > 0 {
                    state(p, second_half(round - 1, last_writer(index)), index)
                } else {
                    // Initial state: cv, IV[0..4], counter 0, block length, flags.
                    match index {
                        0..=7 => previous(WordColumn::DXor, CHAINING_POS + index),
                        8..=11 => Source::Const(IV[index - 8]),
                        12 | 13 => Source::Zero,
                        14 => previous(WordColumn::Message, NEXT_LEN_POS),
                        _ => previous(WordColumn::Message, NEXT_FLAGS_POS),
                    }
                }
            }
        };
    }
    if p < CHALLENGE_POS {
        let j = p - CHAINING_POS;
        return match slot {
            WordSlot::Din => final_state(p, j),
            WordSlot::Word(WiredWord::AIn) => final_state(p, j + 8),
            WordSlot::Bin | WordSlot::Word(_) => Source::Zero,
        };
    }
    if p < CHALLENGE_POS + CHALLENGE_ROWS {
        let i = p - CHALLENGE_POS;
        let next = |group, weights| Source::Cell {
            group,
            weights,
            delta: -1,
        };
        return match slot {
            WordSlot::Din => final_state(p, 8 + 2 * i),
            WordSlot::Bin => final_state(p, 9 + 2 * i),
            WordSlot::Word(WiredWord::CIn) => previous(WordColumn::DXor, CHAINING_POS + 2 * i + 1),
            WordSlot::Word(WiredWord::MIn) => Source::Cell {
                group: WordColumn::Message,
                weights: Weights::Rot(0),
                delta: 0,
            },
            WordSlot::Word(WiredWord::AIn) if i == 0 => next(WordColumn::DXor, Weights::Rot(0)),
            WordSlot::Word(WiredWord::XIn) if i == 0 => next(WordColumn::BXor, Weights::Mask(29)),
            WordSlot::Word(WiredWord::YIn) if i == 0 => next(WordColumn::DXor, Weights::Bswap),
            WordSlot::Word(WiredWord::ZIn) if i == 0 => next(WordColumn::BXor, Weights::Bswap),
            WordSlot::Word(_) => Source::Zero,
        };
    }
    Source::Zero
}

/// The 32-bit word held in a committed column group of `row`.
pub fn word(bits: &[Vec<u8>], group: WordColumn, row: usize) -> u32 {
    let base = group.base();
    (0..WORD_BITS).fold(0, |acc, k| acc | u32::from(bits[base + k][row]) << k)
}

/// The value `slot` of `row` copies.
pub fn read(bits: &[Vec<u8>], public: &PublicInputs, row: usize, slot: WordSlot) -> u32 {
    let (cell, p) = (row >> LOG_CELL, row & (CELL_ROWS - 1));
    match source(p, slot) {
        Source::Zero => 0,
        Source::Const(value) => value,
        Source::Cell {
            group,
            weights,
            delta,
        } => weights.apply(word(
            bits,
            group,
            (row as isize - isize::from(delta)) as usize,
        )),
        Source::Previous {
            group,
            weights,
            position,
        } => weights.apply(if cell == 0 {
            public.previous_word(usize::from(position))
        } else {
            word(bits, group, (cell - 1) * CELL_ROWS + usize::from(position))
        }),
        Source::Next {
            group,
            weights,
            position,
        } => {
            let next = (cell + 1) * CELL_ROWS + usize::from(position);
            weights.apply(if next < bits[0].len() {
                word(bits, group, next)
            } else {
                0
            })
        }
    }
}

/// Every wired column from the committed bits: the 64 `din` / `bin` bits,
/// then the words in `WiredWord::ALL` order.
pub fn materialize(bits: &[Vec<u8>], public: &PublicInputs) -> (Vec<Vec<u8>>, Vec<Vec<u32>>) {
    let rows = bits[0].len();
    let mut wired_bits = vec![vec![0u8; rows]; WIRED_BITS];
    let mut wired_words = vec![vec![0u32; rows]; WIRED_WORDS];
    for row in 0..rows {
        let din = read(bits, public, row, WordSlot::Din);
        let bin = read(bits, public, row, WordSlot::Bin);
        for k in 0..WORD_BITS {
            wired_bits[k][row] = ((din >> k) & 1) as u8;
            wired_bits[WORD_BITS + k][row] = ((bin >> k) & 1) as u8;
        }
        for (column, word) in wired_words.iter_mut().zip(WiredWord::ALL) {
            column[row] = read(bits, public, row, WordSlot::Word(word));
        }
    }
    (wired_bits, wired_words)
}

/// Verifier-key column evaluations at the bound row point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VkEvals {
    pub lo_is_const: Fr,
    pub lo_const: Fr,
    pub hi_is_const: Fr,
    pub hi_const: Fr,
}

/// `eq(x, y)` for two big-endian points of equal length.
pub fn eq_points(x: &[Fr], y: &[Fr]) -> Fr {
    x.iter().zip(y).fold(Fr::one(), |acc, (a, b)| {
        let ab = *a * *b;
        acc * (Fr::one() - *a - *b + ab + ab)
    })
}

/// `eq(x, 0)`.
fn eq_zero(x: &[Fr]) -> Fr {
    x.iter().fold(Fr::one(), |acc, a| acc * (Fr::one() - *a))
}

/// The verifier side of the wiring zero-check.
pub struct WiringStatement<'a> {
    /// `WIRING_TERMS` batching coefficients: wired slots, then the low / high
    /// half-word pins.
    pub gammas: &'a [Fr],
    pub log_rows: usize,
}

impl WiringStatement<'_> {
    fn gamma_lo(&self) -> Fr {
        self.gammas[WIRED_BITS + WIRED_WORDS]
    }

    fn gamma_hi(&self) -> Fr {
        self.gammas[WIRED_BITS + WIRED_WORDS + 1]
    }

    /// Batched weight of bit `k` of a value copied into `slot`: per-bit
    /// coefficients for the XOR operands, the slot coefficient times `2^k`
    /// for words.
    pub fn slot_weight(&self, slot: WordSlot, k: usize) -> Fr {
        match slot {
            WordSlot::Din | WordSlot::Bin => self.gammas[slot.gamma_index() + k],
            WordSlot::Word(_) => self.gammas[slot.gamma_index()].mul_pow_2(k),
        }
    }

    /// `Σ_k weight(slot, k) · bit_k(value)`.
    fn weighted_constant(&self, slot: WordSlot, value: u32) -> Fr {
        (0..WORD_BITS)
            .filter(|k| (value >> k) & 1 == 1)
            .fold(Fr::zero(), |acc, k| acc + self.slot_weight(slot, k))
    }

    /// `Σ_j weight(slot, coefficient(j)) · bit_j(r)` of a source word group.
    fn source_value(
        &self,
        evals: &ColumnEvals,
        slot: WordSlot,
        group: WordColumn,
        weights: Weights,
    ) -> Fr {
        let base = group.base();
        (0..WORD_BITS)
            .filter_map(|j| weights.coefficient(j).map(|k| (j, k)))
            .fold(Fr::zero(), |acc, (j, k)| {
                acc + self.slot_weight(slot, k) * evals.committed[base + j]
            })
    }

    /// The public part of the batched sum: constants and the first cell's
    /// previous-cell reads (`Σ eq(τ, row) · Σ_s weight(source constant)`).
    pub fn input_claim(&self, tau: &[Fr], public: &PublicInputs) -> Fr {
        let (tau_hi, tau_lo) = tau.split_at(self.log_rows - LOG_CELL);
        let eq_tau_lo = EqPolynomial::<Fr>::evals(tau_lo, None);
        let first_cell = eq_zero(tau_hi);
        let mut claim = Fr::zero();
        for (p, eq_tau_p) in eq_tau_lo.iter().enumerate() {
            for slot in WordSlot::all() {
                match source(p, slot) {
                    Source::Const(value) => {
                        claim += *eq_tau_p * self.weighted_constant(slot, value);
                    }
                    Source::Previous {
                        weights, position, ..
                    } => {
                        let value = weights.apply(public.previous_word(usize::from(position)));
                        claim += first_cell * *eq_tau_p * self.weighted_constant(slot, value);
                    }
                    Source::Zero | Source::Cell { .. } | Source::Next { .. } => {}
                }
            }
        }
        claim
    }

    /// The expected final claim at the point bound from `challenges` (round
    /// order), given the column, verifier-key and public evaluations.
    pub fn final_check(
        &self,
        tau: &[Fr],
        challenges: &[Fr],
        evals: &ColumnEvals,
        vk: &VkEvals,
        public: &PublicInputs,
    ) -> Fr {
        let [eq, wired, pins, sources] = self.final_parts(tau, challenges, evals, vk, public);
        eq * (wired + pins) - sources
    }

    /// `(eq(τ, r), wired − pinned constants, pin products, Σ_κ K_κ · V_κ)`.
    ///
    /// # Panics
    ///
    /// Panics unless `tau.len() == challenges.len() == log_rows`.
    pub fn final_parts(
        &self,
        tau: &[Fr],
        challenges: &[Fr],
        evals: &ColumnEvals,
        vk: &VkEvals,
        public: &PublicInputs,
    ) -> [Fr; 4] {
        assert_eq!(tau.len(), self.log_rows, "one τ per row variable");
        assert_eq!(
            challenges.len(),
            self.log_rows,
            "one challenge per row variable"
        );
        let r: Vec<Fr> = challenges.iter().rev().copied().collect();
        let split = self.log_rows - LOG_CELL;
        let (tau_hi, tau_lo) = tau.split_at(split);
        let (r_hi, r_lo) = r.split_at(split);
        let eq_tau_lo = EqPolynomial::<Fr>::evals(tau_lo, None);
        let eq_r_lo = EqPolynomial::<Fr>::evals(r_lo, None);
        let same_cell = eq_points(tau_hi, r_hi);
        let previous_cell = EqPlusOnePolynomial::new(r_hi.to_vec()).evaluate(tau_hi);
        let next_cell = EqPlusOnePolynomial::new(tau_hi.to_vec()).evaluate(r_hi);
        let eq_full = same_cell
            * eq_tau_lo
                .iter()
                .zip(&eq_r_lo)
                .fold(Fr::zero(), |acc, (a, b)| acc + *a * *b);
        let r_first_cell = eq_zero(r_hi);

        // Wired side: Σ_s γ_s · w_s(r).
        let mut wired = Fr::zero();
        for (k, value) in evals.wired_bits.iter().enumerate() {
            wired += self.gammas[k] * *value;
        }
        for (word, value) in WiredWord::ALL.iter().zip(&evals.wired_words) {
            wired += self.gammas[WordSlot::Word(*word).gamma_index()] * *value;
        }
        // Pins: γ_lo · (is_lo(r) · lo(m)(r) − const_lo(r)) + the high half; the
        // public tail adds its half-words to both the selector and the constant.
        let (mut is_lo, mut is_hi) = (vk.lo_is_const, vk.hi_is_const);
        let (mut const_lo, mut const_hi) = (vk.lo_const, vk.hi_const);
        for (w, high, value) in public.tail_halves() {
            let weight = r_first_cell * eq_r_lo[w];
            let (is, constant) = if high {
                (&mut is_hi, &mut const_hi)
            } else {
                (&mut is_lo, &mut const_lo)
            };
            *is += weight;
            *constant += weight * Fr::from_u64(u64::from(value));
        }
        let half = |from: usize| {
            (0..WORD_BITS / 2).fold(Fr::zero(), |acc, k| {
                acc + evals.committed[MESSAGE + from + k].mul_pow_2(k)
            })
        };
        let wired = wired - self.gamma_lo() * const_lo - self.gamma_hi() * const_hi;
        let pins =
            self.gamma_lo() * is_lo * half(0) + self.gamma_hi() * is_hi * half(WORD_BITS / 2);

        // Source side: Σ_κ K_κ(τ, r) · V_κ(r).
        let mut sources = Fr::zero();
        for (p, eq_tau_p) in eq_tau_lo.iter().enumerate() {
            for slot in WordSlot::all() {
                match source(p, slot) {
                    Source::Cell {
                        group,
                        weights,
                        delta,
                    } => {
                        let from = (p as isize - isize::from(delta)) as usize;
                        sources += same_cell
                            * *eq_tau_p
                            * eq_r_lo[from]
                            * self.source_value(evals, slot, group, weights);
                    }
                    Source::Previous {
                        group,
                        weights,
                        position,
                    } => {
                        sources += previous_cell
                            * *eq_tau_p
                            * eq_r_lo[usize::from(position)]
                            * self.source_value(evals, slot, group, weights);
                    }
                    Source::Next {
                        group,
                        weights,
                        position,
                    } => {
                        sources += next_cell
                            * *eq_tau_p
                            * eq_r_lo[usize::from(position)]
                            * self.source_value(evals, slot, group, weights);
                    }
                    Source::Zero | Source::Const(_) => {}
                }
            }
        }
        [eq_full, wired, pins, sources]
    }
}
