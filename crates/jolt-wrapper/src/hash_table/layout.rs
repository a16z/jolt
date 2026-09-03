//! Row layout and the aligned quadratic row relation.
//!
//! A row holds one half-G step (or one chaining / challenge XOR row).
//! Committed columns are bits: the add outputs `A'`, `C'`, the un-rotated XOR
//! outputs `D' = d ^ A'`, `B' = b ^ C'`, the three add carries and the row's
//! message word `m` (meaningful on round-0 rows, free elsewhere). Wired
//! columns are committed copies of other rows' words supplied by the wiring
//! kernels ([`super::wiring`]): the XOR operands `din`, `bin` as bits and
//! fourteen 32-bit words — the add operands `a_in`, `c_in`, `rot_d`, `m_in`
//! and ten decoder helper words that only carry meaning on challenge / wire
//! rows.
//!
//! The batched constraint `Σ_j γ_j C_j(row)` is written as
//! `Σ_j γ̃_j v_j² + γ̃'_j v_j w_j + L1_j v_j + L2_j w_j` over the committed
//! vector `v` and the wired vector `w` sharing one column-index space (a wired
//! XOR operand sits at its committed partner's index), so the final claim is a
//! quadratic form a column sumcheck can reduce (lane N3). Degree 2 in the
//! columns; 3 with the row `eq` factor.

use jolt_field::{Fr, Ring, Zero};

pub const WORD_BITS: usize = 32;

/// Committed column groups: bit `k` of a word is column `base + k`.
pub const A_OUT: usize = 0;
pub const D_XOR: usize = 32;
pub const C_OUT: usize = 64;
pub const B_XOR: usize = 96;
pub const CARRY_A_LO: usize = 128;
pub const CARRY_A_HI: usize = 129;
pub const CARRY_C: usize = 130;
pub const MESSAGE: usize = 131;
pub const COMMITTED: usize = 163;

/// Wired word columns in column space start right after the committed bits
/// (`din_k` is at `A_OUT + k`, `bin_k` at `C_OUT + k`).
pub const WIRED_WORD_BASE: usize = COMMITTED;
pub const LOG_COLUMNS: usize = 8;

pub const WIRED_BITS: usize = 2 * WORD_BITS;
pub const WIRED_WORDS: usize = 15;
/// Booleanity × 163, XOR × 64, ternary add, binary add.
pub const CONSTRAINTS: usize = COMMITTED + WIRED_BITS + 2;
/// Sumcheck degree of the row relation including the `eq` factor.
pub const DEGREE: usize = 3;

/// A committed 32-bit word group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WordColumn {
    AOut,
    DXor,
    COut,
    BXor,
    Message,
}

impl WordColumn {
    pub fn base(self) -> usize {
        match self {
            Self::AOut => A_OUT,
            Self::DXor => D_XOR,
            Self::COut => C_OUT,
            Self::BXor => B_XOR,
            Self::Message => MESSAGE,
        }
    }
}

/// The wired 32-bit word columns, in column order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WiredWord {
    /// Ternary-add operand `a`.
    AIn,
    /// Binary-add operand `c`.
    CIn,
    /// This row's `D'` rotated by the half's first rotation.
    RotD,
    /// The message word of the step (a copy of a round-0 row's `m`).
    MIn,
    /// Challenge row 0: `out[11] mod 2^29` (Challenge125's masked top word).
    XIn,
    /// Challenge row 0: `bswap(out[10])`.
    YIn,
    /// Challenge row 0: `bswap(out[11])`.
    ZIn,
    /// Wire rows: `bswap(m)` of the row `i` steps later (`i` = 1..=7).
    FrNext(u8),
    /// Wire rows: `bswap16` of the low half-word of `m` eight rows later (the
    /// last two bytes of a field element absorbed two bytes into its word).
    FrTail,
}

impl WiredWord {
    pub const ALL: [Self; WIRED_WORDS] = [
        Self::AIn,
        Self::CIn,
        Self::RotD,
        Self::MIn,
        Self::XIn,
        Self::YIn,
        Self::ZIn,
        Self::FrNext(1),
        Self::FrNext(2),
        Self::FrNext(3),
        Self::FrNext(4),
        Self::FrNext(5),
        Self::FrNext(6),
        Self::FrNext(7),
        Self::FrTail,
    ];

    /// Index among the wired words (and offset from `WIRED_WORD_BASE`).
    pub fn index(self) -> usize {
        match self {
            Self::AIn => 0,
            Self::CIn => 1,
            Self::RotD => 2,
            Self::MIn => 3,
            Self::XIn => 4,
            Self::YIn => 5,
            Self::ZIn => 6,
            Self::FrNext(i) => 6 + usize::from(i),
            Self::FrTail => 14,
        }
    }

    pub fn column(self) -> usize {
        WIRED_WORD_BASE + self.index()
    }
}

/// Column-space index of every wired column: the 64 wired bits (`din`, then
/// `bin`), then the wired words.
pub fn wired_columns() -> Vec<usize> {
    (0..WORD_BITS)
        .map(|k| A_OUT + k)
        .chain((0..WORD_BITS).map(|k| C_OUT + k))
        .chain(WiredWord::ALL.iter().map(|w| w.column()))
        .collect()
}

/// Claimed evaluations of every column at one row point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnEvals {
    pub committed: Vec<Fr>,
    /// `din` bits, then `bin` bits.
    pub wired_bits: Vec<Fr>,
    /// `WiredWord::ALL` order.
    pub wired_words: Vec<Fr>,
}

impl ColumnEvals {
    /// Column-space vectors `(v, w)`.
    pub fn column_space(&self) -> (Vec<Fr>, Vec<Fr>) {
        let mut v = vec![Fr::zero(); 1 << LOG_COLUMNS];
        let mut w = vec![Fr::zero(); 1 << LOG_COLUMNS];
        v[..COMMITTED].copy_from_slice(&self.committed);
        for (value, j) in self
            .wired_bits
            .iter()
            .chain(&self.wired_words)
            .zip(wired_columns())
        {
            w[j] = *value;
        }
        (v, w)
    }

    /// `Σ_k 2^k · bit_k` of a committed word group.
    pub fn word(&self, group: WordColumn) -> Fr {
        let base = group.base();
        (0..WORD_BITS).fold(Fr::zero(), |acc, k| {
            acc + self.committed[base + k].mul_pow_2(k)
        })
    }

    pub fn wired_word(&self, word: WiredWord) -> Fr {
        self.wired_words[word.index()]
    }
}

/// The row relation's coefficient vectors over the column space, for one
/// choice of the 229 batching coefficients.
#[derive(Clone, Debug)]
pub struct Relation {
    pub gamma_sq: Vec<Fr>,
    pub gamma_cross: Vec<Fr>,
    pub l1: Vec<Fr>,
    pub l2: Vec<Fr>,
}

impl Relation {
    /// # Panics
    ///
    /// Panics unless `gammas.len() == CONSTRAINTS`.
    pub fn new(gammas: &[Fr]) -> Self {
        assert_eq!(gammas.len(), CONSTRAINTS, "one coefficient per constraint");
        let n = 1 << LOG_COLUMNS;
        let mut rel = Self {
            gamma_sq: vec![Fr::zero(); n],
            gamma_cross: vec![Fr::zero(); n],
            l1: vec![Fr::zero(); n],
            l2: vec![Fr::zero(); n],
        };
        let (booleanity, rest) = gammas.split_at(COMMITTED);
        for (j, gamma) in booleanity.iter().enumerate() {
            rel.gamma_sq[j] += *gamma;
            rel.l1[j] -= *gamma;
        }
        let (xor_d, rest) = rest.split_at(WORD_BITS);
        for (k, gamma) in xor_d.iter().enumerate() {
            rel.xor(*gamma, D_XOR + k, A_OUT + k);
        }
        let (xor_b, rest) = rest.split_at(WORD_BITS);
        for (k, gamma) in xor_b.iter().enumerate() {
            rel.xor(*gamma, B_XOR + k, C_OUT + k);
        }
        // Ternary add: Σ A'_k 2^k + 2^32 κ0 + 2^33 κ1 − a_in − Σ bin_k 2^k − m_in.
        let gamma = rest[0];
        for k in 0..WORD_BITS {
            rel.l1[A_OUT + k] += gamma.mul_pow_2(k);
            rel.l2[C_OUT + k] -= gamma.mul_pow_2(k);
        }
        rel.l1[CARRY_A_LO] += gamma.mul_pow_2(32);
        rel.l1[CARRY_A_HI] += gamma.mul_pow_2(33);
        rel.l2[WiredWord::AIn.column()] -= gamma;
        rel.l2[WiredWord::MIn.column()] -= gamma;
        // Binary add: Σ C'_k 2^k + 2^32 κ2 − c_in − rot_d.
        let gamma = rest[1];
        for k in 0..WORD_BITS {
            rel.l1[C_OUT + k] += gamma.mul_pow_2(k);
        }
        rel.l1[CARRY_C] += gamma.mul_pow_2(32);
        rel.l2[WiredWord::CIn.column()] -= gamma;
        rel.l2[WiredWord::RotD.column()] -= gamma;
        rel
    }

    /// `out = x ^ y` with `x` wired at the committed operand's index:
    /// `out − w − v + 2·w·v`.
    fn xor(&mut self, gamma: Fr, out: usize, operand: usize) {
        self.l1[out] += gamma;
        self.l1[operand] -= gamma;
        self.l2[operand] -= gamma;
        self.gamma_cross[operand] += gamma + gamma;
    }

    /// The quadratic form at column-space vectors `(v, w)`.
    pub fn evaluate(&self, v: &[Fr], w: &[Fr]) -> Fr {
        let mut acc = Fr::zero();
        for j in 0..1 << LOG_COLUMNS {
            acc += v[j] * (self.gamma_sq[j] * v[j] + self.l1[j]);
            acc += w[j] * (self.gamma_cross[j] * v[j] + self.l2[j]);
        }
        acc
    }

    /// The verifier's expected final sumcheck claim: `eq(τ, r) · Q(v(r), w(r))`
    /// for the row point `r` bound from `challenges` (round order; round `i`
    /// binds row-index bit `i`, i.e. `τ[n − 1 − i]`).
    pub fn final_check(&self, tau: &[Fr], challenges: &[Fr], evals: &ColumnEvals) -> Fr {
        let (v, w) = evals.column_space();
        eq_rounds(tau, challenges) * self.evaluate(&v, &w)
    }
}

/// `eq(τ, r)` for a point bound in round order (round `i` binds `τ[n − 1 − i]`).
///
/// # Panics
///
/// Panics unless `tau.len() == challenges.len()`.
pub fn eq_rounds(tau: &[Fr], challenges: &[Fr]) -> Fr {
    assert_eq!(
        tau.len(),
        challenges.len(),
        "one challenge per row variable"
    );
    let one = Fr::from_u64(1);
    tau.iter().rev().zip(challenges).fold(one, |acc, (t, r)| {
        let tr = *t * *r;
        acc * (one - *t - *r + tr + tr)
    })
}
