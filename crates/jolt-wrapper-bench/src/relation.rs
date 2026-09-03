//! Aligned quadratic row relations. Every layout's batched constraint
//! `Σ_j γ_j·C_j(row)` is written as
//! `Σ_j γ̃_j·v_j² + γ̃'_j·v_j·w_j + L1_j·v_j + L2_j·w_j`
//! over a committed vector `v` (indices `0..committed`) and a wired vector `w`
//! sharing one column-index space of `2^log_columns` slots, so the final
//! claim is a quadratic form the column sumcheck can reduce.

use jolt_field::{Fr, Ring, Zero};

#[derive(Clone, Copy, Debug)]
pub enum Layout {
    /// Lane M3's half-G row: 163 committed bits (A', D', C', B', 3 carries,
    /// 32 message bits), 64 wired bits (din, bin), 3 wired words (a_in, c_in,
    /// the parity-rotated D' input of the binary add).
    Word,
    /// Bit-sliced G-step rows holding `b` bit positions each: per position 16
    /// committed bits (8 outputs, 6 carry bits, 2 message bits) and 7 wired
    /// bits, plus 6 wired carry-in bits for the row's first position.
    Bits(usize),
}

const WORD: usize = 32;
pub const WORD_A: usize = 0;
pub const WORD_D: usize = 32;
pub const WORD_C: usize = 64;
pub const WORD_B: usize = 96;
pub const WORD_CARRY: usize = 128;
pub const WORD_M: usize = 131;
pub const WORD_COMMITTED: usize = 163;
const WORD_AIN: usize = 163;
const WORD_CIN: usize = 164;
const WORD_ROTD: usize = 165;

const SLOT: usize = 16;
const A1: usize = 0;
const D1: usize = 1;
const C1: usize = 2;
const B1: usize = 3;
const A2: usize = 4;
const D2: usize = 5;
const C2: usize = 6;
const B2: usize = 7;
const K1_LO: usize = 8;
const K1_HI: usize = 9;
const K2: usize = 10;
const K3_LO: usize = 11;
const K3_HI: usize = 12;
const K4: usize = 13;
const M0: usize = 14;
const M1: usize = 15;
/// Wired inputs per position beyond the four aligned XOR operands.
const EXTRA_WIRED: usize = 3;
const CARRY_INS: usize = 6;

impl Layout {
    pub fn committed(&self) -> usize {
        match self {
            Self::Word => WORD_COMMITTED,
            Self::Bits(b) => SLOT * b,
        }
    }

    /// Column-space indices of the wired bit inputs.
    pub fn wired_bits(&self) -> Vec<usize> {
        match self {
            Self::Word => (0..WORD)
                .map(|k| WORD_A + k)
                .chain((0..WORD).map(|k| WORD_C + k))
                .collect(),
            Self::Bits(b) => {
                let mut out = Vec::new();
                for i in 0..*b {
                    out.extend([A1, C1, A2, C2].map(|c| SLOT * i + c));
                    out.extend((0..EXTRA_WIRED).map(|e| self.extra(i, e)));
                }
                out.extend((0..CARRY_INS).map(|c| self.carry_in(c)));
                out
            }
        }
    }

    /// Column-space indices of the wired 32-bit word inputs.
    pub fn wired_ints(&self) -> Vec<usize> {
        match self {
            Self::Word => vec![WORD_AIN, WORD_CIN, WORD_ROTD],
            Self::Bits(_) => Vec::new(),
        }
    }

    pub fn log_columns(&self) -> usize {
        let used = match self {
            Self::Word => WORD_ROTD + 1,
            Self::Bits(_) => self.carry_in(CARRY_INS - 1) + 1,
        };
        used.next_power_of_two().trailing_zeros() as usize
    }

    pub fn constraints(&self) -> usize {
        match self {
            Self::Word => WORD_COMMITTED + 2 * WORD + 2,
            Self::Bits(b) => 24 * b,
        }
    }

    fn extra(&self, position: usize, which: usize) -> usize {
        match self {
            Self::Bits(b) => SLOT * b + EXTRA_WIRED * position + which,
            Self::Word => unreachable!(),
        }
    }

    fn carry_in(&self, which: usize) -> usize {
        match self {
            Self::Bits(b) => SLOT * b + EXTRA_WIRED * b + which,
            Self::Word => unreachable!(),
        }
    }

    #[expect(
        clippy::unwrap_used,
        reason = "the gamma count is asserted to equal the constraint count"
    )]
    pub fn relation(&self, gammas: &[Fr]) -> Relation {
        assert_eq!(gammas.len(), self.constraints());
        let n = 1 << self.log_columns();
        let mut rel = Relation {
            log_columns: self.log_columns(),
            committed: self.committed(),
            wired_bits: self.wired_bits(),
            wired_ints: self.wired_ints(),
            gamma_sq: vec![Fr::zero(); n],
            gamma_cross: vec![Fr::zero(); n],
            l1: vec![Fr::zero(); n],
            l2: vec![Fr::zero(); n],
        };
        let mut g = gammas.iter().copied();
        for j in 0..self.committed() {
            let gamma = g.next().unwrap();
            rel.gamma_sq[j] += gamma;
            rel.l1[j] -= gamma;
        }
        match self {
            Self::Word => {
                for k in 0..WORD {
                    rel.xor(g.next().unwrap(), WORD_D + k, WORD_A + k);
                }
                for k in 0..WORD {
                    rel.xor(g.next().unwrap(), WORD_B + k, WORD_C + k);
                }
                let gamma = g.next().unwrap();
                for k in 0..WORD {
                    rel.l1[WORD_A + k] += gamma.mul_pow_2(k);
                    rel.l2[WORD_C + k] -= gamma.mul_pow_2(k);
                    rel.l1[WORD_M + k] -= gamma.mul_pow_2(k);
                }
                rel.l1[WORD_CARRY] += gamma.mul_pow_2(32);
                rel.l1[WORD_CARRY + 1] += gamma.mul_pow_2(33);
                rel.l2[WORD_AIN] -= gamma;
                let gamma = g.next().unwrap();
                for k in 0..WORD {
                    rel.l1[WORD_C + k] += gamma.mul_pow_2(k);
                }
                rel.l1[WORD_CARRY + 2] += gamma.mul_pow_2(32);
                rel.l2[WORD_CIN] -= gamma;
                rel.l2[WORD_ROTD] -= gamma;
            }
            Self::Bits(b) => {
                for i in 0..*b {
                    let s = |c: usize| SLOT * i + c;
                    rel.xor(g.next().unwrap(), s(D1), s(A1));
                    rel.xor(g.next().unwrap(), s(B1), s(C1));
                    rel.xor(g.next().unwrap(), s(D2), s(A2));
                    rel.xor(g.next().unwrap(), s(B2), s(C2));
                    // Carry-in of each add: the previous position's carry-out
                    // (committed) or, at position 0, a wired bit.
                    let prev = |c: usize| SLOT * (i - 1) + c;
                    // a1 = a + b + m0
                    let gamma = g.next().unwrap();
                    rel.l2[self.extra(i, 0)] += gamma;
                    rel.l2[s(C1)] += gamma;
                    rel.l1[s(M0)] += gamma;
                    if i == 0 {
                        rel.l2[self.carry_in(0)] += gamma;
                        rel.l2[self.carry_in(1)] += gamma + gamma;
                    } else {
                        rel.l1[prev(K1_LO)] += gamma;
                        rel.l1[prev(K1_HI)] += gamma + gamma;
                    }
                    rel.l1[s(A1)] -= gamma;
                    rel.l1[s(K1_LO)] -= gamma.mul_pow_2(1);
                    rel.l1[s(K1_HI)] -= gamma.mul_pow_2(2);
                    // c1 = c + rot16(d1)
                    let gamma = g.next().unwrap();
                    rel.l2[self.extra(i, 1)] += gamma;
                    rel.l2[s(A2)] += gamma;
                    if i == 0 {
                        rel.l2[self.carry_in(2)] += gamma;
                    } else {
                        rel.l1[prev(K2)] += gamma;
                    }
                    rel.l1[s(C1)] -= gamma;
                    rel.l1[s(K2)] -= gamma.mul_pow_2(1);
                    // a2 = a1 + rot12(b1) + m1
                    let gamma = g.next().unwrap();
                    rel.l1[s(A1)] += gamma;
                    rel.l2[s(C2)] += gamma;
                    rel.l1[s(M1)] += gamma;
                    if i == 0 {
                        rel.l2[self.carry_in(3)] += gamma;
                        rel.l2[self.carry_in(4)] += gamma + gamma;
                    } else {
                        rel.l1[prev(K3_LO)] += gamma;
                        rel.l1[prev(K3_HI)] += gamma + gamma;
                    }
                    rel.l1[s(A2)] -= gamma;
                    rel.l1[s(K3_LO)] -= gamma.mul_pow_2(1);
                    rel.l1[s(K3_HI)] -= gamma.mul_pow_2(2);
                    // c2 = c1 + rot8(d2)
                    let gamma = g.next().unwrap();
                    rel.l1[s(C1)] += gamma;
                    rel.l2[self.extra(i, 2)] += gamma;
                    if i == 0 {
                        rel.l2[self.carry_in(5)] += gamma;
                    } else {
                        rel.l1[prev(K4)] += gamma;
                    }
                    rel.l1[s(C2)] -= gamma;
                    rel.l1[s(K4)] -= gamma.mul_pow_2(1);
                }
            }
        }
        assert!(g.next().is_none());
        rel
    }
}

pub struct Relation {
    pub log_columns: usize,
    pub committed: usize,
    pub wired_bits: Vec<usize>,
    pub wired_ints: Vec<usize>,
    pub gamma_sq: Vec<Fr>,
    pub gamma_cross: Vec<Fr>,
    pub l1: Vec<Fr>,
    pub l2: Vec<Fr>,
}

impl Relation {
    /// `out = x ⊕ y` with `x` wired at the committed operand's index:
    /// `out − w − v + 2·w·v`.
    fn xor(&mut self, gamma: Fr, out: usize, operand: usize) {
        self.l1[out] += gamma;
        self.l1[operand] -= gamma;
        self.l2[operand] -= gamma;
        self.gamma_cross[operand] += gamma + gamma;
    }

    /// Column-space value vectors at one point: `v` committed, `w` wired.
    pub fn evaluate(&self, v: &[Fr], w: &[Fr]) -> Fr {
        let mut acc = Fr::zero();
        for j in 0..1 << self.log_columns {
            acc += v[j] * (self.gamma_sq[j] * v[j] + self.l1[j]);
            acc += w[j] * (self.gamma_cross[j] * v[j] + self.l2[j]);
        }
        acc
    }

    /// Column-space multilinear evaluations of the four coefficient vectors at `s` (big-endian).
    pub fn coefficients_at(&self, eq_s: &[Fr]) -> [Fr; 4] {
        let dot = |c: &[Fr]| {
            c.iter()
                .zip(eq_s)
                .fold(Fr::zero(), |acc, (a, e)| acc + *a * *e)
        };
        [
            dot(&self.gamma_sq),
            dot(&self.gamma_cross),
            dot(&self.l1),
            dot(&self.l2),
        ]
    }
}
