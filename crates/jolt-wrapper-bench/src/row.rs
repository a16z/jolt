//! The row sumcheck `Σ_row eq(τ,row)·Q(row)` for an aligned quadratic
//! relation: round 0 runs on the 0/1 columns (no field multiplications),
//! later rounds on bound field columns; `s(X) = c·l(X)·t(X)` with `t(1)`
//! recovered from the running claim.

use jolt_field::{Field, Fr, One, Ring, Zero};
use jolt_poly::EqPolynomial;
use rayon::prelude::*;

use crate::relation::Relation;
use crate::sumcheck::{horner, Instance};
use crate::table::Table;

pub struct RowInstance<'a> {
    rel: &'a Relation,
    tau: Vec<Fr>,
    round: usize,
    c: Fr,
    running: Fr,
    claim: Fr,
    round0: [Fr; 4],
    last: [Fr; 4],
    bits: Vec<Vec<u8>>,
    wired_bits: Vec<Vec<u8>>,
    wired_ints: Vec<Vec<u64>>,
    /// Bound committed columns (`0..committed`) and wired columns (bits, then words).
    v: Vec<Vec<Fr>>,
    w: Vec<Vec<Fr>>,
    /// Column-space index of every wired column (bits, then words).
    wired_index: Vec<usize>,
    /// `γ̃_j + L1_j`: a set committed bit's contribution.
    bool_weight: Vec<Fr>,
    /// `(γ̃'_j + L2_j, L2_j)`: a set wired bit's contribution with its committed partner set / clear.
    wired_weight: Vec<(Fr, Fr)>,
}

fn eq_scalar(a: Fr, b: Fr) -> Fr {
    let ab = a * b;
    Fr::one() - a - b + ab + ab
}

/// `s(X) = (l0 + (l1 − l0)X)·(t0 + (t1 − t0 − t2)X + t2X²)`.
fn round_poly(l0: Fr, l1: Fr, t0: Fr, t1: Fr, t2: Fr) -> [Fr; 4] {
    let m = l1 - l0;
    let u = t1 - t0 - t2;
    [l0 * t0, l0 * u + m * t0, l0 * t2 + m * u, m * t2]
}

impl<'a> RowInstance<'a> {
    pub fn new(rel: &'a Relation, table: Table, tau: Vec<Fr>) -> Self {
        let n = tau.len();
        assert_eq!(table.bits.len(), rel.committed);
        let wired_index: Vec<usize> = rel
            .wired_bits
            .iter()
            .chain(&rel.wired_ints)
            .copied()
            .collect();
        let bool_weight: Vec<Fr> = (0..rel.committed)
            .map(|j| rel.gamma_sq[j] + rel.l1[j])
            .collect();
        let wired_weight: Vec<(Fr, Fr)> = rel
            .wired_bits
            .iter()
            .map(|&j| (rel.gamma_cross[j] + rel.l2[j], rel.l2[j]))
            .collect();
        let mut this = Self {
            rel,
            tau,
            round: 0,
            c: Fr::one(),
            running: Fr::zero(),
            claim: Fr::zero(),
            round0: [Fr::zero(); 4],
            last: [Fr::zero(); 4],
            bits: table.bits,
            wired_bits: table.wired_bits,
            wired_ints: table.wired_ints,
            v: Vec::new(),
            w: Vec::new(),
            wired_index,
            bool_weight,
            wired_weight,
        };
        this.round0 = this.round_on_bits(n);
        this
    }

    /// Final values: committed columns, then wired columns (bits, words).
    pub fn claims(&self) -> (Vec<Fr>, Vec<Fr>) {
        (
            self.v.iter().map(|c| c[0]).collect(),
            self.w.iter().map(|c| c[0]).collect(),
        )
    }

    /// Column-space vectors `(v, w)` from claim lists.
    pub fn column_space(rel: &Relation, committed: &[Fr], wired: &[Fr]) -> (Vec<Fr>, Vec<Fr>) {
        let n = 1 << rel.log_columns;
        let mut v = vec![Fr::zero(); n];
        let mut w = vec![Fr::zero(); n];
        v[..committed.len()].copy_from_slice(committed);
        for (value, &j) in wired
            .iter()
            .zip(rel.wired_bits.iter().chain(&rel.wired_ints))
        {
            w[j] = *value;
        }
        (v, w)
    }

    fn q_bits(&self, row: usize) -> Fr {
        let mut acc = Fr::zero();
        for (column, weight) in self.bits.iter().zip(&self.bool_weight) {
            if column[row] == 1 {
                acc += *weight;
            }
        }
        for ((column, &j), weight) in self
            .wired_bits
            .iter()
            .zip(&self.rel.wired_bits)
            .zip(&self.wired_weight)
        {
            if column[row] == 1 {
                let partner = j < self.rel.committed && self.bits[j][row] == 1;
                acc += if partner { weight.0 } else { weight.1 };
            }
        }
        for (column, &j) in self.wired_ints.iter().zip(&self.rel.wired_ints) {
            acc += self.rel.l2[j] * Fr::from_u64(column[row]);
        }
        acc
    }

    fn q2_bits(&self, lo: usize, hi: usize) -> Fr {
        let rel = self.rel;
        let mut acc = Fr::zero();
        for (j, column) in self.bits.iter().enumerate() {
            if column[lo] != column[hi] {
                acc += rel.gamma_sq[j];
            }
        }
        for (column, &j) in self.wired_bits.iter().zip(&rel.wired_bits) {
            if j >= rel.committed {
                continue;
            }
            let dw =
                i8::try_from(column[hi]).expect("bit") - i8::try_from(column[lo]).expect("bit");
            let dv = i8::try_from(self.bits[j][hi]).expect("bit")
                - i8::try_from(self.bits[j][lo]).expect("bit");
            match dv * dw {
                1 => acc += rel.gamma_cross[j],
                -1 => acc -= rel.gamma_cross[j],
                _ => {}
            }
        }
        acc
    }

    fn q(&self, v: &[Fr], w: &[Fr]) -> Fr {
        let rel = self.rel;
        let mut acc = Fr::zero();
        for (j, &x) in v.iter().enumerate() {
            acc += x * (rel.gamma_sq[j] * x + rel.l1[j]);
        }
        for (&y, &j) in w.iter().zip(&self.wired_index) {
            let partner = if j < rel.committed { v[j] } else { Fr::zero() };
            acc += y * (rel.gamma_cross[j] * partner + rel.l2[j]);
        }
        acc
    }

    fn q2(&self, dv: &[Fr], dw: &[Fr]) -> Fr {
        let rel = self.rel;
        let mut acc = Fr::zero();
        for (j, &x) in dv.iter().enumerate() {
            acc += rel.gamma_sq[j] * x.square();
        }
        for (&y, &j) in dw.iter().zip(&rel.wired_bits) {
            if j < rel.committed {
                acc += rel.gamma_cross[j] * dv[j] * y;
            }
        }
        acc
    }

    fn round_on_bits(&mut self, n: usize) -> [Fr; 4] {
        let tau_v = self.tau[n - 1];
        let e = EqPolynomial::<Fr>::evals(&self.tau[..n - 1], None);
        let (t0, t1, t2) = e
            .par_iter()
            .enumerate()
            .fold(
                || (Fr::zero(), Fr::zero(), Fr::zero()),
                |(s0, s1, s2), (i, &weight)| {
                    (
                        s0 + weight * self.q_bits(2 * i),
                        s1 + weight * self.q_bits(2 * i + 1),
                        s2 + weight * self.q2_bits(2 * i, 2 * i + 1),
                    )
                },
            )
            .reduce(
                || (Fr::zero(), Fr::zero(), Fr::zero()),
                |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
            );
        let (l0, l1) = (Fr::one() - tau_v, tau_v);
        self.claim = l0 * t0 + l1 * t1;
        self.running = self.claim;
        round_poly(l0, l1, t0, t1, t2)
    }

    fn round_on_field(&self) -> [Fr; 4] {
        let n = self.tau.len();
        let tau_v = self.tau[n - 1 - self.round];
        let e = EqPolynomial::<Fr>::evals(&self.tau[..n - 1 - self.round], None);
        let (nv, nw) = (self.v.len(), self.w.len());
        let (t0, t2) = e
            .par_iter()
            .enumerate()
            .fold(
                || {
                    (
                        Fr::zero(),
                        Fr::zero(),
                        vec![Fr::zero(); nv],
                        vec![Fr::zero(); nw],
                        vec![Fr::zero(); nv],
                        vec![Fr::zero(); nw],
                    )
                },
                |(mut s0, mut s2, mut v, mut w, mut dv, mut dw), (i, &weight)| {
                    for (j, column) in self.v.iter().enumerate() {
                        v[j] = column[2 * i];
                        dv[j] = column[2 * i + 1] - column[2 * i];
                    }
                    for (j, column) in self.w.iter().enumerate() {
                        w[j] = column[2 * i];
                        dw[j] = column[2 * i + 1] - column[2 * i];
                    }
                    s0 += weight * self.q(&v, &w);
                    s2 += weight * self.q2(&dv, &dw);
                    (s0, s2, v, w, dv, dw)
                },
            )
            .map(|(s0, s2, ..)| (s0, s2))
            .reduce(|| (Fr::zero(), Fr::zero()), |a, b| (a.0 + b.0, a.1 + b.1));
        let l0 = self.c * (Fr::one() - tau_v);
        let l1 = self.c * tau_v;
        let t1 = (self.running - l0 * t0) * l1.inverse().expect("nonzero eq factor");
        round_poly(l0, l1, t0, t1, t2)
    }
}

impl Instance for RowInstance<'_> {
    fn rounds(&self) -> usize {
        self.tau.len()
    }

    fn input_claim(&self) -> Fr {
        self.claim
    }

    fn round_poly(&mut self) -> Vec<Fr> {
        self.last = if self.round == 0 {
            self.round0
        } else {
            self.round_on_field()
        };
        self.last.to_vec()
    }

    fn bind(&mut self, r: Fr) {
        let n = self.tau.len();
        let tau_v = self.tau[n - 1 - self.round];
        if self.round == 0 {
            let lut = [Fr::zero(), Fr::one() - r, r, Fr::one()];
            let fold_bits = |column: &Vec<u8>| -> Vec<Fr> {
                (0..column.len() / 2)
                    .map(|i| lut[usize::from(column[2 * i] | (column[2 * i + 1] << 1))])
                    .collect()
            };
            self.v = self.bits.par_iter().map(fold_bits).collect();
            let mut w: Vec<Vec<Fr>> = self.wired_bits.par_iter().map(fold_bits).collect();
            w.par_extend(self.wired_ints.par_iter().map(|column| {
                (0..column.len() / 2)
                    .map(|i| {
                        let (lo, hi) =
                            (Fr::from_u64(column[2 * i]), Fr::from_u64(column[2 * i + 1]));
                        lo + r * (hi - lo)
                    })
                    .collect()
            }));
            self.w = w;
            self.bits = Vec::new();
            self.wired_bits = Vec::new();
            self.wired_ints = Vec::new();
        } else {
            self.v
                .par_iter_mut()
                .chain(self.w.par_iter_mut())
                .for_each(|column| {
                    let half = column.len() / 2;
                    for i in 0..half {
                        column[i] = column[2 * i] + r * (column[2 * i + 1] - column[2 * i]);
                    }
                    column.truncate(half);
                });
        }
        self.running = horner(&self.last, r);
        self.c *= eq_scalar(tau_v, r);
        self.round += 1;
    }
}
