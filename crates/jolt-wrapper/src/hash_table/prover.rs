//! The row sumcheck `Σ_row eq(τ, row) · Q(row)` as a batch member: round 0
//! runs on the borrowed 0/1 columns (no field multiplications), later rounds
//! on the bound field columns; `s(X) = c · l(X) · t(X)` with `t(1)` recovered
//! from the running claim. Row-index bits are bound high-to-low like every
//! stream column (round `i` binds the top remaining bit against `τ[i]`).

use jolt_field::{Field, Fr, One, Ring, Zero};
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_sumcheck::prover::ProveRounds;
use jolt_sumcheck::SumcheckError;
use rayon::prelude::*;

use super::layout::{wired_columns, ColumnEvals, Relation, COMMITTED, WIRED_BITS};
use super::table::HashTable;

/// Borrows the table's columns until round 0 is bound; the stream packs and
/// commits the same columns without a copy.
pub struct HashTableProver<'a> {
    relation: &'a Relation,
    tau: Vec<Fr>,
    round: usize,
    /// `Π eq(τ_bound, r_bound)` over the rounds bound so far.
    eq_prefix: Fr,
    claim: Fr,
    round0: [Fr; 4],
    table: &'a HashTable,
    /// Bound committed columns and wired columns (bits, then words).
    v: Vec<Vec<Fr>>,
    w: Vec<Vec<Fr>>,
    wired_index: Vec<usize>,
    /// `γ̃_j + L1_j`: a set committed bit's contribution.
    bool_weight: Vec<Fr>,
    /// `(γ̃'_j + L2_j, L2_j)`: a set wired bit's contribution with its
    /// committed partner set / clear.
    wired_weight: Vec<(Fr, Fr)>,
}

fn eq_scalar(a: Fr, b: Fr) -> Fr {
    let ab = a * b;
    Fr::one() - a - b + ab + ab
}

/// `s(X) = (l0 + (l1 − l0)X) · (t0 + (t1 − t0 − t2)X + t2X²)`.
fn round_poly(l0: Fr, l1: Fr, t0: Fr, t1: Fr, t2: Fr) -> [Fr; 4] {
    let m = l1 - l0;
    let u = t1 - t0 - t2;
    [l0 * t0, l0 * u + m * t0, l0 * t2 + m * u, m * t2]
}

impl<'a> HashTableProver<'a> {
    /// # Panics
    ///
    /// Panics unless `tau.len() == table.log_rows`.
    pub fn new(relation: &'a Relation, table: &'a HashTable, tau: Vec<Fr>) -> Self {
        assert_eq!(tau.len(), table.log_rows, "one τ per row variable");
        let bool_weight = (0..COMMITTED)
            .map(|j| relation.gamma_sq[j] + relation.l1[j])
            .collect();
        let wired_index = wired_columns();
        let wired_weight = wired_index[..WIRED_BITS]
            .iter()
            .map(|&j| (relation.gamma_cross[j] + relation.l2[j], relation.l2[j]))
            .collect();
        let mut this = Self {
            relation,
            tau,
            round: 0,
            eq_prefix: Fr::one(),
            claim: Fr::zero(),
            round0: [Fr::zero(); 4],
            table,
            v: Vec::new(),
            w: Vec::new(),
            wired_index,
            bool_weight,
            wired_weight,
        };
        this.round0 = this.round_on_bits();
        this
    }

    /// `Σ_row eq(τ, row) · Q(row)`: zero for a table satisfying the relation.
    pub fn input_claim(&self) -> Fr {
        self.claim
    }

    /// Every column's value at the bound point; valid after `finish_rounds`.
    pub fn column_evals(&self) -> ColumnEvals {
        let (bits, words) = self.w.split_at(WIRED_BITS);
        ColumnEvals {
            committed: self.v.iter().map(|c| c[0]).collect(),
            wired_bits: bits.iter().map(|c| c[0]).collect(),
            wired_words: words.iter().map(|c| c[0]).collect(),
        }
    }

    fn q_bits(&self, row: usize) -> Fr {
        let table = self.table;
        let mut acc = Fr::zero();
        for (column, weight) in table.bits.iter().zip(&self.bool_weight) {
            if column[row] == 1 {
                acc += *weight;
            }
        }
        for ((column, &j), weight) in table
            .wired_bits
            .iter()
            .zip(&self.wired_index)
            .zip(&self.wired_weight)
        {
            if column[row] == 1 {
                acc += if table.bits[j][row] == 1 {
                    weight.0
                } else {
                    weight.1
                };
            }
        }
        for (column, &j) in table
            .wired_words
            .iter()
            .zip(&self.wired_index[WIRED_BITS..])
        {
            acc += self.relation.l2[j] * Fr::from_u32(column[row]);
        }
        acc
    }

    /// The quadratic coefficient of `Q` along the pair `(lo, hi)`.
    fn q2_bits(&self, lo: usize, hi: usize) -> Fr {
        let (rel, table) = (self.relation, self.table);
        let mut acc = Fr::zero();
        for (j, column) in table.bits.iter().enumerate() {
            if column[lo] != column[hi] {
                acc += rel.gamma_sq[j];
            }
        }
        for (column, &j) in table.wired_bits.iter().zip(&self.wired_index) {
            let dw = i8::from(column[hi] != 0) - i8::from(column[lo] != 0);
            let dv = i8::from(table.bits[j][hi] != 0) - i8::from(table.bits[j][lo] != 0);
            match dv * dw {
                1 => acc += rel.gamma_cross[j],
                -1 => acc -= rel.gamma_cross[j],
                _ => {}
            }
        }
        acc
    }

    fn q(&self, v: &[Fr], w: &[Fr]) -> Fr {
        let rel = self.relation;
        let mut acc = Fr::zero();
        for (j, &x) in v.iter().enumerate() {
            acc += x * (rel.gamma_sq[j] * x + rel.l1[j]);
        }
        for (&y, &j) in w.iter().zip(&self.wired_index) {
            let partner = if j < COMMITTED { v[j] } else { Fr::zero() };
            acc += y * (rel.gamma_cross[j] * partner + rel.l2[j]);
        }
        acc
    }

    fn q2(&self, dv: &[Fr], dw: &[Fr]) -> Fr {
        let rel = self.relation;
        let mut acc = Fr::zero();
        for (j, &x) in dv.iter().enumerate() {
            acc += rel.gamma_sq[j] * x.square();
        }
        for (&y, &j) in dw.iter().zip(&self.wired_index[..WIRED_BITS]) {
            acc += rel.gamma_cross[j] * dv[j] * y;
        }
        acc
    }

    fn round_on_bits(&mut self) -> [Fr; 4] {
        let tau_v = self.tau[0];
        let e = EqPolynomial::<Fr>::evals(&self.tau[1..], None);
        let half = e.len();
        let (t0, t1, t2) = e
            .par_iter()
            .enumerate()
            .fold(
                || (Fr::zero(), Fr::zero(), Fr::zero()),
                |(s0, s1, s2), (i, &weight)| {
                    (
                        s0 + weight * self.q_bits(i),
                        s1 + weight * self.q_bits(i + half),
                        s2 + weight * self.q2_bits(i, i + half),
                    )
                },
            )
            .reduce(
                || (Fr::zero(), Fr::zero(), Fr::zero()),
                |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
            );
        let (l0, l1) = (Fr::one() - tau_v, tau_v);
        self.claim = l0 * t0 + l1 * t1;
        round_poly(l0, l1, t0, t1, t2)
    }

    fn round_on_field(&self, running: Fr) -> Result<[Fr; 4], SumcheckError<Fr>> {
        let tau_v = self.tau[self.round];
        let e = EqPolynomial::<Fr>::evals(&self.tau[self.round + 1..], None);
        let half = e.len();
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
                        v[j] = column[i];
                        dv[j] = column[i + half] - column[i];
                    }
                    for (j, column) in self.w.iter().enumerate() {
                        w[j] = column[i];
                        dw[j] = column[i + half] - column[i];
                    }
                    s0 += weight * self.q(&v, &w);
                    s2 += weight * self.q2(&dv, &dw);
                    (s0, s2, v, w, dv, dw)
                },
            )
            .map(|(s0, s2, ..)| (s0, s2))
            .reduce(|| (Fr::zero(), Fr::zero()), |a, b| (a.0 + b.0, a.1 + b.1));
        let l0 = self.eq_prefix * (Fr::one() - tau_v);
        let l1 = self.eq_prefix * tau_v;
        let l1_inv = l1.inverse().ok_or(SumcheckError::RoundCheckFailed {
            round: self.round,
            expected: running,
            actual: l0 * t0,
        })?;
        let t1 = (running - l0 * t0) * l1_inv;
        Ok(round_poly(l0, l1, t0, t1, t2))
    }

    fn bind(&mut self, r: Fr) {
        let tau_v = self.tau[self.round];
        if self.round == 0 {
            let table = self.table;
            let lut = [Fr::zero(), Fr::one() - r, r, Fr::one()];
            let fold_bits = |column: &Vec<u8>| -> Vec<Fr> {
                let half = column.len() / 2;
                (0..half)
                    .map(|i| lut[usize::from(column[i] | (column[i + half] << 1))])
                    .collect()
            };
            self.v = table.bits.par_iter().map(fold_bits).collect();
            let mut w: Vec<Vec<Fr>> = table.wired_bits.par_iter().map(fold_bits).collect();
            w.par_extend(table.wired_words.par_iter().map(|column| {
                let half = column.len() / 2;
                (0..half)
                    .map(|i| {
                        let (lo, hi) = (Fr::from_u32(column[i]), Fr::from_u32(column[i + half]));
                        lo + r * (hi - lo)
                    })
                    .collect()
            }));
            self.w = w;
        } else {
            self.v
                .par_iter_mut()
                .chain(self.w.par_iter_mut())
                .for_each(|column| {
                    let half = column.len() / 2;
                    let (lo, hi) = column.split_at_mut(half);
                    for (lo, hi) in lo.iter_mut().zip(hi.iter()) {
                        *lo += r * (*hi - *lo);
                    }
                    column.truncate(half);
                });
        }
        self.eq_prefix *= eq_scalar(tau_v, r);
        self.round += 1;
    }
}

impl ProveRounds<Fr> for HashTableProver<'_> {
    fn num_rounds(&self) -> usize {
        self.tau.len()
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        if let Some(r) = bind {
            self.bind(r);
        }
        if round != self.round {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: Fr::from_u64(self.round as u64),
                actual: Fr::from_u64(round as u64),
            });
        }
        let coefficients = if round == 0 {
            self.round0
        } else {
            self.round_on_field(previous_claim)?
        };
        Ok(UnivariatePoly::new(coefficients.to_vec()))
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        self.bind(bind);
        Ok(())
    }
}
