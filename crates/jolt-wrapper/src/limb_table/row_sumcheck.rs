//! The row member: the row-major matrix of every claimed and public column
//! folded round by round, and the `eq(τ, ·)` column it carries.

use jolt_field::{Fr, Ring, Zero};
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_sumcheck::prover::ProveRounds;
use jolt_sumcheck::SumcheckError;
use rayon::prelude::*;

use super::layout::LOG_ROWS;
use super::relation::{Col, PublicEvals, RowRelation};

/// Linear extrapolation of a row pair to `X = x`: `even + x·(odd − even)`.
fn extrapolate(out: &mut [Fr], even: &[Fr], odd: &[Fr], x: Fr) {
    for ((slot, &e), &o) in out.iter_mut().zip(even).zip(odd) {
        *slot = e + x * (o - e);
    }
}

/// The row member as a batch member: a row-major matrix of every claimed and
/// public column, folded round by round (low bit first). The `eq(τ,·)`
/// column is part of the matrix, so the round polynomial is the plain
/// degree-`d+1` evaluation of the summand at `d + 2` points.
pub struct RowSumcheck<'a> {
    relation: &'a RowRelation,
    matrix: Vec<Fr>,
    scratch: Vec<Fr>,
    rows: usize,
    round: usize,
    points: Vec<Fr>,
}

impl<'a> RowSumcheck<'a> {
    /// `columns[i]` is matrix column `i` (`Col::WIDTH` columns of `2^LOG_ROWS`
    /// rows: claimed columns then the public ones).
    pub fn new(relation: &'a RowRelation, columns: &[Vec<Fr>]) -> Self {
        assert_eq!(columns.len(), Col::WIDTH);
        let rows = 1usize << LOG_ROWS;
        for column in columns {
            assert_eq!(column.len(), rows);
        }
        let mut matrix = vec![Fr::zero(); rows * Col::WIDTH];
        matrix
            .par_chunks_mut(Col::WIDTH)
            .enumerate()
            .for_each(|(row, slot)| {
                for (value, column) in slot.iter_mut().zip(columns) {
                    *value = column[row];
                }
            });
        Self {
            relation,
            matrix,
            scratch: Vec::new(),
            rows,
            round: 0,
            points: (0..=(Self::degree() + 1) as u64)
                .map(Fr::from_u64)
                .collect(),
        }
    }

    /// Summand degree in each variable (`eq` included).
    pub const fn degree() -> usize {
        RowRelation::degree()
    }

    /// `Σ_row summand(row)` — zero for an honest witness.
    pub fn input_claim(&self) -> Fr {
        self.matrix
            .par_chunks(Col::WIDTH)
            .map(|row| self.relation.summand(row))
            .sum()
    }

    fn round_poly(&self) -> Vec<Fr> {
        let relation = self.relation;
        let width = Col::WIDTH;
        let evals: Vec<Fr> = self.matrix[..self.rows * width]
            .par_chunks(2 * width)
            .fold(
                || (vec![Fr::zero(); self.points.len()], vec![Fr::zero(); width]),
                |(mut acc, mut scratch), pair| {
                    let (even, odd) = pair.split_at(width);
                    acc[0] += relation.summand(even);
                    acc[1] += relation.summand(odd);
                    for (i, &point) in self.points.iter().enumerate().skip(2) {
                        extrapolate(&mut scratch, even, odd, point);
                        acc[i] += relation.summand(&scratch);
                    }
                    (acc, scratch)
                },
            )
            .map(|(acc, _)| acc)
            .reduce(
                || vec![Fr::zero(); self.points.len()],
                |a, b| a.iter().zip(&b).map(|(x, y)| *x + *y).collect(),
            );
        let mut coefficients = UnivariatePoly::from_evals(&evals).into_coefficients();
        while coefficients.len() > 2 && coefficients.last() == Some(&Fr::zero()) {
            let _ = coefficients.pop();
        }
        coefficients
    }

    fn bind(&mut self, r: Fr) {
        let width = Col::WIDTH;
        let half = self.rows / 2;
        if self.scratch.len() < half * width {
            self.scratch.resize(half * width, Fr::zero());
        }
        self.scratch[..half * width]
            .par_chunks_mut(width)
            .zip(self.matrix[..self.rows * width].par_chunks(2 * width))
            .for_each(|(out, pair)| {
                let (even, odd) = pair.split_at(width);
                extrapolate(out, even, odd, r);
            });
        std::mem::swap(&mut self.matrix, &mut self.scratch);
        self.rows = half;
        self.round += 1;
    }

    /// Final evaluations after every round: the claimed columns, then the
    /// public columns (which the verifier recomputes).
    pub fn final_row(&self) -> &[Fr] {
        assert_eq!(self.rows, 1, "sumcheck not finished");
        &self.matrix[..Col::WIDTH]
    }

    /// Claimed column evaluations at the point (`Col::CLAIMED` values).
    pub fn claims(&self) -> Vec<Fr> {
        self.final_row()[..Col::CLAIMED].to_vec()
    }

    /// The public columns' final values (the verifier's oracle in tests).
    pub fn public_evals(&self) -> PublicEvals {
        let p = &self.final_row()[Col::CLAIMED..];
        PublicEvals {
            eq_tau: p[0],
            copy_kernel: p[1],
            sel: p[2],
            is_gt: p[3],
            is_g1: p[4],
            is_g2: p[5],
            s0: p[6],
            coord: p[7],
            constancy: p[8],
            small: p[9],
            id: p[10],
        }
    }
}

impl PublicEvals {
    /// Matrix row layout helper: the public part of a final row.
    pub fn to_row(&self) -> [Fr; Col::PUBLIC] {
        self.as_array()
    }
}

impl ProveRounds<Fr> for RowSumcheck<'_> {
    fn num_rounds(&self) -> usize {
        LOG_ROWS
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        _previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        if let Some(r) = bind {
            self.bind(r);
        }
        debug_assert_eq!(round, self.round);
        Ok(UnivariatePoly::new(self.round_poly()))
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        self.bind(bind);
        Ok(())
    }
}

/// `eq(τ, x)` over the rows for the big-endian `tau`.
pub fn eq_tau_column(tau: &[Fr]) -> Vec<Fr> {
    EqPolynomial::<Fr>::evals(tau, None)
}
