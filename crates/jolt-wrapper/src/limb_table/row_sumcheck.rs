//! The row member: the row-major matrix of every claimed and public column
//! folded round by round, and the `eq(τ, ·)` column it carries.

use jolt_field::{Fr, Ring, Zero};
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_sumcheck::prover::ProveRounds;
use jolt_sumcheck::SumcheckError;
use rayon::prelude::*;

use crate::stream::Column;

use super::layout::LOG_ROWS;
use super::relation::{Col, PublicEvals, RowRelation};

/// Linear extrapolation of a row pair to `X = x`: `lo + x·(hi − lo)`.
fn extrapolate(out: &mut [Fr], lo: &[Fr], hi: &[Fr], x: Fr) {
    for ((slot, &l), &h) in out.iter_mut().zip(lo).zip(hi) {
        *slot = l + x * (h - l);
    }
}

/// The row member's columns in their commitment-time scalar types.
pub struct RowMatrix {
    indices: Vec<MatrixIndex>,
    small: Vec<u32>,
    field: Vec<Fr>,
    small_width: usize,
    field_width: usize,
    rows: usize,
}

#[derive(Clone, Copy)]
enum MatrixIndex {
    Small(usize, SmallType),
    Field(usize),
}

#[derive(Clone, Copy)]
enum SmallType {
    Bit,
    U16,
    U32,
}

enum MatrixSource<'a> {
    Typed(&'a RowMatrix),
    Field(&'a [Vec<Fr>]),
}

impl MatrixSource<'_> {
    fn fill_row(&self, row: usize, values: &mut [Fr]) {
        match self {
            Self::Typed(matrix) => matrix.fill_row(row, values),
            Self::Field(columns) => {
                for (value, column) in values.iter_mut().zip(*columns) {
                    *value = column[row];
                }
            }
        }
    }

    fn bind_rows(&self, lo: &mut [Fr], hi: &mut [Fr], rows: [usize; 4], bind: Fr) {
        match self {
            Self::Typed(matrix) => matrix.bind_rows(lo, hi, rows, bind),
            Self::Field(columns) => {
                for (column, values) in columns.iter().enumerate() {
                    let a = values[rows[0]];
                    let b = values[rows[2]];
                    lo[column] = a + bind * (b - a);
                    let a = values[rows[1]];
                    let b = values[rows[3]];
                    hi[column] = a + bind * (b - a);
                }
            }
        }
    }
}

impl RowMatrix {
    pub(crate) fn new(columns: Vec<Column>) -> Self {
        assert_eq!(columns.len(), Col::WIDTH);
        let rows = 1usize << LOG_ROWS;
        assert!(columns.iter().all(|column| column.len() == rows));
        let mut small_width = 0;
        let mut field_width = 0;
        let indices = columns
            .iter()
            .map(|column| match column {
                Column::Bits(_) => {
                    let index = MatrixIndex::Small(small_width, SmallType::Bit);
                    small_width += 1;
                    index
                }
                Column::U16(_) => {
                    let index = MatrixIndex::Small(small_width, SmallType::U16);
                    small_width += 1;
                    index
                }
                Column::U32(_) => {
                    let index = MatrixIndex::Small(small_width, SmallType::U32);
                    small_width += 1;
                    index
                }
                Column::Fr(_) => {
                    let index = MatrixIndex::Field(field_width);
                    field_width += 1;
                    index
                }
            })
            .collect();
        let small_sources = columns
            .iter()
            .filter(|column| !matches!(column, Column::Fr(_)))
            .collect::<Vec<_>>();
        let field_sources = columns
            .iter()
            .filter_map(|column| match column {
                Column::Fr(values) => Some(values),
                Column::Bits(_) | Column::U16(_) | Column::U32(_) => None,
            })
            .collect::<Vec<_>>();
        let mut small = vec![0u32; rows * small_width];
        small
            .par_chunks_mut(small_width)
            .enumerate()
            .for_each(|(row, target)| {
                for (value, source) in target.iter_mut().zip(&small_sources) {
                    *value = match source {
                        Column::Bits(values) => u32::from(values[row]),
                        Column::U16(values) => u32::from(values[row]),
                        Column::U32(values) => values[row],
                        Column::Fr(_) => unreachable!("filtered small columns"),
                    };
                }
            });
        let mut field = vec![Fr::zero(); rows * field_width];
        field
            .par_chunks_mut(field_width)
            .enumerate()
            .for_each(|(row, target)| {
                for (value, source) in target.iter_mut().zip(&field_sources) {
                    *value = source[row];
                }
            });
        Self {
            indices,
            small,
            field,
            small_width,
            field_width,
            rows,
        }
    }

    pub fn column(&self, column: usize) -> Column {
        match self.indices[column] {
            MatrixIndex::Small(index, SmallType::Bit) => Column::Bits(
                (0..self.rows)
                    .map(|row| self.small[row * self.small_width + index] as u8)
                    .collect(),
            ),
            MatrixIndex::Small(index, SmallType::U16) => Column::U16(
                (0..self.rows)
                    .map(|row| self.small[row * self.small_width + index] as u16)
                    .collect(),
            ),
            MatrixIndex::Small(index, SmallType::U32) => Column::U32(
                (0..self.rows)
                    .map(|row| self.small[row * self.small_width + index])
                    .collect(),
            ),
            MatrixIndex::Field(_) => Column::Fr(self.field_column(column)),
        }
    }

    pub fn value(&self, column: usize, row: usize) -> Fr {
        match self.indices[column] {
            MatrixIndex::Small(index, _) => {
                Fr::from_u64(u64::from(self.small[row * self.small_width + index]))
            }
            MatrixIndex::Field(index) => self.field[row * self.field_width + index],
        }
    }

    pub fn field_column(&self, column: usize) -> Vec<Fr> {
        (0..self.rows)
            .into_par_iter()
            .map(|row| self.value(column, row))
            .collect()
    }

    fn fill_row(&self, row: usize, values: &mut [Fr]) {
        let small = &self.small[row * self.small_width..(row + 1) * self.small_width];
        let field = &self.field[row * self.field_width..(row + 1) * self.field_width];
        for (column, index) in self.indices.iter().enumerate() {
            values[column] = match *index {
                MatrixIndex::Small(index, _) => Fr::from_u64(u64::from(small[index])),
                MatrixIndex::Field(index) => field[index],
            };
        }
    }

    fn bind_rows(&self, lo: &mut [Fr], hi: &mut [Fr], rows: [usize; 4], bind: Fr) {
        for (column, index) in self.indices.iter().enumerate() {
            let value = |row| match *index {
                MatrixIndex::Small(index, _) => {
                    Fr::from_u64(u64::from(self.small[row * self.small_width + index]))
                }
                MatrixIndex::Field(index) => self.field[row * self.field_width + index],
            };
            let a = value(rows[0]);
            let b = value(rows[2]);
            lo[column] = a + bind * (b - a);
            let a = value(rows[1]);
            let b = value(rows[3]);
            hi[column] = a + bind * (b - a);
        }
    }
}

/// The row member as a batch member: a row-major matrix of every claimed and
/// public column, folded round by round from the most significant row bit
/// (round `i` pairs row `j` with `j + rows/2` and writes the bound row at
/// `j`, so the stage point is big-endian like the stream's). The `eq(τ,·)`
/// column is part of the matrix, so the round polynomial evaluates the
/// degree-`d` summand at `d + 1` points.
pub struct RowSumcheck<'a> {
    relation: &'a RowRelation,
    source: MatrixSource<'a>,
    matrix: Vec<Fr>,
    rows: usize,
    round: usize,
    points: Vec<Fr>,
}

impl<'a> RowSumcheck<'a> {
    pub fn new(relation: &'a RowRelation, columns: &'a [Vec<Fr>]) -> Self {
        assert_eq!(columns.len(), Col::WIDTH);
        assert!(columns
            .iter()
            .all(|column| column.len() == 1usize << LOG_ROWS));
        Self::from_source(relation, MatrixSource::Field(columns))
    }

    pub fn new_typed(relation: &'a RowRelation, matrix: &'a RowMatrix) -> Self {
        Self::from_source(relation, MatrixSource::Typed(matrix))
    }

    fn from_source(relation: &'a RowRelation, source: MatrixSource<'a>) -> Self {
        let rows = 1usize << LOG_ROWS;
        Self {
            relation,
            source,
            matrix: Vec::new(),
            rows,
            round: 0,
            points: (0..Self::degree() as u64).map(Fr::from_u64).collect(),
        }
    }

    /// Summand degree in each variable (`eq` included).
    pub const fn degree() -> usize {
        RowRelation::degree()
    }

    /// The zero-check's declared input claim; every round pins
    /// `p(0) + p(1)` to its running claim.
    pub fn input_claim(&self) -> Fr {
        Fr::zero()
    }

    fn coefficients(evals: &[Fr], leading: Fr) -> Vec<Fr> {
        let mut coefficients = UnivariatePoly::from_evals(evals).into_coefficients();
        coefficients.resize(evals.len() + 1, Fr::zero());
        let mut vanishing = vec![Fr::from_u64(1)];
        for root in (0u64..).take(evals.len()) {
            let root = Fr::from_u64(root);
            let mut next = vec![Fr::zero(); vanishing.len() + 1];
            for (power, coefficient) in vanishing.into_iter().enumerate() {
                next[power] -= root * coefficient;
                next[power + 1] += coefficient;
            }
            vanishing = next;
        }
        for (coefficient, value) in coefficients.iter_mut().zip(vanishing) {
            *coefficient += leading * value;
        }
        while coefficients.len() > 2 && coefficients.last() == Some(&Fr::zero()) {
            let _ = coefficients.pop();
        }
        coefficients
    }

    fn round_poly_typed(&self, previous_claim: Fr) -> Vec<Fr> {
        let relation = self.relation;
        let width = Col::WIDTH;
        let half = self.rows / 2;
        let mut evals = (0..half)
            .into_par_iter()
            .fold(
                || {
                    (
                        vec![Fr::zero(); self.points.len()],
                        Fr::zero(),
                        vec![Fr::zero(); width],
                        vec![Fr::zero(); width],
                        vec![Fr::zero(); width],
                    )
                },
                |(mut acc, mut leading, mut lo, mut hi, mut scratch), row| {
                    self.source.fill_row(row, &mut lo);
                    self.source.fill_row(row + half, &mut hi);
                    leading += relation.leading_coefficient(&lo, &hi);
                    // Before any bind, every row-local constraint vanishes;
                    // only the global linear identities contribute at 0/1.
                    acc[0] += relation.linear_summand(&lo);
                    for (i, &point) in self.points.iter().enumerate().skip(2) {
                        extrapolate(&mut scratch, &lo, &hi, point);
                        acc[i] += relation.summand(&scratch);
                    }
                    (acc, leading, lo, hi, scratch)
                },
            )
            .map(|(acc, leading, _, _, _)| (acc, leading))
            .reduce(
                || (vec![Fr::zero(); self.points.len()], Fr::zero()),
                |(a, a_leading), (b, b_leading)| {
                    (
                        a.iter().zip(&b).map(|(x, y)| *x + *y).collect(),
                        a_leading + b_leading,
                    )
                },
            );
        evals.0[1] = previous_claim - evals.0[0];
        Self::coefficients(&evals.0, evals.1)
    }

    fn bind_typed_and_round_poly(&mut self, r: Fr, previous_claim: Fr) -> Vec<Fr> {
        let relation = self.relation;
        let width = Col::WIDTH;
        let half = self.rows / 2;
        let quarter = half / 2;
        let mut matrix = vec![Fr::zero(); half * width];
        let (lo, hi) = matrix.split_at_mut(quarter * width);
        let mut evals = lo
            .par_chunks_mut(width)
            .zip(hi.par_chunks_mut(width))
            .enumerate()
            .fold(
                || {
                    (
                        vec![Fr::zero(); self.points.len()],
                        Fr::zero(),
                        vec![Fr::zero(); width],
                    )
                },
                |(mut acc, mut leading, mut scratch), (row, (lo, hi))| {
                    self.source.bind_rows(
                        lo,
                        hi,
                        [row, row + quarter, row + half, row + quarter + half],
                        r,
                    );
                    leading += relation.leading_coefficient(lo, hi);
                    acc[0] += relation.summand(lo);
                    for (i, &point) in self.points.iter().enumerate().skip(2) {
                        extrapolate(&mut scratch, lo, hi, point);
                        acc[i] += relation.summand(&scratch);
                    }
                    (acc, leading, scratch)
                },
            )
            .map(|(acc, leading, _)| (acc, leading))
            .reduce(
                || (vec![Fr::zero(); self.points.len()], Fr::zero()),
                |(a, a_leading), (b, b_leading)| {
                    (
                        a.iter().zip(&b).map(|(x, y)| *x + *y).collect(),
                        a_leading + b_leading,
                    )
                },
            );
        evals.0[1] = previous_claim - evals.0[0];
        self.matrix = matrix;
        self.rows = half;
        self.round += 1;
        Self::coefficients(&evals.0, evals.1)
    }

    fn bind_dense_and_round_poly(&mut self, r: Fr, previous_claim: Fr) -> Vec<Fr> {
        let relation = self.relation;
        let width = Col::WIDTH;
        let half = self.rows / 2;
        let quarter = half / 2;
        let (q0, rest) = self.matrix[..self.rows * width].split_at_mut(quarter * width);
        let (q1, rest) = rest.split_at_mut(quarter * width);
        let (q2, q3) = rest.split_at_mut(quarter * width);
        let mut evals = q0
            .par_chunks_mut(width)
            .zip(q1.par_chunks_mut(width))
            .zip(q2.par_chunks(width))
            .zip(q3.par_chunks(width))
            .fold(
                || {
                    (
                        vec![Fr::zero(); self.points.len()],
                        Fr::zero(),
                        vec![Fr::zero(); width],
                    )
                },
                |(mut acc, mut leading, mut scratch), (((lo, hi), upper_lo), upper_hi)| {
                    for column in 0..width {
                        lo[column] += r * (upper_lo[column] - lo[column]);
                        hi[column] += r * (upper_hi[column] - hi[column]);
                    }
                    leading += relation.leading_coefficient(lo, hi);
                    acc[0] += relation.summand(lo);
                    for (i, &point) in self.points.iter().enumerate().skip(2) {
                        extrapolate(&mut scratch, lo, hi, point);
                        acc[i] += relation.summand(&scratch);
                    }
                    (acc, leading, scratch)
                },
            )
            .map(|(acc, leading, _)| (acc, leading))
            .reduce(
                || (vec![Fr::zero(); self.points.len()], Fr::zero()),
                |(a, a_leading), (b, b_leading)| {
                    (
                        a.iter().zip(&b).map(|(x, y)| *x + *y).collect(),
                        a_leading + b_leading,
                    )
                },
            );
        evals.0[1] = previous_claim - evals.0[0];
        self.rows = half;
        self.round += 1;
        Self::coefficients(&evals.0, evals.1)
    }

    fn bind_final(&mut self, r: Fr) {
        let width = Col::WIDTH;
        let half = self.rows / 2;
        let (lo, hi) = self.matrix[..self.rows * width].split_at_mut(half * width);
        lo.par_chunks_mut(width)
            .zip(hi.par_chunks(width))
            .for_each(|(lo, hi)| {
                for (l, &h) in lo.iter_mut().zip(hi) {
                    *l += r * (h - *l);
                }
            });
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
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        if let Some(r) = bind {
            let coefficients = if self.matrix.is_empty() {
                self.bind_typed_and_round_poly(r, previous_claim)
            } else {
                self.bind_dense_and_round_poly(r, previous_claim)
            };
            debug_assert_eq!(round, self.round);
            return Ok(UnivariatePoly::new(coefficients));
        }
        debug_assert_eq!(round, self.round);
        Ok(UnivariatePoly::new(self.round_poly_typed(previous_claim)))
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        self.bind_final(bind);
        Ok(())
    }

    fn append_bound_values(&self, values: &mut Vec<Fr>) -> Result<(), SumcheckError<Fr>> {
        values.extend_from_slice(&self.final_row()[..Col::CLAIMED]);
        Ok(())
    }
}

/// `eq(τ, x)` over the rows for the big-endian `tau`.
pub fn eq_tau_column(tau: &[Fr]) -> Vec<Fr> {
    EqPolynomial::<Fr>::evals(tau, None)
}
