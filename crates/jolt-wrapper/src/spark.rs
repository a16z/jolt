//! Sparse matrix MLE checking with LogUp row/column lookups.

use std::collections::BTreeMap;

use jolt_crypto::{Bn254, HomomorphicCommitment};
use jolt_field::{Field, Fr, One, Ring, Zero};
use jolt_hyperkzg::HyperKZGProverSetup;
use jolt_poly::{BindingOrder, EqPolynomial, Polynomial, UnivariatePoly};
use jolt_r1cs::{ConstraintMatrices, SparseRow};
use jolt_sumcheck::prover::ProveRounds;
use jolt_sumcheck::SumcheckError;
use rayon::prelude::*;
use thiserror::Error;

use crate::stream::{commit_packed, Column, Commitment, PackedColumns, PackingLayout, StreamError};

pub const FIXED_COLUMNS: usize = 7;
pub const WITNESS_COLUMNS: usize = 6;
pub const TOTAL_COLUMNS: usize = FIXED_COLUMNS + WITNESS_COLUMNS;
pub const DEGREE: usize = 3;

const ROW: usize = 0;
const COL: usize = 1;
const VAL_A: usize = 2;
const VAL_B: usize = 3;
const VAL_C: usize = 4;
const ROW_MULTIPLICITY: usize = 5;
const COL_MULTIPLICITY: usize = 6;

const E_ROW: usize = 0;
const E_COL: usize = 1;
const H_LEFT_ROW: usize = 2;
const H_LEFT_COL: usize = 3;
const H_RIGHT_ROW: usize = 4;
const H_RIGHT_COL: usize = 5;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SparkError {
    #[error("SPARK column range [{start}, {end}) exceeds {total} R1CS columns")]
    ColumnRange {
        start: usize,
        end: usize,
        total: usize,
    },
    #[error("SPARK row point has {actual} coordinates, expected {expected}")]
    RowPoint { expected: usize, actual: usize },
    #[error("SPARK column point has {actual} coordinates, expected {expected}")]
    ColumnPoint { expected: usize, actual: usize },
    #[error("SPARK challenge point has {actual} coordinates, expected {expected}")]
    ChallengePoint { expected: usize, actual: usize },
    #[error("SPARK LogUp denominator is zero")]
    ZeroDenominator,
    #[error("SPARK relation input claim mismatch")]
    ClaimMismatch,
    #[error("SPARK common row domain has {actual} rows, need at least {minimum}")]
    CommonRows { minimum: usize, actual: usize },
    #[error("SPARK packed columns: {0}")]
    Stream(#[from] StreamError),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparkTables {
    fixed: [Vec<Fr>; FIXED_COLUMNS],
    indices: [Vec<usize>; 2],
    row_vars: usize,
    col_vars: usize,
    entry_vars: usize,
}

impl SparkTables {
    pub fn new(
        matrices: &ConstraintMatrices<Fr>,
        column_start: usize,
        column_count: usize,
    ) -> Result<Self, SparkError> {
        let column_end = column_start
            .checked_add(column_count)
            .ok_or(SparkError::ColumnRange {
                start: column_start,
                end: usize::MAX,
                total: matrices.num_vars,
            })?;
        if column_count == 0 || column_end > matrices.num_vars {
            return Err(SparkError::ColumnRange {
                start: column_start,
                end: column_end,
                total: matrices.num_vars,
            });
        }
        let mut entries = BTreeMap::<(usize, usize), [Fr; 3]>::new();
        add_matrix(&mut entries, &matrices.a, column_start, column_end, 0);
        add_matrix(
            &mut entries,
            &matrices.b,
            column_start,
            column_end,
            VAL_B - VAL_A,
        );
        add_matrix(
            &mut entries,
            &matrices.c,
            column_start,
            column_end,
            VAL_C - VAL_A,
        );
        entries.retain(|_, values| values.iter().any(|value| !value.is_zero()));

        let row_count = matrices.num_constraints.next_power_of_two();
        let col_count = column_count.next_power_of_two();
        let entry_count = entries
            .len()
            .max(row_count)
            .max(col_count)
            .next_power_of_two();
        let mut fixed: [Vec<Fr>; FIXED_COLUMNS] =
            std::array::from_fn(|_| Vec::with_capacity(entry_count));
        let mut indices: [Vec<usize>; 2] = std::array::from_fn(|_| Vec::with_capacity(entry_count));
        for ((row, column), values) in entries {
            let column = column - column_start;
            indices[ROW].push(row);
            indices[COL].push(column);
            fixed[ROW].push(Fr::from_u64(row as u64));
            fixed[COL].push(Fr::from_u64(column as u64));
            fixed[VAL_A].push(values[0]);
            fixed[VAL_B].push(values[1]);
            fixed[VAL_C].push(values[2]);
        }
        while fixed[ROW].len() < entry_count {
            indices[ROW].push(0);
            indices[COL].push(0);
            fixed[ROW].push(Fr::zero());
            fixed[COL].push(Fr::zero());
            fixed[VAL_A].push(Fr::zero());
            fixed[VAL_B].push(Fr::zero());
            fixed[VAL_C].push(Fr::zero());
        }
        fixed[ROW_MULTIPLICITY].resize(entry_count, Fr::zero());
        fixed[COL_MULTIPLICITY].resize(entry_count, Fr::zero());
        for (&row, &col) in indices[ROW].iter().zip(&indices[COL]) {
            fixed[ROW_MULTIPLICITY][row] += Fr::one();
            fixed[COL_MULTIPLICITY][col] += Fr::one();
        }
        Ok(Self {
            fixed,
            indices,
            row_vars: row_count.trailing_zeros() as usize,
            col_vars: col_count.trailing_zeros() as usize,
            entry_vars: entry_count.trailing_zeros() as usize,
        })
    }

    pub fn entry_vars(&self) -> usize {
        self.entry_vars
    }

    pub fn entry_count(&self) -> usize {
        1 << self.entry_vars
    }

    pub fn fixed_evaluations(&self) -> &[Vec<Fr>; FIXED_COLUMNS] {
        &self.fixed
    }

    pub fn fixed_columns(&self, common_rows: usize) -> Result<Vec<Column>, SparkError> {
        let mut columns = self
            .fixed
            .iter()
            .map(|values| embed(values, common_rows).map(Column::Fr))
            .collect::<Result<Vec<_>, _>>()?;
        columns.extend((0..WITNESS_COLUMNS).map(|_| Column::Fr(vec![Fr::zero(); common_rows])));
        Ok(columns)
    }

    pub fn columns(
        &self,
        witness: &SparkWitness,
        common_rows: usize,
    ) -> Result<Vec<Column>, SparkError> {
        let mut columns = self
            .fixed
            .iter()
            .map(|values| embed(values, common_rows).map(Column::Fr))
            .collect::<Result<Vec<_>, _>>()?;
        columns.extend(
            witness
                .columns
                .iter()
                .map(|values| embed(values, common_rows).map(Column::Fr))
                .collect::<Result<Vec<_>, _>>()?,
        );
        Ok(columns)
    }

    fn witness_columns(
        &self,
        witness: &SparkWitness,
        common_rows: usize,
    ) -> Result<Vec<Column>, SparkError> {
        let mut columns: Vec<Column> = (0..FIXED_COLUMNS)
            .map(|_| Column::Fr(vec![Fr::zero(); common_rows]))
            .collect();
        columns.extend(
            witness
                .columns
                .iter()
                .map(|values| embed(values, common_rows).map(Column::Fr))
                .collect::<Result<Vec<_>, _>>()?,
        );
        Ok(columns)
    }

    fn virtual_evaluations(&self, point: &[Fr], rx: &[Fr], ry: &[Fr]) -> [Fr; 4] {
        let row_point = &point[self.entry_vars - self.row_vars..];
        let col_point = &point[self.entry_vars - self.col_vars..];
        [
            identity_mle(row_point),
            identity_mle(col_point),
            EqPolynomial::<Fr>::mle(rx, row_point),
            EqPolynomial::<Fr>::mle(ry, col_point),
        ]
    }
}

#[derive(Clone, Debug)]
pub struct SparkVerifierKey {
    pub fixed_commitments: Vec<Commitment>,
    pub layout: PackingLayout,
    pub row_vars: usize,
    pub col_vars: usize,
    pub entry_vars: usize,
}

impl SparkVerifierKey {
    pub fn combine_commitments(
        &self,
        witness_commitments: &[Commitment],
    ) -> Result<Vec<Commitment>, SparkError> {
        if witness_commitments.len() != self.fixed_commitments.len() {
            return Err(StreamError::StageCount.into());
        }
        Ok(self
            .fixed_commitments
            .iter()
            .zip(witness_commitments)
            .map(|(fixed, witness)| <Commitment as HomomorphicCommitment<Fr>>::add(fixed, witness))
            .collect())
    }
}

#[derive(Clone, Debug)]
pub struct SparkProverKey {
    pub tables: SparkTables,
    fixed: PackedColumns,
}

impl SparkProverKey {
    pub fn new(
        tables: SparkTables,
        common_rows: usize,
        packing: usize,
        setup: &HyperKZGProverSetup<Bn254>,
    ) -> Result<(Self, SparkVerifierKey), SparkError> {
        let fixed = commit_packed(&tables.fixed_columns(common_rows)?, packing, setup)?;
        let verifier = SparkVerifierKey {
            fixed_commitments: fixed.commitments.clone(),
            layout: fixed.layout,
            row_vars: tables.row_vars,
            col_vars: tables.col_vars,
            entry_vars: tables.entry_vars,
        };
        Ok((Self { tables, fixed }, verifier))
    }

    pub fn commit_witness(
        &self,
        witness: &SparkWitness,
        setup: &HyperKZGProverSetup<Bn254>,
    ) -> Result<(PackedColumns, Vec<Commitment>), SparkError> {
        let witness_packed = commit_packed(
            &self
                .tables
                .witness_columns(witness, self.fixed.layout.rows)?,
            self.fixed.layout.k,
            setup,
        )?;
        let commitments = self
            .fixed
            .commitments
            .iter()
            .zip(&witness_packed.commitments)
            .map(|(fixed, witness)| <Commitment as HomomorphicCommitment<Fr>>::add(fixed, witness))
            .collect();
        let evaluations = self
            .fixed
            .evaluations
            .iter()
            .zip(&witness_packed.evaluations)
            .map(|(fixed, witness)| {
                fixed
                    .iter()
                    .zip(witness)
                    .map(|(&fixed, &witness)| fixed + witness)
                    .collect()
            })
            .collect();
        Ok((
            PackedColumns {
                layout: self.fixed.layout,
                evaluations,
                commitments,
            },
            witness_packed.commitments,
        ))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparkChallenges {
    pub alpha_row: Fr,
    pub beta_row: Fr,
    pub alpha_col: Fr,
    pub beta_col: Fr,
    pub matrix_weights: [Fr; 3],
    pub relation_weights: [Fr; 6],
    pub tau: Vec<Fr>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparkWitness {
    columns: [Vec<Fr>; WITNESS_COLUMNS],
    matrix_evaluations: [Fr; 3],
}

impl SparkWitness {
    pub fn new(
        tables: &SparkTables,
        rx: &[Fr],
        ry: &[Fr],
        challenges: &SparkChallenges,
    ) -> Result<Self, SparkError> {
        validate_points(tables, rx, ry, challenges)?;
        let row_eq = EqPolynomial::<Fr>::evals(rx, None);
        let col_eq = EqPolynomial::<Fr>::evals(ry, None);
        let entry_count = tables.entry_count();
        let mut columns: [Vec<Fr>; WITNESS_COLUMNS] =
            std::array::from_fn(|_| Vec::with_capacity(entry_count));
        let mut matrix_evaluations = [Fr::zero(); 3];
        for index in 0..entry_count {
            let row = tables.indices[ROW][index];
            let col = tables.indices[COL][index];
            let e_row = row_eq[row];
            let e_col = col_eq[col];
            let row_address = index & ((1 << tables.row_vars) - 1);
            let col_address = index & ((1 << tables.col_vars) - 1);
            let h_left_row = inverse(
                challenges.alpha_row - tables.fixed[ROW][index] - challenges.beta_row * e_row,
            )?;
            let h_left_col = inverse(
                challenges.alpha_col - tables.fixed[COL][index] - challenges.beta_col * e_col,
            )?;
            let h_right_row = inverse(
                challenges.alpha_row
                    - Fr::from_u64(row_address as u64)
                    - challenges.beta_row * row_eq[row_address],
            )?;
            let h_right_col = inverse(
                challenges.alpha_col
                    - Fr::from_u64(col_address as u64)
                    - challenges.beta_col * col_eq[col_address],
            )?;
            for (acc, values) in matrix_evaluations
                .iter_mut()
                .zip(&tables.fixed[VAL_A..=VAL_C])
            {
                *acc += values[index] * e_row * e_col;
            }
            for (column, value) in [
                e_row,
                e_col,
                h_left_row,
                h_left_col,
                h_right_row,
                h_right_col,
            ]
            .into_iter()
            .enumerate()
            {
                columns[column].push(value);
            }
        }
        Ok(Self {
            columns,
            matrix_evaluations,
        })
    }

    pub fn matrix_evaluations(&self) -> [Fr; 3] {
        self.matrix_evaluations
    }

    pub fn evaluations(&self) -> &[Vec<Fr>; WITNESS_COLUMNS] {
        &self.columns
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparkEvaluations {
    pub fixed: [Fr; FIXED_COLUMNS],
    pub witness: [Fr; WITNESS_COLUMNS],
}

pub struct SparkProver {
    fixed: [Polynomial<Fr>; FIXED_COLUMNS],
    witness: [Polynomial<Fr>; WITNESS_COLUMNS],
    virtuals: [Polynomial<Fr>; 4],
    eq_tau: Polynomial<Fr>,
    challenges: SparkChallenges,
    input_claim: Fr,
    rounds: usize,
}

impl SparkProver {
    pub fn new(
        tables: &SparkTables,
        witness: &SparkWitness,
        rx: &[Fr],
        ry: &[Fr],
        challenges: SparkChallenges,
    ) -> Result<Self, SparkError> {
        validate_points(tables, rx, ry, &challenges)?;
        let row_eq = EqPolynomial::<Fr>::evals(rx, None);
        let col_eq = EqPolynomial::<Fr>::evals(ry, None);
        let entry_count = tables.entry_count();
        let row_mask = (1 << tables.row_vars) - 1;
        let col_mask = (1 << tables.col_vars) - 1;
        let virtuals: [Vec<Fr>; 4] = std::array::from_fn(|virtual_index| {
            (0..entry_count)
                .map(|index| {
                    let row_address = index & row_mask;
                    let col_address = index & col_mask;
                    match virtual_index {
                        0 => Fr::from_u64(row_address as u64),
                        1 => Fr::from_u64(col_address as u64),
                        2 => row_eq[row_address],
                        3 => col_eq[col_address],
                        _ => unreachable!("four SPARK virtual columns"),
                    }
                })
                .collect()
        });
        let eq_tau = EqPolynomial::<Fr>::evals(&challenges.tau, None);
        let input_claim = challenges
            .matrix_weights
            .iter()
            .zip(witness.matrix_evaluations)
            .map(|(&weight, evaluation)| weight * evaluation)
            .sum();
        let actual: Fr = (0..entry_count)
            .into_par_iter()
            .map(|index| {
                relation_value(
                    &challenges,
                    std::array::from_fn(|column| tables.fixed[column][index]),
                    std::array::from_fn(|column| witness.columns[column][index]),
                    std::array::from_fn(|column| virtuals[column][index]),
                    eq_tau[index],
                )
            })
            .sum();
        if actual != input_claim {
            return Err(SparkError::ClaimMismatch);
        }
        Ok(Self {
            fixed: std::array::from_fn(|column| Polynomial::new(tables.fixed[column].clone())),
            witness: std::array::from_fn(|column| Polynomial::new(witness.columns[column].clone())),
            virtuals: virtuals.map(Polynomial::new),
            eq_tau: Polynomial::new(eq_tau),
            challenges,
            input_claim,
            rounds: tables.entry_vars,
        })
    }

    pub fn input_claim(&self) -> Fr {
        self.input_claim
    }

    pub fn final_evaluations(&self) -> SparkEvaluations {
        SparkEvaluations {
            fixed: std::array::from_fn(|column| self.fixed[column].evals()[0]),
            witness: std::array::from_fn(|column| self.witness[column].evals()[0]),
        }
    }

    fn bind(&mut self, challenge: Fr) {
        for polynomial in self
            .fixed
            .iter_mut()
            .chain(&mut self.witness)
            .chain(&mut self.virtuals)
            .chain(std::iter::once(&mut self.eq_tau))
        {
            polynomial.bind_with_order(challenge, BindingOrder::HighToLow);
        }
    }
}

impl ProveRounds<Fr> for SparkProver {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        let half = self.eq_tau.len() / 2;
        let evaluations = (0..half)
            .into_par_iter()
            .map(|index| {
                let mut local = [Fr::zero(); DEGREE + 1];
                for (x, evaluation) in local.iter_mut().enumerate() {
                    let x = Fr::from_u64(x as u64);
                    *evaluation = relation_value(
                        &self.challenges,
                        round_evaluations(&self.fixed, index, x),
                        round_evaluations(&self.witness, index, x),
                        round_evaluations(&self.virtuals, index, x),
                        self.eq_tau.sumcheck_round_eval(index, x),
                    );
                }
                local
            })
            .reduce(
                || [Fr::zero(); DEGREE + 1],
                |mut sum, local| {
                    for (sum, value) in sum.iter_mut().zip(local) {
                        *sum += value;
                    }
                    sum
                },
            );
        if evaluations[0] + evaluations[1] != previous_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual: evaluations[0] + evaluations[1],
            });
        }
        Ok(UnivariatePoly::from_evals(&evaluations))
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        self.bind(bind);
        Ok(())
    }
}

pub fn final_claim(
    tables: &SparkTables,
    rx: &[Fr],
    ry: &[Fr],
    challenges: &SparkChallenges,
    point: &[Fr],
    evaluations: &SparkEvaluations,
) -> Result<Fr, SparkError> {
    validate_points(tables, rx, ry, challenges)?;
    if point.len() != tables.entry_vars {
        return Err(SparkError::ChallengePoint {
            expected: tables.entry_vars,
            actual: point.len(),
        });
    }
    Ok(relation_value(
        challenges,
        evaluations.fixed,
        evaluations.witness,
        tables.virtual_evaluations(point, rx, ry),
        EqPolynomial::<Fr>::mle(&challenges.tau, point),
    ))
}

fn relation_value(
    challenges: &SparkChallenges,
    fixed: [Fr; FIXED_COLUMNS],
    witness: [Fr; WITNESS_COLUMNS],
    virtuals: [Fr; 4],
    eq_tau: Fr,
) -> Fr {
    let matrix = challenges.matrix_weights[0] * fixed[VAL_A]
        + challenges.matrix_weights[1] * fixed[VAL_B]
        + challenges.matrix_weights[2] * fixed[VAL_C];
    let inverse_relations = [
        witness[H_LEFT_ROW]
            * (challenges.alpha_row - fixed[ROW] - challenges.beta_row * witness[E_ROW])
            - Fr::one(),
        witness[H_LEFT_COL]
            * (challenges.alpha_col - fixed[COL] - challenges.beta_col * witness[E_COL])
            - Fr::one(),
        witness[H_RIGHT_ROW]
            * (challenges.alpha_row - virtuals[0] - challenges.beta_row * virtuals[2])
            - Fr::one(),
        witness[H_RIGHT_COL]
            * (challenges.alpha_col - virtuals[1] - challenges.beta_col * virtuals[3])
            - Fr::one(),
    ];
    let logup = [
        witness[H_LEFT_ROW] - fixed[ROW_MULTIPLICITY] * witness[H_RIGHT_ROW],
        witness[H_LEFT_COL] - fixed[COL_MULTIPLICITY] * witness[H_RIGHT_COL],
    ];
    let mut value = matrix * witness[E_ROW] * witness[E_COL];
    for (&weight, relation) in challenges.relation_weights.iter().zip(
        inverse_relations
            .into_iter()
            .map(|relation| eq_tau * relation)
            .chain(logup),
    ) {
        value += weight * relation;
    }
    value
}

fn round_evaluations<const N: usize>(
    polynomials: &[Polynomial<Fr>; N],
    index: usize,
    point: Fr,
) -> [Fr; N] {
    std::array::from_fn(|column| polynomials[column].sumcheck_round_eval(index, point))
}

fn add_matrix(
    entries: &mut BTreeMap<(usize, usize), [Fr; 3]>,
    rows: &[SparseRow<Fr>],
    column_start: usize,
    column_end: usize,
    matrix: usize,
) {
    for (row, entries_in_row) in rows.iter().enumerate() {
        for &(column, coefficient) in entries_in_row {
            if column_start <= column && column < column_end {
                entries.entry((row, column)).or_default()[matrix] += coefficient;
            }
        }
    }
}

fn embed(values: &[Fr], common_rows: usize) -> Result<Vec<Fr>, SparkError> {
    if !common_rows.is_power_of_two() || common_rows < values.len() {
        return Err(SparkError::CommonRows {
            minimum: values.len(),
            actual: common_rows,
        });
    }
    let repetitions = common_rows / values.len();
    let mut embedded = Vec::with_capacity(common_rows);
    for &value in values {
        embedded.extend(std::iter::repeat_n(value, repetitions));
    }
    Ok(embedded)
}

fn identity_mle(point: &[Fr]) -> Fr {
    point
        .iter()
        .enumerate()
        .map(|(index, &coordinate)| coordinate * Fr::from_u64(1 << (point.len() - index - 1)))
        .sum()
}

fn validate_points(
    tables: &SparkTables,
    rx: &[Fr],
    ry: &[Fr],
    challenges: &SparkChallenges,
) -> Result<(), SparkError> {
    if rx.len() != tables.row_vars {
        return Err(SparkError::RowPoint {
            expected: tables.row_vars,
            actual: rx.len(),
        });
    }
    if ry.len() != tables.col_vars {
        return Err(SparkError::ColumnPoint {
            expected: tables.col_vars,
            actual: ry.len(),
        });
    }
    if challenges.tau.len() != tables.entry_vars {
        return Err(SparkError::ChallengePoint {
            expected: tables.entry_vars,
            actual: challenges.tau.len(),
        });
    }
    Ok(())
}

fn inverse(value: Fr) -> Result<Fr, SparkError> {
    value.inverse().ok_or(SparkError::ZeroDenominator)
}
