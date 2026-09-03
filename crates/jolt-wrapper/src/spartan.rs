//! Spartan for plain R1CS with columns `[0, 1 + public_inputs.len())` public.

use jolt_field::{Fr, One, Ring, Zero};
use jolt_hyperkzg::{NoopVerifierObserver, VerifierObserver};
use jolt_poly::{BindingOrder, EqPolynomial, Polynomial, UnivariatePoly};
use jolt_r1cs::{
    ConstraintMatrices, ConstraintMatrixEvalError, MatrixColumnContributions, SparseRow,
};
use jolt_sumcheck::prover::ProveRounds;
use jolt_sumcheck::SumcheckError;
use jolt_transcript::Transcript;
use rayon::prelude::*;
use thiserror::Error;

use crate::stream::{
    AffineForm, Column, ColumnId, StageMemberSpec, StreamError, Term, TermContext, TermExporter,
    TermObserver,
};

pub(crate) const OUTER_DEGREE: usize = 3;
pub(crate) const INNER_DEGREE: usize = 2;

#[derive(Debug, Error)]
pub enum SpartanError {
    #[error("R1CS constraint count must be nonzero and paddable to a power of two")]
    InvalidConstraintCount,
    #[error("public input range has {public} columns, but the R1CS has {total}")]
    PublicInputRange { public: usize, total: usize },
    #[error("witness has {actual} entries, expected {expected}")]
    WitnessLength { expected: usize, actual: usize },
    #[error("committed witness length must be a nonzero power of two")]
    InvalidWitnessLength,
    #[error("common row domain has {rows} rows, but the padded witness needs {minimum}")]
    InvalidSharedRowDomain { rows: usize, minimum: usize },
    #[error("common row point has {actual} coordinates, expected {expected}")]
    SharedPointDimension { expected: usize, actual: usize },
    #[error("R1CS witness fails at row {0}")]
    Unsatisfied(usize),
    #[error("outer sumcheck final claim mismatch")]
    OuterFinalClaim,
    #[error("inner sumcheck input claim mismatch")]
    InnerInputClaim,
    #[error("inner sumcheck final claim mismatch")]
    InnerFinalClaim,
    #[error("stream: {0}")]
    Stream(#[from] StreamError),
    #[error("matrix evaluation: {0}")]
    Matrix(#[from] ConstraintMatrixEvalError),
}

/// Spartan's witness polynomial embedded in the wrapper's common row domain.
///
/// Witness values occupy the first rows; the common-domain suffix is zero.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SharedWitnessColumn {
    column: Column,
    inner_rounds: usize,
    common_row_vars: usize,
}

impl SharedWitnessColumn {
    pub fn new(witness: &[Fr], common_rows: usize) -> Result<Self, SpartanError> {
        let minimum = witness
            .len()
            .checked_next_power_of_two()
            .filter(|minimum| *minimum != 0)
            .ok_or(SpartanError::InvalidWitnessLength)?;
        if !common_rows.is_power_of_two() || common_rows < minimum {
            return Err(SpartanError::InvalidSharedRowDomain {
                rows: common_rows,
                minimum,
            });
        }
        let mut evaluations = vec![Fr::zero(); common_rows];
        evaluations[..witness.len()].copy_from_slice(witness);
        Ok(Self {
            column: Column::Fr(evaluations),
            inner_rounds: minimum.trailing_zeros() as usize,
            common_row_vars: common_rows.trailing_zeros() as usize,
        })
    }

    pub fn inner_member(&self) -> StageMemberSpec {
        StageMemberSpec {
            rounds: self.inner_rounds,
            degree: INNER_DEGREE,
            offset: 0,
        }
    }

    pub fn source_point(&self, inner_point: &[Fr]) -> Result<Vec<Fr>, SpartanError> {
        if inner_point.len() != self.inner_rounds {
            return Err(SpartanError::SharedPointDimension {
                expected: self.inner_rounds,
                actual: inner_point.len(),
            });
        }
        let mut point = vec![Fr::zero(); self.common_row_vars - self.inner_rounds];
        point.extend_from_slice(inner_point);
        Ok(point)
    }

    pub fn evaluations(&self) -> &[Fr] {
        match &self.column {
            Column::Fr(evaluations) => evaluations,
            Column::Bits(_) | Column::U16(_) | Column::U32(_) => {
                unreachable!("Spartan witness is a field column")
            }
        }
    }

    pub fn inner_evaluations(&self) -> &[Fr] {
        &self.evaluations()[..1 << self.inner_rounds]
    }

    pub fn into_column(self) -> Column {
        self.column
    }
}

pub(crate) struct CarryTermExporter<'a> {
    pub source_point: &'a [Fr],
    pub wire: ColumnId,
    pub member_index: usize,
}

impl CarryTermExporter<'_> {
    fn terms_observed<O: TermObserver + ?Sized>(
        &self,
        context: &TermContext<'_>,
        observer: &mut O,
    ) -> Vec<Term> {
        let eq = self.source_point.iter().zip(context.row_point).fold(
            Fr::one(),
            |value, (&left, &right)| {
                let both = observer.fr_mul(left, right);
                let neither = observer.fr_mul(Fr::one() - left, Fr::one() - right);
                observer.fr_mul(value, both + neither)
            },
        );
        vec![Term {
            coefficient: observer.fr_mul(context.batching_coefficients[self.member_index], eq),
            factors: vec![AffineForm {
                constant: Fr::zero(),
                weights: vec![(self.wire, Fr::one())],
            }],
        }]
    }
}

impl TermExporter for CarryTermExporter<'_> {
    fn max_factors(&self) -> usize {
        1
    }

    fn terms(&self, context: &TermContext<'_>) -> Vec<Term> {
        self.terms_observed(context, &mut NoopVerifierObserver)
    }

    fn terms_observed(
        &self,
        context: &TermContext<'_>,
        observer: &mut dyn TermObserver,
    ) -> Vec<Term> {
        CarryTermExporter::terms_observed(self, context, observer)
    }
}

pub(crate) fn validate_dimensions(
    r1cs: &ConstraintMatrices<Fr>,
    public_input_count: usize,
    witness_len: usize,
) -> Result<(), SpartanError> {
    if r1cs.num_constraints == 0 || r1cs.num_constraints.checked_next_power_of_two().is_none() {
        return Err(SpartanError::InvalidConstraintCount);
    }
    let public_columns = 1 + public_input_count;
    if public_columns > r1cs.num_vars {
        return Err(SpartanError::PublicInputRange {
            public: public_columns,
            total: r1cs.num_vars,
        });
    }
    let expected = r1cs.num_vars - public_columns;
    if witness_len != expected {
        return Err(SpartanError::WitnessLength {
            expected,
            actual: witness_len,
        });
    }
    if witness_len == 0 || witness_len.checked_next_power_of_two().is_none() {
        return Err(SpartanError::InvalidWitnessLength);
    }
    Ok(())
}

pub(crate) fn assignment(public_inputs: &[Fr], witness: &[Fr]) -> Vec<Fr> {
    let mut z = Vec::with_capacity(1 + public_inputs.len() + witness.len());
    z.push(Fr::one());
    z.extend_from_slice(public_inputs);
    z.extend_from_slice(witness);
    z
}

pub(crate) fn draw_point<T: Transcript<Challenge = Fr>>(
    rounds: usize,
    transcript: &mut T,
) -> Vec<Fr> {
    (0..rounds).map(|_| transcript.challenge()).collect()
}

pub(crate) fn draw_matrix_weights<T: Transcript<Challenge = Fr>>(transcript: &mut T) -> [Fr; 3] {
    [
        transcript.challenge(),
        transcript.challenge(),
        transcript.challenge(),
    ]
}

pub(crate) fn public_contributions(
    r1cs: &ConstraintMatrices<Fr>,
    row_weights: &[Fr],
    public_inputs: &[Fr],
) -> Result<MatrixColumnContributions<Fr>, ConstraintMatrixEvalError> {
    let mut values = Vec::with_capacity(1 + public_inputs.len());
    values.push(Fr::one());
    values.extend_from_slice(public_inputs);
    r1cs.column_range_contributions(row_weights, 0, &values)
}

pub(crate) fn public_contributions_observed<O: VerifierObserver>(
    r1cs: &ConstraintMatrices<Fr>,
    row_weights: &[Fr],
    public_inputs: &[Fr],
    observer: &mut O,
) -> Result<MatrixColumnContributions<Fr>, ConstraintMatrixEvalError> {
    let mut values = Vec::with_capacity(1 + public_inputs.len());
    values.push(Fr::one());
    values.extend_from_slice(public_inputs);
    let evaluate = |matrix: &[SparseRow<Fr>], observer: &mut O| {
        let mut sum = Fr::zero();
        for (row, &row_weight) in matrix.iter().zip(row_weights) {
            for &(column, coefficient) in row {
                if let Some(&value) = values.get(column) {
                    let term = observer.fr_mul(row_weight, coefficient);
                    sum += observer.fr_mul(term, value);
                }
            }
        }
        sum
    };
    Ok(MatrixColumnContributions {
        a: evaluate(&r1cs.a, observer),
        b: evaluate(&r1cs.b, observer),
        c: evaluate(&r1cs.c, observer),
    })
}

pub(crate) fn witness_linear_eval_observed<O: VerifierObserver>(
    r1cs: &ConstraintMatrices<Fr>,
    row_weights: &[Fr],
    column_weights: &[Fr],
    public_columns: usize,
    matrix_weights: [Fr; 3],
    observer: &mut O,
) -> Result<Fr, ConstraintMatrixEvalError> {
    if row_weights.len() < r1cs.num_constraints {
        return Err(ConstraintMatrixEvalError::RowWeightsLengthMismatch {
            expected: r1cs.num_constraints,
            actual: row_weights.len(),
        });
    }
    let witness_len = r1cs.num_vars.checked_sub(public_columns).ok_or(
        ConstraintMatrixEvalError::ColumnRangeOverflow {
            start: public_columns,
            count: 0,
        },
    )?;
    if column_weights.len() < witness_len {
        return Err(ConstraintMatrixEvalError::ColumnWeightsLengthMismatch {
            expected: witness_len,
            actual: column_weights.len(),
        });
    }
    let mut sum = Fr::zero();
    for (matrix, matrix_weight) in [
        (&r1cs.a, matrix_weights[0]),
        (&r1cs.b, matrix_weights[1]),
        (&r1cs.c, matrix_weights[2]),
    ] {
        for (row, &row_weight) in matrix.iter().zip(row_weights) {
            for &(column, coefficient) in row {
                if let Some(offset) = column.checked_sub(public_columns) {
                    let column_weight = *column_weights.get(offset).ok_or(
                        ConstraintMatrixEvalError::ColumnOutOfBounds {
                            column,
                            num_vars: r1cs.num_vars,
                        },
                    )?;
                    let term = observer.fr_mul(row_weight, column_weight);
                    let term = observer.fr_mul(term, coefficient);
                    sum += observer.fr_mul(term, matrix_weight);
                }
            }
        }
    }
    Ok(sum)
}

pub fn matrix_nnz(r1cs: &ConstraintMatrices<Fr>) -> usize {
    r1cs.a
        .iter()
        .chain(&r1cs.b)
        .chain(&r1cs.c)
        .map(Vec::len)
        .sum()
}

pub(crate) fn project_witness_columns(
    r1cs: &ConstraintMatrices<Fr>,
    row_weights: &[Fr],
    start_column: usize,
    witness_len: usize,
    weights: [Fr; 3],
) -> Vec<Fr> {
    let mut projected = vec![Fr::zero(); witness_len];
    project_matrix(
        &mut projected,
        &r1cs.a,
        row_weights,
        start_column,
        weights[0],
    );
    project_matrix(
        &mut projected,
        &r1cs.b,
        row_weights,
        start_column,
        weights[1],
    );
    project_matrix(
        &mut projected,
        &r1cs.c,
        row_weights,
        start_column,
        weights[2],
    );
    projected
}

fn project_matrix(
    projected: &mut [Fr],
    rows: &[SparseRow<Fr>],
    row_weights: &[Fr],
    start_column: usize,
    matrix_weight: Fr,
) {
    if matrix_weight.is_zero() {
        return;
    }
    for (row, &row_weight) in rows.iter().zip(row_weights) {
        let scale = matrix_weight * row_weight;
        for &(column, coefficient) in row {
            if let Some(entry) = column
                .checked_sub(start_column)
                .and_then(|offset| projected.get_mut(offset))
            {
                *entry += scale * coefficient;
            }
        }
    }
}

fn matrix_vector_product(rows: &[SparseRow<Fr>], z: &[Fr], padded_rows: usize) -> Vec<Fr> {
    let mut values = rows
        .par_iter()
        .map(|row| {
            row.iter()
                .map(|&(column, coefficient)| z[column] * coefficient)
                .sum()
        })
        .collect::<Vec<_>>();
    values.resize(padded_rows, Fr::zero());
    values
}

pub(crate) struct OuterSumcheck {
    eq_tau: Polynomial<Fr>,
    az: Polynomial<Fr>,
    bz: Polynomial<Fr>,
    cz: Polynomial<Fr>,
    rounds: usize,
}

impl OuterSumcheck {
    pub(crate) fn new(r1cs: &ConstraintMatrices<Fr>, z: &[Fr], tau: &[Fr]) -> Self {
        let padded_rows = 1 << tau.len();
        Self {
            eq_tau: Polynomial::new(EqPolynomial::<Fr>::evals(tau, None)),
            az: Polynomial::new(matrix_vector_product(&r1cs.a, z, padded_rows)),
            bz: Polynomial::new(matrix_vector_product(&r1cs.b, z, padded_rows)),
            cz: Polynomial::new(matrix_vector_product(&r1cs.c, z, padded_rows)),
            rounds: tau.len(),
        }
    }

    pub(crate) fn finals(&self) -> [Fr; 3] {
        [self.az.evals()[0], self.bz.evals()[0], self.cz.evals()[0]]
    }

    fn bind(&mut self, challenge: Fr) {
        for polynomial in [&mut self.eq_tau, &mut self.az, &mut self.bz, &mut self.cz] {
            polynomial.bind_with_order(challenge, BindingOrder::HighToLow);
        }
    }
}

impl ProveRounds<Fr> for OuterSumcheck {
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
        let half = self.az.len() / 2;
        let evaluations = (0..half)
            .into_par_iter()
            .map(|index| {
                let mut local = [Fr::zero(); OUTER_DEGREE + 1];
                for (x, evaluation) in local.iter_mut().enumerate() {
                    let x = Fr::from_u64(x as u64);
                    *evaluation = {
                        let eq = self.eq_tau.sumcheck_round_eval(index, x);
                        let az = self.az.sumcheck_round_eval(index, x);
                        let bz = self.bz.sumcheck_round_eval(index, x);
                        let cz = self.cz.sumcheck_round_eval(index, x);
                        eq * (az * bz - cz)
                    };
                }
                local
            })
            .reduce(
                || [Fr::zero(); OUTER_DEGREE + 1],
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

pub(crate) struct InnerSumcheck {
    linear_form: Polynomial<Fr>,
    witness: Polynomial<Fr>,
    rounds: usize,
}

impl InnerSumcheck {
    pub(crate) fn new(
        linear_form: Vec<Fr>,
        witness: Vec<Fr>,
        claim: Fr,
    ) -> Result<Self, SpartanError> {
        let actual: Fr = linear_form
            .iter()
            .zip(&witness)
            .map(|(&left, &right)| left * right)
            .sum();
        if actual != claim {
            return Err(SpartanError::InnerInputClaim);
        }
        let rounds = witness.len().trailing_zeros() as usize;
        Ok(Self {
            linear_form: Polynomial::new(linear_form),
            witness: Polynomial::new(witness),
            rounds,
        })
    }

    pub(crate) fn finals(&self) -> [Fr; 2] {
        [self.linear_form.evals()[0], self.witness.evals()[0]]
    }

    fn bind(&mut self, challenge: Fr) {
        self.linear_form
            .bind_with_order(challenge, BindingOrder::HighToLow);
        self.witness
            .bind_with_order(challenge, BindingOrder::HighToLow);
    }
}

impl ProveRounds<Fr> for InnerSumcheck {
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
        let half = self.witness.len() / 2;
        let evaluations = (0..half)
            .into_par_iter()
            .map(|index| {
                let mut local = [Fr::zero(); INNER_DEGREE + 1];
                for (x, evaluation) in local.iter_mut().enumerate() {
                    let x = Fr::from_u64(x as u64);
                    *evaluation = self.linear_form.sumcheck_round_eval(index, x)
                        * self.witness.sumcheck_round_eval(index, x);
                }
                local
            })
            .reduce(
                || [Fr::zero(); INNER_DEGREE + 1],
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
