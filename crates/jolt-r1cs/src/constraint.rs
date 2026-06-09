//! Sparse per-cycle R1CS constraint matrices.

use jolt_field::Field;
use jolt_poly::EqPolynomial;
use serde::{Deserialize, Serialize};
use thiserror::Error as ThisError;

/// Sparse row: `[(variable_index, coefficient)]`.
pub type SparseRow<F> = Vec<(usize, F)>;

#[derive(Clone, Debug, ThisError, PartialEq, Eq)]
pub enum ConstraintMatrixEvalError {
    #[error("row point length mismatch: expected {expected}, got {actual}")]
    RowPointLengthMismatch { expected: usize, actual: usize },
    #[error("column point length mismatch: expected {expected}, got {actual}")]
    ColumnPointLengthMismatch { expected: usize, actual: usize },
    #[error("row weights length mismatch: expected at least {expected}, got {actual}")]
    RowWeightsLengthMismatch { expected: usize, actual: usize },
    #[error("column weights length mismatch: expected {expected}, got {actual}")]
    ColumnWeightsLengthMismatch { expected: usize, actual: usize },
    #[error("column {column} out of bounds for {num_vars} variables")]
    ColumnOutOfBounds { column: usize, num_vars: usize },
    #[error("matrix {dimension} dimension {value} cannot be padded to a power of two")]
    PaddedDimensionOverflow {
        dimension: &'static str,
        value: usize,
    },
    #[error("matrix column range overflow: start {start}, count {count}")]
    ColumnRangeOverflow { start: usize, count: usize },
}

/// Per-cycle sparse R1CS constraint matrices.
///
/// Represents the local A, B, C matrices for a single cycle in a uniform
/// R1CS. For a valid witness z, the constraint system requires:
///
/// $$A z \circ B z = C z$$
///
/// Each matrix is stored as a list of sparse rows, one per constraint.
/// Entry `a[k] = [(v, α), ...]` means constraint k's A side has
/// coefficient α at variable index v.
///
/// Deserialization routes through [`RawConstraintMatrices`] and revalidates
/// the same invariants as [`ConstraintMatrices::new`]; malformed input is
/// rejected before any consumer sees the struct.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "", try_from = "RawConstraintMatrices<F>")]
pub struct ConstraintMatrices<F: Field> {
    pub num_constraints: usize,
    pub num_vars: usize,
    pub a: Vec<SparseRow<F>>,
    pub b: Vec<SparseRow<F>>,
    pub c: Vec<SparseRow<F>>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WeightedMatrixColumns<F: Field> {
    pub a: Vec<F>,
    pub b: Vec<F>,
    pub c: Vec<F>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MatrixColumnContributions<F: Field> {
    pub a: F,
    pub b: F,
    pub c: F,
}

/// Deserialization helper; never exposed directly.
#[derive(Deserialize)]
#[serde(bound = "")]
struct RawConstraintMatrices<F: Field> {
    num_constraints: usize,
    num_vars: usize,
    a: Vec<SparseRow<F>>,
    b: Vec<SparseRow<F>>,
    c: Vec<SparseRow<F>>,
}

impl<F: Field> TryFrom<RawConstraintMatrices<F>> for ConstraintMatrices<F> {
    type Error = String;

    fn try_from(raw: RawConstraintMatrices<F>) -> Result<Self, Self::Error> {
        let RawConstraintMatrices {
            num_constraints,
            num_vars,
            a,
            b,
            c,
        } = raw;
        check_invariants(num_constraints, num_vars, &a, &b, &c)?;
        Ok(Self {
            num_constraints,
            num_vars,
            a,
            b,
            c,
        })
    }
}

fn check_invariants<F: Field>(
    num_constraints: usize,
    num_vars: usize,
    a: &[SparseRow<F>],
    b: &[SparseRow<F>],
    c: &[SparseRow<F>],
) -> Result<(), String> {
    for (label, rows) in [("a", a), ("b", b), ("c", c)] {
        if rows.len() != num_constraints {
            return Err(format!(
                "ConstraintMatrices.{label}.len() = {}, expected num_constraints = {num_constraints}",
                rows.len(),
            ));
        }
        for (k, row) in rows.iter().enumerate() {
            for &(col, _) in row {
                if col >= num_vars {
                    return Err(format!(
                        "ConstraintMatrices.{label}[{k}] references variable {col} >= num_vars = {num_vars}"
                    ));
                }
            }
        }
    }
    Ok(())
}

impl<F: Field> ConstraintMatrices<F> {
    /// Builds constraint matrices from sparse rows.
    ///
    /// # Panics
    ///
    /// Panics if any matrix has a different number of rows than
    /// `num_constraints`, or if any row references a variable index
    /// `>= num_vars`.
    #[expect(
        clippy::expect_used,
        reason = "constructor invariant violation indicates a programmer error"
    )]
    pub fn new(
        num_constraints: usize,
        num_vars: usize,
        a: Vec<SparseRow<F>>,
        b: Vec<SparseRow<F>>,
        c: Vec<SparseRow<F>>,
    ) -> Self {
        check_invariants(num_constraints, num_vars, &a, &b, &c)
            .expect("ConstraintMatrices::new invariant violated");
        Self {
            num_constraints,
            num_vars,
            a,
            b,
            c,
        }
    }

    /// Checks whether a per-cycle witness satisfies all constraints.
    ///
    /// Returns `Ok(())` if Az ∘ Bz = Cz for every row, or the index
    /// of the first violated constraint.
    pub fn check_witness(&self, witness: &[F]) -> Result<(), usize> {
        for k in 0..self.num_constraints {
            let az = dot(&self.a[k], witness);
            let bz = dot(&self.b[k], witness);
            let cz = dot(&self.c[k], witness);
            if az * bz != cz {
                return Err(k);
            }
        }
        Ok(())
    }

    pub fn public_column_contributions(
        &self,
        row_weights: &[F],
        column: usize,
        scalar: F,
    ) -> Result<MatrixColumnContributions<F>, ConstraintMatrixEvalError> {
        Ok(MatrixColumnContributions {
            a: matrix_column_eval(&self.a, row_weights, column)? * scalar,
            b: matrix_column_eval(&self.b, row_weights, column)? * scalar,
            c: matrix_column_eval(&self.c, row_weights, column)? * scalar,
        })
    }

    pub fn weighted_columns(
        &self,
        row_weights: &[F],
        columns: &[usize],
    ) -> Result<WeightedMatrixColumns<F>, ConstraintMatrixEvalError> {
        if row_weights.len() < self.num_constraints {
            return Err(ConstraintMatrixEvalError::RowWeightsLengthMismatch {
                expected: self.num_constraints,
                actual: row_weights.len(),
            });
        }

        let mut weighted = WeightedMatrixColumns {
            a: Vec::with_capacity(columns.len()),
            b: Vec::with_capacity(columns.len()),
            c: Vec::with_capacity(columns.len()),
        };

        for &column in columns {
            if column >= self.num_vars {
                return Err(ConstraintMatrixEvalError::ColumnOutOfBounds {
                    column,
                    num_vars: self.num_vars,
                });
            }

            weighted
                .a
                .push(matrix_column_eval(&self.a, row_weights, column)?);
            weighted
                .b
                .push(matrix_column_eval(&self.b, row_weights, column)?);
            weighted
                .c
                .push(matrix_column_eval(&self.c, row_weights, column)?);
        }

        Ok(weighted)
    }

    pub fn evaluate_matrix_mles(
        &self,
        row_point: &[F],
        column_point: &[F],
    ) -> Result<MatrixColumnContributions<F>, ConstraintMatrixEvalError> {
        let expected_row_vars = log_padded_dimension("rows", self.num_constraints)?;
        if row_point.len() != expected_row_vars {
            return Err(ConstraintMatrixEvalError::RowPointLengthMismatch {
                expected: expected_row_vars,
                actual: row_point.len(),
            });
        }

        let expected_column_vars = log_padded_dimension("columns", self.num_vars)?;
        if column_point.len() != expected_column_vars {
            return Err(ConstraintMatrixEvalError::ColumnPointLengthMismatch {
                expected: expected_column_vars,
                actual: column_point.len(),
            });
        }

        let row_eq = EqPolynomial::new(row_point.to_vec()).evaluations();
        let column_eq = EqPolynomial::new(column_point.to_vec()).evaluations();

        Ok(MatrixColumnContributions {
            a: matrix_bilinear_eval_columns(
                &self.a,
                &row_eq,
                &column_eq[..self.num_vars],
                0,
                self.num_vars,
            )?,
            b: matrix_bilinear_eval_columns(
                &self.b,
                &row_eq,
                &column_eq[..self.num_vars],
                0,
                self.num_vars,
            )?,
            c: matrix_bilinear_eval_columns(
                &self.c,
                &row_eq,
                &column_eq[..self.num_vars],
                0,
                self.num_vars,
            )?,
        })
    }

    pub fn linear_form_bilinear_eval(
        &self,
        row_weights: &[F],
        column_weights: &[F],
        start_col: usize,
        col_count: usize,
        weights: [F; 3],
    ) -> Result<F, ConstraintMatrixEvalError> {
        let a = matrix_bilinear_eval_columns(
            &self.a,
            row_weights,
            column_weights,
            start_col,
            col_count,
        )?;
        let b = matrix_bilinear_eval_columns(
            &self.b,
            row_weights,
            column_weights,
            start_col,
            col_count,
        )?;
        let c = matrix_bilinear_eval_columns(
            &self.c,
            row_weights,
            column_weights,
            start_col,
            col_count,
        )?;
        Ok(weights[0] * a + weights[1] * b + weights[2] * c)
    }
}

fn log_padded_dimension(
    dimension: &'static str,
    raw: usize,
) -> Result<usize, ConstraintMatrixEvalError> {
    let padded = raw.max(1).checked_next_power_of_two().ok_or(
        ConstraintMatrixEvalError::PaddedDimensionOverflow {
            dimension,
            value: raw,
        },
    )?;
    Ok(padded.trailing_zeros() as usize)
}

#[inline]
fn dot<F: Field>(row: &[(usize, F)], witness: &[F]) -> F {
    let mut acc = F::zero();
    for &(col, coeff) in row {
        acc += coeff * witness[col];
    }
    acc
}

fn matrix_column_eval<F: Field>(
    rows: &[SparseRow<F>],
    row_weights: &[F],
    column: usize,
) -> Result<F, ConstraintMatrixEvalError> {
    if row_weights.len() < rows.len() {
        return Err(ConstraintMatrixEvalError::RowWeightsLengthMismatch {
            expected: rows.len(),
            actual: row_weights.len(),
        });
    }

    let mut acc = F::zero();
    for (row, &row_weight) in rows.iter().zip(row_weights) {
        for &(col, coeff) in row {
            if col == column {
                acc += row_weight * coeff;
            }
        }
    }
    Ok(acc)
}

fn matrix_bilinear_eval_columns<F: Field>(
    rows: &[SparseRow<F>],
    row_weights: &[F],
    column_weights: &[F],
    start_col: usize,
    col_count: usize,
) -> Result<F, ConstraintMatrixEvalError> {
    if row_weights.len() < rows.len() {
        return Err(ConstraintMatrixEvalError::RowWeightsLengthMismatch {
            expected: rows.len(),
            actual: row_weights.len(),
        });
    }
    if column_weights.len() != col_count {
        return Err(ConstraintMatrixEvalError::ColumnWeightsLengthMismatch {
            expected: col_count,
            actual: column_weights.len(),
        });
    }

    let end_col =
        start_col
            .checked_add(col_count)
            .ok_or(ConstraintMatrixEvalError::ColumnRangeOverflow {
                start: start_col,
                count: col_count,
            })?;
    let mut acc = F::zero();
    for (row, &row_weight) in rows.iter().zip(row_weights) {
        for &(col, coeff) in row {
            if (start_col..end_col).contains(&col) {
                acc += row_weight * column_weights[col - start_col] * coeff;
            }
        }
    }
    Ok(acc)
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests should fail loudly")]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    #[test]
    fn satisfied_constraint() {
        // x * x = y with witness [1, x=3, y=9]
        let m = ConstraintMatrices::new(
            1,
            3,
            vec![vec![(1, Fr::from_u64(1))]],
            vec![vec![(1, Fr::from_u64(1))]],
            vec![vec![(2, Fr::from_u64(1))]],
        );
        let w = vec![Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(9)];
        assert!(m.check_witness(&w).is_ok());
    }

    #[test]
    fn violated_constraint() {
        let m = ConstraintMatrices::new(
            1,
            3,
            vec![vec![(1, Fr::from_u64(1))]],
            vec![vec![(1, Fr::from_u64(1))]],
            vec![vec![(2, Fr::from_u64(1))]],
        );
        let w = vec![Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(10)];
        assert_eq!(m.check_witness(&w), Err(0));
    }

    #[test]
    fn matrix_mles_evaluate_sparse_entries() {
        let m = ConstraintMatrices::new(
            2,
            3,
            vec![vec![(1, Fr::from_u64(2))], vec![(2, Fr::from_u64(3))]],
            vec![vec![(0, Fr::from_u64(5))], vec![]],
            vec![vec![], vec![(1, Fr::from_u64(7))]],
        );

        let row_point = [Fr::from_u64(11)];
        let column_point = [Fr::from_u64(13), Fr::from_u64(17)];
        let evals = m
            .evaluate_matrix_mles(&row_point, &column_point)
            .expect("matrix MLE evaluation should accept matching point sizes");

        let row_0 = Fr::from_u64(1) - row_point[0];
        let row_1 = row_point[0];
        let col_0 = (Fr::from_u64(1) - column_point[0]) * (Fr::from_u64(1) - column_point[1]);
        let col_1 = (Fr::from_u64(1) - column_point[0]) * column_point[1];
        let col_2 = column_point[0] * (Fr::from_u64(1) - column_point[1]);

        assert_eq!(
            evals.a,
            row_0 * col_1 * Fr::from_u64(2) + row_1 * col_2 * Fr::from_u64(3)
        );
        assert_eq!(evals.b, row_0 * col_0 * Fr::from_u64(5));
        assert_eq!(evals.c, row_1 * col_1 * Fr::from_u64(7));
    }

    #[test]
    fn matrix_mles_reject_wrong_point_lengths() {
        let m = ConstraintMatrices::new(2, 3, vec![vec![]; 2], vec![vec![]; 2], vec![vec![]; 2]);

        assert_eq!(
            m.evaluate_matrix_mles(&[], &[Fr::from_u64(1), Fr::from_u64(2)]),
            Err(ConstraintMatrixEvalError::RowPointLengthMismatch {
                expected: 1,
                actual: 0
            })
        );
        assert_eq!(
            m.evaluate_matrix_mles(&[Fr::from_u64(1)], &[Fr::from_u64(2)]),
            Err(ConstraintMatrixEvalError::ColumnPointLengthMismatch {
                expected: 2,
                actual: 1
            })
        );
    }

    #[test]
    fn matrix_mles_reject_unpaddable_dimensions() {
        let m: ConstraintMatrices<Fr> =
            ConstraintMatrices::new(1, usize::MAX, vec![vec![]], vec![vec![]], vec![vec![]]);

        assert_eq!(
            m.evaluate_matrix_mles(&[], &[]),
            Err(ConstraintMatrixEvalError::PaddedDimensionOverflow {
                dimension: "columns",
                value: usize::MAX
            })
        );
    }

    #[test]
    #[should_panic(expected = "num_constraints")]
    fn new_rejects_row_count_mismatch() {
        let _ = ConstraintMatrices::new(
            2,
            3,
            vec![vec![(1, Fr::from_u64(1))]],
            vec![vec![(1, Fr::from_u64(1))]],
            vec![vec![(2, Fr::from_u64(1))]],
        );
    }

    #[test]
    #[should_panic(expected = "num_vars")]
    fn new_rejects_oob_column() {
        let _ = ConstraintMatrices::new(
            1,
            3,
            vec![vec![(99, Fr::from_u64(1))]],
            vec![vec![(1, Fr::from_u64(1))]],
            vec![vec![(2, Fr::from_u64(1))]],
        );
    }

    #[test]
    fn public_column_contributions_projects_abc_column() {
        let m = ConstraintMatrices::new(
            2,
            3,
            vec![
                vec![(0, Fr::from_u64(2)), (1, Fr::from_u64(99))],
                vec![(0, Fr::from_u64(3))],
            ],
            vec![vec![(0, Fr::from_u64(5))], vec![(2, Fr::from_u64(13))]],
            vec![vec![(1, Fr::from_u64(17))], vec![(0, Fr::from_u64(7))]],
        );
        let row_weights = [Fr::from_u64(11), Fr::from_u64(19), Fr::from_u64(23)];

        let contributions = m
            .public_column_contributions(&row_weights, 0, Fr::from_u64(3))
            .expect("row weights cover all constraints");

        assert_eq!(
            contributions,
            MatrixColumnContributions {
                a: Fr::from_u64(237),
                b: Fr::from_u64(165),
                c: Fr::from_u64(399),
            }
        );
    }

    #[test]
    fn weighted_columns_projects_multiple_abc_columns() {
        let m = ConstraintMatrices::new(
            2,
            3,
            vec![
                vec![(0, Fr::from_u64(2)), (1, Fr::from_u64(99))],
                vec![(0, Fr::from_u64(3)), (2, Fr::from_u64(4))],
            ],
            vec![
                vec![(0, Fr::from_u64(5)), (2, Fr::from_u64(7))],
                vec![(2, Fr::from_u64(13))],
            ],
            vec![vec![(1, Fr::from_u64(17))], vec![(0, Fr::from_u64(7))]],
        );
        let row_weights = [Fr::from_u64(11), Fr::from_u64(19)];

        let weighted = m
            .weighted_columns(&row_weights, &[0, 2])
            .expect("row weights cover all constraints");

        assert_eq!(
            weighted,
            WeightedMatrixColumns {
                a: vec![Fr::from_u64(79), Fr::from_u64(76)],
                b: vec![Fr::from_u64(55), Fr::from_u64(324)],
                c: vec![Fr::from_u64(133), Fr::from_u64(0)],
            }
        );
    }

    #[test]
    fn linear_form_bilinear_eval_combines_weighted_matrices() {
        let m = ConstraintMatrices::new(
            2,
            4,
            vec![
                vec![(0, Fr::from_u64(100)), (1, Fr::from_u64(2))],
                vec![(2, Fr::from_u64(3))],
            ],
            vec![
                vec![(1, Fr::from_u64(5)), (3, Fr::from_u64(7))],
                vec![(2, Fr::from_u64(11))],
            ],
            vec![vec![(3, Fr::from_u64(13))], vec![(1, Fr::from_u64(17))]],
        );
        let row_weights = [Fr::from_u64(2), Fr::from_u64(3)];
        let column_weights = [Fr::from_u64(5), Fr::from_u64(7), Fr::from_u64(11)];

        let value = m
            .linear_form_bilinear_eval(
                &row_weights,
                &column_weights,
                1,
                3,
                [Fr::from_u64(19), Fr::from_u64(23), Fr::from_u64(29)],
            )
            .expect("weights match the selected columns");

        assert_eq!(value, Fr::from_u64(27271));
    }

    #[test]
    fn matrix_eval_rejects_short_row_weights() {
        let m = ConstraintMatrices::new(
            2,
            2,
            vec![vec![(0, Fr::from_u64(1))], vec![(1, Fr::from_u64(2))]],
            vec![Vec::new(), Vec::new()],
            vec![Vec::new(), Vec::new()],
        );

        let error = m
            .public_column_contributions(&[Fr::from_u64(1)], 0, Fr::from_u64(1))
            .expect_err("one row weight cannot cover two constraints");

        assert_eq!(
            error,
            ConstraintMatrixEvalError::RowWeightsLengthMismatch {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn matrix_eval_rejects_column_weight_mismatch() {
        let m = ConstraintMatrices::new(
            1,
            3,
            vec![vec![(1, Fr::from_u64(1))]],
            vec![Vec::new()],
            vec![Vec::new()],
        );

        let error = m
            .linear_form_bilinear_eval(
                &[Fr::from_u64(1)],
                &[Fr::from_u64(1)],
                1,
                2,
                [Fr::from_u64(1), Fr::from_u64(1), Fr::from_u64(1)],
            )
            .expect_err("selected column count must match weights");

        assert_eq!(
            error,
            ConstraintMatrixEvalError::ColumnWeightsLengthMismatch {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn matrix_eval_rejects_column_range_overflow() {
        let m = ConstraintMatrices::new(0, 1, Vec::new(), Vec::new(), Vec::new());

        let error = m
            .linear_form_bilinear_eval(
                &[],
                &[Fr::from_u64(1), Fr::from_u64(1)],
                usize::MAX,
                2,
                [Fr::from_u64(1), Fr::from_u64(1), Fr::from_u64(1)],
            )
            .expect_err("column range must not overflow");

        assert_eq!(
            error,
            ConstraintMatrixEvalError::ColumnRangeOverflow {
                start: usize::MAX,
                count: 2,
            }
        );
    }
}
