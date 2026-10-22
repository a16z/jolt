//! The BlindFold statement: the residual relation lowered to relaxed R1CS
//! over a row-committed witness.
//!
//! The witness is laid out in rows of `row_len` values:
//!
//! - the pad rows, in pad order, whose commitments are the transcript's
//!   pad commitment (never recommitted);
//! - two dedicated rows holding the residual evaluation y and its blind ρ,
//!   bound to the residual (evaluation) commitment;
//! - the product rows: one witness-only intermediate w per pair of a
//!   multi-pair product row but its last, in row order;
//! - zero padding up to a power of two.
//!
//! Variable 0 is the relaxed instance's u, so every constant of a row
//! multiplies u: an affine row Σ a_k x_k + b becomes
//! (Σ a_k x_k + b·u) · u = 0, and a product pair x · y becomes
//! (u·m̂_x − p_x) · y with the constants of y and z on u. A product row
//! Σ_{k≤n} x_k · y_k = z takes n constraints: x_k · y_k = w_k for k < n and
//! x_n · y_n = z − Σ_{k<n} w_k.

use std::ops::Range;

use jolt_algebra::JoltField;
use jolt_crypto::HomomorphicCommitment;
use jolt_r1cs::{ConstraintMatrices, LinearCombination, Variable};

use super::{RelaxedError, RelaxedInstance};
use crate::masking::{
    AffineRow, ProductPair, ProductRow, ResidualRelation, ResidualVar, Row, ZkError,
};

#[derive(Clone, Debug)]
pub struct BlindFoldProtocol<F: JoltField, Com> {
    pub r1cs: ConstraintMatrices<F>,
    pub dimensions: BlindFoldDimensions,
    pub pad_row_commitments: Vec<Com>,
    pub eval_commitment: Com,
    pub final_opening: FinalOpeningWitnessCoordinates,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowDimensions {
    pub row_len: usize,
    pub row_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlindFoldDimensions {
    pub witness: RowDimensions,
    pub error: RowDimensions,
    pub witness_rows: WitnessRowLayout,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WitnessRowLayout {
    pub pads: Range<usize>,
    /// The y and ρ rows, then the product rows.
    pub auxiliary: Range<usize>,
    /// The intermediates of multi-pair product rows.
    pub products: Range<usize>,
    pub padding: Range<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WitnessCoordinate {
    pub row: usize,
    pub column: usize,
}

/// Where the residual evaluation and its blind sit in the witness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FinalOpeningWitnessCoordinates {
    pub evaluation: WitnessCoordinate,
    pub blinding: WitnessCoordinate,
}

/// The number of pad rows a layout of `num_pads` pads takes at `row_len`.
pub fn pad_row_count(num_pads: usize, row_len: usize) -> usize {
    num_pads.div_ceil(row_len.max(1))
}

/// The pairs of `row` that take a witness-only intermediate: all but the
/// last.
fn intermediates<F>(row: &ProductRow<F>) -> impl Iterator<Item = &ProductPair<F>> {
    row.pairs().take(row.rest.len())
}

/// The intermediates of `relation`'s multi-pair product rows on an
/// assignment, in their witness order: `x_k · y_k` for every pair of a row
/// but its last, row by row.
///
/// # Errors
///
/// [`ZkError`] if a row names a variable outside the assignment.
pub(super) fn product_intermediates<F: JoltField>(
    relation: &ResidualRelation<F>,
    pads: &[F],
    witness: &[F],
) -> Result<Vec<F>, ZkError> {
    relation
        .products
        .iter()
        .flat_map(intermediates)
        .map(|pair| Ok(pair.x.form().evaluate(pads, witness)? * pair.y.evaluate(pads, witness)?))
        .collect()
}

impl<F: JoltField, Com> BlindFoldProtocol<F, Com> {
    /// Lowers `relation` at witness row length `row_len` (a power of two).
    /// Its one witness coordinate is the evaluation committed in
    /// `eval_commitment`.
    ///
    /// # Errors
    ///
    /// [`RelaxedError`] if the dimensions are inconsistent: a row length that
    /// is not a power of two, a pad-commitment count that differs from the
    /// layout, or a relation whose witness is not the single evaluation.
    pub fn new(
        relation: &ResidualRelation<F>,
        row_len: usize,
        pad_row_commitments: Vec<Com>,
        eval_commitment: Com,
    ) -> Result<Self, RelaxedError> {
        if row_len == 0 || !row_len.is_power_of_two() {
            return Err(RelaxedError::DimensionOverflow {
                name: "witness row length",
                value: row_len,
            });
        }
        if relation.num_witness != 1 {
            return Err(RelaxedError::LengthMismatch {
                name: "residual witness coordinates",
                expected: 1,
                actual: relation.num_witness,
            });
        }
        let pad_rows = pad_row_count(relation.layout.num_pads, row_len);
        ensure_len("pad row commitments", pad_rows, pad_row_commitments.len())?;
        let evaluation_row = pad_rows;
        let blinding_row = checked_sum("witness rows", [pad_rows, 1])?;
        let products_row = checked_sum("witness rows", [pad_rows, 2])?;
        let num_intermediates = relation
            .products
            .iter()
            .map(|row| row.rest.len())
            .try_fold(0usize, |sum, count| sum.checked_add(count))
            .ok_or(RelaxedError::DimensionOverflow {
                name: "product intermediates",
                value: relation.products.len(),
            })?;
        let occupied = checked_sum(
            "witness rows",
            [products_row, num_intermediates.div_ceil(row_len)],
        )?;
        let row_count = checked_next_power_of_two("witness rows", occupied)?;
        let witness_values =
            row_count
                .checked_mul(row_len)
                .ok_or(RelaxedError::DimensionOverflow {
                    name: "witness values",
                    value: row_count,
                })?;

        let variables = Variables {
            row_len,
            num_pads: relation.layout.num_pads,
            evaluation_row,
        };
        let mut next_intermediate = products_row * row_len;
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut c = Vec::new();
        for row in relation.rows() {
            match row {
                Row::Affine(row) => {
                    a.push(variables.lower(row)?.into_sparse_row());
                    b.push(LinearCombination::one().into_sparse_row());
                    c.push(LinearCombination::zero().into_sparse_row());
                }
                Row::Product(row) => {
                    let last = row.rest.last().unwrap_or(&row.first);
                    let mut z = variables.lower(&row.z)?;
                    for pair in intermediates(row) {
                        let w =
                            Variable::new(checked_sum("R1CS variable", [next_intermediate, 1])?);
                        next_intermediate += 1;
                        a.push(variables.lower(&pair.x.form())?.into_sparse_row());
                        b.push(variables.lower(&pair.y)?.into_sparse_row());
                        c.push(
                            LinearCombination {
                                terms: vec![(w, F::one())],
                            }
                            .into_sparse_row(),
                        );
                        z.terms.push((w, -F::one()));
                    }
                    a.push(variables.lower(&last.x.form())?.into_sparse_row());
                    b.push(variables.lower(&last.y)?.into_sparse_row());
                    c.push(z.into_sparse_row());
                }
            }
        }
        let num_constraints = a.len();
        let num_vars = checked_sum("R1CS variables", [witness_values, 1])?;
        let r1cs = ConstraintMatrices::new(num_constraints, num_vars, a, b, c);

        let padded_constraints = checked_next_power_of_two("error values", num_constraints.max(1))?;
        let error_row_len = row_len.min(padded_constraints);
        let dimensions = BlindFoldDimensions {
            witness: RowDimensions { row_len, row_count },
            error: RowDimensions {
                row_len: error_row_len,
                row_count: padded_constraints / error_row_len,
            },
            witness_rows: WitnessRowLayout {
                pads: 0..pad_rows,
                auxiliary: pad_rows..occupied,
                products: products_row..occupied,
                padding: occupied..row_count,
            },
        };

        Ok(Self {
            r1cs,
            dimensions,
            pad_row_commitments,
            eval_commitment,
            final_opening: FinalOpeningWitnessCoordinates {
                evaluation: WitnessCoordinate {
                    row: evaluation_row,
                    column: 0,
                },
                blinding: WitnessCoordinate {
                    row: blinding_row,
                    column: 0,
                },
            },
        })
    }
}

/// The R1CS variable of each residual variable: pad i is variable 1 + i, the
/// evaluation the first entry of its dedicated row.
struct Variables {
    row_len: usize,
    num_pads: usize,
    evaluation_row: usize,
}

impl Variables {
    fn variable(&self, variable: ResidualVar) -> Result<Variable, RelaxedError> {
        let index = match variable {
            ResidualVar::Pad(pad) if pad.0 < self.num_pads => pad.0,
            ResidualVar::Witness(0) => self.evaluation_row * self.row_len,
            ResidualVar::Pad(_) | ResidualVar::Witness(_) => {
                return Err(RelaxedError::InconsistentDimensions {
                    name: "residual row variable",
                    total: self.num_pads,
                    used: 0,
                })
            }
        };
        Ok(Variable::new(checked_sum("R1CS variable", [index, 1])?))
    }

    fn lower<F: JoltField>(
        &self,
        row: &AffineRow<F>,
    ) -> Result<LinearCombination<F>, RelaxedError> {
        let mut terms = Vec::with_capacity(row.terms.len() + 1);
        for &(variable, coefficient) in &row.terms {
            terms.push((self.variable(variable)?, coefficient));
        }
        terms.push((Variable::ONE, row.constant));
        Ok(LinearCombination { terms })
    }
}

impl<F, Com> BlindFoldProtocol<F, Com>
where
    F: JoltField,
    Com: Clone + HomomorphicCommitment<F>,
{
    /// The committed instance: u = 1, the pad rows from the pad commitment,
    /// the dedicated rows from `auxiliary_row_commitments`, identity padding,
    /// and a zero error.
    pub fn committed_relaxed_instance(
        &self,
        auxiliary_row_commitments: &[Com],
    ) -> Result<RelaxedInstance<F, Com>, RelaxedError> {
        Ok(RelaxedInstance::new(
            F::one(),
            self.witness_row_commitments(&self.pad_row_commitments, auxiliary_row_commitments)?,
            vec![Com::default(); self.dimensions.error.row_count],
            vec![self.eval_commitment.clone()],
        ))
    }

    pub fn random_relaxed_instance(
        &self,
        pad_row_commitments: &[Com],
        auxiliary_row_commitments: &[Com],
        error_row_commitments: &[Com],
        eval_commitment: &Com,
        u: F,
    ) -> Result<RelaxedInstance<F, Com>, RelaxedError> {
        ensure_len(
            "random error row commitments",
            self.dimensions.error.row_count,
            error_row_commitments.len(),
        )?;
        Ok(RelaxedInstance::new(
            u,
            self.witness_row_commitments(pad_row_commitments, auxiliary_row_commitments)?,
            error_row_commitments.to_vec(),
            vec![eval_commitment.clone()],
        ))
    }

    pub fn validate_cross_term_error_rows(
        &self,
        cross_term_error_row_commitments: &[Com],
    ) -> Result<(), RelaxedError> {
        ensure_len(
            "cross-term error row commitments",
            self.dimensions.error.row_count,
            cross_term_error_row_commitments.len(),
        )
    }

    fn witness_row_commitments(
        &self,
        pad_rows: &[Com],
        auxiliary_rows: &[Com],
    ) -> Result<Vec<Com>, RelaxedError> {
        let layout = &self.dimensions.witness_rows;
        ensure_len("pad row commitments", layout.pads.len(), pad_rows.len())?;
        ensure_len(
            "auxiliary row commitments",
            layout.auxiliary.len(),
            auxiliary_rows.len(),
        )?;
        let mut rows = Vec::with_capacity(self.dimensions.witness.row_count);
        rows.extend_from_slice(pad_rows);
        rows.extend_from_slice(auxiliary_rows);
        rows.resize_with(self.dimensions.witness.row_count, Com::default);
        Ok(rows)
    }
}

pub(crate) fn ensure_len(
    name: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), RelaxedError> {
    if expected != actual {
        return Err(RelaxedError::LengthMismatch {
            name,
            expected,
            actual,
        });
    }
    Ok(())
}

fn checked_next_power_of_two(name: &'static str, value: usize) -> Result<usize, RelaxedError> {
    value
        .checked_next_power_of_two()
        .ok_or(RelaxedError::DimensionOverflow { name, value })
}

fn checked_sum<const N: usize>(
    name: &'static str,
    values: [usize; N],
) -> Result<usize, RelaxedError> {
    values.into_iter().try_fold(0usize, |sum, value| {
        sum.checked_add(value)
            .ok_or(RelaxedError::DimensionOverflow { name, value: sum })
    })
}
