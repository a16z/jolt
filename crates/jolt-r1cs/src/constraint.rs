//! Sparse per-cycle R1CS constraint matrices.

use jolt_field::Field;
use serde::{Deserialize, Serialize};

/// Sparse row: `[(variable_index, coefficient)]`.
pub type SparseRow<F> = Vec<(usize, F)>;

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
}

#[inline]
fn dot<F: Field>(row: &[(usize, F)], witness: &[F]) -> F {
    let mut acc = F::zero();
    for &(col, coeff) in row {
        acc += coeff * witness[col];
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;

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
}
