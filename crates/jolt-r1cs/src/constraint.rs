//! Sparse per-cycle R1CS constraint matrices.

use jolt_field::Field;

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
#[derive(Clone, Debug)]
pub struct ConstraintMatrices<F: Field> {
    pub num_constraints: usize,
    pub num_vars: usize,
    pub a: Vec<SparseRow<F>>,
    pub b: Vec<SparseRow<F>>,
    pub c: Vec<SparseRow<F>>,
}

impl<F: Field> ConstraintMatrices<F> {
    /// Builds constraint matrices from sparse rows.
    ///
    /// # Panics
    ///
    /// Panics if any matrix has a different number of rows than `num_constraints`.
    pub fn new(
        num_constraints: usize,
        num_vars: usize,
        a: Vec<SparseRow<F>>,
        b: Vec<SparseRow<F>>,
        c: Vec<SparseRow<F>>,
    ) -> Self {
        assert_eq!(a.len(), num_constraints);
        assert_eq!(b.len(), num_constraints);
        assert_eq!(c.len(), num_constraints);
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
}
