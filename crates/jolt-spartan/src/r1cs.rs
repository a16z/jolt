//! R1CS trait and concrete sparse implementation.
//!
//! A Rank-1 Constraint System (R1CS) consists of three matrices $A$, $B$, $C$
//! such that for a valid witness $z$:
//! $$Az \circ Bz = Cz$$
//! where $\circ$ denotes the Hadamard (element-wise) product.

use jolt_field::Field;

/// A Rank-1 Constraint System over a field $\mathbb{F}$.
///
/// Defines three sparse matrices $A$, $B$, $C$ with `num_constraints` rows
/// and `num_variables` columns. A witness vector $z \in \mathbb{F}^m$ is
/// *satisfying* iff $Az \circ Bz = Cz$ element-wise.
pub trait R1CS<F: Field> {
    /// Number of constraints (rows in each matrix).
    fn num_constraints(&self) -> usize;

    /// Number of variables (columns in each matrix), including the constant `1` variable.
    fn num_variables(&self) -> usize;

    /// Computes the matrix-vector products $(Az, Bz, Cz)$.
    ///
    /// Returns three vectors, each of length [`num_constraints`](R1CS::num_constraints).
    fn multiply_witness(&self, witness: &[F]) -> (Vec<F>, Vec<F>, Vec<F>);
}

/// A sparse R1CS representation for testing and small circuits.
///
/// Each matrix is stored as a list of `(row, col, value)` triples.
/// No ordering or deduplication is required; duplicate entries for the
/// same `(row, col)` pair are summed during multiplication.
pub struct SimpleR1CS<F: Field> {
    num_constraints: usize,
    num_variables: usize,
    a_entries: Vec<(usize, usize, F)>,
    b_entries: Vec<(usize, usize, F)>,
    c_entries: Vec<(usize, usize, F)>,
}

impl<F: Field> SimpleR1CS<F> {
    /// Creates a new sparse R1CS instance.
    ///
    /// # Arguments
    ///
    /// * `num_constraints` - Number of constraint rows.
    /// * `num_variables` - Number of witness variables (including the constant term).
    /// * `a_entries` - Sparse entries for the $A$ matrix as `(row, col, value)` triples.
    /// * `b_entries` - Sparse entries for the $B$ matrix.
    /// * `c_entries` - Sparse entries for the $C$ matrix.
    pub fn new(
        num_constraints: usize,
        num_variables: usize,
        a_entries: Vec<(usize, usize, F)>,
        b_entries: Vec<(usize, usize, F)>,
        c_entries: Vec<(usize, usize, F)>,
    ) -> Self {
        Self {
            num_constraints,
            num_variables,
            a_entries,
            b_entries,
            c_entries,
        }
    }
}

/// Sparse matrix-vector product: given `(row, col, val)` entries and a witness,
/// compute the result vector of length `num_rows`.
fn sparse_matvec<F: Field>(
    entries: &[(usize, usize, F)],
    witness: &[F],
    num_rows: usize,
) -> Vec<F> {
    let mut result = vec![F::zero(); num_rows];
    for &(row, col, val) in entries {
        result[row] += val * witness[col];
    }
    result
}

impl<F: Field> R1CS<F> for SimpleR1CS<F> {
    fn num_constraints(&self) -> usize {
        self.num_constraints
    }

    fn num_variables(&self) -> usize {
        self.num_variables
    }

    fn multiply_witness(&self, witness: &[F]) -> (Vec<F>, Vec<F>, Vec<F>) {
        let az = sparse_matvec(&self.a_entries, witness, self.num_constraints);
        let bz = sparse_matvec(&self.b_entries, witness, self.num_constraints);
        let cz = sparse_matvec(&self.c_entries, witness, self.num_constraints);
        (az, bz, cz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use jolt_field::Field;
    use num_traits::Zero;

    #[test]
    fn simple_r1cs_x_squared_eq_y() {
        // x * x = y with witness [1, x=3, y=9]
        let r1cs = SimpleR1CS::<Fr>::new(
            1,
            3,
            vec![(0, 1, Fr::from_u64(1))],
            vec![(0, 1, Fr::from_u64(1))],
            vec![(0, 2, Fr::from_u64(1))],
        );
        let witness = vec![Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(9)];
        let (az, bz, cz) = r1cs.multiply_witness(&witness);
        assert_eq!(az[0] * bz[0], cz[0]);
    }

    #[test]
    fn unsatisfied_r1cs_detected() {
        let r1cs = SimpleR1CS::<Fr>::new(
            1,
            3,
            vec![(0, 1, Fr::from_u64(1))],
            vec![(0, 1, Fr::from_u64(1))],
            vec![(0, 2, Fr::from_u64(1))],
        );
        let witness = vec![Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(10)];
        let (az, bz, cz) = r1cs.multiply_witness(&witness);
        assert_ne!(az[0] * bz[0], cz[0]);
    }

    #[test]
    fn empty_matrix_gives_zero() {
        let r1cs = SimpleR1CS::<Fr>::new(2, 3, vec![], vec![], vec![]);
        let witness = vec![Fr::from_u64(1), Fr::from_u64(5), Fr::from_u64(7)];
        let (az, bz, cz) = r1cs.multiply_witness(&witness);
        for i in 0..2 {
            assert!(az[i].is_zero());
            assert!(bz[i].is_zero());
            assert!(cz[i].is_zero());
        }
    }
}
