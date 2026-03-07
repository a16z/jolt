//! Spartan proving/verification key derived from an R1CS instance.
//!
//! The key precomputes multilinear extensions (MLEs) of the constraint
//! matrices, enabling efficient evaluation during the sumcheck protocol.

use jolt_field::Field;
use jolt_poly::Polynomial;
use serde::{Deserialize, Serialize};

use crate::r1cs::R1CS;

/// Precomputed Spartan key containing MLEs of the R1CS matrices.
///
/// Given an R1CS with $m$ constraints and $n$ variables, each matrix is
/// encoded as a multilinear polynomial in $\log_2 m + \log_2 n$ variables.
/// Entry $(i, j)$ of the matrix maps to hypercube point
/// $(i_{\log m - 1}, \ldots, i_0, j_{\log n - 1}, \ldots, j_0)$.
///
/// Both dimensions are padded to the next power of two. Unused entries are zero.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SpartanKey<F: Field> {
    /// Number of constraints (pre-padding).
    pub num_constraints: usize,
    /// Number of variables (pre-padding).
    pub num_variables: usize,
    /// Padded number of constraints (power of two).
    pub num_constraints_padded: usize,
    /// Padded number of variables (power of two).
    pub num_variables_padded: usize,
    /// MLE of the $A$ matrix.
    a_mle: Polynomial<F>,
    /// MLE of the $B$ matrix.
    b_mle: Polynomial<F>,
    /// MLE of the $C$ matrix.
    c_mle: Polynomial<F>,
}

impl<F: Field> SpartanKey<F> {
    /// Builds a [`SpartanKey`] from an R1CS instance.
    ///
    /// Uses the R1CS sparse entry API to construct dense MLE tables in a
    /// single pass over each matrix's nonzero entries, avoiding the $O(n)$
    /// basis-vector probing of the default implementation.
    #[tracing::instrument(skip_all, name = "SpartanKey::from_r1cs")]
    pub fn from_r1cs(r1cs: &impl R1CS<F>) -> Self {
        let m = r1cs.num_constraints();
        let n = r1cs.num_variables();
        let m_padded = m.next_power_of_two();
        let n_padded = n.next_power_of_two();
        let total = m_padded * n_padded;

        let a_mle = sparse_entries_to_mle(&r1cs.sparse_entries_a(), n_padded, total);
        let b_mle = sparse_entries_to_mle(&r1cs.sparse_entries_b(), n_padded, total);
        let c_mle = sparse_entries_to_mle(&r1cs.sparse_entries_c(), n_padded, total);

        Self {
            num_constraints: m,
            num_variables: n,
            num_constraints_padded: m_padded,
            num_variables_padded: n_padded,
            a_mle,
            b_mle,
            c_mle,
        }
    }

    pub fn a_mle(&self) -> &Polynomial<F> {
        &self.a_mle
    }

    pub fn b_mle(&self) -> &Polynomial<F> {
        &self.b_mle
    }

    pub fn c_mle(&self) -> &Polynomial<F> {
        &self.c_mle
    }

    /// Number of sumcheck variables: $\log_2(m')$ where $m'$ is the padded
    /// constraint count. This determines the number of rounds in the outer
    /// Spartan sumcheck.
    pub fn num_sumcheck_vars(&self) -> usize {
        self.num_constraints_padded.trailing_zeros() as usize
    }

    /// Number of variables in the witness MLE: $\log_2(n')$. This determines
    /// the dimensionality of witness opening evaluation points.
    pub fn num_witness_vars(&self) -> usize {
        self.num_variables_padded.trailing_zeros() as usize
    }

    /// Evaluates all three matrix MLEs at a combined point $(r_x, r_y)$.
    ///
    /// Each matrix MLE has $\log_2(m') + \log_2(n')$ variables, where the first
    /// $\log_2(m')$ correspond to the constraint index and the last $\log_2(n')$
    /// correspond to the variable index.
    pub fn evaluate_matrix_mles(&self, r_x: &[F], r_y: &[F]) -> (F, F, F) {
        let mut point = Vec::with_capacity(r_x.len() + r_y.len());
        point.extend_from_slice(r_x);
        point.extend_from_slice(r_y);
        (
            self.a_mle.evaluate(&point),
            self.b_mle.evaluate(&point),
            self.c_mle.evaluate(&point),
        )
    }
}

/// Builds a dense MLE from sparse `(row, col, value)` entries.
///
/// Computes the flat index `row * n_padded + col` for each entry and scatters
/// into a dense evaluation table of size `total = m_padded * n_padded`.
fn sparse_entries_to_mle<F: Field>(
    entries: &[(usize, usize, F)],
    n_padded: usize,
    total: usize,
) -> Polynomial<F> {
    let mut evals = vec![F::zero(); total];
    for &(row, col, val) in entries {
        evals[row * n_padded + col] = val;
    }
    Polynomial::new(evals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r1cs::SimpleR1CS;
    use jolt_field::Field;
    use jolt_field::Fr;

    #[test]
    fn key_dimensions_match_r1cs() {
        let r1cs = SimpleR1CS::<Fr>::new(
            1,
            3,
            vec![(0, 1, Fr::from_u64(1))],
            vec![(0, 1, Fr::from_u64(1))],
            vec![(0, 2, Fr::from_u64(1))],
        );
        let key = SpartanKey::from_r1cs(&r1cs);

        assert_eq!(key.num_constraints, 1);
        assert_eq!(key.num_variables, 3);
        assert_eq!(key.num_constraints_padded, 1);
        assert_eq!(key.num_variables_padded, 4);
        // MLE has log2(1) + log2(4) = 0 + 2 = 2 vars, so 4 evaluations
        assert_eq!(key.a_mle().len(), 4);
    }
}
