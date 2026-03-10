//! Uniform Spartan key for repeated-constraint R1CS instances.
//!
//! A **uniform** R1CS has the same constraints replicated across `num_cycles`
//! cycles, making it natural for execution traces. Instead of storing dense
//! matrix MLEs (which would be `O(total_rows * total_cols)`), the key stores
//! only the per-block sparse constraint matrices. Matrix MLE evaluations are
//! computed on-the-fly using equality polynomial evaluations.
//!
//! # Layout
//!
//! The full R1CS has `num_cycles * num_constraints_per_cycle` rows and
//! `num_cycles * num_vars_per_cycle` columns. Row index `i` decomposes as
//! `(cycle, constraint) = (i / K_padded, i % K_padded)`, column index `j`
//! as `(cycle, variable) = (j / V_padded, j % V_padded)`.
//!
//! Matrix entry: `M[cycle*K + k, cycle*V + v] = M_local[k, v]`
//! (only same-cycle entries are nonzero).
//!
//! # MLE evaluation
//!
//! By the uniform structure:
//! $$\tilde{M}(r_x, r_y) = \widetilde{eq}(r_x^{\text{cycle}}, r_y^{\text{cycle}}) \cdot \tilde{M}_{\text{local}}(r_x^{\text{constr}}, r_y^{\text{var}})$$
//!
//! where the local MLE is evaluated from the sparse representation:
//! $$\tilde{M}_{\text{local}}(r_c, r_v) = \sum_{k} \widetilde{eq}(k, r_c) \sum_{(j, \alpha) \in M[k]} \alpha \cdot \widetilde{eq}(j, r_v)$$

use jolt_field::Field;
use jolt_poly::EqPolynomial;
use serde::{Deserialize, Serialize};

/// A Spartan key for uniform (repeated-constraint) R1CS instances.
///
/// Stores the per-cycle sparse constraint matrices and dimensional metadata.
/// All three matrices share the same sparsity pattern: `a_sparse[k]` is a
/// list of `(variable_index, coefficient)` pairs for constraint row `k`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct UniformSpartanKey<F: Field> {
    /// Number of execution cycles (padded to power of two).
    pub num_cycles: usize,
    /// Number of constraints per cycle (pre-padding).
    pub num_constraints: usize,
    /// Number of variables per cycle (pre-padding).
    pub num_vars: usize,
    /// Padded constraint count per cycle (power of two).
    pub num_constraints_padded: usize,
    /// Padded variable count per cycle (power of two).
    pub num_vars_padded: usize,
    /// Sparse $A$ matrix per cycle: `a_sparse[constraint] = [(var, coeff), ...]`.
    pub a_sparse: Vec<Vec<(usize, F)>>,
    /// Sparse $B$ matrix per cycle.
    pub b_sparse: Vec<Vec<(usize, F)>>,
    /// Sparse $C$ matrix per cycle.
    pub c_sparse: Vec<Vec<(usize, F)>>,
}

impl<F: Field> UniformSpartanKey<F> {
    /// Creates a new uniform Spartan key.
    ///
    /// # Arguments
    ///
    /// * `num_cycles` — Number of execution cycles (must be a power of two).
    /// * `num_constraints` — Constraints per cycle (pre-padding).
    /// * `num_vars` — Variables per cycle (pre-padding).
    /// * `a_sparse`, `b_sparse`, `c_sparse` — Per-constraint sparse entries.
    ///   Each inner vec has `(variable_index, coefficient)` pairs. Must have
    ///   exactly `num_constraints` rows.
    pub fn new(
        num_cycles: usize,
        num_constraints: usize,
        num_vars: usize,
        a_sparse: Vec<Vec<(usize, F)>>,
        b_sparse: Vec<Vec<(usize, F)>>,
        c_sparse: Vec<Vec<(usize, F)>>,
    ) -> Self {
        assert!(
            num_cycles.is_power_of_two(),
            "num_cycles must be a power of two"
        );
        assert_eq!(a_sparse.len(), num_constraints);
        assert_eq!(b_sparse.len(), num_constraints);
        assert_eq!(c_sparse.len(), num_constraints);

        let num_constraints_padded = num_constraints.next_power_of_two();
        let num_vars_padded = num_vars.next_power_of_two();

        Self {
            num_cycles,
            num_constraints,
            num_vars,
            num_constraints_padded,
            num_vars_padded,
            a_sparse,
            b_sparse,
            c_sparse,
        }
    }

    /// Number of bits for the cycle dimension: $\log_2(\text{num\_cycles})$.
    #[inline]
    pub fn num_cycle_vars(&self) -> usize {
        self.num_cycles.trailing_zeros() as usize
    }

    /// Number of bits for the within-cycle constraint dimension.
    #[inline]
    pub fn num_constraint_vars(&self) -> usize {
        self.num_constraints_padded.trailing_zeros() as usize
    }

    /// Number of bits for the within-cycle variable dimension.
    #[inline]
    pub fn num_var_vars(&self) -> usize {
        self.num_vars_padded.trailing_zeros() as usize
    }

    /// Total number of rows: `num_cycles * num_constraints_padded`.
    #[inline]
    pub fn total_rows(&self) -> usize {
        self.num_cycles * self.num_constraints_padded
    }

    /// Total number of columns: `num_cycles * num_vars_padded`.
    #[inline]
    pub fn total_cols(&self) -> usize {
        self.num_cycles * self.num_vars_padded
    }

    /// Number of row variables (outer sumcheck rounds).
    #[inline]
    pub fn num_row_vars(&self) -> usize {
        self.num_cycle_vars() + self.num_constraint_vars()
    }

    /// Number of column variables (inner sumcheck rounds).
    #[inline]
    pub fn num_col_vars(&self) -> usize {
        self.num_cycle_vars() + self.num_var_vars()
    }

    /// Evaluates the local (per-block) matrix MLEs at points
    /// `(constraint_point, var_point)`.
    ///
    /// Returns $(A_{\text{local}}, B_{\text{local}}, C_{\text{local}})$ where:
    /// $$M_{\text{local}}(r_c, r_v) = \sum_k \widetilde{eq}(k, r_c) \sum_{(j, \alpha)} \alpha \cdot \widetilde{eq}(j, r_v)$$
    pub fn evaluate_local_mles(&self, constraint_point: &[F], var_point: &[F]) -> (F, F, F) {
        assert_eq!(constraint_point.len(), self.num_constraint_vars());
        assert_eq!(var_point.len(), self.num_var_vars());

        let eq_constraint = EqPolynomial::new(constraint_point.to_vec()).evaluations();
        let eq_var = EqPolynomial::new(var_point.to_vec()).evaluations();

        let mut a_eval = F::zero();
        let mut b_eval = F::zero();
        let mut c_eval = F::zero();

        for (k, (a_row_entries, (b_row_entries, c_row_entries))) in self
            .a_sparse
            .iter()
            .zip(self.b_sparse.iter().zip(self.c_sparse.iter()))
            .enumerate()
        {
            let w = eq_constraint[k];
            if w.is_zero() {
                continue;
            }

            let mut a_row = F::zero();
            for &(j, coeff) in a_row_entries {
                a_row += coeff * eq_var[j];
            }
            a_eval += w * a_row;

            let mut b_row = F::zero();
            for &(j, coeff) in b_row_entries {
                b_row += coeff * eq_var[j];
            }
            b_eval += w * b_row;

            let mut c_row = F::zero();
            for &(j, coeff) in c_row_entries {
                c_row += coeff * eq_var[j];
            }
            c_eval += w * c_row;
        }

        (a_eval, b_eval, c_eval)
    }

    /// Evaluates the full matrix MLEs at points $(r_x, r_y)$.
    ///
    /// Splits $r_x = (r_x^{\text{cycle}} \| r_x^{\text{constr}})$ and
    /// $r_y = (r_y^{\text{cycle}} \| r_y^{\text{var}})$, then computes:
    /// $$\tilde{M}(r_x, r_y) = \widetilde{eq}(r_x^{\text{cycle}}, r_y^{\text{cycle}}) \cdot \tilde{M}_{\text{local}}(r_x^{\text{constr}}, r_y^{\text{var}})$$
    pub fn evaluate_matrix_mles(&self, r_x: &[F], r_y: &[F]) -> (F, F, F) {
        let cycle_vars = self.num_cycle_vars();
        let constraint_vars = self.num_constraint_vars();
        let var_vars = self.num_var_vars();

        assert_eq!(r_x.len(), cycle_vars + constraint_vars);
        assert_eq!(r_y.len(), cycle_vars + var_vars);

        let (r_x_cycle, r_x_constraint) = r_x.split_at(cycle_vars);
        let (r_y_cycle, r_y_var) = r_y.split_at(cycle_vars);

        let cycle_eq = EqPolynomial::new(r_x_cycle.to_vec()).evaluate(r_y_cycle);
        let (a_local, b_local, c_local) = self.evaluate_local_mles(r_x_constraint, r_y_var);

        (cycle_eq * a_local, cycle_eq * b_local, cycle_eq * c_local)
    }

    /// Evaluates the combined matrix
    /// $M = \rho_A \cdot A + \rho_B \cdot B + \rho_C \cdot C$
    /// at a point and returns $M(r_x, r_y) \cdot z_{\text{eval}}$.
    ///
    /// Used by the inner sumcheck check: the verifier compares this
    /// against the final sumcheck evaluation.
    pub fn inner_check(
        &self,
        r_x: &[F],
        r_y: &[F],
        rho_a: F,
        rho_b: F,
        rho_c: F,
        witness_eval: F,
    ) -> F {
        let (a_eval, b_eval, c_eval) = self.evaluate_matrix_mles(r_x, r_y);
        let combined = rho_a * a_eval + rho_b * b_eval + rho_c * c_eval;
        combined * witness_eval
    }

    /// Evaluates `Az(r_constraint)`, `Bz(r_constraint)`, `Cz(r_constraint)`
    /// using the sparse constraint representation and per-variable witness
    /// evaluations.
    ///
    /// This is the "inner sum product" evaluation: for each constraint `k`,
    /// computes `dot(M_row_k, witness_evals)` weighted by the eq polynomial
    /// at the constraint point.
    pub fn evaluate_sparse_matvec(&self, constraint_point: &[F], witness_evals: &[F]) -> (F, F, F) {
        assert_eq!(constraint_point.len(), self.num_constraint_vars());
        assert!(witness_evals.len() >= self.num_vars);

        let eq_constraint = EqPolynomial::new(constraint_point.to_vec()).evaluations();

        let mut az = F::zero();
        let mut bz = F::zero();
        let mut cz = F::zero();

        for (k, (a_row, (b_row, c_row))) in self
            .a_sparse
            .iter()
            .zip(self.b_sparse.iter().zip(self.c_sparse.iter()))
            .enumerate()
        {
            let w = eq_constraint[k];
            if w.is_zero() {
                continue;
            }

            let mut a_dot = F::zero();
            for &(j, coeff) in a_row {
                a_dot += coeff * witness_evals[j];
            }

            let mut b_dot = F::zero();
            for &(j, coeff) in b_row {
                b_dot += coeff * witness_evals[j];
            }

            let mut c_dot = F::zero();
            for &(j, coeff) in c_row {
                c_dot += coeff * witness_evals[j];
            }

            az += w * a_dot;
            bz += w * b_dot;
            cz += w * c_dot;
        }

        (az, bz, cz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Field, Fr};
    use num_traits::{One, Zero};

    /// Creates a simple 2-constraint, 3-variable uniform key:
    /// Constraint 0: x * x = y  →  A[0] = [(1, 1)], B[0] = [(1, 1)], C[0] = [(2, 1)]
    /// Constraint 1: y * x = z  →  A[1] = [(2, 1)], B[1] = [(1, 1)], C[1] = [(3, 1)]
    ///
    /// Wire layout per cycle: [1, x, y, z] (4 vars, 0=constant)
    fn test_key(num_cycles: usize) -> UniformSpartanKey<Fr> {
        let one = Fr::from_u64(1);
        UniformSpartanKey::new(
            num_cycles,
            2,
            4,
            vec![vec![(1, one)], vec![(2, one)]], // A
            vec![vec![(1, one)], vec![(1, one)]], // B
            vec![vec![(2, one)], vec![(3, one)]], // C
        )
    }

    #[test]
    fn key_dimensions() {
        let key = test_key(4);
        assert_eq!(key.num_cycle_vars(), 2);
        assert_eq!(key.num_constraint_vars(), 1);
        assert_eq!(key.num_var_vars(), 2);
        assert_eq!(key.total_rows(), 4 * 2); // 4 cycles * 2 constraints
        assert_eq!(key.total_cols(), 4 * 4); // 4 cycles * 4 vars
        assert_eq!(key.num_row_vars(), 3); // 2 cycle + 1 constraint
        assert_eq!(key.num_col_vars(), 4); // 2 cycle + 2 var
    }

    #[test]
    fn local_mle_at_boolean_point() {
        let key = test_key(1);

        // EqPolynomial indexing: index = x0 * 2^(n-1) + ... + x_{n-1} * 2^0
        // (MSB-first: first coordinate in point → high bit of index)
        //
        // For 2-var eq: eq_var[idx] where idx = x0*2 + x1:
        //   eq_var[0] = (1-r0)(1-r1)
        //   eq_var[1] = (1-r0)*r1
        //   eq_var[2] = r0*(1-r1)
        //   eq_var[3] = r0*r1

        // Evaluate at constraint=0, var_point=[0,0]
        // eq_constraint[0] = 1, only constraint 0 contributes: A[0] = [(1, 1)]
        // eq_var[1] at [0,0] = (1-0)*0 = 0 → A_local = 0
        let (a, _b, _c) = key.evaluate_local_mles(&[Fr::zero()], &[Fr::zero(), Fr::zero()]);
        assert_eq!(a, Fr::zero());

        // To select variable index 1: need eq_var[1] = 1
        // eq_var[1] = (1-r0)*r1, so r0=0, r1=1 → var_point = [0, 1]
        let (a, _b, _c) = key.evaluate_local_mles(&[Fr::zero()], &[Fr::zero(), Fr::one()]);
        assert_eq!(a, Fr::from_u64(1));

        // Constraint 0, variable 2 (y): A[0] doesn't reference var 2, so expect 0.
        // Var index 2: eq_var[2] = r0*(1-r1), so r0=1, r1=0 → var_point = [1, 0]
        let (a, _b, c) = key.evaluate_local_mles(&[Fr::zero()], &[Fr::one(), Fr::zero()]);
        assert_eq!(a, Fr::zero());
        // C[0] = [(2, 1)] → C_local should be eq_var[2] = 1
        assert_eq!(c, Fr::from_u64(1));
    }

    #[test]
    fn sparse_matvec_matches_dense() {
        let key = test_key(1);

        // Witness per cycle: [1, x=3, y=9, z=27]
        // Constraint 0: A*z = x = 3, B*z = x = 3, C*z = y = 9
        // Constraint 1: A*z = y = 9, B*z = x = 3, C*z = z = 27
        let witness_evals = [
            Fr::from_u64(1),
            Fr::from_u64(3),
            Fr::from_u64(9),
            Fr::from_u64(27),
        ];

        // At constraint=0: should return (3, 3, 9)
        let (az, bz, cz) = key.evaluate_sparse_matvec(&[Fr::zero()], &witness_evals);
        assert_eq!(az, Fr::from_u64(3));
        assert_eq!(bz, Fr::from_u64(3));
        assert_eq!(cz, Fr::from_u64(9));
        // Satisfaction check: az * bz = 9 = cz
        assert_eq!(az * bz, cz);

        // At constraint=1: should return (9, 3, 27)
        let (az, bz, cz) = key.evaluate_sparse_matvec(&[Fr::one()], &witness_evals);
        assert_eq!(az, Fr::from_u64(9));
        assert_eq!(bz, Fr::from_u64(3));
        assert_eq!(cz, Fr::from_u64(27));
        assert_eq!(az * bz, cz);
    }

    #[test]
    fn evaluate_matrix_mles_factored() {
        let key = test_key(2);

        // r_x = [r_cycle_0, r_constraint_0]  (1 cycle var + 1 constraint var)
        // r_y = [r_cycle_0, r_var_0, r_var_1] (1 cycle var + 2 var vars)
        let r_x = [Fr::from_u64(5), Fr::from_u64(7)];
        let r_y = [Fr::from_u64(5), Fr::from_u64(11), Fr::from_u64(13)];

        let (a_eval, b_eval, c_eval) = key.evaluate_matrix_mles(&r_x, &r_y);

        // Verify factorization: M(r_x, r_y) = eq(r_cycle_x, r_cycle_y) * M_local(r_constr, r_var)
        let cycle_eq = EqPolynomial::new(vec![Fr::from_u64(5)]).evaluate(&[Fr::from_u64(5)]);
        let (a_local, b_local, c_local) =
            key.evaluate_local_mles(&[Fr::from_u64(7)], &[Fr::from_u64(11), Fr::from_u64(13)]);

        assert_eq!(a_eval, cycle_eq * a_local);
        assert_eq!(b_eval, cycle_eq * b_local);
        assert_eq!(c_eval, cycle_eq * c_local);
    }

    #[test]
    fn inner_check_computation() {
        let key = test_key(2);
        let r_x = [Fr::from_u64(3), Fr::from_u64(7)];
        let r_y = [Fr::from_u64(3), Fr::from_u64(11), Fr::from_u64(13)];

        let rho_a = Fr::from_u64(2);
        let rho_b = Fr::from_u64(5);
        let rho_c = Fr::from_u64(9);
        let witness_eval = Fr::from_u64(42);

        let result = key.inner_check(&r_x, &r_y, rho_a, rho_b, rho_c, witness_eval);

        let (a, b, c) = key.evaluate_matrix_mles(&r_x, &r_y);
        let expected = (rho_a * a + rho_b * b + rho_c * c) * witness_eval;
        assert_eq!(result, expected);
    }
}
