//! Preprocessed uniform R1CS key.
//!
//! Combines per-cycle [`ConstraintMatrices`] with a runtime-determined
//! `num_cycles` to form the complete R1CS instance. Provides matrix MLE
//! evaluation for the verifier and combined-row materialization for the
//! inner sumcheck.
//!
//! # Uniform structure
//!
//! The full R1CS has `num_cycles × num_constraints_padded` rows and
//! `num_cycles × num_vars_padded` columns. Row `i` decomposes as
//! `(cycle, constraint) = (i / K_pad, i % K_pad)`, column `j` as
//! `(cycle, variable) = (j / V_pad, j % V_pad)`.
//!
//! Matrix MLE factors as:
//! $$\tilde{M}(r_x, r_y) = \widetilde{eq}(r_x^{cyc}, r_y^{cyc}) \cdot \tilde{M}_{local}(r_x^{con}, r_y^{var})$$

use jolt_field::Field;
use jolt_poly::EqPolynomial;
use serde::{Deserialize, Serialize};

use crate::constraint::ConstraintMatrices;

/// Preprocessed uniform R1CS key for runtime consumption.
///
/// Stores per-cycle sparse constraint matrices and dimensional metadata.
/// All evaluation methods exploit the uniform (repeated-constraint) structure.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct R1csKey<F: Field> {
    pub matrices: ConstraintMatrices<F>,
    pub num_cycles: usize,
    pub num_constraints_padded: usize,
    pub num_vars_padded: usize,
}

impl<F: Field> R1csKey<F> {
    /// Creates a new key from per-cycle constraints and cycle count.
    ///
    /// # Panics
    ///
    /// Panics if `num_cycles` is not a power of two.
    pub fn new(matrices: ConstraintMatrices<F>, num_cycles: usize) -> Self {
        assert!(
            num_cycles.is_power_of_two(),
            "num_cycles must be a power of two, got {num_cycles}"
        );
        Self {
            num_constraints_padded: matrices.num_constraints.next_power_of_two(),
            num_vars_padded: matrices.num_vars.next_power_of_two(),
            matrices,
            num_cycles,
        }
    }

    #[inline]
    pub fn num_cycle_vars(&self) -> usize {
        self.num_cycles.trailing_zeros() as usize
    }

    #[inline]
    pub fn num_constraint_vars(&self) -> usize {
        self.num_constraints_padded.trailing_zeros() as usize
    }

    #[inline]
    pub fn num_var_vars(&self) -> usize {
        self.num_vars_padded.trailing_zeros() as usize
    }

    #[inline]
    pub fn total_rows(&self) -> usize {
        self.num_cycles * self.num_constraints_padded
    }

    #[inline]
    pub fn total_cols(&self) -> usize {
        self.num_cycles * self.num_vars_padded
    }

    #[inline]
    pub fn num_row_vars(&self) -> usize {
        self.num_cycle_vars() + self.num_constraint_vars()
    }

    #[inline]
    pub fn num_col_vars(&self) -> usize {
        self.num_cycle_vars() + self.num_var_vars()
    }

    /// Evaluates the per-cycle (local) matrix MLEs at `(constraint_point, var_point)`.
    ///
    /// $$\tilde{M}_{local}(r_c, r_v) = \sum_k \widetilde{eq}(k, r_c) \sum_{(j, \alpha)} \alpha \cdot \widetilde{eq}(j, r_v)$$
    pub fn evaluate_local_mles(&self, constraint_point: &[F], var_point: &[F]) -> (F, F, F) {
        debug_assert_eq!(constraint_point.len(), self.num_constraint_vars());
        debug_assert_eq!(var_point.len(), self.num_var_vars());

        let eq_con = EqPolynomial::new(constraint_point.to_vec()).evaluations();
        let eq_var = EqPolynomial::new(var_point.to_vec()).evaluations();

        let mut a_eval = F::zero();
        let mut b_eval = F::zero();
        let mut c_eval = F::zero();

        for (k, ((a_row, b_row), c_row)) in self
            .matrices
            .a
            .iter()
            .zip(&self.matrices.b)
            .zip(&self.matrices.c)
            .enumerate()
        {
            let w = eq_con[k];
            if w.is_zero() {
                continue;
            }

            let mut a_row_eval = F::zero();
            for &(j, coeff) in a_row {
                a_row_eval += coeff * eq_var[j];
            }
            a_eval += w * a_row_eval;

            let mut b_row_eval = F::zero();
            for &(j, coeff) in b_row {
                b_row_eval += coeff * eq_var[j];
            }
            b_eval += w * b_row_eval;

            let mut c_row_eval = F::zero();
            for &(j, coeff) in c_row {
                c_row_eval += coeff * eq_var[j];
            }
            c_eval += w * c_row_eval;
        }

        (a_eval, b_eval, c_eval)
    }

    /// Evaluates the full matrix MLEs at `(r_x, r_y)`.
    ///
    /// Splits `r_x = (r_cycle ‖ r_constraint)` and `r_y = (r_cycle ‖ r_var)`,
    /// then uses the factorization:
    /// $$\tilde{M}(r_x, r_y) = \widetilde{eq}(r_x^{cyc}, r_y^{cyc}) \cdot \tilde{M}_{local}(r_x^{con}, r_y^{var})$$
    pub fn evaluate_matrix_mles(&self, r_x: &[F], r_y: &[F]) -> (F, F, F) {
        let cv = self.num_cycle_vars();
        debug_assert_eq!(r_x.len(), cv + self.num_constraint_vars());
        debug_assert_eq!(r_y.len(), cv + self.num_var_vars());

        let (rx_cycle, rx_con) = r_x.split_at(cv);
        let (ry_cycle, ry_var) = r_y.split_at(cv);

        let cycle_eq = EqPolynomial::new(rx_cycle.to_vec()).evaluate(ry_cycle);
        let (a_local, b_local, c_local) = self.evaluate_local_mles(rx_con, ry_var);

        (cycle_eq * a_local, cycle_eq * b_local, cycle_eq * c_local)
    }

    /// Evaluates Az(r_con), Bz(r_con), Cz(r_con) from per-variable witness evaluations.
    ///
    /// For each constraint k, computes `dot(M_row_k, witness_evals)` weighted
    /// by the eq polynomial at the constraint point.
    pub fn evaluate_sparse_matvec(&self, constraint_point: &[F], witness_evals: &[F]) -> (F, F, F) {
        debug_assert_eq!(constraint_point.len(), self.num_constraint_vars());
        debug_assert!(witness_evals.len() >= self.matrices.num_vars);

        let eq_con = EqPolynomial::new(constraint_point.to_vec()).evaluations();

        let mut az = F::zero();
        let mut bz = F::zero();
        let mut cz = F::zero();

        for (k, ((a_row, b_row), c_row)) in self
            .matrices
            .a
            .iter()
            .zip(&self.matrices.b)
            .zip(&self.matrices.c)
            .enumerate()
        {
            let w = eq_con[k];
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

    /// Materializes the combined row polynomial for the inner sumcheck.
    ///
    /// $$M(r_x, y) = \rho_A A(r_x, y) + \rho_B B(r_x, y) + \rho_C C(r_x, y)$$
    ///
    /// Returns a dense polynomial over `total_cols() = num_cycles · num_vars_padded`
    /// column indices (already a power of two by construction).
    pub fn combined_row(&self, r_x: &[F], rho_a: F, rho_b: F, rho_c: F) -> Vec<F> {
        let cv = self.num_cycle_vars();
        let (rx_cycle, rx_con) = r_x.split_at(cv);

        let eq_con = EqPolynomial::new(rx_con.to_vec()).evaluations();
        let eq_cycle = EqPolynomial::new(rx_cycle.to_vec()).evaluations();

        // Build combined local row: M_local(r_con, v) for each variable v
        let mut local_row = vec![F::zero(); self.num_vars_padded];
        for (k, ((a_row, b_row), c_row)) in self
            .matrices
            .a
            .iter()
            .zip(&self.matrices.b)
            .zip(&self.matrices.c)
            .enumerate()
        {
            let w = eq_con[k];
            if w.is_zero() {
                continue;
            }
            for &(j, coeff) in a_row {
                local_row[j] += w * rho_a * coeff;
            }
            for &(j, coeff) in b_row {
                local_row[j] += w * rho_b * coeff;
            }
            for &(j, coeff) in c_row {
                local_row[j] += w * rho_c * coeff;
            }
        }

        let v_pad = self.num_vars_padded;
        let mut combined = vec![F::zero(); self.total_cols()];

        let fill_cycle = |(c, chunk): (usize, &mut [F])| {
            let eq_c = eq_cycle[c];
            if eq_c.is_zero() {
                return;
            }
            for (v, &local_val) in local_row.iter().enumerate() {
                if !local_val.is_zero() {
                    chunk[v] = eq_c * local_val;
                }
            }
        };

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            combined
                .par_chunks_mut(v_pad)
                .enumerate()
                .for_each(fill_cycle);
        }
        #[cfg(not(feature = "parallel"))]
        {
            combined.chunks_mut(v_pad).enumerate().for_each(fill_cycle);
        }

        combined
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::ConstraintMatrices;
    use jolt_field::{Field, Fr};
    use num_traits::{One, Zero};

    /// x * x = y, y * x = z — 2 constraints, 4 vars [1, x, y, z]
    fn test_matrices() -> ConstraintMatrices<Fr> {
        let one = Fr::one();
        ConstraintMatrices::new(
            2,
            4,
            vec![vec![(1, one)], vec![(2, one)]],
            vec![vec![(1, one)], vec![(1, one)]],
            vec![vec![(2, one)], vec![(3, one)]],
        )
    }

    fn test_key(num_cycles: usize) -> R1csKey<Fr> {
        R1csKey::new(test_matrices(), num_cycles)
    }

    #[test]
    fn dimensions() {
        let key = test_key(4);
        assert_eq!(key.num_cycle_vars(), 2);
        assert_eq!(key.num_constraint_vars(), 1);
        assert_eq!(key.num_var_vars(), 2);
        assert_eq!(key.total_rows(), 8);
        assert_eq!(key.total_cols(), 16);
        assert_eq!(key.num_row_vars(), 3);
        assert_eq!(key.num_col_vars(), 4);
    }

    #[test]
    fn local_mle_boolean_points() {
        let key = test_key(1);

        // Constraint 0, var 0 → A entry is at (1, 1), so A(0, [0,0]) = 0
        let (a, _, _) = key.evaluate_local_mles(&[Fr::zero()], &[Fr::zero(), Fr::zero()]);
        assert!(a.is_zero());

        // Constraint 0, var 1 → A has (1, 1), eq([0,1], [0,1]) = 1, so A(0, [0,1]) = 1
        let (a, _, _) = key.evaluate_local_mles(&[Fr::zero()], &[Fr::zero(), Fr::one()]);
        assert_eq!(a, Fr::one());
    }

    #[test]
    fn matrix_mle_factorizes() {
        let key = test_key(2);

        let r_x = [Fr::from_u64(5), Fr::from_u64(7)];
        let r_y = [Fr::from_u64(5), Fr::from_u64(11), Fr::from_u64(13)];

        let (a_eval, b_eval, c_eval) = key.evaluate_matrix_mles(&r_x, &r_y);

        let cycle_eq = EqPolynomial::new(vec![Fr::from_u64(5)]).evaluate(&[Fr::from_u64(5)]);
        let (a_local, b_local, c_local) =
            key.evaluate_local_mles(&[Fr::from_u64(7)], &[Fr::from_u64(11), Fr::from_u64(13)]);

        assert_eq!(a_eval, cycle_eq * a_local);
        assert_eq!(b_eval, cycle_eq * b_local);
        assert_eq!(c_eval, cycle_eq * c_local);
    }

    #[test]
    fn sparse_matvec_satisfies() {
        let key = test_key(1);
        // Witness: [1, 3, 9, 27]
        let w = [
            Fr::from_u64(1),
            Fr::from_u64(3),
            Fr::from_u64(9),
            Fr::from_u64(27),
        ];

        // Constraint 0: 3*3 = 9 ✓
        let (az, bz, cz) = key.evaluate_sparse_matvec(&[Fr::zero()], &w);
        assert_eq!(az, Fr::from_u64(3));
        assert_eq!(bz, Fr::from_u64(3));
        assert_eq!(cz, Fr::from_u64(9));
        assert_eq!(az * bz, cz);

        // Constraint 1: 9*3 = 27 ✓
        let (az, bz, cz) = key.evaluate_sparse_matvec(&[Fr::one()], &w);
        assert_eq!(az, Fr::from_u64(9));
        assert_eq!(bz, Fr::from_u64(3));
        assert_eq!(cz, Fr::from_u64(27));
        assert_eq!(az * bz, cz);
    }

    #[test]
    fn combined_row_consistency() {
        use rand_chacha::ChaCha20Rng;
        use rand_core::SeedableRng;

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let key = test_key(4);

        let r_x: Vec<Fr> = (0..key.num_row_vars())
            .map(|_| Fr::random(&mut rng))
            .collect();
        let r_y: Vec<Fr> = (0..key.num_col_vars())
            .map(|_| Fr::random(&mut rng))
            .collect();
        let rho_a = Fr::random(&mut rng);
        let rho_b = Fr::random(&mut rng);
        let rho_c = Fr::random(&mut rng);

        let combined = key.combined_row(&r_x, rho_a, rho_b, rho_c);

        // Evaluate the dense combined row polynomial at r_y
        let eq_y = EqPolynomial::new(r_y.clone()).evaluations();
        let dense_eval: Fr = combined.iter().zip(eq_y.iter()).map(|(&c, &e)| c * e).sum();

        // Compare with direct matrix MLE evaluation
        let (a_eval, b_eval, c_eval) = key.evaluate_matrix_mles(&r_x, &r_y);
        let mle_eval = rho_a * a_eval + rho_b * b_eval + rho_c * c_eval;

        assert_eq!(dense_eval, mle_eval);
    }

    #[test]
    fn brute_force_eq_factorization() {
        use rand_chacha::ChaCha20Rng;
        use rand_core::SeedableRng;

        let mut rng = ChaCha20Rng::seed_from_u64(99);

        // Trivial key: A[k] = [(0, 1)], B/C empty
        let one = Fr::one();
        let m = ConstraintMatrices::new(
            24,
            41,
            (0..24).map(|_| vec![(0, one)]).collect(),
            (0..24).map(|_| vec![]).collect(),
            (0..24).map(|_| vec![]).collect(),
        );
        let key = R1csKey::new(m, 32);

        let r_x: Vec<Fr> = (0..key.num_row_vars())
            .map(|_| Fr::random(&mut rng))
            .collect();
        let r_y: Vec<Fr> = (0..key.num_col_vars())
            .map(|_| Fr::random(&mut rng))
            .collect();

        let cv = key.num_cycle_vars();
        let (rx_cycle, rx_con) = r_x.split_at(cv);
        let (ry_cycle, _) = r_y.split_at(cv);

        // Check eq factorization: eq_full[c*K+k] == eq_cycle[c] * eq_sub[k]
        let eq_row = EqPolynomial::new(r_x.clone()).evaluations();
        let eq_x_cycle = EqPolynomial::new(rx_cycle.to_vec()).evaluations();
        let eq_con = EqPolynomial::new(rx_con.to_vec()).evaluations();

        let k_pad = key.num_constraints_padded;
        for (c, &eq_xc) in eq_x_cycle.iter().enumerate().take(32) {
            for (k, &eq_ck) in eq_con.iter().enumerate().take(24) {
                assert_eq!(eq_row[c * k_pad + k], eq_xc * eq_ck);
            }
        }

        // Check key MLE vs brute force
        let eq_col = EqPolynomial::new(r_y.clone()).evaluations();
        let (a_local, _, _) = key.evaluate_local_mles(rx_con, &r_y[cv..]);
        let cycle_eq = EqPolynomial::new(rx_cycle.to_vec()).evaluate(ry_cycle);
        let key_a = cycle_eq * a_local;

        let v_pad = key.num_vars_padded;
        let mut brute_a = Fr::zero();
        for c in 0..32usize {
            for k in 0..24usize {
                brute_a += eq_row[c * k_pad + k] * eq_col[c * v_pad];
            }
        }
        assert_eq!(brute_a, key_a);
    }
}
