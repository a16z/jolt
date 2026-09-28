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

use jolt_field::JoltField;
use jolt_poly::EqPolynomial;
use serde::{Deserialize, Serialize};

use crate::constraint::ConstraintMatrices;

/// Preprocessed uniform R1CS key for runtime consumption.
///
/// Stores per-cycle sparse constraint matrices and dimensional metadata.
/// All evaluation methods exploit the uniform (repeated-constraint) structure.
///
/// Dimensional invariants (power-of-two `num_cycles`, padded dimensions
/// re-derivable from the matrices, non-overflowing `total_rows`/`total_cols`
/// products) are established at construction. Deserialization routes through
/// `RawR1csKey` and revalidates them; malformed input is rejected before
/// any consumer sees the struct.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(
    bound(serialize = "F: Serialize", deserialize = "F: for<'a> Deserialize<'a>"),
    try_from = "RawR1csKey<F>"
)]
pub struct R1csKey<F: JoltField> {
    pub(crate) matrices: ConstraintMatrices<F>,
    pub(crate) num_cycles: usize,
    pub(crate) num_constraints_padded: usize,
    pub(crate) num_vars_padded: usize,
}

/// Deserialization helper; never exposed directly.
#[derive(Deserialize)]
#[serde(bound(deserialize = "F: for<'a> Deserialize<'a>"))]
struct RawR1csKey<F: JoltField> {
    matrices: ConstraintMatrices<F>,
    num_cycles: usize,
    num_constraints_padded: usize,
    num_vars_padded: usize,
}

impl<F: JoltField> TryFrom<RawR1csKey<F>> for R1csKey<F> {
    type Error = String;

    fn try_from(raw: RawR1csKey<F>) -> Result<Self, Self::Error> {
        let RawR1csKey {
            matrices,
            num_cycles,
            num_constraints_padded,
            num_vars_padded,
        } = raw;
        check_key_invariants(
            &matrices,
            num_cycles,
            num_constraints_padded,
            num_vars_padded,
        )?;
        Ok(Self {
            matrices,
            num_cycles,
            num_constraints_padded,
            num_vars_padded,
        })
    }
}

fn check_key_invariants<F: JoltField>(
    matrices: &ConstraintMatrices<F>,
    num_cycles: usize,
    num_constraints_padded: usize,
    num_vars_padded: usize,
) -> Result<(), String> {
    if !num_cycles.is_power_of_two() {
        return Err(format!(
            "num_cycles must be a power of two, got {num_cycles}"
        ));
    }
    let expected_num_constraints_padded = matrices
        .num_constraints
        .checked_next_power_of_two()
        .ok_or_else(|| {
            format!(
                "next power of two overflows usize for {} constraints",
                matrices.num_constraints
            )
        })?;
    if num_constraints_padded != expected_num_constraints_padded {
        return Err(format!(
            "num_constraints_padded = {num_constraints_padded}, expected {} (next power of two of {} constraints)",
            expected_num_constraints_padded,
            matrices.num_constraints,
        ));
    }
    let expected_num_vars_padded =
        matrices
            .num_vars
            .checked_next_power_of_two()
            .ok_or_else(|| {
                format!(
                    "next power of two overflows usize for {} variables",
                    matrices.num_vars
                )
            })?;
    if num_vars_padded != expected_num_vars_padded {
        return Err(format!(
            "num_vars_padded = {num_vars_padded}, expected {} (next power of two of {} variables)",
            expected_num_vars_padded, matrices.num_vars,
        ));
    }
    // Guarantees total_rows()/total_cols() cannot overflow downstream.
    if num_cycles.checked_mul(num_constraints_padded).is_none() {
        return Err(format!(
            "total row count overflows usize: {num_cycles} cycles * {num_constraints_padded} padded constraints"
        ));
    }
    if num_cycles.checked_mul(num_vars_padded).is_none() {
        return Err(format!(
            "total column count overflows usize: {num_cycles} cycles * {num_vars_padded} padded variables"
        ));
    }
    Ok(())
}

impl<F: JoltField> R1csKey<F> {
    /// Creates a new key from per-cycle constraints and cycle count.
    ///
    /// # Panics
    ///
    /// Panics if `num_cycles` is not a power of two, or if a padded dimension
    /// or total row/column count overflows `usize`.
    #[expect(
        clippy::expect_used,
        reason = "constructor invariant violation indicates a programmer error"
    )]
    pub fn new(matrices: ConstraintMatrices<F>, num_cycles: usize) -> Self {
        let num_constraints_padded = matrices
            .num_constraints
            .checked_next_power_of_two()
            .expect("R1csKey constraint dimension exceeds the maximum power of two");
        let num_vars_padded = matrices
            .num_vars
            .checked_next_power_of_two()
            .expect("R1csKey variable dimension exceeds the maximum power of two");
        check_key_invariants(
            &matrices,
            num_cycles,
            num_constraints_padded,
            num_vars_padded,
        )
        .expect("R1csKey::new invariant violated");
        Self {
            matrices,
            num_cycles,
            num_constraints_padded,
            num_vars_padded,
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
        assert_eq!(
            constraint_point.len(),
            self.num_constraint_vars(),
            "constraint point dimension mismatch"
        );
        assert_eq!(
            var_point.len(),
            self.num_var_vars(),
            "variable point dimension mismatch"
        );

        let eq_con = EqPolynomial::new(constraint_point.to_vec()).evaluations();
        let eq_var = EqPolynomial::new(var_point.to_vec()).evaluations();

        let mut a_eval = F::zero();
        let mut b_eval = F::zero();
        let mut c_eval = F::zero();

        for (((a_row, b_row), c_row), &w) in self
            .matrices
            .a
            .iter()
            .zip(&self.matrices.b)
            .zip(&self.matrices.c)
            .zip(&eq_con)
        {
            if w.is_zero() {
                continue;
            }

            a_eval += w * sparse_row_dot(a_row, &eq_var);
            b_eval += w * sparse_row_dot(b_row, &eq_var);
            c_eval += w * sparse_row_dot(c_row, &eq_var);
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
        assert_eq!(r_x.len(), self.num_row_vars(), "r_x dimension mismatch");
        assert_eq!(r_y.len(), self.num_col_vars(), "r_y dimension mismatch");

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
        assert_eq!(
            constraint_point.len(),
            self.num_constraint_vars(),
            "constraint point dimension mismatch"
        );
        assert!(
            witness_evals.len() >= self.matrices.num_vars,
            "witness evals must cover all {} variables, got {}",
            self.matrices.num_vars,
            witness_evals.len()
        );

        let eq_con = EqPolynomial::new(constraint_point.to_vec()).evaluations();

        let mut az = F::zero();
        let mut bz = F::zero();
        let mut cz = F::zero();

        for (((a_row, b_row), c_row), &w) in self
            .matrices
            .a
            .iter()
            .zip(&self.matrices.b)
            .zip(&self.matrices.c)
            .zip(&eq_con)
        {
            if w.is_zero() {
                continue;
            }

            az += w * sparse_row_dot(a_row, witness_evals);
            bz += w * sparse_row_dot(b_row, witness_evals);
            cz += w * sparse_row_dot(c_row, witness_evals);
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
        assert_eq!(r_x.len(), self.num_row_vars(), "r_x dimension mismatch");
        let (rx_cycle, rx_con) = r_x.split_at(cv);

        let eq_con = EqPolynomial::new(rx_con.to_vec()).evaluations();
        let eq_cycle = EqPolynomial::new(rx_cycle.to_vec()).evaluations();

        // Build combined local row: M_local(r_con, v) for each variable v
        let mut local_row = vec![F::zero(); self.num_vars_padded];
        #[expect(
            clippy::indexing_slicing,
            reason = "column indices are below num_vars <= num_vars_padded = local_row.len() by the ConstraintMatrices invariant"
        )]
        for (((a_row, b_row), c_row), &w) in self
            .matrices
            .a
            .iter()
            .zip(&self.matrices.b)
            .zip(&self.matrices.c)
            .zip(&eq_con)
        {
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

        let fill_cycle = |(chunk, eq_c): (&mut [F], &F)| {
            if eq_c.is_zero() {
                return;
            }
            for (slot, &local_val) in chunk.iter_mut().zip(&local_row) {
                if !local_val.is_zero() {
                    *slot = *eq_c * local_val;
                }
            }
        };

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            combined
                .par_chunks_mut(v_pad)
                .zip(eq_cycle.par_iter())
                .for_each(fill_cycle);
        }
        #[cfg(not(feature = "parallel"))]
        {
            combined
                .chunks_mut(v_pad)
                .zip(eq_cycle.iter())
                .for_each(fill_cycle);
        }

        combined
    }
}

/// Dot product of a sparse row with a dense evaluation table.
///
/// Callers guarantee the table covers `num_vars` entries; every column index
/// is below `num_vars` by the [`ConstraintMatrices`] invariant.
#[expect(
    clippy::indexing_slicing,
    reason = "column indices are below num_vars by the ConstraintMatrices invariant and tables cover num_vars"
)]
fn sparse_row_dot<F: JoltField>(row: &[(usize, F)], table: &[F]) -> F {
    let mut acc = F::zero();
    for &(j, coeff) in row {
        acc += coeff * table[j];
    }
    acc
}

#[cfg(test)]
#[expect(clippy::indexing_slicing, reason = "tests index fixture data")]
mod tests {
    use super::*;
    use crate::constraint::ConstraintMatrices;
    use jolt_field::{Field, Fr, Ring};
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
    #[should_panic(expected = "power of two")]
    fn new_rejects_non_power_of_two_cycles() {
        let _ = R1csKey::new(test_matrices(), 3);
    }

    #[test]
    #[should_panic(expected = "overflows usize")]
    fn new_rejects_total_size_overflow() {
        // 2 constraints pad to 2; (1 << 63) * 2 overflows usize.
        let _ = R1csKey::new(test_matrices(), 1 << 63);
    }

    fn raw_key(
        num_cycles: usize,
        num_constraints_padded: usize,
        num_vars_padded: usize,
    ) -> RawR1csKey<Fr> {
        RawR1csKey {
            matrices: test_matrices(),
            num_cycles,
            num_constraints_padded,
            num_vars_padded,
        }
    }

    #[test]
    #[expect(clippy::expect_used, reason = "test should fail loudly")]
    fn try_from_accepts_consistent_dimensions() {
        let key = R1csKey::try_from(raw_key(4, 2, 4)).expect("consistent raw key");
        assert_eq!(key.num_cycles, 4);
        assert_eq!(key.num_constraints_padded, 2);
        assert_eq!(key.num_vars_padded, 4);
    }

    #[test]
    fn try_from_rejects_dimensional_invariant_violations() {
        // Zero num_cycles would make num_cycle_vars() return 64.
        assert!(R1csKey::try_from(raw_key(0, 2, 4)).is_err());
        assert!(R1csKey::try_from(raw_key(3, 2, 4)).is_err());
        // Padded dimensions inconsistent with the embedded matrices.
        assert!(R1csKey::try_from(raw_key(4, 1, 4)).is_err());
        assert!(R1csKey::try_from(raw_key(4, 4, 4)).is_err());
        assert!(R1csKey::try_from(raw_key(4, 2, 2)).is_err());
        // total_rows()/total_cols() products must not overflow.
        assert!(R1csKey::try_from(raw_key(1 << 63, 2, 4)).is_err());
    }

    #[test]
    fn try_from_rejects_padded_dimension_overflow() {
        let matrices = ConstraintMatrices::<Fr>::new(0, usize::MAX, vec![], vec![], vec![]);
        let raw = RawR1csKey {
            matrices,
            num_cycles: 1,
            num_constraints_padded: 1,
            num_vars_padded: 0,
        };

        assert!(R1csKey::try_from(raw).is_err());
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
    #[should_panic(expected = "r_x dimension mismatch")]
    fn matrix_mles_reject_short_r_x() {
        let key = test_key(4);
        let r_x = vec![Fr::one(); key.num_row_vars() - 1];
        let r_y = vec![Fr::one(); key.num_col_vars()];
        let _ = key.evaluate_matrix_mles(&r_x, &r_y);
    }

    #[test]
    #[should_panic(expected = "r_y dimension mismatch")]
    fn matrix_mles_reject_long_r_y() {
        let key = test_key(4);
        let r_x = vec![Fr::one(); key.num_row_vars()];
        let r_y = vec![Fr::one(); key.num_col_vars() + 1];
        let _ = key.evaluate_matrix_mles(&r_x, &r_y);
    }

    #[test]
    #[should_panic(expected = "constraint point dimension mismatch")]
    fn sparse_matvec_rejects_wrong_constraint_point() {
        let key = test_key(1);
        let w = vec![Fr::one(); 4];
        let _ = key.evaluate_sparse_matvec(&[Fr::zero(), Fr::zero()], &w);
    }

    #[test]
    #[should_panic(expected = "witness evals must cover")]
    fn sparse_matvec_rejects_short_witness_evals() {
        let key = test_key(1);
        let w = vec![Fr::one(); key.matrices.num_vars - 1];
        let _ = key.evaluate_sparse_matvec(&[Fr::zero()], &w);
    }

    #[test]
    #[should_panic(expected = "r_x dimension mismatch")]
    fn combined_row_rejects_wrong_r_x() {
        let key = test_key(4);
        let r_x = vec![Fr::one(); key.num_cycle_vars()];
        let _ = key.combined_row(&r_x, Fr::one(), Fr::one(), Fr::one());
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
