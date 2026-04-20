//! R1CS polynomial materialization for the prover runtime.
//!
//! [`R1csSource`] wraps an [`R1csKey`] and a witness slice, computing
//! R1CS-derived polynomials (Az, Bz, Cz, combined row) on demand.
//!
//! Used internally by `ProverData` — not a standalone `BufferProvider`.

use jolt_field::Field;

use crate::column::R1csColumn;
use crate::key::R1csKey;

/// Spartan challenge state needed for combined row materialization.
#[derive(Clone, Debug)]
pub struct SpartanChallenges<F> {
    pub rho_a: F,
    pub rho_b: F,
    pub rho_c: F,
    /// Outer sumcheck challenge point (row variables).
    pub r_x: Vec<F>,
}

/// Computes R1CS-derived polynomials from constraint matrices and witness.
///
/// Handles `PolySource::R1cs(column)`: Az, Bz, Cz (sparse matvec),
/// CombinedRow (linear combination with Spartan challenges), and
/// Variable(i) column extraction from the per-cycle witness vector.
pub struct R1csSource<'a, F: Field> {
    key: &'a R1csKey<F>,
    witness: &'a [F],
    challenges: Option<SpartanChallenges<F>>,
}

impl<'a, F: Field> R1csSource<'a, F> {
    pub fn new(key: &'a R1csKey<F>, witness: &'a [F]) -> Self {
        Self {
            key,
            witness,
            challenges: None,
        }
    }

    /// Sets the Spartan challenges needed for combined row materialization.
    ///
    /// Must be called before computing a `CombinedRow` polynomial.
    pub fn set_challenges(&mut self, challenges: SpartanChallenges<F>) {
        self.challenges = Some(challenges);
    }

    /// Sparse matrix-vector product for one R1CS matrix.
    ///
    /// For each cycle c and constraint k:
    /// `Mz[c * K_pad + k] = Σ_v M_local[k][v] * witness[c * V_pad + v]`
    fn compute_matvec(&self, matrix: &[Vec<(usize, F)>]) -> Vec<F> {
        let k_pad = self.key.num_constraints_padded;
        let v_pad = self.key.num_vars_padded;
        let total = self.key.num_cycles * k_pad;
        let mut result = vec![F::zero(); total];

        for c in 0..self.key.num_cycles {
            let w_base = c * v_pad;
            let r_base = c * k_pad;
            for (k, row) in matrix.iter().enumerate() {
                let mut acc = F::zero();
                for &(j, coeff) in row {
                    acc += coeff * self.witness[w_base + j];
                }
                result[r_base + k] = acc;
            }
        }

        result
    }

    /// Compute an R1CS-derived polynomial by column.
    #[expect(
        clippy::expect_used,
        reason = "CombinedRow requires prior set_challenges() — enforced as an API contract"
    )]
    pub fn compute(&self, column: R1csColumn) -> Vec<F> {
        match column {
            R1csColumn::Az => {
                let _s = tracing::info_span!("r1cs::Az").entered();
                self.compute_matvec(&self.key.matrices.a)
            }
            R1csColumn::Bz => {
                let _s = tracing::info_span!("r1cs::Bz").entered();
                self.compute_matvec(&self.key.matrices.b)
            }
            R1csColumn::Cz => {
                let _s = tracing::info_span!("r1cs::Cz").entered();
                self.compute_matvec(&self.key.matrices.c)
            }
            R1csColumn::CombinedRow => {
                let _s = tracing::info_span!("r1cs::CombinedRow").entered();
                let ch = self
                    .challenges
                    .as_ref()
                    .expect("set_challenges() must be called before computing CombinedRow");
                let total_cols_padded = self.key.total_cols().next_power_of_two();
                self.key
                    .combined_row(&ch.r_x, ch.rho_a, ch.rho_b, ch.rho_c, total_cols_padded)
            }
            R1csColumn::Variable(var_idx) => {
                let _s = tracing::info_span!("r1cs::Variable").entered();
                let v_pad = self.key.num_vars_padded;
                (0..self.key.num_cycles)
                    .map(|c| self.witness[c * v_pad + var_idx])
                    .collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::ConstraintMatrices;
    use jolt_field::{Field, Fr};
    use num_traits::{One, Zero};

    #[test]
    fn compute_az_matches_manual() {
        let one = Fr::one();
        let m = ConstraintMatrices::new(
            1,
            3,
            vec![vec![(1, one)]],
            vec![vec![(1, one)]],
            vec![vec![(2, one)]],
        );
        let key = R1csKey::new(m, 2);

        let v_pad = key.num_vars_padded;
        let mut witness = vec![Fr::zero(); 2 * v_pad];
        witness[0] = Fr::one();
        witness[1] = Fr::from_u64(3);
        witness[2] = Fr::from_u64(9);
        witness[v_pad] = Fr::one();
        witness[v_pad + 1] = Fr::from_u64(5);
        witness[v_pad + 2] = Fr::from_u64(25);

        let source = R1csSource::new(&key, &witness);

        let az = source.compute_matvec(&key.matrices.a);
        assert_eq!(az[0], Fr::from_u64(3));
        assert_eq!(az[key.num_constraints_padded], Fr::from_u64(5));
    }
}
