//! R1CS buffer provider for runtime integration.
//!
//! [`R1csProvider`] wraps an [`R1csKey`] and a witness slice, implementing
//! [`BufferProvider`] so the runtime can load R1CS-derived polynomials
//! (Az, Bz, Cz, combined row) as device buffers on demand.

use jolt_compute::{Buf, BufferProvider, ComputeBackend, DeviceBuffer};
use jolt_field::Field;

use crate::key::R1csKey;

/// Polynomial indices for R1CS-derived buffers.
pub const POLY_AZ: usize = 0;
pub const POLY_BZ: usize = 1;
pub const POLY_CZ: usize = 2;
pub const POLY_COMBINED_ROW: usize = 3;

/// Spartan challenge state needed for combined row materialization.
#[derive(Clone, Debug)]
pub struct SpartanChallenges<F> {
    pub rho_a: F,
    pub rho_b: F,
    pub rho_c: F,
    /// Outer sumcheck challenge point (row variables).
    pub r_x: Vec<F>,
}

/// Provides R1CS-derived polynomial buffers to the runtime.
///
/// Created from an [`R1csKey`] and a witness reference. Computes Az, Bz, Cz
/// and the combined row polynomial on demand via [`BufferProvider::load`].
///
/// # Buffer indices
///
/// | Index | Polynomial | When available |
/// |-------|------------|----------------|
/// | [`POLY_AZ`] | Az (A × witness) | Always |
/// | [`POLY_BZ`] | Bz (B × witness) | Always |
/// | [`POLY_CZ`] | Cz (C × witness) | Always |
/// | [`POLY_COMBINED_ROW`] | ρ_A·A(r_x,·) + ρ_B·B(r_x,·) + ρ_C·C(r_x,·) | After `set_challenges` |
pub struct R1csProvider<'a, F: Field> {
    key: &'a R1csKey<F>,
    witness: &'a [F],
    challenges: Option<SpartanChallenges<F>>,
}

impl<'a, F: Field> R1csProvider<'a, F> {
    pub fn new(key: &'a R1csKey<F>, witness: &'a [F]) -> Self {
        Self {
            key,
            witness,
            challenges: None,
        }
    }

    /// Sets the Spartan challenges needed for combined row materialization.
    ///
    /// Must be called before loading [`POLY_COMBINED_ROW`].
    pub fn set_challenges(&mut self, challenges: SpartanChallenges<F>) {
        self.challenges = Some(challenges);
    }

    /// Computes the full Az, Bz, or Cz polynomial for the outer sumcheck.
    ///
    /// For each cycle c and constraint k, computes:
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
}

impl<B: ComputeBackend, F: Field> BufferProvider<usize, B, F> for R1csProvider<'_, F> {
    fn load(&mut self, poly_index: usize, backend: &B) -> Buf<B, F> {
        let buf = match poly_index {
            POLY_AZ => {
                let az = self.compute_matvec(&self.key.matrices.a);
                backend.upload(&az)
            }
            POLY_BZ => {
                let bz = self.compute_matvec(&self.key.matrices.b);
                backend.upload(&bz)
            }
            POLY_CZ => {
                let cz = self.compute_matvec(&self.key.matrices.c);
                backend.upload(&cz)
            }
            POLY_COMBINED_ROW => {
                let ch = self
                    .challenges
                    .as_ref()
                    .expect("set_challenges() must be called before loading POLY_COMBINED_ROW");
                let total_cols_padded = self.key.total_cols().next_power_of_two();
                let row =
                    self.key
                        .combined_row(&ch.r_x, ch.rho_a, ch.rho_b, ch.rho_c, total_cols_padded);
                backend.upload(&row)
            }
            _ => panic!("unknown R1CS polynomial index: {poly_index}"),
        };
        DeviceBuffer::Field(buf)
    }

    fn as_slice(&self, poly_index: usize) -> &[F] {
        panic!(
            "R1CS polynomials (index {poly_index}) are computed on-the-fly and have no host-side slice"
        );
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
        // Single constraint: A = [(1, 1)], B = [(1, 1)], C = [(2, 1)]
        // x * x = y, witness per cycle: [1, x, y, 0]
        let m = ConstraintMatrices::new(
            1,
            3,
            vec![vec![(1, one)]],
            vec![vec![(1, one)]],
            vec![vec![(2, one)]],
        );
        let key = R1csKey::new(m, 2);

        // Two cycles: cycle 0 witness [1, 3, 9, 0], cycle 1 witness [1, 5, 25, 0]
        let v_pad = key.num_vars_padded; // 4
        let mut witness = vec![Fr::zero(); 2 * v_pad];
        witness[0] = Fr::one();
        witness[1] = Fr::from_u64(3);
        witness[2] = Fr::from_u64(9);
        witness[v_pad] = Fr::one();
        witness[v_pad + 1] = Fr::from_u64(5);
        witness[v_pad + 2] = Fr::from_u64(25);

        let provider = R1csProvider::new(&key, &witness);

        let az = provider.compute_matvec(&key.matrices.a);
        // Az[0] = A_local · [1, 3, 9, 0] = 3
        // Az[1] = 0 (padding constraint)
        // Az[2] = A_local · [1, 5, 25, 0] = 5
        assert_eq!(az[0], Fr::from_u64(3));
        assert_eq!(az[key.num_constraints_padded], Fr::from_u64(5));
    }
}
