#![allow(missing_docs)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

use super::ark_field::ArkFr;
use crate::error::DoryError;
use crate::mode::Mode;
use crate::primitives::arithmetic::{DoryRoutines, Field, Group, PairingCurve};
use crate::primitives::poly::{MultilinearLagrange, Polynomial};
use crate::setup::ProverSetup;

/// Simple polynomial implementation wrapping coefficient vector
#[derive(Clone, Debug)]
pub struct ArkworksPolynomial {
    coefficients: Vec<ArkFr>,
    num_vars: usize,
}

impl ArkworksPolynomial {
    pub fn new(coefficients: Vec<ArkFr>) -> Self {
        let len = coefficients.len();
        let num_vars = (len as f64).log2() as usize;
        assert_eq!(
            1 << num_vars,
            len,
            "Coefficient length must be a power of 2"
        );
        Self {
            coefficients,
            num_vars,
        }
    }
}

impl Polynomial<ArkFr> for ArkworksPolynomial {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    #[tracing::instrument(skip_all, name = "ArkworksPolynomial::evaluate", fields(num_vars = self.num_vars))]
    fn evaluate(&self, point: &[ArkFr]) -> ArkFr {
        assert_eq!(point.len(), self.num_vars, "Point dimension mismatch");

        // Compute multilinear Lagrange basis
        let mut basis = vec![ArkFr::zero(); 1 << self.num_vars];
        crate::primitives::poly::multilinear_lagrange_basis(&mut basis, point);

        // Evaluate: sum_i coeff[i] * basis[i]
        let mut result = ArkFr::zero();
        for (coeff, basis_val) in self.coefficients.iter().zip(basis.iter()) {
            result = result + coeff.mul(basis_val);
        }
        result
    }

    #[tracing::instrument(skip_all, name = "ArkworksPolynomial::commit", fields(nu, sigma, num_rows = 1 << nu, num_cols = 1 << sigma))]
    #[allow(clippy::type_complexity)]
    fn commit<E, Mo, M1>(
        &self,
        nu: usize,
        sigma: usize,
        setup: &ProverSetup<E>,
    ) -> Result<(E::GT, Vec<E::G1>, ArkFr), DoryError>
    where
        E: PairingCurve,
        Mo: Mode,
        M1: DoryRoutines<E::G1>,
        E::G1: Group<Scalar = ArkFr>,
        E::GT: Group<Scalar = ArkFr>,
    {
        let expected_len = 1 << (nu + sigma);
        if self.coefficients.len() != expected_len {
            return Err(DoryError::InvalidSize {
                expected: expected_len,
                actual: self.coefficients.len(),
            });
        }

        let num_rows = 1 << nu;
        let num_cols = 1 << sigma;
        let g1 = &setup.g1_vec[..num_cols];

        // Row commitments are always unblinded (internal to prover)
        let row_commitments: Vec<E::G1> = (0..num_rows)
            .map(|i| {
                let row = &self.coefficients[i * num_cols..(i + 1) * num_cols];
                M1::msm(g1, row)
            })
            .collect();

        let tier_2 = E::multi_pair_g2_setup(&row_commitments, &setup.g2_vec[..num_rows]);

        // Single GT-level blind: commitment += r_d1 * HT
        let r_d1: ArkFr = Mo::sample();
        let commitment = Mo::mask(tier_2, &setup.ht, &r_d1);

        Ok((commitment, row_commitments, r_d1))
    }
}

impl MultilinearLagrange<ArkFr> for ArkworksPolynomial {
    #[tracing::instrument(skip_all, name = "ArkworksPolynomial::vector_matrix_product", fields(nu, sigma, num_rows = 1 << nu, num_cols = 1 << sigma))]
    fn vector_matrix_product(&self, left_vec: &[ArkFr], nu: usize, sigma: usize) -> Vec<ArkFr> {
        let num_cols = 1 << sigma;
        let num_rows = 1 << nu;
        let mut v_vec = vec![ArkFr::zero(); num_cols];

        for (j, v) in v_vec.iter_mut().enumerate() {
            let mut sum = ArkFr::zero();
            for (i, left_val) in left_vec.iter().enumerate().take(num_rows) {
                let coeff_idx = i * num_cols + j;
                if coeff_idx < self.coefficients.len() {
                    sum = sum + left_val.mul(&self.coefficients[coeff_idx]);
                }
            }
            *v = sum;
        }

        v_vec
    }
}
