#![allow(clippy::too_many_arguments)]
use crate::poly::eq_poly::EqPolynomial;
use crate::utils::thread::{drop_in_background_thread, unsafe_allocate_zero_vec};
use crate::utils::{self, compute_dotproduct, compute_dotproduct_low_optimized};

use crate::field::JoltField;
use crate::utils::math::Math;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use core::ops::Index;
use rand_core::{CryptoRng, RngCore};
use rayon::prelude::*;

#[derive(Default, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct DensePolynomial<F: JoltField> {
    num_vars: usize, // the number of variables in the multilinear polynomial
    len: usize,
    pub Z: Vec<F>, // evaluations of the polynomial in all the 2^num_vars Boolean inputs
}

impl<F: JoltField> DensePolynomial<F> {
    pub fn new(Z: Vec<F>) -> Self {
        assert!(
            utils::is_power_of_two(Z.len()),
            "Dense multi-linear polynomials must be made from a power of 2 (not {})",
            Z.len()
        );

        DensePolynomial {
            num_vars: Z.len().log_2(),
            len: Z.len(),
            Z,
        }
    }

    pub fn new_padded(evals: Vec<F>) -> Self {
        // Pad non-power-2 evaluations to fill out the dense multilinear polynomial
        let mut poly_evals = evals;
        while !(utils::is_power_of_two(poly_evals.len())) {
            poly_evals.push(F::zero());
        }

        DensePolynomial {
            num_vars: poly_evals.len().log_2(),
            len: poly_evals.len(),
            Z: poly_evals,
        }
    }

    pub fn get_num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn is_bound(&self) -> bool {
        self.len != self.Z.len()
    }

    pub fn bound_poly_var_top(&mut self, r: &F) {
        let n = self.len() / 2;
        let (left, right) = self.Z.split_at_mut(n);

        left.iter_mut().zip(right.iter()).for_each(|(a, b)| {
            *a += *r * (*b - *a);
        });

        self.num_vars -= 1;
        self.len = n;
    }

    pub fn bound_poly_var_top_many_ones(&mut self, r: &F) {
        let n = self.len() / 2;
        let (left, right) = self.Z.split_at_mut(n);

        left.iter_mut()
            .zip(right.iter())
            .filter(|(&mut a, &b)| a != b)
            .for_each(|(a, b)| {
                let m = *b - *a;
                if m.is_one() {
                    *a += *r;
                } else {
                    *a += *r * m;
                }
            });

        self.num_vars -= 1;
        self.len = n;
    }

    /// Bounds the polynomial's most significant index bit to 'r' optimized for a
    /// high P(eval = 0).
    #[tracing::instrument(skip_all)]
    pub fn bound_poly_var_top_zero_optimized(&mut self, r: &F) {
        let n = self.len() / 2;

        let (left, right) = self.Z.split_at_mut(n);

        left.par_iter_mut()
            .zip(right.par_iter())
            .filter(|(&mut a, &b)| a != b)
            .for_each(|(a, b)| {
                *a += *r * (*b - *a);
            });

        self.num_vars -= 1;
        self.len = n;
    }

    #[tracing::instrument(skip_all)]
    pub fn new_poly_from_bound_poly_var_top(&self, r: &F) -> Self {
        let n = self.len() / 2;
        let mut new_evals: Vec<F> = unsafe_allocate_zero_vec(n);

        for i in 0..n {
            // let low' = low + r * (high - low)
            let low = self.Z[i];
            let high = self.Z[i + n];
            if !(low.is_zero() && high.is_zero()) {
                let m = high - low;
                new_evals[i] = low + *r * m;
            }
        }
        let num_vars = self.num_vars - 1;
        let len = n;

        Self {
            num_vars,
            len,
            Z: new_evals,
        }
    }

    #[tracing::instrument(skip_all)]
    pub fn new_poly_from_bound_poly_var_top_flags(&self, r: &F) -> Self {
        let n = self.len() / 2;
        let mut new_evals: Vec<F> = unsafe_allocate_zero_vec(n);

        for i in 0..n {
            // let low' = low + r * (high - low)
            // Special truth table here
            //         high 0   high 1
            // low 0     0        r
            // low 1   (1-r)      1
            let low = self.Z[i];
            let high = self.Z[i + n];

            if low.is_zero() {
                if high.is_one() {
                    new_evals[i] = *r;
                } else if !high.is_zero() {
                    panic!("Shouldn't happen for a flag poly");
                }
            } else if low.is_one() {
                if high.is_one() {
                    new_evals[i] = F::one();
                } else if high.is_zero() {
                    new_evals[i] = F::one() - *r;
                } else {
                    panic!("Shouldn't happen for a flag poly");
                }
            }
        }
        let num_vars = self.num_vars - 1;
        let len = n;

        Self {
            num_vars,
            len,
            Z: new_evals,
        }
    }

    /// Note: does not truncate
    #[tracing::instrument(skip_all)]
    pub fn bound_poly_var_bot(&mut self, r: &F) {
        let n = self.len() / 2;
        for i in 0..n {
            self.Z[i] = self.Z[2 * i] + *r * (self.Z[2 * i + 1] - self.Z[2 * i]);
        }

        self.num_vars -= 1;
        self.len = n;
    }

    pub fn bound_poly_var_bot_01_optimized(&mut self, r: &F) {
        let n = self.len() / 2;
        let mut new_z = unsafe_allocate_zero_vec(n);
        new_z.par_iter_mut().enumerate().for_each(|(i, z)| {
            let m = self.Z[2 * i + 1] - self.Z[2 * i];
            *z = if m.is_zero() {
                self.Z[2 * i]
            } else if m.is_one() {
                self.Z[2 * i] + r
            } else {
                self.Z[2 * i] + *r * m
            }
        });

        let old_Z = std::mem::replace(&mut self.Z, new_z);
        drop_in_background_thread(old_Z);

        self.num_vars -= 1;
        self.len = n;
    }

    // returns Z(r) in O(n) time
    pub fn evaluate(&self, r: &[F]) -> F {
        // r must have a value for each variable
        assert_eq!(r.len(), self.get_num_vars());
        let chis = EqPolynomial::evals(r);
        assert_eq!(chis.len(), self.Z.len());
        compute_dotproduct(&self.Z, &chis)
    }

    pub fn evaluate_at_chi(&self, chis: &[F]) -> F {
        compute_dotproduct(&self.Z, chis)
    }

    pub fn evaluate_at_chi_low_optimized(&self, chis: &[F]) -> F {
        assert_eq!(self.Z.len(), chis.len());
        compute_dotproduct_low_optimized(&self.Z, chis)
    }

    pub fn evals(&self) -> Vec<F> {
        self.Z.clone()
    }

    pub fn evals_ref(&self) -> &[F] {
        self.Z.as_ref()
    }

    #[tracing::instrument(skip_all, name = "DensePolynomial::from")]
    pub fn from_usize(Z: &[usize]) -> Self {
        DensePolynomial::new(
            (0..Z.len())
                .map(|i| F::from_u64(Z[i] as u64))
                .collect::<Vec<F>>(),
        )
    }

    #[tracing::instrument(skip_all, name = "DensePolynomial::from")]
    pub fn from_u64(Z: &[u64]) -> Self {
        DensePolynomial::new((0..Z.len()).map(|i| F::from_u64(Z[i])).collect::<Vec<F>>())
    }

    pub fn random<R: RngCore + CryptoRng>(num_vars: usize, mut rng: &mut R) -> Self {
        Self::new(
            std::iter::from_fn(|| Some(F::random(&mut rng)))
                .take(1 << num_vars)
                .collect(),
        )
    }
}

impl<F: JoltField> Clone for DensePolynomial<F> {
    fn clone(&self) -> Self {
        Self::new(self.Z[0..self.len].to_vec())
    }
}

impl<F: JoltField> Index<usize> for DensePolynomial<F> {
    type Output = F;

    #[inline(always)]
    fn index(&self, _index: usize) -> &F {
        &(self.Z[_index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_std::test_rng;

    pub fn compute_chis_at_r<F: JoltField>(r: &[F]) -> Vec<F> {
        let ell = r.len();
        let n = ell.pow2();
        let mut chis: Vec<F> = Vec::new();
        for i in 0..n {
            let mut chi_i = F::one();
            for j in 0..r.len() {
                let bit_j = (i & (1 << (r.len() - j - 1))) > 0;
                if bit_j {
                    chi_i *= r[j];
                } else {
                    chi_i *= F::one() - r[j];
                }
            }
            chis.push(chi_i);
        }
        chis
    }

    #[test]
    fn check_memoized_chis() {
        check_memoized_chis_helper::<Fr>()
    }

    fn check_memoized_chis_helper<F: JoltField>() {
        let mut prng = test_rng();

        let s = 10;
        let mut r: Vec<F> = Vec::new();
        for _i in 0..s {
            r.push(F::random(&mut prng));
        }
        let chis = compute_chis_at_r::<F>(&r);
        let chis_m = EqPolynomial::<F>::evals(&r);
        assert_eq!(chis, chis_m);
    }

    #[test]
    fn evaluation() {
        let num_evals = 4;
        let mut evals: Vec<Fr> = Vec::with_capacity(num_evals);
        for _ in 0..num_evals {
            evals.push(Fr::from(8));
        }
        let dense_poly: DensePolynomial<Fr> = DensePolynomial::new(evals.clone());

        // Evaluate at 3:
        // (0, 0) = 1
        // (0, 1) = 1
        // (1, 0) = 1
        // (1, 1) = 1
        // g(x_0,x_1) => c_0*(1 - x_0)(1 - x_1) + c_1*(1-x_0)(x_1) + c_2*(x_0)(1-x_1) + c_3*(x_0)(x_1)
        // g(3, 4) = 8*(1 - 3)(1 - 4) + 8*(1-3)(4) + 8*(3)(1-4) + 8*(3)(4) = 48 + -64 + -72 + 96  = 8
        // g(5, 10) = 8*(1 - 5)(1 - 10) + 8*(1 - 5)(10) + 8*(5)(1-10) + 8*(5)(10) = 96 + -16 + -72 + 96  = 8
        assert_eq!(
            dense_poly.evaluate(vec![Fr::from(3), Fr::from(4)].as_slice()),
            Fr::from(8)
        );
    }
}
