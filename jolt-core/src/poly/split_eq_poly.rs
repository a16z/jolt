//! Implements the Dao-Thaler optimization for EQ polynomial evaluations
//! https://eprint.iacr.org/2024/1210.pdf
#[cfg(test)]
use super::dense_mlpoly::DensePolynomial;
use crate::{field::JoltField, poly::eq_poly::EqPolynomial};
use rayon::prelude::*;

use crate::utils::{math::Math, thread::unsafe_allocate_zero_vec};

#[derive(Debug)]
pub struct SplitEqPolynomial<F> {
    E_1: Vec<F>,
    E_1_len: usize,
    E_2: Vec<F>,
    E_2_len: usize,
}

impl<F: JoltField> SplitEqPolynomial<F> {
    pub fn new(w: Vec<F>) -> Self {
        let m = w.len() / 2;
        let (w_2, w_1) = w.split_at(m);
        let (E_2, E_1) = rayon::join(|| EqPolynomial::evals(w_2), || EqPolynomial::evals(w_1));
        let E_1_len = E_1.len();
        let E_2_len = E_2.len();
        Self {
            E_1,
            E_1_len,
            E_2,
            E_2_len,
        }
    }

    pub fn bind(&mut self, r: F) {
        if self.E_1_len == 1 {
            // E_1 is already completely bound, so we bind E_2
            let n = self.E_2_len / 2;
            for i in 0..n {
                self.E_2[i] = self.E_2[2 * i] + r * (self.E_2[2 * i + 1] - self.E_2[2 * i]);
            }
            self.E_2_len = n;
        } else {
            // Bind E_1
            let n = self.E_1_len / 2;
            for i in 0..n {
                self.E_1[i] = self.E_1[2 * i] + r * (self.E_1[2 * i + 1] - self.E_1[2 * i]);
            }
            self.E_1_len = n;

            // If E_1 is now completely bound, we will be switching over to the
            // linear-time sumcheck prover, using E_1 * E_2:
            if self.E_1_len == 1 {
                self.E_2.iter_mut().for_each(|eval| *eval *= self.E_1[0]);
            }
        }
    }

    #[cfg(test)]
    fn merge(&self) -> DensePolynomial<F> {
        if self.E_1_len == 1 {
            DensePolynomial::new(self.E_2[..self.E_2_len].to_vec())
        } else {
            let mut merged = vec![];
            for i in 0..self.E_2_len {
                for j in 0..self.E_1_len {
                    merged.push(self.E_2[i] * self.E_1[j])
                }
            }
            DensePolynomial::new(merged)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_std::test_rng;

    #[test]
    fn bind() {
        const NUM_VARS: usize = 10;
        let mut rng = test_rng();
        let w: Vec<Fr> = std::iter::repeat_with(|| Fr::random(&mut rng))
            .take(NUM_VARS)
            .collect();

        let mut regular_eq = DensePolynomial::new(EqPolynomial::evals(&w));
        let mut split_eq = SplitEqPolynomial::new(w);
        assert_eq!(regular_eq, split_eq.merge());

        for _ in 0..NUM_VARS {
            let r = Fr::random(&mut rng);
            regular_eq.bound_poly_var_bot(&r);
            split_eq.bind(r);

            let merged = split_eq.merge();
            assert_eq!(regular_eq.Z[..regular_eq.len()], merged.Z[..merged.len()]);
        }
    }
}
