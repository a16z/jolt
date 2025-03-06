//! Implements the Dao-Thaler optimization for EQ polynomial evaluations
//! https://eprint.iacr.org/2024/1210.pdf
#[cfg(test)]
use super::dense_mlpoly::DensePolynomial;
use crate::{field::JoltField, poly::eq_poly::EqPolynomial};

#[derive(Debug, Clone, PartialEq)]
/// A struct holding the equality polynomial evaluations for use in sum-check,
/// when incorporating both the Gruen and Dao-Thaler optimizations.
///
/// In this optimization, for the `i`th round of sum-check, we want the quantities:
///
/// - `current_index = i`
/// - `current_scalar = \prod_{j < i} eq(w[..j],r[..j])`
/// - If `i < n/2`, then `E1.last().unwrap() = [eq(w[(i + 1)..n/2], x) for all x in {0, 1}^{n/2 - i - 1}]`; else `E1` is empty
/// - `E2.last().unwrap() = [eq(w[max(i, n/2)..n], x) for all x in {0, 1}^{n - max(i, n/2)}]`
pub struct SplitEqPolynomial<F> {
    pub(crate) current_index: usize,
    pub(crate) current_scalar: F,
    pub(crate) w: Vec<F>,
    pub(crate) E1: Vec<Vec<F>>,
    pub(crate) E2: Vec<Vec<F>>,
}

pub struct OldSplitEqPolynomial<F> {
    num_vars: usize,
    pub(crate) E1: Vec<F>,
    pub(crate) E1_len: usize,
    pub(crate) E2: Vec<F>,
    pub(crate) E2_len: usize,
}

impl<F: JoltField> SplitEqPolynomial<F> {
    #[tracing::instrument(skip_all, name = "SplitEqPolynomial::new")]
    pub fn new(w: &[F]) -> Self {
        let m = w.len() / 2;
        let (_, wprime) = w.split_first().unwrap();
        let (w2, w1) = wprime.split_at(m);
        let (E2, E1) = rayon::join(
            || EqPolynomial::evals_cached(w2),
            || EqPolynomial::evals_cached(w1),
        );
        Self {
            current_index: 0,
            current_scalar: F::one(),
            w: w.to_vec(),
            E1,
            E2,
        }
    }

    pub fn get_num_vars(&self) -> usize {
        self.w.len()
    }

    pub fn len(&self) -> usize {
        1 << (self.get_num_vars() - self.current_index + 1)
    }

    #[tracing::instrument(skip_all, name = "SplitEqPolynomial::bind")]
    pub fn bind(&mut self, r: F) {
        // multiply `current_scalar` by `eq(w[i], r) = (1 - w[i]) * (1 - r) + w[i] * r`
        self.current_scalar *= F::one() - self.w[self.current_index] - r
            + self.w[self.current_index] * r
            + self.w[self.current_index] * r;
        // pop the last vector from `E1` or `E2` (since we don't need it anymore)
        if self.current_index < self.w.len() / 2 {
            self.E1.pop();
        } else {
            self.E2.pop();
        }
        // increment `current_index`
        self.current_index += 1;
    }

    #[cfg(test)]
    pub fn merge(&self) -> DensePolynomial<F> {
        DensePolynomial::new(EqPolynomial::evals(&self.w[self.current_index..]))
    }
}

impl<F: JoltField> OldSplitEqPolynomial<F> {
    #[tracing::instrument(skip_all, name = "OldSplitEqPolynomial::new")]
    pub fn new(w: &[F]) -> Self {
        let m = w.len() / 2;
        let (w2, w1) = w.split_at(m);
        let (E2, E1) = rayon::join(|| EqPolynomial::evals(w2), || EqPolynomial::evals(w1));
        let E1_len = E1.len();
        let E2_len = E2.len();
        Self {
            num_vars: w.len(),
            E1,
            E1_len,
            E2,
            E2_len,
        }
    }

    pub fn get_num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn len(&self) -> usize {
        if self.E1_len == 1 {
            self.E2_len
        } else {
            self.E1_len * self.E2_len
        }
    }

    #[tracing::instrument(skip_all, name = "OldSplitEqPolynomial::bind")]
    pub fn bind(&mut self, r: F) {
        if self.E1_len == 1 {
            // E_1 is already completely bound, so we bind E_2
            let n = self.E2_len / 2;
            for i in 0..n {
                self.E2[i] = self.E2[2 * i] + r * (self.E2[2 * i + 1] - self.E2[2 * i]);
            }
            self.E2_len = n;
        } else {
            // Bind E_1
            let n = self.E1_len / 2;
            for i in 0..n {
                self.E1[i] = self.E1[2 * i] + r * (self.E1[2 * i + 1] - self.E1[2 * i]);
            }
            self.E1_len = n;

            // If E_1 is now completely bound, we will be switching over to the
            // linear-time sumcheck prover, using E_1 * E_2:
            if self.E1_len == 1 {
                self.E2[..self.E2_len]
                    .iter_mut()
                    .for_each(|eval| *eval *= self.E1[0]);
            }
        }
    }

    #[cfg(test)]
    pub fn merge(&self) -> DensePolynomial<F> {
        if self.E1_len == 1 {
            DensePolynomial::new(self.E2[..self.E2_len].to_vec())
        } else {
            let mut merged = vec![];
            for i in 0..self.E2_len {
                for j in 0..self.E1_len {
                    merged.push(self.E2[i] * self.E1[j])
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
        const NUM_VARS: usize = 9;
        let mut rng = test_rng();
        let w: Vec<Fr> = std::iter::repeat_with(|| Fr::random(&mut rng))
            .take(NUM_VARS)
            .collect();

        let mut regular_eq = DensePolynomial::new(EqPolynomial::evals(&w));
        let mut split_eq = SplitEqPolynomial::new(&w);
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
