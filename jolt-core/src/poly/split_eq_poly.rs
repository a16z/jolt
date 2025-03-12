//! Implements the Dao-Thaler optimization for EQ polynomial evaluations
//! https://eprint.iacr.org/2024/1210.pdf
#[cfg(test)]
use super::dense_mlpoly::DensePolynomial;
use crate::{field::JoltField, poly::eq_poly::EqPolynomial};

#[derive(Debug, Clone, PartialEq)]
/// A struct holding the equality polynomial evaluations for use in sum-check, when incorporating
/// both the Gruen and Dao-Thaler optimizations.
///
/// For the `i = 0..n`-th round of sum-check, we want the following invariants:
///
/// - `current_index = n - i` (where `n = w.len()`)
/// - `current_scalar = eq(w[(n - i)..],r[..i])`
/// - `E2.last().unwrap() = [eq(w[..min(i, n/2)], x) for all x in {0, 1}^{n - min(i, n/2)}]`
/// - If `i < n/2`, then `E1.last().unwrap() = [eq(w[n/2..(n/2 + i + 1)], x) for all x in {0,
///   1}^{n/2 - i - 1}]`; else `E1` is empty
///
/// Note: all current applications of `SplitEqPolynomial` use the `LowToHigh` binding order. This
/// means that we are iterating over `w` in the reverse order: `w.len()` down to `0`.
pub struct SplitEqPolynomial<F> {
    pub(crate) current_index: usize,
    pub(crate) current_scalar: F,
    pub(crate) w: Vec<F>,
    pub(crate) E1: Vec<Vec<F>>,
    pub(crate) E2: Vec<Vec<F>>,
}

/// Old struct for equality polynomial, without Gruen's optimization
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
        //   w = [w2, w1, w_last]
        //        ↑   ↑    ↑
        //        |   |    |
        //        |   |    last element
        //        |   second half of remaining elements (for E1)
        //        first half of remaining elements (for E2)
        let (_, wprime) = w.split_last().unwrap();
        let (w2, w1) = wprime.split_at(m);
        let (E2, E1) = rayon::join(
            || EqPolynomial::evals_cached(w2),
            || EqPolynomial::evals_cached(w1),
        );
        Self {
            current_index: w.len(),
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
        1 << self.current_index
    }

    pub fn E1_len(&self) -> usize {
        self.E1.last().unwrap().len()
    }

    pub fn E2_len(&self) -> usize {
        self.E2.last().unwrap().len()
    }

    /// Return the last vector from `E1` as a slice
    pub fn E1_current(&self) -> &[F] {
        self.E1.last().unwrap()
    }

    pub fn to_E1_old(&self) -> Vec<F> {
        if self.current_index > self.w.len() / 2 {
            let wi = self.w[self.current_index - 1];
            let E1_old_odd: Vec<F> = self
                .E1
                .last()
                .unwrap()
                .iter()
                .map(|x| *x * (F::one() - wi))
                .collect();
            let E1_old_even: Vec<F> = self.E1.last().unwrap().iter().map(|x| *x * wi).collect();
            // Interleave the two vectors
            let mut E1_old = vec![];
            for i in 0..E1_old_odd.len() {
                E1_old.push(E1_old_odd[i]);
                E1_old.push(E1_old_even[i]);
            }
            E1_old
        } else {
            println!("Don't expect to call this");
            vec![self.current_scalar; 1]
        }
    }

    /// Return the last vector from `E2` as a slice
    pub fn E2_current(&self) -> &[F] {
        self.E2.last().unwrap()
    }

    #[tracing::instrument(skip_all, name = "SplitEqPolynomial::bind")]
    pub fn bind(&mut self, r: F) {
        // multiply `current_scalar` by `eq(w[i], r) = (1 - w[i]) * (1 - r) + w[i] * r`
        self.current_scalar *= F::one() - self.w[self.current_index - 1] - r
            + self.w[self.current_index - 1] * r
            + self.w[self.current_index - 1] * r;
        // decrement `current_index`
        self.current_index -= 1;
        // pop the last vector from `E1` or `E2` (since we don't need it anymore)
        if self.w.len() / 2 < self.current_index {
            self.E1.pop();
        } else if 0 < self.current_index {
            self.E2.pop();
        }
        // println!(
        //     "current_index: {}, E1_len: {}, E2_len: {}",
        //     self.current_index,
        //     self.E1.len(),
        //     self.E2.len()
        // );
    }

    #[cfg(test)]
    pub fn merge(&self) -> DensePolynomial<F> {
        let evals = EqPolynomial::evals(&self.w[..self.current_index])
            .iter()
            .map(|x| *x * self.current_scalar)
            .collect();
        DensePolynomial::new(evals)
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
        const NUM_VARS: usize = 10;
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

    #[test]
    fn equal_old_and_new_split_eq() {
        const NUM_VARS: usize = 4;
        let mut rng = test_rng();
        let w: Vec<Fr> = std::iter::repeat_with(|| Fr::random(&mut rng))
            .take(NUM_VARS)
            .collect();

        let mut old_split_eq = OldSplitEqPolynomial::new(&w);
        let mut new_split_eq = SplitEqPolynomial::new(&w);

        assert_eq!(old_split_eq.get_num_vars(), new_split_eq.get_num_vars());
        assert_eq!(old_split_eq.len(), new_split_eq.len());
        assert_eq!(old_split_eq.E1, *new_split_eq.to_E1_old());
        assert_eq!(old_split_eq.E2, *new_split_eq.E2.last().unwrap());
        assert_eq!(old_split_eq.merge(), new_split_eq.merge());
        // Show that they are the same after binding
        for i in (0..NUM_VARS).rev() {
            println!("i: {}", i);
            let r = Fr::random(&mut rng);
            old_split_eq.bind(r);
            new_split_eq.bind(r);
            assert_eq!(old_split_eq.merge(), new_split_eq.merge());
            if NUM_VARS / 2 < i {
                assert_eq!(old_split_eq.E1_len, new_split_eq.E1_len() * 2);
                assert_eq!(old_split_eq.E2_len, new_split_eq.E2_len());
            } else if i > 0 {
                assert_eq!(old_split_eq.E1_len, new_split_eq.E1_len());
                assert_eq!(old_split_eq.E2_len, new_split_eq.E2_len() * 2);
            }
        }
    }

    #[test]
    fn bench_old_and_new_split_eq() {
        let mut rng = test_rng();
        for num_vars in 5..30 {
            let w: Vec<Fr> = std::iter::repeat_with(|| Fr::random(&mut rng))
                .take(num_vars)
                .collect();
            println!("Testing for {} variables", num_vars);

            let start_old_split_eq_time = std::time::Instant::now();
            let _old_split_eq = OldSplitEqPolynomial::new(&w);
            let end_old_split_eq_time = std::time::Instant::now();
            println!(
                "Time taken for creating old split eq: {:?}",
                end_old_split_eq_time.duration_since(start_old_split_eq_time)
            );

            let start_new_split_eq_time = std::time::Instant::now();
            let _new_split_eq = SplitEqPolynomial::new(&w);
            let end_new_split_eq_time = std::time::Instant::now();
            println!(
                "Time taken for creating new split eq: {:?}",
                end_new_split_eq_time.duration_since(start_new_split_eq_time)
            );
        }
    }
}
