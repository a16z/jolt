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
/// - `E_out_vec.last().unwrap() = [eq(w[..min(i, n/2)], x) for all x in {0, 1}^{n - min(i, n/2)}]`
/// - If `i < n/2`, then `E_in_vec.last().unwrap() = [eq(w[n/2..(n/2 + i + 1)], x) for all x in {0,
///   1}^{n/2 - i - 1}]`; else `E_in_vec` is empty
///
/// Note: all current applications of `SplitEqPolynomial` use the `LowToHigh` binding order. This
/// means that we are iterating over `w` in the reverse order: `w.len()` down to `0`.
pub struct GruenSplitEqPolynomial<F> {
    pub(crate) current_index: usize,
    pub(crate) current_scalar: F,
    pub(crate) w: Vec<F>,
    pub(crate) E_in_vec: Vec<Vec<F>>,
    pub(crate) E_out_vec: Vec<Vec<F>>,
}

/// Old struct for split equality polynomial, without Gruen's optimization
/// TODO: remove all usage of this struct with the new one
pub struct SplitEqPolynomial<F> {
    num_vars: usize,
    pub(crate) E1: Vec<F>,
    pub(crate) E1_len: usize,
    pub(crate) E2: Vec<F>,
    pub(crate) E2_len: usize,
}

impl<F: JoltField> GruenSplitEqPolynomial<F> {
    #[tracing::instrument(skip_all, name = "GruenSplitEqPolynomial::new")]
    pub fn new(w: &[F]) -> Self {
        let m = w.len() / 2;
        //   w = [w_out, w_in, w_last]
        //         ↑      ↑      ↑
        //         |      |      |
        //         |      |      last element
        //         |      second half of remaining elements (for E_in)
        //         first half of remaining elements (for E_out)
        let (_, wprime) = w.split_last().unwrap();
        let (w_out, w_in) = wprime.split_at(m);
        let (E_out_vec, E_in_vec) = rayon::join(
            || EqPolynomial::evals_cached(w_out),
            || EqPolynomial::evals_cached(w_in),
        );
        Self {
            current_index: w.len(),
            current_scalar: F::one(),
            w: w.to_vec(),
            E_in_vec,
            E_out_vec,
        }
    }

    /// Compute the split equality polynomial for the small value optimization
    ///
    /// The split is done as follows: (here `l = num_small_value_rounds`)
    ///
    /// 0 ..... (n/2 - l) ..... (n - l) ..... n
    ///
    ///           <-- E_in -->
    ///
    /// E_out --->                <--- E_out
    ///
    /// where the first E_out part (0 to n/2 - l) corresponds to x_out, and the second E_out part
    /// (n/2 - l to n) corresponds to y_suffix
    ///
    /// Returns E_out which contains `l` vectors of eq evals for the same x_out part, with decreasing
    /// length for y_suffix, and E_in which contains the single vector of eq evals for the x_in part.
    ///
    /// Note the differences between this and the `new` constructor: this is specialized for the
    /// small value optimization.
    pub fn new_for_small_value(
        w: &[F],
        num_x_out_vars: usize,
        num_x_in_vars: usize,
        num_small_value_rounds: usize,
    ) -> Self {
        // Split w into the slices: (l = num_small_value_rounds)
        // (n/2 - l) ..... (n - l)
        // 0..(n/2 - l - 1) concatenated with (n - l)...n
        // Then invoke the evals_cached constructor on the concatenated slice, producing E_out
        // Invoke the evals constructor (no caching) on the middle slice, producing E_in
        // In other words, there is only 1 vector in E_in, and l vectors in E_out
        // (we may drop the rest of the vectors after evals_cached)
        let n = w.len();

        assert!(
            n > 0,
            "length of w must be positive for the split to be valid."
        );
        assert!(num_x_out_vars + num_x_in_vars + num_small_value_rounds == n, "num_x_out_vars + num_x_in_vars + num_small_value_rounds must be == n for the split to be valid.");

        // This should be `min(num_steps, n/2 - num_small_value_rounds)`, computed externally before calling this function.
        let split_point_x_out = num_x_out_vars;
        let split_point_x_in = split_point_x_out + num_x_in_vars;

        let w_E_in_vars: Vec<F> = w[split_point_x_out..split_point_x_in].to_vec();

        // Determine the end index for the suffix part of w_E_out_vars
        let suffix_slice_end = if num_small_value_rounds == 0 {
            split_point_x_in // Results in an empty suffix, e.g., w[n..n]
        } else {
            n - 1 // Use up to n-1, excluding the last variable of w (tau)
        };

        let num_actual_suffix_vars = suffix_slice_end.saturating_sub(split_point_x_in);

        let mut w_E_out_vars: Vec<F> = Vec::with_capacity(num_x_out_vars + num_actual_suffix_vars);
        w_E_out_vars.extend_from_slice(&w[0..split_point_x_out]);
        if split_point_x_in < suffix_slice_end {
            // Add suffix only if range is valid and non-empty
            w_E_out_vars.extend_from_slice(&w[split_point_x_in..suffix_slice_end]);
        }

        let (mut E_out_vec, E_in) = rayon::join(
            || EqPolynomial::evals_cached(&w_E_out_vars),
            || EqPolynomial::evals(&w_E_in_vars),
        );

        // Take only the first `num_small_value_rounds` vectors from E_out_vec (after reversing)
        // Recall that at this point, E_out_vec[0] = `eq(w[0..split_point_x_out] ++ w[split_point_x_in..n-1], x)`
        E_out_vec.reverse();
        E_out_vec.truncate(num_small_value_rounds);

        Self {
            current_index: num_x_out_vars,
            current_scalar: F::one(),
            w: w.to_vec(),
            E_in_vec: vec![E_in],
            E_out_vec,
        }
    }

    pub fn get_num_vars(&self) -> usize {
        self.w.len()
    }

    pub fn len(&self) -> usize {
        1 << self.current_index
    }

    pub fn E_in_current_len(&self) -> usize {
        self.E_in_vec.last().unwrap().len()
    }

    pub fn E_out_current_len(&self) -> usize {
        self.E_out_vec.last().unwrap().len()
    }

    /// Return the last vector from `E1` as a slice
    pub fn E_in_current(&self) -> &[F] {
        self.E_in_vec.last().unwrap()
    }

    /// Return the last vector from `E2` as a slice
    pub fn E_out_current(&self) -> &[F] {
        self.E_out_vec.last().unwrap()
    }

    #[tracing::instrument(skip_all, name = "GruenSplitEqPolynomial::bind")]
    pub fn bind(&mut self, r: F) {
        // multiply `current_scalar` by `eq(w[i], r) = (1 - w[i]) * (1 - r) + w[i] * r`
        // which is the same as `1 - w[i] - r + 2 * w[i] * r`
        let prod_w_r = self.w[self.current_index - 1] * r;
        self.current_scalar *= F::one() - self.w[self.current_index - 1] - r + prod_w_r + prod_w_r;
        // decrement `current_index`
        self.current_index -= 1;
        // pop the last vector from `E_in_vec` or `E_out_vec` (since we don't need it anymore)
        if self.w.len() / 2 < self.current_index {
            self.E_in_vec.pop();
        } else if 0 < self.current_index {
            self.E_out_vec.pop();
        }
    }

    #[cfg(test)]
    fn to_E1_old(&self) -> Vec<F> {
        if self.current_index > self.w.len() / 2 {
            let wi = self.w[self.current_index - 1];
            let E1_old_odd: Vec<F> = self
                .E_in_vec
                .last()
                .unwrap()
                .iter()
                .map(|x| *x * (F::one() - wi))
                .collect();
            let E1_old_even: Vec<F> = self
                .E_in_vec
                .last()
                .unwrap()
                .iter()
                .map(|x| *x * wi)
                .collect();
            // Interleave the two vectors
            let mut E1_old = vec![];
            for i in 0..E1_old_odd.len() {
                E1_old.push(E1_old_odd[i]);
                E1_old.push(E1_old_even[i]);
            }
            E1_old
        } else {
            // println!("Don't expect to call this");
            vec![self.current_scalar; 1]
        }
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

impl<F: JoltField> SplitEqPolynomial<F> {
    #[tracing::instrument(skip_all, name = "SplitEqPolynomial::new")]
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

    #[tracing::instrument(skip_all, name = "SplitEqPolynomial::bind")]
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
        let mut split_eq = GruenSplitEqPolynomial::new(&w);
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
        const NUM_VARS: usize = 15;
        let mut rng = test_rng();
        let w: Vec<Fr> = std::iter::repeat_with(|| Fr::random(&mut rng))
            .take(NUM_VARS)
            .collect();

        let mut old_split_eq = SplitEqPolynomial::new(&w);
        let mut new_split_eq = GruenSplitEqPolynomial::new(&w);

        assert_eq!(old_split_eq.get_num_vars(), new_split_eq.get_num_vars());
        assert_eq!(old_split_eq.len(), new_split_eq.len());
        assert_eq!(old_split_eq.E1, *new_split_eq.to_E1_old());
        assert_eq!(old_split_eq.E2, *new_split_eq.E_out_current());
        assert_eq!(old_split_eq.merge(), new_split_eq.merge());
        // Show that they are the same after binding
        for i in (0..NUM_VARS).rev() {
            let r = Fr::random(&mut rng);
            old_split_eq.bind(r);
            new_split_eq.bind(r);
            assert_eq!(old_split_eq.merge(), new_split_eq.merge());
            if NUM_VARS / 2 < i {
                assert_eq!(old_split_eq.E1_len, new_split_eq.E_in_current_len() * 2);
                assert_eq!(old_split_eq.E2_len, new_split_eq.E_out_current_len());
            } else if i > 0 {
                assert_eq!(old_split_eq.E1_len, new_split_eq.E_in_current_len());
                assert_eq!(old_split_eq.E2_len, new_split_eq.E_out_current_len() * 2);
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
            let _old_split_eq = SplitEqPolynomial::new(&w);
            let end_old_split_eq_time = std::time::Instant::now();
            println!(
                "Time taken for creating old split eq: {:?}",
                end_old_split_eq_time.duration_since(start_old_split_eq_time)
            );

            let start_new_split_eq_time = std::time::Instant::now();
            let _new_split_eq = GruenSplitEqPolynomial::new(&w);
            let end_new_split_eq_time = std::time::Instant::now();
            println!(
                "Time taken for creating new split eq: {:?}",
                end_new_split_eq_time.duration_since(start_new_split_eq_time)
            );
        }
    }

    #[test]
    fn test_new_for_small_value() {
        let mut rng = test_rng();
        const N: usize = 10; // Total variables
        const L0: usize = 3; // SVO rounds

        // Test case 1: Standard setup
        let num_x_out_vars_1 = 2; // Example split for x_out part
        let w1: Vec<Fr> = (0..N).map(|i| Fr::from(i as u64)).collect(); // Use predictable values

        let num_x_in_vars_1 = N - num_x_out_vars_1 - L0;
        let split_eq1 =
            GruenSplitEqPolynomial::new_for_small_value(&w1, num_x_out_vars_1, num_x_in_vars_1, L0);

        // Verify split points and variable slices
        let split_point1_expected1 = num_x_out_vars_1; // Should be 2
        let split_point_x_in_expected1 = num_x_out_vars_1 + num_x_in_vars_1;
        assert_eq!(split_eq1.current_index, split_point1_expected1); // repurposed current_index

        let w_E_in_vars_expected1: Vec<Fr> =
            w1[split_point1_expected1..split_point_x_in_expected1].to_vec(); // w[2..7] = [2,3,4,5,6]
        let mut w_E_out_vars_expected1: Vec<Fr> = Vec::new();
        w_E_out_vars_expected1.extend_from_slice(&w1[0..split_point1_expected1]); // w[0..2] = [0,1]
                                                                                  // Suffix slice is w[split_point_x_in .. N-1] = w[7..9] for N=10, L0=3.
        if split_point_x_in_expected1 < N - 1 {
            // Match logic in main code for L0 > 0
            w_E_out_vars_expected1.extend_from_slice(&w1[split_point_x_in_expected1..N - 1]);
            // w[7..9] = [7,8]
        }
        // Combined = [0, 1, 7, 8]

        // Verify E_in content
        assert_eq!(split_eq1.E_in_vec.len(), 1);
        let expected_E_in1 = EqPolynomial::evals(&w_E_in_vars_expected1);
        assert_eq!(split_eq1.E_in_vec[0], expected_E_in1);

        // Verify E_out content (structure and count)
        assert_eq!(split_eq1.E_out_vec.len(), L0); // Should have L0 = 3 vectors

        // Verify E_out content requires understanding evals_cached internal structure
        // evals_cached(w_E_out) returns [ T(w_E_out[0..k], x), T(w_E_out[0..k-1], x), ..., T(w_E_out[0], x), T([], x) ]
        // where k = w_E_out.len(). Let k=4 here ([0,1,7,8]). Returns 5 vectors.
        // new_for_small_value takes the *last* L0=3 vectors and reverses them.
        // Last 3 vectors from evals_cached([0,1,7,8]) correspond to challenges w=[0,1,7], w=[0,1], w=[0]
        // After reversal: E_out_vec[0] is cache for w=[0], E_out_vec[1] for w=[0,1], E_out_vec[2] for w=[0,1,7]

        let cached_E_out1 = EqPolynomial::evals_cached(&w_E_out_vars_expected1);
        // Expected: cached_E_out1 has len k+1 = 5
        assert_eq!(cached_E_out1.len(), w_E_out_vars_expected1.len() + 1);

        // E_out_vec[0] should be cached_E_out1[4] (evals for w=[0])
        assert_eq!(
            split_eq1.E_out_vec[0],
            cached_E_out1[w_E_out_vars_expected1.len() - 0]
        );
        // E_out_vec[1] should be cached_E_out1[3] (evals for w=[0,1])
        assert_eq!(
            split_eq1.E_out_vec[1],
            cached_E_out1[w_E_out_vars_expected1.len() - 1]
        );
        // E_out_vec[2] should be cached_E_out1[2] (evals for w=[0,1,7])
        assert_eq!(
            split_eq1.E_out_vec[2],
            cached_E_out1[w_E_out_vars_expected1.len() - 2]
        );

        // Test case 2: Edge case L0 = 0
        let num_x_out_vars_2 = N / 2; // Max possible value for num_x_out_vars if num_x_in_vars is also N/2 and L0=0
        let w2: Vec<Fr> = (0..N).map(|_| Fr::random(&mut rng)).collect();
        let num_x_in_vars_2 = N - num_x_out_vars_2 - 0; // L0 is 0
        let split_eq2 =
            GruenSplitEqPolynomial::new_for_small_value(&w2, num_x_out_vars_2, num_x_in_vars_2, 0);
        assert_eq!(split_eq2.E_out_vec.len(), 0);
        assert_eq!(split_eq2.E_in_vec.len(), 1); // E_in should cover w[N/2 .. N/2 + num_x_in_vars_2 -1]
        let split_point1_expected2 = num_x_out_vars_2;
        let split_point_x_in_expected2 = num_x_out_vars_2 + num_x_in_vars_2;
        let w_E_in_vars_expected2: Vec<Fr> =
            w2[split_point1_expected2..split_point_x_in_expected2].to_vec();
        assert!(w_E_in_vars_expected2.len() == num_x_in_vars_2);
        let expected_E_in2 = EqPolynomial::evals(&w_E_in_vars_expected2); // evals of N/2 vars
        assert_eq!(split_eq2.E_in_vec[0], expected_E_in2);

        // Test case 3: Panic case N = 0
        let w3: Vec<Fr> = vec![];
        let l0_3 = 0;
        let num_x_out_vars_3 = 0;
        let n3 = w3.len();
        let num_x_in_vars_3 = n3 - num_x_out_vars_3 - l0_3; // 0 - 0 - 0 = 0
        let result3 = std::panic::catch_unwind(|| {
            GruenSplitEqPolynomial::new_for_small_value(
                &w3,
                num_x_out_vars_3,
                num_x_in_vars_3,
                l0_3,
            );
        });
        assert!(result3.is_err());
    }
}
