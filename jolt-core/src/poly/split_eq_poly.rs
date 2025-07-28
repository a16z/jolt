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

    #[tracing::instrument(skip_all, name = "GruenSplitEqPolynomial::new_rev")]
    pub fn new_rev(w: &[F]) -> Self {
        let mut w_rev = w.to_vec();
        w_rev.reverse();
        let w = &w_rev;
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
            w: w_rev,
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

    /// Compute the sumcheck cubic sumcheck evaluations (i.e., the evaluations at {0, 2, 3}) of a
    /// polynomial s(X) = l(X) * q(X), where l(X) is the current (linear) eq polynomial and
    /// q(X) = c + dX + eX^2, given the following:
    /// - c, the constant term of q
    /// - e, the quadratic term of q
    /// - the previous round claim, s(0) + s(1)
    pub fn gruen_evals_deg_3(
        &self,
        q_constant: F,
        q_quadratic_coeff: F,
        s_0_plus_s_1: F,
    ) -> [F; 3] {
        // We want to compute the evaluations of the cubic polynomial s(X) = l(X) * q(X), where
        // l is linear, and q is quadratic, at the points {0, 2, 3}.
        //
        // At this point, we have
        // - the linear polynomial, l(X) = a + bX
        // - the quadratic polynomial, q(X) = c + dX + eX^2
        // - the previous round's claim s(0) + s(1) = a * c + (a + b) * (c + d + e)
        //
        // Both l and q are represented by their evaluations at 0 and infinity. I.e., we have a, b, c,
        // and e, but not d. We compute s by first computing l and q at points 2 and 3.

        // Evaluations of the linear polynomial
        let eq_eval_1 = self.current_scalar * self.w[self.current_index - 1];
        let eq_eval_0 = self.current_scalar - eq_eval_1;
        let eq_m = eq_eval_1 - eq_eval_0;
        let eq_eval_2 = eq_eval_1 + eq_m;
        let eq_eval_3 = eq_eval_2 + eq_m;

        // Evaluations of the quadratic polynomial
        let quadratic_eval_0 = q_constant;
        let cubic_eval_0 = eq_eval_0 * quadratic_eval_0;
        let cubic_eval_1 = s_0_plus_s_1 - cubic_eval_0;
        // q(1) = c + d + e
        let quadratic_eval_1 = cubic_eval_1 / eq_eval_1;
        // q(2) = c + 2d + 4e = q(1) + q(1) - q(0) + 2e
        let e_times_2 = q_quadratic_coeff + q_quadratic_coeff;
        let quadratic_eval_2 = quadratic_eval_1 + quadratic_eval_1 - quadratic_eval_0 + e_times_2;
        // q(3) = c + 3d + 9e = q(2) + q(1) - q(0) + 4e
        let quadratic_eval_3 =
            quadratic_eval_2 + quadratic_eval_1 - quadratic_eval_0 + e_times_2 + e_times_2;

        [
            cubic_eval_0,
            eq_eval_2 * quadratic_eval_2,
            eq_eval_3 * quadratic_eval_3,
        ]
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
        if self.current_index == 0 {
            return DensePolynomial::new(vec![self.current_scalar]);
        }

        let e_in = self.E_in_current();
        let e_out = self.E_out_current();

        // Low-to-high: remaining variables are w_0...w_{i-1}.
        // The initial split is w = [w_out, w_in, w_last].
        // w_out corresponds to lower-order bits of the hypercube, w_in to higher-order.
        // So e_out is the inner loop, e_in is the outer loop.
        let mut merged_evals = vec![F::zero(); e_in.len() * e_out.len()];
        for i in 0..e_in.len() {
            for j in 0..e_out.len() {
                merged_evals[i * e_out.len() + j] = e_in[i] * e_out[j];
            }
        }

        for val in &mut merged_evals {
            *val *= self.current_scalar;
        }

        DensePolynomial::new(merged_evals)
    }
}

/// GruenSplitEqPolynomial for High-to-Low (MSB to LSB) binding order
/// This binds variables starting from the most significant bit (index 0) down to the least significant bit
#[derive(Clone, Debug)]
pub struct GruenSplitEqPolynomialHighToLow<F> {
    pub(crate) current_index: usize,
    pub(crate) current_scalar: F,
    pub(crate) w: Vec<F>,
    pub(crate) E_in_vec: Vec<Vec<F>>,
    pub(crate) E_out_vec: Vec<Vec<F>>,
}

impl<F: JoltField> GruenSplitEqPolynomialHighToLow<F> {
    #[tracing::instrument(skip_all, name = "GruenSplitEqPolynomialHighToLow::new")]
    pub fn new(w: &[F]) -> Self {
        // For high-to-low binding, we bind from MSB (index 0) to LSB (index n-1).
        // We need to split the variables appropriately.
        // For a polynomial with variables [x0, x1, x2], in high-to-low:
        // - First we bind x0
        // - Then x1
        // - Finally x2
        // 
        // The split should be: w_in = first half, w_out = second half

        let (_, wprime) = w.split_first().unwrap();
        let m = w.len() / 2;
        let (w_in, w_out) = wprime.split_at(m);
        // [w_first <- w[0], w_in, w_out]
        
        let (E_in_vec, E_out_vec) = rayon::join(
            || EqPolynomial::evals_cached_rev(w_in),
            || EqPolynomial::evals_cached_rev(w_out),
        );

        Self {
            current_index: 0, // Start from 0 for high-to-low
            current_scalar: F::one(),
            w: w.to_vec(),
            E_in_vec,
            E_out_vec,
        }
    }

    #[tracing::instrument(skip_all, name = "GruenSplitEqPolynomialHighToLow::bind")]
    pub fn bind(&mut self, r: F) {
        // multiply `current_scalar` by `eq(w[i], r) = (1 - w[i]) * (1 - r) + w[i] * r`
        // which is the same as `1 - w[i] - r + 2 * w[i] * r`
        let prod_w_r = self.w[self.current_index] * r;
        self.current_scalar *= F::one() - self.w[self.current_index] - r + prod_w_r + prod_w_r;

        // increment `current_index` (going from 0 to n-1)
        self.current_index += 1;

        // pop the last vector from `E_in_vec` or `E_out_vec` (since we don't need it anymore)
        // For high-to-low, we bind variables in the first half first (E_in),
        // then variables in the second half (E_out)
        if self.current_index <= self.w.len() / 2 {
            // We're binding variables from the first half (E_in)
            self.E_in_vec.pop();
        } else if self.current_index <= self.w.len() {
            // We're binding variables from the second half (E_out)
            self.E_out_vec.pop();
        }
    }

    #[cfg(test)]
    pub fn merge(&self) -> DensePolynomial<F> {
        if self.current_index >= self.w.len() {
            return DensePolynomial::new(vec![self.current_scalar]);
        }
        
        let e_in = self.E_in_current();
        let e_out = self.E_out_current();
        
        // Handle the case where we haven't bound any variables yet
        if self.current_index == 0 {
            // E_in and E_out were built from w[1:] due to split_first
            // We need to expand to include w[0]
            
            // First compute eq(w[0], x) evaluations for x in {0, 1}
            let w0 = self.w[0];
            let eq_w0_evals = vec![F::one() - w0, w0]; // [eq(w[0], 0), eq(w[0], 1)]
            
            // Now create the full tensor product: eq(w[0], x0) * e_in[i] * e_out[j]
            let mut merged_evals = vec![F::zero(); 2 * e_in.len() * e_out.len()];
            
            for x0 in 0..2 {
                let offset = x0 * e_in.len() * e_out.len();
                for i in 0..e_in.len() {
                    for j in 0..e_out.len() {
                        merged_evals[offset + i * e_out.len() + j] = 
                            eq_w0_evals[x0] * e_in[i] * e_out[j] * self.current_scalar;
                    }
                }
            }
            
            return DensePolynomial::new(merged_evals);
        }
        
        // For other cases, the split handling has already incorporated w[0] via current_scalar
        let mut merged_evals = vec![F::zero(); e_in.len() * e_out.len()];
        
        if self.current_index <= self.w.len() / 2 {
            // Still binding in the first half
            for i in 0..e_in.len() {
                for j in 0..e_out.len() {
                    merged_evals[i * e_out.len() + j] = e_in[i] * e_out[j] * self.current_scalar;
                }
            }
        } else {
            // Binding in the second half - E_in is fully bound
            for j in 0..e_out.len() {
                for i in 0..e_in.len() {
                    merged_evals[j * e_in.len() + i] = e_in[i] * e_out[j] * self.current_scalar;
                }
            }
        }
        
        DensePolynomial::new(merged_evals)
    }

    pub fn len(&self) -> usize {
        1 << (self.w.len() - self.current_index)
    }

    pub fn get_bound_coeff(&self, index: usize) -> F {
        // This is equivalent to calling merge() and then getting the coefficient at index
        // For efficiency, we compute it directly using E_in and E_out
        
        if self.current_index == 0 {
            // Initial state - need to account for w[0] that was split off
            let e_in = self.E_in_current();
            let e_out = self.E_out_current();
            let e_out_len = e_out.len();
            let e_in_len = e_in.len();
            let total_half_size = e_in_len * e_out_len;
            
            // Determine which half of the hypercube we're in
            let w0_bit = index / total_half_size;
            let local_index = index % total_half_size;
            
            let i = local_index / e_out_len;
            let j = local_index % e_out_len;
            
            let w0 = self.w[0];
            let eq_w0_eval = if w0_bit == 0 {
                F::one() - w0  // eq(w[0], 0)
            } else {
                w0  // eq(w[0], 1)
            };
            
            return eq_w0_eval * e_in[i] * e_out[j] * self.current_scalar;
        }
        
        if self.current_index >= self.w.len() {
            return self.current_scalar;
        }

        if self.current_index <= self.w.len() / 2 {
            // Still binding in the first half
            let e_in = self.E_in_current();
            let e_out = self.E_out_current();
            let e_out_len = e_out.len();
            let i = index / e_out_len;
            let j = index % e_out_len;
            e_in[i] * e_out[j] * self.current_scalar
        } else {
            // Binding in the second half
            let e_in = self.E_in_current();
            let e_out = self.E_out_current();
            let e_in_len = e_in.len();
            let j = index / e_in_len;
            let i = index % e_in_len;
            e_in[i] * e_out[j] * self.current_scalar
        }
    }


    pub fn E_in_current_len(&self) -> usize {
        self.E_in_vec.last().map_or(0, |v| v.len())
    }

    pub fn E_out_current_len(&self) -> usize {
        self.E_out_vec.last().map_or(0, |v| v.len())
    }

    pub fn E_in_current(&self) -> &[F] {
        self.E_in_vec.last().map_or(&[], |v| v.as_slice())
    }

    pub fn E_out_current(&self) -> &[F] {
        self.E_out_vec.last().map_or(&[], |v| v.as_slice())
    }

    pub fn sumcheck_evals_array<const DEGREE: usize>(&self, index: usize) -> [F; DEGREE] {
        debug_assert!(DEGREE > 0);
        debug_assert!(index < self.len() / 2);

        let mut evals = [F::zero(); DEGREE];
        // For high-to-low binding order
        evals[0] = self.get_bound_coeff(index);
        if DEGREE == 1 {
            return evals;
        }
        let mut eval = self.get_bound_coeff(index + self.len() / 2);
        let m = eval - evals[0];
        for i in 1..DEGREE {
            eval += m;
            evals[i] = eval;
        }
        evals
    }

    pub fn gruen_evals_deg_2(&self, q_0: F, previous_claim: F) -> [F; 2] {
        // We want to compute the evaluations of the quadratic polynomial s(X) = l(X) * q(X), where
        // l is linear, and q is linear, at the points {0, 2}.
        //
        // At this point, we have:
        // - the linear polynomial, l(X) = a + bX
        // - the linear polynomial, q(X) = c + dX
        // - the previous round's claim s(0) + s(1) = a * c + (a + b) * (c + d)
        //
        // We have q(0) = c, and we need to compute q(1) and q(2).

        // Evaluations of the linear eq polynomial
        // For high-to-low, we bind from index 0 upward
        println!("W ARRAY:{:?}", self.w);
        let wi = self.w[self.current_index];
        let eq_eval_0 = self.current_scalar * (F::one() - wi);
        println!("wi: {:?}", wi);
        println!("current_scalar: {:?}", self.current_scalar);
        let eq_eval_1 = self.current_scalar * wi;
        let eq_m = eq_eval_1 - eq_eval_0;
        let eq_eval_2 = eq_eval_1 + eq_m;

        // Evaluations of the linear q polynomial
        let linear_eval_0 = q_0;
        let quadratic_eval_0 = eq_eval_0 * linear_eval_0;
        let quadratic_eval_1 = previous_claim - quadratic_eval_0;
        // q(1) = c + d
        let linear_eval_1 = quadratic_eval_1 / eq_eval_1;
        // q(2) = c + 2d = 2*q(1) - q(0)
        let linear_eval_2 = linear_eval_1 + linear_eval_1 - linear_eval_0;

        [quadratic_eval_0, eq_eval_2 * linear_eval_2]
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
        const NUM_VARS: usize = 3;
        let mut rng = test_rng();
        let w: Vec<Fr> = std::iter::repeat_with(|| Fr::random(&mut rng))
            .take(NUM_VARS)
            .collect();

        let mut regular_eq = DensePolynomial::new(EqPolynomial::evals(&w));
        let mut split_eq = GruenSplitEqPolynomial::new(&w);
        assert_eq!(regular_eq, split_eq.merge());

        println!("\n=== Initial state ===");
        println!("w vector length: {}", w.len());
        println!("Regular EQ poly len: {}", regular_eq.len());
        println!("Split EQ state:");
        println!("  current_index: {}", split_eq.current_index);
        println!("  current_scalar: {:?}", split_eq.current_scalar);
        println!("  w length: {}", split_eq.w.len());
        println!("  E_in_vec length: {}", split_eq.E_in_vec.len());
        println!("  E_out_vec length: {}", split_eq.E_out_vec.len());
        if !split_eq.E_in_vec.is_empty() {
            println!("  E_in_vec[0] length: {}", split_eq.E_in_vec[0].len());
        }
        if !split_eq.E_out_vec.is_empty() {
            println!("  E_out_vec[0] length: {}", split_eq.E_out_vec[0].len());
        }
        println!(
            "First few evals of regular_eq: {:?}",
            &regular_eq.Z[..8.min(regular_eq.len())]
        );

        for i in 0..NUM_VARS {
            println!("\n=== Iteration {} ===", i + 1);
            let r = Fr::random(&mut rng);
            println!("Binding with r = {:?}", r);

            println!("Before binding:");
            println!("  Regular EQ len: {}", regular_eq.len());
            println!(
                "  Split EQ - current_index: {}, current_scalar: {:?}",
                split_eq.current_index, split_eq.current_scalar
            );
            println!(
                "  E_in_vec.len(): {}, E_out_vec.len(): {}",
                split_eq.E_in_vec.len(),
                split_eq.E_out_vec.len()
            );

            regular_eq.bound_poly_var_bot(&r);
            split_eq.bind(r);

            println!("After binding:");
            println!("  Regular EQ len: {}", regular_eq.len());
            println!(
                "  Split EQ - current_index: {}, current_scalar: {:?}",
                split_eq.current_index, split_eq.current_scalar
            );
            println!(
                "  E_in_vec.len(): {}, E_out_vec.len(): {}",
                split_eq.E_in_vec.len(),
                split_eq.E_out_vec.len()
            );

            let merged = split_eq.merge();
            println!("  Merged poly len: {}", merged.len());

            if regular_eq.len() <= 8 {
                println!(
                    "  Regular EQ evals: {:?}",
                    &regular_eq.Z[..regular_eq.len()]
                );
                println!("  Merged evals: {:?}", &merged.Z[..merged.len()]);
            } else {
                println!("  First 4 regular EQ evals: {:?}", &regular_eq.Z[..4]);
                println!("  First 4 merged evals: {:?}", &merged.Z[..4]);
            }

            assert_eq!(regular_eq.Z[..regular_eq.len()], merged.Z[..merged.len()]);
        }
        println!("\n=== Test completed successfully ===");
    }

    #[test]
    fn bind_rev() {
        const NUM_VARS: usize = 10;
        let mut rng = test_rng();
        let w: Vec<Fr> = std::iter::repeat_with(|| Fr::random(&mut rng))
            .take(NUM_VARS)
            .collect();

        // Create regular polynomial with original w
        // @TODO(markosg04) this fails when trying to use regular_eq high to low
        let mut regular_eq = DensePolynomial::new(EqPolynomial::evals(&w));

        // Create split eq polynomial using new_rev with original w
        let mut split_eq_rev = GruenSplitEqPolynomial::new_rev(&w);

        // Verify they start equal
        assert_eq!(regular_eq, split_eq_rev.merge());

        // Bind with same random values
        for _ in 0..NUM_VARS {
            let r = Fr::random(&mut rng);
            regular_eq.bound_poly_var_top(&r);
            split_eq_rev.bind(r);

            let merged = split_eq_rev.merge();
            assert_eq!(regular_eq.Z[..regular_eq.len()], merged.Z[..merged.len()]);
        }
    }

    #[test]
    fn bind_new() {
        const NUM_VARS: usize = 3;
        let mut rng = test_rng();
        let w: Vec<Fr> = std::iter::repeat_with(|| Fr::random(&mut rng))
            .take(NUM_VARS)
            .collect();

        // Create regular polynomial with original w
        let mut regular_eq = DensePolynomial::new(EqPolynomial::evals(&w));

        // Create high-to-low split eq polynomial
        let mut split_eq_high_to_low = GruenSplitEqPolynomialHighToLow::new(&w);

        // Debug initial state
        println!("Initial state debug:");
        println!("  w.len() = {}", w.len());
        println!("  E_in_vec.len() = {}", split_eq_high_to_low.E_in_vec.len());
        println!("  E_out_vec.len() = {}", split_eq_high_to_low.E_out_vec.len());
        
        // Print all vector lengths in E_in_vec
        for (i, vec) in split_eq_high_to_low.E_in_vec.iter().enumerate() {
            println!("  E_in_vec[{}].len() = {}", i, vec.len());
        }
        
        // Print all vector lengths in E_out_vec
        for (i, vec) in split_eq_high_to_low.E_out_vec.iter().enumerate() {
            println!("  E_out_vec[{}].len() = {}", i, vec.len());
        }
        
        if !split_eq_high_to_low.E_in_vec.is_empty() {
            println!("  E_in_current().len() = {}", split_eq_high_to_low.E_in_current().len());
        }
        if !split_eq_high_to_low.E_out_vec.is_empty() {
            println!("  E_out_current().len() = {}", split_eq_high_to_low.E_out_current().len());
        }
        println!("  current_index = {}", split_eq_high_to_low.current_index);
        
        // Verify they start equal
        assert_eq!(regular_eq, split_eq_high_to_low.merge());

        println!("\n=== Testing High-to-Low binding ===");
        println!("w vector length: {}", w.len());
        println!("Regular EQ poly initial len: {}", regular_eq.len());
        println!("Split EQ High-to-Low state:");
        println!("  current_index: {}", split_eq_high_to_low.current_index);
        println!(
            "  current_scalar: {:?}",
            split_eq_high_to_low.current_scalar
        );

        // Bind with same random values, but regular_eq uses top and split uses new high-to-low
        for i in 0..NUM_VARS {
            println!("\n=== Iteration {} ===", i + 1);
            let r = Fr::random(&mut rng);
            println!("Binding with r = {:?}", r);

            println!("Before binding:");
            println!("  Regular EQ len: {}", regular_eq.len());
            println!(
                "  Split EQ - current_index: {}",
                split_eq_high_to_low.current_index
            );

            regular_eq.bound_poly_var_top(&r);
            split_eq_high_to_low.bind(r);

            println!("After binding:");
            println!("  Regular EQ len: {}", regular_eq.len());
            println!(
                "  Split EQ - current_index: {}",
                split_eq_high_to_low.current_index
            );

            let merged = split_eq_high_to_low.merge();
            println!("  Merged poly len: {}", merged.len());

            if regular_eq.len() <= 8 {
                println!(
                    "  Regular EQ evals: {:?}",
                    &regular_eq.Z[..regular_eq.len()]
                );
                println!("  Merged evals: {:?}", &merged.Z[..merged.len()]);
            } else {
                println!("  First 4 regular EQ evals: {:?}", &regular_eq.Z[..4]);
                println!("  First 4 merged evals: {:?}", &merged.Z[..4]);
            }

            assert_eq!(regular_eq.Z[..regular_eq.len()], merged.Z[..merged.len()]);
        }
        println!("\n=== High-to-Low test completed successfully ===");
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
            println!("Testing for {num_vars} variables");

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
            cached_E_out1[w_E_out_vars_expected1.len()]
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
        let num_x_in_vars_2 = N - num_x_out_vars_2; // L0 is 0
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
