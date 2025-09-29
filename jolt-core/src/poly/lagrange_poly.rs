use crate::field::JoltField;
use std::marker::PhantomData;
use std::ops::{Add, Mul};

/// Lagrange polynomials over zero-centered, symmetric, consecutive-integer domain, i.e.
/// grids like [-6, -5, ..., 6, 7].
/// This is the high-degree univariate analogue of EqPolynomial, which are for multilinear polynomials.
/// We use this in the univariate skip optimization in Spartan's outer sum-check.
pub struct LagrangePolynomial<F: JoltField>(PhantomData<F>);

impl<F: JoltField> LagrangePolynomial<F> {
    /// Evaluate a Lagrange-interpolated polynomial at point `r` given values on symmetric grid.
    /// Input: `values[i] = p(start + i)` where `start = -floor((N-1)/2)`.
    /// Returns: `p(r)` using Lagrange interpolation.
    #[inline]
    pub fn evaluate<const N: usize>(values: &[F; N], r: &F) -> F {
        let basis = Self::evals::<N>(r);
        values.iter().zip(basis.iter()).map(|(v, b)| *v * b).sum()
    }

    /// Compute all Lagrange basis polynomial values `[L_i(r)]` at point `r` for symmetric grid of size `N`.
    /// Grid nodes are `{start, start+1, ..., start+N-1}` where `start = -floor((N-1)/2)`.
    /// Returns: `[L_0(r), L_1(r), ..., L_{N-1}(r)]` such that `p(r) = sum_i L_i(r) * p(x_i)`.
    ///
    /// **Constraint**: N must be ≤ 2^30 to avoid i64 overflow in symmetric domain calculations.
    #[inline]
    pub fn evals<const N: usize>(r: &F) -> [F; N] {
        const MAX_N: usize = 1 << 32; // 2^32, ensures (N-1)/2 fits in i64
        debug_assert!(N <= MAX_N, "N={N} exceeds maximum safe value {MAX_N}");
        debug_assert!(N > 0, "N must be positive");
        let d = N - 1;
        let start: i64 = -((d / 2) as i64);

        // Build nodes xs
        let mut xs = [F::zero(); N];
        let mut i = 0usize;
        while i < N {
            let t = start + (i as i64);
            xs[i] = F::from_i64(t);
            i += 1;
        }

        // If r equals some node, return one-hot
        let mut j = 0usize;
        while j < N {
            if *r == xs[j] {
                let mut out = [F::zero(); N];
                out[j] = F::one();
                return out;
            }
            j += 1;
        }

        // Compute 1/(r - x_i) for all i using prefix/suffix products with a single inversion
        let mut dists = [F::zero(); N];
        i = 0;
        while i < N {
            dists[i] = *r - xs[i];
            i += 1;
        }

        let mut p: Vec<F> = Vec::with_capacity(N + 1);
        p.push(F::one());
        i = 0;
        while i < N {
            let next = p[i] * dists[i];
            p.push(next);
            i += 1;
        }
        let mut s: Vec<F> = vec![F::one(); N + 1];
        let mut tix: isize = (N as isize) - 2;
        while tix >= 0 {
            let ui = (tix + 1) as usize;
            s[tix as usize] = s[ui] * dists[ui];
            tix -= 1;
        }
        let inv_prod = p[N].inverse().unwrap();

        // Compute barycentric weights w_i = 1 / prod_{j!=i} (x_i - x_j) in O(N).
        // For consecutive-integer nodes (unit spacing),
        //   prod_{j!=i} (x_i - x_j) = i! * (-1)^{N-1-i} * (N-1-i)!
        // which is independent of the shift `start`.
        let mut fact = [F::one(); N];
        let mut k: usize = 1;
        while k < N {
            fact[k] = fact[k - 1].mul_u64(k as u64);
            k += 1;
        }
        let mut denoms = [F::zero(); N];
        i = 0;
        while i < N {
            let mut denom = fact[i] * fact[N - 1 - i];
            if ((N - 1 - i) & 1) == 1 {
                denom = -denom;
            }
            denoms[i] = denom;
            i += 1;
        }
        // Invert all denominators with one inversion via prefix/suffix products
        let mut pref: Vec<F> = Vec::with_capacity(N + 1);
        pref.push(F::one());
        i = 0;
        while i < N {
            let next = pref[i] * denoms[i];
            pref.push(next);
            i += 1;
        }
        let mut suff: Vec<F> = vec![F::one(); N + 1];
        let mut idx: isize = (N as isize) - 1;
        while idx >= 0 {
            let ui = (idx + 1) as usize;
            suff[idx as usize] = suff[ui] * denoms[idx as usize];
            idx -= 1;
        }
        let inv_total = pref[N].inverse().unwrap();
        let mut ws = [F::zero(); N];
        i = 0;
        while i < N {
            ws[i] = pref[i] * suff[i + 1] * inv_total; // = 1 / denoms[i]
            i += 1;
        }

        let mut num = [F::zero(); N];
        let mut sum = F::zero();
        i = 0;
        while i < N {
            let inv_di = p[i] * s[i] * inv_prod;
            let term = ws[i] * inv_di;
            num[i] = term;
            sum += term;
            i += 1;
        }
        let inv_sum = sum.inverse().unwrap();
        let mut outv = [F::zero(); N];
        i = 0;
        while i < N {
            outv[i] = num[i] * inv_sum;
            i += 1;
        }
        outv
    }

    /// Compute evaluations of the interpolated polynomial at multiple points.
    /// Input: `values` on symmetric grid, `points` to evaluate at.
    /// Returns: `[p(points[0]), p(points[1]), ...]`.
    pub fn evaluate_many<const N: usize>(values: &[F; N], points: &[F]) -> Vec<F> {
        if points.is_empty() {
            return Vec::new();
        }

        // For many evaluation points, it's more efficient to interpolate to monomial form once
        // and then evaluate using Horner's method, rather than computing Lagrange basis for each point
        if points.len() > N {
            let coeffs = Self::interpolate_coeffs(values);
            points
                .iter()
                .map(|r| {
                    // Horner's method: p(r) = c_0 + r*(c_1 + r*(c_2 + ... + r*c_{n-1}))
                    let mut result = coeffs[N - 1];
                    for i in (0..N - 1).rev() {
                        result = result * r + coeffs[i];
                    }
                    result
                })
                .collect()
        } else {
            // For few evaluation points, direct Lagrange evaluation is faster
            points
                .iter()
                .map(|r| {
                    let basis = Self::evals::<N>(r);
                    values.iter().zip(basis.iter()).map(|(v, b)| *v * b).sum()
                })
                .collect()
        }
    }

    /// Interpolate monomial coefficients from values on the symmetric consecutive-integer grid.
    /// Input: `values[i] = p(start + i)` where `start = -floor((N-1)/2)`.
    /// Output: coefficients `[c_0, c_1, ..., c_{N-1}]` with `p(x) = sum_j c_j * x^j`.
    ///
    /// **Constraint**: N must be ≤ 2^30 to avoid i64 overflow in symmetric domain calculations.
    #[inline]
    pub fn interpolate_coeffs<const N: usize>(values: &[F; N]) -> [F; N] {
        const MAX_N: usize = 1 << 32; // 2^32, ensures (N-1)/2 fits in i64
        debug_assert!(N <= MAX_N, "N={MAX_N} exceeds maximum safe value {MAX_N}");
        debug_assert!(N > 0, "N must be positive");
        let d = N - 1;
        let start: i64 = -((d / 2) as i64);

        // Precompute inverses of small integers 1..d using one inversion (batch/Montgomery trick)
        // Use mul_u64 semantics wherever possible.
        let mut smalls = [0u64; N];
        let mut pref = [F::one(); N]; // pref[i] = product_{k=1..i} k, pref[0]=1
        let mut m: usize = 1;
        while m <= d {
            smalls[m] = m as u64;
            pref[m] = pref[m - 1].mul_u64(smalls[m]);
            m += 1;
        }
        let inv_total = pref[d].inverse().unwrap(); // one inversion
        let mut right = F::one();
        let mut invs = [F::zero(); N];
        // invs[m] = 1/m for m=1..d
        let mut i: isize = d as isize;
        while i >= 1 {
            let idx = i as usize;
            // 1/m = (pref[m-1] * right) * (1/pref[d])
            invs[idx] = pref[idx - 1] * right * inv_total;
            right = right.mul_u64(smalls[idx]);
            i -= 1;
        }

        // Divided differences (Newton coefficients)
        let mut dd = *values; // dd[i] holds current order differences
        let mut newton = [F::zero(); N];
        newton[0] = dd[0];
        let mut order: usize = 1;
        while order <= d {
            let inv = invs[order];
            let mut i: usize = 0;
            while i + order < N {
                // denominator (x_{i+order} - x_i) = order (since nodes are consecutive)
                dd[i] = (dd[i + 1] - dd[i]) * inv;
                i += 1;
            }
            newton[order] = dd[0];
            order += 1;
        }

        // Convert Newton form to monomial coefficients
        let mut coeffs = [F::zero(); N];
        let mut basis = [F::zero(); N];
        basis[0] = F::one();
        let mut deg: usize = 0;
        let mut k: usize = 0;
        while k < N {
            // coeffs += newton[k] * basis
            let scale = newton[k];
            let mut j: usize = 0;
            while j <= deg {
                coeffs[j] += scale * basis[j];
                j += 1;
            }

            if k == d {
                break;
            }

            // Update basis ← basis * (x - (start + k))
            let a: i64 = start + (k as i64);
            // Save old highest coef (becomes new leading term)
            let last = basis[deg];
            let mut t: isize = deg as isize;
            while t >= 1 {
                let idx = t as usize;
                let old = basis[idx];
                let term = old.mul_i64(a);
                basis[idx] = basis[idx - 1] - term;
                t -= 1;
            }
            // t == 0 case
            basis[0] = -basis[0].mul_i64(a);
            deg += 1;
            basis[deg] = last;

            k += 1;
        }

        coeffs
    }
}

/// A collection of helper and utility functions for Lagrange interpolation,
/// binomial coefficients, and polynomial extrapolation over integer domains.
pub struct LagrangeHelper;

impl LagrangeHelper {
    // ===== Binomial coefficient utilities =====

    /// Compute binomial coefficient C(n, k) with k reduced to min(k, n-k).
    #[inline]
    pub const fn binomial_coeff(n: usize, k: usize) -> u64 {
        let kk = if k <= n - k { k } else { n - k };
        let mut i = 0usize;
        let mut res: u128 = 1u128;
        while i < kk {
            // res = res * (n - i) / (i + 1)
            let num = (n - i) as u128;
            let den = (i + 1) as u128;
            res = (res * num) / den;
            i += 1;
        }
        res as u64
    }

    /// Generalized binomial coefficient for integer t and k >= 0.
    /// Supports negative t via identity: C(t, k) = (-1)^k C(-t + k - 1, k).
    #[inline]
    pub const fn generalized_binomial(t: i64, k: usize) -> i128 {
        if k == 0 {
            return 1;
        }
        if t >= 0 {
            let tt = t as i128;
            if (k as i128) > tt {
                return 0;
            }
            let mut num: i128 = 1;
            let mut den: i128 = 1;
            let mut j: usize = 0;
            while j < k {
                num *= tt - (j as i128);
                den *= (j as i128) + 1;
                j += 1;
            }
            num / den
        } else {
            let sign = if (k & 1) == 1 { -1i128 } else { 1i128 };
            let tt = (-t) as i128 + (k as i128) - 1;
            let mut num: i128 = 1;
            let mut den: i128 = 1;
            let mut j: usize = 0;
            while j < k {
                num *= tt - (j as i128);
                den *= (j as i128) + 1;
                j += 1;
            }
            sign * (num / den)
        }
    }

    /// Lagrange shift coefficients for evaluating at integer `shift` from a window of length N.
    /// Given base values p(0), p(1), ..., p(N-1), returns alphas such that:
    /// p(shift) = sum_{i=0}^{N-1} alpha[i] * p(i)
    #[inline]
    pub const fn shift_coeffs_i32<const N: usize>(shift: i64) -> [i32; N] {
        let mut out = [0i32; N];
        let n_minus_1 = (N - 1) as i64;
        let mut i: usize = 0;
        while i < N {
            let s1 = Self::generalized_binomial(shift, i);
            let s2 = Self::generalized_binomial(shift - (i as i64) - 1, (N - 1) - i);
            let sign = if (((n_minus_1 as usize) - i) & 1) == 1 {
                -1i128
            } else {
                1i128
            };
            let val = sign * s1 * s2;
            out[i] = val as i32;
            i += 1;
        }
        out
    }

    /// Same as shift_coeffs_i32 but returns i128 for higher precision.
    #[inline]
    pub const fn shift_coeffs_i128<const N: usize>(shift: i64) -> [i128; N] {
        let mut out = [0i128; N];
        let n_minus_1 = (N - 1) as i64;
        let mut i = 0usize;
        while i < N {
            let s1 = Self::generalized_binomial(shift, i);
            let s2 = Self::generalized_binomial(shift - (i as i64) - 1, (N - 1) - i);
            let sign = if (((n_minus_1 as usize) - i) & 1) == 1 {
                -1i128
            } else {
                1i128
            };
            out[i] = sign * s1 * s2;
            i += 1;
        }
        out
    }

    // ===== Extension and extrapolation utilities =====

    /// Extrapolate the next value p(x+n) from n consecutive values p(x), p(x+1), ..., p(x+n-1).
    #[inline]
    pub fn extrapolate_next<F: JoltField>(prev: &[F]) -> F {
        let n = prev.len();
        let mut acc = F::zero();
        for i in 0..n {
            let c = Self::binomial_coeff(n, i);
            let coef = F::from_u64(c);
            let term = prev[i] * coef;
            if ((n - 1 - i) & 1) == 1 {
                acc -= term;
            } else {
                acc += term;
            }
        }
        acc
    }

    /// Compute p(x-1) from a length-N window [p(x), p(x+1), ..., p(x+N-1)].
    #[inline]
    pub fn backward_step<const N: usize, F: JoltField>(window: &[F; N]) -> F {
        let mut acc = F::zero();
        for j in 0..N {
            let c = Self::binomial_coeff(N, j + 1);
            let coef = F::from_u64(c);
            let term = window[j] * coef;
            if (j & 1) == 1 {
                acc -= term;
            } else {
                acc += term;
            }
        }
        acc
    }

    /// Extend consecutive symmetric window by N new values (alternating left/right).
    #[inline]
    pub fn extend_consecutive_symmetric<const N: usize, F: JoltField>(base: &[F; N]) -> [F; N] {
        debug_assert!(N >= 1 && (N % 2 == 1));

        // Build forward-difference diagonal at left edge (i = 0): diffs[k] = Δ^k f(0), k=1..N-1
        let mut work = *base;
        let mut diffs = [F::zero(); N];
        for k in 1..N {
            for i in 0..(N - k) {
                work[i] = work[i + 1] - work[i];
            }
            diffs[k] = work[0];
        }

        // Left-side extension: step N_left = (N+1)/2 times backwards
        let left_cnt = N.div_ceil(2);
        let mut out = [F::zero(); N];
        let mut left_d = diffs;
        let mut left_anchor = base[0];
        let mut left_vals = [F::zero(); N];
        for s in 0..left_cnt {
            // Backward update of differences: d_k <- d_k - d_{k+1}
            for k in (1..(N - 1)).rev() {
                left_d[k] -= left_d[k + 1];
            }
            left_anchor -= left_d[1];
            left_vals[s] = left_anchor;
        }

        // Right-side extension: first advance differences to the right boundary (i = N-1)
        let right_cnt = N / 2;
        let mut right_d = diffs;
        for _ in 0..(N - 1) {
            for k in (1..(N - 1)).rev() {
                right_d[k] += right_d[k + 1];
            }
        }
        let mut right_anchor = base[N - 1];
        let mut right_vals = [F::zero(); N];
        for s in 0..right_cnt {
            right_anchor += right_d[1];
            right_vals[s] = right_anchor;
            for k in (1..(N - 1)).rev() {
                right_d[k] += right_d[k + 1];
            }
        }

        // Interleave: [-1, N, -2, N+1, ...]
        for i in 0..right_cnt {
            out[2 * i] = left_vals[i];
            out[2 * i + 1] = right_vals[i];
        }
        if left_cnt > right_cnt {
            out[2 * right_cnt] = left_vals[right_cnt];
        }
        out
    }

    /// N is the number of nodes, degree D = N-1. Returns [S_0, S_1, ..., S_D] as i64.
    /// S_k = sum_{t=start..start+D} t^k, where start = -floor(D/2).
    /// For N <= 16 and k <= 15, values safely fit in i64.
    #[inline]
    pub const fn power_sums<const N: usize>() -> [i64; N] {
        debug_assert!(N <= 16, "N exceeds maximum safe value 16");
        let d = N - 1;
        let start: i64 = -((d / 2) as i64);
        let mut sums = [0i64; N];
        let mut j: usize = 0;
        while j < N {
            let t = start + (j as i64);
            // k = 0
            sums[0] += 1;
            // k >= 1 up to D
            let mut pow = t; // t^1
            let mut k: usize = 1;
            while k <= d {
                sums[k] += pow;
                pow *= t;
                k += 1;
            }
            j += 1;
        }
        sums
    }

    /// Extend evaluations from symmetric base window by D points using integer coefficients.
    /// Input: `base_evals[i] = p(start + i)` where `start = -floor(D/2)`, `N = D+1`.
    /// Output: extended evaluations of length D at points outside the base window.
    #[inline]
    pub fn extend_evals_symmetric<const D: usize, const N: usize, T>(base_evals: &[T; N]) -> [T; D]
    where
        T: Copy + Add<Output = T> + Mul<i32, Output = T> + Default,
    {
        debug_assert!(N == D + 1);
        let d_i64: i64 = D as i64;
        let start: i64 = -((D / 2) as i64);

        // Split negative/positive counts relative to the base window
        let neg_cnt: usize = (d_i64 + start) as usize; // number of points in [-D .. start-1]
        let pos_cnt: usize = D - neg_cnt; // number of points in [start+D+1 .. D]

        let mut out: [T; D] = [Default::default(); D];

        // Negative side: t in [-D .. start-1]
        let mut j: usize = 0;
        while j < neg_cnt {
            let t = -d_i64 + (j as i64);
            let shift = t - start; // evaluate g(shift) where g(i) = p(start + i)
            let coeffs = Self::shift_coeffs_i32::<N>(shift);
            let mut acc: T = Default::default();
            let mut i = 0usize;
            while i < N {
                acc = acc + (base_evals[i] * coeffs[i]);
                i += 1;
            }
            out[j] = acc;
            j += 1;
        }

        // Positive side: t in [start+D+1 .. D]
        let mut k: usize = 0;
        while k < pos_cnt {
            let t = start + d_i64 + 1 + (k as i64);
            let shift = t - start;
            let coeffs = Self::shift_coeffs_i32::<N>(shift);
            let mut acc: T = Default::default();
            let mut i = 0usize;
            while i < N {
                acc = acc + (base_evals[i] * coeffs[i]);
                i += 1;
            }
            out[neg_cnt + k] = acc;
            k += 1;
        }

        out
    }
}

/// Specialized optimizations for degree-13 Lagrange polynomials.
/// Kept separate for performance-critical paths in univariate skip.
/// NOTE: if we add more constraints in the future, we will need to change this to degree 14, 15, etc.
/// But no other changes should be needed.
pub struct Degree13Lagrange;

impl Degree13Lagrange {
    /// Build the full row [C(n,0), C(n,1), ..., C(n,n)] as an array of length N = n+1.
    const fn row_const<const N: usize>() -> [u64; N] {
        // N must be at least 1 (so n = N-1 is valid). Our usages satisfy this.
        let n = N - 1;
        let mut out = [0u64; N];
        let mut i = 0usize;
        while i < N {
            out[i] = LagrangeHelper::binomial_coeff(n, i);
            i += 1;
        }
        out
    }

    /// Precomputed binomial row C(14,k) for recurrence over N=14 window.
    pub const BINOMIAL_ROW_14: [u64; 15] = Self::row_const::<15>();

    /// Precomputed Lagrange coefficients for common degree-13 evaluation points
    pub const AT_NEG7: [i32; 14] = LagrangeHelper::shift_coeffs_i32::<14>(-1);
    pub const AT_NEG8: [i32; 14] = LagrangeHelper::shift_coeffs_i32::<14>(-2);
    pub const AT_NEG9: [i32; 14] = LagrangeHelper::shift_coeffs_i32::<14>(-3);
    pub const AT_NEG10: [i32; 14] = LagrangeHelper::shift_coeffs_i32::<14>(-4);
    pub const AT_NEG11: [i32; 14] = LagrangeHelper::shift_coeffs_i32::<14>(-5);
    pub const AT_NEG12: [i32; 14] = LagrangeHelper::shift_coeffs_i32::<14>(-6);
    pub const AT_NEG13: [i32; 14] = LagrangeHelper::shift_coeffs_i32::<14>(-7);
    pub const AT_9: [i32; 14] = LagrangeHelper::shift_coeffs_i32::<14>(15);
    pub const AT_10: [i32; 14] = LagrangeHelper::shift_coeffs_i32::<14>(16);
    pub const AT_11: [i32; 14] = LagrangeHelper::shift_coeffs_i32::<14>(17);
    pub const AT_12: [i32; 14] = LagrangeHelper::shift_coeffs_i32::<14>(18);
    pub const AT_13: [i32; 14] = LagrangeHelper::shift_coeffs_i32::<14>(19);

    /// Fast extension for boolean evaluations (returns i32 for exact arithmetic).
    #[inline]
    pub fn extend_bool_evals(base_evals: &[bool; 14]) -> [i32; 13] {
        let c = &Self::BINOMIAL_ROW_14;

        // Current windows for left/backward and right/forward stepping
        let mut left_w: [i32; 14] = [0; 14];
        let mut right_w: [i32; 14] = [0; 14];
        for i in 0..14 {
            let v = if base_evals[i] { 1 } else { 0 };
            left_w[i] = v;
            right_w[i] = v;
        }

        // p(x-1) using N=14 backward recurrence with explicit signs
        let prev_left = |w: &[i32; 14]| -> i32 {
            let c1 = c[1] as i32;
            let c2 = c[2] as i32;
            let c3 = c[3] as i32;
            let c4 = c[4] as i32;
            let c5 = c[5] as i32;
            let c6 = c[6] as i32;
            let c7 = c[7] as i32;
            let c8 = c[8] as i32;
            let c9 = c[9] as i32;
            let c10 = c[10] as i32;
            let c11 = c[11] as i32;
            let c12 = c[12] as i32;
            let c13 = c[13] as i32;
            let c14 = c[14] as i32;
            c1 * w[0] - c2 * w[1] + c3 * w[2] - c4 * w[3] + c5 * w[4] - c6 * w[5] + c7 * w[6]
                - c8 * w[7]
                + c9 * w[8]
                - c10 * w[9]
                + c11 * w[10]
                - c12 * w[11]
                + c13 * w[12]
                - c14 * w[13]
        };

        // p(x+14) using N=14 forward recurrence grouped like ex8/ex16
        let next_right = |w: &[i32; 14]| -> i32 {
            let c0 = c[0] as i32;
            let c1 = c[1] as i32;
            let c2 = c[2] as i32;
            let c3 = c[3] as i32;
            let c4 = c[4] as i32;
            let c5 = c[5] as i32;
            let c6 = c[6] as i32;
            let c7 = c[7] as i32;
            let sp1 = w[1] + w[13];
            let sp3 = w[3] + w[11];
            let sp5 = w[5] + w[9];
            let sn2 = w[2] + w[12];
            let sn4 = w[4] + w[10];
            let sn6 = w[6] + w[8];
            c1 * sp1 + c3 * sp3 + c5 * sp5 + c7 * w[7] - c0 * w[0] - c2 * sn2 - c4 * sn4 - c6 * sn6
        };

        let mut out: [i32; 13] = [0; 13];
        // Produce 6 interleaved pairs, then one final left value
        for i in 0..6 {
            // left: x -> x-1
            let l = prev_left(&left_w);
            out[2 * i] = l;
            // shift left window right and insert new left at position 0
            for k in (1..14).rev() {
                left_w[k] = left_w[k - 1];
            }
            left_w[0] = l;

            // right: x+13 -> x+14
            let r = next_right(&right_w);
            out[2 * i + 1] = r;
            // shift right window left and append new right at the end
            for k in 0..13 {
                right_w[k] = right_w[k + 1];
            }
            right_w[13] = r;
        }
        // Final left value (-13)
        let l_last = prev_left(&left_w);
        out[12] = l_last;

        out
    }

    /// Fast extension for i128 evaluations.
    #[inline]
    pub fn extend_i128_evals(base_evals: &[i128; 14]) -> [i128; 13] {
        let c = &Self::BINOMIAL_ROW_14;

        let mut left_w: [i128; 14] = [0; 14];
        let mut right_w: [i128; 14] = [0; 14];
        for i in 0..14 {
            left_w[i] = base_evals[i];
            right_w[i] = base_evals[i];
        }

        let prev_left = |w: &[i128; 14]| -> i128 {
            let c1 = c[1] as i128;
            let c2 = c[2] as i128;
            let c3 = c[3] as i128;
            let c4 = c[4] as i128;
            let c5 = c[5] as i128;
            let c6 = c[6] as i128;
            let c7 = c[7] as i128;
            let c8 = c[8] as i128;
            let c9 = c[9] as i128;
            let c10 = c[10] as i128;
            let c11 = c[11] as i128;
            let c12 = c[12] as i128;
            let c13 = c[13] as i128;
            let c14 = c[14] as i128;
            c1 * w[0] - c2 * w[1] + c3 * w[2] - c4 * w[3] + c5 * w[4] - c6 * w[5] + c7 * w[6]
                - c8 * w[7]
                + c9 * w[8]
                - c10 * w[9]
                + c11 * w[10]
                - c12 * w[11]
                + c13 * w[12]
                - c14 * w[13]
        };

        let next_right = |w: &[i128; 14]| -> i128 {
            let c0 = c[0] as i128;
            let c1 = c[1] as i128;
            let c2 = c[2] as i128;
            let c3 = c[3] as i128;
            let c4 = c[4] as i128;
            let c5 = c[5] as i128;
            let c6 = c[6] as i128;
            let c7 = c[7] as i128;
            let sp1 = w[1] + w[13];
            let sp3 = w[3] + w[11];
            let sp5 = w[5] + w[9];
            let sn2 = w[2] + w[12];
            let sn4 = w[4] + w[10];
            let sn6 = w[6] + w[8];
            c1 * sp1 + c3 * sp3 + c5 * sp5 + c7 * w[7] - c0 * w[0] - c2 * sn2 - c4 * sn4 - c6 * sn6
        };

        let mut out: [i128; 13] = [0; 13];
        for i in 0..6 {
            let l = prev_left(&left_w);
            out[2 * i] = l;
            for k in (1..14).rev() {
                left_w[k] = left_w[k - 1];
            }
            left_w[0] = l;

            let r = next_right(&right_w);
            out[2 * i + 1] = r;
            for k in 0..13 {
                right_w[k] = right_w[k + 1];
            }
            right_w[13] = r;
        }
        let l_last = prev_left(&left_w);
        out[12] = l_last;
        out
    }
}
