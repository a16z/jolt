use std::ops::{Add, Mul, Sub};

/// Barycentric Lagrange interpolation evaluators and small-degree weight tables.
pub mod barycentric {
    use crate::field::JoltField;

    /// Compute naive barycentric weights given arbitrary distinct nodes `xs`.
    /// Complexity: O(n^2). For symmetric integer grids, prefer closed-form weights.
    pub fn compute_weights<F: JoltField>(xs: &[F]) -> Vec<F> {
        let n = xs.len();
        let mut ws = vec![F::zero(); n];
        for i in 0..n {
            let mut denom = F::one();
            let xi = xs[i];
            for j in 0..n {
                if i != j {
                    denom *= xi - xs[j];
                }
            }
            ws[i] = denom.inverse().unwrap();
        }
        ws
    }

    /// Safe evaluator: if T == 0 in the unnormalized ratio, fall back to a derivative-based
    /// evaluation using one additional inversion. Returns the correct value for all `x`.
    pub fn eval_safe<F: JoltField>(x: &F, xs: &[F], ys: &[F], ws: &[F]) -> F {
        debug_assert_eq!(xs.len(), ys.len());
        debug_assert_eq!(xs.len(), ws.len());
        let n = xs.len();

        // d[i] = x - x_i
        let mut d = Vec::with_capacity(n);
        for i in 0..n {
            let di = *x - xs[i];
            if di == F::zero() {
                return ys[i];
            }
            d.push(di);
        }

        // Prefix products P[0..=n]: P[0]=1, P[i+1] = P[i] * d[i]
        let mut p = Vec::with_capacity(n + 1);
        p.push(F::one());
        for i in 0..n {
            p.push(p[i] * d[i]);
        }

        // Suffix products S[0..=n]: S[n]=1, S[i] = S[i+1] * d[i+1]
        let mut s = vec![F::one(); n + 1];
        for i in (0..n).rev() {
            let next = if i + 1 < n { d[i + 1] } else { F::one() };
            s[i] = s[i + 1] * next;
        }

        let mut numerator = F::zero();
        let mut denominator = F::zero();
        for i in 0..n {
            let denom_except_i = p[i] * s[i];
            numerator += ws[i] * ys[i] * denom_except_i;
            denominator += ws[i] * denom_except_i;
        }

        if denominator != F::zero() {
            return numerator * denominator.inverse().unwrap();
        }

        // Fallback: compute inv_d_i via Montgomery trick using one inversion of D = prod d_i
        let d_prod = p[n];
        let inv_d_prod = d_prod.inverse().unwrap();

        let mut s2 = F::zero();
        let mut s1 = F::zero();
        let mut s2p = F::zero();
        let mut s1p = F::zero();
        for i in 0..n {
            let denom_except_i = p[i] * s[i];
            let inv_di = denom_except_i * inv_d_prod;
            s2 += ws[i] * inv_di;
            s1 += ws[i] * ys[i] * inv_di;
            let inv_di2 = inv_di * inv_di;
            s2p -= ws[i] * inv_di2;
            s1p -= ws[i] * ys[i] * inv_di2;
        }

        if s2 != F::zero() {
            return s1 * s2.inverse().unwrap();
        }
        s1p * s2p.inverse().unwrap()
    }

    /// Const-generic convenience wrapper (safe variant) over arrays of length N (nodes count).
    /// Note: polynomial degree is D = N - 1.
    pub fn eval_safe_const<F: JoltField, const N: usize>(
        x: &F,
        xs: &[F; N],
        ys: &[F; N],
        ws: &[F; N],
    ) -> F {
        eval_safe(x, xs, ys, ws)
    }

    /// Precomputed rational weight tables for small degrees over the integer node grids.
    /// Node convention:
    /// - Let degree be `d`, number of nodes is `n = d + 1`.
    /// - Nodes are the consecutive integers from `start = -floor(d/2)` to `start + d`.
    ///   Examples: d=2 -> [-1,0,1]; d=3 -> [-1,0,1,2]; d=4 -> [-2,-1,0,1,2].
    pub mod tables {
        use crate::field::JoltField;

        #[derive(Copy, Clone)]
        pub struct RationalWeight {
            pub num: i32,
            pub den: i32,
        }

        pub const D2_WEIGHTS_RATIONAL: [RationalWeight; 3] = [
            RationalWeight { num: 1, den: 2 },
            RationalWeight { num: -1, den: 1 },
            RationalWeight { num: 1, den: 2 },
        ];

        pub const D3_WEIGHTS_RATIONAL: [RationalWeight; 4] = [
            RationalWeight { num: -1, den: 6 },
            RationalWeight { num: 1, den: 2 },
            RationalWeight { num: -1, den: 2 },
            RationalWeight { num: 1, den: 6 },
        ];

        pub const D4_WEIGHTS_RATIONAL: [RationalWeight; 5] = [
            RationalWeight { num: 1, den: 24 },
            RationalWeight { num: -1, den: 6 },
            RationalWeight { num: 1, den: 4 },
            RationalWeight { num: -1, den: 6 },
            RationalWeight { num: 1, den: 24 },
        ];

        pub const D5_WEIGHTS_RATIONAL: [RationalWeight; 6] = [
            RationalWeight { num: -1, den: 120 },
            RationalWeight { num: 1, den: 24 },
            RationalWeight { num: -1, den: 12 },
            RationalWeight { num: 1, den: 12 },
            RationalWeight { num: -1, den: 24 },
            RationalWeight { num: 1, den: 120 },
        ];

        pub const D6_WEIGHTS_RATIONAL: [RationalWeight; 7] = [
            RationalWeight { num: 1, den: 720 },
            RationalWeight { num: -1, den: 120 },
            RationalWeight { num: 1, den: 48 },
            RationalWeight { num: -1, den: 36 },
            RationalWeight { num: 1, den: 48 },
            RationalWeight { num: -1, den: 120 },
            RationalWeight { num: 1, den: 720 },
        ];

        pub const D7_WEIGHTS_RATIONAL: [RationalWeight; 8] = [
            RationalWeight { num: -1, den: 5040 },
            RationalWeight { num: 1, den: 720 },
            RationalWeight { num: -1, den: 240 },
            RationalWeight { num: 1, den: 144 },
            RationalWeight { num: -1, den: 144 },
            RationalWeight { num: 1, den: 240 },
            RationalWeight { num: -1, den: 720 },
            RationalWeight { num: 1, den: 5040 },
        ];

        /// Materialize d=2 weights as field elements. One inversion.
        pub fn weights_d2<F: JoltField>() -> [F; 3] {
            let inv2 = F::from_u64(2).inverse().unwrap();
            let neg_one = -F::one();
            [inv2, neg_one, inv2]
        }

        /// Materialize d=3 weights as field elements. One inversion (invert 6).
        pub fn weights_d3<F: JoltField>() -> [F; 4] {
            let inv6 = F::from_u64(6).inverse().unwrap();
            let inv2 = inv6.mul_u64(3); // (1/6) * 3 = 1/2
            [-inv6, inv2, -inv2, inv6]
        }

        /// Materialize d=4 weights as field elements. One inversion (invert 24).
        pub fn weights_d4<F: JoltField>() -> [F; 5] {
            let inv24 = F::from_u64(24).inverse().unwrap();
            let inv6 = inv24.mul_u64(4); // (1/24) * 4 = 1/6
            let inv4 = inv24.mul_u64(6); // (1/24) * 6 = 1/4
            [inv24, -inv6, inv4, -inv6, inv24]
        }

        /// Materialize d=5 weights as field elements. One inversion (invert 120).
        pub fn weights_d5<F: JoltField>() -> [F; 6] {
            let inv120 = F::from_u64(120).inverse().unwrap();
            let p5 = inv120.mul_u64(5); // 5/120 = 1/24
            let p10 = inv120.mul_u64(10); // 10/120 = 1/12
            [-inv120, p5, -p10, p10, -p5, inv120]
        }

        /// Materialize d=6 weights as field elements. One inversion (invert 720).
        pub fn weights_d6<F: JoltField>() -> [F; 7] {
            let inv720 = F::from_u64(720).inverse().unwrap();
            let p6 = inv720.mul_u64(6); // 6/720 = 1/120
            let p15 = inv720.mul_u64(15); // 15/720 = 1/48
            let p20 = inv720.mul_u64(20); // 20/720 = 1/36
            [inv720, -p6, p15, -p20, p15, -p6, inv720]
        }

        /// Materialize d=7 weights as field elements. One inversion (invert 5040).
        pub fn weights_d7<F: JoltField>() -> [F; 8] {
            let inv5040 = F::from_u64(5040).inverse().unwrap();
            let p7 = inv5040.mul_u64(7); // 7/5040 = 1/720
            let p21 = inv5040.mul_u64(21); // 21/5040 = 1/240
            let p35 = inv5040.mul_u64(35); // 35/5040 = 1/144
            [-inv5040, p7, -p21, p35, -p35, p21, -p7, inv5040]
        }
    }
}

/// Binomial coefficients C(n, k) for 0 <= n <= 16 and 0 <= k <= floor(n/2).
/// Stored as a jagged constant table: each row `BINOMIAL_ROW_n` has length floor(n/2)+1.
pub mod binomial {
    /// Compute C(n, k) with k reduced to min(k, n-k). Uses u128 intermediates.
    #[inline]
    pub const fn binom_small(n: usize, k: usize) -> u64 {
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

    /// Build the full row [C(n,0), C(n,1), ..., C(n,n)] as an array of length N = n+1.
    /// Using [u64; N] avoids unstable generic const expressions in types.
    #[inline]
    pub const fn row_const<const N: usize>() -> [u64; N] {
        // N must be at least 1 (so n = N-1 is valid). Our usages satisfy this.
        let n = N - 1;
        let mut out = [0u64; N];
        let mut i = 0usize;
        while i < N {
            out[i] = binom_small(n, i);
            i += 1;
        }
        out
    }

    pub const BINOMIAL_ROW_13: [u64; 14] = row_const::<14>();
    pub const BINOMIAL_ROW_14: [u64; 15] = row_const::<15>();

    /// Generalized binomial for integer t and k >= 0, as i128.
    /// Supports negative t via identity: C(t, k) = (-1)^k C(-t + k - 1, k).
    #[inline]
    pub fn binom_int_i128(t: i64, k: usize) -> i128 {
        if k == 0 {
            return 1;
        }
        if t >= 0 {
            // C(t, k) with t >= 0; if k > t then 0
            let tt = t as i128;
            if (k as i128) > tt {
                return 0;
            }
            let mut num: i128 = 1;
            let mut den: i128 = 1;
            // Compute product_{j=0}^{k-1} (t - j) / (j + 1)
            let mut j = 0usize;
            while j < k {
                num = num * (tt - (j as i128));
                den = den * ((j as i128) + 1);
                j += 1;
            }
            // Exact division (combinatorial integer)
            num / den
        } else {
            // Use identity for negative t
            let sign = if (k & 1) == 1 { -1i128 } else { 1i128 };
            let tt = (-t) as i128 + (k as i128) - 1; // -t + k - 1 >= 0
                                                     // Compute C(tt, k)
            let mut num: i128 = 1;
            let mut den: i128 = 1;
            let mut j = 0usize;
            while j < k {
                num = num * (tt - (j as i128));
                den = den * ((j as i128) + 1);
                j += 1;
            }
            sign * (num / den)
        }
    }

    /// Lagrange coefficients for evaluating at integer shift `shift` from a window of length N.
    /// Given base values p(0), p(1), ..., p(N-1) of deg < N, returns alphas such that:
    ///   p(shift) = sum_{i=0}^{N-1} alpha[i] * p(i)
    /// Closed-form: alpha_i = (-1)^{N-1-i} C(shift, i) C(shift - i - 1, N - 1 - i)
    #[inline]
    pub fn lagrange_shift_coeffs_i128_const<const N: usize>(shift: i64) -> [i128; N] {
        let mut out = [0i128; N];
        let n_minus_1 = (N - 1) as i64;
        let mut i = 0usize;
        while i < N {
            let s1 = binom_int_i128(shift, i);
            let s2 = binom_int_i128(shift - (i as i64) - 1, (N - 1) - i);
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

    /// Const-evaluable generalized binomial for integer t and k >= 0, as i128.
    #[inline]
    pub const fn binom_int_i128_const(t: i64, k: usize) -> i128 {
        if k == 0 { return 1; }
        if t >= 0 {
            let tt = t as i128;
            if (k as i128) > tt { return 0; }
            let mut num: i128 = 1;
            let mut den: i128 = 1;
            let mut j: usize = 0;
            while j < k {
                num = num * (tt - (j as i128));
                den = den * ((j as i128) + 1);
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
                num = num * (tt - (j as i128));
                den = den * ((j as i128) + 1);
                j += 1;
            }
            sign * (num / den)
        }
    }

    /// Const-evaluable Lagrange coefficients at integer `shift` for a window of size N
    /// representing samples p(0),...,p(N-1). Returns i32 assuming it fits.
    #[inline]
    pub const fn lagrange_shift_coeffs_i32_const<const N: usize>(shift: i64) -> [i32; N] {
        let mut out = [0i32; N];
        let n_minus_1 = (N - 1) as i64;
        let mut i: usize = 0;
        while i < N {
            let s1 = binom_int_i128_const(shift, i);
            let s2 = binom_int_i128_const(shift - (i as i64) - 1, (N - 1) - i);
            let sign = if (((n_minus_1 as usize) - i) & 1) == 1 { -1i128 } else { 1i128 };
            let val = sign * s1 * s2;
            out[i] = val as i32;
            i += 1;
        }
        out
    }

    /// Constants for degree-13 window p(-6..=7):
    /// To evaluate p(S), define g(i)=p(i-6) on i=0..13, then compute g(S+6) with N=14.
    pub const LAGRANGE_DEG13_AT_NEG7: [i32; 14] = lagrange_shift_coeffs_i32_const::<14>(-1);
    pub const LAGRANGE_DEG13_AT_NEG8: [i32; 14] = lagrange_shift_coeffs_i32_const::<14>(-2);
    pub const LAGRANGE_DEG13_AT_NEG9: [i32; 14] = lagrange_shift_coeffs_i32_const::<14>(-3);
    pub const LAGRANGE_DEG13_AT_NEG10: [i32; 14] = lagrange_shift_coeffs_i32_const::<14>(-4);
    pub const LAGRANGE_DEG13_AT_NEG11: [i32; 14] = lagrange_shift_coeffs_i32_const::<14>(-5);
    pub const LAGRANGE_DEG13_AT_NEG12: [i32; 14] = lagrange_shift_coeffs_i32_const::<14>(-6);
    pub const LAGRANGE_DEG13_AT_NEG13: [i32; 14] = lagrange_shift_coeffs_i32_const::<14>(-7);

    pub const LAGRANGE_DEG13_AT_9: [i32; 14] = lagrange_shift_coeffs_i32_const::<14>(15);
    pub const LAGRANGE_DEG13_AT_10: [i32; 14] = lagrange_shift_coeffs_i32_const::<14>(16);
    pub const LAGRANGE_DEG13_AT_11: [i32; 14] = lagrange_shift_coeffs_i32_const::<14>(17);
    pub const LAGRANGE_DEG13_AT_12: [i32; 14] = lagrange_shift_coeffs_i32_const::<14>(18);
    pub const LAGRANGE_DEG13_AT_13: [i32; 14] = lagrange_shift_coeffs_i32_const::<14>(19);
}

/// Extrapolate the next value p(x+n) of a polynomial sequence from n consecutive values
/// p(x), p(x+1), ..., p(x+n-1).
///
/// For any polynomial of degree <= n-1, the next value satisfies the order-n recurrence:
///   p(x+n) = sum_{i=0}^{n-1} (-1)^{n-1-i} * C(n, i) * p(x+i)
/// where C(n, i) are binomial coefficients.
use crate::field::JoltField;

#[inline]
pub fn extrapolate_next_from_consecutive<F: JoltField>(prev: &[F]) -> F {
    let n = prev.len();
    debug_assert!(n >= 1 && n <= 16);
    let mut acc = F::zero();
    for i in 0..n {
        let c = binomial::binom_small(n, i);
        let coef = F::from_u64(c as u64);
        let term = prev[i] * coef;
        if ((n - 1 - i) & 1) == 1 {
            acc -= term;
        } else {
            acc += term;
        }
    }
    acc
}

/// Const-generic convenience wrapper for fixed window size N (N <= 16).
#[inline]
pub fn extrapolate_next_from_consecutive_const<F: JoltField, const N: usize>(prev: &[F; N]) -> F {
    extrapolate_next_from_consecutive(prev)
}

/// Optimized ex2 on consecutive grid (no infinity):
/// Given f[0] = p(x), f[1] = p(x+1) for deg <= 1, returns f[2] = p(x+2) = 2 f[1] - f[0].
#[inline(always)]
pub fn ex2_consecutive<F: JoltField>(f: &[F; 2]) -> F {
    f[1] + f[1] - f[0]
}

/// Optimized ex3 on consecutive grid (no infinity):
/// Given f[0..3) = [p(x), p(x+1), p(x+2)] for deg <= 2, returns f[3] = p(x+3)
/// = 3 f[2] - 3 f[1] + f[0].
#[inline(always)]
pub fn ex3_consecutive<F: JoltField>(f: &[F; 3]) -> F {
    let diff21 = f[2] - f[1];
    F::linear_combination_i64(&[(diff21, 3)], &[], &[f[0]], &[])
}

/// Optimized ex4 on consecutive grid (no infinity):
/// Given f[0..4) = [p(x), p(x+1), p(x+2), p(x+3)] for deg <= 3, returns p(x+4)
/// = -f0 + 4 f1 - 6 f2 + 4 f3.
#[inline(always)]
pub fn ex4_consecutive<F: JoltField>(f: &[F; 4]) -> F {
    let s13 = f[1] + f[3];
    F::linear_combination_i64(&[(s13, 4)], &[(f[0], 1), (f[2], 6)], &[], &[])
}

/// Optimized ex8 on consecutive grid (no infinity):
/// Given f[0..8) for deg <= 7, returns p(x+8) with binomial coefficients.
/// Coeff pattern: [-1, 8, -28, 56, -70, 56, -28, 8].
#[inline(always)]
pub fn ex8_consecutive<F: JoltField>(f: &[F; 8]) -> F {
    let s17 = f[1] + f[7];
    let s35 = f[3] + f[5];
    let s26 = f[2] + f[6];
    F::linear_combination_i64(
        &[(s17, 8), (s35, 56)],
        &[(f[0], 1), (s26, 28), (f[4], 70)],
        &[],
        &[],
    )
}

/// Optimized ex16 on consecutive grid (no infinity):
/// Given f[0..16) for deg <= 15, returns p(x+16) with binomial coefficients.
/// Coeff pattern: [-1, 16, -120, 560, -1820, 4368, -8008, 11440, -12870, 11440, -8008, 4368, -1820, 560, -120, 16].
#[inline(always)]
pub fn ex16_consecutive<F: JoltField>(f: &[F; 16]) -> F {
    let sp1 = f[1] + f[15];
    let sp3 = f[3] + f[13];
    let sp5 = f[5] + f[11];
    let sp7 = f[7] + f[9];
    let sn2 = f[2] + f[14];
    let sn4 = f[4] + f[12];
    let sn6 = f[6] + f[10];
    F::linear_combination_i64(
        &[(sp1, 16), (sp3, 560), (sp5, 4368), (sp7, 11440)],
        &[
            (f[0], 1),
            (sn2, 120),
            (sn4, 1820),
            (sn6, 8008),
            (f[8], 12870),
        ],
        &[],
        &[],
    )
}

/// Extend an odd-length consecutive window by N new values without multiplications.
///
/// Input: base[i] = p(i) for i = 0..N-1, where deg(p) <= N-1 and N is odd.
/// Output: out[j] = p(t_j) for j = 0..N-1 with t sequence:
///   t = [-1, N, -2, N+1, -3, N+2, ...] (alternating left and right), length N.
#[inline]
pub fn extend_consecutive_symmetric_const<F: JoltField, const N: usize>(base: &[F; N]) -> [F; N] {
    debug_assert!(N >= 1 && (N % 2 == 1));

    // Build forward-difference diagonal at left edge (i = 0): diffs[k] = Δ^k f(0), k=1..N-1
    let mut work = base.clone();
    let mut diffs = [F::zero(); N];
    for k in 1..N {
        for i in 0..(N - k) {
            work[i] = work[i + 1] - work[i];
        }
        diffs[k] = work[0];
    }

    // Left-side extension: step N_left = (N+1)/2 times backwards
    let left_cnt = (N + 1) / 2;
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

/// Compute p(x-1) from a length-N window [p(x), p(x+1), ..., p(x+N-1)] using
/// binomial-weighted add/sub only. Valid for N <= 16.
#[inline]
pub fn backward_step_exn<F: JoltField, const N: usize>(w: &[F; N]) -> F {
    let mut acc = F::zero();
    for j in 0..N {
        let c = crate::utils::interpolation::binomial::binom_small(N, j + 1);
        let coef = F::from_u64(c as u64);
        let term = w[j] * coef;
        if (j & 1) == 1 {
            acc -= term;
        } else {
            acc += term;
        }
    }
    acc
}

/// Symmetric extension by sliding-window exN: produce N values interleaved
/// [-1, N, -2, N+1, ...] using backward_step_exn on the left and specialized
/// exN (or general recurrence fallback) on the right.
#[inline]
pub fn extend_consecutive_symmetric_exn_const<F: JoltField, const N: usize>(
    base: &[F; N],
) -> [F; N] {
    // Left side via backward steps
    let left_cnt = (N + 1) / 2;
    let mut left_window = *base;
    let mut left_vals = [F::zero(); N];
    for i in 0..left_cnt {
        let prev = backward_step_exn::<F, N>(&left_window);
        left_vals[i] = prev;
        for k in (1..N).rev() {
            left_window[k] = left_window[k - 1];
        }
        left_window[0] = prev;
    }

    // Right side via forward steps
    let right_cnt = N / 2;
    let mut right_window = *base;
    let mut right_vals = [F::zero(); N];
    for i in 0..right_cnt {
        let next = match N {
            2 => ex2_consecutive::<F>(unsafe { &*(right_window.as_ptr() as *const [F; 2]) }),
            3 => ex3_consecutive::<F>(unsafe { &*(right_window.as_ptr() as *const [F; 3]) }),
            4 => ex4_consecutive::<F>(unsafe { &*(right_window.as_ptr() as *const [F; 4]) }),
            8 => ex8_consecutive::<F>(unsafe { &*(right_window.as_ptr() as *const [F; 8]) }),
            16 => ex16_consecutive::<F>(unsafe { &*(right_window.as_ptr() as *const [F; 16]) }),
            _ => extrapolate_next_from_consecutive::<F>(&right_window),
        };
        right_vals[i] = next;
        for k in 0..(N - 1) {
            right_window[k] = right_window[k + 1];
        }
        right_window[N - 1] = next;
    }

    // Interleave outputs
    let mut out = [F::zero(); N];
    for i in 0..right_cnt {
        out[2 * i] = left_vals[i];
        out[2 * i + 1] = right_vals[i];
    }
    if left_cnt > right_cnt {
        out[2 * right_cnt] = left_vals[right_cnt];
    }
    out
}

#[inline]
pub fn extend_consecutive_symmetric_3<F: JoltField>(base: &[F; 3]) -> [F; 3] {
    extend_consecutive_symmetric_const::<F, 3>(base)
}

#[inline]
pub fn extend_consecutive_symmetric_7<F: JoltField>(base: &[F; 7]) -> [F; 7] {
    extend_consecutive_symmetric_const::<F, 7>(base)
}

// Let's start simple. First derive generic implemenetation of extrapolating from a degree-D to a degree-2D polynomial
// with our symmetric zero-based domain
pub fn extend_evals_symmetric<const D: usize, F: Copy + Add + Sub + Mul<u32> + Default>(
    base_evals: &[F],
) -> [F; D] {
    // Requires that base_evals.len() == D + 1 (an interpolating set for the degree-D polynomial)
    assert_eq!(base_evals.len(), D + 1);
    let mut extended_evals = [Default::default(); D];
    for i in 0..D {
        extended_evals[i] = base_evals[i];
    }
    extended_evals
}

// Hard-coded implementation for degree-13 to degree-26 for maximum efficiency
// Also hard-coded depends on the input type?
// Tested to be equal to the generic implementation

// First case: Az are boolean (first group), returns i32
// Assumed layout of base_evals: [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
// Output order (interleaved): [-7, 8, -8, 9, -9, 10, -10, 11, -11, 12, -12, 13, -13]
pub fn extend_evals_deg_13_bool(base_evals: &[bool; 14]) -> [i32; 13] {
    let c = &crate::utils::interpolation::binomial::BINOMIAL_ROW_13;

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

// Second case: Bz are i128 with roughly 64-bit magnitude (first group), returns i128
pub fn extend_evals_deg_13_i128(base_evals: &[i128; 14]) -> [i128; 13] {
    let c = &crate::utils::interpolation::binomial::BINOMIAL_ROW_13;

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

#[cfg(test)]
mod tests {
    use crate::field::JoltField;
    use ark_bn254::Fr;
    use ark_std::{One, Zero};

    #[test]
    fn barycentric_eval_example_d3() {
        // polynomial: 5 + x + 3x^2 + 9x^3
        let coeffs = [
            Fr::from_u64(5),
            Fr::from_u64(1),
            Fr::from_u64(3),
            Fr::from_u64(9),
        ];
        let eval_poly = |x: Fr| -> Fr {
            let mut acc = coeffs[0];
            let mut pow = Fr::one();
            for i in 1..coeffs.len() {
                pow *= x;
                acc += pow * coeffs[i];
            }
            acc
        };

        // nodes: {-1, 0, 1, 2}
        let xs = [
            -Fr::from_u64(1),
            Fr::from_u64(0),
            Fr::from_u64(1),
            Fr::from_u64(2),
        ];
        let ys = [
            eval_poly(xs[0]),
            eval_poly(xs[1]),
            eval_poly(xs[2]),
            eval_poly(xs[3]),
        ];

        // Precompute weights (naively for the test)
        let ws_vec = crate::utils::interpolation::barycentric::compute_weights(&xs);
        let ws = [ws_vec[0], ws_vec[1], ws_vec[2], ws_vec[3]];

        // Query points
        let q = Fr::from_u64(7);
        let eval_expected = eval_poly(q);
        let eval_got =
            crate::utils::interpolation::barycentric::eval_safe_const::<Fr, 4>(&q, &xs, &ys, &ws);
        assert_eq!(eval_got, eval_expected);

        // x equal to a node returns y_i
        let eval_node = crate::utils::interpolation::barycentric::eval_safe_const::<Fr, 4>(
            &xs[2], &xs, &ys, &ws,
        );
        assert_eq!(eval_node, ys[2]);
    }

    #[test]
    fn barycentric_weight_tables_match_compute_weights() {
        // d = 2, nodes [-1,0,1]
        let xs2 = [-Fr::from_u64(1), Fr::from_u64(0), Fr::from_u64(1)];
        let ws2_generic = crate::utils::interpolation::barycentric::compute_weights(&xs2);
        let ws2_table = crate::utils::interpolation::barycentric::tables::weights_d2::<Fr>();
        assert_eq!(ws2_generic, ws2_table);

        // d = 3, nodes [-1,0,1,2]
        let xs3 = [
            -Fr::from_u64(1),
            Fr::from_u64(0),
            Fr::from_u64(1),
            Fr::from_u64(2),
        ];
        let ws3_generic = crate::utils::interpolation::barycentric::compute_weights(&xs3);
        let ws3_table = crate::utils::interpolation::barycentric::tables::weights_d3::<Fr>();
        assert_eq!(ws3_generic, ws3_table);

        // d = 4, nodes [-2,-1,0,1,2]
        let xs4 = [
            -Fr::from_u64(2),
            -Fr::from_u64(1),
            Fr::from_u64(0),
            Fr::from_u64(1),
            Fr::from_u64(2),
        ];
        let ws4_generic = crate::utils::interpolation::barycentric::compute_weights(&xs4);
        let ws4_table = crate::utils::interpolation::barycentric::tables::weights_d4::<Fr>();
        assert_eq!(ws4_generic, ws4_table);

        // d = 5, nodes [-2,-1,0,1,2,3]
        let xs5 = [
            -Fr::from_u64(2),
            -Fr::from_u64(1),
            Fr::from_u64(0),
            Fr::from_u64(1),
            Fr::from_u64(2),
            Fr::from_u64(3),
        ];
        let ws5_generic = crate::utils::interpolation::barycentric::compute_weights(&xs5);
        let ws5_table = crate::utils::interpolation::barycentric::tables::weights_d5::<Fr>();
        assert_eq!(ws5_generic, ws5_table);

        // d = 6, nodes [-3,-2,-1,0,1,2,3]
        let xs6 = [
            -Fr::from_u64(3),
            -Fr::from_u64(2),
            -Fr::from_u64(1),
            Fr::from_u64(0),
            Fr::from_u64(1),
            Fr::from_u64(2),
            Fr::from_u64(3),
        ];
        let ws6_generic = crate::utils::interpolation::barycentric::compute_weights(&xs6);
        let ws6_table = crate::utils::interpolation::barycentric::tables::weights_d6::<Fr>();
        assert_eq!(ws6_generic, ws6_table);

        // d = 7, nodes [-3,-2,-1,0,1,2,3,4]
        let xs7 = [
            -Fr::from_u64(3),
            -Fr::from_u64(2),
            -Fr::from_u64(1),
            Fr::from_u64(0),
            Fr::from_u64(1),
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(4),
        ];
        let ws7_generic = crate::utils::interpolation::barycentric::compute_weights(&xs7);
        let ws7_table = crate::utils::interpolation::barycentric::tables::weights_d7::<Fr>();
        assert_eq!(ws7_generic, ws7_table);
    }

    #[test]
    fn extrapolate_next_matches_polynomial_values() {
        // For each window size n = 1..16, build a degree-(n-1) polynomial with simple
        // integer coefficients a_k = k+1, evaluate at x=0..n, and check prediction.
        for n in 1usize..=16usize {
            // Coefficients a_k = k+1
            let mut coeffs: Vec<Fr> = Vec::with_capacity(n);
            for k in 0..n {
                coeffs.push(Fr::from_u64((k as u64) + 1));
            }

            // Evaluate poly at integer points 0..=n
            let eval_at = |x: u64| -> Fr {
                let mut acc = Fr::zero();
                let mut pow = Fr::one();
                for c in coeffs.iter() {
                    acc += *c * pow;
                    pow *= Fr::from_u64(x);
                }
                acc
            };

            let mut prev: Vec<Fr> = Vec::with_capacity(n);
            for x in 0..(n as u64) {
                prev.push(eval_at(x));
            }
            let truth = eval_at(n as u64);
            let pred = crate::utils::interpolation::extrapolate_next_from_consecutive::<Fr>(&prev);
            assert_eq!(pred, truth, "n={} failed", n);
        }
    }

    #[test]
    fn optimized_ex_consecutive_random_inputs_consistency() {
        use crate::utils::interpolation::{
            ex16_consecutive, ex2_consecutive, ex3_consecutive, ex4_consecutive, ex8_consecutive,
            extrapolate_next_from_consecutive,
        };
        use ark_std::test_rng;
        use rand_core::RngCore;

        let mut rng = test_rng();

        // N = 2
        for _ in 0..500 {
            let f0 = Fr::from_u64(rng.next_u64());
            let f1 = Fr::from_u64(rng.next_u64());
            let prev = [f0, f1];
            let a = ex2_consecutive::<Fr>(&prev);
            let b = extrapolate_next_from_consecutive::<Fr>(&prev);
            assert_eq!(a, b);
        }

        // N = 3
        for _ in 0..500 {
            let f0 = Fr::from_u64(rng.next_u64());
            let f1 = Fr::from_u64(rng.next_u64());
            let f2 = Fr::from_u64(rng.next_u64());
            let prev = [f0, f1, f2];
            let a = ex3_consecutive::<Fr>(&prev);
            let b = extrapolate_next_from_consecutive::<Fr>(&prev);
            assert_eq!(a, b);
        }

        // N = 4
        for _ in 0..300 {
            let f0 = Fr::from_u64(rng.next_u64());
            let f1 = Fr::from_u64(rng.next_u64());
            let f2 = Fr::from_u64(rng.next_u64());
            let f3 = Fr::from_u64(rng.next_u64());
            let prev = [f0, f1, f2, f3];
            let a = ex4_consecutive::<Fr>(&prev);
            let b = extrapolate_next_from_consecutive::<Fr>(&prev);
            assert_eq!(a, b);
        }

        // N = 8
        for _ in 0..100 {
            let prev = [
                Fr::from_u64(rng.next_u64()),
                Fr::from_u64(rng.next_u64()),
                Fr::from_u64(rng.next_u64()),
                Fr::from_u64(rng.next_u64()),
                Fr::from_u64(rng.next_u64()),
                Fr::from_u64(rng.next_u64()),
                Fr::from_u64(rng.next_u64()),
                Fr::from_u64(rng.next_u64()),
            ];
            let a = ex8_consecutive::<Fr>(&prev);
            let b = extrapolate_next_from_consecutive::<Fr>(&prev);
            assert_eq!(a, b);
        }

        // N = 16
        for _ in 0..50 {
            let prev = [
                Fr::from_u64(rng.next_u64()),
                Fr::from_u64(rng.next_u64()),
                Fr::from_u64(rng.next_u64()),
                Fr::from_u64(rng.next_u64()),
                Fr::from_u64(rng.next_u64()),
                Fr::from_u64(rng.next_u64()),
                Fr::from_u64(rng.next_u64()),
                Fr::from_u64(rng.next_u64()),
                Fr::from_u64(rng.next_u64()),
                Fr::from_u64(rng.next_u64()),
                Fr::from_u64(rng.next_u64()),
                Fr::from_u64(rng.next_u64()),
                Fr::from_u64(rng.next_u64()),
                Fr::from_u64(rng.next_u64()),
                Fr::from_u64(rng.next_u64()),
                Fr::from_u64(rng.next_u64()),
            ];
            let a = ex16_consecutive::<Fr>(&prev);
            let b = extrapolate_next_from_consecutive::<Fr>(&prev);
            assert_eq!(a, b);
        }

        // Symmetric extension N=3
        for _ in 0..200 {
            // Random degree-2 polynomial over integers mod Fr
            let c0 = Fr::from_u64(rng.next_u64());
            let c1 = Fr::from_u64(rng.next_u64());
            let c2 = Fr::from_u64(rng.next_u64());
            let eval = |t: i64| -> Fr {
                let x = if t >= 0 {
                    Fr::from_u64(t as u64)
                } else {
                    -Fr::from_u64((-t) as u64)
                };
                c0 + c1 * x + c2 * (x * x)
            };
            let base = [eval(0), eval(1), eval(2)];
            let ext = crate::utils::interpolation::extend_consecutive_symmetric_3::<Fr>(&base);
            // Expect [-1, 3, -2]
            assert_eq!(ext[0], eval(-1));
            assert_eq!(ext[1], eval(3));
            assert_eq!(ext[2], eval(-2));
        }

        // Symmetric extension N=7
        for _ in 0..100 {
            // Random degree-6 polynomial via Horner
            let mut coeffs = [Fr::from_u64(0); 7];
            for i in 0..7 {
                coeffs[i] = Fr::from_u64(rng.next_u64());
            }
            let eval = |t: i64| -> Fr {
                let x = if t >= 0 {
                    Fr::from_u64(t as u64)
                } else {
                    -Fr::from_u64((-t) as u64)
                };
                let mut acc = coeffs[6];
                for k in (0..6).rev() {
                    acc = acc * x + coeffs[k];
                }
                acc
            };
            let base = [
                eval(0),
                eval(1),
                eval(2),
                eval(3),
                eval(4),
                eval(5),
                eval(6),
            ];
            let ext = crate::utils::interpolation::extend_consecutive_symmetric_7::<Fr>(&base);
            // Expect [-1, 7, -2, 8, -3, 9, -4]
            let exp = [
                eval(-1),
                eval(7),
                eval(-2),
                eval(8),
                eval(-3),
                eval(9),
                eval(-4),
            ];
            for i in 0..7 {
                assert_eq!(ext[i], exp[i]);
            }
        }
    }
}
