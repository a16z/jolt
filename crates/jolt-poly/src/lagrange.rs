//! Lagrange interpolation utilities over integer domains.
//!
//! Provides building blocks for the univariate skip optimization in sumcheck
//! protocols. All functions are generic over [`Field`] and operate on
//! integer-indexed domains (symmetric or arbitrary).

use std::{fmt, marker::PhantomData};

use jolt_field::Field;

/// Evaluates all Lagrange basis polynomials $L_0(r), \ldots, L_{N-1}(r)$ over
/// the domain $\{s, s+1, \ldots, s+N-1\}$ where $s$ = `domain_start`.
///
/// Uses the barycentric formula with $O(N^2)$ work for weight computation
/// and $O(N)$ per-element inversions.
///
/// # Panics
/// Panics if `domain_size` is zero.
#[expect(clippy::expect_used)]
pub fn lagrange_evals<F: Field>(domain_start: i64, domain_size: usize, r: F) -> Vec<F> {
    assert!(domain_size > 0, "domain_size must be positive");

    // Check if r coincides with a grid point (early exit)
    let nodes: Vec<F> = (0..domain_size)
        .map(|k| F::from_i64(domain_start + k as i64))
        .collect();

    for (i, &node) in nodes.iter().enumerate() {
        if r == node {
            let mut result = vec![F::zero(); domain_size];
            result[i] = F::one();
            return result;
        }
    }

    // Compute (r - x_0)(r - x_1)...(r - x_{N-1})
    let diffs: Vec<F> = nodes.iter().map(|&x| r - x).collect();
    let full_product: F = diffs.iter().copied().product();

    // Barycentric weights: w_i = 1 / prod_{j != i} (x_i - x_j)
    // For consecutive integers {s, s+1, ..., s+N-1}, the denominator is
    // prod_{j != i} (i - j) which equals (-1)^{N-1-i} * i! * (N-1-i)!
    let mut weights = vec![F::one(); domain_size];
    for (i, wi) in weights.iter_mut().enumerate() {
        for j in 0..domain_size {
            if i != j {
                let diff = (i as i64) - (j as i64);
                *wi *= F::from_i64(diff);
            }
        }
        *wi = wi.inverse().expect("Lagrange weights must be invertible");
    }

    // L_i(r) = full_product * w_i / (r - x_i)
    let mut result = Vec::with_capacity(domain_size);
    for i in 0..domain_size {
        let diff_inv = diffs[i]
            .inverse()
            .expect("r should not coincide with a node");
        result.push(full_product * weights[i] * diff_inv);
    }

    result
}

/// Evaluates all Lagrange basis polynomials over the centered consecutive
/// integer domain used by univariate-skip protocols.
pub fn centered_lagrange_evals<F: Field>(
    domain_size: usize,
    r: F,
) -> Result<Vec<F>, CenteredIntegerDomainError> {
    Ok(lagrange_evals(
        centered_domain_start(domain_size)?,
        domain_size,
        r,
    ))
}

pub fn centered_lagrange_evals_array<F: Field, const N: usize>(
    r: F,
) -> Result<[F; N], CenteredIntegerDomainError> {
    let evals = centered_lagrange_evals(N, r)?;
    let mut result = [F::zero(); N];
    for (dst, src) in result.iter_mut().zip(evals) {
        *dst = src;
    }
    Ok(result)
}

/// Computes `sum_i L_i(x) * L_i(y)` over the centered consecutive integer
/// domain used by univariate-skip protocols.
pub fn centered_lagrange_kernel<F: Field>(
    domain_size: usize,
    x: F,
    y: F,
) -> Result<F, CenteredIntegerDomainError> {
    let x_evals = centered_lagrange_evals(domain_size, x)?;
    let y_evals = centered_lagrange_evals(domain_size, y)?;
    Ok(x_evals
        .into_iter()
        .zip(y_evals)
        .map(|(left, right)| left * right)
        .sum())
}

/// Computes power sums $S_k = \sum_{t=-D}^{D} t^k$ for $k = 0, 1, \ldots, \text{num\_powers}-1$
/// over the symmetric integer domain $\{-D, \ldots, D\}$ of size $2D+1$.
///
/// Returns integer power sums as `i128`. Odd-power sums are zero by symmetry.
///
/// Used by the verifier to check $\sum_{Y \in W} p(Y) = \text{claimed\_sum}$
/// without evaluating $p$ at every domain point.
pub fn symmetric_power_sums(half_width: i64, num_powers: usize) -> Vec<i128> {
    let mut sums = vec![0i128; num_powers];
    for t in -half_width..=half_width {
        let mut power = 1i128;
        for s in &mut sums {
            *s += power;
            power *= t as i128;
        }
    }
    sums
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CenteredIntegerDomainError {
    EmptyDomain,
    DomainTooLarge { domain_size: usize },
    PowerSumOverflow { domain_size: usize, power: usize },
}

impl fmt::Display for CenteredIntegerDomainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDomain => write!(f, "centered integer domain must be non-empty"),
            Self::DomainTooLarge { domain_size } => {
                write!(
                    f,
                    "centered integer domain size {domain_size} exceeds i64::MAX"
                )
            }
            Self::PowerSumOverflow { domain_size, power } => write!(
                f,
                "centered integer domain size {domain_size} overflowed i128 at power {power}"
            ),
        }
    }
}

impl std::error::Error for CenteredIntegerDomainError {}

pub struct LagrangeHelper;

impl LagrangeHelper {
    #[inline]
    pub const fn fact(n: usize) -> u64 {
        let mut acc = 1u64;
        let mut i = 2usize;
        while i <= n {
            acc *= i as u64;
            i += 1;
        }
        acc
    }

    pub const FACT_U64_0_TO_20: [u64; 21] = {
        let mut out = [0u64; 21];
        let mut i = 0usize;
        while i <= 20 {
            out[i] = Self::fact(i);
            i += 1;
        }
        out
    };

    #[inline]
    pub const fn den_row_i64<const N: usize>() -> [i64; N] {
        let mut out = [0i64; N];
        let mut i = 0usize;
        while i < N {
            let left = Self::FACT_U64_0_TO_20[i] as i128;
            let right = Self::FACT_U64_0_TO_20[N - 1 - i] as i128;
            let mut value = left * right;
            if ((N - 1 - i) & 1) == 1 {
                value = -value;
            }
            out[i] = value as i64;
            i += 1;
        }
        out
    }
}

pub struct LagrangePolynomial<F: Field>(PhantomData<F>);

impl<F: Field> LagrangePolynomial<F> {
    #[inline]
    fn start_i64<const N: usize>() -> i64 {
        -(((N - 1) / 2) as i64)
    }

    #[inline]
    fn distances<const N: usize>(r: F) -> ([F; N], Option<usize>) {
        let mut dists = [F::zero(); N];
        let mut base = r - F::from_i64(Self::start_i64::<N>());
        let mut hit = None;
        for (i, dist) in dists.iter_mut().enumerate() {
            let current = base;
            if current.is_zero() {
                hit = Some(i);
            }
            *dist = current;
            base -= F::one();
        }
        (dists, hit)
    }

    #[inline]
    #[expect(clippy::expect_used)]
    fn inv_denom<const N: usize>() -> [F; N] {
        let den_i64 = LagrangeHelper::den_row_i64::<N>();
        let mut denom = [F::zero(); N];
        for (dst, &src) in denom.iter_mut().zip(den_i64.iter()) {
            *dst = F::from_i64(src);
        }

        let mut left = [F::one(); N];
        for i in 1..N {
            left[i] = left[i - 1] * denom[i - 1];
        }
        let inv_total = (left[N - 1] * denom[N - 1])
            .inverse()
            .expect("Lagrange denominator product is invertible");

        let mut inv_denom = [F::zero(); N];
        let mut right = F::one();
        for i in (0..N).rev() {
            inv_denom[i] = left[i] * right * inv_total;
            right *= denom[i];
        }
        inv_denom
    }

    #[inline]
    #[expect(clippy::expect_used)]
    fn bary_terms_from_dists<const N: usize>(dists: &[F; N], inv_denom: &[F; N]) -> ([F; N], F) {
        let mut prefix = [F::one(); N];
        for i in 1..N {
            prefix[i] = prefix[i - 1] * dists[i - 1];
        }
        let inv_prod = (prefix[N - 1] * dists[N - 1])
            .inverse()
            .expect("off-domain Lagrange distance product is invertible");

        let mut suffix = [F::one(); N];
        for i in (0..N.saturating_sub(1)).rev() {
            suffix[i] = suffix[i + 1] * dists[i + 1];
        }

        let mut terms = [F::zero(); N];
        let mut sum = F::zero();
        for i in 0..N {
            let inv_di = prefix[i] * suffix[i] * inv_prod;
            let term = inv_denom[i] * inv_di;
            terms[i] = term;
            sum += term;
        }
        (terms, sum)
    }

    #[inline]
    #[expect(clippy::expect_used)]
    pub fn evaluate<const N: usize>(values: &[F; N], r: F) -> F {
        debug_assert!(N > 0, "N must be positive");
        debug_assert!(N <= 20, "evaluate is intended for small N");
        let (dists, hit) = Self::distances::<N>(r);
        if let Some(i) = hit {
            return values[i];
        }
        let inv_denom = Self::inv_denom::<N>();
        let (terms, sum) = Self::bary_terms_from_dists::<N>(&dists, &inv_denom);
        let inv_sum = sum
            .inverse()
            .expect("off-domain Lagrange term sum is invertible");
        let mut numerator = F::zero();
        for i in 0..N {
            numerator += values[i] * terms[i];
        }
        numerator * inv_sum
    }

    #[inline]
    #[expect(clippy::expect_used)]
    pub fn evals<const N: usize>(r: F) -> [F; N] {
        debug_assert!(N > 0, "N must be positive");
        debug_assert!(N <= 20, "evals is intended for small N");
        let (dists, hit) = Self::distances::<N>(r);
        if let Some(i) = hit {
            let mut out = [F::zero(); N];
            out[i] = F::one();
            return out;
        }
        let inv_denom = Self::inv_denom::<N>();
        let (terms, sum) = Self::bary_terms_from_dists::<N>(&dists, &inv_denom);
        let inv_sum = sum
            .inverse()
            .expect("off-domain Lagrange term sum is invertible");
        let mut out = [F::zero(); N];
        for i in 0..N {
            out[i] = terms[i] * inv_sum;
        }
        out
    }

    #[inline]
    #[expect(clippy::expect_used)]
    pub fn lagrange_kernel<const N: usize>(x: F, y: F) -> F {
        debug_assert!(N > 0, "N must be positive");
        debug_assert!(N <= 20, "lagrange_kernel is intended for small N");
        let (dists_x, hit_x) = Self::distances::<N>(x);
        let (dists_y, hit_y) = Self::distances::<N>(y);

        if let (Some(ix), Some(iy)) = (hit_x, hit_y) {
            return if ix == iy { F::one() } else { F::zero() };
        }

        let inv_denom = Self::inv_denom::<N>();
        if let Some(ix) = hit_x {
            let (terms_y, sum_y) = Self::bary_terms_from_dists::<N>(&dists_y, &inv_denom);
            return terms_y[ix]
                * sum_y
                    .inverse()
                    .expect("off-domain Lagrange term sum is invertible");
        }
        if let Some(iy) = hit_y {
            let (terms_x, sum_x) = Self::bary_terms_from_dists::<N>(&dists_x, &inv_denom);
            return terms_x[iy]
                * sum_x
                    .inverse()
                    .expect("off-domain Lagrange term sum is invertible");
        }

        let (terms_x, sum_x) = Self::bary_terms_from_dists::<N>(&dists_x, &inv_denom);
        let (terms_y, sum_y) = Self::bary_terms_from_dists::<N>(&dists_y, &inv_denom);
        let mut numerator = F::zero();
        for i in 0..N {
            numerator += terms_x[i] * terms_y[i];
        }
        numerator
            * (sum_x * sum_y)
                .inverse()
                .expect("off-domain Lagrange kernel denominator is invertible")
    }

    pub fn evaluate_many<const N: usize>(values: &[F; N], points: &[F]) -> Vec<F> {
        if points.is_empty() {
            return Vec::new();
        }

        if points.len() > N {
            let coeffs = Self::interpolate_coeffs(values);
            points
                .iter()
                .map(|&point| {
                    let mut result = coeffs[N - 1];
                    for i in (0..N - 1).rev() {
                        result = result * point + coeffs[i];
                    }
                    result
                })
                .collect()
        } else {
            points
                .iter()
                .map(|&point| Self::evaluate::<N>(values, point))
                .collect()
        }
    }

    #[inline]
    #[expect(clippy::expect_used)]
    pub fn interpolate_coeffs<const N: usize>(values: &[F; N]) -> [F; N] {
        debug_assert!(N > 0, "N must be positive");
        let degree = N - 1;
        let start = Self::start_i64::<N>();

        let mut smalls = [0u64; N];
        let mut prefix = [F::one(); N];
        for m in 1..=degree {
            smalls[m] = m as u64;
            prefix[m] = prefix[m - 1].mul_u64(smalls[m]);
        }
        let inv_total = prefix[degree]
            .inverse()
            .expect("factorial product is invertible");
        let mut right = F::one();
        let mut inverses = [F::zero(); N];
        for idx in (1..=degree).rev() {
            inverses[idx] = prefix[idx - 1] * right * inv_total;
            right = right.mul_u64(smalls[idx]);
        }

        let mut dd = *values;
        let mut newton = [F::zero(); N];
        newton[0] = dd[0];
        for order in 1..=degree {
            let inv = inverses[order];
            for i in 0..(N - order) {
                dd[i] = (dd[i + 1] - dd[i]) * inv;
            }
            newton[order] = dd[0];
        }

        let mut coeffs = [F::zero(); N];
        let mut basis = [F::zero(); N];
        basis[0] = F::one();
        let mut basis_degree = 0usize;
        for (k, &scale) in newton.iter().enumerate() {
            for j in 0..=basis_degree {
                coeffs[j] += scale * basis[j];
            }

            if k == degree {
                break;
            }

            let node = start + k as i64;
            let last = basis[basis_degree];
            for idx in (1..=basis_degree).rev() {
                let old = basis[idx];
                basis[idx] = basis[idx - 1] - old.mul_i64(node);
            }
            basis[0] = -basis[0].mul_i64(node);
            basis_degree += 1;
            basis[basis_degree] = last;
        }

        coeffs
    }
}

pub fn centered_lagrange_evaluate<F: Field, const N: usize>(
    values: &[F; N],
    r: F,
) -> Result<F, CenteredIntegerDomainError> {
    let _ = centered_domain_start(N)?;
    Ok(LagrangePolynomial::<F>::evaluate::<N>(values, r))
}

pub fn centered_lagrange_evaluate_many<F: Field, const N: usize>(
    values: &[F; N],
    points: &[F],
) -> Result<Vec<F>, CenteredIntegerDomainError> {
    let _ = centered_domain_start(N)?;
    Ok(LagrangePolynomial::<F>::evaluate_many::<N>(values, points))
}

pub fn centered_interpolate_coeffs_array<F: Field, const N: usize>(
    values: &[F; N],
) -> Result<[F; N], CenteredIntegerDomainError> {
    let _ = centered_domain_start(N)?;
    Ok(LagrangePolynomial::<F>::interpolate_coeffs::<N>(values))
}

/// Start of the centered consecutive-integer domain used by core univariate skip.
///
/// The domain has `domain_size` consecutive integer points
/// `{start, start + 1, ..., start + domain_size - 1}` where
/// `start = -floor((domain_size - 1) / 2)`.
pub fn centered_domain_start(domain_size: usize) -> Result<i64, CenteredIntegerDomainError> {
    if domain_size == 0 {
        return Err(CenteredIntegerDomainError::EmptyDomain);
    }
    if domain_size > i64::MAX as usize {
        return Err(CenteredIntegerDomainError::DomainTooLarge { domain_size });
    }
    Ok(-(((domain_size - 1) / 2) as i64))
}

/// Computes `S_k = sum_t t^k` over the centered consecutive-integer domain.
///
/// This matches core's univariate-skip window convention for both odd and even
/// domain sizes. For example, size `3` is `{-1, 0, 1}` and size `4` is
/// `{-1, 0, 1, 2}`.
pub fn centered_power_sums(
    domain_size: usize,
    num_powers: usize,
) -> Result<Vec<i128>, CenteredIntegerDomainError> {
    let start = centered_domain_start(domain_size)?;
    let mut sums = vec![0i128; num_powers];
    if num_powers == 0 {
        return Ok(sums);
    }

    for offset in 0..domain_size {
        let offset = i64::try_from(offset)
            .map_err(|_| CenteredIntegerDomainError::DomainTooLarge { domain_size })?;
        let t = i128::from(
            start
                .checked_add(offset)
                .ok_or(CenteredIntegerDomainError::DomainTooLarge { domain_size })?,
        );
        let mut pow = 1i128;
        for (power, sum) in sums.iter_mut().enumerate() {
            *sum = sum
                .checked_add(pow)
                .ok_or(CenteredIntegerDomainError::PowerSumOverflow { domain_size, power })?;
            if power + 1 < num_powers {
                pow = pow
                    .checked_mul(t)
                    .ok_or(CenteredIntegerDomainError::PowerSumOverflow {
                        domain_size,
                        power: power + 1,
                    })?;
            }
        }
    }

    Ok(sums)
}

/// Polynomial multiplication in coefficient form.
///
/// Given $p(x) = \sum a_i x^i$ and $q(x) = \sum b_j x^j$, returns
/// coefficients of $p \cdot q$ of length `a.len() + b.len() - 1`.
///
/// Returns empty if either input is empty.
pub fn poly_mul<F: Field>(a: &[F], b: &[F]) -> Vec<F> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let n = a.len() + b.len() - 1;
    let mut result = vec![F::zero(); n];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            result[i + j] += ai * bj;
        }
    }
    result
}

/// Interpolates evaluations at consecutive integers to monomial coefficients.
///
/// Given values $[f(s), f(s+1), \ldots, f(s+N-1)]$ where $s$ = `domain_start`,
/// returns the unique polynomial of degree $\leq N-1$ in coefficient form
/// $[c_0, c_1, \ldots, c_{N-1}]$ such that $p(x) = \sum c_i x^i$.
///
/// Uses Newton's divided differences for $O(N^2)$ work.
///
/// # Panics
/// Panics if `values` is empty.
#[expect(clippy::expect_used)]
pub fn interpolate_to_coeffs<F: Field>(domain_start: i64, values: &[F]) -> Vec<F> {
    let n = values.len();
    assert!(n > 0, "cannot interpolate zero values");

    // Newton's divided differences: dd[i] = f[x_i, ..., x_{i-step}]
    // For consecutive integer nodes x_k = s+k, the denominator is always `step`.
    let mut dd = values.to_vec();
    for step in 1..n {
        let denom_inv = F::from_i64(step as i64)
            .inverse()
            .expect("divided difference denominator must be invertible");
        for i in (step..n).rev() {
            dd[i] = (dd[i] - dd[i - 1]) * denom_inv;
        }
    }

    // Convert Newton form (with nodes s, s+1, ...) to monomial form.
    // p(x) = dd[0] + dd[1]*(x-s) + dd[2]*(x-s)*(x-s-1) + ...
    let mut coeffs = vec![F::zero(); n];
    // basis[k] = coefficient-form of (x-s)(x-s-1)...(x-s-k+1)
    let mut basis = vec![F::zero(); n];
    basis[0] = F::one();

    for (k, &dd_k) in dd.iter().enumerate() {
        // Add dd[k] * basis to coeffs
        for (i, &b) in basis.iter().enumerate().take(k + 1) {
            coeffs[i] += dd_k * b;
        }
        // Update basis: multiply by (x - (s + k))
        if k < n - 1 {
            let shift = F::from_i64(-(domain_start + k as i64));
            // basis = basis * (x + shift) = basis * x + basis * shift
            // Process in reverse to avoid overwriting
            for i in (1..=k + 1).rev() {
                basis[i] = basis[i - 1] + basis[i] * shift;
            }
            basis[0] *= shift;
        }
    }

    coeffs
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests should fail loudly")]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};
    use num_traits::{One, Zero};

    #[test]
    fn lagrange_evals_partition_of_unity() {
        // Sum of all Lagrange basis values at any point must be 1
        let r = Fr::from_u64(42);
        let evals = lagrange_evals(0, 5, r);
        let sum: Fr = evals.iter().copied().sum();
        assert_eq!(sum, Fr::one());
    }

    #[test]
    fn lagrange_evals_at_node_is_indicator() {
        for i in 0..5u64 {
            let r = Fr::from_u64(i);
            let evals = lagrange_evals(0, 5, r);
            for (j, &val) in evals.iter().enumerate() {
                if j == i as usize {
                    assert_eq!(val, Fr::one(), "L_{j}({i}) should be 1");
                } else {
                    assert!(val.is_zero(), "L_{j}({i}) should be 0");
                }
            }
        }
    }

    #[test]
    fn lagrange_evals_symmetric_domain() {
        // Domain {-2, -1, 0, 1, 2}
        let r = Fr::from_u64(7);
        let evals = lagrange_evals(-2, 5, r);
        let sum: Fr = evals.iter().copied().sum();
        assert_eq!(sum, Fr::one());
    }

    #[test]
    fn lagrange_evals_symmetric_at_node() {
        // r = -1 is the second node in {-2, -1, 0, 1, 2}
        let r = Fr::from_i64(-1);
        let evals = lagrange_evals(-2, 5, r);
        assert_eq!(evals[1], Fr::one());
        for (i, &val) in evals.iter().enumerate() {
            if i != 1 {
                assert!(val.is_zero());
            }
        }
    }

    #[test]
    fn symmetric_power_sums_basic() {
        // Domain {-1, 0, 1}: S_0 = 3, S_1 = 0, S_2 = 2
        let sums = symmetric_power_sums(1, 4);
        assert_eq!(sums[0], 3);
        assert_eq!(sums[1], 0); // symmetric
        assert_eq!(sums[2], 2); // (-1)^2 + 0 + 1^2
        assert_eq!(sums[3], 0); // symmetric
    }

    #[test]
    fn symmetric_power_sums_width_2() {
        // Domain {-2, -1, 0, 1, 2}: S_0 = 5, S_1 = 0, S_2 = 10
        let sums = symmetric_power_sums(2, 3);
        assert_eq!(sums[0], 5);
        assert_eq!(sums[1], 0);
        assert_eq!(sums[2], 10); // 4 + 1 + 0 + 1 + 4
    }

    #[test]
    fn centered_domain_start_matches_core_uniskip_convention() {
        assert_eq!(centered_domain_start(1), Ok(0));
        assert_eq!(centered_domain_start(3), Ok(-1));
        assert_eq!(centered_domain_start(4), Ok(-1));
        assert_eq!(centered_domain_start(10), Ok(-4));
    }

    #[test]
    fn centered_lagrange_helpers_match_centered_domain() {
        let r = Fr::from_u64(7);
        let evals = centered_lagrange_evals(5, r).unwrap();
        let evals_array = centered_lagrange_evals_array::<_, 5>(r).unwrap();

        assert_eq!(evals, lagrange_evals(-2, 5, r));
        assert_eq!(evals_array.as_slice(), evals.as_slice());
        assert_eq!(
            centered_lagrange_kernel(5, Fr::from_i64(-1), Fr::from_i64(-1)),
            Ok(Fr::one())
        );
        assert_eq!(
            centered_lagrange_kernel(5, Fr::from_i64(-1), Fr::from_i64(2)),
            Ok(Fr::zero())
        );
    }

    #[test]
    fn centered_power_sums_handle_even_and_odd_windows() {
        assert_eq!(centered_power_sums(3, 4), Ok(vec![3, 0, 2, 0]));
        assert_eq!(centered_power_sums(4, 4), Ok(vec![4, 2, 6, 8]));
    }

    #[test]
    fn centered_power_sums_reject_invalid_or_overflowing_inputs() {
        assert_eq!(
            centered_power_sums(0, 2),
            Err(CenteredIntegerDomainError::EmptyDomain)
        );
        assert!(matches!(
            centered_power_sums(10, 100),
            Err(CenteredIntegerDomainError::PowerSumOverflow { .. })
        ));
    }

    #[test]
    fn poly_mul_basic() {
        // (1 + 2x) * (3 + x) = 3 + 7x + 2x^2
        let a = [Fr::from_u64(1), Fr::from_u64(2)];
        let b = [Fr::from_u64(3), Fr::from_u64(1)];
        let c = poly_mul(&a, &b);
        assert_eq!(c.len(), 3);
        assert_eq!(c[0], Fr::from_u64(3));
        assert_eq!(c[1], Fr::from_u64(7));
        assert_eq!(c[2], Fr::from_u64(2));
    }

    #[test]
    fn poly_mul_empty() {
        let a: [Fr; 0] = [];
        let b = [Fr::from_u64(1)];
        assert!(poly_mul(&a, &b).is_empty());
    }

    #[test]
    fn interpolate_to_coeffs_constant() {
        // f(0) = f(1) = f(2) = 5 → p(x) = 5
        let vals = [Fr::from_u64(5), Fr::from_u64(5), Fr::from_u64(5)];
        let coeffs = interpolate_to_coeffs(0, &vals);
        assert_eq!(coeffs[0], Fr::from_u64(5));
        assert!(coeffs[1].is_zero());
        assert!(coeffs[2].is_zero());
    }

    #[test]
    fn interpolate_to_coeffs_linear() {
        // f(0) = 1, f(1) = 3 → p(x) = 1 + 2x
        let vals = [Fr::from_u64(1), Fr::from_u64(3)];
        let coeffs = interpolate_to_coeffs(0, &vals);
        assert_eq!(coeffs[0], Fr::from_u64(1));
        assert_eq!(coeffs[1], Fr::from_u64(2));
    }

    #[test]
    fn interpolate_to_coeffs_quadratic() {
        // f(0) = 1, f(1) = 4, f(2) = 11 → p(x) = 1 + x + 2x^2
        // p(0)=1, p(1)=1+1+2=4, p(2)=1+2+8=11 ✓
        let vals = [Fr::from_u64(1), Fr::from_u64(4), Fr::from_u64(11)];
        let coeffs = interpolate_to_coeffs(0, &vals);
        // Verify by evaluating at each point
        for (i, &expected) in vals.iter().enumerate() {
            let x = Fr::from_u64(i as u64);
            let mut val = Fr::zero();
            let mut x_pow = Fr::one();
            for &c in &coeffs {
                val += c * x_pow;
                x_pow *= x;
            }
            assert_eq!(val, expected, "mismatch at x={i}");
        }
    }

    #[test]
    fn interpolate_to_coeffs_symmetric_domain() {
        // Domain {-1, 0, 1}: f(-1)=2, f(0)=1, f(1)=2 → p(x) = 1 + x^2
        let vals = [Fr::from_u64(2), Fr::from_u64(1), Fr::from_u64(2)];
        let coeffs = interpolate_to_coeffs(-1, &vals);

        // Verify at each domain point
        for (k, &expected) in vals.iter().enumerate() {
            let x = Fr::from_i64(-1 + k as i64);
            let mut val = Fr::zero();
            let mut x_pow = Fr::one();
            for &c in &coeffs {
                val += c * x_pow;
                x_pow *= x;
            }
            assert_eq!(val, expected, "mismatch at x={}", -1 + k as i64);
        }
    }

    #[test]
    fn interpolate_roundtrip_with_poly_mul() {
        // Interpolate, multiply by (x - 5), check evaluations
        let vals = [Fr::from_u64(3), Fr::from_u64(7), Fr::from_u64(13)];
        let coeffs = interpolate_to_coeffs(0, &vals);
        let linear = [Fr::from_i64(-5), Fr::one()]; // (x - 5)
        let product = poly_mul(&coeffs, &linear);

        // Verify product at x = 0,1,2
        for (i, &f_val) in vals.iter().enumerate() {
            let x = Fr::from_u64(i as u64);
            let mut val = Fr::zero();
            let mut x_pow = Fr::one();
            for &c in &product {
                val += c * x_pow;
                x_pow *= x;
            }
            let expected = f_val * (x - Fr::from_u64(5));
            assert_eq!(val, expected, "product mismatch at x={i}");
        }
    }

    #[test]
    fn lagrange_evals_agrees_with_interpolation() {
        // Verify that lagrange_evals computes the same as interpolating
        // indicator values and evaluating at r
        let r = Fr::from_u64(17);
        let domain_start = -3i64;
        let domain_size = 7;
        let evals = lagrange_evals(domain_start, domain_size, r);

        for i in 0..domain_size {
            // Indicator values: 1 at position i, 0 elsewhere
            let mut indicator = vec![Fr::zero(); domain_size];
            indicator[i] = Fr::one();
            let coeffs = interpolate_to_coeffs(domain_start, &indicator);
            let mut val = Fr::zero();
            let mut x_pow = Fr::one();
            for &c in &coeffs {
                val += c * x_pow;
                x_pow *= r;
            }
            assert_eq!(evals[i], val, "L_{i}(17) mismatch");
        }
    }
}
