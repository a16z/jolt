//! Lagrange evaluation and interpolation over integer domains and distinct nodes.
//!
//! Provides building blocks for the univariate skip optimization in sumcheck
//! protocols. The `binary` feature adds raw-ordered embedded `F8` domains.

use std::fmt;

use jolt_field::Field;
#[cfg(feature = "binary")]
use jolt_field::F8;
use thiserror::Error;

/// Invalid nodes or values supplied to node-generic interpolation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum LagrangeNodesError {
    /// At least one node is required.
    #[error("Lagrange node list must be non-empty")]
    EmptyNodes,
    /// The lexicographically first pair of equal nodes, with `first < second`.
    #[error("Lagrange nodes at positions {first} and {second} are equal")]
    RepeatedNode { first: usize, second: usize },
    /// There must be exactly one value per node.
    #[error("expected {nodes} Lagrange values, got {values}")]
    LengthMismatch { nodes: usize, values: usize },
}

/// Rejects an empty node list or its lexicographically first repeated pair.
pub fn validate_nodes<F: Field>(nodes: &[F]) -> Result<(), LagrangeNodesError> {
    if nodes.is_empty() {
        return Err(LagrangeNodesError::EmptyNodes);
    }
    for (first, x) in nodes.iter().enumerate() {
        for (second, y) in nodes.iter().enumerate().skip(first + 1) {
            if x == y {
                return Err(LagrangeNodesError::RepeatedNode { first, second });
            }
        }
    }
    Ok(())
}

/// Evaluates the Lagrange basis at `r` over arbitrary distinct `nodes`.
///
/// Returns the node indicator when `r` is a node. Validates the node list
/// before examining `r`, rejecting emptiness or the first repeated pair.
/// Uses O(N²) multiplications and N inversions.
pub fn lagrange_evals_at_nodes<F: Field>(nodes: &[F], r: F) -> Result<Vec<F>, LagrangeNodesError> {
    validate_nodes(nodes)?;
    Ok(nodes
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let mut numerator = F::one();
            let mut denominator = F::one();
            for (j, &y) in nodes.iter().enumerate() {
                if i != j {
                    numerator *= r - y;
                    denominator *= x - y;
                }
            }
            // Distinct nodes make the denominator nonzero.
            numerator * denominator.inv_or_zero()
        })
        .collect())
}

/// Interpolates `values` at arbitrary distinct `nodes`, with O(N²)
/// multiplications and N inversions.
///
/// Returns exactly `nodes.len()` monomial coefficients, low degree first,
/// retaining trailing zeros. Rejects an empty or repeated node list before
/// checking that the value count matches the node count.
pub fn interpolate_nodes_to_coeffs<F: Field>(
    nodes: &[F],
    values: &[F],
) -> Result<Vec<F>, LagrangeNodesError> {
    validate_nodes(nodes)?;
    if nodes.len() != values.len() {
        return Err(LagrangeNodesError::LengthMismatch {
            nodes: nodes.len(),
            values: values.len(),
        });
    }

    // vanishing = prod_j (X - x_j), low degree first.
    let mut vanishing = Vec::with_capacity(nodes.len() + 1);
    vanishing.push(F::one());
    for &x in nodes {
        vanishing.push(F::zero());
        let mut lower = F::zero();
        for coefficient in &mut vanishing {
            let old = *coefficient;
            *coefficient = lower - x * old;
            lower = old;
        }
    }

    // p = sum_i values_i * q_i / q_i(x_i), with q_i = vanishing / (X - x_i).
    let mut coefficients = vec![F::zero(); nodes.len()];
    let mut quotient = vec![F::zero(); nodes.len()];
    for (&x, &value) in nodes.iter().zip(values) {
        let mut higher = F::zero();
        for (q, &v) in quotient.iter_mut().rev().zip(vanishing.iter().rev()) {
            higher = v + x * higher;
            *q = higher;
        }
        let denominator = quotient.iter().rev().fold(F::zero(), |acc, &q| acc * x + q);
        // Distinct nodes make the denominator nonzero.
        let scale = value * denominator.inv_or_zero();
        for (coefficient, &q) in coefficients.iter_mut().zip(&quotient) {
            *coefficient += scale * q;
        }
    }
    Ok(coefficients)
}

/// Maximum number of distinct points in an embedded `F8` domain.
#[cfg(feature = "binary")]
pub const F8_DOMAIN_MAX_SIZE: usize = 256;

/// An embedded `F8` domain size outside `1..=F8_DOMAIN_MAX_SIZE`.
#[cfg(feature = "binary")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
#[error("F8 domain size must be between 1 and {F8_DOMAIN_MAX_SIZE}, got {size}")]
pub struct F8DomainSizeError {
    /// Requested number of domain points.
    pub size: usize,
}

/// Returns the first `size` images of `F8` elements in raw-byte order.
///
/// `From<F8>` must be a field embedding; the implementations for `F64`,
/// `F128`, and `F192` satisfy this requirement. It makes the points distinct
/// and each smaller domain a prefix of every larger domain.
/// Returns [`F8DomainSizeError`] unless `1 <= size <= 256`.
#[cfg(feature = "binary")]
pub fn f8_domain_nodes<F: Field + From<F8>>(size: usize) -> Result<Vec<F>, F8DomainSizeError> {
    if !(1..=F8_DOMAIN_MAX_SIZE).contains(&size) {
        return Err(F8DomainSizeError { size });
    }
    Ok((0..size).map(|i| F::from(F8::from_raw(i as u8))).collect())
}

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

    let mut coeffs = vec![F::zero(); n];
    let mut basis = vec![F::zero(); n];
    basis[0] = F::one();

    for (k, &dd_k) in dd.iter().enumerate() {
        for (i, &b) in basis.iter().enumerate().take(k + 1) {
            coeffs[i] += dd_k * b;
        }
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
    use jolt_field::{Fr, Ring};
    use num_traits::{One, Zero};

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
        let r = Fr::from_u64(7);
        let evals = lagrange_evals(-2, 5, r);
        let sum: Fr = evals.iter().copied().sum();
        assert_eq!(sum, Fr::one());
    }

    #[test]
    fn lagrange_evals_symmetric_at_node() {
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

        assert_eq!(evals, lagrange_evals(-2, 5, r));
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
    fn interpolate_to_coeffs_linear() {
        let vals = [Fr::from_u64(1), Fr::from_u64(3)];
        let coeffs = interpolate_to_coeffs(0, &vals);
        assert_eq!(coeffs[0], Fr::from_u64(1));
        assert_eq!(coeffs[1], Fr::from_u64(2));
    }

    #[test]
    fn interpolate_to_coeffs_symmetric_domain() {
        let vals = [Fr::from_u64(2), Fr::from_u64(1), Fr::from_u64(2)];
        let coeffs = interpolate_to_coeffs(-1, &vals);

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
    fn lagrange_evals_agrees_with_interpolation() {
        let r = Fr::from_u64(17);
        let domain_start = -3i64;
        let domain_size = 7;
        let evals = lagrange_evals(domain_start, domain_size, r);

        for i in 0..domain_size {
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
