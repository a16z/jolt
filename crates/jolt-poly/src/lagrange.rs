//! Lagrange interpolation utilities over integer domains.
//!
//! Provides building blocks for the univariate skip optimization in sumcheck
//! protocols. All functions are generic over [`Field`] and operate on
//! integer-indexed domains (symmetric or arbitrary).

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
mod tests {
    use super::*;
    use jolt_field::Fr;
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
