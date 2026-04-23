//! Backend-aware verifier primitives.
//!
//! Free functions that compute verifier-side quantities (eq evaluation,
//! Lagrange basis, univariate Horner, indicator MLEs, sparse block eval)
//! through a [`FieldBackend`]. They live here rather than in `jolt-poly` /
//! `jolt-sumcheck` because adding `jolt-verifier-backend` as a dependency
//! to every leaf crate would create unnecessary coupling: only the
//! verifier and its tests need the backend abstraction at all.
//!
//! The native fallback for each helper matches its `jolt-poly` /
//! verifier-internal counterpart bit-for-bit (modulo associativity of the
//! accumulator). See the per-function tests for parity proofs.

use jolt_openings::{BackendError, FieldBackend};

/// Evaluates the multilinear equality polynomial
/// `eq(a, b) = Π_i (a_i b_i + (1 - a_i)(1 - b_i))`.
///
/// Uses `2n + 1` multiplications, `2n` subtractions, `n` additions through
/// `backend`. With [`Native`](crate::Native) this is bit-identical to
/// [`EqPolynomial::evaluate`](https://docs.rs/jolt-poly/) (modulo associativity
/// of accumulator); with Tracing each op is recorded.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
pub fn eq_eval<B: FieldBackend>(backend: &mut B, a: &[B::Scalar], b: &[B::Scalar]) -> B::Scalar {
    assert_eq!(a.len(), b.len(), "eq dimension mismatch");

    let one = backend.const_one();
    let mut acc = backend.const_one();
    for (a_i, b_i) in a.iter().zip(b.iter()) {
        let one_minus_a = backend.sub(&one, a_i);
        let one_minus_b = backend.sub(&one, b_i);
        let ab = backend.mul(a_i, b_i);
        let cross = backend.mul(&one_minus_a, &one_minus_b);
        let term = backend.add(&ab, &cross);
        acc = backend.mul(&acc, &term);
    }
    acc
}

/// Evaluates a univariate polynomial in coefficient form via Horner's rule.
///
/// `coefficients` is in ascending degree order: `coefficients[i]` is the
/// coefficient of `x^i`. An empty coefficient slice represents the zero
/// polynomial and yields `backend.const_zero()`.
pub fn univariate_horner<B: FieldBackend>(
    backend: &mut B,
    coefficients: &[B::Scalar],
    point: &B::Scalar,
) -> B::Scalar {
    let mut iter = coefficients.iter().rev();
    let Some(first) = iter.next() else {
        return backend.const_zero();
    };
    let mut acc = first.clone();
    for c in iter {
        let scaled = backend.mul(&acc, point);
        acc = backend.add(&scaled, c);
    }
    acc
}

/// Materializes the table `[eq(r, x) : x ∈ {0,1}^n]` of length `2^n`.
///
/// Mirrors [`EqPolynomial::evaluations`](https://docs.rs/jolt-poly/) using the
/// same big-endian index convention (`r[0]` is the most-significant bit), but
/// every multiplication and subtraction routes through `backend`. Used by the
/// backend-aware Spartan matrix MLE evaluator.
pub fn eq_evals_table<B: FieldBackend>(backend: &mut B, point: &[B::Scalar]) -> Vec<B::Scalar> {
    let one = backend.const_one();
    let mut table: Vec<B::Scalar> = vec![one.clone()];

    for r_i in point {
        let one_minus_r = backend.sub(&one, r_i);
        let prev_len = table.len();
        let mut next: Vec<B::Scalar> = Vec::with_capacity(prev_len * 2);
        for v in &table {
            next.push(backend.mul(v, &one_minus_r));
            next.push(backend.mul(v, r_i));
        }
        table = next;
    }

    table
}

/// Computes `base^exp` via repeated squaring through the backend.
pub fn pow_u64<B: FieldBackend>(backend: &mut B, base: &B::Scalar, exp: u64) -> B::Scalar {
    if exp == 0 {
        return backend.const_one();
    }
    let mut result = backend.const_one();
    let mut squared = base.clone();
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result = backend.mul(&result, &squared);
        }
        e >>= 1;
        if e > 0 {
            squared = backend.square(&squared);
        }
    }
    result
}

/// Lifts a backend constant for an integer index in a Lagrange domain.
#[inline]
fn const_i64<B: FieldBackend>(backend: &mut B, x: i64) -> B::Scalar {
    backend.const_i128(i128::from(x))
}

/// Backend-aware [`lagrange_evals`](jolt_poly::lagrange::lagrange_evals).
///
/// Evaluates all Lagrange basis polynomials `L_0(r), ..., L_{N-1}(r)` over
/// the integer domain `{s, s+1, ..., s+N-1}` (where `s = domain_start`)
/// through `backend`. Mirrors the native version's barycentric formula.
///
/// The "is `r` exactly a node?" early-exit cannot be checked symbolically,
/// so this routes through the full barycentric formula in every backend
/// (Tracing-accurate). Native callers that already hit the early-exit
/// path stay on `jolt_poly::lagrange::lagrange_evals` directly; this
/// function is only used inside backend-aware claim formula evaluation,
/// where the early exit doesn't fire because `r` is a Fiat-Shamir
/// challenge.
///
/// # Errors
/// Returns [`BackendError::InverseOfZero`] if `r` happens to coincide with
/// a domain node (which is a Fiat-Shamir-soundness issue at the field
/// level — vanishingly unlikely on the prime fields Jolt uses).
///
/// # Panics
/// Panics if `domain_size == 0`.
pub fn lagrange_evals<B: FieldBackend>(
    backend: &mut B,
    domain_start: i64,
    domain_size: usize,
    r: &B::Scalar,
) -> Result<Vec<B::Scalar>, BackendError> {
    assert!(domain_size > 0, "domain_size must be positive");

    // diffs[i] = r - (domain_start + i)
    let mut diffs: Vec<B::Scalar> = Vec::with_capacity(domain_size);
    for k in 0..domain_size {
        let node = const_i64(backend, domain_start + k as i64);
        diffs.push(backend.sub(r, &node));
    }

    // full_product = ∏ diffs[i]
    let mut full_product = backend.const_one();
    for d in &diffs {
        full_product = backend.mul(&full_product, d);
    }

    // Native barycentric weights w_i = 1 / ∏_{j≠i} (i - j) baked at
    // `i128` precision and lifted as a backend constant.
    let mut weights_native: Vec<i128> = vec![1; domain_size];
    for (i, w) in weights_native.iter_mut().enumerate() {
        for j in 0..domain_size {
            if i != j {
                *w *= (i as i128) - (j as i128);
            }
        }
    }

    let mut result = Vec::with_capacity(domain_size);
    for (i, w_native) in weights_native.iter().enumerate() {
        let w_const = backend.const_i128(*w_native);
        let w_inv = backend.inverse(&w_const, "lagrange_evals_weight_inv")?;
        let diff_inv = backend.inverse(&diffs[i], "lagrange_evals_diff_inv")?;
        let scaled = backend.mul(&full_product, &w_inv);
        result.push(backend.mul(&scaled, &diff_inv));
    }

    Ok(result)
}

/// Backend-aware single-basis Lagrange evaluation.
///
/// Computes `L_k(r) = ∏_{j ≠ k} (r - (s+j)) / ((s+k) - (s+j))` through the
/// backend. Mirrors [`jolt_poly::lagrange::lagrange_basis_eval`].
///
/// # Errors
/// Returns [`BackendError::InverseOfZero`] if the constant denominator is
/// zero (impossible for a well-formed Lagrange domain).
pub fn lagrange_basis_eval<B: FieldBackend>(
    backend: &mut B,
    domain_start: i64,
    domain_size: usize,
    k: usize,
    r: &B::Scalar,
) -> Result<B::Scalar, BackendError> {
    let mut numer = backend.const_one();
    let mut denom_native: i128 = 1;
    for j in 0..domain_size {
        if j == k {
            continue;
        }
        let node = const_i64(backend, domain_start + j as i64);
        let factor = backend.sub(r, &node);
        numer = backend.mul(&numer, &factor);
        denom_native *= (k as i128) - (j as i128);
    }
    let denom = backend.const_i128(denom_native);
    let denom_inv = backend.inverse(&denom, "lagrange_basis_eval_denom_inv")?;
    Ok(backend.mul(&numer, &denom_inv))
}

/// Backend-aware Lagrange kernel evaluation `L(τ, r) = Σ L_k(τ) · L_k(r)`.
///
/// Mirrors [`jolt_poly::lagrange::lagrange_kernel_eval`].
///
/// # Errors
/// Forwards [`BackendError::InverseOfZero`] from the underlying
/// [`lagrange_evals`] calls (see that function's docs).
pub fn lagrange_kernel_eval<B: FieldBackend>(
    backend: &mut B,
    domain_start: i64,
    domain_size: usize,
    tau: &B::Scalar,
    r: &B::Scalar,
) -> Result<B::Scalar, BackendError> {
    let tau_evals = lagrange_evals(backend, domain_start, domain_size, tau)?;
    let r_evals = lagrange_evals(backend, domain_start, domain_size, r)?;
    let mut acc = backend.const_zero();
    for (a, b) in tau_evals.iter().zip(r_evals.iter()) {
        let prod = backend.mul(a, b);
        acc = backend.add(&acc, &prod);
    }
    Ok(acc)
}

/// Backend-aware MLE of the indicator `{x < threshold}` over the Boolean
/// hypercube of dimension `r.len()`.
///
/// Mirrors `lt_mle` in `jolt-verifier::verifier`. Scans threshold bits MSB
/// first, accumulating the running probability that a point is less than
/// the threshold.
///
/// # Panics
/// In debug mode, panics if `threshold >= 2^r.len()`.
pub fn lt_mle<B: FieldBackend>(backend: &mut B, r: &[B::Scalar], threshold: u128) -> B::Scalar {
    let n = r.len();
    debug_assert!(
        n == 128 || threshold < (1u128 << n),
        "threshold exceeds domain"
    );
    let one = backend.const_one();
    let mut lt = backend.const_zero();
    let mut eq = backend.const_one();
    for (i, ri) in r.iter().enumerate() {
        let bit = (threshold >> (n - 1 - i)) & 1;
        if bit == 1 {
            // (1 - r_i) branch contributes to lt; r_i branch carries eq forward.
            let one_minus_ri = backend.sub(&one, ri);
            let term = backend.mul(&eq, &one_minus_ri);
            lt = backend.add(&lt, &term);
            eq = backend.mul(&eq, ri);
        } else {
            let one_minus_ri = backend.sub(&one, ri);
            eq = backend.mul(&eq, &one_minus_ri);
        }
    }
    lt
}

/// Backend-aware MLE of the identity polynomial `f(x) = ∑ r_i · 2^(n-1-i)`.
///
/// Mirrors `identity_mle` in `jolt-verifier::verifier`.
pub fn identity_mle<B: FieldBackend>(backend: &mut B, r: &[B::Scalar]) -> B::Scalar {
    let n = r.len();
    let mut sum = backend.const_zero();
    for (i, ri) in r.iter().enumerate() {
        let weight = backend.const_i128(1i128 << (n - 1 - i));
        let term = backend.mul(ri, &weight);
        sum = backend.add(&sum, &term);
    }
    sum
}

/// Backend-aware `eq(idx, r)` where `idx` is a Boolean hypercube point
/// (MSB-first integer encoding).
///
/// Mirrors `eq_at_index` in `jolt-verifier::verifier`. Each bit of `idx`
/// chooses between `r_i` and `1 - r_i`.
pub fn eq_at_index<B: FieldBackend>(backend: &mut B, idx: usize, r: &[B::Scalar]) -> B::Scalar {
    let n = r.len();
    let one = backend.const_one();
    let mut prod = backend.const_one();
    for (i, ri) in r.iter().enumerate() {
        let bit = (idx >> (n - 1 - i)) & 1;
        let factor = if bit == 1 {
            ri.clone()
        } else {
            backend.sub(&one, ri)
        };
        prod = backend.mul(&prod, &factor);
    }
    prod
}

/// Backend-aware sparse block evaluation `Σ_j values[j] · eq(start + j, r)`.
///
/// `values` is a list of native `u64` scalars (typically I/O bytes packed
/// into words). Zero entries are skipped, matching the native sparse
/// path in `jolt-verifier::verifier::sparse_block_eval`.
pub fn sparse_block_eval<B: FieldBackend>(
    backend: &mut B,
    start: usize,
    values: &[u64],
    r: &[B::Scalar],
) -> B::Scalar {
    let mut acc = backend.const_zero();
    for (j, &val) in values.iter().enumerate() {
        if val == 0 {
            continue;
        }
        let eq = eq_at_index(backend, start + j, r);
        let v_const = backend.const_i128(i128::from(val));
        let term = backend.mul(&eq, &v_const);
        acc = backend.add(&acc, &term);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::Native;
    use jolt_field::{Field, Fr};
    use num_traits::{One, Zero};
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;

    fn wrap_slice<B: FieldBackend>(backend: &mut B, xs: &[B::F]) -> Vec<B::Scalar> {
        xs.iter().map(|x| backend.wrap_proof(*x, "x")).collect()
    }

    #[test]
    fn eq_eval_matches_direct_eq() {
        let mut rng = ChaCha8Rng::seed_from_u64(0xeed);
        for n in 0..6 {
            let a: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
            let b: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

            let direct: Fr = a.iter().zip(b.iter()).fold(Fr::one(), |acc, (a_i, b_i)| {
                acc * (*a_i * *b_i + (Fr::one() - *a_i) * (Fr::one() - *b_i))
            });

            let mut backend = Native::<Fr>::new();
            let aw = wrap_slice(&mut backend, &a);
            let bw = wrap_slice(&mut backend, &b);
            let via_backend = eq_eval(&mut backend, &aw, &bw);

            assert_eq!(direct, via_backend, "n = {n}");
        }
    }

    #[test]
    fn eq_eval_at_boolean_points() {
        let mut backend = Native::<Fr>::new();
        let r = wrap_slice(&mut backend, &[Fr::from_u64(3), Fr::from_u64(5)]);
        let zero = backend.const_zero();
        let one = backend.const_one();

        let at_00 = eq_eval(&mut backend, &r, &[zero, zero]);
        let at_11 = eq_eval(&mut backend, &r, &[one, one]);
        let direct_00 = (Fr::one() - Fr::from_u64(3)) * (Fr::one() - Fr::from_u64(5));
        let direct_11 = Fr::from_u64(3) * Fr::from_u64(5);
        assert_eq!(at_00, direct_00);
        assert_eq!(at_11, direct_11);
    }

    #[test]
    fn eq_eval_empty_is_one() {
        let mut backend = Native::<Fr>::new();
        let v = eq_eval(&mut backend, &[], &[]);
        assert_eq!(v, Fr::one());
    }

    #[test]
    fn univariate_horner_matches_naive() {
        let mut rng = ChaCha8Rng::seed_from_u64(0xeed);
        for d in 0..8 {
            let coeffs: Vec<Fr> = (0..=d).map(|_| Fr::random(&mut rng)).collect();
            let pt = Fr::random(&mut rng);

            let mut naive = Fr::zero();
            let mut x_pow = Fr::one();
            for c in &coeffs {
                naive += *c * x_pow;
                x_pow *= pt;
            }

            let mut backend = Native::<Fr>::new();
            let cw = wrap_slice(&mut backend, &coeffs);
            let pw = backend.wrap_challenge(pt, "pt");
            let via_backend = univariate_horner(&mut backend, &cw, &pw);

            assert_eq!(naive, via_backend, "deg {d}");
        }
    }

    #[test]
    fn univariate_horner_empty_is_zero() {
        let mut backend = Native::<Fr>::new();
        let pt = backend.const_one();
        let v = univariate_horner(&mut backend, &[], &pt);
        assert_eq!(v, Fr::zero());
    }

    #[test]
    fn eq_evals_table_matches_jolt_poly() {
        use jolt_poly::EqPolynomial;
        let mut rng = ChaCha8Rng::seed_from_u64(0xeed);
        for n in 0..5 {
            let r: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
            let direct = EqPolynomial::new(r.clone()).evaluations();

            let mut backend = Native::<Fr>::new();
            let rw = wrap_slice(&mut backend, &r);
            let via_backend = eq_evals_table(&mut backend, &rw);

            assert_eq!(direct, via_backend, "n = {n}");
        }
    }

    #[test]
    fn pow_u64_zero_is_one() {
        let mut backend = Native::<Fr>::new();
        let b = backend.wrap_proof(Fr::from_u64(7), "b");
        let v = pow_u64(&mut backend, &b, 0);
        assert_eq!(v, Fr::one());
    }

    #[test]
    fn lagrange_evals_matches_jolt_poly() {
        use jolt_poly::lagrange::lagrange_evals as direct_lagrange_evals;
        let mut rng = ChaCha8Rng::seed_from_u64(0xeed);
        for &(start, n) in &[(0i64, 1), (0, 3), (-2, 5), (-3, 7), (5, 4)] {
            for _ in 0..4 {
                let r = Fr::random(&mut rng);
                let direct = direct_lagrange_evals(start, n, r);

                let mut backend = Native::<Fr>::new();
                let rw = backend.wrap_challenge(r, "r");
                let via_backend = lagrange_evals(&mut backend, start, n, &rw).unwrap();

                assert_eq!(direct, via_backend, "start={start}, n={n}");
            }
        }
    }

    #[test]
    fn lagrange_basis_eval_matches_jolt_poly() {
        use jolt_poly::lagrange::lagrange_basis_eval as direct_basis;
        let mut rng = ChaCha8Rng::seed_from_u64(0xeed);
        for &(start, n) in &[(0i64, 3), (-2, 5), (5, 4)] {
            for k in 0..n {
                let r = Fr::random(&mut rng);
                let direct = direct_basis(start, n, k, r);

                let mut backend = Native::<Fr>::new();
                let rw = backend.wrap_challenge(r, "r");
                let via_backend = lagrange_basis_eval(&mut backend, start, n, k, &rw).unwrap();

                assert_eq!(direct, via_backend, "start={start}, n={n}, k={k}");
            }
        }
    }

    #[test]
    fn lagrange_kernel_eval_matches_jolt_poly() {
        use jolt_poly::lagrange::lagrange_kernel_eval as direct_kernel;
        let mut rng = ChaCha8Rng::seed_from_u64(0xeed);
        for &(start, n) in &[(0i64, 3), (-2, 5), (5, 4)] {
            for _ in 0..4 {
                let tau = Fr::random(&mut rng);
                let r = Fr::random(&mut rng);
                let direct = direct_kernel(start, n, tau, r);

                let mut backend = Native::<Fr>::new();
                let tau_w = backend.wrap_challenge(tau, "tau");
                let r_w = backend.wrap_challenge(r, "r");
                let via_backend =
                    lagrange_kernel_eval(&mut backend, start, n, &tau_w, &r_w).unwrap();

                assert_eq!(direct, via_backend, "start={start}, n={n}");
            }
        }
    }

    #[test]
    fn lt_mle_matches_naive() {
        let mut rng = ChaCha8Rng::seed_from_u64(0xeed);
        for n in 1..=5 {
            let r: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
            for threshold in 0u128..(1u128 << n) {
                // Naive sum over the hypercube.
                let mut naive = Fr::zero();
                for x in 0u128..(1u128 << n) {
                    if x < threshold {
                        let mut term = Fr::one();
                        for (i, ri) in r.iter().enumerate() {
                            let bit = (x >> (n - 1 - i)) & 1;
                            term *= if bit == 1 { *ri } else { Fr::one() - *ri };
                        }
                        naive += term;
                    }
                }

                let mut backend = Native::<Fr>::new();
                let rw = wrap_slice(&mut backend, &r);
                let via_backend = lt_mle(&mut backend, &rw, threshold);

                assert_eq!(naive, via_backend, "n={n}, threshold={threshold}");
            }
        }
    }

    #[test]
    fn identity_mle_matches_native() {
        let mut rng = ChaCha8Rng::seed_from_u64(0xeed);
        for n in 1..=8 {
            let r: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
            let mut naive = Fr::zero();
            for (i, ri) in r.iter().enumerate() {
                naive += *ri * Fr::from_u128(1u128 << (n - 1 - i));
            }

            let mut backend = Native::<Fr>::new();
            let rw = wrap_slice(&mut backend, &r);
            let via_backend = identity_mle(&mut backend, &rw);

            assert_eq!(naive, via_backend, "n={n}");
        }
    }

    #[test]
    fn eq_at_index_matches_naive() {
        let mut rng = ChaCha8Rng::seed_from_u64(0xeed);
        for n in 1..=5 {
            let r: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
            for idx in 0usize..(1 << n) {
                let mut naive = Fr::one();
                for (i, ri) in r.iter().enumerate() {
                    let bit = (idx >> (n - 1 - i)) & 1;
                    naive *= if bit == 1 { *ri } else { Fr::one() - *ri };
                }

                let mut backend = Native::<Fr>::new();
                let rw = wrap_slice(&mut backend, &r);
                let via_backend = eq_at_index(&mut backend, idx, &rw);

                assert_eq!(naive, via_backend, "n={n}, idx={idx}");
            }
        }
    }

    #[test]
    fn sparse_block_eval_matches_naive() {
        let mut rng = ChaCha8Rng::seed_from_u64(0xeed);
        for n in 2..=5 {
            let r: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
            let values = [0u64, 7, 0, 3, 5, 0, 1, 2];
            let start = 1usize;

            let mut naive = Fr::zero();
            for (j, &val) in values.iter().enumerate() {
                if val == 0 {
                    continue;
                }
                let idx = start + j;
                if idx >= (1 << n) {
                    continue;
                }
                let mut eq = Fr::one();
                for (i, ri) in r.iter().enumerate() {
                    let bit = (idx >> (n - 1 - i)) & 1;
                    eq *= if bit == 1 { *ri } else { Fr::one() - *ri };
                }
                naive += eq * Fr::from_u64(val);
            }

            let mut backend = Native::<Fr>::new();
            let rw = wrap_slice(&mut backend, &r);
            let truncated: Vec<u64> = values
                .iter()
                .copied()
                .take((1usize << n).saturating_sub(start))
                .collect();
            let via_backend = sparse_block_eval(&mut backend, start, &truncated, &rw);

            assert_eq!(naive, via_backend, "n={n}");
        }
    }

    #[test]
    fn pow_u64_matches_naive() {
        let mut rng = ChaCha8Rng::seed_from_u64(0xeed);
        for _ in 0..8 {
            let base = Fr::random(&mut rng);
            for e in [1u64, 2, 3, 4, 7, 16, 31, 64, 100, 1023] {
                let mut naive = Fr::one();
                for _ in 0..e {
                    naive *= base;
                }

                let mut backend = Native::<Fr>::new();
                let bw = backend.wrap_proof(base, "b");
                let via_backend = pow_u64(&mut backend, &bw, e);

                assert_eq!(naive, via_backend, "e = {e}");
            }
        }
    }
}
