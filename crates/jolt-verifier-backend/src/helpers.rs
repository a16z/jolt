//! Backend-aware verifier primitives.
//!
//! Free functions that compute verifier-side quantities (eq evaluation,
//! univariate Horner evaluation, repeated squaring) through a
//! [`FieldBackend`]. They live here rather than in `jolt-poly` /
//! `jolt-sumcheck` because adding `jolt-verifier-backend` as a dependency to
//! every leaf crate would create unnecessary coupling: only the verifier and
//! its tests need the backend abstraction at all.

use crate::backend::FieldBackend;

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
