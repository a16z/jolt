//! `eq`-family evaluations and challenge powers with every field
//! multiplication routed through the verifier's operation counter
//! (`stream::TermObserver::fr_mul`). A verifier that derives T1's statement
//! through `adapter::StreamTermExporter::input_claims` and its terms through
//! `terms_observed` reports an execution-derived `VerifierCost`; the plain
//! `T1Challenges::from_challenges` / `input_claims` are the prover's. Powers
//! of two are constants of the relation, not verifier arithmetic;
//! multiplying by one is.

use jolt_field::{Fr, One, Ring, Zero};

/// A field multiplication as the verifier performs it.
pub type Mul<'a> = &'a mut dyn FnMut(Fr, Fr) -> Fr;

/// The uncounted multiplication (prover side, test oracles).
pub fn plain(left: Fr, right: Fr) -> Fr {
    left * right
}

/// `2^k` as a relation constant.
pub fn pow2(k: usize) -> Fr {
    Fr::one().mul_pow_2(k)
}

/// `eq(a, b) = ab + (1 − a)(1 − b)` (one multiplication).
pub fn eq_scalar_with(a: Fr, b: Fr, mul: Mul<'_>) -> Fr {
    let ab = mul(a, b);
    Fr::one() - a - b + ab + ab
}

/// `eq(x, y) = Π_i eq(x_i, y_i)`.
pub fn eq_points_with(x: &[Fr], y: &[Fr], mul: Mul<'_>) -> Fr {
    debug_assert_eq!(x.len(), y.len());
    let mut acc = Fr::one();
    for (i, (a, b)) in x.iter().zip(y).enumerate() {
        let factor = eq_scalar_with(*a, *b, mul);
        acc = if i == 0 { factor } else { mul(acc, factor) };
    }
    acc
}

/// `eq(x, 0) = Π_i (1 − x_i)`.
pub fn eq_zero_with(x: &[Fr], mul: Mul<'_>) -> Fr {
    let mut acc = Fr::one();
    for (i, a) in x.iter().enumerate() {
        let factor = Fr::one() - *a;
        acc = if i == 0 { factor } else { mul(acc, factor) };
    }
    acc
}

/// The table `eq(point, j)` over `j ∈ {0,1}^n`, big-endian (`point[0]` is the
/// top bit of `j`): `2^n − 1` multiplications.
pub fn eq_evals_with(point: &[Fr], mul: Mul<'_>) -> Vec<Fr> {
    let mut evals = vec![Fr::one()];
    for r in point {
        let mut next = Vec::with_capacity(2 * evals.len());
        for e in &evals {
            let hi = mul(*e, *r);
            next.push(*e - hi);
            next.push(hi);
        }
        evals = next;
    }
    evals
}

/// `eq+1(x, y)`: 1 iff `y = x + 1` on the hypercube (`jolt_poly::EqPlusOnePolynomial`
/// semantics, `x` the polynomial's point). Writing bit `k` LSB-first, the
/// increment flips bit `k` (`x_k = 0`, `y_k = 1`), clears the lower bits
/// (`x_i = 1`, `y_i = 0`) and keeps the higher ones (`eq`).
pub fn eq_plus_one_with(x: &[Fr], y: &[Fr], mul: Mul<'_>) -> Fr {
    let l = x.len();
    debug_assert_eq!(y.len(), l);
    // LSB-first views.
    let bit = |v: &[Fr], k: usize| v[l - 1 - k];
    let mut lower = Vec::with_capacity(l);
    let mut acc = Fr::one();
    for k in 0..l {
        lower.push(acc);
        let cleared = mul(bit(x, k), Fr::one() - bit(y, k));
        acc = if k == 0 { cleared } else { mul(acc, cleared) };
    }
    let mut higher = vec![Fr::one(); l];
    let mut acc = Fr::one();
    for k in (0..l).rev() {
        higher[k] = acc;
        let kept = eq_scalar_with(bit(x, k), bit(y, k), mul);
        acc = if k == l - 1 { kept } else { mul(acc, kept) };
    }
    (0..l).fold(Fr::zero(), |sum, k| {
        let flip = mul(Fr::one() - bit(x, k), bit(y, k));
        let with_lower = mul(lower[k], flip);
        sum + mul(with_lower, higher[k])
    })
}

/// `[1, base, …, base^(count − 1)]`.
pub fn powers_with(base: Fr, count: usize, mul: Mul<'_>) -> Vec<Fr> {
    let mut powers = Vec::with_capacity(count);
    let mut acc = Fr::one();
    for i in 0..count {
        powers.push(acc);
        if i + 1 < count {
            acc = if i == 0 { base } else { mul(acc, base) };
        }
    }
    powers
}
