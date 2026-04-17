//! Gruen split-eq utilities for cubic sumcheck round-polynomial assembly.
//!
//! Ports the `gruen_poly_deg_3` scalar formula from jolt-core's
//! `split_eq_poly.rs`. See https://eprint.iacr.org/2024/1210.pdf (Dao-Thaler
//! + Gruen) for the underlying technique.
//!
//! This is a pure scalar helper used by the modular stack's outer-Spartan
//! sumcheck prover to assemble the round cubic `s(X) = l(X) · q(X)` from
//! the linear eq factor `l(X) = a + bX` and the quadratic `q(X) = c + dX + eX²`,
//! where `d` is inferred from the previous-round claim `s(0) + s(1)` without
//! requiring a second pass over the witness to evaluate `q` at `X = 1`.

use jolt_compute::BindingOrder;
use jolt_field::Field;

/// Assemble cubic polynomial `s(X) = l(X) · q(X)` evaluated at `{0, 1, 2, 3}`.
///
/// The linear factor `l(X)` is encoded as `(current_scalar, w_current)` where
/// `l(0) = current_scalar · (1 - w_current)` and `l(1) = current_scalar · w_current`.
/// The quadratic factor `q(X) = c + dX + eX²` is passed as just `c` and `e`;
/// `d` is recovered from `s(0) + s(1)`.
///
/// The caller is responsible for resolving `w_current` — for `LowToHigh`
/// binding this is `w[current_index - 1]`, for `HighToLow` it is
/// `w[current_index]` (see `GruenSplitEqPolynomial::gruen_poly_deg_3` in core).
pub fn gruen_cubic_evals<F: Field>(
    current_scalar: F,
    w_current: F,
    q_constant: F,
    q_quadratic_coeff: F,
    s_0_plus_s_1: F,
) -> [F; 4] {
    let eq_eval_1 = current_scalar * w_current;
    let eq_eval_0 = current_scalar - eq_eval_1;
    let eq_m = eq_eval_1 - eq_eval_0;
    let eq_eval_2 = eq_eval_1 + eq_m;
    let eq_eval_3 = eq_eval_2 + eq_m;

    let quadratic_eval_0 = q_constant;
    let cubic_eval_0 = eq_eval_0 * quadratic_eval_0;
    let cubic_eval_1 = s_0_plus_s_1 - cubic_eval_0;
    // q(1) = s(1) / l(1).
    let quadratic_eval_1 = cubic_eval_1 / eq_eval_1;
    let e_times_2 = q_quadratic_coeff + q_quadratic_coeff;
    // q(k+1) = q(k) + (q(k) - q(k-1)) + 2e, for k ≥ 1.
    let quadratic_eval_2 = quadratic_eval_1 + quadratic_eval_1 - quadratic_eval_0 + e_times_2;
    let quadratic_eval_3 =
        quadratic_eval_2 + quadratic_eval_1 - quadratic_eval_0 + e_times_2 + e_times_2;

    [
        cubic_eval_0,
        cubic_eval_1,
        eq_eval_2 * quadratic_eval_2,
        eq_eval_3 * quadratic_eval_3,
    ]
}

/// Fused reduce producing `(q(0), coeff_of_X²_in_q)` for a bilinear product
/// `q(X) = Σ_i e_active[i] · a_i(X) · b_i(X)` scaled by `scalar`.
///
/// Each `a_i(X)`, `b_i(X)` is the multilinear extension of a polynomial at
/// the pair `(lo_i, hi_i)`, so `a_i(X) = a_lo + X · (a_hi - a_lo)`. Their
/// product is quadratic:
///
/// - `q(0)        = Σ e_active[i] · a_lo · b_lo`
/// - `q(∞) coeff  = Σ e_active[i] · (a_hi - a_lo) · (b_hi - b_lo)`
///
/// Both outputs are multiplied by `scalar` at the end (typically the batching
/// challenge). The missing linear coefficient is recovered by the caller
/// from the previous-round claim via [`gruen_cubic_evals`].
///
/// # Layout
///
/// - `e_active.len() = half` — weights indexed by pair.
/// - `factor_a.len() = factor_b.len() = 2 * half` — pairs laid out per
///   `order`: `LowToHigh` → `(buf[2i], buf[2i+1])`; `HighToLow` →
///   `(buf[i], buf[i+half])`.
pub fn reduce_dense_gruen_deg2<F: Field>(
    e_active: &[F],
    factor_a: &[F],
    factor_b: &[F],
    scalar: F,
    order: BindingOrder,
) -> (F, F) {
    let half = e_active.len();
    debug_assert_eq!(factor_a.len(), 2 * half, "factor_a must have length 2·half");
    debug_assert_eq!(factor_b.len(), 2 * half, "factor_b must have length 2·half");

    let load_pair = |buf: &[F], i: usize| -> (F, F) {
        match order {
            BindingOrder::LowToHigh => (buf[2 * i], buf[2 * i + 1]),
            BindingOrder::HighToLow => (buf[i], buf[i + half]),
        }
    };

    let mut q_const = F::zero();
    let mut q_quad = F::zero();
    for (i, &w) in e_active.iter().enumerate() {
        let (a_lo, a_hi) = load_pair(factor_a, i);
        let (b_lo, b_hi) = load_pair(factor_b, i);
        q_const += w * a_lo * b_lo;
        q_quad += w * (a_hi - a_lo) * (b_hi - b_lo);
    }
    (scalar * q_const, scalar * q_quad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    fn eval_l(eq_eval_0: Fr, eq_eval_1: Fr, x: Fr) -> Fr {
        eq_eval_0 + x * (eq_eval_1 - eq_eval_0)
    }

    fn eval_q(c: Fr, d: Fr, e: Fr, x: Fr) -> Fr {
        c + x * d + x * x * e
    }

    #[test]
    fn gruen_cubic_matches_direct_multiplication() {
        let mut rng = ChaCha20Rng::seed_from_u64(0x0919_e100_dead_beef);
        for _ in 0..32 {
            let current_scalar = Fr::random(&mut rng);
            let w_current = Fr::random(&mut rng);
            let q_c = Fr::random(&mut rng);
            let q_d = Fr::random(&mut rng);
            let q_e = Fr::random(&mut rng);

            let eq_eval_1 = current_scalar * w_current;
            let eq_eval_0 = current_scalar - eq_eval_1;

            let direct =
                |x: Fr| -> Fr { eval_l(eq_eval_0, eq_eval_1, x) * eval_q(q_c, q_d, q_e, x) };

            let expected_0 = direct(Fr::from_u64(0));
            let expected_1 = direct(Fr::from_u64(1));
            let expected_2 = direct(Fr::from_u64(2));
            let expected_3 = direct(Fr::from_u64(3));

            let prev_claim = expected_0 + expected_1;

            let actual = gruen_cubic_evals(current_scalar, w_current, q_c, q_e, prev_claim);

            assert_eq!(actual[0], expected_0, "s(0) mismatch");
            assert_eq!(actual[1], expected_1, "s(1) mismatch");
            assert_eq!(actual[2], expected_2, "s(2) mismatch");
            assert_eq!(actual[3], expected_3, "s(3) mismatch");
        }
    }

    /// Naive reference: sum e_active[i] * a_i(X) * b_i(X) pointwise at X in
    /// {0, ∞-coeff} and return (scalar * sum_const, scalar * sum_quad).
    fn naive_reduce(
        e_active: &[Fr],
        factor_a: &[Fr],
        factor_b: &[Fr],
        scalar: Fr,
        order: BindingOrder,
    ) -> (Fr, Fr) {
        let half = e_active.len();
        let load = |buf: &[Fr], i: usize| match order {
            BindingOrder::LowToHigh => (buf[2 * i], buf[2 * i + 1]),
            BindingOrder::HighToLow => (buf[i], buf[i + half]),
        };
        let mut c = Fr::from_u64(0);
        let mut e = Fr::from_u64(0);
        for (i, &w) in e_active.iter().enumerate() {
            let (a_lo, a_hi) = load(factor_a, i);
            let (b_lo, b_hi) = load(factor_b, i);
            c += w * a_lo * b_lo;
            e += w * (a_hi - a_lo) * (b_hi - b_lo);
        }
        (scalar * c, scalar * e)
    }

    #[test]
    fn reduce_gruen_deg2_matches_naive_both_orders() {
        let mut rng = ChaCha20Rng::seed_from_u64(0xf00d_d15e_c001_b00b);
        for order in [BindingOrder::LowToHigh, BindingOrder::HighToLow] {
            for &half in &[1usize, 3, 8, 17] {
                let e_active: Vec<Fr> = (0..half).map(|_| Fr::random(&mut rng)).collect();
                let factor_a: Vec<Fr> = (0..2 * half).map(|_| Fr::random(&mut rng)).collect();
                let factor_b: Vec<Fr> = (0..2 * half).map(|_| Fr::random(&mut rng)).collect();
                let scalar = Fr::random(&mut rng);

                let (q_c, q_e) =
                    reduce_dense_gruen_deg2(&e_active, &factor_a, &factor_b, scalar, order);
                let (ref_c, ref_e) = naive_reduce(&e_active, &factor_a, &factor_b, scalar, order);

                assert_eq!(
                    q_c, ref_c,
                    "q_const mismatch for order {order:?}, half={half}"
                );
                assert_eq!(
                    q_e, ref_e,
                    "q_quad mismatch for order {order:?}, half={half}"
                );
            }
        }
    }

    #[test]
    fn reduce_gruen_deg2_composes_with_cubic_assembly() {
        // End-to-end Gruen flow: reduce a random bilinear sum, then feed
        // (q_const, q_quad) through gruen_cubic_evals along with a previous
        // round claim computed from the *same* witness. The four cubic
        // evals must equal direct per-pair cubic accumulation at X ∈ {0..3}.
        let mut rng = ChaCha20Rng::seed_from_u64(0xabad_cafe_b00b_dead);
        let half = 4usize;
        let order = BindingOrder::LowToHigh;

        let e_active: Vec<Fr> = (0..half).map(|_| Fr::random(&mut rng)).collect();
        let factor_a: Vec<Fr> = (0..2 * half).map(|_| Fr::random(&mut rng)).collect();
        let factor_b: Vec<Fr> = (0..2 * half).map(|_| Fr::random(&mut rng)).collect();
        let scalar = Fr::random(&mut rng);

        let (q_const, q_quad) =
            reduce_dense_gruen_deg2(&e_active, &factor_a, &factor_b, scalar, order);

        // Build a random l(X) factor and derive prev_claim from it.
        let current_scalar = Fr::random(&mut rng);
        let w_current = Fr::random(&mut rng);
        let eq_eval_1 = current_scalar * w_current;
        let eq_eval_0 = current_scalar - eq_eval_1;

        let load = |buf: &[Fr], i: usize| -> (Fr, Fr) { (buf[2 * i], buf[2 * i + 1]) };
        let direct_cubic = |x: Fr| -> Fr {
            let mut acc = Fr::from_u64(0);
            for (i, &w) in e_active.iter().enumerate() {
                let (a_lo, a_hi) = load(&factor_a, i);
                let (b_lo, b_hi) = load(&factor_b, i);
                let a_x = a_lo + x * (a_hi - a_lo);
                let b_x = b_lo + x * (b_hi - b_lo);
                acc += w * a_x * b_x;
            }
            let l_x = eq_eval_0 + x * (eq_eval_1 - eq_eval_0);
            scalar * acc * l_x
        };

        let expected_0 = direct_cubic(Fr::from_u64(0));
        let expected_1 = direct_cubic(Fr::from_u64(1));
        let expected_2 = direct_cubic(Fr::from_u64(2));
        let expected_3 = direct_cubic(Fr::from_u64(3));
        let prev_claim = expected_0 + expected_1;

        let evals = gruen_cubic_evals(current_scalar, w_current, q_const, q_quad, prev_claim);
        assert_eq!(evals[0], expected_0, "s(0) mismatch");
        assert_eq!(evals[1], expected_1, "s(1) mismatch");
        assert_eq!(evals[2], expected_2, "s(2) mismatch");
        assert_eq!(evals[3], expected_3, "s(3) mismatch");
    }

    #[test]
    fn gruen_cubic_interpolates_beyond_domain() {
        // The four evals describe a cubic: Lagrange-interpolating through
        // s(0..3) must reproduce direct multiplication at X = 4.
        let mut rng = ChaCha20Rng::seed_from_u64(0xcafe_babe_1234_5678);
        for _ in 0..8 {
            let current_scalar = Fr::random(&mut rng);
            let w_current = Fr::random(&mut rng);
            let q_c = Fr::random(&mut rng);
            let q_d = Fr::random(&mut rng);
            let q_e = Fr::random(&mut rng);

            let eq_eval_1 = current_scalar * w_current;
            let eq_eval_0 = current_scalar - eq_eval_1;
            let direct =
                |x: Fr| -> Fr { eval_l(eq_eval_0, eq_eval_1, x) * eval_q(q_c, q_d, q_e, x) };

            let prev_claim = direct(Fr::from_u64(0)) + direct(Fr::from_u64(1));
            let evals = gruen_cubic_evals(current_scalar, w_current, q_c, q_e, prev_claim);

            // Lagrange coefficients L_i(4) for nodes {0, 1, 2, 3} are {-1, 4, -6, 4}.
            let at_4 = -evals[0] + Fr::from_u64(4) * evals[1] - Fr::from_u64(6) * evals[2]
                + Fr::from_u64(4) * evals[3];
            assert_eq!(at_4, direct(Fr::from_u64(4)), "interpolated s(4) mismatch");
        }
    }
}
