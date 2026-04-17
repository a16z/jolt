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
