//! Univariate skip optimization for the first sumcheck round.
//!
//! The outer Spartan sumcheck proves
//! $\sum_x \widetilde{eq}(x,\tau) \cdot (\widetilde{Az}(x) \cdot \widetilde{Bz}(x) - \widetilde{Cz}(x)) = 0$.
//!
//! The standard first round evaluates the round polynomial at 4 points by
//! iterating over all $2^{n-1}$ hypercube assignments for each point.
//!
//! The **univariate skip** exploits the factored structure of $\widetilde{eq}$:
//! $$g(x_1, x') = \widetilde{eq}_1(x_1, \tau_1) \cdot \widetilde{eq}_{\text{rest}}(x', \tau') \cdot [\widetilde{Az}(x_1, x') \cdot \widetilde{Bz}(x_1, x') - \widetilde{Cz}(x_1, x')]$$
//!
//! Define $t_1(Y) = \sum_{x'} \widetilde{eq}_{\text{rest}}(x', \tau') \cdot [\widetilde{Az}(Y, x') \cdot \widetilde{Bz}(Y, x') - \widetilde{Cz}(Y, x')]$.
//!
//! For a satisfying witness, $t_1(0) = t_1(1) = 0$, so $t_1(Y) = \alpha \cdot Y(Y-1)$
//! for some scalar $\alpha$. The prover computes $\alpha$ from a single evaluation
//! ($t_1(2)$) instead of four, reducing the first round's work by ~4x.
//!
//! The round polynomial $s_1(Y) = \widetilde{eq}_1(Y, \tau_1) \cdot t_1(Y)$ is then
//! constructed analytically from $\alpha$ and $\tau_1$.

use jolt_field::Field;
use jolt_poly::UnivariatePoly;

/// Strategy for the first round of the outer sumcheck.
///
/// Controls whether the prover uses the standard Boolean-hypercube
/// enumeration or the factored univariate skip optimization.
#[derive(Clone, Copy, Debug, Default)]
pub enum FirstRoundStrategy {
    /// Standard first-round evaluation over the Boolean hypercube.
    ///
    /// Iterates over all $2^{n-1}$ variable assignments, evaluating the
    /// round polynomial at $\{0, 1, 2, 3\}$. This is the baseline approach.
    #[default]
    Standard,

    /// Factored univariate skip.
    ///
    /// Exploits the identity $t_1(0) = t_1(1) = 0$ for satisfying witnesses
    /// to compute the first round polynomial from a single evaluation point
    /// instead of four. Produces the same proof as `Standard`.
    UnivariateSkip,
}

/// Builds the degree-3 first-round polynomial analytically from $t_1(2)$
/// and $\tau_1$.
///
/// Given $\alpha = t_1(2) / 2$ and the eq factor
/// $\widetilde{eq}_1(X, \tau_1) = (1-\tau_1) + (2\tau_1-1) X$, constructs:
///
/// $$s_1(X) = \widetilde{eq}_1(X, \tau_1) \cdot \alpha \cdot X(X-1)$$
///
/// This is a utility used by any zero-check sumcheck that wants univariate
/// skip. The caller computes $t_1(2)$ using its formula-specific logic.
pub fn uniskip_round_poly<F: Field>(t1_at_2: F, tau_1: F) -> UnivariatePoly<F> {
    let two = F::from_u64(2);
    let alpha = t1_at_2 * two.inverse().expect("2 is invertible in any prime field");

    let one_minus_tau = F::one() - tau_1;
    let two_tau_minus_one = two * tau_1 - F::one();
    let two_minus_3tau = two - F::from_u64(3) * tau_1;

    UnivariatePoly::new(vec![
        F::zero(),                 // X^0
        -(alpha * one_minus_tau),  // X^1
        alpha * two_minus_3tau,    // X^2
        alpha * two_tau_minus_one, // X^3
    ])
}

/// Computes the first outer sumcheck round polynomial using the factored
/// univariate skip.
///
/// Given the evaluation tables for eq, Az, Bz, Cz (each of length $2^n$),
/// and the first Fiat-Shamir challenge $\tau_1$:
///
/// 1. Computes $\alpha$ from $t_1(2) = \sum_{i'} \widetilde{eq}_{\text{rest}}(i') \cdot [(2az_h - az_l)(2bz_h - bz_l) - (2cz_h - cz_l)]$
/// 2. Constructs $s_1(X) = \widetilde{eq}_1(X, \tau_1) \cdot \alpha \cdot X(X-1)$ analytically
///
/// The result is identical to the standard 4-point evaluation approach.
pub fn uniskip_first_round<F: Field>(
    eq_evals: &[F],
    az_evals: &[F],
    bz_evals: &[F],
    cz_evals: &[F],
    tau_1: F,
) -> UnivariatePoly<F> {
    let half = eq_evals.len() / 2;
    let two = F::from_u64(2);

    // t_1(2) = sum_{i'} eq_rest[i'] * [(2*az_hi - az_lo)(2*bz_hi - bz_lo) - (2*cz_hi - cz_lo)]
    // where eq_rest[i'] = eq_lo[i'] + eq_hi[i'] (no division needed)
    let t1_at_2 = compute_t1_at_2(half, eq_evals, az_evals, bz_evals, cz_evals);

    // t_1(X) = alpha * X * (X - 1), so t_1(2) = alpha * 2 * 1 = 2*alpha
    let alpha = t1_at_2 * two.inverse().expect("2 is invertible in any prime field");

    // s_1(X) = eq_factor(X) * alpha * X * (X - 1)
    // eq_factor(X) = (1 - tau_1) + (2*tau_1 - 1)*X
    //
    // s_1(X) = alpha * [(2*tau_1 - 1)*X^3 + (2 - 3*tau_1)*X^2 - (1 - tau_1)*X]
    let one_minus_tau = F::one() - tau_1;
    let two_tau_minus_one = two * tau_1 - F::one();
    let two_minus_3tau = two - F::from_u64(3) * tau_1;

    UnivariatePoly::new(vec![
        F::zero(),                 // X^0
        -(alpha * one_minus_tau),  // X^1
        alpha * two_minus_3tau,    // X^2
        alpha * two_tau_minus_one, // X^3
    ])
}

fn compute_t1_at_2<F: Field>(
    half: usize,
    eq_evals: &[F],
    az_evals: &[F],
    bz_evals: &[F],
    cz_evals: &[F],
) -> F {
    let two = F::from_u64(2);

    #[cfg(feature = "parallel")]
    {
        if half >= super::prover::PAR_THRESHOLD {
            use rayon::prelude::*;
            return (0..half)
                .into_par_iter()
                .map(|i| {
                    let eq_rest = eq_evals[i] + eq_evals[i + half];
                    let az_at_2 = two * az_evals[i + half] - az_evals[i];
                    let bz_at_2 = two * bz_evals[i + half] - bz_evals[i];
                    let cz_at_2 = two * cz_evals[i + half] - cz_evals[i];
                    eq_rest * (az_at_2 * bz_at_2 - cz_at_2)
                })
                .sum();
        }
    }

    let mut sum = F::zero();
    for i in 0..half {
        let eq_rest = eq_evals[i] + eq_evals[i + half];
        let az_at_2 = two * az_evals[i + half] - az_evals[i];
        let bz_at_2 = two * bz_evals[i + half] - bz_evals[i];
        let cz_at_2 = two * cz_evals[i + half] - cz_evals[i];
        sum += eq_rest * (az_at_2 * bz_at_2 - cz_at_2);
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Field, Fr};
    use jolt_poly::{EqPolynomial, Polynomial};
    use num_traits::{One, Zero};

    /// Computes the standard first-round polynomial by evaluating at {0,1,2,3}.
    fn standard_first_round(
        eq: &Polynomial<Fr>,
        az: &Polynomial<Fr>,
        bz: &Polynomial<Fr>,
        cz: &Polynomial<Fr>,
    ) -> UnivariatePoly<Fr> {
        let half = eq.evaluations().len() / 2;
        let mut evals = [Fr::zero(); 4];

        let eq_e = eq.evaluations();
        let az_e = az.evaluations();
        let bz_e = bz.evaluations();
        let cz_e = cz.evaluations();

        for i in 0..half {
            let eq_lo = eq_e[i];
            let eq_hi = eq_e[i + half];
            let az_lo = az_e[i];
            let az_hi = az_e[i + half];
            let bz_lo = bz_e[i];
            let bz_hi = bz_e[i + half];
            let cz_lo = cz_e[i];
            let cz_hi = cz_e[i + half];

            let eq_d = eq_hi - eq_lo;
            let az_d = az_hi - az_lo;
            let bz_d = bz_hi - bz_lo;
            let cz_d = cz_hi - cz_lo;

            for (t, eval) in evals.iter_mut().enumerate() {
                let x = Fr::from_u64(t as u64);
                let eq_val = eq_lo + x * eq_d;
                let az_val = az_lo + x * az_d;
                let bz_val = bz_lo + x * bz_d;
                let cz_val = cz_lo + x * cz_d;
                *eval += eq_val * (az_val * bz_val - cz_val);
            }
        }

        let points: Vec<(Fr, Fr)> = (0..4).map(|t| (Fr::from_u64(t as u64), evals[t])).collect();
        UnivariatePoly::interpolate(&points)
    }

    /// Builds test polynomials for a 2-constraint system.
    fn test_polynomials(
        tau: &[Fr],
    ) -> (
        Polynomial<Fr>,
        Polynomial<Fr>,
        Polynomial<Fr>,
        Polynomial<Fr>,
    ) {
        // x*x = y (constraint 0), y*x = z (constraint 1)
        // witness: [1, 3, 9, 27] → az = [3, 9], bz = [3, 3], cz = [9, 27]
        let m_padded = 2usize;
        let az = Polynomial::new(vec![Fr::from_u64(3), Fr::from_u64(9)]);
        let bz = Polynomial::new(vec![Fr::from_u64(3), Fr::from_u64(3)]);
        let cz = Polynomial::new(vec![Fr::from_u64(9), Fr::from_u64(27)]);
        let eq = Polynomial::new(EqPolynomial::new(tau.to_vec()).evaluations());
        assert_eq!(eq.len(), m_padded);
        (eq, az, bz, cz)
    }

    #[test]
    fn uniskip_matches_standard() {
        let tau = vec![Fr::from_u64(7)]; // 1 sumcheck variable for 2 constraints
        let (eq, az, bz, cz) = test_polynomials(&tau);

        let standard = standard_first_round(&eq, &az, &bz, &cz);
        let factored = uniskip_first_round(
            eq.evaluations(),
            az.evaluations(),
            bz.evaluations(),
            cz.evaluations(),
            tau[0],
        );

        // Both should evaluate identically at several test points
        for x in 0..10u64 {
            let x_f = Fr::from_u64(x);
            assert_eq!(
                standard.evaluate(x_f),
                factored.evaluate(x_f),
                "mismatch at x={x}"
            );
        }
    }

    #[test]
    fn uniskip_sum_is_zero() {
        let tau = vec![Fr::from_u64(42)];
        let (eq, az, bz, cz) = test_polynomials(&tau);

        let poly = uniskip_first_round(
            eq.evaluations(),
            az.evaluations(),
            bz.evaluations(),
            cz.evaluations(),
            tau[0],
        );

        // s(0) + s(1) must equal 0 (satisfying witness)
        let sum = poly.evaluate(Fr::zero()) + poly.evaluate(Fr::one());
        assert_eq!(sum, Fr::zero());
    }

    #[test]
    fn uniskip_matches_4_constraint_system() {
        // x^2=y, xy=z, xz=w, xw=v with x=2
        let az_vals = vec![
            Fr::from_u64(2),
            Fr::from_u64(4),
            Fr::from_u64(8),
            Fr::from_u64(16),
        ];
        let bz_vals = vec![
            Fr::from_u64(2),
            Fr::from_u64(2),
            Fr::from_u64(2),
            Fr::from_u64(2),
        ];
        let cz_vals = vec![
            Fr::from_u64(4),
            Fr::from_u64(8),
            Fr::from_u64(16),
            Fr::from_u64(32),
        ];

        let tau = vec![Fr::from_u64(13), Fr::from_u64(37)];
        let eq = Polynomial::new(EqPolynomial::new(tau.clone()).evaluations());
        let az = Polynomial::new(az_vals);
        let bz = Polynomial::new(bz_vals);
        let cz = Polynomial::new(cz_vals);

        let standard = standard_first_round(&eq, &az, &bz, &cz);
        let factored = uniskip_first_round(
            eq.evaluations(),
            az.evaluations(),
            bz.evaluations(),
            cz.evaluations(),
            tau[0],
        );

        for x in 0..10u64 {
            let x_f = Fr::from_u64(x);
            assert_eq!(
                standard.evaluate(x_f),
                factored.evaluate(x_f),
                "mismatch at x={x}"
            );
        }
    }
}
