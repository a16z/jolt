//! Test-only reference statistics: the normal CDF (from `libm::erfc`) and the
//! p-values of the chi-square and Kolmogorov–Smirnov tests.

#![expect(
    clippy::indexing_slicing,
    reason = "test statistics over paired slices"
)]

use std::f64::consts::SQRT_2;

pub(crate) fn normal_cdf(x: f64) -> f64 {
    0.5 * libm::erfc(-x / SQRT_2)
}

/// Upper tail of Pearson's statistic over cells with expected count ≥ 5;
/// sparser cells are pooled, in order, until they reach 5.
pub(crate) fn chi_square_p(observed: &[u64], expected: &[f64]) -> f64 {
    assert_eq!(observed.len(), expected.len());
    let mut statistic = 0.0;
    let mut cells = 0usize;
    let (mut pooled_o, mut pooled_e) = (0.0, 0.0);
    for i in 0..observed.len() {
        pooled_o += observed[i] as f64;
        pooled_e += expected[i];
        if pooled_e >= 5.0 {
            statistic += (pooled_o - pooled_e).powi(2) / pooled_e;
            cells += 1;
            (pooled_o, pooled_e) = (0.0, 0.0);
        }
    }
    if pooled_e > 0.0 {
        statistic += (pooled_o - pooled_e).powi(2) / pooled_e;
        cells += 1;
    }
    upper_regularized_gamma((cells - 1) as f64 / 2.0, statistic / 2.0)
}

/// Asymptotic Kolmogorov distribution with Stephens' small-sample correction.
pub(crate) fn ks_p(d: f64, n: usize) -> f64 {
    let root = (n as f64).sqrt();
    let lambda = (root + 0.12 + 0.11 / root) * d;
    let tail: f64 = (1..=100)
        .map(|j| {
            let sign = if j % 2 == 1 { 1.0 } else { -1.0 };
            sign * (-2.0 * f64::from(j * j) * lambda * lambda).exp()
        })
        .sum();
    (2.0 * tail).clamp(0.0, 1.0)
}

/// Two-sided p-value of a standard normal score.
pub(crate) fn normal_two_sided_p(z: f64) -> f64 {
    libm::erfc(z.abs() / SQRT_2)
}

/// `Q(a, x) = Γ(a, x)/Γ(a)`, by the series for `x < a + 1` and Lentz's
/// continued fraction otherwise (Numerical Recipes §6.2).
fn upper_regularized_gamma(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let log_prefactor = a * x.ln() - x - libm::lgamma(a);
    if x < a + 1.0 {
        let (mut term, mut sum, mut ap) = (1.0 / a, 1.0 / a, a);
        for _ in 0..10_000 {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if term.abs() < sum.abs() * 1e-16 {
                break;
            }
        }
        1.0 - sum * log_prefactor.exp()
    } else {
        let tiny = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..10_000 {
            let an = -f64::from(i) * (f64::from(i) - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 1e-16 {
                break;
            }
        }
        h * log_prefactor.exp()
    }
}

#[test]
fn reference_statistics_match_tables() {
    // χ²(10) upper 5% point 18.307; Φ(1.959964) = 0.975; Kolmogorov
    // λ = 1.358 has upper tail 0.05.
    assert!((upper_regularized_gamma(5.0, 18.307 / 2.0) - 0.05).abs() < 1e-4);
    assert!((normal_cdf(1.959_964) - 0.975).abs() < 1e-6);
    assert!((ks_p(1.358 / 1000.0, 1_000_000) - 0.05).abs() < 1e-3);
}
