//! [`SumcheckCompute`] implementation for the RA virtual sumcheck.
//!
//! Wraps [`RaPolynomial`] and [`SplitEqEvaluator`] to compute
//! `g(X) = Σ_j eq(·) · Σ_i γ^i · Π_k ra_{i·m+k}(X, j)` per round.

use jolt_field::Field;
use jolt_poly::{BindingOrder, UnivariatePoly};
use jolt_sumcheck::{SplitEqEvaluator, SumcheckCompute};

use super::mles_product_sum::{compute_mles_product_sum, compute_mles_weighted_sop};
use super::ra_poly::RaPolynomial;

/// Sumcheck witness for the RA virtual sumcheck.
///
/// When `n_products == 1`, delegates to [`compute_mles_product_sum`] (single product).
/// When `n_products > 1`, uses [`compute_mles_weighted_sop`] for the gamma-weighted
/// sum of products `Σ_i γ^i · Π_k ra_{i·m+k}`.
pub struct RaVirtualCompute<F: Field> {
    /// RA chunk polynomials (one per committed chunk across all virtual polys).
    pub mles: Vec<RaPolynomial<u8, F>>,
    /// Split-eq evaluator factoring `eq(w, x)`.
    pub eq_poly: SplitEqEvaluator<F>,
    /// Current claimed sum `g(0) + g(1)` — updated after each bind.
    pub claim: F,
    /// Binding order for RA polynomial variables.
    pub binding_order: BindingOrder,
    /// Gamma power coefficients for combining virtual polynomials.
    pub gamma_powers: Vec<F>,
    /// Number of virtual RA polynomials (product groups).
    pub n_products: usize,
}

impl<F: Field> SumcheckCompute<F> for RaVirtualCompute<F> {
    fn set_claim(&mut self, claim: F) {
        self.claim = claim;
    }

    fn round_polynomial(&self) -> UnivariatePoly<F> {
        if self.n_products == 1 {
            compute_mles_product_sum(&self.mles, self.claim, &self.eq_poly)
        } else {
            compute_mles_weighted_sop(
                &self.mles,
                &self.gamma_powers,
                self.n_products,
                self.claim,
                &self.eq_poly,
            )
        }
    }

    fn bind(&mut self, challenge: F) {
        self.eq_poly.bind(challenge);
        for mle in &mut self.mles {
            mle.bind(challenge, self.binding_order);
        }
    }
}
