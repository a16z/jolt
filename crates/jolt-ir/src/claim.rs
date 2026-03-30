use jolt_field::Field;

use crate::composition::CompositionFormula;
use crate::expr::Expr;
use crate::polynomial_id::PolynomialId;

/// Maps an opening variable index to a concrete polynomial identity.
///
/// `var_id` matches `Var::Opening(id)` in the expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpeningBinding {
    pub var_id: u32,
    pub polynomial: PolynomialId,
}

/// A complete claim definition: expression + binding metadata.
///
/// This is the single source of truth for a sumcheck claim formula. All
/// backends (evaluation, R1CS, Lean4, circuit, compute) consume this
/// structure — either via `Expr` (visitor-based backends) or via
/// `CompositionFormula` (compute backend).
///
/// # Example
///
/// ```
/// use jolt_ir::{ExprBuilder, ClaimDefinition, OpeningBinding, PolynomialId};
///
/// let b = ExprBuilder::new();
/// let h = b.opening(0);
/// let gamma = b.challenge(0);
/// let expr = b.build(gamma * (h * h - h));
///
/// let claim = ClaimDefinition {
///     expr,
///     opening_bindings: vec![
///         OpeningBinding { var_id: 0, polynomial: PolynomialId::HammingWeight },
///     ],
///     num_challenges: 1,
/// };
///
/// assert_eq!(claim.opening_bindings.len(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct ClaimDefinition {
    pub expr: Expr,
    pub opening_bindings: Vec<OpeningBinding>,
    pub num_challenges: u32,
}

impl ClaimDefinition {
    /// Evaluate the claim expression with concrete opening and challenge values.
    pub fn evaluate<F: Field>(&self, openings: &[F], challenges: &[F]) -> F {
        self.expr.evaluate(openings, challenges)
    }

    /// All polynomials referenced by the formula's opening bindings.
    pub fn polynomials(&self) -> Vec<PolynomialId> {
        self.opening_bindings.iter().map(|b| b.polynomial).collect()
    }

    /// Normalize the claim's expression into a [`CompositionFormula`].
    ///
    /// The returned formula uses `Factor::Input(i)` for `Var::Opening(i)` and
    /// `Factor::Challenge(i)` for `Var::Challenge(i)`. This is the raw claim
    /// formula — it does NOT include the eq weight polynomial. The caller
    /// (orchestrator or backend) is responsible for incorporating eq.
    pub fn to_composition_formula(&self) -> CompositionFormula {
        self.expr.to_composition_formula()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::ExprBuilder;
    use jolt_field::{Field, Fr};

    #[test]
    fn claim_definition_construction() {
        let b = ExprBuilder::new();
        let h = b.opening(0);
        let gamma = b.challenge(0);
        let expr = b.build(gamma * (h * h - h));

        let claim = ClaimDefinition {
            expr,
            opening_bindings: vec![OpeningBinding {
                var_id: 0,
                polynomial: PolynomialId::HammingWeight,
            }],
            num_challenges: 1,
        };

        assert_eq!(claim.opening_bindings.len(), 1);
        assert_eq!(claim.num_challenges, 1);
    }

    #[test]
    fn to_composition_formula_hamming() {
        let claim = crate::zkvm::claims::ram::hamming_booleanity();
        let f = claim.to_composition_formula();
        assert!(f.is_hamming_booleanity());
        assert_eq!(f.degree(), 2);
    }

    #[test]
    fn to_composition_formula_registers_reduction() {
        let claim = crate::zkvm::claims::reductions::registers_claim_reduction();
        let f = claim.to_composition_formula();
        assert!(f.is_linear_combination());
    }

    #[test]
    fn to_composition_formula_increment_reduction() {
        let claim = crate::zkvm::claims::reductions::increment_claim_reduction();
        let f = claim.to_composition_formula();
        assert!(f.is_linear_combination());
    }

    #[test]
    fn to_composition_formula_ram_rw() {
        let claim = crate::zkvm::claims::ram::ram_read_write_checking();
        let f = claim.to_composition_formula();
        // Not linear, not product-sum, not hamming — general formula
        assert!(!f.is_linear_combination());
        assert!(f.as_product_sum().is_none());
        assert!(!f.is_hamming_booleanity());
        assert_eq!(f.degree(), 2); // ra * val terms
    }

    #[test]
    fn to_composition_formula_ram_ra_virtual() {
        let claim = crate::zkvm::claims::ram::ram_ra_virtual(4);
        let f = claim.to_composition_formula();
        assert_eq!(f.degree(), 4); // product of 4 inputs
    }

    #[test]
    fn to_composition_formula_raf_evaluation() {
        let claim = crate::zkvm::claims::ram::ram_raf_evaluation();
        let f = claim.to_composition_formula();
        assert!(f.is_linear_combination());
    }

    #[test]
    fn claim_with_batching_coefficients() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let alpha = b.challenge(0);
        let expr = b.build(a + alpha * bv);

        let claim = ClaimDefinition {
            expr,
            opening_bindings: vec![
                OpeningBinding {
                    var_id: 0,
                    polynomial: PolynomialId::RamInc,
                },
                OpeningBinding {
                    var_id: 1,
                    polynomial: PolynomialId::RdInc,
                },
            ],
            num_challenges: 1,
        };

        assert_eq!(claim.opening_bindings.len(), 2);
    }

    #[test]
    fn composition_formula_evaluation_matches_expr() {
        let claim = crate::zkvm::claims::ram::hamming_booleanity();
        let h = Fr::from_u64(5);
        let eq_eval = Fr::from_u64(7);
        let neg_eq = -eq_eval;
        let challenges = vec![eq_eval, neg_eq];

        let direct = claim.evaluate::<Fr>(&[h], &challenges);
        let f = claim.to_composition_formula();
        let via_formula = f.evaluate(&[h], &challenges);
        assert_eq!(direct, via_formula);
    }

    #[test]
    fn linear_combination_weights_match_evaluation() {
        let claim = crate::zkvm::claims::reductions::registers_claim_reduction();
        let eq_eval = Fr::from_u64(7);
        let gamma = Fr::from_u64(11);
        let gamma_sq = gamma * gamma;
        let challenges = vec![eq_eval, gamma, gamma_sq];

        let f = claim.to_composition_formula();
        let weights = f.linear_combination_weights(&challenges);

        let openings = vec![Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)];
        let direct = claim.evaluate::<Fr>(&openings, &challenges);
        let via_weights: Fr = weights
            .iter()
            .zip(openings.iter())
            .map(|(w, o)| *w * *o)
            .sum();
        assert_eq!(direct, via_weights);
    }

    #[test]
    fn hamming_eq_scale_correctness() {
        let claim = crate::zkvm::claims::ram::hamming_booleanity();
        let eq_eval = Fr::from_u64(7);
        let neg_eq = -eq_eval;
        let challenges = vec![eq_eval, neg_eq];

        let f = claim.to_composition_formula();
        let scale = f.hamming_eq_scale(&challenges);

        // scale * (H² - H) must match direct evaluation
        let h = Fr::from_u64(5);
        let direct = claim.evaluate::<Fr>(&[h], &challenges);
        let via_scale = scale * (h * h - h);
        assert_eq!(direct, via_scale);
    }
}
