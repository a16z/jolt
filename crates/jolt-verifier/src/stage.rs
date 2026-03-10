//! Stage descriptors for config-driven verification.
//!
//! Each sumcheck stage (S2–S7) is described by a [`StageDescriptor`] — a plain
//! data struct that encodes the claim formula, eq evaluation point, and
//! commitment mapping. The [`verify`](crate::verifier::verify) loop processes
//! all stages generically using expression evaluation from `jolt-ir`.
//!
//! No per-stage hand-written verification code is needed. The single
//! verification loop handles all stage types: claim reductions (linear
//! combinations), booleanity checks, product-based checking, etc.

use jolt_field::Field;
use jolt_ir::{ChallengeBinding, ChallengeSource, ClaimDefinition, Expr, ExprBuilder, OpeningBinding};

/// Describes one sumcheck stage for verification.
///
/// The verifier checks each stage as follows:
///
/// 1. Build a [`SumcheckClaim`](jolt_sumcheck::SumcheckClaim) from
///    `num_vars`, `degree`, and `claimed_sum`
/// 2. Run the batched sumcheck verifier → `(final_eval, challenges)`
/// 3. Compute `eval_point` from challenges (applying [`binding_order`](Self::reverse_challenges))
/// 4. Check: `eq(eq_point, eval_point) × output_expr(evaluations, output_challenges) == final_eval`
/// 5. Produce [`VerifierClaim`](jolt_openings::VerifierClaim)s at `eval_point` for batch opening
///
/// # Expression convention
///
/// `output_expr` represents the composition `g(x)` in the sumcheck
/// `Σ_x eq(r, x) · g(x)`. At the evaluation point:
///
/// - `Opening(i)` resolves to `evaluations[i]` from the proof
/// - `Challenge(i)` resolves to `output_challenges[i]`
/// - `Constant(v)` is promoted to the field via `F::from_i128(v)`
pub struct StageDescriptor<F: Field> {
    /// Number of sumcheck variables.
    pub num_vars: usize,
    /// Sumcheck polynomial degree (2 for eq × linear, 3 for eq × quadratic, etc.).
    pub degree: usize,
    /// Expected sumcheck claimed value.
    pub claimed_sum: F,
    /// Eq polynomial evaluation point (typically from Spartan's challenge vectors).
    pub eq_point: Vec<F>,
    /// Whether to reverse sumcheck challenges to obtain the evaluation point.
    ///
    /// - `true` (LowToHigh binding): round 0 binds the last variable, so
    ///   `eval_point = challenges.reverse()`. Used by the production pipeline.
    /// - `false` (standard MSB-first): `eval_point = challenges`. Used in tests
    ///   with the standard `SumcheckCompute` bind order.
    pub reverse_challenges: bool,
    /// Expression for the composition g(evaluations, challenges).
    ///
    /// The verifier checks: `eq(eq_point, eval_point) × g_eval == final_eval`.
    pub output_expr: Expr,
    /// Challenge values plugged into the output expression.
    pub output_challenges: Vec<F>,
    /// Maps each evaluation slot to a commitment index in `proof.commitments`.
    ///
    /// `commitment_indices[i]` is the index of the commitment for `evaluations[i]`.
    pub commitment_indices: Vec<usize>,
}

impl<F: Field> StageDescriptor<F> {
    /// Descriptor for an eq-weighted linear combination.
    ///
    /// Verifies: `Σ_x eq(r, x) · Σᵢ cᵢ · pᵢ(x) = claimed_sum`
    ///
    /// This covers claim reduction stages (S3) and Hamming reduction (S7).
    ///
    /// # Panics
    ///
    /// Panics if `coefficients` and `commitment_indices` have different lengths.
    pub fn claim_reduction(
        eq_point: Vec<F>,
        coefficients: Vec<F>,
        claimed_sum: F,
        commitment_indices: Vec<usize>,
    ) -> Self {
        assert_eq!(
            coefficients.len(),
            commitment_indices.len(),
            "coefficients and commitment_indices length mismatch"
        );

        let n = coefficients.len();
        let num_vars = eq_point.len();

        // Σ challenge(i) * opening(i)
        let b = ExprBuilder::new();
        let expr_handle = (0..n)
            .map(|i| b.challenge(i as u32) * b.opening(i as u32))
            .reduce(|acc, term| acc + term)
            .unwrap_or_else(|| b.zero());

        Self {
            num_vars,
            degree: 2,
            claimed_sum,
            eq_point,
            reverse_challenges: false,
            output_expr: b.build(expr_handle),
            output_challenges: coefficients,
            commitment_indices,
        }
    }

    /// Descriptor for booleanity check: `Σ_x eq(r, x) · h(x) · (h(x) − 1)`.
    ///
    /// `claimed_sum` is typically zero (all h values are boolean).
    pub fn booleanity(
        eq_point: Vec<F>,
        claimed_sum: F,
        commitment_index: usize,
    ) -> Self {
        let num_vars = eq_point.len();

        // opening(0) * (opening(0) - 1)
        let b = ExprBuilder::new();
        let h = b.opening(0);
        let expr_handle = h * (h - 1);

        Self {
            num_vars,
            degree: 3,
            claimed_sum,
            eq_point,
            reverse_challenges: false,
            output_expr: b.build(expr_handle),
            output_challenges: vec![],
            commitment_indices: vec![commitment_index],
        }
    }

    /// Descriptor using an arbitrary `ClaimDefinition` from `jolt-ir`.
    ///
    /// The expression and challenge values are taken from the claim definition.
    /// This is the most general form — any sumcheck composition can be described.
    pub fn from_claim_definition(
        num_vars: usize,
        degree: usize,
        claimed_sum: F,
        eq_point: Vec<F>,
        claim: &ClaimDefinition,
        challenge_values: Vec<F>,
        commitment_indices: Vec<usize>,
    ) -> Self {
        Self {
            num_vars,
            degree,
            claimed_sum,
            eq_point,
            reverse_challenges: false,
            output_expr: claim.expr.clone(),
            output_challenges: challenge_values,
            commitment_indices,
        }
    }

    /// Builder-style method to enable LowToHigh challenge reversal.
    pub fn with_reverse_challenges(mut self) -> Self {
        self.reverse_challenges = true;
        self
    }
}

/// Build a [`ClaimDefinition`] for a linear combination: `Σ cᵢ · pᵢ`.
///
/// Useful when downstream code needs the full claim definition (with binding
/// metadata) rather than just a `StageDescriptor`.
pub fn linear_combination_claim(n: usize) -> ClaimDefinition {
    let b = ExprBuilder::new();
    let expr_handle = (0..n)
        .map(|i| b.challenge(i as u32) * b.opening(i as u32))
        .reduce(|acc, term| acc + term)
        .unwrap_or_else(|| b.zero());

    ClaimDefinition {
        expr: b.build(expr_handle),
        opening_bindings: (0..n)
            .map(|i| OpeningBinding {
                var_id: i as u32,
                polynomial_tag: 0,
                sumcheck_tag: 0,
            })
            .collect(),
        challenge_bindings: (0..n)
            .map(|i| ChallengeBinding {
                var_id: i as u32,
                source: ChallengeSource::Derived,
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;

    #[test]
    fn claim_reduction_descriptor() {
        let eq_point = vec![Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)];
        let coeffs = vec![Fr::from_u64(5), Fr::from_u64(7)];
        let desc = StageDescriptor::claim_reduction(
            eq_point.clone(),
            coeffs.clone(),
            Fr::from_u64(42),
            vec![0, 1],
        );

        assert_eq!(desc.num_vars, 3);
        assert_eq!(desc.degree, 2);
        assert!(!desc.reverse_challenges);

        // Evaluate: 5*10 + 7*20 = 50 + 140 = 190
        let openings = [Fr::from_u64(10), Fr::from_u64(20)];
        let result: Fr = desc.output_expr.evaluate(&openings, &desc.output_challenges);
        assert_eq!(result, Fr::from_u64(190));
    }

    #[test]
    fn booleanity_descriptor() {
        let eq_point = vec![Fr::from_u64(1), Fr::from_u64(2)];
        let desc = StageDescriptor::<Fr>::booleanity(eq_point, Fr::from_u64(0), 0);

        assert_eq!(desc.degree, 3);

        // h=0: 0*(0-1) = 0
        let result: Fr = desc.output_expr.evaluate(&[Fr::from_u64(0)], &[]);
        assert_eq!(result, Fr::from_u64(0));

        // h=1: 1*(1-1) = 0
        let result: Fr = desc.output_expr.evaluate(&[Fr::from_u64(1)], &[]);
        assert_eq!(result, Fr::from_u64(0));

        // h=3: 3*(3-1) = 6
        let result: Fr = desc.output_expr.evaluate(&[Fr::from_u64(3)], &[]);
        assert_eq!(result, Fr::from_u64(6));
    }

    #[test]
    fn with_reverse_challenges_builder() {
        let desc = StageDescriptor::<Fr>::claim_reduction(
            vec![Fr::from_u64(1)],
            vec![Fr::from_u64(1)],
            Fr::from_u64(0),
            vec![0],
        )
        .with_reverse_challenges();
        assert!(desc.reverse_challenges);
    }

    #[test]
    fn linear_combination_claim_definition() {
        let claim = linear_combination_claim(3);
        assert_eq!(claim.opening_bindings.len(), 3);
        assert_eq!(claim.challenge_bindings.len(), 3);

        let openings = [Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)];
        let challenges = [Fr::from_u64(1), Fr::from_u64(10), Fr::from_u64(100)];
        // 1*2 + 10*3 + 100*5 = 2 + 30 + 500 = 532
        let result: Fr = claim.evaluate(&openings, &challenges);
        assert_eq!(result, Fr::from_u64(532));
    }
}
