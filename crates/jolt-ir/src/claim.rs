use jolt_field::Field;

use crate::expr::Expr;

/// Where a challenge value originates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChallengeSource {
    /// A batching coefficient (α_i for combining multiple instances).
    BatchingCoefficient(usize),
    /// A sumcheck challenge round value.
    SumcheckChallenge(usize),
    /// Derived from other values (e.g., eq polynomial evaluation, gamma power).
    Derived,
}

/// Maps a challenge variable index to its semantic origin.
///
/// `var_id` matches `Var::Challenge(id)` in the expression. Downstream code
/// uses `source` to resolve the variable to a concrete field value at
/// evaluation time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChallengeBinding {
    pub var_id: u32,
    pub source: ChallengeSource,
}

/// Maps an opening variable index to a concrete polynomial identity.
///
/// `var_id` matches `Var::Opening(id)` in the expression. Tags are opaque
/// `u64` so `jolt-ir` never depends on `jolt-zkvm` types — the downstream
/// consumer interprets them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpeningBinding {
    pub var_id: u32,
    pub polynomial_tag: u64,
    pub sumcheck_tag: u64,
}

/// A complete claim definition: expression + binding metadata.
///
/// This is the single source of truth for a sumcheck claim formula. All
/// backends (evaluation, R1CS, Lean4, circuit) consume this structure.
///
/// # Example
///
/// ```
/// use jolt_ir::{ExprBuilder, ClaimDefinition, OpeningBinding, ChallengeBinding, ChallengeSource};
///
/// let b = ExprBuilder::new();
/// let h = b.opening(0);
/// let gamma = b.challenge(0);
/// let expr = b.build(gamma * (h * h - h));
///
/// let claim = ClaimDefinition {
///     expr,
///     opening_bindings: vec![
///         OpeningBinding { var_id: 0, polynomial_tag: 1, sumcheck_tag: 2 },
///     ],
///     challenge_bindings: vec![
///         ChallengeBinding { var_id: 0, source: ChallengeSource::Derived },
///     ],
/// };
///
/// assert_eq!(claim.opening_bindings.len(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct ClaimDefinition {
    pub expr: Expr,
    pub opening_bindings: Vec<OpeningBinding>,
    pub challenge_bindings: Vec<ChallengeBinding>,
}

impl ClaimDefinition {
    /// Evaluate the claim expression with concrete opening and challenge values.
    ///
    /// Convenience wrapper around `Expr::evaluate`. The caller is responsible
    /// for ordering `openings` and `challenges` to match the `var_id` indices
    /// in the binding metadata.
    pub fn evaluate<F: Field>(&self, openings: &[F], challenges: &[F]) -> F {
        self.expr.evaluate(openings, challenges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::ExprBuilder;

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
                polynomial_tag: 100,
                sumcheck_tag: 200,
            }],
            challenge_bindings: vec![ChallengeBinding {
                var_id: 0,
                source: ChallengeSource::Derived,
            }],
        };

        assert_eq!(claim.opening_bindings.len(), 1);
        assert_eq!(claim.opening_bindings[0].polynomial_tag, 100);
        assert_eq!(claim.challenge_bindings.len(), 1);
        assert_eq!(claim.challenge_bindings[0].source, ChallengeSource::Derived);
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
                    polynomial_tag: 1,
                    sumcheck_tag: 10,
                },
                OpeningBinding {
                    var_id: 1,
                    polynomial_tag: 2,
                    sumcheck_tag: 10,
                },
            ],
            challenge_bindings: vec![ChallengeBinding {
                var_id: 0,
                source: ChallengeSource::BatchingCoefficient(0),
            }],
        };

        assert_eq!(claim.opening_bindings.len(), 2);
        assert_eq!(
            claim.challenge_bindings[0].source,
            ChallengeSource::BatchingCoefficient(0)
        );
    }

    #[test]
    fn claim_with_no_challenges() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let expr = b.build(a * a);

        let claim = ClaimDefinition {
            expr,
            opening_bindings: vec![OpeningBinding {
                var_id: 0,
                polynomial_tag: 1,
                sumcheck_tag: 1,
            }],
            challenge_bindings: vec![],
        };

        assert!(claim.challenge_bindings.is_empty());
    }
}
