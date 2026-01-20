//! Output Claim Constraints for BlindFold R1CS
//!
//! This module defines the constraint representation for binding sumcheck final output claims
//! to polynomial evaluations. Each sumcheck instance describes how its final claim relates
//! to the polynomial openings via an `OutputClaimConstraint`.

use crate::poly::opening_proof::OpeningId;

/// Identifies a value source in the constraint expression.
/// Values can come from polynomial openings, challenges, or be constants.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueSource {
    /// Polynomial evaluation from opening accumulator (witness variable).
    /// The OpeningId identifies which polynomial evaluation this refers to.
    Opening(OpeningId),
    /// Challenge-derived value (public input).
    /// Index into the challenge/public values array provided at constraint evaluation time.
    Challenge(usize),
    /// Constant value represented as i128 for flexibility with negative values.
    Constant(i128),
}

impl ValueSource {
    pub fn opening(id: OpeningId) -> Self {
        Self::Opening(id)
    }

    pub fn challenge(idx: usize) -> Self {
        Self::Challenge(idx)
    }

    pub fn constant(val: i128) -> Self {
        Self::Constant(val)
    }

    pub fn one() -> Self {
        Self::Constant(1)
    }

    pub fn neg_one() -> Self {
        Self::Constant(-1)
    }
}

/// A term in a sum-of-products expression.
/// Represents: coeff * factor[0] * factor[1] * ... * factor[n-1]
#[derive(Clone, Debug)]
pub struct ProductTerm {
    /// Coefficient (can be a constant, challenge, or opening).
    pub coeff: ValueSource,
    /// Factors to multiply together (can be empty for just the coefficient).
    pub factors: Vec<ValueSource>,
}

impl ProductTerm {
    pub fn new(coeff: ValueSource, factors: Vec<ValueSource>) -> Self {
        Self { coeff, factors }
    }

    /// Simple product with coefficient 1.
    pub fn product(factors: Vec<ValueSource>) -> Self {
        Self {
            coeff: ValueSource::one(),
            factors,
        }
    }

    /// Scaled product with explicit coefficient.
    pub fn scaled(coeff: ValueSource, factors: Vec<ValueSource>) -> Self {
        Self { coeff, factors }
    }

    /// Single value (coefficient only, no factors).
    pub fn single(value: ValueSource) -> Self {
        Self {
            coeff: value,
            factors: vec![],
        }
    }
}

/// Describes the final output claim constraint for a sumcheck instance.
///
/// General form: output = Σᵢ coeffᵢ * ∏ⱼ factorᵢⱼ
///
/// This single representation handles all constraint patterns:
/// - Linear: coeff * factor (single factor per term)
/// - Product: 1 * (a * b * c) (single term, multiple factors)
/// - Sum-of-products: α₁*(a*b) + α₂*(c*d) (multiple terms)
/// - Differences: α*(a) + (-α)*(b) (negative coefficients)
/// - Squares: 1*(a*a) (repeated factors)
#[derive(Clone, Debug, Default)]
pub struct OutputClaimConstraint {
    /// List of product terms to sum.
    /// output = Σᵢ terms[i].coeff * ∏ⱼ terms[i].factors[j]
    pub terms: Vec<ProductTerm>,

    /// Which openings are needed (for witness allocation).
    pub required_openings: Vec<OpeningId>,

    /// Number of challenge-derived values needed.
    pub num_challenges: usize,
}

impl OutputClaimConstraint {
    pub fn new(terms: Vec<ProductTerm>, required_openings: Vec<OpeningId>) -> Self {
        let num_challenges = terms
            .iter()
            .flat_map(|t| {
                std::iter::once(&t.coeff).chain(t.factors.iter())
            })
            .filter_map(|v| match v {
                ValueSource::Challenge(idx) => Some(*idx + 1),
                _ => None,
            })
            .max()
            .unwrap_or(0);

        Self {
            terms,
            required_openings,
            num_challenges,
        }
    }

    fn collect_unique_openings(terms: &[ProductTerm]) -> Vec<OpeningId> {
        let mut openings = Vec::new();
        for term in terms {
            for vs in std::iter::once(&term.coeff).chain(term.factors.iter()) {
                if let ValueSource::Opening(id) = vs {
                    if !openings.contains(id) {
                        openings.push(*id);
                    }
                }
            }
        }
        openings
    }

    /// Simple product: output = a * b * c * ...
    pub fn product(factors: Vec<ValueSource>) -> Self {
        let terms = vec![ProductTerm::product(factors)];
        let required_openings = Self::collect_unique_openings(&terms);
        Self::new(terms, required_openings)
    }

    /// Linear combination: output = Σᵢ αᵢ * evalᵢ
    /// Takes pairs of (coefficient, value).
    pub fn linear(terms: Vec<(ValueSource, ValueSource)>) -> Self {
        let product_terms: Vec<ProductTerm> = terms
            .into_iter()
            .map(|(coeff, val)| ProductTerm::scaled(coeff, vec![val]))
            .collect();

        let required_openings = Self::collect_unique_openings(&product_terms);
        Self::new(product_terms, required_openings)
    }

    /// Scaled linear: output = multiplier * Σᵢ αᵢ * evalᵢ
    /// Expands to: Σᵢ (multiplier * αᵢ) * evalᵢ if multiplier is constant.
    /// For non-constant multiplier, creates product terms.
    pub fn scaled_linear(
        multiplier: ValueSource,
        terms: Vec<(ValueSource, ValueSource)>,
    ) -> Self {
        let product_terms: Vec<ProductTerm> = terms
            .into_iter()
            .map(|(coeff, val)| {
                ProductTerm::scaled(coeff, vec![multiplier.clone(), val])
            })
            .collect();

        let required_openings = Self::collect_unique_openings(&product_terms);
        Self::new(product_terms, required_openings)
    }

    /// Sum of products: output = Σᵢ coeffᵢ * ∏ⱼ factorsᵢⱼ
    pub fn sum_of_products(product_terms: Vec<ProductTerm>) -> Self {
        let required_openings = Self::collect_unique_openings(&product_terms);
        Self::new(product_terms, required_openings)
    }

    /// Direct evaluation: output = eval (single opening, coefficient 1)
    pub fn direct(opening: OpeningId) -> Self {
        Self::new(
            vec![ProductTerm::single(ValueSource::Opening(opening))],
            vec![opening],
        )
    }

    /// Combine multiple constraints with batching coefficients.
    ///
    /// Given constraints C_j and coefficients α_j, produces a combined constraint:
    /// `output = Σⱼ αⱼ * C_j`
    ///
    /// Challenge index layout:
    /// - Challenge(0..num_instances) = batching coefficients α₀, α₁, ...
    /// - Challenge(num_instances..) = individual constraint challenges, offset per constraint
    ///
    /// Returns None if any constraint is None.
    pub fn batch(
        constraints: &[Option<OutputClaimConstraint>],
        _num_batching_coefficients: usize,
    ) -> Option<Self> {
        // Check all constraints are present
        if constraints.iter().any(|c| c.is_none()) {
            return None;
        }

        let constraints: Vec<&OutputClaimConstraint> =
            constraints.iter().map(|c| c.as_ref().unwrap()).collect();

        let num_instances = constraints.len();

        // Combine all terms, scaling each constraint's terms by its batching coefficient
        let mut combined_terms = Vec::new();
        let mut combined_openings = Vec::new();

        // Track challenge offset for each constraint
        let mut challenge_offset = num_instances;

        for (j, constraint) in constraints.iter().enumerate() {
            // Batching coefficient αⱼ is Challenge(j)
            let alpha_j = ValueSource::Challenge(j);

            for term in &constraint.terms {
                // Offset the challenge indices in this term
                let offset_coeff = Self::offset_challenge(&term.coeff, challenge_offset);
                let offset_factors: Vec<_> = term
                    .factors
                    .iter()
                    .map(|f| Self::offset_challenge(f, challenge_offset))
                    .collect();

                // New term: αⱼ * (offset_coeff * offset_factors)
                let mut new_factors = vec![alpha_j.clone()];
                new_factors.extend(offset_factors);

                combined_terms.push(ProductTerm::new(offset_coeff, new_factors));
            }

            // Collect all required openings
            for opening in &constraint.required_openings {
                if !combined_openings.contains(opening) {
                    combined_openings.push(*opening);
                }
            }

            // Move offset past this constraint's challenges
            challenge_offset += constraint.num_challenges;
        }

        Some(Self::new(combined_terms, combined_openings))
    }

    fn offset_challenge(value: &ValueSource, offset: usize) -> ValueSource {
        match value {
            ValueSource::Challenge(idx) => ValueSource::Challenge(idx + offset),
            other => other.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::opening_proof::SumcheckId;
    use crate::zkvm::witness::CommittedPolynomial;

    fn test_opening(idx: usize) -> OpeningId {
        OpeningId::Committed(CommittedPolynomial::RamRa(idx), SumcheckId::RamReadWriteChecking)
    }

    #[test]
    fn test_product_constraint() {
        let a = ValueSource::Opening(test_opening(0));
        let b = ValueSource::Opening(test_opening(1));

        let constraint = OutputClaimConstraint::product(vec![a.clone(), b.clone()]);

        assert_eq!(constraint.terms.len(), 1);
        assert_eq!(constraint.terms[0].factors.len(), 2);
        assert_eq!(constraint.required_openings.len(), 2);
    }

    #[test]
    fn test_linear_constraint() {
        let alpha = ValueSource::Challenge(0);
        let beta = ValueSource::Challenge(1);
        let y0 = ValueSource::Opening(test_opening(0));
        let y1 = ValueSource::Opening(test_opening(1));

        let constraint = OutputClaimConstraint::linear(vec![
            (alpha, y0),
            (beta, y1),
        ]);

        assert_eq!(constraint.terms.len(), 2);
        assert_eq!(constraint.required_openings.len(), 2);
        assert_eq!(constraint.num_challenges, 2);
    }

    #[test]
    fn test_sum_of_products() {
        let eq = ValueSource::Opening(test_opening(0));
        let ra = ValueSource::Opening(test_opening(1));
        let val = ValueSource::Opening(test_opening(2));
        let inc = ValueSource::Opening(test_opening(3));
        let gamma = ValueSource::Challenge(0);

        let constraint = OutputClaimConstraint::sum_of_products(vec![
            ProductTerm::product(vec![eq.clone(), ra.clone(), val.clone()]),
            ProductTerm::scaled(gamma.clone(), vec![eq.clone(), ra.clone(), val.clone()]),
            ProductTerm::scaled(gamma, vec![eq, ra, inc]),
        ]);

        assert_eq!(constraint.terms.len(), 3);
        assert_eq!(constraint.required_openings.len(), 4); // 4 unique openings
        assert_eq!(constraint.num_challenges, 1);
    }
}
