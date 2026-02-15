use crate::poly::opening_proof::OpeningId;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueSource {
    Opening(OpeningId),
    Challenge(usize),
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
    pub coeff: ValueSource,
    pub factors: Vec<ValueSource>,
}

impl ProductTerm {
    pub fn new(coeff: ValueSource, factors: Vec<ValueSource>) -> Self {
        Self { coeff, factors }
    }

    pub fn product(factors: Vec<ValueSource>) -> Self {
        Self {
            coeff: ValueSource::one(),
            factors,
        }
    }

    pub fn scaled(coeff: ValueSource, factors: Vec<ValueSource>) -> Self {
        Self { coeff, factors }
    }

    pub fn single(value: ValueSource) -> Self {
        Self {
            coeff: value,
            factors: vec![],
        }
    }
}

/// General form: output = Σᵢ coeffᵢ * ∏ⱼ factorᵢⱼ
///
/// Handles all constraint patterns:
/// - Linear: coeff * factor (single factor per term)
/// - Product: 1 * (a * b * c) (single term, multiple factors)
/// - Sum-of-products: α₁*(a*b) + α₂*(c*d) (multiple terms)
/// - Differences: α*(a) + (-α)*(b) (negative coefficients)
/// - Squares: 1*(a*a) (repeated factors)
#[derive(Clone, Debug, Default)]
pub struct OutputClaimConstraint {
    pub terms: Vec<ProductTerm>,
    pub required_openings: Vec<OpeningId>,
    pub num_challenges: usize,
}

impl OutputClaimConstraint {
    pub fn new(terms: Vec<ProductTerm>, required_openings: Vec<OpeningId>) -> Self {
        let num_challenges = terms
            .iter()
            .flat_map(|t| std::iter::once(&t.coeff).chain(t.factors.iter()))
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
        let mut seen = std::collections::HashSet::new();
        let mut openings = Vec::new();
        for term in terms {
            for vs in std::iter::once(&term.coeff).chain(term.factors.iter()) {
                if let ValueSource::Opening(id) = vs {
                    if seen.insert(*id) {
                        openings.push(*id);
                    }
                }
            }
        }
        openings
    }

    pub fn evaluate<F: crate::field::JoltField>(
        &self,
        opening_values: &[F],
        challenge_values: &[F],
    ) -> F {
        let mut visitor = EvaluateVisitor::new(self, opening_values, challenge_values);
        let mut acc = F::zero();
        self.visit(&mut visitor, &mut acc);
        acc
    }

    pub fn product(factors: Vec<ValueSource>) -> Self {
        let terms = vec![ProductTerm::product(factors)];
        let required_openings = Self::collect_unique_openings(&terms);
        Self::new(terms, required_openings)
    }

    pub fn linear(terms: Vec<(ValueSource, ValueSource)>) -> Self {
        let product_terms: Vec<ProductTerm> = terms
            .into_iter()
            .map(|(coeff, val)| ProductTerm::scaled(coeff, vec![val]))
            .collect();

        let required_openings = Self::collect_unique_openings(&product_terms);
        Self::new(product_terms, required_openings)
    }

    pub fn scaled_linear(multiplier: ValueSource, terms: Vec<(ValueSource, ValueSource)>) -> Self {
        let product_terms: Vec<ProductTerm> = terms
            .into_iter()
            .map(|(coeff, val)| ProductTerm::scaled(coeff, vec![multiplier.clone(), val]))
            .collect();

        let required_openings = Self::collect_unique_openings(&product_terms);
        Self::new(product_terms, required_openings)
    }

    pub fn sum_of_products(product_terms: Vec<ProductTerm>) -> Self {
        let required_openings = Self::collect_unique_openings(&product_terms);
        Self::new(product_terms, required_openings)
    }

    pub fn estimate_aux_var_count(&self) -> usize {
        let mut visitor = CountVisitor;
        let mut count = 0usize;
        self.visit(&mut visitor, &mut count);
        count
    }

    pub fn visit<V: SumOfProductsVisitor>(&self, visitor: &mut V, acc: &mut V::Acc) {
        for term in &self.terms {
            let coeff = visitor.resolve(&term.coeff);
            match term.factors.len() {
                0 => visitor.on_no_factors(acc, coeff),
                1 => {
                    let factor = visitor.resolve(&term.factors[0]);
                    visitor.on_single_factor(acc, coeff, factor);
                }
                _ => {
                    let f0 = visitor.resolve(&term.factors[0]);
                    let f1 = visitor.resolve(&term.factors[1]);
                    visitor.on_chain_start(acc, f0, f1);
                    for factor in &term.factors[2..] {
                        let f = visitor.resolve(factor);
                        visitor.on_chain_step(acc, f);
                    }
                    visitor.on_chain_finalize(acc, coeff);
                }
            }
        }
    }

    pub fn direct(opening: OpeningId) -> Self {
        Self::new(
            vec![ProductTerm::single(ValueSource::Opening(opening))],
            vec![opening],
        )
    }

    /// Builds: opening_0 + Challenge(0)*opening_1 + Challenge(1)*opening_2 + ...
    pub fn weighted_openings(openings: &[OpeningId]) -> Self {
        let mut terms = vec![ProductTerm::single(ValueSource::Opening(openings[0]))];
        for (i, opening) in openings[1..].iter().enumerate() {
            terms.push(ProductTerm::scaled(
                ValueSource::Challenge(i),
                vec![ValueSource::Opening(*opening)],
            ));
        }
        Self::sum_of_products(terms)
    }

    /// Builds: Challenge(0)*opening_0 + Challenge(1)*opening_1 + ...
    pub fn all_weighted_openings(openings: &[OpeningId]) -> Self {
        let terms = openings
            .iter()
            .enumerate()
            .map(|(i, opening)| {
                ProductTerm::scaled(
                    ValueSource::Challenge(i),
                    vec![ValueSource::Opening(*opening)],
                )
            })
            .collect();
        Self::sum_of_products(terms)
    }

    pub fn batch(
        constraints: &[Option<OutputClaimConstraint>],
        _num_batching_coefficients: usize,
    ) -> Option<Self> {
        if constraints.iter().any(|c| c.is_none()) {
            return None;
        }

        let refs: Vec<&OutputClaimConstraint> =
            constraints.iter().map(|c| c.as_ref().unwrap()).collect();
        Some(Self::batch_inner(&refs))
    }

    fn batch_inner(constraints: &[&OutputClaimConstraint]) -> Self {
        let num_instances = constraints.len();
        let mut combined_terms = Vec::new();
        let mut combined_openings = Vec::new();
        let mut challenge_offset = num_instances;

        for (j, constraint) in constraints.iter().enumerate() {
            let alpha_j = ValueSource::Challenge(j);

            for term in &constraint.terms {
                let offset_coeff = Self::offset_challenge(&term.coeff, challenge_offset);
                let offset_factors: Vec<_> = term
                    .factors
                    .iter()
                    .map(|f| Self::offset_challenge(f, challenge_offset))
                    .collect();

                let mut new_factors = vec![alpha_j.clone()];
                new_factors.extend(offset_factors);

                combined_terms.push(ProductTerm::new(offset_coeff, new_factors));
            }

            for opening in &constraint.required_openings {
                if !combined_openings.contains(opening) {
                    combined_openings.push(*opening);
                }
            }

            challenge_offset += constraint.num_challenges;
        }

        Self::new(combined_terms, combined_openings)
    }

    fn offset_challenge(value: &ValueSource, offset: usize) -> ValueSource {
        match value {
            ValueSource::Challenge(idx) => ValueSource::Challenge(idx + offset),
            other => other.clone(),
        }
    }
}

pub trait SumOfProductsVisitor {
    type Resolved;
    type Acc;

    fn resolve(&self, vs: &ValueSource) -> Self::Resolved;
    fn on_no_factors(&mut self, acc: &mut Self::Acc, coeff: Self::Resolved);
    fn on_single_factor(
        &mut self,
        acc: &mut Self::Acc,
        coeff: Self::Resolved,
        factor: Self::Resolved,
    );
    fn on_chain_start(&mut self, acc: &mut Self::Acc, f0: Self::Resolved, f1: Self::Resolved);
    fn on_chain_step(&mut self, acc: &mut Self::Acc, factor: Self::Resolved);
    fn on_chain_finalize(&mut self, acc: &mut Self::Acc, coeff: Self::Resolved);
}

struct CountVisitor;

impl SumOfProductsVisitor for CountVisitor {
    type Resolved = ();
    type Acc = usize;

    fn resolve(&self, _vs: &ValueSource) {}
    fn on_no_factors(&mut self, acc: &mut usize, _coeff: ()) {
        *acc += 1;
    }
    fn on_single_factor(&mut self, acc: &mut usize, _coeff: (), _factor: ()) {
        *acc += 1;
    }
    fn on_chain_start(&mut self, acc: &mut usize, _f0: (), _f1: ()) {
        *acc += 1;
    }
    fn on_chain_step(&mut self, acc: &mut usize, _factor: ()) {
        *acc += 1;
    }
    fn on_chain_finalize(&mut self, acc: &mut usize, _coeff: ()) {
        *acc += 1;
    }
}

pub(crate) struct EvaluateVisitor<'a, F> {
    opening_map: std::collections::HashMap<OpeningId, usize>,
    opening_values: &'a [F],
    challenge_values: &'a [F],
    current_product: F,
}

impl<'a, F: crate::field::JoltField> EvaluateVisitor<'a, F> {
    pub fn new(
        constraint: &OutputClaimConstraint,
        opening_values: &'a [F],
        challenge_values: &'a [F],
    ) -> Self {
        let opening_map = constraint
            .required_openings
            .iter()
            .enumerate()
            .map(|(i, id)| (*id, i))
            .collect();
        Self {
            opening_map,
            opening_values,
            challenge_values,
            current_product: F::zero(),
        }
    }
}

impl<F: crate::field::JoltField> SumOfProductsVisitor for EvaluateVisitor<'_, F> {
    type Resolved = F;
    type Acc = F;

    fn resolve(&self, vs: &ValueSource) -> F {
        match vs {
            ValueSource::Opening(id) => {
                let idx = *self.opening_map.get(id).expect("Opening not found");
                self.opening_values[idx]
            }
            ValueSource::Challenge(idx) => self.challenge_values[*idx],
            ValueSource::Constant(val) => F::from_i128(*val),
        }
    }

    fn on_no_factors(&mut self, acc: &mut F, coeff: F) {
        *acc += coeff;
    }

    fn on_single_factor(&mut self, acc: &mut F, coeff: F, factor: F) {
        *acc += coeff * factor;
    }

    fn on_chain_start(&mut self, _acc: &mut F, f0: F, f1: F) {
        self.current_product = f0 * f1;
    }

    fn on_chain_step(&mut self, _acc: &mut F, factor: F) {
        self.current_product *= factor;
    }

    fn on_chain_finalize(&mut self, acc: &mut F, coeff: F) {
        *acc += coeff * self.current_product;
    }
}

pub type InputClaimConstraint = OutputClaimConstraint;

impl InputClaimConstraint {
    pub fn batch_required(
        constraints: &[InputClaimConstraint],
        num_batching_coefficients: usize,
    ) -> Self {
        assert_eq!(num_batching_coefficients, constraints.len());
        let refs: Vec<&InputClaimConstraint> = constraints.iter().collect();
        Self::batch_inner(&refs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::opening_proof::SumcheckId;
    use crate::zkvm::witness::CommittedPolynomial;

    fn test_opening(idx: usize) -> OpeningId {
        OpeningId::committed(
            CommittedPolynomial::RamRa(idx),
            SumcheckId::RamReadWriteChecking,
        )
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

        let constraint = OutputClaimConstraint::linear(vec![(alpha, y0), (beta, y1)]);

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
        assert_eq!(constraint.required_openings.len(), 4);
        assert_eq!(constraint.num_challenges, 1);
    }
}
