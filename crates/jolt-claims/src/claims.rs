use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

use crate::util::extend_unique;

/// An atomic value used inside a symbolic claim expression.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Source<O, P = (), C = usize> {
    Opening(O),
    Challenge(C),
    Public(P),
}

/// One product term: `coefficient * product(factors)`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Term<F, O, P = (), C = usize> {
    pub coefficient: F,
    pub factors: Vec<Source<O, P, C>>,
}

impl<F, O, P, C> Term<F, O, P, C> {
    pub fn constant(coefficient: F) -> Self {
        Self {
            coefficient,
            factors: Vec::new(),
        }
    }
}

impl<F: RingCore, O, P, C> Term<F, O, P, C> {
    pub fn source(source: Source<O, P, C>) -> Self {
        Self {
            coefficient: F::one(),
            factors: vec![source],
        }
    }
}

/// A symbolic sum-of-products expression.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Expr<F, O, P = (), C = usize> {
    pub terms: Vec<Term<F, O, P, C>>,
}

impl<F, O, P, C> Expr<F, O, P, C> {
    pub fn zero() -> Self {
        Self { terms: Vec::new() }
    }

    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }
}

impl<F, O, P, C: Clone + Eq> Expr<F, O, P, C> {
    pub fn required_challenges(&self) -> Vec<C> {
        let mut challenges = Vec::new();
        for source in self.terms.iter().flat_map(|term| term.factors.iter()) {
            if let Source::Challenge(id) = source {
                if !challenges.contains(id) {
                    challenges.push(id.clone());
                }
            }
        }
        challenges
    }

    pub fn num_challenges(&self) -> usize {
        self.required_challenges().len()
    }
}

impl<F: RingCore, O, P, C> Expr<F, O, P, C> {
    pub fn one() -> Self {
        Self {
            terms: vec![Term::constant(F::one())],
        }
    }

    pub fn constant(value: F) -> Self {
        if value.is_zero() {
            Self::zero()
        } else {
            Self {
                terms: vec![Term::constant(value)],
            }
        }
    }

    pub fn evaluate<OpeningValue, ChallengeValue, PublicValue>(
        &self,
        mut resolve_opening: OpeningValue,
        mut resolve_challenge: ChallengeValue,
        mut resolve_public: PublicValue,
    ) -> F
    where
        OpeningValue: FnMut(&O) -> F,
        ChallengeValue: FnMut(&C) -> F,
        PublicValue: FnMut(&P) -> F,
    {
        let mut result = F::zero();
        for term in &self.terms {
            let mut value = term.coefficient;
            for factor in &term.factors {
                value *= match factor {
                    Source::Opening(id) => resolve_opening(id),
                    Source::Challenge(id) => resolve_challenge(id),
                    Source::Public(id) => resolve_public(id),
                };
            }
            result += value;
        }
        result
    }

    pub fn try_evaluate<OpeningValue, ChallengeValue, PublicValue, Error>(
        &self,
        mut resolve_opening: OpeningValue,
        mut resolve_challenge: ChallengeValue,
        mut resolve_public: PublicValue,
    ) -> Result<F, Error>
    where
        OpeningValue: FnMut(&O) -> Result<F, Error>,
        ChallengeValue: FnMut(&C) -> Result<F, Error>,
        PublicValue: FnMut(&P) -> Result<F, Error>,
    {
        let mut result = F::zero();
        for term in &self.terms {
            let mut value = term.coefficient;
            for factor in &term.factors {
                value *= match factor {
                    Source::Opening(id) => resolve_opening(id)?,
                    Source::Challenge(id) => resolve_challenge(id)?,
                    Source::Public(id) => resolve_public(id)?,
                };
            }
            result += value;
        }
        Ok(result)
    }
}

impl<F: RingCore, O: Clone, P: Clone, C: Clone> Expr<F, O, P, C> {
    pub fn pow(self, mut exponent: usize) -> Self {
        let mut result = Self::one();
        let mut base = self;

        while exponent > 0 {
            if exponent % 2 == 1 {
                result = result * base.clone();
            }
            exponent /= 2;
            if exponent > 0 {
                base = base.clone() * base;
            }
        }

        result
    }
}

impl<F: RingCore, O, C> Expr<F, O, (), C> {
    pub fn evaluate_without_public<OpeningValue, ChallengeValue>(
        &self,
        opening_value: OpeningValue,
        challenge_value: ChallengeValue,
    ) -> F
    where
        OpeningValue: FnMut(&O) -> F,
        ChallengeValue: FnMut(&C) -> F,
    {
        self.evaluate(opening_value, challenge_value, |()| F::zero())
    }
}

impl<F, O: Clone + Eq, P, C> Expr<F, O, P, C> {
    pub fn required_openings(&self) -> Vec<O> {
        let mut openings = Vec::new();
        for source in self.terms.iter().flat_map(|term| term.factors.iter()) {
            if let Source::Opening(id) = source {
                if !openings.contains(id) {
                    openings.push(id.clone());
                }
            }
        }
        openings
    }
}

impl<F, O, P: Clone + Eq, C> Expr<F, O, P, C> {
    pub fn required_publics(&self) -> Vec<P> {
        let mut publics = Vec::new();
        for source in self.terms.iter().flat_map(|term| term.factors.iter()) {
            if let Source::Public(id) = source {
                if !publics.contains(id) {
                    publics.push(id.clone());
                }
            }
        }
        publics
    }
}

/// Builds an opening source expression.
pub fn opening<F: RingCore, O, P, C>(id: O) -> Expr<F, O, P, C> {
    Expr {
        terms: vec![Term::source(Source::Opening(id))],
    }
}

/// Builds a Fiat-Shamir challenge source expression.
pub fn challenge<F: RingCore, O, P, C>(id: C) -> Expr<F, O, P, C> {
    Expr {
        terms: vec![Term::source(Source::Challenge(id))],
    }
}

/// Builds a named public-value source expression.
pub fn public<F: RingCore, O, P, C>(id: P) -> Expr<F, O, P, C> {
    Expr {
        terms: vec![Term::source(Source::Public(id))],
    }
}

/// Builds a constant expression.
pub fn constant<F: RingCore, O, P, C>(value: F) -> Expr<F, O, P, C> {
    Expr::constant(value)
}

/// Expression metadata used by claim-check protocols.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimExpression<F, O, P = (), C = usize> {
    expression: Expr<F, O, P, C>,
    pub required_openings: Vec<O>,
    pub required_publics: Vec<P>,
    pub required_challenges: Vec<C>,
}

impl<F, O, P, C> ClaimExpression<F, O, P, C> {
    pub fn expression(&self) -> &Expr<F, O, P, C> {
        &self.expression
    }
}

impl<F, O: Clone + Eq, P: Clone + Eq, C: Clone + Eq> From<Expr<F, O, P, C>>
    for ClaimExpression<F, O, P, C>
{
    fn from(expression: Expr<F, O, P, C>) -> Self {
        let required_openings = expression.required_openings();
        let required_publics = expression.required_publics();
        let required_challenges = expression.required_challenges();
        Self {
            expression,
            required_openings,
            required_publics,
            required_challenges,
        }
    }
}

impl<F, O, P, C: Eq> ClaimExpression<F, O, P, C> {
    pub fn challenge_index(&self, id: &C) -> Option<usize> {
        self.required_challenges
            .iter()
            .position(|challenge| challenge == id)
    }

    pub fn num_challenges(&self) -> usize {
        self.required_challenges.len()
    }
}

impl<F, O, P, C: Eq> ClaimExpression<F, O, P, C> {
    pub fn pull_challenge_for_transcript_sync(&mut self, id: C) {
        if !self.required_challenges.contains(&id) {
            self.required_challenges.push(id);
        }
    }

    pub fn pull_challenges_for_transcript_sync<I>(&mut self, ids: I)
    where
        I: IntoIterator<Item = C>,
    {
        for id in ids {
            self.pull_challenge_for_transcript_sync(id);
        }
    }
}

pub type InputClaimExpression<F, O, P = (), C = usize> = ClaimExpression<F, O, P, C>;
pub type OutputClaimExpression<F, O, P = (), C = usize> = ClaimExpression<F, O, P, C>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SameEvaluation<O> {
    pub left: O,
    pub right: O,
}

impl<O> SameEvaluation<O> {
    pub fn new(left: O, right: O) -> Self {
        Self { left, right }
    }
}

pub trait SameEvaluationAs<Rhs = Self> {
    type Output;

    fn same_evaluation_as(self, rhs: Rhs) -> Self::Output;
}

impl<O> SameEvaluationAs for O {
    type Output = SameEvaluation<O>;

    fn same_evaluation_as(self, rhs: Self) -> Self::Output {
        SameEvaluation::new(self, rhs)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsistencyClaim<F, O, P = (), C = usize> {
    EqualExpressions {
        left: Expr<F, O, P, C>,
        right: Expr<F, O, P, C>,
    },
}

impl<F: RingCore, O, P, C> ConsistencyClaim<F, O, P, C> {
    pub fn same_evaluation(left: O, right: O) -> Self {
        Self::EqualExpressions {
            left: opening(left),
            right: opening(right),
        }
    }

    pub fn equal_expressions(left: Expr<F, O, P, C>, right: Expr<F, O, P, C>) -> Self {
        Self::EqualExpressions { left, right }
    }
}

impl<F: RingCore, O, P, C> From<SameEvaluation<O>> for ConsistencyClaim<F, O, P, C> {
    fn from(value: SameEvaluation<O>) -> Self {
        Self::same_evaluation(value.left, value.right)
    }
}

impl<F, O: Clone + Eq, P, C> ConsistencyClaim<F, O, P, C> {
    pub fn required_openings(&self) -> Vec<O> {
        match self {
            Self::EqualExpressions { left, right } => {
                let mut openings = left.required_openings();
                extend_unique(&mut openings, &right.required_openings());
                openings
            }
        }
    }
}

impl<F, O, P: Clone + Eq, C> ConsistencyClaim<F, O, P, C> {
    pub fn required_publics(&self) -> Vec<P> {
        match self {
            Self::EqualExpressions { left, right } => {
                let mut publics = left.required_publics();
                extend_unique(&mut publics, &right.required_publics());
                publics
            }
        }
    }
}

impl<F, O, P, C: Clone + Eq> ConsistencyClaim<F, O, P, C> {
    pub fn required_challenges(&self) -> Vec<C> {
        match self {
            Self::EqualExpressions { left, right } => {
                let mut challenges = left.required_challenges();
                extend_unique(&mut challenges, &right.required_challenges());
                challenges
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt, RingCore};

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    enum Opening {
        A,
        B,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    enum Public {
        Offset,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    enum Challenge {
        Alpha,
        Beta,
    }

    #[test]
    fn expression_evaluates_with_resolvers() {
        let expr: Expr<Fr, Opening> =
            constant(Fr::from_u64(2)) * opening(Opening::A) * opening(Opening::B)
                + challenge(1) * opening(Opening::A)
                - constant(Fr::from_u64(5));

        let value = expr.evaluate_without_public(
            |opening| match opening {
                Opening::A => Fr::from_u64(3),
                Opening::B => Fr::from_u64(7),
            },
            |index| match *index {
                1 => Fr::from_u64(11),
                _ => Fr::from_u64(0),
            },
        );

        assert_eq!(value, Fr::from_u64(70));
    }

    #[test]
    fn expression_evaluates_public_sources() {
        let expr: Expr<Fr, Opening, Public> =
            opening(Opening::A) * public(Public::Offset) + constant(Fr::from_u64(4));

        let value = expr.evaluate(
            |opening| match opening {
                Opening::A => Fr::from_u64(3),
                Opening::B => Fr::from_u64(0),
            },
            |_| Fr::from_u64(0),
            |public| match public {
                Public::Offset => Fr::from_u64(9),
            },
        );

        assert_eq!(value, Fr::from_u64(31));
    }

    #[test]
    fn pow2_builds_field_native_powers() {
        assert_eq!(Fr::pow2(0), Fr::from_u64(1));
        assert_eq!(Fr::pow2(1), Fr::from_u64(2));
        assert_eq!(Fr::pow2(63), Fr::from_u64(1u64 << 63));
    }

    #[test]
    fn expression_powers_are_structural_products() {
        let gamma: Expr<Fr, Opening, Public, Challenge> = challenge(Challenge::Alpha);
        let expr = gamma.pow(3) * opening(Opening::A);

        assert_eq!(expr.required_challenges(), vec![Challenge::Alpha]);
        assert_eq!(expr.required_openings(), vec![Opening::A]);
        assert_eq!(
            expr.evaluate(
                |_| Fr::from_u64(5),
                |_| Fr::from_u64(7),
                |_| Fr::from_u64(0)
            ),
            Fr::from_u64(1715)
        );
    }

    #[test]
    fn expression_zero_power_is_one() {
        let expr: Expr<Fr, Opening, Public, Challenge> = Expr::zero().pow(0);

        assert_eq!(
            expr.evaluate(
                |_| Fr::from_u64(0),
                |_| Fr::from_u64(0),
                |_| Fr::from_u64(0)
            ),
            Fr::from_u64(1)
        );
    }

    #[test]
    fn zero_coefficient_terms_keep_non_constant_metadata() {
        let expr: Expr<Fr, Opening, Public, Challenge> = Term {
            coefficient: Fr::from_u64(0),
            factors: vec![Source::Opening(Opening::A)],
        }
        .into();

        assert_eq!(expr.required_openings(), vec![Opening::A]);
        assert!(
            expr.evaluate(
                |_| Fr::from_u64(9),
                |_| Fr::from_u64(0),
                |_| Fr::from_u64(0)
            ) == Fr::from_u64(0)
        );
    }

    #[test]
    fn claim_expression_derives_metadata() {
        let expression: Expr<Fr, Opening, Public> =
            challenge(2) * opening(Opening::B) * public(Public::Offset) + opening(Opening::A);
        let claim = ClaimExpression::from(expression.clone());

        assert_eq!(claim.expression(), &expression);
        assert_eq!(claim.required_openings, vec![Opening::B, Opening::A]);
        assert_eq!(claim.required_publics, vec![Public::Offset]);
        assert_eq!(claim.required_challenges, vec![2]);
        assert_eq!(claim.num_challenges(), 1);
        assert_eq!(claim.challenge_index(&2), Some(0));
        assert_eq!(claim.challenge_index(&1), None);
    }

    #[test]
    fn same_evaluation_claim_requires_both_openings() {
        let consistency: ConsistencyClaim<Fr, Opening, Public, Challenge> =
            ConsistencyClaim::same_evaluation(Opening::A, Opening::B);

        assert_eq!(
            consistency,
            ConsistencyClaim::same_evaluation(Opening::A, Opening::B)
        );
        assert_eq!(
            consistency.required_openings(),
            vec![Opening::A, Opening::B]
        );
        assert!(consistency.required_publics().is_empty());
        assert!(consistency.required_challenges().is_empty());
    }

    #[test]
    fn expression_consistency_claim_derives_metadata() {
        let consistency: ConsistencyClaim<Fr, Opening, Public, Challenge> =
            ConsistencyClaim::equal_expressions(
                opening(Opening::A) + challenge(Challenge::Alpha) * public(Public::Offset),
                opening(Opening::B) + challenge(Challenge::Alpha) + challenge(Challenge::Beta),
            );

        assert_eq!(
            consistency.required_openings(),
            vec![Opening::A, Opening::B]
        );
        assert_eq!(consistency.required_publics(), vec![Public::Offset]);
        assert_eq!(
            consistency.required_challenges(),
            vec![Challenge::Alpha, Challenge::Beta]
        );
    }

    #[test]
    fn typed_challenges_have_canonical_order() {
        let expression: Expr<Fr, Opening, Public, Challenge> = challenge(Challenge::Beta)
            * opening(Opening::B)
            + challenge(Challenge::Alpha) * public(Public::Offset)
            + challenge(Challenge::Beta);
        let claim = ClaimExpression::from(expression.clone());

        assert_eq!(
            expression.required_challenges(),
            vec![Challenge::Beta, Challenge::Alpha]
        );
        assert_eq!(
            claim.required_challenges,
            vec![Challenge::Beta, Challenge::Alpha]
        );
        assert_eq!(claim.challenge_index(&Challenge::Beta), Some(0));
        assert_eq!(claim.challenge_index(&Challenge::Alpha), Some(1));
    }
}
