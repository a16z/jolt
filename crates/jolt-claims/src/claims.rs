use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

/// A claim that a polynomial evaluates to `value` at `point`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvaluationClaim<F> {
    pub point: Vec<F>,
    pub value: F,
}

impl<F> EvaluationClaim<F> {
    pub fn new(point: Vec<F>, value: F) -> Self {
        Self { point, value }
    }
}

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
        mut opening_value: OpeningValue,
        mut challenge_value: ChallengeValue,
        mut public_value: PublicValue,
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
                    Source::Opening(id) => opening_value(id),
                    Source::Challenge(id) => challenge_value(id),
                    Source::Public(id) => public_value(id),
                };
            }
            result += value;
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
    pub expression: Expr<F, O, P, C>,
    pub required_openings: Vec<O>,
    pub required_publics: Vec<P>,
    pub required_challenges: Vec<C>,
    pub num_challenges: usize,
}

impl<F, O: Clone + Eq, P: Clone + Eq, C: Clone + Eq> From<Expr<F, O, P, C>>
    for ClaimExpression<F, O, P, C>
{
    fn from(expression: Expr<F, O, P, C>) -> Self {
        let required_openings = expression.required_openings();
        let required_publics = expression.required_publics();
        let required_challenges = expression.required_challenges();
        let num_challenges = required_challenges.len();
        Self {
            expression,
            required_openings,
            required_publics,
            required_challenges,
            num_challenges,
        }
    }
}

impl<F, O, P, C: Eq> ClaimExpression<F, O, P, C> {
    pub fn challenge_index(&self, id: &C) -> Option<usize> {
        self.required_challenges
            .iter()
            .position(|challenge| challenge == id)
    }
}

pub type InputClaimExpression<F, O, P = (), C = usize> = ClaimExpression<F, O, P, C>;
pub type OutputClaimExpression<F, O, P = (), C = usize> = ClaimExpression<F, O, P, C>;

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

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
    fn claim_expression_derives_metadata() {
        let expression: Expr<Fr, Opening, Public> =
            challenge(2) * opening(Opening::B) * public(Public::Offset) + opening(Opening::A);
        let claim = ClaimExpression::from(expression.clone());

        assert_eq!(claim.expression, expression);
        assert_eq!(claim.required_openings, vec![Opening::B, Opening::A]);
        assert_eq!(claim.required_publics, vec![Public::Offset]);
        assert_eq!(claim.required_challenges, vec![2]);
        assert_eq!(claim.num_challenges, 1);
        assert_eq!(claim.challenge_index(&2), Some(0));
        assert_eq!(claim.challenge_index(&1), None);
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
