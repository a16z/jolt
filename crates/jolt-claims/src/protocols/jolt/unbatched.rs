use jolt_field::Ring;
use std::ops::{Add, Mul, Sub};

use crate::{challenge, derived, opening};

use super::{
    JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId, JoltPolynomialId, JoltRelationId,
};

/// A pointwise polynomial expression before a relation's claims are folded by
/// its batching challenge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnbatchedClaimExpr {
    Constant(i64),
    Polynomial(JoltPolynomialId),
    Add(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
}

impl UnbatchedClaimExpr {
    pub fn constant(value: i64) -> Self {
        Self::Constant(value)
    }

    pub fn polynomial(polynomial: impl Into<JoltPolynomialId>) -> Self {
        Self::Polynomial(polynomial.into())
    }

    pub fn visit_polynomials(&self, visit: &mut impl FnMut(JoltPolynomialId)) {
        match self {
            Self::Constant(_) => {}
            Self::Polynomial(polynomial) => visit(*polynomial),
            Self::Add(lhs, rhs) | Self::Mul(lhs, rhs) | Self::Sub(lhs, rhs) => {
                lhs.visit_polynomials(visit);
                rhs.visit_polynomials(visit);
            }
        }
    }

    fn at_relation<F: Ring>(&self, relation: JoltRelationId) -> JoltExpr<F> {
        match self {
            Self::Constant(value) => JoltExpr::constant(F::from_i64(*value)),
            Self::Polynomial(polynomial) => {
                opening(JoltOpeningId::polynomial(*polynomial, relation))
            }
            Self::Add(lhs, rhs) => lhs.at_relation(relation) + rhs.at_relation(relation),
            Self::Mul(lhs, rhs) => lhs.at_relation(relation) * rhs.at_relation(relation),
            Self::Sub(lhs, rhs) => lhs.at_relation(relation) - rhs.at_relation(relation),
        }
    }
}

impl Add for UnbatchedClaimExpr {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::Add(Box::new(self), Box::new(rhs))
    }
}

impl Mul for UnbatchedClaimExpr {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::Mul(Box::new(self), Box::new(rhs))
    }
}

impl Sub for UnbatchedClaimExpr {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::Sub(Box::new(self), Box::new(rhs))
    }
}

/// One unfused identity contributing to a gamma-batched sumcheck relation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnbatchedClaim {
    pub input_relation: JoltRelationId,
    pub input: UnbatchedClaimExpr,
    pub output: UnbatchedClaimExpr,
    pub output_weight: JoltDerivedId,
    pub offset: bool,
}

/// The pointwise identities and fold metadata for one sumcheck relation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnbatchedRelation {
    pub output_relation: JoltRelationId,
    pub gamma: JoltChallengeId,
    pub claims: Vec<UnbatchedClaim>,
}

impl UnbatchedRelation {
    pub fn folded_input<F: Ring>(&self) -> JoltExpr<F> {
        let gamma: JoltExpr<F> = challenge(self.gamma);
        self.claims
            .iter()
            .enumerate()
            .fold(JoltExpr::zero(), |folded, (index, claim)| {
                folded + gamma.clone().pow(index) * claim.input.at_relation(claim.input_relation)
            })
    }

    pub fn folded_output<F: Ring>(&self) -> JoltExpr<F> {
        let gamma: JoltExpr<F> = challenge(self.gamma);
        self.claims
            .iter()
            .enumerate()
            .fold(JoltExpr::zero(), |folded, (index, claim)| {
                folded
                    + gamma.clone().pow(index)
                        * derived(claim.output_weight)
                        * claim.output.at_relation(self.output_relation)
            })
    }
}
