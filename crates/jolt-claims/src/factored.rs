//! Claim formulas in factored form: `Σ_t coefficient_t · ∏_j factor_{t,j}`.
//!
//! [`Expr`] is a flat sum of monomials, so a product of sums such as
//! `τ · Az · Bz` expands into one monomial per pair of summands. Masked
//! verification (`jolt-zk::masking`) proves each product of hidden factors
//! with one product row per factor, so it consumes this form instead: a
//! relation whose formula is a product of sums states it once here, and its
//! [`Expr`] is the [`expand`](FactoredExpr::expand)ed form.

use std::ops::Add;

use jolt_algebra::Ring;

use crate::{Expr, Source, Term};

/// One product: `coefficient · ∏ factors`. Each factor is an [`Expr`]; the
/// masked lowering requires every factor to be affine in its openings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FactoredTerm<F, O, P = (), C = usize> {
    pub coefficient: F,
    pub factors: Vec<Expr<F, O, P, C>>,
}

/// A sum of [`FactoredTerm`]s.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FactoredExpr<F, O, P = (), C = usize> {
    pub terms: Vec<FactoredTerm<F, O, P, C>>,
}

impl<F, O, P, C> FactoredExpr<F, O, P, C> {
    pub fn zero() -> Self {
        Self { terms: Vec::new() }
    }
}

impl<F: Ring, O, P, C> FactoredExpr<F, O, P, C> {
    /// The single term `∏ factors`.
    pub fn product(factors: Vec<Expr<F, O, P, C>>) -> Self {
        Self {
            terms: vec![FactoredTerm {
                coefficient: F::one(),
                factors,
            }],
        }
    }

    /// Multiplies every term's coefficient by `scale`.
    pub fn scale(mut self, scale: F) -> Self {
        for term in &mut self.terms {
            term.coefficient *= scale;
        }
        self
    }

    /// Rewrites every id, factor by factor.
    pub fn map_sources<O2, P2, C2>(
        self,
        mut map: impl FnMut(Source<O, P, C>) -> Source<O2, P2, C2>,
    ) -> FactoredExpr<F, O2, P2, C2> {
        FactoredExpr {
            terms: self
                .terms
                .into_iter()
                .map(|term| FactoredTerm {
                    coefficient: term.coefficient,
                    factors: term
                        .factors
                        .into_iter()
                        .map(|factor| Expr {
                            terms: factor
                                .terms
                                .into_iter()
                                .map(|inner| Term {
                                    coefficient: inner.coefficient,
                                    factors: inner.factors.into_iter().map(&mut map).collect(),
                                })
                                .collect(),
                        })
                        .collect(),
                })
                .collect(),
        }
    }
}

impl<F: Ring, O: Clone, P: Clone, C: Clone> FactoredExpr<F, O, P, C> {
    /// The flat monomial form: each term multiplied out.
    pub fn expand(self) -> Expr<F, O, P, C> {
        self.terms.into_iter().fold(Expr::zero(), |sum, term| {
            sum + term
                .factors
                .into_iter()
                .fold(Expr::constant(term.coefficient), |product, factor| {
                    product * factor
                })
        })
    }
}

/// Every monomial as a term whose factors are its single sources.
impl<F: Ring, O, P, C> From<Expr<F, O, P, C>> for FactoredExpr<F, O, P, C> {
    fn from(expr: Expr<F, O, P, C>) -> Self {
        Self {
            terms: expr
                .terms
                .into_iter()
                .map(|term| FactoredTerm {
                    coefficient: term.coefficient,
                    factors: term
                        .factors
                        .into_iter()
                        .map(|source| Expr {
                            terms: vec![Term::source(source)],
                        })
                        .collect(),
                })
                .collect(),
        }
    }
}

impl<F, O, P, C> Add for FactoredExpr<F, O, P, C> {
    type Output = Self;

    fn add(mut self, mut rhs: Self) -> Self::Output {
        self.terms.append(&mut rhs.terms);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{challenge, derived, opening};
    use jolt_algebra::Fr;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Opening {
        A,
        B,
        C,
    }

    type E = Expr<Fr, Opening, u8, u8>;

    fn evaluate(expr: &E) -> Fr {
        expr.evaluate(
            |id| match id {
                Opening::A => Fr::from_u64(3),
                Opening::B => Fr::from_u64(5),
                Opening::C => Fr::from_u64(7),
            },
            |_| Fr::from_u64(11),
            |_| Fr::from_u64(13),
        )
    }

    #[test]
    fn expansion_multiplies_out_products_of_sums() {
        let left: E = opening(Opening::A) + challenge(0u8) * opening(Opening::B);
        let right: E = opening(Opening::C) + derived(1u8);
        let factored = FactoredExpr::product(vec![derived(2u8), left.clone(), right.clone()])
            .scale(Fr::from_u64(2));
        let expected =
            Fr::from_u64(2) * evaluate(&derived(2u8)) * evaluate(&left) * evaluate(&right);
        assert_eq!(evaluate(&factored.expand()), expected);
    }

    #[test]
    fn monomial_conversion_keeps_one_source_per_factor() {
        let expr: E = opening(Opening::A) * opening(Opening::B) * challenge(0u8) + derived(1u8);
        let factored = FactoredExpr::from(expr.clone());
        assert_eq!(factored.terms.len(), 2);
        assert!(factored
            .terms
            .iter()
            .flat_map(|term| &term.factors)
            .all(|factor| factor.terms.len() == 1 && factor.terms[0].factors.len() == 1));
        assert_eq!(evaluate(&factored.expand()), evaluate(&expr));
    }
}
