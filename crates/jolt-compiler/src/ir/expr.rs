//! Expression types for SNARK protocol composition.
//!
//! `Poly` and `Challenge` are handles into a [`Protocol`](crate::Protocol).
//! Arithmetic ops on them produce [`Expr`]. Upstream claims enter
//! expressions directly via `ClaimId` arithmetic.
//!
//! All expressions are maintained in canonical form: factors sorted within
//! each term, terms sorted lexicographically by factors, like terms merged,
//! zero-coefficient terms eliminated.

use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

use serde::{Deserialize, Serialize};

use super::ClaimId;

/// Handle to a polynomial in the protocol. Index into `Protocol.polynomials`.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct Poly(pub usize);

/// Handle to a Fiat-Shamir challenge. Index into `Protocol.challenge_names`.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct Challenge(pub usize);

impl Challenge {
    /// `self^exp` as a single symbolic factor. `exp=0` → constant 1, `exp=1` → plain challenge.
    #[inline]
    pub fn pow(self, exp: u32) -> Expr {
        if exp == 0 {
            return Expr::from(1i64);
        }
        let factors = vec![Factor::Challenge(self.0); exp as usize];
        Expr(vec![Term { coeff: 1, factors }])
    }
}

/// A symbolic factor in a product term — never a constant.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum Factor {
    Poly(usize),
    Challenge(usize),
    Claim(ClaimId),
}

/// A single product term: coefficient times a product of symbolic factors.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Term {
    pub coeff: i64,
    /// Sorted in canonical order (Poly < Challenge < Claim, then by index).
    pub factors: Vec<Factor>,
}

/// Sum-of-products expression: `Σ coeff_i · ∏ factors_i`.
///
/// Always in canonical form: factors sorted within each term, terms sorted
/// by factors, like terms merged, zero-coefficient terms removed.
/// Empty vec = zero. Semantic equality = structural equality.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct Expr(pub Vec<Term>);

impl Expr {
    fn canonicalize(&mut self) {
        for term in &mut self.0 {
            term.factors.sort_unstable();
        }
        self.0.sort_unstable_by(|a, b| a.factors.cmp(&b.factors));
        self.0.dedup_by(|b, a| {
            if a.factors == b.factors {
                a.coeff += b.coeff;
                true
            } else {
                false
            }
        });
        self.0.retain(|t| t.coeff != 0);
    }

    pub fn claim_deps(&self) -> Vec<ClaimId> {
        let mut deps: Vec<ClaimId> = self
            .0
            .iter()
            .flat_map(|term| {
                term.factors.iter().filter_map(|f| match f {
                    Factor::Claim(id) => Some(*id),
                    Factor::Poly(_) | Factor::Challenge(_) => None,
                })
            })
            .collect();
        deps.sort_unstable();
        deps.dedup();
        deps
    }
}

// --- From conversions ---

impl From<Poly> for Expr {
    #[inline]
    fn from(p: Poly) -> Self {
        Expr(vec![Term {
            coeff: 1,
            factors: vec![Factor::Poly(p.0)],
        }])
    }
}

impl From<Challenge> for Expr {
    #[inline]
    fn from(c: Challenge) -> Self {
        Expr(vec![Term {
            coeff: 1,
            factors: vec![Factor::Challenge(c.0)],
        }])
    }
}

impl From<ClaimId> for Expr {
    #[inline]
    fn from(c: ClaimId) -> Self {
        Expr(vec![Term {
            coeff: 1,
            factors: vec![Factor::Claim(c)],
        }])
    }
}

impl From<i64> for Expr {
    #[inline]
    fn from(v: i64) -> Self {
        if v == 0 {
            Expr(vec![])
        } else {
            Expr(vec![Term {
                coeff: v,
                factors: vec![],
            }])
        }
    }
}

// --- Neg ---

impl Neg for Expr {
    type Output = Expr;
    #[inline]
    fn neg(mut self) -> Expr {
        for term in &mut self.0 {
            term.coeff = -term.coeff;
        }
        self
    }
}

impl Neg for Poly {
    type Output = Expr;
    #[inline]
    fn neg(self) -> Expr {
        -Expr::from(self)
    }
}

impl Neg for Challenge {
    type Output = Expr;
    #[inline]
    fn neg(self) -> Expr {
        -Expr::from(self)
    }
}

impl Neg for ClaimId {
    type Output = Expr;
    #[inline]
    fn neg(self) -> Expr {
        -Expr::from(self)
    }
}

// --- Add ---

impl Add for Expr {
    type Output = Expr;
    #[inline]
    fn add(mut self, rhs: Expr) -> Expr {
        self.0.extend(rhs.0);
        self.canonicalize();
        self
    }
}

macro_rules! impl_add {
    ($lhs:ty, $rhs:ty) => {
        impl Add<$rhs> for $lhs {
            type Output = Expr;
            #[inline]
            fn add(self, rhs: $rhs) -> Expr {
                Expr::from(self) + Expr::from(rhs)
            }
        }
    };
}

impl_add!(Poly, Poly);
impl_add!(Poly, Challenge);
impl_add!(Poly, ClaimId);
impl_add!(Poly, Expr);
impl_add!(Challenge, Poly);
impl_add!(Challenge, Challenge);
impl_add!(Challenge, ClaimId);
impl_add!(Challenge, Expr);
impl_add!(ClaimId, Poly);
impl_add!(ClaimId, Challenge);
impl_add!(ClaimId, ClaimId);
impl_add!(ClaimId, Expr);
impl_add!(Expr, Poly);
impl_add!(Expr, Challenge);
impl_add!(Expr, ClaimId);

// --- Sub ---

impl Sub for Expr {
    type Output = Expr;
    #[inline]
    fn sub(self, rhs: Expr) -> Expr {
        self + (-rhs)
    }
}

macro_rules! impl_sub {
    ($lhs:ty, $rhs:ty) => {
        impl Sub<$rhs> for $lhs {
            type Output = Expr;
            #[inline]
            fn sub(self, rhs: $rhs) -> Expr {
                Expr::from(self) - Expr::from(rhs)
            }
        }
    };
}

impl_sub!(Poly, Poly);
impl_sub!(Poly, Challenge);
impl_sub!(Poly, ClaimId);
impl_sub!(Poly, Expr);
impl_sub!(Challenge, Poly);
impl_sub!(Challenge, Challenge);
impl_sub!(Challenge, ClaimId);
impl_sub!(Challenge, Expr);
impl_sub!(ClaimId, Poly);
impl_sub!(ClaimId, Challenge);
impl_sub!(ClaimId, ClaimId);
impl_sub!(ClaimId, Expr);
impl_sub!(Expr, Poly);
impl_sub!(Expr, Challenge);
impl_sub!(Expr, ClaimId);

// --- Mul ---

impl Mul for Expr {
    type Output = Expr;
    fn mul(self, rhs: Expr) -> Expr {
        let mut terms = Vec::with_capacity(self.0.len() * rhs.0.len());
        for lt in &self.0 {
            for rt in &rhs.0 {
                let coeff = lt.coeff * rt.coeff;
                if coeff == 0 {
                    continue;
                }
                let mut factors = Vec::with_capacity(lt.factors.len() + rt.factors.len());
                factors.extend_from_slice(&lt.factors);
                factors.extend_from_slice(&rt.factors);
                terms.push(Term { coeff, factors });
            }
        }
        let mut expr = Expr(terms);
        expr.canonicalize();
        expr
    }
}

macro_rules! impl_mul {
    ($lhs:ty, $rhs:ty) => {
        impl Mul<$rhs> for $lhs {
            type Output = Expr;
            #[inline]
            fn mul(self, rhs: $rhs) -> Expr {
                Expr::from(self) * Expr::from(rhs)
            }
        }
    };
}

impl_mul!(Poly, Poly);
impl_mul!(Poly, Challenge);
impl_mul!(Poly, ClaimId);
impl_mul!(Poly, Expr);
impl_mul!(Challenge, Poly);
impl_mul!(Challenge, Challenge);
impl_mul!(Challenge, ClaimId);
impl_mul!(Challenge, Expr);
impl_mul!(ClaimId, Poly);
impl_mul!(ClaimId, Challenge);
impl_mul!(ClaimId, ClaimId);
impl_mul!(ClaimId, Expr);
impl_mul!(Expr, Poly);
impl_mul!(Expr, Challenge);
impl_mul!(Expr, ClaimId);

// --- Display ---

impl fmt::Display for Factor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Factor::Poly(i) => write!(f, "p{i}"),
            Factor::Challenge(i) => write!(f, "r{i}"),
            Factor::Claim(id) => write!(f, "claim_{}", id.0),
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return write!(f, "0");
        }
        for (i, term) in self.0.iter().enumerate() {
            let abs = term.coeff.unsigned_abs();

            if i > 0 {
                if term.coeff < 0 {
                    write!(f, " - ")?;
                } else {
                    write!(f, " + ")?;
                }
            } else if term.coeff < 0 {
                write!(f, "-")?;
            }

            if term.factors.is_empty() {
                write!(f, "{abs}")?;
            } else {
                if abs != 1 {
                    write!(f, "{abs} · ")?;
                }
                for (j, factor) in term.factors.iter().enumerate() {
                    if j > 0 {
                        write!(f, " · ")?;
                    }
                    write!(f, "{factor}")?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn term(coeff: i64, factors: Vec<Factor>) -> Term {
        Term { coeff, factors }
    }

    #[test]
    fn poly_sub_produces_negated_term() {
        let expr = Poly(0) - Poly(1);
        assert_eq!(
            expr.0,
            vec![
                term(1, vec![Factor::Poly(0)]),
                term(-1, vec![Factor::Poly(1)]),
            ]
        );
    }

    #[test]
    fn mul_distributes_over_add() {
        let expr = Poly(0) * (Poly(1) + Poly(2));
        assert_eq!(
            expr.0,
            vec![
                term(1, vec![Factor::Poly(0), Factor::Poly(1)]),
                term(1, vec![Factor::Poly(0), Factor::Poly(2)]),
            ]
        );
    }

    #[test]
    fn mul_distributes_over_sub() {
        let expr = Poly(0) * (Poly(1) - Poly(2));
        assert_eq!(
            expr.0,
            vec![
                term(1, vec![Factor::Poly(0), Factor::Poly(1)]),
                term(-1, vec![Factor::Poly(0), Factor::Poly(2)]),
            ]
        );
    }

    #[test]
    fn challenge_mul_poly() {
        let expr = Challenge(0) * Poly(0);
        // Canonical factor order: Poly < Challenge
        assert_eq!(
            expr.0,
            vec![term(1, vec![Factor::Poly(0), Factor::Challenge(0)])]
        );
    }

    #[test]
    fn spartan_outer_composition() {
        let eq = Poly(0);
        let az = Poly(1);
        let bz = Poly(2);
        let cz = Poly(3);
        let expr = eq * (az * bz - cz);
        assert_eq!(
            expr.0,
            vec![
                term(1, vec![Factor::Poly(0), Factor::Poly(1), Factor::Poly(2)]),
                term(-1, vec![Factor::Poly(0), Factor::Poly(3)]),
            ]
        );
    }

    #[test]
    fn booleanity_composition() {
        let eq = Poly(0);
        let gamma = Challenge(0);
        let h = Poly(1);
        let expr = eq * gamma * (h * h - h);
        // Canonical order: [P(0),P(1),P(1),C(0)] < [P(0),P(1),C(0)]
        // because P(1) < C(0) at index 2
        assert_eq!(
            expr.0,
            vec![
                term(
                    1,
                    vec![
                        Factor::Poly(0),
                        Factor::Poly(1),
                        Factor::Poly(1),
                        Factor::Challenge(0),
                    ]
                ),
                term(
                    -1,
                    vec![Factor::Poly(0), Factor::Poly(1), Factor::Challenge(0)]
                ),
            ]
        );
    }

    #[test]
    fn claim_in_expr() {
        let rho = Challenge(0);
        let c0 = ClaimId(0);
        let c1 = ClaimId(1);
        let expr = rho * c0 + c1;
        assert_eq!(
            expr.0,
            vec![
                term(1, vec![Factor::Challenge(0), Factor::Claim(ClaimId(0))]),
                term(1, vec![Factor::Claim(ClaimId(1))]),
            ]
        );
    }

    #[test]
    fn claim_deps_collected() {
        let rho = Challenge(0);
        let c0 = ClaimId(0);
        let c1 = ClaimId(1);
        let expr = rho * c0 + c1;
        assert_eq!(expr.claim_deps(), vec![ClaimId(0), ClaimId(1)]);
    }

    #[test]
    fn zero_from_i64() {
        assert_eq!(Expr::from(0i64), Expr(vec![]));
    }

    #[test]
    fn constant_from_i64() {
        assert_eq!(Expr::from(5i64).0, vec![term(5, vec![])]);
    }

    #[test]
    fn double_neg_cancels() {
        assert_eq!(-(-Expr::from(Poly(0))), Expr::from(Poly(0)));
    }

    #[test]
    fn neg_constant_folds() {
        assert_eq!((-Expr::from(5i64)).0, vec![term(-5, vec![])]);
    }

    #[test]
    fn mul_folds_constants() {
        let expr = Expr::from(2i64) * Expr::from(3i64);
        assert_eq!(expr.0, vec![term(6, vec![])]);
    }

    #[test]
    fn mul_neg_neg_cancels() {
        let expr = (-Expr::from(Poly(0))) * (-Expr::from(Poly(1)));
        assert_eq!(
            expr.0,
            vec![term(1, vec![Factor::Poly(0), Factor::Poly(1)])]
        );
    }

    // --- Canonicalization tests ---

    #[test]
    fn canonicalize_merges_like_terms() {
        let a = Poly(0);
        let expr = a + a;
        assert_eq!(expr.0, vec![term(2, vec![Factor::Poly(0)])]);
    }

    #[test]
    fn canonicalize_cancels_opposite_terms() {
        let a = Poly(0);
        let expr = a - a;
        assert_eq!(expr, Expr::from(0i64));
    }

    #[test]
    fn canonicalize_sorts_factors() {
        // Challenge * Poly should sort to [Poly, Challenge]
        let expr = Challenge(0) * Poly(0);
        assert_eq!(
            expr.0[0].factors,
            vec![Factor::Poly(0), Factor::Challenge(0)]
        );
    }

    #[test]
    fn canonicalize_sorts_terms() {
        // b + a should sort to a + b
        let expr = Poly(1) + Poly(0);
        assert_eq!(
            expr.0,
            vec![
                term(1, vec![Factor::Poly(0)]),
                term(1, vec![Factor::Poly(1)]),
            ]
        );
    }

    #[test]
    fn semantic_equality() {
        let a = Poly(0);
        let b = Poly(1);
        // a*b and b*a are the same expression
        assert_eq!(a * b, b * a);
        // a + b and b + a are the same expression
        assert_eq!(a + b, b + a);
    }

    // --- Display tests ---

    #[test]
    fn display_zero() {
        assert_eq!(Expr::from(0i64).to_string(), "0");
    }

    #[test]
    fn display_subtraction() {
        let expr = Poly(0) - Poly(1);
        assert_eq!(expr.to_string(), "p0 - p1");
    }

    #[test]
    fn display_sum_of_products() {
        let expr = Poly(0) * (Poly(1) + Poly(2));
        assert_eq!(expr.to_string(), "p0 · p1 + p0 · p2");
    }

    #[test]
    fn display_spartan() {
        let eq = Poly(0);
        let az = Poly(1);
        let bz = Poly(2);
        let cz = Poly(3);
        let expr = eq * (az * bz - cz);
        assert_eq!(expr.to_string(), "p0 · p1 · p2 - p0 · p3");
    }

    #[test]
    fn display_with_challenge_and_claim() {
        let expr = Challenge(0) * ClaimId(0) + ClaimId(1);
        assert_eq!(expr.to_string(), "r0 · claim_0 + claim_1");
    }

    #[test]
    fn display_scaled() {
        let expr = Expr::from(3i64) * Poly(0);
        assert_eq!(expr.to_string(), "3 · p0");
    }

    #[test]
    fn display_negative_constant() {
        let expr = -(Expr::from(5i64));
        assert_eq!(expr.to_string(), "-5");
    }
}
