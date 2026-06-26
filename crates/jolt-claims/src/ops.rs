use std::ops::{Add, Mul, Neg, Sub};

use jolt_field::{FromPrimitiveInt, RingCore};

use crate::{Expr, Term};

impl<F: RingCore, O, P, C> From<Term<F, O, P, C>> for Expr<F, O, P, C> {
    fn from(term: Term<F, O, P, C>) -> Self {
        if term.coefficient.is_zero() && term.factors.is_empty() {
            Self::zero()
        } else {
            Self { terms: vec![term] }
        }
    }
}

impl<F: RingCore + FromPrimitiveInt, O, P, C> From<i128> for Expr<F, O, P, C> {
    fn from(value: i128) -> Self {
        Self::constant(F::from_i128(value))
    }
}

impl<F, O, P, C> Add for Expr<F, O, P, C> {
    type Output = Self;

    fn add(mut self, mut rhs: Self) -> Self::Output {
        self.terms.append(&mut rhs.terms);
        self
    }
}

impl<F: Clone, O: Clone, P: Clone, C: Clone> Add<&Expr<F, O, P, C>> for Expr<F, O, P, C> {
    type Output = Self;

    fn add(mut self, rhs: &Expr<F, O, P, C>) -> Self::Output {
        self.terms.extend(rhs.terms.iter().cloned());
        self
    }
}

impl<F: Clone, O: Clone, P: Clone, C: Clone> Add<Expr<F, O, P, C>> for &Expr<F, O, P, C> {
    type Output = Expr<F, O, P, C>;

    fn add(self, rhs: Expr<F, O, P, C>) -> Self::Output {
        (*self).clone() + rhs
    }
}

impl<F: Clone, O: Clone, P: Clone, C: Clone> Add<&Expr<F, O, P, C>> for &Expr<F, O, P, C> {
    type Output = Expr<F, O, P, C>;

    fn add(self, rhs: &Expr<F, O, P, C>) -> Self::Output {
        (*self).clone() + rhs
    }
}

impl<F: RingCore + FromPrimitiveInt, O, P, C> Add<i128> for Expr<F, O, P, C> {
    type Output = Self;

    fn add(self, rhs: i128) -> Self::Output {
        self + Self::from(rhs)
    }
}

impl<F: RingCore + FromPrimitiveInt, O, P, C> Add<Expr<F, O, P, C>> for i128 {
    type Output = Expr<F, O, P, C>;

    fn add(self, rhs: Expr<F, O, P, C>) -> Self::Output {
        Expr::from(self) + rhs
    }
}

impl<F: RingCore, O, P, C> Sub for Expr<F, O, P, C> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}

impl<F: RingCore + Clone, O: Clone, P: Clone, C: Clone> Sub<&Expr<F, O, P, C>>
    for Expr<F, O, P, C>
{
    type Output = Self;

    fn sub(self, rhs: &Expr<F, O, P, C>) -> Self::Output {
        self - rhs.clone()
    }
}

impl<F: RingCore + Clone, O: Clone, P: Clone, C: Clone> Sub<Expr<F, O, P, C>>
    for &Expr<F, O, P, C>
{
    type Output = Expr<F, O, P, C>;

    fn sub(self, rhs: Expr<F, O, P, C>) -> Self::Output {
        (*self).clone() - rhs
    }
}

impl<F: RingCore + Clone, O: Clone, P: Clone, C: Clone> Sub<&Expr<F, O, P, C>>
    for &Expr<F, O, P, C>
{
    type Output = Expr<F, O, P, C>;

    fn sub(self, rhs: &Expr<F, O, P, C>) -> Self::Output {
        (*self).clone() - rhs
    }
}

impl<F: RingCore + FromPrimitiveInt, O, P, C> Sub<i128> for Expr<F, O, P, C> {
    type Output = Self;

    fn sub(self, rhs: i128) -> Self::Output {
        self - Self::from(rhs)
    }
}

impl<F: RingCore + FromPrimitiveInt, O, P, C> Sub<Expr<F, O, P, C>> for i128 {
    type Output = Expr<F, O, P, C>;

    fn sub(self, rhs: Expr<F, O, P, C>) -> Self::Output {
        Expr::from(self) - rhs
    }
}

impl<F: RingCore, O, P, C> Neg for Expr<F, O, P, C> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        for term in &mut self.terms {
            term.coefficient = -term.coefficient;
        }
        self
    }
}

impl<F: RingCore + Clone, O: Clone, P: Clone, C: Clone> Neg for &Expr<F, O, P, C> {
    type Output = Expr<F, O, P, C>;

    fn neg(self) -> Self::Output {
        -(*self).clone()
    }
}

impl<F: RingCore, O: Clone, P: Clone, C: Clone> Mul for Expr<F, O, P, C> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.is_zero() || rhs.is_zero() {
            return Self::zero();
        }

        let mut terms = Vec::with_capacity(self.terms.len() * rhs.terms.len());
        for lhs_term in self.terms {
            for rhs_term in &rhs.terms {
                let mut factors = lhs_term.factors.clone();
                factors.extend(rhs_term.factors.clone());
                terms.push(Term {
                    coefficient: lhs_term.coefficient * rhs_term.coefficient,
                    factors,
                });
            }
        }
        Self { terms }
    }
}

impl<F: RingCore, O: Clone, P: Clone, C: Clone> Mul<&Expr<F, O, P, C>> for Expr<F, O, P, C> {
    type Output = Self;

    fn mul(self, rhs: &Expr<F, O, P, C>) -> Self::Output {
        self * rhs.clone()
    }
}

impl<F: RingCore, O: Clone, P: Clone, C: Clone> Mul<Expr<F, O, P, C>> for &Expr<F, O, P, C> {
    type Output = Expr<F, O, P, C>;

    fn mul(self, rhs: Expr<F, O, P, C>) -> Self::Output {
        (*self).clone() * rhs
    }
}

impl<F: RingCore, O: Clone, P: Clone, C: Clone> Mul<&Expr<F, O, P, C>> for &Expr<F, O, P, C> {
    type Output = Expr<F, O, P, C>;

    fn mul(self, rhs: &Expr<F, O, P, C>) -> Self::Output {
        (*self).clone() * rhs
    }
}

impl<F: RingCore + FromPrimitiveInt, O, P, C> Mul<i128> for Expr<F, O, P, C> {
    type Output = Self;

    fn mul(mut self, rhs: i128) -> Self::Output {
        let rhs = F::from_i128(rhs);
        if rhs.is_zero() {
            return Self::zero();
        }
        for term in &mut self.terms {
            term.coefficient *= rhs;
        }
        self
    }
}

impl<F: RingCore + FromPrimitiveInt, O, P, C> Mul<Expr<F, O, P, C>> for i128 {
    type Output = Expr<F, O, P, C>;

    fn mul(self, rhs: Expr<F, O, P, C>) -> Self::Output {
        rhs * self
    }
}

#[cfg(test)]
mod tests {
    use crate::{challenge, constant, opening, Expr};
    use jolt_field::{Fr, FromPrimitiveInt};

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    enum Opening {
        A,
        B,
    }

    #[test]
    fn expression_ops_build_sum_of_products() {
        let expr: Expr<Fr, Opening> =
            opening(Opening::A) * opening(Opening::B) + challenge(0usize) * opening(Opening::A) - 3;

        assert_eq!(expr.terms.len(), 3);
        assert_eq!(expr.required_openings(), vec![Opening::A, Opening::B]);
        assert_eq!(expr.num_challenges(), 1);
    }

    #[test]
    fn field_constants_scale_terms() {
        let expr: Expr<Fr, Opening> =
            opening(Opening::A) * constant(Fr::from_u64(7)) + constant(Fr::from_u64(2));

        let value = expr.evaluate_without_derived(|_| Fr::from_u64(3), |_| Fr::from_u64(0));

        assert_eq!(value, Fr::from_u64(23));
    }

    #[test]
    fn zero_terms_are_canonical_empty_expression() {
        let multiplier = i128::from(0);
        let expr: Expr<Fr, Opening> = opening(Opening::A) * multiplier;
        assert!(expr.is_zero());
        let constant: Expr<Fr, Opening> = constant(Fr::from_u64(0));
        assert_eq!(constant, Expr::zero());
    }
}
