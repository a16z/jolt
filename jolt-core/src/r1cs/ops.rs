//! Defines the Linear Combination (LC) object and associated operations.
//! A LinearCombination is a vector of Terms, where each Term is a pair of a Variable and a coefficient.

#[cfg(test)]
use super::inputs::ConstraintInput;
use crate::{
    field::{JoltField, OptimizedMul},
    poly::dense_mlpoly::DensePolynomial,
    utils::thread::unsafe_allocate_zero_vec,
};
use rayon::prelude::*;
use std::fmt::Debug;
#[cfg(test)]
use std::fmt::Write as _;
use std::hash::Hash;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Variable {
    Input(usize),
    Auxiliary(usize),
    Constant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Term(pub Variable, pub i64);
impl Term {
    #[cfg(test)]
    fn pretty_fmt<const C: usize, I: ConstraintInput>(&self, f: &mut String) -> std::fmt::Result {
        match self.0 {
            Variable::Input(var_index) | Variable::Auxiliary(var_index) => match self.1.abs() {
                1 => write!(f, "{:?}", I::from_index::<C>(var_index)),
                _ => write!(f, "{}⋅{:?}", self.1, I::from_index::<C>(var_index)),
            },
            Variable::Constant => write!(f, "{}", self.1),
        }
    }
}

/// Linear Combination of terms.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LC(Vec<Term>);

impl LC {
    pub fn new(terms: Vec<Term>) -> Self {
        #[cfg(test)]
        Self::assert_no_duplicate_terms(&terms);

        let mut sorted_terms = terms;
        sorted_terms.sort_by(|a, b| a.0.cmp(&b.0));
        LC(sorted_terms)
    }

    pub fn zero() -> Self {
        LC::new(vec![])
    }

    pub fn terms(&self) -> &[Term] {
        &self.0
    }

    pub fn constant_term(&self) -> Option<&Term> {
        self.0
            .last()
            .filter(|term| matches!(term.0, Variable::Constant))
    }

    pub fn constant_term_field<F: JoltField>(&self) -> F {
        if let Some(term) = self.constant_term() {
            F::from_i64(term.1)
        } else {
            F::zero()
        }
    }

    pub fn to_field_elements<F: JoltField>(&self) -> Vec<F> {
        self.terms()
            .iter()
            .map(|term| F::from_i64(term.1))
            .collect()
    }

    pub fn num_terms(&self) -> usize {
        self.0.len()
    }

    pub fn num_vars(&self) -> usize {
        self.0
            .iter()
            .filter(|term| matches!(term.0, Variable::Auxiliary(_) | Variable::Input(_)))
            .count()
    }

    pub fn evaluate<F: JoltField>(&self, values: &[F]) -> F {
        let num_vars = self.num_vars();
        assert_eq!(num_vars, values.len());

        let mut var_index = 0;
        let mut result = F::zero();
        for term in self.terms().iter() {
            match term.0 {
                Variable::Input(_) | Variable::Auxiliary(_) => {
                    result += values[var_index] * F::from_i64(term.1);
                    var_index += 1;
                }
                Variable::Constant => result += F::from_i64(term.1),
            }
        }
        result
    }

    // JP: Why does `evaluate` exist?
    pub fn evaluate_row<F: JoltField>(
        &self,
        flattened_polynomials: &[&DensePolynomial<F>],
        row: usize,
    ) -> F {
        self.terms()
            .iter()
            .map(|term| match term.0 {
                Variable::Input(var_index) | Variable::Auxiliary(var_index) => {
                    F::from_i64(term.1).mul_01_optimized(flattened_polynomials[var_index][row])
                }
                Variable::Constant => F::from_i64(term.1),
            })
            .sum()
    }

    pub fn evaluate_batch<F: JoltField>(
        &self,
        flattened_polynomials: &[&DensePolynomial<F>],
        batch_size: usize,
    ) -> Vec<F> {
        let mut output = unsafe_allocate_zero_vec(batch_size);
        self.evaluate_batch_mut::<F>(flattened_polynomials, &mut output);
        output
    }

    pub fn evaluate_batch_mut<F: JoltField>(
        &self,
        flattened_polynomials: &[&DensePolynomial<F>],
        output: &mut [F],
    ) {
        output.par_iter_mut().enumerate().for_each(|(i, eval)| {
            *eval = self
                .terms()
                .iter()
                .map(|term| match term.0 {
                    Variable::Input(var_index) | Variable::Auxiliary(var_index) => {
                        F::from_i64(term.1).mul_01_optimized(flattened_polynomials[var_index][i])
                    }
                    Variable::Constant => F::from_i64(term.1),
                })
                .sum()
        });
    }

    #[cfg(test)]
    pub fn pretty_fmt<const C: usize, I: ConstraintInput>(
        &self,
        f: &mut String,
    ) -> std::fmt::Result {
        if self.0.is_empty() {
            write!(f, "0")
        } else {
            if self.0.len() > 1 {
                write!(f, "(")?;
            }
            for (index, term) in self.0.iter().enumerate() {
                if term.1 == 0 {
                    continue;
                }
                if index > 0 {
                    if term.1 < 0 {
                        write!(f, " - ")?;
                    } else {
                        write!(f, " + ")?;
                    }
                }
                term.pretty_fmt::<C, I>(f)?;
            }
            if self.0.len() > 1 {
                write!(f, ")")?;
            }
            Ok(())
        }
    }

    #[cfg(test)]
    fn assert_no_duplicate_terms(terms: &[Term]) {
        let mut term_vec = Vec::new();
        for term in terms {
            if term_vec.contains(&term.0) {
                panic!("Duplicate variable found in terms: {:?}", term.0);
            } else {
                term_vec.push(term.0);
            }
        }
    }
}

// Arithmetic for LC

impl<T> std::ops::Add<T> for LC
where
    T: Into<LC>,
{
    type Output = Self;

    fn add(self, other: T) -> Self::Output {
        let other_lc: LC = other.into();
        let mut combined_terms = self.0;
        // TODO(sragss): Can be made more efficient by assuming sorted
        for other_term in other_lc.terms() {
            if let Some(term) = combined_terms
                .iter_mut()
                .find(|term| term.0 == other_term.0)
            {
                term.1 += other_term.1;
            } else {
                combined_terms.push(*other_term);
            }
        }
        LC::new(combined_terms)
    }
}

impl<T> std::ops::Add<T> for Term
where
    T: Into<LC>,
{
    type Output = LC;

    fn add(self, other: T) -> Self::Output {
        let lc: LC = self.into();
        let other_lc: LC = other.into();
        lc + other_lc
    }
}

impl<T> std::ops::Add<T> for Variable
where
    T: Into<LC>,
{
    type Output = LC;

    fn add(self, other: T) -> Self::Output {
        let lc: LC = self.into();
        let other_lc: LC = other.into();
        lc + other_lc
    }
}

impl std::ops::Neg for LC {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let negated_terms: Vec<Term> = self.0.into_iter().map(|term| -term).collect();
        LC::new(negated_terms)
    }
}

impl<T: Into<LC>> std::ops::Sub<T> for LC {
    type Output = Self;

    fn sub(self, other: T) -> Self::Output {
        let other: LC = other.into();
        let negated_other = -other;
        self + negated_other
    }
}

// Arithmetic for Term

impl std::ops::Neg for Term {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Term(self.0, -self.1)
    }
}

impl From<i64> for Term {
    fn from(val: i64) -> Self {
        Term(Variable::Constant, val)
    }
}

impl From<Variable> for Term {
    fn from(val: Variable) -> Self {
        Term(val, 1)
    }
}

impl std::ops::Sub for Variable {
    type Output = LC;

    fn sub(self, other: Self) -> Self::Output {
        let lhs: LC = self.into();
        let rhs: LC = other.into();
        lhs - rhs
    }
}

// Into<LC>

impl From<i64> for LC {
    fn from(val: i64) -> Self {
        LC::new(vec![Term(Variable::Constant, val)])
    }
}

impl From<Variable> for LC {
    fn from(val: Variable) -> Self {
        LC::new(vec![Term(val, 1)])
    }
}

impl From<Term> for LC {
    fn from(val: Term) -> Self {
        LC::new(vec![val])
    }
}

impl From<Vec<Term>> for LC {
    fn from(val: Vec<Term>) -> Self {
        LC::new(val)
    }
}

// Generic arithmetic for Variable

impl std::ops::Mul<i64> for Variable {
    type Output = Term;

    fn mul(self, other: i64) -> Self::Output {
        Term(self, other)
    }
}

impl std::ops::Mul<Variable> for i64 {
    type Output = Term;

    fn mul(self, other: Variable) -> Self::Output {
        Term(other, self)
    }
}

/// Conversions and arithmetic for concrete ConstraintInput
#[macro_export]
macro_rules! impl_r1cs_input_lc_conversions {
    ($ConcreteInput:ty, $C:expr) => {
        impl Into<$crate::r1cs::ops::Variable> for $ConcreteInput {
            fn into(self) -> $crate::r1cs::ops::Variable {
                $crate::r1cs::ops::Variable::Input(self.to_index::<$C>())
            }
        }

        impl Into<$crate::r1cs::ops::Term> for $ConcreteInput {
            fn into(self) -> $crate::r1cs::ops::Term {
                $crate::r1cs::ops::Term(
                    $crate::r1cs::ops::Variable::Input(self.to_index::<$C>()),
                    1,
                )
            }
        }

        impl Into<$crate::r1cs::ops::LC> for $ConcreteInput {
            fn into(self) -> $crate::r1cs::ops::LC {
                $crate::r1cs::ops::Term(
                    $crate::r1cs::ops::Variable::Input(self.to_index::<$C>()),
                    1,
                )
                .into()
            }
        }

        impl Into<$crate::r1cs::ops::LC> for Vec<$ConcreteInput> {
            fn into(self) -> $crate::r1cs::ops::LC {
                let terms: Vec<$crate::r1cs::ops::Term> =
                    self.into_iter().map(Into::into).collect();
                $crate::r1cs::ops::LC::new(terms)
            }
        }

        impl<T: Into<$crate::r1cs::ops::LC>> std::ops::Add<T> for $ConcreteInput {
            type Output = $crate::r1cs::ops::LC;

            fn add(self, rhs: T) -> Self::Output {
                let lhs_lc: $crate::r1cs::ops::LC = self.into();
                let rhs_lc: $crate::r1cs::ops::LC = rhs.into();
                lhs_lc + rhs_lc
            }
        }

        impl<T: Into<$crate::r1cs::ops::LC>> std::ops::Sub<T> for $ConcreteInput {
            type Output = $crate::r1cs::ops::LC;

            fn sub(self, rhs: T) -> Self::Output {
                let lhs_lc: $crate::r1cs::ops::LC = self.into();
                let rhs_lc: $crate::r1cs::ops::LC = rhs.into();
                lhs_lc - rhs_lc
            }
        }

        impl std::ops::Mul<i64> for $ConcreteInput {
            type Output = $crate::r1cs::ops::Term;

            fn mul(self, rhs: i64) -> Self::Output {
                $crate::r1cs::ops::Term(
                    $crate::r1cs::ops::Variable::Input(self.to_index::<$C>()),
                    rhs,
                )
            }
        }

        impl std::ops::Mul<$ConcreteInput> for i64 {
            type Output = $crate::r1cs::ops::Term;

            fn mul(self, rhs: $ConcreteInput) -> Self::Output {
                $crate::r1cs::ops::Term(
                    $crate::r1cs::ops::Variable::Input(rhs.to_index::<$C>()),
                    self,
                )
            }
        }
        impl std::ops::Add<$ConcreteInput> for i64 {
            type Output = $crate::r1cs::ops::LC;

            fn add(self, rhs: $ConcreteInput) -> Self::Output {
                let term1 = $crate::r1cs::ops::Term(
                    $crate::r1cs::ops::Variable::Input(rhs.to_index::<$C>()),
                    1,
                );
                let term2 = $crate::r1cs::ops::Term($crate::r1cs::ops::Variable::Constant, self);
                $crate::r1cs::ops::LC::new(vec![term1, term2])
            }
        }
    };
}
